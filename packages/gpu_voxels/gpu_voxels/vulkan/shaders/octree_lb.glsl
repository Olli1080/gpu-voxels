// Load balancing framework of the octree traversals: port of load_balancer/kernel_config/LoadBalance.cuh (kernelLBWorkConcept,
// AbstractKernelConfig::abortLoop) to Vulkan. The including shader declares the push constant block `pc` with at least
//   uint stack_size;       items per stack
//   uint stack_threshold;  stack_size + LB_ITEMS - LB_THREADS: a work group stops when its stack holds that many items
//   uint idle_threshold;   number of idle work groups at which every work group stops
// and then writes its main() with the loop
//   if (tid == 0) lb_num = stack_counts[gl_WorkGroupID.x]; barrier();
//   while (true) { if (lb_should_stop()) break; items = lb_pop(); ...work...; prefix, write, lb_commit; }
//   if (tid == 0) stack_counts[gl_WorkGroupID.x] = lb_num;
// Every work group owns one stack of work items sorted by level (deepest on top) and processes it depth first, LB_ITEMS items per
// iteration with LB_LANES threads per item (one per child). The work groups never wait for each other: they only read the
// idle counter, so no forward progress guarantee is needed (a group that starts late simply stops at once and keeps its stack).
// The CUDA kernel uses warp votes for the prefix count of new items; here it is a Hillis-Steele scan in shared memory, so no
// assumption about the subgroup size is made.
const uint LB_THREADS = 64u;
const uint LB_LANES = 8u;                  // threads per work item = branching factor
const uint LB_ITEMS = LB_THREADS / LB_LANES;
const uint LB_WORDS = 6u;                  // words of one work item

layout(local_size_x = 64) in;
layout(std430, set = 0, binding = 0) buffer Stacks { uint stacks[]; };
layout(std430, set = 0, binding = 1) buffer StackCounts { uint stack_counts[]; };
layout(std430, set = 0, binding = 2) buffer Idle { uint idle_count; };

shared uint lb_num;                        // items on the stack of this work group
shared uint lb_cache[LB_ITEMS * LB_WORDS]; // the items grabbed for the current iteration
shared uint lb_flag;
shared uint lb_scan[LB_THREADS];

uint lb_base() { return gl_WorkGroupID.x * pc.stack_size * LB_WORDS; }

// abortLoop: empty or full stack (counted as idle) or enough idle work groups. Uniform control flow; ends with a barrier.
bool lb_should_stop()
{
    const uint tid = gl_LocalInvocationID.x;
    if (lb_num == 0u || lb_num >= pc.stack_threshold)
    {
        if (tid == 0u)
            atomicAdd(idle_count, 1u);
        barrier();
        return true;
    }
    if (tid == 0u)
        lb_flag = atomicAdd(idle_count, 0u) >= pc.idle_threshold ? 1u : 0u;
    barrier();
    const bool stop = lb_flag != 0u;
    barrier();
    return stop;
}

// Takes the top min(LB_ITEMS, stack) items into lb_cache; returns their number (uniform).
uint lb_pop()
{
    const uint tid = gl_LocalInvocationID.x;
    const uint items = min(LB_ITEMS, lb_num);
    const uint first = lb_num - items;
    for (uint w = tid; w < items * LB_WORDS; w += LB_THREADS)
        lb_cache[w] = stacks[lb_base() + first * LB_WORDS + w];
    barrier();
    if (tid == 0u)
        lb_num -= items;
    barrier();
    return items;
}

// Exclusive prefix count of `insert` over the threads of the work group; total = number of inserting threads. Ends with a barrier.
uint lb_prefix(bool insert, out uint total)
{
    const uint tid = gl_LocalInvocationID.x;
    lb_scan[tid] = insert ? 1u : 0u;
    barrier();
    for (uint offset = 1u; offset < LB_THREADS; offset <<= 1u)
    {
        const uint v = tid >= offset ? lb_scan[tid - offset] : 0u;
        barrier();
        lb_scan[tid] += v;
        barrier();
    }
    total = lb_scan[LB_THREADS - 1u];
    const uint exclusive = lb_scan[tid] - (insert ? 1u : 0u);
    barrier();
    return exclusive;
}

// Makes `total` items written above the stack top part of the stack.
void lb_commit(uint total)
{
    barrier();
    if (gl_LocalInvocationID.x == 0u)
        lb_num += total;
    barrier();
}
