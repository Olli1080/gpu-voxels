If ($Args.Count -ne 2)
{
    echo "Usage: voxelize.sh path_to_mesh_file scaling_factor"
    exit 1
}

$Path = Get-Location

$MESH_FILE = $Args[0]
[double] $SCALING = $Args[1]

echo "==================================================="
echo "====== Step 1: Executing BinVox prerun... ========="
echo "==================================================="

$PREVOX_OUTPUT = $(.\binvox.exe -d 20 -rotx "$MESH_FILE")

echo "         --------- Pre-Run Output -----------"
echo "$PREVOX_OUTPUT"
echo "         -------- END Pre-Run Output --------"
echo ""

$pre_output = $PREVOX_OUTPUT | Select-String -Pattern "longest length: (?<len>[^ ]+)" -CaseSensitive -AllMatches

[double] $LENGTH = $pre_output.Matches.Groups[1].Value

$bb_match = $PREVOX_OUTPUT | Select-String -Pattern "Mesh::normalize, bounding box: \[([^, ]+), ([^, ]+), ([^, ]+), 1\] - \[([^, ]+), ([^, ]+), ([^, ]+), 1\]" -CaseSensitive -AllMatches

[double] $MIN_X = $bb_match.Matches.Groups[1].Value
[double] $MIN_Y = $bb_match.Matches.Groups[2].Value
[double] $MIN_Z = $bb_match.Matches.Groups[3].Value
[double] $MAX_X = $bb_match.Matches.Groups[4].Value
[double] $MAX_Y = $bb_match.Matches.Groups[5].Value
[double] $MAX_Z = $bb_match.Matches.Groups[6].Value

#echo $MIN_X, $MIN_Y, $MIN_Z     $MAX_X, $MAX_Y, $MAX_Z

[double] $res0 = ($LENGTH + $SCALING) / $SCALING

#echo "$LENGTH" "+" "$SCALING" "/" "$SCALING"
#echo $res0

$NUM_VOXELS = [math]::ceiling($res0)

#echo $NUM_VOXELS

echo "==================================================="
echo "====== Step 2: Executing BinVox final run... ======"
echo "==================================================="
echo "Params are:"
echo "Bounding Box of input mesh: Min: [ ${MIN_X}, ${MIN_Y}, ${MIN_Z} ] Max: [ ${MAX_X}, ${MAX_Y}, ${MAX_Z} ]"
echo "Bounding Box maximum side length: ${LENGTH} ==> Voxelcube sidelength in Voxels: ${NUM_VOXELS}"
echo ""

echo "         --------- Final-Run Output -----------"
#Add the -e option to generate hollow models.
echo ".\binvox.exe -d $NUM_VOXELS -bb ${MIN_X} ${MIN_Y} ${MIN_Z} ${MAX_X} ${MAX_Y} ${MAX_Z} -rotx $MESH_FILE"
.\binvox.exe -d $NUM_VOXELS -bb ${MIN_X} ${MIN_Y} ${MIN_Z} ${MAX_X} ${MAX_Y} ${MAX_Z} -rotx $MESH_FILE
echo "         -------- END Final-Run Output --------"
