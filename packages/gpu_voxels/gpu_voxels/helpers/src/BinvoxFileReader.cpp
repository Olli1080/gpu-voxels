// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
//
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE.txt in the top
// directory of the source code.
//
// © Copyright 2014 FZI Forschungszentrum Informatik, Karlsruhe, Germany
//
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Andreas Hermann
 * \author  Michael Kazhdan
 * \date    2014-06-12
 *
 * Parser for the Binvox file format. http://www.google.com/search?q=binvox
 *
 * Software from Fakir Nooruddin and Greg Turk: "Simplification and Repair
 * of Polygonal Models Using Volumetric Techniques",
 * GVU technical report 99-37 (later published in IEEE Trans. on
 * Visualization and Computer Graphics, vol. 9, nr. 2, April 2003,
 * pages 191-205)
 *
 */
//----------------------------------------------------------------------
#include "BinvoxFileReader.h"

#include <fstream>

#include <gpu_voxels/logging/logging_gpu_voxels_helpers.h>

namespace gpu_voxels
{
    namespace file_handling
	{
        using namespace std;
        typedef unsigned char byte;

        bool BinvoxFileReader::readPointCloud(const std::filesystem::path& filename, std::vector<Vector3f>& points)
        {
            // 0 = empty voxel
            // 1 = filled voxel
            // A newline is output after every "dim" voxels (depth = height = width = dim)

            static int version;
            static int depth, height, width;
            static int size;
            static float tx, ty, tz;
            static float scale;

            auto input = ifstream(filename, ios::in | ios::binary);
            if (!input.good())
            {
                LOGGING_ERROR(Gpu_voxels_helpers, "Binvox: Could not open file " << filename.c_str() << " !" << endl);
                return false;
            }

            // read header
            string line;
            input >> line;  // #binvox
            if (line != "#binvox") 
            {
                LOGGING_ERROR(Gpu_voxels_helpers, "Binvox: First line reads [" << line << "] instead of [#binvox]" << endl);
                return false;
            }
            input >> version;
            LOGGING_DEBUG(Gpu_voxels_helpers, "Binvox: Reading version " << version << endl);

            depth = -1;
            int done = 0;
            while (input.good() && !done) 
            {
                input >> line;
                if (line == "data") done = 1;
                else if (line == "dim") 
                {
                    input >> depth >> height >> width;
                }
                else if (line == "translate") 
                {
                    input >> tx >> ty >> tz;
                }
                else if (line == "scale") 
                {
                    input >> scale;
                    scale = scale / width;
                }
                else 
                {
                    LOGGING_WARNING(Gpu_voxels_helpers, "Binvox: unrecognized keyword [" << line << "], skipping" << endl);
                    char c;
                    do {  // skip until end of line
                        c = input.get();
                    } while (input.good() && (c != '\n'));

                }
            }
            if (!done) 
            {
                LOGGING_ERROR(Gpu_voxels_helpers, "Binvox: Error reading header" << endl);
                return false;
            }
            if (depth == -1) 
            {
                LOGGING_ERROR(Gpu_voxels_helpers, "Binvox: Missing dimensions in header" << endl);
                return false;
            }

            size = width * height * depth;
            std::vector<byte> voxels;
            try
            {
                voxels.resize(size);
            }
            catch (...)
            {
                LOGGING_ERROR(Gpu_voxels_helpers, "Binvox: Error allocating memory" << endl);
                return false;
            }

            // read voxel data
            byte value;
            byte count;
            int index = 0;
            int end_index = 0;
            int nr_voxels = 0;

            input.unsetf(ios::skipws);  // need to read every byte now (!)
            input >> value;  // read the linefeed char

            while ((end_index < size) && input.good()) 
            {
                input >> value >> count;

                if (input.good()) 
                {
                    end_index = index + count;
                    if (end_index > size) return false;
                    for (int i = index; i < end_index; i++)
                    {
                        voxels[i] = value;
                    }
                    if (value) nr_voxels += count;
                    index = end_index;
                }  // if file still ok
            }
            input.close();

            // convert occupied voxels index to points in a pointcloud
            LOGGING_DEBUG(Gpu_voxels_helpers, "Binvox: Generating pointcloud from " << nr_voxels << " occupied Voxels" << endl);

            // The x-axis is the most significant axis, then the z-axis, then the y-axis.
            index = 0;
            //std::cout << "xyz size: " << depth << ", " << height << ", " << width << std::endl;
            for (int x = 0; x < depth; x++)
            {
                for (int z = 0; z < height; z++)
                {
                    for (int y = 0; y < width; y++)
                    {
                        if (voxels[index] == 1)
                        {
                            //std::cout << "Point at " << scale*x << ", " << scale*y << ", " << scale*z << std::endl;
                            points.emplace_back(scale * x + tx, scale * y + ty, scale * z + tz);
                        }
                        index++;
                    }
                }
            }

            LOGGING_DEBUG(
                Gpu_voxels_helpers,
                "Binvox Handler: loaded " << points.size() << " points (" << (points.size() * sizeof(Vector3f)) * cBYTE2MBYTE << " MB on CPU) from " << filename.c_str() << "." << endl);

            return true;
        }
    } // end of namespace
} // end of namespace