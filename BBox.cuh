#include <GL/freeglut.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <bitset>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>
#include "test/test_util.h"
#include <intrin.h>
#include <chrono>
#pragma once
using namespace std;
#define vRaw(v) &v.x
struct BBox
{
	float xmin, xmax, ymin, ymax, zmin, zmax;
	float x00,x01,x02,x10,x11,x12,x20,x21,x22;
	__device__ __host__ string toString(){
		return (std::to_string(xmin) + " " + std::to_string(xmax)
			+ " " + std::to_string(ymin) + " " + std::to_string(ymax) + " " + std::to_string(zmin) + " " + std::to_string(zmax) + "\n");
	};
	__device__ __host__ BBox(){ xmin = FLT_MAX; xmax = -FLT_MAX; ymin = FLT_MAX; ymax = -FLT_MAX; zmin = FLT_MAX; zmax = -FLT_MAX; }

    void draw(float rgb[3])
    {
        float3 corners[8];
        corners[0] = make_float3(xmin, ymax, zmax);
        corners[1] = make_float3(xmax, ymax, zmax);
        corners[2] = make_float3(xmax, ymin, zmax);
        corners[3] = make_float3(xmin, ymin, zmax);
        corners[4] = make_float3(xmin, ymax, zmin);
        corners[5] = make_float3(xmax, ymax, zmin);
        corners[6] = make_float3(xmax, ymin, zmin);
        corners[7] = make_float3(xmin, ymin, zmin);
        glBegin(GL_QUADS);
        glColor3fv(rgb);
        glVertex3fv(vRaw(corners[0]));
        glVertex3fv(vRaw(corners[1]));
        glVertex3fv(vRaw(corners[2]));
        glVertex3fv(vRaw(corners[3]));
        glVertex3fv(vRaw(corners[1]));
        glVertex3fv(vRaw(corners[5]));
        glVertex3fv(vRaw(corners[6]));
        glVertex3fv(vRaw(corners[2]));
        glVertex3fv(vRaw(corners[0]));
        glVertex3fv(vRaw(corners[4]));
        glVertex3fv(vRaw(corners[5]));
        glVertex3fv(vRaw(corners[1]));
        glVertex3fv(vRaw(corners[4]));
        glVertex3fv(vRaw(corners[5]));
        glVertex3fv(vRaw(corners[6]));
        glVertex3fv(vRaw(corners[7]));
        glVertex3fv(vRaw(corners[0]));
        glVertex3fv(vRaw(corners[4]));
        glVertex3fv(vRaw(corners[7]));
        glVertex3fv(vRaw(corners[3]));
        glVertex3fv(vRaw(corners[3]));
        glVertex3fv(vRaw(corners[7]));
        glVertex3fv(vRaw(corners[6]));
        glVertex3fv(vRaw(corners[2]));
        glEnd();
    }
};