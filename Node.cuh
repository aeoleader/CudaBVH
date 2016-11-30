#pragma once

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
#include "BBox.cuh"
#include "NodeType.cuh"

class Node
{
protected:
	NodeType type;
	int parent;
	NodeType parentType;
	BBox bbox;
	int idx;
public:
	
    __device__ __host__ Node() : bbox(BBox()), parent(-1) { type = NODE; }
	__device__ __host__ Node(int bbox) : bbox(BBox()), parent(-1) { type = NODE; }
	__device__ __host__ void setParent(int p, NodeType pt){ parent = p; parentType = pt; }
	__device__ __host__ int getParent(){ return parent; }
	__device__ __host__ BBox getBBox(){ return bbox; }
	__device__ __host__ void setBBox(BBox box) { bbox = box; }
	__device__ __host__ int checkType() { return type; }
	__device__ __host__ int getIdx() { return idx; }
	__device__ __host__ void setIdx(int id) { idx = id; }
};
