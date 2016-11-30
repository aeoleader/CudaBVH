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
#include "Node.cuh"
#include "NodeType.cuh"
#pragma once
class LeafNode : public Node
{
private:
	int objectID;
public:
	__device__ __host__ LeafNode(){ objectID = NULL; type = LEAFNODE; }
	__device__ __host__ LeafNode(int id, BBox bb) : objectID(id){ bbox = bb; type = LEAFNODE; }
	__device__ __host__ LeafNode(int id) : objectID(id) { type = LEAFNODE; }
	__device__ __host__ int getObjectID(){ return objectID; }
	__device__ __host__ void setObjectID(int id) { objectID = id; }
	__device__ __host__ int checkType() { return type; }
	__device__ __host__ void setType() { type = LEAFNODE; }
};