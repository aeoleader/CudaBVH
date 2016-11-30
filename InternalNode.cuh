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
class InternalNode : public Node
{
private:
	int left;	NodeType leftType;
	int right;	NodeType rightType;
public:
	__device__ __host__ InternalNode(){ left = -1; right = -1; type = INTERNALNODE; idx = -1; }
	__device__ __host__ InternalNode(int l, int r) : left(l), right(r) { type = INTERNALNODE; idx = -1; }

	__device__ __host__ int getLeftNodeIdx(){ return left; }
	__device__ __host__ NodeType getLeftNodeType() { return leftType; }
	__device__ __host__ void setLeftNode(int l, NodeType t) { left = l; leftType = t; }

	__device__ __host__ int getRightNodeIdx(){ return right; }
	__device__ __host__ NodeType getRightNodeType() { return rightType; }
	__device__ __host__ void setRightNode(int r, NodeType t) { right = r; rightType = t; }

	__device__ __host__ int checkType() { return type; }
	__device__ __host__ void setType() { type = INTERNALNODE; }
};