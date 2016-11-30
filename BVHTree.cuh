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
#include "LeafNode.cuh"
#include "InternalNode.cuh"

struct BVHTree
{
	InternalNode *internalNodes;
	LeafNode *leafNodes;
	__device__ __host__ InternalNode* getRoot(){ return &internalNodes[0]; }
	__device__ __host__ Node* getLeftChild(int idx) {
		Node* ptr;
		if (internalNodes[idx].getLeftNodeType() == NodeType(1))
			ptr = &leafNodes[internalNodes[idx].getLeftNodeIdx()];
		if (internalNodes[idx].getLeftNodeType() == NodeType(2))
			ptr = &internalNodes[internalNodes[idx].getLeftNodeIdx()];
		return ptr;
	}
	__device__ __host__ Node* getRightChild(int idx) {
		Node* ptr;
		if (internalNodes[idx].getRightNodeType() == NodeType(1))
			ptr = &leafNodes[internalNodes[idx].getRightNodeIdx()];
		if (internalNodes[idx].getRightNodeType() == NodeType(2))
			ptr = &internalNodes[internalNodes[idx].getRightNodeIdx()];
		return ptr;
	}

	__device__ __host__ BBox getBBox(Node* node)
	{
		if (node->checkType() == NodeType(1))
			return leafNodes[node->getIdx()].getBBox();
		else
			return internalNodes[node->getIdx()].getBBox();
	}

	__device__ __host__ bool isLeaf(Node* node)
	{
		if (node->checkType() == NodeType(1))
			return true;
		return false;
	}

	__device__ __host__ int getObjectIdx(Node* node)
	{
		return leafNodes[node->getIdx()].getObjectID();
	}

	__device__ __host__ Node* getLeaf(int idx)
	{
		return &leafNodes[idx];
	}
};