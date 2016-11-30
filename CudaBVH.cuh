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
#include <Windows.h>
#include "BBox.cuh"
#include "Node.cuh"
#include "LeafNode.cuh"
#include "InternalNode.cuh"
#include "BVHTree.cuh"
#include <vector>
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>

using namespace std;
using namespace cub;

CachingDeviceAllocator  g_allocator(true);

__device__ int findSplit(unsigned int* sortedMortonCodes,
	int           first,
	int           last)
{
	// Identical Morton codes => split the range in the middle.

	unsigned int firstCode = sortedMortonCodes[first];
	unsigned int lastCode = sortedMortonCodes[last];

	if (firstCode == lastCode)
		return (first + last) >> 1;

	// Calculate the number of highest bits that are the same
	// for all objects, using the count-leading-zeros intrinsic.

	int commonPrefix = __clz(firstCode ^ lastCode);

	// Use binary search to find where the next bit differs.
	// Specifically, we are looking for the highest object that
	// shares more than commonPrefix bits with the first one.

	int split = first; // initial guess
	int step = last - first;

	do
	{
		step = (step + 1) >> 1; // exponential decrease
		int newSplit = split + step; // proposed new position

		if (newSplit < last)
		{
			unsigned int splitCode = sortedMortonCodes[newSplit];
			int splitPrefix = __clz(firstCode ^ splitCode);
			if (splitPrefix > commonPrefix)
				split = newSplit; // accept proposal
		}
	} while (step > 1);

	return split;
}

__device__ int delta(unsigned int* sortedMortonCodes, int x, int y, int numObjects)
{
	if (x >= 0 && x <= numObjects - 1 && y >= 0 && y <= numObjects - 1)
		return __clz(sortedMortonCodes[x] ^ sortedMortonCodes[y]);
	return -1;
}

__device__ int sign(int x)
{
	return (x > 0) - (x < 0);
}

__device__ int2 determineRange(unsigned int* sortedMortonCodes, int numObjects, int idx)
{
	int d = sign(delta(sortedMortonCodes, idx, idx + 1, numObjects) - delta(sortedMortonCodes, idx, idx - 1, numObjects));
	int dmin = delta(sortedMortonCodes, idx, idx - d, numObjects);
	int lmax = 2;
	while (delta(sortedMortonCodes, idx, idx + lmax * d, numObjects) > dmin)
		lmax = lmax * 2;
	int l = 0;
	for (int t = lmax / 2; t >= 1; t /= 2)
	{
		if (delta(sortedMortonCodes, idx, idx + (l + t)*d, numObjects) > dmin)
			l += t;
	}
	int j = idx + l*d;
	int2 range;
	range.x = min(idx, j);
	range.y = max(idx, j);
	return range;
}

__device__ unsigned int expandBits(unsigned int v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

__device__ bool checkOverlap(BBox& query, BBox& target)
{
	float e = 1e-10;
	float xa = query.xmax - target.xmin;
	float xb = target.xmax - query.xmin;
	float ya = query.ymax - target.ymin;
	float yb = target.ymax - query.ymin;
	float za = query.zmax - target.zmin;
	float zb = target.zmax - query.zmin;

	return ((xa>0|| abs(xa) < e) &&
		(xb>0 || abs(xb) < e) &&
		(ya>0 || abs(ya) < e) &&
		(yb>0 || abs(yb) < e) &&
		(za>0 || abs(za) < e) &&
		(zb>0 || abs(zb) < e) 
		);
}

__device__ void traverseIterative(int2* list,
	BVHTree     bvh,
	BBox    queryAABB,
	int     queryObjectIdx,
	int POTENTIAL_COLLIDE_PER_LEAF,
	int* collide_list_begin_end)
{
	Node* stack[64];
	Node** stackPtr = stack;
	*stackPtr++ = NULL; // push
	//printf("Query xmax: %12f xmin: %12f Target xmax: %12f xmin: %12f\n", queryAABB.xmax, bvh.getLeaf(queryObjectIdx)->getBBox().xmin);
	// Traverse nodes starting from the root.
	Node* node = bvh.getRoot();
	int begin = queryObjectIdx*POTENTIAL_COLLIDE_PER_LEAF;
	int end = begin;
	do
	{
		// Check each child node for overlap.
		Node *childL = bvh.getLeftChild(node->getIdx());
		Node *childR = bvh.getRightChild(node->getIdx());
		bool overlapL = (checkOverlap(queryAABB,
			bvh.getBBox(childL)));
		bool overlapR = (checkOverlap(queryAABB,
			bvh.getBBox(childR)));
		// Query overlaps a leaf node => report collision.
		if (overlapL && bvh.isLeaf(childL))
		{
			if (queryObjectIdx != bvh.getObjectIdx(childL)){
				int2 buf;
				buf.x = queryObjectIdx;
				buf.y = bvh.getObjectIdx(childL);
				list[end] = buf;
				end++;
			}
		}
		if (overlapR && bvh.isLeaf(childR))
		{
			if (queryObjectIdx != bvh.getObjectIdx(childR)){
				int2 buf;
				buf.x = queryObjectIdx;
				buf.y = bvh.getObjectIdx(childR);
				list[end] = buf;
				end++;
			}
		}
		// Query overlaps an internal node => traverse.
		bool traverseL = (overlapL && !bvh.isLeaf(childL));
		bool traverseR = (overlapR && !bvh.isLeaf(childR));

		if (!traverseL && !traverseR)
			node = *--stackPtr; // pop
		else
		{
			node = (traverseL) ? childL : childR;
			if (traverseL && traverseR){
				//modify
				*stackPtr = childR; // push
				stackPtr++;
			}
				
		}
	} while (node != NULL);
	collide_list_begin_end[queryObjectIdx * 2 + 0] = begin;
	collide_list_begin_end[queryObjectIdx * 2 + 1] = end;
}

__global__ void assignInternalNodes(int SAMPLE_SIZE, unsigned int *sortedMortonCodes, LeafNode* leafNodes, InternalNode* internalNodes)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < SAMPLE_SIZE - 1)
	{
		int2 range = determineRange(sortedMortonCodes, SAMPLE_SIZE, idx);
		int first = range.x;
		int last = range.y;

		// Determine where to split the range.

		int split = findSplit(sortedMortonCodes, first, last);

		// Select childA.
		Node* childA;
		int childAIdx;
		NodeType childAType;
		if (split == first)
		{
			childA = &leafNodes[split];
			childAIdx = split;
			childAType = NodeType(1);
		}
		else
		{
			childA = &internalNodes[split];
			childAIdx = split;
			childAType = NodeType(2);
		}
		// Select childB.
		Node* childB;
		int childBIdx;
		NodeType childBType;
		if (split + 1 == last)
		{
			childB = &leafNodes[split + 1];
			childBIdx = split + 1;
			childBType = NodeType(1);
		}
		else
		{
			childB = &internalNodes[split + 1];
			childBIdx = split + 1;
			childBType = NodeType(2);
		}

		// Record parent-child relationships.
		internalNodes[idx].setType();
		internalNodes[idx].setLeftNode(childAIdx, childAType);
		internalNodes[idx].setRightNode(childBIdx, childBType);
		internalNodes[idx].setIdx(idx);
		//internalNodes[idx].setParent(-1, NodeType(0));
		childA->setParent(idx, NodeType(2));
		childB->setParent(idx, NodeType(2));
		//printf("%d : %d\n", idx, internalNodes[idx].getParent());
		//printf("%d %d %d %d %d %d\n", idx, first, last, split, childA->getParent(), childB->getParent());
	}
}

__global__ void morton3DCuda(int SAMPLE_SIZE, unsigned int *c, BBox *objects)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < SAMPLE_SIZE)
	{

		float x, y, z;
		x = (objects[idx].x00 + objects[idx].x10 + objects[idx].x20) / 3;
		y = (objects[idx].x01 + objects[idx].x11 + objects[idx].x21) / 3;
		z = (objects[idx].x02 + objects[idx].x12 + objects[idx].x22) / 3;
		//printf("%f ,%f ,%f\n", x, y, z);
		x = fmin(fmax(x * 1024.0f, 0.0f), 1023.0f);
		y = fmin(fmax(y * 1024.0f, 0.0f), 1023.0f);
		z = fmin(fmax(z * 1024.0f, 0.0f), 1023.0f);
		unsigned int xx = expandBits((unsigned int) x);
		unsigned int yy = expandBits((unsigned int) y);
		unsigned int zz = expandBits((unsigned int) z);
		c[idx] = xx * 4 + yy * 2 + zz;
	}
}

__global__ void valuesKernel(int SAMPLE_SIZE, int *keys)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < SAMPLE_SIZE)
		keys[index] = index;
}

__global__ void internalNodeBBox(int SAMPLE_SIZE, int* atom, InternalNode* internalNodes, LeafNode* leafNodes)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < SAMPLE_SIZE)
	{
		Node* ptr = &leafNodes[idx];
		InternalNode* parent = &internalNodes[ptr->getParent()];
		while (parent->getIdx() < SAMPLE_SIZE - 1 && parent->getIdx() > -1 && atomicCAS(&atom[parent->getIdx()], 0, 1) == 1)
		{
			BBox buf;
			float left, right;
			//printf("In while %d\n", parent->getIdx());

			left = parent->getLeftNodeType() == INTERNALNODE ? internalNodes[parent->getLeftNodeIdx()].getBBox().xmax : leafNodes[parent->getLeftNodeIdx()].getBBox().xmax;
			right = parent->getRightNodeType() == INTERNALNODE ? internalNodes[parent->getRightNodeIdx()].getBBox().xmax : leafNodes[parent->getRightNodeIdx()].getBBox().xmax;
			buf.xmax = fmax(left, right);
			left = parent->getLeftNodeType() == INTERNALNODE ? internalNodes[parent->getLeftNodeIdx()].getBBox().xmin : leafNodes[parent->getLeftNodeIdx()].getBBox().xmin;
			right = parent->getRightNodeType() == INTERNALNODE ? internalNodes[parent->getRightNodeIdx()].getBBox().xmin : leafNodes[parent->getRightNodeIdx()].getBBox().xmin;
			buf.xmin = fmin(left, right);
			left = parent->getLeftNodeType() == INTERNALNODE ? internalNodes[parent->getLeftNodeIdx()].getBBox().ymax : leafNodes[parent->getLeftNodeIdx()].getBBox().ymax;
			right = parent->getRightNodeType() == INTERNALNODE ? internalNodes[parent->getRightNodeIdx()].getBBox().ymax : leafNodes[parent->getRightNodeIdx()].getBBox().ymax;
			buf.ymax = fmax(left, right);
			left = parent->getLeftNodeType() == INTERNALNODE ? internalNodes[parent->getLeftNodeIdx()].getBBox().ymin : leafNodes[parent->getLeftNodeIdx()].getBBox().ymin;
			right = parent->getRightNodeType() == INTERNALNODE ? internalNodes[parent->getRightNodeIdx()].getBBox().ymin : leafNodes[parent->getRightNodeIdx()].getBBox().ymin;
			buf.ymin = fmin(left, right);
			left = parent->getLeftNodeType() == INTERNALNODE ? internalNodes[parent->getLeftNodeIdx()].getBBox().zmax : leafNodes[parent->getLeftNodeIdx()].getBBox().zmax;
			right = parent->getRightNodeType() == INTERNALNODE ? internalNodes[parent->getRightNodeIdx()].getBBox().zmax : leafNodes[parent->getRightNodeIdx()].getBBox().zmax;
			buf.zmax = fmax(left, right);
			left = parent->getLeftNodeType() == INTERNALNODE ? internalNodes[parent->getLeftNodeIdx()].getBBox().zmin : leafNodes[parent->getLeftNodeIdx()].getBBox().zmin;
			right = parent->getRightNodeType() == INTERNALNODE ? internalNodes[parent->getRightNodeIdx()].getBBox().zmin : leafNodes[parent->getRightNodeIdx()].getBBox().zmin;
			buf.zmin = fmin(left, right);
			//if (idx == 7 || idx == 13)
			//printf("Index: %d Parent: %d\n", idx, parent->getParent());
			//printf("Index %d -> %d with value: %f %f %f %f %f %f; \n", idx, parent->getIdx(), buf.xmax, buf.xmin, buf.ymax, buf.ymin, buf.zmax, buf.zmin);
			parent->setBBox(buf);
			ptr = parent;
			if (ptr->getParent() > -1) parent = &internalNodes[ptr->getParent()];
			else return;
		}
	}
}

__global__ void assignLeafNodes(int SAMPLE_SIZE, LeafNode* leafNodes, int* sortedObjectIDs, BBox* bbox)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < SAMPLE_SIZE)
	{
		leafNodes[idx].setType();
		leafNodes[idx].setIdx(idx);
		leafNodes[idx].setObjectID(sortedObjectIDs[idx]);
		leafNodes[idx].setBBox(bbox[sortedObjectIDs[idx]]);
		//printf("Index: %d ObjectID: %d\n", idx, leafNodes[idx].getObjectID());
	}
}

__global__ void findPotentialCollisions(int SAMPLE_SIZE, int2* list, int POTENTIAL_COLLIDE_PER_LEAF, int* collide_list_begin_end, BVHTree* bvh)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < SAMPLE_SIZE)
	{
		Node* leaf = bvh->getLeaf(idx);
		//printf("Enter global %d objectID %d bbox xmin %12f xmax %12f\n", idx, bvh->getObjectIdx(leaf), bvh->getBBox(leaf).xmax, bvh->getBBox(leaf).xmin);
		
		traverseIterative(list, *bvh,
			bvh->getBBox(leaf),
			bvh->getObjectIdx(leaf), POTENTIAL_COLLIDE_PER_LEAF, collide_list_begin_end);

	}
}

class CudaBVH{
private:

	int SAMPLE_SIZE;
	int COLLISION_LIMIT;
	int POTENTIAL_COLLIDE_PER_LEAF = 50;
	int THREADS_PER_BLOCK;
	BVHTree myTree;
	int2* d_collisionList;
	int* d_collide_list_begin_end;
	int2* h_collisionList;
	int* h_collide_list_begin_end;
	vector<int2> hv_collisionList;
	vector<int> hv_collide_list_begin_end;
	void generateValues(int *keys)
	{
		int *d_keys;
		int size = SAMPLE_SIZE*sizeof(unsigned int);
		checkCudaErrors(cudaMalloc((void **) &d_keys, size));

		valuesKernel << <SAMPLE_SIZE / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(SAMPLE_SIZE, d_keys);
		getLastCudaError("value err");

		checkCudaErrors(cudaMemcpy(keys, d_keys, size, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_keys));
	}

	BVHTree generateBVHTree(int* values, BBox *objects)
	{
		unsigned int* d_objKeys;
		int* d_objValues;
		BBox* d_objects;

		cudaMalloc((void **) &d_objValues, SAMPLE_SIZE*sizeof(int));
		cudaMalloc((void **) &d_objKeys, SAMPLE_SIZE*sizeof(unsigned int));
		cudaMalloc((void **) &d_objects, SAMPLE_SIZE*sizeof(BBox));


		cudaMemcpy(d_objects, objects, SAMPLE_SIZE*sizeof(BBox), cudaMemcpyHostToDevice);
		cudaMemcpy(d_objValues, values, SAMPLE_SIZE*sizeof(int), cudaMemcpyHostToDevice);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);
		morton3DCuda << <(SAMPLE_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(SAMPLE_SIZE, d_objKeys, d_objects);
		cudaEventRecord(stop);

		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		printf("It took me %f milliseconds to generate morton codes.\n", milliseconds);
		/*for (int i = 0; i < SAMPLE_SIZE; i++)
		printf("%d\n", d_keys[i]);*/

		cudaEventCreate(&start);
		cudaEventCreate(&stop);


		///Radix sort
		int num_items = SAMPLE_SIZE;
		int size = SAMPLE_SIZE * sizeof(unsigned int);
		DoubleBuffer<unsigned int> d_sortedKeys;
		DoubleBuffer<int> d_sortedValues;
		CubDebugExit(g_allocator.DeviceAllocate((void**) &d_sortedKeys.d_buffers[0], sizeof(unsigned int) * num_items));
		CubDebugExit(g_allocator.DeviceAllocate((void**) &d_sortedKeys.d_buffers[1], sizeof(unsigned int) * num_items));
		CubDebugExit(g_allocator.DeviceAllocate((void**) &d_sortedValues.d_buffers[0], sizeof(int) * num_items));
		CubDebugExit(g_allocator.DeviceAllocate((void**) &d_sortedValues.d_buffers[1], sizeof(int) * num_items));

		// Allocate temporary storage
		size_t  temp_storage_bytes = 0;
		void    *d_temp_storage = NULL;
		cudaEventRecord(start);
		CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_sortedKeys, d_sortedValues, num_items));
		CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
		CubDebugExit(cudaMemcpy(d_sortedKeys.d_buffers[d_sortedKeys.selector], d_objKeys, sizeof(unsigned int) * num_items, cudaMemcpyDeviceToDevice));
		CubDebugExit(cudaMemcpy(d_sortedValues.d_buffers[d_sortedValues.selector], d_objValues, sizeof(int) * num_items, cudaMemcpyDeviceToDevice));
		CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_sortedKeys, d_sortedValues, num_items));
		cudaEventRecord(stop);
		checkCudaErrors(cudaMemcpy(d_objValues, d_sortedValues.Current(), SAMPLE_SIZE*sizeof(int), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(d_objKeys, d_sortedKeys.Current(), SAMPLE_SIZE*sizeof(unsigned int), cudaMemcpyDeviceToDevice));


		cudaEventSynchronize(stop);
		milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		printf("It took me %f milliseconds to run parallel radix sort.\n", milliseconds);
		///debug
		int* vbuf = new int[SAMPLE_SIZE];
		unsigned int* kbuf = new unsigned int[SAMPLE_SIZE];

		cudaMemcpy(vbuf, d_objValues, SAMPLE_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(kbuf, d_objKeys, SAMPLE_SIZE*sizeof(unsigned int), cudaMemcpyDeviceToHost);

		/*for (int i = 0; i < SAMPLE_SIZE; i++)
		cout << "Sorted value: " << vbuf[i] << " key: " << bitset<30>(kbuf[i]) << "\n";*/
		
		/////Generate hierachy

		LeafNode* d_leafNodes;
		InternalNode* d_internalNodes;
		// Construct leaf nodes.
		// Note: This step can be avoided by storing
		// the tree in a slightly different way.
		cudaMalloc((void **) &d_leafNodes, SAMPLE_SIZE*sizeof(LeafNode));
		cudaMallocManaged((void **) &d_internalNodes, (SAMPLE_SIZE - 1)*sizeof(InternalNode));
		for (int i = 0; i < SAMPLE_SIZE - 1; i++)
			d_internalNodes[i] = InternalNode();
		cudaDeviceSynchronize();
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);
		assignLeafNodes << <(SAMPLE_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(SAMPLE_SIZE, d_leafNodes, d_objValues, d_objects);

		assignInternalNodes << <(SAMPLE_SIZE + THREADS_PER_BLOCK - 2) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(SAMPLE_SIZE, d_objKeys, d_leafNodes, d_internalNodes);
		cudaEventRecord(stop);

		cudaEventSynchronize(stop);
		milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		printf("It took me %f milliseconds to generate hierachy.\n", milliseconds);
		cudaFree(d_objKeys); cudaFree(d_objValues); cudaFree(d_objects);

		/*LeafNode* leafNodes = new LeafNode[SAMPLE_SIZE];
		InternalNode* internalNodes = new InternalNode[SAMPLE_SIZE - 1];
		cudaMemcpy(leafNodes, d_leafNodes, SAMPLE_SIZE*sizeof(LeafNode), cudaMemcpyDeviceToHost);
		cudaMemcpy(internalNodes, d_internalNodes, (SAMPLE_SIZE-1)*sizeof(InternalNode), cudaMemcpyDeviceToHost);
		for (int i = 0; i < SAMPLE_SIZE - 1; i++)
		cout << internalNodes[i].getIdx() << " " << internalNodes[i].getParent() << "\n";*/
		/////Assign bounding box to internal nodes
		int* atom;
		cudaMalloc((void **) &atom, SAMPLE_SIZE*sizeof(int));
		cudaMemset(atom, 0, SAMPLE_SIZE*sizeof(int));

		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);
		internalNodeBBox << <(SAMPLE_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(SAMPLE_SIZE, atom, d_internalNodes, d_leafNodes);
		cudaEventRecord(stop);

		cudaEventSynchronize(stop);
		milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		printf("It took me %f milliseconds to assign bounding box.\n", milliseconds);
		LeafNode* leafNodes = new LeafNode[SAMPLE_SIZE];
		InternalNode* internalNodes = new InternalNode[SAMPLE_SIZE - 1];

		cudaMemcpy(leafNodes, d_leafNodes, SAMPLE_SIZE*sizeof(LeafNode), cudaMemcpyDeviceToHost);
		cudaMemcpy(internalNodes, d_internalNodes, (SAMPLE_SIZE - 1)*sizeof(InternalNode), cudaMemcpyDeviceToHost);
		//printBVH(internalNodes, leafNodes, 0);
		cudaFree(d_internalNodes); cudaFree(d_leafNodes);
		BVHTree buf;
		buf.internalNodes = internalNodes;
		buf.leafNodes = leafNodes;
		return buf;
	}

	void findAllCollision()
	{
		//thrust::device_vector<thrust::device_vector<int2>> d_collisionList;
		cudaMallocManaged((void**) &d_collisionList, (COLLISION_LIMIT)*sizeof(int2));
		cudaMallocManaged((void**)&d_collide_list_begin_end, (SAMPLE_SIZE*2)*sizeof(int));
		cudaDeviceSynchronize();
		InternalNode* d_internalNodes;
		LeafNode* d_leafNodes;
		cudaMalloc((void**) &d_internalNodes, (SAMPLE_SIZE-1)*sizeof(InternalNode));
		cudaMalloc((void**) &d_leafNodes, SAMPLE_SIZE*sizeof(LeafNode));
		cudaMemcpy(d_internalNodes, myTree.internalNodes, (SAMPLE_SIZE - 1)*sizeof(InternalNode), cudaMemcpyHostToDevice);
		cudaMemcpy(d_leafNodes, myTree.leafNodes, SAMPLE_SIZE * sizeof(LeafNode), cudaMemcpyHostToDevice);
		BVHTree* d_buf;
		cudaMallocManaged((void**) &d_buf, sizeof(BVHTree));
		d_buf->internalNodes = d_internalNodes;
		d_buf->leafNodes = d_leafNodes;
		cudaDeviceSynchronize();
		findPotentialCollisions << <(SAMPLE_SIZE + THREADS_PER_BLOCK - 1), THREADS_PER_BLOCK >> > (SAMPLE_SIZE, d_collisionList, POTENTIAL_COLLIDE_PER_LEAF, d_collide_list_begin_end,d_buf);
		//thrust::host_vector<thrust::host_vector<int2>> h_collisionList = d_collisionList;
		h_collisionList=new int2[COLLISION_LIMIT];
		h_collide_list_begin_end = new int[SAMPLE_SIZE * 2];
		cudaMemcpy(h_collisionList, d_collisionList, COLLISION_LIMIT*sizeof(int2), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_collide_list_begin_end, d_collide_list_begin_end, SAMPLE_SIZE*2*sizeof(int), cudaMemcpyDeviceToHost);
		

		for (int i = 0; i < SAMPLE_SIZE; i++){
			hv_collide_list_begin_end.push_back(hv_collisionList.size());
			for (int j = h_collide_list_begin_end[i * 2 + 0]; j < h_collide_list_begin_end[i * 2 + 1]; j++){
				hv_collisionList.push_back(h_collisionList[j]);
			}
		}
		hv_collide_list_begin_end.push_back(hv_collisionList.size());
		delete[] h_collisionList;
		h_collisionList = hv_collisionList.data();
		delete[] h_collide_list_begin_end;
		h_collide_list_begin_end = hv_collide_list_begin_end.data();

		cudaDeviceSynchronize();
		
		cudaFree(d_collisionList);
		cudaFree(d_collide_list_begin_end);
		cudaFree(d_internalNodes);
		cudaFree(d_leafNodes);
	}
public:
	CudaBVH()
	{
		SAMPLE_SIZE = 10;
		THREADS_PER_BLOCK = 5;
		BBox* dummy = new BBox[SAMPLE_SIZE];
		generateSampleDataset(dummy);
		Init(dummy, SAMPLE_SIZE, THREADS_PER_BLOCK);
		free(dummy);
	}
	CudaBVH(int sample_size, int threads_per_block)
	{
		SAMPLE_SIZE = sample_size;
		THREADS_PER_BLOCK = threads_per_block;
		BBox* dummy = new BBox[SAMPLE_SIZE];
		generateSampleDataset(dummy);
		Init(dummy, SAMPLE_SIZE, THREADS_PER_BLOCK);
		free(dummy);
	}
	CudaBVH(BBox* objects, int sample_size, int threads_per_block)
	{
		SAMPLE_SIZE = sample_size;
		THREADS_PER_BLOCK = threads_per_block;
		COLLISION_LIMIT = SAMPLE_SIZE*POTENTIAL_COLLIDE_PER_LEAF;
		Init(objects, SAMPLE_SIZE, THREADS_PER_BLOCK);
		
	}
	void Init(BBox* objects, int sample_size, int threads_per_block)
	{
		SAMPLE_SIZE = sample_size;
		THREADS_PER_BLOCK = threads_per_block;
		int *values;
		values = (int*) malloc(SAMPLE_SIZE*sizeof(unsigned int));
		generateValues(values);
		myTree = generateBVHTree(values, objects);
		findAllCollision();
		printf("Collision found: %d\n", h_collide_list_begin_end[SAMPLE_SIZE]);
		/*for (int i = 0; i < SAMPLE_SIZE - 1; i++)
		printf("%d : %d\n", i, myTree.internalNodes[i].getParent());*/
		free(values);
	}

	void generateSampleDataset(BBox *objects)
	{
		float* buf = new float[6];
		for (int i = 0; i < SAMPLE_SIZE; i++)
		{
			for (int j = 0; j < 6; j++)
				buf[j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			objects[i].xmax = max(buf[0], buf[1]);
			objects[i].xmin = min(buf[0], buf[1]);
			objects[i].ymax = max(buf[2], buf[3]);
			objects[i].ymin = min(buf[2], buf[3]);
			objects[i].zmax = max(buf[4], buf[5]);
			objects[i].zmin = min(buf[4], buf[5]);
		}
	}


#if 1
	void printBVH(int idx, int level)
	{
		cout << "Internal (Parent " << myTree.internalNodes[idx].getParent() << ") " << idx << " " << myTree.internalNodes[idx].getBBox().toString() << "\n";
		if (myTree.internalNodes[idx].getLeftNodeType() == LEAFNODE)
		{
			cout << "Leaf (l) " << "Parent " << myTree.leafNodes[myTree.internalNodes[idx].getLeftNodeIdx()].getParent() << " " << myTree.leafNodes[myTree.internalNodes[idx].getLeftNodeIdx()].getObjectID() << " "
				<< myTree.leafNodes[myTree.internalNodes[idx].getLeftNodeIdx()].getBBox().toString() << "\n";
		}
		else printBVH(myTree.internalNodes[idx].getLeftNodeIdx(), level + 1);

		if (myTree.internalNodes[idx].getRightNodeType() == LEAFNODE)
		{
			cout << "Leaf (r) " << "Parent " << myTree.leafNodes[myTree.internalNodes[idx].getRightNodeIdx()].getParent() << " " <<  myTree.leafNodes[myTree.internalNodes[idx].getRightNodeIdx()].getObjectID() << " "
				<< myTree.leafNodes[myTree.internalNodes[idx].getRightNodeIdx()].getBBox().toString() << "\n";
		}
		else printBVH(myTree.internalNodes[idx].getRightNodeIdx(), level + 1);
	}
#else
	void printBVH(int dummy, int dummy2) {
		int visit_stack[64] = { 0 };
		int level_stack[64] = { 0 };
		int stack_ptr = 1;

		while (stack_ptr > 0) {
			--stack_ptr;
			int idx = visit_stack[stack_ptr];
			int level = level_stack[stack_ptr];
			cout << "Internal (" << level << ") " << idx << " " << myTree.internalNodes[idx].getBBox().toString() << endl;

			if (myTree.internalNodes[idx].getRightNodeType() == LEAFNODE) {
				cout << "Leaf (r) " << myTree.leafNodes[myTree.internalNodes[idx].getRightNodeIdx()].getObjectID() << " " << myTree.leafNodes[myTree.internalNodes[idx].getRightNodeIdx()].getBBox().toString() << endl;
			}
			else {
				visit_stack[stack_ptr] = myTree.internalNodes[idx].getRightNodeIdx();
				level_stack[stack_ptr] = level + 1;
				stack_ptr++;
			}
			if (myTree.internalNodes[idx].getLeftNodeType() == LEAFNODE) {
				cout << "Leaf (l) " << myTree.leafNodes[myTree.internalNodes[idx].getLeftNodeIdx()].getObjectID() << " " << myTree.leafNodes[myTree.internalNodes[idx].getLeftNodeIdx()].getBBox().toString() << endl;
			}
			else {
				visit_stack[stack_ptr] = myTree.internalNodes[idx].getLeftNodeIdx();
				level_stack[stack_ptr] = level + 1;
				stack_ptr++;
			}
		}
	}
#endif

#if 0
	void drawBVHRecursive(int idx, int level)
	{
		if (level > 32) return;

		float color[3] = { level * 0.03, 1 - level * 0.03, 0 };

		myTree.internalNodes[idx].getBBox().draw(color);

		if (myTree.internalNodes[idx].getLeftNodeType() == LEAFNODE)
		{
			myTree.leafNodes[myTree.internalNodes[idx].getLeftNodeIdx()].getBBox().draw(color);
		}
		else drawBVHRecursive(myTree.internalNodes[idx].getLeftNodeIdx(), level + 1);

		if (myTree.internalNodes[idx].getRightNodeType() == LEAFNODE)
		{
			myTree.leafNodes[myTree.internalNodes[idx].getRightNodeIdx()].getBBox().draw(color);
		}
		else drawBVHRecursive(myTree.internalNodes[idx].getRightNodeIdx(), level + 1);
	}
	void draw() {
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

		drawBVHRecursive(0, 0);

		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
#else
	void draw() {
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

		int visit_stack[64] = { 0 };
		int level_stack[64] = { 0 };
		int stack_ptr = 1;

		while (stack_ptr > 0) {
			--stack_ptr;
			int idx = visit_stack[stack_ptr];
			int level = level_stack[stack_ptr];

			if (level > 32) return;
			float color[3] = { level * 0.03, 1 - level * 0.03, 0 };

			myTree.internalNodes[idx].getBBox().draw(color);

			if (myTree.internalNodes[idx].getRightNodeType() == LEAFNODE) {
				myTree.leafNodes[myTree.internalNodes[idx].getRightNodeIdx()].getBBox().draw(color);
			}
			else {
				visit_stack[stack_ptr] = myTree.internalNodes[idx].getRightNodeIdx();
				level_stack[stack_ptr] = level + 1;
				stack_ptr++;
			}
			if (myTree.internalNodes[idx].getLeftNodeType() == LEAFNODE) {
				myTree.leafNodes[myTree.internalNodes[idx].getLeftNodeIdx()].getBBox().draw(color);
			}
			else {
				visit_stack[stack_ptr] = myTree.internalNodes[idx].getLeftNodeIdx();
				level_stack[stack_ptr] = level + 1;
				stack_ptr++;
			}
		}

		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
#endif
	void printCollisionList()
	{
		/*printf("Collision Pairs: \n");
		if (collisionList == NULL) return;
		for (int i = 1; i <= collisionList[0].x; i++)
			printf("%d %d\n", collisionList[i].x, collisionList[i].y);*/
	}
};