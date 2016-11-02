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

// #include <helper_math.h>

using namespace std;
using namespace cub;

CachingDeviceAllocator  g_allocator(true);

#define HASH_64 1

#if HASH_64
    typedef unsigned __int64 HashType;
#else
    typedef unsigned int HashType;
#endif

__device__ int findSplit(HashType* sortedMortonCodes,
	int           first,
	int           last)
{
	// Identical Morton codes => split the range in the middle.

	HashType firstCode = sortedMortonCodes[first];
	HashType lastCode = sortedMortonCodes[last];

	if (firstCode == lastCode)
		return (first + last) >> 1;

	// Calculate the number of highest bits that are the same
	// for all objects, using the count-leading-zeros intrinsic.

// 	int commonPrefix = __clz(firstCode ^ lastCode);
#if HASH_64
    int commonPrefix = __clzll(firstCode ^ lastCode);
#else
    int commonPrefix = __clz(firstCode ^ lastCode);
#endif

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
			HashType splitCode = sortedMortonCodes[newSplit];
// 			int splitPrefix = __clz(firstCode ^ splitCode);
#if HASH_64
            int splitPrefix = __clzll(firstCode ^ splitCode);
#else
            int splitPrefix = __clz(firstCode ^ splitCode);
#endif
			if (splitPrefix > commonPrefix)
				split = newSplit; // accept proposal
		}
	} while (step > 1);

	return split;
}

__device__ int delta(HashType* sortedMortonCodes, int x, int y, int numObjects)
{
	if (x >= 0 && x <= numObjects - 1 && y >= 0 && y <= numObjects - 1)
    {
//         return __clz(sortedMortonCodes[x] ^ sortedMortonCodes[y]);
#if HASH_64
        return __clzll(sortedMortonCodes[x] ^ sortedMortonCodes[y]);
#else
        return __clz(sortedMortonCodes[x] ^ sortedMortonCodes[y]);
#endif
    }
	return -1;
}

__device__ int sign(int x)
{
	return (x > 0) - (x < 0);
}

__device__ int2 determineRange(HashType* sortedMortonCodes, int numObjects, int idx)
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
    if (idx == 38043 || idx == 38044 || idx == 38045 || idx == 38046 || idx == 38047 || idx == 38048)
        printf("idx %d range :%d - %d j: %d morton: %d\n", idx, range.x, range.y, j, sortedMortonCodes[idx]);
	return range;
}

#if HASH_64
__device__ HashType expandBits(HashType v)
{
    v = (v * 0x000100000001u) & 0xFFFF00000000FFFFu;
    v = (v * 0x000000010001u) & 0x00FF0000FF0000FFu;
    v = (v * 0x000000000101u) & 0xF00F00F00F00F00Fu;
    v = (v * 0x000000000011u) & 0x30C30C30C30C30C3u;
    v = (v * 0x000000000005u) & 0x9249249249249249u;

    return v;
}
#else
__device__ HashType expandBits(HashType v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;

    return v;
}
#endif

// __device__ HashType expandBits(HashType v)
// {
//     unsigned int vh = (v >> 32u) & 0xFFFFFFFFu;
//     unsigned int vl = (v)& 0xFFFFFFFFu;
// 
// 	vh = (vh * 0x00010001u) & 0xFF0000FFu;
// 	vh = (vh * 0x00000101u) & 0x0F00F00Fu;
// 	vh = (vh * 0x00000011u) & 0xC30C30C3u;
// 	vh = (vh * 0x00000005u) & 0x49249249u;
// 
//     vh = 0;
// 
//     vl = (vl * 0x00010001u) & 0xFF0000FFu;
//     vl = (vl * 0x00000101u) & 0x0F00F00Fu;
//     vl = (vl * 0x00000011u) & 0xC30C30C3u;
//     vl = (vl * 0x00000005u) & 0x49249249u;
//     
//     return ((HashType)(vh) << 30u) | ((HashType)(vl) & 0x3FFFFFFFu);
// }

__global__ void assignInternalNodes(int SAMPLE_SIZE, HashType *sortedMortonCodes, LeafNode* leafNodes, InternalNode* internalNodes, int* sortedObjectIDs)
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
			childAType = LEAFNODE;
		}
		else
		{
			childA = &internalNodes[split];
			childAIdx = split;
			childAType = INTERNALNODE;
		}
		// Select childB.
		Node* childB;
		int childBIdx;
		NodeType childBType;
		if (split + 1 == last)
		{
			childB = &leafNodes[split + 1];
			childBIdx = split + 1;
			childBType = LEAFNODE;
		}
		else
		{
			childB = &internalNodes[split + 1];
			childBIdx = split + 1;
			childBType = INTERNALNODE;
		}

		// Record parent-child relationships.
		internalNodes[idx].setType();
		internalNodes[idx].setLeftNode(childAIdx, childAType);
		internalNodes[idx].setRightNode(childBIdx, childBType);
		internalNodes[idx].setIdx(idx);

        // the initialization is moved outside and using constructors to avoid overwrite
//         if (0)        {
//             //         if ()
//             //             internalNodes[idx].setParent(-1, NODE);
//         }

        childA->setParent(idx, INTERNALNODE);
		childB->setParent(idx, INTERNALNODE);
		//printf("%d %d %d %d %d %d\n", idx, first, last, split, childA->getParent(), childB->getParent());
	}
}

#if 0
__global__ void morton3DCuda(int SAMPLE_SIZE, HashType *c, const BBox *objects)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < SAMPLE_SIZE)
	{

		float x, y, z;
		x = (objects[idx].xmax + objects[idx].xmin) / 2;
		y = (objects[idx].ymax + objects[idx].ymin) / 2;
		z = (objects[idx].zmax + objects[idx].zmin) / 2;
		x = fmin(fmax(x * 1024.0f, 0.0f), 1023.0f);
		y = fmin(fmax(y * 1024.0f, 0.0f), 1023.0f);
		z = fmin(fmax(z * 1024.0f, 0.0f), 1023.0f);
		HashType xx = expandBits((HashType) x);
		HashType yy = expandBits((HashType) y);
		HashType zz = expandBits((HashType) z);
		c[idx] = xx * 4 + yy * 2 + zz;
	}
}
#else

struct MortonRec {
    float x, y, z;
    float xx, yy, zz;
    HashType ex, ey, ez;
    HashType m;
};

__global__ void morton3DCuda(int SAMPLE_SIZE, HashType *c, const BBox *objects, MortonRec *mor)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < SAMPLE_SIZE)
	{

        bool needprint = idx >= 38040 && idx <= 38050;

		float x, y, z;
        x = (objects[idx].xmax + objects[idx].xmin) / 2;  //if (needprint) printf("idx: %d  x: %f\n", idx, x);
        y = (objects[idx].ymax + objects[idx].ymin) / 2;  //if (needprint) printf("idx: %d  y: %f\n", idx, y);
        z = (objects[idx].zmax + objects[idx].zmin) / 2;  //if (needprint) printf("idx: %d  z: %f\n", idx, z);
        mor[idx].x = x;
        mor[idx].y = y;
        mor[idx].z = z;
//         x = fmin(fmax(x * 1023.0f, 0.0f), 1023.0f);
//         y = fmin(fmax(y * 1023.0f, 0.0f), 1023.0f);
//         z = fmin(fmax(z * 1023.0f, 0.0f), 1023.0f);
//         x = fmin(fmax(x * 1024.0f, 0.0f), 1024.0f);
//         y = fmin(fmax(y * 1024.0f, 0.0f), 1024.0f);
//         z = fmin(fmax(z * 1024.0f, 0.0f), 1024.0f);
//         x = fmin(fmax(x * 1023.99f, 0.0f), 1023.0f);
//         y = fmin(fmax(y * 1023.99f, 0.0f), 1023.0f);
//         z = fmin(fmax(z * 1023.99f, 0.0f), 1023.0f);
#if HASH_64
        x = x * 1024.0f * 1024.0f;  //if (needprint) printf("idx: %d  xx: %f\n", idx, x);
        y = y * 1024.0f * 1024.0f;  //if (needprint) printf("idx: %d  yy: %f\n", idx, y);
        z = z * 1024.0f * 1024.0f;  //if (needprint) printf("idx: %d  zz: %f\n", idx, z);
#else
        x = x * 1023.0f;  //if (needprint) printf("idx: %d  xx: %f\n", idx, x);
        y = y * 1023.0f;  //if (needprint) printf("idx: %d  yy: %f\n", idx, y);
        z = z * 1023.0f;  //if (needprint) printf("idx: %d  zz: %f\n", idx, z);
#endif
//         x = x * 1023.0f;  //if (needprint) printf("idx: %d  xx: %f\n", idx, x);
//         y = y * 1023.0f;  //if (needprint) printf("idx: %d  yy: %f\n", idx, y);
//         z = z * 1023.0f;  //if (needprint) printf("idx: %d  zz: %f\n", idx, z);
//         x = x * 1024.0f;  //if (needprint) printf("idx: %d  xx: %f\n", idx, x);
//         y = y * 1024.0f;  //if (needprint) printf("idx: %d  yy: %f\n", idx, y);
//         z = z * 1024.0f;  //if (needprint) printf("idx: %d  zz: %f\n", idx, z);
        mor[idx].xx = x;
        mor[idx].yy = y;
        mor[idx].zz = z;

		HashType xx = expandBits((HashType) ((double)x));  //if (needprint) printf("idx: %d  expand x: %d\n", idx, xx);
		HashType yy = expandBits((HashType) ((double)y));  //if (needprint) printf("idx: %d  expand y: %d\n", idx, yy);
		HashType zz = expandBits((HashType) ((double)z));  //if (needprint) printf("idx: %d  expand z: %d\n", idx, zz);
		c[idx] = xx * 4 + yy * 2 + zz;
        //if (needprint) printf("idx: %d  hash: %d\n", idx, c[idx]);

        mor[idx].ex = xx;
        mor[idx].ey = yy;
        mor[idx].ez = zz;
        mor[idx].m = c[idx];
	}
}
#endif

__global__ void valuesKernel(int SAMPLE_SIZE, int *keys)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < SAMPLE_SIZE)
		keys[index] = index;
}

__global__ void internalNodeBBox(int SAMPLE_SIZE, int* atom, InternalNode* internalNodes, LeafNode* leafNodes, BBox *d_myBBox)
{
#if 1
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

			parent->setBBox(buf);
			ptr = parent;
			if (ptr->getParent() > -1) parent = &internalNodes[ptr->getParent()];
			else return;
		}
	}
#else
#endif
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

__host__ __device__ bool IntersectRayAABB(const float3& start, const float3& dir, const float3& bmin, const float3& bmax, float& t)
{
    //! calculate candidate plane on each axis
    float tx = -1.0f, ty = -1.0f, tz = -1.0f;
    bool inside = true;

    //! use unrolled loops

    //! x
    if (start.x < bmin.x)
    {
        if (dir.x != 0.0f)
            tx = (bmin.x - start.x) / dir.x;
        inside = false;
    }
    else if (start.x > bmax.x)
    {
        if (dir.x != 0.0f)
            tx = (bmax.x - start.x) / dir.x;
        inside = false;
    }

    //! y
    if (start.y < bmin.y)
    {
        if (dir.y != 0.0f)
            ty = (bmin.y - start.y) / dir.y;
        inside = false;
    }
    else if (start.y > bmax.y)
    {
        if (dir.y != 0.0f)
            ty = (bmax.y - start.y) / dir.y;
        inside = false;
    }

    //! z
    if (start.z < bmin.z)
    {
        if (dir.z != 0.0f)
            tz = (bmin.z - start.z) / dir.z;
        inside = false;
    }
    else if (start.z > bmax.z)
    {
        if (dir.z != 0.0f)
            tz = (bmax.z - start.z) / dir.z;
        inside = false;
    }

    //! if point inside all planes
    if (inside)
    {
        t = 0.0f;
        return true;
    }

    //! we now have t values for each of possible intersection planes
    //! find the maximum to get the intersection point
    float tmax = tx;
    int taxis = 0;

    if (ty > tmax)
    {
        tmax = ty;
        taxis = 1;
    }
    if (tz > tmax)
    {
        tmax = tz;
        taxis = 2;
    }

    if (tmax < 0.0f)
        return false;

    //! check that the intersection point lies on the plane we picked
    //! we don't test the axis of closest intersection for precision reasons

    //! no eps for now
    float eps = 0.0f;

    float3 hit = make_float3(start.x + dir.x * tmax, start.y + dir.y * tmax, start.z + dir.z * tmax);

    if ((hit.x < bmin.x - eps || hit.x > bmax.x + eps) && taxis != 0)
        return false;
    if ((hit.y < bmin.y - eps || hit.y > bmax.y + eps) && taxis != 1)
        return false;
    if ((hit.z < bmin.z - eps || hit.z > bmax.z + eps) && taxis != 2)
        return false;

    //! output results
    t = tmax;

    return true;
}

__host__ __device__ inline float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__host__ __device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float3 normalize(const float3& v)
{
    float invLen = 1.0f / sqrtf(dot(v, v));
    return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

// Moller and Trumbore's method
__host__ __device__ bool IntersectRayTriTwoSided(const float3& p, const float3& dir, const float3& a, const float3& b, const float3& c, float& t, float& u, float& v)
{
    float3 ab = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
    float3 ac = make_float3(c.x - a.x, c.y - a.y, c.z - a.z);
//     float3 n = normalize(cross(ab, ac));
    float3 n = (cross(ab, ac));

    float3 ndir = make_float3(-dir.x, -dir.y, -dir.z);
    float d = dot(ndir, n);
    float ood = 1.0f / d; // No need to check for division by zero here as infinity aritmetic will save us...
    float3 ap = make_float3(p.x - a.x, p.y - a.y, p.z - a.z);

    t = dot(ap, n) * ood;
//     cout << "t = " << t << endl;
    if (t < 0.0f)
        return false;

    float3 e = cross(ndir, ap);
    v = dot(ac, e) * ood;
//     cout << "v = " << v << " | " << dot(ac, e) << " * " << ood << endl;
    if (v < 0.0f || v > 1.0f) // ...here...
    {
        return false;
    }
    float w = -dot(ab, e) * ood;
//     cout << "w = " << w << " | " << -dot(ab, e) << " * " << ood << endl;
    if (w < 0.0f || v + w > 1.0f) // ...and here
    {
        return false;
    }

    u = 1.0f - v - w;

    return true;
}

__device__ bool intersect(const float3& ray_orig, const float3& ray_dir, float& outT, float&outU, float& outV, int& outIdx,
    InternalNode *internalNodes, LeafNode *leafNodes, Triangle *myMesh) {
    //         float3 rcp_dir = make_float3(1.0f / ray_dir.x, 1.0f / ray_dir.y, 1.0f / ray_dir.z);
    outT = FLT_MAX;
    outU = 0;
    outV = 0;
    outIdx = -1;

    int visit_stack[64] = { 0 };
    int level_stack[64] = { 0 };
    int stack_ptr = 1;

    float near_t = FLT_MAX;
    float near_u = 0, near_v = 0;
    int near_idx = -1; // initially invalid

    while (stack_ptr > 0) {
        --stack_ptr;
        int idx = visit_stack[stack_ptr];
        int level = level_stack[stack_ptr];

        BBox box = internalNodes[idx].getBBox();
        float bbox_t = FLT_MAX;
        bool bbox_hit = IntersectRayAABB(ray_orig, ray_dir, make_float3(box.xmin, box.ymin, box.zmin), make_float3(box.xmax, box.ymax, box.zmax), bbox_t);

        if (!bbox_hit) {
            //                 cout << "not hit at level " << level << endl;
            continue;
        }
        else{
            //                 cout << "hit at level " << level << endl;
        }

        ///
        if (internalNodes[idx].getRightNodeType() == LEAFNODE) {

            // this is incorrect
            //                 const float3& a = myMesh[internalNodes[idx].getRightNodeIdx()].a;
            //                 const float3& b = myMesh[internalNodes[idx].getRightNodeIdx()].b;
            //                 const float3& c = myMesh[internalNodes[idx].getRightNodeIdx()].c;

            // corrected
            const float3& a = myMesh[leafNodes[internalNodes[idx].getRightNodeIdx()].getObjectID()].a;
            const float3& b = myMesh[leafNodes[internalNodes[idx].getRightNodeIdx()].getObjectID()].b;
            const float3& c = myMesh[leafNodes[internalNodes[idx].getRightNodeIdx()].getObjectID()].c;

#if 0 // debugging
            BBox boxleft = leafNodes[internalNodes[idx].getRightNodeIdx()].getBBox();
            if (boxleft.contains(a) && boxleft.contains(b) && boxleft.contains(c)) {
                cout << "contains" << endl;
            }
            else{
                cout << "not contains" << endl;

                cout << boxleft.toString();
                cout << a.x << "," << a.y << "," << a.z << endl;
                cout << b.x << "," << b.y << "," << b.z << endl;
                cout << c.x << "," << c.y << "," << c.z << endl;
                cout << level << endl;
            }
#endif

            float t = FLT_MAX;
            float u, v;
            //                 cout << "ray: " << "(" << ray_orig.x << "," << ray_orig.y << "," << ray_orig.z << ") "
            //                     << "(" << ray_dir.x << "," << ray_dir.y << "," << ray_dir.z << ")" << endl;
            bool hit = IntersectRayTriTwoSided(ray_orig, normalize(ray_dir), a, b, c, t, u, v);
            //                 cout << "testing right leaf: " << near_idx << ", " << hit << endl;
            if (hit && t < near_t) {
                near_t = t;
                near_u = u;
                near_v = v;
                near_idx = internalNodes[idx].getRightNodeIdx();
            }
        }
        else{
            visit_stack[stack_ptr] = internalNodes[idx].getRightNodeIdx();
            level_stack[stack_ptr] = level + 1;
            stack_ptr++;
        }

        ///
        if (internalNodes[idx].getLeftNodeType() == LEAFNODE) {
            // this is incorrect
            //                 const float3& a = myMesh[internalNodes[idx].getLeftNodeIdx()].a;
            //                 const float3& b = myMesh[internalNodes[idx].getLeftNodeIdx()].b;
            //                 const float3& c = myMesh[internalNodes[idx].getLeftNodeIdx()].c;

            // corrected
            const float3& a = myMesh[leafNodes[internalNodes[idx].getLeftNodeIdx()].getObjectID()].a;
            const float3& b = myMesh[leafNodes[internalNodes[idx].getLeftNodeIdx()].getObjectID()].b;
            const float3& c = myMesh[leafNodes[internalNodes[idx].getLeftNodeIdx()].getObjectID()].c;

#if 0 // debugging
            BBox boxleft = leafNodes[internalNodes[idx].getLeftNodeIdx()].getBBox();
            if (boxleft.contains(a) && boxleft.contains(b) && boxleft.contains(c)) {
                cout << "contains" << endl;
            }
            else{
                cout << "not contains" << endl;

                cout << boxleft.toString();
                cout << a.x << "," << a.y << "," << a.z << endl;
                cout << b.x << "," << b.y << "," << b.z << endl;
                cout << c.x << "," << c.y << "," << c.z << endl;
                cout << level << endl;
            }
#endif


            float t = FLT_MAX;
            float u, v;
            //                 cout << "ray: " << "(" << ray_orig.x << "," << ray_orig.y << "," << ray_orig.z << ") "
            //                     << "(" << ray_dir.x << "," << ray_dir.y << "," << ray_dir.z << ")" << endl;
            bool hit = IntersectRayTriTwoSided(ray_orig, normalize(ray_dir), a, b, c, t, u, v);
            //                 cout << "testing left leaf: " << near_idx << ", " << hit << endl;
            if (hit && t < near_t) {
                near_t = t;
                near_u = u;
                near_v = v;
                near_idx = internalNodes[idx].getLeftNodeIdx();
            }
        }
        else{
            visit_stack[stack_ptr] = internalNodes[idx].getLeftNodeIdx();
            level_stack[stack_ptr] = level + 1;
            stack_ptr++;
        }
    }

    outU = near_u;
    outV = near_v;
    outT = near_t;
    outIdx = near_idx;
    return outT != FLT_MAX; // ray epsilon to mitigate self intersection
}

__global__ void intersectKernel(
    int total_rays,
    float3 *ray_orig, float3 *ray_dir,
    float *outT, float *outU, float *outV, int *outIdx, int *outHit,
    InternalNode *internalNodes, LeafNode *leafNodes, Triangle *myMesh)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= total_rays)
        return;

    if (intersect(ray_orig[idx], ray_dir[idx], outT[idx], outU[idx], outV[idx], outIdx[idx], internalNodes, leafNodes, myMesh)) {
        outHit[idx] = 1;
    }
    else{
        outHit[idx] = 0;
    }
}

class CudaBVH{
public:

	int SAMPLE_SIZE;
	int THREADS_PER_BLOCK;
	BVHTree myTree, d_myTree;
    vector<Triangle> myMesh;
    vector<BBox> myBBox;
    Triangle *d_myMesh = nullptr;
    BBox *d_myBBox = nullptr;
    MortonRec *mor = nullptr;

	void generateValues(int *keys)
	{
		int *d_keys;
		int size = SAMPLE_SIZE*sizeof(HashType);
		checkCudaErrors(cudaMalloc((void **) &d_keys, size));

		valuesKernel << <(SAMPLE_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(SAMPLE_SIZE, d_keys);
		getLastCudaError("value err");

		checkCudaErrors(cudaMemcpy(keys, d_keys, size, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_keys));
	}

	BVHTree generateBVHTree(int* values, // element index in the unsorted array
        BBox *objects, Triangle *tris)
	{
//         for (int n = 0 ; n < SAMPLE_SIZE; n++) {
//             BBox b = objects[n];
//             Triangle t = tris[n];
//             if (b.contains(t.a) && b.contains(t.b) && b.contains(t.c)) {
//                 cout << "contains" << endl;
//             }
//             else{
//                 cout << "not contains" << endl;
// 
//                 cout << b.toString();
//                 cout << t.a.x << "," << t.a.y << "," << t.a.z << endl;
//                 cout << t.b.x << "," << t.b.y << "," << t.b.z << endl;
//                 cout << t.c.x << "," << t.c.y << "," << t.c.z << endl;
//             }
//         }


        // the unsorted element index
//         for (int n = 0; n < SAMPLE_SIZE; n++) {
//             cout << values[n] << endl;
//         }

        myMesh.resize(SAMPLE_SIZE);
        for (int n = 0; n < SAMPLE_SIZE; n++) {
            myMesh[n] = tris[n];
        }
        myBBox.resize(SAMPLE_SIZE);
        for (int n = 0; n < SAMPLE_SIZE; n++) {
            myBBox[n] = objects[n];
        }

        checkCudaErrors(cudaMalloc((void**)&d_myMesh, SAMPLE_SIZE * sizeof(Triangle)));
        checkCudaErrors(cudaMemcpy(d_myMesh, &myMesh[0], SAMPLE_SIZE * sizeof(Triangle), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc((void**)&d_myBBox, SAMPLE_SIZE * sizeof(BBox)));
        checkCudaErrors(cudaMemcpy(d_myBBox, &myBBox[0], SAMPLE_SIZE * sizeof(BBox), cudaMemcpyHostToDevice));

		HashType* d_objKeys;
		int* d_objValues;
		BBox* d_objects;

		cudaMalloc((void **) &d_objValues, SAMPLE_SIZE*sizeof(int));
		cudaMalloc((void **) &d_objKeys, SAMPLE_SIZE*sizeof(HashType));
		cudaMalloc((void **) &d_objects, SAMPLE_SIZE*sizeof(BBox));

		cudaMemcpy(d_objects, objects, SAMPLE_SIZE*sizeof(BBox), cudaMemcpyHostToDevice);
		cudaMemcpy(d_objValues, values, SAMPLE_SIZE*sizeof(int), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

        cudaMallocManaged((void**) &mor, SAMPLE_SIZE * sizeof(MortonRec));
        cudaDeviceSynchronize();
		cudaEventRecord(start);
		morton3DCuda << <(SAMPLE_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(SAMPLE_SIZE, d_objKeys, d_objects, mor);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();

//         {
//             HashType *keys = new HashType[SAMPLE_SIZE];
//             BBox *boxes = new BBox[SAMPLE_SIZE];
//             checkCudaErrors(cudaMemcpy(keys, d_objKeys, sizeof(HashType) * SAMPLE_SIZE, cudaMemcpyDeviceToHost));
//             checkCudaErrors(cudaMemcpy(boxes, d_objects, sizeof(BBox) * SAMPLE_SIZE, cudaMemcpyDeviceToHost));
//             for (int n = 0; n < SAMPLE_SIZE; n++) {
//                 cout << "key " << n << " = " << keys[n] << endl;
//             }
// 
//             for (;;);
//         }

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
		int size = SAMPLE_SIZE * sizeof(HashType);
		DoubleBuffer<HashType> d_sortedKeys;
		DoubleBuffer<int> d_sortedValues;
		CubDebugExit(g_allocator.DeviceAllocate((void**) &d_sortedKeys.d_buffers[0], sizeof(HashType) * num_items));
		CubDebugExit(g_allocator.DeviceAllocate((void**) &d_sortedKeys.d_buffers[1], sizeof(HashType) * num_items));
		CubDebugExit(g_allocator.DeviceAllocate((void**) &d_sortedValues.d_buffers[0], sizeof(int) * num_items));
		CubDebugExit(g_allocator.DeviceAllocate((void**) &d_sortedValues.d_buffers[1], sizeof(int) * num_items));

		// Allocate temporary storage
		size_t  temp_storage_bytes = 0;
		void    *d_temp_storage = NULL;
		cudaEventRecord(start);
		CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_sortedKeys, d_sortedValues, num_items));
        CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
		CubDebugExit(cudaMemcpy(d_sortedKeys.d_buffers[d_sortedKeys.selector], d_objKeys, sizeof(HashType) * num_items, cudaMemcpyDeviceToDevice));
		CubDebugExit(cudaMemcpy(d_sortedValues.d_buffers[d_sortedValues.selector], d_objValues, sizeof(int) * num_items, cudaMemcpyDeviceToDevice));
		CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_sortedKeys, d_sortedValues, num_items));
		cudaEventRecord(stop);
		checkCudaErrors(cudaMemcpy(d_objValues, d_sortedValues.Current(), SAMPLE_SIZE*sizeof(int), cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_objKeys, d_sortedKeys.Current(), SAMPLE_SIZE*sizeof(HashType), cudaMemcpyDeviceToDevice));

        //         {
        //             HashType *keys = new HashType[SAMPLE_SIZE];
        //             int *values = new int[SAMPLE_SIZE];
        //             checkCudaErrors(cudaMemcpy(keys, d_objKeys, sizeof(HashType) * SAMPLE_SIZE, cudaMemcpyDeviceToHost));
        //             checkCudaErrors(cudaMemcpy(values, d_objValues, sizeof(int) * SAMPLE_SIZE, cudaMemcpyDeviceToHost));
        //             for (int n = 0; n < SAMPLE_SIZE; n++) {
        //                 cout << "key, value " << n << " = " << keys[n] << ", " << values[n] << endl;
        //             }
        // 
        //             for (;;);
        //         }


        // (FALSE) also apply the sorted order to the triangle array
//         {
//             // the sorted element index
//             cudaMemcpy(values, d_objValues, SAMPLE_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
//             //for (int n = 0; n < SAMPLE_SIZE; n++) {
//             //    cout << values[n] << endl;
//             //}
// 
//             vector<Triangle> oldMesh = myMesh;
//             for (int n = 0; n < SAMPLE_SIZE; n++) {
//                 myMesh[n] = oldMesh[values[n]];
//             }
//         }

		cudaEventSynchronize(stop);
		milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		printf("It took me %f milliseconds to run parallel radix sort.\n", milliseconds);
		///debug
		/*int* vbuf = new int[SAMPLE_SIZE];
		HashType* kbuf = new HashType[SAMPLE_SIZE];

		cudaMemcpy(vbuf, d_objValues, SAMPLE_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(kbuf, d_objKeys, SAMPLE_SIZE*sizeof(HashType), cudaMemcpyDeviceToHost);

		for (int i = 0; i < SAMPLE_SIZE; i++)
		cout << "Sorted value: " << vbuf[i] << " key: " << bitset<30>(kbuf[i]) << "\n";
		*/
		/////Generate hierachy

		LeafNode* d_leafNodes;
		InternalNode* d_internalNodes;
		// Construct leaf nodes.
		// Note: This step can be avoided by storing
		// the tree in a slightly different way.
		cudaMalloc((void **) &d_leafNodes, SAMPLE_SIZE*sizeof(LeafNode));
		cudaMallocManaged((void **) &d_internalNodes, (SAMPLE_SIZE - 1)*sizeof(InternalNode));

        for (int n = 0; n < SAMPLE_SIZE; n++) {
            d_internalNodes[n] = InternalNode();
        }
        checkCudaErrors(cudaDeviceSynchronize());




		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);
		assignLeafNodes << <(SAMPLE_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(SAMPLE_SIZE, d_leafNodes, d_objValues, d_objects);
//         {
//             // the sorted element index
//             cudaMemcpy(values, d_objValues, SAMPLE_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
//             cudaMemcpy(objects, d_objects, SAMPLE_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
// //             for (int n = 0; n < SAMPLE_SIZE; n++) {
// //                 cout << values[n] << endl;
// //             }
//             
//             for (int n = 0; n < SAMPLE_SIZE; n++) {
//                 BBox b = objects[values[n]];
//                 Triangle t = myMesh[values[n]];
//                 if (b.contains(t.a) && b.contains(t.b) && b.contains(t.c)) {
// //                     cout << "contains" << endl;
//                 }
//                 else{
//                     cout << "not contains" << endl;
// 
//                     cout << b.toString();
//                     cout << t.a.x << "," << t.a.y << "," << t.a.z << endl;
//                     cout << t.b.x << "," << t.b.y << "," << t.b.z << endl;
//                     cout << t.c.x << "," << t.c.y << "," << t.c.z << endl;
//                 }
//             }
//             for(;;);
//         }

		assignInternalNodes << <(SAMPLE_SIZE + THREADS_PER_BLOCK - 2) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(SAMPLE_SIZE, d_objKeys, d_leafNodes, d_internalNodes, d_objValues);
		cudaEventRecord(stop);

		cudaEventSynchronize(stop);
		milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		printf("It took me %f milliseconds to generate hierachy.\n", milliseconds);
		cudaFree(d_objKeys); cudaFree(d_objValues); cudaFree(d_objects);

        ///

		/*LeafNode* leafNodes = new LeafNode[SAMPLE_SIZE];
		InternalNode* internalNodes = new InternalNode[SAMPLE_SIZE - 1];
		cudaMemcpy(leafNodes, d_leafNodes, SAMPLE_SIZE*sizeof(LeafNode), cudaMemcpyDeviceToHost);
		cudaMemcpy(internalNodes, d_internalNodes, (SAMPLE_SIZE-1)*sizeof(InternalNode), cudaMemcpyDeviceToHost);
		for (int i = 0; i < SAMPLE_SIZE; i++)
		cout << internalNodes[i].getIdx() << " " << internalNodes[i].getParent() << "\n";*/
		/////Assign bounding box to internal nodes
		int* atom;
		cudaMalloc((void **) &atom, SAMPLE_SIZE*sizeof(int));
		cudaMemset(atom, 0, SAMPLE_SIZE*sizeof(int));

		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);
		internalNodeBBox << <(SAMPLE_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(SAMPLE_SIZE, atom, d_internalNodes, d_leafNodes, d_myBBox);
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
		//cudaFree(d_internalNodes); cudaFree(d_leafNodes);
		BVHTree buf;
		buf.internalNodes = internalNodes;
		buf.leafNodes = leafNodes;
        myTree = buf;

        BVHTree d_buf;
        d_buf.internalNodes = d_internalNodes;
        d_buf.leafNodes = d_leafNodes;
        d_myTree = d_buf;
	}

public:
    ~CudaBVH() {
        checkCudaErrors(cudaFree(d_myTree.internalNodes));
        checkCudaErrors(cudaFree(d_myTree.leafNodes));
        checkCudaErrors(cudaFree(d_myMesh));
        checkCudaErrors(cudaFree(d_myBBox));
        cudaFree(mor);
    }

// 	CudaBVH()
// 	{
// 		SAMPLE_SIZE = 10;
// 		THREADS_PER_BLOCK = 5;
// 		BBox* dummy = new BBox[SAMPLE_SIZE];
// 		generateSampleDataset(dummy);
// 		CudaBVH(dummy, SAMPLE_SIZE, THREADS_PER_BLOCK);
// 		free(dummy);
// 	}
// 	CudaBVH(int sample_size, int threads_per_block)
// 	{
// 		SAMPLE_SIZE = sample_size;
// 		THREADS_PER_BLOCK = threads_per_block;
// 		BBox* dummy = new BBox[SAMPLE_SIZE];
// 		generateSampleDataset(dummy);
// 		Init(dummy, SAMPLE_SIZE, THREADS_PER_BLOCK);
//         delete[] dummy;
// 	}
	CudaBVH(BBox* objects, Triangle *tris, int sample_size, int threads_per_block)
	{
		SAMPLE_SIZE = sample_size;
		THREADS_PER_BLOCK = threads_per_block;
		Init(objects, tris, SAMPLE_SIZE, THREADS_PER_BLOCK);
	}
	void Init(BBox* objects, Triangle *tris, int sample_size, int threads_per_block)
	{
		SAMPLE_SIZE = sample_size;
		THREADS_PER_BLOCK = threads_per_block;
        cout << SAMPLE_SIZE << " " << THREADS_PER_BLOCK << endl;
		int *values;
		values = new int[SAMPLE_SIZE];

// 		generateValues(values);
//         for (int n = 0; n < SAMPLE_SIZE; n++) {
//             cout << "dfhgsd" << values[n] << endl;
//         }
        for (int n = 0; n < SAMPLE_SIZE; n++) {
            values[n] = n;
        }

		generateBVHTree(values, objects, tris);
		delete [] values;
	}

	void generateSampleDataset(BBox *objects)
	{
		float buf[6];
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

#if 0
	void printBVH(int idx, int level)
	{
		cout << "Internal (" << level << ") " << idx << " " << myTree.internalNodes[idx].getBBox().toString() << "\n";
		if (myTree.internalNodes[idx].getLeftNodeType() == LEAFNODE)
		{
			cout << "Leaf (l) " << myTree.leafNodes[myTree.internalNodes[idx].getLeftNodeIdx()].getObjectID() << " "
                << myTree.leafNodes[myTree.internalNodes[idx].getLeftNodeIdx()].getBBox().toString() << "\n";
		}
		else printBVH(myTree.internalNodes[idx].getLeftNodeIdx(), level + 1);

		if (myTree.internalNodes[idx].getRightNodeType() == LEAFNODE)
		{
			cout << "Leaf (r) " << myTree.leafNodes[myTree.internalNodes[idx].getRightNodeIdx()].getObjectID() << " "
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
//             int pid = myTree.internalNodes[idx].getParent();
//             cout << idx << ", " << level << ", " << stack_ptr << endl;
//             if (pid == -1) {
//                 cout << "Internal (" << level << ") " << idx << " " << myTree.internalNodes[idx].getBBox().toString() << endl;
//             }

            if (level > 100 /*idx > 38000*/) {
                cout << "===============================" << endl;
                cout << "node info {" << endl;
                cout << "  level: " << level << endl;
                cout << "  node idx: " << idx << ", " << myTree.internalNodes[idx].getIdx() << endl;
                cout << "  node type: " << myTree.internalNodes[idx].checkType() << endl;
                cout << "  node.parent idx: " << myTree.internalNodes[idx].getParent() << endl;
                cout << "  ndoe.parent type: " << myTree.internalNodes[idx].getParentType() << endl;
                cout << "  node.left idx: " << myTree.internalNodes[idx].getLeftNodeIdx() << endl;
                cout << "  node.left type: " << myTree.internalNodes[idx].getLeftNodeType() << endl;
                cout << "  node.right idx: " << myTree.internalNodes[idx].getRightNodeIdx() << endl;
                cout << "  node.right type: " << myTree.internalNodes[idx].getRightNodeType() << endl;
                cout << "}" << endl;
            }

            if (myTree.internalNodes[idx].getRightNodeType() == LEAFNODE) {
//                 if (myTree.leafNodes[myTree.internalNodes[idx].getRightNodeIdx()].getParent() == -1) {
//                     cout << "Leaf (r) " << myTree.leafNodes[myTree.internalNodes[idx].getRightNodeIdx()].getObjectID() << " " << myTree.leafNodes[myTree.internalNodes[idx].getRightNodeIdx()].getBBox().toString() << endl;
//                 }
            }
            else {
                visit_stack[stack_ptr] = myTree.internalNodes[idx].getRightNodeIdx();
                level_stack[stack_ptr] = level + 1;
                stack_ptr++;
            }
            if (myTree.internalNodes[idx].getLeftNodeType() == LEAFNODE) {
//                 if (myTree.leafNodes[myTree.internalNodes[idx].getLeftNodeIdx()].getParent() == -1) {
//                     cout << "Leaf (l) " << myTree.leafNodes[myTree.internalNodes[idx].getLeftNodeIdx()].getObjectID() << " " << myTree.leafNodes[myTree.internalNodes[idx].getLeftNodeIdx()].getBBox().toString() << endl;
//                 }
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

    void drawTriangles() {
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

            if (myTree.internalNodes[idx].getRightNodeType() == LEAFNODE) {
                myMesh[myTree.internalNodes[idx].getRightNodeIdx()].draw(color);
            }
            else {
                visit_stack[stack_ptr] = myTree.internalNodes[idx].getRightNodeIdx();
                level_stack[stack_ptr] = level + 1;
                stack_ptr++;
            }
            if (myTree.internalNodes[idx].getLeftNodeType() == LEAFNODE) {
                myMesh[myTree.internalNodes[idx].getLeftNodeIdx()].draw(color);
            }
            else {
                visit_stack[stack_ptr] = myTree.internalNodes[idx].getLeftNodeIdx();
                level_stack[stack_ptr] = level + 1;
                stack_ptr++;
            }
        }

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    void drawTrianglesDEBUG() {
//         glDisable(GL_CULL_FACE);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        int visit_stack[64] = { 0 };
        int level_stack[64] = { 0 };
        int stack_ptr = 1;

        while (stack_ptr > 0) {
            --stack_ptr;
            int idx = visit_stack[stack_ptr];
            int level = level_stack[stack_ptr];

            if (level > 64) return;
            float color[3] = { level * 0.03, 1 - level * 0.03, 0 };

//             myTree.internalNodes[idx].getBBox().draw(color);

            if (myTree.internalNodes[idx].getRightNodeType() == LEAFNODE) {
//                 myMesh[myTree.internalNodes[idx].getRightNodeIdx()].draw(color); // also works with large mesh
//                 myMesh[myTree.internalNodes[idx].getRightNodeIdx()].getBBox().draw(color);
//                 myBBox[myTree.internalNodes[idx].getRightNodeIdx()].draw(color);
//                 myTree.leafNodes[myTree.internalNodes[idx].getRightNodeIdx()].bbox_debug.draw(color);//=
//                 myTree.leafNodes[myTree.internalNodes[idx].getRightNodeIdx()].getBBox().draw(color);//-
                myMesh[myTree.leafNodes[myTree.internalNodes[idx].getRightNodeIdx()].getObjectID()].draw(color);//- only works with small mesh
            }
            else {
                visit_stack[stack_ptr] = myTree.internalNodes[idx].getRightNodeIdx();
                level_stack[stack_ptr] = level + 1;
                stack_ptr++;
            }
            if (myTree.internalNodes[idx].getLeftNodeType() == LEAFNODE) {
//                 myMesh[myTree.internalNodes[idx].getLeftNodeIdx()].draw(color); // also works with large mesh
//                 myMesh[myTree.internalNodes[idx].getLeftNodeIdx()].getBBox().draw(color);
//                 myBBox[myTree.internalNodes[idx].getLeftNodeIdx()].draw(color);
//                 myTree.leafNodes[myTree.internalNodes[idx].getLeftNodeIdx()].bbox_debug.draw(color);//=
//                 myTree.leafNodes[myTree.internalNodes[idx].getLeftNodeIdx()].getBBox().draw(color);//-
                myMesh[myTree.leafNodes[myTree.internalNodes[idx].getLeftNodeIdx()].getObjectID()].draw(color);//- only works with small mesh
            }
            else {
                visit_stack[stack_ptr] = myTree.internalNodes[idx].getLeftNodeIdx();
                level_stack[stack_ptr] = level + 1;
                stack_ptr++;
            }
        }

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    void batchIntersect(float3 *ray_orig, float3 *ray_dir, int ray_count,
        float *outT, float *outU, float *outV, int *outIdx, int *outHit) {
        intersectKernel<< <(ray_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >
            (ray_count, ray_orig, ray_dir, outT, outU, outV, outIdx, outHit, d_myTree.internalNodes, d_myTree.leafNodes, d_myMesh);

        // for debugging
//         for (int n = 0; n < ray_count; n++) {
//             outHit[n] = intersect(ray_orig[n], ray_dir[n], outT[n], outU[n], outV[n], outIdx[n]);
//         }
    }

    bool intersect(const float3& ray_orig, const float3& ray_dir, float& outT, float&outU, float& outV, int& outIdx) {
        float3 *d_rayorig;
        float3 *d_raydir;
        float *d_t;
        float *d_u;
        float *d_v;
        int *d_idx;
        int *d_hit;

        checkCudaErrors(cudaMallocManaged((void**)&d_rayorig, sizeof(float3)));
        checkCudaErrors(cudaMallocManaged((void**)&d_raydir, sizeof(float3)));
        checkCudaErrors(cudaMallocManaged((void**)&d_t, sizeof(float)));
        checkCudaErrors(cudaMallocManaged((void**)&d_u, sizeof(float)));
        checkCudaErrors(cudaMallocManaged((void**)&d_v, sizeof(float)));
        checkCudaErrors(cudaMallocManaged((void**)&d_idx, sizeof(int)));
        checkCudaErrors(cudaMallocManaged((void**)&d_hit, sizeof(int)));

        d_rayorig[0] = ray_orig;
        d_raydir[0] = ray_dir;

        checkCudaErrors(cudaDeviceSynchronize());
        intersectKernel << <1, 1 >> >(1, d_rayorig, d_raydir, d_t, d_u, d_v, d_idx, d_hit, d_myTree.internalNodes, d_myTree.leafNodes, d_myMesh);
        checkCudaErrors(cudaDeviceSynchronize());

        outT = d_t[0];
        outU = d_u[0];
        outV = d_v[0];
        outIdx = d_idx[0];
        int outHit = d_hit[0];

        checkCudaErrors(cudaFree(d_rayorig));
        checkCudaErrors(cudaFree(d_raydir));
        checkCudaErrors(cudaFree(d_t));
        checkCudaErrors(cudaFree(d_u));
        checkCudaErrors(cudaFree(d_v));
        checkCudaErrors(cudaFree(d_idx));
        checkCudaErrors(cudaFree(d_hit));

        return outHit;
    }
};