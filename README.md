# CudaBVH
This is an implementation of LBVH on CUDA following the detailed explanation offered by Tero Karras from Nvidia https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/. 
This implemenation of LBVH is fully parrallel both in construction and collision pairs finding. The program is divided into two parts. The first part is tree construction and the second part is tree traversal.
For tree construction, the workflow is the following. 
1. Convert AABBs to morton codes (in the implementation AABBs are BBoxs). The first implementation used 32 bit uint to represent hash codes of AABBs while this version improved the hash code using 64 bit uint.
2. Sort converted morton codes using Radix sort. This implemenation utilizes the radix sort from CUB package, which is a package of algorithms implemented to fully harness the power of our GPU.
3. Create tree hierarchy. Create all leaf nodes first and then create all internal nodes in parallel. The detailed explanation can be found from the link above. Tero explained the algorithm pretty well. If you want to know more about his algorithm, you can go to his paper which can be found in the link above.
4. Assign BBox to each internal node. This process uses the power of atomic operation in CUDA. 

The second step is tree traversal.

Detailed implementation details can be found in my code and in Tero's paper.

Finally, thanks to Yulong Guo for his help in patiently giving me advices about CUDA (I had no previous knowledge of CUDA programming before this project) and testing my code after my first implementation. He also helped me find many errors and bugs of my implementation.
