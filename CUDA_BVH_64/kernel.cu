#include <GL/freeglut.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include "CudaBVH.cuh"
#include "objloader.h"

using namespace std;
CudaBVH* myBVH = nullptr;

struct Timer{
    LARGE_INTEGER _begin, _end, _freq;
    Timer() {
        QueryPerformanceFrequency(&_freq);
    }
    void tick() {
        QueryPerformanceCounter(&_begin);
    }
    void tock() {
        QueryPerformanceCounter(&_end);
    }
    float interval() {
        return (_end.QuadPart - _begin.QuadPart) * 1e-6f;
    }
};
Timer timer;

struct Ray {
    Vector3 orig;
    Vector3 dir;
};
vector<Ray> rays;

float3 *d_rayorig;
float3 *d_raydir;
float *d_t;
float *d_u;
float *d_v;
int *d_idx;
int *d_hit;
const int raycount = 1 << 10;

float rot = 0;
void display_cb() {
    glClearColor(1, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glRotatef(rot, 0, 1, 0);
    glTranslatef(-0.5, -0.5, -0.5);

    myBVH->draw();
    myBVH->drawTrianglesDEBUG();
    //myBVH->drawTriangles();
    rot += 1;

#if 1
    ///
    // #pragma omp parallel for schedule(dynamic)
    for (int n = 0; n < raycount; n++) {
        Ray ray;
//         Vector3 v1 = sampleSphere(randf(), randf()) * 0.5f + Vector3(0.5, 0.5, 0.5);
//         Vector3 v2 = sampleSphere(randf(), randf()) * 0.5f + Vector3(0.5, 0.5, 0.5);
        Vector3 v1 = sampleSphere(randf(), randf()) * 1.0f + Vector3(0.5, 0.5, 0.5);
        Vector3 v2 = sampleSphere(randf(), randf()) * 1.0f + Vector3(0.5, 0.5, 0.5);
        ray.orig = v1;
        ray.dir = Normalize(v2 - v1);

        d_rayorig[n] = make_float3(ray.orig.x, ray.orig.y, ray.orig.z);
        d_raydir[n] = make_float3(ray.dir.x, ray.dir.y, ray.dir.z);
    }

    timer.tick();
    checkCudaErrors(cudaDeviceSynchronize());
    myBVH->batchIntersect(d_rayorig, d_raydir, raycount, d_t, d_u, d_v, d_idx, d_hit);
    checkCudaErrors(cudaDeviceSynchronize());
    timer.tock();
    cout << float(raycount) * 1e-6f / timer.interval() << " M rays / s" << endl;

    int hit = 0;
    rays.clear();
    rays.resize(raycount);
    for (int n = 0; n < raycount; n++) {
        if (d_hit[n]) {
            Ray tr;
            tr.orig = Vector3(d_rayorig[n].x, d_rayorig[n].y, d_rayorig[n].z);
            tr.dir = Vector3(d_rayorig[n].x, d_rayorig[n].y, d_rayorig[n].z) + Vector3(d_raydir[n].x, d_raydir[n].y, d_raydir[n].z) * d_t[n];
            rays[hit++] = tr;
        }
    }
#else
    ///
    int hit = 0;
    rays.clear();
    rays.resize(raycount);

    timer.tick();
    for (int n = 0; n < raycount; n++) {
        Ray ray;
        Vector3 v1 = sampleSphere(randf(), randf()) * 0.5f + Vector3(0.5, 0.5, 0.5);
        Vector3 v2 = sampleSphere(randf(), randf()) * 0.5f + Vector3(0.5, 0.5, 0.5);
        ray.orig = v1;
        ray.dir = Normalize(v2 - v1);

        float t, u, v;
        int idx;
        if (myBVH->intersect(make_float3(ray.orig.x, ray.orig.y, ray.orig.z), make_float3(ray.dir.x, ray.dir.y, ray.dir.z), t, u, v, idx)) 
        {
            Ray tr;
            tr.orig = ray.orig;
            tr.dir = ray.orig + ray.dir * t;
            rays[hit++] = tr;
        }
    }
    timer.tock();

    cout << float(raycount) * 1e-6f / timer.interval() << " M rays / s" << endl;
#endif


    cout << hit << " out of " << rays.size() << " rays" << endl;

    glPointSize(5);
    for (int n = 0; n < hit; n++) {
        auto v = rays[n];
        glBegin(GL_LINES);
        glColor3f(0, 0, 1);
        glVertex3fv(vRaw(v.orig));
        glVertex3fv(vRaw(v.dir));
        glEnd();
     
        glBegin(GL_POINTS);
        glColor3f(1, 0, 0);
        glVertex3fv(vRaw(v.orig));
        glColor3f(0, 1, 0);
        glVertex3fv(vRaw(v.dir));
        glEnd();
    }

    glPopMatrix();


//     bvh->DebugDraw();
// 
//     rays.clear();
//     rays.resize(10000);
//     timer.tick();
//     // #pragma omp parallel for schedule(dynamic)
//     for (int n = 0; n < rays.size(); n++) {
//         Ray ray;
//         Vector3 v1 = sampleSphere(nextFloat(), nextFloat());
//         Vector3 v2 = sampleSphere(nextFloat(), nextFloat());
//         ray.orig = v1;
//         ray.dir = Normalize(v2 - v1);
//         float t, u, v, w, sgn;
//         uint32_t idx;
//         if (bvh->TraceRay(ray.orig, ray.dir, t, u, v, w, sgn, idx)) {
//             Ray tr;
//             tr.orig = ray.orig;
//             tr.dir = ray.orig + ray.dir * t;
//             rays[n] = tr;
//         }
//     }
//     timer.tock();
//     cout << float(rays.size()) * 1e-6f / timer.interval() << " M rays / s" << endl;
// 
//     //     cout << rays.size() << endl;
//     glBegin(GL_LINES);
//     glColor3f(0, 0, 1);
//     for (auto v : rays) {
//         glVertex3fv(Raw(v.orig));
//         glVertex3fv(Raw(v.dir));
//     }
//     glEnd();

    glutSwapBuffers();
    glutPostRedisplay();
}

void keyboard_cb(unsigned char key, int x, int y) {
    if (key == 'q') exit(0);
}

///
int sample_count = 0;
const int blockSize = 128;

int main(int argc, char **argv)
{
#if 1
    obj *mesh = new obj;
    objLoader obj("feline.obj", mesh);

    auto faces = mesh->getFaces();
    auto verts = mesh->getPoints();

    vector<Vector3> my_verts;
    for (auto v : *verts) {
        my_verts.push_back(Vector3(v.x, v.y, v.z));
    }

    vector<BBox> aabbs;
    vector<Triangle> tris;

    for (auto f : *faces) {
        if (f.size() == 3) {
            Vector3 va = my_verts[f[0]];
            Vector3 vb = my_verts[f[1]];
            Vector3 vc = my_verts[f[2]];

            BBox b;
            b.xmin = fmin(fmin(va.x, vb.x), vc.x);
            b.xmax = fmax(fmax(va.x, vb.x), vc.x);
            b.ymin = fmin(fmin(va.y, vb.y), vc.y);
            b.ymax = fmax(fmax(va.y, vb.y), vc.y);
            b.zmin = fmin(fmin(va.z, vb.z), vc.z);
            b.zmax = fmax(fmax(va.z, vb.z), vc.z);
            aabbs.push_back(b);

            Triangle t;
            t.a = make_float3(va.x, va.y, va.z);
            t.b = make_float3(vb.x, vb.y, vb.z);
            t.c = make_float3(vc.x, vc.y, vc.z);
            tris.push_back(t);
        }
    }
    sample_count = aabbs.size();

    // must be bounded to unit cube
    float bounds[6] = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
    for (auto& b : aabbs) {
        bounds[0] = fmin(bounds[0], b.xmin);
        bounds[1] = fmax(bounds[1], b.xmax);
        bounds[2] = fmin(bounds[2], b.ymin);
        bounds[3] = fmax(bounds[3], b.ymax);
        bounds[4] = fmin(bounds[4], b.zmin);
        bounds[5] = fmax(bounds[5], b.zmax);
    }

//     float _scale = fmin(fmin(1.0f / (bounds[1], - bounds[0]), 1.0f / (bounds[3] - bounds[2])), 1.0f / (bounds[5] - bounds[4]));
//     for (auto& b : aabbs) {
//         b.xmin = fmax(0.01, fmin(0.99, (b.xmin - bounds[0]) * _scale));
//         b.xmax = fmax(0.01, fmin(0.99, (b.xmax - bounds[0]) * _scale));
//         b.ymin = fmax(0.01, fmin(0.99, (b.ymin - bounds[2]) * _scale));
//         b.ymax = fmax(0.01, fmin(0.99, (b.ymax - bounds[2]) * _scale));
//         b.zmin = fmax(0.01, fmin(0.99, (b.zmin - bounds[4]) * _scale));
//         b.zmax = fmax(0.01, fmin(0.99, (b.zmax - bounds[4]) * _scale));
//     }

//     for (auto b : aabbs) {
//         cout << b.toString() << endl;
//     }
#else
    vector<BBox> aabbs(sample_count);
    float buf[6];
    for (int i = 0; i < sample_count; i++)
    {
        for (int j = 0; j < 6; j++)
            buf[j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        aabbs[i].xmax = max(buf[0], buf[1]);
        aabbs[i].xmin = min(buf[0], buf[1]);
        aabbs[i].ymax = max(buf[2], buf[3]);
        aabbs[i].ymin = min(buf[2], buf[3]);
        aabbs[i].zmax = max(buf[4], buf[5]);
        aabbs[i].zmin = min(buf[4], buf[5]);
    }
#endif

    ///

	OutputDebugStringA(std::to_string(sample_count).c_str());
	myBVH = new CudaBVH(&aabbs[0], &tris[0], sample_count, blockSize);
    system("pause");
    BVHTree myTree = myBVH->myTree;
    for (int n = 38040; n < 38050; n++) {
//         cout << "   aabb " << n << " : " << aabbs[myTree.leafNodes[n].getObjectID()].toString() << endl;

        int idx = n;
        MortonRec m = myBVH->mor[myTree.leafNodes[idx].getObjectID()];
        printf("idx: %d  x: %f\n", idx, m.x);
        printf("idx: %d  y: %f\n", idx, m.y);
        printf("idx: %d  z: %f\n", idx, m.z);

        printf("idx: %d  xx: %f\n", idx, m.xx);
        printf("idx: %d  yy: %f\n", idx, m.yy);
        printf("idx: %d  zz: %f\n", idx, m.zz);

        printf("idx: %d  expand x: %lld\n", idx, m.ex);
        printf("idx: %d  expand y: %lld\n", idx, m.ey);
        printf("idx: %d  expand z: %lld\n", idx, m.ez);
        printf("idx: %d  hash: %lld\n", idx, m.m);
        printf("\n");
    }
    system("pause");
	myBVH->printBVH(0, 0);
    system("pause");

    checkCudaErrors(cudaMallocManaged((void**)&d_rayorig, raycount * sizeof(float3)));
    checkCudaErrors(cudaMallocManaged((void**)&d_raydir, raycount * sizeof(float3)));
    checkCudaErrors(cudaMallocManaged((void**)&d_t, raycount * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void**)&d_u, raycount * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void**)&d_v, raycount * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void**)&d_idx, raycount * sizeof(int)));
    checkCudaErrors(cudaMallocManaged((void**)&d_hit, raycount * sizeof(int)));

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(900, 900);
    glutCreateWindow("LBVH");
    glEnable(GL_DEPTH_TEST);
//     gluOrtho2D(-2, 2, -2, 2);
    //gluOrtho2D(0, 1, 0, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
//     gluLookAt(0, 0, 1.5, 0, 0, 0, 0, 1, 0);
    gluLookAt(0, 0, 2.5, 0, 0, 0, 0, 1, 0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45, 1, 0.1, 100);
    glutDisplayFunc(display_cb);
    glutKeyboardFunc(keyboard_cb);
    glutMainLoop();

    delete (myBVH);
    checkCudaErrors(cudaFree(d_rayorig));
    checkCudaErrors(cudaFree(d_raydir));
    checkCudaErrors(cudaFree(d_t));
    checkCudaErrors(cudaFree(d_u));
    checkCudaErrors(cudaFree(d_v));
    checkCudaErrors(cudaFree(d_idx));
    checkCudaErrors(cudaFree(d_hit));
    checkCudaErrors(cudaDeviceReset());
	return 0;
}