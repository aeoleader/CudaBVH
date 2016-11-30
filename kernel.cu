#include <GL/freeglut.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "CudaBVH.cuh"
#include "objloader.h"

using namespace std;
CudaBVH* myBVH = nullptr;

float rot = 0;
void display_cb() {
    glClearColor(1, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glRotatef(rot, 0, 1, 0);
    glTranslatef(-0.5, -0.5, -0.5);
    myBVH->draw();
    glPopMatrix();
    rot += 1;

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

int sample_count = 20;
const int blockSize = 1;

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
			//float* coodinates = new float[]{ va.x, va.y, va.z, vb.x, vb.y, vb.z, vc.x, vc.y, vc.z };
			//b.tri_coodinate = new float[9]{ va.x, va.y, va.z, vb.x, vb.y, vb.z, vc.x, vc.y, vc.z };
			b.x00 = va.x;
			b.x01 = va.y;
			b.x02 = va.z;
			b.x10 = vb.x;
			b.x11 = vb.y;
			b.x12 = vb.z;
			b.x20 = vc.x;
			b.x21 = vc.y;
			b.x22 = vc.z;
			
            aabbs.push_back(b);
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
	myBVH = new CudaBVH(&aabbs[0], sample_count, blockSize);
// 	myBVH->printBVH(0, 0);

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
    gluLookAt(0, 0, 1.5, 0, 0, 0, 0, 1, 0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45, 1, 0.1, 100);
    glutDisplayFunc(display_cb);
    glutKeyboardFunc(keyboard_cb);
	//myBVH->printCollisionList();
	//myBVH->printBVH(0, 1);
    glutMainLoop();
	
    free(myBVH);
    checkCudaErrors(cudaDeviceReset());
	return 0;
}