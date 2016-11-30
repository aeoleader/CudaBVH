#include <random>
#include "mymath.h"
using namespace std;
default_random_engine rng;
uniform_real_distribution<float> dist(0.0f, 1.0f);
float randf() {
    return dist(rng);
}

Vector3 sampleSphere(float r1, float r2)
{
    float cos_theta = 2.0f * r1 - 1.0f;
    float phi = 2.0f * M_PI * r2;
    float r = sqrtf(1.0f - cos_theta * cos_theta);
    return Vector3(r * cos(phi), cos_theta, r * sin(phi));
}
