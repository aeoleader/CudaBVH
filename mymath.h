#ifndef math_h__
#define math_h__

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

typedef glm::vec2 Vector2;
typedef glm::vec3 Vector3;
typedef glm::vec4 Vector4;
typedef glm::mat4x4 Matrix4x4;
typedef glm::mat3x3 Matrix3x3;

#define M_PI 3.14159265358979323846

#define Min(a, b) glm::min(a, b)
#define Max(a, b) glm::max(a, b)
#define Normalize(x) glm::normalize(x)
#define Cross(a, b) glm::cross(a, b)
#define Dot(a, b) glm::dot(a, b)
#define Raw(x) glm::value_ptr(x)

float randf();

Vector3 sampleSphere(float r1, float r2);

#endif // math_h__
