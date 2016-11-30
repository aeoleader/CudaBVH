////////////////////////////////////////////////////////////////////////////////////////////////////
// OBJCORE: A Simple Obj Library
// by Yining Karl Li
//
// obj.h
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef OBJ
#define OBJ

#include "mymath.h"
#include <string>
#include <vector>

using namespace std;

class obj{
private:
	vector<Vector4> points;
	vector<vector<int> > faces; 
	vector<vector<int> > facenormals; 
	vector<vector<int> > facetextures; 
    vector<float*> faceboxes;   //bounding boxes for each face are stored in vbo-format!
	vector<Vector4> normals;
	vector<Vector4> texturecoords;
	int vbosize;
	int nbosize;
	int cbosize;
	int ibosize;
	float* vbo;
	float* nbo;
	float* cbo;
	unsigned short* ibo;
	float* boundingbox;
	float top;
	Vector3 defaultColor;
	bool maxminSet;
	float xmax; float xmin; float ymax; float ymin; float zmax; float zmin; 
public:
	obj();
	~obj();  

	//-------------------------------
	//-------Mesh Operations---------
	//-------------------------------
	void buildVBOs();
	void addPoint(Vector3);
	void addFace(vector<int>);
	void addNormal(Vector3);
	void addTextureCoord(Vector3);
	void addFaceNormal(vector<int>);
	void addFaceTexture(vector<int>);
	void compareMaxMin(float, float, float);
	bool isConvex(vector<int>);
	void recenter();

	//-------------------------------
	//-------Get/Set Operations------
	//-------------------------------
	float* getBoundingBox();    //returns vbo-formatted bounding box
	float getTop();
	void setColor(Vector3);
	Vector3 getColor();
	float* getVBO();
	float* getCBO();
	float* getNBO();
	unsigned short* getIBO();
	int getVBOsize();
	int getNBOsize();
	int getIBOsize();
	int getCBOsize();
    vector<Vector4>* getPoints();
	vector<vector<int> >* getFaces(); 
	vector<vector<int> >* getFaceNormals(); 
	vector<vector<int> >* getFaceTextures(); 
	vector<Vector4>* getNormals();
	vector<Vector4>* getTextureCoords();
    vector<float*>* getFaceBoxes();
	int GetNumTris();
	Vector3 getMin();
	Vector3 getMax();
};

#endif