#ifndef PIECEWISEAFFINEWARP_AAMODEL_H
#define PIECEWISEAFFINEWARP_AAMODEL_H

#include "stdafx.h"
#include "Utils.h"
#include "ShapeModel.h"
#include "opencv2\imgproc.hpp"

class BasicAAModel;
class CombinedAAModel;

class piecewiseAffineWarpAAModel {
	friend class BasicAAModel;
	friend class CombinedAAModel;
	friend class Commons;
public:
	piecewiseAffineWarpAAModel();
	~piecewiseAffineWarpAAModel();

	void trainModel(const ShapeModel& ReferenceShape, CvMat* Points, CvMemStorage* Storage, const std::vector<std::vector<int> >* tri = 0,
		bool buildVtri = true);

	void readStream(std::ifstream& is);
	void writeStream(std::ofstream& os);
	inline const int nPoints()const { return __pointsCount; }
	inline const int nPix()const { return __nPixels; }
	inline const int nTri()const { return __trianglesCount; }

	inline const CvPoint2D32f Vertex(int i)const { return __referenceshape[i]; }
	inline const std::vector<std::vector<int> >* GetTri()const { return &__triangles; }
	inline const int Tri(int i, int j)const { return __triangles[i][j]; }
	inline const int PixTri(int i)const { return __pixTri[i]; }
	inline const double Alpha(int i)const { return __alpha[i]; }
	inline const double Belta(int i)const { return __belta[i]; }
	inline const double Gamma(int i)const { return __gamma[i]; }
	inline const int Width()const { return __width; }
	inline const int Height()const { return __height; }
	inline const int Rect(int i, int j)const { return __rect[i][j]; }
	void computeWarpedTexture(const CvMat* s, const IplImage* image, CvMat* t)const;
	void convertTextureToImage(IplImage* image, const CvMat* t)const;
	void storeWarpedTextureAsImage(const char* filename, const CvMat* t)const;
	static void computeWarpedParams(double x, double y, double x1, double y1, double x2, double y2, double x3, double y3,
		double &alpha, double &belta, double &gamma);
	static void Warp(double x, double y, double x1, double y1, double x2, double y2, double x3, double y3,
		double& X, double& Y, double X1, double Y1, double X2, double Y2, double X3, double Y3);

private:
	void Delaunay(const CvSubdiv2D* Subdiv, const CvMat *ConvexHull);
	void computePixelPoint(const CvRect rect, CvMat *ConvexHull);
	void FastCalcPixelPoint(const CvRect rect);
	static bool IsEdgeIn(int ind1, int ind2, const std::vector<std::vector<int> > &edges);
	static bool IsTriangleNotIn(const std::vector<int>& one_tri, const std::vector<std::vector<int> > &tris);
	void FindVTri();
	int FastFillConvexPoly(CvPoint2D32f pts[3], void* data);

private:
	int __pointsCount;
	int __nPixels;
	int __trianglesCount;
	int __width, __height, __xmin, __ymin; //warped area

	std::vector<std::vector<int> > __triangles;	// vertices of triangles
	std::vector<std::vector<int> > __vtri;
	std::vector<int> __pixTri;
	std::vector<double> __alpha, __belta, __gamma;
	std::vector<std::vector<int> > __rect;

	ShapeModel __referenceshape;
};

#endif
