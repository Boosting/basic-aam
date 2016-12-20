#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include "opencv/cv.h"
#include "opencv/highgui.h"

#include "ShapeModel.h"
#include "FaceDetector.h"

#ifndef byte
#define byte unsigned char
#endif

#define gettime cvGetTickCount() / (cvGetTickFrequency()*1000.)
#define LOGD(...) fprintf(stdout,  __VA_ARGS__)
#define LOGI(...) fprintf(stdout,  __VA_ARGS__)
#define LOGW(...) fprintf(stderr,  __VA_ARGS__)

typedef std::vector<std::string> inputFiles;
void ReadCvMat(std::istream &is, CvMat* mat);
void WriteCvMat(std::ostream &os, const CvMat* mat);

class piecewiseAffineWarpAAModel;

class Commons {
public:
	static inputFiles parseInputDir(const std::string &dirPath, const std::string &fileExt);
	static void checkIfShapeInBoundary(CvMat* s, int w, int h);
	static void DrawPoints(IplImage* image, const ShapeModel& Shape);
	static void DrawPointsInAlignedShape(IplImage* image, const ShapeModel& Shape);
	static void DrawTriangles(IplImage* image, const ShapeModel& Shape, const std::vector<std::vector<int> >&tris);
	static void DrawAppearance(IplImage*image, const ShapeModel& Shape,
		const CvMat* t, const piecewiseAffineWarpAAModel& paw, const piecewiseAffineWarpAAModel& refpaw);
	static int createDirectory(const char* dirname);
};

class AAModel {
public:
	AAModel();
	virtual ~AAModel() = 0;
	virtual const int GetType()const = 0;
	virtual void Build(const inputFiles& inputPts, const inputFiles& inputImages, double scale = 1.0) = 0;
	virtual bool Fit(const IplImage* image, ShapeModel& Shape, int max_iter = 30, bool showprocess = false) = 0;
	virtual void SetAllParamsZero() = 0;
	virtual void initializeParameteres(const IplImage* image) = 0;
	virtual void Draw(IplImage* image, const ShapeModel& Shape, int type) = 0;
	virtual void readStream(std::ifstream& is) = 0;
	virtual void writeStream(std::ofstream& os) = 0;
	virtual const ShapeModel GetMeanShape()const = 0;
	virtual const ShapeModel GetReferenceShape()const = 0;
};

class PyramidModel {
public:
	PyramidModel();
	~PyramidModel();

	void Build(const inputFiles& inputPts, const inputFiles& inputImages, int type, int level);
	bool Fit(const IplImage* image, ShapeModel& Shape, int max_iter = 30, bool showprocess = false);
	void BuildDetectMapping(const inputFiles& inputPts, const inputFiles& inputImages, FaceDetector& facedetect,
		double refWidth = 100);
	bool initializeShapeFromImage(ShapeModel& Shape, FaceDetector& facedetect, const IplImage* image);
	void initializeShapeFromDetectionBox(ShapeModel &Shape, const ShapeModel& detectionBox);

	bool WriteModel(const std::string& filename);
	bool ReadModel(const std::string& filename);
	void Draw(IplImage* image, const ShapeModel& Shape, int type);
	const ShapeModel GetMeanShape()const;

private:
	std::vector<AAModel*> __model;
	ShapeModel ShapeDetector;
	double __referenceWidth;
};

#endif