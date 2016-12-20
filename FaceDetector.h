#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "ShapeModel.h"

class FaceDetector {
public:
	FaceDetector();
	~FaceDetector();

	// Face detection/estimation for an initial shape
	bool DetectFace(std::vector<ShapeModel> &Shape, const IplImage* image);

	// Loading adaboost XML file for face detection
	bool loadXML(const char* cascade_name);

private:
	CvMemStorage* initStorage;
	CvHaarClassifierCascade* initCascade;
};

#endif

