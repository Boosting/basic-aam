#include "stdafx.h"
#include "FaceDetector.h"
#include "Utils.h"

using namespace std;

FaceDetector::FaceDetector() {
	initCascade = 0;
	initStorage = 0;
}

FaceDetector::~FaceDetector() {
	cvReleaseMemStorage(&initStorage);
	cvReleaseHaarClassifierCascade(&initCascade);
}

bool FaceDetector::loadXML(const char* inputXML) {
	if (initStorage)
		cvReleaseMemStorage(&initStorage);
	if (initCascade)
		cvReleaseHaarClassifierCascade(&initCascade);

	initCascade = (CvHaarClassifierCascade*)cvLoad(inputXML, 0, 0, 0);
	if (initCascade == 0) {
		LOGW("ERROR(%s, %d): Can't load XML file!\n", __FILE__, __LINE__);
		return false;
	}
	initStorage = cvCreateMemStorage(0);
	LOGD("Successfully loaded Haar Cascade XML file ... \n");
	return true;
}

bool FaceDetector::DetectFace(std::vector<ShapeModel> &Shape, const IplImage* image) {
	IplImage* small_image = cvCreateImage(cvSize(image->width / 2, image->height / 2), image->depth, image->nChannels);
	cvPyrDown(image, small_image, CV_GAUSSIAN_5x5);

	CvSeq* pFaces = cvHaarDetectObjects(small_image, initCascade, initStorage,
		1.1, 3, CV_HAAR_DO_CANNY_PRUNING);

	cvReleaseImage(&small_image);

	if (0 == pFaces->total)//can't find a face
		return false;

	Shape.resize(pFaces->total);
	for (int i = 0; i < pFaces->total; i++)
	{
		Shape[i].resize(2);
		CvRect* r = (CvRect*)cvGetSeqElem(pFaces, i);

		CvPoint pt1, pt2;
		pt1.x = r->x * 2;
		pt2.x = (r->x + r->width) * 2;
		pt1.y = r->y * 2;
		pt2.y = (r->y + r->height) * 2;

		Shape[i][0].x = r->x*2.0;
		Shape[i][0].y = r->y*2.0;
		Shape[i][1].x = Shape[i][0].x + 2.0*r->width;
		Shape[i][1].y = Shape[i][0].y + 2.0*r->height;
	}
	return true;
}
