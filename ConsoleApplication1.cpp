#include "stdafx.h"
#include "BasicAAModel.h"
#include "Utils.h"

using namespace std;

int main(int argc, char* argv[]) {

	// TRAINING PART --- Stores the model in AMF file. This file is used for fitting purposes thereafter

	/* int type = 0;
	int level = 3;	// Pyrmaidal levels for training and testing

	inputFiles inputImages = Commons::parseInputDir("./helen/smallTrain", "jpg");
	inputFiles inputPoints = Commons::parseInputDir("./helen/smallTrain", "pts");

	if (inputPoints.size() != inputImages.size()) {
		printf("number of shapes is not equal to number of points");
		exit(0);
	}

	FaceDetector fDetect;
	fDetect.loadXML("./haarcascade_frontalface_alt2.xml");
	PyramidModel model;
	model.Build(inputPoints, inputImages, type, level);
	model.BuildDetectMapping(inputPoints, inputImages, fDetect);
	model.WriteModel("basic.amf"); */	// Change this to ic.amf -- use 2 pyramidal levels for that


	// FITTING PART -- comment out the training part for fitting. 

	inputFiles inputImages = Commons::parseInputDir("./TestImages/Test", "jpg");
	ShapeModel shapeModel;
	PyramidModel pyramidModel;
	pyramidModel.ReadModel("basicNew200.amf");
	FaceDetector facedetector;
	facedetector.loadXML("./haarcascade_frontalface_alt2.xml");

	for (int i = 0; i < inputImages.size(); i++) {
		IplImage* image = cvLoadImage(inputImages[i].c_str(), -1);
		//IplImage* image = cvLoadImage("./TestImages/15.jpg", -1);
		bool flag = flag = pyramidModel.initializeShapeFromImage(shapeModel, facedetector, image);
		if (flag == false) {
			fprintf(stderr, "Stop messing with the system !! The image doesn't contain any faces\n");
			exit(0);
		}
		pyramidModel.Fit(image, shapeModel, 30, true);
		pyramidModel.Draw(image, shapeModel, 2);
		cvNamedWindow("Fitting");
		cvShowImage("Fitting", image);
		cvWaitKey(0);
		cvReleaseImage(&image);

	}
	return 0;
}