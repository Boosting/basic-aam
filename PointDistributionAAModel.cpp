#include "stdafx.h"
#include "PointDistributionAAModel.h"
using namespace std;

PointDistributionAAModel::PointDistributionAAModel() {
	__MeanShape = 0;
	__ShapesEigenVectors = 0;
	__ShapesEigenValues = 0;
	__matshape = 0;
}

PointDistributionAAModel::~PointDistributionAAModel() {
	cvReleaseMat(&__MeanShape);
	cvReleaseMat(&__ShapesEigenVectors);
	cvReleaseMat(&__ShapesEigenValues);
	cvReleaseMat(&__matshape);
}

void PointDistributionAAModel::trainModel(const std::vector<ShapeModel> &completeShapeList, double scale, double shapeEstimate) {
	int sampleCount = completeShapeList.size();
	int pointsCount = completeShapeList[0].getPointsCount();
	__matshape = cvCreateMat(1, pointsCount * 2, CV_64FC1);

	std::vector<ShapeModel> alignedShapesList = completeShapeList;
	for (int s = 0; s < alignedShapesList.size(); s++)
		alignedShapesList[s] *= scale;

	PointDistributionAAModel::alignAllShapes(alignedShapesList);

	CvMat *alignedCVShapes = cvCreateMat(sampleCount, pointsCount*2, CV_64FC1);
	for (int i = 0; i < sampleCount; i++) {
		for (int j = 0; j < pointsCount; j++) {
			CV_MAT_ELEM(*alignedCVShapes, double, i, 2*j) = alignedShapesList[i][j].x;
			CV_MAT_ELEM(*alignedCVShapes, double, i, 2*j+1) = alignedShapesList[i][j].y;
		}
	}

	// All centered shape data on top of each --- result 2
	// Superimposing all scaled landmark points on one image
	/* IplImage* imgScribble = cvCreateImage(cvSize(1000, 1000), 8, 3);
	cvSet(imgScribble, cvScalar(0, 0, 0));
	for (std::vector<ShapeModel>::iterator iter = alignedShapesList.begin(); iter != alignedShapesList.end(); ++iter) {
		cvNamedWindow("tempWin", CV_WINDOW_AUTOSIZE);
		Commons::DrawPointsInAlignedShape(imgScribble, *iter);
		cvShowImage("tempWin", imgScribble);
		cvWaitKey(0);
	}
	cvReleaseImage(&imgScribble); */

	// End Result

	computePCA(alignedCVShapes, shapeEstimate);
	__AAMRefShape.Mat2Point(__MeanShape);
	__AAMRefShape.translateShape(-__AAMRefShape.MinX(), -__AAMRefShape.MinY());
	cvReleaseMat(&alignedCVShapes);
}

void PointDistributionAAModel::alignAllShapes(std::vector<ShapeModel> &completeShapeList) {
	LOGD("Aligning all shapes...\n");
	int sampleCount = completeShapeList.size();
	int pointsCount = completeShapeList[0].getPointsCount();
	for (int j = 0; j < sampleCount; j++) {
		completeShapeList[j].centerShape();	//Center each shape to its origin
	}
	ShapeModel meanShape;
	PointDistributionAAModel::computeMeanShape(meanShape, completeShapeList);
	ShapeModel referenceShape(meanShape);
	const ShapeModel constReferenceShape(meanShape);
	ShapeModel updatedMeanShape(meanShape);

	// Generalized Procrustes Analysis Algorithms
	// do a number of alignment iterations until the mean shape estimate is stable
	double diff, diff_max = 0.0001;
	const int maxIter = 30;	// Max iterations till we converge
	for (int iter=0; iter<maxIter; iter++) {
		//align all shapes to the mean shape estimate
		for (int i = 0; i < sampleCount; i++) {
			completeShapeList[i].alignShapeTo(referenceShape);
		}
		// Re-estimate new mean shape from aligned shapes
		PointDistributionAAModel::computeMeanShape(updatedMeanShape, completeShapeList);
		// Constrain new mean shape by aligning it to ref shape
		updatedMeanShape.alignShapeTo(constReferenceShape);
		diff = (updatedMeanShape - referenceShape).GetNorm2();
		LOGD("Shape Alignment Iteration #%i, estimate difference = %g\n", iter, diff);

		if (diff <= diff_max) break;
		referenceShape = updatedMeanShape;
	}
	PointDistributionAAModel::computeMeanShape(meanShape, completeShapeList);
}

void PointDistributionAAModel::computeMeanShape(ShapeModel &meanShape, const std::vector<ShapeModel> &completeShapeList) {
	meanShape.resize(completeShapeList[0].getPointsCount());
	meanShape = 0;
	for (int i = 0; i < completeShapeList.size(); i++) {
		meanShape += completeShapeList[i];
	}
	meanShape /= completeShapeList.size();
}

void PointDistributionAAModel::computePCA(const CvMat* completeShapeList, double percentage) {
	LOGD("Performing PCA for all shape data ...");

	int sampleCount = completeShapeList->rows;
	int nPointsby2 = completeShapeList->cols;
	int nEigenAtMost = MIN(sampleCount, nPointsby2);

	CvMat* tempEigenVals = cvCreateMat(1, nEigenAtMost, CV_64FC1);
	CvMat* tempEigenVectors = cvCreateMat(nEigenAtMost, nPointsby2, CV_64FC1);
	__MeanShape = cvCreateMat(1, nPointsby2, CV_64FC1);
	cvCalcPCA(completeShapeList, __MeanShape, tempEigenVals, tempEigenVectors, CV_PCA_DATA_AS_ROW);
	double allSum = cvSum(tempEigenVals).val[0];
	double partSum = 0.0;
	int nTruncated = 0;
	double biggestEigenVal = cvmGet(tempEigenVals, 0, 0);
	for (int i = 0; i < nEigenAtMost; i++) {
		double eigenVal = cvmGet(tempEigenVals, 0, i);
		if (eigenVal / biggestEigenVal < 0.0001) break;
		partSum += eigenVal;
		++nTruncated;
		if (partSum / allSum >= percentage)	break;
	}

	__ShapesEigenValues = cvCreateMat(1, nTruncated, CV_64FC1);
	__ShapesEigenVectors = cvCreateMat(nTruncated, nPointsby2, CV_64FC1);

	CvMat G;
	cvGetCols(tempEigenVals, &G, 0, nTruncated);
	cvCopy(&G, __ShapesEigenValues);
	cvGetRows(tempEigenVectors, &G, 0, nTruncated);
	cvCopy(&G, __ShapesEigenVectors);
	cvReleaseMat(&tempEigenVectors);
	cvReleaseMat(&tempEigenVals);
	LOGD("Done (%d/%d)\n", nTruncated, nEigenAtMost);
}

void PointDistributionAAModel::computeLocalShape(const CvMat* p, CvMat* s) {
	cvBackProjectPCA(p, __MeanShape, __ShapesEigenVectors, s);
}

void PointDistributionAAModel::computeGlobalShape(const CvMat* q, CvMat* s) {
	int npoints = nPoints();
	double* fasts = s->data.db;
	double a = cvmGet(q, 0, 0) + 1, b = cvmGet(q, 0, 1),
	tx = cvmGet(q, 0, 2), ty = cvmGet(q, 0, 3);
	double x, y;
	for (int i = 0; i < npoints; i++) {
		x = fasts[2 * i];
		y = fasts[2 * i + 1];
		fasts[2 * i] = a*x - b*y + tx;
		fasts[2 * i + 1] = b*x + a*y + ty;
	}
}

void PointDistributionAAModel::computeShape(const CvMat* p, const CvMat* q, CvMat* s) {
	computeLocalShape(p, s);
	computeGlobalShape(q, s);
}

void PointDistributionAAModel::computeShape(const CvMat* pq, CvMat* s) {
	CvMat p, q;
	cvGetCols(pq, &q, 0, 4);
	cvGetCols(pq, &p, 4, 4 + nModes());
	computeShape(&p, &q, s);
}

void PointDistributionAAModel::computeParams(const CvMat* s, CvMat* pq) {
	CvMat p, q;
	cvGetCols(pq, &q, 0, 4);
	cvGetCols(pq, &p, 4, 4 + nModes());

	computeParams(s, &p, &q);
}

void PointDistributionAAModel::computeParams(const CvMat* s, CvMat* p, CvMat* q) {
	int nmodes = nModes(), npoints = nPoints();

	double a, b, tx, ty;
	double a_, b_, tx_, ty_;
	double norm;

	__y.Mat2Point(s);
	__y.getCenterOfGravity(tx, ty);
	__y.translateShape(-tx, -ty);
	cvmSet(q, 0, 2, tx);
	cvmSet(q, 0, 3, ty);

	// do a few iterations to get (s, theta, p)
	cvZero(p);
	for (int iter = 0; iter < 2; iter++) {
		cvBackProjectPCA(p, __MeanShape, __ShapesEigenVectors, __matshape);
		__x.Mat2Point(__matshape);

		__x.getAlignTransformation(__y, a, b, tx, ty); //in fact, tx = ty = 0

		norm = a*a + b*b;
		a_ = a / norm; b_ = -b / norm;
		tx_ = (-a*tx - b*ty) / norm; ty_ = (b*tx - a*ty) / norm;
		__x = __y;
		__x.transformShape(a_, b_, tx_, ty_);

		__x.Point2Mat(__matshape);
		cvProjectPCA(__matshape, __MeanShape, __ShapesEigenVectors, p);
	}

	cvmSet(q, 0, 0, a - 1);
	cvmSet(q, 0, 1, b);
	Clamp(p, 1.8);
}

void PointDistributionAAModel::Clamp(CvMat* p, double s_d) {
	double* fastp = p->data.db;
	double* fastv = __ShapesEigenValues->data.db;
	int nmodes = nModes();
	double limit;

	for (int i = 0; i < nmodes; i++) {
		limit = s_d*sqrt(fastv[i]);
		if (fastp[i] > limit) fastp[i] = limit;
		else if (fastp[i] < -limit) fastp[i] = -limit;
	}
}

void PointDistributionAAModel::writeStream(std::ofstream& os) {
	int _nPoints = nPoints();
	int _nModes = nModes();
	os.write((char*)&_nPoints, sizeof(_nPoints));
	os.write((char*)&_nModes, sizeof(_nModes));

	WriteCvMat(os, __MeanShape);
	WriteCvMat(os, __ShapesEigenValues);
	WriteCvMat(os, __ShapesEigenVectors);
	__AAMRefShape.writeStream(os);
}

void PointDistributionAAModel::readStream(std::ifstream& is) {
	int _nPoints, _nModes;
	is >> _nPoints >> _nModes;
	// is.read((char*)&_nPoints, sizeof(_nPoints));
	// is.read((char*)&_nModes, sizeof(_nModes));

	__MeanShape = cvCreateMat(1, _nPoints * 2, CV_64FC1);
	__ShapesEigenValues = cvCreateMat(1, _nModes, CV_64FC1);
	__ShapesEigenVectors = cvCreateMat(_nModes, _nPoints * 2, CV_64FC1);
	__AAMRefShape.resize(_nPoints);

	ReadCvMat(is, __MeanShape);
	ReadCvMat(is, __ShapesEigenValues);
	ReadCvMat(is, __ShapesEigenVectors);
	__AAMRefShape.readStream(is);

	__matshape = cvCreateMat(1, nPoints()*2, CV_64FC1);
}