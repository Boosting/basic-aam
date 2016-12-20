#include "stdafx.h"
#include "textureDistributionAAModel.h"
#include "piecewiseAffineWarpAAModel.h"
#include "Utils.h"

textureDistributionAAModel::textureDistributionAAModel() {
	__meanTextureMat = 0;
	__EigenVectorsOfTexture = 0;
	__EigenValsOfTexture = 0;
}

textureDistributionAAModel::~textureDistributionAAModel() {
	cvReleaseMat(&__meanTextureMat);
	cvReleaseMat(&__EigenVectorsOfTexture);
	cvReleaseMat(&__EigenValsOfTexture);
}

void textureDistributionAAModel::trainModel(const inputFiles& inputPoints, const inputFiles& inputImages,
	const piecewiseAffineWarpAAModel& warpedModel, double textureEstimate, bool registration) {
	int pointsCount = warpedModel.nPoints();
	int pixelCount = warpedModel.nPix() * 3;
	int sampleCount = inputPoints.size();

	CvMat *completeTextureMat = cvCreateMat(sampleCount, pixelCount, CV_64FC1);
	CvMat * matshape = cvCreateMat(1, pointsCount * 2, CV_64FC1);
	for (int i = 0; i < sampleCount; i++) {
		IplImage* image = cvLoadImage(inputImages[i].c_str(), -1);
		ShapeModel trueshape;
		if (!trueshape.readData(inputPoints[i]))
			trueshape.ScaleXY(image->width, image->height);
		trueshape.Point2Mat(matshape);
		Commons::checkIfShapeInBoundary(matshape, image->width, image->height);

		CvMat t; cvGetRow(completeTextureMat, &t, i);
		warpedModel.computeWarpedTexture(matshape, image, &t);
		cvReleaseImage(&image);
	}
	cvReleaseMat(&matshape);

	textureDistributionAAModel::alignAllTextureMatrices(completeTextureMat);
	computePCA(completeTextureMat, textureEstimate);
	if (registration) SaveSeriesTemplate(completeTextureMat, warpedModel);
	cvReleaseMat(&completeTextureMat);
}

void textureDistributionAAModel::computePCA(const CvMat* completeTextureMat, double percentage) {
	LOGD("Computing PCA of textures data...");

	int sampleCount = completeTextureMat->rows;
	int pixelCount = completeTextureMat->cols;
	int nEigenAtMost = MIN(sampleCount, pixelCount);

	CvMat* tmpEigenValues = cvCreateMat(1, nEigenAtMost, CV_64FC1);
	CvMat* tmpEigenVectors = cvCreateMat(nEigenAtMost, pixelCount, CV_64FC1);
	__meanTextureMat = cvCreateMat(1, pixelCount, CV_64FC1);

	cvCalcPCA(completeTextureMat, __meanTextureMat, tmpEigenValues, tmpEigenVectors, CV_PCA_DATA_AS_ROW);

	double allSum = cvSum(tmpEigenValues).val[0];
	double partSum = 0.0;
	int nTruncated = 0;
	double biggestEigenValue = cvmGet(tmpEigenValues, 0, 0);
	for (int i = 0; i < nEigenAtMost; i++) {
		double currEigenValue = cvmGet(tmpEigenValues, 0, i);
		if (currEigenValue / biggestEigenValue < 0.0001) break;
		partSum += currEigenValue;
		++nTruncated;
		if (partSum / allSum >= percentage)	break;
	}
	__EigenValsOfTexture = cvCreateMat(1, nTruncated, CV_64FC1);
	__EigenVectorsOfTexture = cvCreateMat(nTruncated, pixelCount, CV_64FC1);

	CvMat G;
	cvGetCols(tmpEigenValues, &G, 0, nTruncated);
	cvCopy(&G, __EigenValsOfTexture);
	cvGetRows(tmpEigenVectors, &G, 0, nTruncated);
	cvCopy(&G, __EigenVectorsOfTexture);
	cvReleaseMat(&tmpEigenVectors);
	cvReleaseMat(&tmpEigenValues);

	LOGD("Done (%d/%d)\n", nTruncated, nEigenAtMost);
}

void textureDistributionAAModel::computeTextureFromLamdaValues(const CvMat* lamda, CvMat* t) {
	cvBackProjectPCA(lamda, __meanTextureMat, __EigenVectorsOfTexture, t);
}

void textureDistributionAAModel::computeParams(const CvMat* t, CvMat* lamda) {
	cvProjectPCA(t, __meanTextureMat, __EigenVectorsOfTexture, lamda);
}

void textureDistributionAAModel::Clamp(CvMat* lamda, double s_d) {
	double* fastp = lamda->data.db;
	double* fastv = __EigenValsOfTexture->data.db;
	int nmodes = nModes();
	double limit;

	for (int i = 0; i < nmodes; i++) {
		limit = s_d*sqrt(fastv[i]);
		if (fastp[i] > limit) fastp[i] = limit;
		else if (fastp[i] < -limit) fastp[i] = -limit;
	}
}

void textureDistributionAAModel::alignAllTextureMatrices(CvMat* completeTextureMat) {
	LOGD("Aligning textures to account for illumination changes...\n");

	int sampleCount = completeTextureMat->rows;
	int pixelCount = completeTextureMat->cols;
	CvMat* meanTexture = cvCreateMat(1, pixelCount, CV_64FC1);
	CvMat* lastMeanEstimate = cvCreateMat(1, pixelCount, CV_64FC1);
	CvMat* constmeanTexture = cvCreateMat(1, pixelCount, CV_64FC1);
	CvMat textureImage;
	textureDistributionAAModel::computeMeanTexture(completeTextureMat, meanTexture);
	textureDistributionAAModel::ZeroMeanUnitLength(meanTexture);
	cvCopy(meanTexture, constmeanTexture);

	double diff, diff_max = 1e-6;
	const int max_iter = 15;
	for (int iter = 0; iter < max_iter; iter++)	{	//Keep on iterating until convergance / max iterations reached
		cvCopy(meanTexture, lastMeanEstimate);
		for (int i = 0; i < sampleCount; i++) {
			cvGetRow(completeTextureMat, &textureImage, i);
			textureDistributionAAModel::NormalizeTexture(meanTexture, &textureImage);
		}
		// Compute the new mean texture 
		textureDistributionAAModel::computeMeanTexture(completeTextureMat, meanTexture);
		// Normalize it
		textureDistributionAAModel::NormalizeTexture(constmeanTexture, meanTexture);
		diff = cvNorm(meanTexture, lastMeanEstimate);
		LOGD("Texture Alignment Iteration #%i, estimationf difference = %g\n", iter, diff);
		if (diff <= diff_max) break;
	}
	cvReleaseMat(&meanTexture);
	cvReleaseMat(&lastMeanEstimate);
	cvReleaseMat(&constmeanTexture);
}

void textureDistributionAAModel::computeMeanTexture(const CvMat* completeTextureMat, CvMat* meanTexture) {
	CvMat submat;
	for (int i = 0; i < meanTexture->cols; i++) {
		cvGetCol(completeTextureMat, &submat, i);
		cvmSet(meanTexture, 0, i, cvAvg(&submat).val[0]);
	}
}

void textureDistributionAAModel::NormalizeTexture(const CvMat* refTextrure, CvMat* Texture) {
	textureDistributionAAModel::ZeroMeanUnitLength(Texture);
	double alpha = cvDotProduct(Texture, refTextrure);
	if (alpha != 0)	cvConvertScale(Texture, Texture, 1.0 / alpha, 0);
}

void textureDistributionAAModel::ZeroMeanUnitLength(CvMat* Texture) {
	CvScalar mean = cvAvg(Texture);
	cvSubS(Texture, mean, Texture);
	double norm = cvNorm(Texture);
	cvConvertScale(Texture, Texture, 1.0 / norm);
}

void textureDistributionAAModel::SaveSeriesTemplate(const CvMat* completeTextureMat, const piecewiseAffineWarpAAModel& warpedModel) {
	LOGD("Saving the face template image...\n");
	Commons::createDirectory("registration");
	Commons::createDirectory("Modes");
	Commons::createDirectory("Tri");
	char filename[100];

	int i;
	for (i = 0; i < completeTextureMat->rows; i++) {
		CvMat oneTexture;
		cvGetRow(completeTextureMat, &oneTexture, i);
		sprintf(filename, "registration/%d.jpg", i);
		warpedModel.storeWarpedTextureAsImage(filename, &oneTexture);
	}

	for (int nmodes = 0; nmodes < nModes(); nmodes++) {
		CvMat oneVar;
		cvGetRow(__EigenVectorsOfTexture, &oneVar, nmodes);

		sprintf(filename, "Modes/A%03d.jpg", nmodes + 1);
		warpedModel.storeWarpedTextureAsImage(filename, &oneVar);
	}

	IplImage* templateimg = cvCreateImage
	(cvSize(warpedModel.Width(), warpedModel.Height()), IPL_DEPTH_8U, 3);
	IplImage* convexImage = cvCreateImage
	(cvSize(warpedModel.Width(), warpedModel.Height()), IPL_DEPTH_8U, 3);
	IplImage* TriImage = cvCreateImage
	(cvSize(warpedModel.Width(), warpedModel.Height()), IPL_DEPTH_8U, 3);

	warpedModel.storeWarpedTextureAsImage("Modes/Template.jpg", __meanTextureMat);
	warpedModel.convertTextureToImage(templateimg, __meanTextureMat);

	cvSetZero(convexImage);
	for (i = 0; i < warpedModel.nTri(); i++) {
		CvPoint p, q;
		int ind1, ind2;
		cvCopy(templateimg, TriImage);
		ind1 = warpedModel.Tri(i, 0); ind2 = warpedModel.Tri(i, 1);
		p = cvPointFrom32f(warpedModel.Vertex(ind1));
		q = cvPointFrom32f(warpedModel.Vertex(ind2));
		cvLine(TriImage, p, q, CV_RGB(255, 255, 255));
		cvLine(convexImage, p, q, CV_RGB(255, 255, 255));

		ind1 = warpedModel.Tri(i, 1); ind2 = warpedModel.Tri(i, 2);
		p = cvPointFrom32f(warpedModel.Vertex(ind1));
		q = cvPointFrom32f(warpedModel.Vertex(ind2));
		cvLine(TriImage, p, q, CV_RGB(255, 255, 255));
		cvLine(convexImage, p, q, CV_RGB(255, 255, 255));

		ind1 = warpedModel.Tri(i, 2); ind2 = warpedModel.Tri(i, 0);
		p = cvPointFrom32f(warpedModel.Vertex(ind1));
		q = cvPointFrom32f(warpedModel.Vertex(ind2));
		cvLine(TriImage, p, q, CV_RGB(255, 255, 255));
		cvLine(convexImage, p, q, CV_RGB(255, 255, 255));

		sprintf(filename, "Tri/%03i.jpg", i + 1);
		cvSaveImage(filename, TriImage);
	}
	cvSaveImage("Tri/convex.jpg", convexImage);

	cvReleaseImage(&templateimg);
	cvReleaseImage(&convexImage);
	cvReleaseImage(&TriImage);
}

void textureDistributionAAModel::writeStream(std::ofstream& os) {
	int _npixels = nPixels();
	int _nModes = nModes();

	os.write((char*)&_npixels, sizeof(int));
	os.write((char*)&_nModes, sizeof(int));

	WriteCvMat(os, __meanTextureMat);
	WriteCvMat(os, __EigenValsOfTexture);
	WriteCvMat(os, __EigenVectorsOfTexture);
}

void textureDistributionAAModel::readStream(std::ifstream& is) {
	int _npixels, _nModes;
	is >> _npixels >> _nModes;
	// is.read((char*)&_npixels, sizeof(int));
	// is.read((char*)&_nModes, sizeof(int));

	__meanTextureMat = cvCreateMat(1, _npixels, CV_64FC1);
	__EigenValsOfTexture = cvCreateMat(1, _nModes, CV_64FC1);
	__EigenVectorsOfTexture = cvCreateMat(_nModes, _npixels, CV_64FC1);

	ReadCvMat(is, __meanTextureMat);
	ReadCvMat(is, __EigenValsOfTexture);
	ReadCvMat(is, __EigenVectorsOfTexture);
}
