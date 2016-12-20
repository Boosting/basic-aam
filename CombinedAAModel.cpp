#include "stdafx.h"
#include "CombinedAAModel.h"

CombinedAAModel::CombinedAAModel() {
	__MeanAppearance = 0;
	__AppearanceEigenValues = 0;
	__AppearanceEigenVectors = 0;
	__Qs = 0;
	__Qg = 0;
	__MeanS = 0;
	__MeanG = 0;
	__Points = 0;
	__Storage = 0;
	__pq = 0;
	__a = 0;
}

CombinedAAModel::~CombinedAAModel() {
	cvReleaseMat(&__MeanAppearance);
	cvReleaseMat(&__AppearanceEigenValues);
	cvReleaseMat(&__AppearanceEigenVectors);
	cvReleaseMat(&__Qg);
	cvReleaseMat(&__Qs);
	cvReleaseMat(&__MeanS);
	cvReleaseMat(&__MeanG);
	cvReleaseMat(&__Points);
	cvReleaseMemStorage(&__Storage);
	cvReleaseMat(&__pq);
	cvReleaseMat(&__a);
}

void CombinedAAModel::trainModel(const inputFiles& inputPoints, const inputFiles& inputImages,
	double scale, double shapeEstimate, double textureEstimate, double appearanceEstimate) {
	
	//building shape and texture distribution model
	std::vector<ShapeModel> completeShapeList;
	for (int k = 0; k<inputPoints.size(); k++) {
		ShapeModel shapeModel;
		bool flag = shapeModel.readData(inputPoints[k]);
		if (!flag) {
			IplImage* image = cvLoadImage(inputImages[k].c_str(), -1);
			shapeModel.ScaleXY(image->width, image->height);
			cvReleaseImage(&image);
		}
		completeShapeList.push_back(shapeModel);
	}

	// Result 1 --- Plotting raw data
	// Superimposing all scaled landmark points on one image
	/* IplImage* imgScribble = cvCreateImage(cvSize(1000, 1000), 8, 3);
	cvSet(imgScribble, cvScalar(0, 0, 0));
	for (std::vector<ShapeModel>::iterator iter = completeShapeList.begin(); iter != completeShapeList.end(); ++iter) {
		cvNamedWindow("tempWin", CV_WINDOW_AUTOSIZE);
		Commons::DrawPoints(imgScribble, *iter);
		cvShowImage("tempWin", imgScribble);
		cvWaitKey(0);
	}
	cvReleaseImage(&imgScribble); */

	// End Result

	LOGD("Building point distribution model...\n");
	__pointDistributionModel.trainModel(completeShapeList, scale, shapeEstimate);

	LOGD("Building warping information for the mean shape...");
	__Points = cvCreateMat(1, __pointDistributionModel.nPoints(), CV_32FC2);
	__Storage = cvCreateMemStorage(0);
	ShapeModel referenceShape = __pointDistributionModel.__AAMRefShape;
	
	__piecewiseAffineWarp.trainModel(referenceShape, __Points, __Storage);
	LOGD("[%d by %d -- dimensions, %d -- no. of triangles, %d*3 -- no. of pixels]\n", __piecewiseAffineWarp.Width(), __piecewiseAffineWarp.Height(), __piecewiseAffineWarp.nTri(), __piecewiseAffineWarp.nPix());

	LOGD("Building texture distribution model...\n");
	__textureDistributionModel.trainModel(inputPoints, inputImages, __piecewiseAffineWarp, textureEstimate, true);
	__pq = cvCreateMat(1, __pointDistributionModel.nModes() + 4, CV_64FC1);

	LOGD("Build combined appearance model...\n");
	int nsamples = inputPoints.size();
	int npointsby2 = __pointDistributionModel.nPoints() * 2;
	int npixels = __textureDistributionModel.nPixels();
	int nfeatures = __pointDistributionModel.nModes() + __textureDistributionModel.nModes();
	CvMat* completeAppearanceList = cvCreateMat(nsamples, nfeatures, CV_64FC1);
	CvMat* shapeInstance = cvCreateMat(1, npointsby2, CV_64FC1);
	CvMat* textureInstance = cvCreateMat(1, npixels, CV_64FC1);
	__MeanS = cvCreateMat(1, npointsby2, CV_64FC1);
	__MeanG = cvCreateMat(1, npixels, CV_64FC1);
	cvCopy(__pointDistributionModel.GetMean(), __MeanS);
	cvCopy(__textureDistributionModel.GetMean(), __MeanG);

	//calculate ratio of shape to appearance
	CvScalar Sum1 = cvSum(__pointDistributionModel.__ShapesEigenValues);
	CvScalar Sum2 = cvSum(__textureDistributionModel.__EigenValsOfTexture);
	__ratio_shapes_texture = sqrt(Sum2.val[0] / Sum1.val[0]);

	LOGD("Combine shape and texture parameters...\n");
	for (int i=0; i<nsamples; i++) {
		//Get shapeModel and Texture respectively
		IplImage* image = cvLoadImage(inputImages[i].c_str(), -1);

		ShapeModel shapeModel;
		if (!shapeModel.readData(inputPoints[i]))
			shapeModel.ScaleXY(image->width, image->height);
		shapeModel.Point2Mat(shapeInstance);
		Commons::checkIfShapeInBoundary(shapeInstance, image->width, image->height);

		__piecewiseAffineWarp.computeWarpedTexture(shapeInstance, image, textureInstance);
		__textureDistributionModel.NormalizeTexture(__MeanG, textureInstance);

		//combine shape and texture parameters
		CvMat OneAppearance;
		cvGetRow(completeAppearanceList, &OneAppearance, i);
		ShapeTexture2Combined(shapeInstance, textureInstance, &OneAppearance);

		cvReleaseImage(&image);
	}

	//Do PCA of appearances
	computePCA(completeAppearanceList, appearanceEstimate);

	int np = __AppearanceEigenVectors->rows;

	LOGD("Extracting the shape and texture part of the combined eigen vectors..\n");

	// extract the shape part of the combined eigen vectors
	CvMat Ps;
	cvGetCols(__AppearanceEigenVectors, &Ps, 0, __pointDistributionModel.nModes());
	__Qs = cvCreateMat(np, npointsby2, CV_64FC1);
	cvMatMul(&Ps, __pointDistributionModel.GetBases(), __Qs);
	cvConvertScale(__Qs, __Qs, 1.0 / __ratio_shapes_texture);

	// extract the texture part of the combined eigen vectors
	CvMat Pg;
	cvGetCols(__AppearanceEigenVectors, &Pg, __pointDistributionModel.nModes(), nfeatures);
	__Qg = cvCreateMat(np, npixels, CV_64FC1);
	cvMatMul(&Pg, __textureDistributionModel.GetBases(), __Qg);

	__a = cvCreateMat(1, __AppearanceEigenVectors->cols, CV_64FC1);
}

void CombinedAAModel::ShapeTexture2Combined(const CvMat* shapeModel, const CvMat* Texture, CvMat* Appearance) {
	__pointDistributionModel.computeParams(shapeModel, __pq);
	CvMat mat1, mat2;
	cvGetCols(__pq, &mat1, 4, 4 + __pointDistributionModel.nModes());
	cvGetCols(Appearance, &mat2, 0, __pointDistributionModel.nModes());
	cvCopy(&mat1, &mat2);
	cvConvertScale(&mat2, &mat2, __ratio_shapes_texture);

	cvGetCols(Appearance, &mat2, __pointDistributionModel.nModes(), __pointDistributionModel.nModes() + __textureDistributionModel.nModes());
	__textureDistributionModel.computeParams(Texture, &mat2);
}

void CombinedAAModel::computePCA(const CvMat* AllAppearances, double percentage) {
	LOGD("Doing PCA of appearance datas...");

	int nSamples = AllAppearances->rows;
	int nfeatures = AllAppearances->cols;
	int nEigenAtMost = MIN(nSamples, nfeatures);

	CvMat* tmpEigenValues = cvCreateMat(1, nEigenAtMost, CV_64FC1);
	CvMat* tmpEigenVectors = cvCreateMat(nEigenAtMost, nfeatures, CV_64FC1);
	__MeanAppearance = cvCreateMat(1, nfeatures, CV_64FC1);

	cvCalcPCA(AllAppearances, __MeanAppearance,
		tmpEigenValues, tmpEigenVectors, CV_PCA_DATA_AS_ROW);

	double allSum = cvSum(tmpEigenValues).val[0];
	double partSum = 0.0;
	int nTruncated = 0;
	double largesteigval = cvmGet(tmpEigenValues, 0, 0);
	for (int i = 0; i < nEigenAtMost; i++)
	{
		double thiseigval = cvmGet(tmpEigenValues, 0, i);
		if (thiseigval / largesteigval < 0.0001) break; // firstly check
		partSum += thiseigval;
		++nTruncated;
		if (partSum / allSum >= percentage)	break;    //secondly check
	}

	__AppearanceEigenValues = cvCreateMat(1, nTruncated, CV_64FC1);
	__AppearanceEigenVectors = cvCreateMat(nTruncated, nfeatures, CV_64FC1);

	CvMat G;
	cvGetCols(tmpEigenValues, &G, 0, nTruncated);
	cvCopy(&G, __AppearanceEigenValues);

	cvGetRows(tmpEigenVectors, &G, 0, nTruncated);
	cvCopy(&G, __AppearanceEigenVectors);

	cvReleaseMat(&tmpEigenVectors);
	cvReleaseMat(&tmpEigenValues);
	LOGD("Done (%d/%d)\n", nTruncated, nEigenAtMost);
}

void CombinedAAModel::computeLocalShape(CvMat* shapeInstance, const CvMat* combinedParams) {
	cvMatMul(combinedParams, __Qs, shapeInstance);
	cvAdd(shapeInstance, __MeanS, shapeInstance);
}

void CombinedAAModel::computeGlobalShape(CvMat* shapeInstance, const CvMat* pose) {
	int npoints = shapeInstance->cols / 2;
	double* fasts = shapeInstance->data.db;
	double a = cvmGet(pose, 0, 0) + 1, b = cvmGet(pose, 0, 1),
		tx = cvmGet(pose, 0, 2), ty = cvmGet(pose, 0, 3);
	double x, y;
	for (int i = 0; i < npoints; i++)
	{
		x = fasts[2 * i];
		y = fasts[2 * i + 1];

		fasts[2 * i] = a*x - b*y + tx;
		fasts[2 * i + 1] = b*x + a*y + ty;
	}
}

void CombinedAAModel::computeTextureFromLamdaValues(CvMat* textureInstance, const CvMat* combinedParams) {
	cvMatMul(combinedParams, __Qg, textureInstance);
	cvAdd(textureInstance, __MeanG, textureInstance);
}

void CombinedAAModel::computeParams(CvMat* combinedParams, const CvMat* bs, const CvMat* bg) {
	double* fasta = __a->data.db;
	double* fastbs = bs->data.db;
	double* fastbg = bg->data.db;

	int i;
	for (i = 0; i < bs->cols; i++)	fasta[i] = __ratio_shapes_texture * fastbs[i];
	for (i = 0; i < bg->cols; i++)   fasta[i + bs->cols] = fastbg[i];

	cvProjectPCA(__a, __MeanAppearance, __AppearanceEigenVectors, combinedParams);
}

void CombinedAAModel::Clamp(CvMat* combinedParams, double s_d) {
	double* fastc = combinedParams->data.db;
	double* fastv = __AppearanceEigenValues->data.db;
	int nmodes = nModes();
	double limit;

	for (int i = 0; i < nmodes; i++)
	{
		limit = s_d*sqrt(fastv[i]);
		if (fastc[i] > limit) fastc[i] = limit;
		else if (fastc[i] < -limit) fastc[i] = -limit;
	}
}

void CombinedAAModel::DrawAppearance(IplImage* image, const ShapeModel& shapeModel, CvMat* Texture) {
	piecewiseAffineWarpAAModel paw;
	int x1, x2, y1, y2, idx1 = 0, idx2 = 0;
	int tri_idx, v1, v2, v3;
	int minx, miny, maxx, maxy;
	paw.trainModel(shapeModel, __Points, __Storage, __piecewiseAffineWarp.GetTri(), false);
	ShapeModel referenceShape = __piecewiseAffineWarp.__referenceshape;
	double minV, maxV;
	cvMinMaxLoc(Texture, &minV, &maxV);
	cvConvertScale(Texture, Texture, 1 / (maxV - minV) * 255, -minV * 255 / (maxV - minV));

	minx = shapeModel.MinX(); miny = shapeModel.MinY();
	maxx = shapeModel.MaxX(); maxy = shapeModel.MaxY();
	for (int y = miny; y < maxy; y++)
	{
		y1 = y - miny;
		for (int x = minx; x < maxx; x++)
		{
			x1 = x - minx;
			idx1 = paw.Rect(y1, x1);
			if (idx1 >= 0)
			{
				tri_idx = paw.PixTri(idx1);
				v1 = paw.Tri(tri_idx, 0);
				v2 = paw.Tri(tri_idx, 1);
				v3 = paw.Tri(tri_idx, 2);

				x2 = paw.Alpha(idx1)*referenceShape[v1].x + paw.Belta(idx1)*referenceShape[v2].x +
					paw.Gamma(idx1)*referenceShape[v3].x;
				y2 = paw.Alpha(idx1)*referenceShape[v1].y + paw.Belta(idx1)*referenceShape[v2].y +
					paw.Gamma(idx1)*referenceShape[v3].y;

				idx2 = __piecewiseAffineWarp.Rect(y2, x2);
				if (idx2 < 0) continue;

				CV_IMAGE_ELEM(image, byte, y, 3 * x) = cvmGet(Texture, 0, 3 * idx2);
				CV_IMAGE_ELEM(image, byte, y, 3 * x + 1) = cvmGet(Texture, 0, 3 * idx2 + 1);
				CV_IMAGE_ELEM(image, byte, y, 3 * x + 2) = cvmGet(Texture, 0, 3 * idx2 + 2);
			}
		}
	}
}


void CombinedAAModel::writeStream(std::ofstream& os) {
	__pointDistributionModel.writeStream(os);
	__textureDistributionModel.writeStream(os);
	__piecewiseAffineWarp.writeStream(os);

	os.write((char*)&__AppearanceEigenVectors->rows, sizeof(int));
	os.write((char*)&__AppearanceEigenVectors->cols, sizeof(int));
	os.write((char*)&__ratio_shapes_texture, sizeof(__ratio_shapes_texture));
	WriteCvMat(os, __MeanAppearance);
	WriteCvMat(os, __AppearanceEigenValues);
	WriteCvMat(os, __AppearanceEigenVectors);
	WriteCvMat(os, __Qs);
	WriteCvMat(os, __Qg);
	WriteCvMat(os, __MeanS);
	WriteCvMat(os, __MeanG);
}

void CombinedAAModel::readStream(std::ifstream& is) {
	__pointDistributionModel.readStream(is);
	__textureDistributionModel.readStream(is);
	__piecewiseAffineWarp.readStream(is);

	int np, nfeatures;
	is >> np >> nfeatures >> __ratio_shapes_texture;
	// is.read((char*)&np, sizeof(int));
	// is.read((char*)& nfeatures, sizeof(int));
	// is.read((char*)& __ratio_shapes_texture, sizeof(__ratio_shapes_texture));

	__MeanAppearance = cvCreateMat(1, nfeatures, CV_64FC1);
	__AppearanceEigenValues = cvCreateMat(1, np, CV_64FC1);
	__AppearanceEigenVectors = cvCreateMat(np, nfeatures, CV_64FC1);
	__Qs = cvCreateMat(np, __pointDistributionModel.nPoints() * 2, CV_64FC1);
	__Qg = cvCreateMat(np, __textureDistributionModel.nPixels(), CV_64FC1);
	__MeanS = cvCreateMat(1, __pointDistributionModel.nPoints() * 2, CV_64FC1);
	__MeanG = cvCreateMat(1, __textureDistributionModel.nPixels(), CV_64FC1);

	ReadCvMat(is, __MeanAppearance);
	ReadCvMat(is, __AppearanceEigenValues);
	ReadCvMat(is, __AppearanceEigenVectors);
	ReadCvMat(is, __Qs);
	ReadCvMat(is, __Qg);
	ReadCvMat(is, __MeanS);
	ReadCvMat(is, __MeanG);

	__Points = cvCreateMat(1, __pointDistributionModel.nPoints(), CV_32FC2);
	__Storage = cvCreateMemStorage(0);
	__pq = cvCreateMat(1, __pointDistributionModel.nModes() + 4, CV_64FC1);
	__a = cvCreateMat(1, __AppearanceEigenVectors->cols, CV_64FC1);
}

static CombinedAAModel *g_cam;
static const int n = 6;//appearance modes
static int b_c[n];
static const int offset = 40;
static CvMat* combinedParams = 0;
static CvMat* shapeInstance = 0;
static CvMat* textureInstance = 0;
static IplImage* image = 0;
static ShapeModel shapeModel;

void ontrackcam(int pos) {
	if (combinedParams == 0) {
		combinedParams = cvCreateMat(1, g_cam->nModes(), CV_64FC1); cvZero(combinedParams);
		shapeInstance = cvCreateMat(1, g_cam->__pointDistributionModel.nPoints() * 2, CV_64FC1);
		textureInstance = cvCreateMat(1, g_cam->__textureDistributionModel.nPixels(), CV_64FC1);
	}

	double variance;
	//registrate appearance parameters
	for (int i = 0; i < n; i++) {
		variance = 3 * sqrt(g_cam->computeVariance(i))*(double(b_c[i]) / offset - 1.0);
		cvmSet(combinedParams, 0, i, variance);
	}

	g_cam->computeLocalShape(shapeInstance, combinedParams);
	g_cam->computeTextureFromLamdaValues(textureInstance, combinedParams);

	shapeModel.Mat2Point(shapeInstance);
	int w = shapeModel.GetWidth(), h = shapeModel.MaxY() - shapeModel.MinY();
	shapeModel.translateShape(w, h);
	if (image == 0)image = cvCreateImage(cvSize(w * 2, h * 2), 8, 3);
	cvSet(image, cvScalar(128, 128, 128));
	g_cam->DrawAppearance(image, shapeModel, textureInstance);

	cvNamedWindow("Combined Appearance Model", 1);
	cvShowImage("Combined Appearance Model", image);

	if (cvWaitKey(10) == '27') {
		cvReleaseImage(&image);
		cvReleaseMat(&shapeInstance);
		cvReleaseMat(&textureInstance);
		cvReleaseMat(&combinedParams);
		cvDestroyWindow("Parameters");
		cvDestroyWindow("Combined Appearance Model");
	}
}

void CombinedAAModel::ShowVariation() {
	printf("Show modes of appearance variations...\n");
	cvNamedWindow("Parameters", 1);

	//create trackbars for appearance
	for (int i = 0; i < n; i++) {
		char barname[100];
		sprintf(barname, "a %d", i);
		//teddt
		b_c[i] = offset;
		cvCreateTrackbar(barname, "Parameters", &b_c[i], 2 * offset + 1, ontrackcam);
	}

	g_cam = this;
	ontrackcam(1);
	cvWaitKey(0);
}