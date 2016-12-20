#include "stdafx.h"
#include <ctime> 
#include "BasicAAModel.h"

BasicAAModel::BasicAAModel() {
	__G = 0;
	__curr_appearance_pose = 0;
	__update_appearance_pose = 0;
	__delta_appearance_pose = 0;
	__appearanceParams = 0;
	__shapeParams = 0;
	__poseParams = 0;
	__lamda = 0;
	__pointDistributionModel = 0;
	__modelledTexture = 0;
	__warpedTexture = 0;
	__delta_warped_model_texture = 0;
}

BasicAAModel::~BasicAAModel() {
	cvReleaseMat(&__G);	cvReleaseMat(&__curr_appearance_pose);
	cvReleaseMat(&__update_appearance_pose); cvReleaseMat(&__delta_appearance_pose);
	cvReleaseMat(&__shapeParams); cvReleaseMat(&__poseParams);
	cvReleaseMat(&__appearanceParams);	cvReleaseMat(&__lamda);
	cvReleaseMat(&__pointDistributionModel);	cvReleaseMat(&__warpedTexture);
	cvReleaseMat(&__modelledTexture); cvReleaseMat(&__delta_warped_model_texture);
}

void BasicAAModel::trainModel(const inputFiles& inputPoints,
	const inputFiles& inputImages, double scale, double shapeEstimate, double textureEstimate, double appearanceEstimate) {

	LOGD("################################################\n");
	__cam.trainModel(inputPoints, inputImages, scale, shapeEstimate, textureEstimate, appearanceEstimate);

	LOGD("Computing Jacobian Matrix ...\n");
	__G = cvCreateMat(__cam.nModes() + 4, __cam.__textureDistributionModel.nPixels(), CV_64FC1);
	ComputeJacobian(inputPoints, inputImages);

	__curr_appearance_pose = cvCreateMat(1, __cam.nModes() + 4, CV_64FC1);
	__update_appearance_pose = cvCreateMat(1, __cam.nModes() + 4, CV_64FC1);
	__delta_appearance_pose = cvCreateMat(1, __cam.nModes() + 4, CV_64FC1);
	__appearanceParams = cvCreateMat(1, __cam.nModes(), CV_64FC1);
	__shapeParams = cvCreateMat(1, __cam.__pointDistributionModel.nModes(), CV_64FC1);
	__poseParams = cvCreateMat(1, 4, CV_64FC1);
	__lamda = cvCreateMat(1, __cam.__textureDistributionModel.nModes(), CV_64FC1);
	__pointDistributionModel = cvCreateMat(1, __cam.__pointDistributionModel.nPoints() * 2, CV_64FC1);
	__warpedTexture = cvCreateMat(1, __cam.__textureDistributionModel.nPixels(), CV_64FC1);
	__modelledTexture = cvCreateMat(1, __cam.__textureDistributionModel.nPixels(), CV_64FC1);
	__delta_warped_model_texture = cvCreateMat(1, __cam.__textureDistributionModel.nPixels(), CV_64FC1);

	LOGD("################################################\n\n");
}

static double rand_in_between(double a, double b) {
	int A = rand() % 50;
	return a + (b - a)*A / 49;
}

void BasicAAModel::ComputeJacobian(const inputFiles& inputPoints,
	const inputFiles& inputImages, double disp_scale, double disp_angle, double disp_trans, double disp_std, int nExp) {
	CvMat* J = cvCreateMat(__cam.nModes() + 4, __cam.__textureDistributionModel.nPixels(), CV_64FC1);
	CvMat* d = cvCreateMat(1, __cam.nModes() + 4, CV_64FC1);
	CvMat* o = cvCreateMat(1, __cam.nModes() + 4, CV_64FC1);
	CvMat* oo = cvCreateMat(1, __cam.nModes() + 4, CV_64FC1);
	CvMat* textureInstance = cvCreateMat(1, __cam.__textureDistributionModel.nPixels(), CV_64FC1);
	CvMat* t_m = cvCreateMat(1, __cam.__textureDistributionModel.nPixels(), CV_64FC1);
	CvMat* t_s = cvCreateMat(1, __cam.__textureDistributionModel.nPixels(), CV_64FC1);
	CvMat* t1 = cvCreateMat(1, __cam.__textureDistributionModel.nPixels(), CV_64FC1);
	CvMat* t2 = cvCreateMat(1, __cam.__textureDistributionModel.nPixels(), CV_64FC1);
	CvMat* u = cvCreateMat(1, __cam.nModes() + 4, CV_64FC1);
	CvMat* combinedParams = cvCreateMat(1, __cam.nModes(), CV_64FC1);
	CvMat* shapeInstance = cvCreateMat(1, __cam.__pointDistributionModel.nPoints() * 2, CV_64FC1);
	CvMat* q = cvCreateMat(1, 4, CV_64FC1);
	CvMat* p = cvCreateMat(1, __cam.__pointDistributionModel.nModes(), CV_64FC1);
	CvMat* lamda = cvCreateMat(1, __cam.__textureDistributionModel.nModes(), CV_64FC1);

	double theta = disp_angle * CV_PI / 180;
	double aa = MAX(fabs(disp_scale*cos(theta)), fabs(disp_scale*sin(theta)));
	cvmSet(d, 0, 0, aa); cvmSet(d, 0, 1, aa); cvmSet(d, 0, 2, disp_trans); cvmSet(d, 0, 3, disp_trans);
	for (int nmode = 0; nmode < __cam.nModes(); nmode++)
		cvmSet(d, 0, 4 + nmode, disp_std*sqrt(__cam.computeVariance(nmode)));

	srand(unsigned(time(0)));
	cvSetZero(u); cvSetZero(J);
	for (int i = 0; i < inputPoints.size(); i++) {
		IplImage* image = cvLoadImage(inputImages[i].c_str(), -1);
		ShapeModel Shape;
		if (!Shape.readData(inputPoints[i]))
			Shape.ScaleXY(image->width, image->height);
		Shape.Point2Mat(shapeInstance);
		Commons::checkIfShapeInBoundary(shapeInstance, image->width, image->height);

		//calculate current texture vector
		__cam.__piecewiseAffineWarp.computeWarpedTexture(shapeInstance, image, textureInstance);
		__cam.__textureDistributionModel.NormalizeTexture(__cam.__MeanG, textureInstance);

		__cam.__pointDistributionModel.computeParams(shapeInstance, p, q);
		__cam.__textureDistributionModel.computeParams(textureInstance, lamda);
		__cam.computeParams(combinedParams, p, lamda);

		CvMat subo;
		cvGetCols(o, &subo, 0, 4); cvCopy(q, &subo);
		cvGetCols(o, &subo, 4, 4 + __cam.nModes()); cvCopy(combinedParams, &subo);

		estimateResidual(image, o, shapeInstance, t_m, t_s, t1);
		for (int j = 0; j < nExp; j++) {
			printf("Pertubing (%d/%d) for image (%d/%d)...\r", j, nExp, i, inputPoints.size());
			for (int l = 0; l < 4 + __cam.nModes(); l++) {
				double D = cvmGet(d, 0, l);
				double v = rand_in_between(-D, D);
				cvCopy(o, oo); CV_MAT_ELEM(*oo, double, 0, l) += v;
				estimateResidual(image, oo, shapeInstance, t_m, t_s, t2);

				cvSub(t1, t2, t2);
				cvConvertScale(t2, t2, 1.0 / v);
				CvMat Jl; cvGetRow(J, &Jl, l);
				cvAdd(&Jl, t2, &Jl);
				CV_MAT_ELEM(*u, double, 0, l) += 1.0;
			}
		}
		cvReleaseImage(&image);
	}

	for (int l = 0; l < __cam.nModes() + 4; l++) {
		CvMat Jl; cvGetRow(J, &Jl, l);
		cvConvertScale(&Jl, &Jl, 1.0 / cvmGet(u, 0, l));
	}

	CvMat* JtJ = cvCreateMat(__cam.nModes() + 4, __cam.nModes() + 4, CV_64FC1);
	CvMat* InvJtJ = cvCreateMat(__cam.nModes() + 4, __cam.nModes() + 4, CV_64FC1);
	cvGEMM(J, J, 1, NULL, 0, JtJ, CV_GEMM_B_T);
	cvInvert(JtJ, InvJtJ, CV_SVD);
	cvMatMul(InvJtJ, J, __G);

	cvReleaseMat(&J); cvReleaseMat(&d); cvReleaseMat(&o); cvReleaseMat(&oo); cvReleaseMat(&textureInstance); cvReleaseMat(&t_s);
	cvReleaseMat(&t_m);cvReleaseMat(&t1); cvReleaseMat(&t2); cvReleaseMat(&u); cvReleaseMat(&combinedParams); cvReleaseMat(&shapeInstance);
	cvReleaseMat(&q); cvReleaseMat(&p); cvReleaseMat(&lamda); cvReleaseMat(&JtJ); cvReleaseMat(&InvJtJ);
}

double BasicAAModel::estimateResidual(const IplImage* image, const CvMat* c_q, CvMat* shapeInstance, CvMat* t_m, CvMat* t_s, CvMat* deltat) {
	CvMat combinedParams, q;
	cvGetCols(c_q, &q, 0, 4);
	cvGetCols(c_q, &combinedParams, 4, 4 + __cam.nModes());

	__cam.computeTextureFromLamdaValues(t_m, &combinedParams);
	__cam.computeShape(shapeInstance, c_q);
	Commons::checkIfShapeInBoundary(shapeInstance, image->width, image->height);
	__cam.__piecewiseAffineWarp.computeWarpedTexture(shapeInstance, image, t_s);
	__cam.__textureDistributionModel.NormalizeTexture(__cam.__MeanG, t_s);
	cvSub(t_m, t_s, deltat);
	return cvNorm(deltat);
}

void BasicAAModel::SetAllParamsZero() {
	cvSetZero(__poseParams);
	cvSetZero(__appearanceParams);
}

void BasicAAModel::initializeParameteres(const IplImage* image) {
	__cam.__pointDistributionModel.computeParams(__pointDistributionModel, __shapeParams, __poseParams);
	__cam.__piecewiseAffineWarp.computeWarpedTexture(__pointDistributionModel, image, __warpedTexture);
	__cam.__textureDistributionModel.NormalizeTexture(__cam.__MeanG, __warpedTexture);
	__cam.__textureDistributionModel.computeParams(__warpedTexture, __lamda);
	__cam.computeParams(__appearanceParams, __shapeParams, __lamda);
}

bool BasicAAModel::Fit(const IplImage* image, ShapeModel& shapeModel, int max_iter, bool showprocess) {

	double timeVal = gettime;
	double e1, e2;
	const int np = 5;
	double k_values[np] = { 1, 0.5, 0.25, 0.125, 0.0625 };
	int k;
	IplImage* Drawimg = 0;

	shapeModel.Point2Mat(__pointDistributionModel);
	initializeParameteres(image);
	CvMat subcq;
	cvGetCols(__curr_appearance_pose, &subcq, 0, 4); cvCopy(__poseParams, &subcq);
	cvGetCols(__curr_appearance_pose, &subcq, 4, 4 + __cam.nModes()); cvCopy(__appearanceParams, &subcq);
	e1 = estimateResidual(image, __curr_appearance_pose, __pointDistributionModel, __modelledTexture, __warpedTexture, __delta_warped_model_texture);

	for (int iter = 0; iter <max_iter; iter++) {
		bool converge = false;
		if (showprocess) {
			if (Drawimg == 0)	Drawimg = cvCloneImage(image);
			else cvCopy(image, Drawimg);
			__cam.computeShape(__pointDistributionModel, __curr_appearance_pose);
			Commons::checkIfShapeInBoundary(__pointDistributionModel, image->width, image->height);
			shapeModel.Mat2Point(__pointDistributionModel);
			Draw(Drawimg, shapeModel, 2);
			Commons::createDirectory("result");
			char filename[100];
			sprintf(filename, "result/Iter-%02d.jpg", iter);
			cvSaveImage(filename, Drawimg);
		}

		cvGEMM(__delta_warped_model_texture, __G, 1, NULL, 0, __delta_appearance_pose, CV_GEMM_B_T);

		if (iter == 0) {
			cvAdd(__curr_appearance_pose, __delta_appearance_pose, __curr_appearance_pose);
			CvMat combinedParams; cvGetCols(__curr_appearance_pose, &combinedParams, 4, 4 + __cam.nModes());
			__cam.Clamp(&combinedParams);
			e1 = estimateResidual(image, __curr_appearance_pose, __pointDistributionModel, __modelledTexture, __warpedTexture, __delta_warped_model_texture);
		} else {
			for (k = 0; k < np; k++) {
				cvScaleAdd(__delta_appearance_pose, cvScalar(k_values[k]), __curr_appearance_pose, __update_appearance_pose);
				CvMat combinedParams; cvGetCols(__update_appearance_pose, &combinedParams, 4, 4 + __cam.nModes());
				__cam.Clamp(&combinedParams);

				e2 = estimateResidual(image, __update_appearance_pose, __pointDistributionModel, __modelledTexture, __warpedTexture, __delta_warped_model_texture);
				if (e2 <= e1) {
					converge = true;
					break;
				}
			}

			if (converge) {
				e1 = e2;
				cvCopy(__update_appearance_pose, __curr_appearance_pose);
			}
			else
				break;
		}
	}


	cvReleaseImage(&Drawimg);
	__cam.computeShape(__pointDistributionModel, __curr_appearance_pose);
	Commons::checkIfShapeInBoundary(__pointDistributionModel, image->width, image->height);
	shapeModel.Mat2Point(__pointDistributionModel);
	timeVal = gettime - timeVal;
	LOGI("AAM-Basic Fitting: time cost=%.3f millisec, measure=%.2f\n", timeVal, e1);

	if (e1 >= 1.75) return false;
	return true;
}


void BasicAAModel::Draw(IplImage* image, const ShapeModel& Shape, int type) {
	if (type == 0) Commons::DrawPoints(image, Shape);
	else if (type == 1) Commons::DrawTriangles(image, Shape, __cam.__piecewiseAffineWarp.__triangles);
	else if (type == 2) {
		double minV, maxV;
		cvMinMaxLoc(__modelledTexture, &minV, &maxV);
		cvConvertScale(__modelledTexture, __modelledTexture, 255 / (maxV - minV), -minV * 255 / (maxV - minV));
		piecewiseAffineWarpAAModel paw;
		paw.trainModel(Shape, __cam.__Points, __cam.__Storage, __cam.__piecewiseAffineWarp.GetTri(), false);
		Commons::DrawAppearance(image, Shape, __modelledTexture, paw, __cam.__piecewiseAffineWarp);
	}
	else LOGW("ERROR(%s, %d): Unsupported drawing type\n", __FILE__, __LINE__);
}

void BasicAAModel::writeStream(std::ofstream& os) {
	__cam.writeStream(os);
	WriteCvMat(os, __G);
}

void BasicAAModel::readStream(std::ifstream& is) {
	__cam.readStream(is);
	__G = cvCreateMat(__cam.nModes() + 4, __cam.__textureDistributionModel.nPixels(), CV_64FC1);
	ReadCvMat(is, __G);

	__curr_appearance_pose = cvCreateMat(1, __cam.nModes() + 4, CV_64FC1);
	__update_appearance_pose = cvCreateMat(1, __cam.nModes() + 4, CV_64FC1);
	__delta_appearance_pose = cvCreateMat(1, __cam.nModes() + 4, CV_64FC1);
	__appearanceParams = cvCreateMat(1, __cam.nModes(), CV_64FC1);
	__shapeParams = cvCreateMat(1, __cam.__pointDistributionModel.nModes(), CV_64FC1);
	__poseParams = cvCreateMat(1, 4, CV_64FC1);
	__lamda = cvCreateMat(1, __cam.__textureDistributionModel.nModes(), CV_64FC1);
	__pointDistributionModel = cvCreateMat(1, __cam.__pointDistributionModel.nPoints() * 2, CV_64FC1);
	__warpedTexture = cvCreateMat(1, __cam.__textureDistributionModel.nPixels(), CV_64FC1);
	__modelledTexture = cvCreateMat(1, __cam.__textureDistributionModel.nPixels(), CV_64FC1);
	__delta_warped_model_texture = cvCreateMat(1, __cam.__textureDistributionModel.nPixels(), CV_64FC1);
}
