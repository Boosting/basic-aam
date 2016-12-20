#ifndef BASIC_AAMODEL_H
#define BASIC_AAMODEL_H

#include "Utils.h"
#include "textureDistributionAAModel.h"
#include "CombinedAAModel.h"

class BasicAAModel : public AAModel {
public:
	BasicAAModel();
	~BasicAAModel();

	virtual const int GetType()const { return 0; }

	void trainModel(const inputFiles& inputPoints,
		const inputFiles& inputImages,
		double scale = 1.0, double shapeEstimate = 0.975, double texturEstimate = 0.975, double appearanceEstimate = 0.975);
	virtual void Build(const inputFiles& inputPoints, const inputFiles& inputImages, double scale = 1.0) {
		trainModel(inputPoints, inputImages, scale);
	}

	virtual bool Fit(const IplImage* image, ShapeModel& Shape, int max_iter = 30, bool showprocess = false);
	virtual const ShapeModel GetMeanShape()const { return __cam.__pointDistributionModel.GetMeanShape(); }
	const ShapeModel GetReferenceShape()const { return __cam.__piecewiseAffineWarp.__referenceshape; }

	virtual void Draw(IplImage* image, const ShapeModel& Shape, int type);
	virtual void readStream(std::ifstream& is);
	virtual void writeStream(std::ofstream& os);
	virtual void SetAllParamsZero();
	virtual void initializeParameteres(const IplImage* image);

private:

	double estimateResidual(const IplImage* image, const CvMat* c_q, CvMat* shapeInstance, CvMat* t_m, CvMat* t_s, CvMat* deltat);
	void ComputeJacobian(const inputFiles& inputPoints,
		const inputFiles& inputImages, double disp_scale = 0.2, double disp_angle = 20, double disp_trans = 5.0, double disp_std = 1.0, int nExp = 30);

private:
	CombinedAAModel __cam;
	CvMat*  __G;
	CvMat*  __curr_appearance_pose;
	CvMat*  __update_appearance_pose;
	CvMat*  __delta_appearance_pose;
	CvMat*	__appearanceParams;
	CvMat*	__shapeParams;
	CvMat*	__poseParams;
	CvMat*	__lamda;
	CvMat*	__pointDistributionModel;
	CvMat*	__modelledTexture;
	CvMat*	__warpedTexture;
	CvMat*	__delta_warped_model_texture;
};

#endif