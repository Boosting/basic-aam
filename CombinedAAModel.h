#ifndef Combined_AAModel_H
#define Combined_AAModel_H

#include "textureDistributionAAModel.h"
#include "PointDistributionAAModel.h"

class BasicAAModel;

//combined appearance model
class CombinedAAModel
{
	friend class BasicAAModel;
public:
	CombinedAAModel();
	~CombinedAAModel();

	void trainModel(const inputFiles& inputPoints, const inputFiles& inputImages, double scale = 1.0, double shapeEstimate = 0.975,
		double textureEstimate = 0.975, double appearanceEstimate = 0.975);

	inline const int nParameters()const { return __AppearanceEigenVectors->cols; }
	inline const int nModes()const { return __AppearanceEigenVectors->rows; }
	inline double computeVariance(int i)const { return cvmGet(__AppearanceEigenValues, 0, i); }
	inline const CvMat* GetMean()const { return __MeanAppearance; }
	inline const CvMat* GetBases()const { return __AppearanceEigenVectors; }

	void ShowVariation();
	friend void ontrackcam(int pos);
	void DrawAppearance(IplImage* image, const ShapeModel& Shape, CvMat* Texture);
	void computeLocalShape(CvMat* shapeInstance, const CvMat* combinedParams);
	void computeGlobalShape(CvMat* shapeInstance, const CvMat* pose);
	inline void computeShape(CvMat* shapeInstance, const CvMat* c_q) {
		CvMat combinedParams; cvGetCols(c_q, &combinedParams, 4, 4 + nModes()); computeLocalShape(shapeInstance, &combinedParams);
		CvMat q; cvGetCols(c_q, &q, 0, 4); computeGlobalShape(shapeInstance, &q);
	}
	inline void computeShape(CvMat* shapeInstance, const CvMat* combinedParams, const CvMat* pose) {
		computeLocalShape(shapeInstance, combinedParams); computeGlobalShape(shapeInstance, pose);
	}

	void computeTextureFromLamdaValues(CvMat* textureInstance, const CvMat* combinedParams);	// Computing texture based on appearance parameters
	void computeParams(CvMat* combinedParams, const CvMat* p, const CvMat* lamda);	//compute combined appearance parameters
	void Clamp(CvMat* combinedParams, double s_d = 3.0);
	void readStream(std::ifstream& is);
	void writeStream(std::ofstream& os);

private:
	void computePCA(const CvMat* AllAppearances, double percentage);
	void ShapeTexture2Combined(const CvMat* Shape, const CvMat* Texture, CvMat* Appearance);

private:
	PointDistributionAAModel __pointDistributionModel;
	textureDistributionAAModel	__textureDistributionModel;
	piecewiseAffineWarpAAModel	__piecewiseAffineWarp;
	double __ratio_shapes_texture;
	CvMat* __MeanAppearance;
	CvMat* __AppearanceEigenValues;
	CvMat* __AppearanceEigenVectors;
	CvMat* __Qs;
	CvMat* __Qg;
	CvMat* __MeanS;
	CvMat* __MeanG;
	CvMat* __a;
	CvMat* __Points;
	CvMemStorage* __Storage;
	CvMat* __pq;
};

#endif
