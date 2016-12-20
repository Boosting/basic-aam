#ifndef TEXTUREDISTRIBUTION_AAMODEL_H
#define TEXTUREDISTRIBUTION_AAMODEL_H

#include "Utils.h"
#include "ShapeModel.h"
#include "piecewiseAffineWarpAAModel.h"

class CombinedAAModel;

class textureDistributionAAModel {
	friend class CombinedAAModel;
public:
	textureDistributionAAModel();
	~textureDistributionAAModel();

	void trainModel(const inputFiles& inputPts, const inputFiles& inputImages,
		const piecewiseAffineWarpAAModel& m_warp, double textureEstimate = 0.975,
		bool registration = true);

	void readStream(std::ifstream& is);
	void writeStream(std::ofstream& os);
	void computeTextureFromLamdaValues(const CvMat* lamda, CvMat* t);
	void computeParams(const CvMat* t, CvMat* lamda);
	void Clamp(CvMat* lamda, double s_d = 3.0);
	static void ZeroMeanUnitLength(CvMat* Texture);
	static void NormalizeTexture(const CvMat* refTextrure, CvMat* Texture);
	inline const int nPixels()const { return __meanTextureMat->cols; }
	inline const int nModes()const { return __EigenVectorsOfTexture->rows; }
	inline const CvMat* GetMean()const { return __meanTextureMat; }
	inline const CvMat* GetBases()const { return __EigenVectorsOfTexture; }
	inline const double computeVariance(int i)const { return cvmGet(__EigenValsOfTexture, 0, i); }

private:

	void computePCA(const CvMat* AllTextures, double percentage);
	static void alignAllTextureMatrices(CvMat* AllTextures);
	static void computeMeanTexture(const CvMat* AllTextures, CvMat* meanTexture);
	void SaveSeriesTemplate(const CvMat* AllTextures, const piecewiseAffineWarpAAModel& m_warp);

private:

	CvMat* __meanTextureMat;
	CvMat* __EigenVectorsOfTexture;
	CvMat* __EigenValsOfTexture;
};

#endif // 
