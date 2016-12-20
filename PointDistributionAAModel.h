#ifndef PointDistribution_AAModel_H
#define PointDistribution_AAModel_H

#include "Utils.h"
#include "ShapeModel.h"

class CombinedAAModel;

//2D point distribution model
class PointDistributionAAModel
{
	friend class CombinedAAModel;
public:
	PointDistributionAAModel();
	~PointDistributionAAModel();

	// Build shape distribution model
	void trainModel(const std::vector<ShapeModel> &completeShapeList, double scale = 1.0, double percentage = 0.975);

	// Read data from stream 
	void readStream(std::ifstream& is);

	// write data to stream
	void writeStream(std::ofstream& os);

	// Calculate shape according to parameters p and q
	void computeLocalShape(const CvMat* p, CvMat* s);
	void computeGlobalShape(const CvMat* q, CvMat* s);
	void computeShape(const CvMat* p, const CvMat* q, CvMat* s);
	void computeShape(const CvMat* pq, CvMat* s);

	// Calculate parameters p and q according to shape 
	void computeParams(const CvMat* s, CvMat* p, CvMat* q);
	void computeParams(const CvMat* s, CvMat* pq);

	// Limit shape parameters.
	void Clamp(CvMat* p, double s_d = 3.0);

	// Get number of points in shape model
	inline const int nPoints()const { return __MeanShape->cols / 2; }

	// Get number of modes of shape variation
	inline const int nModes()const { return __ShapesEigenVectors->rows; }

	// Get mean shape
	inline const CvMat* GetMean()const { return __MeanShape; }
	inline const ShapeModel GetMeanShape()const { return __AAMRefShape; }

	// Get shape eigen-vectors of PCA (shape modes)
	inline const CvMat* GetBases()const { return __ShapesEigenVectors; }

	inline const double computeVariance(int i)const { return cvmGet(__ShapesEigenValues, 0, i); }

private:
	// Align shapes using procrustes analysis
	static void alignAllShapes(std::vector<ShapeModel> &completeShapeList);

	// Calculate mean shape of all shapes
	static void computeMeanShape(ShapeModel &MeanShape,
		const std::vector<ShapeModel> &completeShapeList);

	// Do PCA of shape data
	void computePCA(const CvMat* completeShapeList, double percentage);

private:

	CvMat*		__MeanShape;
	CvMat*		__ShapesEigenVectors;
	CvMat*		__ShapesEigenValues;
	ShapeModel	__AAMRefShape;

	CvMat*		__matshape;
	ShapeModel   __x;
	ShapeModel   __y;
};

#endif
