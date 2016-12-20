#ifndef AAM_SHAPE_H
#define AAM_SHAPE_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include "opencv/cv.h"
#include "opencv/highgui.h"

class ShapeModel {
private:
	// points vector
	std::vector<CvPoint2D32f> pointsVector;

public:
	// Constructors and Destructor
	ShapeModel() { resize(0); }
	ShapeModel(const ShapeModel &v);
	~ShapeModel() { clear(); }

	//access elements
	const CvPoint2D32f operator[] (int i)const { return pointsVector[i]; }
	CvPoint2D32f& operator[] (int i) { return pointsVector[i]; }

	inline const int getPointsCount()const { return	pointsVector.size(); }

	// operators
	ShapeModel& operator=(const ShapeModel &s);
	ShapeModel& operator=(double value);
	ShapeModel operator+(const ShapeModel &s)const;
	ShapeModel& operator+=(const ShapeModel &s);
	ShapeModel operator-(const ShapeModel &s)const;
	ShapeModel& operator-=(const ShapeModel &s);
	ShapeModel operator*(double value)const;
	ShapeModel& operator*=(double value);
	double operator*(const ShapeModel &s)const;
	ShapeModel operator/(double value)const;
	ShapeModel& operator/=(double value);
	bool operator==(double value);

	void clear() { resize(0); }
	void resize(int length) { pointsVector.resize(length); }
	void readStream(std::ifstream& is);
	void writeStream(std::ofstream& os);
	bool readData(const std::string &filename);
	void readASFfile(const std::string &filename);
	void readPointsFile(const std::string &filename);

	const double MinX()const;
	const double MinY()const;
	const double MaxX()const;
	const double MaxY()const;
	inline const double GetWidth()const { return MaxX() - MinX(); }
	inline const double GetHeight()const { return MaxY() - MinY(); }

	// Transformations
	void getCenterOfGravity(double &x, double &y)const;
	void centerShape();
	void translateShape(double x, double y);
	void Scale(double s);
	void Rotate(double theta);
	void ScaleXY(double sx, double sy);
	double Normalize();

	// Align the shapes to reference shape 
	//													[a -b Tx]
	// returns the similarity transform: T(a,b,tx,ty) = [b  a Ty]
	//													[0  0  1]
	void getAlignTransformation(const ShapeModel &ref, double &a, double &b, double &tx, double &ty)const;

	// Align the shapes to reference shape as above, but no returns
	void alignShapeTo(const ShapeModel &ref);

	// Transform Shape using similarity transform T(a,b,tx,ty)
	void transformShape(double a, double b, double tx, double ty);

	// Euclidean norm
	double GetNorm2()const;

	// conversion between CvMat and ShapeModel
	void Mat2Point(const CvMat* res);
	void Point2Mat(CvMat* res)const;

private:
	void CopyData(const ShapeModel &s);
	void applyTransformation(double c00, double c01, double c10, double c11);
};

#endif