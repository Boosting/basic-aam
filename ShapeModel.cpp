#include "stdafx.h"
#include <string>
#include <iostream>
#include <fstream>

#include "ShapeModel.h"

using namespace std;

ShapeModel::ShapeModel(const ShapeModel &s) {
	CopyData(s);
}

ShapeModel& ShapeModel::operator=(const ShapeModel &s) {
	CopyData(s);
	return *this;
}

ShapeModel& ShapeModel::operator=(double value) {
	for (int i = 0, size = pointsVector.size(); i < size; i++)
	{
		pointsVector[i].x = value;
		pointsVector[i].y = value;
	}
	return *this;
}

ShapeModel ShapeModel::operator+(const ShapeModel &s)const {
	//    return ShapeModel(*this) += s;

	ShapeModel res(*this);
	for (int i = 0, size = pointsVector.size(); i < size; i++)
	{
		res.pointsVector[i].x += s.pointsVector[i].x;
		res.pointsVector[i].y += s.pointsVector[i].y;
	}
	return res;

}

ShapeModel& ShapeModel::operator+=(const ShapeModel &s) {
	for (int i = 0, size = pointsVector.size(); i < size; i++)
	{
		pointsVector[i].x += s.pointsVector[i].x;
		pointsVector[i].y += s.pointsVector[i].y;
	}
	return *this;
}


ShapeModel ShapeModel::operator-(const ShapeModel &s)const {
	//	return ShapeModel(*this) -= s;

	ShapeModel res(*this);
	for (int i = 0, size = pointsVector.size(); i < size; i++)
	{
		res.pointsVector[i].x -= s.pointsVector[i].x;
		res.pointsVector[i].y -= s.pointsVector[i].y;
	}
	return res;
}


ShapeModel& ShapeModel::operator-=(const ShapeModel &s) {
	for (int i = 0, size = pointsVector.size(); i < size; i++)
	{
		pointsVector[i].x -= s.pointsVector[i].x;
		pointsVector[i].y -= s.pointsVector[i].y;
	}
	return *this;
}


ShapeModel ShapeModel::operator*(double value)const {
	//	return ShapeModel(*this) *= value;

	ShapeModel res(*this);
	for (int i = 0, size = pointsVector.size(); i < size; i++)
	{
		res.pointsVector[i].x *= value;
		res.pointsVector[i].y *= value;
	}
	return res;
}


ShapeModel& ShapeModel::operator*=(double value) {
	for (int i = 0, size = pointsVector.size(); i < size; i++)
	{
		pointsVector[i].x *= value;
		pointsVector[i].y *= value;
	}
	return *this;
}


double ShapeModel::operator*(const ShapeModel &s)const {
	double result = 0.0;
	for (int i = 0, size = pointsVector.size(); i < size; i++)
	{
		result += pointsVector[i].x * s.pointsVector[i].x +
			pointsVector[i].y * s.pointsVector[i].y;
	}
	return result;
}


ShapeModel ShapeModel::operator/(double value)const {
	assert(value != 0);

	ShapeModel res(*this);
	for (int i = 0, size = pointsVector.size(); i < size; i++)
	{
		res.pointsVector[i].x /= value;
		res.pointsVector[i].y /= value;
	}
	return res;
}


ShapeModel& ShapeModel::operator/=(double value) {
	assert(value != 0);

	for (int i = 0, size = pointsVector.size(); i < size; i++)
	{
		pointsVector[i].x /= value;
		pointsVector[i].y /= value;
	}
	return *this;
}

bool ShapeModel::operator==(double value) {
	for (int i = 0, size = pointsVector.size(); i < size; i++)
	{
		if (fabs(pointsVector[i].x - value) > 1e-6)
			return false;

		if (fabs(pointsVector[i].y - value) > 1e-6)
			return false;
	}

	return true;
}

void ShapeModel::CopyData(const ShapeModel &s) {
	if (pointsVector.size() != s.pointsVector.size())
		pointsVector.resize(s.pointsVector.size());
	pointsVector = s.pointsVector;
}

double ShapeModel::GetNorm2()const {
	double norm = 0.0;

	// Normalize the vector to unit length, using the 2-norm.
	for (int i = 0, size = pointsVector.size(); i < size; i++)
	{
		norm += pow(pointsVector[i].x, 2);
		norm += pow(pointsVector[i].y, 2);
	}
	norm = sqrt(norm);
	return norm;
}

void ShapeModel::getCenterOfGravity(double &x, double &y)const {
	x = y = 0.0;

	for (int i = 0, size = pointsVector.size(); i < size; i++) {
		x=x+pointsVector[i].x;
		y=y+pointsVector[i].y;
	}
	x = x/pointsVector.size();
	y = y/pointsVector.size();
}

void ShapeModel::translateShape(double x, double y) {
	for (int i = 0, size = pointsVector.size(); i < size; i++) {
		pointsVector[i].x += x;
		pointsVector[i].y += y;
	}
}

void ShapeModel::centerShape() {
	double xSum, ySum;
	getCenterOfGravity(xSum, ySum);
	translateShape(-xSum, -ySum);	//Center the shape based on the center of gravity coordinates
}

void ShapeModel::Scale(double s) {
	for (int i = 0, size = pointsVector.size(); i < size; i++)
	{
		pointsVector[i].x *= s;
		pointsVector[i].y *= s;
	}
}

void ShapeModel::ScaleXY(double sx, double sy) {
	for (int i = 0, size = pointsVector.size(); i < size; i++) {
		pointsVector[i].x *= sx;
		pointsVector[i].y *= sy;
	}
}

double ShapeModel::Normalize() {
	double x, y;
	getCenterOfGravity(x, y);
	translateShape(-x, -y);
	double norm = GetNorm2();
	norm = (norm < 1e-10) ? 1 : norm;
	Scale(1. / norm);
	return norm;
}

void ShapeModel::applyTransformation(double c00, double c01, double c10, double c11) {
	double x, y;
	for (int i=0, size=pointsVector.size(); i<size; i++) {
		x = pointsVector[i].x;
		y = pointsVector[i].y;
		pointsVector[i].x = c00*x + c01*y;
		pointsVector[i].y = c10*x + c11*y;
	}
}

//----------------------------------------------------------------------
// Rotate Shape as theta
//----------------------------------------------------------------------
void ShapeModel::Rotate(double theta) {
	// set up transform matrix of rotation
	double c00 = cos(theta);
	double c01 = -sin(theta);
	double c10 = sin(theta);
	double c11 = cos(theta);

	applyTransformation(c00, c01, c10, c11);
}

void ShapeModel::getAlignTransformation(const ShapeModel &ref, double &a, double &b, double &tx, double &ty)const {

	double X1 = 0, Y1 = 0, X2 = 0, Y2 = 0, Z = 0, C1 = 0, C2 = 0;
	double W = pointsVector.size();
	double x1, y1, x2, y2;

	for (int i=0, size=pointsVector.size(); i < size; i++) {
		x1 = ref.pointsVector[i].x;
		y1 = ref.pointsVector[i].y;
		x2 = pointsVector[i].x;
		y2 = pointsVector[i].y;

		Z += x2 * x2 + y2 * y2;
		X1 += x1;
		Y1 += y1;
		X2 += x2;
		Y2 += y2;
		C1 += x1 * x2 + y1 * y2;
		C2 += y1 * x2 - x1 * y2;
	}

	{
		double SolnA[] = { X2, -Y2, W, 0, Y2, X2, 0, W, Z, 0, X2, Y2, 0, Z, -Y2, X2 };
		CvMat A = cvMat(4, 4, CV_64FC1, SolnA);
		double SolnB[] = { X1, Y1, C1, C2 };
		CvMat B = cvMat(4, 1, CV_64FC1, SolnB);

		static CvMat* Soln = cvCreateMat(4, 1, CV_64FC1);
		cvSolve(&A, &B, Soln, CV_SVD);

		a = CV_MAT_ELEM(*Soln, double, 0, 0);  b = CV_MAT_ELEM(*Soln, double, 1, 0);
		tx = CV_MAT_ELEM(*Soln, double, 2, 0);	 ty = CV_MAT_ELEM(*Soln, double, 3, 0);
	}
}

void ShapeModel::alignShapeTo(const ShapeModel &ref) {
	double a, b, tx, ty;
	getAlignTransformation(ref, a, b, tx, ty);
	transformShape(a, b, tx, ty);
}

void ShapeModel::transformShape(double a, double b, double tx, double ty) {
	applyTransformation(a, -b, b, a);
	translateShape(tx, ty);
}

const double ShapeModel::MinX()const {
	double val, min = 1.7E+308;

	for (int i = 0, size = pointsVector.size(); i < size; i++)
	{
		val = pointsVector[i].x;
		min = val<min ? val : min;
	}
	return min;
}

const double ShapeModel::MinY()const {
	double val, min = 1.7E+308;
	for (int i = 0, size = pointsVector.size(); i < size; i++)
	{
		val = pointsVector[i].y;
		min = val<min ? val : min;
	}
	return min;
}


const double ShapeModel::MaxX()const {
	double val, max = -1.7E+308;
	for (int i = 0, size = pointsVector.size(); i < size; i++)
	{
		val = pointsVector[i].x;
		max = val>max ? val : max;
	}
	return max;
}


const double ShapeModel::MaxY()const {
	double val, max = -1.7E+308;
	for (int i = 0, size = pointsVector.size(); i < size; i++)
	{
		val = pointsVector[i].y;
		max = val>max ? val : max;
	}
	return max;
}

void ShapeModel::Mat2Point(const CvMat* res) {
	int nPoints = res->cols / 2;
	double *B = (double*)(res->data.ptr + res->step * 0);
	if (pointsVector.size() != nPoints)		resize(nPoints);
	for (int i = 0; i < nPoints; i++)
	{
		pointsVector[i].x = B[2 * i];
		pointsVector[i].y = B[2 * i + 1];
	}
}

void ShapeModel::Point2Mat(CvMat* res)const {
	int nPoints = res->cols / 2;
	double *B = (double*)(res->data.ptr + res->step * 0);
	for (int i = 0; i < nPoints; i++)
	{
		B[2 * i] = pointsVector[i].x;
		B[2 * i + 1] = pointsVector[i].y;
	}
}

bool ShapeModel::readData(const std::string &filename) {
	bool isPointsFile;
	if (strstr(filename.c_str(), ".asf")) {
		isPointsFile = false;
		readASFfile(filename);
	}
	else if (strstr(filename.c_str(), ".pts")) {
		isPointsFile = true;
		readPointsFile(filename);
	}
	return isPointsFile;
}

void ShapeModel::readASFfile(const std::string &filename) {
	fstream fp;
	fp.open(filename.c_str(), ios::in);

	string temp;

	getline(fp, temp);
	getline(fp, temp);
	getline(fp, temp);
	getline(fp, temp);
	getline(fp, temp);
	getline(fp, temp);
	getline(fp, temp);
	getline(fp, temp);
	getline(fp, temp);
	getline(fp, temp);

	int pointsCount = atoi(temp.c_str());

	getline(fp, temp);
	getline(fp, temp);
	getline(fp, temp);
	getline(fp, temp);
	getline(fp, temp);
	getline(fp, temp);

	CvPoint2D32f tempCvPoint;
	pointsVector.clear();
	for (int i = 0; i < pointsCount; i++)
	{
		getline(fp, temp, ' ');
		getline(fp, temp, ' ');
		getline(fp, temp, ' ');
		// In DTU IMM , x means rows from left to right
		tempCvPoint.x = atof(temp.c_str());
		getline(fp, temp, ' ');
		// In DTU IMM , y means cols from top to bottom
		tempCvPoint.y = atof(temp.c_str());
		getline(fp, temp, ' ');
		getline(fp, temp, ' ');
		getline(fp, temp);
		// In sum, topleft is (0,0), right bottom is (640,480)
		pointsVector.push_back(tempCvPoint);
	}

	getline(fp, temp);
	getline(fp, temp);
	getline(fp, temp);
	getline(fp, temp);
	getline(fp, temp);
	fp.close();
}

void ShapeModel::readPointsFile(const std::string &filename) {
	fstream fp;
	fp.open(filename.c_str(), ios::in);
	string temp;
	getline(fp, temp);getline(fp, temp, ' ');getline(fp, temp);
	int pointsCount = atoi(temp.c_str());
	getline(fp, temp);

	CvPoint2D32f tempCvPoint;
	pointsVector.clear();
	for (int i=0; i<pointsCount; i++) {
		fp >> tempCvPoint.x >> tempCvPoint.y;
		pointsVector.push_back(tempCvPoint);
	}

	getline(fp, temp);
	fp.close();
}

void ShapeModel::readStream(std::ifstream& is) {
	for (int i = 0, nPoints = getPointsCount(); i < nPoints; i++) {
		float x, y;
		is >> x >> y;
		pointsVector[i] = cvPoint2D32f(x, y);
	}
	// is.read((char*)&pointsVector[i], sizeof(CvPoint2D32f));	// TODO -- CHANGE ??
}

void ShapeModel::writeStream(std::ofstream& os) {
	for (int i = 0, nPoints = getPointsCount(); i < nPoints; i++)
		os.write((char*)&pointsVector[i], sizeof(CvPoint2D32f));
}