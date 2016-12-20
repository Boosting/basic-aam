#include "stdafx.h"
#include "direct.h"
#include "dirent.h"
#include <fstream>
#include "Utils.h"
#include "BasicAAModel.h"

using namespace std;

void ReadCvMat(std::istream &is, CvMat* mat) {
	assert(CV_MAT_TYPE(mat->type) == CV_64FC1);
	double* p = mat->data.db;
	for (int i = 0; i < mat->rows*mat->cols; i++) {
		is >> p[i];
	}
	// is.read((char*)p, mat->rows*mat->cols * sizeof(double));
}

void WriteCvMat(std::ostream &os, const CvMat* mat) {
	assert(CV_MAT_TYPE(mat->type) == CV_64FC1);
	double* p = mat->data.db;
	os.write((char*)p, mat->rows*mat->cols * sizeof(double));
}

// Comparator function 
static int strComparator(const void *arg1, const void *arg2) {
	return strcmp((*(std::string*)arg1).c_str(), (*(std::string*)arg2).c_str());
}

#ifdef WIN32
#include <direct.h>
#include <io.h>
inputFiles Commons::parseInputDir(const std::string &dirPath, const std::string &fileExt) {
	WIN32_FIND_DATAA findFileData;
	HANDLE hFind;
	string searchDirPath, absFilePath;
	inputFiles inFiles;
	int fileCount = 0;

	searchDirPath = dirPath + "/*" + fileExt;
	hFind = FindFirstFileA(searchDirPath.c_str(), &findFileData);
	if (hFind == INVALID_HANDLE_VALUE) {
		printf("FindFirstFile failed (%d)\n", GetLastError());
		exit(0);
	}
	do {
		absFilePath = dirPath + "/" + (char*) findFileData.cFileName;
		inFiles.push_back(absFilePath);
		fileCount++;
	} while (FindNextFileA(hFind, &findFileData));

	FindClose(hFind);
	qsort((void *)&(inFiles[0]), (size_t)fileCount, sizeof(string), strComparator);
	LOGD("Input directory scanned. Absolute path of files loaded in memory ...\n");
	return inFiles;
}

int Commons::createDirectory(const char* dirname) {
	if (_access(dirname, 0)) { return _mkdir(dirname); }
	return 0;
}

#endif

void Commons::DrawPoints(IplImage* image, const ShapeModel& Shape) {
	for (int i = 0; i < Shape.getPointsCount(); i++) {
		cvCircle(image, cvPointFrom32f(Shape[i]), 3, CV_RGB(255, 0, 0));
	}
}

void Commons::DrawPointsInAlignedShape(IplImage* image, const ShapeModel& Shape) {
	for (int i = 0; i < Shape.getPointsCount(); i++) {
		//cvCircle(image, cvPointFrom32f(Shape[i]), 3, CV_RGB(255, 0, 0));
		cvCircle(image, cvPointFrom32f(cvPoint2D32f(Shape[i].x + 450, Shape[i].y + 300)), 3, CV_RGB(255, 0, 0));
	}
}

void Commons::DrawTriangles(IplImage* image, const ShapeModel& Shape, const std::vector<std::vector<int> >&tris) {
	int idx1, idx2, idx3;
	for (int i = 0; i < tris.size(); i++)
	{
		idx1 = tris[i][0];
		idx2 = tris[i][1];
		idx3 = tris[i][2];
		cvLine(image, cvPointFrom32f(Shape[idx1]), cvPointFrom32f(Shape[idx2]),
			CV_RGB(128, 255, 0));
		cvLine(image, cvPointFrom32f(Shape[idx2]), cvPointFrom32f(Shape[idx3]),
			CV_RGB(128, 255, 0));
		cvLine(image, cvPointFrom32f(Shape[idx3]), cvPointFrom32f(Shape[idx1]),
			CV_RGB(128, 255, 0));
	}
}

void Commons::DrawAppearance(IplImage*image, const ShapeModel& Shape, const CvMat* t, const piecewiseAffineWarpAAModel& paw, const piecewiseAffineWarpAAModel& refpaw) {
	int x1, x2, y1, y2, idx1, idx2;
	int xby3, idxby3;
	int minx, miny, maxx, maxy;
	int tri_idx, v1, v2, v3;
	byte* pimg;
	double* fastt = t->data.db;
	int nChannel = image->nChannels;
	int nPoints = Shape.getPointsCount();
	const ShapeModel& refShape = refpaw.__referenceshape;
	const std::vector<std::vector<int> >& tri = paw.__triangles;
	const std::vector<std::vector<int> >& rect1 = paw.__rect;
	const std::vector<std::vector<int> >& rect2 = refpaw.__rect;
	const std::vector<int>& pixTri = paw.__pixTri;
	const std::vector<double>& alpha = paw.__alpha;
	const std::vector<double>& belta = paw.__belta;
	const std::vector<double>& gamma = paw.__gamma;

	minx = Shape.MinX(); miny = Shape.MinY();
	maxx = Shape.MaxX(); maxy = Shape.MaxY();
	for (int y = miny; y < maxy; y++) {
		y1 = y - miny;
		pimg = (byte*)(image->imageData + image->widthStep*y);
		for (int x = minx; x < maxx; x++) {
			x1 = x - minx;
			idx1 = rect1[y1][x1];
			if (idx1 >= 0) {
				tri_idx = pixTri[idx1];
				v1 = tri[tri_idx][0];
				v2 = tri[tri_idx][1];
				v3 = tri[tri_idx][2];

				x2 = alpha[idx1] * refShape[v1].x + belta[idx1] * refShape[v2].x + gamma[idx1] * refShape[v3].x;
				y2 = alpha[idx1] * refShape[v1].y + belta[idx1] * refShape[v2].y + gamma[idx1] * refShape[v3].y;
				// int temp = rect2[y2].size;
				if (y2 < 0 || x2 < 0 || x2 >= refpaw.__width || y2 >= refpaw.__height) continue;
				idx2 = rect2[y2][x2];
				idxby3 = idx2 + (idx2 << 1);

				if (nChannel == 4) {
					xby3 = x << 2;
					pimg[xby3 + 2] = fastt[idxby3];
					pimg[xby3 + 1] = fastt[idxby3 + 1];
					pimg[xby3] = fastt[idxby3 + 2];
				} else if (nChannel == 3) {
					xby3 = x + (x << 1);
					pimg[xby3] = fastt[idxby3];
					pimg[xby3 + 1] = fastt[idxby3 + 1];
					pimg[xby3 + 2] = fastt[idxby3 + 2];
				} else {
					pimg[x] = (fastt[idxby3] + fastt[idxby3 + 1] + fastt[idxby3 + 2]) / 3;
				}
			}
		}
	}
}

void Commons::checkIfShapeInBoundary(CvMat* s, int w, int h) {
	double* fasts = s->data.db;
	int npoints = s->cols / 2;

	for (int i = 0; i < npoints; i++) {
		if (fasts[2 * i] > w - 1) fasts[2 * i] = w - 1;
		else if (fasts[2 * i] < 0) fasts[2 * i] = 0;

		if (fasts[2 * i + 1] > h - 1) fasts[2 * i + 1] = h - 1;
		else if (fasts[2 * i + 1] < 0) fasts[2 * i + 1] = 0;
	}
}

AAModel::AAModel() {}
AAModel::~AAModel() {}

void PyramidModel::BuildDetectMapping(const inputFiles& inputPoints, const inputFiles& inputImages, FaceDetector& FaceDetect, double refWidth) {
	printf("########################################################\n");
	printf("Building Detect Mapping ...\n");

	int total = 0;
	__referenceWidth = refWidth;
	int nPoints = GetMeanShape().getPointsCount();
	ShapeDetector.resize(nPoints);
	for (int i = 0; i < inputPoints.size(); i++)
	{
		printf("%i of %i\r", i, inputPoints.size());

		IplImage* image = cvLoadImage(inputImages[i].c_str(), -1);

		std::vector<ShapeModel> DetShape;
		bool flag = FaceDetect.DetectFace(DetShape, image);
		if (!flag) continue;

		ShapeModel Shape;
		flag = Shape.readData(inputPoints[i]);
		if (!flag)	Shape.ScaleXY(image->width, image->height);

		cvReleaseImage(&image);

		CvPoint2D32f  lt = DetShape[0][0], rb = DetShape[0][1];
		double x = (lt.x + rb.x) / 2., y = (lt.y + rb.y) / 2.;
		double w = (-lt.x + rb.x), h = (-lt.y + rb.y);

		Shape.translateShape(-x, -y);
		Shape.ScaleXY(__referenceWidth / w, __referenceWidth / h);

		ShapeDetector += Shape;
		total++;
	}
	ShapeDetector /= total;

	printf("########################################################\n");
}

void PyramidModel::initializeShapeFromDetectionBox(ShapeModel &Shape, const ShapeModel& detectionBox) {
	CvPoint2D32f  lt = detectionBox[0], RB = detectionBox[1];
	Shape = ShapeDetector;
	Shape.ScaleXY((-lt.x + RB.x) / __referenceWidth, (-lt.y + RB.y) / __referenceWidth);
	Shape.translateShape((lt.x + RB.x) / 2., (lt.y + RB.y) / 2.);
}

bool PyramidModel::initializeShapeFromImage(ShapeModel& Shape, FaceDetector& facedetect, const IplImage* image) {
	std::vector<ShapeModel> DetShape;

	bool flag = facedetect.DetectFace(DetShape, image);
	if (!flag)	return false;
	CvPoint2D32f  lt = DetShape[0][0], RB = DetShape[0][1];
	Shape = ShapeDetector;
	Shape.ScaleXY((-lt.x + RB.x) / __referenceWidth, (-lt.y + RB.y) / __referenceWidth);
	Shape.translateShape((lt.x + RB.x) / 2., (lt.y + RB.y) / 2.);
	return true;
}

PyramidModel::PyramidModel() {
	__model.resize(0);
}

PyramidModel::~PyramidModel() {
	for (int i = 0; i < __model.size(); i++)
		delete __model[i];
}

void PyramidModel::Build(const inputFiles& inputPoints, const inputFiles& inputImages, int type, int level) {
	__model.resize(0);
	LOGD("Starting Training ... \n");
	for (int i = 0; i < level; i++) {	//Build a multi-level AAM Pyramidal model
		__model.push_back(new BasicAAModel);
		LOGD("Building Active Appearance Model for level %d\n", i+1);
		__model[i]->Build(inputPoints, inputImages, 1.0 / pow(2.0, i));
	}
}

bool PyramidModel::Fit(const IplImage* image, ShapeModel& Shape, int max_iter, bool showprocess) {
	// the images used during search
	int w = image->width;
	int h = image->height;
	bool flag;

	double scale = __model[0]->GetReferenceShape().GetWidth() / Shape.GetWidth();
	Shape *= scale;
	int w0 = w*scale;
	int h0 = h*scale;

	int startlev = __model.size() - 1;
	int iter = max_iter / (startlev + 1);
	double PyrScale = pow(2.0, startlev);
	Shape /= PyrScale;

	// for each level in the image pyramid
	for (int iLev = startlev; iLev >= 0; iLev--) {
		LOGD("Level %d: \n", iLev);
		IplImage* fitimage = cvCreateImage(cvSize(w0 / PyrScale, h0 / PyrScale), image->depth, image->nChannels);
		cvResize(image, fitimage);
		ShapeModel shapeBackup = Shape;
		flag = __model[iLev]->Fit(fitimage, Shape, iter, showprocess);
		cvReleaseImage(&fitimage);
		if (!flag) Shape = shapeBackup;
		if (iLev != 0) {
			PyrScale /= 2.0;
			Shape *= 2.0;
		}
	}
	Shape /= scale;
	return flag;
}

const ShapeModel PyramidModel::GetMeanShape()const {
	return __model[0]->GetMeanShape();
}

void PyramidModel::Draw(IplImage* image, const ShapeModel& Shape, int type) {
	__model[0]->Draw(image, Shape, type);
}

bool PyramidModel::WriteModel(const std::string& filename) {
	ofstream outputStream(filename.c_str(), ios::out | ios::binary);
	if (!outputStream) {
		LOGW("ERROR(%s, %d): CANNOT create model \"%s\"\n", __FILE__, __LINE__, filename.c_str());
		return false;
	}

	int level = __model.size();
	int type = __model[0]->GetType();
	outputStream.write((char*)&type, sizeof(type));
	outputStream.write((char*)&level, sizeof(level));

	for (int i = 0; i < __model.size(); i++) {
		LOGD("Writing (level %d) active appearance model to file...\n", i);
		__model[i]->writeStream(outputStream);
	}
	LOGD("Done\n");

	outputStream.write((char*)&__referenceWidth, sizeof(__referenceWidth));
	ShapeDetector.writeStream(outputStream);
	outputStream.close();
	return true;
}

bool PyramidModel::ReadModel(const std::string& filename) {
	ifstream is(filename.c_str(), ios::in | ios::binary);
	if (!is) {
		LOGW("ERROR(%s, %d): CANNOT load model \"%s\"\n", __FILE__, __LINE__, filename.c_str());
		return false;
	}
			
	int level, type;
	is >> type >> level;
	//is.read((char*)&type, sizeof(type));
	//is.read((char*)&level, sizeof(level));

	for (int i = 0; i < level; i++) {
		__model.push_back(new BasicAAModel);
		LOGI("Reading (level %d) active appearance model from file...\n", i);
		__model[i]->readStream(is);
	}
	LOGI("Done\n");

	is >> __referenceWidth;
	// is.read((char*)&__referenceWidth, sizeof(__referenceWidth));
	ShapeDetector.resize(GetMeanShape().getPointsCount());
	ShapeDetector.readStream(is);
	is.close();

	return true;
}
