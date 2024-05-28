#include "stdafx.h"
#include "shi_tomasi_FeatureDetection.h"

using namespace cv;
using namespace std;

vector<Point2f> detectShiTomasi(Mat& src_gray)
{
	int maxCorners = 100;
	double qualityLevel = 0.01;
	double minDistance = 10;
	int blockSize = 3, gradientSize = 3;
	bool useHarrisDetector = false;
	double k = 0.04;
	vector<Point2f> c;
	goodFeaturesToTrack(src_gray, c, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
	return c;
}

void showFeature(Mat dst, vector<Point2f> corners)
{
	RNG rng(12345);
	cout << "** Number of corners detected: " << corners.size() << endl;
	int radius = 4;
	for (size_t i = 0; i < corners.size(); i++)
	{
		circle(dst, corners[i], radius, Scalar(rng.uniform(0, 255), rng.uniform(0, 256), rng.uniform(0, 256)), FILLED);
	}

	imshow("source_window", dst);
}