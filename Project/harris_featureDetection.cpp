#include "stdafx.h"
#include "harris_FeatureDetection.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp> 

void detectCornerHarris(Mat& src_gray, Mat& dst)
{
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
}

vector<Point2f> detectHarris(Mat& src_gray)
{
	int maxCorners = 100;
	double qualityLevel = 0.01;
	double minDistance = 10;
	int blockSize = 3, gradientSize = 3;
	bool useHarrisDetector = true;
	double k = 0.04;
	vector<Point2f> c;
	goodFeaturesToTrack(src_gray, c, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);

	return c;
}

vector<KeyPoint> harrisDetectKeyPoints(Mat& image)
{
	vector<Point2f> corners = detectHarris(image);
	vector<KeyPoint> keyPoints;
	for (Point2f corner : corners) {
		keyPoints.push_back(KeyPoint(corner, 1.f));
	}
	return keyPoints;
}