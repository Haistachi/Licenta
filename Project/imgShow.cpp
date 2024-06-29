#include "stdafx.h"
#include "imgShow.h"

using namespace cv;
using namespace std;

void showFeature(const string& title, Mat dst, vector<Point2f> corners)
{
	RNG rng(12345);
	cout << "** Number of corners detected: " << corners.size() << endl;
	int radius = 4;
	for (size_t i = 0; i < corners.size(); i++)
	{
		circle(dst, corners[i], radius, Scalar(rng.uniform(0, 255), rng.uniform(0, 256), rng.uniform(0, 256)), FILLED);
	}

	imshow(title, dst);
}

void showFeature(const string& title, Mat& dst)
{
	Mat dst_norm, dst_norm_scaled;
	int thresh = 200;

	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);
	imshow(title, dst_norm_scaled);
}

void showFeature(const string& title, Mat& dst, vector<KeyPoint> keyPoints)
{
	Mat img_keypoints;
	drawKeypoints(dst, keyPoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow(title, img_keypoints);
}