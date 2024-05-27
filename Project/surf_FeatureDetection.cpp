#include "stdafx.h"
#include "surf_FeatureDetection.h"
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace std;
using namespace cv;


Mat detectSurf(Mat& src_gray)
{
	/*
	int minHessian = 400;
	Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);
	vector<KeyPoint> keypoints;
	Mat descriptors;
	detector->detectAndCompute(src_gray, noArray(), keypoints, descriptors);
	Mat img_keypoints;
	drawKeypoints(src_gray, keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	return img_keypoints;
	*/
	Mat m;
	return m;
}
