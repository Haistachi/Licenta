#include "shift_FeatureDetection.h"
#include "stdafx.h"
#include "imageControl.h"

Mat detectShif(Mat& src_gray)
{
	Ptr<SIFT> detector = SIFT::create();
	vector<KeyPoint> keypoints;
	Mat descriptors;
	detector->detectAndCompute(src_gray, noArray(), keypoints, descriptors);

	//limitKeyPoints(keypoints, 100);

	Mat img_keypoints;
	drawKeypoints(src_gray, keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	return img_keypoints;
}

vector<KeyPoint> shifDetectKeyPoints(Mat& image)
{
	Ptr<SIFT> detector = SIFT::create();
	vector<KeyPoint> keypoints;
	Mat descriptors;
	detector->detectAndCompute(image, noArray(), keypoints, descriptors);
	return keypoints;
}