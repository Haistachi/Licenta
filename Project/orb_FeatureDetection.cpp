#include "stdafx.h"
#include "orb_FeatureDetection.h"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "imageControl.h"

Mat detectOrb(Mat& src_gray)
{
	Ptr<ORB> orb = ORB::create();
	vector<KeyPoint> keypoints;
	orb->detect(src_gray, keypoints);

	limitKeyPoints(keypoints, 100);

	Mat img_keypoints;
	drawKeypoints(src_gray, keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	return img_keypoints;
}

vector<KeyPoint> orbDetectKeyPoints(Mat& image)
{
	Ptr<ORB> orb = ORB::create();
	vector<KeyPoint> keypoints;
	orb->detect(image, keypoints);
	return keypoints;
}