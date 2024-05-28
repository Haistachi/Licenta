#include "stdafx.h"
#include "orb_FeatureDetection.h"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;


Mat detectOrb(Mat& src_gray)
{
	Ptr<ORB> orb = ORB::create();
	vector<KeyPoint> keypoints;
	orb->detect(src_gray, keypoints);
	Mat descriptors;
	orb->compute(src_gray, keypoints, descriptors);

	sort(keypoints.begin(), keypoints.end(), [](const KeyPoint& a, const KeyPoint& b) {
		return a.response > b.response;
		});
	int maxKeypoints = 100; // Set the maximum number of keypoints you want
	if (keypoints.size() > maxKeypoints) {
		keypoints.resize(maxKeypoints);
		descriptors = descriptors.rowRange(0, maxKeypoints);
	}

	Mat img_keypoints;
	drawKeypoints(src_gray, keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	return img_keypoints;
}
