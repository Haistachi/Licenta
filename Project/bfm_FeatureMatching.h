#include "stdafx.h"
#pragma once

vector<DMatch> bfmFeatureMatching(Mat& src_gray1, Mat& src_gray2,
	Mat& descriptors1, Mat& descriptors2,
	vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, string alg);
vector<DMatch> bfmFeatureMatching(Mat& src_gray1, Mat& src_gray2,
	Mat& descriptors1, Mat& descriptors2, string alg);