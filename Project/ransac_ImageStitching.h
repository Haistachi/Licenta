#include "stdafx.h"
#pragma once

Mat ransacHomography(Mat& img1, Mat& img2, vector<cv::DMatch> goodMatches,
    vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2);
Mat ransacHomography(Mat& img1, Mat& img2, vector<DMatch> goodMatches,
	vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2,
	double& inlierRatio, double& reprojectionError);