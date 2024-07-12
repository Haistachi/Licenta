#include "stdafx.h"
#pragma once

Mat ransacStitchImages(Mat& img1, Mat& img2, vector<cv::DMatch> goodMatches,
	vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2);