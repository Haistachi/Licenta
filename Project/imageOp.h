#include "stdafx.h"
#pragma once

using namespace cv;
using namespace std;

std::string openFileDialog();
Mat readImage();
void convertToGray(Mat& src, Mat& dst);
void limitKeyPoints(vector<KeyPoint> &keypoints, int maxKeypoints);