#include "stdafx.h"
#pragma once

using namespace cv;

std::string openFileDialog();
Mat readImage();
void convertToGray(Mat& src, Mat& dst);