#include "stdafx.h"
#pragma once

Mat detectLogGabor(Mat& src_gray, double sig_fs, double lam);
void centering_transform(Mat img);
Mat fourierTransform(Mat& src_gray);