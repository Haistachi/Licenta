#include "stdafx.h"
#pragma once

Mat detectLogGabor(Mat& src_gray, double angl, double sig_fs, double lam, double theta_o);
vector<Mat> logGaborFilterBank(int rows, int cols, int numScales, int numOrientations);
void applyFilterBankToImage(const Mat& graySrc, int numScales, int numOrientations);
vector<KeyPoint> logGaborDetectKeyPoints(Mat& image);