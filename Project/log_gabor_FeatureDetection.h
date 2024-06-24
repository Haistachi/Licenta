#include "stdafx.h"
#pragma once

Mat detectLogGabor(Mat& src_gray, double sig_fs, double lam, double theta_o);
Mat testLogGaborFillter(Mat& src_gray);
Mat detectLogGaborV2(Mat& src_gray, double sig_fs, double lam, double theta_o);
vector<Mat> logGaborFilterBank(int rows, int cols, int numScales, int numOrientations);
void applyFilterBankToImage(const Mat& graySrc, int numScales, int numOrientations);