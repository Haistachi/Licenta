#include "stdafx.h"
#pragma once

vector<Point2f> detectShiTomasi(Mat& src_gray);
void showFeature(Mat dst, std::vector<Point2f> corners);
vector<KeyPoint> shiTomasiDetectKeyPoints(Mat& image);