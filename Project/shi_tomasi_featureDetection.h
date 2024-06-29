#include "stdafx.h"
#pragma once

std::vector<Point2f> detectShiTomasi(Mat& src_gray);
void showFeature(Mat dst, std::vector<Point2f> corners);
vector<KeyPoint> shiTomasiDetectKeyPoints(Mat& image);