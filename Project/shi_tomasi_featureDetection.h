#include "stdafx.h"  
#pragma once  

vector<Point2f> detectShiTomasi(Mat& src_gray);
vector<KeyPoint> shiTomasiDetectKeyPoints(Mat& image);
