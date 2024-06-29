#include "stdafx.h"
#pragma once

Mat detectOrb(Mat& src_gray);
vector<KeyPoint> orbDetectKeyPoints(Mat& image);