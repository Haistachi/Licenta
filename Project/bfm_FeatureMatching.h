#include "stdafx.h"
#pragma once

vector<DMatch> bfmFeatureMatchingEvaluated(Mat& src_gray1, Mat& src_gray2,
    Mat& descriptors1, Mat& descriptors2, string alg);