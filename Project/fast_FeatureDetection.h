#include "stdafx.h"
#pragma once

/**
 * @brief Detects FAST features in a grayscale image.
 *
 * @param src_gray The source grayscale image.
 * @return Mat The image with detected FAST features.
 */
Mat detectFast(Mat& src_gray);

/**
 * @brief Detects keypoints using the FAST algorithm.
 *
 * @param src_gray The source grayscale image.
 * @return vector<KeyPoint> The detected keypoints.
 */
vector<KeyPoint> fastDetectKeyPoints(Mat& src_gray);
