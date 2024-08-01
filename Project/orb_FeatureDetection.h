#include "stdafx.h"
#pragma once

/**
 * @brief Detects ORB features in a grayscale image.
 *
 * @param src_gray The source grayscale image.
 * @return Mat The image with detected ORB features.
 */
Mat detectOrb(Mat& src_gray);

/**
 * @brief Detects keypoints using the ORB algorithm.
 *
 * @param image The source image.
 * @return vector<KeyPoint> The detected keypoints.
 */
vector<KeyPoint> orbDetectKeyPoints(Mat& image);
