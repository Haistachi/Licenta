#include "stdafx.h"
#pragma once

/**
 * @brief Detects SIFT features in a grayscale image.
 *
 * @param src_gray The source grayscale image.
 * @return Mat The image with detected SIFT features.
 */
Mat detectSift(Mat& src_gray);

/**
 * @brief Detects keypoints using the SIFT algorithm.
 *
 * @param image The source image.
 * @return vector<KeyPoint> The detected keypoints.
 */
vector<KeyPoint> siftDetectKeyPoints(Mat& image);
