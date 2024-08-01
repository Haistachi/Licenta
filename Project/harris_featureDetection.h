#include "stdafx.h"
#pragma once

/**
 * @brief Detects corners in a grayscale image using the Harris corner detection method.
 *
 * @param src_gray The source grayscale image.
 * @param dst The destination image with detected corners.
 */
void detectCornerHarris(Mat& src_gray, Mat& dst);

/**
 * @brief Detects Harris corners in a grayscale image.
 *
 * @param src_gray The source grayscale image.
 * @return vector<Point2f> The detected Harris corners.
 */
vector<Point2f> detectHarris(Mat& src_gray);

/**
 * @brief Detects keypoints using the Harris corner detection method.
 *
 * @param image The source image.
 * @return vector<KeyPoint> The detected keypoints.
 */
vector<KeyPoint> harrisDetectKeyPoints(Mat& image);
