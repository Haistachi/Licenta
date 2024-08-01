
#include "stdafx.h"  
#pragma once  

/**
 * @brief Detects corners in a grayscale image using the Shi-Tomasi corner detection method.
 *
 * @param src_gray The source grayscale image.
 * @return vector<Point2f> The detected Shi-Tomasi corners.
 */
vector<Point2f> detectShiTomasi(Mat& src_gray);

/**
 * @brief Displays an image with detected Shi-Tomasi corners highlighted.
 *
 * @param dst The image to display.
 * @param corners The detected Shi-Tomasi corners.
 */
void showFeature(Mat dst, std::vector<Point2f> corners);

/**
 * @brief Detects keypoints using the Shi-Tomasi corner detection method.
 *
 * @param image The source image.
 * @return vector<KeyPoint> The detected keypoints.
 */
vector<KeyPoint> shiTomasiDetectKeyPoints(Mat& image);
