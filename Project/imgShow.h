#include "stdafx.h"

#pragma once

/**
 * @brief Displays an image with detected features highlighted.
 *
 * @param title The title of the window.
 * @param dst The image to display.
 * @param corners The detected feature points.
 */
void showFeature(const string& title, Mat& dst, vector<Point2f> corners);

/**
 * @brief Displays an image.
 *
 * @param title The title of the window.
 * @param dst The image to display.
 */
void showFeature(const string& title, Mat& dst);

/**
 * @brief Displays an image with detected keypoints highlighted.
 *
 * @param title The title of the window.
 * @param dst The image to display.
 * @param keyPoints The detected keypoints.
 */
void showFeature(const string& title, Mat& dst, vector<KeyPoint> keyPoints);
