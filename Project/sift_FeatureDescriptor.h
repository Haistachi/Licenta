#include "stdafx.h"
#pragma once

/**
 * @brief Computes SIFT descriptors for the given keypoints in an image.
 *
 * @param image The source image.
 * @param keyPoints The keypoints for which to compute the descriptors.
 * @return Mat The computed SIFT descriptors.
 */
Mat siftDescriptors(const Mat& image, vector<KeyPoint>& keyPoints);
