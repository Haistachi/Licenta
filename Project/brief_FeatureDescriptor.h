#include "stdafx.h"
#pragma once

/**
 * @brief Computes BRIEF descriptors for the given keypoints in an image.
 *
 * @param image The source image.
 * @param keyPoints The keypoints for which to compute the descriptors.
 * @return Mat The computed BRIEF descriptors.
 */
Mat briefDescriptors(const Mat& image, vector<KeyPoint>& keyPoints);
