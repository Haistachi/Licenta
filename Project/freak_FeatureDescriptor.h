#include "stdafx.h"
#pragma once

/**
 * @brief Computes FREAK descriptors for the given keypoints in an image.
 *
 * @param image The source image.
 * @param keyPoints The keypoints for which to compute the descriptors.
 * @return Mat The computed FREAK descriptors.
 */
Mat freakDescriptors(const Mat& image, vector<KeyPoint>& keyPoints);
