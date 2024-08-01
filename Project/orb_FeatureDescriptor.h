#include "stdafx.h"
#pragma once

/**
 * @brief Computes ORB descriptors for the given keypoints in an image.
 *
 * @param image The source image.
 * @param keyPoints The keypoints for which to compute the descriptors.
 * @return Mat The computed ORB descriptors.
 */
Mat orbDescriptors(const Mat& image, vector<KeyPoint>& keyPoints);
