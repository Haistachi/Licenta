#include "stdafx.h"
#pragma once

/**
 * @brief Matches features between two images using the FLANN-based matcher.
 *
 * @param src_gray1 The first source grayscale image.
 * @param src_gray2 The second source grayscale image.
 * @param descriptors1 The descriptors of the first image.
 * @param descriptors2 The descriptors of the second image.
 * @param keypoints1 The keypoints of the first image.
 * @param keypoints2 The keypoints of the second image.
 * @return vector<DMatch> The matched features.
 */
vector<DMatch> flannfeatureMatching(Mat& src_gray1, Mat& src_gray2,
    Mat& descriptors1, Mat& descriptors2,
    vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2);

/**
 * @brief Matches features between two images using the FLANN-based matcher.
 *
 * @param src_gray1 The first source grayscale image.
 * @param src_gray2 The second source grayscale image.
 * @param descriptors1 The descriptors of the first image.
 * @param descriptors2 The descriptors of the second image.
 * @return vector<DMatch> The matched features.
 */
vector<DMatch> flannfeatureMatching(Mat& src_gray1, Mat& src_gray2,
    Mat& descriptors1, Mat& descriptors2);
