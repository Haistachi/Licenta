#include "stdafx.h"
#pragma once

/**
 * @brief Stitches two images together using RANSAC to find the best homography.
 *
 * @param img1 The first image.
 * @param img2 The second image.
 * @param goodMatches The good matches between keypoints in the two images.
 * @param keypoints1 The keypoints from the first image.
 * @param keypoints2 The keypoints from the second image.
 * @return Mat The stitched image.
 */
Mat ransacStitchImages(Mat& img1, Mat& img2, vector<cv::DMatch> goodMatches,
    vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2);
