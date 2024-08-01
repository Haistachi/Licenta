#include "stdafx.h"
#pragma once

/**
 * @brief Warps the perspective of the second image to align with the first image using the provided homography matrix,
 *        and blends the two images together.
 *
 * @param img1 The first image.
 * @param img2 The second image.
 * @param homography The homography matrix used to warp the perspective of the second image.
 * @return Mat The resulting image after warping and blending.
 */
Mat warpPerspectiveAndBlend(Mat& img1, Mat& img2, Mat& homography);
