#include "stdafx.h"
#include "warpPerspective&Blending_ImageStitching.h"

Mat warpPerspectiveAndBlend(Mat& img1, Mat& img2, Mat& homography)
{
    Mat result;
    warpPerspective(img2, result, homography, Size(img1.cols + img2.cols, img1.rows));
    Mat half(result, Rect(0, 0, img1.cols, img1.rows));
    img1.copyTo(half);
    return result;
}