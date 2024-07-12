#include "stdafx.h"
#include "warpPerspective&Blending_ImageStitching.h"

Mat warpPerspectiveAndBlend(Mat& img1, Mat& img2, Mat& homography)
{
	Mat img1_warped;
	warpPerspective(img1, img1_warped, homography, Size(img1.cols + img2.cols, img1.rows));
	Mat half(img1_warped, Rect(0, 0, img2.cols, img2.rows));
	img2.copyTo(half);

	for (int y = 0; y < img1_warped.rows; y++) {
		for (int x = 0; x < img1_warped.cols; x++) {
			if (img1_warped.at<Vec3b>(y, x) == Vec3b(0, 0, 0)) {
				img1_warped.at<Vec3b>(y, x) = img2.at<Vec3b>(y, x);
			}
		}
	}

	imshow("Stitched Image", img1_warped);
	waitKey(0);
	return img1_warped;
}