#include "stdafx.h"
#include "ransac_ImageStitching.h"

Mat ransacStitchImages(Mat& img1, Mat& img2, vector<cv::DMatch> goodMatches,
	vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2)
{
	// Extract location of good matches
	vector<Point2f> points1, points2;
	for (size_t i = 0; i < goodMatches.size(); i++) {
		points1.push_back(keypoints1[goodMatches[i].queryIdx].pt);
		points2.push_back(keypoints2[goodMatches[i].trainIdx].pt);
	}

	// Find Homography using RANSAC
	Mat H = findHomography(points1, points2, RANSAC);

	// Show the stitched image
	imshow("Homography", H);
	waitKey(0);
	return H;
}