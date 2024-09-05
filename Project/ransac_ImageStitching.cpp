#include "stdafx.h"
#include "ransac_ImageStitching.h"
#include "imageOp.h"
#include "evaluation.h"


Mat ransacHomography(Mat& img1, Mat& img2, vector<DMatch> goodMatches,
	vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2)
{
	// Extract location of good matches
	vector<Point2f> points1, points2;
	for (size_t i = 0; i < goodMatches.size(); i++) {
		points1.push_back(keypoints1[goodMatches[i].queryIdx].pt);
		points2.push_back(keypoints2[goodMatches[i].trainIdx].pt);
	}

	double inlierRatio = calculateInlierRatio(goodMatches, points1, points2);

	// Find Homography using RANSAC
	Mat H = findHomography(points2, points1, RANSAC);

	double reprojectionError = calculateReprojectionError(H, points1, points2);
	
	return H;
}

Mat ransacHomography(Mat& img1, Mat& img2, vector<DMatch> goodMatches,
	vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2,
	double& inlierRatio, double& reprojectionError
	)
{
	// Extract location of good matches
	vector<Point2f> points1, points2;
	for (size_t i = 0; i < goodMatches.size(); i++) {
		points1.push_back(keypoints1[goodMatches[i].queryIdx].pt);
		points2.push_back(keypoints2[goodMatches[i].trainIdx].pt);
	}

	inlierRatio = calculateInlierRatio(goodMatches, points1, points2);

	// Find Homography using RANSAC
	Mat H = findHomography(points2, points1, RANSAC);

	reprojectionError = calculateReprojectionError(H, points1, points2);

	return H;
}