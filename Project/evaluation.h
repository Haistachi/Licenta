#include "stdafx.h"  
#pragma once  

// Evaluate the feature matching performance
// this functions requer a initial set of valid features and their matches
double calculateRecall(const std::vector<KeyPoint>& keypoints1, const std::vector<KeyPoint>& keypoints2,
	const std::vector<DMatch>& goodMatches, const Mat& homography, double epsilon_d);
double calculateSpreadOverlap(const std::vector<KeyPoint>& keypoints1, const std::vector<KeyPoint>& keypoints2,
	const std::vector<DMatch>& goodMatches, const Mat& homography, const Size& imageSize);
double calculateFeatureDistance(const  vector<KeyPoint>& keypoints1, const  vector<KeyPoint>& keypoints2,
	const  vector<DMatch>& goodMatches, const Mat& homography);
double calculateFeatureDensity(const  vector<KeyPoint>& keypoints1, const  vector<KeyPoint>& keypoints2,
	const  vector<DMatch>& goodMatches, const Mat& homography, const Size& imageSize);
double calculateFeatureMatchingScore(const  vector<KeyPoint>& keypoints1, const  vector<KeyPoint>& keypoints2,
	const  vector<DMatch>& goodMatches, const Mat& homography, const Size& imageSize);
double calculatePrecision(const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2,
	const vector<DMatch>& goodMatches, const Mat& homography, double epsilon_d);
double calculateF1Score(double recall, double precision);
void evaluateFeatureMatching(const Mat& img1, const Mat& img2, const Mat& homography,
	const  vector<KeyPoint>& keypoints1, const  vector<KeyPoint>& keypoints2,
	const  vector<DMatch>& goodMatches, const Size& imageSize);
void evaluateFeatureMatchingToFile(const Mat& img1, const Mat& img2, const Mat& homography,
	const  vector<KeyPoint>& keypoints1, const  vector<KeyPoint>& keypoints2,
	const  vector<DMatch>& goodMatches, const Size& imageSize);

// Evaluate the feature matching performance by using the homography matrix
// it dose not require a initial set of valid features and their matches
vector<DMatch> crossValidationCheck(const Mat& descriptors1, const Mat& descriptors2,
	const vector<DMatch>& matches1to2, const vector<DMatch>& matches2to1);
double calculateInlierRatio(const vector<DMatch>& goodMatches,
	const vector<Point2f>& points1, const vector<Point2f>& points2);
double calculateReprojectionError(const Mat& H,
	const vector<Point2f>& points1, const vector<Point2f>& points2);
vector<DMatch> filterMatchesByDistance(const vector<DMatch>& matches, double threshold);
vector<DMatch> filterMatchesByDynamicThreshold(const vector<DMatch>& matches);