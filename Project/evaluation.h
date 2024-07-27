#include "stdafx.h"
#pragma once

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