#include "stdafx.h"  
#pragma once  

/**
 * @brief Calculates the recall of feature matching.
 *
 * @param keypoints1 The keypoints from the first image.
 * @param keypoints2 The keypoints from the second image.
 * @param goodMatches The good matches between keypoints.
 * @param homography The homography matrix.
 * @param epsilon_d The distance threshold for considering a match as correct.
 * @return double The recall value.
 */
double calculateRecall(const std::vector<KeyPoint>& keypoints1, const std::vector<KeyPoint>& keypoints2,
	const std::vector<DMatch>& goodMatches, const Mat& homography, double epsilon_d);

/**
 * @brief Calculates the spread overlap of feature matching.
 *
 * @param keypoints1 The keypoints from the first image.
 * @param keypoints2 The keypoints from the second image.
 * @param goodMatches The good matches between keypoints.
 * @param homography The homography matrix.
 * @param imageSize The size of the image.
 * @return double The spread overlap value.
 */
double calculateSpreadOverlap(const std::vector<KeyPoint>& keypoints1, const std::vector<KeyPoint>& keypoints2,
	const std::vector<DMatch>& goodMatches, const Mat& homography, const Size& imageSize);

/**
 * @brief Calculates the feature distance of feature matching.
 *
 * @param keypoints1 The keypoints from the first image.
 * @param keypoints2 The keypoints from the second image.
 * @param goodMatches The good matches between keypoints.
 * @param homography The homography matrix.
 * @return double The feature distance value.
 */
double calculateFeatureDistance(const  vector<KeyPoint>& keypoints1, const  vector<KeyPoint>& keypoints2,
	const  vector<DMatch>& goodMatches, const Mat& homography);

/**
 * @brief Calculates the feature density of feature matching.
 *
 * @param keypoints1 The keypoints from the first image.
 * @param keypoints2 The keypoints from the second image.
 * @param goodMatches The good matches between keypoints.
 * @param homography The homography matrix.
 * @param imageSize The size of the image.
 * @return double The feature density value.
 */
double calculateFeatureDensity(const  vector<KeyPoint>& keypoints1, const  vector<KeyPoint>& keypoints2,
	const  vector<DMatch>& goodMatches, const Mat& homography, const Size& imageSize);

/**
 * @brief Calculates the feature matching score.
 *
 * @param keypoints1 The keypoints from the first image.
 * @param keypoints2 The keypoints from the second image.
 * @param goodMatches The good matches between keypoints.
 * @param homography The homography matrix.
 * @param imageSize The size of the image.
 * @return double The feature matching score.
 */
double calculateFeatureMatchingScore(const  vector<KeyPoint>& keypoints1, const  vector<KeyPoint>& keypoints2,
	const  vector<DMatch>& goodMatches, const Mat& homography, const Size& imageSize);

/**
 * @brief Calculates the precision of feature matching.
 *
 * @param keypoints1 The keypoints from the first image.
 * @param keypoints2 The keypoints from the second image.
 * @param goodMatches The good matches between keypoints.
 * @param homography The homography matrix.
 * @param epsilon_d The distance threshold for considering a match as correct.
 * @return double The precision value.
 */
double calculatePrecision(const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2,
	const vector<DMatch>& goodMatches, const Mat& homography, double epsilon_d);

/**
 * @brief Calculates the F1 score based on recall and precision.
 *
 * @param recall The recall value.
 * @param precision The precision value.
 * @return double The F1 score.
 */
double calculateF1Score(double recall, double precision);

/**
 * @brief Evaluates feature matching and prints the results.
 *
 * @param img1 The first image.
 * @param img2 The second image.
 * @param homography The homography matrix.
 * @param keypoints1 The keypoints from the first image.
 * @param keypoints2 The keypoints from the second image.
 * @param goodMatches The good matches between keypoints.
 * @param imageSize The size of the image.
 */
void evaluateFeatureMatching(const Mat& img1, const Mat& img2, const Mat& homography,
	const  vector<KeyPoint>& keypoints1, const  vector<KeyPoint>& keypoints2,
	const  vector<DMatch>& goodMatches, const Size& imageSize);

/**
 * @brief Evaluates feature matching and writes the results to a file.
 *
 * @param img1 The first image.
 * @param img2 The second image.
 * @param homography The homography matrix.
 * @param keypoints1 The keypoints from the first image.
 * @param keypoints2 The keypoints from the second image.
 * @param goodMatches The good matches between keypoints.
 * @param imageSize The size of the image.
 */
void evaluateFeatureMatchingToFile(const Mat& img1, const Mat& img2, const Mat& homography,
	const  vector<KeyPoint>& keypoints1, const  vector<KeyPoint>& keypoints2,
	const  vector<DMatch>& goodMatches, const Size& imageSize);
