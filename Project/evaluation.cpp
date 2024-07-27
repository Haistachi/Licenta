#include "stdafx.h"
#include "evaluation.h"
#include <fstream>

/*
* calculateRecall
* Definition: Recall is the ratio of correctly matched keypoints (inliers) to the total number of matches.
* Formula: Recall = Nm / Nc, where Nm is the number of inliers and Nc is the total number of matches.
* Interpretation:
* Low Recall: Indicates that a low proportion of the matches are correct.
* High Recall: Indicates that a high proportion of the matches are correct.
* epsilon_d: The maximum distance between a matched keypoint and its corresponding projected keypoint to be considered an inlier.
* recomended recall score: above 0.7 (70%)
*/
double calculateRecall(const  vector<KeyPoint>& keypoints1, const  vector<KeyPoint>& keypoints2,
    const  vector<DMatch>& goodMatches, const Mat& homography, double epsilon_d)
{
    int Nm = 0; // Number of inliers
    int Nc = goodMatches.size(); // Number of corresponding points

    for (const auto& match : goodMatches) {
        Point2f pt1 = keypoints1[match.queryIdx].pt;
        Point2f pt2 = keypoints2[match.trainIdx].pt;

        // Project pt2 to pt1 using the homography
         vector<Point2f> pts2 = { pt2 };
         vector<Point2f> projectedPts;
        perspectiveTransform(pts2, projectedPts, homography);

        // Calculate the distance between pt1 and the projected point
        double distance = norm(pt1 - projectedPts[0]);

        if (distance < epsilon_d) {
            Nm++;
        }
    }
    double recall = static_cast<double>(Nm) / static_cast<double>(Nc);
    return recall;
}

/*	
*	calculateSpreadOverlap
* Definition: Spread Overlap is the ratio of the number of grid cells that contain at least one correct match to the total number of grid cells.
* Formula: Spread Overlap = numValidCells / totalCells, where numValidCells is the number of grid cells that contain at least one correct match and totalCells is the total number of grid cells.
* Interpretation:
* Low Spread Overlap: Indicates that the correct matches are clustered in a few regions of the image.
* High Spread Overlap: Indicates that the correct matches are spread across the image.
* Recomended spread overlap score: above 0.6 (60%)
*/
double calculateSpreadOverlap(const  vector<KeyPoint>& keypoints1, const  vector<KeyPoint>& keypoints2,
    const  vector<DMatch>& goodMatches, const Mat& homography, const Size& imageSize) {

    // Calculate the grid cell size (0.25% of the total image area)
    double cellArea = 0.0025 * imageSize.area();
    int cellSize = static_cast<int>(sqrt(cellArea));
    int numCellsX = imageSize.width / cellSize;
    int numCellsY = imageSize.height / cellSize;
    int totalCells = numCellsX * numCellsY;

     set<pair<int, int>> validCells;

    for (const auto& match : goodMatches) {
        Point2f pt1 = keypoints1[match.queryIdx].pt;
        Point2f pt2 = keypoints2[match.trainIdx].pt;

        // Project pt2 to pt1 using the homography
         vector<Point2f> pts2 = { pt2 };
         vector<Point2f> projectedPts;
        perspectiveTransform(pts2, projectedPts, homography);

        Point2f projectedPt = projectedPts[0];
        if (projectedPt.x >= 0 && projectedPt.x < imageSize.width && projectedPt.y >= 0 && projectedPt.y < imageSize.height) {
            int cellX = static_cast<int>(projectedPt.x) / cellSize;
            int cellY = static_cast<int>(projectedPt.y) / cellSize;
            validCells.insert({ cellX, cellY });
        }
    }

    double numValidCells = validCells.size();
    double spreadOverlap = numValidCells / totalCells;
    return spreadOverlap;
}

/*	
*	calculateFeatureDistance
* Definition: Feature Distance is the average distance between the matched keypoints and their corresponding projected keypoints.
* Formula: Feature Distance = totalDistance / numMatches, where totalDistance is the sum of the distances between the matched keypoints and their corresponding projected keypoints, and numMatches is the total number of matches.
* Interpretation:
* Low Feature Distance: Indicates that the matched keypoints are close to their corresponding projected keypoints.
* High Feature Distance: Indicates that the matched keypoints are far from their corresponding projected keypoints.
* Recommended feature distance score: below 5.0 pixels
*/
double calculateFeatureDistance(const  vector<KeyPoint>& keypoints1, const  vector<KeyPoint>& keypoints2,
	const  vector<DMatch>& goodMatches, const Mat& homography) {

	
	double totalDistance = 0.0;
	int numMatches = 0;

	for (const auto& match : goodMatches) {
		Point2f pt1 = keypoints1[match.queryIdx].pt;
		Point2f pt2 = keypoints2[match.trainIdx].pt;

		// Project pt2 to pt1 using the homography
		 vector<Point2f> pts2 = { pt2 };
		 vector<Point2f> projectedPts;
		perspectiveTransform(pts2, projectedPts, homography);

		Point2f projectedPt = projectedPts[0];
		double distance = norm(pt1 - projectedPt);
		totalDistance += distance;
		numMatches++;
	}

	double featureDistance = totalDistance / numMatches;
	return featureDistance;
}

/*
* calculateFeatureDensity
* Definition: Feature Density is the ratio of the number of grid cells that contain at least one correct match to the total number of grid cells.
* Formula: Feature Density = numValidCells / totalCells, where numValidCells is the number of grid cells that contain at least one correct match and totalCells is the total number of grid cells.
* Interpretation:
* Low Feature Density: Indicates that the correct matches are clustered in a few regions of the image.
* High Feature Density: Indicates that the correct matches are spread across the image.
* Recommended feature density score: above 0.6 (60%)
*/
double calculateFeatureDensity(const  vector<KeyPoint>& keypoints1, const  vector<KeyPoint>& keypoints2,
	const  vector<DMatch>& goodMatches, const Mat& homography, const Size& imageSize) {
	// Calculate the grid cell size (0.25% of the total image area)
	double cellArea = 0.0025 * imageSize.area();
	int cellSize = static_cast<int>(sqrt(cellArea));
	int numCellsX = imageSize.width / cellSize;
	int numCellsY = imageSize.height / cellSize;
	int totalCells = numCellsX * numCellsY;

	 set<pair<int, int>> validCells;

	for (const auto& match : goodMatches) {
		Point2f pt1 = keypoints1[match.queryIdx].pt;
		Point2f pt2 = keypoints2[match.trainIdx].pt;

		// Project pt2 to pt1 using the homography
		 vector<Point2f> pts2 = { pt2 };
		 vector<Point2f> projectedPts;
		perspectiveTransform(pts2, projectedPts, homography);

		Point2f projectedPt = projectedPts[0];
		if (projectedPt.x >= 0 && projectedPt.x < imageSize.width && projectedPt.y >= 0 && projectedPt.y < imageSize.height) {
			int cellX = static_cast<int>(projectedPt.x) / cellSize;
			int cellY = static_cast<int>(projectedPt.y) / cellSize;
			validCells.insert({ cellX, cellY });
		}
	}

	double numValidCells = validCells.size();
	double featureDensity = numValidCells / totalCells;
	return featureDensity;
}

/*
* calculateFeatureMatchingScore
* Definition: Feature Matching Score is a weighted combination of Recall, Spread Overlap, Feature Distance, and Feature Density.
* Formula: Score = 0.5 * Recall + 0.3 * Spread Overlap + 0.1 * Feature Distance + 0.1 * Feature Density
* Interpretation:
* High Score: Indicates that the feature matching is accurate, the correct matches are spread across the image, the matched keypoints are close to their corresponding projected keypoints, and the correct matches are spread across the image.
* Low Score: Indicates that the feature matching is inaccurate, the correct matches are clustered in a few regions of the image, the matched keypoints are far from their corresponding projected keypoints, and the correct matches are clustered in a few regions of the image.
* Recommended score: Above 0.7
*/
double calculateFeatureMatchingScore(const  vector<KeyPoint>& keypoints1, const  vector<KeyPoint>& keypoints2,
	const  vector<DMatch>& goodMatches, const Mat& homography, const Size& imageSize) {
	double recall = calculateRecall(keypoints1, keypoints2, goodMatches, homography, 3.0);
	double spreadOverlap = calculateSpreadOverlap(keypoints1, keypoints2, goodMatches, homography, imageSize);
	double featureDistance = calculateFeatureDistance(keypoints1, keypoints2, goodMatches, homography);
	double featureDensity = calculateFeatureDensity(keypoints1, keypoints2, goodMatches, homography, imageSize);

	double score = 0.5 * recall + 0.3 * spreadOverlap + 0.1 * featureDistance + 0.1 * featureDensity;
	return score;
}

/*
* calculatePrecision
* Definition: Precision is the ratio of correctly matched keypoints (inliers) to the total number of matches.
* Formula: Precision = Nm / Nc, where Nm is the number of inliers and Nc is the total number of matches.
* Interpretation:
* High Precision: Indicates that a high proportion of the matches are correct.
* Low Precision: Indicates that a low proportion of the matches are correct.
* Recommended precision score: Above 0.7
*/
double calculatePrecision(const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2,
	const vector<DMatch>& goodMatches, const Mat& homography, double epsilon_d) {
	int Nm = 0; // Number of inliers
	int Nc = goodMatches.size(); // Number of corresponding points

	for (const auto& match : goodMatches) {
		Point2f pt1 = keypoints1[match.queryIdx].pt;
		Point2f pt2 = keypoints2[match.trainIdx].pt;

		vector<Point2f> pts2 = { pt2 };
		vector<Point2f> projectedPts;
		perspectiveTransform(pts2, projectedPts, homography);

		double distance = norm(pt1 - projectedPts[0]);

		if (distance < epsilon_d) {
			Nm++;
		}
	}
	double precision = static_cast<double>(Nm) / static_cast<double>(Nc);
	return precision;
}

/*
* calculateF1Score
* Definition: F1 Score is the harmonic mean of Recall and Precision.
* Formula: F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
* Interpretation:
* High F1 Score: Indicates that both Recall and Precision are high.
* Low F1 Score: Indicates that either Recall or Precision is low.
* Recommended F1 Score: Above 0.75
*/
double calculateF1Score(double recall, double precision) {
	return 2 * (precision * recall) / (precision + recall);
}

void evaluateFeatureMatching(const Mat& img1, const Mat& img2, const Mat& homography,
	const  vector<KeyPoint>& keypoints1, const  vector<KeyPoint>& keypoints2,
	const  vector<DMatch>& goodMatches, const Size& imageSize) {
	double recall = calculateRecall(keypoints1, keypoints2, goodMatches, homography, 3.0);
	double spreadOverlap = calculateSpreadOverlap(keypoints1, keypoints2, goodMatches, homography, imageSize);
	double featureDistance = calculateFeatureDistance(keypoints1, keypoints2, goodMatches, homography);
	double featureDensity = calculateFeatureDensity(keypoints1, keypoints2, goodMatches, homography, imageSize);
	double score = calculateFeatureMatchingScore(keypoints1, keypoints2, goodMatches, homography, imageSize);
	double precision = calculatePrecision(keypoints1, keypoints2, goodMatches, homography, 3.0);
	double f1Score = calculateF1Score(recall, precision);

	cout << "Feature Matching Evaluation:" << endl;
	cout << "Recall: " << recall << endl;
	cout << "Spread Overlap: " << spreadOverlap << endl;
	cout << "Feature Distance: " << featureDistance << endl;
	cout << "Feature Density: " << featureDensity << endl;
	cout << "Score: " << score << endl;
	cout << "Precision: " << precision << endl;
	cout << "F1 Score: " << f1Score << endl;
}

void evaluateFeatureMatchingToFile(const Mat& img1, const Mat& img2, const Mat& homography,
	const  vector<KeyPoint>& keypoints1, const  vector<KeyPoint>& keypoints2,
	const  vector<DMatch>& goodMatches, const Size& imageSize){

	ofstream file("evaluation.txt");
	double recall = calculateRecall(keypoints1, keypoints2, goodMatches, homography, 3.0);
	double spreadOverlap = calculateSpreadOverlap(keypoints1, keypoints2, goodMatches, homography, imageSize);
	double featureDistance = calculateFeatureDistance(keypoints1, keypoints2, goodMatches, homography);
	double featureDensity = calculateFeatureDensity(keypoints1, keypoints2, goodMatches, homography, imageSize);
	double score = calculateFeatureMatchingScore(keypoints1, keypoints2, goodMatches, homography, imageSize);
	double precision = calculatePrecision(keypoints1, keypoints2, goodMatches, homography, 3.0);
	double f1Score = calculateF1Score(recall, precision);

	cout << "Feature Matching Evaluation:" << endl;
	cout << "Recall: " << recall << endl;
	cout << "Spread Overlap: " << spreadOverlap << endl;
	cout << "Feature Distance: " << featureDistance << endl;
	cout << "Feature Density: " << featureDensity << endl;
	cout << "Score: " << score << endl;
	cout << "Precision: " << precision << endl;
	cout << "F1 Score: " << f1Score << endl;

	file << "Feature Matching Evaluation:" << endl;
	file << "Recall: " << recall << endl;
	file << "Spread Overlap: " << spreadOverlap << endl;
	file << "Feature Distance: " << featureDistance << endl;
	file << "Feature Density: " << featureDensity << endl;
	file << "Score: " << score << endl;
	file << "Precision: " << precision << endl;
	file << "F1 Score: " << f1Score << endl;
	file.close();
}