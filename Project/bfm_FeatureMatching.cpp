#include "stdafx.h"
#include "bfm_FeatureMatching.h"

vector<DMatch> bfmFeatureMatching(Mat& src_gray1, Mat& src_gray2,
	Mat& descriptors1, Mat& descriptors2,
	vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, string alg)
{
	int normType;
	if(alg._Equal("ORB") || alg._Equal("FREAK") || alg._Equal("BRIEF"))
		normType = NORM_HAMMING;
	if(alg._Equal("SIFT") || alg._Equal("SURF")) 
		normType = NORM_L2;
	else
		normType = NORM_L1;

	//Match descriptors using BFMatcher
	BFMatcher matcher(normType);
	vector<DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);

	//Filter matches.
	double max_dist = 0; double min_dist = 100;
	for (int i = 0; i < descriptors1.rows; i++) {
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	vector<DMatch> good_matches;
	for (int i = 0; i < descriptors1.rows; i++) {
		if (matches[i].distance <= max(2 * min_dist, 0.02)) {
			good_matches.push_back(matches[i]);
		}
	}

	//Draw matches
	Mat img_matches;
	drawMatches(src_gray1, keypoints1, src_gray2, keypoints2,
		good_matches, img_matches,
		Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//Show detected matches
	imshow("Good Matches", img_matches);

	return good_matches;
}

vector<DMatch> bfmFeatureMatching(Mat& src_gray1, Mat& src_gray2,
	Mat& descriptors1, Mat& descriptors2, string alg)
{
	int normType;
	if (alg._Equal("ORB") || alg._Equal("FREAK") || alg._Equal("BRIEF"))
		normType = NORM_HAMMING;
	if (alg._Equal("SIFT") || alg._Equal("SURF"))
		normType = NORM_L2;
	else
		normType = NORM_L1;

	//Match descriptors using BFMatcher
	BFMatcher matcher(normType);
	vector<DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);

	// Apply ratio test
	vector<vector<DMatch>> knnMatches;
	matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);

	const float ratio_thresh = 0.75f;
	vector<DMatch> goodMatches;
	for (size_t i = 0; i < knnMatches.size(); i++) {
		if (knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance) {
			goodMatches.push_back(knnMatches[i][0]);
		}
	}

	return goodMatches;
}