#include "stdafx.h"
#include "bfm_FeatureMatching.h"
#include "evaluation.h"

vector<DMatch> bfmFeatureMatchingEvaluated(Mat& src_gray1, Mat& src_gray2,
	Mat& descriptors1, Mat& descriptors2, string alg)
{
	int normType;
	if (alg._Equal("ORB") || alg._Equal("FREAK") || alg._Equal("BRIEF"))
		normType = NORM_HAMMING;
	if (alg._Equal("SIFT") || alg._Equal("SURF"))
		normType = NORM_L2;
	else
		normType = NORM_L1;

	BFMatcher matcher(normType, true);
	vector<DMatch> matches1to2, matches2to1, consistentMatches;
	matcher.match(descriptors1, descriptors2, matches1to2);
	matcher.match(descriptors2, descriptors1, matches2to1);
	consistentMatches = crossValidationCheck(descriptors1, descriptors2, matches1to2, matches2to1);
	vector<DMatch> goodMatches = filterMatchesByDynamicThreshold(consistentMatches);

	return goodMatches;
}