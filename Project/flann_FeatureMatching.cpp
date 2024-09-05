#include "stdafx.h"
#include "flann_FeatureMatching.h"
#include "evaluation.h"

vector<DMatch> flannfeatureMatchingEvaluated(Mat& src_gray1, Mat& src_gray2,
    Mat& descriptors1, Mat& descriptors2)
{
    if (descriptors1.type() != CV_32F) {
        descriptors1.convertTo(descriptors1, CV_32F);
    }
    if (descriptors2.type() != CV_32F) {
        descriptors2.convertTo(descriptors2, CV_32F);
    }

    // Use FLANN matcher
    FlannBasedMatcher matcher;
    vector<DMatch> matches1to2, matches2to1, consistentMatches;
    matcher.match(descriptors1, descriptors2, matches1to2);
    matcher.match(descriptors2, descriptors1, matches2to1);
    consistentMatches = crossValidationCheck(descriptors1, descriptors2, matches1to2, matches2to1);
    vector<DMatch> goodMatches = filterMatchesByDynamicThreshold(consistentMatches);
    return goodMatches;
}