#include "stdafx.h"
#include "flann_FeatureMatching.h"

vector<DMatch> flannfeatureMatching(Mat& src_gray1, Mat& src_gray2,
	Mat& descriptors1, Mat& descriptors2,
	vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2)
{
    // Convert descriptors to the type CV_32F needed for FLANN
    if (descriptors1.type() != CV_32F) {
        descriptors1.convertTo(descriptors1, CV_32F);
    }
    if (descriptors2.type() != CV_32F) {
        descriptors2.convertTo(descriptors2, CV_32F);
    }

    // Use FLANN matcher
    FlannBasedMatcher matcher;
    vector<vector<DMatch>> knnMatches;
    matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);

    // Apply Lowe's ratio test
    const float ratio_thresh = 0.7f; // Commonly used threshold
    vector<DMatch> goodMatches;
    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance) {
            goodMatches.push_back(knnMatches[i][0]);
        }
    }

    return goodMatches;
}

vector<DMatch> flannfeatureMatching(Mat& src_gray1, Mat& src_gray2,
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
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
}