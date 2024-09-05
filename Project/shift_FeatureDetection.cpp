#include "sift_FeatureDetection.h"
#include "stdafx.h"

// Function to detect SIFT keypoints and return them
vector<KeyPoint> siftDetectKeyPoints(Mat& src_gray) {
    Ptr<SIFT> detector = SIFT::create();
    vector<KeyPoint> keypoints;
    Mat descriptors;
    detector->detectAndCompute(src_gray, noArray(), keypoints, descriptors);
    return keypoints;
}