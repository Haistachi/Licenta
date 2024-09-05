#include "stdafx.h"
#include "harris_FeatureDetection.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp> 

vector<KeyPoint> harrisDetectKeyPoints(Mat& src_gray) {
    const int maxCorners = 100000;
    const double qualityLevel = 0.01;
    const double minDistance = 10;
    const int blockSize = 3, gradientSize = 3;
    const bool useHarrisDetector = true;
    const double k = 0.04;

    vector<Point2f> corners;
    goodFeaturesToTrack(src_gray, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);

    vector<KeyPoint> keyPoints;
    for (Point2f corner : corners) {
        keyPoints.push_back(KeyPoint(corner, 1.f));
    }
    return keyPoints;
}