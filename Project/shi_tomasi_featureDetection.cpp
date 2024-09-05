#include "stdafx.h"
#include "shi_tomasi_FeatureDetection.h"

// Function to detect Shi-Tomasi corners and return them
vector<Point2f> detectShiTomasi(Mat& src_gray) {
    const int maxCorners = 100000;
    const double qualityLevel = 0.01;
    const double minDistanceST = 10;
    const int blockSize = 3, gradientSize = 3;
    const bool useHarrisDetector = false;
    const double k = 0.04;
    vector<Point2f> corners;
    goodFeaturesToTrack(src_gray, corners, maxCorners, qualityLevel, minDistanceST, Mat(), blockSize, useHarrisDetector, k);
    return corners;
}

// Function to convert Shi-Tomasi corners to KeyPoints
vector<KeyPoint> shiTomasiDetectKeyPoints(Mat& image) {
    vector<Point2f> corners = detectShiTomasi(image);
    vector<KeyPoint> keyPoints;
    for (Point2f corner : corners) {
        keyPoints.push_back(KeyPoint(corner, 1.f));
    }
    return keyPoints;
}