#include "stdafx.h"
#include "fast_FeatureDetection.h"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "imageControl.h"

using namespace cv::xfeatures2d;

vector<KeyPoint> fastDetectKeyPoints(Mat& src_gray) {
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
    vector<KeyPoint> keypoints;
    detector->detect(src_gray, keypoints);
    return keypoints;
}