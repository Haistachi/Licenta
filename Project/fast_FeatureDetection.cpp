#include "stdafx.h"
#include "fast_FeatureDetection.h"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

Mat detectFast(Mat& src_gray)
{
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
    vector<KeyPoint> keypoints;
    detector->detect(src_gray, keypoints);

    sort(keypoints.begin(), keypoints.end(), [](const KeyPoint& a, const KeyPoint& b) {
        return a.response > b.response;
        });
    int maxKeypoints = 100; // Set the maximum number of keypoints you want
    if (keypoints.size() > maxKeypoints) {
        keypoints.resize(maxKeypoints);
    }

    Mat img_keypoints;
    drawKeypoints(src_gray, keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return img_keypoints;
}