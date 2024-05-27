#include "stdafx.h"
#include "fast_FeatureDetection.h"
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

Mat detectFast(Mat src_gray)
{
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
    vector<KeyPoint> keypoints;
    detector->detect(src_gray, keypoints);
    Mat img_keypoints;
    drawKeypoints(src_gray, keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return img_keypoints;
}