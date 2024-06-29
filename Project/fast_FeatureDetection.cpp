#include "stdafx.h"
#include "fast_FeatureDetection.h"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "imageControl.h"

using namespace cv::xfeatures2d;

Mat detectFast(Mat& src_gray)
{
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
    vector<KeyPoint> keypoints;
    detector->detect(src_gray, keypoints);

    //limitKeyPoints(keypoints, 100);

    Mat img_keypoints;
    drawKeypoints(src_gray, keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return img_keypoints;
}

vector<KeyPoint> fastDetectKeyPoints(Mat& src_gray)
{
	Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
    vector<KeyPoint> keypoints;
	detector->detect(src_gray, keypoints);
	return keypoints;
}