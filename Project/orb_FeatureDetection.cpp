#include "stdafx.h"
#include "orb_FeatureDetection.h"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "imageControl.h"

vector<KeyPoint> orbDetectKeyPoints(Mat& src_gray) {
    Ptr<ORB> orb = ORB::create();
    vector<KeyPoint> keypoints;
    orb->detect(src_gray, keypoints);
    return keypoints;
}