#include "stdafx.h"
#include "brief_FeatureDescriptor.h"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv::xfeatures2d;

Mat briefDescriptors(const Mat& image, vector<KeyPoint>& keyPoints)
{
    Ptr<BriefDescriptorExtractor> featureExtractor = BriefDescriptorExtractor::create();
    Mat descriptors;
    featureExtractor->compute(image, keyPoints, descriptors);
    return descriptors;
}