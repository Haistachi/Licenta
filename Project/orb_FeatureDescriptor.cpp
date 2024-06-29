#include "stdafx.h"
#include "orb_FeatureDescriptor.h"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv::xfeatures2d;

Mat orbDescriptors(const Mat& image, vector<KeyPoint>& keyPoints)
{
	Ptr<ORB> featureExtractor = ORB::create();
	Mat descriptors;
	featureExtractor->compute(image, keyPoints, descriptors);
	return descriptors;
}