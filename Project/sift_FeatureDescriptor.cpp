#include "stdafx.h"
#include "sift_FeatureDescriptor.h"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv::xfeatures2d;

Mat siftDescriptors(const Mat& image, vector<KeyPoint>& keyPoints)
{
	Ptr<SIFT> featureExtractor = SIFT::create();
	Mat descriptors;
	featureExtractor->compute(image, keyPoints, descriptors);
	return descriptors;
}