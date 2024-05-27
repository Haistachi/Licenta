#include "shift_FeatureDetection.h"
#include "stdafx.h"

using namespace std;
using namespace cv;

Mat detectShif(Mat& src_gray)
{
	//Creare SIFT
	Ptr<SIFT> detector = SIFT::create();
	//Detectare si descriptor
	vector<KeyPoint> keypoints;
	Mat descriptors;
	detector->detectAndCompute(src_gray, noArray(), keypoints, descriptors);
	//Desenare keypoints
	Mat img_keypoints;
	drawKeypoints(src_gray, keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	return img_keypoints;
}
