#include "stdafx.h"
#include "imgShow.h"
#include "imageOp.h"

using namespace cv;
using namespace std;

void showFeature(const string& title, Mat& dst, vector<Point2f> corners)
{
	RNG rng(12345);
	cout << "** Number of corners detected: " << corners.size() << endl;
	int radius = 4;
	for (size_t i = 0; i < corners.size(); i++)
	{
		circle(dst, corners[i], radius, Scalar(rng.uniform(0, 255), rng.uniform(0, 256), rng.uniform(0, 256)), FILLED);
	}
	Mat displayImg = resizeForDisplay(dst);
	imshow(title, displayImg);
}

void showFeature(const string& title, Mat& dst)
{
	Mat dst_norm, dst_norm_scaled;
	int thresh = 200;

	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);
	Mat displayImg = resizeForDisplay(dst_norm_scaled);
	imshow(title, displayImg);
}

void showFeature(const string& title, Mat& dst, vector<KeyPoint> keyPoints)
{
	Mat img_keypoints;
	drawKeypoints(dst, keyPoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	Mat displayImg = resizeForDisplay(img_keypoints);
	imshow(title, displayImg);
}

void drawKeypointsOnImage(Mat& image, const vector<KeyPoint>& keypoints, const Scalar& color, int thickness) {
	for (const KeyPoint& kp : keypoints) {
		Point center(cvRound(kp.pt.x), cvRound(kp.pt.y));
		int radius = cvRound(kp.size / 2);
		circle(image, center, radius, color, thickness);
	}
}

void drawKeypointsOnImageShow(Mat& image, const vector<KeyPoint>& keypoints, const Scalar& color, int thickness) {
	for (const KeyPoint& kp : keypoints) {
		Point center(cvRound(kp.pt.x), cvRound(kp.pt.y));
		int radius = cvRound(kp.size / 2);
		circle(image, center, radius, color, thickness);
	}
	imshow("Keypoints", image);
	waitKey(0);
}

void showImagesProcesed(Mat& imgMatches, Mat& result)
{
	Mat imgMatchesDisplay = resizeForDisplay(imgMatches);
	Mat resultDisplay = resizeForDisplay(result);
	imshow("Matches", imgMatchesDisplay);
	imshow("Stitched Image", resultDisplay);
}