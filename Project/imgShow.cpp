#include "stdafx.h"
#include "imgShow.h"

using namespace cv;
using namespace std;

void showFeature(const std::string& title, Mat dst, vector<Point2f> corners)
{
	RNG rng(12345);
	cout << "** Number of corners detected: " << corners.size() << endl;
	int radius = 4;
	for (size_t i = 0; i < corners.size(); i++)
	{
		circle(dst, corners[i], radius, Scalar(rng.uniform(0, 255), rng.uniform(0, 256), rng.uniform(0, 256)), FILLED);
	}

	imshow(title, dst);
}

void showFeature(const std::string& title, Mat& dst)
{
	Mat dst_norm, dst_norm_scaled;
	int thresh = 200;

	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	for (int j = 0; j < dst_norm.rows; j++)
	{
		for (int i = 0; i < dst_norm.cols; i++)
		{
			if ((int)dst_norm.at<float>(j, i) > thresh)
			{
				circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
			}
		}
	}
	imshow(title, dst_norm_scaled);
}