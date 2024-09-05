#include "stdafx.h"  
#pragma once  

string openFileDialog();
Mat readImage();
Mat readImage(string& name);
void convertToGray(Mat& src, Mat& dst);
void limitKeyPoints(vector<KeyPoint>& keypoints, int maxKeypoints);
Mat resizeForDisplay(const Mat image);
void customFeatureDetection(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors, string detector, string descriptor, const string& imageName);
vector<DMatch> customFeatureMatching(Mat& src_gray1, Mat& src_gray2,
    Mat& descriptors1, Mat& descriptors2,
    vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2,
    string matcher, string detector);
