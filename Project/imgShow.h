#include "stdafx.h"
#include "imageOp.h"
#pragma once

void showFeature(const string& title, Mat& dst, vector<Point2f> corners);
void showFeature(const string& title, Mat& dst);
void showFeature(const string& title, Mat& dst, vector<KeyPoint> keyPoints);

void drawKeypointsOnImage(Mat& image, const vector<KeyPoint>& keypoints, const Scalar& color = Scalar(0, 255, 0), int thickness = 2);
void drawKeypointsOnImageShow(Mat& image, const vector<KeyPoint>& keypoints, const Scalar& color = Scalar(0, 255, 0), int thickness = 2);

void showImagesProcesed(Mat& imgMatches, Mat& result);