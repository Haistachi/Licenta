#include "stdafx.h"

#pragma once

void showFeature(const string& title, Mat& dst, vector<Point2f> corners);
void showFeature(const string& title, Mat& dst);
void showFeature(const string& title, Mat& dst, vector<KeyPoint> keyPoints);