#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/base.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>

#include <stdio.h>
#include <tchar.h>
#include <SDKDDKVer.h>
#define NOMINMAX
#include <windows.h>
#include <CommDlg.h>
#include <ShlObj.h>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

#define PI 3.14159265
#define POS_INFINITY 1e30
#define NEG_INFINITY -1e30
#define max_(x,y) ((x) > (y) ? (x) : (y))
#define min_(x,y) ((x) < (y) ? (x) : (y))
#define isNan(x) ((x) != (x) ? 1 : 0)
#define isInf(x) ((x) == POS_INFINITY || (x) == NEG_INFINITY ? 1 : 0)
#define MAX_DYSPLAY_HEIGHT 540.0
#define MAX_DYSPLAY_WIDTH 960.0
#define MAX_IMAGE_HEIGHT 1080.0
#define MAX_IMAGE_WIDTH 1920.0
#define MAX_IMAGE_SIZE 2073600.0
#define MAX_DISPLAY_SIZE 518400.0
#define DISTANCE_THRESHOLD 1.0

const double thresholdKeypoint = 0.7;    // Keypoint detection threshold
#pragma once

