#include <iostream>
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
#include <windows.h>
#include <CommDlg.h>
#include <ShlObj.h>
#include <vector>
#include <algorithm>

using namespace cv;

#define PI 3.14159265
#define POS_INFINITY 1e30
#define NEG_INFINITY -1e30
#define max_(x,y) ((x) > (y) ? (x) : (y))
#define min_(x,y) ((x) < (y) ? (x) : (y))
#define isNan(x) ((x) != (x) ? 1 : 0)

#pragma once

