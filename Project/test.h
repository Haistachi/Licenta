#include "stdafx.h"
#include <complex>
#include <cmath>
#include <iostream>

#pragma once
using namespace std;

Mat createGaborKernel(int filterSizeX, int filterSizeY, double theta, double sigma, double gamma, double psi, double Fx);
void test(Mat& originalImage);