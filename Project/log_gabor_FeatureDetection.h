
#include "stdafx.h"  
#pragma once  

/**
 * @brief Detects features in a grayscale image using the Log-Gabor filter.
 *
 * @param src_gray The source grayscale image.
 * @param angl The angle parameter for the Log-Gabor filter.
 * @param sig_fs The sigma parameter for the frequency spread.
 * @param lam The wavelength parameter for the Log-Gabor filter.
 * @param theta_o The orientation parameter for the Log-Gabor filter.
 * @return Mat The image with detected Log-Gabor features.
 */
Mat detectLogGabor(Mat& src_gray, double angl, double sig_fs, double lam, double theta_o);

/**
 * @brief Creates a Log-Gabor filter bank.
 *
 * @param rows The number of rows in the filter bank.
 * @param cols The number of columns in the filter bank.
 * @param numScales The number of scales in the filter bank.
 * @param numOrientations The number of orientations in the filter bank.
 * @return vector<Mat> The Log-Gabor filter bank.
 */
vector<Mat> logGaborFilterBank(int rows, int cols, int numScales, int numOrientations);

/**
 * @brief Applies a Log-Gabor filter bank to an image.
 *
 * @param graySrc The source grayscale image.
 * @param numScales The number of scales in the filter bank.
 * @param numOrientations The number of orientations in the filter bank.
 */
void applyFilterBankToImage(const Mat& graySrc, int numScales, int numOrientations);

/**
 * @brief Detects keypoints in an image using the Log-Gabor filter.
 *
 * @param image The source image.
 * @return vector<KeyPoint> The detected keypoints.
 */
vector<KeyPoint> logGaborDetectKeyPoints(Mat& image);
