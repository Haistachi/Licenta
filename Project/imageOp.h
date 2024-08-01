#include "stdafx.h"  
#pragma once  

using namespace cv;
using namespace std;

/**
 * @brief Opens a file dialog to select a file.
 *
 * @return std::string The path of the selected file.
 */
std::string openFileDialog();

/**
 * @brief Reads an image from a file selected via a file dialog.
 *
 * @return Mat The loaded image.
 */
Mat readImage();

/**
 * @brief Converts a color image to grayscale.
 *
 * @param src The source color image.
 * @param dst The destination grayscale image.
 */
void convertToGray(Mat& src, Mat& dst);

/**
 * @brief Limits the number of keypoints to a maximum value.
 *
 * @param keypoints The vector of keypoints to be limited.
 * @param maxKeypoints The maximum number of keypoints to retain.
 */
void limitKeyPoints(vector<KeyPoint>& keypoints, int maxKeypoints);

/**
 * @brief Resizes an image for display, maintaining aspect ratio.
 *
 * @param image The source image.
 * @return Mat The resized image.
 */
Mat resizeForDisplay(const Mat& image);

/**
 * @brief Detects features and computes descriptors for an image using specified algorithms.
 *
 * @param image The source image.
 * @param keypoints The detected keypoints.
 * @param descriptors The computed descriptors.
 * @param detector The feature detection algorithm to use.
 * @param descriptor The feature descriptor algorithm to use.
 */
void customFeatureDetection(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors, string detector, string descriptor);

/**
 * @brief Matches features between two images using specified algorithms.
 *
 * @param src_gray1 The first source grayscale image.
 * @param src_gray2 The second source grayscale image.
 * @param descriptors1 The descriptors of the first image.
 * @param descriptors2 The descriptors of the second image.
 * @param keypoints1 The keypoints of the first image.
 * @param keypoints2 The keypoints of the second image.
 * @param matcher The feature matching algorithm to use.
 * @param detector The algorithm to use for matching.
 * @return vector<DMatch> The matched features.
 */
vector<DMatch> featureMatching(Mat& src_gray1, Mat& src_gray2,
    Mat& descriptors1, Mat& descriptors2,
    vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2,
    string matcher, string detector);
