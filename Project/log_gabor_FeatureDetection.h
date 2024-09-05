#include "stdafx.h"  
#pragma once  

//For LogGaborFilter
const double angl = 0.0;         // Orientation
const double sig_fs = 0.55;      // Standard deviation of the Gaussian function used to create the Log-Gabor filter
const double lam = 10.0;         // Wavelength
const double theta_o = 1.0;      // Angular bandwidth
// Threshold for feature detection is in stdafx.h for global use
//For FeatureDetection and filterKeypoints
const int maxKeypoints = 20000;    // Maximum number of keypoints to detect
const float minDistance = 5.0f; // Minimum distance between keypoints
const int borderWidth = 10;          // Border around the image
//For MultiScaleKeypoints
const int numOrientations = 6;    // 6 orientations (0, 30, 60, 90, 120, 150 degrees)
const int numScales = 4;          // 4 scales (wavelengths)
const double minWavelength = 3.0; // Minimum wavelength for filters
const double maxWavelength = 15.0;// Maximum wavelength for filters
const double sigmaOnf = 0.55;     // Bandwidth parameter for Log-Gabor
const double thetaSigma = 1.0;    // Angular spread
const double cutoff = 0.45;       // Low-pass filter cutoff
const double sharpness = 15.0;    // Low-pass filter sharpness

struct FilterParameters {
	double orientation; // Angle of orientation (in radians)
	double wavelength;  // Scale (wavelength of the sinusoid)
	double sigmaOnf;    // Bandwidth parameter
	double thetaSigma;  // Angular spread parameter
	double cutoff;      // Low-pass filter cutoff frequency
	double sharpness;   // Low-pass filter sharpness
};

Mat createNormalizedRadius(int rows, int cols);
Mat createLogGaborFilter(const Mat& radius, double wavelength, double sigmaOnf);
Mat createLowPassFilter(const Mat& radius, float cutoff, float sharpness);
Mat createAngularComponent(int cols, int rows, float angl, float thetaSigma);
void fftShift(const Mat& input, Mat& output);
void applyFilterManually(Mat& complexI, const Mat& filter);
void checkAndResizeIfNeeded(Mat& mat1, Mat& mat2, const std::string& name1, const std::string& name2);

vector<KeyPoint> findKeypoints(const Mat& response, double threshold, int maxKeypoints = 500);
void filterKeypointsByDistance(vector<KeyPoint>& keypoints, float minDistanceKP = 10.0f);

vector<FilterParameters> generateFilterBankParameters(
	int numOrientations, int numScales, double minWavelength, double maxWavelength,
	double sigmaOnf, double thetaSigma, double cutoff, double sharpness);
Mat applyLogGaborFilter(Mat& src_gray, double angl, double sig_fs, double lam, double theta_o, double threshold,
	float cutoff, float sharpness, const string& imageName, int id);
vector<KeyPoint> detectLogGaborKeypoints(Mat& src_gray, double angl, double sig_fs, double lam, double theta_o,
	double threshold, float cutoff, float sharpness, const string& imageName, int id);
vector<KeyPoint> detectLogGaborMultiScaleKeypoints(Mat& src_gray, const string& imageName);
