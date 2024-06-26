#include "stdafx.h"
#include "imageOp.h"
#include "imageControl.h"
#include "featureDetection.h"

int main()
{
	Mat src, dst, ch, st, gray, har;
	std::vector<Point2f> stc;
	src = readImage().clone();

	double lam = 3; // wavelength, lambda
	double sig_fs = 0.7; // sigmaOnf, radian bandwidth of Bf
	double C = 3.0; //scaling factor
	double k = 1.3;
	double angl = CV_PI/4; //filter orientation angle
	double theta_o = 0.5;

	//procesare initiala
	imshow("src", src);
	convertToGray(src, gray);
	imshow("dst", gray);

	//Harris
	cout << "Harris" << endl;
	Mat har_gray = gray.clone();
	stc = detectHarris(har_gray);
	//showFeature("Harris feature", har_gray, stc);

	//Shi-Tomasi
	cout << "Shi-Tomasi" << endl;
	Mat sitm_gray = gray.clone();
	stc = detectShiTomasi(sitm_gray);
	//showFeature("ShiTomasi feature", sitm_gray, stc);

	//SIFT
	cout << "SIFT" << endl;
	Mat sift_gray = gray.clone();
	Mat sift = detectShif(sift_gray);
	//showFeature("SIFT feature", sift);

	//ORB
	cout << "ORB" << endl;
	Mat orb_gray = gray.clone();
	Mat orb = detectOrb(orb_gray);
	//showFeature("ORB feature", orb);

	//FAST
	cout << "FAST" << endl;
	Mat fast_gray = gray.clone();
	Mat fast = detectFast(fast_gray);
	//showFeature("FAST feature", fast);

	//Log-Gabor
	cout << "Log-Gabor" << endl;
	Mat lg_gray = gray.clone();
	Mat lg = detectLogGabor(lg_gray, angl, sig_fs, lam, theta_o);
	showFeature("Log-Gabor feature", lg);

	//applyFilterBankToImage(gray, 5, 8);
	waitKey(0);
	return 0;
}