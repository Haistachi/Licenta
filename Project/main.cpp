#include "stdafx.h"
#include "imageOp.h"
#include "imageControl.h"
#include "featureDetection.h"

using namespace std;


int main()
{
	Mat src, dst, ch, st, gray, har;
	std::vector<Point2f> stc;
	src = readImage().clone();

	//procesare initiala
	imshow("src", src);
	convertToGray(src, gray);
	imshow("dst", gray);

	//Harris
	cout << "Harris" << endl;
	Mat har_gray = gray.clone();
	stc = detectHarris(har_gray);
	showFeature("Harris feature", har_gray, stc);
	//Shi-Tomasi
	cout << "Shi-Tomasi" << endl;
	Mat sitm_gray = gray.clone();
	stc = detectShiTomasi(sitm_gray);
	showFeature("ShiTomasi feature", sitm_gray, stc);
	//SIFT
	cout << "SIFT" << endl;
	Mat sift_gray = gray.clone();
	Mat sift = detectShif(sift_gray);
	showFeature("SIFT feature", sift);
	/*
	//SURF
	cout << "SURF" << endl;
	Mat surf_gray = gray.clone();
	Mat surf = detectSurf(surf_gray);
	showFeature("SURF feature", surf);
	*/
	//FAST
	cout << "FAST" << endl;
	Mat fast_gray = gray.clone();
	Mat fast = detectSurf(fast_gray);
	showFeature("FAST feature", fast);
	waitKey(0);
	return 0;
}