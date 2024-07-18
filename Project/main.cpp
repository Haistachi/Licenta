#include "stdafx.h"
#include "imageOp.h"
#include "imageControl.h"
#include "featureDetection.h"

int main()
{
	Mat src, dst, ch, st, gray, har;
	std::vector<Point2f> stc;
	//src = readImage().clone();

	// Resize image for display if it's larger than a specific size (e.g., 1080p)
	//Mat displayImg = resizeForDisplay(src);

	double lam = 3; // wavelength, lambda
	double sig_fs = 0.7; // sigmaOnf, radian bandwidth of Bf
	double C = 3.0; //scaling factor
	double k = 1.3;
	double angl = CV_PI/4; //filter orientation angle
	double theta_o = 0.5;

	//Log-Gabor
	//cout << "Log-Gabor" << endl;
	//Mat lg_gray = gray.clone();
	//Mat lg = detectLogGabor(lg_gray, angl, sig_fs, lam, theta_o);
	//showFeature("Log-Gabor feature", lg);

	//applyFilterBankToImage(gray, 5, 8);

    Mat img1, img2;
    img1 = readImage().clone();
    img2 = readImage().clone();
    Mat img1_gray, img2_gray;
    convertToGray(img1, img1_gray);
    convertToGray(img2, img2_gray);
    Mat img1_grayDisplay = resizeForDisplay(img1_gray);
    Mat img2_grayDisplay = resizeForDisplay(img2_gray);
    imshow("img1", img1_grayDisplay);
    imshow("img2", img2_grayDisplay);

    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    std::string detector = "ORB";
    std::string descriptor = "ORB";
    std::string matcher = "BFM";

    customFeatureDetection(img1_gray, keypoints1, descriptors1, detector, descriptor);
    customFeatureDetection(img2_gray, keypoints2, descriptors2, detector, descriptor);

    std::vector<DMatch> goodMatches =
        featureMatching(img1_gray, img2_gray, descriptors1, descriptors2, keypoints1, keypoints2, matcher, detector);

    Mat imgMatches;
    drawMatches(img1_gray, keypoints1, img2_gray, keypoints2, goodMatches, imgMatches);
    Mat imgMatchesDisplay = resizeForDisplay(imgMatches);
    imshow("Matches", imgMatchesDisplay);

    std::vector<Point2f> points1, points2;
    for (const auto& match : goodMatches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    Mat H = findHomography(points1, points2, RANSAC);
    
    Mat result;
    warpPerspective(img2, result, H, Size(img1.cols + img2.cols, img1.rows));

    Mat half(result, Rect(0, 0, img1.cols, img1.rows));
    img1.copyTo(half);

    Mat blendedDisplay = resizeForDisplay(result);
    imshow("Warped Image", blendedDisplay);

    waitKey(0);
    return 0;
}