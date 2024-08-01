#include "stdafx.h"
#include "imageOp.h"
#include "featureDetection.h"
#include "featureDescriptor.h"
#include "featureMatching.h"

using namespace cv;
using namespace std;

std::string openFileDialog() {
    char filename[MAX_PATH] = "";

    OPENFILENAME ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = "All Files\0*.*\0Text Files\0*.TXT\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
    ofn.lpstrDefExt = "txt";

    if (GetOpenFileName(&ofn)) {
        return std::string(filename);
    }
    else {
        return "";
    }
}

Mat readImage()
{
    Mat src;
    std::string fname = openFileDialog();
    src = imread(fname, IMREAD_COLOR); // CV_LOAD_IMAGE_COLOR is deprecated, use IMREAD_COLOR
    return src;
}

void convertToGray(Mat& src, Mat& dst)
{
    cvtColor(src, dst, COLOR_RGB2GRAY); // CV_RGB2GRAY is deprecated, use COLOR_RGB2GRAY
}

void limitKeyPoints(vector<KeyPoint>& keypoints, int maxKeypoints)
{
    sort(keypoints.begin(), keypoints.end(), [](const KeyPoint& a, const KeyPoint& b) {
        return a.response > b.response;
        });
    if (keypoints.size() > maxKeypoints) {
        keypoints.resize(maxKeypoints);
    }
}

Mat resizeForDisplay(const Mat& image) {
    Mat displayImg;
    if (image.rows > MAX_DYSPLAY_HEIGHT) {
        double ratio = static_cast<double>(MAX_DYSPLAY_HEIGHT) / image.rows;
        cv::resize(image, displayImg, cv::Size(), ratio, ratio);
    }
    else {
        displayImg = image.clone();
    }
    return displayImg;
}

void customFeatureDetection(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors, string detector, string descriptor) {
    Mat src = image.clone();

    if (detector == "FAST") {
        keypoints = fastDetectKeyPoints(src);
    }
    else if (detector == "HARRIS") { // Corrected "HARIS" to "HARRIS"
        keypoints = harrisDetectKeyPoints(src);
    }
    else if (detector == "SHITOMASI") { // Corrected "SHITOM" to "SHITOMASI"
        keypoints = shiTomasiDetectKeyPoints(src);
    }
    else if (detector == "ORB") {
        keypoints = orbDetectKeyPoints(src);
    }
    else if (detector == "SIFT") {
        keypoints = siftDetectKeyPoints(src);
    }
    else {
        keypoints = fastDetectKeyPoints(src);
    }

    if (descriptor == "BRIEF") {
        descriptors = briefDescriptors(src, keypoints);
    }
    else if (descriptor == "FREAK") {
        descriptors = freakDescriptors(src, keypoints);
    }
    else if (descriptor == "ORB") {
        descriptors = orbDescriptors(src, keypoints);
    }
    else if (descriptor == "SIFT") {
        descriptors = siftDescriptors(src, keypoints);
    }
    else {
        descriptors = briefDescriptors(src, keypoints);
    }
}

vector<DMatch> featureMatching(Mat& src_gray1, Mat& src_gray2,
    Mat& descriptors1, Mat& descriptors2,
    vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2,
    string matcher, string alg)
{
    if (matcher == "FLANN") {
        return flannfeatureMatching(src_gray1, src_gray2, descriptors1, descriptors2, keypoints1, keypoints2);
    }
    else if (matcher == "BFM") {
        return bfmFeatureMatching(src_gray1, src_gray2, descriptors1, descriptors2, alg);
    }
    else {
        return bfmFeatureMatching(src_gray1, src_gray2, descriptors1, descriptors2, alg);
    }
}