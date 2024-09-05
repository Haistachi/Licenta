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
    src = imread(fname, IMREAD_COLOR);
    return src;
}

Mat readImage(string& name)
{
    Mat src;
    std::string fname = openFileDialog();

    if (fname.empty()) {
        std::cerr << "No file selected!" << std::endl;
        return src;
    }

    src = imread(fname, IMREAD_COLOR);

    if (src.empty()) {
        std::cerr << "Error: Could not load image: " << fname << std::endl;
    }

    size_t lastSlash = fname.find_last_of("\\/");
    string fileName = fname.substr(lastSlash + 1);
    size_t lastDot = fileName.find_last_of(".");
    name = fileName.substr(0, lastDot);

    return src;
}

void convertToGray(Mat& src, Mat& dst)
{
    cvtColor(src, dst, COLOR_RGB2GRAY);
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

Mat resizeForDisplay(const Mat image) {
    Mat displayImg;
    if (image.rows > MAX_DYSPLAY_HEIGHT) {
        double ratio = static_cast<double>(MAX_DYSPLAY_HEIGHT) / image.rows;
        resize(image, displayImg, cv::Size(), ratio, ratio);
    }
    else {
        displayImg = image.clone();
    }
    return displayImg;
}

void customFeatureDetection(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors, string detector, string descriptor, const string& imageName) {
    Mat src = image.clone();

    if (detector == "FAST") {
        keypoints = fastDetectKeyPoints(src);
    }
    else if (detector == "HARRIS") {
        keypoints = harrisDetectKeyPoints(src);
    }
    else if (detector == "SHITOMASI") {
        keypoints = shiTomasiDetectKeyPoints(src);
    }
    else if (detector == "ORB") {
        keypoints = orbDetectKeyPoints(src);
    }
    else if (detector == "SIFT") {
        keypoints = siftDetectKeyPoints(src);
    }
    else if (detector == "LOGGABOR") {
        keypoints = detectLogGaborMultiScaleKeypoints(src, imageName);
    }
    else {
        keypoints = fastDetectKeyPoints(src);
    }

    vector<KeyPoint> keypointsCopy = keypoints;
    if (descriptor == "BRIEF") {
        descriptors = briefDescriptors(src, keypointsCopy);
    }
    else if (descriptor == "FREAK") {
        descriptors = freakDescriptors(src, keypointsCopy);
    }
    else if (descriptor == "ORB") {
        descriptors = orbDescriptors(src, keypointsCopy);
    }
    else if (descriptor == "SIFT") {
        descriptors = siftDescriptors(src, keypointsCopy);
    }
    else {
        descriptors = briefDescriptors(src, keypointsCopy);
    }
}

vector<DMatch> customFeatureMatching(Mat& src_gray1, Mat& src_gray2,
    Mat& descriptors1, Mat& descriptors2,
    vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2,
    string matcher, string alg)
{
    if (matcher == "FLANN") {
        return flannfeatureMatchingEvaluated(src_gray1, src_gray2, descriptors1, descriptors2);
    }
    else if (matcher == "BFM") {
        return bfmFeatureMatchingEvaluated(src_gray1, src_gray2, descriptors1, descriptors2, alg);
    }
    else {
        return bfmFeatureMatchingEvaluated(src_gray1, src_gray2, descriptors1, descriptors2, alg);
    }
}