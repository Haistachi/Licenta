#include "stdafx.h"
#include "imageOp.h"
#include "imageControl.h"
#include "evaluation.h"
#include "ImageStitching.h"
#include <chrono>

using namespace std::chrono;

void printMenu(const vector<string>& options, const string& prompt) {
    cout << prompt << endl;
    for (size_t i = 0; i < options.size(); ++i) {
        cout << i + 1 << ". " << options[i] << endl;
    }
}

int getUserSelection(int maxOption) {
    int selection;
    while (true) {
        cout << "Enter your choice (1-" << maxOption << "): ";
        cin >> selection;

        if (cin.fail() || selection < 1 || selection > maxOption) {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "Invalid choice. Please try again." << endl;
        }
        else {
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            break;
        }
    }
    return selection - 1;
}

int main() {
    vector<string> detectors = { "SIFT", "LOGGABOR", "HARRIS", "FAST", "ORB", "SHITOMASI" };
    vector<string> descriptors = { "SIFT", "ORB", "FREAK", "BRIEF" };
    vector<string> matchers = { "BFM", "FLANN" };

    String name1, name2;
    Mat img1 = readImage(name1).clone();
    Mat img2 = readImage(name2).clone();

    cout << "Image 1: " << name1 << endl;
    cout << "Image 2: " << name2 << endl;
    if (img1.empty() || img2.empty()) {
        cout << "Could not open or find the images!" << endl;
        return -1;
    }

    printMenu(detectors, "Select a detector:");
    int detectorIndex = getUserSelection(detectors.size());
    string detectorf = detectors[detectorIndex];

    printMenu(descriptors, "Select a descriptor:");
    int descriptorIndex = getUserSelection(descriptors.size());
    string descriptorf = descriptors[descriptorIndex];

    printMenu(matchers, "Select a matcher:");
    int matcherIndex = getUserSelection(matchers.size());
    string matcherf = matchers[matcherIndex];

    // Skip incompatible combinations
    if (detectorf == "SIFT" && descriptorf != "SIFT") {
        cout << "Incompatible combination: SIFT detector must be used with SIFT descriptor." << endl;
        system("pause");
        return -1;
    }
    if (detectorf != "SIFT" && descriptorf == "SIFT") {
        cout << "Incompatible combination: SIFT descriptor must be used with SIFT detector." << endl;
        system("pause");
        return -1;
    }
    if ((descriptorf == "ORB" || descriptorf == "FREAK" || descriptorf == "BRIEF") && matcherf == "FLANN") {
        cout << "Incompatible combination: ORB, FREAK, and BRIEF descriptors cannot be used with FLANN matcher." << endl;
        system("pause");
        return -1;
    }

    Mat gray1, gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);

    cout << "Processing with " << detectorf << " + " << descriptorf << " + " << matcherf << endl;

    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    vector<DMatch> goodMatches;
    Mat imgMatches, homography, result;
    //start time
    auto start = high_resolution_clock::now();

    customFeatureDetection(gray1, keypoints1, descriptors1, detectorf, descriptorf, name1);
    customFeatureDetection(gray2, keypoints2, descriptors2, detectorf, descriptorf, name2);

    if (keypoints1.empty() || keypoints2.empty()) {
        cout << "No keypoints detected!" << endl;
        return -1;
    }
    if (descriptors1.empty() || descriptors2.empty()) {
        cout << "No descriptors computed!" << endl;
        return -1;
    }

    goodMatches = customFeatureMatching(gray1, gray2, descriptors1, descriptors2, keypoints1, keypoints2, matcherf, detectorf);

    double inlierRatio, reprojectionError;
    homography = ransacHomography(img1, img2, goodMatches, keypoints1, keypoints2, inlierRatio, reprojectionError);
    result = warpPerspectiveAndBlend(img1, img2, homography);

    //end time
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count();

    drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, imgMatches,
        Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    // Save the result
    string resultPath = "stitched_result_" + detectorf + "_" + descriptorf + "_" + matcherf;
    imwrite(resultPath + ".jpg", result);
    string matchPath = "match_result_" + detectorf + "_" + descriptorf + "_" + matcherf;
    imwrite(matchPath + ".jpg", imgMatches);

    ofstream Metrics(resultPath + "_Metrics.txt");
    Metrics << "Inlier Ratio: " << inlierRatio << endl;
    Metrics << "Reprojection Error: " << reprojectionError << endl;
    Metrics << "Number of Good Matches: " << goodMatches.size() << endl;
    Metrics << "Processing Time (ms): " << duration << endl;

    Mat keypoints1Img = img1.clone();
    Mat keypoints2Img = img2.clone();
    drawKeypointsOnImage(keypoints1Img, keypoints1, Scalar(0, 255, 0), 5);
    drawKeypointsOnImage(keypoints2Img, keypoints2, Scalar(0, 255, 0), 5);
    string keypointPath1 = "keypoints_" + detectorf + "_" + descriptorf + "_" + matcherf + "_" + name1;
    string keypointPath2 = "keypoints_" + detectorf + "_" + descriptorf + "_" + matcherf + "_" + name2;
    imwrite(keypointPath1 + ".jpg", keypoints1Img);
    imwrite(keypointPath2 + ".jpg", keypoints2Img);
    Metrics << "Number of Keypoints in " << name1 << " : " << keypoints1.size() << endl;
    Metrics << "Number of Keypoints in " << name2 << " : " << keypoints2.size() << endl;
    Metrics << "Keypoint Detection Threshold: " << thresholdKeypoint << endl;
    Metrics << "Number of successful descriptions for " << name1 << " : " << descriptors1.rows << endl;
    Metrics << "Number of successful descriptions for " << name2 << " : " << descriptors2.rows << endl;

    Metrics.close();

    waitKey(0);
    system("pause");
    return 0;
}