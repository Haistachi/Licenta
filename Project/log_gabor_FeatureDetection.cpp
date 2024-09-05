#include "stdafx.h"
#include "log_gabor_FeatureDetection.h"
#include <cmath>

Mat createNormalizedRadius(int rows, int cols) {
    Mat x, y, radius;
    vector<float> x_range(cols);
    vector<float> y_range(rows);

    for (int i = 0; i < cols; ++i) {x_range[i] = (i - cols / 2.0) / cols;}
    for (int i = 0; i < rows; ++i) {y_range[i] = (i - rows / 2.0) / rows;}

    repeat(Mat(y_range).reshape(1, 1), cols, 1, y);
    repeat(Mat(x_range).reshape(1, 1).t(), 1, rows, x);
    sqrt(x.mul(x) + y.mul(y), radius);

    // Set the center point to 1 to avoid log(0) issues
    radius.at<float>(rows / 2, cols / 2) = 1;
    return radius;
}

Mat createLogGaborFilter(const Mat& radius, double wavelength, double sigmaOnf) {
    int rows = radius.rows;
    int cols = radius.cols;
    Mat logGabor(rows, cols, CV_32F);

    double fo = 1.0 / wavelength;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float r = radius.at<float>(i, j);
            logGabor.at<float>(i, j) = exp(-pow(log(r / fo), 2) / (2 * pow(log(sigmaOnf), 2)));
        }
    }

    logGabor.at<float>(rows / 2, cols / 2) = 0;
    return logGabor;
}

Mat createLowPassFilter(const Mat& radius, float cutoff, float sharpness) {
    int rows = radius.rows;
    int cols = radius.cols;
    Mat lowPassFilter(rows, cols, CV_32F);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float r = radius.at<float>(i, j);
            lowPassFilter.at<float>(i, j) = 1.0 / (1.0 + pow(r / cutoff, 2 * sharpness));
        }
    }
    return lowPassFilter;
}

Mat createAngularComponent(int cols, int rows, float angl, float thetaSigma) {
    Mat x = Mat::zeros(rows, cols, CV_32F);
    Mat y = Mat::zeros(rows, cols, CV_32F);

    // Create meshgrid
    float x_val, y_val;
    for (int i = 0; i < rows; i++) {
        y_val = (i - rows / 2.0f) / rows;
        for (int j = 0; j < cols; j++) {
            x_val = (j - cols / 2.0f) / cols;
            x.at<float>(i, j) = x_val;
            y.at<float>(i, j) = y_val;
        }
    }

    // Calculate angle (theta) and angular spread
    Mat theta = Mat::zeros(rows, cols, CV_32F);
    Mat spread = Mat::zeros(rows, cols, CV_32F);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float angle = atan2(-y.at<float>(i, j), x.at<float>(i, j));
            float deltaTheta = abs(angle - angl);
            deltaTheta = min(deltaTheta, float(2 * CV_PI) - deltaTheta);
            spread.at<float>(i, j) = exp(-pow(deltaTheta, 2) / (2 * pow(thetaSigma, 2)));
        }
    }

    return spread;
}

// Function to shift the zero-frequency component to the center of the spectrum
void fftShift(const Mat& input, Mat& output) {
    output = input.clone();
    int cx = output.cols / 2;
    int cy = output.rows / 2;

    Mat q0(output, Rect(0, 0, cx, cy));   // Top-Left
    Mat q1(output, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(output, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(output, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void applyFilterManually(Mat& complexI, const Mat& filter) {
    Mat planes[2];
    split(complexI, planes);
    Mat realPart = planes[0];
    Mat imaginaryPart = planes[1];

    Mat filteredRealPart;
    multiply(realPart, filter, filteredRealPart);
    Mat filteredImaginaryPart;
    multiply(imaginaryPart, filter, filteredImaginaryPart);

    merge(vector<Mat>{filteredRealPart, filteredImaginaryPart}, complexI);
}

vector<KeyPoint> findKeypoints(const Mat& response, double threshold, int maxKeypoints) {
    vector<KeyPoint> keypoints;
    
    // Non-Maximum Suppression and Thresholding
    for (int y = borderWidth; y < response.rows - borderWidth; ++y) {
        for (int x = borderWidth; x < response.cols - borderWidth; ++x) {
            float val = response.at<float>(y, x);
            if (val > threshold &&
                val > response.at<float>(y - 1, x) &&
                val > response.at<float>(y + 1, x) &&
                val > response.at<float>(y, x - 1) &&
                val > response.at<float>(y, x + 1)) {

                keypoints.emplace_back(Point2f(x, y), val);
            }
        }
    }
    sort(keypoints.begin(), keypoints.end(), [](const KeyPoint& a, const KeyPoint& b) {
        return a.response > b.response;
        });

    cout << "Number of keypoints detected: " << keypoints.size() << endl << endl;
    if (keypoints.size() > maxKeypoints) {
        keypoints.resize(maxKeypoints);
    }

    return keypoints;
}

void filterKeypointsByDistance(vector<KeyPoint>& keypoints, float minDistanceKP) {
    vector<KeyPoint> filteredKeypoints;

    for (size_t i = 0; i < keypoints.size(); ++i) {
        bool keep = true;
        for (size_t j = 0; j < filteredKeypoints.size(); ++j) {
            if (norm(keypoints[i].pt - filteredKeypoints[j].pt) < minDistanceKP) {
                keep = false;
                break;
            }
        }
        if (keep) {
            filteredKeypoints.push_back(keypoints[i]);
        }
    }

    keypoints = filteredKeypoints;
}

void checkAndResizeIfNeeded(Mat& mat1, Mat& mat2, const std::string& name1, const std::string& name2) {
    Mat mat1Display, mat2Display;
    if (mat1.size() != mat2.size()) {
        cout << "Size mismatch detected:" << endl;
        cout << name1 << " size: " << mat1.size() << endl;
        cout << name2 << " size: " << mat2.size() << endl;

        if (mat1.total() > mat2.total()) {
            resize(mat2, mat2, mat1.size());
            cout << "Resized " << name2 << " to " << mat1.size() << endl;
        }
        else {
            resize(mat1, mat1, mat2.size());
            cout << "Resized " << name1 << " to " << mat2.size() << endl;
        }
        cout << endl;
    }
}

vector<FilterParameters> generateFilterBankParameters(
    int numOrientations,
    int numScales,
    double minWavelength,
    double maxWavelength,
    double sigmaOnf,
    double thetaSigma,
    double cutoff,
    double sharpness)
{
    vector<FilterParameters> filterBank;

    // Generate orientations evenly spaced in the range [0, 180) degrees
    double orientationStep = CV_PI / numOrientations;

    // Generate scales logarithmically spaced between min and max wavelength
    double logMinWavelength = log(minWavelength);
    double logMaxWavelength = log(maxWavelength);
    double scaleStep = (logMaxWavelength - logMinWavelength) / (numScales - 1);

    for (int i = 0; i < numOrientations; ++i) {
        double orientation = i * orientationStep;

        for (int j = 0; j < numScales; ++j) {
            double wavelength = exp(logMinWavelength + j * scaleStep);

            FilterParameters params;
            params.orientation = orientation;
            params.wavelength = wavelength;
            params.sigmaOnf = sigmaOnf;
            params.thetaSigma = thetaSigma;
            params.cutoff = cutoff;
            params.sharpness = sharpness;

            filterBank.push_back(params);
        }
    }

    return filterBank;
}

vector<KeyPoint> detectLogGaborMultiScaleKeypoints(Mat& src_gray, const string& imageName) {
    vector<FilterParameters> filterBank = generateFilterBankParameters(
        numOrientations, numScales, minWavelength, maxWavelength, sigmaOnf, thetaSigma, cutoff, sharpness);
    vector<KeyPoint> allKeypoints;
    cout<< "Detecting keypoints for " << imageName << endl;
    int id = 0;
    for (const auto& params : filterBank) {
        cout<< "Filter " << id << ": Orientation: " << params.orientation << ", Wavelength: " << params.wavelength << endl;
        vector<KeyPoint> keypoints = detectLogGaborKeypoints(
            src_gray,
            params.orientation,
            params.sigmaOnf,
            params.wavelength,
            params.thetaSigma,
            thresholdKeypoint,
            params.cutoff,
            params.sharpness,
            imageName,
            id);
        allKeypoints.insert(allKeypoints.end(), keypoints.begin(), keypoints.end());
        id++;
    }

    sort(allKeypoints.begin(), allKeypoints.end(), [](const KeyPoint& a, const KeyPoint& b) {
        return a.response > b.response;
        });

    if (allKeypoints.size() > maxKeypoints) {
        allKeypoints.resize(maxKeypoints);
    }

    filterKeypointsByDistance(allKeypoints, minDistance);
    return allKeypoints;
}

vector<KeyPoint> detectLogGaborKeypoints(Mat& src_gray, double angl, double sig_fs, double lam, double theta_o,
    double threshold, float cutoff, float sharpness, const string& imageName, int id) {
    Mat src_gray_copy = src_gray.clone();
    Mat response = applyLogGaborFilter(src_gray_copy, angl, sig_fs, lam, theta_o, threshold, cutoff, sharpness, imageName, id);

    // Detect keypoints
    vector<KeyPoint> keypoints = findKeypoints(response, threshold, maxKeypoints);
    filterKeypointsByDistance(keypoints, minDistance);
    return keypoints;
}

Mat applyLogGaborFilter(Mat& src_gray, double angl, double sig_fs, double lam, double theta_o, double threshold,
    float cutoff, float sharpness, const string& imageName, int id) {
    int rows = src_gray.rows;
    int cols = src_gray.cols;
    Mat radius = createNormalizedRadius(rows, cols);
    Mat logGabor = createLogGaborFilter(radius, lam, sig_fs);

    Mat lowPassFilter = createLowPassFilter(radius, cutoff, sharpness);

    Mat logGaborFiltered;
    checkAndResizeIfNeeded(logGabor, lowPassFilter, "logGabor", "lowPassFilter");
    multiply(logGabor, lowPassFilter, logGaborFiltered);

    Mat spread = createAngularComponent(cols, rows, angl, theta_o);

    Mat filter;
    checkAndResizeIfNeeded(spread, logGaborFiltered, "spread", "logGaborFiltered");
    multiply(spread, logGaborFiltered, filter);

    Mat padded;
    int m = getOptimalDFTSize(rows);
    int n = getOptimalDFTSize(cols);
    copyMakeBorder(src_gray, padded, 0, m - rows, 0, n - cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);

    dft(complexI, complexI);
    fftShift(complexI, complexI);
    checkAndResizeIfNeeded(complexI, filter, "complexI", "filter");
    applyFilterManually(complexI, filter);
    fftShift(complexI, complexI);

    Mat imgf;
    idft(complexI, imgf, DFT_REAL_OUTPUT);
    Mat response;
    imgf.convertTo(response, CV_32F);
    normalize(response, response, 0, 1, NORM_MINMAX);

    ofstream fout("Parameters_" + imageName + "_filter" + to_string(id) + ".txt");
    fout << "Orientation: " << angl << endl;
    fout << "Wavelength: " << lam << endl;
    fout << "SigmaOnf: " << sig_fs << endl;
    fout << "ThetaSigma: " << theta_o << endl;
    fout << "Threshold: " << threshold << endl;
    fout << "Cutoff: " << cutoff << endl;
    fout << "Sharpness: " << sharpness << endl;
    fout.close();
    imwrite(imageName + "_dst_filter" + to_string(id) + ".jpg", response * 255);
    imwrite(imageName + "_filter" + to_string(id) + ".jpg", filter * 255);

    return response;
}