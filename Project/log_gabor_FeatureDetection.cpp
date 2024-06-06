#include "stdafx.h"
#include "log_gabor_FeatureDetection.h"
#include "imageControl.h"
#include <cmath>
using namespace cv;
using namespace std;

void centering_transform(Mat img) {
    // imaginea trebuie să aibă elemente de tip float
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
        }
    }
}

Mat detectLogGabor(Mat& src_gray, double sig_fs, double lam)
{
    //imaginea trebuie să aibă elemente de tip float
    Mat srcf;
    src_gray.convertTo(srcf, CV_32FC1);

    //transformarea de centrare
    centering_transform(srcf);

    //aplicarea transformatei Fourier, se obține o imagine cu valori numere complexe
    Mat fourier;
    dft(srcf, fourier, DFT_COMPLEX_OUTPUT);
    //divizare în două canale: partea reală și partea imaginară
    Mat channels[] = { Mat::zeros(src_gray.size(), CV_32F), Mat::zeros(src_gray.size(), CV_32F) };
    split(fourier, channels); // channels[0] = Re(DFT(I)), channels[1] = Im(DFT(I))
    //calcularea magnitudinii și fazei în imaginile mag, respectiv phi, cu elemente de tip float
    Mat mag, phi, g(srcf.size(), CV_32F);
    magnitude(channels[0], channels[1], mag);
    phase(channels[0], channels[1], phi);


    //aici inserați operații de filtrare aplicate pe coeficienții Fourier
    // 2d L-G (magnitudine * filtru)
    // f=log(mag i,j)
    // ro= magnitu
    // lampda este parametru

    for (int i = 0; i < srcf.rows; i++)
    {
        for (int j = 0; j < srcf.cols; j++)
        {
            double ff = mag.at<float>(i,j);//logRadius 
            double rr = log(ff);
            double fs = log(1 / lam);//logWavelength 
            double sigm = log(sig_fs / fs);
            double tt = phi.at<float>(i, j);

            if (mag.at<float>(i, j) > 0.0001) {
                //formula 6:
                g.at<float>(i, j) = exp(-(((ff-fs) * (ff - fs))/(2 * log(sig_fs/fs) * log(sig_fs / fs))));
                
            }
            else
            {
                g.at<float>(i, j) = 0;
            }
            mag.at<float>(i, j) *= g.at<float>(i, j);
        }
    }

    //aplicarea transformatei Fourier inversă și punerea rezultatului în dstf
    Mat dst, dstf;
    merge(channels, 2, fourier);
    dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
    //transformarea de centrare inversă
    centering_transform(dstf);
    //normalizarea rezultatului în imaginea destinație
    dstf.convertTo(dst, CV_8UC1);

    //absolute balue

    //aprox box fillter

    //non-max supression

    //feature marking
    Mat thresholded;
    double thresholdValue = 0.5; // Example threshold value, adjust as needed
    threshold(dstf, thresholded, thresholdValue, 1.0, THRESH_BINARY);
    //Find and draw keypoints
    vector<KeyPoint> keypoints;
    for (int y = 0; y < thresholded.rows; y++) {
        for (int x = 0; x < thresholded.cols; x++) {
            if (thresholded.at<float>(y, x) > 0) {
                keypoints.push_back(KeyPoint((float)x, (float)y, 1.0f));
            }
        }
    }

    limitKeyPoints(keypoints, 100);

    Mat imgKeypoints;
    drawKeypoints(src_gray, keypoints, imgKeypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    return imgKeypoints;
}



// Function to create a 2D Log-Gabor filter in the frequency domain
Mat createLogGaborFilter(Size size, double wavelength, double sigmaOnf) {
    Mat filter(size, CV_32F);
    Point center = Point(size.width / 2, size.height / 2);

    for (int y = 0; y < size.height; y++) {
        for (int x = 0; x < size.width; x++) {
            double u = (x - center.x) / (double)size.width;
            double v = (y - center.y) / (double)size.height;
            double radius = sqrt(u * u + v * v);

            if (radius > 0) {
                double logRadius = log(radius);
                double logWavelength = log(1.0 / wavelength);
                double logGabor = exp(-(logRadius - logWavelength) * (logRadius - logWavelength) / (2 * log(sigmaOnf) * log(sigmaOnf)));
                filter.at<float>(y, x) = (float)logGabor;
            }
            else {
                filter.at<float>(y, x) = 0;
            }
        }
    }

    return filter;
}

Mat testLogGaborFillter(Mat& src_gray)
{
    Mat padded;
    int m = getOptimalDFTSize(src_gray.rows);
    int n = getOptimalDFTSize(src_gray.cols);
    copyMakeBorder(src_gray, padded, 0, m - src_gray.rows, 0, n - src_gray.cols, BORDER_CONSTANT, Scalar::all(0));

    // 3. Make place for both the complex and the real values
    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);

    // 4. Perform the Discrete Fourier Transform
    dft(complexI, complexI);

    // 5. Create the Log-Gabor filter
    double wavelength = 4.0; // Example wavelength
    double sigmaOnf = 0.55;  // Example sigmaOnf
    Mat logGaborFilter = createLogGaborFilter(padded.size(), wavelength, sigmaOnf);

    // 6. Apply the filter
    Mat planesH[] = { logGaborFilter, Mat::zeros(logGaborFilter.size(), CV_32F) };
    Mat complexH;
    merge(planesH, 2, complexH);
    mulSpectrums(complexI, complexH, complexI, 0);

    // 7. Compute the magnitude and switch to logarithmic scale
    split(complexI, planes);
    magnitude(planes[0], planes[1], planes[0]);
    Mat magI = planes[0];
    magI += Scalar::all(1);
    log(magI, magI);

    // 8. Crop and rearrange the quadrants of the Fourier image so that the origin is at the image center
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    // 9. Normalize the magnitude image for better visualization
    normalize(magI, magI, 0, 1, NORM_MINMAX);

    // 10. Threshold the magnitude image to extract features
    Mat thresholded;
    double thresholdValue = 0.5; // Example threshold value, adjust as needed
    threshold(magI, thresholded, thresholdValue, 1.0, THRESH_BINARY);

    // 11. Find and draw keypoints
    vector<KeyPoint> keypoints;
    for (int y = 0; y < thresholded.rows; y++) {
        for (int x = 0; x < thresholded.cols; x++) {
            if (thresholded.at<float>(y, x) > 0) {
                keypoints.push_back(KeyPoint((float)x, (float)y, 1.0f));
            }
        }
    }

    limitKeyPoints(keypoints, 100);

    Mat imgKeypoints;
    drawKeypoints(src_gray, keypoints, imgKeypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // 12. Display the result
    imshow("Log-Gabor Magnitude Spectrum", magI);
    imshow("Detected Features", imgKeypoints);
    return imgKeypoints;
}