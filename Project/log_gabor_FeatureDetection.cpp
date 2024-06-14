#include "stdafx.h"
#include "log_gabor_FeatureDetection.h"
#include "imageControl.h"
#include <cmath>

using namespace cv;
using namespace std;

void centering_transform(Mat& img) {
    // imaginea trebuie să aibă elemente de tip float
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
        }
    }
}

Mat detectLogGabor(Mat& src_gray, double sig_fs, double lam, double theta_o)
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
    int w = src_gray.size().width,h = src_gray.size().height;
    Point center = Point(w / 2, h / 2);
    for (int i = 0; i < srcf.rows; i++)
    {
        for (int j = 0; j < srcf.cols; j++)
        {
            double u = (i - center.x) / (double)w;
            double v = (j - center.y) / (double)h;
            double radius = sqrt(u * u + v * v);
            double f = mag.at<float>(i,j);//Radius 

            //log-polar coordinates
            double rho = log(radius);
            double theta = phi.at<float>(i, j);

            double fs = (1 / lam);//logWavelength, filter's radial frequency 
            double rho_s = log(fs);
            double sigm_rho = log(sig_fs / fs);

            double sigm_theta_o = 0.2;// angular bandwidth in octaves

            double go = exp(-(pow(theta - theta_o,2))/(2 * sigm_theta_o * sigm_theta_o));// exp(-((tt - or_ang) * (tt - or_ang)) / (2 * sigm * sigm));
            
            if (radius > 0.0001) {
                //formula 6:
                g.at<float>(i, j) = exp(-pow(rho - rho_s, 2)/(2 * sigm_rho * sigm_rho));
                g.at<float>(i, j) *= go;
            }
            else
            {
                g.at<float>(i, j) = 0;
            }
            //formula 7
            mag.at<float>(i, j) *= g.at<float>(i, j);
        }
    }
    imshow("gabor filter", g);
    //descompunere parte reala si imaginara
    polarToCart(mag, phi, channels[0], channels[1]);

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
    double thresholdValue = 1.5; // Example threshold value, adjust as needed
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

    //limitKeyPoints(keypoints, 100);

    Mat imgKeypoints;
    drawKeypoints(src_gray, keypoints, imgKeypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    return dstf;
}

Mat createNormalizedRadius(int rows, int cols) {
    Mat x, y, radius;
    vector<float> x_range(cols);
    vector<float> y_range(rows);

    for (int i = 0; i < cols; ++i) {x_range[i] = (i - cols / 2.0) / cols;}
    for (int i = 0; i < rows; ++i) {y_range[i] = (i - rows / 2.0) / rows;}

    repeat(Mat(y_range).reshape(1, 1), cols, 1, y);
    repeat(Mat(x_range).reshape(1, 1).t(), 1, rows, x);

    // Calculate radius
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

    // Set the center point to zero
    logGabor.at<float>(rows / 2, cols / 2) = 0;
    return logGabor;
}

Mat createLowPassFilter(const cv::Mat& radius, float cutoff, float sharpness) {
    int rows = radius.rows;
    int cols = radius.cols;
    Mat lowPassFilter(rows, cols, CV_32F);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float r = radius.at<float>(i, j);
            //Low-pass Filter Butterworth 
            lowPassFilter.at<float>(i, j) = 1.0 / (1.0 + pow(r / cutoff, 2 * sharpness));
        }
    }
    return lowPassFilter;
}

Mat createAngularComponent(int cols, int rows, float angl, float thetaSigma) {
    
    Mat x, y;
    std::vector<float> x_range(cols);
    std::vector<float> y_range(rows);
    for (int i = 0; i < cols; ++i) { x_range[i] = (i - cols / 2.0) / cols; }
    for (int i = 0; i < rows; ++i) { y_range[i] = (i - rows / 2.0) / rows; }
    repeat(Mat(y_range).reshape(1, 1), cols, 1, y);
    repeat(Mat(x_range).reshape(1, 1).t(), 1, rows, x);

    Mat theta(rows, cols, CV_32F);
    Mat sintheta(rows, cols, CV_32F);
    Mat costheta(rows, cols, CV_32F);
    Mat spread(rows, cols, CV_32F);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            theta.at<float>(i, j) = atan2(-y.at<float>(i, j), x.at<float>(i, j));
            sintheta.at<float>(i, j) = sin(theta.at<float>(i, j));
            costheta.at<float>(i, j) = cos(theta.at<float>(i, j));
        }
    }

    float cosAngl = cos(angl);
    float sinAngl = sin(angl);

    Mat ds = sintheta * cosAngl - costheta * sinAngl; // Difference in sine.
    Mat dc = costheta * cosAngl + sintheta * sinAngl; // Difference in cosine.
    Mat dtheta(rows, cols, CV_32F);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dtheta.at<float>(i, j) = abs(atan2(ds.at<float>(i, j), dc.at<float>(i, j)));
        }
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            spread.at<float>(i, j) = exp(-pow(dtheta.at<float>(i, j), 2) / (2 * pow(thetaSigma, 2)));
        }
    }
    return spread;
}

Mat detectLogGaborV2(Mat& src_gray, double sig_fs, double lam, double theta_o)
{
    int rows = src_gray.rows;
    int cols = src_gray.cols;
    Mat radius=createNormalizedRadius(rows, cols);

    double wavelength = 10.0; // wavelength
    double sigmaOnf = 0.55; // sigmaOnf
    Mat logGabor = createLogGaborFilter(radius, wavelength, sigmaOnf);

    float cutoff = 0.4;
    float sharpness = 10;
    Mat lowPassFilter = createLowPassFilter(radius, cutoff, sharpness);

    Mat logGaborFiltered;
    multiply(logGabor, lowPassFilter, logGaborFiltered);

    float angl = CV_PI / 4; // Example angle in radians
    float thetaSigma = 0.6; // Example thetaSigma
    // Create the angular component
    Mat spread = createAngularComponent(rows, cols, angl, thetaSigma);

    Mat filter;
    multiply(spread, logGaborFiltered, filter);

    Mat srcf = src_gray.clone();
    srcf.convertTo(srcf, CV_32FC1);
    centering_transform(srcf);
    Mat fourier;
    dft(srcf, fourier, DFT_COMPLEX_OUTPUT);
    Mat channels[] = { Mat::zeros(src_gray.size(), CV_32F), Mat::zeros(src_gray.size(), CV_32F) };
    split(fourier, channels); // channels[0] = Re(DFT(I)), channels[1] = Im(DFT(I))
    Mat mag, phi, g(srcf.size(), CV_32F);
    magnitude(channels[0], channels[1], mag);
    phase(channels[0], channels[1], phi);

    multiply(mag, filter, mag);
    multiply(phi, filter, phi);

    //centering_transform(mag);
    //centering_transform(phi);

    polarToCart(mag, phi, channels[0], channels[1]);
    Mat dst, dstf;
    merge(channels, 2, fourier);
    dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
    //transformarea de centrare inversă
    centering_transform(dstf);
    //normalizarea rezultatului în imaginea destinație
    dstf.convertTo(dst, CV_8UC1);
    mag.convertTo(mag, CV_8UC1);
    phi.convertTo(phi, CV_8UC1);

    imshow("Image", dst);
    imshow("Real Part (Even-symmetric)", mag);
    imshow("Imaginary Part (Odd-symmetric)", phi);
    imshow("Normalized Radius", radius);
    imshow("Log-Gabor Filter", logGabor);
    imshow("Low-Pass Filter", lowPassFilter);
    imshow("Log-Gabor Filtered", logGaborFiltered);
    imshow("Angular Component", spread);
    imshow("Combined Filter", filter);
    return logGabor;
}