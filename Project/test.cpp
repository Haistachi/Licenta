#include "test.h"
#include "stdafx.h"
#include "imageControl.h"
using namespace std;

cv::Mat createGaborKernel(int filterSizeX, int filterSizeY, double theta, double sigma, double gamma, double psi, double Fx) {
    // Number of different orientation angles to use
    int nAngles = 5;

    int middleX = filterSizeX / 2;
    int middleY = filterSizeY / 2;
    double sigma_x = sigma;
    double sigma_y = sigma / gamma;
    double sigma_x2 = sigma_x * sigma_x;
    double sigma_y2 = sigma_y * sigma_y;
    cv::Mat kernel(filterSizeY, filterSizeX, CV_32F);

    for (int x = -middleX; x <= middleX; x++) {
        for (int y = -middleY; y <= middleY; y++) {
            double xPrime = x * cos(theta) + y * sin(theta);
            double yPrime = y * cos(theta) - x * sin(theta);
            double a = 1.0 / (2.0 * CV_PI * sigma_x * sigma_y) *
                exp(-0.5 * (xPrime * xPrime / sigma_x2 + yPrime * yPrime / sigma_y2));
            double c = cos(2.0 * CV_PI * (Fx * xPrime) / filterSizeX + psi);
            kernel.at<float>(y + middleY, x + middleX) = (float)(a * c);
        }
    }
    
    return kernel;
}


void test(Mat & originalImage) {
    // Sigma defining the size of the Gaussian envelope
    double sigma = 8.0;
    // Aspect ratio of the Gaussian curves
    double gamma = 0.25;
    // Phase
    double psi = CV_PI / 4.0 * 0;
    // Frequency of the sinusoidal component
    double Fx = 3.0;
    // Number of different orientation angles to use
    int nAngles = 5;
    // Convert image to float
    originalImage.convertTo(originalImage, CV_32F);

    int width = originalImage.cols;
    int height = originalImage.rows;

    // Determine the size of the filters based on sigma
    double sigma_x = sigma;
    double sigma_y = sigma / gamma;
    int largerSigma = max((int)sigma_x, (int)sigma_y);
    if (largerSigma < 1)
        largerSigma = 1;

    int filterSizeX = 19; //6 * largerSigma + 1;
    int filterSizeY = 19; //6 * largerSigma + 1;

    double rotationAngle = CV_PI / (double)nAngles;

    std::vector<cv::Mat> kernels;
    for (int i = 0; i < nAngles; i++) {
        double theta = rotationAngle * i;
        cv::Mat kernel = createGaborKernel(filterSizeX, filterSizeY, theta, sigma, gamma, psi, Fx);
        kernels.push_back(kernel);
        cv::normalize(kernel, kernel, 0, 255, cv::NORM_MINMAX);
        std::string windowName = "Kernel - Angle " + std::to_string(i);
        imshow(windowName, kernel);
    }

    // Apply kernels
    std::vector<cv::Mat> filteredImages;
    for (int i = 0; i < nAngles; i++) {
        cv::Mat filteredImage;
        cv::filter2D(originalImage, filteredImage, CV_32F, kernels[i]);
        filteredImages.push_back(filteredImage);
    }

    // Normalize and enhance contrast
    for (auto& img : filteredImages) {
        cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
        img.convertTo(img, CV_8U);
    }

    // Display filtered images
    for (int i = 0; i < filteredImages.size(); i++) {
        std::string windowName = "Gabor Filtered Image - Angle " + std::to_string(i);
        cv::imshow(windowName, filteredImages[i]);
    }

    cv::waitKey(0);
}