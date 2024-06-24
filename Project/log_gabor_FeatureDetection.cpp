#include "stdafx.h"
#include "log_gabor_FeatureDetection.h"
#include "imageControl.h"
#include <cmath>

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

Mat createLowPassFilter(const Mat& radius, float cutoff, float sharpness) {
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
            deltaTheta = min(deltaTheta, float(2 * CV_PI) - deltaTheta); // Correct difference in angles
            spread.at<float>(i, j) = exp(-pow(deltaTheta, 2) / (2 * pow(thetaSigma, 2)));
        }
    }

    return spread;
}

// Function to shift the zero-frequency component to the center of the spectrum
void fftShift(const Mat& input, Mat& output) {
    output = input.clone(); // Ensure the output is a separate copy
    int cx = output.cols / 2;
    int cy = output.rows / 2;

    // Create ROI for each quadrant
    Mat q0(output, Rect(0, 0, cx, cy));   // Top-Left
    Mat q1(output, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(output, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(output, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp; // Temporary matrix for swapping

    // Swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    // Swap quadrants (Top-Right with Bottom-Left)
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void visualizeFourierComponents(const Mat& complexI) {
    // Split the complex image into its real and imaginary parts
    Mat planes[2];
    split(complexI, planes);
    Mat realPart = planes[0];
    Mat imaginaryPart = planes[1];

    // Calculate the magnitude
    Mat magnitudeImage;
    magnitude(realPart, imaginaryPart, magnitudeImage);

    // Calculate the phase
    Mat phaseImage;
    phase(realPart, imaginaryPart, phaseImage);

    // Switch to logarithmic scale to enhance visibility
    magnitudeImage += Scalar::all(1); // Avoid log(0) by adding 1
    log(magnitudeImage, magnitudeImage);

    // Normalize the magnitude image to [0, 1] for display
    normalize(magnitudeImage, magnitudeImage, 0, 1, NORM_MINMAX);

    // Normalize the phase image to [0, 1] for display
    normalize(phaseImage, phaseImage, 0, 1, NORM_MINMAX);

    // Display the magnitude and phase
    imshow("Magnitude", magnitudeImage);
    imshow("Phase", phaseImage);
}

void applyFilterManually(Mat& complexI, const Mat& filter) {
    // Split the complex image into real and imaginary parts
    Mat planes[2];
    split(complexI, planes);
    Mat realPart = planes[0];
    Mat imaginaryPart = planes[1];

    // The filter is real; thus, we can directly multiply it
    // Multiply the real part of the Fourier-transformed image with the filter
    Mat filteredRealPart;
    multiply(realPart, filter, filteredRealPart);

    // Multiply the imaginary part of the Fourier-transformed image with the filter
    Mat filteredImaginaryPart;
    multiply(imaginaryPart, filter, filteredImaginaryPart);

    // Merge the filtered real and imaginary parts back into a complex matrix
    merge(vector<Mat>{filteredRealPart, filteredImaginaryPart}, complexI);
}

Mat detectLogGaborV2(Mat& src_gray, double sig_fs, double lam, double theta_o)
{
    //Make the filter

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
    Mat spread = createAngularComponent(rows, cols, angl, thetaSigma);

    Mat filter;
    multiply(spread, logGaborFiltered, filter);


    // Perform Fourier Transform on the image

    //Expand the image to an optimal size
    Mat padded;
    int m = getOptimalDFTSize(rows);
    int n = getOptimalDFTSize(cols); // On the border add zero values
    copyMakeBorder(src_gray, padded, 0, m - rows, 0, n - cols, BORDER_CONSTANT, Scalar::all(0));
    //Make place for both the complex and the real values
    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    // Fourier Transform
    dft(complexI, complexI); 
    visualizeFourierComponents(complexI);
    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    // crop the spectrum, if it has an odd number of rows or columns

    // Shift the Fourier image
    fftShift(complexI, complexI);
    // Apply the filter
     Mat filterPlanes[] = { filter, filter };
     Mat complexFilter;
     merge(filterPlanes, 2, complexFilter);

     //mulSpectrums(complexI, complexFilter, complexI, 0);
     applyFilterManually(complexI, filter);

    // Shift back
    fftShift(complexI, complexI);
    // Perform inverse Fourier Transform
    idft(complexI, complexI,  DFT_SCALE |  DFT_REAL_OUTPUT);
    // Split the real and imaginary parts
     split(complexI, planes);
     Mat realPart = planes[0]; // Even-symmetric component
     Mat imaginaryPart = planes[1]; // Odd-symmetric component
    // Normalize for display
     normalize(realPart, realPart, 0, 1,  NORM_MINMAX);
     normalize(imaginaryPart, imaginaryPart, 0, 1,  NORM_MINMAX);

    imshow("Real Part (Even-symmetric)", realPart);
    imshow("Imaginary Part (Odd-symmetric)", imaginaryPart);
    imshow("Combined Filter", filter);
    return logGabor;
}

vector<Mat> logGaborFilterBank(int rows, int cols, int numScales, int numOrientations)
{
    vector<Mat> filterBank;
    Mat finalFilter, logGaborFilter, radius, angularComponent, lowPass;
    //Example values
    //Log-Gabor filter
    double lambda = 0.5; // wavelength, lambda
    double sigma_fs = 0.85; // sigmaOnf, radian bandwidth of Bf
    //Angular component filter
    double thetaSigma = 0.6; //filter orientation angle
    float angl = CV_PI / 4; // angle in radians
    //LowPass filter
    float cutoff = 0.4;
    float sharpness = 10;

    radius = createNormalizedRadius(rows, cols);
    for (int s = 0; s < numScales; ++s) {

        lambda = pow(2.0, s + 2);
        logGaborFilter = createLogGaborFilter(radius, lambda, 0.55);

        for (int o = 0; o < numOrientations; ++o) {

            angl = o * CV_PI / numOrientations;
            angularComponent = createAngularComponent(cols, rows, angl, CV_PI / numOrientations);
            //lowPass = createLowPassFilter(radius, cutoff, sharpness);
            //multiply(logGaborFilter, lowPass, logGaborFilter);
            multiply(logGaborFilter, angularComponent, finalFilter);
            filterBank.push_back(finalFilter);
        }
    }
    return filterBank;
}

void applyFilterBankToImage(const Mat& graySrc, int numScales, int numOrientations) {

    // Convert to float and normalize
    Mat srcFloat;
    graySrc.convertTo(srcFloat, CV_32F, 1.0 / 255.0);

    // Compute the DFT of the image
    Mat dftImage;
    dft(srcFloat, dftImage, DFT_COMPLEX_OUTPUT);

    // Generate the filter bank
    vector<Mat> filterBank = logGaborFilterBank(graySrc.rows, graySrc.cols, numScales, numOrientations);

    // Apply each filter
    for (const Mat& filter : filterBank) {
        // Multiply DFT of image with the filter
        Mat filteredDft;
        mulSpectrums(dftImage, filter, filteredDft, 0);

        // Compute the inverse DFT to get the filtered image
        Mat filteredImage;
        idft(filteredDft, filteredImage, DFT_SCALE | DFT_REAL_OUTPUT);

        // Process the filtered image to extract features
        // For example, thresholding
        Mat thresholdedImage;
        threshold(filteredImage, thresholdedImage, 0.5, 1.0, THRESH_BINARY);

        // Display or process the thresholded image
        imshow("Filtered Image", thresholdedImage);
        waitKey(0);

        // Here you can also implement non-maximum suppression, feature marking, etc.
    }
}