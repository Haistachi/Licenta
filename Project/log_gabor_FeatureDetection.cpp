#include "stdafx.h"
#include "log_gabor_FeatureDetection.h"

using namespace cv;
using namespace std;

Mat detectLogGabor(Mat& src_gray)
{
    //Expand the image to an optimal size
    Mat padded;
    int m = getOptimalDFTSize(src_gray.rows);
    int n = getOptimalDFTSize(src_gray.cols); // on the border add zero values
    copyMakeBorder(src_gray, padded, 0, m - src_gray.rows, 0, n - src_gray.cols, BORDER_CONSTANT, Scalar::all(0));

    //Make place for both the complex and the real values
    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);

    //Perform the Discrete Fourier Transform
    dft(complexI, complexI);

    //Compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]); // planes[0] = magnitude
    Mat magI = planes[0];
    magI += Scalar::all(1); // switch to logarithmic scale
    log(magI, magI);

    //Crop and rearrange the quadrants of the Fourier image so that the origin is at the image center
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

    //Normalize the magnitude image for better visualization
    normalize(magI, magI, 0, 1, NORM_MINMAX);

    imshow("spectrum magnitude", magI);

    //Perform the inverse DFT
    Mat inverseTransform;
    idft(complexI, inverseTransform, DFT_SCALE | DFT_REAL_OUTPUT);
    //Normalize the inverse DFT result for better visualization
    normalize(inverseTransform, inverseTransform, 0, 1, NORM_MINMAX);
    //Display the inverse transform result
    imshow("Reconstructed Image", inverseTransform);

	return inverseTransform;
}

Mat fourierTransform(Mat& src_gray)
{
    //Expand the image to an optimal size
    Mat padded;
    int m = getOptimalDFTSize(src_gray.rows);
    int n = getOptimalDFTSize(src_gray.cols); // on the border add zero values
    copyMakeBorder(src_gray, padded, 0, m - src_gray.rows, 0, n - src_gray.cols, BORDER_CONSTANT, Scalar::all(0));

    //Make place for both the complex and the real values
    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);

    //Perform the Discrete Fourier Transform
    dft(complexI, complexI);

    //Compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]); // planes[0] = magnitude
    Mat magI = planes[0];
    magI += Scalar::all(1); // switch to logarithmic scale
    log(magI, magI);

    //Crop and rearrange the quadrants of the Fourier image so that the origin is at the image center
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

    //Normalize the magnitude image for better visualization
    normalize(magI, magI, 0, 1, NORM_MINMAX);

    imshow("spectrum magnitude", magI);

    //Perform the inverse DFT
    Mat inverseTransform;
    idft(complexI, inverseTransform, DFT_SCALE | DFT_REAL_OUTPUT);
    //Normalize the inverse DFT result for better visualization
    normalize(inverseTransform, inverseTransform, 0, 1, NORM_MINMAX);

    imshow("Reconstructed Image", inverseTransform);

    return inverseTransform;
}

Mat convoluteLogGaborFillter(Mat& src_gray)
{
    Mat tmp;
    return tmp;
}