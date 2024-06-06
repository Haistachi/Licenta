#include "stdafx.h"
#include "log_gabor_FeatureDetection.h"

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
    Mat mag, phi, g;
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

            if (ff > 0.0001) {
                //formula 6
                g.at<float>(i, j) = exp(-(((ff-fs) * (ff - fs))/(2 * log(sig_fs/fs) * log(sig_fs / fs))));

                //formula 7
                mag.at<float>(i, j) *= g.at<float>(i, j);
            }
            else
            {
                g.at<float>(i, j) = 0;
            }
        }
    }
    //memorați partea reală în channels[0] și partea imaginară în channels[1]
    // ......
    //aplicarea transformatei Fourier inversă și punerea rezultatului în dstf
    Mat dst, dstf;
    merge(channels, 2, fourier);
    dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
    //transformarea de centrare inversă
    centering_transform(dstf);
    //normalizarea rezultatului în imaginea destinație
    //normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    

    //Notă: normalizarea distorsionează rezultatul oferind o afișare îmbunătățită în intervalul
    //[0,255]. Dacă se dorește afișarea rezultatului cu exactitate (vezi Activitatea 3) se va
    //folosi în loc de normalizare conversia:
    dstf.convertTo(dst, CV_8UC1);

    //absolute balue

    //aprox box fillter

    //non-max supression

    return dst;
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