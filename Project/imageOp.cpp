#include "stdafx.h"
#include "imageOp.h"

using namespace cv;

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
    std::string fname= openFileDialog();
	src = imread(fname, CV_LOAD_IMAGE_COLOR);
	return src;
}

void convertToGray(Mat& src, Mat& dst)
{
	cvtColor(src, dst, CV_RGB2GRAY);
}
