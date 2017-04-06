#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

Mat preprocess(Mat src) {
    GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

    /// Convert it to gray
    cvtColor( src, src, CV_BGR2GRAY );

    return src;
}

Mat sobel(Mat src_gray) {
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    Mat grad, grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );

    /// Gradient Y
    Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );

    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

    return grad;
}

Mat scharr(Mat src_gray) {
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    Mat grad, grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );

    /// Gradient Y
    Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );

    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

    return grad;
}

/** @function main */
int main( int argc, char** argv ) {

    // string window_name = "Sobel Demo - Simple Edge Detector";

    /// Load an image
    Mat original = imread( argv[1] );

    if( !original.data )
    { return -1; }

    /// Create windows
    namedWindow( "Original", CV_WINDOW_AUTOSIZE );
    namedWindow( "Sobel", CV_WINDOW_AUTOSIZE );
    namedWindow( "Scharr", CV_WINDOW_AUTOSIZE );

    Mat sobe = sobel(preprocess(original));
    Mat schar = scharr(preprocess(original));

    imshow( "Original", original );
    imshow( "Sobel", sobe );
    imshow( "Scharr", schar );

    waitKey(0);
}
