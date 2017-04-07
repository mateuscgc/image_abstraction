#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <stdio.h>
#include "lime/sources/lime.hpp"

using namespace cv;
using namespace std;

#define KERNEL_SIZE 5

struct vec2d {
    double x,y;
    vec2d() {}
    vec2d(double xx, double yy) : x(xx), y(yy) {}
    double length() {
        return sqrt(x*x + y*y);
    }
    void normalize() {
        double len = length();
        if(len) {
            x = x/len;
            y = y/len;
        } else {
            x = 0;
            y = 0;
        }
    }
    vec2d operator*(const vec2d& rhs) {
        return vec2d(x*rhs.x, y*rhs.y);
    }
    vec2d operator*(double rhs) {
        return vec2d(x*rhs, y*rhs);
    }
    vec2d operator*(int rhs) {
        return vec2d(x*rhs, y*rhs);
    }
    vec2d& operator+=(const vec2d& rhs) {
        this->x += rhs.x;
        this->y += rhs.y;
        return *this;
    }
};

vec2d operator* (int lhs, vec2d rhs) {
    return vec2d(rhs.x*lhs, rhs.y*lhs);
}

vec2d operator* (double lhs, vec2d rhs) {
    return vec2d(rhs.x*lhs, rhs.y*lhs);
}

struct mat2d {
    vector< vector<vec2d> > vecs;
    mat2d(Mat gx, Mat gy) {
        Scalar intensity;
        // cout << gx.rows << " " << gx.cols << endl;
        // cout << gy.rows << " " << gy.cols << endl;
        int n = gx.rows;
        int m = gx.cols;
        for(int i = 0; i < n; i++) {
            vecs.push_back(vector<vec2d>());
            for(int j = 0; j < m; j++) {
                vec2d vec;
                // cout << "( ";
                intensity = gx.at<uchar>(i, j);
                vec.x = (double)intensity.val[0];
                // cout << "(" << vec.x; 
                    // cout << vec.x;

                intensity = gy.at<uchar>(i, j);
                vec.y = (double)intensity.val[0];
                // cout << ", " << vec.y << ") ";

                // cout << " ) ";
                vecs[i].push_back(vec);
            }
            // cout << endl;
        }
    }
    void normalize() {
        for(int n = vecs.size(), i = 0; i < n; i++) {
            for(int m = vecs[i].size(), j = 0; j < m; j++) {
                vecs[i][j].normalize();
            }
        }
    }
};

Mat preprocess(Mat src) {
    GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

    /// Convert it to gray
    cvtColor( src, src, CV_BGR2GRAY );

    return src;
}

mat2d sobel(Mat src_gray) {
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    Mat grad, grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    // imshow( "Sobel_x", grad_x );
    convertScaleAbs( grad_x, abs_grad_x );
    imshow( "Sobel_x", abs_grad_x );

    /// Gradient Y
    Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    // imshow( "Sobel_y", grad_y );
    convertScaleAbs( grad_y, abs_grad_y );
    imshow( "Sobel_y", abs_grad_y );


    mat2d gmap(grad_x, grad_y);

    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    imshow( "Sobel", grad );

    return gmap;
}

double dot_product(vec2d a, vec2d b) {
    return a.x*b.x + a.y*b.y;
}

int sign(vec2d c, vec2d n) {
    if(dot_product(c,n) > 0)
        return 1;
    return -1;
}

int ws(double x1, double y1, double x2, double y2) {
    double length = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
    if(length < KERNEL_SIZE)
        return 1;
    return 0;
}

double wm(double mc, double mn) {
    return (tanh(mn-mc)+1)/2;
    //return (mn-mc+1)/2;
}

double wd(vec2d c, vec2d n) {
    return fabs(dot_product(c,n));
}

mat2d ETF_iteration(mat2d tn, const vector< vector<double> >& mag) {
    mat2d tn1 = tn;
    for(int sn = tn.vecs.size(), i = KERNEL_SIZE; i < sn-KERNEL_SIZE; i++) {
        for(int sm = tn.vecs[i].size(), j = KERNEL_SIZE; j < sm-KERNEL_SIZE; j++) {
            // For each possible point to apply the kernell

            vec2d new_vec(0,0);
            vec2d center_vec = tn.vecs[i][j];

            for(int di = -KERNEL_SIZE; di <= KERNEL_SIZE; di++) {
                for(int dj = -KERNEL_SIZE; dj <= KERNEL_SIZE; dj++) {
                    // cout << di << " " << dj << endl;
                    if(di != 0 || dj != 0) {
                        vec2d neighbor_vec = tn.vecs[i+di][j+dj];


                        new_vec += sign(center_vec, neighbor_vec)
                                    *neighbor_vec
                                    *ws(i, j, i+di, j+dj)
                                    *wm(mag[i][j], mag[i+di][j+dj])
                                    *wd(center_vec, neighbor_vec);
                    // cout << di << " " << dj << endl;
                    }
                }
            }
            tn1.vecs[i][j] = new_vec;
            tn1.vecs[i][j].normalize();
        }
    }

    return tn1;
}

mat2d ETF(mat2d t0, int num_iterations, mat2d grad) {
    grad.normalize();
    vector< vector<double> > mag;
    for(int n = grad.vecs.size(), i = 0; i < n; i++) {
        mag.push_back(vector<double>());
        for(int m = grad.vecs[i].size(), j = 0; j < m; j++) {
            mag[i].push_back(grad.vecs[i][j].length());

            // cout << "(" << grad.vecs[i][j].x << ", " << grad.vecs[i][j].y << ") ";
            // cout << mag[i][j] << " ";
        }
        // cout << endl;
    }

    mat2d tni = t0;
    for(int ni = 0; ni < num_iterations; ni++) {
        tni = ETF_iteration(tni, mag);
    }
    return tni;
}

mat2d get_init_t(mat2d grad) {
    for(int n = grad.vecs.size(), i = 0; i < n; i++) {
        for(int m = grad.vecs[i].size(), j = 0; j < m; j++) {
            double aux = grad.vecs[i][j].x;
            grad.vecs[i][j].x = grad.vecs[i][j].y;
            grad.vecs[i][j].y = -aux;
        }
    }
    grad.normalize();
    return grad;
}

/** @function main */
int main( int argc, char** argv ) {

    // string window_name = "Sobel Demo - Simple Edge Detector";

    /// Load an image
    Mat original = imread( argv[1] );
    Mat fm;

    original.convertTo(fm,CV_32F);

    if( !original.data )
    { return -1; }

    /// Create windows
    namedWindow( "Original", CV_WINDOW_AUTOSIZE );
    namedWindow( "Sobel", CV_WINDOW_AUTOSIZE );

    Mat gray = preprocess(original);
    gray.convertTo(fm,CV_32F);
    //gray.convertTo(gray, CV_64FC1);

    mat2d grad = sobel(preprocess (original));
    mat2d t0 = get_init_t(grad);
    
    // cout << grad.vecs[425][148].x << " " << grad.vecs[425][148].y << endl;
    // cout << t0.vecs[425][148].x << " " << t0.vecs[425][148].y << endl;

    mat2d etf = ETF(t0, 3, grad);

    for(int n = etf.vecs.size(), i = 0; i < n; i++) {
        for(int n = etf.vecs[i].size(), j = 0; j < n; j++) {
            cout << "(" << etf.vecs[i][j].x << ",  " << etf.vecs[i][j].y << ") ";
        }
        cout << endl;
    }

    imshow( "Original", original );
    // imshow( "Sobel", sobe );

    Mat final;
    Mat tangent(etf.vecs);

    vector< vector< pair<double, double> > > test;

    for(int n = etf.vecs.size(), i = 0; i < n; i++) {
        test.push_back(vector< pair<double, double> >());
        for(int n = etf.vecs[i].size(), j = 0; j < n; j++) {
            //cout << "(" << etf.vecs[i][j].x << ",  " << etf.vecs[i][j].y << ") ";
            //Scalar intensity = tangent.at<uchar>(i, j);
            test[i].push_back(pair<double, double>(etf.vecs[i][j].x, etf.vecs[i][j].y));
            //cout << tangent.at<double>(i,j) << " " << " ";
            //f.at<double>(i,j)
        }
        //cout << endl;
    }
    int sz = test.size()*2;
    //lime::LIC(fm, final, Mat(test), sz, lime::LIC_EULERIAN);

    //imshow( "final", final );

    waitKey(0);
}
