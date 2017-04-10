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
        int n = gx.rows;
        int m = gx.cols;
        for(int i = 0; i < n; i++) {
            vecs.push_back(vector<vec2d>());
            for(int j = 0; j < m; j++) {
                vec2d vec;
                vec.x = gx.at<double>(i,j);
                // cout << "(" << vec.x; 

                vec.y = gy.at<double>(i,j);
                // cout << ", " << vec.y << ") ";

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
    vector< vector<double> > coord(char c) {
        vector< vector<double> > ans;
        for(int n = vecs.size(), i = 0; i < n; i++) {
            ans.push_back(vector<double>());
            for(int m = vecs[i].size(), j = 0; j < m; j++) {
                ans[i].push_back((c == 'x' ? vecs[i][j].x : vecs[i][j].y));
            }
        }
        return ans;
    }
    vector< vector<double> > x() { return this->coord('x'); }
    vector< vector<double> > y() { return this->coord('y'); }

    vector< vector<double> > angle() {
        vector< vector<double> > ans;
        for(int n = vecs.size(), i = 0; i < n; i++) {
            ans.push_back(vector<double>());
            for(int m = vecs[i].size(), j = 0; j < m; j++) {
                ans[i].push_back(fmod(atan2(vecs[i][j].y, vecs[i][j].x)+M_PI, M_PI)/M_PI);
                // cout << "(" << vecs[i][j].y << ", " << vecs[i][j].x << ") " << fmod(atan2(vecs[i][j].y, vecs[i][j].x)+M_PI, M_PI)/M_PI << " ";
            }
            // cout << endl;
        }
        return ans;
    }
    Mat get_mat_coord(char coord) {
        vector< vector<double> > c;
        if(coord == 'x')
            c = x();
        else if(coord == 'y')
            c = y();
        else
            c = angle();
        Mat mat(c.size(), c.at(0).size(), CV_64FC1);
        for(int i=0; i<mat.rows; ++i)
            for(int j=0; j<mat.cols; ++j)
                mat.at<double>(i, j) = c[i][j];

        return mat;
    }
    Mat get_mat_x() { return get_mat_coord('x'); }
    Mat get_mat_y() { return get_mat_coord('y'); }
    Mat get_mat_angle() { return get_mat_coord('a'); }
    Mat get_mat() {
        Mat mat;
        vector<cv::Mat> marray = { get_mat_x(), get_mat_y() };
        merge(marray.data(), marray.size(), mat);
        return mat;
    }
    Mat LIC() {
        Mat mat = get_mat();
        Mat vfield(mat.rows, mat.cols, CV_32FC2);
        for (int y = 0; y < mat.rows; y++) {
            for (int x = 0; x < mat.cols; x++) {
                Vec2d theta = mat.at<Vec2d>(y, x);
                vfield.at<float>(y, x * 2 + 0) = static_cast<float>(theta[0]);
                vfield.at<float>(y, x * 2 + 1) = static_cast<float>(theta[1]);
            }
        }

        Mat noise, lic;
        lime::randomNoise(noise, cv::Size(mat.cols, mat.rows));
        lime::LIC(noise, lic, vfield, 20, lime::LIC_RUNGE_KUTTA);
        return lic;
    }
};

Mat preprocess(Mat src) {
    GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

    /// Convert it to gray
    cvtColor( src, src, CV_BGR2GRAY );

    return src;
}

mat2d sobel(Mat src_gray, Mat& mag) {
    int ddepth = CV_64F;
    Mat grad_x, grad_y;
    // Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    Sobel( src_gray, grad_x, ddepth, 1, 0, 3);
    // convertScaleAbs( grad_x, abs_grad_x );
    // imshow( "Sobel_x", abs_grad_x );

    /// Gradient Y
    Sobel( src_gray, grad_y, ddepth, 0, 1, 3);
    // convertScaleAbs( grad_y, abs_grad_y );
    // imshow( "Sobel_y", abs_grad_y );

    /// Total Gradient (approximate)
    mag = Mat(grad_x.rows, grad_x.cols, CV_64FC1);
    magnitude( grad_x, grad_y, mag );
    normalize(mag, mag, 1.0, 0.0, cv::NORM_MINMAX);
    imshow( "Sobel", mag );

    mat2d gmap(grad_x, grad_y);

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

mat2d ETF_iteration(mat2d tn, const Mat& mag) {
    mat2d tn1 = tn;
    for(int sn = tn.vecs.size(), i = KERNEL_SIZE; i < sn-KERNEL_SIZE; i++) {
        for(int sm = tn.vecs[i].size(), j = KERNEL_SIZE; j < sm-KERNEL_SIZE; j++) {
            // For each possible point to apply the kernell

            vec2d new_vec(0,0);
            vec2d center_vec = tn.vecs[i][j];

            for(int di = -KERNEL_SIZE; di <= KERNEL_SIZE; di++) {
                for(int dj = -KERNEL_SIZE; dj <= KERNEL_SIZE; dj++) {
                    if(di != 0 || dj != 0) {
                        vec2d neighbor_vec = tn.vecs[i+di][j+dj];


                        new_vec += sign(center_vec, neighbor_vec)
                                    *neighbor_vec
                                    *ws(i, j, i+di, j+dj)
                                    *wm(mag.at<double>(i, j), mag.at<double>(i+di, j+dj))
                                    *wd(center_vec, neighbor_vec);
                    }
                }
            }
            tn1.vecs[i][j] = new_vec;
            tn1.vecs[i][j].normalize();
        }
    }

    return tn1;
}

mat2d ETF(const mat2d& t0, int num_iterations, const Mat& mag) {

    mat2d tni = t0;
    for(int ni = 1; ni <= num_iterations; ni++) {
        tni = ETF_iteration(tni, mag);

        Mat lic = tni.LIC();

        imshow("ETF "+to_string(ni), lic);
    }
    return tni;
}

mat2d get_init_t(mat2d grad) {

    for(int n = grad.vecs.size(), i = 0; i < n; i++) {
        for(int m = grad.vecs[i].size(), j = 0; j < m; j++) {
            double aux = grad.vecs[i][j].x;
            grad.vecs[i][j].x = grad.vecs[i][j].y;
            grad.vecs[i][j].y = -1*aux;
        }
    }

    grad.normalize();

    Mat init = grad.LIC();

    imshow("Initial t", init);

    return grad;
}

cv::Mat drawimg64(const cv::Mat &m) {
    cv::Mat n;
    m.convertTo(n, CV_8UC1, 255, 0);
    return n;
}

/** @function main */
int main( int argc, char** argv ) {

    /// Load an image
    Mat original = imread( argv[1] );

    if( !original.data )
    { return -1; }

    /// Create windows
    namedWindow( "Original", CV_WINDOW_AUTOSIZE );

    Mat gray = preprocess (original);
    Mat mag;
    mat2d grad = sobel(gray, mag);
    mat2d t0 = get_init_t(grad);

    mat2d etf = ETF(t0, 3, mag);

    // for(int n = etf.vecs.size(), i = 0; i < n; i++) {
    //     for(int n = etf.vecs[i].size(), j = 0; j < n; j++) {
    //         cout << "(" << etf.vecs[i][j].x << ",  " << etf.vecs[i][j].y << ") ";
    //     }
    //     cout << endl;
    // }

    // Mat etfa = etf.get_mat_angle();
    // imshow("ETF ANGLE", etfa);

    imshow( "Original", original );

    // Mat lic = etf.LIC();
    // imshow( "Final", lic );
    // imwrite("etf_"+string(argv[1]), drawimg64(lic));

    waitKey(0);
}
