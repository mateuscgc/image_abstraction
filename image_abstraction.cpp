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

string img_name;

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
    double angle(double signal) const {
        return atan2(signal*y, signal*x);
    }
    double perp(double signal) const {
        return atan2(signal*x, signal*-y);
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

cv::Mat drawimg64(const cv::Mat &m) {
    cv::Mat n;
    m.convertTo(n, CV_8UC1, 255, 0);
    return n;
}

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
    // imshow( "Sobel", mag );

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


        // Mat lic = tni.LIC();
        // imshow("ETF "+to_string(ni), lic);
        // imwrite("ETF_"+to_string(ni)+"_"+img_name, drawimg64(lic));
    }

    Mat lic = tni.LIC();
    imshow("ETF", lic);
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

    // imshow("Initial t", init);

    return grad;
}

pair<int, int> flow_neighbor(double angle) {
    double limits[] = { 0, 45, 90, 135, 180, -135, -90, -45 };
    vector< pair<int, int> > d = { {0,1}, {-1,1}, {-1,0}, {-1,-1}, {0,-1}, {1,-1}, {1,0}, {1,1} };
    int closest = 0;
    for(int i = 0; i < 8; i++) {
        if(abs(angle-limits[i]) < abs(angle-limits[closest]))
            closest = i;
    }
    return d[closest];
}

double gaussian(int s, double sig) {
    return (1/sqrt(2*M_PI*sig))*pow(M_E, -(s*s)/(2*0.58*sig));
}

double spatial_weight(int s, double sig) {
    return gaussian(s, sig);
}

double color_dist(Vec3d c1, Vec3d c2) {
    // GBR
    double dr = c1[2]-c2[2];
    double dg = c1[0]-c2[0];
    double db = c1[1]-c2[1];
    return sqrt(2*(dr*dr) + 4*(dg*dg) + 3*(db*db));
}

double similarity_weight(Vec3d c1, Vec3d c2, double sig) {
    return gaussian(color_dist(c1, c2), sig);
}


Mat FBL_iteration(const Mat& img, const mat2d& etf) {
    Mat smoothed = img;
    for(int sn = img.rows, i = 0; i < sn; i++) {
        for(int sm = img.cols, j = 0; j < sm; j++) {

            // vec2d new_vec(0,0);
            // vec2d center_vec = tn.vecs[i][j];
            double sig = 2.0;

            double ve = spatial_weight(0, sig)*similarity_weight(img.at<Vec3d>(i,j), img.at<Vec3d>(i,j), sig);
            Vec3d new_c = img.at<Vec3d>(i,j)*ve;

            for(double m = -1; m <= 1; m += 2) {
                int di = i, dj = j;
                for(int s = 1; s <= KERNEL_SIZE; s++) {
                    double angle = etf.vecs[di][dj].angle(m);
                    // cout << "angle " << s*m << " " << angle*180/M_PI << endl;
                    pair<int, int> d = flow_neighbor(angle*180/M_PI);
                    
                    di += d.first;
                    dj += d.second;
                    if(di >= 0 && di < img.rows && dj >= 0 && dj < img.cols) {
                        double weight = spatial_weight(s*m, sig)*similarity_weight(img.at<Vec3d>(i,j), img.at<Vec3d>(di,dj), sig);
                        ve += weight;
                        new_c += img.at<Vec3d>(di, dj)*weight;
                    } else {
                        break;
                    }
                }
            }

            smoothed.at<Vec3d>(i,j) = new_c/ve;
        }
    }

    Mat smoothed2 = smoothed;

    for(int sn = smoothed.rows, i = 0; i < sn; i++) {
        for(int sm = smoothed.cols, j = 0; j < sm; j++) {

            // vec2d new_vec(0,0);
            // vec2d center_vec = tn.vecs[i][j];
            double sig = 0.3;

            double ve = spatial_weight(0, sig)*similarity_weight(smoothed.at<Vec3d>(i,j), smoothed.at<Vec3d>(i,j), sig);
            Vec3d new_c = smoothed.at<Vec3d>(i,j)*ve;

            for(double m = -1; m <= 1; m += 2) {
                double angle = etf.vecs[i][j].perp(m);
                int di = i, dj = j;

                for(int s = 1; s <= KERNEL_SIZE; s++) {
                    // cout << "angle " << s*m << " " << angle*180/M_PI << endl;
                    pair<int, int> d = flow_neighbor(angle*180/M_PI);
                    
                    di += d.first;
                    dj += d.second;
                    if(di >= 0 && di < smoothed.rows && dj >= 0 && dj < smoothed.cols) {
                        double weight = spatial_weight(s*m, sig)*similarity_weight(smoothed.at<Vec3d>(i,j), smoothed.at<Vec3d>(di,dj), sig);
                        ve += weight;
                        new_c += smoothed.at<Vec3d>(di, dj)*weight;
                    } else {
                        break;
                    }
                }
            }
            smoothed2.at<Vec3d>(i,j) = new_c/ve;
        }
    }
    return smoothed2;
}

Mat FBL(Mat cur, int num_iterations, mat2d etf) {

    for(int ni = 1; ni <= num_iterations; ni++) {
        cur = FBL_iteration(cur, etf);


        // imwrite("FBL_"+to_string(ni)+"_"+img_name, drawimg64(cur));

    }
    imshow("FBL", cur);
    return cur;
}

struct mcq {
    Vec3d c;
    pair<int, int> pos;
};

struct cmp {
    int color;
    cmp(int c) : color(c) {}
    bool operator () (mcq a, mcq b) {
        return a.c[color] < b.c[color];
    }
};

void set_average_color(mcq* quant, Mat& img, int ini, int end) {
    Vec3d new_color;
    // cout << new_color[1] << endl;
    for(int i = ini; i < end; i++) {
        new_color += quant[i].c;
    }
    new_color /= end-ini;
    for(int i = ini; i < end; i++) {
        img.at<Vec3d>(quant[i].pos.first, quant[i].pos.second) = new_color;
    }
}

void divaide(mcq* quant, int ini, int end, int remain, Mat &cur) {
    // cout << remain << endl;
    if(!remain) {
        // cout << ini << " " << end << endl;
        set_average_color(quant, cur, ini, end);
        return;
    }
    double maxi[] = {0,0,0};
    double mini[] = {1,1,1};
    for(int i = ini; i < end; i++) {
        for(int c = 0; c < 3; c++) {
            maxi[c] = max(maxi[c], quant[i].c[c]);
            mini[c] = min(mini[c], quant[i].c[c]);
        }
    }

    int cur_color = 0;
    for(int c = 0; c < 3; c++)
        if(maxi[c]-mini[c] > maxi[cur_color]-mini[cur_color])
            cur_color = c;

    sort(quant+ini, quant+end, cmp(cur_color));

    int mid = (ini + end)/2;
    divaide(quant, ini, mid, remain-1, cur);
    divaide(quant, mid, end, remain-1, cur);
}

Mat median_cut_quantization(Mat cur, int number_of_colors) {
    int i;
    for(i = 0; (1 << i) <= number_of_colors; i++);
    number_of_colors = (1 << (i-1));
    int number_of_iterations = i-1;
    
    int n = cur.rows*cur.cols;
    // cout << n << endl;
    mcq* quant = new mcq[n];
    // cout << n << endl;

    for(int i = 0; i < cur.rows; i++) {
        // cout << i << endl;
        for(int j = 0; j < cur.cols; j++) {
            mcq aux;
            aux.c = cur.at<Vec3d>(i,j);
            aux.pos = make_pair(i, j);
            quant[i*cur.cols + j] = aux;
        }
    }

    divaide(quant, 0, n, number_of_iterations, cur);

    //Define colors
    // for(int i = 0; i < number_of_colors; i++)
        // set_average_color(&quant, cur, i*(n/number_of_colors)) 
    //Set new colors in image

    return cur; 
}


/** @function main */
int main( int argc, char** argv ) {

    /// Load an image
    Mat original = imread( argv[1] );
    img_name = string(argv[1]);

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
    //imwrite("etf_"+string(argv[1]), drawimg64(lic));

    // ======================

    Mat dcolor;
    // cvtColor( original, dgray, CV_BGR2GRAY );
    original.convertTo(dcolor, CV_64FC3);
    dcolor /= 255;
    imshow("test", dcolor);
    Mat smoothed = FBL(dcolor, 5, etf);

    Mat quantized = median_cut_quantization(smoothed, 64);
    imshow("Quantized", quantized);
    imwrite("Quantized_"+string(argv[1]), drawimg64(quantized));

    waitKey(0);
}
