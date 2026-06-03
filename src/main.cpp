/**
 * @file main.cpp
 * @brief Manual Implementation of Canny Edge Detector - Version 3.
 * * Course: Image Processing (ZPOe)
 * Institution: Brno University of Technology (BUT), Faculty of Information Technology 
 * * @author Kevin KUZU (xkuzuke00)
 * @date May 10, 2026
 * * @details This file implements the complete Canny pipeline from scratch, and compare
 * * the result of the manual implementation and the OpenCv implementation
 */


#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio> 

using namespace cv;
using namespace std;

// --- Global Variables ---
Mat src_original, src_processing, result_display;
int sigma_int = 14;
int low_threshold = 40;
int high_threshold = 120;

// --- Manual Gaussian Blur ---
void manualGaussianBlur(const Mat& src, Mat& dst, float sigma) {
    int size = (int)(6 * sigma); if (size % 2 == 0) size++;
    int center = size / 2;
    vector<float> kernel(size);
    float sum = 0;

    // Generate 1D Kernel
    for (int i = 0; i < size; i++) {
        float x = i - center;
        kernel[i] = exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    for (int i = 0; i < size; i++) kernel[i] /= sum; // Normalization

    Mat temp = Mat::zeros(src.size(), CV_32F);
    dst = Mat::zeros(src.size(), CV_32F);

    // Horizontal Pass
    for (int y = 0; y < src.rows; y++) {
        for (int x = center; x < src.cols - center; x++) {
            float val = 0;
            for (int k = -center; k <= center; k++)
                val += src.at<uchar>(y, x + k) * kernel[k + center];
            temp.at<float>(y, x) = val;
        }
    }
    
    // Vertical Pass
    for (int x = 0; x < src.cols; x++) {
        for (int y = center; y < src.rows - center; y++) {
            float val = 0;
            for (int k = -center; k <= center; k++) {
                val += temp.at<float>(y + k, x) * kernel[k + center];
            }
            dst.at<float>(y, x) = val;
        }
    }
}

void manualSobel(const Mat& src, Mat& magnitude, Mat& angle) {
    magnitude = Mat::zeros(src.size(), CV_32F);
    angle = Mat::zeros(src.size(), CV_32F);

    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            float gx = -1*src.at<float>(y-1,x-1) + 1*src.at<float>(y-1,x+1)
                       -2*src.at<float>(y,x-1)   + 2*src.at<float>(y,x+1)
                       -1*src.at<float>(y+1,x-1) + 1*src.at<float>(y+1,x+1);
            
            float gy = -1*src.at<float>(y-1,x-1) - 2*src.at<float>(y-1,x) - 1*src.at<float>(y-1,x+1)
                       +1*src.at<float>(y+1,x-1) + 2*src.at<float>(y+1,x) + 1*src.at<float>(y+1,x+1);

            magnitude.at<float>(y, x) = sqrt(gx*gx + gy*gy);
            angle.at<float>(y, x) = atan2(gy, gx) * 180.0 / CV_PI;
        }
    }
}

void nonMaximumSuppression(const Mat& magnitude, const Mat& angle, Mat& dst) {
    dst = Mat::zeros(magnitude.size(), CV_32F);
    for (int y = 1; y < magnitude.rows - 1; y++) {
        for (int x = 1; x < magnitude.cols - 1; x++) {
            float ang = angle.at<float>(y, x);
            if (ang < 0) ang += 180;

            float q = 255, r = 255;
            if ((ang >= 0 && ang < 22.5) || (ang >= 157.5 && ang <= 180)) {
                q = magnitude.at<float>(y, x + 1);
                r = magnitude.at<float>(y, x - 1);
            }
            else if (ang >= 67.5 && ang < 112.5) {
                q = magnitude.at<float>(y + 1, x);
                r = magnitude.at<float>(y - 1, x);
            }
            else if (ang >= 22.5 && ang < 67.5) {
                q = magnitude.at<float>(y + 1, x + 1);
                r = magnitude.at<float>(y - 1, x - 1);
            }
            else if (ang >= 112.5 && ang < 157.5) {
                q = magnitude.at<float>(y - 1, x + 1);
                r = magnitude.at<float>(y + 1, x - 1);
            }

            if (magnitude.at<float>(y, x) >= q && magnitude.at<float>(y, x) >= r)
                dst.at<float>(y, x) = magnitude.at<float>(y, x);
        }
    }
}

void doubleThreshold(const Mat& src, Mat& dst, int low, int high) {
    dst = Mat::zeros(src.size(), CV_8U);

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            float pixel = src.at<float>(y, x);
            if (pixel >= high) {
                dst.at<uchar>(y, x) = 255;
            } 
            else if (pixel >= low) {
                dst.at<uchar>(y, x) = 100;
            }
        }
    }
}

void hysteresis(Mat& img) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (int y = 1; y < img.rows - 1; y++) {
            for (int x = 1; x < img.cols - 1; x++) {
                if (img.at<uchar>(y, x) == 100) {
                    for (int i = -1; i <= 1; i++) {
                        for (int j = -1; j <= 1; j++) {
                            if (img.at<uchar>(y + i, x + j) == 255) {
                                img.at<uchar>(y, x) = 255;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
    }
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            if (img.at<uchar>(y, x) == 100) img.at<uchar>(y, x) = 0;
        }
    }
}

string openFileExplorer() {
    char buffer[1024];
    FILE* pipe = popen("zenity --file-selection --title='Choose your image (PNG/JPG)' --file-filter='*.jpg *.png *.jpeg'", "r");
    if (!pipe) return "";
    if (fgets(buffer, 1024, pipe) != NULL) {
        string path = buffer;
        path.erase(path.find_last_not_of("\n\r") + 1);
        pclose(pipe);
        return path;
    }
    pclose(pipe);
    return "";
}

// --- Pipeline Update Function ---
void updateCanny(int, void*) {
    if (src_processing.empty()) return;

    float sigma = (float)sigma_int / 10.0f;
    if (sigma < 0.1) sigma = 0.1;

    // --- 1. YOUR MANUAL RESULT ---
    Mat blurred, magnitude, angle, nms, result_manual;
    manualGaussianBlur(src_processing, blurred, sigma);
    manualSobel(blurred, magnitude, angle);
    nonMaximumSuppression(magnitude, angle, nms);
    doubleThreshold(nms, result_manual, low_threshold, high_threshold);
    hysteresis(result_manual);

    // --- 2. OPENCV OFFICIAL RESULT ---
    Mat result_opencv, blurred_cv;
    int ksize = (int)(6 * sigma); if (ksize % 2 == 0) ksize++;
    GaussianBlur(src_processing, blurred_cv, Size(ksize, ksize), sigma);
    Canny(blurred_cv, result_opencv, low_threshold, high_threshold);

    // --- 3. CALCULATE DIFFERENCE (ERROR MAP) ---
    Mat diff_img;
    absdiff(result_manual, result_opencv, diff_img);

    int total_pixels = diff_img.rows * diff_img.cols;
    int diff_count = countNonZero(diff_img); 
    double error_rate = (double)diff_count / total_pixels * 100.0;

    cout << "\033[1;32m--- Comparison Analysis ---\033[0m" << endl;
    cout << "Total Pixels: " << total_pixels << endl;
    cout << "Divergent Pixels: " << diff_count << endl;
    cout << "Error Rate: " << error_rate << " %" << endl;
    cout << "Accuracy: " << (100.0 - error_rate) << " %" << endl << endl;

    // --- 4. TRIPLE DISPLAY ---
    Mat comparison;
    vector<Mat> images = { result_manual, result_opencv, diff_img };
    hconcat(images, comparison);

    putText(comparison, "MANUAL", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255), 2);
    putText(comparison, "OPENCV", Point(result_manual.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255), 2);
    putText(comparison, "DIFF (ERRORS)", Point(result_manual.cols * 2 + 10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255), 2);

    imshow("Canny Detector", comparison);
}

void loadAndPrepareImage(string path) {
    src_original = imread(path, IMREAD_GRAYSCALE);
    if (src_original.empty()) return;

    int max_dim = 800; 
    float scale = min((float)max_dim / src_original.cols, (float)max_dim / src_original.rows);
    if (scale < 1.0) {
        resize(src_original, src_processing, Size(), scale, scale, INTER_AREA);
    } else {
        src_processing = src_original.clone();
    }
    updateCanny(0, 0);
}

void saveHighResResult() {
    if (src_original.empty()) return;

    cout << "Saving..." << endl;

    float sigma = (float)sigma_int / 10.0f;
    if (sigma < 0.1) sigma = 0.1;

    Mat blurred, magnitude, angle, nms, result;
    manualGaussianBlur(src_original, blurred, sigma);
    manualSobel(blurred, magnitude, angle);
    nonMaximumSuppression(magnitude, angle, nms);
    doubleThreshold(nms, result, low_threshold, high_threshold);
    hysteresis(result);

    string filename = "canny_result_high_res.png";
    if (imwrite(filename, result)) {
        cout << "SUCCESS: Image saved as " << filename << " in the project folder." << endl;
    } else {
        cout << "ERROR: Failed to save the image. So bad..." << endl;
    }
}

int main(int argc, char** argv) {
    string path;
    if (argc < 2) {
        cout << "No file provided. Opening file explorer..." << endl;
        path = openFileExplorer();
    } else {
        path = argv[1];
    }

    if (path.empty()) return -1;

    namedWindow("Canny Detector", WINDOW_AUTOSIZE);
    createTrackbar("Sigma x10", "Canny Detector", &sigma_int, 50, updateCanny);
    createTrackbar("Low Thr", "Canny Detector", &low_threshold, 255, updateCanny);
    createTrackbar("High Thr", "Canny Detector", &high_threshold, 255, updateCanny);

    loadAndPrepareImage(path);

    cout << "-----------------------------------------" << endl;
    cout << "KEYBOARD COMMANDS:" << endl;
    cout << " 'o' : Open new image" << endl;
    cout << " 's' : SAVE High-Res Result :) " << endl;
    cout << " 'q' : Quit :(" << endl;
    cout << "-----------------------------------------" << endl;

    while (true) {
        char key = (char)waitKey(10);
        if (key == 'q' || key == 27) break;
        
        if (key == 'o') { 
            string newPath = openFileExplorer();
            if (!newPath.empty()) loadAndPrepareImage(newPath);
        }

        if (key == 's') { 
            saveHighResResult();
        }
    }
    cout << "Bye bye !" << endl;
    return 0;
}