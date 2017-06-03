#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/ximgproc.hpp>
#include <string>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ximgproc;

std::vector<std::vector<Point2d>> getSEEDSuperpixels(const Mat& src) {
    int num_iterations = 8;
    int prior = 2;
    bool double_step = true;
    int num_superpixels = 1000;
    int num_levels = 4;
    int num_histogram_bins = 5;
    Ptr<SuperpixelSEEDS> seeds;
    int width = src.size().width;
    int height = src.size().height;
    seeds = createSuperpixelSEEDS(width, height, src.channels(), num_superpixels,
                                  num_levels, prior, num_histogram_bins, double_step);
    Mat converted;
    cvtColor(src, converted, COLOR_BGR2HSV);
    seeds -> iterate(converted, num_iterations);

    Mat labels;
    seeds -> getLabels(labels);
    if (0) {
        Mat result, mask;
        result = src;
        seeds->getLabelContourMask(mask, false);
        result.setTo(Scalar(0, 0, 255), mask);
        imwrite("/home/sanity-seeker/Programming/Projects/Source/seg/top_mosaic_09cm_area1_SEED.tif", result);
        result.release();
    }
    int quantity = seeds->getNumberOfSuperpixels();
    std::vector<std::vector<Point2d>> superpixels(quantity, std::vector<Point2d>(0));
    for (size_t i = 0; i != labels.rows; ++i) {
        for (size_t j = 0; j != labels.cols; ++j) {
            superpixels[labels.at<int>(i, j)].push_back(Point(j, i));
        }
    }
    return superpixels;
}

std::vector<std::vector<Point2d>> getSLICSuperpixels(const Mat& src) {
    Ptr<SuperpixelSLIC> slics;
    int num_iterations = 10;
    slics = createSuperpixelSLIC(src, SLICO);
    slics -> iterate(num_iterations);
    Mat labels;
    slics -> getLabels(labels);
    if (0) {
        Mat result, mask;
        result = src;
        slics -> getLabelContourMask(mask, false);
        result.setTo(Scalar(0, 0, 255), mask);
        imwrite("/home/sanity-seeker/Programming/Projects/Source/seg/top_mosaic_09cm_area1_SLIC.tif", result);
        result.release();
    }
    int quantity = slics->getNumberOfSuperpixels();
    std::vector<std::vector<Point2d>> superpixels(quantity, std::vector<Point2d>(0));
    for (size_t i = 0; i != labels.rows; ++i) {
        for (size_t j = 0; j != labels.cols; ++j) {
            superpixels[labels.at<int>(i, j)].push_back(Point(j, i));
        }
    }
    return superpixels;
}
