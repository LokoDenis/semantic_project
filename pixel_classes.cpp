#include <iostream>
#include <numeric>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/ximgproc.hpp>
#include <cmath>
#include <string>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ximgproc;

int determineColour(Vec3b input) {
    if (input.val[0] && input.val[1] && input.val[2])
        return 0;  // white imp_surf
    if (input.val[0] && !input.val[1] && !input.val[2])
        return 1;  // blue buildings
    if (input.val[0] && input.val[1] && !input.val[2])
        return 2; // cian low_veg
    if (!input.val[0] && input.val[1] && !input.val[2])
        return 3;  // green tree
    if (!input.val[0] && input.val[1] && input.val[2])
        return 4;  // yellow cars
    if (!input.val[0] && !input.val[1] && input.val[2])
        return 5;  // red clutter
}

class Pixel {
public:
    double x;
    double y;
    int green;
    int red;
    int infred;
    int intensity;
    int ndsm;
    double ndvi;
    int number;
    int colour;

    explicit Pixel(const Point2d p, Mat& src)
        : x(p.x)
        , y(p.y)
        , green(src.at<Vec3b>(Point(p.x, p.y))[0])
        , red(src.at<Vec3b>(Point(p.x, p.y))[1])
        , infred(src.at<Vec3b>(Point(p.x, p.y))[2])
    {
        ndvi = (static_cast<double>(this->infred) - this->red)/(this->infred + this->red);
        intensity = src.at<uchar>(Point(x, y));
    }

    void setNDSM(const Mat &src_ndsm) {
        ndsm = src_ndsm.at<uchar>(Point(x, y));
    }

    void setColour(const Mat &truth) {
        colour = determineColour(truth.at<Vec3b>(Point(x, y)));
    }

    void writeToCsvTable(std::string &path) {
        std::ofstream file_stream;
        file_stream.open(path, std::ofstream::out | std::ofstream::app);
        file_stream << number << "," << green <<"," << red << "," << infred << "," << intensity << ","
                    << ndsm << "," << ndvi << "," << colour <<",\n";
        file_stream.close();
    }

    void setNumber(int i) {
        number = i;
    }
};

double calculateMean(const std::vector<double> &input) {
    if (!input.size()) return 0;
    double sum = 0.0;
    for (auto elem : input) {
        sum += elem;
    }
    sum /= input.size();
    if (std::isnan(sum)) return 0;
    return sum;
}

double calculateStD(const std::vector<double> &input) {
    if (!input.size()) return 0;
    double mean = calculateMean(input);
    double accum = 0.0;
    for (auto elem : input) {
        accum += (elem - mean) * (elem - mean);
    }
    accum = std::sqrt(abs(accum / input.size()));
    if (std::isnan(accum)) return 0;
    return accum;
}

class SuperPixel {
public:
    double mean_green;
    double std_green;
    double mean_red;
    double std_red;
    double mean_infred;
    double std_infred;
    double mean_intensity;
    double std_intensity;
    double mean_ndsm;
    double std_ndsm;
    double mean_ndvi;
    double std_ndvi;
    int number;
    int colour;

    explicit SuperPixel (const std::vector<Pixel> &pixels) {
        std::vector<double> green_coll, red_coll, infred_coll, intensity_coll, ndsm_coll, ndvi_coll;
        for (auto elem : pixels) {
            green_coll.push_back(elem.green);
            red_coll.push_back(elem.red);
            infred_coll.push_back(elem.infred);
            intensity_coll.push_back(elem.intensity);
            ndsm_coll.push_back(elem.ndsm);
            ndvi_coll.push_back(elem.ndvi);
        }
        mean_green = calculateMean(green_coll);
        std_green = calculateStD(green_coll);
        mean_red = calculateMean(red_coll);
        std_red = calculateStD(red_coll);
        mean_infred = calculateMean(infred_coll);
        std_infred = calculateStD(infred_coll);
        mean_intensity = calculateMean(intensity_coll);
        std_intensity = calculateStD(intensity_coll);
        mean_ndsm = calculateMean(ndsm_coll);
        std_ndsm = calculateStD(ndsm_coll);
        mean_ndvi = calculateMean(ndvi_coll);
        std_ndvi = calculateStD(ndvi_coll);
        // finding majority class to be sure that colour is set correctly
        int maj_index = 0;
        int count = 1;
        for (size_t i = 1; i < pixels.size(); ++i) {
            if (pixels[maj_index].colour == pixels[i].colour) {
                ++count;
            } else --count;
            if (!count) {
                maj_index = i;
                count = 1;
            }
        }
        colour = pixels[maj_index].colour;
    }

    void writeToCsvTable(const std::string& path) {
        std::ofstream file_stream;
        file_stream.open(path, std::ofstream::out | std::ofstream::app);
        file_stream << number << "," << mean_green << "," << std_green << "," << mean_red << "," << std_red << ","
                    << mean_infred << "," << std_infred << "," << mean_intensity << "," << std_intensity << ","
                    << mean_ndsm << "," << std_ndsm << "," << mean_ndvi << "," << std_ndvi << "," << colour << ",\n";
        file_stream.close();
    }

    void setNumber(int i) {
        number = i;
    }
};
