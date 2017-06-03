#include <iostream>
#include "opencv2/core/core.hpp"
#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;

void createConfig() {
    FileStorage fs("/home/sanity-seeker/Programming/Projects/semantic_project/config.yml", FileStorage::WRITE);
    fs << "truth" << "/home/sanity-seeker/Programming/Projects/Source/truth/";
    fs << "top" << "/home/sanity-seeker/Programming/Projects/Source/original/";
    fs << "ndsm" << "/home/sanity-seeker/Programming/Projects/Source/ndsm/";
    fs << "training_data" << "/home/sanity-seeker/Programming/Projects/semantic_project/training data/";
    fs << "testing_data" << "/home/sanity-seeker/Programming/Projects/semantic_project/testing data/";
    fs.release();
}

void loadConfig(std::string &truth, std::string &top, std::string &ndsm, std::string &training, std::string &testing) {
    FileStorage fs("/home/sanity-seeker/Programming/Projects/segmentation/semantic_project/config.yml", FileStorage::READ);
    fs["truth"] >> truth;
    fs["top"] >> top;
    fs["ndsm"] >> ndsm;
    fs["training_data"] >> training;
    fs["testing_data"] >> testing;
    fs.release();
}

std::string getFilePath (const std::string& str) {
    size_t found_slash = str.find_last_of("/");
    return str.substr(0, found_slash + 1);
}

std::string getFileName (const std::string& str) {
    size_t found_slash = str.find_last_of("/");
    size_t found_dot = str.find_last_of(".");
    return str.substr(found_slash + 1, found_dot);
}

std::string getFileExtension(const std::string& str) {
    size_t found_dot = str.find_last_of(".");
    return str.substr(found_dot);
}

void writePixelHatCsv(const std::string& path) {
    std::ofstream file_stream;
    file_stream.open(path, std::ofstream::out | std::ofstream::trunc);
    file_stream << "N,Green,Red,Infrared,Intensity,NDSM,NDVI,Colour, \n";
    file_stream.close();
}

void writeSuperPixelHatCsv(const std::string& path) {
    std::ofstream file_stream;
    file_stream.open(path, std::ofstream::out | std::ofstream::trunc);
    file_stream << "N,MeanGreen,StdGreen,MeanRed,StdRed,MeanInfrared,StdInfrared,MeanIntensity,StdIntensity"
                << ",MeanNDSM,StdNDSM,MeanNDVI,StdNDVI,Colour, \n";
    file_stream.close();
}

void rotate(cv::Mat& src, double angle, cv::Mat& dst) {
    Point2d ptCp(src.cols * 0.5, src.rows * 0.5);
    Mat M = getRotationMatrix2D(ptCp, angle, 1.0);
    warpAffine(src, dst, M, src.size(), INTER_CUBIC);
}
