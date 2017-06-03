#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/ximgproc.hpp>
#include "useful_functions.cpp"
#include "yaml-cpp/yaml.h"

Vec3b getColour(int code) {
    switch (code) {
        case 0: {
            Vec3b point(255, 255, 255);
            return point;
        }
        case 1: {
            Vec3b point(255, 0, 0);
            return point;
        }
        case 2: {
            Vec3b point(255, 255, 0);
            return point;
        }
        case 3: {
            Vec3b point(0, 255, 0);
            return point;
        }
        case 4: {
            Vec3b point(0, 255, 255);
            return point;
        }
        case 5: {
            Vec3b point(0, 0, 255);
            return point;
        }
        default: {
            Vec3b point(0, 0, 0);
            return point;
        }
    }
}

int main(int argc, char** argv) {
    std::string original_path, groundtruth_path, training_path, testing_path, ndsm_path;
    loadConfig(groundtruth_path, original_path, ndsm_path, training_path, testing_path);
    std::string path = argv[1];
    FileStorage fs_pixels(testing_path + path, FileStorage::READ);
    path = argv[2];
    YAML::Node fs_answers = YAML::LoadFile(testing_path + path);
    std::string base_str = "SuperPixel_";
    int height, width, count;
    fs_pixels["width"] >> width;
    fs_pixels["height"] >> height;
    fs_pixels["quantity"] >> count;

    Mat image(height, width, CV_8UC3, Scalar(0, 0, 0));
    std::vector<cv::Point2d> curr_segment;
    int curr_colour;
    for (size_t i = 1; i != count; ++i) {
        FileNode vecFileNode = fs_pixels[base_str + std::to_string(i)];
        read(vecFileNode, curr_segment);
        curr_colour = fs_answers[std::to_string(i)].as<int>();
        for (auto elem : curr_segment) {
            image.at<Vec3b>(elem.y, elem.x) = getColour(curr_colour);
        }
    }
    fs_pixels.release();
    GaussianBlur(image, image, Size(3, 3), 0, 0);
    imshow("example", image);
    waitKey(0);
    imwrite(testing_path + getFileName(argv[2]) + ".jpg", image);
    image.release();
}