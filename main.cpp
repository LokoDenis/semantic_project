#include <iostream>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/ximgproc.hpp>
#include <string>
#include "segmentations.cpp"
#include "pixel_classes.cpp"
#include "useful_functions.cpp"
#include <fstream>

void createCSV(const std::string& result_path, const std::string& original_path,
               const std::string& truth_path, const std::string& ndsm_path,
               int number_required=1, int start=1, int direction=-1,
               bool enable_rotations=false, int rotation_angle=60) {

    writeSuperPixelHatCsv(result_path);
    std::string base_str = "top_mosaic_09cm_area";  // .tif
    std::string base_ndsm = "ndsm_09cm_matching_area"; // .bmp

    int different_images = 0;

    int sup_number = 1;

    while (different_images < number_required) {
        Mat curr_truth = imread(truth_path + base_str + std::to_string(start) + ".tif", CV_LOAD_IMAGE_UNCHANGED);
        if (curr_truth.empty()) {
            start += direction;
            continue;
        }
        Mat curr_image = imread(original_path + base_str + std::to_string(start) + ".tif", CV_LOAD_IMAGE_UNCHANGED);
        Mat curr_ndsm = imread(ndsm_path + base_ndsm + std::to_string(start) + ".bmp", CV_LOAD_IMAGE_UNCHANGED);
        int rot_count = 0;
        int rot_desired = (enable_rotations ? 360 / rotation_angle : 1);
        ++different_images;

        FileStorage fs_pixels(getFilePath(result_path) + base_str + std::to_string(start) + ".yml", FileStorage::WRITE);
        fs_pixels << "width" << curr_image.cols;
        fs_pixels << "height" << curr_image.rows;
        bool have_written = false;

        while (rot_count++ < rot_desired) {
            std::cout << "starting " + std::to_string(start) + " picture\n";
            std::cout << "\tstarting segmentation \n";
            std::vector<std::vector<Point2d>> raw_superpixels = getSLICSuperpixels(curr_image);
            std::cout << "\tfinishing segmentation \n";
            std::vector<Pixel> superpixel;

            std::cout << "\tstarting counting superpixels \n";
            for (auto array : raw_superpixels) {
                for (auto pixel : array) {
                    Pixel curr_pixel(pixel, curr_image);
                    curr_pixel.setColour(curr_truth);
                    curr_pixel.setNDSM(curr_ndsm);
                    superpixel.push_back(curr_pixel);
                }
                SuperPixel curr_superpixel(superpixel);
                if (!have_written) {
                    write(fs_pixels, "SuperPixel_" + std::to_string(sup_number), array);
                }
                curr_superpixel.setNumber(sup_number++);
                curr_superpixel.writeToCsvTable(result_path);
                superpixel.clear();
            }
            std::cout << "\tfinishing counting superpixels \n";
            fs_pixels << "quantity" << sup_number - 1;
            have_written = true;
            fs_pixels.release();
            rotate(curr_truth, rotation_angle, curr_truth);
            rotate(curr_image, rotation_angle, curr_image);
            rotate(curr_ndsm, rotation_angle, curr_ndsm);
        }
        curr_truth.release();
        curr_image.release();
        curr_ndsm.release();
        std::cout << std::to_string(start) + " picture finished\n";
        start += direction;
    }
}

int main() {
    std::string original_path, groundtruth_path, pixeltable_path, superpixeltable_path, training_path, testing_path, ndsm_path;
    loadConfig(groundtruth_path, original_path, ndsm_path, training_path, testing_path);
    superpixeltable_path = testing_path + "32.csv";
    createCSV(superpixeltable_path, original_path, groundtruth_path, ndsm_path, 1, 32, -1);
    return 0;
}