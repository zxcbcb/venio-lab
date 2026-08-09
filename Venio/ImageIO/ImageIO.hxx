#pragma once
#include <Eigen/Core>
#include <string>

// Tiny image I/O wrapper around stb_image / stb_image_write.
// Grayscale, values in [0,1]. Rows = image height, cols = image width.
namespace ImageIO
{
    Eigen::MatrixXd loadGray(const std::string &path); // empty matrix on failure
    bool saveGray(const std::string &path, const Eigen::MatrixXd &img);
}
