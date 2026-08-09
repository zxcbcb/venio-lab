#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "ImageIO.hxx"
#include <vector>

namespace ImageIO
{
    Eigen::MatrixXd loadGray(const std::string &path)
    {
        int w = 0, h = 0, ch = 0;
        unsigned char *data = stbi_load(path.c_str(), &w, &h, &ch, 1); // force 1 channel
        if (!data)
            return Eigen::MatrixXd();
        Eigen::MatrixXd img(h, w);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                img(y, x) = data[y * w + x] / 255.0;
        stbi_image_free(data);
        return img;
    }

    bool saveGray(const std::string &path, const Eigen::MatrixXd &img)
    {
        const int h = static_cast<int>(img.rows());
        const int w = static_cast<int>(img.cols());
        std::vector<unsigned char> buf(static_cast<size_t>(w) * h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
            {
                double v = img(y, x);
                v = v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
                buf[static_cast<size_t>(y) * w + x] = static_cast<unsigned char>(v * 255.0 + 0.5);
            }
        return stbi_write_png(path.c_str(), w, h, 1, buf.data(), w) != 0;
    }
}
