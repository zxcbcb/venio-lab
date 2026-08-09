// Task 4: test layers on NOISE and on a PHOTO.
// Runs the ConvolutionLayer with a 3x3 edge kernel on (a) a random noise image
// and (b) a real photo loaded from disk, saving the filtered photo as PNG.
// argv[1] = input photo path, argv[2] = output png path.
#include "Venio.hxx"
#include "ImageIO.hxx"
#include <iostream>
#include <iomanip>
#include <random>
#include <string>

int main(int argc, char **argv)
{
    Eigen::setNbThreads(1);
    LinearFunction linear;

    std::string inPath = (argc > 1) ? argv[1] : "Tests/data/photo.jpg";
    std::string outPath = (argc > 2) ? argv[2] : "Tests/data/photo_edges.png";

    Matrixd edge(3, 3);
    edge << -1, -1, -1,
            -1,  8, -1,
            -1, -1, -1;

    ConvolutionLayer conv(3, 3, &linear);

    std::cout << std::fixed << std::setprecision(6);

    // ---- NOISE ----
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const int N = 64;
    Matrixd noise(N, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            noise(i, j) = uni(rng);

    conv.setKernel(edge);
    conv.setConvBias(0.0);
    conv.propogateLayer(noise);
    Matrixd nout = conv.getLayerActiveValues();
    bool noiseOK = (nout.rows() == N - 2 && nout.cols() == N - 2 && nout.allFinite());
    std::cout << "NOISE  in " << N << "x" << N << " -> out " << nout.rows() << "x" << nout.cols()
              << " finite=" << nout.allFinite() << " sum=" << nout.sum() << "\n";

    // ---- PHOTO ----
    Matrixd img = ImageIO::loadGray(inPath);
    if (img.size() == 0)
    {
        std::cout << "PHOTO load FAILED: " << inPath << "\n";
        return 1;
    }
    std::cout << "PHOTO  loaded " << img.rows() << "x" << img.cols() << "\n";

    conv.setKernel(edge);
    conv.setConvBias(0.0);
    conv.propogateLayer(img);
    Matrixd pout = conv.getLayerActiveValues();

    // normalize to [0,1] for viewing
    double mn = pout.minCoeff(), mx = pout.maxCoeff();
    Matrixd vis = (pout.array() - mn) / (mx - mn + 1e-9);
    bool saved = ImageIO::saveGray(outPath, vis);
    bool photoOK = (pout.rows() == img.rows() - 2 && pout.cols() == img.cols() - 2 && pout.allFinite() && saved);
    std::cout << "PHOTO  edges -> out " << pout.rows() << "x" << pout.cols()
              << " finite=" << pout.allFinite() << " saved=" << saved << " (" << outPath << ")\n";

    std::cout << ((noiseOK && photoOK) ? "LAYER IMAGE TEST OK\n" : "LAYER IMAGE TEST FAIL\n");
    return (noiseOK && photoOK) ? 0 : 1;
}
