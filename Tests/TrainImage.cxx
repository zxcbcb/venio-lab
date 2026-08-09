// Task 5: train an end-to-end model on a photo.
// A coordinate-MLP maps (x,y)+positional features -> pixel intensity and is
// trained with GD + SquareError to reconstruct a downsampled photo. Reports the
// loss dropping over epochs and saves the reconstruction as PNG.
// argv[1] = input photo, argv[2] = output reconstruction png.
#include "Venio.hxx"
#include "ImageIO.hxx"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>

static const double PI = 3.14159265358979323846;

static Matrixd feats(double xn, double yn)
{
    Matrixd f(1, 10);
    f(0, 0) = xn;
    f(0, 1) = yn;
    f(0, 2) = std::sin(PI * xn);       f(0, 3) = std::cos(PI * xn);
    f(0, 4) = std::sin(2 * PI * xn);   f(0, 5) = std::cos(2 * PI * xn);
    f(0, 6) = std::sin(PI * yn);       f(0, 7) = std::cos(PI * yn);
    f(0, 8) = std::sin(2 * PI * yn);   f(0, 9) = std::cos(2 * PI * yn);
    return f;
}

int main(int argc, char **argv)
{
    Eigen::setNbThreads(4);
    std::string inPath = (argc > 1) ? argv[1] : "Tests/data/photo.jpg";
    std::string outPath = (argc > 2) ? argv[2] : "Tests/data/photo_recon.png";
    const int S = 40;          // downsample target
    const int epochs = 300;
    const double lr = 0.2;

    Matrixd full = ImageIO::loadGray(inPath);
    if (full.size() == 0) { std::cout << "load fail: " << inPath << "\n"; return 1; }

    Matrixd img(S, S);
    for (int y = 0; y < S; ++y)
        for (int x = 0; x < S; ++x)
            img(y, x) = full(static_cast<int>((double)y / S * full.rows()),
                             static_cast<int>((double)x / S * full.cols()));

    TH tanhf;
    LogisticFunction logistic;
    LinearFunction linear;
    SquareErrorFunction square;

    std::vector<std::shared_ptr<Layer>> layers{
        std::make_shared<SequentialLayer>(10, &linear),
        std::make_shared<SequentialLayer>(64, &tanhf),
        std::make_shared<SequentialLayer>(64, &tanhf),
        std::make_shared<SequentialLayer>(1, &logistic),
    };
    Model net(&square, layers);

    // Symmetric small-weight init (the default init is all-positive and trains poorly).
    for (int i = 1; i < (int)net.getLayersSize(); ++i)
    {
        Matrixd w = net.getLayerWeights(i);
        net.setLayerWeights(i, RandomGenerator::generateRandomMatrix(-0.3, 0.3, w.rows(), w.cols()));
        Matrixd b = net.getLayerBias(i);
        net.setLayerBias(i, Matrixd::Zero(b.rows(), b.cols()));
    }

    GD opt(net);

    auto avgLoss = [&]() -> double {
        double s = 0.0;
        for (int y = 0; y < S; ++y)
            for (int x = 0; x < S; ++x)
            {
                double xn = 2.0 * x / (S - 1) - 1.0, yn = 2.0 * y / (S - 1) - 1.0;
                Matrixd in = feats(xn, yn);
                net.setInput(in);
                net.forwardPropogation();
                Matrixd t(1, 1); t(0, 0) = img(y, x);
                s += net.getAverageLoss(t);
            }
        return s / (S * S);
    };

    std::cout << std::fixed << std::setprecision(6);
    double l0 = avgLoss();
    std::cout << "INIT_LOSS " << l0 << "\n";

    for (int e = 1; e <= epochs; ++e)
    {
        for (int y = 0; y < S; ++y)
            for (int x = 0; x < S; ++x)
            {
                double xn = 2.0 * x / (S - 1) - 1.0, yn = 2.0 * y / (S - 1) - 1.0;
                Matrixd in = feats(xn, yn);
                net.setInput(in);
                net.forwardPropogation();
                Matrixd t(1, 1); t(0, 0) = img(y, x);
                net.backPropogation(t);
                opt.updateWeights(lr, e);
            }
        if (e % 50 == 0)
            std::cout << "epoch " << e << " loss " << avgLoss() << "\n";
    }
    double lf = avgLoss();

    Matrixd recon(S, S);
    for (int y = 0; y < S; ++y)
        for (int x = 0; x < S; ++x)
        {
            double xn = 2.0 * x / (S - 1) - 1.0, yn = 2.0 * y / (S - 1) - 1.0;
            Matrixd in = feats(xn, yn);
            net.setInput(in);
            net.forwardPropogation();
            recon(y, x) = net.getOutput()(0, 0);
        }
    bool saved = ImageIO::saveGray(outPath, recon);

    std::cout << "FINAL_LOSS " << lf << "\n";
    std::cout << "SAVED " << saved << " (" << outPath << ")\n";
    std::cout << ((lf < l0 * 0.5) ? "TRAIN OK (loss more than halved)\n" : "TRAIN WEAK\n");
    return 0;
}
