// Compares the project's optimizers on the same task: fit a downsampled photo
// with a coordinate-MLP. Identical deterministic weight init for every run, so
// only the optimizer differs. Reports init/final average loss per optimizer.
#include "Venio.hxx"
#include "ImageIO.hxx"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <memory>
#include <string>

static const double PI = 3.14159265358979323846;

static Matrixd feats(double xn, double yn)
{
    Matrixd f(1, 10);
    f(0, 0) = xn; f(0, 1) = yn;
    f(0, 2) = std::sin(PI * xn);     f(0, 3) = std::cos(PI * xn);
    f(0, 4) = std::sin(2 * PI * xn); f(0, 5) = std::cos(2 * PI * xn);
    f(0, 6) = std::sin(PI * yn);     f(0, 7) = std::cos(PI * yn);
    f(0, 8) = std::sin(2 * PI * yn); f(0, 9) = std::cos(2 * PI * yn);
    return f;
}

static void initDet(Model &net)
{
    for (int i = 1; i < (int)net.getLayersSize(); ++i)
    {
        Matrixd w = net.getLayerWeights(i);
        for (int a = 0; a < w.rows(); ++a)
            for (int b = 0; b < w.cols(); ++b)
                w(a, b) = 0.15 * std::sin(1.0 * a + 2.0 * b + 3.0 * i);
        net.setLayerWeights(i, w);
        Matrixd bb = net.getLayerBias(i);
        net.setLayerBias(i, Matrixd::Zero(bb.rows(), bb.cols()));
    }
}

static double avgLoss(Model &net, const Matrixd &img, int S)
{
    double s = 0.0;
    for (int y = 0; y < S; ++y)
        for (int x = 0; x < S; ++x)
        {
            double xn = 2.0 * x / (S - 1) - 1.0, yn = 2.0 * y / (S - 1) - 1.0;
            net.setInput(feats(xn, yn));
            net.forwardPropogation();
            Matrixd t(1, 1); t(0, 0) = img(y, x);
            s += net.getAverageLoss(t);
        }
    return s / (S * S);
}

static double trainWith(int optType, const char *name, double lr,
                        const Matrixd &img, int S, int epochs)
{
    TH tanhf; LogisticFunction logistic; LinearFunction linear; SquareErrorFunction square;
    std::vector<std::shared_ptr<Layer>> layers{
        std::make_shared<SequentialLayer>(10, &linear),
        std::make_shared<SequentialLayer>(48, &tanhf),
        std::make_shared<SequentialLayer>(48, &tanhf),
        std::make_shared<SequentialLayer>(1, &logistic),
    };
    Model net(&square, layers);
    initDet(net);

    Optimizer *opt = nullptr;
    switch (optType)
    {
        case 0: opt = new GD(net); break;
        case 1: opt = new ADAM(net); break;
        case 2: opt = new RMSProp(net); break;
        case 3: opt = new Adagrad(net); break;
        case 4: opt = new Adadelta(net); break;
    }

    double l0 = avgLoss(net, img, S);
    for (int e = 1; e <= epochs; ++e)
        for (int y = 0; y < S; ++y)
            for (int x = 0; x < S; ++x)
            {
                double xn = 2.0 * x / (S - 1) - 1.0, yn = 2.0 * y / (S - 1) - 1.0;
                net.setInput(feats(xn, yn));
                net.forwardPropogation();
                Matrixd t(1, 1); t(0, 0) = img(y, x);
                net.backPropogation(t);
                opt->updateWeights(lr, e);
            }
    double lf = avgLoss(net, img, S);
    delete opt;

    std::cout << std::left << std::setw(10) << name << std::right
              << "  lr=" << std::fixed << std::setprecision(4) << lr
              << "  init=" << std::setprecision(5) << l0
              << "  final=" << lf
              << "  x" << std::setprecision(1) << (l0 / (lf + 1e-12)) << " better\n";
    return lf;
}

int main(int argc, char **argv)
{
    Eigen::setNbThreads(4);
    std::string inPath = (argc > 1) ? argv[1] : "Tests/data/photo.jpg";
    const int S = 32, epochs = 120;

    Matrixd full = ImageIO::loadGray(inPath);
    if (full.size() == 0) { std::cout << "load fail\n"; return 1; }
    Matrixd img(S, S);
    for (int y = 0; y < S; ++y)
        for (int x = 0; x < S; ++x)
            img(y, x) = full(static_cast<int>((double)y / S * full.rows()),
                             static_cast<int>((double)x / S * full.cols()));

    std::cout << "Optimizer comparison (coordinate-MLP on photo, " << S << "x" << S
              << ", " << epochs << " epochs, identical init)\n";
    trainWith(0, "GD",       0.20, img, S, epochs);
    trainWith(1, "ADAM",     0.02, img, S, epochs);
    trainWith(2, "RMSProp",  0.005, img, S, epochs);
    trainWith(3, "Adagrad",  0.10, img, S, epochs);
    trainWith(4, "Adadelta", 1.00, img, S, epochs);
    return 0;
}
