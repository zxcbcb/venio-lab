#include "TransformerBlock.hxx"
#include "RandomGenerator.hxx"
#include <cmath>

TransformerBlock::TransformerBlock(int d_model, int hidden)
    : _d(d_model), _hidden(hidden), _attn(d_model)
{
    _W1 = RandomGenerator::generateRandomMatrix(-0.1, 0.1, _d, _hidden);
    _W2 = RandomGenerator::generateRandomMatrix(-0.1, 0.1, _hidden, _d);
    _b1 = Eigen::RowVectorXd::Zero(_hidden);
    _b2 = Eigen::RowVectorXd::Zero(_d);
}

Matrixd TransformerBlock::layerNorm(const Matrixd &X, double eps)
{
    Matrixd out(X.rows(), X.cols());
    for (int i = 0; i < X.rows(); ++i)
    {
        double m = X.row(i).mean();
        Eigen::RowVectorXd c = X.row(i).array() - m;
        double var = c.array().square().sum() / static_cast<double>(X.cols());
        out.row(i) = c / std::sqrt(var + eps);
    }
    return out;
}

Matrixd TransformerBlock::relu(const Matrixd &X)
{
    return X.cwiseMax(0.0);
}

Matrixd TransformerBlock::forward(const Matrixd &X)
{
    Matrixd a = _attn.forward(X);           // seq x d
    Matrixd r1 = layerNorm(X + a);          // residual + norm
    Matrixd h = relu((r1 * _W1).rowwise() + _b1); // seq x hidden
    Matrixd f = (h * _W2).rowwise() + _b2;  // seq x d
    Matrixd out = layerNorm(r1 + f);        // residual + norm
    return out;
}

void TransformerBlock::setW1(const Matrixd &m) { _W1 = m; }
void TransformerBlock::setW2(const Matrixd &m) { _W2 = m; }
void TransformerBlock::setB1(const Eigen::RowVectorXd &v) { _b1 = v; }
void TransformerBlock::setB2(const Eigen::RowVectorXd &v) { _b2 = v; }
