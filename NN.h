//
// Created by hbass on 11/29/2022.
//

#ifndef MATH233_FP_NN_H
#define MATH233_FP_NN_H

#include "eigen3/Eigen/Eigen"
#include <iostream>
#include <vector>

typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;

class NN {
public:
    NN(std::vector<int> neurons, Scalar learningRate = Scalar(0.01));

    void propagateForward(RowVector& input);
    void propagateBackward(RowVector& output);
    void calcErrors(RowVector& output);
    void updateWeights();
    void ReadCSV(std::string filename, std::vector<RowVector*>& data);
    void train(std::vector<RowVector *> input_data, std::vector<RowVector *> output_data);

    std::vector<RowVector*> neuronLayers;
    std::vector<RowVector*> cacheLayers;
    std::vector<RowVector*> deltas;
    std::vector<Matrix*> weights;
    Scalar learningRate;
    std::vector<int> neurons;
};

#endif //MATH233_FP_NN_H
