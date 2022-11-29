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

class NeuralNetwork {
public:
    NeuralNetwork(std::vector<int> neurons, Scalar learningRate = Scalar(0.005));

    void propagateForward(RowVector& input);
    void propagateBackward(RowVector& output);
    
    void calcErrors(RowVector& output);
    void updateWeights();
    void train(std::vector<RowVector*> data);

    std::vector<RowVector*> neuronLayers;
    std::vector<RowVector*> cacheLayers;
    std::vector<RowVector*> deltas;
    std::vector<Matrix*> weights;
    Scalar learningRate;
};

#endif //MATH233_FP_NN_H
