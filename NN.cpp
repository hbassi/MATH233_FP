//
// Created by hbass on 11/29/2022.
//
#include "NN.h"
// constructor of neural network class
NN::NN(std::vector<int> neurons, Scalar learningRate){
    this->neurons = neurons;
    this->learningRate = learningRate;
    for (int i = 0; i < neurons.size(); i++) {
        // initialize neuron layers
        if (i == neurons.size() - 1)
            neuronLayers.push_back(new RowVector(neurons[i]));
        else
            neuronLayers.push_back(new RowVector(neurons[i] + 1));
        cacheLayers.push_back(new RowVector(neuronLayers.size()));
        deltas.push_back(new RowVector(neuronLayers.size()));

        if (i != neurons.size() - 1) {
            neuronLayers.back()->coeffRef(neurons[i]) = 1.0;
            cacheLayers.back()->coeffRef(neurons[i]) = 1.0;
        }

        // initialize weights matrix
        if (i > 0) {
            if (i != neurons.size() - 1) {
                weights.push_back(new Matrix(neurons[i - 1] + 1, neurons[i] + 1));
                weights.back()->setRandom();
                weights.back()->col(neurons[i]).setZero();
                weights.back()->coeffRef(neurons[i - 1], neurons[i]) = 1.0;
            } else {
                weights.push_back(new Matrix(neurons[i - 1] + 1, neurons[i]));
                weights.back()->setRandom();
            }
        }
    }
};
