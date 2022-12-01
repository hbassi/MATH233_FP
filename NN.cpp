//
// Created by hbass on 11/29/2022.
//
#include <fstream>
#include "NN.h"
#include "math.h"
#include <iostream>
NN::NN(std::vector<int> neurons, Scalar learningRate){
    this->neurons = neurons;
    this->learningRate = learningRate;
    for (int i = 0; i < neurons.size(); i++) {
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
}

Scalar relu(Scalar x) {
    if (x <= 0)
        return 0;
    else
        return x;
}

Scalar reluDer(Scalar x) {
    if (x <= 0)
        return 0;
    else
        return 1;
}

void NN::propagateForward(RowVector& input) {
    neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size() - 1) = input;
#pragma omp  parallel for
    for (int i = 1; i < neurons.size(); i++) {
        (*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
        neuronLayers[i]->block(0, 0, 1, neurons[i]).unaryExpr(std::ptr_fun(relu));
    }
}

void NN::calcErrors(RowVector& output) {
    (*deltas.back()) = output - (*neuronLayers.back());
#pragma omp  parallel for
    for (int i = neurons.size() - 2; i > 0; i--) {
        (*deltas[i]) = (*deltas[i + 1]) * (weights[i]->transpose());
    }
}

void NN::updateWeights() {
    for (int i = 0; i < neurons.size() - 1; i++) {
        if (i != neurons.size() - 2) {
            for (int c = 0; c < weights[i]->cols() - 1; c++) {
#pragma omp  parallel for
                for (int r = 0; r < weights[i]->rows(); r++) {
                    weights[i]->coeffRef(r, c) += learningRate * deltas[i + 1]->coeffRef(c) * reluDer(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
                }
            }
        }
        else {
            for (int c = 0; c < weights[i]->cols(); c++) {
#pragma omp  parallel for
                for (int r = 0; r < weights[i]->rows(); r++) {
                    weights[i]->coeffRef(r, c) += learningRate * deltas[i + 1]->coeffRef(c) * reluDer(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
                }
            }
        }
    }
}

void NN::propagateBackward(RowVector& output) {
    calcErrors(output);
    updateWeights();
}

void NN::train(std::vector<RowVector*> input_data, std::vector<RowVector*> output_data) {
    for (int i = 0; i < input_data.size(); i++) {
        std::cout << "Input to neural network is : " << *input_data[i] << std::endl;
        propagateForward(*input_data[i]);
        std::cout << "Expected output is : " << *output_data[i] << std::endl;
        std::cout << "Output produced is : " << *neuronLayers.back() << std::endl;
        propagateBackward(*output_data[i]);
        std::cout << "MSE : " << std::sqrt((*deltas.back()).dot((*deltas.back())) / deltas.back()->size()) << std::endl;
    }
}


void NN::ReadCSV(std::string filename, std::vector<RowVector*>& data) {
    data.clear();
    std::ifstream file(filename);
    std::string line, word;
    getline(file, line, '\n');
    std::stringstream ss(line);
    std::vector<Scalar> parsed_vec;
    while (getline(ss, word, ',')) {
        parsed_vec.push_back(Scalar(std::stof(&word[0])));
    }
    int cols = parsed_vec.size();
    data.push_back(new RowVector(cols));
    for (int i = 0; i < cols; i++) {
        data.back()->coeffRef(1, i) = parsed_vec[i];
    }
    if (file.is_open()) {
        while (getline(file, line, '\n')) {
            std::stringstream ss(line);
            data.push_back(new RowVector(1, cols));
            int i = 0;
            while (getline(ss, word, ',')) {
                data.back()->coeffRef(i) = Scalar(std::stof(&word[0]));
                i++;
            }
        }
    }
}
