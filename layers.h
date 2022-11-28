//
// Created by Hardeep on 11/28/2022.
//

#ifndef MATH233_FP_LAYERS_H
#define MATH233_FP_LAYERS_H
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>

class layers {
private:
    double num_inputs;
    std::tuple<double> input_shape;
    double output_dim;
public:
    // Constructor to create object
    layers();

   // layers(double inputs_dim, double outputs_dim);

    std::vector<double> affine_forward(std::vector<double> x, std::vector<double> w, std::vector<double> b);
    std::vector<double> linspace(double a, double b, int N);
    double input_size3D(double batch_size, std::tuple<double,double,double> shape);
    double input_size2D(double batch_size, std::tuple<double,double> shape);
};
#endif //MATH233_FP_LAYERS_H
