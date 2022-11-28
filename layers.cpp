//
// Created by Hardeep on 11/28/2022.
//
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include "layers.h"

layers::layers() {

}

std::vector<double> layers::linspace(double a, double b, int N) {
    std::vector<double> x;
    for (int i = 0; i < N; i++) {
        x.push_back(i);
        x[i] = a + (b - a) * (i / (N - 1.0));
    }
    return x;
}

double layers::input_size3D(double batch_size, std::tuple<double,double,double> shape) {
    return batch_size * std::get<0>(shape) * std::get<1>(shape) * std::get<2>(shape) ;
}
double layers::input_size2D(double batch_size, std::tuple<double,double> shape) {
    return batch_size * std::get<0>(shape) * std::get<1>(shape);
}

