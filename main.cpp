#include <iostream>
#include <tuple>
#include "layers.h"
int main() {

    layers L = layers();
    double num_inputs = 2;
    std::tuple<double,double,double> input_shape = std::make_tuple(4, 5, 6);
    double output_dim = 3;
    double input_size = L.input_size3D(num_inputs, input_shape);
    double weight_size =  L.input_size3D(output_dim,input_shape);
    //std::cout << input_size;

    std::vector<double> x = L.linspace(-0.1,0.5, input_size);
    std::vector<double> w = L.linspace(-0.2,0.3, weight_size);
    std::vector<double> b = L.linspace(-0.3,0.1, output_dim);
    return 0;
}
