#include <iostream>
#include <tuple>
#include "layers.h"
#include "NN.h"

void genData(std::string filename) {
    std::ofstream file1(filename + "-in");
    std::ofstream file2(filename + "-out");
    for (int r = 0; r < 10000; r++) {
        Scalar x = rand() / Scalar(RAND_MAX);
        file1 << x  << std::endl;
        file2 << 2 * x + 10 << std::endl;
    }
    file1.close();
    file2.close();
}

typedef std::vector<RowVector*> data;
int main() {
    NN n({ 1, 16, 16, 1 });
    data in_dat, out_dat;
    genData("test");
    n.ReadCSV("test-in", in_dat);
    n.ReadCSV("test-out", out_dat);
    n.train(in_dat, out_dat);
    return 0;
}


