#include <iostream>
#include "Tensor.hpp"
#include "Network.hpp"

int main(){
    // x
    Tensor input(2,{2,2},false);
    float data[4]={3.,7.,2.,8.};
    input.init(data);
    input.print();
    // fc1
    Tensor fc1_weight(2,fc1_w_schema,false);
    fc1_weight.init(fc1_w);
    Tensor fc1_bias(1,fc1_b_schema,false);
    fc1_bias.init(fc1_b);
    // fc2
    Tensor fc2_weight(2,fc2_w_schema,false);
    fc2_weight.init(fc2_w);
    Tensor fc2_bias(1,fc2_b_schema,false);
    fc2_bias.init(fc2_b);
    // fc2(fc1(x))
    Tensor* mid = matmul(&fc1_weight, &input);
    Tensor* result = matmul(&fc2_weight, mid);
    result->print();
}