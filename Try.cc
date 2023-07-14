#include <iostream>
#include "Tensor.hpp"
#include "Network.hpp"

int main(){
    // x
    Tensor input(2,{1,2},false);
    float data[2]={3.,7.};
    input.init(data);
    input.reshape(3,{1,2,1});
    input.print();
    input.printShape();
    // fc1
    Tensor fc1_weight(Schema(2,fc1_w_schema,false),fc1_w);
    Tensor fc1_bias(Schema(1,fc1_b_schema,false),fc1_b);
    // fc2
    Tensor fc2_weight(Schema(2,fc2_w_schema,false),fc2_w);
    Tensor fc2_bias(Schema(1,fc2_b_schema,false),fc2_b);
    // reshape
    fc1_weight.reshape(3,{1,4,2});
    fc2_weight.reshape(3,{1,1,4});
    fc1_bias.reshape(3,{1,4,1});
    fc2_bias.reshape(3,{1,1,1});
    // fc2(fc1(x))
    Tensor* mid = matmul(fc1_weight, input);
    mid = (*mid) + fc1_bias;
    Tensor* result = matmul(fc2_weight, *mid);
    result = (*result) + fc2_bias;
    result->printShape();
    result->print();
}