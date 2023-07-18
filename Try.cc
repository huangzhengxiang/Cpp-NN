#include <iostream>
#include "CppNN/Tensor.hpp"
#include "Network.hpp"

#ifdef COMPLETE_TEST

#include "include/nn.hpp"

int main(){
    // x
    Tensor input({2,2,1},false);
    float data[4]={3.,7.,4.,8.}; // 2 inference data points.
    input.init(data);
    input.printShape();
    input.print();
    // fc1
    Schema fc1_weight_schema(2,fc1_w_schema,false);
    Schema fc1_bias_schema(1,fc1_b_schema,false);
    Tensor fc1_weight(fc1_weight_schema,fc1_w);
    Tensor fc1_bias(fc1_bias_schema,fc1_b);
    Linear fc1(2,4,true,&fc1_weight,&fc1_bias);
    // fc2
    Schema fc2_weight_schema(2,fc2_w_schema,false);
    Schema fc2_bias_schema(1,fc2_b_schema,false)
    Tensor fc2_weight(fc2_weight_schema,fc2_w);
    Tensor fc2_bias(fc2_bias_schema,fc2_b);
    Linear fc2(4,1,true,&fc2_weight,&fc2_bias);
    // fc2(fc1(x))
    Tensor* mid = fc1.forward(input);
    Tensor* result = fc2.forward(*mid);
    result->printShape();
    result->print();
    return 0;
}

#else

#ifdef UNIT_TEST
// Unit Test

int main(){
    // x
    Tensor input({3,2,1},false);
    float data[6]={3.,7.,4.,8.,5.,5.}; // 2 inference data points.
    input.init(data);
    input.printShape();
    input.print();
    // test
    printf("item: %.4f\n",input.get({1,0,0}));
    input.view({1,1,3,2,1});
    printf("item: %.4f\n",input.get({0,0,1,0,0}));
    input.deview();
    input.view({3,2,1,1});
    printf("item: %.4f\n",input.get({1,0,0,0}));
    input.deview();
    input.view({3,1,2,1});
    printf("item: %.4f\n",input.get({1,0,0,0}));
    input.deview();
    input.view({6,1,1});
    printf("item: %.4f\n",input.get({2,0,0}));
    input.deview();
    input.view({6});
    printf("item: %.4f\n",input.get({2}));
    input.deview();
    input.view({1,6});
    printf("item: %.4f\n",input.get({0,2}));
    input.deview();
    input.permute({1,0,2});
    input.printShape();
    printf("item: %.4f\n",input.get({0,1,0}));
    input.view({1,6});
    printf("item: %.4f\n",input.get({0,1}));
    input.deview();
    return 0;
}

#else

int main(){
    printf("Hello World!\n");
    return 0;
}

#endif

#endif