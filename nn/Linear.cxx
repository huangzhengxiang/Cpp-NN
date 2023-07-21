#include <stdio.h>
#include "Linear.hpp"
#include "CppNN/Tensor.hpp"

Tensor* Linear::forward(Tensor& input){
    /*  input (Not a view): (N,in_features,...), 
        weight (Not a view): (out_features, in_features),
        bias (Not a view): (out_features),
        results (Not a view): (N,out_features,...)
    */
    // mid_result: (N,out_features,...)
    Schema resSchema(input.getShape(),this->weight->require_grad());
    resSchema.setKdim(1,this->out_features);
    Tensor* mid_result = newTensor(resSchema);
    // input: (N,in_features,...) -> (N,...,in_features,1) (view)
    // mid_result: (N,out_features,...) -> (N,...,out_features,1) (view)
    std::vector<int> permute_shape(input.getDim(),0);
    for (int j=1;j<permute_shape.size()-1;++j) permute_shape[j] = j+1;
    permute_shape[permute_shape.size()-1] = 1;
    input.permute(permute_shape);
    mid_result->permute(permute_shape);
    permute_shape = std::vector<int>(input.getShape());
    permute_shape.push_back(1);
    input.view(permute_shape);
    permute_shape = std::vector<int>(mid_result->getShape());
    permute_shape.push_back(1);
    mid_result->view(permute_shape);
    // mid_result = weight @ input (N,...,out_features)
    permute_shape = std::vector<int>(permute_shape.size(),1);
    permute_shape[permute_shape.size()-2] = this->out_features;
    permute_shape[permute_shape.size()-1] = this->in_features;
    this->weight->view(permute_shape);
    mid_result = matmul(*(this->weight), input, mid_result);
    mid_result->deview();
    this->weight->deview();
    // result = mid_result + bias (N,...,out_features)
    Tensor* result = NULL;
    if (this->require_bias){
        permute_shape = std::vector<int>(mid_result->getDim(),1);
        permute_shape[permute_shape.size()-1] = this->out_features;
        this->bias->view(permute_shape);
        result = (*mid_result) + *(this->bias);
        this->bias->deview();
        mid_result->destruct();
    } else{
        result = mid_result;
    }
    // result: (N,out_features,...), input: (N,in_features,...)
    input.deview(); // (N,...,in_features,1) -> (N,...,in_features)
    permute_shape = std::vector<int>(input.getDim(),0);
    for (int j=1;j<permute_shape.size()-1;++j) permute_shape[j+1] = j;
    permute_shape[1] = permute_shape.size()-1; 
    input.permute(permute_shape);
    result->permute(permute_shape);
    return result;
}