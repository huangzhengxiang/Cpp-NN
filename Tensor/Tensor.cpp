/*
 * Tensor.cpp
 *
 *  Created on: 2023/7/12
 *      Author: hzx
 */
#include <cmath>
#include <vector>
#include "Schema.hpp"
#include "Tensor.hpp"

Tensor* newTensor(Schema& schema){
    Tensor* result = new Tensor(schema);
    result->content = new float[result->getSize()];
    return result;
}
bool isCompatible(Tensor& t1, Tensor& t2){
    return isCompatible(t1.schema,t2.schema);
}

Tensor::Tensor(std::vector<int> s, bool grad):schema(s, grad){}
Tensor::Tensor(int d, int* s, bool grad):schema(d, s, grad){}
Tensor::Tensor(std::vector<int> s, std::vector<int> perm, bool grad):schema(s, perm, grad){}
Tensor::Tensor(Schema other):schema(other){}
Tensor::Tensor(Schema other, float* pointer):schema(other),content(pointer){}
void Tensor::init(float* pointer){
    this->content = pointer;
}

// reshaping
// void Tensor::reshape(int d, std::vector<int> s){
//     this->schema = Schema(d,s,this->require_grad());
// }
// void Tensor::reshape(int d, int* s){
//     this->schema = Schema(d,s,this->require_grad());
// }

float& Tensor::get(std::vector<int> index){
    int real_idx = 0;
    int offset = 1;
    for(int j=this->getDim()-1;j>=0;j--){
        real_idx += index[this->schema.getMap(j)]*offset;
        offset *= this->getKdim(j);
    }
    return this->content[real_idx];
}

// Exp and Log
Tensor* log(Tensor& tensor){
    Tensor* result = newTensor(tensor.schema);
    for (int j=0; j<tensor.getSize(); ++j){
        result->content[j] = std::log(tensor.content[j]);
    }
    return result;
}
Tensor* log(Tensor& tensor, float a){
    float k = log(a);
    Tensor* result = newTensor(tensor.schema);
    for (int j=0; j<tensor.getSize(); ++j){
        result->content[j] = std::log(tensor.content[j])/k;
    }
    return result;
}
Tensor* exp(Tensor& tensor){
    Tensor* result = newTensor(tensor.schema);
    for (int j=0; j<tensor.getSize(); ++j){
        result->content[j] = std::exp(tensor.content[j]);
    }
    return result;
}
// a is exponent.
Tensor* pow(Tensor& tensor, float a){
    Tensor* result = newTensor(tensor.schema);
    for (int j=0; j<tensor.getSize(); ++j){
        result->content[j] = std::pow(tensor.content[j],a);
    }
    return result;
}
// a is base.
Tensor* pow(float a, Tensor& tensor){
    Tensor* result = newTensor(tensor.schema);
    for (int j=0; j<tensor.getSize(); ++j){
        result->content[j] = std::pow(a,tensor.content[j]);
    }
    return result;
}
void Tensor::log(){
    for (int j=0; j<this->getSize(); ++j){
        this->content[j] = std::log(this->content[j]);
    }
}
void Tensor::log(float a){
    float k = std::log(a);
    for (int j=0; j<this->getSize(); ++j){
        this->content[j] = std::log(this->content[j])/k;
    }
}
void Tensor::exp(){
    for (int j=0; j<this->getSize(); ++j){
        this->content[j] = std::exp(this->content[j]);
    }
}

// sqrt
Tensor* sqrt(Tensor& tensor){
    Tensor* result = newTensor(tensor.schema);
    for (int j=0; j<tensor.getSize(); ++j){
        result->content[j] = std::sqrt(tensor.content[j]);
    }
    return result;
}
void Tensor::sqrt(){
    for (int j=0; j<this->getSize(); ++j){
        this->content[j] = std::sqrt(this->content[j]);
    }
}

// test print! Not for real purpose!
void Tensor::print(){
    printf("Tensor: [");
    for (int j=0; j<this->getSize(); ++j){
        printf("%f, ", this->content[j]);
    }
    printf("]\n");
}
void Tensor::printShape(){
    this->schema.print();
}