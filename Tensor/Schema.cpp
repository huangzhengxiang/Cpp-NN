/*
 * Schema.cpp
 *
 *  Created on: 2023/7/12
 *      Author: hzx
 */
#include "Schema.hpp"

Schema::Schema(int d, std::vector<int> s, bool grad){
    this->dim=d;
    this->Shape=s;
    this->isGrad=grad;
}
Schema::Schema(int d, int* s, bool grad){
    this->dim=d;
    for(int j=0;j<d;++j){
        this->Shape.push_back(s[j]);
    }
    this->isGrad=grad;
}
Schema::Schema(Schema& other){
    this->dim=other.dim;
    this->Shape=other.Shape;
    this->isGrad=other.isGrad;
}

int Schema::getDim(){
    return this->dim;
}
int Schema::getSize(){
    int result = 1;
    for (int j=0;j<this->dim;++j){
        result *= this->Shape[j];
    }
    return result;
}
int Schema::getKdim(int k){
    k = (k>=0) ? (k) : (this->dim + k);
    return this->Shape[k];
}
void Schema::setKdim(int k, int d){
    k = (k>=0) ? (k) : (this->dim + k);
    this->Shape[k] = d;
    return;
}