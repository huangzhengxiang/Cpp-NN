/*
 * Schema.cpp
 *
 *  Created on: 2023/7/12
 *      Author: hzx
 */
#include <stdio.h>
#include "Schema.hpp"

bool isCompatible(Schema& t1, Schema& t2){
    if (t1.getDim()!=t2.getDim()) return false;
    for (int j=0;j<t1.getDim();++j){
        if (t1.getKdim(j)!=t2.getKdim(j)) return false;
    }
    return true;
}

std::vector<int> Schema::axis_map_inv(){
    std::vector<int> result(this->getDim(),0);
    for (int j=0;j<this->getDim();++j){
        result[this->axis_map[j]]=j;
    }
    return result;
}

// initialization
Schema::Schema(std::vector<int> shape, bool grad){
    this->real_shape=shape;
    for (int j=0; j<shape.size();++j){
        this->axis_map.push_back(j);
    }
    this->a_inv = this->axis_map_inv();
    this->isGrad=grad;
}
Schema::Schema(int d, int* shape, bool grad){
    for (int j=0; j<d;++j){
        this->real_shape.push_back(shape[j]);
        this->axis_map.push_back(j);
    }
    this->a_inv = this->axis_map_inv();
    this->isGrad=grad;
}
Schema::Schema(std::vector<int> shape, std::vector<int> axis_perm, bool grad){
    this->real_shape=shape;
    this->axis_map=axis_perm;
    this->a_inv = this->axis_map_inv();
    this->isGrad=grad;
}
Schema::Schema(Schema& other){
    this->real_shape=other.real_shape;
    this->axis_map=other.axis_map;
    this->a_inv = this->axis_map_inv();
    this->isGrad=other.isGrad;
}

int Schema::getDim(){
    return this->real_shape.size();
}
int Schema::getSize(){
    int result = 1;
    for (int j=0;j<this->getDim();++j){
        result *= this->real_shape[j];
    }
    return result;
}
int Schema::getKdim(int k){
    k = (k>=0) ? (k) : (this->getDim() + k);
    return this->real_shape[this->a_inv[k]];
}
void Schema::setKdim(int k, int d){
    k = (k>=0) ? (k) : (this->getDim() + k);
    this->real_shape[this->a_inv[k]] = d;
    return;
}

void Schema::print(){
    printf("Real Shape: [");
    for(int j=0;j<this->getDim();++j){
        printf("%d, ", this->real_shape[j]);
    }
    printf("]\n");
    printf("External Shape: [");
    std::vector<int> external_shape = this->getShape();
    for(int j=0;j<this->getDim();++j){
        printf("%d, ", external_shape[j]);
    }
    printf("]\n\n");
}

std::vector<int> Schema::getShape(){
    std::vector<int> external_shape(this->getDim(),0);
    for(int j=0;j<this->getDim();++j){
        external_shape[j]=this->real_shape[this->a_inv[j]];
    }
    return external_shape;
}
std::vector<int> Schema::realShape(){
    return this->real_shape;
}

// permute upon the external shape.
void Schema::permute(std::vector<int> axis_perm){
    // permute the inverse.
    std::vector<int> new_inv(this->getDim(),0);
    for(int j=0;j<this->getDim();++j){
        new_inv[j]=this->a_inv[axis_perm[j]];
    }
    this->a_inv=new_inv;
    // resolve the origin.
    for(int j=0;j<this->getDim();++j){
        this->axis_map[new_inv[j]]=j;
    }
    return;
}