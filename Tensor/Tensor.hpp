/*
 * Tensor.hpp
 *
 *  Created on: 2023/7/12
 *      Author: hzx
 */

#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include "Schema.hpp"


class Tensor;

Tensor* newTensor(Schema& schema);

// Inplace Operators are allowed!
// Exp and Log
Tensor* log(Tensor& tensor);
Tensor* log(Tensor& tensor, float a);
Tensor* exp(Tensor& tensor);
Tensor* pow(Tensor& tensor, float a);
Tensor* pow(float a, Tensor& tensor);
// sqrt
Tensor* sqrt(Tensor& tensor);

// isCompatible
bool isCompatible(Tensor& t1, Tensor& t2);

class Tensor{
private:
    Schema schema;
    float* content;
public:
    // initialization
    Tensor(std::vector<int> s, bool grad);
    Tensor(int d, int* s, bool grad);
    Tensor(std::vector<int> s, std::vector<int> perm, bool grad);
    Tensor(Schema other);
    Tensor(Schema other, float* pointer);
    // set the content
    void init(float* pointer);
    // reshaping
    // void reshape(int d, std::vector<int> s);
    // void reshape(int d, int* s);
    // getDim, getSize, getKdim
    int getDim(){return this->schema.getDim();}
    int getSize(){return this->schema.getSize();}
    int getKdim(int k){return this->schema.getKdim(k);}
    void setKdim(int k, int d){this->schema.setKdim(k,d);}
    std::vector<int> getShape(){return this->schema.getShape();}
    Schema getSchema(){return this->schema;}
    bool& require_grad(){return this->schema.require_grad();}
    float& get(std::vector<int> index);
    // friends
    friend Tensor* newTensor(Schema& schema);
    friend bool isCompatible(Tensor& t1, Tensor& t2);
    // Only inplace operator are allowed!
    // exp and log
    friend Tensor* log(Tensor& tensor);
    friend Tensor* log(Tensor& tensor, float a);
    friend Tensor* exp(Tensor& tensor);
    friend Tensor* pow(Tensor& tensor, float a);
    friend Tensor* pow(float a, Tensor& tensor);
    void log();
    void log(float a);
    void exp();
    // sqrt
    friend Tensor* sqrt(Tensor& tensor);
    void sqrt();

    // test print
    void print();
    void printShape();
    // destructor
    void destruct(){delete content;}
    ~Tensor(){} 
};



#endif /* TENSOR_HPP_ */
