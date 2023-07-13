/*
 * Tensor.hpp
 *
 *  Created on: 2023/7/12
 *      Author: hzx
 */

#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include "Schema.hpp"

// Tensor* operator+(Tensor* t1, Tensor* t2);
// Tensor* operator+(int a, Tensor* tensor);
// Tensor* operator+(float a, Tensor* tensor);
// Tensor* operator+(double a, Tensor* tensor);
// Tensor* operator+(Tensor* tensor, int a);
// Tensor* operator+(Tensor* tensor, float a);
// Tensor* operator+(Tensor* tensor, double a);
// Tensor* operator-(Tensor* t1, Tensor* t2);
// Tensor* operator-(int a, Tensor* tensor);
// Tensor* operator-(float a, Tensor* tensor);
// Tensor* operator-(double a, Tensor* tensor);
// Tensor* operator-(Tensor* tensor, int a);
// Tensor* operator-(Tensor* tensor, float a);
// Tensor* operator-(Tensor* tensor, double a);
// Tensor* operator*(Tensor* t1, Tensor* t2);
// Tensor* operator*(int a, Tensor* tensor);
// Tensor* operator*(float a, Tensor* tensor);
// Tensor* operator*(double a, Tensor* tensor);
// Tensor* operator*(Tensor* tensor, int a);
// Tensor* operator*(Tensor* tensor, float a);
// Tensor* operator*(Tensor* tensor, double a);
// Tensor* operator/(Tensor* t1, Tensor* t2);
// Tensor* operator/(Tensor* tensor, int a);
// Tensor* operator/(Tensor* tensor, float a);
// Tensor* operator/(Tensor* tensor,double a);

class Tensor;

// Matrix Multiplication
Tensor* matmul(Tensor* t1, Tensor* t2);

class Tensor{
private:
    Schema schema;
    float* content;
public:
    // initialization
    Tensor(int d, std::vector<int> s, bool grad);
    Tensor(int d, int* s, bool grad);
    Tensor(Schema other);
    // set the content
    void init(float* pointer);
    // reshaping
    // void reshape(int d, std::vector<int> s);
    // void reshape(int d, int* s);
    // getDim, getSize, getKdim
    int getDim();
    int getSize();
    int getKdim(int k);
    // friends
    // Matrix Multiplication
    friend Tensor* matmul(Tensor* t1, Tensor* t2);

    // test print
    void print();
    
    // destructor
    void destructor(){delete content;}
    ~Tensor(){} 
};



#endif /* TENSOR_HPP_ */
