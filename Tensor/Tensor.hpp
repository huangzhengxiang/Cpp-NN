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

// // Basic Op
Tensor* operator+(Tensor& t1, Tensor& t2);
// Tensor* operator+(int a, Tensor& tensor);
// Tensor* operator+(float a, Tensor& tensor);
// Tensor* operator+(double a, Tensor& tensor);
// Tensor* operator+(Tensor& tensor, int a);
// Tensor* operator+(Tensor& tensor, float a);
// Tensor* operator+(Tensor& tensor, double a);
Tensor* operator-(Tensor& t1, Tensor& t2);
// Tensor* operator-(int a, Tensor& tensor);
// Tensor* operator-(float a, Tensor& tensor);
// Tensor* operator-(double a, Tensor& tensor);
// Tensor* operator-(Tensor& tensor, int a);
// Tensor* operator-(Tensor& tensor, float a);
// Tensor* operator-(Tensor& tensor, double a);
Tensor* operator*(Tensor& t1, Tensor& t2);
// Tensor* operator*(int a, Tensor& tensor);
// Tensor* operator*(float a, Tensor& tensor);
// Tensor* operator*(double a, Tensor& tensor);
// Tensor* operator*(Tensor& tensor, int a);
// Tensor* operator*(Tensor& tensor, float a);
// Tensor* operator*(Tensor& tensor, double a);
// Tensor* operator/(Tensor& t1, Tensor& t2);
// Tensor* operator/(Tensor& tensor, int a);
// Tensor* operator/(Tensor& tensor, float a);
// Tensor* operator/(Tensor& tensor, double a);

// Matrix Multiplication
Tensor* matmul(Tensor& t1, Tensor& t2);

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
    Tensor(int d, std::vector<int> s, bool grad);
    Tensor(int d, int* s, bool grad);
    Tensor(Schema other);
    Tensor(Schema other, float* pointer);
    // set the content
    void init(float* pointer);
    // reshaping
    void reshape(int d, std::vector<int> s);
    void reshape(int d, int* s);
    // getDim, getSize, getKdim
    int getDim();
    int getSize();
    int getKdim(int k);
    bool& require_grad();
    float& get(int* index);
    float& get(std::vector<int> index);
    // friends
    friend Tensor* newTensor(Schema& schema);
    friend bool isCompatible(Tensor& t1, Tensor& t2);
    // Matrix Multiplication
    friend Tensor* matmul(Tensor& t1, Tensor& t2);
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
    // Basic Op
    friend Tensor* operator+(Tensor& t1, Tensor& t2);
    // friend Tensor* operator+(int a, Tensor& tensor);
    // friend Tensor* operator+(float a, Tensor& tensor);
    // friend Tensor* operator+(double a, Tensor& tensor);
    // friend Tensor* operator+(Tensor& tensor, int a);
    // friend Tensor* operator+(Tensor& tensor, float a);
    // friend Tensor* operator+(Tensor& tensor, double a);
    friend Tensor* operator-(Tensor& t1, Tensor& t2);
    // friend Tensor* operator-(int a, Tensor& tensor);
    // friend Tensor* operator-(float a, Tensor& tensor);
    // friend Tensor* operator-(double a, Tensor& tensor);
    // friend Tensor* operator-(Tensor& tensor, int a);
    // friend Tensor* operator-(Tensor& tensor, float a);
    // friend Tensor* operator-(Tensor& tensor, double a);
    friend Tensor* operator*(Tensor& t1, Tensor& t2);
    // friend Tensor* operator*(int a, Tensor& tensor);
    // friend Tensor* operator*(float a, Tensor& tensor);
    // friend Tensor* operator*(double a, Tensor& tensor);
    // friend Tensor* operator*(Tensor& tensor, int a);
    // friend Tensor* operator*(Tensor& tensor, float a);
    // friend Tensor* operator*(Tensor& tensor, double a);
    // friend Tensor* operator/(Tensor& t1, Tensor& t2);
    // friend Tensor* operator/(Tensor& tensor, int a);
    // friend Tensor* operator/(Tensor& tensor, float a);
    // friend Tensor* operator/(Tensor& tensor, double a);

    // test print
    void print();
    void printShape();
    // destructor
    void destruct(){delete content;}
    ~Tensor(){} 
};



#endif /* TENSOR_HPP_ */
