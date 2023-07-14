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

Tensor::Tensor(int d, std::vector<int> s, bool grad):schema(d, s, grad){}
Tensor::Tensor(int d, int* s, bool grad):schema(d, s, grad){}
Tensor::Tensor(Schema other):schema(other){}
Tensor::Tensor(Schema other, float* pointer):schema(other),content(pointer){}
void Tensor::init(float* pointer){
    this->content = pointer;
}

// size
int Tensor::getDim(){return this->schema.getDim();}
int Tensor::getSize(){return this->schema.getSize();}
int Tensor::getKdim(int k){return this->schema.getKdim(k);}
bool& Tensor::require_grad(){
    return this->schema.require_grad();
}
void Tensor::reshape(int d, std::vector<int> s){
    this->schema = Schema(d,s,this->require_grad());
}
void Tensor::reshape(int d, int* s){
    this->schema = Schema(d,s,this->require_grad());
}

float& Tensor::get(int* index){
    int real_idx = 0;
    int offset = 1;
    for(int j=this->getDim()-1;j>=0;j--){
        real_idx += index[j]*offset;
        offset *= this->getKdim(j);
    }
    return this->content[real_idx];
}

float& Tensor::get(std::vector<int> index){
    int real_idx = 0;
    int offset = 1;
    for(int j=this->getDim()-1;j>=0;j--){
        real_idx += index[j]*offset;
        offset *= this->getKdim(j);
    }
    return this->content[real_idx];
}

Tensor* matmul(Tensor& t1, Tensor& t2){
    // (...,m,n) @ (...,n,k)
    int m = t1.getKdim(-2);
    int n = t1.getKdim(-1);
    if (n!=t2.getKdim(-2)) return NULL;
    int k = t2.getKdim(-1);
    int N1 = t1.getSize()/(m*n);
    int N2 = t2.getSize()/(n*k);
    if ((N1!=N2) || (t1.getDim()!=t2.getDim())) return NULL;// No broadcasting for now!

    // reshape to (N1,m,n) @ (N2,n,k)
    std::vector<int> resShape = t1.schema.getShape();
    Tensor* tensor1 = new Tensor(t1.schema,t1.content);
    Tensor* tensor2 = new Tensor(t2.schema,t2.content);
    tensor1->reshape(3,{N1,m,n});
    tensor2->reshape(3,{N2,n,k});
    
    // initialize the result matrix (N1,m,k)
    Tensor* result = new Tensor(tensor1->schema);
    result->schema.setKdim(-2, m);
    result->schema.setKdim(-1, k);
    result->content = new float[result->getSize()];

    for(int i=0;i<N1;++i){
        // conduct N1 times matmul
        for (int jl=0;jl<m;++jl){
            for (int jr=0;jr<k;++jr){
                float temp = 0.0;
                for (int b=0;b<n;++b){
                    temp += tensor1->get({i,jl,b}) * tensor2->get({i,b,jr});
                    // temp += tensor1->content[i*m*n+jl*n+b] * tensor2->content[i*n*k+b*k+jr];
                }
                result->get({i,jl,jr}) = temp;
            }
        }
    }

    // (N1,m,k) -> (...,m,k)
    result->schema.setShape(resShape);
    result->schema.setKdim(-2, m);
    result->schema.setKdim(-1, k);
    // (...,m,k)
    return result;
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

// Basic Op
Tensor* operator+(Tensor& t1, Tensor& t2){
    if (!isCompatible(t1,t2)) return NULL;
    Tensor* result = newTensor(t1.schema);
    for (int j=0;j<t1.getSize();++j){
        result->content[j] = t1.content[j] + t2.content[j];
    }
    return result;
}
Tensor* operator-(Tensor& t1, Tensor& t2){
    if (!isCompatible(t1,t2)) return NULL;
    Tensor* result = newTensor(t1.schema);
    for (int j=0;j<t1.getSize();++j){
        result->content[j] = t1.content[j] - t2.content[j];
    }
    return result;
}
Tensor* operator*(Tensor& t1, Tensor& t2){
    if (!isCompatible(t1,t2)) return NULL;
    Tensor* result = newTensor(t1.schema);
    for (int j=0;j<t1.getSize();++j){
        result->content[j] = t1.content[j] * t2.content[j];
    }
    return result;
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