/*
 * Tensor.cpp
 *
 *  Created on: 2023/7/12
 *      Author: hzx
 */
#include "Schema.hpp"
#include "Tensor.hpp"

Tensor::Tensor(int d, std::vector<int> s, bool grad):schema(d, s, grad){}
Tensor::Tensor(int d, int* s, bool grad):schema(d, s, grad){}
Tensor::Tensor(Schema other):schema(other){}
void Tensor::init(float* pointer){
    this->content = pointer;
}

// size
int Tensor::getDim(){return this->schema.getDim();}
int Tensor::getSize(){return this->schema.getSize();}
int Tensor::getKdim(int k){return this->schema.getKdim(k);}

// void reshape(int d, std::vector<int> s){
    
// }
// void reshape(int d, int* s){

// }

Tensor* matmul(Tensor* t1, Tensor* t2){
    // (...,m,n) @ (...,n,k)
    int m = t1->getKdim(-2);
    int n = t1->getKdim(-1);
    if (n!=t2->getKdim(-2)) return NULL;
    int k = t2->getKdim(-1);

    int N1 = t1->getSize()/(m*n);
    int N2 = t2->getSize()/(n*k);
    if ((N1!=N2) || (t1->getDim()!=t2->getDim())) return NULL;// No broadcasting for now!
    
    // initialize the result matrix 
    Schema newSchema(t1->schema);
    newSchema.setKdim(-2, m);
    newSchema.setKdim(-1, k);
    Tensor* result = new Tensor(newSchema);
    result->content = new float[result->getSize()];

    for(int i=0;i<N1;++i){
        // conduct N1 times matmul
        for (int jl=0;jl<m;++jl){
            for (int jr=0;jr<k;++jr){
                float temp = 0.0;
                for (int b=0;b<n;++b){
                    temp += t1->content[i*m*n+jl*n+b] * t2->content[i*n*k+b*k+jr];
                }
                result->content[i*m*k+jl*k+jr] = temp;
            }
        }
    }

    // (...,m,k)
    return result;
}

void Tensor::print(){
    printf("[");
    for (int j=0; j<this->getSize(); ++j){
        printf("%f, ", this->content[j]);
    }
    printf("]\n");
}