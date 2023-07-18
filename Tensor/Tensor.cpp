/*
 * Tensor.cpp
 *
 *  Created on: 2023/7/12
 *      Author: hzx
 */
#include <cmath>
#include <vector>
#include "Schema.hpp"
#include "CppNN/Tensor.hpp"

Tensor* newTensor(Schema& schema){
    Tensor* result = new Tensor(schema);
    result->content = new float[result->getSize()];
    return result;
}
bool isCompatible(Tensor& t1, Tensor& t2){
    return isCompatible(t1.schema,t2.schema);
}

// Constructors
Tensor::Tensor(std::vector<int> s, bool grad):schema(s, grad),isViewed(false){}
Tensor::Tensor(int d, int* s, bool grad):schema(d, s, grad),isViewed(false){}
Tensor::Tensor(std::vector<int> s, std::vector<int> perm, bool grad):schema(s, perm, grad),isViewed(false){}
Tensor::Tensor(Schema other):schema(other),isViewed(false){}
Tensor::Tensor(Schema other, float* pointer):schema(other),content(pointer),isViewed(false){}
void Tensor::init(float* pointer){
    this->content = pointer;
}

int Tensor::getKdim(int k){
    return this->schema.getKdim(k);
}
void Tensor::setKdim(int k, int d){
    if (!this->isViewed) this->schema.setKdim(k,d);
}
std::vector<int> Tensor::getShape(){
    if (!this->isViewed)
        return this->schema.getShape();
    else{
        return this->user_view;
    }
}
int Tensor::getKthdim(int k){
    if (!this->isViewed){
        return this->schema.getKdim(k);
    }        
    else{
        return this->user_view[k];
    }
}
// reshaping
// void Tensor::reshape(int d, std::vector<int> s){
//     this->schema = Schema(d,s,this->require_grad());
// }
// void Tensor::reshape(int d, int* s){
//     this->schema = Schema(d,s,this->require_grad());
// }

// view
void Tensor::view(std::vector<int> new_view){
    // default as compatible shape! (contiguous reshaping!)
    this->occupancy_map = std::vector<int>();
    this->user_view = new_view;
    this->isViewed=true;
    int schema_ptr = 0, new_ptr = 0;
    while (new_ptr<new_view.size()-1){
        if (this->getKdim(schema_ptr)==new_view[new_ptr]){
            // the same, place 1
            this->occupancy_map.push_back(1);
            schema_ptr++; new_ptr++;
        }
        else if (new_view[new_ptr]==1){
            // redundant, place 0
            this->occupancy_map.push_back(0);
            new_ptr++;
        } else{
            // merged, place merge number.
            int cnt = 0;
            while (new_view[new_ptr]>1){
                new_view[new_ptr] /= this->getKdim(schema_ptr++);
                cnt++;
            }
            this->occupancy_map.push_back(cnt); new_ptr++;
        }
        if (schema_ptr>=this->getDim())
            break;
    }
    if (schema_ptr>=this->getDim()){
        // redundant 1's at the end!
        while (new_ptr<new_view.size()){
            this->occupancy_map.push_back(0);
            new_ptr++;
        }
    }else{
        this->occupancy_map.push_back(this->getDim() - schema_ptr);
    }
    // debug
    #ifdef UNIT_TEST
    this->printShape();
    printf("map: ");
    for (int j=0; j<this->occupancy_map.size();++j){
        printf("%d ",this->occupancy_map[j]);
    }
    printf("\n");
    #endif
}
// permute
bool Tensor::permute(std::vector<int> axis_perm){
    if (this->isViewed) return false; // failed!
    this->schema.permute(axis_perm);
    return true;
}
// internal get
float& Tensor::internal_get(std::vector<int> index){
    int real_idx = 0;
    int offset = 1;
    for(int j=this->getDim()-1;j>=0;j--){
        real_idx += index[this->schema.getMap(j)]*offset;
        offset *= this->schema.realKdim(j);
    }
    return this->content[real_idx];
}
// external get (user-level view)
float& Tensor::get(std::vector<int> index){
    if (!this->isViewed) return this->internal_get(index); // No view at all.
    std::vector<int> internal_index;
    int user_ptr = 0;
    int schema_ptr = 0;
    while (user_ptr < this->occupancy_map.size()){
        switch (this->occupancy_map[user_ptr]){
            case 0: 
                user_ptr++;
                break;
            case 1: 
                internal_index.push_back(index[user_ptr]); 
                schema_ptr++; user_ptr++;
                break;
            default:
                int offset = this->user_view[user_ptr];
                for (int j=0;j<this->occupancy_map[user_ptr];++j){
                    offset /= this->getKdim(schema_ptr+j);
                    internal_index.push_back(index[user_ptr]/offset); // +1-1, +1: the very column, -1: the very index.
                    index[user_ptr] = index[user_ptr] - (index[user_ptr]/offset)*offset; // +1-1, inverse operation.
                }
                schema_ptr += this->occupancy_map[user_ptr];
                user_ptr++;
                break;
        }
        #ifdef UNIT_TEST
        printf("%d,%d ",schema_ptr,user_ptr);
        #endif
    }
    #ifdef UNIT_TEST
    printf("\n");
    for (int j=0; j<internal_index.size(); ++j){
        printf("%d ",internal_index[j]);
    }
    printf("\n");
    #endif
    return this->internal_get(internal_index);
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
    // tensor1->reshape(3,{N1,m,n});
    // tensor2->reshape(3,{N2,n,k});
    
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
    // result->schema.setShape(resShape);
    result->schema.setKdim(-2, m);
    result->schema.setKdim(-1, k);
    // (...,m,k)
    return result;
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
        printf("%.5f, ", this->content[j]);
    }
    printf("]\n");
}
void Tensor::printShape(){
    if (!this->isViewed)
        this->schema.print();
    else{
        printf("View Shape: ");
        for (int j=0;j<this->user_view.size();++j){
            printf("%d ",this->user_view[j]);
        }
        printf("\n");
    }
}