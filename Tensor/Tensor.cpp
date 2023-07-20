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
#include "TensorIterator.hpp"

Tensor* newTensor(Tensor& tensor){
    Tensor* result = new Tensor(tensor.schema);
    result->content = new float[result->getSize()];
    if(tensor.viewed()) result->view(tensor.getShape());
    return result;
}
Tensor* newTensor(Schema& schema){
    Tensor* result = new Tensor(schema);
    result->content = new float[result->getSize()];
    return result;
}
Tensor* zero_like(Tensor& tensor){
    Tensor* result = new Tensor(tensor.schema);
    result->content = new float[result->getSize()]{0};
    if(tensor.viewed()) result->view(tensor.getShape());
    return result;
}
bool isCompatible(Tensor& t1, Tensor& t2){
    if (t1.getDim()!=t2.getDim()) return false;
    for (int j=0;j<t1.getDim();++j){
        if (t1.getKthdim(j)!=t2.getKthdim(j)) return false;
    }
    return true;
}
bool isBroadcastCompatible(Tensor& t_big, Tensor& t_small){
    if (t_big.getDim()!=t_small.getDim()) return false;
    for (int j=0;j<t_big.getDim();++j){
        if (t_big.getKthdim(j)!=t_small.getKthdim(j) && t_small.getKthdim(j)!=1) 
            return false;
    }
    return true;
}
bool isBroadcastMulCompatible(Tensor& t_left, Tensor& t_right){
    // (...,m,n) @ (...,n,k)
    if (t_left.getDim()!=t_right.getDim()) return false;
    for (int j=0;j<t_left.getDim()-2;++j){
        if (t_left.getKthdim(j)!=t_left.getKthdim(j) && \
             t_left.getKthdim(j)!=1 && t_right.getKthdim(j)!=1) 
            return false;
    }
    if (t_left.getKthdim(-1)!=t_right.getKthdim(-2)) return false;
    return true;
}
void initBroadcastMul(Tensor& t_left, Tensor& t_right, FullIterator* it_left, FullIterator* it_right, FullIterator* it_result){
    // (...,m,n) @ (...,n,k)
    for (int j=0;j<t_left.getDim()-2;++j){
        if (t_left.getKthdim(j)!=t_right.getKthdim(j)){
            if(t_left.getKthdim(j)!=1){
                it_right->repeat(j,t_left.getKthdim(j));
            }else{ // t_right.getKthdim(j)!=1
                it_left->repeat(j,t_right.getKthdim(j));
            }
        }
    }
    it_left->setSubTensor(-2);
    it_right->setSubTensor(-2);
    it_result->setSubTensor(-2);
    it_left->openSubTensorIterator();
    it_right->openSubTensorIterator();
    it_result->openSubTensorIterator();
}

Tensor* initBroadcast(Tensor& t1, Tensor& t2, FullIterator* it1, FullIterator* it2){
    Tensor* result=NULL;
    // The physical shape of the result is the same as that of the bigger tensor or the left one if the same shape.
    if (isBroadcastCompatible(t1,t2)){
        // t1 is bigger.
        result = newTensor(t1);
        it2->broadcast(t1);
    }else if(isBroadcastCompatible(t2,t1)){
        // t1 is bigger.
        result = newTensor(t2);
        it1->broadcast(t2);
    }else{
        return NULL;
    }
    it1->open();
    it2->open();
    return result;
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

int Tensor::getDim(){
    if (!this->isViewed){
        return this->schema.getDim();
    }else{
        return this->user_view.size();
    }
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
        k = (k>=0) ? (k) : (this->getDim() + k);
        return this->user_view[k];
    }
}

// view
void Tensor::view(){
    if (this->isViewed) return;
    // default view = original view
    this->occupancy_map = std::vector<int>(this->getDim(),1);
    this->user_view = this->getShape();
    this->isViewed=true;
}
void Tensor::view(std::vector<int> new_view){
    // default as compatible shape! (contiguous reshaping!)
    this->occupancy_map = std::vector<int>();
    this->user_view = new_view;
    this->isViewed=true;
    // calculate occupancy map!
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
        if (schema_ptr>=this->schema.getDim())
            break;
    }
    if (schema_ptr>=this->schema.getDim()){
        // redundant 1's at the end!
        while (new_ptr<new_view.size()){
            this->occupancy_map.push_back(0);
            new_ptr++;
        }
    }else{
        this->occupancy_map.push_back(this->schema.getDim() - schema_ptr);
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
    for(int j=this->schema.getDim()-1;j>=0;j--){
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
    Tensor* result = newTensor(tensor);
    FullIterator* resultIterator = new FullIterator(result);
    FullIterator* sourceIterator = new FullIterator(&tensor);
    resultIterator->open();
    sourceIterator->open();
    while(sourceIterator->hasNext()) resultIterator->next()=std::log(sourceIterator->next());
    return result;
}
Tensor* log(Tensor& tensor, float a){
    float k = log(a);
    Tensor* result = newTensor(tensor);
    FullIterator* resultIterator = new FullIterator(result);
    FullIterator* sourceIterator = new FullIterator(&tensor);
    resultIterator->open();
    sourceIterator->open();
    while(sourceIterator->hasNext()) resultIterator->next()=std::log(sourceIterator->next())/k;
    return result;
}
Tensor* exp(Tensor& tensor){
    Tensor* result = newTensor(tensor);
    FullIterator* resultIterator = new FullIterator(result);
    FullIterator* sourceIterator = new FullIterator(&tensor);
    resultIterator->open();
    sourceIterator->open();
    while(sourceIterator->hasNext()) resultIterator->next()=std::exp(sourceIterator->next());
    return result;
}
// a is exponent.
Tensor* pow(Tensor& tensor, float a){
    Tensor* result = newTensor(tensor);
    FullIterator* resultIterator = new FullIterator(result);
    FullIterator* sourceIterator = new FullIterator(&tensor);
    resultIterator->open();
    sourceIterator->open();
    while(sourceIterator->hasNext()) resultIterator->next()=std::pow(sourceIterator->next(),a);
    return result;
}
// a is base.
Tensor* pow(float a, Tensor& tensor){
    Tensor* result = newTensor(tensor);
    FullIterator* resultIterator = new FullIterator(result);
    FullIterator* sourceIterator = new FullIterator(&tensor);
    resultIterator->open();
    sourceIterator->open();
    while(sourceIterator->hasNext())  resultIterator->next()=std::pow(a,sourceIterator->next());
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
    Tensor* result = newTensor(tensor);
    FullIterator* resultIterator = new FullIterator(result);
    FullIterator* sourceIterator = new FullIterator(&tensor);
    resultIterator->open();
    sourceIterator->open();
    while(sourceIterator->hasNext()) resultIterator->next()=std::sqrt(sourceIterator->next());
    return result;
}
void Tensor::sqrt(){
    for (int j=0; j<this->getSize(); ++j){
        this->content[j] = std::sqrt(this->content[j]);
    }
}

Tensor* matmul(Tensor& t1, Tensor& t2, Tensor* out){
    // Do not induce isViewed change.
    // (...,m,n) @ (...,n,k)
    if (!isBroadcastMulCompatible(t1,t2)) return NULL;
    FullIterator* tensor1 = new FullIterator(&t1);
    FullIterator* tensor2 = new FullIterator(&t2);
    FullIterator* resultIterator = new FullIterator(out);
    initBroadcastMul(t1,t2,tensor1,tensor2,resultIterator);
    // matmul
    while(resultIterator->hasNextSubTensor()){
        // conduct N times matmul
        tensor1->nextSubTensor();
        tensor2->nextSubTensor();
        resultIterator->nextSubTensor();
        for (int jl=0;jl<t1.getKthdim(-2);++jl){
            for (int jr=0;jr<t2.getKthdim(-1);++jr){
                float temp = 0.0;
                for (int b=0;b<t1.getKthdim(-1);++b){
                    temp += tensor1->subTensor_get({jl,b}) * tensor2->subTensor_get({b,jr});
                }
                resultIterator->subTensor_get({jl,jr}) = temp;
            }
        }
    }
    // return the result
    return out;
}

// Basic Op
Tensor* operator+(Tensor& t1, Tensor& t2){
    FullIterator* tensor1 = new FullIterator(&t1);
    FullIterator* tensor2 = new FullIterator(&t2);
    Tensor* result = initBroadcast(t1,t2,tensor1,tensor2);
    FullIterator* resultIterator = new FullIterator(result);
    resultIterator->open();
    while(resultIterator->hasNext()) {
        resultIterator->next() = tensor1->next() + tensor2->next();
    }
    return result;
}
Tensor* operator-(Tensor& t1, Tensor& t2){
    FullIterator* tensor1 = new FullIterator(&t1);
    FullIterator* tensor2 = new FullIterator(&t2);
    Tensor* result = initBroadcast(t1,t2,tensor1,tensor2);
    FullIterator* resultIterator = new FullIterator(result);
    resultIterator->open();
    while(resultIterator->hasNext()) resultIterator->next() = tensor1->next() - tensor2->next();
    return result;
}
Tensor* operator*(Tensor& t1, Tensor& t2){
    FullIterator* tensor1 = new FullIterator(&t1);
    FullIterator* tensor2 = new FullIterator(&t2);
    Tensor* result = initBroadcast(t1,t2,tensor1,tensor2);
    FullIterator* resultIterator = new FullIterator(result);
    resultIterator->open();
    while(resultIterator->hasNext()) resultIterator->next() = tensor1->next() * tensor2->next();
    return result;
}
Tensor* operator/(Tensor& t1, Tensor& t2){
    FullIterator* tensor1 = new FullIterator(&t1);
    FullIterator* tensor2 = new FullIterator(&t2);
    Tensor* result = initBroadcast(t1,t2,tensor1,tensor2);
    FullIterator* resultIterator = new FullIterator(result);
    resultIterator->open();
    while(resultIterator->hasNext()) resultIterator->next() = tensor1->next() / tensor2->next();
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