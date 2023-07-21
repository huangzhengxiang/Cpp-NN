#include <stdio.h>
#include <vector>
#include "TensorIterator.hpp"
#include "CppNN/Tensor.hpp"

void FullIterator::singleDimNext(int j){
    bool isBroadcast = (this->broadcast_map[j]>1);
    int dimLimit = ((isBroadcast) ? this->broadcast_map[j] : this->tensor->getKthdim(j)) -1;
    if (this->current_index[j] == dimLimit){
        this->current_index[j] = 0; // reset
        if (j>0) singleDimNext(j-1); // When hasNext==false, rewind.
    }else{
        this->current_index[j]++;
    }
}
void FullIterator::singleDimSubTensorNext(int j){
    bool isBroadcast = (this->broadcast_map[j]>1);
    int dimLimit = ((isBroadcast) ? this->broadcast_map[j] : this->tensor->getKthdim(j)) -1;
    if (this->subTensor_index[j] == dimLimit){
        this->subTensor_index[j] = 0; // reset
        if (j>0) singleDimSubTensorNext(j-1); // When hasNext==false, rewind.
    }else{
        this->subTensor_index[j]++;
    }
}
std::vector<int> FullIterator::deBroadcast(std::vector<int>& broadcasted){
    std::vector<int> result(broadcasted);
    for (int j=0;j<broadcasted.size();++j){
        if (this->broadcast_map[j]>1) result[j]=0;
    }
    return result;
}

void FullIterator::repeats(std::vector<int> repeat_map){
    this->broadcast_map=repeat_map;
    for (int j=0;j<this->broadcast_map.size();++j){
        if (this->broadcast_map[j]!=1 && this->tensor->getKthdim(j)!=1)
            // error!
            this->broadcast_map[j]=0;
    }
}
void FullIterator::repeat(int k, int r){
    if(this->tensor->getKthdim(k)==1)
        this->broadcast_map[k]=r;
}
void FullIterator::broadcast(Tensor& other){
    if (!isBroadcastCompatible(other, *(this->tensor))) return; // do nothing!
    for (int j=0;j<other.getDim();++j){
        if (this->tensor->getKthdim(j)!=other.getKthdim(j)) this->repeat(j,other.getKthdim(j));
    }
}
// item iterator
void FullIterator::open(){
    this->isOpen = true;
    this->isFirstNext = true;
    this->current_index = std::vector<int>(this->tensor->getDim(),0);
}
float& FullIterator::next(){
    if (this->isFirstNext){
        this->isFirstNext=false;
    }else{
        singleDimNext(this->tensor->getDim()-1);
    }    
    std::vector<int> result_index = this->deBroadcast(this->current_index);
    return this->tensor->get(result_index);
}
bool FullIterator::hasNext(){
    if (!(this->isOpen)) return false;
    for (int j=0;j<this->broadcast_map.size();++j){
        if (this->current_index[j] < (this->broadcast_map[j]*this->tensor->getKthdim(j) - 1))
            return true;
    }
    // If no element left, close it.
    this->close();
    return false;
}
void FullIterator::close(){
    isOpen = false;
}
// sub tensor iterator
void FullIterator::openSubTensorIterator(){
    this->isSubTensorOpen=true;
    this->isSubTensorFristNext=true;
    this->subTensor_index = std::vector<int>(this->tensor->getDim()+this->subTensorDim,0);
}
void FullIterator::nextSubTensor(){
    if (this->isSubTensorFristNext){
        this->isSubTensorFristNext = false;
        return;
    }
    singleDimSubTensorNext(this->tensor->getDim()+this->subTensorDim-1);
}
bool FullIterator::hasNextSubTensor(){
    if (!(this->isSubTensorOpen)) return false;
    for (int j=0;j<this->subTensor_index.size();++j){
        if (this->subTensor_index[j] < (this->broadcast_map[j]*this->tensor->getKthdim(j) - 1))
            return true;
    }
    // If no element left, close it.
    this->closeSubTensorIterator();
    return false;
}
float& FullIterator::subTensor_get(std::vector<int> sub_index){
    std::vector<int> result_index = this->deBroadcast(this->subTensor_index);
    for (int j=0;j<sub_index.size();++j){
        // extend
        result_index.push_back(sub_index[j]);
    }
    return this->tensor->get(result_index);
}
void FullIterator::closeSubTensorIterator(){
    isSubTensorOpen = false;
}

void FullIterator::print_index(){
    if (this->isOpen){
        printf("current index: ");
        for (int j=0;j<this->current_index.size();++j){
            printf("%d ",this->current_index[j]);
        }
        printf("\n");
    }
}

void FullIterator::print_subTensor_index(){
    if (this->isSubTensorOpen){
        printf("current subTensor index: ");
        for (int j=0;j<this->subTensor_index.size();++j){
            printf("%d ",this->subTensor_index[j]);
        }
        printf("\n");
    }
}

void FullIterator::print_broadcast_map(){
    printf("broadcast map: ");
    for (int j=0;j<this->broadcast_map.size();++j){
        printf("%d ",this->broadcast_map[j]);
    }
    printf("\n");
}