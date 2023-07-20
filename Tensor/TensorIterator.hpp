#ifndef TENSOR_ITERATOR_HPP_
#define TENSOR_ITERATOR_HPP_

#include "CppNN/Tensor.hpp"

class Tensor;

class FullIterator{
private:
    Tensor* tensor;
    // iterator repeat view.
    std::vector<int> broadcast_map;
    // current index
    std::vector<int> current_index;
    std::vector<int> subTensor_index;
    int subTensorDim; // This must be a negative number! [-this->tensor->getDim()+1,-1]
    bool isOpen; 
    bool isSubTensorOpen;
    bool isFirstNext;
    bool isSubTensorFristNext;
    // private function
    void singleDimNext(int j);
    void singleDimSubTensorNext(int j);
    std::vector<int> deBroadcast(std::vector<int>& broadcasted);
public:
    FullIterator(Tensor* t):tensor(t),broadcast_map(t->getDim(),1),isOpen(false),isSubTensorOpen(false){}
    void repeats(std::vector<int> repeat_map);
    void repeat(int k, int r);
    void broadcast(Tensor& other);
    // item iterator
    void open();
    float& next();
    bool hasNext();
    void close();
    void rewind(){
        this->close();
        this->open();
    }
    // sub tensor iterator
    void openSubTensorIterator();
    void setSubTensor(int j){this->subTensorDim=j;}
    void nextSubTensor();
    bool hasNextSubTensor();
    float& subTensor_get(std::vector<int> sub_index);
    void closeSubTensorIterator();
    void rewindSubTensorIterator(){
        this->closeSubTensorIterator();
        this->openSubTensorIterator();
    }
    // test print
    void print_index();
    void print_subTensor_index();
    void print_broadcast_map();
    ~FullIterator(){}
};

// class BroadcastIterator{
// private:
//     Tensor* tensor;
//     std::vector<int> broadcast_map;
// public:
//     BroadcastIterator(Tensor* t_small, Tensor& t_big):t1(t_small){

//     }
//     BroadcastIterator(Tensor* t_small, std::vector<int> repeats):t1(t_small),broadcast_map(repeats){}
//     void open();
//     float& next();
//     bool hasNext();
//     void close();
//     void rewind(){
//         this->close();
//         this->open();
//     }
//     ~BroadcastIterator(){}
// };

#endif