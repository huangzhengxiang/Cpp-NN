#ifndef LINEAR_HPP_
#define LINEAR_HPP_

#include "CppNN/Tensor.hpp"

class Linear;

/*  input: (N,in_features,...), 
    weight: (out_features, in_features), 
    bias: (out_features)
    results: (N,out_features,...)
*/
class Linear{
private:
    int in_features;
    int out_features;
    bool require_bias;
    Tensor* weight; // (out_features, in_features)
    Tensor* bias; // (out_features)
    bool require_grad;
public:
    Linear(){}
    Linear(int in_dim, int out_dim, bool bias_on):in_features(in_dim),out_features(out_dim),require_bias(bias_on){}
    Linear(int in_dim, int out_dim, bool bias_on, Tensor* w, Tensor* b):in_features(in_dim),out_features(out_dim),require_bias(bias_on){
        this->init(w,b);
    }
    void init(Tensor* w, Tensor* b){
        this->weight=w;
        this->bias=b;
        this->no_grad();
        // this->weight->view({this->out_features,this->in_features});
        // this->bias->view({this->out_features,1});
    }
    void grad(){
        this->require_grad=true;
        this->weight->require_grad()=true;
        this->bias->require_grad()=true;
    }
    void no_grad(){
        this->require_grad=false;
        this->weight->require_grad()=false;
        this->bias->require_grad()=false;
    }
    Tensor* forward(Tensor& input);
    ~Linear(){}
};

#endif