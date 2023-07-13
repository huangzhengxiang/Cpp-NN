/*
 * Schema.hpp
 *
 *  Created on: 2023/7/12
 *      Author: hzx
 */

#ifndef SCHEMA_HPP_
#define SCHEMA_HPP_

#include <vector>

class Schema;

// isCompatible
bool isCompatible(Schema& t1, Schema& t2){
    if (t1.getDim()!=t2.getDim()) return false;
    for (int j=0;j<t1.dim;++j){
        if (t1.getKdim(j)!=t2.getKdim(j)) return false;
    }
    return true;
}

class Schema{
private:
    int dim;
    std::vector<int> Shape;
    bool isGrad;
public:
    // initialization
    Schema(int d, int* s, bool grad);
    Schema(int d, std::vector<int> s, bool grad);
    Schema(Schema& other);
    // setting
    void setKdim(int k, int d);
    // get info
    int getDim();
    int getSize();
    int getKdim(int k);
    bool& require_grad(){return this->isGrad;}
    std::vector<int> getShape();
    void setShape(std::vector<int> shape);
    // test print
    void print();
};



#endif /* SCHEMA_HPP_ */
