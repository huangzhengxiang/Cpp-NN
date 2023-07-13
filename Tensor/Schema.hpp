/*
 * Schema.hpp
 *
 *  Created on: 2023/7/12
 *      Author: hzx
 */

#ifndef SCHEMA_HPP_
#define SCHEMA_HPP_

#include <vector>

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
};



#endif /* SCHEMA_HPP_ */
