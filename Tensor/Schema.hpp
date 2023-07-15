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
bool isCompatible(Schema& t1, Schema& t2);

class Schema{
private:
    // All 2 vectors shall be of the same shape!!!
    std::vector<int> real_shape;    // The underlying physical shape. (required)
    std::vector<int> axis_map;      // The axis map against permutation. (default: No permute, 0,1,2,...)
    std::vector<int> a_inv;         // We'd better use a cache.
    bool isGrad;                    // Whether the Tensor require_grad
    std::vector<int> axis_map_inv();//get a_inv
public:
    // initialization
    Schema(std::vector<int> shape, bool grad);
    Schema(int d, int* shape, bool grad);
    Schema(std::vector<int> shape, std::vector<int> axis_perm, bool grad);
    Schema(Schema& other);
    // setting
    void setKdim(int k, int d);
    // get info
    int getDim();
    int getSize();
    int getKdim(int k);
    bool& require_grad(){return this->isGrad;}
    std::vector<int> getShape(); // external shape
    std::vector<int> realShape();//real shape
    int getMap(int j){return this->axis_map[j];}
    // advanced manipulation
    void permute(std::vector<int> axis_perm); // permute upon the external shape.
    // test print
    void print();
};



#endif /* SCHEMA_HPP_ */
