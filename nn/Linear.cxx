#include "Linear.hpp"
#include "CppNN/Tensor.hpp"

// Tensor* Linear::forward(Tensor& input){
//     /*  input: (N,in_features,...), 
//         weight: (out_features, in_features), 
//         bias: (out_features)
//         results: (N,out_features,...)
//     */
//     Schema resSchema = Schema(input.getSchema());
//     resSchema.setKdim(1,this->out_features);
//     Tensor* result = newTensor(resSchema);
//     // input: (N,in_features,...) -> (N,...,in_features,1)

//     // result = weight @ input (N,...,out_features,1)

//     // result += bias (N,...,out_features)

//     // result: (N,out_features,...)

//     return result;
// }