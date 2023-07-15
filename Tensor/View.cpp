#include "Tensor.hpp"
#include "View.hpp"
#include <vector>

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
    tensor1->reshape(3,{N1,m,n});
    tensor2->reshape(3,{N2,n,k});
    
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
    result->schema.setShape(resShape);
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