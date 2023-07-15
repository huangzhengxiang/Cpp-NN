#ifndef VIEW_HPP_
#define VIEW_HPP_

#include "Tensor.hpp"
#include "Schema.hpp"
#include <vector>

// // Basic Op
View* operator+(View& t1, View& t2);
// View* operator+(int a, View& view);
// View* operator+(float a, View& view);
// View* operator+(double a, View& view);
// View* operator+(View& view, int a);
// View* operator+(View& view, float a);
// View* operator+(View& view, double a);
View* operator-(View& t1, View& t2);
// View* operator-(int a, View& view);
// View* operator-(float a, View& view);
// View* operator-(double a, View& view);
// View* operator-(View& view, int a);
// View* operator-(View& view, float a);
// View* operator-(View& view, double a);
View* operator*(View& t1, View& t2);
// View* operator*(int a, View& view);
// View* operator*(float a, View& view);
// View* operator*(double a, View& view);
// View* operator*(View& view, int a);
// View* operator*(View& view, float a);
// View* operator*(View& view, double a);
// View* operator/(View& t1, View& t2);
// View* operator/(View& view, int a);
// View* operator/(View& view, float a);
// View* operator/(View& view, double a);

// Matrix Multiplication
View* matmul(View& t1, View& t2);

class View(public Tensor){
private:

public:
    // Matrix Multiplication
    friend View* matmul(View& t1, View& t2);
    // Basic Op
    friend View* operator+(View& t1, View& t2);
    // friend View* operator+(int a, View& view);
    // friend View* operator+(float a, View& view);
    // friend View* operator+(double a, View& view);
    // friend View* operator+(View& view, int a);
    // friend View* operator+(View& view, float a);
    // friend View* operator+(View& view, double a);
    friend View* operator-(View& t1, View& t2);
    // friend View* operator-(int a, View& view);
    // friend View* operator-(float a, View& view);
    // friend View* operator-(double a, View& view);
    // friend View* operator-(View& view, int a);
    // friend View* operator-(View& view, float a);
    // friend View* operator-(View& view, double a);
    friend View* operator*(View& t1, View& t2);
    // friend View* operator*(int a, View& view);
    // friend View* operator*(float a, View& view);
    // friend View* operator*(double a, View& view);
    // friend View* operator*(View& view, int a);
    // friend View* operator*(View& view, float a);
    // friend View* operator*(View& view, double a);
    // friend View* operator/(View& t1, View& t2);
    // friend View* operator/(View& view, int a);
    // friend View* operator/(View& view, float a);
    // friend View* operator/(View& view, double a);

};

#endif