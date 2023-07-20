## C++ Nerual Network Practice
### 1. Train the Network with Pytorch
Run the following command. This will train the network with Pytorch and simultaneously convert both the model and the weight into C style format in the header which can be subsequently built in our framework.

~~~
python converter.py
~~~

Both the network structure and pretrained weights are built into Network.hpp

### 2. Build & Run the Code
Then you can build the Inference framework from source by running the following commands. 
(For Windows)
~~~
mkdir build
cd build
cmake .. -DCOMPLETE_TEST=ON
cmake --build .
cd ..
build\Debug\Try.exe
~~~

### 3. Supported Operators
- [x] weight converter
- [ ] model converter (TorchScript)
- [x] Tensor::permute()
- [x] Tensor::view()
- [x] Tensor::repeat()
- [ ] Tensor::copy()
- [x] Linear (no_grad)
- [ ] ReLU (no_grad)
- [ ] ReLU6 (no_grad)
- [ ] Sigmoid (no_grad)
- [ ] Tanh (no_grad)
- [ ] Conv1d (no_grad)
- [ ] Conv2d (no_grad)