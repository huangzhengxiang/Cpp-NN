## 3 levels of abstraction for Tensor operation
### 1. Schema (physical)
The real shape / physical shape of the Tensor is stored here, while also the permuted shape / external shape is stored here.
- internal access method: real shape Op
- external access method: external (permuted) shape Op
- manipulation: permute()
### 2. Tensor (logical / external)
The data pointer is stored here.
- internal access method: external shape Op
- external access method: user-level shape Op (after view)
- manipulation: inplace operators / element-wise Op, and all kinds of arithmetic Ops, view, and repeat.
- caution: therefore, only 1 view can be created at a time.
- after view(), no permutation is allowed.