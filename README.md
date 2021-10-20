
# LibQPEP Multi-Precision
The QPEP (Quadratic Pose Estimation Problems) library with multi-precision support. The current repo is written in type-oriented C++ templates, rather than the double-precision version in https://github.com/zarathustr/LibQPEP. It can be merged to low-configuration platforms with only single-precision support.


## Compilation
```bash
git clone https://github.com/zarathustr/LibQPEP
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make install
```

## Demo Program
Just run
```bash
./LibQPEP-test
```


## Publication
Wu, J., Zheng, Y., Gao, Z., Jiang, Y., Hu, X., Zhu, Y., Jiao, J., Liu*, M. (2020)
           Quadratic Pose Estimation Problems: Globally Optimal Solutions, 
           Solvability/Observability Analysis and Uncertainty Description.
