cd ../FBGEMM/build
cmake ..  -DFBGEMM_LIBRARY_TYPE=shared -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_BUILD_TYPE=Debug
make -j 16
make install
cd ../../tvm-hhl
source ins.sh
