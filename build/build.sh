find . -mindepth 1 ! -name "build.sh" -exec rm -rf {} +

# cmake ..
cmake -DCMAKE_BUILD_TYPE=Release ..

make -j8

# ctest --output-on-failure

# ./tests/vadd_tests
# ./samples/vadd_demo