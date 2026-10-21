#!/bin/bash
set -e

echo "Compiling nsphere_cuda.cu for NVIDIA Tesla T4 (sm_75)..."
nvcc -O3 -arch=sm_75 nsphere_cuda.cu -o nsphere_cuda

echo ""
echo "======================================================================"
echo "1. ASSIGNMENT BRIEF VERIFICATION EXAMPLES"
echo "======================================================================"
./nsphere_cuda 1 25.5 1 1
./nsphere_cuda 2 2.05 1 1
./nsphere_cuda 3 1.5  1 1

echo ""
echo "======================================================================"
echo "2. RADIUS SCALING IN 2D & 3D (Comparing Naive Atomic vs Shared Reduction)"
echo "======================================================================"
./nsphere_cuda 2 500.5  1 1
./nsphere_cuda 2 2000.5 1 1
./nsphere_cuda 2 5000.5 1 1
./nsphere_cuda 3 50.5   1 1
./nsphere_cuda 3 100.5  1 1
./nsphere_cuda 3 200.5  1 1

echo ""
echo "======================================================================"
echo "3. DIMENSIONALITY SCALING (N = 2 to 11, Radius = 2.5 -> Base = 5)"
echo "======================================================================"
for n in 2 3 4 5 6 7 8 9 10 11; do
    ./nsphere_cuda $n 2.5 1 1
done

echo ""
echo "======================================================================"
echo "4. HIGH-DIMENSIONAL / BILLION-POINT GPU STRESS TESTS (GPU Only)"
echo "======================================================================"
./nsphere_cuda 12 2.5 0 1
./nsphere_cuda 13 2.5 0 1
./nsphere_cuda 4  50.5 0 1
./nsphere_cuda 18 1.5 0 1
