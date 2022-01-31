#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

# echo "Building sigmoid_focal_loss op..."
# cd mmdet/ops/nms
# if [ -d "build" ]; then
#     rm -r build
# fi
# $PYTHON setup.py build_ext --inplace

# echo "Building fcosr_tools op..."
# cd mmdet/ops/fcosr_tool
# if [ -d "build" ]; then
#     rm -r build
# fi
# $PYTHON setup_tools.py build_ext --inplace

# echo "Building sigmoid_focal_loss op..."
# cd mmdet/ops/sigmoid_focal_loss
# if [ -d "build" ]; then
#     rm -r build
# fi
# $PYTHON setup.py build_ext --inplace

# echo "Building poly_nms op..."
# cd mmdet/ops/poly_nms
# if [ -d "build" ]; then
#     rm -r build
# fi
# $PYTHON setup.py build_ext --inplace

# echo "Building roi align rotated op..."
# cd mmdet/ops/roi_align_rotated
# if [ -d "build" ]; then
#     rm -r build
# fi
# $PYTHON setup.py build_ext --inplace

# echo "Building ps roi align rotated op..."
# cd ../psroi_align_rotated
# if [ -d "build" ]; then
#     rm -r build
# fi
# $PYTHON setup.py build_ext --inplace
# 
echo "Building poly_nms op..."
cd ../poly_nms
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace
# 
# echo "Building cpu_nms..."
# cd ../../core/bbox
# $PYTHON setup_linux.py build_ext --inplace

