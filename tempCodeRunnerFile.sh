
cd mmdet/ops/nms
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace