
cd mmdet/ops/bbox
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup_linux.py build_ext --inplace