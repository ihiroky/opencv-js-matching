#!/bin/bash

test -d opencv || git clone https://github.com/opencv/opencv.git
cd opencv
git apply ../opencv-20201012.diff

docker run -it --rm -v $(pwd):/src -u $(id -u):$(id -g) emscripten/emsdk bash -c \
  'cd /src/opencv && python3 platforms/js/build_js.py --emscripten_dir=$EMSDK/upstream/emscripten --build_wasm build_wsam'
