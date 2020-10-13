#!/bin/bash

cd $(dirname $0)

SRC=$(pwd)/..
docker run -it --rm -v $SRC:/src -u $(id -u):$(id -g) emscripten/emsdk bash "$@"
