#!/bin/bash

cd $(dirname $0)

SRC=$(pwd)/../src
docker run --rm -v $SRC:/src -u $(id -u):$(id -g) emscripten/emsdk emcc "$@"
