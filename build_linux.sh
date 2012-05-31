#!/bin/bash

cd $(dirname $0)

mkdir -p bin

g++ -Isrc -lcv -lhighgui -lm -o bin/video_textures src/main.cpp src/video_textures.cpp src/randomlib.c -I/usr/include/opencv/
