#!/bin/bash

cd $(dirname $0)

mkdir -p bin

/usr/bin/g++-4.2 -headerpad_max_install_names -arch x86_64 -Xarch_x86_64 -mmacosx-version-min=10.6 -Isrc $(pkg-config --cflags opencv) -o bin/video_textures src/main.cpp src/video_textures.cpp src/randomlib.c $(pkg-config --libs opencv) -lm
