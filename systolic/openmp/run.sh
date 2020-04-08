#!/bin/bash
export LD_LIBRARY_PATH=~/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export PATH=~/local/cuda-10.0/bin:$PATH 
###########################################
name="5 9 13 17 21 25 64 81 121"
exe=./Release/ssai-2dstencil
for s in $name; do
	$exe 8192 $s double
	$exe 8192 $s float
done
