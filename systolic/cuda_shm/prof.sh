#!/bin/bash
export LD_LIBRARY_PATH=~/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export PATH=~/local/cuda-10.0/bin:$PATH 

function get_gpu(){
	nvidia-smi | cat | head -n 8 | tail -n 1 | awk '{print $4}' | awk -F '-' '{print $1}' 
}
###########################################
#name="tmp_5 tmp_9 5 9 13 17 21 25 64 81 121"
name="9"

rm -f nvcc.results
GPU=$(get_gpu)
echo "GPU="${GPU}
rm -f ${GPU}.float32.nvcc.results ${GPU}.double.nvcc.results
exe=./Release/ssai-2dstencil
for s in $name; do
	echo "prof : " $s, double, float
	for ((i=0; i<1; i++)); do $exe 8192 $s double; done
	nvprof --normalized-time-unit ms --print-gpu-trace $exe 8192 $s double > /dev/null 2>>${GPU}.double.nvcc.results
	for ((i=0; i<1; i++)); do $exe 8192 $s float; done
	nvprof --normalized-time-unit ms --print-gpu-trace $exe 8192 $s float > /dev/null 2>>${GPU}.float32.nvcc.results
done
cat *.nvcc.results | grep 'void' | awk '{print $2, $17 ,$21}'
