#!/bin/bash

source func.sh

GPU=$(get_gpu)
dirs='../ssai-j3d7pt ../ssai-j3d27pt ../ssai-j3d125pt ../ssai-poisson'

function prof_ssai3d(){
	for s in $dirs; do
		echo "cur_dir="$s
		pushd $s
			make CONFIG=Release clean all
			./run.sh > log.tmp
			dtype=$(cat log.tmp | grep "dtype" | awk -F "=" '{print $2}')	
			echo "GPU=" ${GPU}", dtype="${dtype}
			cat log.tmp
			fname=${GPU}.${dtype}.nvcc.results
			rm -f nvcc.results $fname
					
			./prof.sh	
			mv nvcc.results $fname
			cat $fname
		pushd
    done
}


function prof_double(){
	echo "###########Test 3d stenci double"
	set_double
	prof_ssai3d
}

function prof_float(){
	echo "###########Test 3d stenci float"
	set_float32
	prof_ssai3d
}

prof_double
prof_float
