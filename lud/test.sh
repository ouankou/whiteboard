#!/bin/bash

echo "Problem Size,Parallel For,SIMD,Target Parallel For, Target SIMD" > lud.csv
for i in 64 128 256 512 1024
  do
    echo -n "${i}," >> lud.csv
    ./lud_parallel_for.out $i >> lud.csv
    echo -n "," >> lud.csv
    ./lud_simd.out $i >> lud.csv
    echo -n "," >> lud.csv
    ./lud_target_parallel_for.out $i >> lud.csv
    echo -n "," >> lud.csv
    ./lud_target_simd.out $i >> lud.csv
    echo -n "," >> lud.csv
    echo "" >> lud.csv
  done
