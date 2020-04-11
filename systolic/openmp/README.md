
# Issues

1. While using ROSE/REX compiler to generate new source code, the `target` directive has to be in the `main` function instead of an individual function.
Otherwise, it will complain some error related to function wrapper.

1. To generate CUDA code, the compilation parameter `-rose:skipfinalCompileStep` should be specified.
The command line would be like `~/Projects/rose_local/rose_install/bin/rose-compiler -l$HOME/Projects/rose_local/rose_install/lib/librose.so.0 -rose:openmp:lowering -rose:skipfinalCompileStep -c target_parallel_for.c`
