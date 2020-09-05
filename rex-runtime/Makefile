ifeq (${LLVM_OPENMP_INSTALL},)
	LLVM_OPENMP_INSTALL := ${LLVM_PATH}
endif
ifeq (${LLVM_OPENMP_SOURCE},)
	LLVM_OPENMP_SOURCE := ${LLVM_SRC}/openmp
endif
ifeq (${CUDA_ARCH},)
	CUDA_ARCH := sm_60
endif


default: cudakernel_fatbinwrapper
	clang++ -g -fopenmp axpy_libomptarget.c -I${LLVM_OPENMP_SOURCE}/libomptarget/include -o axpy_libomptarget.out -L${LLVM_OPENMP_INSTALL}/lib -lomptarget

# This target compiles a CUDA program into binary, store the binary in an array and generate the wrapper for using with LLVM OpenMP target library
cudakernel_fatbinwrapper: axpy_cudakernel.cu
	nvcc -cubin -g -arch ${CUDA_ARCH} axpy_cudakernel.cu

clean: 
		rm -rf ${OBJS} *.cubin *.out

