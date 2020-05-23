#include <stdio.h>
#include <string.h>
#include <dlfcn.h>

extern int stencil_9pt(int argc, char** argv);
extern int hub(int, int);

int main(int argc, char** argv)
{
	for (int i = 0; i < argc; i++){
		printf("%s ", argv[i]);
		if (i == argc - 1) printf("\n");
	}
	const char* pts  = argv[2];
	const char* dtype = argv[3];
	
#define CALL_FUNC(num)\
	if (strcmp(pts, #num) == 0) {\
		if (strcmp(dtype, "double") == 0) {\
			extern int stencil_##num##pt_double(int argc, char** argv);\
			return stencil_##num##pt_double(argc, argv);\
		}\
		else if (strcmp(dtype, "float") == 0) {\
			extern int stencil_##num##pt_float(int argc, char** argv);\
			return stencil_##num##pt_float(argc, argv);\
		}else {\
			printf("error!\n");\
		}\
	}
	
    // List all the supported runtime and check in order
    // Check if a CUDA runtime is loaded
    //void* cuda_runtime = dlopen("libcudart.so.10.1", RTLD_LAZY);
    //if (cuda_runtime == NULL)
    //    printf("We do not have CUDA SHUFFLE 10.1.\n");

    //cuda_runtime = dlopen("libcudart.so.10.3", RTLD_LAZY);
    //if (cuda_runtime != NULL)
    //    printf("We have CUDA SHUFFLE 10.3.\n");

    // If there's no CUDA runtime loaded, use CPU kernel instead
    //if (cuda_runtime == NULL) {
    //    int size = 4096;
    //    hub(size, size);
    //}
    //else {
	    CALL_FUNC(9);
    //};
	printf("error, do not call functions\n");
	return 0;
}

//{
	//	if (strcmp(pts, "5pt")) {
//		if (strcmp(dtype, "double") == 0) {
//			extern int stencil_5pt_double(int argc, char** argv);
//			return stencil_5pt_double(argc, argv);
//		}
//		else if (strcmp(dtype, "float") == 0) {
//			extern int stencil_5pt_float(int argc, char** argv);
//			return stencil_5pt_float(argc, argv);
//		}
//		else {
//			printf("error!\n");
//		}		
//	}
//}
