#include <stdio.h>
#include <string.h>

extern int stencil_9pt(int argc, char** argv);

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
		}else {\
			printf("error!\n");\
		}\
	}
	
	CALL_FUNC(9);
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
