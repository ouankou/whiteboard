#include <stdio.h>
#include <string.h>

extern int stencil_3d7pt_v3(int argc, char** argv);
extern int stencil_poisson(int argc, char** argv);

int main(int argc, char** argv)
{
	if (argc < 3) {
		printf("error, argc = %d\n", argc);
		return 1;
	}
	const char* pfun = argv[1];
	if (0) {}
	else if (strstr(pfun, "7pt")) return stencil_3d7pt_v3(argc, argv);
//	else if (strstr(pfun, "poisson")) return stencil_poisson(argc, argv);
	else {
		printf("invalid function name!\n");
	}
	//stencil_3d7pt_v1(argc, argv);
	//stencil_3d7pt_v2(argc, argv);
	//stencil_3d7pt_v3(argc, argv);
	////stencil_3d13pt(argc, argv);
	//stencil_poisson(argc, argv);
	
	return 0;
}
