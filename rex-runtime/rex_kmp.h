
typedef struct ident {
    int reserved_1;
    int flags;
    int reserved_2;
    int reserved_3;
    char const *psource;
} ident_t;

void __kmpc_fork_call(ident_t*, int, void*, ...);

