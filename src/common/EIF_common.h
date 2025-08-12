//
//  EIF_common.h – common helpers for EIF MEX simulations
//
#ifndef EIF_COMMON_H
#define EIF_COMMON_H

/* ---------- System / MATLAB includes ---------- */
#include "mex.h"
#include "matrix.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---------- Build metadata ---------- */
#ifndef EIF_COMMON_VERSION
#define EIF_COMMON_VERSION "1.0.0"
#endif

/* ---------- Compile-time toggles ---------- */
#ifndef EIF_DEBUG
#define EIF_DEBUG 0
#endif

/* ---------- Numeric guards ---------- */
#ifndef EIF_MIN_DT
#define EIF_MIN_DT 1e-8
#endif

#ifndef EIF_MAX_DT
#define EIF_MAX_DT 1.0
#endif

#ifndef EIF_MAX_T
#define EIF_MAX_T 1e7
#endif

/* ---------- Convenience macros ---------- */
#if EIF_DEBUG
#  define DBG(...) mexPrintf("[DBG] " __VA_ARGS__)
#else
#  define DBG(...) do {} while(0)
#endif

#define STR2(x) #x
#define STR(x) STR2(x)

#define FAIL(msg) mexErrMsgIdAndTxt("EIF:Error", "%s", (msg))
#define FAIL_IF(cond, msg) do { if (cond) mexErrMsgIdAndTxt("EIF:Error", "%s", (msg)); } while(0)

#define REQUIRE(cond, what) do { if(!(cond)) mexErrMsgIdAndTxt("EIF:Missing", "Missing or invalid: %s", (what)); } while(0)

#define SAFE_MALLOC(ptr, n, type)                                                     \
    do {                                                                              \
        (ptr) = (type*)mxMalloc((mwSize)(n) * (mwSize)sizeof(type));                  \
        if(!(ptr)) mexErrMsgIdAndTxt("EIF:OOM", "Out of memory for " #ptr);           \
    } while(0)

#define SAFE_CALLOC(ptr, n, type)                                                     \
    do {                                                                              \
        (ptr) = (type*)mxCalloc((mwSize)(n), (mwSize)sizeof(type));                   \
        if(!(ptr)) mexErrMsgIdAndTxt("EIF:OOM", "Out of memory for " #ptr);           \
    } while(0)

#define SAFE_FREE(ptr) do { if(ptr){ mxFree(ptr); (ptr)=NULL; } } while(0)

/* ---------- Types shared by all variants ---------- */

/* Populations (if you use them across files) */
typedef enum { EXCITATORY = 0, INHIBITORY = 1, NUM_POP = 2 } Population;

/* Network-wide parameters commonly needed by all variants.
   Keep this minimal; extend in your variant .c files if needed. */
typedef struct {
    /* Counts (optional but commonly used) */
    int32_t Ne, Ni, Nx;      /* sizes of populations */
    int32_t N;               /* Ne + Ni (derived if not provided) */

    /* Simulation controls */
    double  T;               /* total time (ms or s; your convention) */
    double  dt;              /* time step */
    int32_t maxns;           /* spike buffer cap (optional) */

    /* Misc flags */
    int32_t attarea;         /* optional attention-area flag */
} NetworkParams;

/* ---------- Fast math ---------- */
/* Exponential with overflow/underflow clamps to avoid NaNs in inner loops. */
static inline double eif_fast_exp(double x) {
    if (x > 700.0) return INFINITY;
    if (x < -700.0) return 0.0;
    return exp(x);
}

/* ---------- RNG helpers (simple, reproducible when seeded) ---------- */
void eif_seed(uint64_t seed);
double eif_rand_uniform(void);                 /* in (0,1) */
double eif_rand_normal(void);                  /* mean 0, std 1 (Box–Muller) */

/* ---------- MATLAB / mxArray helpers ---------- */
const mxArray* eif_require_field(const mxArray* S, const char* name);
const mxArray* eif_get_field(const mxArray* S, const char* name); /* may return NULL */

double  eif_get_scalar_double(const mxArray* A, const char* what);
int32_t eif_get_scalar_int32 (const mxArray* A, const char* what);

double  eif_get_field_double_req(const mxArray* S, const char* name);
double  eif_get_field_double_opt(const mxArray* S, const char* name, double defv);

int32_t eif_get_field_int32_req (const mxArray* S, const char* name);
int32_t eif_get_field_int32_opt (const mxArray* S, const char* name, int32_t defv);

void    eif_require_type(const mxArray* A, mxClassID cls, const char* what);
void    eif_require_vector(const mxArray* A, const char* what);
void    eif_require_matrix(const mxArray* A, const char* what);
void    eif_require_length_at_least(const mxArray* A, mwSize nmin, const char* what);
void    eif_require_size(const mxArray* A, mwSize m, mwSize n, const char* what);

/* Cast a MATLAB numeric array to a newly allocated int32* (column-major) */
int32_t* eif_cast_to_int32_copy(const mxArray* A, const char* what);

/* ---------- Parsing & validation (generic) ---------- */
/* Parse a MATLAB struct of general network parameters.
   Only a few fields are *required* to keep this generic across variants. */
void parse_params(const mxArray* S, NetworkParams* P);

/* Validate feedforward spikes or any time vector: monotone non-decreasing and within [0, T]. */
void validate_time_vector(const double* t, mwSize nt, double T, const char* what);

/* Validate that an index array (1-based for MATLAB) is in [1, Nmax]. */
void validate_indices_1based(const int32_t* idx, mwSize n, int32_t Nmax, const char* what);

/* Convenience: check connectivity pairs (i,j) in two equal-length int32 arrays. */
void validate_edges_1based(const int32_t* pre, const int32_t* post, mwSize nEdges,
                           int32_t Npre, int32_t Npost, const char* what);

#endif /* EIF_COMMON_H */