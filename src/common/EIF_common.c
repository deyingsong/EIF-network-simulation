//
//  EIF_common.c – implementation of common helpers for EIF simulations
//
#include "EIF_common.h"

/* ---------- RNG (portable, simple) ---------- */
static uint64_t g_rng_state = 0x9E3779B97F4A7C15ULL;

/* xorshift64* */
static inline uint64_t eif_next_u64(void) {
    uint64_t x = g_rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    g_rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

void eif_seed(uint64_t seed) {
    if (seed == 0) seed = (uint64_t)time(NULL);
    /* scramble zero seeds as well */
    g_rng_state = seed ^ 0xD2B74407B1CE6E93ULL;
    /* warm up */
    for (int i = 0; i < 10; ++i) (void)eif_next_u64();
}

double eif_rand_uniform(void) {
    /* return in (0,1) open interval to avoid log(0) in Box–Muller */
    const double k = 1.0 / (double)UINT64_MAX;
    double u = (double)eif_next_u64() * k;         /* in [0,1] */
    if (u <= 0.0) u = 1.0 / (double)UINT64_MAX;    /* push into (0,1) */
    if (u >= 1.0) u = 1.0 - 1.0 / (double)UINT64_MAX;
    return u;
}

double eif_rand_normal(void) {
    /* Box-Muller */
    double u1 = eif_rand_uniform();
    double u2 = eif_rand_uniform();
    double r  = sqrt(-2.0 * log(u1));
    double th = 2.0 * M_PI * u2;
    return r * cos(th); /* mean 0, std 1 */
}

/* ---------- MATLAB helpers ---------- */

const mxArray* eif_get_field(const mxArray* S, const char* name) {
    if (!S || !mxIsStruct(S)) return NULL;
    return mxGetField(S, 0, name);
}

const mxArray* eif_require_field(const mxArray* S, const char* name) {
    REQUIRE(S && mxIsStruct(S), "Params must be a struct");
    const mxArray* f = mxGetField(S, 0, name);
    if (!f) {
        mexErrMsgIdAndTxt("EIF:MissingField", "Missing field '%s' in params struct", name);
    }
    return f;
}

double eif_get_scalar_double(const mxArray* A, const char* what) {
    REQUIRE(A, what);
    eif_require_type(A, mxDOUBLE_CLASS, what);
    FAIL_IF(mxIsComplex(A), "Complex values are not supported");
    FAIL_IF(mxGetNumberOfElements(A) != 1, "Expected a scalar");
    return mxGetScalar(A);
}

int32_t eif_get_scalar_int32(const mxArray* A, const char* what) {
    REQUIRE(A, what);
    if (mxIsInt32(A) && mxGetNumberOfElements(A) == 1) {
        return *(int32_t*)mxGetData(A);
    }
    /* Allow double scalar that is an integer */
    double v = eif_get_scalar_double(A, what);
    double r = floor(v + 0.5);
    FAIL_IF(fabs(v - r) > 0.0, "Expected an integer-valued scalar");
    FAIL_IF(r < (double)INT32_MIN || r > (double)INT32_MAX, "Integer out of int32 range");
    return (int32_t)r;
}

double eif_get_field_double_req(const mxArray* S, const char* name) {
    const mxArray* f = eif_require_field(S, name);
    return eif_get_scalar_double(f, name);
}

double eif_get_field_double_opt(const mxArray* S, const char* name, double defv) {
    const mxArray* f = eif_get_field(S, name);
    if (!f) return defv;
    return eif_get_scalar_double(f, name);
}

int32_t eif_get_field_int32_req(const mxArray* S, const char* name) {
    const mxArray* f = eif_require_field(S, name);
    return eif_get_scalar_int32(f, name);
}

int32_t eif_get_field_int32_opt(const mxArray* S, const char* name, int32_t defv) {
    const mxArray* f = eif_get_field(S, name);
    if (!f) return defv;
    return eif_get_scalar_int32(f, name);
}

void eif_require_type(const mxArray* A, mxClassID cls, const char* what) {
    REQUIRE(A, what);
    if (mxGetClassID(A) != cls) {
        mexErrMsgIdAndTxt("EIF:Type", "Expected %s to be of type %d", what, (int)cls);
    }
}

void eif_require_vector(const mxArray* A, const char* what) {
    REQUIRE(A, what);
    FAIL_IF(mxGetNumberOfDimensions(A) != 2, "Expected 2D array");
    mwSize m = mxGetM(A), n = mxGetN(A);
    if (!(m == 1 || n == 1)) {
        mexErrMsgIdAndTxt("EIF:Shape", "Expected %s to be a vector", what);
    }
}

void eif_require_matrix(const mxArray* A, const char* what) {
    REQUIRE(A, what);
    FAIL_IF(mxGetNumberOfDimensions(A) != 2, "Expected 2D array");
}

void eif_require_length_at_least(const mxArray* A, mwSize nmin, const char* what) {
    REQUIRE(A, what);
    mwSize nel = mxGetNumberOfElements(A);
    if (nel < nmin) {
        mexErrMsgIdAndTxt("EIF:Length", "Expected %s length >= %zu", what, (size_t)nmin);
    }
}

void eif_require_size(const mxArray* A, mwSize m, mwSize n, const char* what) {
    REQUIRE(A, what);
    if (mxGetM(A) != m || mxGetN(A) != n) {
        mexErrMsgIdAndTxt("EIF:Size", "Expected %s of size %zu x %zu",
                          what, (size_t)m, (size_t)n);
    }
}

int32_t* eif_cast_to_int32_copy(const mxArray* A, const char* what) {
    REQUIRE(A, what);
    mwSize nel = mxGetNumberOfElements(A);
    int32_t* out = NULL;
    SAFE_MALLOC(out, nel, int32_t);

    if (mxIsInt32(A)) {
        memcpy(out, mxGetData(A), nel * sizeof(int32_t));
        return out;
    }
    /* Accept double array with integer values (MATLAB default) */
    eif_require_type(A, mxDOUBLE_CLASS, what);
    double* p = (double*)mxGetData(A);
    for (mwSize i = 0; i < nel; ++i) {
        double v = p[i];
        double r = floor(v + 0.5);
        if (fabs(v - r) > 0.0) {
            SAFE_FREE(out);
            mexErrMsgIdAndTxt("EIF:Cast", "Non-integer value in %s at index %zu", what, (size_t)i+1);
        }
        if (r < (double)INT32_MIN || r > (double)INT32_MAX) {
            SAFE_FREE(out);
            mexErrMsgIdAndTxt("EIF:Cast", "Value out of int32 range in %s at index %zu", what, (size_t)i+1);
        }
        out[i] = (int32_t)r;
    }
    return out;
}

/* ---------- Parsing & validation ---------- */

void parse_params(const mxArray* S, NetworkParams* P) {
    REQUIRE(P, "NetworkParams output");
    memset(P, 0, sizeof(*P));

    REQUIRE(S && mxIsStruct(S), "params (struct)");

    /* Required: T and dt */
    P->T  = eif_get_field_double_req(S, "T");
    P->dt = eif_get_field_double_req(S, "dt");

    FAIL_IF(!(P->dt > 0.0 && P->dt >= EIF_MIN_DT && P->dt <= EIF_MAX_DT), "Invalid dt");
    FAIL_IF(!(P->T  > 0.0 && P->T  <= EIF_MAX_T), "Invalid T");

    /* Optional: sizes; if not present, caller may set later. */
    P->Ne = eif_get_field_int32_opt(S, "Ne", 0);
    P->Ni = eif_get_field_int32_opt(S, "Ni", 0);
    P->Nx = eif_get_field_int32_opt(S, "Nx", 0);

    int64_t Nsum = (int64_t)P->Ne + (int64_t)P->Ni;
    FAIL_IF(Nsum < 0 || Nsum > INT32_MAX, "Population sizes overflow");
    P->N = (int32_t)Nsum;

    /* Optional buffer size and flags */
    P->maxns  = eif_get_field_int32_opt(S, "maxns", 0);
    P->attarea = eif_get_field_int32_opt(S, "attarea", 0);

    DBG("Parsed params: T=%.6g dt=%.6g Ne=%d Ni=%d Nx=%d N=%d maxns=%d attarea=%d\n",
        P->T, P->dt, P->Ne, P->Ni, P->Nx, P->N, P->maxns, P->attarea);
}

void validate_time_vector(const double* t, mwSize nt, double T, const char* what) {
    if (nt == 0) return;
    double prev = t[0];
    if (!(prev >= 0.0 && prev <= T)) {
        mexErrMsgIdAndTxt("EIF:Time", "%s out of range [0,T] at idx 1", what);
    }
    for (mwSize i = 1; i < nt; ++i) {
        double cur = t[i];
        if (cur < prev) {
            mexErrMsgIdAndTxt("EIF:Time", "%s not non-decreasing at idx %zu", what, (size_t)i+1);
        }
        if (!(cur >= 0.0 && cur <= T)) {
            mexErrMsgIdAndTxt("EIF:Time", "%s out of range [0,T] at idx %zu", what, (size_t)i+1);
        }
        prev = cur;
    }
}

void validate_indices_1based(const int32_t* idx, mwSize n, int32_t Nmax, const char* what) {
    for (mwSize i = 0; i < n; ++i) {
        int32_t v = idx[i];
        if (v < 1 || v > Nmax) {
            mexErrMsgIdAndTxt("EIF:Index", "%s out of bounds at idx %zu: %d not in [1,%d]",
                              what, (size_t)i+1, v, Nmax);
        }
    }
}

void validate_edges_1based(const int32_t* pre, const int32_t* post, mwSize nEdges,
                           int32_t Npre, int32_t Npost, const char* what) {
    for (mwSize i = 0; i < nEdges; ++i) {
        int32_t u = pre[i];
        int32_t v = post[i];
        if (u < 1 || u > Npre) {
            mexErrMsgIdAndTxt("EIF:Edge", "%s: pre index out of bounds at %zu: %d not in [1,%d]",
                              what, (size_t)i+1, u, Npre);
        }
        if (v < 1 || v > Npost) {
            mexErrMsgIdAndTxt("EIF:Edge", "%s: post index out of bounds at %zu: %d not in [1,%d]",
                              what, (size_t)i+1, v, Npost);
        }
    }
}
