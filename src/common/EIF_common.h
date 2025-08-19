
// EIF_common.h
// Shared utilities for Exponential Integrate-and-Fire (EIF) MEX simulations
// Style aligned with EIF_normalization_default.c
#ifndef EIF_COMMON_H
#define EIF_COMMON_H

#include "mex.h"
#include "matrix.h"
#include <stdint.h>
#include <math.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef DEBUG
#define DEBUG 0
#endif

#if DEBUG
#define DEBUG_PRINT(fmt, ...) mexPrintf("[DEBUG] " fmt "\n", ##__VA_ARGS__)
#else
#define DEBUG_PRINT(fmt, ...)
#endif

#define ERROR_CHECK(cond, msg) do { if(!(cond)) mexErrMsgTxt("ERROR: " msg); } while(0)

typedef enum {
    EXCITATORY = 0,
    INHIBITORY = 1,
    NUM_POPULATIONS = 2
} PopulationType;

typedef enum {
    SYNAPSE_FEEDFORWARD = 0,
    SYNAPSE_EXCITATORY = 1,
    SYNAPSE_INHIBITORY = 2,
    NUM_SYNAPSE_TYPES = 3
} SynapseType;

typedef struct {
    // Counts
    int32_t Ne, Ni, Nx;
    int32_t Ne1, Ni1, Nx1;

    // Base strengths (used when not using broad weights)
    double Jex, Jix, Jex1, Jix1;
    double Jee, Jie, Jei, Jii;

    // Fanouts
    int32_t Kex, Kix;
    int32_t Kee, Kie, Kei, Kii;

    // Neuron params
    double C[NUM_POPULATIONS];
    double gl[NUM_POPULATIONS];
    double Vleak[NUM_POPULATIONS];
    double DeltaT[NUM_POPULATIONS];
    double VT[NUM_POPULATIONS];
    double tref[NUM_POPULATIONS];
    double Vth[NUM_POPULATIONS];
    double Vre[NUM_POPULATIONS];
    double Vlb[NUM_POPULATIONS];

    // Sim params
    double T, dt;
    int32_t maxns;
    int32_t attarea;

    // Optional extras
    // Noise (Box-Muller) with sigma per population common scalar (original uses one sigma)
    double sigma_current;
    int use_current_noise; // 1 if using per-timestep noise

    // Gaussian input current (baseline + time series)
    double me_current, mi_current;
    const double* L_current; // length Nt1 (user provides); used as L_current[floor(i*dt)]
    int use_gaussian_input; // 1 if using me/mi + L_current

    // Broad weights (per-connection)
    const double* Jrf; // length == num entries of Wrf
    const double* Jrr; // length == num entries of Wrr
    int use_broad_weights; // 1 if using Jrf/Jrr arrays
} NetworkParams;

typedef struct {
    int32_t Nsyntype;
    const double* taursyn; // 3 x Nsyntype
    const double* taudsyn; // 3 x Nsyntype
    const double* Psyn;    // 3 x Nsyntype

    // Active synapses (Psyn > 0) flattened
    int32_t Nsyn;
    int32_t* syntype;  // length <= 3*Nsyntype; entries in {0,1,2}
    double* Psyn2;     // same length
    double* temp1;     // same length: 1/tau_d + 1/tau_r
    double* temp2;     // same length: 1/(tau_d * tau_r)
} SynapseParams;

// Memory helpers
void* eif_safe_malloc(size_t sz, const char* ctx);

// Extractors (fill defaults if fields missing)
void eif_extract_network_params(const mxArray* params_struct, NetworkParams* p);
void eif_extract_synapse_params(const mxArray* params_struct, SynapseParams* sp);
void eif_free_synapse_params(SynapseParams* sp);

// Attention
void eif_init_attention(const NetworkParams* p, double* attyne, double* attyni);

// Validation
void eif_validate_spike_data(const double* sx, int32_t Nsx, int32_t Nx);
void eif_validate_connectivity(const int32_t* W, int32_t Nw, int32_t N);

// Core simulation (variant-agnostic).
// - Wrf and Wrr are 1-based index arrays from MATLAB, column vectors.
// - Output arrays must be pre-allocated by the caller: s(2 x maxns), Isynrecord(Nrec*rec_cols x Nt), vr(Nrec x Nt)
// - rec_cols is Nsyn (or Nsyn+1 for noise variants); if rec_cols > Nsyn, extra columns will be left as zeros by core.
void eif_simulate_core(
    // inputs
    const double* sx, int32_t Nsx,
    const int32_t* Wrf, int32_t Nwrf,
    const int32_t* Wrr, int32_t Nwrr,
    const NetworkParams* P,
    const SynapseParams* S,
    const double* V0,
    const double* Irecord, int32_t Nrecord,
    int32_t rec_cols,
    // outputs
    double* spikes_out,          // 2 x P->maxns
    double* Isynrecord_out,      // Nrecord*rec_cols x Nt
    double* Vrecord_out          // Nrecord x Nt
);

#ifdef __cplusplus
}
#endif

#endif // EIF_COMMON_H
