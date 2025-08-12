//
//  EIF_normalization_default.c
//  Exponential Integrate-and-Fire Network Simulation for normalization and selective spatial attention
//
//  Created by Deying Song on 2022/6/29.
//  Revised for improved robustness and clarity
//

/*
 * MATLAB Usage Instructions:
 * 
 * To compile this MEX file:
 * >> mex -DDEBUG=0 EIF_normalization_default.c
 * 
 * For debug mode:
 * >> mex -DDEBUG=1 EIF_normalization_default.c
 * 
 * To call the function:
 * >> [spikes, Isyn_recorded, V_recorded] = EIF_normalization_default(sx, Wrf, Wrr, params);
 * 
 * Input Parameters:
 * - sx: 2×Nsx matrix of feedforward spikes [spike_times; neuron_indices]
 * - Wrf: Feedforward connectivity matrix (column vector)
 * - Wrr: Recurrent connectivity matrix (column vector) 
 * - params: Structure with network parameters (see NetworkParams struct)
 * 
 * Output Parameters:
 * - spikes: 2×maxns matrix [spike_times; neuron_indices]
 * - Isyn_recorded: Recorded synaptic currents for specified neurons
 * - V_recorded: Recorded membrane potentials for specified neurons
 * 
 */


#include "mex.h"
#include "math.h"
#include "time.h"
#include "matrix.h"
#include <stdint.h>
#include <string.h>

// Constants and configuration
#define DEBUG 0  // Set to 1 for debug mode
#define MAX_NEURONS 100000
#define MAX_SPIKES_DEFAULT 1000000
#define MIN_DT 1e-6
#define MAX_DT 1.0
#define MIN_SIMULATION_TIME 1.0
#define MAX_SIMULATION_TIME 5e5

// Neuron population types
typedef enum {
    EXCITATORY = 0,
    INHIBITORY = 1,
    NUM_POPULATIONS = 2
} PopulationType;

// Synapse types
typedef enum {
    SYNAPSE_FEEDFORWARD = 0,
    SYNAPSE_EXCITATORY = 1,
    SYNAPSE_INHIBITORY = 2,
    NUM_SYNAPSE_TYPES = 3
} SynapseType;

// Attention areas
typedef enum {
    ATTENTION_AREA_1 = 1,
    ATTENTION_AREA_2 = 2
} AttentionArea;

// Network parameters structure
typedef struct {
    int32_t Ne, Ni, Nx;           // Number of neurons
    int32_t Ne1, Ni1, Nx1;        // Grid dimensions
    double Jex, Jix, Jex1, Jix1;  // Connection strengths
    double Jee, Jei, Jie, Jii;    // Recurrent connections
    int32_t Kex, Kix;             // Feedforward connections
    int32_t Kee, Kie, Kei, Kii;   // Recurrent connections
    double C[NUM_POPULATIONS];     // Membrane capacitance
    double gl[NUM_POPULATIONS];    // Leak conductance
    double Vleak[NUM_POPULATIONS]; // Leak potential
    double DeltaT[NUM_POPULATIONS];// EIF slope factor
    double VT[NUM_POPULATIONS];    // EIF threshold
    double tref[NUM_POPULATIONS];  // Refractory period
    double Vth[NUM_POPULATIONS];   // Spike threshold
    double Vre[NUM_POPULATIONS];   // Reset potential
    double Vlb[NUM_POPULATIONS];   // Lower bound potential
    double T;                      // Simulation time
    double dt;                     // Time step
    int32_t maxns;                 // Maximum spikes
    int32_t attarea;               // Attention area
} NetworkParams;

// Synapse parameters structure
typedef struct {
    int32_t Nsyntype;
    double *taursyn;    // Rise time constants
    double *taudsyn;    // Decay time constants
    double *Psyn;       // Synapse percentages
    int32_t *syntype;   // Synapse types
    double *Psyn2;      // Active synapse percentages
    double *temp1, *temp2; // Precomputed constants
    int32_t Nsyn;       // Number of active synapses
} SynapseParams;

// Debug and logging macros
#if DEBUG
#define DEBUG_PRINT(fmt, ...) mexPrintf("[DEBUG] " fmt "\n", ##__VA_ARGS__)
#else
#define DEBUG_PRINT(fmt, ...)
#endif

#define ERROR_CHECK(condition, message) \
    do { \
        if (!(condition)) { \
            mexErrMsgTxt("ERROR: " message); \
        } \
    } while(0)

#define WARN_PRINT(fmt, ...) mexWarnMsgTxt(fmt)

// Fast exponential approximation (platform-independent)
static inline double fast_exp(double x) {
    // Clamp input to prevent overflow
    if (x > 700.0) return INFINITY;
    if (x < -700.0) return 0.0;
    return exp(x); // Use standard exp for better accuracy and portability
}

// Validation functions
static void validate_spike_data(const double *sx, int32_t Nsx, int32_t Nx) {
    ERROR_CHECK(sx != NULL, "Spike data is NULL");
    ERROR_CHECK(Nsx >= 0, "Number of spikes cannot be negative");
    
    for (int32_t i = 0; i < Nsx; i++) {
        ERROR_CHECK(sx[2*i] >= 0, "Spike times must be non-negative");
        ERROR_CHECK(sx[2*i + 1] >= 1 && sx[2*i + 1] <= Nx, 
                   "Spike neuron indices out of bounds");
    }
}

static void validate_connectivity(const int32_t *W, int32_t Nw, int32_t N) {
    ERROR_CHECK(W != NULL, "Connectivity matrix is NULL");
    ERROR_CHECK(Nw >= 0, "Number of connections cannot be negative");
    
    for (int32_t i = 0; i < Nw; i++) {
        ERROR_CHECK(W[i] >= 1 && W[i] <= N, 
                   "Connection indices out of bounds");
    }
}

static void validate_network_params(const NetworkParams *params) {
    ERROR_CHECK(params != NULL, "Network parameters are NULL");
    ERROR_CHECK(params->Ne > 0 && params->Ne <= MAX_NEURONS, 
               "Invalid number of excitatory neurons");
    ERROR_CHECK(params->Ni > 0 && params->Ni <= MAX_NEURONS, 
               "Invalid number of inhibitory neurons");
    ERROR_CHECK(params->Nx > 0 && params->Nx <= MAX_NEURONS, 
               "Invalid number of feedforward neurons");
    ERROR_CHECK(params->dt >= MIN_DT && params->dt <= MAX_DT, 
               "Invalid time step");
    ERROR_CHECK(params->T >= MIN_SIMULATION_TIME && params->T <= MAX_SIMULATION_TIME, 
               "Invalid simulation time");
    ERROR_CHECK(params->maxns > 0, "Maximum spikes must be positive");
    
    // Validate neuron parameters
    for (int32_t i = 0; i < NUM_POPULATIONS; i++) {
        ERROR_CHECK(params->C[i] > 0, "Capacitance must be positive");
        ERROR_CHECK(params->gl[i] > 0, "Leak conductance must be positive");
        ERROR_CHECK(params->tref[i] >= 0, "Refractory period cannot be negative");
        ERROR_CHECK(params->DeltaT[i] > 0, "Delta T must be positive");
    }
}

// Memory allocation with error checking
static void* safe_malloc(size_t size, const char *context) {
    void *ptr = mxMalloc(size);
    if (ptr == NULL) {
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg), 
                "Memory allocation failed for %s (size: %zu bytes)", 
                context, size);
        mexErrMsgTxt(error_msg);
    }
    memset(ptr, 0, size); // Initialize to zero
    return ptr;
}

// Extract and validate parameters from MATLAB struct
static void extract_network_params(const mxArray *param_struct, NetworkParams *params) {
    const mxArray *field;
    
    // Extract integer parameters
    field = mxGetField(param_struct, 0, "Ne");
    ERROR_CHECK(field != NULL, "Missing field: Ne");
    params->Ne = (int32_t)mxGetScalar(field);
    params->Ne1 = (int32_t)sqrt(params->Ne / 2);
    
    field = mxGetField(param_struct, 0, "Ni");
    ERROR_CHECK(field != NULL, "Missing field: Ni");
    params->Ni = (int32_t)mxGetScalar(field);
    params->Ni1 = (int32_t)sqrt(params->Ni / 2);
    
    field = mxGetField(param_struct, 0, "Nx");
    ERROR_CHECK(field != NULL, "Missing field: Nx");
    params->Nx = (int32_t)mxGetScalar(field);
    params->Nx1 = (int32_t)sqrt(params->Nx / 2);
    
    // Extract connection strengths
    field = mxGetField(param_struct, 0, "Jx");
    ERROR_CHECK(field != NULL && mxGetNumberOfElements(field) >= 4, 
               "Invalid Jx field");
    double *Jx = mxGetPr(field);
    params->Jex = Jx[0];
    params->Jix = Jx[1];
    params->Jex1 = Jx[2];
    params->Jix1 = Jx[3];
    
    field = mxGetField(param_struct, 0, "Jr");
    ERROR_CHECK(field != NULL && mxGetNumberOfElements(field) >= 4, 
               "Invalid Jr field");
    double *Jr = mxGetPr(field);
    params->Jee = Jr[0];
    params->Jie = Jr[1];
    params->Jei = Jr[2];
    params->Jii = Jr[3];
    
    // Extract connectivity parameters
    field = mxGetField(param_struct, 0, "Kx");
    ERROR_CHECK(field != NULL && mxGetNumberOfElements(field) >= 2, 
               "Invalid Kx field");
    double *Kx = mxGetPr(field);
    params->Kex = (int32_t)Kx[0];
    params->Kix = (int32_t)Kx[1];
    
    field = mxGetField(param_struct, 0, "Kr");
    ERROR_CHECK(field != NULL && mxGetNumberOfElements(field) >= 4, 
               "Invalid Kr field");
    double *Kr = mxGetPr(field);
    params->Kee = (int32_t)Kr[0];
    params->Kie = (int32_t)Kr[1];
    params->Kei = (int32_t)Kr[2];
    params->Kii = (int32_t)Kr[3];
    
    // Extract neuron parameters (2-element arrays)
    const char* param_names[] = {"Cm", "gl", "vl", "DeltaT", "vT", "tref", "vth", "vre", "vlb"};
    double* param_arrays[] = {params->C, params->gl, params->Vleak, params->DeltaT, 
                             params->VT, params->tref, params->Vth, params->Vre, params->Vlb};
    
    for (int32_t i = 0; i < 9; i++) {
        field = mxGetField(param_struct, 0, param_names[i]);
        ERROR_CHECK(field != NULL && mxGetNumberOfElements(field) == 2, 
                   "Invalid neuron parameter");
        double *values = mxGetPr(field);
        param_arrays[i][0] = values[0];
        param_arrays[i][1] = values[1];
    }
    
    // Extract simulation parameters
    field = mxGetField(param_struct, 0, "T");
    ERROR_CHECK(field != NULL, "Missing field: T");
    params->T = mxGetScalar(field);
    
    field = mxGetField(param_struct, 0, "dt");
    ERROR_CHECK(field != NULL, "Missing field: dt");
    params->dt = mxGetScalar(field);
    
    field = mxGetField(param_struct, 0, "maxns");
    ERROR_CHECK(field != NULL, "Missing field: maxns");
    params->maxns = (int32_t)mxGetScalar(field);
    
    field = mxGetField(param_struct, 0, "attarea");
    ERROR_CHECK(field != NULL, "Missing field: attarea");
    params->attarea = (int32_t)mxGetScalar(field);
}

// Extract synapse parameters
static void extract_synapse_params(const mxArray *param_struct, SynapseParams *syn_params) {
    const mxArray *field;
    
    // Extract synapse time constants
    field = mxGetField(param_struct, 0, "taursyn");
    ERROR_CHECK(field != NULL && mxGetM(field) == 3, "Invalid taursyn field");
    syn_params->Nsyntype = (int32_t)mxGetN(field);
    syn_params->taursyn = mxGetPr(field);
    
    field = mxGetField(param_struct, 0, "taudsyn");
    ERROR_CHECK(field != NULL && mxGetM(field) == 3 && 
               mxGetN(field) == syn_params->Nsyntype, "Invalid taudsyn field");
    syn_params->taudsyn = mxGetPr(field);
    
    // Extract synapse percentages
    field = mxGetField(param_struct, 0, "Psyn");
    ERROR_CHECK(field != NULL && mxGetM(field) == 3 && 
               mxGetN(field) == syn_params->Nsyntype, "Invalid Psyn field");
    syn_params->Psyn = mxGetPr(field);
    
    // Allocate temporary arrays
    int32_t total_synapses = syn_params->Nsyntype * 3;
    syn_params->syntype = (int32_t*)safe_malloc(total_synapses * sizeof(int32_t), "syntype");
    syn_params->Psyn2 = (double*)safe_malloc(total_synapses * sizeof(double), "Psyn2");
    syn_params->temp1 = (double*)safe_malloc(total_synapses * sizeof(double), "temp1");
    syn_params->temp2 = (double*)safe_malloc(total_synapses * sizeof(double), "temp2");
    
    // Process active synapses
    syn_params->Nsyn = 0;
    for (int32_t isyn = 0; isyn < total_synapses; isyn++) {
        if (syn_params->Psyn[isyn] > 0) {
            syn_params->syntype[syn_params->Nsyn] = isyn % 3;
            syn_params->Psyn2[syn_params->Nsyn] = syn_params->Psyn[isyn];
            
            double tau_r = syn_params->taursyn[isyn];
            double tau_d = syn_params->taudsyn[isyn];
            ERROR_CHECK(tau_r > 0 && tau_d > 0, "Synapse time constants must be positive");
            
            syn_params->temp1[syn_params->Nsyn] = 1.0/tau_d + 1.0/tau_r;
            syn_params->temp2[syn_params->Nsyn] = 1.0/(tau_d * tau_r);
            syn_params->Nsyn++;
        }
    }
    
    DEBUG_PRINT("Active synapses: %d", syn_params->Nsyn);
}

// Initialize attention weights
static void initialize_attention_weights(const NetworkParams *params, 
                                       double *attyne, double *attyni) {
    int32_t Nx1 = params->Nx1;
    double att_factor_e = (params->Jex1 != 0) ? params->Jex1 / params->Jex : 1.0;
    double att_factor_i = (params->Jix1 != 0) ? params->Jix1 / params->Jix : 1.0;
    
    for (int32_t jj = 0; jj < Nx1; jj++) {
        for (int32_t kk = 0; kk < Nx1 * 2; kk++) {
            int32_t idx = jj * Nx1 * 2 + kk;
            
            if (params->attarea == ATTENTION_AREA_1) {
                // First half gets attention
                if (kk < Nx1) {
                    attyne[idx] = att_factor_e;
                    attyni[idx] = att_factor_i;
                } else {
                    attyne[idx] = 1.0;
                    attyni[idx] = 1.0;
                }
            } else {
                // Second half gets attention
                if (kk >= Nx1) {
                    attyne[idx] = att_factor_e;
                    attyni[idx] = att_factor_i;
                } else {
                    attyne[idx] = 1.0;
                    attyni[idx] = 1.0;
                }
            }
        }
    }
}

// Main MEX function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    // Validate inputs
    ERROR_CHECK(nrhs == 4, "Exactly 4 inputs required: sx, Wrf, Wrr, params");
    ERROR_CHECK(nlhs <= 3, "Maximum 3 outputs supported");
    
    // Extract and validate spike data
    ERROR_CHECK(mxGetM(prhs[0]) == 2, "sx must be 2xNsx matrix");
    double *sx = mxGetPr(prhs[0]);
    int32_t Nsx = (int32_t)mxGetN(prhs[0]);
    
    // Extract connectivity matrices
    ERROR_CHECK(mxGetN(prhs[1]) == 1, "Wrf must be a column vector");
    int32_t *Wrf = (int32_t*)mxGetData(prhs[1]);
    int32_t Nwrf = (int32_t)mxGetM(prhs[1]);
    
    ERROR_CHECK(mxGetN(prhs[2]) == 1, "Wrr must be a column vector");
    int32_t *Wrr = (int32_t*)mxGetData(prhs[2]);
    int32_t Nwrr = (int32_t)mxGetM(prhs[2]);
    
    ERROR_CHECK(mxIsStruct(prhs[3]), "Fourth input must be parameter struct");
    
    // Extract parameters
    NetworkParams params = {0};
    extract_network_params(prhs[3], &params);
    validate_network_params(&params);
    
    int32_t N = params.Ne + params.Ni;
    validate_spike_data(sx, Nsx, params.Nx);
    validate_connectivity(Wrf, Nwrf, N);
    validate_connectivity(Wrr, Nwrr, N);
    
    // Extract synapse parameters
    SynapseParams syn_params = {0};
    extract_synapse_params(prhs[3], &syn_params);
    
    // Extract recording parameters
    const mxArray *field = mxGetField(prhs[3], 0, "Irecord");
    ERROR_CHECK(field != NULL && mxGetM(field) == 1, "Invalid Irecord field");
    double *Irecord = mxGetPr(field);
    int32_t Nrecord = (int32_t)mxGetN(field);
    
    // Validate recording indices
    for (int32_t i = 0; i < Nrecord; i++) {
        ERROR_CHECK(Irecord[i] >= 1 && Irecord[i] <= N, 
                   "Recording indices out of bounds");
    }
    
    field = mxGetField(prhs[3], 0, "V0");
    ERROR_CHECK(field != NULL && mxGetNumberOfElements(field) == N, 
               "Invalid V0 field");
    double *V0 = mxGetPr(field);
    
    // Simulation parameters
    int32_t Nt = (int32_t)(params.T / params.dt);
    int32_t Ntref[NUM_POPULATIONS] = {
        (int32_t)round(params.tref[0] / params.dt),
        (int32_t)round(params.tref[1] / params.dt)
    };
    
    DEBUG_PRINT("Simulation: N=%d, Nt=%d, Nsyn=%d", N, Nt, syn_params.Nsyn);
    
    // Allocate output arrays
    plhs[0] = mxCreateDoubleMatrix(2, params.maxns, mxREAL);
    double *s = mxGetPr(plhs[0]);
    
    plhs[1] = mxCreateDoubleMatrix(Nrecord * syn_params.Nsyn, Nt, mxREAL);
    double *Isynrecord = mxGetPr(plhs[1]);
    
    plhs[2] = mxCreateDoubleMatrix(Nrecord, Nt, mxREAL);
    double *vr = mxGetPr(plhs[2]);
    
    // Allocate simulation arrays
    double *v = (double*)safe_malloc(N * sizeof(double), "membrane potential");
    int32_t *refstate = (int32_t*)safe_malloc(N * sizeof(int32_t), "refractory state");
    double *Isyn = (double*)safe_malloc(syn_params.Nsyn * N * sizeof(double), "synaptic currents");
    double *Isynprime = (double*)safe_malloc(syn_params.Nsyn * N * sizeof(double), "synaptic derivatives");
    
    // Connectivity arrays
    int32_t Kx = params.Kex + params.Kix;
    int32_t Ke = params.Kee + params.Kie;
    int32_t Ki = params.Kei + params.Kii;
    
    int32_t *postcellX = (int32_t*)safe_malloc(Kx * sizeof(int32_t), "feedforward targets");
    int32_t *postcellE = (int32_t*)safe_malloc(Ke * sizeof(int32_t), "excitatory targets");
    int32_t *postcellI = (int32_t*)safe_malloc(Ki * sizeof(int32_t), "inhibitory targets");
    
    // Attention weights
    double *attyne = (double*)safe_malloc(params.Nx * sizeof(double), "attention weights E");
    double *attyni = (double*)safe_malloc(params.Nx * sizeof(double), "attention weights I");
    initialize_attention_weights(&params, attyne, attyni);
    
    // Initialize simulation state
    memcpy(v, V0, N * sizeof(double));
    
    // Record initial state
    for (int32_t jj = 0; jj < Nrecord; jj++) {
        int32_t neuron_idx = (int32_t)round(Irecord[jj]) - 1;
        for (int32_t isyn = 0; isyn < syn_params.Nsyn; isyn++) {
            Isynrecord[isyn * Nrecord + jj] = Isyn[neuron_idx * syn_params.Nsyn + isyn];
        }
        vr[jj] = v[neuron_idx];
    }
    
    // Main simulation loop
    int32_t ns = 0;  // Number of spikes
    int32_t iXspike = 0;  // Current feedforward spike index
    double *iXspkInd = &sx[0];  // Pointer to current spike
    
    for (int32_t i = 1; i < Nt && ns < params.maxns; i++) {
        double current_time = i * params.dt;
        
        // Update synaptic variables
        for (int32_t jj = 0; jj < N * syn_params.Nsyn; jj++) {
            int32_t isyn = jj % syn_params.Nsyn;
            Isyn[jj] += Isynprime[jj] * params.dt;
            Isynprime[jj] -= params.dt * (Isynprime[jj] * syn_params.temp1[isyn] + 
                                         Isyn[jj] * syn_params.temp2[isyn]);
        }
        
        // Process feedforward spikes
        while (iXspike < Nsx && *iXspkInd <= current_time) {
            iXspkInd++;  // Move to neuron index
            int32_t jspike = (int32_t)round(*iXspkInd) - 1;
            
            ERROR_CHECK(jspike >= 0 && jspike < params.Nx, "Feedforward spike index out of bounds");
            
            // Get postsynaptic targets
            int32_t *Pr = &Wrf[jspike * Kx];
            for (int32_t k = 0; k < Kx; k++) {
                postcellX[k] = Pr[k] - 1;
                ERROR_CHECK(postcellX[k] >= 0 && postcellX[k] < N, 
                           "Feedforward target out of bounds");
            }
            
            // Update synaptic inputs
            for (int32_t isyn = 0; isyn < syn_params.Nsyn; isyn++) {
                if (syn_params.syntype[isyn] == SYNAPSE_FEEDFORWARD) {
                    for (int32_t k = 0; k < Kx; k++) {
                        double strength = (postcellX[k] < params.Ne) ? 
                            params.Jex * attyne[jspike] : params.Jix * attyni[jspike];
                        Isynprime[postcellX[k] * syn_params.Nsyn + isyn] += 
                            strength * syn_params.temp2[isyn];
                    }
                }
            }
            
            iXspike++;
            iXspkInd++;
        }
        
        // Update neurons
        int32_t *Pr = &Wrr[0];
        for (int32_t j = 0; j < N; j++) {
            PopulationType pop_type = (j < params.Ne) ? EXCITATORY : INHIBITORY;
            
            // Update membrane potential
            if (refstate[j] <= 0) {
                // Calculate total synaptic input
                double Isyntot = 0.0;
                for (int32_t isyn = 0; isyn < syn_params.Nsyn; isyn++) {
                    Isyntot += syn_params.Psyn2[isyn] * Isyn[j * syn_params.Nsyn + isyn];
                }
                
                // EIF equation
                double exp_term = fast_exp((v[j] - params.VT[pop_type]) / params.DeltaT[pop_type]);
                double dv = (Isyntot - params.gl[pop_type] * (v[j] - params.Vleak[pop_type]) +
                           params.gl[pop_type] * params.DeltaT[pop_type] * exp_term) * 
                           params.dt / params.C[pop_type];
                
                v[j] += fmax(dv, params.Vlb[pop_type] - v[j]);
            } else {
                // In refractory period
                v[j] = (refstate[j] > 1) ? params.Vth[pop_type] : params.Vre[pop_type];
                refstate[j]--;
            }
            
            // Check for spike
            if (v[j] >= params.Vth[pop_type] && refstate[j] <= 0 && ns < params.maxns) {
                refstate[j] = Ntref[pop_type];
                v[j] = params.Vth[pop_type];
                
                // Record spike
                s[2*ns] = current_time;
                s[2*ns + 1] = j + 1;
                ns++;
                
                // Update postsynaptic targets
                if (pop_type == EXCITATORY) {
                    for (int32_t k = 0; k < Ke; k++) {
                        postcellE[k] = Pr[k] - 1;
                        ERROR_CHECK(postcellE[k] >= 0 && postcellE[k] < N, 
                                   "Excitatory target out of bounds");
                    }
                    
                    for (int32_t isyn = 0; isyn < syn_params.Nsyn; isyn++) {
                        if (syn_params.syntype[isyn] == SYNAPSE_EXCITATORY) {
                            for (int32_t k = 0; k < Ke; k++) {
                                double strength = (postcellE[k] < params.Ne) ? params.Jee : params.Jie;
                                Isynprime[postcellE[k] * syn_params.Nsyn + isyn] += 
                                    strength * syn_params.temp2[isyn];
                            }
                        }
                    }
                    Pr += Ke;
                } else {
                    // Inhibitory neuron
                    int32_t *PrI = &Wrr[(j - params.Ne) * Ki + params.Ne * Ke];
                    for (int32_t k = 0; k < Ki; k++) {
                        postcellI[k] = PrI[k] - 1;
                        ERROR_CHECK(postcellI[k] >= 0 && postcellI[k] < N, 
                                   "Inhibitory target out of bounds");
                    }
                    
                    for (int32_t isyn = 0; isyn < syn_params.Nsyn; isyn++) {
                        if (syn_params.syntype[isyn] == SYNAPSE_INHIBITORY) {
                            for (int32_t k = 0; k < Ki; k++) {
                                double strength = (postcellI[k] < params.Ne) ? params.Jei : params.Jii;
                                Isynprime[postcellI[k] * syn_params.Nsyn + isyn] += 
                                    strength * syn_params.temp2[isyn];
                            }
                        }
                    }
                    Pr += Ki;
                }
            } else {
                // No spike, advance pointer
                if (pop_type == EXCITATORY) {
                    Pr += Ke;
                } else {
                    Pr += Ki;
                }
            }
        }
        
        // Record variables
        for (int32_t jj = 0; jj < Nrecord; jj++) {
            int32_t neuron_idx = (int32_t)round(Irecord[jj]) - 1;
            for (int32_t isyn = 0; isyn < syn_params.Nsyn; isyn++) {
                Isynrecord[isyn * Nrecord + jj + i * Nrecord * syn_params.Nsyn] = 
                    Isyn[neuron_idx * syn_params.Nsyn + isyn];
            }
            vr[jj + Nrecord * i] = v[neuron_idx];
        }
    }
    
    // Issue warning if maximum spikes reached
    if (ns >= params.maxns) {
        WARN_PRINT("Maximum number of spikes reached, simulation terminated.");
    }
    
    DEBUG_PRINT("Simulation completed: %d spikes generated", ns);
    
    // Clean up allocated memory
    mxFree(v);
    mxFree(refstate);
    mxFree(Isyn);
    mxFree(Isynprime);
    mxFree(postcellX);
    mxFree(postcellE);
    mxFree(postcellI);
    mxFree(attyne);
    mxFree(attyni);
    mxFree(syn_params.syntype);
    mxFree(syn_params.Psyn2);
    mxFree(syn_params.temp1);
    mxFree(syn_params.temp2);
}

