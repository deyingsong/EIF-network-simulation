
// EIF_common.c
#include "EIF_common.h"
#include <stdlib.h>
#include <time.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline double fast_exp(double x) {
    if (x > 700.0) return INFINITY;
    if (x < -700.0) return 0.0;
    return exp(x);
}

// --- helpers 
static const mxArray* get_field2(const mxArray* ps,
                                const char* n1, const char* n2) {
    const mxArray* f = mxGetField(ps, 0, n1);
    if (!f && n2) f = mxGetField(ps, 0, n2);
    return f;
}

void* eif_safe_malloc(size_t sz, const char* ctx) {
    void* p = mxMalloc(sz);
    if (!p) {
        char buf[256];
        snprintf(buf, sizeof(buf), "Memory allocation failed for %s (size=%zu)", ctx, sz);
        mexErrMsgTxt(buf);
    }
    memset(p, 0, sz);
    return p;
}

void eif_validate_spike_data(const double* sx, int32_t Nsx, int32_t Nx) {
    ERROR_CHECK(sx != NULL, "sx is NULL");
    ERROR_CHECK(Nsx >= 0, "Nsx negative");
    for (int32_t i = 0; i < Nsx; i++) {
        ERROR_CHECK(sx[2*i] >= 0.0, "Spike time negative");
        ERROR_CHECK(sx[2*i+1] >= 1 && sx[2*i+1] <= Nx, "Spike neuron index out of bounds");
    }
}

void eif_validate_connectivity(const int32_t* W, int32_t Nw, int32_t N) {
    ERROR_CHECK(W != NULL, "Connectivity vector NULL");
    ERROR_CHECK(Nw >= 0, "Connectivity length negative");
    for (int32_t i = 0; i < Nw; i++) {
        ERROR_CHECK(W[i] >= 1 && W[i] <= N, "Connectivity index out of bounds");
    }
}

void eif_extract_network_params(const mxArray* ps, NetworkParams* p) {
    memset(p, 0, sizeof(*p));
    const mxArray* f = NULL;

    #define GRAB_SCALAR(name) do{ \
        f=mxGetField(ps,0,#name); \
        ERROR_CHECK(f!=NULL, "Missing field: " #name); \
        p->name = mxGetScalar(f); \
    }while(0)


    #define GRAB_VEC2_STRICT(name, dst) do{ \
        f = mxGetField(ps,0,#name); \
        ERROR_CHECK(f != NULL, "Missing field: " #name); \
        ERROR_CHECK(mxGetNumberOfElements(f) == 2, "Field " #name " must have 2 elements"); \
        double* v = mxGetPr(f); (dst)[0] = v[0]; (dst)[1] = v[1]; \
    }while(0)

    // --- alias-friendly versions (accept legacy and new names) ---
    #define GRAB_SCALAR2(n1,n2, lvalue, label) do{ \
        f = get_field2(ps,(n1),(n2)); \
        ERROR_CHECK(f != NULL, "Missing field: " label); \
        lvalue = mxGetScalar(f); \
    }while(0)

    #define GRAB_VEC2_2(n1,n2, dst, label) do{ \
        f = get_field2(ps,(n1),(n2)); \
        ERROR_CHECK(f != NULL, "Missing field: " label); \
        ERROR_CHECK(mxGetNumberOfElements(f) == 2, "Field " label " must have 2 elements"); \
        double* v = mxGetPr(f); (dst)[0] = v[0]; (dst)[1] = v[1]; \
    }while(0)


    // Counts
    GRAB_SCALAR(Ne);
    GRAB_SCALAR(Ni);
    GRAB_SCALAR(Nx);
    p->Ne1 = (int32_t)sqrt(p->Ne/2.0);
    p->Ni1 = (int32_t)sqrt(p->Ni/2.0);
    p->Nx1 = (int32_t)sqrt(p->Nx/2.0);

    // Jx (Jex, Jix, Jex1, Jix1)
    f = mxGetField(ps,0,"Jx");
    ERROR_CHECK(f && mxGetNumberOfElements(f)>=4, "Invalid field: Jx");
    {
        double* Jx = mxGetPr(f);
        p->Jex = Jx[0]; p->Jix = Jx[1]; p->Jex1 = Jx[2]; p->Jix1 = Jx[3];
    }

    // Jr (Jee, Jie, Jei, Jii)
    f = mxGetField(ps,0,"Jr");
    ERROR_CHECK(f && mxGetNumberOfElements(f)>=4, "Invalid field: Jr");
    {
        double* Jr = mxGetPr(f);
        p->Jee = Jr[0]; p->Jie = Jr[1]; p->Jei = Jr[2]; p->Jii = Jr[3];
    }

    // Kx
    f = mxGetField(ps,0,"Kx");
    ERROR_CHECK(f && mxGetNumberOfElements(f)>=2, "Invalid field: Kx");
    {
        double* Kx = mxGetPr(f);
        p->Kex = (int32_t)Kx[0]; p->Kix = (int32_t)Kx[1];
    }

    // Kr
    f = mxGetField(ps,0,"Kr");
    ERROR_CHECK(f && mxGetNumberOfElements(f)>=4, "Invalid field: Kr");
    {
        double* Kr = mxGetPr(f);
        p->Kee = (int32_t)Kr[0]; p->Kie = (int32_t)Kr[1]; p->Kei = (int32_t)Kr[2]; p->Kii = (int32_t)Kr[3];
    }



    GRAB_VEC2_2("C",     "Cm",  p->C,     "C/Cm");
    GRAB_VEC2_2("Vleak", "vl",  p->Vleak,"Vleak/vl");
    GRAB_VEC2_2("VT",    "vT",  p->VT,    "VT/vT");
    GRAB_VEC2_2("Vth",   "vth", p->Vth,   "Vth/vth");
    GRAB_VEC2_2("Vre",   "vre", p->Vre,   "Vre/vre");
    GRAB_VEC2_2("Vlb",   "vlb", p->Vlb,   "Vlb/vlb");
    GRAB_VEC2_STRICT(gl,          p->gl);
    GRAB_VEC2_STRICT(tref,        p->tref);
    GRAB_VEC2_STRICT(DeltaT,      p->DeltaT);

    GRAB_SCALAR(T);
    GRAB_SCALAR(dt);
    GRAB_SCALAR(maxns);

    // Optional: attarea (required in originals)
    f = mxGetField(ps,0,"attarea");
    ERROR_CHECK(f!=NULL, "Missing field: attarea");
    p->attarea = (int32_t)mxGetScalar(f);

    // Optional fields (silently default if absent)
    f = mxGetField(ps,0,"sigma_current");
    p->sigma_current = f ? mxGetScalar(f) : 0.0;

    f = mxGetField(ps,0,"m_current");
    if (f && mxGetNumberOfElements(f)>=2) {
        double* m = mxGetPr(f);
        p->me_current = m[0];
        p->mi_current = m[1];
    } else {
        p->me_current = 0.0;
        p->mi_current = 0.0;
    }

    // Flags default off; wrappers set them appropriately
    p->use_current_noise = 0;
    p->use_gaussian_input = 0;
    p->use_broad_weights = 0;

    #undef GRAB_SCALAR
    #undef GRAB_VEC2_STRICT
    #undef GRAB_SCALAR2
    #undef GRAB_VEC2_2
}

void eif_extract_synapse_params(const mxArray* ps, SynapseParams* sp) {
    memset(sp, 0, sizeof(*sp));
    const mxArray *f = NULL;

    f = mxGetField(ps,0,"taursyn");
    ERROR_CHECK(f && mxGetM(f)==3, "Invalid taursyn");
    sp->Nsyntype = (int32_t)mxGetN(f);
    sp->taursyn = mxGetPr(f);

    f = mxGetField(ps,0,"taudsyn");
    ERROR_CHECK(f && mxGetM(f)==3 && (int32_t)mxGetN(f)==sp->Nsyntype, "Invalid taudsyn");
    sp->taudsyn = mxGetPr(f);

    f = mxGetField(ps,0,"Psyn");
    ERROR_CHECK(f && mxGetM(f)==3 && (int32_t)mxGetN(f)==sp->Nsyntype, "Invalid Psyn");
    sp->Psyn = mxGetPr(f);

    int32_t total = sp->Nsyntype * 3;
    sp->syntype = (int32_t*)eif_safe_malloc(total * sizeof(int32_t), "syntype");
    sp->Psyn2   = (double*)eif_safe_malloc(total * sizeof(double), "Psyn2");
    sp->temp1   = (double*)eif_safe_malloc(total * sizeof(double), "temp1");
    sp->temp2   = (double*)eif_safe_malloc(total * sizeof(double), "temp2");

    sp->Nsyn = 0;
    for (int32_t isyn = 0; isyn < total; isyn++) {
        double psyn = sp->Psyn[isyn];
        if (psyn > 0.0) {
            double tau_r = sp->taursyn[isyn];
            double tau_d = sp->taudsyn[isyn];
            ERROR_CHECK(tau_r > 0.0 && tau_d > 0.0, "Non-positive tau in synapses");

            sp->syntype[sp->Nsyn] = isyn % 3; // (X,E,I)
            sp->Psyn2  [sp->Nsyn] = psyn;
            sp->temp1  [sp->Nsyn] = 1.0/tau_d + 1.0/tau_r;
            sp->temp2  [sp->Nsyn] = 1.0/(tau_d * tau_r);
            sp->Nsyn++;
        }
    }
}

void eif_free_synapse_params(SynapseParams* sp) {
    if (!sp) return;
    if (sp->syntype) mxFree(sp->syntype);
    if (sp->Psyn2)   mxFree(sp->Psyn2);
    if (sp->temp1)   mxFree(sp->temp1);
    if (sp->temp2)   mxFree(sp->temp2);
    memset(sp, 0, sizeof(*sp));
}

void eif_init_attention(const NetworkParams* p, double* attyne, double* attyni) {
    double ae = (p->Jex != 0.0) ? (p->Jex1 / p->Jex) : 1.0;
    double ai = (p->Jix != 0.0) ? (p->Jix1 / p->Jix) : 1.0;
    int32_t Nx1 = p->Nx1;
    for (int32_t jj = 0; jj < Nx1; jj++) {
        for (int32_t kk = 0; kk < Nx1 * 2; kk++) {
            int32_t idx = jj * Nx1 * 2 + kk;
            if (p->attarea == 1) {
                // First half attended
                attyne[idx] = (kk < Nx1) ? ae : 1.0;
                attyni[idx] = (kk < Nx1) ? ai : 1.0;
            } else {
                // Second half attended
                attyne[idx] = (kk >= Nx1) ? ae : 1.0;
                attyni[idx] = (kk >= Nx1) ? ai : 1.0;
            }
        }
    }
}

// RNG using rand() as originals; Box-Muller to match original noise variant
static inline double box_muller(void) {
    double u1 = (double)rand() / (double)RAND_MAX;
    if (u1 == 0.0) u1 = 1e-6;
    double u2 = (double)rand() / (double)RAND_MAX;
    if (u2 == 0.0) u2 = 1e-6;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

void eif_simulate_core(
    const double* sx, int32_t Nsx,
    const int32_t* Wrf, int32_t Nwrf,
    const int32_t* Wrr, int32_t Nwrr,
    const NetworkParams* P,
    const SynapseParams* S,
    const double* V0,
    const double* Irecord, int32_t Nrecord,
    int32_t rec_cols,
    double* s_out,
    double* Isynrecord_out,
    double* vr_out
) {
    const int32_t N = P->Ne + P->Ni;
    const int32_t Nt = (int32_t)(P->T / P->dt);
    const int32_t Ntref[2] = {
        (int32_t)round(P->tref[0] / P->dt),
        (int32_t)round(P->tref[1] / P->dt)
    };

    // Validate
    eif_validate_spike_data(sx, Nsx, P->Nx);
    eif_validate_connectivity(Wrf, Nwrf, N);
    eif_validate_connectivity(Wrr, Nwrr, N);
    ERROR_CHECK(S->Nsyn >= 0, "S->Nsyn invalid");

    // Allocate state
    double* v = (double*)eif_safe_malloc(N * sizeof(double), "membrane potential");
    int32_t* refstate = (int32_t*)eif_safe_malloc(N * sizeof(int32_t), "refstate");
    double* Isyn = (double*)eif_safe_malloc((size_t)N * S->Nsyn * sizeof(double), "Isyn");
    double* Isynprime = (double*)eif_safe_malloc((size_t)N * S->Nsyn * sizeof(double), "Isynprime");
    double* Isynrandom = NULL;
    if (P->use_current_noise) {
        Isynrandom = (double*)eif_safe_malloc(N * sizeof(double), "Isynrandom");
    }

    int32_t Kx = P->Kex + P->Kix;
    int32_t Ke = P->Kee + P->Kie;
    int32_t Ki = P->Kei + P->Kii;

    int32_t* postcellX  = (int32_t*)eif_safe_malloc(Kx * sizeof(int32_t), "postcellX");
    int32_t* postcellE  = (int32_t*)eif_safe_malloc(Ke * sizeof(int32_t), "postcellE");
    int32_t* postcellI  = (int32_t*)eif_safe_malloc(Ki * sizeof(int32_t), "postcellI");

    // For broad weight indexing, we also need the indices into Wrf/Wrr (Pr - Wrf etc)
    int32_t* postcellX_idx = NULL;
    int32_t* postcellE_idx = NULL;
    int32_t* postcellI_idx = NULL;
    if (P->use_broad_weights) {
        postcellX_idx = (int32_t*)eif_safe_malloc(Kx * sizeof(int32_t), "postcellX_idx");
        postcellE_idx = (int32_t*)eif_safe_malloc(Ke * sizeof(int32_t), "postcellE_idx");
        postcellI_idx = (int32_t*)eif_safe_malloc(Ki * sizeof(int32_t), "postcellI_idx");
    }

    double* attyne = (double*)eif_safe_malloc(P->Nx * sizeof(double), "attyne");
    double* attyni = (double*)eif_safe_malloc(P->Nx * sizeof(double), "attyni");
    eif_init_attention(P, attyne, attyni);

    // Seed RNG once to mimic original behavior (time-based)
    if (P->use_current_noise) {
        unsigned int seed = (unsigned int)time(NULL);
        srand(seed);
    }

    // Initialize
    memcpy(v, V0, N * sizeof(double));
    memset(refstate, 0, N * sizeof(int32_t));
    memset(Isyn, 0, (size_t)N * S->Nsyn * sizeof(double));
    memset(Isynprime, 0, (size_t)N * S->Nsyn * sizeof(double));
    if (Isynrandom) memset(Isynrandom, 0, N * sizeof(double));

    // Record initial state
    for (int32_t jj = 0; jj < Nrecord; jj++) {
        int32_t idx = (int32_t)round(Irecord[jj]) - 1;
        for (int32_t is = 0; is < S->Nsyn; is++) {
            Isynrecord_out[is * Nrecord + jj] = Isyn[idx * S->Nsyn + is];
        }
        // Extra record columns (e.g., noise) left as 0 to match originals
        vr_out[jj] = v[idx];
    }

    int32_t ns = 0;
    int32_t iXspike = 0;
    const double* iXspkInd = &sx[0];

    // Main loop
    for (int32_t it = 1; it < Nt && ns < P->maxns; it++) {
        double tnow = it * P->dt;

        // Update synaptic ODEs
        for (int32_t jj = 0; jj < N * S->Nsyn; jj++) {
            int32_t is = jj % S->Nsyn;
            Isyn[jj]      += Isynprime[jj] * P->dt;
            Isynprime[jj] -= P->dt * (Isynprime[jj] * S->temp1[is] + Isyn[jj] * S->temp2[is]);
        }

        // Process feedforward spikes at this bin
        while (iXspike < Nsx && *iXspkInd <= tnow) {
            iXspkInd++; // neuron id
            int32_t jspike = (int32_t)round(*iXspkInd) - 1;
            ERROR_CHECK(jspike >= 0 && jspike < P->Nx, "sx neuron id OOB");

            const int32_t* Pr = &Wrf[jspike * Kx];
            for (int32_t k = 0; k < Kx; k++) {
                postcellX[k] = Pr[k] - 1;
                ERROR_CHECK(postcellX[k] >= 0 && postcellX[k] < N, "Wrf postcell OOB");
                if (P->use_broad_weights) postcellX_idx[k] = (int32_t)((&Pr[k]) - Wrf);
            }

            for (int32_t is = 0; is < S->Nsyn; is++) {
                if (S->syntype[is] == SYNAPSE_FEEDFORWARD) {
                    for (int32_t k = 0; k < Kx; k++) {
                        double strength = 0.0;
                        if (P->use_broad_weights && P->Jrf) {
                            int32_t idx = postcellX_idx[k];
                            strength = P->Jrf[idx];
                        } else {
                            strength = (postcellX[k] < P->Ne) ? (P->Jex * attyne[jspike])
                                                              : (P->Jix * attyni[jspike]);
                        }
                        Isynprime[postcellX[k] * S->Nsyn + is] += strength * S->temp2[is];
                    }
                }
            }

            iXspike++;
            iXspkInd++; // move to next spike time
        }

        // Walk recurrent neurons
        const int32_t* Pr = &Wrr[0];
        for (int32_t j = 0; j < N; j++) {
            PopulationType pop = (j < P->Ne) ? EXCITATORY : INHIBITORY;
            int32_t Keff = (pop == EXCITATORY) ? Ke : Ki;

            // Update V
            if (refstate[j] <= 0) {
                double Isyntot = 0.0;

                // Optional noise current (added exactly like original variants)
                if (P->use_current_noise && Isynrandom) {
                    Isynrandom[j] = P->sigma_current * box_muller();
                    Isyntot += Isynrandom[j];
                }

                // Optional Gaussian input with baseline + time series
                if (P->use_gaussian_input) {
                    double bias = (pop == EXCITATORY) ? P->me_current : P->mi_current;
                    Isyntot += bias;
                    if (P->L_current) {
                        // Original code uses floor(i*dt) as index (1-based vector in MATLAB passed as double* here 0-based)
                        int32_t ti = (int32_t)floor(tnow);
                        if (ti >= 0) Isyntot += P->L_current[ti];
                    }
                }

                // Sum synaptic currents
                for (int32_t is = 0; is < S->Nsyn; is++) {
                    Isyntot += S->Psyn2[is] * Isyn[j * S->Nsyn + is];
                }

                double exp_term = fast_exp((v[j] - P->VT[pop]) / P->DeltaT[pop]);
                double dv = (Isyntot - P->gl[pop] * (v[j] - P->Vleak[pop]) +
                            P->gl[pop] * P->DeltaT[pop] * exp_term) * P->dt / P->C[pop];
                double vnext = v[j] + dv;
                double vmin  = P->Vlb[pop];
                v[j] = (vnext < vmin) ? vmin : vnext;
            } else {
                v[j] = (refstate[j] > 1) ? P->Vth[pop] : P->Vre[pop];
                refstate[j]--;
            }

            // Spike?
            if (v[j] >= P->Vth[pop] && refstate[j] <= 0 && ns < P->maxns) {
                refstate[j] = Ntref[pop];
                v[j] = P->Vth[pop];

                s_out[2*ns]   = tnow;
                s_out[2*ns+1] = j + 1;
                ns++;

                // Postsyn targets and syn updates
                if (pop == EXCITATORY) {
                    for (int32_t k = 0; k < Ke; k++) {
                        postcellE[k] = Pr[k] - 1;
                        ERROR_CHECK(postcellE[k] >= 0 && postcellE[k] < N, "Wrr E postcell OOB");
                        if (P->use_broad_weights) postcellE_idx[k] = (int32_t)((&Pr[k]) - Wrr);
                    }
                    for (int32_t is = 0; is < S->Nsyn; is++) {
                        if (S->syntype[is] == SYNAPSE_EXCITATORY) {
                            for (int32_t k = 0; k < Ke; k++) {
                                double strength = 0.0;
                                if (P->use_broad_weights && P->Jrr) {
                                    int32_t idx = postcellE_idx[k];
                                    strength = P->Jrr[idx];
                                } else {
                                    strength = (postcellE[k] < P->Ne) ? P->Jee : P->Jie;
                                }
                                Isynprime[postcellE[k] * S->Nsyn + is] += strength * S->temp2[is];
                            }
                        }
                    }
                    Pr += Ke;
                } else {
                    const int32_t* PrI = &Wrr[(j - P->Ne) * Ki + P->Ne * Ke];
                    for (int32_t k = 0; k < Ki; k++) {
                        postcellI[k] = PrI[k] - 1;
                        ERROR_CHECK(postcellI[k] >= 0 && postcellI[k] < N, "Wrr I postcell OOB");
                        if (P->use_broad_weights) postcellI_idx[k] = (int32_t)((&PrI[k]) - Wrr);
                    }
                    for (int32_t is = 0; is < S->Nsyn; is++) {
                        if (S->syntype[is] == SYNAPSE_INHIBITORY) {
                            for (int32_t k = 0; k < Ki; k++) {
                                double strength = 0.0;
                                if (P->use_broad_weights && P->Jrr) {
                                    int32_t idx = postcellI_idx[k];
                                    strength = P->Jrr[idx];
                                } else {
                                    strength = (postcellI[k] < P->Ne) ? P->Jei : P->Jii;
                                }
                                Isynprime[postcellI[k] * S->Nsyn + is] += strength * S->temp2[is];
                            }
                        }
                    }
                }
            } else {
                // Advance pointer even if no spike (to preserve traversal parity)
                if (pop == EXCITATORY) Pr += Ke;
            }
        }

        // Record this bin
        for (int32_t jj = 0; jj < Nrecord; jj++) {
            int32_t idx = (int32_t)round(Irecord[jj]) - 1;
            for (int32_t is = 0; is < S->Nsyn; is++) {
                Isynrecord_out[it * (Nrecord * rec_cols) + is * Nrecord + jj] =
                    Isyn[idx * S->Nsyn + is];
            }
            // Leave any extra columns (e.g., noise) as zeros to match originals.
            vr_out[it * Nrecord + jj] = v[idx];
        }
    }
}

