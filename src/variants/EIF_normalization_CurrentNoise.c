//  Variant: CurrentNoise
//  Inputs:  prhs[0]=sx(2xNsx double), prhs[1]=Wrf(int/double Nx*(Kex+Kix)x1),
//           prhs[2]=Wrr(int/double [Ne*(Kee+Kei)+Ni*(Kie+Kii)]x1), prhs[3]=params(struct with sigma_current)
//  Outputs: plhs[0]=spikes(2 x maxns), plhs[1]=Isyn_recorded(Nrecord*Nsyn x Nt), plhs[2]=V_recorded(Nrecord x Nt)

#include "EIF_common.h"

enum { SY_X = 0, SY_E = 1, SY_I = 2, NSYN = 3 };

static void get_conn_blocks(int32_t Ne, int32_t Ni, int32_t Kee, int32_t Kei, int32_t Kie, int32_t Kii,
                            int32_t *Ke_block, int32_t *Ki_block, int64_t *WrrE_base, int64_t *WrrI_base)
{
    *Ke_block  = Kee + Kei;
    *Ki_block  = Kie + Kii;
    *WrrE_base = 0;
    *WrrI_base = (int64_t)Ne * (*Ke_block);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* ---- Inputs sanity ---- */
    if (nrhs != 4) mexErrMsgIdAndTxt("EIF:Args","Expected 4 inputs: sx, Wrf, Wrr, params");
    if (nlhs != 3) mexErrMsgIdAndTxt("EIF:Args","Expected 3 outputs");

    /* sx (2 x Nsx) */
    eif_require_matrix(prhs[0], "sx");
    if (mxGetM(prhs[0]) != 2) mexErrMsgIdAndTxt("EIF:sx","sx must be 2xNsx");
    const double *sx = mxGetPr(prhs[0]);
    const mwSize Nsx = mxGetN(prhs[0]);

    /* Wrf & Wrr (post lists) */
    int32_t *Wrf = eif_cast_to_int32_copy(prhs[1], "Wrf");
    int32_t *Wrr = eif_cast_to_int32_copy(prhs[2], "Wrr");

    /* params */
    NetworkParams P; parse_params(prhs[3], &P);

    /* ---- Extract model params from struct (matching your default) ---- */
    const mxArray *mx;

    /* sizes for grid (not used directly but preserved) */
    // int32_t Ne1 = P.Ne1, Ni1 = P.Ni1, Nx1 = P.Nx1;
    // mx = eif_get_field(prhs[3],"Ne1"); if(mx) Ne1 = (int32_t)mxGetScalar(mx);
    // mx = eif_get_field(prhs[3],"Ni1"); if(mx) Ni1 = (int32_t)mxGetScalar(mx);
    // mx = eif_get_field(prhs[3],"Nx1"); if(mx) Nx1 = (int32_t)mxGetScalar(mx);

    /* Jx = [Jex; Jix; Jex1; Jix1] */
    mx = eif_require_field(prhs[3],"Jx");
    if (mxGetNumberOfElements(mx) < 2) mexErrMsgIdAndTxt("EIF:Jx","Jx needs at least 2 elems");
    const double *Jx = mxGetPr(mx);
    const double Jex = Jx[0], Jix = Jx[1];
    double Jex1 = Jex, Jix1 = Jix;
    if (mxGetNumberOfElements(mx) >= 4) { Jex1 = Jx[2]; Jix1 = Jx[3]; }

    /* Jr = [Jee, Jei; Jie, Jii] flattened */
    mx = eif_require_field(prhs[3],"Jr");
    if (mxGetNumberOfElements(mx) < 4) mexErrMsgIdAndTxt("EIF:Jr","Jr needs 4 elements");
    const double *Jr = mxGetPr(mx);
    const double Jee = Jr[0], Jei = Jr[1], Jie = Jr[2], Jii = Jr[3];

    /* Kx = [Kex; Kix], Kr = [Kee, Kei; Kie, Kii] */
    mx = eif_require_field(prhs[3],"Kx"); const double *Kx = mxGetPr(mx);
    const int32_t Kex = (int32_t)Kx[0], Kix = (int32_t)Kx[1];
    mx = eif_require_field(prhs[3],"Kr"); const double *Kr = mxGetPr(mx);
    const int32_t Kee = (int32_t)Kr[0], Kei = (int32_t)Kr[1], Kie = (int32_t)Kr[2], Kii = (int32_t)Kr[3];

    /* neuron arrays (2-element each) */
    double *C     = (double*)mxCalloc(2,sizeof(double));
    double *gl    = (double*)mxCalloc(2,sizeof(double));
    double *Vleak = (double*)mxCalloc(2,sizeof(double));
    double *DeltaT= (double*)mxCalloc(2,sizeof(double));
    double *VT    = (double*)mxCalloc(2,sizeof(double));
    double *tref  = (double*)mxCalloc(2,sizeof(double));
    double *Vth   = (double*)mxCalloc(2,sizeof(double));
    double *Vre   = (double*)mxCalloc(2,sizeof(double));
    double *Vlb   = (double*)mxCalloc(2,sizeof(double));
    const char* names[]={"Cm","gl","vl","DeltaT","vT","tref","vth","vre","vlb"};
    double* dsts[]={C,gl,Vleak,DeltaT,VT,tref,Vth,Vre,Vlb};
    for(int i=0;i<9;i++){
        mx = eif_require_field(prhs[3],names[i]);
        if (mxGetNumberOfElements(mx)!=2) mexErrMsgIdAndTxt("EIF:param","%s must have 2 elems",names[i]);
        double* v = mxGetPr(mx); dsts[i][0]=v[0]; dsts[i][1]=v[1];
    }

    /* tausyn (rise/decay) per syn type */
    mx = eif_require_field(prhs[3],"taursyn"); if(mxGetNumberOfElements(mx)<NSYN) mexErrMsgIdAndTxt("EIF:taursyn","need %d",NSYN);
    const double *taur = mxGetPr(mx);
    mx = eif_require_field(prhs[3],"taudsyn"); if(mxGetNumberOfElements(mx)<NSYN) mexErrMsgIdAndTxt("EIF:taudsyn","need %d",NSYN);
    const double *taud = mxGetPr(mx);

    /* recording set */
    mx = eif_require_field(prhs[3],"Irecord"); eif_require_vector(mx,"Irecord");
    const double *Irecord = mxGetPr(mx); const int32_t Nrecord = (int32_t)mxGetNumberOfElements(mx);

    /* initial V0 (N x 1) */
    mx = eif_require_field(prhs[3],"V0");
    if ((int)mxGetNumberOfElements(mx) != P.N) mexErrMsgIdAndTxt("EIF:V0","V0 must be N x 1");
    const double *V0 = mxGetPr(mx);

    /* variant: sigma_current */
    mx = eif_require_field(prhs[3],"sigma_current");
    const double sigma_current = mxGetScalar(mx);

    /* ---- Derived sizes ---- */
    const int32_t N  = P.N;
    const int32_t Nt = (int32_t)floor(P.T / P.dt) + 1;
    int32_t Ke_block, Ki_block; int64_t WrrE_base, WrrI_base;
    get_conn_blocks(P.Ne,P.Ni,Kee,Kei,Kie,Kii,&Ke_block,&Ki_block,&WrrE_base,&WrrI_base);

    /* ---- Output buffers ---- */
    plhs[0] = mxCreateDoubleMatrix(2, P.maxns, mxREAL);       double *s   = mxGetPr(plhs[0]);
    plhs[1] = mxCreateDoubleMatrix(Nrecord*NSYN, Nt, mxREAL); double *Irec= mxGetPr(plhs[1]);
    plhs[2] = mxCreateDoubleMatrix(Nrecord, Nt, mxREAL);      double *Vrec= mxGetPr(plhs[2]);

    /* ---- State ---- */
    double *v = (double*)mxMalloc(sizeof(double)*N);
    int32_t *ref = (int32_t*)mxCalloc(N,sizeof(int32_t));
    double *Isyn     = (double*)mxCalloc((size_t)N*NSYN,sizeof(double));
    double *Isynprim = (double*)mxCalloc((size_t)N*NSYN,sizeof(double));

    int32_t NtrefE = (int32_t)round(tref[0]/P.dt);
    int32_t NtrefI = (int32_t)round(tref[1]/P.dt);

    /* precompute ODE constants per syn */
    double a1[NSYN], a2[NSYN];
    for (int sy=0; sy<NSYN; ++sy) {
        a1[sy] = 1.0/taud[sy];
        a2[sy] = 1.0/(taur[sy]*taud[sy]);
    }

    /* init */
    for (int j=0;j<N;++j) { v[j]=V0[j]; ref[j]=0; }
    for (int r=0;r<Nrecord;++r) {
        int idx=(int)floor(Irecord[r]-1.0);
        if (idx<0 || idx>=N) mexErrMsgIdAndTxt("EIF:Irecord","index out of range");
        Vrec[r + 0*Nrecord] = v[idx];
        for(int sy=0; sy<NSYN; ++sy) Irec[sy*Nrecord + r] = Isyn[idx*NSYN+sy];
    }

    /* seed RNG for noise */
    eif_seed((uint64_t)time(NULL));

    /* feedforward spike cursor */
    int64_t ix = 0;      /* index into columns of sx */
    double  tnext_x = (Nsx>0 ? sx[0] : INFINITY);

    int32_t ns = 0;
    /* ---- Main loop ---- */
    for (int it=1; it<Nt && ns<P.maxns; ++it) {
        double t = it * P.dt;

        /* syn ODE step */
        for (int64_t jj=0; jj<(int64_t)N*NSYN; ++jj) {
            int sy = (int)(jj % NSYN);
            Isyn[jj]     += Isynprim[jj] * P.dt;
            Isynprim[jj] += (-Isynprim[jj]*a1[sy] - Isyn[jj]*a2[sy]) * P.dt;
        }

        /* process all X spikes up to time t */
        while (ix < (int64_t)Nsx && tnext_x <= t) {
            int x_id = (int)floor(sx[1 + 2*ix] - 1.0);
            if (x_id < 0 || x_id >= P.Nx) mexErrMsgIdAndTxt("EIF:sx","feedforward index out of range");
            int64_t base = (int64_t)x_id * (Kex + Kix);
            for (int k=0;k<Kex;++k) {
                int post = Wrf[base + k] - 1; if (post<0||post>=N) mexErrMsgIdAndTxt("EIF:Wrf","post<0");
                Isynprim[(int64_t)post*NSYN + SY_X] += Jex; /* attention factor hook could go here */
            }
            for (int k=0;k<Kix;++k) {
                int post = Wrf[base + Kex + k] - 1; if (post<0||post>=N) mexErrMsgIdAndTxt("EIF:Wrf","post<0");
                Isynprim[(int64_t)post*NSYN + SY_X] += Jix; /* attention factor hook could go here */
            }
            ++ix;
            tnext_x = (ix < (int64_t)Nsx ? sx[0 + 2*ix] : INFINITY);
        }

        /* recurrent loop with membrane update + spike propagation at END of bin */
        int64_t rptr = 0; /* pointer into Wrr, advanced coherently */
        for (int j=0;j<N;++j) {
            const int pop = (j < P.Ne ? 0 : 1);
            const double glj = gl[pop], Cj=C[pop], Vlj=Vleak[pop], DTj=DeltaT[pop], VTj=VT[pop], Vthj=Vth[pop], Vrej=Vre[pop], Vlbj=Vlb[pop];
            int32_t *refj = &ref[j];

            /* integrate */
            if (*refj<=0) {
                double I = 0.0;
                for (int sy=0; sy<NSYN; ++sy) I += Isyn[(int64_t)j*NSYN + sy];
                /* add white current noise */
                I += sigma_current * eif_rand_normal();

                double dv = (-glj*(v[j]-Vlj) + glj*DTj*eif_fast_exp((v[j]-VTj)/DTj) + I)/Cj;
                v[j] += dv * P.dt;
                if (v[j] < Vlbj) v[j] = Vlbj;
            } else {
                v[j] = (*refj>1 ? Vthj : Vrej);
                (*refj)--;
            }

            /* spike? */
            if (v[j] >= Vthj && *refj<=0 && ns < P.maxns) {
                *refj = (pop==0 ? NtrefE : NtrefI);
                v[j]  = Vthj;
                s[0 + 2*ns] = t;
                s[1 + 2*ns] = j + 1;
                ns++;

                /* propagate at end of bin */
                if (pop==0) {
                    int64_t base = (int64_t)j * Ke_block;
                    for (int k=0;k<Kee;++k) {
                        int post = Wrr[base + k] - 1; if (post<0||post>=N) mexErrMsgIdAndTxt("EIF:Wrr","post<0");
                        Isynprim[(int64_t)post*NSYN + SY_E] += Jee;
                    }
                    for (int k=0;k<Kei;++k) {
                        int post = Wrr[base + Kee + k] - 1; if (post<0||post>=N) mexErrMsgIdAndTxt("EIF:Wrr","post<0");
                        Isynprim[(int64_t)post*NSYN + SY_I] += Jei;
                    }
                } else {
                    int64_t base = WrrI_base + (int64_t)(j - P.Ne) * Ki_block;
                    for (int k=0;k<Kie;++k) {
                        int post = Wrr[base + k] - 1; if (post<0||post>=N) mexErrMsgIdAndTxt("EIF:Wrr","post<0");
                        Isynprim[(int64_t)post*NSYN + SY_E] += Jie;
                    }
                    for (int k=0;k<Kii;++k) {
                        int post = Wrr[base + Kie + k] - 1; if (post<0||post>=N) mexErrMsgIdAndTxt("EIF:Wrr","post<0");
                        Isynprim[(int64_t)post*NSYN + SY_I] += Jii;
                    }
                }
            }
        }

        /* record */
        for (int r=0;r<Nrecord;++r) {
            int idx = (int)floor(Irecord[r]-1.0);
            Vrec[r + it*(mwSize)Nrecord] = v[idx];
            for (int sy=0; sy<NSYN; ++sy) {
                Irec[sy*Nrecord + r + it*(mwSize)(Nrecord*NSYN)] = Isyn[(int64_t)idx*NSYN + sy];
            }
        }
    }

    /* shrink spikes second dimension to actual count */
    if (ns < P.maxns) {
        mxArray *S = mxCreateDoubleMatrix(2, ns, mxREAL);
        memcpy(mxGetPr(S), s, sizeof(double)*2*ns);
        plhs[0] = S;
    }

    /* free */
    SAFE_FREE(Wrf); SAFE_FREE(Wrr);
    SAFE_FREE(v); SAFE_FREE(ref); SAFE_FREE(Isyn); SAFE_FREE(Isynprim);
    SAFE_FREE(C); SAFE_FREE(gl); SAFE_FREE(Vleak); SAFE_FREE(DeltaT); SAFE_FREE(VT);
    SAFE_FREE(tref); SAFE_FREE(Vth); SAFE_FREE(Vre); SAFE_FREE(Vlb);
}
