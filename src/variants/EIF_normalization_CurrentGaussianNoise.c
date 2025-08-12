// Variant: CurrentGaussianNoise (global time-varying current + DC per pop)
// Inputs: prhs[0]=sx, [1]=Wrf(idx), [2]=Wrr(idx), [3]=params(with m_current=[mE mI]),
//         [4]=L_current (Nt1x1)
// Outputs: [spikes, Isyn_recorded, V_recorded]

#include "EIF_common.h"

enum { SY_X=0, SY_E=1, SY_I=2, NSYN=3 };

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 5) mexErrMsgIdAndTxt("EIF:Args","Need 5 inputs: sx,Wrf,Wrr,params,L_current");
    if (nlhs != 3) mexErrMsgIdAndTxt("EIF:Args","3 outputs");

    const double *sx = mxGetPr(prhs[0]); if (mxGetM(prhs[0])!=2) mexErrMsgIdAndTxt("EIF:sx","sx 2xNsx");
    const mwSize Nsx = mxGetN(prhs[0]);
    int32_t *Wrf = eif_cast_to_int32_copy(prhs[1],"Wrf");
    int32_t *Wrr = eif_cast_to_int32_copy(prhs[2],"Wrr");

    NetworkParams P; parse_params(prhs[3], &P);

    /* L_current(t) */
    eif_require_vector(prhs[4],"L_current");
    const double *Lcur = mxGetPr(prhs[4]);
    const int32_t Nt1  = (int32_t)mxGetNumberOfElements(prhs[4]);

    const mxArray *mx;

    /* Jx, Jr, Kx, Kr */
    mx=eif_require_field(prhs[3],"Jx"); const double* Jx=mxGetPr(mx);
    const double Jex=Jx[0], Jix=Jx[1];
    double mE=0.0, mI=0.0;
    mx=eif_require_field(prhs[3],"m_current"); const double* mcur=mxGetPr(mx); mE=mcur[0]; mI=mcur[1];

    mx=eif_require_field(prhs[3],"Jr"); const double* Jr=mxGetPr(mx);
    const double Jee=Jr[0], Jei=Jr[1], Jie=Jr[2], Jii=Jr[3];

    mx=eif_require_field(prhs[3],"Kx"); const double* Kx=mxGetPr(mx);
    const int32_t Kex=(int32_t)Kx[0], Kix=(int32_t)Kx[1];
    mx=eif_require_field(prhs[3],"Kr"); const double* Kr=mxGetPr(mx);
    const int32_t Kee=(int32_t)Kr[0], Kei=(int32_t)Kr[1], Kie=(int32_t)Kr[2], Kii=(int32_t)Kr[3];

    /* neuron 2-vectors */
    const char* names[]={"Cm","gl","vl","DeltaT","vT","tref","vth","vre","vlb"};
    double C[2],gl[2],Vleak[2],DeltaT[2],VT[2],tref[2],Vth[2],Vre[2],Vlb[2];
    for(int i=0;i<9;i++){ mx=eif_require_field(prhs[3],names[i]); if(mxGetNumberOfElements(mx)!=2) mexErrMsgIdAndTxt("EIF:param","%s need 2",names[i]); double* v=mxGetPr(mx); (&C)[i][0]=v[0]; (&C)[i][1]=v[1]; }

    /* tausyn */
    mx=eif_require_field(prhs[3],"taursyn"); const double* taur=mxGetPr(mx);
    mx=eif_require_field(prhs[3],"taudsyn"); const double* taud=mxGetPr(mx);

    /* recording + V0 */
    mx=eif_require_field(prhs[3],"Irecord"); const double* Irecord=mxGetPr(mx); const int32_t Nrecord=(int32_t)mxGetNumberOfElements(mx);
    mx=eif_require_field(prhs[3],"V0"); const double* V0=mxGetPr(mx); if((int)mxGetNumberOfElements(mx)!=P.N) mexErrMsgIdAndTxt("EIF:V0","V0 N x 1");

    const int32_t N=P.N, Nt=(int32_t)floor(P.T/P.dt)+1;
    plhs[0]=mxCreateDoubleMatrix(2,P.maxns,mxREAL); double *s=mxGetPr(plhs[0]);
    plhs[1]=mxCreateDoubleMatrix(Nrecord*NSYN,Nt,mxREAL); double *Irec=mxGetPr(plhs[1]);
    plhs[2]=mxCreateDoubleMatrix(Nrecord,Nt,mxREAL); double *Vrec=mxGetPr(plhs[2]);

    double *v=(double*)mxMalloc(sizeof(double)*N);
    int32_t* ref=(int32_t*)mxCalloc(N,sizeof(int32_t));
    double *Isyn=(double*)mxCalloc((size_t)N*NSYN,sizeof(double));
    double *Isynp=(double*)mxCalloc((size_t)N*NSYN,sizeof(double));

    int32_t NtrefE=(int32_t)round(tref[0]/P.dt), NtrefI=(int32_t)round(tref[1]/P.dt);
    double a1[NSYN],a2[NSYN]; for(int sy=0;sy<NSYN;++sy){a1[sy]=1.0/taud[sy]; a2[sy]=1.0/(taur[sy]*taud[sy]);}
    for(int j=0;j<N;++j){v[j]=V0[j]; ref[j]=0;}
    for(int r=0;r<Nrecord;++r){int idx=(int)floor(Irecord[r]-1.0); if(idx<0||idx>=N) mexErrMsgIdAndTxt("EIF:Irecord","bad");
        Vrec[r]=v[idx]; for(int sy=0;sy<NSYN;++sy) Irec[sy*Nrecord+r]=Isyn[(int64_t)idx*NSYN+sy];}

    /* ff spike cursor */
    int64_t ix=0; double tnext=(Nsx>0? sx[0]:INFINITY); int ns=0;

    for(int it=1; it<Nt && ns<P.maxns; ++it){
        double t=it*P.dt;

        /* syn dynamics */
        for(int64_t jj=0;jj<(int64_t)N*NSYN;++jj){int sy=(int)(jj%NSYN);
            Isyn[jj]+=Isynp[jj]*P.dt; Isynp[jj]+=(-Isynp[jj]*a1[sy]-Isyn[jj]*a2[sy])*P.dt;}

        /* ff updates (constant Jex/Jix) */
        while(ix<(int64_t)Nsx && tnext<=t){
            int x=(int)floor(sx[1+2*ix]-1.0); if(x<0||x>=P.Nx) mexErrMsgIdAndTxt("EIF:sx","idx");
            int64_t b=(int64_t)x*(Kex+Kix);
            for(int k=0;k<Kex;++k){int post=Wrf[b+k]-1; Isynp[(int64_t)post*NSYN+SY_X]+=Jex;}
            for(int k=0;k<Kix;++k){int post=Wrf[b+Kex+k]-1; Isynp[(int64_t)post*NSYN+SY_X]+=Jix;}
            ++ix; tnext=(ix<(int64_t)Nsx? sx[0+2*ix] : INFINITY);
        }

        /* recurrent + membrane */
        for(int j=0;j<N;++j){
            int pop=(j<P.Ne?0:1);
            double glj=gl[pop],Cj=C[pop],Vlj=Vleak[pop],DTj=DeltaT[pop],VTj=VT[pop],Vthj=Vth[pop],Vrej=Vre[pop],Vlbj=Vlb[pop];
            if(ref[j]<=0){
                double I=0.0; for(int sy=0;sy<NSYN;++sy) I+=Isyn[(int64_t)j*NSYN+sy];
                /* add global time signal + DC per pop */
                I += (pop==0? mE : mI);
                I += Lcur[(it < Nt1 ? it : Nt1-1)];
                double dv=(-glj*(v[j]-Vlj)+glj*DTj*eif_fast_exp((v[j]-VTj)/DTj)+I)/Cj; v[j]+=dv*P.dt; if(v[j]<Vlbj) v[j]=Vlbj;
            }else{ v[j]=(ref[j]>1?Vthj:Vrej); ref[j]--; }

            if(v[j]>=Vthj && ref[j]<=0 && ns<P.maxns){
                ref[j]=(pop==0?NtrefE:NtrefI); v[j]=Vthj; s[0+2*ns]=t; s[1+2*ns]=j+1; ns++;
                if(pop==0){
                    int64_t b=(int64_t)j*(Kee+Kei);
                    for(int k=0;k<Kee;++k){int post=Wrr[b+k]-1; Isynp[(int64_t)post*NSYN+SY_E]+=Jee;}
                    for(int k=0;k<Kei;++k){int post=Wrr[b+Kee+k]-1; Isynp[(int64_t)post*NSYN+SY_I]+=Jei;}
                }else{
                    int64_t b=(int64_t)P.Ne*(Kee+Kei)+(int64_t)(j-P.Ne)*(Kie+Kii);
                    for(int k=0;k<Kie;++k){int post=Wrr[b+k]-1; Isynp[(int64_t)post*NSYN+SY_E]+=Jie;}
                    for(int k=0;k<Kii;++k){int post=Wrr[b+Kie+k]-1; Isynp[(int64_t)post*NSYN+SY_I]+=Jii;}
                }
            }
        }

        for(int r=0;r<Nrecord;++r){int idx=(int)floor(Irecord[r]-1.0);
            Vrec[r + it*(mwSize)Nrecord]=v[idx];
            for(int sy=0;sy<NSYN;++sy) Irec[sy*Nrecord+r + it*(mwSize)(Nrecord*NSYN)] = Isyn[(int64_t)idx*NSYN+sy];}
    }

    if(ns<P.maxns){ mxArray* S=mxCreateDoubleMatrix(2,ns,mxREAL); memcpy(mxGetPr(S),s,sizeof(double)*2*ns); plhs[0]=S; }

    SAFE_FREE(Wrf); SAFE_FREE(Wrr); SAFE_FREE(v); SAFE_FREE(ref); SAFE_FREE(Isyn); SAFE_FREE(Isynp);
}
