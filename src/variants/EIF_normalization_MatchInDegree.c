// Variant: InDegree (variable out-degree; use seqindX/E/I from params to segment W lists)
// Inputs: prhs[0]=sx, prhs[1]=Wrf(post idx), prhs[2]=Wrr(post idx), prhs[3]=params(with seqindX/E/I and outdegree* arrays)
// Outputs: [spikes, Isyn_recorded, V_recorded]

#include "EIF_common.h"

enum { SY_X=0, SY_E=1, SY_I=2, NSYN=3 };

static void get_seq(const mxArray* Pmx, const char* name, int32_t expect, const int32_t **ptr, int32_t *len)
{
    const mxArray *mx = eif_require_field(Pmx, name);
    *ptr = (const int32_t*)mxGetData(mx);
    *len = (int32_t)mxGetNumberOfElements(mx);
    if (*len != expect) mexErrMsgIdAndTxt("EIF:seq","%s must have %d elements", name, expect);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 4) mexErrMsgIdAndTxt("EIF:Args","Need 4 inputs: sx,Wrf,Wrr,params");
    if (nlhs != 3) mexErrMsgIdAndTxt("EIF:Args","3 outputs");

    const double *sx = mxGetPr(prhs[0]); if (mxGetM(prhs[0])!=2) mexErrMsgIdAndTxt("EIF:sx","sx 2xNsx");
    const mwSize Nsx = mxGetN(prhs[0]);
    int32_t *Wrf = eif_cast_to_int32_copy(prhs[1], "Wrf");
    int32_t *Wrr = eif_cast_to_int32_copy(prhs[2], "Wrr");
    NetworkParams P; parse_params(prhs[3], &P);

    const mxArray *mx;

    /* K, J */
    mx=eif_require_field(prhs[3],"Jx"); const double* Jx=mxGetPr(mx); const double Jex=Jx[0], Jix=Jx[1];
    mx=eif_require_field(prhs[3],"Jr"); const double* Jr=mxGetPr(mx); const double Jee=Jr[0], Jei=Jr[1], Jie=Jr[2], Jii=Jr[3];

    /* neuron params */
    const char* names[]={"Cm","gl","vl","DeltaT","vT","tref","vth","vre","vlb"};
    double C[2],gl[2],Vleak[2],DeltaT[2],VT[2],tref[2],Vth[2],Vre[2],Vlb[2];
    for(int i=0;i<9;i++){ mx=eif_require_field(prhs[3],names[i]); if(mxGetNumberOfElements(mx)!=2) mexErrMsgIdAndTxt("EIF:param","%s need 2",names[i]); double* v=mxGetPr(mx); (&C)[i][0]=v[0]; (&C)[i][1]=v[1]; }

    /* tausyn */
    mx=eif_require_field(prhs[3],"taursyn"); const double* taur=mxGetPr(mx);
    mx=eif_require_field(prhs[3],"taudsyn"); const double* taud=mxGetPr(mx);

    /* recording + V0 */
    mx=eif_require_field(prhs[3],"Irecord"); const double* Irecord=mxGetPr(mx); const int32_t Nrecord=(int32_t)mxGetNumberOfElements(mx);
    mx=eif_require_field(prhs[3],"V0"); const double* V0=mxGetPr(mx); if((int)mxGetNumberOfElements(mx)!=P.N) mexErrMsgIdAndTxt("EIF:V0","V0 N x 1");

    /* seqind for variable degree (1-based segment starts like MATLAB’s cs array) */
    const int32_t *seqX, *seqE, *seqI; int32_t Lx,Le,Li;
    get_seq(prhs[3],"seqindX", P.Nx+1, &seqX, &Lx);
    get_seq(prhs[3],"seqindE", P.Ne+1, &seqE, &Le);
    get_seq(prhs[3],"seqindI", P.Ni+1, &seqI, &Li);

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

    int ns=0; int64_t ix=0; double tnext=(Nsx>0? sx[0]:INFINITY);

    for(int it=1; it<Nt && ns<P.maxns; ++it){
        double t=it*P.dt;
        for(int64_t jj=0;jj<(int64_t)N*NSYN;++jj){int sy=(int)(jj%NSYN); Isyn[jj]+=Isynp[jj]*P.dt; Isynp[jj]+=(-Isynp[jj]*a1[sy]-Isyn[jj]*a2[sy])*P.dt;}
        /* ff spikes: use seqindX to segment Wrf */
        while(ix<(int64_t)Nsx && tnext<=t){
            int x=(int)floor(sx[1+2*ix]-1.0); if(x<0||x>=P.Nx) mexErrMsgIdAndTxt("EIF:sx","idx");
            int64_t b = (int64_t)seqX[x]; int64_t e = (int64_t)seqX[x+1];
            for(int64_t p=b; p<e; ++p){ int post=Wrf[p]-1; if(post<0||post>=N) mexErrMsgIdAndTxt("EIF:Wrf","post");
                /* assume X syn kernel (SY_X) and split weight by post’s pop using Jex/Jix */
                if (post < P.Ne) Isynp[(int64_t)post*NSYN+SY_X]+=Jex; else Isynp[(int64_t)post*NSYN+SY_X]+=Jix;
            }
            ++ix; tnext=(ix<(int64_t)Nsx? sx[0+2*ix] : INFINITY);
        }
        /* recurrent + membrane */
        for(int j=0;j<N;++j){
            int pop=(j<P.Ne?0:1);
            const double glj=gl[pop],Cj=C[pop],Vlj=Vleak[pop],DTj=DeltaT[pop],VTj=VT[pop],Vthj=Vth[pop],Vrej=Vre[pop],Vlbj=Vlb[pop];
            if(ref[j]<=0){
                double I=0.0; for(int sy=0;sy<NSYN;++sy) I+=Isyn[(int64_t)j*NSYN+sy];
                double dv=(-glj*(v[j]-Vlj)+glj*DTj*eif_fast_exp((v[j]-VTj)/DTj)+I)/Cj; v[j]+=dv*P.dt; if(v[j]<Vlbj)v[j]=Vlbj;
            }else{ v[j]=(ref[j]>1?Vthj:Vrej); ref[j]--; }
            if(v[j]>=Vthj && ref[j]<=0 && ns<P.maxns){
                ref[j]=(pop==0?NtrefE:NtrefI); v[j]=Vthj; s[0+2*ns]=t; s[1+2*ns]=j+1; ns++;
                if(pop==0){
                    int64_t b=(int64_t)seqE[j], e=(int64_t)seqE[j+1];
                    int64_t split = b; /* first E targets then I targets in your data layout */
                    /* We don't have explicit split; if you keep E then I contiguous, set split=b+countE. If not, it still works with weight by post pop: */
                    for(int64_t p=b;p<e;++p){ int post=Wrr[p]-1; if(post<0||post>=N) mexErrMsgIdAndTxt("EIF:Wrr","post");
                        if (post < P.Ne) Isynp[(int64_t)post*NSYN+SY_E]+=Jee; else Isynp[(int64_t)post*NSYN+SY_I]+=Jei; }
                }else{
                    int jj = j - P.Ne;
                    int64_t b=(int64_t)seqI[jj], e=(int64_t)seqI[jj+1];
                    for(int64_t p=b;p<e;++p){ int post=Wrr[p]-1; if(post<0||post>=N) mexErrMsgIdAndTxt("EIF:Wrr","post");
                        if (post < P.Ne) Isynp[(int64_t)post*NSYN+SY_E]+=Jie; else Isynp[(int64_t)post*NSYN+SY_I]+=Jii; }
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
