
// Gaussian input current variant: adds me_current/mi_current and time series L_current(floor(t))
// Signature preserved: [s, Isyn_record, v_record] = EIF1DRFfastslowSynAtttSpatRec_CurrentGaussianNoise(sx, Wrf, Wrr, params, L_current)
#include "EIF_common.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    ERROR_CHECK(nrhs == 5, "Expected 5 inputs: sx, Wrf, Wrr, params, L_current");
    ERROR_CHECK(nlhs <= 3, "Up to 3 outputs supported");

    ERROR_CHECK(mxGetM(prhs[0]) == 2, "sx must be 2xNsx");
    double* sx = mxGetPr(prhs[0]);
    int32_t Nsx = (int32_t)mxGetN(prhs[0]);

    ERROR_CHECK(mxGetN(prhs[1]) == 1, "Wrf must be column vector");
    ERROR_CHECK(mxGetN(prhs[2]) == 1, "Wrr must be column vector");
    int32_t* Wrf = (int32_t*)mxGetData(prhs[1]);
    int32_t* Wrr = (int32_t*)mxGetData(prhs[2]);
    int32_t Nwrf = (int32_t)mxGetM(prhs[1]);
    int32_t Nwrr = (int32_t)mxGetM(prhs[2]);

    ERROR_CHECK(mxIsStruct(prhs[3]), "params must be struct");

    NetworkParams P; eif_extract_network_params(prhs[3], &P);
    P.use_gaussian_input = 1;

    // L_current as column vector (Nt1 x 1) double
    ERROR_CHECK(mxGetN(prhs[4]) == 1, "L_current must be column vector");
    P.L_current = mxGetPr(prhs[4]);

    SynapseParams S; eif_extract_synapse_params(prhs[3], &S);

    const mxArray* f = mxGetField(prhs[3],0,"Irecord");
    ERROR_CHECK(f && mxGetM(f)==1, "Irecord must be 1xN");
    double* Irecord = mxGetPr(f);
    int32_t Nrecord = (int32_t)mxGetN(f);

    f = mxGetField(prhs[3],0,"V0");
    ERROR_CHECK(f && (int32_t)mxGetNumberOfElements(f) == P.Ne + P.Ni, "Invalid V0 size");
    double* V0 = mxGetPr(f);

    int32_t Nt = (int32_t)(P.T / P.dt);

    plhs[0] = mxCreateDoubleMatrix(2, P.maxns, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(Nrecord * S.Nsyn, Nt, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(Nrecord, Nt, mxREAL);

    eif_simulate_core(
        sx, Nsx, Wrf, Nwrf, Wrr, Nwrr,
        &P, &S,
        V0, Irecord, Nrecord,
        /*rec_cols=*/S.Nsyn,
        mxGetPr(plhs[0]), mxGetPr(plhs[1]), mxGetPr(plhs[2])
    );

    eif_free_synapse_params(&S);
}
