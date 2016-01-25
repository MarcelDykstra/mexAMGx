#include <cuda.h>
#include <mex.h>
#include <stdbool.h>
#include <stdio.h>
#include <gpu/mxGPUArray.h>
#include "amgx_capi.h"
#define EXPORT_FCNS
#include "mex_export.h"

#define MEX_AMGX_SAFE_CALL(rc) \
{ \
  AMGX_RC err;     \
  char amg_msg[4096];   \
  char err_msg[4096]; \
  switch(err = (rc)) {    \
  case AMGX_RC_OK: \
    break; \
  default: \
    AMGX_get_error_string(err, amg_msg, 4096); \
    sprintf(err_msg, "AMGx: ERROR: %s\n " \
      "file %s line %6d\nCUDA: LAST ERROR: %s\n", \
      amg_msg, __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError())); \
    mexErrMsgIdAndTxt("AMGx:SafeCall", err_msg); \
    AMGX_abort(NULL, 1); \
    break; \
  } \
}

#ifdef __cplusplus
extern "C" {
#endif

void print_amgx(const char *msg, int length);

//------------------------------------------------------------------------------
// Status handling.
AMGX_SOLVE_STATUS status;
bool is_verbose;

// Library handles.
void *lib_handle;
AMGX_Mode mode;
AMGX_config_handle cfg;
AMGX_resources_handle rsrc;
AMGX_matrix_handle A;
AMGX_vector_handle b;
AMGX_vector_handle x = NULL;
AMGX_solver_handle solver;

//------------------------------------------------------------------------------
EXPORTED_FUNCTION void mexAMGxInitialize(const mxArray *cfg_str,
                                         bool is_file, bool is_set_verbose)
{
  int major, minor;
  char *ver, *date, *time;
  unsigned int buf_len;
  char *buf_cfg_str;

  is_verbose = is_set_verbose;

#ifdef _WIN32
  mxInitGPU();
#endif

#ifdef _WIN32
  lib_handle = amgx_libopen("amgxsh.dll");
  if (lib_handle == NULL) {
    mexErrMsgIdAndTxt("AMGx:LibOpen",
      "AMGx: ERROR: failed loading, with system error code: %d\n",
              GetLastError());
    return;
  }
#else
  lib_handle = amgx_libopen("./libamgxsh.so");
  if (lib_handle == NULL) {
    mexErrMsgIdAndTxt("AMGx:LibOpen",
      "AMGx: ERROR: failed loading: %s\n", dlerror());
    return;
  }
#endif

  // Load all the dynamic link library routines.
  if (amgx_liblink_all(lib_handle) == 0) {
    amgx_libclose(lib_handle);
    mexErrMsgIdAndTxt("AMGx:LibLink",
      "AMGx: ERROR: corrupted library loaded.\n");
  }

  // Init.
  MEX_AMGX_SAFE_CALL(AMGX_initialize());
  MEX_AMGX_SAFE_CALL(AMGX_initialize_plugins());

  // System.
  MEX_AMGX_SAFE_CALL(AMGX_register_print_callback(&print_amgx));
  MEX_AMGX_SAFE_CALL(AMGX_install_signal_handler());

  // Get API and build info.
  AMGX_get_api_version(&major, &minor);
  if (is_verbose) mexPrintf("AMGx: API version: %d.%d\n", major, minor);
  AMGX_get_build_info_strings(&ver, &date, &time);
  if (is_verbose) mexPrintf("AMGx: build version: %s\nAMGx: " \
                            "build date and time: %s %s\n",
                            ver, date, time);

  // Set mode.
  mode = AMGX_mode_dDDI;

  // Create config.
  buf_len = mxGetN(cfg_str) * sizeof(mxChar) + 1;
  buf_cfg_str = (char *) mxMalloc(buf_len);
  mxGetString(cfg_str, buf_cfg_str, (mwSize)buf_len);

  if (is_file) {
    MEX_AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg,
                   buf_cfg_str));
  }
  else {
    MEX_AMGX_SAFE_CALL(AMGX_config_create(&cfg,
                   buf_cfg_str));
  }

  // Create resources, matrix, vector and solver.
  MEX_AMGX_SAFE_CALL(AMGX_resources_create_simple(&rsrc, cfg));
  MEX_AMGX_SAFE_CALL(AMGX_matrix_create(&A, rsrc, mode));
  MEX_AMGX_SAFE_CALL(AMGX_vector_create(&x, rsrc, mode));
  MEX_AMGX_SAFE_CALL(AMGX_vector_create(&b, rsrc, mode));
  MEX_AMGX_SAFE_CALL(AMGX_solver_create(&solver, rsrc, mode, cfg));

  mxFree(buf_cfg_str);
}

//------------------------------------------------------------------------------
EXPORTED_FUNCTION void mexAMGxMatrixUploadA(const mxArray *mxA)
{
  size_t n;
  mwIndex *jc;
  mwIndex *ir;
  int *amgx_jc;
  int *amgx_ir;
  double *pr;
  int nnz;

  n = mxGetM(mxA);
  if (!mxIsSparse(mxA) || mxGetN(mxA) != mxGetM(mxA) ||
      n == 0 || mxIsComplex(mxA) || !mxIsDouble(mxA)) {
    mexErrMsgIdAndTxt("mexAMGx:mexAMGxMatrixUpload",
                      "mexAMGx: ERROR: bad matrix.");
  }

  jc = mxGetJc(mxA);
  ir = mxGetIr(mxA);
  pr = mxGetPr(mxA);
  nnz = (int) jc[n];

  // Cast array from 'mwIndex' to 'int'.
  amgx_jc = new int[n + 1];
  for (int j = 0; j <= n; j++) {
    amgx_jc[j] = (int) jc[j];
  }
  MEX_AMGX_SAFE_CALL(AMGX_pin_memory(amgx_jc, (n + 1) * sizeof(int)));

  // Cast array from 'mwIndex' to 'int'.
  amgx_ir = new int[nnz];
  for (int j = 0; j < nnz; j++) {
    amgx_ir[j] = (int) ir[j];
  }
  MEX_AMGX_SAFE_CALL(AMGX_pin_memory(amgx_ir, nnz * sizeof(int)));

  MEX_AMGX_SAFE_CALL(AMGX_matrix_upload_all(A, n, nnz, 1, 1,
                                        amgx_jc, amgx_ir, pr, NULL));
  MEX_AMGX_SAFE_CALL(AMGX_unpin_memory(amgx_jc));
  MEX_AMGX_SAFE_CALL(AMGX_unpin_memory(amgx_ir));
  delete amgx_jc;
  delete amgx_ir;
}

//------------------------------------------------------------------------------
EXPORTED_FUNCTION void mexAMGxMatrixReplaceCoeffA(const mxArray *mxA)
{
  size_t n;
  mwIndex *jc;
  double *pr;
  int nnz;

  n = mxGetM(mxA);
  if (!mxIsSparse(mxA) || mxGetN(mxA) != mxGetM(mxA) ||
      n == 0 || mxIsComplex(mxA) || !mxIsDouble(mxA)) {
    mexErrMsgIdAndTxt("mexAMGx:mexAMGxMatrixReplaceCoeff",
                      "mexAMGx: ERROR: bad matrix.");
  }

  jc = mxGetJc(mxA);
  pr = mxGetPr(mxA);
  nnz = jc[n];

  MEX_AMGX_SAFE_CALL(AMGX_matrix_replace_coefficients(A, n, nnz, pr, NULL));
}

//------------------------------------------------------------------------------
EXPORTED_FUNCTION void mexAMGxVectorUploadX(const mxArray *mxX)
{
  size_t n;
  double *pr;

  n = mxGetM(mxX);
  if (mxIsSparse(mxX) || mxGetN(mxX) != 1 || n == 0 || mxIsComplex(mxX) ||
      !mxIsDouble(mxX)) {
    mexErrMsgIdAndTxt("mexAMGx:mexAMGxVectorUploadX",
                      "mexAMGx: ERROR: bad vector.");
  }

  pr = mxGetPr(mxX);

  MEX_AMGX_SAFE_CALL(AMGX_vector_upload(x, n, 1, pr));
}

//------------------------------------------------------------------------------
EXPORTED_FUNCTION void mexAMGxVectorUploadB(const mxArray *mxB)
{
  size_t n;
  double *pr;

  n = mxGetM(mxB);
  if (mxIsSparse(mxB) || mxGetN(mxB) != 1 || n == 0 || mxIsComplex(mxB) ||
      !mxIsDouble(mxB)) {
    mexErrMsgIdAndTxt("mexAMGx:mexAMGxVectorUploadX",
                      "mexAMGx: ERROR: bad vector.");
  }

  pr = mxGetPr(mxB);

  MEX_AMGX_SAFE_CALL(AMGX_vector_upload(b, n, 1, pr));
}

//------------------------------------------------------------------------------
EXPORTED_FUNCTION void mexAMGxVectorSetZeroX(void)
{
  int n, block_dim;

  if (x == NULL) return;
  MEX_AMGX_SAFE_CALL(AMGX_vector_get_size(x, &n, &block_dim));
  MEX_AMGX_SAFE_CALL(AMGX_vector_set_zero(x, n, block_dim));
}

//------------------------------------------------------------------------------
EXPORTED_FUNCTION mxArray *mexAMGxVectorDownloadX(void)
{
  int n, block_dim;
  double *data;
  mxArray *mxX;

  MEX_AMGX_SAFE_CALL(AMGX_vector_get_size(x, &n, &block_dim));
  mxX = mxCreateNumericMatrix(n * block_dim, 1, mxDOUBLE_CLASS, mxREAL);
  data = (double *) mxMalloc(n * block_dim * sizeof(double));
  MEX_AMGX_SAFE_CALL(AMGX_vector_download(x, data));
  mxSetPr(mxX, data);
  return mxX;
}

//------------------------------------------------------------------------------
EXPORTED_FUNCTION void mexAMGxSolverSetup(void)
{
  // Solver setup.
  MEX_AMGX_SAFE_CALL(AMGX_solver_setup(solver, A));
}

//------------------------------------------------------------------------------
EXPORTED_FUNCTION void mexAMGxSolverSolve(void)
{
  // Solver solve.
  MEX_AMGX_SAFE_CALL(AMGX_solver_solve(solver, b, x));
}

//------------------------------------------------------------------------------
EXPORTED_FUNCTION void mexAMGxFinalize(void)
{
  // Destroy resources, matrix, vector and solver.
  MEX_AMGX_SAFE_CALL(AMGX_solver_destroy(solver));
  MEX_AMGX_SAFE_CALL(AMGX_vector_destroy(x));
  MEX_AMGX_SAFE_CALL(AMGX_vector_destroy(b));
  MEX_AMGX_SAFE_CALL(AMGX_matrix_destroy(A));
  MEX_AMGX_SAFE_CALL(AMGX_resources_destroy(rsrc));

  // Destroy config.
  MEX_AMGX_SAFE_CALL(AMGX_config_destroy(cfg));

  // Shutdown and exit.
  MEX_AMGX_SAFE_CALL(AMGX_finalize_plugins());
  MEX_AMGX_SAFE_CALL(AMGX_finalize());

  amgx_libclose(lib_handle);
  return;
}

//------------------------------------------------------------------------------
void print_amgx(const char *msg, int length){
  if (is_verbose) mexPrintf("AMGx: %s", msg);
}

//------------------------------------------------------------------------------
void mexFunction(int nlhs, mxArray * plhs[],
                 int nrhs,const mxArray * prhs[])
{
  mexPrintf("mexAMGx: Use loadlibrary.\n");
  return;
}

#ifdef __cplusplus
}
#endif
