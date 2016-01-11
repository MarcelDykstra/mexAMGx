#include "mex_export.h"
#define EXTERN_C EXPORTED_FUNCTION
#include <mex.h>

#ifdef __cplusplus
extern "C" {
#endif

EXPORTED_FUNCTION void mexAMGxInitialize(const mxArray *cfg_str,
                                         bool is_file);
EXPORTED_FUNCTION void mexAMGxFinalize(void);

EXPORTED_FUNCTION void mexAMGxMatrixUploadA(const mxArray *mxA);
EXPORTED_FUNCTION void mexAMGxMatrixReplaceCoeffA(const mxArray *mxA);
EXPORTED_FUNCTION void mexAMGxVectorUploadX(const mxArray *mxX);
EXPORTED_FUNCTION void mexAMGxVectorUploadB(const mxArray *mxB);
EXPORTED_FUNCTION void mexAMGxVectorSetZeroX(void);
EXPORTED_FUNCTION mxArray *mexAMGxVectorDownloadX(void);

EXPORTED_FUNCTION void mexAMGxSolverSetup(void);
EXPORTED_FUNCTION void mexAMGxSolverSolve(void);

EXPORTED_FUNCTION void mexFunction(int nlhs, mxArray *plhs[],
                                   int nrhs, const mxArray *prhs[]);

#ifdef __cplusplus
}
#endif
