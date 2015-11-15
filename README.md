Matlab interface for NVidia AMGx
================================

Obtain NVidia AMGx
------------------
* Register as
  [CUDA registered developer](https://developer.nvidia.com/cuda-registered-developer-program).
* Read the NVidia Software License Agreement (AmgX Pre-Release Software).
* Download NVidia AMGx for the target platform.
* Download NVidia AMGx trial license key.

Overlay NVidia AMGx
-------------------
After cloning this repository, before compiling the Matlab interface,
it is required to overlay NVidia AMGx. From the extracted
download:
* Copy the `\configs` directory to the repository directory `\`.
* Copy the NVidia AMGx dynamic-link library (Windows: `amgxsh.dll` ) to the
  repository directory `\`.
* Copy `amgx_capi.h` and `amgx_config.h` to the repository
  source-directory `\src`.

Match NVidia CUDA Runtime Libraries
-----------------------------------
When NVidia AMGx has dependency on lower CUDA version then the installed CUDA
Toolkit it is possible to copy and rename required dependencies
(make symbolic links on Linux):
* Copy `cublas64_??.dll` and `cusparse64_??.dll` in `Program Files\NVidia GPU
Computing Toolkit\CUDA\v?.?\bin` to the repository directory `\`.
* Rename the version number in `_??.dll` to the version NVidia AMGx depends on;
assuming hardly any change in the ABI (application binary interface) of
CuBLAS and CuSPARSE.

Safe Error Messages
-------------------
In `\src\amgx_capi.h` overwrite `AMGX_SAFE_CALL`, giving proper NVidia AMGx error
messages in Matlab without seg-faults:
```c
#define AMGX_SAFE_CALL(rc) \
{ \
  AMGX_RC err;     \
  char msg[4096];   \
  switch(err = (rc)) {    \
  case AMGX_RC_OK: \
    break; \
  default: \
    AMGX_get_error_string(err, msg, 4096); \
    mexErrMsgIdAndTxt("AMGx:SafeCall", "AMGx: ERROR: %s\n " \
                      "file %s line %6d\nCUDA: LAST ERROR: %s\n", \
      msg, __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError())); \
    AMGX_abort(NULL, 1); \
    break; \
  } \
}
```

Configure License
-----------------
On Windows open `System Properties` and click `Environment Variables`.
Add `LM_LICENSE_FILE` and provide the pathname to the downloaded license file.

Compile and Validate
--------------------
In Matlab run `compile.m` in the repository source directory `\src`.
Read further instructions inside `compile.m`. After compilation
run `test_mex_amgx.m` in the `\validate` directory.
