#ifndef __MEX_EXPORT_H
  #define __MEX_EXPORT_H

  typedef enum {FALSE =0, TRUE = 1} boolean;

  #ifdef _WIN32
    #ifdef EXPORT_FCNS
      #define EXPORTED_FUNCTION __declspec(dllexport)
    #else
      #define EXPORTED_FUNCTION __declspec(dllimport)
    #endif
  #else
    #define EXPORTED_FUNCTION
  #endif

#endif
