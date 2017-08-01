#ifndef __MEX_EXPORT_H
    #define __MEX_EXPORT_H

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
