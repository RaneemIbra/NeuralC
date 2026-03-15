#ifndef NEURALC_COMMON_H
#define NEURALC_COMMON_H

#include <stdio.h>

/**
 * Global Status Codes for Error Handling
 */
typedef enum {
    NC_SUCCESS = 0,
    NC_ERROR_OUT_OF_MEMORY,
    NC_ERROR_SHAPE_MISMATCH,
    NC_ERROR_INVALID_DTYPE,
    NC_ERROR_NDIM_EXCEEDED,
    NC_ERROR_NULL_POINTER,
    NC_ERROR_INTERNAL
} NcStatus;

/**
 * Simple Logger / Error Reporter
 */
static inline const char* nc_status_to_string(NcStatus status) {
    switch (status) {
        case NC_SUCCESS:            return "Success";
        case NC_ERROR_OUT_OF_MEMORY: return "Out of Memory";
        case NC_ERROR_SHAPE_MISMATCH: return "Shape Mismatch";
        case NC_ERROR_INVALID_DTYPE: return "Invalid Data Type";
        case NC_ERROR_NDIM_EXCEEDED: return "Number of Dimensions Exceeded";
        case NC_ERROR_NULL_POINTER:  return "Null Pointer Encountered";
        case NC_ERROR_INTERNAL:      return "Internal Engine Error";
        default:                    return "Unknown Error";
    }
}

#define NC_REPORT_ERROR(status) fprintf(stderr, "[NeuralC Error] %s at %s:%d\n", nc_status_to_string(status), __FILE__, __LINE__)

#endif //NEURALC_COMMON_H
