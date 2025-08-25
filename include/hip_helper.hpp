#pragma once
// #include <rccl/rccl.h>  // RCCL

// #define CHECK_RCCL(call)                                                       \
//   do {                                                                         \
//     rcclResult_t status_ = call;                                               \
//     if (status_ != ncclSuccess && status_ != ncclInProgress) {                 \
//       fprintf(stderr, "NCCL error (%s:%d): %s\n", __FILE__, __LINE__,          \
//               ncclGetErrorString(status_));                                    \
//       exit(EXIT_FAILURE);                                                      \
//     }                                                                          \
//   } while (0)

#define MAX_NUM_SUPPORTED_GPUS 32
#define WARP_SIZE 64
#define MAX_BLOCK_SIZE 1024
