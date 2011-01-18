package cublas

// structs related to memory management

// #include <cublas.h>
// #include <cuda.h>
// #include <cuda_runtime_api.h>
// #include <stdlib.h>
import "C"

import (
	"os"
	"strconv"
)

const(
 // cublas errors
 CUBLAS_STATUS_SUCCESS           = 0x00000000
 CUBLAS_STATUS_NOT_INITIALIZED   = 0x00000001
 CUBLAS_STATUS_ALLOC_FAILED      = 0x00000003
 CUBLAS_STATUS_INVALID_VALUE     = 0x00000007
 CUBLAS_STATUS_ARCH_MISMATCH     = 0x00000008
 CUBLAS_STATUS_MAPPING_ERROR     = 0x0000000B
 CUBLAS_STATUS_EXECUTION_FAILED  = 0x0000000D
 CUBLAS_STATUS_INTERNAL_ERROR    = 0x0000000E
 // CUDA errors
/*
     * The API call returned with no errors. In the case of query calls, this
     * can also mean that the operation being queried is complete (see
     * ::cuEventQuery() and ::cuStreamQuery()).
     */
    CUDA_SUCCESS                              = 0

    /*
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    CUDA_ERROR_INVALID_VALUE                  = 1

    /*
     * The API call failed because it was unable to allocate enough memory to
     * perform the requested operation.
     */
    CUDA_ERROR_OUT_OF_MEMORY                  = 2

    /*
     * This indicates that the CUDA driver has not been initialized with
     * ::cuInit() or that initialization has failed.
     */
    CUDA_ERROR_NOT_INITIALIZED                = 3

    /*
     * This indicates that the CUDA driver is in the process of shutting down.
     */
    CUDA_ERROR_DEINITIALIZED                  = 4


    /*
     * This indicates that no CUDA-capable devices were detected by the installed
     * CUDA driver.
     */
    CUDA_ERROR_NO_DEVICE                      = 100

    /*
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid CUDA device.
     */
    CUDA_ERROR_INVALID_DEVICE                 = 101


    /*
     * This indicates that the device kernel image is invalid. This can also
     * indicate an invalid CUDA module.
     */
    CUDA_ERROR_INVALID_IMAGE                  = 200

    /*
     * This most frequently indicates that there is no context bound to the
     * current thread. This can also be returned if the context passed to an
     * API call is not a valid handle (such as a context that has had
     * ::cuCtxDestroy() invoked on it). This can also be returned if a user
     * mixes different API versions (i.e. 3010 context with 3020 API calls).
     * See ::cuCtxGetApiVersion() for more details.
     */
    CUDA_ERROR_INVALID_CONTEXT                = 201

    /*
     * This indicated that the context being supplied as a parameter to the
     * API call was already the active context.
     * \deprecated
     * This error return is deprecated as of CUDA 3.2. It is no longer an
     * error to attempt to push the active context via ::cuCtxPushCurrent().
     */
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT        = 202

    /*
     * This indicates that a map or register operation has failed.
     */
    CUDA_ERROR_MAP_FAILED                     = 205

    /*
     * This indicates that an unmap or unregister operation has failed.
     */
    CUDA_ERROR_UNMAP_FAILED                   = 206

    /*
     * This indicates that the specified array is currently mapped and thus
     * cannot be destroyed.
     */
    CUDA_ERROR_ARRAY_IS_MAPPED                = 207

    /*
     * This indicates that the resource is already mapped.
     */
    CUDA_ERROR_ALREADY_MAPPED                 = 208

    /*
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular CUDA source file that do not include the
     * corresponding device configuration.
     */
    CUDA_ERROR_NO_BINARY_FOR_GPU              = 209

    /*
     * This indicates that a resource has already been acquired.
     */
    CUDA_ERROR_ALREADY_ACQUIRED               = 210

    /*
     * This indicates that a resource is not mapped.
     */
    CUDA_ERROR_NOT_MAPPED                     = 211

    /*
     * This indicates that a mapped resource is not available for access as an
     * array.
     */
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY            = 212

    /*
     * This indicates that a mapped resource is not available for access as a
     * pointer.
     */
    CUDA_ERROR_NOT_MAPPED_AS_POINTER          = 213

    /*
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     */
    CUDA_ERROR_ECC_UNCORRECTABLE              = 214

    /*
     * This indicates that the ::CUlimit passed to the API call is not
     * supported by the active device.
     */
    CUDA_ERROR_UNSUPPORTED_LIMIT              = 215


    /*
     * This indicates that the device kernel source is invalid.
     */
    CUDA_ERROR_INVALID_SOURCE                 = 300

    /*
     * This indicates that the file specified was not found.
     */
    CUDA_ERROR_FILE_NOT_FOUND                 = 301

    /*
     * This indicates that a link to a shared object failed to resolve.
     */
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302

    /*
     * This indicates that initialization of a shared object failed.
     */
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      = 303

    /*
     * This indicates that an OS call failed.
     */
    CUDA_ERROR_OPERATING_SYSTEM               = 304


    /*
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::CUstream and ::CUevent.
     */
    CUDA_ERROR_INVALID_HANDLE                 = 400


    /*
     * This indicates that a named symbol was not found. Examples of symbols
     * are global/constant variable names, texture names, and surface names.
     */
    CUDA_ERROR_NOT_FOUND                      = 500


    /*
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::CUDA_SUCCESS (which indicates completion). Calls that
     * may return this value include ::cuEventQuery() and ::cuStreamQuery().
     */
    CUDA_ERROR_NOT_READY                      = 600


    /*
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. The context cannot be used, so it must
     * be destroyed (and a new one should be created). All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using CUDA.
     */
    CUDA_ERROR_LAUNCH_FAILED                  = 700

    /*
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. This error usually indicates that the user has
     * attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernels register
     * count. Passing arguments of the wrong size (i.e. a 64-bit pointer
     * when a 32-bit int is expected) is equivalent to passing too many
     * arguments and can also result in this error.
     */
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        = 701

    /*
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device attribute
     * ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. The
     * context cannot be used (and must be destroyed similar to
     * ::CUDA_ERROR_LAUNCH_FAILED). All existing device memory allocations from
     * this context are invalid and must be reconstructed if the program is to
     * continue using CUDA.
     */
    CUDA_ERROR_LAUNCH_TIMEOUT                 = 702

    /*
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     */
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  = 703


    /*
     * This indicates that an unknown internal error has occurred.
     */
    CUDA_ERROR_UNKNOWN                        = 999
 
)
var(
 SUCCESS        	os.Error = os.NewError("SUCCESS")
 NOT_INITIALIZED       	os.Error = os.NewError("CUBLAS NOT INITIALIZED")
 ALLOC_FAILED        	os.Error = os.NewError("CUBLAS ALLOC FAILED")
 INVALID_VALUE        	os.Error = os.NewError("CUBLAS INVALID VALUE")
 MAPPING_ERROR        	os.Error = os.NewError("CUBLAS MAPPING ERROR")
 EXECUTION_FAILED     	os.Error = os.NewError("CUBLAS EXECUTION FAILED")
 INTERNAL_ERROR		os.Error = os.NewError("CUBLAS INTERNAL ERROR")
)


// returns the last error
func LastErr() os.Error {
	return CublasErr(C.cublasGetError())
}

func CublasErr(stat C.cublasStatus) os.Error {
  switch stat {
   case CUBLAS_STATUS_SUCCESS: return SUCCESS
   case CUBLAS_STATUS_NOT_INITIALIZED: return NOT_INITIALIZED
   case CUBLAS_STATUS_ALLOC_FAILED: return ALLOC_FAILED
   case CUBLAS_STATUS_INVALID_VALUE: return INVALID_VALUE
   case CUBLAS_STATUS_MAPPING_ERROR: return MAPPING_ERROR
   case CUBLAS_STATUS_EXECUTION_FAILED: return EXECUTION_FAILED
   case CUBLAS_STATUS_INTERNAL_ERROR: return INTERNAL_ERROR
  }
  // default
  return os.Error(nil)
}

func CudaErrT(stat C.cudaError_t) os.Error {
	if (stat == 0) { return SUCCESS }
	return os.NewError("Cuda Err: " + strconv.Itoa(int(stat)))
}

func CudaErr(stat C.CUresult) os.Error {
	if (stat == 0) { return SUCCESS }
        return os.NewError("Cuda Err: " + strconv.Itoa(int(stat)))
}

