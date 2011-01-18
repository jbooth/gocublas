package cublas
// memory management

// #include <cublas.h>
// #include <cuda.h>
import "C"

import (
	"unsafe"
	"os"
	"reflect"
)

var(
 floatSize              _Ctype_int       = _Ctype_int(unsafe.Sizeof(float(1)))
 doubleSize             _Ctype_int       = _Ctype_int(unsafe.Sizeof(float64(1)))
 floatSliceType = reflect.Typeof(([]float)(nil))
)

func Init() {
  var stat = C.cublasInit()
  if (stat != CUBLAS_STATUS_SUCCESS) {
    panic("cublasInit() had error " + CublasErr(stat).String())
  }
  //fmt.Printf("sync: " + CudaErrT(C.cudaThreadSynchronize()).String())
}

type Vector interface {
	data() unsafe.Pointer
	Len() int
	elemSize() int
}

// Vector of floats in host memory.  Pinned for optimal transfer speed with GPU.  
type VectorFH struct {
	Ptr unsafe.Pointer // ptr to pinned memory
	Slice []float // Slice representation of _Ctype_ints, could be broken on your platform
	Length int // length of pinned memory section
}

func (v VectorFH)  elemSize() int { return int(floatSize) }
func (v VectorFH) data() unsafe.Pointer { return v.Ptr }
func (v VectorFH) Len() int { return v.Length }

func (v VectorFH) Free() os.Error {
	return CudaErr(C.cuMemFreeHost(v.Ptr))
}

// allocates a vector of host pinned memory.  Must be Free()d
func MallocVecFH(length int) (VectorFH,os.Error) {
	// indirect pointer we'll copy alloc'd address to
        var ptrptr *unsafe.Pointer = new(unsafe.Pointer)
	var err = CudaErr(C.cuMemAllocHost(ptrptr,_Ctypedef_size_t(length)))
	// dereference - this is our memory now
	var ptr = (*ptrptr)
	// wrap in Slice
	var h = reflect.SliceHeader{uintptr(ptr), length, length}
        var Slice []float =unsafe.Unreflect(floatSliceType, unsafe.Pointer(&h)).([]float)
	
	return VectorFH { ptr, Slice, length }, err
}


type DevicePointer struct {
	Ptr unsafe.Pointer
	Length int
}

func (d DevicePointer) data() unsafe.Pointer { return d.Ptr }
func (d DevicePointer) Len() int { return d.Length }

func (d DevicePointer) Free() os.Error {
	return CublasErr(C.cublasFree(d.data()))
}

func (d DevicePointer) CopyFrom(host Vector) os.Error {
        if (host.Len() != d.Length) { 
                return os.NewError("Host and device length not equal") 
        }
        var stat = C.cublasSetVector(
                                _Ctype_int(host.Len()),
                                _Ctype_int(host.elemSize()), 
                                host.data(),
                                _Ctype_int(1), 
                                d.data(),
                                _Ctype_int(1))
	return CublasErr(stat)
}

func (d DevicePointer) CopyTo(host Vector) os.Error {
        if (host.Len() != d.Length) { 
                return os.NewError("Host and device length not equal") 
        }
        var stat = C.cublasGetVector(
                                _Ctype_int(host.Len()),
                                _Ctype_int(host.elemSize()), 
                                d.data(),
                                _Ctype_int(1), 
                                host.data(),
                                _Ctype_int(1))
        return CublasErr(stat)
}

// mallocs a vector of floats on the device
func MallocDevicePointerVF(length int) (DevicePointer, os.Error) {
        var ptrptr *unsafe.Pointer = new(unsafe.Pointer)
        var err = CublasErr(C.cublasAlloc(_Ctype_int(length), floatSize, ptrptr,))
        var ptr = (*ptrptr)
        return DevicePointer { ptr, length }, err
}
