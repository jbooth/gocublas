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

// represents a region of memory
type Memory struct {
	ptr unsafe.Pointer
	Length int
	elemSize _Ctype_int
}

// functions to allocate/manage pinned memory for fast transfer with GPU
// Vector and Matrix types expose slices
type HostMemory struct {
	Memory
}
func (v HostMemory) Free() {
	err := CudaErr(C.cuMemFreeHost(v.ptr))
	if (err != SUCCESS) { panic(err.String()) }
}

func (v HostMemory) sliceFloat() []float {
	h := reflect.SliceHeader{uintptr(v.ptr), v.Length, v.Length}
        return unsafe.Unreflect(floatSliceType, unsafe.Pointer(&h)).([]float)
}

func mallocHostPinned(length int, elemSize _Ctype_int) HostMemory {
	// indirect pointer we'll copy alloc'd address to
        var ptrptr *unsafe.Pointer = new(unsafe.Pointer)
	// malloc length * elemSize bytes
        var err = CudaErr(C.cuMemAllocHost(ptrptr,_Ctypedef_size_t(length * int(elemSize))))
        if (err != SUCCESS) { panic (err.String()) }
        // dereference - this is our memory now
        ptr := (*ptrptr)
        // wrap in HostMemory
        return HostMemory { Memory {ptr, length,elemSize }}
}

type VectorFH struct {
	HostMemory
	Slice []float // Slice representation of _Ctype_floats, could be broken on your platform
}

// allocates a vector of host pinned memory.  Must be Free()d
func MallocVecFH(length int) (VectorFH) {
	hp := mallocHostPinned ( length , floatSize )
	return VectorFH { hp, hp.sliceFloat() }
}

type MatrixFH struct {
	HostMemory
	Slice [][]float
	Width int
	Height int
}

//func MallocMatFH(width int, height int) {
	//length int64 := width*height
	//HostMemory mem := mallocHostPinned(length, floatSize)
	//Slice [][]float :=make([][]float,10
//}

type DevicePointer struct {
	Memory
}

func (d DevicePointer) Free() {
	err := CublasErr(C.cublasFree(d.ptr))
	if (err != SUCCESS) { panic ( err.String()) }
}

func (d DevicePointer) CopyVectorFrom(host Memory) {
        if (host.Length != d.Length) { 
                panic("Host and device length not equal") 
        }
        err := CublasErr(C.cublasSetVector(
                                _Ctype_int(host.Length),
                                _Ctype_int(host.elemSize), 
                                host.ptr,
                                _Ctype_int(1), 
                                d.ptr,
                                _Ctype_int(1)))
	if (err != SUCCESS) { panic(err.String()) }
}

func (d DevicePointer) CopyVectorTo(host Memory) {
        if (host.Length != d.Length) { 
                panic("Host and device length not equal") 
        }
        err := CublasErr(C.cublasGetVector(
                                _Ctype_int(host.Length),
                                _Ctype_int(host.elemSize), 
                                d.ptr,
                                _Ctype_int(1), 
                                host.ptr,
                                _Ctype_int(1)))
        if (err != SUCCESS) { panic(err.String()) }
}

// mallocs a vector of length elements at elemBytes each on the device
func mallocDevice(length int, elemSize int) (DevicePointer, os.Error) {
	// indirect pointer
        var ptrptr *unsafe.Pointer = new(unsafe.Pointer)
	// malloc
        var err = CublasErr(C.cublasAlloc(_Ctype_int(length), _Ctype_int(elemSize), ptrptr))
	// dereference indirect pointer for direct pointer to our new memory
        var ptr = (*ptrptr)
        return DevicePointer { Memory { ptr, length, _Ctype_int(elemSize) }}, err
}
