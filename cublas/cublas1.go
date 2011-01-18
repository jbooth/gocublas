package cublas

// #include <cublas.h>
import "C"
import "os"

func floatPtr(d DevicePointer) *_Ctype_float {
	return ((*_Ctype_float)(d.data()))
}
// BLAS 1 operations, vector-vector and vector-scalar

func CublasSasum(n int,  x DevicePointer, incx int) (float,os.Error) {
	return float(C.cublasSasum(
		_Ctype_int(n),
		floatPtr(x),
		_Ctype_int(incx))),LastErr()
}

func CublasSdot (n int, x DevicePointer, incx int, y DevicePointer, incy int) (float,os.Error) {
	return float(C.cublasSdot(
		_Ctype_int(n),
		floatPtr(x),
		_Ctype_int(incx),
		floatPtr(y),
		_Ctype_int(incy))),LastErr()
}

func CublasSaxpy (n int, alpha float, x DevicePointer, incx int, y DevicePointer, incy int) os.Error {
	C.cublasSaxpy(
		_Ctype_int(n),
		_Ctype_float(alpha),
		floatPtr(x),
		_Ctype_int(incx),
		floatPtr(y),
		_Ctype_int(incy))
	return LastErr()
}
