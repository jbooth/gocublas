package main

import "cublas"
import "fmt"
import "os"

func checkErr(err os.Error) {
	if (err != cublas.SUCCESS) { panic(err.String()) }
}

func main() {
	cublas.Init()
	var length int = 5
	// set up memory
	h_x, err := cublas.MallocVecFH(length)
	checkErr(err)
	defer h_x.Free()
	h_y, err := cublas.MallocVecFH(length)
	checkErr(err)
	defer h_y.Free()
	d_x,err := cublas.MallocDevicePointerVF(length)
	checkErr(err)
	defer d_x.Free()
	d_y,err := cublas.MallocDevicePointerVF(length)
	checkErr(err)
	defer d_y.Free()


	var i int
	for i = 0 ; i < length ; i++ {
		h_x.Slice[i] = float(i+1)
		h_y.Slice[i] = float(2*i + 1)
	}
	fmt.Println("x: ", h_x.Slice)
	fmt.Println("y: ", h_y.Slice)
	err = d_x.CopyFrom(h_x)
	checkErr(err)
	err = d_y.CopyFrom(h_y)
	checkErr(err)
	
	dotp,err := cublas.CublasSdot(length, d_x, 1, d_y, 1)
	checkErr(err)
	fmt.Println("Dot Product: ", dotp)	

	sum_x,err := cublas.CublasSasum(length, d_x, 1)
	checkErr(err)
	fmt.Println("Sum of x: ", sum_x)

	fmt.Println("y = y + x * 3.14..")
	err = cublas.CublasSaxpy(length, 3.14, d_x, 1, d_y, 1)
	checkErr(err)
	d_y.CopyTo(h_y)
	fmt.Println("y: ", h_y.Slice)

}
