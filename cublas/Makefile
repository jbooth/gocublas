# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include $(GOROOT)/src/Make.inc

TARG=cublas


CGOFILES=error.go memory.go cublas1.go

CGO_CFLAGS=-I/usr/local/cuda/include  
CGO_LDFLAGS=-lcublas -lcuda -L/usr/local/cuda/lib64


# To add flags necessary for locating the library or its include files,
# set CGO_CFLAGS or CGO_LDFLAGS.  For example, to use an
# alternate installation of the library:
#	CGO_CFLAGS=-I/home/rsc/gmp32/include
#	CGO_LDFLAGS+=-L/home/rsc/gmp32/lib
# Note the += on the second line.

CLEANFILES+=cublas

include $(GOROOT)/src/Make.pkg

# cublas library
cublas: install main.go
	$(GC) main.go
	$(LD) -o $@ main.$O

