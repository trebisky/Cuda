# Build a cuda simple calculation
# Tom Trebisky  6-19-2025

# We could do this:
## .EXPORT_ALL_VARIABLES:

export CUDAHOSTCXX=/usr/bin/g++-14
export CC=/usr/bin/gcc-14
export CXX=/usr/bin/g++-14
export NVCC_CCBIN=/usr/bin/g++-14

# These exports won't work (and don't seem to be needed).
# They fail because of how the previous values are referenced.
# In lieu of CPATH we could use a -I line on gcc (but not nvcc)
#export LD_LIBRARY_PATH=/usr/local/cuda-12.9/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
##export CPATH=/usr/local/cuda-12.9/targets/x86_64-linux/include:$CPATH
#export PATH=/usr/local/cuda-12.9/bin:$PATH

#NOPTS = -allow-unsupported-compiler -Wno-deprecated-gpu-targets
NOPTS = -Wno-deprecated-gpu-targets

# This nice options doesn't seem to work for us
#NVCC = /usr/local/cuda-12.9/bin/nvcc --std=c++14 $(NOPTS)
NVCC = /usr/local/cuda-12.9/bin/nvcc $(NOPTS)

# A useless experiment when I was having trouble
#  with cudaMallocManaged
XNOPTS = -Wno-deprecated-gpu-targets -arch=sm_61
XNVCC = /usr/local/cuda-12.9/bin/nvcc $(XNOPTS)

all: simple

simple: simple.cu
	$(NVCC) simple.cu -o simple

profile: simple
	#nsys profile -t cuda --stats=true ./simple
	./nsys_easy ./simple

basic: basic.cu
	$(NVCC) basic.cu -o basic

clean:
	rm -f simple *-rep *.sqlite

# THE END
