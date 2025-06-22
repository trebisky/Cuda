# Build a cuda hello world
# Tom Trebisky  6-19-2025
# Hot! these days. over 110 yesterday.

# We could do this:
## .EXPORT_ALL_VARIABLES:

export CUDAHOSTCXX=/usr/bin/g++-14
export CC=/usr/bin/gcc-14
export CXX=/usr/bin/g++-14
export NVCC_CCBIN=/usr/bin/g++-14

# These won't work (and don't seem to be needed).
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

all: hello

hello: hello.cu
	$(NVCC) hello.cu -o hello

clean:
	rm -f hello

# THE END
