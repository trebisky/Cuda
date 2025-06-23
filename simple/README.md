simple

My first project to actually do "something" on the GPU.

"Something" is to add the contents of one vector onto another, i.e.

X += Y

where X and Y are vectors of float values of size 1<<20 (1M).

This is more or less following "An Even Easier Introduction to CUDA"
which you can find online (an Nvidia Developer article).

basic.cu is verbatim code from "An Even Easier Introduction to CUDA"
that I found useful when I was having strange issues with
cudaMallocManaged()

I see 103,733,432 ns for the 1,1 GPU first case
