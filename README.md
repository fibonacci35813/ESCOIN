# Efficient_Sparse_CNN_Inference

## Introduction
In sparse linear algebra, sparse matrix-vector multiplication is critical.
Sparse operations, in contrast to dense linear algebra's uniform regularity,
deal with a wide range of matrices, from regular to severely irregular. DNNs use a significant amount of storage, memory bandwidth, and
compute resources as their model sizes grow.
Weight pruning has been proposed to address this constraint by
compressing DNN models by deleting superfluous connections in the
networks. However, while pruning can dramatically reduce model size by
deleting an average of 80% of weights, it actually degrades inference
performance (i.e. speed) when using GPUs to execute CNN models.

First, as the calculation becomes sparse after pruning, the expense of
lowering convolution onto matrix multiplication becomes a serious issue.
The lowering method has shown overhead for dense convolution since it
repeats input features several times, wasting memory bandwidth and
limiting data reuse chances.

Second, sparse matrix computation on GPUs is substantially less efficient
than dense matrix calculation.

## Solution
Escort, an effective sparse CNN approach adapted for GPU's data-parallel
architecture, was created to circumvent the restrictions. Rather than
lowering the convolution to matrix multiplication, we compute the sparse
convolution directly. We modify the dataflow and use a number of
optimization approaches based on an understanding of the memory access
pattern to take advantage of the GPU's immense computational
horsepower.
Escort increases arithmetic intensity by directly computing sparse
convolution rather than lowering it to matrix multiplication, and it is
specifically tuned for the GPU architecture by taking advantage of the
parallelism available.

## Table of Contents

- SPARSIFY FUNCTION
- SERIAL IMPLEMENTATION
- PARALLEL UNOPTIMISED IMPLEMENTATION
- PARALLEL OPTIMISED IMPLEMENTATION
- PROFILING VIA CNN PIPELINE
