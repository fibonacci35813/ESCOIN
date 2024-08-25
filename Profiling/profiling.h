#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_DIM 1024
#define NUM_PARAMS 12
// [M C H W R S E F stride numWeights numCols numRows]

class SPConv {
    public:
    float *weights, *d_weights;
    int *colIdx, *rowIdx, num_weights, num_rows, num_cols;
    int *d_colIdx, *d_rowIdx;
    int *d_computeParams;
    int M, C, H, W, R, S;
    // int d_M, d_C, d_H, d_W, d_R, d_S;
    float *d_ifmap, *d_ofmap;
    // int d_stride, d_pad, d_E, d_F;
    
    float *h_ifmap, *h_ofmap, *h_filters;
    float *out_ofmap;
    int stride, pad, E, F;
    
    __host__ SPConv(float *in, float *out, float *w, int stride=1, int pad=0, int M=1, int C=1, int H=6, int W=6, int R=3, int S=3)
    {
        E = (H - R + 2*pad)/stride + 1;
        F = (W - S + 2*pad)/stride + 1;

        h_ifmap = (float *) malloc(C * (H + 2*pad) * (W + 2*pad) * sizeof(float));
        memcpy(h_ifmap, in, C * (H + 2*pad) * (W + 2*pad) * sizeof(float));     // padded input
        out_ofmap = out;
        h_ofmap = (float *) malloc(M * E * F * sizeof(float));
        
        h_filters = (float *) malloc(M * C * R * S * sizeof(float));
        memcpy(h_filters, in, M * C * R * S * sizeof(float)); 
        stride = stride; pad = pad; M = M; C = C; H = H; W = W; R = R; S = S;
        
        getSCR();
    }

    __host__ virtual ~SPConv() {}

    __host__ void getCSR()
    {
        int rowCount=1, wCount=0, colCount=0;
        float *w;
        int *c, *r;
        w = (float *)malloc( M * C * R * S * sizeof(float) );
        c = (int *)malloc( M * R * S * sizeof(int) );
        r = (int *)malloc( (M+1) * sizeof(int) );
        r[0] = 0;
        int NNZ = 0;
        for (int i=0; i<M; ++i) {
            for (int j=0; j<R*S; ++j) {
                if (h_filters[R*S*i + j] != 0) {
                    w[wCount++] = h_filters[R*S*i + j];
                    c[colCount++] = R*S*i + j;
                    ++NNZ;
                }
            }
            r[rowCount++] = NNZ;
        }

        weights = (float *)malloc(wCount*sizeof(float));
        memcpy(weights, w, wCount*sizeof(float));
        num_weights = wCount;
        
        // assert(rowCount == M+1)
        rowIdx = (int *)malloc((M+1)*sizeof(int));
        memcpy(rowIdx, r, (M+1)*sizeof(int));
        num_rows = M+1;
        
        colIdx = (int *)malloc(colCount*sizeof(int));
        memcpy(colIdx, c, colCount*sizeof(int));
        num_cols = colCount;
    }

    __host__ void allocateDeviceMem()
    {
        cudaMalloc((void **)&d_weights, num_weights*sizeof(float));
        cudaMemcpy(d_weights, weights, num_weights*sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc((void **)&d_ifmap, C * H * W *sizeof(float));
        cudaMemcpy(d_ifmap, h_ifmap, C * H * W *sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc((void **)&d_ofmap, M * E * F *sizeof(float));
        
        cudaMalloc((void **)&d_colIdx, num_cols*sizeof(float));
        cudaMemcpy(d_colIdx, colIdx, num_cols*sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc((void **)&d_rowIdx, num_rows*sizeof(float));
        cudaMemcpy(d_rowIdx, rowIdx, num_rows*sizeof(float), cudaMemcpyHostToDevice);

        int *computeParams = {M, C, H + 2*pad, W + 2*pad, R, S, E, F, stride, num_weights, num_cols, num_rows};
        cudaMalloc((void **)&d_computeParams, NUM_PARAMS*sizeof(float));
        cudaMemcpy(d_computeParams, computeParams, NUM_PARAMS*sizeof(float), cudaMemcpyHostToDevice);
    }

    // n = NUM_PARAMS + num_weights + num_cols + num_rows + len_ifmap + len_ofmap
    __global__ void compute(unsigned int n)
    {
        extern __shared__ float arr[];
        // Ref: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared
        float *params = (float *)arr;

        int tid = blockIdx.x * blockDim.x + threadIdx.x,
        toffset = gridDim.x * blockDim.x * blockIdx.y;
        for (int i=threadIdx.x; i<NUM_PARAMS; i+=blockDim.x) {
            params[i] = d_computeParams[i];
        }
        __syncthreads();

        float *w = (float *)&arr[sizeof(int)*NUM_PARAMS];
        int *cols = (int *)&arr[sizeof(int)*NUM_PARAMS + sizeof(float)*(params[9])];
        int *rows = (int *)&arr[sizeof(int)*NUM_PARAMS + sizeof(float)*(params[9]) + sizeof(int)*(params[10])];
        float *ip = (float *)&arr[sizeof(int)*NUM_PARAMS + sizeof(float)*(params[9]) + sizeof(int)*(params[10]) + sizeof(int)*(params[11])];
        float *op = (float *)&arr[sizeof(int)*NUM_PARAMS + sizeof(float)*(params[9]) + sizeof(int)*(params[10]) + sizeof(int)*(params[11]) + sizeof(float)*(params[1] * params[2] * params[3] )];
        /* 
           grid(X, Y, Z): dim3(E*F/BLOCK_DIM, M, 1)
           Y -> diff output layer
           X -> ceil(E*F/BLOCK_DIM)
        */

        for (int i=threadIdx.x; i < params[9]; i += blockDim.x) {
            w[i] = d_weights[i];
        } // Load weights

        for (int i=threadIdx.x; i < params[11]; i += blockDim.x) {
            rows[i] = d_rowIdx[i];
        } // Load rows

        for (int i=threadIdx.x; i < params[10]; i += blockDim.x) {
            cols[i] = d_colIdx[i];
        } // Load cols

        for (int i=threadIdx.x; i < params[1] * params[2] * params[3] ; i += blockDim.x) {
            ip[i] = d_ifmap[i];
        } // Load inputs
        __syncthreads();

        if (tid < params[6] * params[7]) {
            float psum = 0;
            for (int c_idx=rows[blockIdx.y]; c_idx<rows[blockIdx.y+1]; ++c_idx) {
                float val = w[colIdx];
                int col = cols[c_idx];
                int j = col % (params[4] * params[5]); // point: (j%S, j/S)
                for (int i_idx = 0; i_idx < C; ++i_idx) {
                    // top-left corner of weight matrix
                    int i_val = i_idx * params[2] * params[3] + (params[8])*(params[3]*(tid/params[7]) + tid%params[7]); // H*W*c + stride*(W*(tid/F) + tid%F)
                    psum += ip[i_val + params[3]*(j/params[5]) + (j%params[5])] * val;
                }
            }
            op[tid + params[6] * params[7] * blockIdx.y] = psum;
        }
        __syncthreads();

        for (int i=tid; i<(params[0] * params[4] * params[5] )/gridDim.y; ++i) {
            d_ofmap[i] = op[i];
        }
        __syncthreads();

    }

    __host__ void freeDeviceMem() 
    {
        cudaMemcpy(h_ofmap, d_ofmap, C * H * W *sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_ifmap); cudaFree(d_ofmap); cudaFree(d_weights);
        cudaFree(d_colIdx); cudaFree(d_rowIdx); 
        cudaFree(d_computeParams);
    }

};

/*
    Note: changed columns from M*C*R*S to M*R*S, please check in case of errors
*/

#define MP_PARAMS 8
class MaxPool {
    public:
    float *d_in, *d_out, *d_params;
    float *h_in, *h_out; 
    int C, H, W, R, S, E, F, stride;

    __host__ MaxPool(const float *in, float *out, int C, int H, int W, int R, int S, int E, int F, int stride) 
    {
        C = C;
        H = H;
        W = W;
        R = R;
        S = S;
        E = E;
        F = F;
        stride = stride;
        h_in = malloc(C*H*W*sizeof(float));
        memcpy(h_in, in, C * H * W * sizeof(float));
        h_out = malloc(C*E*F*sizeof(float));
    }

    __host__ void allocateDeviceMem()
    {
        cudaMalloc((void **)&d_in, C*H*W*sizeof(float));
        cudaMalloc((void **)&d_out, C*E*F*sizeof(float));
        cudaMalloc((void **)&d_params, MP_PARAMS*sizeof(int));
        
        float *params = {C, H, W, R, S, E, F, stride};

        cudaMemcpy(d_in, h_in, C*H*W*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_params, params, MP_PARAMS*sizeof(int), cudaMemcpyHostToDevice);
    }

    __host__ void freeDeviceMem()
    {
        cudaMemcpy(h_out, d_out, C*E*F*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_params);
    }

    // (blockDim.x,blockDim.y)
    __device__ void compute(unsigned int n)
    {
        // Dynamic access to shared memory was not working on colab GPUs
        // Pseudo code:
        //  Load arrays, params
        //  Map grid to [part of mth layer, mth layer]
        //  Use ExF threads in blocks to find ExF elements of the ofmap 
    }
};