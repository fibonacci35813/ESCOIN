#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "profiling.hpp"

void padding(const float *in, float *out, int C, int H, int W, int pad)
{
    for (int i=0; i<M; ++i) for (int j=0; j<pad; ++j) for (int k=0; k<pad; ++k) out[(H+2*pad)*(W+2*pad)*i + (W+2*pad)*j + k] = 0;
    for (int i=0; i<M; ++i) for (int j=pad; j<H-pad; ++j) for (int k=pad; k<W-pad; ++k) out[(H+2*pad)*(W+2*pad)*i + (W+2*pad)*j + k] = in[H*W*i + W*(j-pad) + (k-pad)];
    for (int i=0; i<M; ++i) for (int j=H-pad; j<H; ++j) for (int k=W-pad; k<W; ++k) out[(H+2*pad)*(W+2*pad)*i + (W+2*pad)*j + k] = 0;
}

void addBias(float *ofmap, float *biases, int M, int E, int F)
{
    for (int i=0; i<M; ++i) for(int j=0; j<E; ++j) for (int k = 0; k<E; ++k) ofmap[E*F*i + F*j + k] = biases[i];
}

int main(void)
{
    /*
        Inputs
        For AlexNet: Define CNNs sequentially, feed ofmap[n-1] as ifmap[n] (n!=1) upto n == N
        Pooling layer ??
        Conv1D x 2 ??
    */
    int M, C, H, W, R, S, E, F, stride, pad;
    float *ifmap_pre, *ifmap, *ofmap, *weights, *biases;
    // IMP: All Inputs are to be in row major format
    // Input format for ifmaps: C H W [ifmaps]
    fprintf(stdout, "Enter input feature map: \n")
    scanf("%d %d %d", &C, &H, &W);
    int ifCount = 0;
    ifmap_pre = (float *)malloc(C*H*W*sizeof(float));
    for (ifcount < C*H*W) scanf("%f", &ifmap_pre[ifCount++]);

    // Input format for weights: M C R S [biases] [weights]
    fprintf(stdout, "Enter biases and weights for the first layer: \n")
    scanf("%d %d %d %d", &M, &C, &R, &S);
    int bCount = 0, wCount = 0;
    biases = (float *) malloc(M*sizeof(float));
    for (bCount < M) scanf("%f", &biases[bCount++]);
    weights = (float *) malloc(M*R*S*sizeof(float));
    for (wCount < M*R*S) scanf("%f", &weights[wCount++]);

    // First convolution
    {
        stride = 4; pad = 1;
        ifmap = (float *)malloc(C*(H+2*pad)*(W+2*pad)*sizeof(float));
        padding(ifmap_pre, ifmap, C, H, W, pad);
        SPConv layer1(ifmap, ifmap, ofmap, weights, stride, pad, M, C, H, W, R, S);
        layer1.allcateDeviceMem();
        dim3 layer1_Gdim(layer1.E*layer1.F/BLOCK_DIM + 1, M, 1), layer1_Bdim(BLOCK_DIM, 1, 1); 
        layer1.compute<<< layer1_Gdim, layer1_Bdim, (NUM_PARAMS + layer1.num_cols + layer1.num_rows)*sizeof(int) + (layer1.num_weights + C*(H+2*pad)*(W+2*pad) + M*R*S)*sizeof(float) >>>();
        layer1.freeDeviceMem();
        E = layer1.E; F = layer1.F;
        ofmap = (float *) malloc(M*E*F*sizeof(float));
        memcpy(ofmap, layer1.h_ofmap, M*E*F*sizeof(float));
    }

    // Second layer setup
    C = M;
    H = E;
    W = F;
    free(ifmap_pre); free(ifmap);
    addBias(ofmap, biases, M, E, F);
    ifmap = (float *) malloc(C*H*W*sizeof(float));
    free(ofmap);
    
    // Add maxPool layer
    // Copy paste code from above in order to add further layers

    return 0;
}