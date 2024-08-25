%%cuda --name Convolution.cu

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define H 6 // dimension of the input features 
#define W 3
#define warp_size 8

int *value;
int *colidx;
int *rowptr;
int *rowidx;
__managed__ int position_value=0;
__managed__ int position_rowptr=1;
__managed__ int position_colidx=0;
__managed__ int position_rowidx=0;

 
void sparesify(int* matrix)
{
    value=(int*)malloc(sizeof(int)*(W*W));
    rowptr=(int*)malloc(sizeof(int)*(W+1));
    colidx=(int*)malloc(sizeof(int)*(W*W));
    rowidx=(int*)malloc(sizeof(int)*(W));
    int NNZ = 0;
    rowptr[0]=0;
    for (int i = 0; i < W; i++) {
        for (int j = 0; j < W; j++) {
            if (matrix[i*W+j] != 0) {
                value[position_value++] = matrix[i*W+j];
                colidx[position_colidx++] = j;
                rowidx[position_rowidx++] = i;
                NNZ++;
            }
        }
        rowptr[position_rowptr++] = NNZ;
    }
}

int *SPConv_serial(int input_features[H*H], int weight[], int window_size)
{
	int j1, j2;
	sparesify(weight);
   
	int *output_matrix = (int *)calloc(window_size*window_size, sizeof(int *)); 
    	
	for (int i=0;i<position_rowptr-1;i++) 
	{
    		int row = i;
    		for (int j=rowptr[i];j<rowptr[i+1];j++) 
    		{
      			int mat_value = value[j];
      			int col = colidx[j];
      			for (j1=0; j1<window_size;j1++) 
      			{
        			for (j2=0; j2<window_size;j2++) 
        			{
          				output_matrix[j1*window_size + j2] = output_matrix[j1*window_size + j2]  + input_features[(j1+row)*H + (j2+col)] * mat_value;
			        }
			       		
		        }
    		}
	}

	return output_matrix;

}

__global__ void conv_mul_parallel(int* d_if, int *val, int *row, int * col_id, int * row_id, int* d_of, int window_size)
{
    extern __shared__ int psum[] ;
    __shared__ int active_tid;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
    {
        for (int idx = 1; idx < position_rowptr; idx++)
            psum[idx] = 0;
    }

    if (tid <= position_rowptr - 1)
    {
        for (int k=0; k < position_value; k++)
        {
            int mat_value = val[k];
            int row = row_id[k];
            int col = col_id[k];

            // printf("\n mat-value: %d", mat_value);
            // printf("\n row num: %d", row);
            // printf("\n col num: %d\n", col);
            
            for (int j1 = 0; j1 < window_size; j1++)
            {
                for (int j2=0; j2 < window_size; j2++)
                {
                    if (tid == j2 % warp_size)
                    {
                        // printf("\nActive threadID: %d\n", j2);
                        active_tid = tid;
                        psum[tid] = d_if[(row + j1) * H + (col + j2)] * mat_value;
                        // printf("\npsum value: %d\n", psum[tid]);
                    }

                    __syncthreads();
                    d_of[j1 * window_size + active_tid] += psum[active_tid];

                }
                
            }
        }

    }

}

int* copy(int* arr, int size)
{
    int * ret = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++)
        ret[i] = arr[i];
    return ret;
    
}

int main(void) 
{
	int w[] = {0, 0, 0, 0, 0, 2, 3, 0, 0};
   
    /*int ifmaps[H * H] = {1,2,3,4,5,6,
            7,8,9,10,11,12,
            13,14,15,16,17,18,
            19,20,21,22,23,24,
            25};*/

    int ifmaps[H * H] = {1,2,3,4,5,6,
            7,8,9,10,11,12,
            13,14,15,16,17,18,
            19,20,21,22,23,24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
	int op_size = H - W + 1;
    sparesify(w);
	//int *ofmaps;
	//ofmaps = SPConv_serial(ifmaps, w, op_size);
    int* d_ifmaps, * d_of;
    int * d_ofmaps = (int*)malloc(op_size * op_size * sizeof(int));

    // printf("\nValue[]: ");
    // for(int i=0;i<position_value;i++) 
	// {
    // 	printf("%d ", value[i]);
  	// }

    // printf("\nRow[]: ");
    // for(int i=0;i<position_rowidx;i++) 
	// {
    // 	printf("%d ", rowidx[i]);
  	// }

    // printf("\nCol[]: ");
    // for(int i=0;i<position_colidx;i++) 
	// {
    // 	printf("%d ", colidx[i]);
  	// }

    // printf("\n\n");

    cudaMalloc((void**)&d_ifmaps, H * H * sizeof(int));
    cudaMalloc((void**)&d_of, op_size * op_size * sizeof(int));

    cudaMemcpy(d_ifmaps, ifmaps, H * H * sizeof(int), cudaMemcpyHostToDevice);

    int* h_value, * h_rowptr, * h_colidx, * h_rowidx;
    int* d_value, * d_rowptr, * d_colidx, * d_rowidx;
  
    h_value = copy(value, position_value);
    h_rowptr = copy(rowptr, position_rowptr);
    h_colidx = copy(colidx, position_colidx);
    h_rowidx = copy(rowidx, position_rowidx);

    cudaMalloc((void**)&d_value, position_value * sizeof(int));
    cudaMalloc((void**)&d_rowptr, position_rowptr * sizeof(int));
    cudaMalloc((void**)&d_colidx, position_colidx * sizeof(int));
    cudaMalloc((void**)&d_rowidx, position_rowidx * sizeof(int));

    cudaMemcpy(d_value, h_value, position_value * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowptr, h_rowptr, position_rowptr * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colidx, h_colidx, position_colidx * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowidx, h_rowidx, position_rowidx * sizeof(int), cudaMemcpyHostToDevice);

    int tpb = 8;
    int bpg = (tpb + position_rowptr - 1) / tpb;

    conv_mul_parallel << <bpg, tpb, position_rowptr *sizeof(int) >> > (d_ifmaps, d_value, d_rowptr, d_colidx, d_rowidx, d_of, op_size);

    cudaMemcpy(d_ofmaps, d_of, op_size * op_size * sizeof(int), cudaMemcpyDeviceToHost);
  
    printf("\nOutput maxtrix: \n");
	for(int i=0;i<op_size;i++) 
	{
    		for(int j=0;j<op_size;j++) 
    		{
      			printf("%d ", d_ofmaps[i* op_size + j]);
    		}
    		printf("\n");
  	}
	return 0;
}
