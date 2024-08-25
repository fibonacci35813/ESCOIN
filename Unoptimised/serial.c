#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define H 5 // dimension of the input features 
#define W 3

int *value;
int *colidx;
int *rowptr;
int position_value=0;
int position_rowptr=1;
int position_colidx=0;

 
void sparesify(int* matrix)
{
    value=(int*)malloc(sizeof(int)*(W*W));
    rowptr=(int*)malloc(sizeof(int)*(W+1));
    colidx=(int*)malloc(sizeof(int)*(W*W));
    int NNZ = 0;
    rowptr[0]=0;
    for (int i = 0; i < W; i++) {
        for (int j = 0; j < W; j++) {
            if (matrix[i*W+j] != 0) {
                value[position_value++] = matrix[i*W+j];
                colidx[position_colidx++] = j;
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


int main(void) 
{
	/*
	int w[] = {0,0,0, 0, 0, 2, 3, 0, 0};
	int ifmaps[H*H] = {1,2,3,4,5,6,
			7,8,9,10,11,12,
			13,14,15,16,17,18,
			19,20,21,22,23,24,
			25, 26, 27, 28, 29, 30, 
			31, 32, 33, 34, 35, 36};*/
	
	
	int w[] = {0,1,0, 0, 0, 2, 3, 0, 0};		
	int ifmaps[H*H] = {1,2,3,4,5,
			6,7,8,9,10,
			11,12, 13,14,15,
			16,17,18,19,20,
			21,22,23,24, 25};
	
	int op_size = H - W + 1;
	
	int *ofmaps;
	ofmaps = SPConv_serial(ifmaps, w, op_size);
  	printf ("\nPrinting output feature map . . . . .\n\n\n");
	for(int i=0;i<op_size;i++) 
	{
    		for(int j=0;j<op_size;j++) 
    		{
      			printf("%d ", ofmaps[i* op_size + j]);
    		}
    		printf("\n");
  	}
  	printf("\n");
	return 0;
}
