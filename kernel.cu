/******************************************************************************

 *cr
 ******************************************************************************/

#include <stdio.h>
   

__global__ void QuantKernel(int n, int M, float *trainX_d,int *trainQ_d, const float *L_d) {

    /********************************************************************
     *
     * Compute C = A + B
     *   where A is a (1 * n) vector
     *   where B is a (1 * n) vector
     *   where C is a (1 * n) vector
     *
     ********************************************************************/
    // INSERT KERNEL CODE HERE


	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int indexMin=0;
	float quantMin=abs(trainX_d[tid]-L_d[indexMin]);
	if(tid<n)
	{
		for (int i=1;i<M;i++)
		{
			if ( (abs(trainX_d[tid]-L_d[i]))<quantMin)
			{
				quantMin=abs(trainX_d[tid]-L_d[i]);
				indexMin=i;
			}
		}
		trainQ_d[tid]=indexMin;
	}

}


void CalcQuantization(float *trainX_d,const float *L_d,int *trainQ_d,int numTrainSamples,int numFeatures,int M)
{

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = 256; 
	int n=numTrainSamples*numFeatures;
    //INSERT CODE HERE
	dim3 dimGrid(ceil(n/BLOCK_SIZE),1,1);
	dim3 dimBlock(BLOCK_SIZE,1,1);
	QuantKernel<<<ceil((float)n/BLOCK_SIZE), BLOCK_SIZE>>>(n,M,trainX_d,trainQ_d,L_d);	
      

}

