/******************************************************************************

 *cr
 ******************************************************************************/

#include <stdio.h>
#include <math.h>
   

__global__ void QuantKernel(int n, int M, float *trainX_d,int *trainQ_d, float *L_d) {

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

__global__ void SampleHVKernel(int n_SHV, int D, int numTrainSamples, int numFeatures, int *trainQ_d, int *SampleHV_d, int *LD_d, int *ID_d) 
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int samp_ind= int(tid/D);
	int D_ind=int(tid%D);
	if (tid<n_SHV)
	{
		SampleHV_d[tid]=0;
		
		for (int i_s=0; i_s<numFeatures; i_s++)
		{
			int LD_ind=trainQ_d[samp_ind*numFeatures+i_s];
			SampleHV_d[tid]=SampleHV_d[tid]+(LD_d[LD_ind*D+D_ind]^ID_d[i_s*D+D_ind]);
		}
		SampleHV_d[tid]=int(SampleHV_d[tid]>=(numFeatures/2));

	}
}

__global__ void ClassHVAddKernel(int i_C, int D, int *trainY_d, int *SampleHV_d, int *ClassHV_d, int *Classes_d) {
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int c_ind=trainY_d[i_C];
	if (tid<D)
	{
		ClassHV_d[c_ind*D+tid]=ClassHV_d[c_ind*D+tid]+SampleHV_d[i_C*D+tid];
	}
	
}


__global__ void ClassHVKernel(int ClassHV_sz, int D, int numClasses,int *ClassHV_d, int *Classes_d) {
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int cid=int(tid/D);
	if (tid<ClassHV_sz)
	{
		ClassHV_d[tid]=int(!(ClassHV_d[tid]<(Classes_d[cid]/2)));
	}
}

__global__ void CheckSumKernel(int n_C, int *QueryHV_d,int *ClassHV_d,int *CheckSumHV_d,int numTestSample,int numClasses, int D)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int samp_id=int(tid/numClasses);
	int class_id=int(tid%numClasses);
	CheckSumHV_d[tid]=0;
	if (tid<n_C)
	{
		for (int i=0; i<D; i++)
		{
			CheckSumHV_d[tid]=CheckSumHV_d[tid]+(QueryHV_d[samp_id*D+i]^ClassHV_d[class_id*D+i]);
		}
	}
}

__global__ void TestingKernel(int n_T,int *CheckSumHV_d,int *testLebel_d, int *testY_d, int *AccuVector_d,int numTestSample,int numClasses)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int samp_id=tid;
	int minIndex=0;
	int minValue = CheckSumHV_d[samp_id*numClasses];
	
	if (tid<n_T)
	{
		for (int i=1; i<numClasses; i++)
		{
			if (minValue>CheckSumHV_d[samp_id*numClasses+i])
			{
				minValue=CheckSumHV_d[samp_id*numClasses+i];
				minIndex=i;
			}
			
		}
		testLebel_d[tid]=minIndex;
		AccuVector_d[tid]=(testLebel_d[tid]==testY_d[tid]);
	}
}

void Training_HV(float *trainX_d,int *trainY_d,int *trainY_h, float *L_d,int *trainQ_d,int numTrainSamples,int numFeatures,int numClasses,int M,int D, int *LD_d, int *ID_d, int *ClassHV_d, int *Classes_h)
{

    // Initialize thread block and kernel grid dimensions ---------------------
	printf("Checkpoint: kernelstart\n");
    const unsigned int BLOCK_SIZE = 256; 
	int n=numTrainSamples*numFeatures;
	int n_SHV=D*numTrainSamples;
    int n_CHV=D*numClasses;
	
	printf("Checkpoint: K0\n");
	dim3 dimGrid(ceil(n/BLOCK_SIZE),1,1);
	dim3 dimBlock(BLOCK_SIZE,1,1);
	
	printf("Checkpoint: K1\n");
	QuantKernel<<<ceil((float)n/BLOCK_SIZE), BLOCK_SIZE>>>(n,M,trainX_d,trainQ_d,L_d);
	cudaDeviceSynchronize();
	
	printf("Checkpoint: K2\n");
	
	int *Classes_d, *SampleHV_d, *SampleHV_h, *trainQ_h;
	cudaMalloc((void **)&Classes_d, numClasses*sizeof(int));
	
	cudaMalloc((void **)&SampleHV_d, numTrainSamples*D*sizeof(int));
	SampleHVKernel<<<ceil((float)n_SHV/BLOCK_SIZE), BLOCK_SIZE>>>(n_SHV,D,numTrainSamples, numFeatures,trainQ_d,SampleHV_d,LD_d, ID_d);
	SampleHV_h = (int*) malloc( sizeof(int)*numTrainSamples*D );
	cudaDeviceSynchronize();
	cudaMemcpy(SampleHV_h, SampleHV_d, sizeof(int)*numTrainSamples*D, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	trainQ_h = (int*) malloc( sizeof(int)*numTrainSamples*numFeatures );
	cudaMemcpy(trainQ_h, trainQ_d, sizeof(int)*numTrainSamples*numFeatures, cudaMemcpyDeviceToHost);
	
	
    
	printf("Checkpoint: K3\n");
	
	for (int i_C=1;i_C<numTrainSamples; i_C++)
	{
		ClassHVAddKernel<<<ceil((float)D/BLOCK_SIZE), BLOCK_SIZE>>>(i_C,D,trainY_d,SampleHV_d,ClassHV_d,Classes_d);
		cudaDeviceSynchronize();
		Classes_h[trainY_h[i_C]]=Classes_h[trainY_h[i_C]]+1;
	}
	cudaMemcpy(Classes_d, Classes_h, numClasses*sizeof(int),cudaMemcpyHostToDevice);
	
	printf("Checkpoint: K4\n");
	int ClassHV_sz=D*numClasses;
	ClassHVKernel<<<ceil((float)ClassHV_sz/BLOCK_SIZE), BLOCK_SIZE>>>(ClassHV_sz,D, numClasses, ClassHV_d,Classes_d);
	cudaDeviceSynchronize();
	
	printf("Checkpoint: K5\n");
	cudaFree(SampleHV_d);
	cudaFree(Classes_d);
	free (SampleHV_h);
	free (trainQ_h);
}

void TestingAccuracy(float *testX_d, int *testY_d, int *testY_h, int* ClassHV_d, float *L_d,int *LD_d, int *ID_d, int D,int M, int numTestSamples, int numFeatures,int numClasses,float accuracy)
{
	printf("\nCreating Query Hypervector....\n");
	const unsigned int BLOCK_SIZE = 256;
	
	int *QueryHV_d,*QueryHV_h, *testQ_d, *testQ_h, *AccuVector_d,*AccuVector_h,*testLebel_h, *testLebel_d,*CheckSumHV_d;
	int n=numTestSamples*numFeatures;
	int n_SHV=numTestSamples*D;
	int n_C=numTestSamples*numClasses;
	int n_T=numTestSamples;
	
	cudaMalloc((void **)&testQ_d, numTestSamples*D*sizeof(int));
	QuantKernel<<<ceil((float)n/BLOCK_SIZE), BLOCK_SIZE>>>(n,M,testX_d,testQ_d,L_d);
	cudaDeviceSynchronize();
	
	cudaMalloc((void **)&QueryHV_d, numTestSamples*D*sizeof(int));
	SampleHVKernel<<<ceil((float)n_SHV/BLOCK_SIZE), BLOCK_SIZE>>>(n_SHV,D,numTestSamples, numFeatures,testQ_d,QueryHV_d,LD_d, ID_d);
	cudaDeviceSynchronize();
	
	QueryHV_h = (int*) malloc( sizeof(int)*numTestSamples*D );
	printf("\n Query Hypervector Created....\n");
	
	accuracy=0;
	cudaMalloc((void **)&AccuVector_d, numTestSamples*sizeof(int));
	cudaMalloc((void **)&testLebel_d, numTestSamples*sizeof(int));
	cudaMalloc((void **)&CheckSumHV_d, numTestSamples*sizeof(int));
	
	CheckSumKernel<<<ceil((float)n_C/BLOCK_SIZE), BLOCK_SIZE>>>(n_C, QueryHV_d,ClassHV_d,CheckSumHV_d,numTestSamples,numClasses, D);
	cudaDeviceSynchronize();
	
	TestingKernel<<<ceil((float)n_T/BLOCK_SIZE), BLOCK_SIZE>>>(n_T,CheckSumHV_d,testLebel_d,testY_d,AccuVector_d,numTestSamples,numClasses);
	cudaDeviceSynchronize();
	
	testLebel_h = (int*) malloc( sizeof(int)*numTestSamples );
	AccuVector_h = (int*) malloc( sizeof(int)*numTestSamples );
	cudaMemcpy(testLebel_h, testLebel_d, numTestSamples*sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(AccuVector_h, AccuVector_d, numTestSamples*sizeof(int),cudaMemcpyDeviceToHost);
	for (int i=0;i<numTestSamples; i++)
	{
		accuracy=accuracy+AccuVector_h[i];
		//printf("\nTest Sample:%d;\tPredictedLabel=%d; \t Actual Label=%d\n",i,testLebel_h[i], testY_h[i]);
	}
	printf("\naccuracy=%f%\n",(accuracy/numTestSamples)*100);
	

	
	
	
	cudaFree(QueryHV_d);
	cudaFree(testQ_d);
	cudaFree(CheckSumHV_d);
	cudaFree(AccuVector_d);
	cudaFree(testLebel_d);
	
	free(AccuVector_h);
	free(testLebel_h);
	
}

