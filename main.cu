/******************************************************************************
 *cr

 *cr
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <functional>
#include <iterator> 
#include <iterator>
#include <time.h>
#include "kernel.cu"


using namespace std;
vector<int> linspace(int start_in, int end_in, int num_in);
void printLinspace(vector<int> v);


int main ()
{
    //set standard seed
    //srand(217);

    //Timer timer;
    //cudaError_t cuda_ret;
	cudaEvent_t start, stop; 

    // Initialize host variables ----------------------------------------------
    //startTime(&timer);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    float *trainX_h, *trainY_h, *testX_h, *testY_h;
    float *trainX_d, *trainY_d, *testX_d, *testY_d;
	int *trainQ_h, *trainQ_d;
	float *L_h;
	float *L_d;
    size_t trainX_sz, trainY_sz, testX_sz, testY_sz, L_sz, trainQ_sz;
	int M=11;
    unsigned numTrainSamples=60000;
	//unsigned numTestSamples=10000;
	unsigned numFeatures=784;
	ifstream fin;
    ofstream fout;
	vector<vector<int> > trainset_array;
    vector<int> trainset_labels(60001);
    vector<vector<int> > testset_array;
    vector<int> testset_labels(10001);
	int row = 0;
	vector<int> rowArray(784);
	int lMax = 0;
    int lMin = 0;
	vector<int> L;
	
	cudaEventRecord(start);
	printf("\nSetting up the problem..."); fflush(stdout);
	
	fin.open("mnist_train.csv");
    if(!fin.is_open())
    {
        printf( "Error: Can't open file containind training X dataset"  );
    }
    else
    {	printf("loading data..");
        while(!fin.eof())
        {
            
            if(row > numTrainSamples)
            {
                break;
            }
            
            fin >> trainset_labels.at(row);
			
            trainset_array.push_back(rowArray);
			
            
            for(int col = 0; col < 784; col++)
            {
                fin.ignore(50000000, ',');
                fin >> trainset_array[row][col];
            }
            row++;
			//printf("checkpoint1: %d",row);
        }
		
    }
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop); 

	printf("Time to generate:  %3.1f ms \n", milliseconds); 

	//printf("\tdata loading done ..\n");
	L_h = (float*) malloc( sizeof(float)*M );
	trainX_h = (float*) malloc( sizeof(float)*numTrainSamples*numFeatures );
	trainQ_h = (int*) malloc( sizeof(int)*numTrainSamples*numFeatures );
	trainY_h = (float*) malloc( sizeof(float)*numTrainSamples );
    for (unsigned int i=0; i < numTrainSamples*numFeatures; i++) { trainX_h[i] = trainset_array[int(i/numFeatures)][int(i%numFeatures)]; }
	for (unsigned int i=0; i < numTrainSamples; i++) { trainY_h[i] = trainset_labels[i]; }
	
	printf("\ndata loading done ..\n");
		//checking if data has been loaded
	//for (int i=0; i<numTrainSamples; i++)
	//{	
	//	printf("\n%d",trainset_labels[i]);
	//	for (int j=0; j<numFeatures; j++)
	//	{
	//		printf("/t%d",trainset_array[i][j]);
	//	}
	//}
	
	// Defining Quantization Levels_________
	lMin= *min_element(trainset_array[0].begin(),trainset_array[0].end());
    lMax= *max_element(trainset_array[0].begin(),trainset_array[0].end());
	L = linspace(lMin, lMax, M);
	for (unsigned int i=0; i < M; i++) { L_h[i] = L[i]; }
	
	
	
	//copy trainX,trainY L to device___________
	trainX_sz= numTrainSamples*numFeatures*sizeof(float);
	trainY_sz=numTrainSamples*sizeof(float);
	L_sz=M*sizeof(float);
	trainQ_sz=numTrainSamples*numFeatures*sizeof(int);
	
	cudaMalloc((void **)&trainX_d, trainX_sz);
	cudaMalloc((void **)&trainY_d, trainY_sz);
	cudaMalloc((void **)&L_d, L_sz);
	cudaMalloc((void **)&trainQ_d, trainQ_sz);
	cudaMemcpy(trainX_d, trainX_h, trainX_sz, cudaMemcpyHostToDevice);
	cudaMemcpy(trainY_d, trainY_h, trainY_sz, cudaMemcpyHostToDevice);
	cudaMemcpy(L_d, L_h, L_sz, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	
	CalcQuantization(trainX_d,L_d,trainQ_d,numTrainSamples,numFeatures,M);
	
	cudaMemcpy(trainQ_h, trainQ_d, trainQ_sz, cudaMemcpyDeviceToHost);
	
	for (int i=0; i<numTrainSamples*numFeatures; i++)
	{
		printf("\t %d",trainQ_h[i]);
		if ((i%numFeatures)==0)
		{
			printf("\n");
		}
	}
	
	
	
	free(trainX_h);
	free(trainY_d);
	free(L_h);
	free(trainQ_h);
	
	cudaFree(trainX_d);
	cudaFree(trainY_d);
	cudaFree(L_d);
	cudaFree(trainQ_d);
}

vector<int> linspace(int start_in, int end_in, int num_in)
{
    vector<int> linspaced;
    
    int start = start_in;
    int end = end_in;
    int num = num_in;
    
    if(num == 0)
    {
        return linspaced;
    }
    
    if(num == 1)
    {
        linspaced.push_back(start);
        return linspaced;
    }
    
    int delta = (end - start) / (num - 1);
    
    for(int i = 0; i < num - 1; i++)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end);
        
    
    return linspaced;
}

void printLinspace(vector<int> v)
{
    cout << "size: " << v.size() << endl;
    for(int i=0; i< v.size(); i++)
    {
        cout << v[i] << " ";
    }
    cout << endl;
}