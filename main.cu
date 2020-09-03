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
vector<float> linSpace(float start_in, float end_in, int num_in);
void printLinspace(vector<float> v);


int main ()
{
    //set standard seed
    //srand(217);

    //Timer timer;
    //cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------
    //startTime(&timer);
		
	int M=11;
	int D=10000;
	int numClasses=10;
    int numTrainSamples=60000;
	int numTestSamples=10000;
	int numFeatures=784;

    float *trainX_h, *testX_h;
    float *trainX_d, *testX_d;
	size_t trainX_sz, trainY_sz, testX_sz, testY_sz, L_sz, trainQ_sz;
	int *trainQ_d, *trainY_h, *trainY_d, *testY_h, *testY_d, *Classes_h;
	
	ifstream fin;
	ifstream ftestin;
    ofstream fout;
	vector<vector<float> > trainset_array;
    vector<int> trainset_labels(60001);
    vector<vector<float> > testset_array;
    vector<int> testset_labels(10001);
	int row = 0;
	vector<float> rowArray(784);
	
	float *L_h;
	float *L_d;
	int *LD_h, *ID_h, *ClassHV_h;
	int *LD_d, *ID_d, *ClassHV_d; 
	float lMax;
    float lMin;
	vector<float>L;

//***** Data Loading ****//

//Loading data from .csv file_________

	printf("\nSetting up the problem...\n"); fflush(stdout);
	
	fin.open("mnist_train.csv");
    if(!fin.is_open())
    {
        printf( "Error: Can't open file containind training X dataset"  );
    }
    else
    {	printf("\nloading train data..\n");
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
	fin.close();
	printf("\ntraining dtata loading done..\n");
	row=0;
	ftestin.open("mnist_test.csv");
    if(!ftestin.is_open())
    {
        printf("Error: Can't open file containind training X dataset");
    }
    else
    {	printf("\nloading test data..\n");
        while(!ftestin.eof())
        {
            
            if(row > numTestSamples)
            {
                break;
            }
            
            ftestin >> testset_labels.at(row);
            testset_array.push_back(rowArray);
			
            
            for(int col = 0; col < 784; col++)
            {
                ftestin.ignore(50000000, ',');
                ftestin >> testset_array[row][col];
            }
            row++;
			//printf("checkpoint1: %d",row);
        }
		
    }
	ftestin.close();

	trainX_h = (float*) malloc( sizeof(float)*numTrainSamples*numFeatures );
	trainY_h = (int*) malloc( sizeof(int)*numTrainSamples );
    for (int i=0; i < numTrainSamples*numFeatures; i++) 
	{ 	
		trainX_h[i] = trainset_array[i/numFeatures][int(i%numFeatures)]; 
	}
	for (int i=0; i < numTrainSamples; i++) { trainY_h[i] = trainset_labels[i]; }
	
	testX_h = (float*) malloc( sizeof(float)*numTestSamples*numFeatures );
	testY_h = (int*) malloc( sizeof(int)*numTestSamples );
    for (int i=0; i < numTestSamples*numFeatures; i++) 
	{ 	
		testX_h[i] = testset_array[i/numFeatures][int(i%numFeatures)]; 
	}
	for (int i=0; i < numTestSamples; i++) { testY_h[i] = testset_labels[i]; }	
	printf("\ndata loading done ..\n");
	
	
	
	

printf("\nSetting up Level and Identity Hypervector....\n");
	
//****Setting up Level and Identity Hypervector****//
	
	//Defining Quantization Levels_________
	
	L_h = (float*) malloc( sizeof(float)*M );
	
	lMin= *min_element(trainset_array[0].begin(),trainset_array[0].end());
    lMax= *max_element(trainset_array[0].begin(),trainset_array[0].end());
	L = linSpace(lMin, lMax, M);

	//Setting up Level Hypervector
	LD_h = (int*) malloc( sizeof(int)*M*D );
	
	for (int i=0; i<D; i++) {LD_h[i]=int(rand()%2);}
	int nAlter[D];
	for (int i=1; i<D; i++)
	{
		nAlter[i]=rand()%10000;
	}
	//random_shuffle(nAlter.begin(), nAlter.end());	
	int jAlter;
	
	for (int i=1; i<M; i++)
	{
		for (int j=0; j<D; j++)
		{
			LD_h[i*D+j]=LD_h[(i-1)*D+j];
		}
		for (int j=0; j<ceil(D/M); j++)
		{
			jAlter=nAlter[int((i-1)*ceil(D/M)+j)];
			LD_h[(i*D)+jAlter]=int(LD_h[(i*D)+jAlter]==0);
		}
	}
	
	//Creating Identity Hypervector ID
	ID_h = (int*) malloc( sizeof(int)*numFeatures*D );
	for (int i=0; i<numFeatures; i++)
	{
		for (int j=0; j<D; j++)
		{
			ID_h[i*D+j]=int(rand()%2);
		}	
	}
	
	//test to see if LD and ID is being populated properly_________
    int LD_test1=0;
    int LD_test2=0;
    for(int jtest = 0; jtest < D; jtest++)
    {
      LD_test1=LD_test1+ (LD_h[0+jtest]^LD_h[D+jtest]) ; //FIX ME: print out all 0s
      LD_test2=LD_test2+ (LD_h[5*D+jtest]^LD_h[0*D+jtest]) ;
    }
    cout <<"LDTEST1: "<< LD_test1<< endl;
    cout << "LDTEST2: "<<LD_test2<< endl;
	
	//test to see if ID is being populated properly
    int ID_test=0;
    for(int j = 0; j < D; j++)
    {
        ID_test=ID_test+ (ID_h[D+j]^ID_h[j]) ; 
	}
    cout <<"IDTEST:"<<ID_test<< endl;
			
	printf("Creating level and Identity Hypervector Done\n");
	
	
	
	

//****************Training ************//
//*************************************//

	printf("\nTraining...\n");
	//copy trainX,trainY L to device___________
	
	trainX_sz= numTrainSamples*numFeatures*sizeof(float);
	trainY_sz=numTrainSamples*sizeof(int);
	L_sz=M*sizeof(float);
	trainQ_sz=numTrainSamples*numFeatures*sizeof(int);
	
	cudaMalloc((void **)&trainX_d, trainX_sz);
	cudaMalloc((void **)&trainY_d, trainY_sz);
	cudaMalloc((void **)&L_d, L_sz);
	cudaMalloc((void **)&trainQ_d, trainQ_sz);
	cudaMalloc((void **)&LD_d, M*D*sizeof(int));
	cudaMalloc((void **)&ID_d, numFeatures*D*sizeof(int));
	cudaMalloc((void **)&ClassHV_d, numClasses*D*sizeof(int));
	
	cudaMemcpy(trainX_d, trainX_h, trainX_sz, cudaMemcpyHostToDevice);
	cudaMemcpy(trainY_d, trainY_h, trainY_sz, cudaMemcpyHostToDevice);
	cudaMemcpy(L_d, L_h, M*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(LD_d, LD_h, M*D*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(ID_d, ID_h, numFeatures*D*sizeof(int), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	
	printf("checkpoint1\n");
	Classes_h=(int*) malloc( sizeof(int)*numClasses );
	for(int i=0; i<numClasses; i++)
	{
		Classes_h[i]=0;
	}
	Training_HV(trainX_d,trainY_d,trainY_h,L_d,trainQ_d,numTrainSamples,numFeatures,numClasses, M,D,LD_d,ID_d,ClassHV_d, Classes_h);
	
	printf("checkpoint2\n");
	ClassHV_h = (int*) malloc( sizeof(int)*numClasses*D );
	cudaMemcpy(ClassHV_h, ClassHV_d, numClasses*D*sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	//Teasting  ClassHypervectors
	int Class_test1=0;
	int Class_test2=0;
	
	for (int i=0;i<D; i++)
	{
		Class_test1=Class_test1+(ClassHV_h[0*D+i]^ClassHV_h[1*D+i]);
		Class_test2=Class_test2+(ClassHV_h[0*D+i]^ClassHV_h[5*D+i]);	
	}
	printf("\n Classtest1=%d", Class_test1);
	printf("\n Classtest2=%d", Class_test2);
	
	printf("\nTraining done.......\n");
	
	
	
	
	
	
//*****Testing _________********//
//******************************//

	testX_sz= numTestSamples*numFeatures*sizeof(float);
	testY_sz=numTestSamples*sizeof(int);
	
	cudaMalloc((void **)&testX_d, testX_sz);
	cudaMalloc((void **)&testY_d, testY_sz);
	
	cudaMemcpy(testX_d, testX_h, testX_sz, cudaMemcpyHostToDevice);
	cudaMemcpy(testY_d, testY_h, testY_sz, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	
	float accuracy;
	TestingAccuracy(testX_d, testY_d, testY_h, ClassHV_d, L_d,LD_d,ID_d, D,M, numTestSamples, numFeatures,numClasses,accuracy);
	
	

	printf("\n Testing done......\n");
	
	cudaFree(trainX_d);
	cudaFree(trainY_d);
	cudaFree(trainQ_d);
	cudaFree(testX_d);
	cudaFree(testY_d);
	
	free(trainX_h);
	free(trainY_h);
	free(testX_h);
	free(testY_h);
	free(L_h);
	free(LD_h);
	free(ID_h);
	free (ClassHV_h);

	
	cudaFree(L_d);
	cudaFree(LD_d);
	cudaFree(ID_d);
	cudaFree(ClassHV_d);
	
}

vector<float> linSpace(float start_in, float end_in, int num_in)
{
    vector<float> linspaced;
    
    float start = start_in;
    float end = end_in;
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
    
    float delta = (end - start) / (num - 1);
    
    for(int i = 0; i < num - 1; i++)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end);
    return linspaced;
}

void printLinspace(vector<float> v)
{
    cout << "size: " << v.size() << endl;
    for(int i=0; i< v.size(); i++)
    {
        cout << v[i] << " ";
    }
    cout << endl;
}