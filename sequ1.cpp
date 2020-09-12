/*BP neural networks trained in sequence*/

#include<stdio.h>
#include<math.h>
#include<stdlib.h>

double sigmod(double x);
double test(double* weight1, double* weight2, double* theta1, double theta2);

constexpr auto PI = 3.1415926;
constexpr auto Hidden =9;
constexpr auto MaxTime = 10000000;

void main( )
{
	/*initial the train data*/
	double TrainInput[9] = { 0 , 0.25 * PI ,0.5 * PI, 0.75 * PI,  PI, 1.25 * PI, 1.5 * PI, 1.75 * PI, 2 * PI };
	double TrainOutput[9] = { 0, 0.707106781186548,1,0.707106781186548, 0 , -0.707106781186548, -1, -0.707106781186548, 0 };
	/*************************/
	/****The train process****/
	//initial the weight and bias(Fix the random data, so we can compare the other elements)
	double W1[Hidden] = { 0.959492426392903, 0.655740699156587, 0.0357116785741896 , 0.849129305868777, 0.933993247757551, 0.678735154857774 , 0.655740699156587, 0.0357116785741896 , 0.849129305868777 };
	double Theta1[Hidden] = { 0.757740130578333, 0.743132468124916, 0.392227019534168 ,0.655477890177557,0.171186687811562,0.706046088019609 , 0.743132468124916, 0.392227019534168,0.655477890177557 };
	double W2[Hidden] = { 0.0318328463774207, 0.276922984960890, 0.0461713906311539 , 0.0971317812358475, 0.823457828327293, 0.694828622975817 , 0.276922984960890, 0.0461713906311539  , 0.0971317812358475 };
	double Theta2 = 0.317099480060861;
	double Alpha = 0.01;
	double Eta = 0.001; 
	double ErrRecord[MaxTime];
	//define the process variable
	double HiddenOutput[Hidden];
	int RandNum;

	for (int count1 = 0; count1 < MaxTime; count1++){
		//RandNum = rand() % 9; //9 reprents the total number of input
		RandNum = count1 % 9;
		for (int count2 = 0; count2 < Hidden; count2++) {
			HiddenOutput[count2] = sigmod(W1[count2] * TrainInput[RandNum] + Theta1[count2]);
		}//calculate the output of Hidden layer
		
		double ActualOutput = 0;
		for (int count3 = 0; count3 < Hidden; count3++) {
			ActualOutput = ActualOutput + W2[count3] * HiddenOutput[count3];
		}//calculate the Actualoutput
		ActualOutput += Theta2;
		double Error = TrainOutput[RandNum] - ActualOutput;
		ErrRecord[count1] = 0.5 * pow(TrainOutput[RandNum] - ActualOutput, 2);
		
		//Update W1,W2, Theta1, Theta2
		Theta2 += (Eta * Error);
		for (int count4 = 0; count4 < Hidden; count4++) {

			W1[count4] = W1[count4] + Alpha * Error * HiddenOutput[count4] * (1 - HiddenOutput[count4]) * W2[count4] * TrainInput[RandNum];
			Theta1[count4] = Theta1[count4] + Eta * Error * HiddenOutput[count4] * (1 - HiddenOutput[count4]) * W2[count4];
		//The above sentences execute first in case W2 changes.
			W2[count4] = W2[count4] + Alpha * Error * HiddenOutput[count4];
		}
		if(count1%100000==0)
		printf("Train Error = %lf\n", ErrRecord[count1]);
		
	}

	double TestErr = test(W1, W2, Theta1, Theta2);
	printf("Test Error = %lf", TestErr);
	/*save data to txt files */
	FILE* fp;
	fp = fopen("D:\\err.txt", "w");
	for (int i = 0; i < MaxTime; i++) {
		fprintf(fp, "%lf\n", ErrRecord[i]);
	}
	fclose(fp);
	
	FILE* fp3;
	fp3 = fopen("D:\\weight.txt", "w");
	for (int i = 0; i < Hidden; i++) {
		fprintf(fp3, "%lf\n", W1[i]);
	}
	for (int i = 0; i < Hidden; i++) {
		fprintf(fp3, "%lf\n", W2[i]);
	}
	for (int i = 0; i < Hidden; i++) {
		fprintf(fp3, "%lf\n", Theta1[i]);
	}
	fprintf(fp3, "%lf\n", Theta2);
	fclose(fp3);

}

/*******************************************/
/*This function is used to calculate sigmod*/
double sigmod(double x) {
	return 1.0 / (1.0 + pow(2.71828183, -x));
}
/*******************************************/
/*This function is used to test and return the correction rate*/
double test(double* weight1, double* weight2, double* theta1, double theta2)
{
	double temp[361] = {0};
	double Error = 0;
	int i = 0;
	for (; i < 361; i++)
	{
		double input = 2.0 * PI *i / 361;
		double HiddenOutput[Hidden];
		for (int count2 = 0; count2 < Hidden; count2++) {
			HiddenOutput[count2] = sigmod(weight1[count2] * input + theta1[count2]);
		}
		double ActualOutput = 0;
		for (int count3 = 0; count3 < Hidden; count3++) {
			ActualOutput += weight2[count3] * HiddenOutput[count3];
		}
		ActualOutput += theta2;
		temp[i - 1] = ActualOutput;
		Error += (0.5 * pow((ActualOutput - sin(input)), 2));
	}
	
	FILE* fp2;
	fp2 = fopen("D:\\approximation.txt","w");
	for (int i = 0; i < 361; i++) {
		fprintf(fp2, "%lf\n", temp[i]);
	}
	fclose(fp2);

	double AveErr = Error * 1.0 / i;
	return AveErr;
}