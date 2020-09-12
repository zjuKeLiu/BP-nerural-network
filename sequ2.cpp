/*BP neural networks trained in sequence*/

#include<stdio.h>
#include<math.h>
#include<stdlib.h>

double sigmod(double x);
double test(double* weight1, double* weight2, double* theta1, double theta2);
double DesiredOutput(double x1, double x2);

constexpr auto Hidden = 24;//6;//9;//12;
constexpr auto InputNum = 2;
constexpr auto MaxTime = 15000000;
constexpr auto eps = 0.0000000000000001;
void main()
{
	/*initial the train data*/
	double X1[22] = { -10, -8, - 6, - 4, - 2, 0, 2, 4,	6, 8, 10,-10, -8, -6, -4, - 2, 0, 2, 4,	6, 8, 10 };
	//output data can be calculated we it is used;
	/*************************/
	/****The train process****/
	//initial the weight and bias(Fix the random data, so we can compare the other elements)
	double W1[Hidden * InputNum];
	double Theta1[Hidden];
	double W2[Hidden];
	double Theta2 = ((rand() % 10) - 5) / 5.0;
	for (int i = 0; i < Hidden; i++) {
		for (int j = 0; j < InputNum; j++) {
			W1[j*Hidden + i] = ((rand() % 10) - 5) / 5.0;
		}
		W2[i] = ((rand() % 10) - 5) / 5.0;
		Theta1[i] = ((rand() % 10) - 5) / 5.0;
	}
	//double W1[Hidden * InputNum] = { 0.959492426392903, 0.655740699156587, 0.0357116785741896 , 0.849129305868777, 0.933993247757551, 0.678735154857774 ,0.959492426392903 , 0.655740699156587, 0.0357116785741896 , 0.849129305868777, 0.933993247757551, 0.678735154857774 };// ,0.959492426392903, 0.655740699156587, 0.0357116785741896 //, 0.849129305868777, 0.933993247757551, 0.678735154857774 };//,0.959492426392903 , 0.655740699156587, 0.0357116785741896 ,// 0.849129305868777, 0.933993247757551, 0.678735154857774  };
	//double Theta1[Hidden] = { 0.757740130578333, 0.743132468124916, 0.392227019534168,0.655477890177557,0.171186687811562,0.706046088019609 };//,0.757740130578333, 0.743132468124916 , 0.392227019534168 };// ,0.655477890177557,0.171186687811562,0.706046088019609 };
	//double W2[Hidden] = { 0.0318328463774207, 0.276922984960890, 0.0461713906311539 , 0.0971317812358475, 0.823457828327293, 0.694828622975817 };// ,0.0318328463774207, 0.276922984960890, 0.0461713906311539 };//, 0.0971317812358475, 0.823457828327293, 0.694828622975817 };
	//double Theta2 = 0.317099480060861;
	double Alpha = 0.001;//0.0001;
	double Eta = 0.0001;//0.00001;
	double ErrRecord[MaxTime];
	//define the process variable
	double HiddenOutput[Hidden];
	double HiddenInput[Hidden];
	double DesiredOut;
	int RandNum1;
	int RandNum2;

	for (int count1 = 0; count1 < MaxTime; count1++) {
		double ErrSum = 0;
		for (RandNum1 = 0; RandNum1 < 11; RandNum1++) {
			for (RandNum2 = 0; RandNum2 < 11; RandNum2++) {
				double Input[2] = { X1[RandNum1],X1[RandNum2] };

				DesiredOut = DesiredOutput(Input[0], Input[1]);

				for (int count2 = 0; count2 < Hidden; count2++) {
					HiddenInput[count2] = Input[0] * W1[count2] + Input[1] * W1[count2 + Hidden] + Theta1[count2];
				}//calculate the input of Hidden layer

				for (int count3 = 0; count3 < Hidden; count3++) {
					HiddenOutput[count3] = sigmod(HiddenInput[count3]);
				}//calculate the output of Hidden layer

				double ActualOutput = 0;
				for (int count4 = 0; count4 < Hidden; count4++) {
					ActualOutput = ActualOutput + W2[count4] * HiddenOutput[count4];
				}//calculate the Actualoutput
				ActualOutput += Theta2;

				double Error = DesiredOut - ActualOutput;

				ErrSum = ErrSum + 0.5 * pow(Error, 2);
				//Update W1,W2, Theta1, Theta2
				Theta2 += (Eta * Error);
				for (int count5 = 0; count5 < Hidden * InputNum; count5++) {
					W1[count5] = W1[count5] + Alpha * Error * HiddenOutput[count5 % Hidden] * (1 - HiddenOutput[count5 % Hidden]) * W2[count5 % Hidden] * Input[count5 / Hidden];
					//The above sentences execute first in case W2 changes.
				}
				for (int count6 = 0; count6 < Hidden; count6++) {
					Theta1[count6] = Theta1[count6] + Eta * Error * HiddenOutput[count6] * (1 - HiddenOutput[count6]) * W2[count6];
					W2[count6] = W2[count6] + Alpha * Error * HiddenOutput[count6];
				}
			}
		} 
		ErrRecord[count1] = ErrSum*1.0/121;
		if(count1%1000000 == 0)
			printf("%lf \n", ErrRecord[count1]);
	}
	printf("TestError = %lf\n", test(W1, W2, Theta1, Theta2));
	FILE* fp1;
	fp1 = fopen("D:\\err.txt", "w");
	for (int i = 0; i < MaxTime; i++) {
		fprintf(fp1, "%lf \n", ErrRecord[i]);
	}
	fclose(fp1);
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
	double Error = 0;
	double d;
	double HiddenInput[Hidden];
	double HiddenOutput[Hidden];
	/*initial the test data sets*/
	double testI[21] = { -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	//double testI[11] = { -10, -8, -6, -4, -2, 0, 2, 4,	6, 8, 10 };

	FILE* fp2;
	fp2 = fopen("D:\\approximation.txt", "w");
	for (int i = 0; i < 21; i++) {
		for (int j = 0; j < 21; j++) {
			d = DesiredOutput(testI[i], testI[j]);
			for (int count2 = 0; count2 < Hidden; count2++) {
				HiddenInput[count2] = testI[i] * weight1[count2] + testI[j] * weight1[count2 + Hidden] + theta1[count2];
			}//calculate the input of Hidden layer
			for (int count3 = 0; count3 < Hidden; count3++) {
				HiddenOutput[count3] = sigmod(HiddenInput[count3]);
			}//calculate the output of Hidden layer
			double ActualOutput = 0;
			for (int count4 = 0; count4 < Hidden; count4++) {
				ActualOutput = ActualOutput + weight2[count4] * HiddenOutput[count4];
			}//calculate the Actualoutput
			ActualOutput = ActualOutput + theta2;
			//printf("%lf\n ", (ActualOutput - d));
			//printf("%lf, %lf, %lf, %lf ,%lf\n", testI[i], testI[j], ActualOutput, d, (ActualOutput - d));
			Error += 0.5 * pow((ActualOutput - d), 2);
			fprintf(fp2, "%lf, %lf, %lf, %lf ,%lf\n", testI[i], testI[j], ActualOutput, d, (ActualOutput - d));
		}
		fprintf(fp2, "\n");
	}
	fclose(fp2);
	double AveErr = Error * 1.0 / 441;
	return AveErr;
}

/*****************************************/
/*This function is used to calculate the desired output of x1,x2*/
double DesiredOutput(double x1, double x2)
{
	if (x1 == 0 || x2 == 0) {
		x1 += eps;
		x2 += eps;
	}
	return  (sin(x1) * sin(x2))* 1.0 / (x1 * x2);
}

