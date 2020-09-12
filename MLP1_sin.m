clear
clc

%%
%Train data
Traindata_I = linspace(0,2*pi,9);
Traindata_T = sin(Traindata_I); 
%test data
Testdata_I = linspace(0,2*pi,361);
Testdata_T = sin(Testdata_I);

%%
%initial
HiddenNeuron = 6; %Number of hidden neurons
MaxTimes = 10000000; %Maximum number of training
MinErr =0.00000001; %Margin of error
W1 = rand(HiddenNeuron,1); %Initial the weight of output of input neurons
W2 = rand(1,HiddenNeuron); %Initial the weight of output of hidden neurons
Theta1 = rand(HiddenNeuron,1);
Theta2 = rand(1);
%Fix the random data, so we can compare the other elements
%W1 = [0.959492426392903;0.655740699156587;0.0357116785741896;0.849129305868777;0.933993247757551;0.678735154857774];
%W2 = [0.757740130578333,0.743132468124916,0.392227019534168,0.655477890177557,0.171186687811562,0.706046088019609];
%Theta1 = [0.0318328463774207;0.276922984960890;0.0461713906311539;0.0971317812358475;0.823457828327293;0.694828622975817];
%Theta2 = 0.317099480060861;

Alpha =0.05;% %0.01;  0.1;            %learning rate
Eta = 0.005; %0.001;   0.01;             %floating item
ErrRecord = zeros(1,MaxTimes);
%Train 
for i = 1:MaxTimes 
   HiddenOutput = logsig(W1 * Traindata_I + Theta1); %Each column is the  
                                                    %output of Hidden layer
   ActualOutput = W2 * HiddenOutput + Theta2; %A row vector
   Err = Traindata_T - ActualOutput;
   ErrorSum = sumsqr(Err)/2; 
   ErrRecord(i) = ErrorSum;
   if ErrorSum <MinErr          %Accurate enough 
       break;
   end
   %Error back propagation
   delta2 = HiddenOutput * Err'; %Column Vector,Output layer
   delta_Theta2 = sum(Err);
   %delta1 = sum(ActualOutput .* (1 - ActualOutput) .*Err )* W2; %Row vetor,
   temp1 = ones(1,9);
   delta1 =  HiddenOutput.*(1 - HiddenOutput).*(W2'*temp1)*(Err .* Traindata_I)';
   delta_Theta1 = HiddenOutput.*(1 - HiddenOutput).*(W2'*temp1)*Err';
   %Change the value of W1,W2 and Theta;
   W2 = W2 + Alpha * delta2';
   W1 = W1 + Alpha * delta1 ; 
   Theta1 = Theta1 + Eta * delta_Theta1;
   Theta2 = Theta2 + Eta * delta_Theta2; 
end

%%
%calculate the results
temp2 = ones(1,361);
ActualOutputOfNN = W2 * logsig(W1 * Testdata_I + Theta1) + Theta2;
TestErr = sum((ActualOutputOfNN - Testdata_T).^2)*0.5/361;
%plot pictures
figure 
plot(Testdata_I,Testdata_T);
hold on
plot(Testdata_I, ActualOutputOfNN,'.-');
title('Actual result and  approximation result')
xlabel('Input data')
ylabel('Output data')
legend('Actual result','Aproximation result');
%Test the accuracy
figure
plot(1:MaxTimes, ErrRecord);
title('error variety')
xlabel('Training times')
ylabel('squared-error')
axis([0,100 , 0,50]);


