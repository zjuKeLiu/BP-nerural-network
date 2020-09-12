clear
clc

approximation = load('approximation.txt');
err = load('err.txt');
weight = load('weight.txt');

figure 
plot(linspace(0,2*pi,361),sin(linspace(0,2*pi,361)));
hold on
plot(linspace(0,2*pi,361), approximation,'.-');
title('Actual result and  approximation result')
xlabel('Input data')
ylabel('Output data')
legend('Actual result','Aproximation result');

%Test the accuracy
figure
plot(1:length(err), err);
title('error variety(6 hidden neurons)')
xlabel('Training times')
ylabel('squared-error')
axis([0,length(err) , 0,0.5]);
