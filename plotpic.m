clear
clc

table = load('approximation.txt');
err = load('err.txt');
x = table(1:end,1);
y = table(1:end, 2);
z = table(1:end, 3);
d = table(1:end, 4);
err = table(1:end,5);

[X,Y,Z] = griddata(x,y,z,linspace(min(x),max(x))',linspace(min(y),max(y)),'v4');
figure
mesh(X,Y,Z)
saveas(gcf , "ActualOutput,png");

[U,V,W] = griddata(x,y,d,linspace(min(x),max(x))',linspace(min(y),max(y)),'v4');
figure
mesh(U,V,W)
saveas(gcf , "DesiredOutput,png");

[B,N,M] =griddata(x,y,err,linspace(min(x),max(x))',linspace(min(y),max(y)),'v4');
figure
mesh(B,N,M)
saveas(gcf , "Error,png");

figure
plot(1:length(err), err);
title('error variety')
xlabel('Training times')
ylabel('squared-error')
saveas(gcf, "Err.png")
