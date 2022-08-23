clear;
clc;
x = 1:10;
y = sin(x);
err = rand(1,10);
bar(x,y);
hold on;
errorbar(x,y,err,'o','linewidth',1.5)
