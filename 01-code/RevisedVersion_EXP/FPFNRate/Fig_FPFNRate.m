clear;
clc;

%% This Code is to process the result of *.csv from \01-code\main\MainSimulator_varygen.py

%% color
color1 = [0.12156862745098039 0.4666666666666667  0.7058823529411765];
color2 = [1.0  0.4980392156862745 0.054901960784313725];
color3 = [0.17254901960784313 0.6274509803921569 0.17254901960784313];
color4 = [0.8392156862745098, 0.15294117647058825, 0.1568627450980392];
color5 = [0.5803921568627451, 0.403921568627451, 0.7411764705882353];
color6 = [0.5490196078431373, 0.33725490196078434, 0.29411764705882354];
% legend('TTPM','Epidemic','Prophet','Spray&Wait','DAS','Smart');
c = [color4;color1];
%% data source
Input = "FPFNrate.csv";

%% get parameters
num = size(Input)

%%tmp para 为了获得参数 先执行一下
%Input(1,i)
all_res = csvread(Input);

figure(1)
hold on;
plot(all_res(:,1),all_res(:,2),'color',color4,'LineWidth',2);
plot(all_res(:,1),all_res(:,3),'color',color1,'LineWidth',2);
grid on;
set(gca, 'Fontname', 'Times New Roman','FontSize',16);
legend('False Positive Rate','False Negative Rate','Location','SouthEast');
xlabel('Threshold');

