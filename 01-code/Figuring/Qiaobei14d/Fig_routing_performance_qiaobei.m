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
c = [color4;color2;color3;color1;color5;color6];
%% data source
Input = ["res_oppnet_varygen_qiaobeidata_20230413011850_full.csv",
    "res_oppnet_varygen_qiaobeidata_20230414192541_full.csv"];

%% get parameters
num = size(Input);
num = num(2);
%%tmp para 为了获得参数 先执行一下
%Input(1,i)
all_res = csvread(Input(1,1));
size_res = size(all_res);
%Our,Epidemic,Prophet,SprayandWait,DAS,Smart. number of routing methods
num_routings = 6;
num_rows = size_res(1,1);
num_columns = size_res(1,2);
num_rates = num_rows/num_routings;  
all_delivery_ratio = zeros(num_routings,num_rates);
all_delivery_latency = zeros(num_routings,num_rates);
all_delivery_load = zeros(num_routings,num_rates);

%% read the attributes
for j=1:num
    %Input(1,i)
    inputfile = Input(1,j)
    all_res = csvread(inputfile);
   
    delivery_ratio = zeros(num_routings,num_rates);
    delivery_latency = zeros(num_routings,num_rates);
    delivery_load = zeros(num_routings,num_rates);
    para_rate = zeros(1,num_rates);
    % 各列属性 succ_ratio,avg_delay,num_comm,num_gen,num_succ,gen_freq,privacy parameter
    for i=1:num_rates
        delivery_ratio(:,i) = all_res((i-1)*num_routings+1 :i*num_routings,1);
        delivery_latency(:,i) = all_res((i-1)*num_routings+1 :i*num_routings,2)./3600;
%        delivery_load(:,i) = all_res((i-1)*num_routings+1:i*num_routings,3)./all_res((i-1)*num_routings+1:i*num_routings,5);
        delivery_load(:,i) = all_res((i-1)*num_routings+1:i*num_routings,3);
        para_rate(1,i) = 3600/all_res((i-1)*num_routings+1,6);
    end
    all_delivery_ratio = all_delivery_ratio + delivery_ratio;
    all_delivery_latency = all_delivery_latency + delivery_latency;
    all_delivery_load = all_delivery_load + delivery_load;
end
delivery_ratio = all_delivery_ratio./num
delivery_latency = all_delivery_latency./num
delivery_load = all_delivery_load./num


%% Performance Figure
%delivery ratio: 0~1
figure(1)
hold on;
h1 = plot(para_rate,delivery_ratio(1,:),'color',c(1,:),'linestyle','-','marker','^','LineWidth',2, 'MarkerSize',8);
h2 = plot(para_rate,delivery_ratio(2,:),'color',c(2,:),'linestyle','--','marker','o','LineWidth',2, 'MarkerSize',8)
h3 = plot(para_rate,delivery_ratio(3,:),'color',c(3,:),'linestyle','-','marker','*','LineWidth',2, 'MarkerSize',8)
h4 = plot(para_rate,delivery_ratio(4,:),'color',c(4,:),'linestyle','--','marker','+','LineWidth',2, 'MarkerSize',8)
h5 = plot(para_rate,delivery_ratio(5,:),'color',c(5,:),'linestyle','-','marker','v','LineWidth',2, 'MarkerSize',8)
h6 = plot(para_rate,delivery_ratio(6,:),'color',c(6,:),'linestyle','--','marker','d','LineWidth',2, 'MarkerSize',8)
grid on;
set(gca, 'Fontname', 'Times New Roman','FontSize',16);
xlim([min(para_rate) max(para_rate)]);
max_ratio = max(max(delivery_ratio))
ylim([0. 1])
ylabel('Delivery Ratio', 'Fontname', 'Times New Roman');
xlabel('Message Generation Rate (pkts/hour)','Fontname', 'Times New Roman');
grid on;
lgd1=legend([h1,h2,h3],'TTPM+NoPvc','Epidemic','Prophet','orientation','horizontal','location','north');
legend boxoff
set(lgd1,'Fontname', 'Times New Roman','FontSize',16);
ah=axes('position',get(gca,'position'),'visible','off');
lgd2=legend(ah,[h4,h5,h6],'Spray & Wait','DAS','SMART','orientation','horizontal','location','north');
legend boxoff
set(lgd2,'Fontname', 'Times New Roman','FontSize',16);

%Average Delivery Latency: seconds 
figure(2)
hold on;
h1 = plot(para_rate,delivery_latency(1,:),'color',c(1,:),'linestyle','-','marker','^','LineWidth',2, 'MarkerSize',8);
h2 = plot(para_rate,delivery_latency(2,:),'color',c(2,:),'linestyle','--','marker','o','LineWidth',2, 'MarkerSize',8)
h3 = plot(para_rate,delivery_latency(3,:),'color',c(3,:),'linestyle','-','marker','*','LineWidth',2, 'MarkerSize',8)
h4 = plot(para_rate,delivery_latency(4,:),'color',c(4,:),'linestyle','--','marker','+','LineWidth',2, 'MarkerSize',8)
h5 = plot(para_rate,delivery_latency(5,:),'color',c(5,:),'linestyle','-','marker','v','LineWidth',2, 'MarkerSize',8)
h6 = plot(para_rate,delivery_latency(6,:),'color',c(6,:),'linestyle','--','marker','d','LineWidth',2, 'MarkerSize',8)
grid on;
set(gca, 'Fontname', 'Times New Roman','FontSize',16);
xlim([min(para_rate) max(para_rate)]);
max_latency = max(max(delivery_latency))
ylim([min(min(delivery_latency)) max_latency*1.18])
ylabel('Average Delivery Latency (hours)');
xlabel('Message Generation Rate (pkts/hour)');
grid on;
lgd1=legend([h1,h2,h3],'TTPM+NoPvc','Epidemic','Prophet','orientation','horizontal','location','north');
legend boxoff
set(lgd1,'Fontname', 'Times New Roman','FontSize',16);
ah=axes('position',get(gca,'position'),'visible','off');
lgd2=legend(ah,[h4,h5,h6],'Spray & Wait','DAS','SMART','orientation','horizontal','location','north');
legend boxoff
set(lgd2,'Fontname', 'Times New Roman','FontSize',16);

%Communication Workload
figure(3)
hold on;
h1 = plot(para_rate,delivery_load(1,:),'color',c(1,:),'linestyle','-','marker','^','LineWidth',2, 'MarkerSize',8);
h2 = plot(para_rate,delivery_load(2,:),'color',c(2,:),'linestyle','--','marker','o','LineWidth',2, 'MarkerSize',8)
h3 = plot(para_rate,delivery_load(3,:),'color',c(3,:),'linestyle','-','marker','*','LineWidth',2, 'MarkerSize',8)
h4 = plot(para_rate,delivery_load(4,:),'color',c(4,:),'linestyle','--','marker','+','LineWidth',2, 'MarkerSize',8)
h5 = plot(para_rate,delivery_load(5,:),'color',c(5,:),'linestyle','-','marker','v','LineWidth',2, 'MarkerSize',8)
h6 = plot(para_rate,delivery_load(6,:),'color',c(6,:),'linestyle','--','marker','d','LineWidth',2, 'MarkerSize',8)
grid on;
set(gca, 'Fontname', 'Times New Roman','FontSize',16);
max_load = max(max(delivery_load));
xlim([min(para_rate) max(para_rate)]);
ylim([0 max_load*1.18])
ylabel('Comm. Load (#. of comms)');
xlabel('Message Generation Rate (pkts/hour)');
grid on;
lgd1=legend([h1,h2,h3],'TTPM+NoPvc','Epidemic','Prophet','orientation','horizontal','location','north');
legend boxoff
set(lgd1,'Fontname', 'Times New Roman','FontSize',16);
ah=axes('position',get(gca,'position'),'visible','off');
lgd2=legend(ah,[h4,h5,h6],'Spray & Wait','DAS','SMART','orientation','horizontal','location','north');
legend boxoff
set(lgd2,'Fontname', 'Times New Roman','FontSize',16);

% small figure
set(groot,'defaultAxesLineStyleOrder','remove','defaultAxesColorOrder','remove');
xyh2=axes('Position',[0.6 0.3 0.3 0.3]);
axes(xyh2);                 
%plot(rand(10,3));
%plot(x,y)
h1 = plot(para_rate,delivery_load(1,:),'color',c(1,:),'linestyle','-','marker','^','LineWidth',2, 'MarkerSize',8);
hold on;
h2 = plot(para_rate,delivery_load(2,:),'color',c(2,:),'linestyle','--','marker','o','LineWidth',2, 'MarkerSize',8)
h3 = plot(para_rate,delivery_load(3,:),'color',c(3,:),'linestyle','-','marker','*','LineWidth',2, 'MarkerSize',8)
h4 = plot(para_rate,delivery_load(4,:),'color',c(4,:),'linestyle','--','marker','+','LineWidth',2, 'MarkerSize',8)
h5 = plot(para_rate,delivery_load(5,:),'color',c(5,:),'linestyle','-','marker','v','LineWidth',2, 'MarkerSize',8)
h6 = plot(para_rate,delivery_load(6,:),'color',c(6,:),'linestyle','--','marker','d','LineWidth',2, 'MarkerSize',8)
set(xyh2,'xlim',[12 18]);
set(xyh2,'ylim',[0 50000]);
set(xyh2,'Fontname', 'Times New Roman');