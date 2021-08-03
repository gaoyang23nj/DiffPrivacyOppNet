clear;
clc;

%% 本程序 对MainSimulator.py得到的*.csv统计处理 并画图

%% 数据来源

%% 获得参数
num_routings = 4;

%% 读取属性值
inputfile = "res_dplapoppnet_20210802173134.csv";
all_res = csvread(inputfile);
% 各列属性 succ_ratio,avg_delay,num_comm,num_gen,num_succ,gen_freq,privacy parameter
delivery_ratio = all_res(1:num_routings,1);
delivery_latency = all_res(1:num_routings,2)./3600;
delivery_load = all_res(1:num_routings,3)./all_res(1:num_routings,5);
para_rate = 24*3600/all_res(1,6);
para_pp = all_res(1:num_routings,8);


%% 性能画图
%delivery ratio: 0~1
figure(1)
hold on;
plot(para_pp,delivery_ratio,'color','red','linestyle','-','marker','^');
xlim([min(para_pp) max(para_pp)]);
ylabel('Delivery Ratio')
xlabel('\epsilon')
% max_ratio = max(max(delivery_ratio)) * 1.5;
% ylim([0 max_ratio])
set(gca, 'Fontname', 'Times New Roman','FontSize',12);

%Average Delivery Latency: seconds 
figure(2)
hold on;
plot(para_pp,delivery_latency,'color','red','linestyle','-','marker','^');
xlim([min(para_pp) max(para_pp)]);
ylabel('Average Delivery Latency (hours)')
xlabel('\epsilon')
set(gca, 'Fontname', 'Times New Roman','FontSize',12);

%Communication Workload
figure(3)
hold on;
plot(para_pp,delivery_load,'color','red','linestyle','-','marker','^');
% max_load = max(max(delivery_load)) * 1.5;
% ylim([0 max_load])
xlim([min(para_pp) max(para_pp)]);
ylabel('Communication Workload (no. comms/pkt)')
xlabel('\epsilon')
set(gca, 'Fontname', 'Times New Roman','FontSize',12);

