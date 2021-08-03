clear;
clc;

%% 本程序 对MainSimulator.py得到的*.csv统计处理 并画图

%% 数据来源
inputfile1 = "res_dpoppnet_20210802220642.csv";
%inputfile2 = "res_dpoppnet_20210726191406.csv";
Input = [inputfile1,inputfile1];

%% 获得参数
num = size(Input);
num = num(2);
%%tmp para 为了获得参数 先执行一下
%Input(1,i)
all_res = csvread(Input(1,1));
size_res = size(all_res);
%Our,Epidemic,Prophet,SparyandWait 路由算法的个数
num_routings = 4;
num_rows = size_res(1,1);
num_columns = size_res(1,2);
num_rates = num_rows/num_routings;  
all_delivery_ratio = zeros(num_routings,num_rates);
all_delivery_latency = zeros(num_routings,num_rates);
all_delivery_load = zeros(num_routings,num_rates);

%% 读取属性值
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
        delivery_ratio(:,i) = all_res((i-1)*4+1:i*4,1);
        delivery_latency(:,i) = all_res((i-1)*4+1:i*4,2)./3600;
        delivery_load(:,i) = all_res((i-1)*4+1:i*4,3)./all_res((i-1)*4+1:i*4,5);
        para_rate(1,i) = 24*3600/all_res((i-1)*4+1,6);
    end
    all_delivery_ratio = all_delivery_ratio + delivery_ratio;
    all_delivery_latency = all_delivery_latency + delivery_latency;
    all_delivery_load = all_delivery_load + delivery_load;
end
delivery_ratio = all_delivery_ratio./num
delivery_latency = all_delivery_latency./num
delivery_load = all_delivery_load./num

%% 性能画图
%delivery ratio: 0~1
figure(1)
hold on;
plot(para_rate,delivery_ratio(1,:),'color','red','linestyle','-','marker','^');
plot(para_rate,delivery_ratio(2,:),'color','black','linestyle','--','marker','o')
plot(para_rate,delivery_ratio(3,:),'color','magenta','linestyle','--','marker','s')
plot(para_rate,delivery_ratio(4,:),'color','blue','linestyle','--','marker','v')
legend('Our','Epidemic','Prophet','Spary & Wait')
xlim([min(para_rate) max(para_rate)]);
ylabel('Delivery Ratio')
xlabel('Message Generation Rate (pkts/day)')
% max_ratio = max(max(delivery_ratio)) * 1.5;
% ylim([0 max_ratio])
set(gca, 'Fontname', 'Times New Roman','FontSize',12);

%Average Delivery Latency: seconds 
figure(2)
hold on;
plot(para_rate,delivery_latency(1,:),'color','red','linestyle','-','marker','^');
plot(para_rate,delivery_latency(2,:),'color','black','linestyle','--','marker','o')
plot(para_rate,delivery_latency(3,:),'color','magenta','linestyle','--','marker','s')
plot(para_rate,delivery_latency(4,:),'color','blue','linestyle','--','marker','v')
legend('Our','Epidemic','Prophet','Spary & Wait')
% max_latency = max(max(delivery_latency)) * 1.5;
% ylim([0 max_latency])
xlim([min(para_rate) max(para_rate)]);
ylabel('Average Delivery Latency (hours)')
xlabel('Message Generation Rate (pkts/day)')
set(gca, 'Fontname', 'Times New Roman','FontSize',12);

%Communication Workload
figure(3)
hold on;
plot(para_rate,delivery_load(1,:),'color','red','linestyle','-','marker','^');
plot(para_rate,delivery_load(2,:),'color','black','linestyle','--','marker','o')
plot(para_rate,delivery_load(3,:),'color','magenta','linestyle','--','marker','s')
plot(para_rate,delivery_load(4,:),'color','blue','linestyle','--','marker','v')
legend('Our','Epidemic','Prophet','Spary & Wait')
% max_load = max(max(delivery_load)) * 1.5;
% ylim([0 max_load])
xlim([min(para_rate) max(para_rate)]);
ylabel('Communication Workload (no. comms/pkt)')
xlabel('Message Generation Rate (pkts/day)')
set(gca, 'Fontname', 'Times New Roman','FontSize',12);

