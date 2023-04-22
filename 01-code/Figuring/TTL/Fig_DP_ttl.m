clear;
clc;
%% color
color1 = [0.12156862745098039 0.4666666666666667  0.7058823529411765];
color2 = [1.0  0.4980392156862745 0.054901960784313725];
color3 = [0.17254901960784313 0.6274509803921569 0.17254901960784313];
color4 = [0.8392156862745098, 0.15294117647058825, 0.1568627450980392];
color5 = [0.5803921568627451, 0.403921568627451, 0.7411764705882353];
color6 = [0.5490196078431373, 0.33725490196078434, 0.29411764705882354];
c = [color1;color4;color6;color3;color2;color5];
% color1 = [0.12156862745098039 0.4666666666666667  0.7058823529411765];
% color2 = [1.0  0.4980392156862745 0.054901960784313725];
% color3 = [0.17254901960784313 0.6274509803921569 0.17254901960784313];
% color4 = [0.8392156862745098, 0.15294117647058825, 0.1568627450980392];
% color5 = [0.5803921568627451, 0.403921568627451, 0.7411764705882353];
% color6 = [0.5490196078431373, 0.33725490196078434, 0.29411764705882354];

%% Files
Inputfiles = ["res_oppnet_varyttl_qiaobeidata_20230420004728_new.csv",
    "res_oppnet_varyttl_qiaobeidata_20230421105959_new.csv"];
number_algs = 5;
number_ttls = 4;
% dr dd cw
number_metrics = 3;
%number_pos_neg_avg = 3;
res = zeros(number_algs*number_ttls, number_metrics);
for i=1:length(Inputfiles)
    tmp_res = csvread(Inputfiles(i));
    tmp_res = tmp_res(:,1:number_metrics);
    res = res+tmp_res;
end
res = res/length(Inputfiles);
err_res_neg = zeros(number_algs*number_ttls, number_metrics);
err_res_pos = zeros(number_algs*number_ttls, number_metrics);
for i=1:length(Inputfiles)
    tmp_res = csvread(Inputfiles(i));
    tmp_res = tmp_res(:,1:number_metrics);
    % neg err
    tmp_err_res = res - tmp_res;
    for j=1:number_algs*number_ttls
        for k=1:number_metrics
            if tmp_err_res(j,k) > err_res_neg(j,k)
                err_res_neg(j,k) = tmp_err_res(j,k);
            end
        end
    end
    % tmp_res - res; pos err
    tmp_err_res = tmp_res - res;
    for j=1:number_algs*number_ttls
        for k=1:number_metrics
            if tmp_err_res(j,k) > err_res_pos(j,k)
                err_res_pos(j,k) = tmp_err_res(j,k);
            end
        end
    end    
end


% len = size(res);
% %逐行筛选Nan
% for i=1:len(1)
%     for j=1:len(2)
%         if isnan(res(i,j))
%             res(i,j)=99999;
%         end
%     end
% end

figure(1);
%投递率
dr = res(:,1);
err_dr_neg = err_res_neg(:,1);
err_dr_pos = err_res_pos(:,1);
%reshape 是按照列读取 按照列摆放
dr_group = (reshape(dr,number_algs,number_ttls))';
err_dr_group_neg = (reshape(err_dr_neg,number_algs,number_ttls))';
err_dr_group_pos = (reshape(err_dr_pos,number_algs,number_ttls))';
h = bar(dr_group);
set(h(1),'FaceColor',c(1,:));
set(h(2),'FaceColor',c(2,:));
set(h(3),'FaceColor',c(3,:)); 
set(h(4),'FaceColor',c(4,:)); 
set(h(5),'FaceColor',c(5,:)); 
hold on;
%location, negative value, postive value
delta = 0.15;
% errorbar([1,2,3,4], dr_group(:,3), err_dr_group_neg(:,3), err_dr_group_pos(:,3), 'k', 'Linestyle', 'None');
% errorbar([1+delta,2+delta,3+delta,4+delta], dr_group(:,4), err_dr_group_neg(:,4), err_dr_group_pos(:,4), 'k', 'Linestyle', 'None');
% errorbar([1-delta,2-delta,3-delta,4-delta], dr_group(:,2), err_dr_group_neg(:,2), err_dr_group_pos(:,2), 'k', 'Linestyle', 'None');
% errorbar([1+2*delta,2+2*delta,3+2*delta,4+2*delta], dr_group(:,5), err_dr_group_neg(:,5), err_dr_group_pos(:,5), 'k', 'Linestyle', 'None');
% errorbar([1-2*delta,2-2*delta,3-2*delta,4-2*delta], dr_group(:,1), err_dr_group_neg(:,1), err_dr_group_pos(:,1), 'k', 'Linestyle', 'None');
set(gca, 'Fontname', 'Times New Roman','FontSize',16);
set(gca, 'XTick',1:4, 'xticklabels',{'1d','2d','3d','4d','Interpreter','LaTex'});
xlabel("Message Lifespan $T_m$" ,'interpreter','latex');
ylabel('Delivery Ratio');
legend('TTPM+Lap','TTPM+DODP','TTPM+GRR','TTPM+StairDP','TTPM+NoPvc','Location','SouthEast');

% improvement ratio
(dr_group(:,2)-dr_group(:,1))./dr_group(:,1)
(dr_group(:,2)-dr_group(:,3))./dr_group(:,3)
(dr_group(:,2)-dr_group(:,4))./dr_group(:,4)
%(dr_group(:,2)-dr_group(:,5))./dr_group(:,5)

figure(2);
%平均投递延迟
dd = res(:,2)./3600;
err_dd_neg = err_res_neg(:,2)./3600;
err_dd_pos = err_res_pos(:,2)./3600;
%reshape 是按照列读取 按照列摆放
dd_group = (reshape(dd,number_algs,number_ttls))';
err_dd_group_neg = (reshape(err_dd_neg,number_algs,number_ttls))';
err_dd_group_pos = (reshape(err_dd_pos,number_algs,number_ttls))';
h = bar(dd_group);
set(h(1),'FaceColor',c(1,:));
set(h(2),'FaceColor',c(2,:));
set(h(3),'FaceColor',c(3,:)); 
set(h(4),'FaceColor',c(4,:)); 
set(h(5),'FaceColor',c(5,:)); 
hold on;
delta = 0.15;
% errorbar([1,2,3,4], dd_group(:,3), err_dd_group_neg(:,3), err_dd_group_pos(:,3), 'k', 'Linestyle', 'None');
% errorbar([1+delta,2+delta,3+delta,4+delta], dd_group(:,4), err_dd_group_neg(:,4), err_dd_group_pos(:,4), 'k', 'Linestyle', 'None');
% errorbar([1-delta,2-delta,3-delta,4-delta], dd_group(:,2), err_dd_group_neg(:,2), err_dd_group_pos(:,2), 'k', 'Linestyle', 'None');
% errorbar([1+2*delta,2+2*delta,3+2*delta,4+2*delta], dd_group(:,5), err_dd_group_neg(:,5), err_dd_group_pos(:,5), 'k', 'Linestyle', 'None');
% errorbar([1-2*delta,2-2*delta,3-2*delta,4-2*delta], dd_group(:,1), err_dd_group_neg(:,1), err_dd_group_pos(:,1), 'k', 'Linestyle', 'None');
set(gca, 'Fontname', 'Times New Roman','FontSize',16);
set(gca, 'XTick',1:4, 'xticklabels',{'1d','2d','3d','4d','Interpreter','LaTex'});
xlabel("Message Lifespan $T_m$" ,'interpreter','latex');
ylabel('Average Delivery Latency (hours)');
legend('TTPM+Lap','TTPM+DODP','TTPM+GRR','TTPM+StairDP','TTPM+NoPvc','Location','SouthEast');

figure(3);
%平均投递延迟
cc = res(:,3);
err_cc_neg = err_res_neg(:,3);
err_cc_pos = err_res_pos(:,3);
%reshape 是按照列读取 按照列摆放
cc_group = (reshape(cc,number_algs,number_ttls))';
err_cc_group_neg = (reshape(err_cc_neg,number_algs,number_ttls))';
err_cc_group_pos = (reshape(err_cc_pos,number_algs,number_ttls))';
h = bar(cc_group);
set(h(1),'FaceColor',c(1,:));
set(h(2),'FaceColor',c(2,:));
set(h(3),'FaceColor',c(3,:)); 
set(h(4),'FaceColor',c(4,:)); 
set(h(5),'FaceColor',c(5,:)); 
hold on;
delta = 0.15;
% errorbar([1,2,3,4], cc_group(:,3), err_cc_group_neg(:,3), err_cc_group_pos(:,3), 'k', 'Linestyle', 'None');
% errorbar([1+delta,2+delta,3+delta,4+delta], cc_group(:,4), err_cc_group_neg(:,4), err_cc_group_pos(:,4), 'k', 'Linestyle', 'None');
% errorbar([1-delta,2-delta,3-delta,4-delta], cc_group(:,2), err_cc_group_neg(:,2), err_cc_group_pos(:,2), 'k', 'Linestyle', 'None');
% errorbar([1+2*delta,2+2*delta,3+2*delta,4+2*delta], cc_group(:,5), err_cc_group_neg(:,5), err_cc_group_pos(:,5), 'k', 'Linestyle', 'None');
% errorbar([1-2*delta,2-2*delta,3-2*delta,4-2*delta], cc_group(:,1), err_cc_group_neg(:,1), err_cc_group_pos(:,1), 'k', 'Linestyle', 'None');

set(gca, 'Fontname', 'Times New Roman','FontSize',16);
set(gca, 'XTick',1:4, 'xticklabels',{'1d','2d','3d','4d','Interpreter','LaTex'});
xlabel("Message Lifespan $T_m$" ,'interpreter','latex');
ylabel('Comm. Load (#. of comms)');
legend('TTPM+Lap','TTPM+DODP','TTPM+GRR','TTPM+StairDP','TTPM+NoPvc','Location','SouthEast');


