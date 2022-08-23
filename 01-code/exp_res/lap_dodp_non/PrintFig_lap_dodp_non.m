clear;
clc;

color1 = [0.12156862745098039 0.4666666666666667  0.7058823529411765];
color2 = [1.0  0.4980392156862745 0.054901960784313725];
color3 = [0.17254901960784313 0.6274509803921569 0.17254901960784313];
% (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)

Inputfiles = ["res_dplapoppnet_20220629192202_new.csv",
    "res_dplapoppnet_20220708144606_new.csv",
    "res_dplapoppnet_20220709080554_new.csv",
    "res_dplapoppnet_20220809174653_new.csv",
    "res_dplapoppnet_20220810094705_new.csv",
    "res_dplapoppnet_20220811093532_new.csv",
    "res_dplapoppnet_20220811095936_new.csv",
    "res_dplapoppnet_20220813113211_new.csv"];
res = zeros(12,9);
for i=1:length(Inputfiles)
    tmp_res = csvread(Inputfiles(i));
    res = res+tmp_res;
end
res = res/length(Inputfiles);
err_res_neg = zeros(12,9);
err_res_pos = zeros(12,9);
for i=1:length(Inputfiles)
    tmp_res = csvread(Inputfiles(i));
    % neg err
    tmp_err_res = res - tmp_res;
    for j=1:12
        for k=1:9
            if tmp_err_res(j,k) > err_res_neg(j,k)
                err_res_neg(j,k) = tmp_err_res(j,k);
            end
        end
    end
    % tmp_res - res; pos err
    tmp_err_res = tmp_res - res;
    for j=1:12
        for k=1:9
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
dr_group = (reshape(dr,3,4))';
err_dr_group_neg = (reshape(err_dr_neg,3,4))';
err_dr_group_pos = (reshape(err_dr_pos,3,4))';
h = bar(dr_group);
set(h(1),'FaceColor',color1);
set(h(2),'FaceColor',color2);
set(h(3),'FaceColor',color3); 

hold on;
errorbar([1,2,3,4], dr_group(:,2), err_dr_group_neg(:,2), err_dr_group_pos(:,2), 'k', 'Linestyle', 'None');
errorbar([1.225,2.225,3.225,4.225], dr_group(:,3), err_dr_group_neg(:,3), err_dr_group_pos(:,3), 'k', 'Linestyle', 'None');
errorbar([0.775,1.775,2.775,3.775], dr_group(:,1), err_dr_group_neg(:,1), err_dr_group_pos(:,1), 'k', 'Linestyle', 'None');
set(gca, 'Fontname', 'Times New Roman','FontSize',16);
set(gca, 'XTick',1:4, 'xticklabels',{'1d','2d','3d','4d','Interpreter','LaTex'});
xlabel("Message Lifespan $T_m$" ,'interpreter','latex');
ylabel('Delivery Ratio');
legend('TTPM+Lap','TTPM+DODP', 'TTPM+NoPvc','Location','SouthEast');

figure(2);
%平均投递延迟
dd = res(:,2)./3600;
err_dd_neg = err_res_neg(:,2)./3600;
err_dd_pos = err_res_pos(:,2)./3600;
%reshape 是按照列读取 按照列摆放
dd_group = (reshape(dd,3,4))';
err_dd_group_neg = (reshape(err_dd_neg,3,4))';
err_dd_group_pos = (reshape(err_dd_pos,3,4))';
bar(dd_group);
hold on;
errorbar([1,2,3,4], dd_group(:,2), err_dd_group_neg(:,2), err_dd_group_pos(:,2), 'k', 'Linestyle', 'None');
errorbar([1.225,2.225,3.225,4.225], dd_group(:,3), err_dd_group_neg(:,3), err_dd_group_pos(:,3), 'k', 'Linestyle', 'None');
errorbar([0.775,1.775,2.775,3.775], dd_group(:,1), err_dd_group_neg(:,1), err_dd_group_pos(:,1), 'k', 'Linestyle', 'None');
set(gca, 'Fontname', 'Times New Roman','FontSize',16);
set(gca, 'XTick',1:4, 'xticklabels',{'1d','2d','3d','4d','Interpreter','LaTex'});
xlabel("Message Lifespan $T_m$" ,'interpreter','latex');
ylabel('Average Delivery Latency (hours)');
legend('TTPM+Lap','TTPM+DODP', 'TTPM+NoPvc','Location','SouthEast');


figure(3);
%平均投递延迟
cc = res(:,3);
err_cc_neg = err_res_neg(:,3);
err_cc_pos = err_res_pos(:,3);
%reshape 是按照列读取 按照列摆放
cc_group = (reshape(cc,3,4))';
err_cc_group_neg = (reshape(err_cc_neg,3,4))';
err_cc_group_pos = (reshape(err_cc_pos,3,4))';
bar(cc_group);
hold on;
errorbar([1,2,3,4], cc_group(:,2), err_cc_group_neg(:,2), err_cc_group_pos(:,2), 'k', 'Linestyle', 'None');
errorbar([1.225,2.225,3.225,4.225], cc_group(:,3), err_cc_group_neg(:,3), err_cc_group_pos(:,3), 'k', 'Linestyle', 'None');
errorbar([0.775,1.775,2.775,3.775], cc_group(:,1), err_cc_group_neg(:,1), err_cc_group_pos(:,1), 'k', 'Linestyle', 'None');
set(gca, 'Fontname', 'Times New Roman','FontSize',16);
set(gca, 'XTick',1:4, 'xticklabels',{'1d','2d','3d','4d','Interpreter','LaTex'});
xlabel("Message Lifespan $T_m$" ,'interpreter','latex');
ylabel('Comm. Load (#. of comms)');
legend('TTPM+Lap','TTPM+DODP', 'TTPM+NoPvc','Location','SouthEast');


