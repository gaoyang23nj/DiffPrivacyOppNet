% 天内 宏观
clear;
clc;
res = csvread('intra_day.csv');
figure('Name','Varying P','Position',[500 500 800 450]);
hold on;
x = res(1,:);
C1 = res(2,:);
C2 = res(3,:);
P1 = res(4,:);
P2 = res(5,:); 
set(gca, 'Fontname', 'Times New Roman','FontSize',18);
%xlim([0, 18]);
[AX,H1,H2] = plotyy(x,C1,x,P1,'plot');
set(H1,'LineStyle','-','marker','o','color','#1F77B4','LineWidth',2,'MarkerSize',8)
set(H2,'LineStyle','-','marker','o','color','#D62728','LineWidth',2,'MarkerSize',8)
axes(AX(1));
hold on;
plot(x,C2,'LineStyle','--','marker','*','color','#1F77B4','LineWidth',2,'MarkerSize',8);
axes(AX(2));
hold on;
plot(x,P2,'LineStyle','--','marker','*','color','#D62728','LineWidth',2,'MarkerSize',8);

set(AX(1),'xlim',[0 23]);
set(AX(2),'xlim',[0 23]);
set(AX(1),'xTick',[0:4:23]);
set(AX(1),'xTicklabel',[0:4:23]);
set(AX(2),'xTick',[0:4:23]);
set(AX(2),'xTicklabel',[0:4:23]);

set(AX(2),'ylim',[0 0.2]);
% set(AX(2),'yTick',[0:0.06:0.2]);
% set(AX(2),'yTicklabel',[0:0.06:0.2]);

set(AX(1),'ylim',[0 1300]);
set(AX(1),'yTick',[0:200:1200]);
set(AX(1),'yTicklabel',[0:200:1200]);



set(gca, 'Fontname', 'Times New Roman','FontSize',18);
% lim1=get(AX(1),'yTick');

legend('$p_{210,183}(x)$','$p_{183,210}(x)$','$210 \rightarrow 183$','$183 \rightarrow 210$','Interpreter','LaTex','fontsize',18)
d1=get(AX(1),'ylabel');
% set(d1,'string','yayacpf','fontsize',18,'FontName','宋体');
%set(d1,'string','# of trips in an hour','fontsize',20,'FontName','Times New Roman','FontWeight','bold');
set(d1,'string','# of trips in an hour','fontsize',20,'FontName','Times New Roman','color','#1F77B4');
d2=get(AX(2),'ylabel');
%set(d2,'string','Prob. density','fontsize',20,'FontName','Times New Roman','FontWeight','bold');
set(d2,'string','Prob. density','fontsize',20,'FontName','Times New Roman','color','#D62728');

d0=get(AX(1),'xlabel');
%set(d0,'string','Hour','fontsize',20,'FontName','Times New Roman','FontWeight','bold','position',[12 -80]);
set(d0,'string','Hour','fontsize',20,'FontName','Times New Roman','position',[12 -80]);

grid on;
