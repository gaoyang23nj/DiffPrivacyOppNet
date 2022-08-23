% Ìì¼ä ºê¹Û
%%
clear;
clc;
res = csvread('inter_day.csv');

x = res(1,:);
C1 = res(2,:);
C2 = res(3,:);
P1 = res(4,:);
P2 = res(5,:); 

figure('Name','Varying P','Position',[500 500 800 340]);
hold on;
set(gca, 'Fontname', 'Times New Roman','FontSize',18);
set(gca, 'YColor', '#4DBEEE');
yyaxis left
y1 = bar(x,C1,'FaceColor','#4DBEEE','BarWidth',1);
ylabel('\# of trips $210 \rightarrow 183$','fontsize',20, 'Interpreter','Latex','color','#1F77B4')
ylim([0 23]);
yyaxis right;
set(gca, 'YColor', '#D62728');
plot(x,P1,'color','#D62728','LineWidth',1.5);
ylabel('Prob. of $210 \rightarrow 183$','fontsize',20, 'Interpreter','Latex','color','#D62728')
ylim([0 1.1]);
legend('$C_{210,183}^{k}$','$P_{210,183}^{k}$','Interpreter','LaTex','fontsize',19,'Position',[0.75 0.6 0.1 0.15]);
xlabel('$k$th day','fontsize',20, 'Interpreter','Latex');
grid on;


figure('Name','Varying P','Position',[500 500 800 340]);
hold on;
set(gca, 'Fontname', 'Times New Roman','FontSize',18);
set(gca, 'YColor','#1F77B4');
yyaxis left
h1 = bar(x,C2,'FaceColor','#4DBEEE','BarWidth',1);
ylabel('\# of trips $35 \rightarrow 4$','fontsize',20, 'Interpreter','Latex','color','#1F77B4')
ylim([0 23])
yyaxis right;
set(gca, 'YColor','#D62728');
plot(x,P2,'color','#D62728','LineWidth',1.5);
ylabel('Prob. of $35 \rightarrow 4$','fontsize',20, 'Interpreter','Latex','color','#D62728')
ylim([0 1.1])
legend('$C_{35,4}^{k}$','$P_{35,4}^{k}$','Interpreter','LaTex','fontsize',19,'Position',[0.75 0.6 0.1 0.15]);
xlabel('$k$th day','fontsize',20, 'Interpreter','Latex');
grid on;

%%
% clear;
% clc;
% res = csvread('inter_day.csv');
% 
% x = res(1,:);
% C1 = res(2,:);
% C2 = res(3,:);
% P1 = res(4,:);
% P2 = res(5,:); 
% %%
% figure(1);
% hold on;
% set(gca, 'Fontname', 'Times New Roman','FontSize',18);
% yyaxis left
% bar(C1,'FaceColor','[0 0 1]');
% yyaxis right;
% plot(P1,'color','blue','LineWidth',2);
% 
% yyaxis left
% bar(C2,'FaceColor','[0.24 0.35 0.67]');
% yyaxis right;
% plot(P2,'color','red','LineWidth',2);
% 
% %%
% figure(2);
% hold on;
% set(gca, 'Fontname', 'Times New Roman','FontSize',18);
% yyaxis left
% bar(C2,'red');
% yyaxis right;
% plot(P2,'color','red','LineWidth',2);
% 
% yyaxis left
% bar(C1,'blue');
% yyaxis right;
% plot(P1,'color','blue','LineWidth',2);


%%
% figure(3);
% hold on;
% set(gca, 'Fontname', 'Times New Roman','FontSize',18);
% yyaxis left
% h1=bar([C1;C2]','grouped','BarWidth',1);
% set(h1(1),'FaceColor','[0 0.4470 0.7410]');
% set(h1(2),'FaceColor','[0.3010 0.7450 0.9330]');
% yyaxis right;
% plot(P1,'color','blue','LineWidth',2);
% 
% yyaxis right;
% plot(P2,'color','red','LineWidth',2);



%%
% xlim([150, 220]);
% [AX,H1,H2] = plotyy(x,C1,x,P1,'plot');
% set(H1,'LineStyle','-','color','blue','LineWidth',2)
% set(H2,'LineStyle','--','color','red','LineWidth',2)
% % axes(AX(1));
% % hold on;
% % plot(x,C2,'LineStyle','-.','color','blue','LineWidth',2);
% % axes(AX(2));
% % hold on;
% % plot(x,P2,'LineStyle','-.','color','red','LineWidth',2);
% set(gca, 'Fontname', 'Times New Roman','FontSize',18);
% legend('$P_{210,183}^{k}$','$C_{210,183}^{k}$','Interpreter','LaTex','fontsize',18)
% 
% % legend('$C_{210,183}^{k}$','$P_{210,183}^{k}$','$C_{35,4}^{k}$','$P_{210,183}^{k}$','Interpreter','LaTex','fontsize',15)
% d1=get(AX(1),'ylabel');
% set(d1,'string','\# of trips from $i$ to $j$','fontsize',19, 'Interpreter','Latex');
% d2=get(AX(2),'ylabel');
% set(d2,'string','Prob. that contacts occur','fontsize',19, 'Interpreter','Latex');
% 
% d0=get(AX(1),'xlabel');
% set(d0,'string','$k$th day','fontsize',19, 'Interpreter','Latex');
% grid on;