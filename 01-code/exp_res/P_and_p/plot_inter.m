% Ìì¼ä ºê¹Û

clear;
clc;
res = csvread('inter_day.csv');
figure(1);
hold on;
x = res(1,:);
C1 = res(2,:);
C2 = res(3,:);
P1 = res(4,:);
P2 = res(5,:); 
set(gca, 'Fontname', 'Times New Roman','FontSize',18);
xlim([150, 220]);
[AX,H1,H2] = plotyy(x,C1,x,P1,'plot');
set(H1,'LineStyle','-','color','blue','LineWidth',2)
set(H2,'LineStyle','-','color','red','LineWidth',2)
axes(AX(1));
hold on;
plot(x,C2,'LineStyle','-.','color','blue','LineWidth',2);
axes(AX(2));
hold on;
plot(x,P2,'LineStyle','-.','color','red','LineWidth',2);

set(gca, 'Fontname', 'Times New Roman','FontSize',18);

legend('$P_{210,183}^{k}$','$P_{35,4}^{k}$','$C_{210,183}^{k}$','$C_{35,4}^{k}$','Interpreter','LaTex','fontsize',18)
% legend('$C_{210,183}^{k}$','$P_{210,183}^{k}$','$C_{35,4}^{k}$','$P_{210,183}^{k}$','Interpreter','LaTex','fontsize',15)
d1=get(AX(1),'ylabel');
set(d1,'string','No. of trips from $i$ to $j$ ($C^{k}_{i,j}$)','fontsize',19, 'Interpreter','Latex');
d2=get(AX(2),'ylabel');
set(d2,'string','Prob. that contacts occur ($P^{k}_{i,j}$)','fontsize',19, 'Interpreter','Latex');

d0=get(AX(1),'xlabel');
set(d0,'string','$k$th day in the year','fontsize',19, 'Interpreter','Latex');
grid on;