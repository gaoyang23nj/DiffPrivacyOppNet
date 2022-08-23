import seaborn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Inputfiles = ["res_dplapoppnet_20220629192202_new.csv",
    "res_dplapoppnet_20220708144606_new.csv",
    "res_dplapoppnet_20220709080554_new.csv",
    "res_dplapoppnet_20220809174653_new.csv",
    "res_dplapoppnet_20220810094705_new.csv",
    "res_dplapoppnet_20220811093532_new.csv",
    "res_dplapoppnet_20220811095936_new.csv",
    "res_dplapoppnet_20220813113211_new.csv"]
res = np.zeros((12,9))
tmp_res = pd.read_csv(Inputfiles[0], header=None)
tmp_res = np.array(tmp_res)
res = tmp_res

# for file in Inputfiles:
#     tmp_res = pd.read_csv(file, header=None)
#     tmp_res = np.array(tmp_res)
#     res = res+tmp_res
#
# res = res/len(Inputfiles)
# err_res_neg = np.zeros((12,9))
# err_res_pos = np.zeros((12,9))
# for file in Inputfiles:
#     tmp_res = pd.read_csv(file, header=None)
#     tmp_res = np.array(tmp_res)
#     # neg err
#     tmp_err_res = res - tmp_res
#     for j in range(12):
#         for k in range(9):
#             if tmp_err_res[j,k] > err_res_neg[j,k]:
#                 err_res_neg[j,k] = tmp_err_res[j,k]
#
#     # % tmp_res - res; pos err
#     tmp_err_res = tmp_res - res
#     for j in range(12):
#         for k in range(9):
#             if tmp_err_res[j,k] > err_res_pos[j,k]:
#                 err_res_pos[j,k] = tmp_err_res[j,k]

color_palette = seaborn.color_palette()

plt.figure(1)
# %投递率
dr = res[:,0]
# err_dr_neg = err_res_neg[:,0]
# err_dr_pos = err_res_pos[:,0]
# %reshape 是按照列读取 按照列摆放
dr_group = np.reshape(dr,(4,3))
# err_dr_group_neg = np.transpose(np.reshape(err_dr_neg,(3,4)))
# err_dr_group_pos = np.transpose(np.reshape(err_dr_pos,(3,4)))
x = np.array([[1,1,1], [2,2,2], [3,3,3], [4,4,4]])
# h = plt.bar(x,height=dr_group)

# plt.bar(np.arange(12), dr)
# plt.show()
plt.figure(1)
ax=plt.axes()
bar_width = 0.2
ax.bar(np.arange(4)-bar_width, dr_group[:,0],label='TTPM+Lap', width=bar_width,color=color_palette[0])
ax.bar(np.arange(4), dr_group[:,1],label='TTPM+DODP',width=bar_width,color=color_palette[1])
ax.bar(np.arange(4)+bar_width, dr_group[:,2],label='TTPM+NoNoise',width=bar_width,color=color_palette[2])

plt.show()

# set(h(1),'FaceColor',color1);
# set(h(2),'FaceColor',color2);
# set(h(3),'FaceColor',color3);
#
# hold on;
# errorbar([1,2,3,4], dr_group(:,2), err_dr_group_neg(:,2), err_dr_group_pos(:,2), 'k', 'Linestyle', 'None');
# errorbar([1.225,2.225,3.225,4.225], dr_group(:,3), err_dr_group_neg(:,3), err_dr_group_pos(:,3), 'k', 'Linestyle', 'None');
# errorbar([0.775,1.775,2.775,3.775], dr_group(:,1), err_dr_group_neg(:,1), err_dr_group_pos(:,1), 'k', 'Linestyle', 'None');
# set(gca, 'Fontname', 'Times New Roman','FontSize',16);
# set(gca, 'XTick',1:4, 'xticklabels',{'1d','2d','3d','4d','Interpreter','LaTex'});
# xlabel("Message Lifespan $T_m$" ,'interpreter','latex');
# ylabel('Delivery Ratio');
# legend('TTPM+Lap','TTPM+DODP', 'TTPM+NoPvc','Location','SouthEast');