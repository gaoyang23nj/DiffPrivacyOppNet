保存实验代码&实验结果图片

1.RGMM day间是按照365天操作的 可能有误

2.graph结构

3.warmup流程设定 [need to do]*

这样对不上！！！
先计算看看
__update_probdensity()
干扰到了 其他的值


1.RGMM day间是按照365天操作的 可能有误

2.graph结构 -- graph 结构比较应该长期保存
每日更新--从邻居获取//从本地更新
各种概率每日更新一次即可; 第三方数据可以 即时更新

3.warmup流程设定 **[need to do]** --ok

4.实验加速,更少的节点(只用pukou stations) **[need to do]** --ok

5.ttl 应该全面使用

6*** path挖掘方案 还需要修改 执行速度太慢了

2021-06-07
1st success: daily update; limited hop; warm up
need to do:
1.加速方案: daily prob计算的结果进行预选择, 之后再使用我们的方法2层prob计算方法

2.高级方案：ttl 随机设置[2day ~ 13day], 突出优势

3.隐私方案：扰动 每日的概率模式  how to do?

4.write


2021-06-11
可能存在的问题
1. Lap噪声可能<0, 从而使得到的p_{24}<0; 想个办法解决
read 2013CCS
2. RTPM方案SpeedUp版本是否有效? RTPMSpdUp 和 RTPM
进行实验
3. 从原理上考虑加速方案
存不存在 权重加和的关系 从而更新起来
4. notify linkup时候 形成一个vertex+edge的graph结构


2021-06-23
可能存在的问题
1.2020NDSS Lap以后一致性问题, sum=1, >=0
2.RTPMSpdUp有效！

*********************************************************
Djk 修复了ttl忽略的问题, 使用了Djk方法进加速, prob seq计算：只计算自己节点对其他节点i的, i对j的都不进行本地更新(第三方i对第三方j只做从其他节点的复制) 
Djk_re2 prob seq计算: 任何i对任何j(即代码上的target_nodeid) 每个节点发起的prob seq (14天) 计算, 利用矩阵方法进更新
Djk_re1 prob seq计算：只是进行重复更新(不采用矩阵)
Djk_Lap 在Djk的基础上进行Lap


2021-07-12
Djk 在216节点情况下 0.456
Djk_Lap 在216节点情况下 0.5


2021-07-13
结论：Djk_re2 在116节点情况下 接近 0.346

2021-07-14
结论：Djk_re1 Djk_re2 在216节点的情况下 都接近 0.104 0.116 差别不大
216 Djk 0.11

2021-07-15
结论：Djk Djk_re2 116节点情况下 0.4 0.41 【没有衰减 正常偏差】
结论：Djk_re2_Lap 116节点情况下 0.39



*********************************************************
probs 的传播方式
7月25日 （1）整改代码 结果显示 应该包含实验时间，变更*.tmp （2）各项实验
计划
100pkt_limit: 3600(1hour)(1pkt/hour) 2400(40min) 1800(30min) 1200(20min)
即12, 24, 36, 48, 72 pkts/day
50pkt_limit: 3600(1hour)(1pkt/hour) 2400(40min) 1800(30min) 1200(20min)

计划 不同的隐私强度 ？？？？

计划 不同的仿真时长 7月1日-7月14日(约完成1小时) 7月1日-7月31日（约4小时完成） 结果差别不大


7月30日
修复了ttl上的bug
仍旧100 limit

计划 ddl=5day 和 无限ddl(原来的实验) 两种情况 性能都可以
ddl=5days 完成

*********************************************************
7月31日
奇怪？？？
ddl=5days gen_freq=1hours dp强度 0.01 0.1 0.5 保持不变 或者说 不降反增
ddl=5days gen_freq=1hours dp强度 0.01 0.1 1 保持不变 或者说 不降反增
ddl=1days gen_freq=2hours dp强度 0.01 0.1 1 保持不变 或者说 不降反增

ddl=1days gen_freq=2hours dp强调 0. 1 10 会怎么样？
难道P才是绝对影响力【扰动P?】














