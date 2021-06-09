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

3.warmup流程设定 [need to do]*
3.warmup流程设定 **[need to do]**

4.实验加速,更少的节点(只用pukou stations) **[need to do]**

5.ttl 应该全面使用

6*** path挖掘方案 还需要修改 执行速度太慢了

2021-06-07
1st success: daily update; limited hop; warm up
need to do:
1.加速方案: daily prob计算的结果进行预选择, 之后再使用我们的方法2层prob计算方法

2.高级方案：ttl 随机设置[2day ~ 13day], 突出优势

3.隐私方案：扰动 每日的概率模式  how to do?

4.write



