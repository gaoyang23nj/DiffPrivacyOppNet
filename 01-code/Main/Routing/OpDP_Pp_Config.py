import numpy as np
from cvxopt import matrix


EPS_TRAN_PDF = '../EncoHistData_NJBike/tran_eps_pdf.csv'

class OpDP_Pp_Config(object):
    def __init__(self, lap_noise_scale):
        self.LapNoiseScale_pintra = lap_noise_scale[0]
        self.LapNoiseScale_Pinter = lap_noise_scale[1]
        # 1.P
        self.delta_P = 0.05
        self.length_P = np.math.floor(1./self.delta_P)
        self.set_st_P()

        # 2.pdf
        # 把pdf分成几段
        self.num_pdf_seg = 6
        oneset = np.zeros(self.num_pdf_seg)
        # 每段的数值可能性 0 0.2 0.4 0.6 0.8 1.//0.1,0.3,0.5,0.7,0.9
        self.choose = np.arange(0.1, 1.1, 0.2)
        # 所有可能的pdf组合
        self.list_pdf = []
        self.getone(oneset, 0)
        self.num_pdf_comb = len(self.list_pdf)
        self.set_st_for_pdf()

        # # 2.pdf 另一种方案
        # # 读取 eps_p_intra pdf 的 转移矩阵
        # npzfile = np.load(EPS_TRAN_PDF)
        # self.tran = npzfile['tran']

    def getone(self, a, i):
        if i == a.size:
            if sum(a) == 1.:
                self.list_pdf.append(a.copy())
            return
        for j in range(self.choose.size):
            a[i:] = 0
            a[i] = self.choose[j]
            if sum(a) > 1.:
                continue
            self.getone(a, i + 1)
        return

    # 为pdf的opt+DP 做出s.t.
    def set_st_for_pdf(self):
        epsp = 1. / self.LapNoiseScale_pintra
        w = np.exp(epsp)
        print('set s.t. ...')
        # 由a c 导出 tranp转成行矩阵对应的位置 c*length+a
        num_st1 = 0
        list_st1 = []
        # s.t.1 隐私保护带来的限制条件
        for c in range(self.num_pdf_comb):
            # 选取a,b
            for a in range(self.num_pdf_comb):
                for b in range(a + 1, self.num_pdf_comb):
                    num_st1 = num_st1 + 1
                    list_st1.append((c, a, b))
        num_st1 = num_st1 * 2
        # print(num_st1)
        num_st2 = 0
        list_st2 = []
        # s.t.2 prob的天然条件 [0,1]
        for a in range(self.num_pdf_comb):
            # 选取a,b
            for c in range(self.num_pdf_comb):
                num_st2 = num_st2 + 1
                list_st2.append((c, a))
        num_st2 = num_st2 * 2
        # print(num_st2)
        num_neq_st = num_st1 + num_st2
        list_st3 = []
        num_st3 = 0
        for a in range(self.num_pdf_comb):
            # 选取a,b
            num_st3 = num_st3 + 1
            list_st3.append(a)
        # print(num_st3)
        # 不等式条件，矩阵
        label = 0
        Gnp = np.zeros((num_neq_st, self.num_pdf_comb * self.num_pdf_comb))
        hnp = np.zeros((num_neq_st, 1))
        for ele in list_st1:
            (c, a, b) = ele
            # p_{a,c} - W p_{b,c} <=0
            Gnp[label, a * self.num_pdf_comb + c] = 1.
            Gnp[label, b * self.num_pdf_comb + c] = -w
            label = label + 1
            # -W p_{a,c} + p_{a,c} <=0
            Gnp[label, a * self.num_pdf_comb + c] = -w
            Gnp[label, b * self.num_pdf_comb + c] = 1.
            label = label + 1
        for ele in list_st2:
            (c, a) = ele
            # -p_{a,c} <= 0
            Gnp[label, a * self.num_pdf_comb + c] = -1.
            label = label + 1
            # p_{a, c} <= 1
            Gnp[label, a * self.num_pdf_comb + c] = 1.
            hnp[label, 0] = 1.
            label = label + 1
        # print(label)
        # 等式条件，矩阵
        Anp = np.zeros((num_st3, self.num_pdf_comb * self.num_pdf_comb))
        bnp = np.ones((num_st3, 1))
        label = 0
        for ele in list_st3:
            (a) = ele
            Anp[label, a * self.num_pdf_comb:(a + 1) * self.num_pdf_comb] = 1.
            label = label + 1
        # print(label)
        self.Gcx_pdf = matrix(Gnp)
        self.hcx_pdf = matrix(hnp)
        self.Acx_pdf = matrix(Anp)
        self.bcx_pdf = matrix(bnp)
        return

    # 做成全局的变量
    def set_st_P(self):
        epsP = 1. / self.LapNoiseScale_Pinter
        W = np.exp(epsP)
        print('set s.t. ...')
        # 由a c 导出 tranP转成行矩阵对应的位置 c*11+a
        num_st1 = 0
        list_st1 = []
        # s.t.1 隐私保护带来的限制条件
        for c in range(self.length_P):
            # 选取a,b
            for a in range(self.length_P):
                for b in range(a + 1, self.length_P):
                    num_st1 = num_st1 + 1
                    list_st1.append((c, a, b))
        num_st1 = num_st1 * 2
        # print(num_st1)
        num_st2 = 0
        list_st2 = []
        # s.t.2 prob的天然条件 [0,1]
        for a in range(self.length_P):
            # 选取a,b
            for c in range(self.length_P):
                num_st2 = num_st2 + 1
                list_st2.append((c, a))
        num_st2 = num_st2 * 2
        # print(num_st2)
        num_neq_st = num_st1 + num_st2
        list_st3 = []
        num_st3 = 0
        for a in range(self.length_P):
            # 选取a,b
            num_st3 = num_st3 + 1
            list_st3.append(a)
        # print(num_st3)
        # 不等式条件，矩阵
        label = 0
        Gnp = np.zeros((num_neq_st, self.length_P * self.length_P))
        hnp = np.zeros((num_neq_st, 1))
        for ele in list_st1:
            (c, a, b) = ele
            # p_{a,c} - W p_{b,c} <=0
            Gnp[label, a * self.length_P + c] = 1.
            Gnp[label, b * self.length_P + c] = -W
            label = label + 1
            # -W p_{a,c} + p_{a,c} <=0
            Gnp[label, a * self.length_P + c] = -W
            Gnp[label, b * self.length_P + c] = 1.
            label = label + 1
        for ele in list_st2:
            (c, a) = ele
            # -p_{a,c} <= 0
            Gnp[label, a * self.length_P + c] = -1.
            label = label + 1
            # p_{a, c} <= 1
            Gnp[label, a * self.length_P + c] = 1.
            hnp[label, 0] = 1.
            label = label + 1
        # print(label)
        # 等式条件，矩阵
        Anp = np.zeros((num_st3, self.length_P * self.length_P))
        bnp = np.ones((num_st3, 1))
        label = 0
        for ele in list_st3:
            (a) = ele
            Anp[label, a * self.length_P:(a + 1) * self.length_P] = 1.
            label = label + 1
        # print(label)
        self.Gcx = matrix(Gnp)
        self.hcx = matrix(hnp)
        self.Acx = matrix(Anp)
        self.bcx = matrix(bnp)
        return

