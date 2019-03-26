# from sklearn.preprocessing import StandardScaler
# import numpy as np

# a = np.array([[1,2,3],[4,5,6]])
# ss = StandardScaler()
# print(a)
# a = ss.fit_transform(a)
# print(a)
# class Data:
#     def __init__(self,v=2):
#         self.v = v
#
# d1 = {1:Data(2),2:Data(3),3:Data(4)}
# d2 = d1.copy()
#
# # for x in d2:
# #     d2[x].v = 0
#
# for x in d1:
#     print(d1[x].v)
f = open('tmp_lxn.txt','r',encoding='utf-8')
f_o = open('tmp_lxn_1.txt','w',encoding='utf-8')
for i in range(1000000):
    print(f.readline().strip(),file=f_o)