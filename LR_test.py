from logistic_regression import LR
import numpy as np
lr = LR(2,4)

x_1 = np.random.uniform(0,1000,size=[100,2])
x_2 = np.random.uniform(0,1000,size=[100,2])
x_3 = np.random.uniform(0,1000,size=[100,2])
x_4 = np.random.uniform(0,1000,size=[100,2])


for i in range(100):
    x_2[i,1]*=-1
    x_4[i,0]*=-1

x_3*=-1

y_1 = np.zeros(shape=[100])
y_2 = np.ones(shape=[100])
y_3 = np.ones(shape=[100])*2
y_4 = np.ones(shape=[100])*3

x = np.concatenate([x_1,x_2,x_3,x_4],axis=0)
y = np.concatenate([y_1,y_2,y_3,y_4],axis=0).astype(np.int32)
print(x.shape)
y = np.eye(4)[y]
print(y.shape)
# print(y)

pair = np.concatenate([x,y],axis=1)
# for i in range(pair.shape[0]):
#     print(pair[i])
np.random.shuffle(pair)
x = pair[:,:2]
y = pair[:,2:]


# np.random.seed(20)
# np.random.shuffle(x)
# np.random.seed(20)
# np.random.shuffle(y)

for i in range(20):
    print(x[i],y[i])

lr.fit(x,y,2000,400,0.00000001,None,None)
# print(lr.backward(np.array([[50,50]]),np.array([[1,0,0,0]])))
# while True:
#     x = input()
#     y = input()
#     print(lr.forward(np.array([[int(x),int(y)]])))

