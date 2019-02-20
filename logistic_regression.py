import numpy as np
# import sklearn as sk
import math


class LR(object):
    def __init__(self,i,j):
        self.w = np.zeros([i,j])
        self.b = np.zeros([1,j])

    def forward(self,x):
        z = np.matmul(x,self.w)+self.b
        predict = 1/(np.exp(-z)+1)
        # predict = np.exp(z)
        # predict = predict / np.sum(predict, axis=-1, keepdims=True)
        return predict

    def backward(self,x,y):
        derivatives = dict()
        # print(y.shape)
        p = self.forward(x)
        # d_p = -y / (p+0.000000001) + (1-y)/(1-p+0.0000000001)
        # print(d_p.shape)
        # d_z = -p * np.tile(np.sum(p * d_p, axis=-1, keepdims=True), [1, p.shape[-1]]) + d_p * p
        # d_z = d_p*p*(1-p)
        # d_z = np.multiply(d_p,np.multiply(p,1-p))
        d_z = p-y
        # print(d_z.shape)
        d_w = np.matmul(np.transpose(x), d_z)
        # print(d_w.shape)
        d_b = np.sum(d_z, axis=0)
        # print(d_b.shape)
        derivatives['w'] = d_w
        derivatives['b'] = d_b
        return derivatives

    def fit(self, train_x, train_y, epoch, minibatch, lr,test_x,test_y):
        acc = 0
        for i in range(epoch):
            # print(i, 'th epoch training loss:', self.cross_entropy_loss(train_x, train_y))
            low = 0
            high = minibatch
            acc = self.acc(test_x, test_y)
            # print('best test acc during training: ', acc)

            for j in range(train_x.shape[0] // minibatch):
                now_x = train_x[low:high]
                now_y = train_y[low:high]
                derivatives = self.backward(now_x, now_y)
                self.w -= lr * derivatives['w']
                self.b -= lr * derivatives['b']
                low += minibatch
                high += minibatch
                if high > train_x.shape[0]:
                    high = train_x.shape[0]

            # print(i, 'th epoch training loss:', self.cross_entropy_loss(train_x,train_y))
        acc = self.acc(test_x,test_y)
        print('best test acc during training: ',acc)

    def l2_loss(self, x, y):
        return np.sum(1 / 2 * np.square((self.forward(x) - y)))

    def cross_entropy_loss(self,x,y):
        return -np.sum(np.log(self.forward(x)+0.0000000001)*y+np.log(1-self.forward(x)+0.0000000001)*(1-y))/x.shape[0]

    def acc(self,x,y):
        p = self.forward(x)
        acc = np.sum(np.equal((p[:,-1]>0.5).astype(np.int32),y[:,-1]).astype(np.float64))/x.shape[0]
        return acc

class DiscreteLR(object):
    def __init__(self,weight):
        self.weight = weight
        self.tag_list = ['n','ns','nr','nz']

    def forward(self,features,isTrain):
        predict = []
        for feature in features:
            p = []
            for t in self.tag_list:
                now_score = self.weight.getFeatureScore(t+'b',isTrain)
                for f in feature[:8]:
                    now_score+=self.weight.getFeatureScore(t+f,isTrain)


                p.append(now_score)
                now_score = 0


            p_sigmoid = list(map(lambda x:1/(np.exp(-x)+1),p))
            # p_exp = list(map(np.exp,p))
            # p_exp_sum = sum(p_exp)
            # p_exp = list(map(lambda x:x/p_exp_sum,p_exp))
            predict.append(p_sigmoid)

            # predict.append(p_exp)

        return np.array(predict)

    def backward(self,features,y):
        predict = self.forward(features,True)
        predict_np = np.array(predict)
        y = np.array(y)
        d_p = -y / (predict+0.00000001)
        # print(d_p.shape)
        d_z = np.zeros(shape=predict_np.shape)
        for i in range(len(features)):
            for j in range(predict_np.shape[1]):
                d_z[i, j] += (-predict_np[i,j]*np.sum(d_p[i] * predict_np[i] ) + d_p[i, j] * predict_np[i, j])
        # d_z = -predict_np * np.tile(np.sum(predict_np * d_p, axis=-1, keepdims=True), [1, predict_np.shape[-1]]) + d_p * predict_np
        return d_z

    def fit(self,features,y,epoch,minibatch,lr,test_feature,test_y):
        for e in range(epoch):
            low = 0
            high = minibatch
            for j in range(len(features) // minibatch):
                d_z = self.backward(features[low:high],y[low:high])
                for i in range(d_z.shape[0]):
                    for j in range(d_z.shape[1]):
                        self.weight.updateFeatureScore(self.tag_list[j]+features[i][j],-lr*d_z[i,j],e)

                    self.weight.updateFeatureScore(self.tag_list[j]+'b',-lr*sum(d_z[i]),e)

                low += minibatch
                high += minibatch
                if high > len(features):
                    high = len(features)

            if e%1 == 0:
                acc = self.acc(test_feature, test_y)
                print(e,'th acc: ',acc)

            # print('best test acc during training: ', acc)


    def acc(self,features,y):
        predict = self.forward(features,False)
        # correct_num = 0
        y = np.array(y)
        acc = np.sum((np.argmax(y, -1) == np.argmax(predict, -1)).astype(np.float64)) / len(features)
        return acc




