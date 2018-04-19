# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 19:29:28 2018

@author: Hp-duan
"""
#向量化编程就需要涉及到numpy这个软件包了，在这个软件包里的加减乘除都会有所不同的！
import numpy as np
from functools import *
import random

class SigmoidActivator:
    def forward(self,weight_input):#注意weight_input是维度为n的array类型
        return 1.0/(1.0+np.exp(-weight_input))#所以这里的除，加，负号，都是针对向量中的每一个分量进行的
    def backward(self,output):
        return output*(1-output)
    
class FullConnectedLayer:
    def __init__(self,input_size,output_size,activator):
        self.activator=activator
        self.input_size=input_size
        self.output_size=output_size
        self.W=np.random.uniform(-0.1,0.1,(input_size,output_size))#m*n
        self.b=np.zeros(output_size)#n
        self.output=np.zeros(output_size)#n
    def forward(self,input):#注意input是一个m的array类型（numpy）必须是array类型！
        self.input=input
        self.output=self.activator.forward(np.dot(input,self.W)+self.b)
        return self.output
    
    def backward(self,delta_array):#这里的delta_array是一个n的纵向向量！
        #matrix必须转变会array吗？
        
        self.W_grad=np.array(np.transpose(np.matrix(self.input))*np.matrix(delta_array))#input是1*m的，delta_array是n*1的，都需要转置
        self.b_grad=delta_array#
        self.delta=self.activator.backward(self.input)*np.dot(self.W,delta_array)#本层的delta应该是1*m的！
    def update(self,rate):
        self.W+=rate*self.W_grad
        self.b+=rate*self.b_grad
    def dump(self):
        print('W:%s\nb:%s'%(self.W,self.b))
        
class Network:
    def __init__(self,layers):
        self.layers=[]
        for i in range(len(layers)-1):
            self.layers.append(FullConnectedLayer(layers[i],layers[i+1],SigmoidActivator()))
    def predict(self,input):##Network本身没有output成员变量！
        output=input
        for layer in self.layers:
            output=layer.forward(output)
        return output
    def train(self,labels,dataset,rate,epoch):
        labels=np.array(labels)
        dataset=np.array(dataset)
        for i in range(epoch):
            for j in range(len(dataset)):
                self.train_one_sample(labels[j],dataset[j],rate)
    def train_one_sample(self,label,data,rate):
        self.predict(data)
        self.calc_gradient(label)
        self.update_weight(rate)
        
    def calc_gradient(self,label):#更新每层的gradient和delta
        delta=self.layers[-1].activator.backward(self.layers[-1].output)*(label-self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta=layer.delta
        return delta
    def update_weight(self,rate):
        for layer in self.layers:
            layer.update(rate)
    
    ##梯度检查
    def loss(self,output,label):
        return 0.5*((label-output)*(label-output)).sum()
    def gradient_check(self,label,feature):
        self.predict(feature)
        self.calc_gradient(label)
        
        epsilon=10e-4#0.001
        for fc in self.layers:
            for i in range(fc.W.shape[0]):
                for j in range(fc.W.shape[1]):
                    fc.W[i][j]+=epsilon
                    output=self.predict(feature)
                    err1=self.loss(label,output)
                    fc.W[i][j]-=2*epsilon
                    output=self.predict(feature)
                    err2=self.loss(label,output)
                    expected_grad=(err2-err1)/(2*epsilon)
                    fc.W[i][j]+=epsilon
                    print('expected:%f\nactual:%f'%(expected_grad,fc.W_grad[i][j]))
                fc.b[i]+=epsilon
                output=self.predict(feature)
                err1=self.loss(label,output)
                fc.b[i]-=2*epsilon
                output=self.predict(feature)
                err2=self.loss(label,output)
                expected_grad=(err2-err1)/(2*epsilon)
                fc.b[i]+=epsilon
                print('expected:%f\nactual:%f'%(expected_grad,fc.b_grad[i]))

def gradient_check():
    net=Network([2,2,2])
    feature=np.array([0.9,0.1])
    label=np.array([0.9,0.1])
    net.gradient_check(label,feature)
    
class Normalizer:
    def __init__(self):
        self.mask=[0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80]
    #把一个0-255之间的数化成了其二进制反过来的形式了
    def norm(self,num):
        return list(map(lambda m: 0.9 if num&m else 0.1,self.mask))
    def denorm(self,vec):
        binary=list(map(lambda v:1 if v>0.5 else 0,vec))
        sum=0
        for i in range(len(binary)):
            sum+=binary[i]*self.mask[i]
        return sum
#生成测试数据集，是32个在0-256之间的数字，输入=输出，
def train_data_set():
    normalizer=Normalizer()
    dataset=[]
    labels=[]
    ##可不可以不这样？
    for i in range(0,256,8):
        n=normalizer.norm(int(random.uniform(0,256)))
        dataset.append(n)
        labels.append(n)
    return labels,dataset

def train(network):
    labels,dataset=train_data_set()
    network.train(labels,dataset,0.3,50)
    
def test_one_example(network,data):
    normalizer=Normalizer()
    norm_data=normalizer.norm(data)
    predict_data=network.predict(norm_data)
    return normalizer.denorm(predict_data)

def correct_ratio(network):
    correct=0
    for i in range(256):
        if test_one_example(network,i)==i:
            correct+=1
    print('correct:%u'%(correct))

if __name__=='__main__':
    #gradient_check()
    net=Network([8,3,8])
    train(net)
    correct_ratio(net)