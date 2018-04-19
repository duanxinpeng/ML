
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 21:10:17 2018

@author: Hp-duan
"""
#面向对象版的全连接神经网络实现
#并生成了一些数据来进行训练并测试，以及梯度测试
import math
import random
from functools import *

def sigmoid(x):
    return 1.0/(1+math.exp(-x))##折腾了这么长时间，原来是激活函数错了，少了一个负号！？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
## node分为输入层node、隐藏层node、输出层node、常量node
class Node:
    def __init__(self,layer_index,node_index):
        self.layer_index=layer_index
        self.node_index=node_index
        self.upConns=[]
        self.downConns=[]
        self.output=0
        self.delta=0
    def append_upConn(self,conn):
        self.upConns.append(conn)
    def append_downConn(self,conn):
        self.downConns.append(conn)
    def set_output(self,output):
        self.output=output
    def calc_output(self):
        #form functools import *
        #sum=reduce(lambda ret,conn:ret+conn.weight*conn.upNode.output,self.upConns,0)
        sum=0
        for conn in self.upConns:
            sum+=conn.upNode.output*conn.weight
        self.output=sigmoid(sum)
    def calc_output_layer_delta(self,label):
        #self.delta=-((label-self.output)*self.output*(1-self.output))
        self.delta=self.output*(1-self.output)*(label-self.output)
    def calc_hidden_layer_delta(self):
        #sum=reduce(lambda ret,conn:ret+conn.downNode.delta*conn.weight,self.downConns,0.0)
        #self.delta=-sum*self.output*(1-self.output)
        down_delta=reduce(lambda ret,conn:ret+conn.downNode.delta*conn.weight,self.downConns,0.0)
        self.delta=self.output*(1-self.output)*down_delta
    def __str__(self):
        node_str='%u-%u:output:%f,delta:%f'%(self.layer_index,self.node_index,self.output,self.delta)
        return node_str
    
class ConstNode:
    def __init__(self,layer_index,node_index):
        self.layer_index=layer_index
        self.node_index=node_index
        self.output=1
        self.delta=0
        self.downConns=[]
    def append_downConn(self,conn):
        self.downConns.append(conn)
    ##不需要这个计算output的函数吗？
    def calc_hidden_layer_delta(self):
        #num=reduce(lambda ret,conn:ret+conn.downNode.delta*conn.weight,self.downConns,0.0)
        #self.delta=-num*self.output*(1-self.output)
        down_delta=reduce(lambda ret,conn:ret+conn.downNode.delta*conn.weight,self.downConns,0.0)
        self.delta=self.output*(1-self.output)*down_delta
    def __str__(self):
        return '%u-%u:output:%f,delta:%f'%(self.layer_index,self.node_index,self.output,self.delta)
    
##layer分为输入层、隐藏层、输出层？好像没考虑这些情况呀
## layer的功能只有设置output和计算output？
##输出层也加上偏置项呗，不用就行了
class Layer:
    def __init__(self,layer_index,nodesCount):
        self.layer_index=layer_index
        self.nodes=[]
        for i in range(nodesCount):
            self.nodes.append(Node(layer_index,i))
        self.nodes.append(ConstNode(layer_index,nodesCount))
    def set_outputs(self,inputs):
        for i in range(len(inputs)):
            self.nodes[i].set_output(inputs[i])
    #ConstNode不需要计算output
    def calc_outputs(self):
        for node in self.nodes[:-1]:
            node.calc_output()
    def dump(self):
        for node in self.nodes:
            print(node)

#这里计算梯度和权重的时候遇到了些问题，
#该怎么解决？
#delta什么时候计算？这里是不用考虑的！竟然不用考虑！
class Connection:
    def __init__(self,upNode,downNode):
        self.gradient=0
        self.weight=random.uniform(-0.1,0.1)
        self.upNode=upNode
        self.downNode=downNode
    #不需要考虑delta的问题，而且这只是一个内函数
    def calc_gradient(self):
        self.gradient=self.upNode.output*self.downNode.delta
    def calc_weight(self,rate):
        self.calc_gradient()
        self.weight+=rate*self.gradient
    def get_gradient(self):
        return self.gradient
    def __str__(self):
        return 'upNode:%u,downNode:%u,gradient:%f,weight:%f'%(self.upNode.node_index,self.downNode.node_index,self.gradient,self.weight)
 #我觉得这个类存在的唯一必要就是打印出所有conn吧？   
class Connections:
    def __init__(self):
        self.connections=[]
    def add_conn(self,conn):
        self.connections.append(conn)
    def dump(self):
        for conn in self.connections:
            print(conn)
#1、初始化：按照一个数组舒适化各层，并且在各层之间建立conn，注意偏置项没有upConn
#2、训练：确定labels、data_set、rate、epoch等元素，
#3、计算输出、计算delta、更新权重等内容都需要在这里实现
            #计算delta：分成输出层和隐藏层，，需要注意输出层的最后一个节点是废的！！没法计算delta
            #以conn为单位向node添加上连接和下连接，就不需要考虑偏置项不能添加上连接的问题了
            #直接用self.conns计算权重可行？
class Network:
    def __init__(self,layers):
        self.conns=Connections()
        self.layers=[]
        for i in range(len(layers)):
            self.layers.append(Layer(i,layers[i]))
        for i in range(len(layers)-1):
            #这里有错，layer不是一个数组！是一个类！
            conns=[Connection(upNode,downNode) for upNode in self.layers[i].nodes for downNode in self.layers[i+1].nodes[:-1]]
            for conn in conns:
                self.conns.add_conn(conn)
                conn.downNode.append_upConn(conn)
                conn.upNode.append_downConn(conn)
    def train(self,labels,dataset,rate,epoch):
        for i in range(epoch):
            for j in range(len(dataset)):
                self.train_one_sample(labels[j],dataset[j],rate)
                print('%d Epoch and %d data:'%(i,j))
                self.dump()
    def train_one_sample(self,labels,data,rate):
        self.predict(data)
        self.calc_deltas(labels)
        self.calc_weights(rate)
    #输入层不需要计算，输出层和隐藏层要分开计算，而且要从后向前
    def calc_deltas(self,labels):
        output_nodes=self.layers[-1].nodes
        for i in range(len(labels)):
            #这里的labels是一个数组！，但是计算delta的时候可不是计算的数组！
            output_nodes[i].calc_output_layer_delta(labels[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()
    #是不是只要把self.connes中的所有连接的权重都更新一遍就行了？我觉得没问题，这也是self.conns的唯一作用吧
    def calc_weights(self,rate):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downConns:
                    conn.calc_weight(rate)
        #for conn in self.conns.connections:
         #   conn.calc_weight(rate)
    def calc_gradient(self):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downConns:
                    conn.calc_gradient()
                    
    def get_gradient(self,label,sample):
        self.predict(sample)
        self.calc_deltas(label)
        self.calc_gradient()
                    
    def predict(self,inputs):
        #self.layers[0].set_outputs(inputs)
        #for layer in self.layers[1:]:
        #    layer.calc_outputs()
        #不要最后一层的最后一个，那是一个废节点，因为输出层不要偏置项
        #return [node.output for node in self.layers[-1].nodes[:-1]]
        self.layers[0].set_outputs(inputs)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_outputs()
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])

    def dump(self):
        for layer in self.layers:
            layer.dump()
        for conn in self.conns.connections:
            print(conn) 
#完成了基本的模型，没有写梯度测试，以后再写  还有一个问题就是关于权重初始化的问题             
#######################################################################
#测试
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
   
##################################
#检查误差
    #检查出了错误，梯度符号错了，为什么呢?推导哪一步出现了问题？
def calc_error(vec1,vec2):   
    vec=list(map(lambda v:(v[0]-v[1])*(v[0]-v[1]),list(zip(vec1,vec2))))
    return 0.5*reduce(lambda a,b:a+b,vec)
def gradient_check(network,sample_label,sample_feature):
    network.get_gradient(sample_label,sample_feature)
   
    for conn in network.conns.connections:
        actual_gradient=conn.gradient
        epsilon=0.0001
        conn.weight+=epsilon
        error1=calc_error(network.predict(sample_feature),sample_label)
        conn.weight-=2*epsilon
        error2=calc_error(network.predict(sample_feature),sample_label)
        expected_gradient=(error2-error1)/(2*epsilon)
        print ('expected gradient:\t%f\nactual gradient:\t%f'%(expected_gradient,actual_gradient))
def gradient_check_test():
    net=Network([2,2,2])
    sample_feature=[0.9,0.1]
    sample_label=[0.9,0.1]
    gradient_check(net,sample_label,sample_feature)

if __name__=='__main__':
    #gradient_check_test() #测试gradient是否正确！
    net = Network([8, 3, 8])
    train(net)
    #net.dump()
    correct_ratio(net)