# encoding: utf-8

import struct
#解析mnist文件，得到训练、测试数据集，并通过散点图把图片可视化出来！【python3.6】

class Loader(object):
	def __init__(self,path,count):
		self.path=path
		self.count=count
		
	def get_file_content(self):
		f=open(self.path,'rb')
		content=f.read()
		f.close()
		return content
	##把8bit的unsigned char byte 转换成int类型的数字数组，然后[0]取出数组中的第一个（也是唯一一个）数字；
	#在python3中不再需要这个函数了！
	#def to_int(self,byte):
	#	return struct.unpack('B',byte)[0]
##	
class ImageLoader(Loader):
	##得到一个28*28的二维数组，是一张图片
	##一个字节代表一个像素，一个字节有八位；而在conteng中是以字节为单位的（8位）
	def get_picture(self,content,index):
		##前16个字节是无用信息
		start=index*28*28+16
		picture=[]
		for i in range(28):
			picture.append([])
			for j in range (28):
				picture[i].append(content[start+i*28+j])
		return picture
	##把picture变成一维向量
	def get_one_sample(self,picture):
		sample=[]
		for i in range(28):
			for j in range(28):
				sample.append(picture[i][j])
		return sample
		
	def load(self):
		content=self.get_file_content()
		data_set=[]
		for index in range(self.count):
			data_set.append(self.get_one_sample(self.get_picture(content,index)))
		return data_set
			
class LabelLoader(Loader):
	##得到一个labels数组，元素是通过norm转变成vec类型的
	def load(self):
		content=self.get_file_content()
		labels=[]
		for index in range(self.count):
			labels.append(self.norm(content[index+8]))
		return labels
	##把labels数组中的一个元素转变成自己想要的形式,比如1=[0.9,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
	def norm(self,label):
		label_vec=[]
		#label_value=self.to_int(label)
		for i in range(10):
			if i ==label:
				label_vec.append(0.9)
			else:
				label_vec.append(0.1)
		return label_vec
#这里得到的dataset已经化为784个分量的一维函数了！label已经化为10个分量的一维函数了！		
def get_training_data_set():
	image_loader=ImageLoader('train-images.idx3-ubyte',6000)
	label_loader=LabelLoader('train-labels.idx1-ubyte',6000)
	return image_loader.load(),label_loader.load()
	
def get_test_data_set():
	image_loader=ImageLoader('t10k-images.idx3-ubyte',1000)
	label_loader=LabelLoader('t10k-labels.idx1-ubyte',1000)
	return image_loader.load(),label_loader.load()

##把vec[0.9,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]转变成1	
def get_result(vec):
	max_value_index=0
	max_value=0
	for i in range(len(vec)):
		if vec[i]>max_value:
			max_value=vec[i]
			max_value_index=i 
	return max_value_index
	


	
##把图片通过散点图的形式画出来，data是784个分量的一维函数！
def plot_picture(data):
	import matplotlib.pyplot as plt
	import numpy as np
	##from matplotlib import mpl
	#data=np.clip(np.random.randn(5,5),-1,1)
	fig=plt.figure()
	ax=fig.add_subplot(111)
	ax.imshow(np.reshape(data,(28,-1)),cmap='Greys',interpolation='nearest')
	plt.show()
	##return np.array(data) 