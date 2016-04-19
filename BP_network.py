# coding=utf-8
import rf
from scipy import *

#sigmoid函数
def sigmoid(inX):  
	return 1.0 / (1 + exp(-inX))

#计算每一层的输出情况
def cal_each_layer_out(d_input, each_layer_w, layer_num):
	e_input = d_input #由于前一层的输出就等于后一层的输入,这里的e_input代表两个含义
	e_output = [] #输出结果数组
	e_output.append(e_input)
	for i in range(layer_num-1):
		e_input = sigmoid(dot(e_input,each_layer_w[i])) #将前一层的输出矩阵与层之间的权重矩阵进行相乘,并通过sigmoid得到结果
		e_output.append(e_input) #将结果保存到返回数组中
	return e_output

#对一个数据进行一次训练,得到新的W
def one_train_for_BP(d_input,d_label,each_layer_dim,each_layer_w,layer_num,neta):
	each_layer_output = cal_each_layer_out(d_input,each_layer_w,layer_num) #计算当前W下的每一层的输出
	#计算输出层的theta值
	theta = each_layer_output[layer_num-1]*(ones((1,each_layer_dim[layer_num-1]))-each_layer_output[layer_num-1])\
			*(d_label-each_layer_output[layer_num-1])
	#进行反向传播
	for i in range(layer_num-1):
		#保存未更改之前的W
		old_w = each_layer_w[layer_num-2-i]
		#更新W
		each_layer_w[layer_num-2-i] = each_layer_w[layer_num-2-i] + dot(each_layer_output[layer_num-2-i].T,theta)*neta
		#计算前一层新的theta
		theta = each_layer_output[layer_num-2-i]*(ones((1,each_layer_dim[layer_num-2-i]))-each_layer_output[layer_num-2-i])\
				*dot(theta,old_w.T)
	#返回结果
	return each_layer_w

#测试训练结果的正确率
def test_result(d_w, d_input, d_label, layer_num, data_num):
	right_num = 0 #初始正确个数为0
	#对于每个输入数据
	for i in range(data_num):
		e_input = array(d_input[i])
		#计算每一层的输出
		out = cal_each_layer_out(e_input,d_w,layer_num)
		#得到最后一层输出的最大值的下标
		[max_index] = out[layer_num-1].argmax(1)
		#如果该下标与实际标签相同,则说明判断正确
		if max_index == d_label[i]:
			right_num+=1
	return right_num


#read the heads of image file and label file and confirm
# image_file = open('train-images.idx3-ubyte','rb')
# label_file = open('train-labels.idx1-ubyte','rb')
# if not rf.confirm_image_file(image_file, 2051):
# 	print "	false image file"
# if not rf.confirm_label_file(label_file, 2049):
# 	print "	flase label file"

# print "reading images..."
# image_data = rf.get_images(image_file)
# label_data = rf.get_labels(label_file)
# print "binaryzation and feature extraction.."
# input_data = []
# for x in range(rf.image_num):
# 	o = zeros((1,rf.rows*rf.cols/4))
# 	for i in range(rf.rows/2):
# 		for j in range(rf.cols/2):
# 			for m in range(2):
# 				for n in range(2):
# 					if image_data[x][(2*i+m)*rf.cols+2*j+n] > 48:
# 						o[0][i*rf.cols/2+j]+=1
# 	input_data.append(o)

#step 1. training

#initial settings:
# training_times = 200
# each_layer_dim = [rf.cols*rf.rows/4, rf.cols+rf.rows, 10]
# each_layer_w = []
# for i in range(len(each_layer_dim)-1):
# 	each_layer_w.append(0.01*random.random((each_layer_dim[i],each_layer_dim[i+1])))
#
# print "1. start training..."
# print "	total training loop: "+str(training_times)
# for i in range(training_times):
# 	print "	loop "+str(i)+" ..."
# 	for j in range(rf.image_num):
# 		d_input = array(input_data[j])
# 		d_label = zeros(10)/1.0
# 		# print label_data[j]
# 		d_label[label_data[j]] = 1.0
# 		each_layer_w = one_train_for_BP(d_input,d_label,each_layer_dim,each_layer_w, 3, 0.01)
# 		# print cal_each_layer_out_4(image,each_layer_dim,each_layer_w)[3]
# 	# print each_layer_w[2]
# 	print "     right_num:	" + str(test_result(each_layer_w,input_data,label_data,3))

#step 2. testing training dataset
