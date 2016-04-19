# coding=utf-8
from BP_network import *
import matplotlib.pyplot as plt

# 读取图片文件信息
image_file = open('train-images.idx3-ubyte','rb')
label_file = open('train-labels.idx1-ubyte','rb')
t_image_file = open('t10k-images.idx3-ubyte','rb')
t_label_file = open('t10k-labels.idx1-ubyte','rb')
[image_num,rows,cols] = rf.confirm_image_file(image_file, 2051)
if not image_num:
	print "	false training image file"
if not rf.confirm_label_file(label_file, 2049):
	print "	flase training label file"
[t_image_num,t_rows,t_cols] = rf.confirm_image_file(t_image_file, 2051)
if not t_image_num:
	print "	false testing image file"
if not rf.confirm_label_file(t_label_file, 2049):
	print "	flase testing label file"

print "reading images..."
image_data = rf.get_images(image_file,image_num,rows,cols)
label_data = rf.get_labels(label_file,image_num)
t_image_data = rf.get_images(t_image_file,t_image_num,t_rows,t_cols)
t_label_data = rf.get_labels(t_label_file,t_image_num)

print "binaryzation and feature extraction.."
input_data = [] # 特征提取后,输入训练数据的数组
for x in range(image_num):
	o = zeros((1,rows*cols/4))
	for i in range(rows/2):
		for j in range(cols/2):
			for m in range(2):
				for n in range(2):
					if image_data[x][(2*i+m)*cols+2*j+n] > 48:
						o[0][i*cols/2+j]+=1
	input_data.append(o)
t_input_data = []  # 特征提取后,输入测试数据的数组
for x in range(t_image_num):
	o = zeros((1, t_rows * t_cols / 4))
	for i in range(t_rows / 2):
		for j in range(t_cols / 2):
			for m in range(2):
				for n in range(2):
					if t_image_data[x][(2 * i + m) * t_cols + 2 * j + n] > 48:
						o[0][i * t_cols / 2 + j] += 1
	t_input_data.append(o)

# 我们进行如下参数设置
neta = 0.01
training_times = 100
each_layer_dim = [cols*rows/4, cols+rows, 10]
each_layer_w = []
for i in range(len(each_layer_dim)-1):
	each_layer_w.append(0.01*random.random((each_layer_dim[i],each_layer_dim[i+1])))

accuracy_rate_training = [] # 记录每次迭代后结果的对于训练数据集准确率
accuracy_rate_testing = [] #记录每次迭代后结果对于测试数据机准确率

print "1. start training..."
print "	total training loop: "+str(training_times)
# 测试没进行训练前的正确率
accurate_num = test_result(each_layer_w,input_data,label_data,3, image_num)
accuracy_rate_training.append(accurate_num*1.0/image_num)
accurate_num = test_result(each_layer_w,t_input_data,t_label_data,3, t_image_num)
accuracy_rate_testing.append(accurate_num*1.0/t_image_num)
for i in range(training_times):
	print "	loop "+str(i)+" ..."
	for j in range(image_num):
		d_input = array(input_data[j])
		d_label = zeros(10)/1.0
		d_label[label_data[j]] = 1.0
		each_layer_w = one_train_for_BP(d_input,d_label,each_layer_dim,each_layer_w, 3, neta)
	accurate_num = test_result(each_layer_w, input_data, label_data, 3, image_num)
	accuracy_rate_training.append(accurate_num * 1.0 / image_num)
	print "     training_data_right_rate:	" + str(accurate_num * 1.0 / image_num)
	accurate_num = test_result(each_layer_w, t_input_data, t_label_data, 3, t_image_num)
	accuracy_rate_testing.append(accurate_num * 1.0 / t_image_num)
	print "     testing_data_right_rate:	" + str(accurate_num * 1.0 / t_image_num)

x = arange(training_times+1)
result_file = open('test_iteration_time_result.txt','w')
result_file.write(str(x)+"\n")
result_file.write(str(accuracy_rate_training)+"\n")
result_file.write(str(accuracy_rate_testing)+"\n")

plt.figure(1)
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
plt.sca(ax1)
plt.plot(x,accuracy_rate_training)
plt.sca(ax2)
plt.plot(x,accuracy_rate_testing)
plt.show()