import struct

def confirm_image_file(image_file, magic_number):
	s = image_file.read(4)
	a, = struct.unpack('>i',s)
	if(a != magic_number):
		print a
		return False
	s = image_file.read(4)
	a, = struct.unpack('>i',s)
	print "number of images: "+str(a)
	image_num = a
	s = image_file.read(4)
	a, = struct.unpack('>i',s)
	print "number of rows: "+str(a)
	rows = a
	s = image_file.read(4)
	a, = struct.unpack('>i',s)
	print "number of cols: "+str(a)
	cols = a
	return [image_num,rows,cols]

def confirm_label_file(label_file, magic_number):
	s = label_file.read(4)
	a, = struct.unpack('>i',s)
	if(a != magic_number):
		print a
		return False
	s = label_file.read(4)
	a, = struct.unpack('>i',s)
	print "number of labels: "+str(a)
	return True

def read_image(image_file,rows,cols):
	pixels = []
	for x in range(cols*rows):
		s = image_file.read(1)
		a, = struct.unpack('B',s)
		pixels.append(a)
	return pixels

def read_label(label_file):
	s = label_file.read(1)
	a, = struct.unpack('B',s)
	return a

def get_images(image_file,image_num,rows,cols):
	image_data = []
	for x in range(image_num):
		image_data.append(read_image(image_file,rows,cols))
	return image_data

def get_labels(label_file,image_num):
	label_data = []
	for x in range(image_num):
		label_data.append(read_label(label_file))
	return label_data