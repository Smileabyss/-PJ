class args():

	# # training args
	epochs = 20000 #the number of training epochs, default is 2"
	batch_size = 1 #"batch size for training, default is 4"
	height = 512
	width = 640
	seed = 42 #"random seed for training"
    
    
	ssim_weight = [1,10,100,1000,10000]
	lr = 1e-3 #"learning rate, default is 0.001"

	model_path = "dic1.pth"
	# model_path = "dic.pth"
	data_path = 'data.npy'
	data_vstack_path = 'data_vstack.npy'    
    
	in_channels = [3, 3]
	out_channels = [1,3]
	kernel_size = [3,3]
	stride = [1,1]
	hidden_sizes = [30, 30]
    
	strategy_type = 'addition'