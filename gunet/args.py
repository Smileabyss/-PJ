class args():


	# # training args
	epochs = 20000 #the number of training epochs, default is 2"
	batch_size = 1 #"batch size for training, default is 4"
	height = 512
	width = 640
	seed = 42 #"random seed for training"
    
    
	ssim_weight = [1,10,100,1000,10000]
	lr = 1e-5 #"learning rate, default is 0.001"

	model_path = "gunet_model.pth"
	data_path = 'data.npy'  
    

    