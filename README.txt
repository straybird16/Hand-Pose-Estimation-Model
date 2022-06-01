Instructions:
		
	
	All the training and testing are in main.py. There is a generated dataset included which has 1398 x 20 x 600 x 3 points. If you would like to use this generated dataset, you can just ignore utils.py. If you wish to generate another dataset with different numbers of frame and points in one frame, you will need to check utils.py, change the variables accordingly, and run the code. Depending on the number of points you are generating, it can take 30 to 60 minutes.

	If you want the original data_set, execute the command in utils.py (wget http://www-rech.telecom-lille.fr/DHGdataset/DHG2016.zip) to download. The dataset is large (5GB)
	
 	For training, first set device to the device you wish to use (by default it is cuda:0), then set the hyper-parameter values. It is suggested that you set save_best to True and num_training to 2. If you have trained a model and decides to change the model parameter (e.g., adding layers), comment out the load model line. Note that hidden size layer is set to be equal to the number of global features extracted by the Point Net. Technically they do not have to be the same but for implementation reasons here they should be set to the same size. If you are testing the model, set "train" to False.

	External libraries can be installed just like other common libraries. (pip install or install inside your IDE)

