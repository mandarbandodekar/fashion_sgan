'''
In most real-world applications, labelled data is scarce. Suppose you are given
the Fashion-MNIST dataset (https://github.com/zalandoresearch/fashion-mnist), but without any labels
in the training set. 
The labels are held in a database, which you may query to
reveal the label of any particular image it contains. Your task is to build a classifier to
>90% accuracy on the test set, using the smallest number of queries to this
>database. 

You may use any combination of techniques you find suitable
(supervised, self-supervised, unsupervised). However, using other datasets or
pre-trained models is not allowed. 
'''

def load_real_samples():
	# load dataset
	(trainX, trainy), (_, _) = load_data()
	# expand to 3d, e.g. add channels
	X = expand_dims(trainX, axis=-1)
	# convert from ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	print(X.shape, trainy.shape)
	return [X, trainy]
 

 # select a supervised subset of the dataset, ensures classes are balanced
def select_supervised_samples(dataset, n_samples=1000, n_classes=10):
	X, y = dataset
	X_list, y_list = list(), list()
	n_per_class = int(n_samples / n_classes)
	for i in range(n_classes):
		# get all images for this class
		X_with_class = X[y == i]
		# choose random instances
		ix = randint(0, len(X_with_class), n_per_class)
		# add to list
		[X_list.append(X_with_class[j]) for j in ix]
		[y_list.append(i) for j in ix]
	return asarray(X_list), asarray(y_list)


