''''
Nearest Classifier is the basic Ml algo in which two function
def train(Images, labels):
    return model
    
def predict(model, test_images):
    return test_labels


web demo of nearest neighbour  http://vision.stanford.edu/teaching/cs231n-demos/knn

Setting hyperparameters: Very problem-dependent. In general need to try them all and
see what works best for our data / task



K-Nearest Neighbor: Universal Approximation
As the number of training samples goes to infinity, nearest
neighbor can represent any(*) function!


Very slow at test time
Distance metrics on pixels are not informative

but Nearest Neighbor with ConvNet features works well 
'''

import numpy as np

class NearestNeighbour:
    def __init__(self) -> None:
        pass
    
    # memories all the training samples of dataset
    def train(self, X, y):
        ''' X is the the N X D where each row is examples of sample and y is the 1-D of size N '''
        self.Xtr = X
        self.ytr =y
        
    def predict(self, X):
        
        '''X in this function is the e the N X D where each row is examples of sample we would predict the labels'''
        
        num_test = X.shape[0]
        # outputs matches with the input type 
        
        Y_predict = np.zeros(num_test, dtype=self.ytr.dtype)
        
        ## iterate the all the test row to predict the nearest training images 
        for i in range(num_test):
            
            # find the nearest training images to the ith test images 
            # using the distacne metics find the sum of the absolute difference b/w train nd test
            distacnes = np.sum(np.abs(self.Xtr - X[i,:]), axis=1)
            min_index =np.argmin(distacnes)
            Y_predict[i] =self.ytr[min_index]
    
        return Y_predict        
        