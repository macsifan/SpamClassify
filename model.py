from scipy.io import loadmat
import utils
import pickle
import os

data = loadmat(os.path.join('Data', 'spamTrain.mat'))
X, y= data['X'].astype(float), data['y'][:, 0]
print('Training Linear SVM (Spam Classification)')
print('This may take 1 to 2 minutes ...\n')
C = 0.1
model = utils.svmTrain(X, y, C, utils.linearKernel)
pickle.dump(model, open('model.pkl','wb'))