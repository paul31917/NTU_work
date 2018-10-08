import keras
import pickle
import sys
from keras.models import Sequential
from keras.layers.core import Dense,Activation, Flatten
from keras.layers.convolutional import Convolution1D,Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dropout
from keras.utils import np_utils
import tensorflow as tf
from keras import backend as K
import numpy as np
from sklearn.utils import class_weight
import json
from keras.models import model_from_json
from keras.models import load_model

def load_data(data_num):
    filename="cv_"+str(data_num)+"_trainX"
    x_train=pickle.load(open(filename,'rb'))
    filename="cv_"+str(data_num)+"_trainY"
    y_train=pickle.load(open(filename,'rb'))

    filename="cv_"+str(data_num)+"_testX"
    x_test=pickle.load(open(filename,'rb'))
    filename="cv_"+str(data_num)+"_testY"
    y_test=pickle.load(open(filename,'rb'))

    filename='cv_'+str(data_num)+'_raw_Y_test'
    Y_raw = pickle.load(open(filename,'rb'))
    filename='cv_'+str(data_num)+'_testA'
    Act = pickle.load(open(filename,'rb'))

    x_train=x_train.reshape(x_train.shape[0],-1)
    x_test=x_test.reshape(x_test.shape[0],-1)
    
    x_train=x_train.reshape(x_train.shape+(1,))
    x_test=x_test.reshape(x_test.shape+(1,))
    print(x_train.shape)
    return x_train,y_train,x_test,y_test,Y_raw,Act

def create_model(input_shape):
    model = Sequential()
    #model.add(Convolution2D(128,3,3, input_shape=(input_shape[1:]), activation='relu'))
    model.add(Convolution1D(32,10,input_shape=input_shape[1:],activation='relu'))


    '''
    model.add(Convolution2D(256,3,3))

    model.add(Activation('relu')                    
    model.add(MaxPooling2D((3,3)))
    model.add(Convolution2D(128,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3,3)))
    '''
    model.add(Flatten())

    #model.add(Dense(output_dim=128))
    #model.add(Activation('relu'))
    model.add(Dense(output_dim=256))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=256))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=2))
    model.add(Activation('softmax'))
    return model

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall





def perf(Y_pre, Y_true, Y_raw, Act):
    TP, TN, FP, FN = 0, 0, 0, 0
    cont_time = 0
    fore_minute = 10
    
    for idx, i in enumerate(Y_pre):
        if cont_time > 0:
            cont_time -= 1
            continue
        if i == 1:
            if 1 in Y_raw[Act[idx] + 1:Act[idx] + fore_minute + 2]:
                TN += 1
                a = 0                
                while Y_raw[Act[idx+a]] != 1:
                    a+=1
                a += 2
                cont_time = a
            else:
                FN += 1
                cont_time = fore_minute
        else:
            if Y_true[idx] == 0:
                TP += 1
            else:
                FP += 1
    return TP, TN, FP, FN



all_TP=[]
all_TN=[]
all_FP=[]
all_FN=[]
ensemble_num=7
for i in (0,1,2):
    x_train,y_train,x_test,y_test,Y_raw,Act =  load_data(i)
    input_shape=x_train.shape
    label_array=np.where(y_train ==1)
    answer=[]
    for j in range(y_test.size):
        answer.append(0)
    
    weight = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
    y_train=np_utils.to_categorical(y_train,2)
    

    for run in range(ensemble_num):
        model=create_model(input_shape)
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=[recall,precision])
        model.summary()
        model.fit(x_train, y_train, nb_epoch=10, batch_size=100, class_weight = weight)
        '''
        model_json = model.to_json()
        with open("cnn_256_3d.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("cnn_256_3d.h5")
        '''

        y_prid=model.predict(x_test)
        print(y_prid)
    
        thresh= min(model.predict(x_train)[label_array][1])/50
        for idx,element in enumerate(y_prid):
            if element[1]>thresh:
                answer[idx]+=1
    
    for idx,ans in enumerate(answer):
        if ans >(3):#ensemble_num/5):
            answer[idx]=1
        else:
            answer[idx]=0



    TP, TN, FP, FN=perf(answer,y_test,Y_raw,Act)
    all_TP.append(TP)
    all_TN.append(TN)
    all_FP.append(FP)
    all_FN.append(FN)



print("TP = "+str(all_TP))
print("TN = "+str(all_TN))
print("FP = "+str(all_FP))
print("FN = "+str(all_FN))

for i in (0,1,2):
    print("precision_"+str(i)+"    = "+str(all_TN[i]/(all_TN[i]+all_FN[i])))
    print("recall_"+str(i)+"       = "+str(all_TN[i]/(all_TN[i]+all_FP[i])))
print ("precision_all = "+str(sum(all_TN)/(sum(all_TN)+sum(all_FN))))
print ("recall_all = "+str(sum(all_TN)/(sum(all_TN)+sum(all_FP))))







