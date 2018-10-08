import os
import gzip
# from datetime import datetime
import time as Time
import pickle
import datetime
import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
import random
from sklearn.model_selection import ShuffleSplit
from keras.layers import Conv1D, Dense, Dropout, BatchNormalization, Flatten, Activation, AlphaDropout, MaxPool1D, AveragePooling1D, CuDNNGRU, Bidirectional
from sklearn.ensemble import ExtraTreesClassifier
import keras
from keras.models import Model, Input
import math
from keras import backend as K

def create_class_weight(labels_dict,mu=1):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        # score = mu*total/float(labels_dict[key])
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight


def info_gain(X, Y, n_est=30, print_flg = False, weight='balanced'):
    # test-------------------------------------------------------------    
    print('info_gain process...')    
    extra_tree = ExtraTreesClassifier(n_estimators=n_est,random_state=0, class_weight=weight)
    extra_tree.fit(X,Y)
    important_score = [(i,j) for (i,j) in enumerate(extra_tree.feature_importances_)]
    order_list = sorted(important_score,key=lambda x:x[1],reverse=True)
    order_list = [i[0] for i in order_list]
    # print(order_list)
    if print_flg:
        for i in range(1,6):
            X_ = X[:,order_list[:i]]
            Y_ = Y
            extra_tree2 = ExtraTreesClassifier(bootstrap=True, oob_score=True,n_estimators=n_est,random_state=0 )
            extra_tree2.fit(X_,Y_)        
            print(i,extra_tree2.oob_score_ )
    return order_list

def discrete_time(str_time):
    #time_V = time.strptime(str_time,'%Y-%m-%d %H:%M:%S')
    time_V = Time.mktime(datetime.datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S').timetuple())
    # print(time_V)
    n_t = int(time_V/300)
    pre_time = n_t*300
    next_time = (n_t+1)*300
    if time_V-pre_time < next_time-time_V:
        return pre_time
    else:
        return next_time

def nn_3layer(input_dim,l1,l2,l3,output_dim, dropout=0.5,alpha=5, gamma=1):    
    # with tf.device('/cpu:0'):
    reg = 0.001
    main_input = Input(shape=(input_dim, ))    
    dense = main_input
    dense = Dense(l1, kernel_initializer='lecun_normal', bias_initializer='zeros', kernel_regularizer=keras.regularizers.l2(reg))(dense)
    dense = Activation('selu')(dense)
    dense = AlphaDropout(dropout)(dense)
    dense = Dense(l2, kernel_initializer='lecun_normal', bias_initializer='zeros', kernel_regularizer=keras.regularizers.l2(reg))(dense)
    dense = Activation('selu')(dense)
    dense = AlphaDropout(dropout)(dense)
    dense = Dense(l3, kernel_initializer='lecun_normal', bias_initializer='zeros', kernel_regularizer=keras.regularizers.l2(reg))(dense)
    dense = Activation('selu')(dense)
    dense = AlphaDropout(2*dropout)(dense)    
    output = Dense(output_dim, activation='sigmoid', kernel_initializer='lecun_normal', bias_initializer='zeros')(dense)
    clf_model = Model(inputs=[main_input], outputs=[output])
    clf_model.summary()

    def loss_F(y_true, y_pred):
        lossFP = y_pred*(1-y_true)
        lossFN = (1-y_pred)*y_true*300
        return K.sum(lossFP, axis=-1) + K.sum(lossFN, axis=-1)
    def focal_loss(y_true, y_pred):
        losses =  - alpha * y_true * K.log(y_pred + K.epsilon()) - (1 - y_true) * (y_pred) ** gamma * K.log(1 - y_pred + K.epsilon())
        return K.mean(losses, axis=-1)

    def loss_0(y_true, y_pred):
        losses =   - (1 - y_true) * (y_pred)**gamma * K.log(1 - y_pred + K.epsilon())
        return K.sum(losses, axis=-1)

    def loss_1(y_true, y_pred):
        losses =  - alpha * y_true * K.log(y_pred + K.epsilon())
        return K.sum(losses, axis=-1)
    
    #parallel_model = multi_gpu_model(clf_model, gpus=2)
    clf_model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0005), loss='binary_crossentropy', metrics=['acc','binary_accuracy',loss_0, loss_1])
    earlyStopping = keras.callbacks.EarlyStopping(monitor='binary_crossentropy', patience=30, mode='min')
    return clf_model, earlyStopping

def nn_2layer(input_dim,l1,l2,output_dim, dropout=0.5,alpha=5, gamma=2):    
    # with tf.device('/cpu:0'):
    reg = 0.001
    main_input = Input(shape=(input_dim, ))    
    dense = main_input
    dense = Dense(l1, kernel_initializer='lecun_normal', bias_initializer='zeros', kernel_regularizer=keras.regularizers.l2(reg))(dense)
    dense = Activation('selu')(dense)
    dense = AlphaDropout(dropout)(dense)
    dense = Dense(l2, kernel_initializer='lecun_normal', bias_initializer='zeros', kernel_regularizer=keras.regularizers.l2(reg))(dense)
    dense = Activation('selu')(dense)
    dense = AlphaDropout(dropout)(dense)       
    output = Dense(output_dim, activation='sigmoid', kernel_initializer='lecun_normal', bias_initializer='zeros')(dense)
    clf_model = Model(inputs=[main_input], outputs=[output])
    clf_model.summary()

    clf_model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0005), loss='binary_crossentropy', metrics=['acc','binary_accuracy'])
    earlyStopping = keras.callbacks.EarlyStopping(monitor='binary_crossentropy', patience=30, mode='min')
    return clf_model, earlyStopping

class ENB:    
    def __init__(self, ENB_ID):
        self.ENB_ID = ENB_ID
        self.aggr_err_time = 0
        self.rel_MBH = []
    
    def relMBH(self, MBH_ID):
        if MBH_ID not in self.rel_MBH:
            self.rel_MBH.append(MBH_ID)
    
    def add_err(self, time):
        self.aggr_err_time += time

if __name__ == '__main__':
    print('ENB_ID')
    ENB_ID_dict = dict()
    min_time = discrete_time('2018-01-01 00:00:00')
    max_time = discrete_time('2018-04-29 23:59:59')
    total_timeslot = int((max_time-min_time)/300)+1

    print('ENB...')
    with open('201801-04-ENB-ALM-org.txt','r') as f:
        lines = f.readlines()        
        for line in lines:
            a = line.strip('\n').split(',')
            ENB_id, MBH_id, _time1, _time2 = a[0], a[1], discrete_time(a[2]), discrete_time(a[3])
            enb_ptr = None
            if ENB_id not in ENB_ID_dict.keys():
                enb_ptr = ENB(ENB_id)
                ENB_ID_dict[ENB_id]=enb_ptr
            else:
                enb_ptr = ENB_ID_dict[ENB_id]
            enb_ptr.rel_MBH(MBH_id)
            enb_ptr.add_err(_time2-_time1)        
        
        # filter err < 20 enb   
        for MBH_id in ENB_ID_dict.keys():
            if ENB_ID_dict[MBH_id].aggr_err_time/300 < 20:
                del ENB_ID_dict[MBH_id]
        cENB = list(ENB_ID_dict.keys())[0]

        # ENB err (label)        
        err_ENB = np.zeros(total_timeslot)
        for line in lines:
            a = line.strip('\n').split(',')
            ENB_id, MBH_id, _timeslot1, _timeslot2 = a[0], a[1], int(discrete_time(a[2])/300), int(discrete_time(a[3])/300)
            if ENB_id == cENB:
                err_ENB[min(_timeslot1,total_timeslot-1):_timeslot2+1]=1

    print('MBH...')    
    with open('201801-04-MBH-ALM-org.txt','r') as f:
        lines = f.readlines()
        MBH_ID = []
        MBH_err = []
        for line in lines:
            a = line.strip('\n').split('.')[0].split(',')            
            _id, err, _timeslot1, _timeslot2 = a[0], a[1], int(discrete_time(a[2])/300), int(discrete_time(a[2])/300)
            if _id in ENB_ID_dict[cENB].rel_MBH:
                if _id not in MBH_ID:
                    MBH_ID.append(_id)
                if err not in MBH_err:
                    MBH_err.append(err)

        err_MBH = np.zeros((total_timeslot,len(MBH_ID)*len(MBH_err)))
        col = lambda _id, err: MBH_ID.index(_id)*len(MBH_err)+MBH_err.index(err)
        for line in lines:
            a = line.strip('\n').split('.')[0].split(',')
            _id, err, _timeslot1, _timeslot2 = a[0], a[1], int(discrete_time(a[2])/300), int(discrete_time(a[2])/300)
            if _id in MBH_ID:                
                err_MBH[min(_timeslot1,total_timeslot-1):_timeslot2+1,col(_id,err)]=1
    print('pm_YYYYMMDD...')
    MBH_attrs_name = ['In','Out','Drop']
    pm_attrs = np.zeros( (total_timeslot,len(ENB_ID_dict[cENB].rel_MBH)*len(MBH_attrs_name)) )*-1
    col = lambda _id, attr_name: ENB_ID_dict[cENB].rel_MBH.index(_id)*len(MBH_attrs_name) + MBH_attrs_name.index(attr_name)
    directory = './data/'    
    files = os.listdir(directory)
    files.sort()
    for file in files:        
        if file.endswith('.gz'):            
            with gzip.open(directory+file,'rb') as f:
                lines = f.readlines()
                for line in lines:
                    a = line.strip('\n').split(',')
                    _id, time, attr_name, value = a[0], discrete_time(a[1]), a[2], int(a[3])
                    if _id in ENB_ID_dict[cENB].rel_MBH:
                        pm_attrs[time, col(_id,attr_name)]=value
    



    
    # Y = Y[:,:5]
    # for _idx in range(5):
    #     n_features = _idx*100+500
    #     print('# of features:',n_features,'-----------------')
    #     for train,test in ss.split(X):
    #         trainX = X[train]
    #         trainY = Y[train]
    #         testX = X[test]
    #         testY = Y[test]
    #         print('X',trainX.shape)
    #         print('Y',trainY.shape )
    #         weight = []
    #         for i in range(trainY.shape[1]):
    #             l_ = trainY[:,i]
    #             # print('test',np.count_nonzero(l_==0),np.count_nonzero(l_==1))
    #             lable_dict = {0:np.count_nonzero(l_==0),1:np.count_nonzero(l_==1)}
    #             print('weight',create_class_weight(lable_dict))
    #             weight.append(create_class_weight(lable_dict))
    #         feature_important_order = info_gain(trainX, trainY,weight = weight)
    #         trainX = trainX[:,feature_important_order[:n_features]]
    #         testX = testX[:,feature_important_order[:n_features]]

    #         input_dim = trainX.shape[1]
    #         output_dim = trainY.shape[1]
    #         clf, earlyStopping = nn_2layer(input_dim,1024,512,output_dim)
    #         clf.fit(trainX, trainY, batch_size=128, epochs=150, callbacks=[earlyStopping], validation_split=0.2, class_weight=weight)

    #         testX = trainX
    #         testY = trainY
            
    #         predY = clf.predict(testX)
    #         predY[predY>0.5]=1
    #         predY[predY<0.5]=0
    #         # result.append([predY,testY])
    #         # print('res',predY.shape, testY.shape)
    #         # print(predY)
    #         # print(testY)
    #         tp = np.sum(predY*testY*1,axis=0)
    #         tn = np.sum((1-predY)*(1-testY)*1,axis=0)
    #         fp = np.sum(predY*(1-testY)*1,axis=0)
    #         fn = np.sum((1-predY)*testY*1,axis=0)
    #         folder+=1
    #         print('res1',sum(tp),sum(tn),sum(fp),sum(fn))
    #         print('res2',tp, tn, fp, fn)
    #         with open('result'+str(folder),'wb') as ff:
    #             pickle.dump((tp, tn, fp, fn),ff)
    #     # with open('res','wb') as ff:
    #     #     pickle.dump(result,ff)
           