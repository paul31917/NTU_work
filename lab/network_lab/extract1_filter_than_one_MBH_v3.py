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
from sklearn.mixture import GaussianMixture as GMM
from sklearn.model_selection import KFold
from sklearn import tree

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




def data_preprocesing():
    print('ENB_ID')
    ENB_ID_list = []
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
                ENB_ID_list.append(ENB_id)
            else:
                enb_ptr = ENB_ID_dict[ENB_id]
            enb_ptr.relMBH(MBH_id)
            enb_ptr.add_err(_time2-_time1)        
        
        # filter err < 20 enb
        filter_id=[]
        for MBH_id in ENB_ID_dict.keys():
            if ENB_ID_dict[MBH_id].aggr_err_time/300 < 20:
                filter_id.append(MBH_id)
        for _id in filter_id:
            del ENB_ID_dict[_id]        
        # choose ENB ID
        def choose_ENB(idx):            
            cENB_idx = 0
            for it in ENB_ID_list:
                if it in ENB_ID_dict.keys():                    
                    cENB_idx += 1
                    if cENB_idx == idx+1:
                        return it
            return None
        cENB = choose_ENB(5)
    timeslot = lambda x: int((x-min_time)/300)
    print(ENB_ID_dict[cENB].rel_MBH)

    with open('201801-04-ENB-ALM-org.txt','r') as f:
        lines = f.readlines()
        print(ENB_ID_dict[cENB].aggr_err_time)
        print(ENB_ID_dict[cENB].ENB_ID)
        # ENB err (label)        
        err_ENB = np.zeros(total_timeslot)        
        for line in lines:            
            a = line.strip('\n').split(',')
            ENB_id, MBH_id, _timeslot1, _timeslot2 = a[0], a[1], timeslot(discrete_time(a[2])), timeslot(discrete_time(a[3]))            
            if ENB_id == cENB:
                print(_timeslot1,_timeslot2)
                err_ENB[min(_timeslot1,total_timeslot-1):_timeslot2+1]=1
        print('check',np.count_nonzero(err_ENB))
        print('check',len(err_ENB))

    print('MBH...')    
    with open('201801-04-MBH-ALM-org.txt','r') as f:
        lines = f.readlines()
        MBH_ID = []
        MBH_err = []
        for line in lines:
            a = line.strip('\n').split('.')[0].split(',')            
            _id, err, _timeslot1, _timeslot2 = a[0], a[1], timeslot(discrete_time(a[2])), timeslot(discrete_time(a[3]))
            if _id in ENB_ID_dict[cENB].rel_MBH:
                if _id not in MBH_ID:
                    MBH_ID.append(_id)
                if err not in MBH_err:
                    MBH_err.append(err)

        err_MBH = np.zeros((total_timeslot,len(MBH_ID)*len(MBH_err)))
        col = lambda _id, err: MBH_ID.index(_id)*len(MBH_err)+MBH_err.index(err)
        for line in lines:
            a = line.strip('\n').split('.')[0].split(',')
            _id, err, _timeslot1, _timeslot2 = a[0], a[1], timeslot(discrete_time(a[2])), timeslot(discrete_time(a[3]))
            if _id in MBH_ID:                
                err_MBH[min(_timeslot1,total_timeslot-1):_timeslot2+1,col(_id,err)]=1

    print('pm_YYYYMMDD...')
    MBH_attrs_name = ['In','Out','Drop']
    pm_attrs = np.ones( (total_timeslot,len(ENB_ID_dict[cENB].rel_MBH)*len(MBH_attrs_name)) )*-1
    col = lambda _id, attr_name: ENB_ID_dict[cENB].rel_MBH.index(_id)*len(MBH_attrs_name) + MBH_attrs_name.index(attr_name)
    directory = './data/'    
    files = os.listdir(directory)
    files.sort()
    print(ENB_ID_dict[cENB].rel_MBH)
    for idx, file in enumerate(files):
        sys.stdout.write('\r%s/%s'%(idx,len(files)))
        sys.stdout.flush()
        if file.endswith('.gz'):            
            with gzip.open(directory+file,'rb') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.decode('utf-8') 
                    a = line.strip('\n').split(',')
                    try:
                        _id, time, attr_name, value = a[0], timeslot(discrete_time(a[1])), a[2], int(a[3])                        
                        if _id in ENB_ID_dict[cENB].rel_MBH:                            
                            pm_attrs[time, col(_id,attr_name)]=value
                    except KeyboardInterrupt:
                        break
                    except:
                        # print('aa1')
                        pass
        # print(pm_attrs)
        # print('aa',np.count_nonzero(pm_attrs!=-1))
        # print('aa',np.count_nonzero(pm_attrs==-1))
    with open('data_preprocessing','wb') as f:
        res = err_ENB, ENB_ID_dict, err_MBH, MBH_ID, MBH_err, pm_attrs
        pickle.dump(res,f)

def unsupervised(n_cluster=8):    
    with open('data_preprocessing','rb') as f:
        err_ENB, ENB_ID_dict, err_MBH, MBH_ID, MBH_err, pm_attrs = pickle.load(f)
    clf = GMM(n_components=n_cluster)
    clf.fit(pm_attrs)
    state = clf.predict(pm_attrs)
    entropy = []
    ENB_err = []
    ENB_and_MBH_err = []
    err_MBH_simp = np.sum(err_MBH,axis=1)
    print(np.count_nonzero(err_MBH_simp!=0))
    print('ENB err number:',np.count_nonzero(err_ENB))
    for cluster in range(n_cluster):
        c_list = state==cluster
        # cal entropy        
        p_abn = np.count_nonzero(err_ENB[c_list])/np.count_nonzero(c_list)
        ENB_err.append((np.count_nonzero(err_ENB[c_list]),np.count_nonzero(c_list)))
        ENB_and_MBH_err.append(np.count_nonzero(err_ENB[c_list]*err_MBH_simp[c_list]))
        entropy_1 = -p_abn*math.log(max(p_abn,0.0000001)) - (1-p_abn)*math.log(max(1-p_abn,0.0000001))
        entropy.append(entropy_1)
    print('ENB_err', ENB_err)
    print('ENB_and_MBH_err', ENB_and_MBH_err)
    print('entropy',entropy)

def unsupervised_abn(n_cluster=8):    
    with open('data_preprocessing','rb') as f:
        err_ENB, ENB_ID_dict, err_MBH, MBH_ID, MBH_err, pm_attrs = pickle.load(f)
    mask = err_ENB!=0
    pm_attrs = pm_attrs[mask,:]
    err_ENB = err_ENB[mask]
    err_MBH = err_MBH.reshape((err_MBH.shape[0],-1))
    err_MBH = err_MBH[mask,:]
    print(pm_attrs)
    clf = GMM(n_components=n_cluster)
    clf.fit(pm_attrs)
    state = clf.predict(pm_attrs)
    print(state.shape,pm_attrs.shape)
    ENB_err = []
    ENB_and_MBH_err = []
    err_MBH_simp = np.sum(err_MBH,axis=1)
    print(np.count_nonzero(err_MBH_simp!=0))
    print('ENB err number:',np.count_nonzero(err_ENB))
    for cluster in range(n_cluster):
        c_list = state==cluster
        # cal entropy
        ENB_err.append((np.count_nonzero(err_ENB[c_list]),np.count_nonzero(c_list)))
        ENB_and_MBH_err.append(np.count_nonzero(err_ENB[c_list]*err_MBH_simp[c_list]))        
        
    print('ENB_err', ENB_err)
    print('ENB_and_MBH_err', ENB_and_MBH_err)
    


def supervised(X,Y,w,d):    
    # cv = KFold(n_splits=4)
    cv = ShuffleSplit(n_splits=3, test_size=0.20)
    clf = RandomForestClassifier(n_estimators=40,class_weight={1:w,0:1})
    # clf = tree.DecisionTreeClassifier(max_depth=d, class_weight={1:w,0:1})
    res = np.zeros(4)
    for train,test in cv.split(X):
        trainX = X[train]
        trainY = Y[train]
        testX = X[test]
        testY = Y[test]
        clf.fit(trainX,trainY)
        predY = clf.predict(testX)

        tp = np.sum(predY*testY*1,axis=0)
        tn = np.sum((1-predY)*(1-testY)*1,axis=0)
        fp = np.sum(predY*(1-testY)*1,axis=0)
        fn = np.sum((1-predY)*testY*1,axis=0)
        
        res += np.array([tp,tn,fp,fn])
    print(res)

def find_longest(arr1D, min_time, filter_V = 0):
    pre_v = 0
    start_idx = 0
    L_time_start = 0
    L_time_interval = 0
    inv_timeslot = lambda _t: min_time + _t*300
    # print('geef',arr1D.shape)
    for _idx, _v in enumerate(arr1D) :        
        if pre_v == filter_V and _v != filter_V:
            start_idx = _idx            
        elif pre_v != filter_V and _v == filter_V:            
            interval = _idx - start_idx
            if interval > L_time_interval:
                L_time_start = start_idx
                L_time_interval = interval
        pre_v = _v
    if pre_v != filter_V:
        interval = len(arr1D) - start_idx
        if interval > L_time_interval:
            L_time_start = start_idx
            L_time_interval = interval
    return inv_timeslot(L_time_start), inv_timeslot(L_time_start+L_time_interval), L_time_interval

class ENB:
    timeslot = lambda self,x: round((x-self.min_time)/300)
    def __init__(self, ENB_ID, min_time=0, max_time = 0):
        self.ENB_ID = ENB_ID
        self.min_time = min_time
        self.max_time = max_time
        self.err = np.zeros( (round((max_time-min_time)/300+1),1) )
        # self.aggr_err_time = 0
        self.rel_MBH = []
    
    def relMBH(self, MBH_ID):
        if MBH_ID not in self.rel_MBH:
            self.rel_MBH.append(MBH_ID)
    
    def assign_err(self, start_time, end_time):        
        self.err[self.timeslot(start_time):self.timeslot(end_time)+1]=1

    def time_series_mask(self,start_time, end_time):
        mask = np.zeros(self.err.shape[0],dtype=bool)
        start_timeslot = self.timeslot(start_time)
        end_timeslot = self.timeslot(end_time)
        mask[start_timeslot:end_timeslot+1]=True
        return mask
        

class MBH:
    timeslot = lambda self,x: round((x-self.min_time)/300)
    # inv_timeslot = lambda self, _t: self.min_time + _t*300    
    
    def __init__(self, MBH_ID, min_time=0, max_time=0):
        self.MBH_ID = MBH_ID
        self.min_time = min_time
        self.max_time = max_time
        self.attr_timeslot = np.ones( (round((max_time-min_time)/300+1),3) )*-1
        self.longest_time_start = 0
        self.longest_time_end = 0
        self.longest_time_interval = 0
        self.alarm = None
    
    def assign_value(self, time, idx, value):
        _t = self.timeslot(time)
        self.attr_timeslot[_t,idx]=value    

    def assign_alarm(self, start_time, end_time, err_idx, total_alarm = None):
        if self.alarm is None:
            self.alarm = np.zeros((self.attr_timeslot.shape[0], total_alarm))
        self.alarm[self.timeslot(start_time): self.timeslot(end_time)+1, err_idx] = 1
    
    def time_series_mask(self,start_time, end_time):
        mask = np.zeros(self.attr_timeslot.shape[0],dtype=bool)
        start_timeslot = self.timeslot(start_time)
        end_timeslot = self.timeslot(end_time)
        mask[start_timeslot:end_timeslot+1]=True
        return mask
        

def pm_file_handler(MBH_dict=None, min_time=0, max_time=0):    
    print('mark non missing time for each MBH...')
    attr_name_list = ['In','Out','Drop']    
    # total_timeslot = round((max_time-min_time)/300)+1
    # timeslot = lambda x: round((x-min_time)/300)    
    directory = './data/'    
    files = os.listdir(directory)
    files.sort()
    if MBH_dict == None:
        MBH_dict = dict()
    for idx, file in enumerate(files):
        sys.stdout.write('\r%s/%s'%(idx,len(files)))
        sys.stdout.flush()
        if file.endswith('.gz'):            
            with gzip.open(directory+file,'rb') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.decode('utf-8') 
                    a = line.strip('\n').split(',')
                    try:
                        _id, time, attr_name, value = a[0], discrete_time(a[1]), a[2], int(a[3])                        
                        if _id not in MBH_dict.keys():                            
                            MBH_dict[_id]=MBH(_id, min_time = min_time, max_time = max_time)
                        MBH_dict[_id].assign_value(time, attr_name_list.index(attr_name),value)
                    except KeyboardInterrupt:
                        break
                    except:
                        # print('aa1')
                        pass
        # if idx==3:
        #     break
        # for key in MBH_dict.keys():
        #     print('ll',np.count_nonzero(MBH_dict[key].attr_timeslot!=-1))
        #     print('ll',np.count_nonzero(MBH_dict[key].attr_timeslot==-1))
        #     break    
        # print('dd',MBH_dict[key].longest_time_start,MBH_dict[key].longest_time_end)
    return MBH_dict

def MBH_file_handler(MBH_dict=None, min_time = None, max_time = None):
    print('MBH handler...')
    if MBH_dict is None:
        MBH_dict = dict()    
    # count number of err type and build error list
    with open('201801-04-MBH-ALM-org.txt','r') as f:
        lines = f.readlines()
        err_list = []
        for line in lines:
            a = line.strip('\n').split('.')[0].split(',')
            err = a[1]
            if err not in err_list:
                err_list.append(err)

    with open('201801-04-MBH-ALM-org.txt','r') as f:
        lines = f.readlines()        
        for line in lines:
            a = line.strip('\n').split('.')[0].split(',')            
            _id, err, _time1, _time2 = a[0], a[1], discrete_time(a[2]), discrete_time(a[3])
            if _id not in MBH_dict.keys():
                MBH_dict[_id] = MBH(_id, min_time = min_time, max_time = max_time)
            MBH_dict[_id].assign_alarm(_time1, _time2, err_list.index(err), total_alarm = len(err_list))
    return MBH_dict

def ENB_file_handler(ENB_dict = None, min_time = None, max_time = None):
    print('ENB handler...')
    if ENB_dict is None:
        ENB_dict = dict()    

    with open('201801-04-ENB-ALM-org.txt','r') as f:
        lines = f.readlines()        
        for line in lines:
            a = line.strip('\n').split('.')[0].split(',')            
            ENB_id, MBH_id, _time1, _time2 = a[0], a[1], discrete_time(a[2]), discrete_time(a[3])
            if ENB_id not in ENB_dict.keys():
                ENB_dict[ENB_id] = ENB(ENB_id, min_time=min_time, max_time=max_time)
            if MBH_id not in ENB_dict[ENB_id].rel_MBH:
                ENB_dict[ENB_id].relMBH(MBH_id)
            ENB_dict[ENB_id].assign_err(_time1, _time2)
    return ENB_dict
    #         if _id not in MBH_dict.keys():
    #             MBH_dict[_id] = MBH(_id, min_time = min_time, max_time = max_time)
    #         MBH_dict[_id].assign_alarm(_time1, _time2, err_list.index(err), total_alarm = len(err_list))
    # return MBH_dict
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

if __name__ == '__main__':
    min_time = discrete_time('2018-01-01 00:00:00')
    max_time = discrete_time('2018-04-29 23:59:59')
    # a,b, interval = find_longest(np.array([0,1,1,1,1,0]),min_time)
    # print('interval',interval)
    
    '''
    # build MBH matrix and get the longest non-missing time interval for each MBH
    MBH_dict = pm_file_handler(min_time = min_time, max_time = max_time)    
    # mark MBH alarm    
    MBH_dict = MBH_file_handler(MBH_dict=MBH_dict, min_time = min_time, max_time = max_time)
    # find ENB and MBH relationship, mark ENB alarm
    ENB_dict = ENB_file_handler(min_time = min_time, max_time = max_time)
    
    with open('result_data','wb') as f:
        pickle.dump( (MBH_dict,ENB_dict), f)
    '''
    
    # MBH_dict = MBH_file_handler(min_time = min_time, max_time = max_time)
    # for mbh in MBH_dict.keys():
    #     arr = MBH_dict[mbh].alarm
    #     print(arr.shape, np.count_nonzero(arr!=0))

    
    
    with open('result_data','rb') as f:
        MBH_dict, ENB_dict = pickle.load(f)
    
    # find ENB is related to only one single MBH
    '''
    if False:
        choosed_enb = []
        pair_list = []
        for enb_id in ENB_dict.keys():
            if len(ENB_dict[enb_id].rel_MBH) == 1:
                choosed_enb.append(enb_id)
        # find longest normal time with non missing mbh sensing value
        print('check enb', len(choosed_enb))
        enb_save = None
        cnt_save = 0
        max_interval = 0
        for enb_id in choosed_enb:
            # enb alarm
            enb_alarm_time_series = ENB_dict[enb_id].err.reshape((-1,))

            relmbh_id = ENB_dict[enb_id].rel_MBH[0]        
            # non missing mbh
            if relmbh_id not in MBH_dict.keys():
                print(relmbh_id, 'is not appear before')
            else:
                relmbh_attr_time_series = MBH_dict[relmbh_id].attr_timeslot
                temp = relmbh_attr_time_series
                temp[temp==-1]=0        
                temp = np.sum(temp, axis=1)
                temp[temp!=0]=1
                relmbh_attr_time_series_summary = temp
                # mbh alarm
                cnt=0
                if MBH_dict[relmbh_id].alarm is None:
                    cnt = np.count_nonzero(enb_alarm_time_series)
                else:                
                    # print(MBH_dict[relmbh_id].alarm.shape)
                    rel_mbh_alarm_time_series = np.sum(MBH_dict[relmbh_id].alarm, axis=1).reshape((-1,))
                    rel_mbh_alarm_time_series[rel_mbh_alarm_time_series!=0]=1                
                    # print(enb_alarm_time_series.shape, np.count_nonzero(enb_alarm_time_series))
                    # print(rel_mbh_alarm_time_series.shape, np.count_nonzero(enb_alarm_time_series))
                    cnt = np.count_nonzero(enb_alarm_time_series*(1-rel_mbh_alarm_time_series))
                if cnt >0 and MBH_dict[relmbh_id].alarm is not None:
                    # print('shape',enb_alarm_time_series.shape, relmbh_attr_time_series_summary.shape)
                    arr = (1-enb_alarm_time_series).reshape((-1,))*relmbh_attr_time_series_summary.reshape((-1,))
                    _,_,interval = find_longest(arr, min_time)
                    print(enb_id,cnt,interval)

                    pair_list.append((enb_id, cnt, interval))

                    if interval> max_interval:
                        max_interval = interval
                        enb_save = enb_id
                        cnt_save = cnt
        print('max',enb_save,cnt_save,max_interval)
        sorted_list = sorted(pair_list, key= lambda x:x[2],reverse=True)
        for _it in sorted_list[:20]:
            print(_it)
    '''
    print ("haha")
    enb_id = '20391c2c1e64008fea91312ae86f9b85'
    rel_mbh = ENB_dict[enb_id].rel_MBH[0]
    mbh_attrs = MBH_dict[rel_mbh].attr_timeslot
    mbh_alarm = MBH_dict[rel_mbh].alarm
    enb_err = ENB_dict[enb_id].err
    enb_alarm_time_series = ENB_dict[enb_id].err.reshape((-1,))    
    temp = mbh_attrs
    temp[temp==-1]=0        
    temp = np.sum(temp, axis=1)
    temp[temp!=0]=1
    relmbh_attr_time_series_summary = temp 
    arr = (1-enb_alarm_time_series).reshape((-1,))*relmbh_attr_time_series_summary.reshape((-1,))
    L_start,L_end,interval = find_longest(arr, min_time)

    print(mbh_attrs.shape, mbh_alarm.shape, enb_err.shape)
    mbh_alarm_s = np.sum(mbh_alarm,axis=1)
    mbh_alarm_s = mbh_alarm_s.reshape((-1,)) # mbh alarm happened (1 happen, 0 not happen)
    enb_err = enb_err.reshape((-1,))

    X = np.concatenate((mbh_alarm,mbh_attrs),axis=1)
    Y = enb_err
    X_val=X[-10000:]
    Y_val=Y[-10000:]
    X=X[:-10000]
    Y=Y[:-10000] 
    
    print(mbh_attrs.shape, enb_err.shape)

    weight = class_weight.compute_class_weight('balanced',np.unique(Y),Y)
    Y=np_utils.to_categorical(Y,2)

    model=create_model(input_shape)
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=[recall,precision])
    model.summary()
    model.fit(X,Y, nb_epoch=10, batch_size=100, class_weight = weight)

    Y_prid=model.predict(X_val)
    for idx,ans in enumerate(Y_val):
        print (Y_prid[idx,ans])


    '''
    for w in range(30,200):
        for d in range(20,45):
            print(w,d)
            supervised(X,Y,w,d)
    '''
    # cv = KFold(n_splits=4)
    # for idx, (train, test) in enumerate(cv.split(X)) :
    #     trainX = mbh_attrs[train]
    #     trainY = enb_err[train]
    #     testX = mbh_attrs[test]
    #     testY = enb_err[test]


                
        

    