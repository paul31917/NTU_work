import pickle

# for cv_0 ###########################################
# Y_pre is the predicrive result of your model
# Y_true = pickle.load(open('cv_0_testY','rb'))
# Y_raw = pickle.load(open('cv_0_raw_Y_test','rb'))
# Act = pickle(open('cv_0_testA','rb'))
######################################################

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
                while Y_raw[Act[idx+a] != 1:
                    a += 1
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