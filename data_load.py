### 读取mnist数据集
import pickle
import numpy as np
import gzip

def load_data():
    '''
    return:
    X_train:[10853, 784]，可视化的时候需要reshape为[28, 28]
    y_train:[10853, 1]，数字1的标签为1， 数字7的标签为-1
    X_test:[2154, 784]
    y_test:[2154, 1]
    '''
    f = gzip.open(r'mnist.pkl.gz', mode='rb')
    tr_d, va_d, te_d = pickle.load(f, encoding='bytes')
    f.close()
    X_train = []
    y_train = []
    for data in zip(tr_d[0], tr_d[1]):
        if data[1] == 1:
            X_train.append(data[0])
            y_train.append(np.array([1]))
        elif data[1] == 7:
            X_train.append(data[0])
            y_train.append(np.array([-1]))
    X_train = np.array(X_train)#可以将一个列表转为一个矩阵
    y_train = np.array(y_train)
    X_test = []
    y_test = []
    for data in zip(va_d[0], va_d[1]):
        if data[1] == 1:
            X_test.append(data[0])
            y_test.append(np.array([1]))
        elif data[1] == 7:
            X_test.append(data[0])
            y_test.append(np.array([-1]))
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X_train, y_train, X_test, y_test