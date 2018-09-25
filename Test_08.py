import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('C:\\Users\\wzq3806295\\Desktop\\train.csv', encoding='gbk')
pm = data[data['測項'] == 'PM2.5'].copy()
pm.drop(['日期','測站','測項'],axis=1,inplace=True)

x_raw = []
y_raw = []

for i in range(15):
    x_t = pm.ix[:, i:i+9]
    x_t.columns = np.array(range(9))
    y_t = pm.ix[:, i+9]
    y_t.columns = ['1']

    x_raw.append(x_t)
    y_raw.append(y_t)

x_data = pd.concat(x_raw)
y_data = pd.concat(y_raw)
x = np.array(x_data,float)
y = np.array(y_data,float)

X_b = np.insert(x, 0, 1, 1)

w = np.zeros(X_b.shape[1])
num_iter = 10000
eta = 0.1
s_grad = np.zeros(X_b.shape[1])
for i in range(num_iter):
    tem = np.dot(X_b, w)
    loss = y - tem
    grad = np.dot(X_b.transpose(), loss)*(-2)
    s_grad += grad**2
    ada = np.sqrt(s_grad)
    w = w - (grad / ada)*eta

data_1 = pd.read_csv('C:\\Users\\wzq3806295\\Desktop\\test.csv', encoding='gbk')
data_1.columns = ['id','測項','1','2','3','4','5','6','7','8','9']
pm2_5 = data_1[data_1['測項'] == 'PM2.5'].ix[:,2:]
x_test = np.array(pm2_5,float)
x_test_b = np.insert(x_test, 0, 1, 1)

y_star = np.dot(x_test_b,w)
y_pre = pd.read_csv('https://ntumlta.github.io/2017fall-ml-hw1/sampleSubmission.csv')
y_pre.value = y_star
real = pd.read_csv('https://ntumlta.github.io/2017fall-ml-hw1/ans.csv')
errors = np.sum(abs(y_pre.value-real.value)) / len(real.value)
print(errors)