import argparse
import matplotlib.pyplot as plt
import numpy as np

import data_load as dl
import svm
import pgd

parser = argparse.ArgumentParser(description='pgd攻击SVM')
parser.add_argument('--C', type=float, default=0.00005, help='svm超参数')
parser.add_argument('--T', type=int, default=1000, help='svm迭代次数')
parser.add_argument('--func_unit', type=int, default=100, help='svm每隔多少次迭代计算一次目标函数')
parser.add_argument('--loss_type', type=str, default='hinge', help='软svm的损失函数采用')
parser.add_argument('--pgd_iter', type=int, default=10, help='pgd攻击的迭代次数')
parser.add_argument('--alpha', type=float, default=0.02, help='pgd攻击的学习率')
parser.add_argument('--epsilou', type=float, default=0.2, help='pgd攻击的误差范围')
parser.add_argument('--pgd_type', type=str, default='l无穷', help='pgd攻击的类型，l无穷/l2')

args = parser.parse_args()

X_train, y_train, X_test, y_test = dl.load_data()
# X_train:10853*784,   y_train=10853*1,    X_test:2154*784,   y_test:2154*1

# 训练得到W,b，测试集的准确度（判断svm训练成功的标准，
W, b, acc = svm.pegasos(X_train, y_train, X_test, y_test, args)

# 输出所有训练样本的梯度
x_grad = pgd.adv_gen(X_train, y_train, W, b, args)

# pgd攻击
x_perturbed = pgd.attack(args, X_train, x_grad)
# 初始化对抗样本的标签为0
y_perturbed = np.zeros(x_perturbed.shape[0])

# 计算攻击成功的概率，也就是对抗样本识别错误的概率
num_attack_success = 0
attack_success_list=[]
for i in range(x_perturbed.shape[0]):
    res = np.dot(W.T, x_perturbed[i].T) + b
    if res >= 0 and y_train[i] == -1:  # 攻击成功
        num_attack_success += 1
        y_perturbed[i] = 1
        attack_success_list.append(i)
    elif res < 0 and y_train[i] == 1:
        num_attack_success += 1
        y_perturbed[i] = -1
        attack_success_list.append(i)
    elif res >= 0 and y_train[i] == 1:
        y_perturbed[i] = 1
    else:
        y_perturbed[i] = -1
rate_attack_success = 100 * num_attack_success / x_perturbed.shape[0]
print(f"攻击成功率：{rate_attack_success:.3f}%")

#可视化原始样本和对抗样本
random_indices=[attack_success_list[i] for i in np.random.randint(0,len(attack_success_list),5)]
samples=[[X_train[random_indices[i],:].reshape((28,28)) for i in range(5)]]
samples+=[[x_perturbed[random_indices[i],:].reshape((28,28)) for i in range(5)]]
lables=[[int(y_train[random_indices[i],:].reshape(1)) for i in range(5)]]
lables+=[[int(y_perturbed[random_indices[i]]) for i in range(5)]]

fig,axes=plt.subplots(nrows=2,ncols=5)
fig.suptitle("original samples and l_2_PGD_preturbed samples",fontsize=12)
for i in range(2):
    for j in range(5):
        axes[i][j].imshow(samples[i][j],cmap='gray')
        axes[i][j].set_title("lable:"+str(lables[i][j]))
        axes[i][j].axis('off')
plt.tight_layout
plt.show()



print("结束！")
