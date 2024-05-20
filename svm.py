import numpy as np
import matplotlib.pyplot as plt
import os


# def plot(func_list, args, acc):
#     """
#     绘制模型在训练过程中的目标函数曲线
#     """
#     ts = [t for t in range(0, len(func_list))]
#     plt.plot(ts, func_list, 'k', label='training_cost')
#     plt.title('{} acc={}% C={} T={}'.format(args.loss_type, acc, args.C, args.T))
#     plt.xlabel('t')
#     plt.ylabel('f(W,b)')
#     if not os.path.exists('./output'):
#         os.makedirs('./output')
#     plt.savefig('./output/2.jpg')
def func(train_x, train_y, W, b, lambda_, loss_type):
    """
    根据当前W、b与loss种类,计算训练集样本的目标函数,平均值或总和均可
    该函数可全部重写,不必严格按照现有代码框架
    """
    num_train = train_x.shape[0]
    func_ = 0
    for i in range(num_train):  # 计算经验损失的总和
        x_i = train_x[[i]].T  # 1899*1
        y_i = train_y[i]  # 1
        if loss_type == 'hinge':
            func_ += max(0, 1 - y_i * (np.dot(W.T, x_i) + b))
        elif loss_type == 'exp':
            func_ += np.exp(-y_i * (np.dot(W.T, x_i) + b))
        elif loss_type == 'log':
            func_ += np.log(1 + np.exp(-y_i * (np.dot(W.T, x_i) + b)))

    func_ /= num_train  # 经验损失的平均
    func_ += 0.5 * lambda_ * np.dot(W.T, W)  # 正则项的平均
    return func_[0][0]


def pegasos(train_x, train_y, test_x, test_y, args):
    """
    func_unit: 每隔func_unit次记录一次当前目标函数值,用于画图
    佩加索斯算法
    """

    num_train = train_x.shape[0]  # 10853
    num_test = test_x.shape[0]  # 2154
    num_features = train_x.shape[1]  # 784

    # 记录目标函数值,用于画图
    # func_list = []

    # 初始化lambda_
    lambda_ = 1 / (num_train * args.C)

    # 高斯初始化权重W和偏置b
    W = np.random.randn(num_features, 1)  # 784*1
    b = np.random.randn(1)  # 1

    # 随机生成一组长度为T,元素范围在[0, num_train-1]的下标(可重复),供算法中随机选取训练样本
    choose = np.random.randint(0, num_train, args.T)  # T

    for t in range(1, args.T + 1):
        # 下降步长eta_t的计算公式
        eta_t = 1 / (lambda_ * t)

        i = choose[t - 1]  # 随机选取的训练样本下标
        # i = np.random.randint(0, num_train)
        x_i = train_x[[i]].T  # 1899*1
        y_i = train_y[i]  # 1

        if args.loss_type == 'hinge':
            # hinge_loss下的梯度更新公式
            if y_i * (np.dot(W.T, x_i)) < 1:
                W = W - eta_t * (lambda_ * W - y_i * x_i)
                b = b - eta_t * (-y_i)
            else:
                W = W - eta_t * (lambda_ * W)
        elif args.loss_type == 'exp':
            # exp_loss下的梯度更新公式
            if np.exp(-y_i * (np.dot(W.T, x_i) + b)) < 10:
                W = W - eta_t * (lambda_ * W - y_i * x_i * np.exp(-y_i * (np.dot(W.T, x_i) + b)))
                b = b - eta_t * (-y_i * np.exp(-y_i * (np.dot(W.T, x_i) + b)))
        elif args.loss_type == 'log':
            # log_loss下的梯度更新公式
            W = W - eta_t * (lambda_ * W - (y_i * x_i * np.exp(-y_i * (np.dot(W.T, x_i) + b))) / (
                    1 + np.exp(-y_i * (np.dot(W.T, x_i) + b))))
            b = b - eta_t * (-y_i * np.exp(-y_i * (np.dot(W.T, x_i) + b))) / (1 + np.exp(-y_i * (np.dot(W.T, x_i) + b)))

        if (t + 1) % args.func_unit == 0:
            # 根据当前W、b与loss种类,计算训练集样本的目标函数,平均值或总和
            func_now = func(train_x, train_y, W, b, lambda_, args.loss_type)
            # func_list.append(func_now)
            print('i = {}, func = {:.4f}'.format(t + 1, func_now))

    # 比对test数据上预测与实际的结果,统计预测对的个数,计算准确率acc
    num_correct = 0
    for i in range(test_x.shape[0]):
        res = np.dot(W.T, test_x[i].T) + b
        if res >= 0 and test_y[i] == 1:
            num_correct += 1
        elif res < 0 and test_y[i] == -1:
            num_correct += 1

    acc = 100 * num_correct / num_test
    print('acc = {:.1f}%'.format(acc))
    # print('func_list = {}'.format(func_list))

    return W, b, acc
