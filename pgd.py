# pgd算法生成对抗样本

import numpy as np


def adv_gen(train_x, train_y, W, b, args):
    # 计算出W和b后计算x_grad用于后续的pgd攻击
    num_train = train_x.shape[0]
    x_grad = np.zeros(train_x.shape)
    for i in range(num_train):
        x_i = train_x[[i]].T  # 1899*1
        y_i = train_y[i]  # 1
        if args.loss_type == 'hinge':
            if y_i * (np.dot(W.T, x_i)) < 1:
                x_grad[i, :] = (-y_i * W).T
            else:
                x_grad[i, :] = 0
        elif args.loss_type == 'exp':
            x_grad[i, :] = -np.exp(-y_i * W).T
        elif args.loss_type == 'log':
            pass
        # x_grad[i, :] = -np.exp(-y_i * W).T
    return x_grad


def attack(args, x, x_grad):
    x_copy = x
    for t in range(args.pgd_iter):
        if args.pgd_type == 'l无穷':
            x = x + args.alpha * np.sign(x_grad)
        elif args.pgd_type == 'l2':
            for i in range(x.shape[0]):
                x[i] = x[i] + args.alpha * x_grad[i] / np.linalg.norm(x_grad[i])
        else:
            print("类型错误")
    if args.pgd_type == 'l无穷' and np.linalg.norm(x - x_copy, np.inf) > args.epsilou:
        x = np.clip(x, x_copy - args.epsilou, x_copy + args.epsilou)
    elif args.pgd_type == 'l2' and np.linalg.norm(x - x_copy) > args.epsilou:
        x = x_copy + args.epsilou * (x - x_copy) / np.linalg.norm(x - x_copy)
    x_adv = np.clip(x, 0, 1)

    return x_adv
