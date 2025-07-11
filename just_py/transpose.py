import numpy as np
# import torch 
import sympy as sym
# import torch.nn as nn
import matplotlib.pyplot as plt
# import scipy.stats as stats
# import sympy.plotting.plot as symplot



def model_par_plot(model_parameters):
    fig, ax = plt.subplots(1, 2, figsize=(12,4))
    for i in range(2):
      ax[i].plot(model_parameters[:, i],'o-')
      ax[i].set_xlabel('Iteration')
      ax[i].set_title(f'Final estimated minimum: {localmin:.5f}')

    ax[0].set_ylabel('Local minimum')
    ax[1].set_ylabel('Derivative')

    plt.show()


# nv = np.array([ [1, 2, 3, 4], 
#                [5, 6, 7, 8]])

# print(nv)

# print(nv.T)



# torch_v = torch.tensor([ [1, 2, 3, 4], 
#                [5, 6, 7, 8]])



# print(torch_v)

# print(torch_v.T)



# npv1 = np.array([1, 2, 3, 4])
# npv2 = np.array([99, 100 , 23123 ,0])

# print(np.dot(npv1, npv2))
# print(np.sum(npv1 * npv2))




# torchv1 = torch.tensor([1, 2, 3, 4])
# torchv2 = torch.tensor([99, 100 , 23123 ,0])

# print(torch.dot(torchv1, torchv2))
# print(torch.sum(torchv1 * torchv2))



# A = np.random.randn(3, 4)
# B = np.random.randn(4, 5)
# C = np.random.randn(5, 4)

# print(A)
# print(B)
# print(C)
# print(np.round(A @ B, 2))
# print(np.round(B @ C, 2))
# print(np.round(A @ C.T, 2))



# A = torch.randn(3, 4)
# B = torch.randn(4, 5)
# C1 = np.random.randn(4, 7)
# C2 = torch.tensor(C1, dtype = torch.float)


# print(np.round( A @ B, 2))
# print(np.round( A @ C1, 2))
# print(np.round( A @ C2, 2))


# # softmax


# z = [1, 2, 3]


# numerator = np.exp(z)
# denum = np.sum(np.exp(z))
# sigma = numerator / denum

# print(sigma)
# print(sum(sigma))

# # plt.plot(z, sigma, 'ko')
# # plt.xlabel('input')
# # plt.ylabel('softmax probs')
# # plt.show()

# softfunc = nn.Softmax(dim=0)

# sigmaT = softfunc(torch.Tensor(z))

# print(sigmaT)


#  #log 

# x = np.linspace(0.0001, 1, 200)

# logx = np.log(x)


# fig = plt.figure(figsize=(10, 4))

# # plt.plot(x, logx, 'ko')
# # plt.show()


# x = np.linspace(0.0001, 1, 20)
# logx = np.log(x)
# expx = np.exp(x)

# # plt.plot(x, x, color=[.8, .8, .8])

# # plt.plot(x, np.exp(logx), 'o', markersize=8)
# # plt.plot(x, np.log(expx), 'x', markersize=8)
# # # plt.show()



# # entropy

# p = [0.25, 0.75]

# H = 0
# for i in p:
#     H -= (p * np.log(p))

# print("Entropy: ", str(H))


# # cross entropy

# p = [1, 0.0000001]
# q = [.25, .75]

# H = 0
# for i, k in zip(p, q):
#     H -= (i * np.log(k))

# print("Cross entropy H(p, q): ", H, "\n\n")

# H = 0
# for i, k in zip(p, q):
#     H -= (k * np.log(i))

# print("Cross entropy H(q, p): ", H, "\n\n")


# # pytorch 
# import torch.nn.functional as F

# q_tensor = torch.Tensor(q)
# p_tensor = torch.Tensor(p)

# F.binary_cross_entropy(q_tensor, p_tensor)


# # min/max argmin argmax

# vv = np.array([1, 40, -2, 3])

# minval = np.min(vv)
# maxval = np.max(vv)


# minindx = np.argmin(vv)
# maxindx = np.argmax(vv)



# matrix = np.array([[0, 1, 10],[20, 8, 5]])

# minvals = np.min(matrix)
# minvals = np.min(matrix, axis=0) # across rows
# minvals = np.min(matrix, axis=1) # across columns

# minvalsind = np.argmin(matrix)
# minvalsind = np.argmin(matrix, axis=0) # across rows
# minvalsind = np.argmin(matrix, axis=1) # across columns


# matrix = torch.Tensor([[0, 1, 10],[20, 8, 5]])

# minvals = torch.min(matrix)
# minvals = torch.min(matrix, axis=0) # across rows
# minvals = torch.min(matrix, axis=1) # across columns

# print('torch min/max')
# print(minvals)
# print(minvals.values)
# print(minvals.indices)


# minvalsind1 = torch.argmin(matrix)
# minvalsind2 = torch.argmin(matrix, axis=0) # across rows
# minvalsind3 = torch.argmin(matrix, axis=1) # across columns


# # seeding

# np.random.randn(5) 

# np.random.seed(17)

# print(np.random.randn(5))
# print(np.random.randn(5))
# np.random.seed(17)
# print(np.random.randn(5))

# # seed function

# randseed1 = np.random.RandomState(5)
# randseed2 = np.random.RandomState(123489751)

# print(randseed1.randn(5))
# print(randseed2.randn(5))
# print(randseed1.randn(5))
# print(randseed2.randn(5))
# print(np.random.randn(5))

# # seed torch

# torch.randn(5)

# torch.manual_seed(111)
# print(torch.randn(5))
# print(torch.randn(5))
# torch.manual_seed(111)
# print(torch.randn(5))


# t test 



# data1 = 1 + np.random.randn(40)
# data2 = 2 + np.random.randn(30)

# t, p = stats.ttest_ind(data1, data2)
# print(t)
# print(p)

# plt.plot(0 + np.random.randn(40)/15, data1, 'ro', markerfacecolor='w', markersize=14)
# plt.plot(1 + np.random.randn(30)/15, data2, 'bs', markerfacecolor='w', markersize=14)
# plt.xlim([-1, 2])
# plt.xticks([0, 1], labels = ["Group1", "Group2"])
# plt.show()


# derivates





# x = sym.symbols('x')

# fx = 2 * x ** 2
# df = sym.diff(fx)

# print(fx)
# print(df)


# # symplot(fx, (x, -4, 4),  title="function")
# # symplot(df, (x, -4, 4),  title="derivate")
# # plt.show()

# def fx(x):
#     return 3 * x ** 2 - 3 * x + 4

# def dx(x):
#     return 6 * x - 3

# x = np.linspace(-2, 2, 2001)

# # random starting point
# localmin = np.random.choice(x, 1)
# learning_rate = 0.1
# training_epochs = 100

# model_paramaters = np.zeros((training_epochs, 2))
# for i in range(0, training_epochs):
#     grad = dx(localmin)
#     localmin = localmin - learning_rate * grad
#     print('grad: ', grad)
#     print('localmin: ', localmin)
#     model_paramaters[i, :] = localmin[0], grad[0]
    

# fig, ax = plt.subplots(1, 2, figsize=(12,4))

# for i in range(2):
#   ax[i].plot(model_paramaters[:, i],'o-')
#   ax[i].set_xlabel('Iteration')
#   ax[i].set_title(f'Final estimated minimum: {localmin[0]:.5f}')

# ax[0].set_ylabel('Local minimum')
# ax[1].set_ylabel('Derivative')

# plt.show()


###
# def fx(x):
#     return 3 * x ** 2 - 3 * x + 4

# def dx(x):
#     return 6 * x - 3

# x = np.linspace(-2, 2, 2001)

# # random starting point
# localmin = np.random.choice(x, 1)
# learning_rate = 0.01
# grad_threshold = 0.000000001
# training_epochs = 100

# model_paramaters = np.zeros((training_epochs, 2))
# for i in range(0, training_epochs):
#     grad = dx(localmin) 
#     localmin = localmin - learning_rate * grad
#     print('grad: ', grad)
#     print('localmin: ', localmin)
#     if abs(grad) < grad_threshold:
#         model_paramaters = np.delete(model_paramaters, list(range(i, training_epochs)), axis=0)
#         break

#     model_paramaters[i, :] = localmin[0], grad[0]
    

# fig, ax = plt.subplots(1, 2, figsize=(12,4))

# for i in range(2):
#   ax[i].plot(model_paramaters[:, i],'o-')
#   ax[i].set_xlabel('Iteration')
#   ax[i].set_title(f'Final estimated minimum: {localmin[0]:.5f}')

# ax[0].set_ylabel('Local minimum')
# ax[1].set_ylabel('Derivative')

# plt.show()




# code challenge
x = sym.symbols('x')
 
funcx = sym.cos(2*sym.pi*x) + x**2
dervx = sym.diff(funcx)

print('function: ', funcx)
print('derv function: ', dervx)


fcx = sym.lambdify(x, funcx, 'numpy')
dvx = sym.lambdify(x, dervx, 'numpy')

invals = np.linspace(-2, 2, 2000)

localmin = np.random.choice(invals)
localmin = 0
training_epochs = 100
learning_rate = 0.01
gradthreshold = 0.00000000001


mpars = np.zeros((training_epochs, 2))
for i in range(training_epochs):
    grad = dvx(localmin)
    if grad == 0:
       grad = (np.random.randint(200) / 100) + gradthreshold
    elif abs(grad) < gradthreshold:
        mpars = np.delete(mpars, list(range(i, training_epochs)), axis = 0)
        break
    localmin = localmin - learning_rate*grad
    print('localmin: ', localmin)
    print('grad: ', grad)
    mpars[i, :] = localmin, grad

model_par_plot(mpars)