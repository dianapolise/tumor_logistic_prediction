import numpy as np
from log_func import sigmoid, z_score, sec_rate
import copy

print("Welcome to tumor prediction model!")
f_name = input("Enter name of the file: ")

n = 3
w_init = []
while n > 0:
    w_inp = input("Enter a value for weight(w): ")
    try:
        w_inp = float(w_inp)
        w_init.append(w_inp)
        n = n - 1
    except:
        print("Please enter a valid value!")
        continue
w_init = np.array(w_init)
b_init = float(input("Enter initial value for bias(b): "))
alpha_tmp = float(input("Enter learning rate value: "))
lambda_tmp = float(input("Enter regularization rate: "))
iterations = int(input("Enter number of iterations: "))

o_file = open(f_name)

tumor_prop = []
kind = []

for line in o_file:
    feature = line.split(":")[0].strip()
    for i in range(3):
        var = float(feature.split(",")[i].strip())
        tumor_prop.append(var)
    type_ = line.split(":")[1].strip()
    if type_ == "malignant":
        kind.append(1)
    elif type_ == "benign":
        kind.append(0)
tumor_prop = z_score(np.array(tumor_prop).reshape(-1,3))
tumor_train = tumor_prop[:100]
tumor_pred = tumor_prop[101:]
kind = np.array(kind)
kind_train = kind[:100]
kind_pred = kind[101:]

def compute_cost_log_reg(X,y,w,b,lambda_):
    m,n = X.shape
    cost = 0
    for i in range(m):
        f_wb = sigmoid(np.dot(X[i],w) + b)
        cost = cost -y[i]*np.log(f_wb) - (1 - y[i])*np.log(1 - f_wb)
    cost = cost / m
    reg_cost = 0
    for j in range(n):
        reg_cost = reg_cost + (w[j])**2
    reg_cost = (lambda_ /(2 * m))*reg_cost
    total_cost = reg_cost + cost
    return cost

def compute_grad_log_reg(X,y,w,b,lambda_):
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        f_wb = sigmoid(np.dot(X[i],w) + b)
        err = f_wb - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i,j]
        dj_db = dj_db + err
        dj_dw = dj_dw / m
        dj_db = dj_db / m
        for j in range(n):
            dj_dw[j] = dj_dw[j] + (lambda_ / m)*w[j]
        return dj_dw, dj_db

def compute_gradient_descent(X,y,w_in,b_in,num_iters,alpha,lambda_):
    m,n = X.shape
    w = copy.deepcopy(w_in)
    b = b_in
    dj_dw, dj_db = compute_grad_log_reg(X,y,w,b,lambda_)
    for i in range(num_iters):
        w = w * (1 - alpha * (lambda_ / m)) - alpha * dj_dw
        b = b - alpha * dj_db
    return w, b

def prediction(X,w,b):
    m,n = X.shape
    y_pred = np.zeros(m)
    for i in range(m):
        f_wb = sigmoid(np.dot(X[i],w) + b)
        if f_wb >= 0.5:
            y_pred[i] = 1
        elif f_wb < 0.5:
            y_pred[i] = 0
    return y_pred

w_i = np.zeros_like(w_init)
w_final, b_final = compute_gradient_descent(tumor_train,kind_train,w_i,b_init,iterations,alpha_tmp,lambda_tmp)

print(f"Final value for w -> {w_final} and final value for b -> {b_final:0.4f}")
print("-------------------------------")
predict = prediction(tumor_pred,w_final,b_final)
tumor_pred_i = tumor_pred[:10]
for i in range(tumor_pred_i.shape[0]):
    print(f"The tumor prediction is {predict[i]} and actual value is {kind_pred[i]}") 
    
ratio = sec_rate(predict,kind_pred)
print(f"The success rate of model is -> %{ratio}")