import numpy as np
from sklearn import linear_model
from sklearn.utils import resample

# β0, β1, β2を返す重回帰分析は関数にしておく
def reg_3_betas(X, y):
  reg = linear_model.LinearRegression()
  reg.fit(X, y)
  beta_1, beta_2 = reg.coef_
  beta_0 = reg.intercept_
  return beta_0, beta_1, beta_2

def bootstrapping_residuals(x1, x2, Y, sample_times=1000):
    X = np.array([x1, x2]).T  # Xを転置して、行数をyの要素数に合わせる
    beta_0, beta_1, beta_2 = reg_3_betas(X, y)
    # 誤差：モデルと実測値との差を算出する
    hat_y = {} # 予測値
    res = [] #残差
    for i in range(len(y)):
    hat_y[i] = beta_0 + beta_1 * x1[i] + beta_2 * x2[i] 
    res.append(y[i] - hat_y[i])
    # bootstrapの開始
    bootstrapped_y = []
    bootstrapped_x1 =[]
    bootstrapped_x2 =[]
    b0_list = []
    b1_list = []
    b2_list = []
    # 何回サンプルを取るかを設定
    for n in range(sample_times): 
        # 残差のサンプル数ぶんだけランダムに残差を取り出す（ε*iを残差の数だけ取得）
        bootstrapped_res_sample = resample(res, replace=True, n_samples=len(res))
        # 取り出した残差が元々の残差の何番目にあるか取得（εiのiを取得）
        res_index = [res.index(i) for i in bootstrapped_res_sample]
        # εiのiに対応したhat_yとx1_i、x2_iを取得し、それぞれをリストに格納
        bootstrapped_y += [hat_y[i] + bootstrapped_res_sample[i] for i in res_index]
        bootstrapped_x1 += [x1[i] for i in res_index]
        bootstrapped_x2 += [x2[i] for i in res_index]
        bootstrapped_X = np.array([bootstrapped_x1, bootstrapped_x2]).T
        b0, b1, b2 = reg_3_betas(bootstrapped_X, bootstrapped_y)
        b0_list.append(b0)
        b1_list.append(b1)
        b2_list.append(b2)
    means = [np.mean(b0_list), np.mean(b1_list),np.mean(b2_list)]
    return means

def bootstrapping_pairs(x1,x2,Y,sample_times=1000):
    b0_list = []
    b1_list = []
    b2_list = []
  
    # 何回サンプルを取るかを設定
    for n in range(sample_times): 
        sample = resample(a, replace=True, n_samples=len(a))
        x1_hat, x2_hat, y_hat = list(zip(*sample))
        X_hat = np.array([x1, x2]).T
        b0,b1,b2 = reg_3_betas(X_hat, y_hat)
        b0_list.append(b0)
        b1_list.append(b1)
        b2_list.append(b2)
    means = [np.mean(b0_list), np.mean(b1_list),np.mean(b2_list)]
    return means
