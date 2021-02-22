

import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

def generate_dataset(d, n, alpha, shift):
    X_good = np.random.normal(0, 1, (int(n-alpha*n),d) )

    X_bad = np.random.normal(0, 1, (int(alpha*n),d))+shift
    X_raw = np.concatenate([X_good, X_bad], axis=0)
    return X_raw

def M(data, n):
    centered_X = data-data.mean(0)
    return 1/n*centered_X.T@centered_X

def output_perturbation(sigma,d):
    vec = np.random.normal(0, sigma, d**2).reshape(d,d)
    
    iu = np.triu_indices(d,1)
    il = (iu[1],iu[0])
    vec[il]=vec[iu]
    return vec


def filter_1d(tail_indices, tau, epsilon, delta, B, d):
    n = tau.shape[0]
    psi = np.sum(tau[tail_indices]-1)/n+np.random.laplace(0,B**2*d/(n*epsilon))
    bins = np.geomspace(1/4, B**2*d, num=2+int(np.log(B**2*d)))
    hist,  _ = np.histogram(tau[tail_indices], bins=bins)
    hist = hist/n
    
    for h in range(len(hist)):
        if hist[h]!=0:
            hist[h] += np.random.laplace(0, 2/(n*epsilon))
        if hist[h]<2*np.log(1/delta)/(epsilon*n)+1/n:
            hist[h] = 0

    for l in range(len(bins)-2, -1,-1):

        if np.sum((bins[l+1:]-bins[l])*hist[l:])>=0.3*psi:
            return bins[l]
    
    return None





        
        
def dp_range(X, epsilon,delta, R):
    n = X.shape[0]
    d = X.shape[1]
    x_bar = np.zeros(d)
    for i in range(d):
        bins = np.linspace(-R-0.5, R+0.5, 2*int(R))
        hist,  _ = np.histogram(X[:,i], bins=bins)
        hist = hist/n
        for h in range(len(hist)):
            if hist[h]!=0:
                hist[h] += np.random.laplace(0, 2/(n*epsilon/d))
            if hist[h]<2*np.log(d/delta)/(epsilon*n/d)+1/n:
                hist[h] = 0
        x_bar[i] = bins[np.argmax(hist)]+bins[np.argmax(hist)+1]
                
    return x_bar, 4*np.sqrt(np.log(d*n/0.9))


def filter_MMW(X, alpha, T1, T2, epsilon,delta, C, B):
    n = X.shape[0]
    d = X.shape[1]
    epsilon_1 = epsilon/(4*T1)
    epsilon_2 = epsilon/(4*T1*T2)
    delta_1 = delta/(4*T1)
    delta_2 = delta/(4*T1*T2)
    
    S = np.arange(0,n)
    
    for _ in range(T1):
        lamb_s = LA.norm(M(X[S], n)-np.identity(d), 2)+np.random.laplace(0,2*B**2*d/(n*epsilon_1))
        
        if len(S)<0.55*n:
            print('failed')
            return None
        if lamb_s<C*alpha*np.log(1/alpha):
            output = np.mean(X[S], axis=0)+np.random.normal(0, 2*B*np.sqrt(2*d*np.log(1.25/delta_1))/(n*epsilon_1), d )
            print('succeded, '+str(S.max())+str("  ")+str(LA.norm(output))+'  '+str(np.sum(S>=n-int(alpha*n))))
            return output
        alpha_s = 1/(10*(0.01/C+1.01)*lamb_s)

        Sigma_list = []
        for _ in range(T2):

            lamb_t = LA.norm(M(X[S], n)-np.identity(d), 2)+np.random.laplace(0,2*B**2*d/(n*epsilon_2))
            if lamb_t<lamb_s*0.5:
                print('next epoch')
                break
            else:
                Sigma = M(X[S],n)+output_perturbation(2*B*B*d*np.sqrt(2*np.log(1/delta_2))/(n*epsilon_2),d)
                Sigma_list.append(Sigma)
                sum_sigma = alpha_s*(np.array(Sigma_list)-np.identity(d)).sum(0)
                U = np.exp(sum_sigma)/np.trace(np.exp(sum_sigma))

                psi = np.trace((M(X[S], n)-np.identity(d))@(U.T))+np.random.laplace(0,2*B**2*d/(n*epsilon_2))
                if psi <= lamb_t/5.5:
                    continue
                else:
                    mu_t = np.mean(X[S], axis=0)+np.random.normal(0, 2*B*np.sqrt(2*d*np.log(1/delta_2))/(n*epsilon_2), d )
                    tau = (((X-mu_t)@U)*(X-mu_t)).sum(1)
                    sorted_tau_thres = np.sort(tau[S])[len(S)-2*int(alpha*n)]
                    
                    tail_indices = []
                    for l in S:
                        if tau[l]>=sorted_tau_thres:
                            tail_indices.append(l)
                    rho = filter_1d(tail_indices, tau, epsilon=epsilon_2, delta=delta_2, B=B, d=d)

                    S_remove = []
                    good = []
                    bad = []
                    for ind in tail_indices:
                        if tau[ind]>= np.random.uniform(0,1)*rho:
                            if ind>=n-int(alpha*n):
                                bad.append(ind)
                            else:
                                good.append(ind)
                            S_remove.append(ind)
                    plt.figure()
                    plt.hist(tau[bad], label='bad',bins=100)
                    plt.hist(tau[good], label='good',bins=100)
                    plt.legend()
                    plt.show()

                    S = np.setdiff1d(S, np.array(S_remove))
                print('next')
    return output



def PRIME(epsilon, delta, X, alpha, R):
    n = X.shape[0]
    d = X.shape[1]
    
    x_bar, B = dp_range(X, 0.01*epsilon,0.01*delta, R=R)

    for i in range(d):
        X[:,i] = np.clip(X[:,i], a_min = x_bar[i]-B, a_max = x_bar[i]+B) 

    C = 2
    if d>1:
        T1 = int(np.log(B*np.sqrt(d)))
        T2 = int(np.log(d))
    else:
        T1 = 2
        T2 = 2

    mean = filter_MMW(X=X, alpha=alpha, T1=T1, T2=T2, epsilon=0.99*epsilon,delta=0.99*delta, C=C, B=B)
        
    return mean
    

    
    
    

    
    
    
def DPmean(epsilon, delta, X, alpha, R):
    n = X.shape[0]
    d = X.shape[1]
    
    x_bar, B = dp_range(X, epsilon,delta, R=R)
    

    S = np.arange(0,n)
    S_bad = []
    for i in range(n):
        for j in range(d):
            if X[i][j]>=x_bar[j]+B or X[i][j]<=x_bar[j]-B:
                
                S_bad.append(i)
    S = np.setdiff1d(S, np.array(S_bad))
        
    
    return np.mean(X[S], axis=0)
    

    
    

if __name__ == "__main__":
    trials = 50
    errors_more = np.zeros((8,trials))
    errors_prime_more = np.zeros((8,trials))
    drange = [1, 10, 20,30,40,50, 75, 100]
    for i in range(len(drange)):
        d = drange[i]
        for j in range(trials):
            X = generate_dataset(d=d, n=1000000, alpha=0.05, shift=1.5)
            x_bar = DPmean(epsilon=10, delta=0.01, X=X, alpha=0.05, R=10)
            errors_more[i][j] = LA.norm(x_bar)
            mean = PRIME(epsilon=10, delta=0.01, X=X, alpha=0.05, R=10)

            errors_prime_more[i][j]= LA.norm(mean)


    plt.figure(figsize=(12,8))
    plt.errorbar([1, 10, 20,30,40,50, 75, 100 ],errors_more.mean(axis=1), np.std(errors_more, axis=1)/np.sqrt(trials), marker='o', label='DP mean')
    plt.errorbar([1, 10, 20,30,40,50, 75, 100], errors_prime_more.mean(axis=1), np.std(errors_prime_more, axis=1)/np.sqrt(trials), marker='D', label='Prime')
    plt.xlabel('Dimension $d$', fontsize=20)
    plt.ylabel('$\ell_2$ error $\|\hat{\mu}-\mu\|_2$', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.ylim([0,0.8])
    plt.savefig('error_dim2.png', dpi=200)


    
    