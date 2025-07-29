import numpy as np
import matplotlib.pyplot as plt


def GeneratePathsMerton(NoOfPaths,NoOfSteps,T,r,sigma,S_0,rate,muJ,sigmaJ):    
    
    # Fixing random seed
    np.random.seed(1)
    dt = T / float(NoOfSteps)   
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    X = np.zeros([NoOfPaths, NoOfSteps+1])
    S = np.zeros([NoOfPaths, NoOfSteps+1])
    time = np.zeros([NoOfSteps+1])
    J=np.random.normal(muJ,sigmaJ,[NoOfPaths,NoOfSteps])
    xp=np.random.poisson(rate*dt,[NoOfPaths,NoOfSteps])
    Ej=np.exp(muJ+0.5*sigmaJ*sigmaJ)
    X[:,0] = np.log(S_0)
    S[:,0] = S_0
    
    
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
            
        X[:,i+1] = X[:,i] + (r - 0.5 * sigma *sigma-rate*(Ej-1) ) * dt + sigma * np.power(dt, 0.5)*Z[:,i]+J[:,i]*xp[:,i]
        time[i+1] = time[i] +dt
        
    #Compute exponent of ABM
    S = np.exp(X)
    paths = {"time":time,"X":X,"S":S}
    return paths

def mainCalculation():
    NoOfPaths = 25
    NoOfSteps = 500
    T = 5
    r = 0.05
    sigma = 0.2
    rate=1
    S0 = 100
    muJ=0
    sigmaJ=0.7
    
    Paths = GeneratePathsMerton(NoOfPaths,NoOfSteps,T,r,sigma,S0,rate,muJ,sigmaJ)
    timeGrid = Paths["time"]
    X = Paths["X"]
    S = Paths["S"]
    
    plt.figure(1)
    plt.plot(timeGrid, np.transpose(X))   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("X(t)")
    
    plt.figure(2)
    plt.plot(timeGrid, np.transpose(S))   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("S(t)")
    plt.show()
mainCalculation()