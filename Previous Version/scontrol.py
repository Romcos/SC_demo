import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


####### SAMPLING #######
def random_rct(N, I, T, T0, r = 2):
    """
    Provides a sample to test the MA-MRSC method
    
    Input: 
        N: number of units / experiments
        I: number of
        T: number of time steps
        TO: begining of intervention
        r: rank (of the latent tensor)
        
    Output: 
        Dataframe of a randomly generated RCT experiment.
    """
    np_data = np.zeros((N,T))
    i_rcv = np.random.randint(0,I, size = (N))
    ids, times, interventions = ["id_"+str(i) for i in range(N)], np.array(["t_"+str(t) for t in range(T)]), np.array(["inter_"+str(i) for i in range(I)])
    U = np.random.normal(size = (N,r))
    V = np.random.normal(size = (T,r))
    F = np.random.normal(size = (I,r))
    for n in range(N):
        for t in range(T0):
            np_data[n,t] = (U[n,:]*V[t,:]*F[i_rcv[0],:]).sum()
        for t in range(T0, T):
            np_data[n,t] = (U[n,:]*V[t,:]*F[i_rcv[n],:]).sum()
    rct_data = pd.DataFrame(data = np_data, columns = times, index = ids)
    rct_data.insert(0,"intervention",interventions[i_rcv])
    return rct_data

####### COMPUTING #######

def hsvt(Z,rank = 2):
    u, s, vh = np.linalg.svd(Z, full_matrices=False)
    s[rank:].fill(0)
    return np.dot(u*s,vh)

def cum_energy(s):
    return (100*(s**2).cumsum()/(s**2).sum())

def pcr(X,y,rank=2):
    """
    Input:
        X (N,T)
        y (T)
    
    Output:
        beta (N) a linear model
    """
    X = hsvt(X,rank=rank)
    beta = np.linalg.pinv(X.T).dot(y)
    return beta

####### QUERYING #######

def counterfactual_query(rct_data, T0, cell):
    """
    Gives the counterfactual observation for an unobserved cell
    
    Input:
        rct_data: typical RCT data as example above
        TO: time of intervention start (a number !)
        cell: cell for which we need a counterfactual estimation
    
    Output:
        Counterfactual estimation of the value of cell
    """

    #Keep relevant experiments
    data = rct_data[rct_data["intervention"] == cell["intervention"]]
    X = np.array(data.drop(columns ="intervention"))
    y1 = np.array(rct_data[rct_data.index==cell["unit"]].drop(columns="intervention"))[0,:T0]
    
    #Build llinear model
    beta = pcr((X[:,:T0]),y1)
    
    #Output predictions
    if "time" in cell.keys():
        return beta.dot(np.array(data[cell["time"]]))
    
    else:
        return (X[:,T0:].T).dot(beta)
    
    
####### PLOTTING ######

def placebo_experiment(rct_data, T0, unit):
    inter = (rct_data[rct_data.index==unit]["intervention"]).values[0]
    y = np.array((rct_data[rct_data.index==unit]).drop(columns="intervention"))[0]
    data = rct_data[(rct_data.intervention==inter) & (rct_data.index !=unit)]
    
    #Keep relevant experiments
    X = np.array(data.drop(columns ="intervention"))
    y1 = y[:T0]
    
    #Build llinear model
    beta = pcr((X[:,:T0]),y1)
    
    #Compute prediction
    yh = (X.T).dot(beta)
    
    plt.axvline(x=T0, color = "black", linestyle = ":", label="T0")
    plt.plot(y, color = "black", label= "real unit")
    plt.plot(yh, label= "synthetic unit", linestyle ="--")
    plt.title(unit + " ---> "+ str(inter))
    plt.legend()
