import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

####### SAMPLING ######

def random_rct(N, I, T, T0, rank = 2, sigma = 0.1):
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
    pre_data = np.zeros((N,T0))
    post_data = np.zeros((N,T-T0))
    i_rcv = np.random.randint(0, I, size = (N))
    ids, times, interventions = ["id_"+str(i) for i in range(N)], np.array(["t_"+str(t) for t in range(T)]), np.array(["inter_"+str(i) for i in range(I)])
    U = np.random.normal(size = (N,rank))
    V = np.random.normal(size = (T,rank))
    F = np.random.normal(size = (I,rank))
    for n in range(N):
        for t in range(T0):
            pre_data[n,t] = (U[n,:]*V[t,:]*F[i_rcv[0],:]).sum() + sigma*np.random.normal()
        for t in range(T0, T):
            post_data[n,t-T0] = (U[n,:]*V[t,:]*F[i_rcv[n],:]).sum() + sigma*np.random.normal()
    pre_df = pd.DataFrame(data = pre_data, columns = times[:T0])
    pre_df.insert(0,"intervention",[interventions[0]]*N)
    pre_df.insert(0,"unit",ids)
    post_df = pd.DataFrame(data = post_data, columns = times[T0:])
    post_df.insert(0,"intervention",interventions[i_rcv])
    post_df.insert(0,"unit",ids)
    return pre_df, post_df

###### COMPUTATION ######

def hsvt(Z, rank = 2):
    u, s, vh = np.linalg.svd(Z, full_matrices=False)
    s[rank:].fill(0)
    return np.dot(u*s,vh)

def cum_energy(s):
    return (100*(s**2).cumsum()/(s**2).sum())

def pcr(X1, X2, y, rank=2, full_matrix_denoise=False):
    """
    Input:
        X (N,T)
        y (T)
    
    Output:
        beta (N) a linear model
    """
    if full_matrix_denoise:
    	X = hsvt(np.concatenate((X1, X2), axis=1), rank=rank)
    else:
    	X = hsvt(X1,rank=rank)
    _, n = X1.shape
    X_pre = X[:, :n]
    beta = np.linalg.pinv(X_pre.T).dot(y)
    return beta


###### DIAGNOSTIC ######

def diagnostic(rct_data, rank = 2):
    """
    Checks if 90% of energy is within rank
    """
    pre_df, post_df = rct_data
    interventions = np.sort(pd.unique(post_df.intervention))
    diagnostics_data = np.zeros(len(interventions))
    
    for i, inter in enumerate(interventions):
        M1 = np.array(pre_df[post_df.intervention == inter].drop(columns = ["unit","intervention"]))
        M2 = np.array(post_df[post_df.intervention == inter].drop(columns = ["unit","intervention"]))
        Mtot = np.concatenate((M1,M2),axis=1)
        _, s, _ = np.linalg.svd(Mtot, full_matrices=False)
        cum_energy(s)
        diagnostics_data[i]=cum_energy(s)[rank-1]
    
    diag = pd.DataFrame(data = diagnostics_data, columns = ["cum. energy at "+str(rank)+" (%)"])
    diag.insert(0,"intervention",interventions)
    diag["Valid (>90)"] = (diag["cum. energy at "+str(rank)+" (%)"] > 90)
    return diag

###### OUTPUT ##########

def fill_tensor(rct_data, rank=2, full_matrix_denoise=False):
    """
    Gives the counterfactual observation for an unobserved cell
    
    Input:
        rct_data: tuple (pre_df, post_df)
    
    Output:
        Counterfactual estimation for all tensor
    """
    # get pre- and post- intervention dataframes
    pre_df, post_df = rct_data

    # get all unique interventions (from post-intervention dataframe)
    interventions = np.sort(pd.unique(post_df.intervention))

    # sort all units (using pre-intervention data)
    units = np.sort(pre_df.unit)

    # get number of units and interventions
    N, I = len(units), len(interventions)

    # check no duplicate units in pre-intervention dataframe
    assert len(pre_df.unit.unique()) == N

    # initialize output dataframe size
    out_data = np.zeros((N*I,post_df.shape[1]-2))
    
    # loop through all interventions 
    for i, inter in enumerate(interventions):
        #Keep relevant experiments

        # extract all units that receive intervention "inter" (P_inter)
        filter_inter = (post_df["intervention"] == inter)

        # get pre-intervention measurements associated with P_inter
        X1_df = pre_df[filter_inter] #.drop(columns=["intervention","unit"]))

        # loop through all units (make sure id's unique)
        for n, unit in enumerate(pre_df.unit):
        	# get target unit pre-intervention measurements
            y1 = np.array(pre_df[pre_df.unit==unit].drop(columns=["intervention","unit"]))[0]

            # get donor unit post-intervention measurements for intervention "inter"
            X2_df = post_df[filter_inter]

            # check if unit is in donor pool 
            if unit in post_df[filter_inter].unit.values: 
            	X1 = X1_df.loc[(X1_df.unit != unit)].drop(columns=['intervention', 'unit']).values
            	X2 = X2_df.loc[(X2_df.unit != unit)].drop(columns=['intervention', 'unit']).values
            else:
            	X1 = X1_df.drop(columns=['intervention', 'unit']).values
            	X2 = X2_df.drop(columns=['intervention', 'unit']).values

            #Build linear model
            beta = pcr(X1, X2, y1, rank=rank, full_matrix_denoise=full_matrix_denoise)

            # forecast counterfactual
            out_data[n*I+i] = (X2.T).dot(beta)
            
    out_units = [units[k//I] for k in range(N*I)]
    out_interventions = [interventions[k%I] for k in range(N*I)]
    out = pd.DataFrame(data = out_data, columns = post_df.drop(columns=["intervention","unit"]).columns)
    out.insert(0, "intervention",out_interventions)
    out.insert(0, "unit",out_units)
    return out