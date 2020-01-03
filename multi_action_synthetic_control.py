import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


####### SAMPLING ######
def enumerate_str(N):
    n_digit = int(np.log(N-1)/np.log(10)) + 1
    return ([str(i).zfill(n_digit) for i in range(N)])
    
def random_rct(N, I, M, T, T0, rank=2, sigma=0.1):
    """
    Provides a sample to test the MA-MRSC method

    Input:
        N: number of units / experiments
        I: number of interventions
        M: number of metrics
        T: number of time steps
        TO: begining of intervention
        r: rank (of the latent tensor)

    Output:
        Dataframe of a randomly generated RCT experiment.
    """

    pre_data = np.zeros((N*M, T0))
    post_data = np.zeros((N*M, (T - T0)))
    i_rcv = np.random.randint(0, I, size=(N))

    ids = ["id_" + i for i in enumerate_str(N)]
    times = np.array(["t_" + t for t in enumerate_str(T)])
    metrics = np.array(["m_"+ m for m in enumerate_str(M)])
    interventions = np.array(["inter_" + i for i in enumerate_str(I)])

    U = np.random.normal(size=(N, rank)) + 1
    V = np.random.normal(size=(T, rank)) + 1
    W = np.random.normal(size=(M, rank)) + 1
    F = np.random.normal(size=(I, rank)) + 1

    for n in range(N):
        for t in range(T0):
            for m in range(M):
                pre_data[n*M+m, t] = (U[n, :] * V[t, :] * W[m,:] * F[i_rcv[0], :]).sum() + sigma * np.random.normal()
        for t in range(T0, T):
            for m in range(M):
                post_data[n*M+m, t-T0] = (U[n, :] * V[t, :] * W[m,:] * F[i_rcv[n], :]).sum() + sigma * np.random.normal()
                         
    def repeat(a_list, k):
        return [a for a in a_list for _ in range(k)]
    
    pre_df = pd.DataFrame(data=pre_data, columns=times[:T0])
    pre_df.insert(0, "metric", list(metrics)*N)
    pre_df.insert(0, "intervention", [interventions[0]] * N * M)
    pre_df.insert(0, "unit", repeat(ids,M))

    post_df = pd.DataFrame(data=post_data, columns=times[T0:])
    post_df.insert(0, "metric", list(metrics)*N)
    post_df.insert(0, "intervention", repeat(interventions[i_rcv],M))
    post_df.insert(0, "unit", repeat(ids,M))

    return pre_df, post_df

###### COMPUTATION ######

def hsvt(Z, rank=2):
    if rank is None:
        return Z
    u, s, vh = np.linalg.svd(Z, full_matrices=False)
    s[rank:].fill(0)
    return np.dot(u * s, vh)

def cum_energy(s):
    return (100 * (s ** 2).cumsum() / (s ** 2).sum())

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
    
def pcr_full_matrix_denoise(X1, X2, y, rank=2, full_matrix_denoise=False):
    if full_matrix_denoise:
        X = hsvt(np.concatenate((X1, X2), axis=1), rank=rank)
    else:
        X = hsvt(X1, rank=rank)
    _, T = X1.shape
    X_pre = X[:, :T]
    beta = np.linalg.pinv(X_pre.T).dot(y)
    return beta

def pcr_normalised_metrics(X1_3d,y_2d,rank):
    """ Input: 
        X1: order 3 representing pre-intervention donnors (NxT0xM)
        y_2d: order 2 representing pre-intervention target (T0xM)
    
    Output: 
        beta (linear model) after metric renormalisation"""
    metric_means = X1_3d.mean(axis=(0,1))
    metric_stds = X1_3d.std(axis=(0,1))
    y_2d = (y_2d - metric_means)/(metric_stds)            
    X1_3d = (X1_3d - metric_means)/(metric_stds)
    N,M,T0 = X1_3d.shape
    return pcr(X1_3d.reshape(N,T0*M),y_2d.reshape(T0*M),rank=rank)

###### DIAGNOSTIC ######

def diagnostic(post_df, df_output):
    # get all unique interventions (from post-intervention dataframe)
    interventions = np.sort(pd.unique(post_df.intervention))

    # sort all units (using post-intervention dataframe)
    units = np.sort(pd.unique(post_df.unit))

    # get number of units and interventions
    N, I = len(units), len(interventions)

    R2_all_interventions = np.zeros(len(interventions))

    # loop through all interventions
    for i, inter in enumerate(interventions):

        unit_ids = np.unique(post_df[post_df['intervention'] == inter]['unit'])

        baseline_error_sum = 0
        estimated_error_sum = 0

        for unit in unit_ids:

            y = post_df.loc[(post_df.unit==unit) & (post_df.intervention==inter)].drop(columns=['unit', 'intervention','metric']).values
            
            y_hat = df_output.loc[(df_output.unit==unit) & (df_output.intervention==inter)].drop(columns=['unit', 'intervention','metric']).values

            baseline_error_sum += ((y.mean(axis=0) - y)**2).sum()

            estimated_error_sum += ((y_hat - y)**2).sum()
          
        R2_all_interventions[i] = 1 - estimated_error_sum / baseline_error_sum

    diag = pd.DataFrame(data= R2_all_interventions, columns = ["Average R^2 Value"])
    diag.insert(0, "intervention", interventions)
    return diag
#             plt.figure()
#             plt.plot(y.flatten(), label='obs')
#             plt.plot(y_hat.flatten(), label='pred')
#             plt.legend(loc='best')
#             plt.show()


###### OUTPUT ##########

def fill_tensor(pre_df, post_df, rank=2, full_matrix_denoise=True):
    """
    Gives the counterfactual observation for an unobserved cell

    Input:
        rct_data: tuple (pre_df, post_df)

    Output:
        Counterfactual estimation for all tensor
    """
    # get all unique interventions (from post-intervention dataframe)
    interventions = np.sort(pd.unique(post_df.intervention))

    # get all units (using pre-intervention data)
    units = list(np.sort(pd.unique(pre_df.unit)))
    
    # get all metrics
    metrics = list(np.sort(pd.unique(pre_df.metric)))

    # get number of units and interventions
    N, I = len(units), len(interventions)
    T0 = pre_df.shape[1]-3
    T = T0 + post_df.shape[1]-3
    M = len(metrics)

    # check no duplicate units in pre-intervention dataframe
    assert len(pre_df.unit.unique()) == N

    # initialize output dataframe size
    out_data = np.zeros((N * I * M, T-T0))

    beta_dict = {}

    for i, inter in enumerate(interventions):
        # Keep relevant experiments

        # extract all units that receive intervention "inter" (P_inter)
        unit_ids = pd.unique(post_df[post_df["intervention"] == inter]['unit'])
        n_i = len(unit_ids)

        # get pre-intervention measurements associated with P_inter
        X1_df = pre_df[pre_df['unit'].isin(unit_ids)].sort_values('unit')  # .drop(columns=["intervention","unit"]))
        
        #loop in all units
        for n,unit in enumerate(units):
            
            # With full_matrix denoise
            #
            ## get target unit pre-intervention measurements
            #y1 = np.array(pre_df[pre_df.unit == unit].drop(columns=["intervention", "unit","metric"]))
            #y1 = y1.reshape((M*T0))
            #
            # get donor unit post-intervention measurements for intervention "inter"
            #X2_df = post_df[post_df['unit'].isin(unit_ids)]
            #X2 = np.array(X2_df.drop(columns=["intervention", "unit","metric"])).reshape(n_i,M*(T-T0))
            #
            ## get donor unit pre-interventon measurements
            #X1_df = pre_df[pre_df["unit"].isin(unit_ids)]
            #X1 = np.array(X1_df.drop(columns=["intervention", "unit","metric"])).reshape(n_i,M*T0)
            #beta = pcr_full_matrix_denoise(X1, X2, y1, rank=rank, full_matrix_denoise=full_matrix_denoise) 
            
            #with multiple metrics
            #
            #get target unit pre-intervention measurements
            y1 = np.array(pre_df[pre_df.unit == unit].drop(columns=["intervention", "unit","metric"]))
            y1 = y1.reshape((T0,M))
            
            # get donor unit pre-interventon measurements
            X1_df = pre_df[pre_df["unit"].isin(unit_ids)]
            X1 = np.array(X1_df.drop(columns=["intervention", "unit","metric"])).reshape(n_i,T0,M)
            beta = pcr_normalised_metrics(X1, y1, rank=rank)
            
            beta_dict[(inter, unit)] = beta
            
            # forecast counterfactual
            X2_df = post_df[post_df['unit'].isin(unit_ids)]
            X2 = np.array(X2_df.drop(columns=["intervention", "unit","metric"])).reshape(n_i,M*(T-T0))
            out_data[n*I*M+i*M:n*I*M+(i+1)*M] = ((X2.T).dot(beta).T).reshape((M,T-T0))      
        
    out = pd.DataFrame(data=out_data, columns=post_df.drop(columns=["intervention", "unit","metric"]).columns)
    out_units = [units[k // (I*M)] for k in range(N * I * M)]
    out_interventions = [interventions[k % I] for k in range(N * I) for _ in range(M)]
    out_metrics = metrics*(I*N)
    out.insert(0, "metric", out_metrics)
    out.insert(0, "intervention", out_interventions)
    out.insert(0, "unit", out_units)

    return out, beta_dict
        