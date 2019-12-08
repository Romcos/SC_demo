import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


####### SAMPLING ######

def random_rct(N, I, T, T0, rank=2, sigma=0.1):
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

    pre_data = np.zeros((N, T0))
    post_data = np.zeros((N, T - T0))
    i_rcv = np.random.randint(0, I, size=(N))

    ids = ["id_" + str(i) for i in range(N)]
    times = np.array(["t_" + str(t) for t in range(T)])
    interventions = np.array(["inter_" + str(i) for i in range(I)])

    U = np.random.normal(size=(N, rank)) + 1
    V = np.random.normal(size=(T, rank)) + 1
    F = np.random.normal(size=(I, rank)) + 1

    for n in range(N):
        for t in range(T0):
            pre_data[n, t] = (U[n, :] * V[t, :] * F[i_rcv[0], :]).sum() + sigma * np.random.normal()
        for t in range(T0, T):
            post_data[n, t - T0] = (U[n, :] * V[t, :] * F[i_rcv[n], :]).sum() + sigma * np.random.normal()

    pre_df = pd.DataFrame(data=pre_data, columns=times[:T0])
    pre_df.insert(0, "intervention", [interventions[0]] * N)
    pre_df.insert(0, "unit", ids)

    post_df = pd.DataFrame(data=post_data, columns=times[T0:])
    post_df.insert(0, "intervention", interventions[i_rcv])
    post_df.insert(0, "unit", ids)

    return pre_df, post_df

###### COMPUTATION ######

def hsvt(Z, rank=2):
    u, s, vh = np.linalg.svd(Z, full_matrices=False)
    s[rank:].fill(0)
    return np.dot(u * s, vh)

def cum_energy(s):
    return (100 * (s ** 2).cumsum() / (s ** 2).sum())

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
        X = hsvt(X1, rank=rank)
    _, T = X1.shape
    X_pre = X[:, :T]
    beta = np.linalg.pinv(X_pre.T).dot(y)
    return beta

###### DIAGNOSTIC ######

def diagnostic(post_df, df_output):
    # get all unique interventions (from post-intervention dataframe)
    interventions = np.sort(pd.unique(post_df.intervention))

    # sort all units (using post-intervention dataframe)
    units = np.sort(post_df.unit)

    # get number of units and interventions
    N, I = len(units), len(interventions)

    R2_all_interventions = np.zeros(len(interventions))

    # loop through all interventions
    for i, inter in enumerate(interventions):

        unit_ids = post_df[post_df['intervention'] == inter]['unit']

        R2_intervention_list = []

        for unit in unit_ids:

            y = post_df.loc[(post_df.unit==unit) & (post_df.intervention==inter)].drop(columns=['unit', 'intervention']).values
            y_hat = df_output.loc[(df_output.unit==unit) & (df_output.intervention==inter)].drop(columns=['unit', 'intervention']).values

            baseline_error = (y.mean() - y)**2
            baseline_error_sum = baseline_error.sum()

            estimated_error = (y_hat - y)**2
            estimate_error_sum = estimated_error.sum()

            R2_unit_intervention = 1 - (estimate_error_sum / baseline_error_sum)

            R2_intervention_list.append(R2_unit_intervention)

        R2_intervention_average = np.mean(R2_intervention_list)
        R2_all_interventions[i] = R2_intervention_average

    diag = pd.DataFrame(data= R2_all_interventions, columns = ["Average R^2 Value"])
    diag.insert(0, "intervention", interventions)
    return diag
#             plt.figure()
#             plt.plot(y.flatten(), label='obs')
#             plt.plot(y_hat.flatten(), label='pred')
#             plt.legend(loc='best')
#             plt.show()


###### OUTPUT ##########

def fill_tensor(pre_df, post_df, cum_energy=0.90, full_matrix_denoise=True):
    """
    Gives the counterfactual observation for an unobserved cell

    Input:
        rct_data: tuple (pre_df, post_df)

    Output:
        Counterfactual estimation for all tensor
    """
    # get pre- and post- intervention dataframes
    # pre_df, post_df = rct_data

    # get all unique interventions (from post-intervention dataframe)
    interventions = np.sort(pd.unique(post_df.intervention))

    # sort all units (using pre-intervention data)
    units = np.sort(pre_df.unit)

    # get number of units and interventions
    N, I = len(units), len(interventions)

    # check no duplicate units in pre-intervention dataframe
    assert len(pre_df.unit.unique()) == N

    # initialize output dataframe size
    out_data = np.zeros((N * I, post_df.shape[1] - 2))

    # loop through all interventions
    for i, inter in enumerate(interventions):
        # Keep relevant experiments

        # extract all units that receive intervention "inter" (P_inter)
        unit_ids = post_df[post_df["intervention"] == inter]['unit']

        # get pre-intervention measurements associated with P_inter
        X1_df = pre_df[pre_df['unit'].isin(unit_ids)]  # .drop(columns=["intervention","unit"]))

        # loop through all units (make sure id's unique)
        for n, unit in enumerate(pre_df.unit):
            # get target unit pre-intervention measurements
            y1 = np.array(pre_df[pre_df.unit == unit].drop(columns=["intervention", "unit"]))[0]

            # get donor unit post-intervention measurements for intervention "inter"
            X2_df = post_df[(post_df['unit'].isin(unit_ids)) & (post_df['intervention'] == inter)]

            # check if unit is in donor pool
            if unit in unit_ids:
                X1 = X1_df.loc[(X1_df.unit != unit)].drop(columns=['intervention', 'unit']).values
                X2 = X2_df.loc[(X2_df.unit != unit)].drop(columns=['intervention', 'unit']).values
            else:
                X1 = X1_df.drop(columns=['intervention', 'unit']).values
                X2 = X2_df.drop(columns=['intervention', 'unit']).values

            Xtot = np.concatenate((X1, X2), axis=1)
            _, s_tot, _ = np.linalg.svd(Xtot, full_matrices=False)
            cum_s_tot = (100 * (s_tot ** 2).cumsum() / (s_tot ** 2).sum())
            post_rank = [index for index, singular_value_cum_energy in enumerate(cum_s_tot) if singular_value_cum_energy > 100 * cum_energy ][0] + 1

            # Build linear model
            beta = pcr(X1, X2, y1, rank=post_rank, full_matrix_denoise=full_matrix_denoise)

            # forecast counterfactual
            out_data[n * I + i] = (X2.T).dot(beta)

    out_units = [units[k // I] for k in range(N * I)]
    out_interventions = [interventions[k % I] for k in range(N * I)]
    out = pd.DataFrame(data=out_data, columns=post_df.drop(columns=["intervention", "unit"]).columns)
    out.insert(0, "intervention", out_interventions)
    out.insert(0, "unit", out_units)

    return out
