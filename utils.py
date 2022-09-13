import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import pprint
matplotlib.use('TkAgg')
matplotlib.rc('image', cmap='cividis')
from mpl_toolkits.mplot3d import Axes3D

import math
import os
import time
from copy import deepcopy

# scipy and scikit learn imports
from sklearn.decomposition import NMF

# tensor related imports
import tensorly as tl
from tensorly.decomposition import non_negative_parafac, non_negative_parafac_hals, non_negative_tucker, non_negative_tucker_hals
from tensorly.decomposition import parafac, tucker
from tensorly.decomposition._nn_cp import initialize_nn_cp
from tensorly.cp_tensor import CPTensor

# set the colour of our walls by setting the
# colour nans are plotted as
current_cmap = matplotlib.cm.get_cmap()
current_cmap.set_bad(color='grey')
actions = ["UP", "RIGHT", "DOWN", "LEFT"]
action_dict = {action : i for i, action in enumerate(actions)}
number_dict = {i : action for i, action in enumerate(actions)}
action_effect = ((-1,0), (0, 1), (1, 0), (0, -1)) # up right down left

four_rooms_map_list = ["oooooxooooo",
                        "oooooxooooo",
                        "ooooooooooo",
                        "oooooxooooo",
                        "oooooxooooo",
                        "xxoxxxooooo",
                        "oooooxxxoxx",
                        "oooooxooooo",
                        "ooooooooooo",
                        "oooooxooooo",
                        "oooooxooooo"]

junction_hard_map_list = ["ooxooooooo",
                          "ooxooooooo",
                          "ooxxxxxxoo",
                          "ooxxxxxxoo",
                          "oooooooooo",
                          "oooooooooo",
                          "ooxxxxxxoo",
                          "ooxxxxxxoo",
                          "ooxooooooo"]

four_way_junction_map_list = ["xxxxoxxxx",
                              "xxxxoxxxx",
                              "xxxxoxxxx",
                              "xxxxoxxxx",
                              "ooooooooo",
                              "xxxxoxxxx",
                              "xxxxoxxxx",
                              "xxxxoxxxx",
                              "xxxxoxxxx"]

open_field_10x10_map_list = ["o"*10 for _ in range(10)]
open_field_50x50_map_list = ["o"*50 for _ in range(50)]
# env generation

def make_transition_functions(env_name: str = "four_rooms"):
    """
    helper function that parses str input
    and calls the gen_t function
    returns a dict of arrays
    """
    if env_name == "four_rooms":
        env = four_rooms_map_list
    elif env_name == "junction_hard":
        env = junction_hard_map_list
    elif env_name == "open_field_10":
        env = open_field_10x10_map_list
    elif env_name == "open_field_50":
        env = open_field_50x50_map_list
    elif env_name == "four_way_junction":
        env = four_way_junction_map_list

    trans_function_dict = gen_t(env)
    return trans_function_dict, env

def gen_t(env: list):
    """
    function takes in a list of strings and returns
    a dict of arrays (transition functions) that with
    actions (ints) as keys
    """
    n_states = sum(row.count("o") for row in env)
    transition_functions_dict = dict()

    # assume fixed rectangular envs -
    # irregular shapes will need to have x padding the boundaries
    row_n = len(env[0])
    col_n = len(env)

    # create dict of unique identifers
    state_id = dict()
    id_counter = 0
    for i, row in enumerate(env):
        for j, state in enumerate(row):
            if state == "x":
                state_id[(i,j)] = np.nan
            else:
                state_id[(i,j)] = id_counter
                id_counter+= 1

    # quick dirty looping
    for action in actions:
        T = np.zeros([n_states, n_states])
        for i, row in enumerate(env):
            for j, state in enumerate(row):
                # check if state is an x
                if state == "x":
                    continue

                # check the i boundary
                i_next = i + action_effect[action_dict[action]][0]
                if i_next < 0 or i_next >= col_n:
                    i_next = i
                elif env[i_next][j] == "x":
                    i_next = i

                # check the j boundary
                j_next = j + action_effect[action_dict[action]][1]
                if j_next < 0 or j_next >= row_n:
                    j_next = j

                elif row[j_next] == "x":
                    j_next = j

                start_id = state_id[(i,j)]
                next_id = state_id[(i_next, j_next)]

                T[start_id][next_id] = 1
        transition_functions_dict[action] = T

    return transition_functions_dict

def get_dims(env: list):
    """
    returns the number of rows, columns,
    occupiable states and total number of elements (states plus walls)
    """

    row_n = len(env[0])
    col_n = len(env)
    elements_n = row_n * col_n
    states_n = sum(row.count("o") for row in env)
    return {"rows": row_n, "columns": col_n, "states": states_n,
            "elements":elements_n }


# plotting

def plot_env(env: list):
    """
    function for plotting transition functions that respect
    the walls of an environment
    """

    # we could be smart by labelling the states and matching them up
    # to the original env, but the way we have things set up we
    # may as well loop through
    state_counter = 0

    row_n = len(env[0])
    col_n = len(env)

    x_o_dict = {"x": 1, "o": 0}
    im = np.array([x_o_dict[state] for row in env for state in row])

    plt.title("Environment")
    plt.imshow(im.reshape((row_n, col_n)))
    plt.show()

def plot_SR(M: np.array, title: str = "Successor Representation", show = True, save = False):
    """
    plot the total successor representation
    """
    savepath = "RLDM"

    if show:
        plt.imshow(M)
     #   plt.title(title)
        if save:

            if not os.path.isdir(savepath):
                os.makedirs(savepath)
            plt.savefig(savepath + "_SR" + ".png", transparent = True)
        plt.show()
    else:
        return M

def plot_SR_row(M: np.array, env:list,  s: int, dims: dict, title: str = "Successor Representation Column", show = False, save = False, savepath = "./results"):
    """
    plot a given column of the successor representation
    """

    row_n = dims["rows"]
    col_n = dims["columns"]

    x_o_dict = {"x": 1, "o": 0}

    count = 0
    x_count = 0
    im = np.zeros(dims["elements"])
    for row in env:
        for state in row:
            if state == "x":
                im[count] = np.nan
                x_count +=1
            else:
                im[count] = M[s, count - x_count]
            count+= 1

    plt.imshow(im.reshape(dims['rows'], dims['columns']))
    plt.title(title, fontsize = 20)
    plt.axis('off')
    if save:
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        filename = title
        plt.savefig(savepath + title + ".png", transparent = True)
    if show:
        #title = title + f"_{s}"
        plt.show()


    else:
        return im.reshape(dims['rows'], dims['columns'])


def plot_SR_column(M: np.array, env:list,  s: int, dims: dict, title: str = "Successor Representation Column", show = False, save = False, savepath = "./results"):
    """
    plot a given column of the successor representation
    """

    row_n = dims["rows"]
    col_n = dims["columns"]

    x_o_dict = {"x": 1, "o": 0}

    count = 0
    x_count = 0
    im = np.zeros(dims["elements"])
    for row in env:
        for state in row:
            if state == "x":
                im[count] = np.nan
                x_count +=1
            else:
                im[count] = M[count - x_count, s]
            count+= 1

    plt.imshow(im.reshape(dims['rows'], dims['columns']))
    plt.title(title, fontsize = 20)
    plt.axis('off')
    if save:
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        filename = title
        plt.savefig(savepath + title + ".png", transparent = True)
    if show:
        #title = title + f"_{s}"
        plt.show()


    else:
        return im.reshape(dims['rows'], dims['columns'])

def plot_SR_columns_superimposed(M: np.array, env:list,  s: list, dims: dict, title: str = "Successor Representation Column", show = False, save = False, savepath = "./results"):
    """
    plot two columns of the successor representation superimposed over an env
    """

    row_n = dims["rows"]
    col_n = dims["columns"]

    x_o_dict = {"x": 1, "o": 0}

    count = 0
    x_count = 0
    im = np.zeros(dims["elements"])
    for row in env:
        for state in row:
            if state == "x":
                im[count] = np.nan
                x_count +=1
            else:
                #im[count] = M[count - x_count, s[0]] + M[count - x_count, s[1]]
                im[count] = sum([M[count - x_count, j] for j in s])
            count+= 1

    plt.imshow(im.reshape(dims['rows'], dims['columns']))
    plt.title(title, fontsize = 20)
    plt.axis('off')
    if save:
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        filename = title
        plt.savefig(savepath + title + ".png")
    if show:
        #title = title + f"_{s}"
        plt.show()


    else:
        return im.reshape(dims['rows'], dims['columns'])



def plot_multiple_skip_SRs(M_J_s: np.array, dims: dict):
    total_skips = M_J_s.shape[0]
    for i, M_j in enumerate(M_J_s):
        plt.subplot(2, (total_skips + 1) // 2 , i+1)
        sr_plot = plot_SR(M_j, title = f"skip {i + 1}", show = False)
        plt.imshow(sr_plot)
        plt.title(f"skip {i+1}")
    plt.suptitle("Successor Representations over skips")
    plt.show()


def plot_multiple_skip_SRs_columns(M_J_s: np.array, env:list, s: int, dims: dict ):
    total_skips = M_J_s.shape[0]
    for i, M_j in enumerate(M_J_s):
        plt.subplot(2, (total_skips + 1) // 2 , i+1)
        sr_plot = plot_SR_column(M_j,env, s, dims = dims, title = f"skip {i + 1}", show = False)
        plt.imshow(sr_plot)
        plt.title(f"skip {i+1}")
    plt.suptitle(f"Successor Representations column {s} over skips")
    plt.show()

def plot_eigenvector(env:list, v: np.array, i:int = 0, show = True):
    """
    plot a given eigenvector distributed across
    an environment
    """
    row_n = len(env[0])
    col_n = len(env)
    elements_n = row_n * col_n

    im = np.full([row_n * col_n], np.nan)

    # make a flat version of the env
    env_unravel = ""
    for row in env:
        env_unravel = env_unravel + row

    v_i = v[:, i]

    im_count = 0
    for a in v_i:
        while env_unravel[im_count] == "x":
            im_count += 1
            if im_count > elements_n:
                break

        im[im_count] = a
        im_count += 1

    im = im.reshape([row_n, col_n])
    if show:
        plt.imshow(im)
        plt.show()
    else:
        return im

def plot_eigenvector_3D(env:list, v: np.array, i:int = 0, show = True):
    """
    plot a given eigenvector distributed across
    an environment
    """
    row_n = len(env[0])
    col_n = len(env)
    elements_n = row_n * col_n

    im = np.full([row_n * col_n], np.nan)

    # make a flat version of the env
    env_unravel = ""
    for row in env:
        env_unravel = env_unravel + row

    v_i = v[:, i]

    im_count = 0
    for a in v_i:
        while env_unravel[im_count] == "x":
            im_count += 1
            if im_count > elements_n:
                break

        im[im_count] = a
        im_count += 1

    im = im.reshape([row_n, col_n])

    fig = plt.figure()
    fig_a = fig.add_subplot(111, projection="3d")

    X, Y = np.meshgrid(np.arange(row_n), np.arange(col_n))
    this_cmap = plt.get_cmap("cividis")
    fig_a.plot_surface(X, Y, np.nan_to_num(im, nan=0), cmap = this_cmap)
    if show:
        plt.show()
    else:
        return fig_a

def plot_eigenvectors(env: list, v: np.array, title:str = "", idx: list = [0], show = True, save = False, filename:str = "results.png", savepath: str = "./results"):
    eig_n = len(idx)
    row_n = math.ceil(eig_n / 8)
    for i in idx:
        plt.subplot(row_n, 8, i+1)
        eig_plot = plot_eigenvector(env, v, i, show=False)
        plt.imshow(eig_plot)
    plt.subplots_adjust(wspace=0.01,hspace=0.01)
    plt.suptitle(title,fontsize=20)
    if save:
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        plt.savefig(savepath + filename)
    if show:
        plt.show()


# SR calculation

def compute_random_walk_SR(T: dict, gamma: float = 0.99, show  = False):
    # random walk therefore uniform policy

    # preserving the action effect doesn't matter here
    # so we can break this into a list
    T_stack = np.stack([T_a for T_a in T.values()])
    T_pi = 0.25 * T_stack.sum(axis=0)
    T_pi = T_pi / T_pi.sum(axis = 1).reshape(-1,1)

    # Neumann sum
    M = np.linalg.inv(np.eye(T_pi.shape[0]) - gamma * T_pi)
    if show:
        plt.imshow(M)
        plt.show()
    return M

def compute_random_walk_temporal_SR(T: dict, M_baseline: np.array, dims: dict, j: int = 2, gamma: float = 0.99, exp_scale = False):
    # when computing the temporally extended SR
    # the effect of the action taken for j steps needs to be considered
    n_states = dims["states"]
    M_skip = np.zeros([j, n_states, n_states]) # for random skip we don't need to store the action

    T_stack = np.stack([T_a for T_a in T.values()])
    T_pi = 0.25 * T_stack.sum(axis=0)
    T_pi = T_pi / T_pi.sum(axis = 1).reshape(-1,1)

    if exp_scale:
        skip_range = np.arange(0, j)
        skip_range = np.power(2, skip_range)

    else:
        skip_range = np.arange(0, j)

    for i, skip in enumerate(skip_range): # note this starts at zero; idx 1 is skip 2

        M_store = []
        for a, T_a in enumerate(T.values()):
            I = np.eye(n_states)
            for n in range(0, skip + 1):
                I+= gamma ** n * np.linalg.matrix_power(T_a, n)
            #I+= np.matmul((gamma ** (skip + 1) * np.linalg.matrix_power(T_pi, skip + 1)), M_baseline)

            M_store.append(I)

        T_sum_j = 0.25 * np.sum(np.stack(M_store), axis = 0)
        T_last_j = np.zeros([n_states, n_states])
        for T_a in T.values():
            T_last_j += np.linalg.matrix_power(T_a, skip + 1)
        T_last_j = 0.25 * gamma ** (skip + 1) * T_last_j
        M_j = T_sum_j + np.matmul(T_last_j, M_baseline)

        M_skip[i, :, :] = M_j
    return M_skip

def compute_macro_action_temporal_SR(T: dict, M_baseline: np.array, dims: dict,
                                     action_seq: list = [0, 0, 1, 1], gamma: float = 0.99):
    """
    function for compute the temporal SR under a defined macroaction:
    the macroaction is defined as taking the actions in action sequence
    """
    n_states = dims["states"]

    I = np.eye(n_states)
    for i, action in enumerate(action_seq):
        if i == 0:
            M_current = gamma * T[number_dict[action]]
            I+= M_current
        else:
            #M_current  = np.matmul(gamma ** ( i+1) * T[number_dict[action]], M_current)
            M_current  = np.matmul(M_current,gamma ** ( i+1) * T[number_dict[action]])
            I += M_current


    #M_macro = I +  gamma * np.matmul(M_current, M_baseline)
    M_macro = I +  gamma * np.matmul(M_baseline, M_current)

    return M_macro

def compute_eigenvectors(M: np.array, take_real = False):
    lam, v = np.linalg.eig(M)
    if take_real:
        v = np.real(v)
    return lam, v

def plot_eigenvalue_distribution(lam: np.array):
    spec_radius = np.max(np.abs(lam))
    print("Spectral Radius", spec_radius)
    plt.hist(np.real(lam), bins = "auto")
    plt.show()


##### Non negative matrix factorisation

def compute_NMF(M: np.array):
    nmf_model = NMF(n_components = None, init='nndsvda', random_state=0, \
                max_iter = 5000, solver="mu", alpha_W = 0.01)
    W = nmf_model.fit_transform(M)
    H = nmf_model.components_

    return W, H

##### Tensor Decomposition

def compute_CP_decomp(tM: np.array, dims: dict, non_negative = False):

    n_states = dims["states"]
    tM_tensor = tl.tensor(tM)

    weights_init, factors_init = initialize_nn_cp(tM_tensor, init='random', rank=n_states)
    cp_init = CPTensor((weights_init, factors_init))
    if non_negative:
        tensor, errors_hals = non_negative_parafac_hals(tM_tensor, rank=n_states, init=deepcopy(cp_init), return_errors=True)
    else:
        tensor, errors_hals = parafac(tM_tensor, rank=n_states, init=deepcopy(cp_init), return_errors=True)

    return tensor

def compute_tucker_decomp(tM: np.array, dims:dict, non_negative = False):
    n_states = dims["states"]
    tM_tensor = tl.tensor(np.stack(tM))
    if non_negative:
        tensor_mu, error_mu = non_negative_tucker(tM_tensor, rank=[1,tM.shape[1], tM.shape[2]], tol=1e-12, n_iter_max=1000, return_errors=True)
    else:
        tensor_mu = tucker(tM_tensor, rank=tM.shape, tol=1e-15, n_iter_max=10000)
    return tensor_mu

##### Other misc utils

def print_macro_actions(results: list):
    # up, right, down, left
    unicode_arrow_map = ["\u2191", "\u2192","\u2193", "\u2190"]

    if isinstance(results, dict):
        # results is a list of arrays of q values
        print("Action Sequences")
        keys = np.arange(0, len(results.keys()))
        seq_string = u""
        for k in keys:
            q = results[k]
            act = int(np.argmax(q))
            seq_string+= unicode_arrow_map[act] + " "
        print(seq_string)

        # results is a explict sequence
    else:
        print("Loss ::: Action Sequence")
        for r in results:
            seq_string = u""
            for act in r[1]:
                seq_string+= unicode_arrow_map[act] + " "
            print(np.round(r[0], 3), seq_string)

