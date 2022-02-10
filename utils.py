import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import os

# set the colour of our walls by setting the
# colour nans are plotted as
current_cmap = matplotlib.cm.get_cmap()
current_cmap.set_bad(color='grey')

actions = ["UP", "RIGHT", "DOWN", "LEFT"]
action_dict = {action : i for i, action in enumerate(actions)}
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
    elif env_name == "open_field_10":
        env = open_field_10x10_map_list
    elif env_name == "open_field_50":
        env = open_field_50x50_map_list

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

    # quick dirty looping
    x_count = 0
    for action in actions:
        T = np.zeros([n_states, n_states])
        for i, row in enumerate(env):
            for j, state in enumerate(row):
                # check if state is an x
                if state == "x":
                    x_count += 1
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
                T[row_n * i + j - x_count][row_n * i_next + j_next - x_count] = 1
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

    plt.imshow(im.reshape((row_n, col_n)))
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
    plt.tight_layout()
    plt.suptitle(title,fontsize=20)
    if save:
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        plt.savefig(savepath + filename)
    if show:
        plt.show()


# SR calculation

def compute_random_walk_SR(T: dict, gamma: float = 0.99):
    # random walk therefore uniform policy

    # preserving the action effect doesn't matter here
    # so we can break this into a list
    T_stack = np.stack([T_a for T_a in T.values()])
    T_pi = 0.25 * T_stack.sum(axis=0)
    T_pi = T_pi / T_pi.sum(axis = 1).reshape(-1,1)
    plt.imshow(T_pi)
    plt.show()

    # Neumann sum
    M = np.linalg.inv(np.eye(T_pi.shape[0]) - gamma * T_pi)
    return M

def compute_random_walk_temporal_SR(T: dict, M_baseline: np.array, dims: dict, j: int = 2, gamma: float = 0.99):
    # when computing the temporally extended SR
    # the effect of the action taken for j steps needs to be considered
    n_states = dims["states"]
    M_skip = np.zeros([j, n_states, n_states]) # for random skip we don't need to store the action

    for skip in range(j): # note this starts at zero; idx 1 is skip 2

        M_store = []
        for a, T_a in enumerate(T.values()):
            I = np.eye(n_states)

            for n in range(1,skip + 1):
                I+= gamma ** n * np.linalg.matrix_power(T_a, n)
            I+= gamma ** (skip + 1) * M_baseline

            M_store.append(I)

        M_skip[skip, :, :] = 0.25 * np.sum(np.stack(M_store), axis = 0)

    return M_skip

def compute_eigenvectors(M: np.array, take_real = True):
    lam, v = np.linalg.eig(M)
    if take_real:
        v = np.real(v)
    return lam, v

