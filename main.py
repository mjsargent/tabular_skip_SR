
import utils
import numpy as np
import matplotlib.pyplot as plt


#env_names: four_rooms, open_field_10, open_field_50
def main():
    max_skip = 7
    env_name = "open_field_50"
    Ts, env = utils.make_transition_functions(env_name)
    dims = utils.get_dims(env)
    utils.plot_env(env)

    M = utils.compute_random_walk_SR(Ts)
    lam_M, V_M = utils.compute_eigenvectors(M)
    utils.plot_eigenvectors(env, V_M, idx =np.arange(32), title = "vanilla_" + env_name)

    M_pi_J = utils.compute_random_walk_temporal_SR(Ts, M, dims,  j = max_skip)
    for sk, M_j in enumerate(M_pi_J):
        print(M_j)
        lam_M_j, V_M_j = utils.compute_eigenvectors(M_j)

        utils.plot_eigenvectors(env, V_M_j, idx = np.arange(32),  title = f"Temporal skip {sk + 1}_" + env_name, save = True, filename = f"skip_{sk}.png", savepath = f"./results/{env_name}/")


if __name__ == '__main__':

    main()
