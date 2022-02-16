
import argparse
import os
import utils
import numpy as np
import matplotlib.pyplot as plt



#env_names: four_rooms, open_field_10, open_field_50
def main():
    #argparse
    parser = argparse.ArgumentParser("tabular skip sr")
    parser.add_argument("--env", type = str, default = "open_field_10",
                        choices = ["open_field_10", "open_field_50", "four_rooms"])
    parser.add_argument("--max_skip", type = int, default = 7)
    parser.add_argument("--gamma", type = float, default = 0.99)
    parser.add_argument("--save_figures", type = lambda x: bool(strtobool(x)), default = True)
    parser.add_argument("--save_dir", type = str, default = "./results")
    parser.add_argument("--show_figures", type = lambda x: bool(strtobool(x)), default = False)
    args = parser.parse_args()

    # process save dir
    save_dir = os.path.join(args.save_dir, args.env)

    # store other args
    max_skip = args.max_skip
    env_name = args.env
    gamma = args.gamma
    save = args.save_figures
    show = args.show_figures

    Ts, env = utils.make_transition_functions(env_name)
    dims = utils.get_dims(env)
    utils.plot_env(env)

    M = utils.compute_random_walk_SR(Ts, gamma = gamma)
    M_pi_J = utils.compute_random_walk_temporal_SR(Ts, M, dims,  j = max_skip, gamma = gamma)

    # plot SRs
    utils.plot_SR(M)
    utils.plot_SR_column(M, s = 10, dims = dims)

    utils.plot_multiple_skip_SRs(M_pi_J, dims = dims)
    utils.plot_multiple_skip_SRs_columns(M_pi_J, s = 25, dims = dims)

    # plot eigenvectors
    lam_M, V_M = utils.compute_eigenvectors(M)
    utils.plot_eigenvectors(env, V_M, idx =np.arange(32), title = "vanilla_" + env_name, show = show, save = save)

    for sk, M_j in enumerate(M_pi_J):
        lam_M_j, V_M_j = utils.compute_eigenvectors(M_j)

        utils.plot_eigenvectors(env, V_M_j, idx = np.arange(32),  title = f"Temporal skip {sk + 1}_" + env_name, save = save, filename = f"skip_{sk}.png", savepath = f"./results/{env_name}/")
    #utils.plot_SR(M_j, title = f"SR skip {sk}")
    #utils.plot_SR_column(M, s = 10, dims = dims, title = f"SR Skip {sk} col ")

if __name__ == '__main__':

    main()
