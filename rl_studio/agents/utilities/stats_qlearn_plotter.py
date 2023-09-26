import argparse
import glob
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_ie_metrics(file):
    df = pd.read_excel(file, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(df)

    ### creating in one canvas
    fig, ax = plt.subplots(2, 2, figsize=(20, 12))
    ax[0, 0].plot(
        df["episode"],
        df["cumulated_reward"],
        color="blue",
        linewidth=2,
        linestyle="-",
        label="rewards",
    )  # Plot some data on the (implicit) axes.
    ax[0, 0].plot(
        df["episode"],
        df["step"],
        color="orange",
        linewidth=2,
        linestyle="--",
        label="steps",
    )  # Plot some data on the (implicit) axes.
    ax[0, 0].set_xlabel("episodes")
    ax[0, 0].set_ylabel("value")
    ax[0, 0].set_title("Rewards/steps per epoch")
    ax[0, 0].legend()

    ax[0, 1].set_yscale("log")
    ax[0, 1].plot(
        df["episode"],
        df["epsilon"],
        color="green",
        linewidth=1,
        linestyle="-",
        label="epsilon",
    )  # Plot some data on the (implicit) axes.
    ax[0, 1].set_xlabel("episodes")
    ax[0, 1].set_ylabel("epsilon value [0 - 0.99]")
    ax[0, 1].set_title("epsilon per epoch")

    ax[1, 0].plot(
        df["episode"],
        df["distance_to_finish"],
        color="red",
        linewidth=2,
        linestyle="-",
        label="distance",
    )  # Plot some data on the (implicit)axes.
    ax[1, 0].set_xlabel("episodes")
    ax[1, 0].set_ylabel("distance")
    ax[1, 0].set_title("Distance to Finish circuit")

    ax[1, 1].plot(
        df["episode"],
        df["epoch_time"],
        color="black",
        linewidth=2,
        linestyle="-",
        label="time",
    )  # Plot some data on the (implicit) axes.
    ax[1, 1].set_xlabel("episodes")
    ax[1, 1].set_ylabel("time")
    ax[1, 1].set_title("time in every epoch")

    # saving
    file_saved = f"{time.strftime('%Y%m%d-%H%M%S')}_ie_metrics"
    plt.savefig(f"{file_saved}.png", dpi=600)
    plt.savefig(f"{file_saved}.jpg", dpi=600)

    ### individuals #####
    # a, b = np.polyfit(df["episode"], df["cumulated_reward"], deg=1)
    a, b, c = np.polyfit(df["episode"], df["cumulated_reward"], deg=2)
    # a, b, c, d = np.polyfit(df["episode"], df["cumulated_reward"], deg=3)
    # y_est = a * df["episode"] + b
    y_est = a * np.square(df["episode"]) + b * df["episode"] + c
    # y_est = (
    #    a * (df["episode"] ** 3) + b * np.square(df["episode"]) + c * df["episode"] + d
    # )
    y_err = df["episode"].std() * np.sqrt(
        1 / len(df["episode"])
        + (df["episode"] - df["episode"].mean()) ** 2
        / np.sum((df["episode"] - df["episode"].mean()) ** 2)
    )

    fig1, axs1 = plt.subplots(1, 1, sharey=True, tight_layout=True)
    axs1.plot(df["episode"], y_est, color="red", linestyle="-")
    axs1.fill_between(df["episode"], y_est - y_err, y_est + y_err, alpha=0.2)

    axs1.plot(
        df["episode"],
        df["cumulated_reward"],
        color="blue",
        linewidth=2,
        linestyle="-",
        label="rewards",
    )  # Plot some data on the (implicit) axes.
    axs1.plot(
        df["episode"],
        df["step"],
        color="orange",
        linewidth=2,
        linestyle="--",
        label="steps",
    )
    # Plot some data on the (implicit) axes.
    axs1.set_xlabel("episodes")
    axs1.set_ylabel("value")
    axs1.set_title("Rewards/steps per epoch")
    axs1.legend()

    file_rewards = f"{time.strftime('%Y%m%d-%H%M%S')}_rewards_steps"
    plt.savefig(f"{file_rewards}.png", dpi=600)
    plt.savefig(f"{file_rewards}.jpg", dpi=600)

    # epsilon
    fig2, axs2 = plt.subplots(1, 1, sharey=True, tight_layout=True)
    axs2.set_yscale("log")
    axs2.plot(
        df["episode"],
        df["epsilon"],
        color="green",
        linewidth=1,
        linestyle="-",
        label="epsilon",
    )  # Plot some data on the (implicit) axes.
    axs2.set_xlabel("episodes")
    axs2.set_ylabel("epsilon value [0 - 0.99]")
    axs2.set_title("epsilon per epoch")

    file_epsilon = f"{time.strftime('%Y%m%d-%H%M%S')}_epsilon"
    plt.savefig(f"{file_epsilon}.png", dpi=600)
    plt.savefig(f"{file_epsilon}.jpg", dpi=600)

    # distance to finish
    fig3, axs3 = plt.subplots(1, 1, sharey=True, tight_layout=True)
    axs3.plot(
        df["episode"],
        df["distance_to_finish"],
        color="red",
        linewidth=2,
        linestyle="-",
        label="distance",
    )  # Plot some data on the (implicit)axes.
    axs3.set_xlabel("episodes")
    axs3.set_ylabel("distance")
    axs3.set_title("Distance to Finish circuit")

    file_dist = f"{time.strftime('%Y%m%d-%H%M%S')}_distance"
    plt.savefig(f"{file_dist}.png", dpi=600)
    plt.savefig(f"{file_dist}.jpg", dpi=600)

    # epoch time
    fig4, axs4 = plt.subplots(1, 1, sharey=True, tight_layout=True)
    axs4.plot(
        df["episode"],
        df["epoch_time"],
        color="black",
        linewidth=2,
        linestyle="-",
        label="epoch time",
    )  # Plot some data on the (implicit) axes.
    axs4.set_xlabel("episodes")
    axs4.set_ylabel("time")
    axs4.set_title("time in every epoch")

    file_time = f"{time.strftime('%Y%m%d-%H%M%S')}_time"
    plt.savefig(f"{file_time}.png", dpi=600)
    plt.savefig(f"{file_time}.jpg", dpi=600)

    # lane changed
    fig5, axs5 = plt.subplots(1, 1, sharey=True, tight_layout=True)
    axs5.plot(
        df["episode"],
        df["lane_changed"],
        color="green",
        linewidth=2,
        linestyle="-",
        label="lane changed",
    )  # Plot some data on the (implicit) axes.
    axs5.set_xlabel("episodes")
    axs5.set_ylabel("lane changed")
    axs5.set_title("Lane changed in every epoch")

    file_lane_changed = f"{time.strftime('%Y%m%d-%H%M%S')}_lane_changed"
    plt.savefig(f"{file_lane_changed}.png", dpi=600)
    plt.savefig(f"{file_lane_changed}.jpg", dpi=600)


def plot_sac_metrics(file):
    df_sac = pd.read_excel(file, usecols=[1, 2, 3, 4])
    print(df_sac)

    ### STATES
    num_states = 5
    list_states = [
        df_sac[df_sac["state"] == i][["counter"]].sum() for i in range(0, num_states)
    ]

    states_df = pd.DataFrame(list_states, columns=["state", "counter"])
    # print(states_df)
    states_df["state"] = states_df.index
    # print(states_df)

    fig1, axs1 = plt.subplots(1, 1, figsize=(10, 6), sharey=True, tight_layout=True)
    axs1.bar(states_df["state"], states_df["counter"], color="dodgerblue")
    axs1.set_xlabel("states")
    axs1.set_ylabel("frequency")
    axs1.set_title("Histogram of States")
    axs1.set_xticks(states_df["state"])

    file_states = f"{time.strftime('%Y%m%d-%H%M%S')}_histogram_states"
    plt.savefig(f"{file_states}.png", dpi=600)
    plt.savefig(f"{file_states}.jpg", dpi=600)

    #### ACTIONS
    num_actions = 5
    list_actions = [
        df_sac[df_sac["action"] == i][["counter"]].sum() for i in range(0, num_actions)
    ]

    actions_df = pd.DataFrame(list_actions, columns=["action", "counter"])
    # print(states_df)
    actions_df["action"] = actions_df.index
    # print(actions_df)

    fig2, axs2 = plt.subplots(1, 1, sharey=True, tight_layout=True)
    axs2.bar(
        actions_df["action"],
        actions_df["counter"],
        tick_label=[
            "0",
            "1",
            "2",
            "3",
            "4",
        ],  # change in function of line 228, var num_actions
        color="teal",
    )
    axs2.set_xlabel("actions")
    axs2.set_ylabel("frequency")
    axs2.set_title("Histogram of Actions")
    axs2.set_xticks(actions_df["action"])

    file_actions = f"{time.strftime('%Y%m%d-%H%M%S')}_histogram_actions"
    plt.savefig(f"{file_actions}.png", dpi=600)
    plt.savefig(f"{file_actions}.jpg", dpi=600)


def plot_qtable(file):
    df_qtable = pd.read_excel(file, usecols=[1, 2, 3])
    # df_sac = pd.read_excel("asac.xlsx")
    print(df_qtable)
    dataf_size = df_qtable.shape[0]

    actions = [0, 1, 2]
    states = [i for i in range(0, 8)]
    z_pos = np.where(df_qtable["q_value"] >= 0, df_qtable["q_value"], 0)
    z_neg = np.where(df_qtable["q_value"] < 0, df_qtable["q_value"], 0)
    fig = plt.figure(figsize=(10, 10))

    ax1 = fig.add_subplot(111, projection="3d")
    # ax1.bar3d(df_qtable['state'], df_qtable['action'], np.zeros(dataf_size), np.ones(dataf_size), np.ones(dataf_size), df_qtable['q_value'])
    ax1.bar3d(df_qtable["state"], df_qtable["action"], 0.1, 0.1, 0.1, z_pos)
    ax1.bar3d(df_qtable["state"], df_qtable["action"], 0.1, 0.1, 0.1, z_neg)
    # for y in enumerate(actions):
    #  ax1.bar(df_qtable['state'], df_qtable['q_value'], zs=y, zdir='y')

    ax1.set_xlabel("states")
    ax1.set_ylabel("actions")
    ax1.set_zlabel("value")
    ax1.set_title("Q-Table Values", fontsize=14, fontweight="bold")

    ax1.set_yticks(actions[::-1])
    ax1.set_xticks(states)

    file_qtable = f"{time.strftime('%Y%m%d-%H%M%S')}_qtable"
    plt.savefig(f"{file_qtable}.png", dpi=600)
    plt.savefig(f"{file_qtable}.jpg", dpi=600)


##############################################################


def generate_plotter(args):
    # print(f"{args=}")
    if args.type == "ie_metrics":
        # df = pd.read_excel(args.file)
        # print(f"{df=}")
        """
        ie_metrics means for intrinsic and extrinsic metrics
        """
        plot_ie_metrics(args.file)

    elif args.type == "sac":
        print("SAC")
        """sac stands for states and actions counter"""
        plot_sac_metrics(args.file)
    else:
        print("q_table")
        """q-table"""
        plot_qtable(args.file)


def main():
    argparser = argparse.ArgumentParser(description="Plot Q-Learn Carla Stats")
    argparser.add_argument(
        "-t",
        "--type",
        required=True,
        help="type of data: q-table, sac, ie_metrics",
    )
    argparser.add_argument(
        "-f",
        "--file",
        required=True,
        help="load data file",
    )

    args = argparser.parse_args()
    # args2 = vars(argparser.parse_args())

    print(f"{args=}")
    # print(f"{args2=}")

    try:
        generate_plotter(args)

    except KeyboardInterrupt:
        print("\nCancelled by user. Bye!")


if __name__ == "__main__":
    """
    How to use it:

    options:
    -t, --type: you have 3 options, q-table, sac, ie_metrics
    -f, --file: depending on -t option, you have to choose the correct file

    """

    main()
