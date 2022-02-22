import matplotlib.pyplot as plt
import pandas as pd


def update_line(axes, runs_rewards):
    plot_rewards_per_run(axes, runs_rewards)
    plt.draw()
    plt.pause(0.01)

def get_stats_figure(runs_rewards):
    fig, axes = plt.subplots()
    fig.set_size_inches(12, 4)
    plot_rewards_per_run(axes, runs_rewards)
    plt.ion()
    plt.show()
    return fig, axes

def plot_rewards_per_run(axes, runs_rewards):
    rewards_graph=pd.DataFrame(runs_rewards)
    ax=rewards_graph.plot(ax=axes, title="steps per run");
    ax.set_xlabel("runs")
    ax.set_ylabel("steps")
    ax.legend().set_visible(False)
