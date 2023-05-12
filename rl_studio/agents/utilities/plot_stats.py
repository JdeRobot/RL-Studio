import os
import time

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


class StatsDataFrame:
    def save_dataframe_stats(self, environment, outdir, stats, q_table=False):
        os.makedirs(f"{outdir}", exist_ok=True)

        if q_table:
            file_csv = f"{outdir}/{time.strftime('%Y%m%d-%H%M%S')}_QTABLE_Circuit-{environment['town']}_States-{environment['states']}_Actions-{environment['action_space']}_Rewards-{environment['reward_function']}.csv"
            file_excel = f"{outdir}/{time.strftime('%Y%m%d')}_QTABLE_Circuit-{environment['town']}_States-{environment['states']}_Actions-{environment['action_space']}_Rewards-{environment['reward_function']}.xlsx"
        else:
            file_csv = f"{outdir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{environment['town']}_States-{environment['states']}_Actions-{environment['action_space']}_Rewards-{environment['reward_function']}.csv"
            file_excel = f"{outdir}/{time.strftime('%Y%m%d')}_Circuit-{environment['town']}_States-{environment['states']}_Actions-{environment['action_space']}_Rewards-{environment['reward_function']}.xlsx"

        df = pd.DataFrame(stats)
        df.to_csv(file_csv, mode="a", index=False, header=None)

        # with pd.ExcelWriter(file_excel, mode="a") as writer:
        #    df.to_excel(writer)
        df.to_excel(file_excel)


class MetricsPlot:
    def __init__(self, outdir, line_color="blue", **args):
        """
        renders graphs of intrinsic and extrinsic metrics
        Args:

        """
        self.outdir = outdir
        self.line_color = line_color

        # styling options
        matplotlib.rcParams["toolbar"] = "None"
        plt.style.use("ggplot")
        fig = plt.gcf().canvas.set_window_title("simulation_graph")

    # def plot(self, env):
