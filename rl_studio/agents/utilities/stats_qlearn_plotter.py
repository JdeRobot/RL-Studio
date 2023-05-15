import glob
import os
import sys

import argparse
import random
import time
import numpy as np
import pandas as pd


def main():
    argparser = argparse.ArgumentParser(description="Plot Q-Learn Carla Stats")
    argparser.add_argument(
        "-f",
        "--generalfile",
        metavar="G",
        default="127.0.0.1",
        help="IP of the host server (default: 127.0.0.1)",
    )
    argparser.add_argument(
        "-p",
        "--port",
        metavar="P",
        default=2000,
        type=int,
        help="TCP port to listen to (default: 2000)",
    )
    argparser.add_argument(
        "--sync", action="store_true", help="Synchronous mode execution"
    )
    argparser.add_argument(
        "--async", dest="sync", action="store_false", help="Asynchronous mode execution"
    )

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split("x")]

    try:
        run_simulation(args, client)

    except KeyboardInterrupt:
        print("\nCancelled by user. Bye!")


if __name__ == "__main__":
    main()
