#!/usr/bin/env python
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from collections import Counter
import glob
import os
import sys

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass

import carla

import argparse


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--host",
        metavar="H",
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
        "-t",
        "--town",
        metavar="T",
        default="Town01",
        help="load Town to work with",
    )
    argparser.add_argument(
        "-o",
        "--only_road",
        default=False,
        metavar="R",
        help="only show roads",
    )
    args = argparser.parse_args()

    # Approximate distance between the waypoints
    WAYPOINT_DISTANCE = 5.0  # in meters

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        # world = client.get_world()
        world = client.load_world(args.town)
        carla_map = world.get_map()

        import matplotlib.pyplot as plt

        # plt.subplot(211)
        # Invert the y axis since we follow UE4 coordinates
        plt.gca().invert_yaxis()
        plt.margins(x=0.7, y=0)

        # GET WAYPOINTS IN THE MAP ##########################################
        # Returns a list of waypoints positioned on the center of the lanes
        # all over the map with an approximate distance between them.
        waypoint_list = carla_map.generate_waypoints(WAYPOINT_DISTANCE)

        print(f"{len(waypoint_list) = }")
        wtown_road = []
        wtown_lane = []
        wtown = []
        y = 1
        for x in waypoint_list:
            #    print(f"{x.transform.location.x = }")
            #    print(f"{x.transform.location.y = }")
            #    print(f"{x.transform.location.z = }")
            #    print(f"{x.transform.rotation.pitch = }")
            #    print(f"{x.transform.rotation.yaw = }")
            #    print(f"{x.transform.rotation.roll = }")
            #    print(f"{x.lane_id = }")
            #    print(f"{x.road_id = }")
            wtown_road.append(x.road_id)
            wtown_lane.append(x.lane_id)
            if args.only_road:
                cadena = f"[{x.road_id}]"
            else:
                cadena = f"[{x.road_id},{x.lane_id},#{y}]"
            wtown.append(cadena)
            y += 1

        counter_town_lanes = Counter(wtown_lane)
        counter_town_roads = Counter(wtown_road)
        print(f"\n{counter_town_lanes = }")
        print(f"\n{counter_town_roads = }")
        print(f"\n{counter_town_roads.keys() = }")
        print(f"\n{counter_town_roads.values() = }")

        x = [wp.transform.location.x for wp in waypoint_list]
        y = [wp.transform.location.y for wp in waypoint_list]
        plt.scatter(x, y, label="Values")
        # plt.xlabel("X")
        # plt.ylabel("Y")
        for i, label in enumerate(wtown):
            if i % 2 == 0:
                plt.annotate(label, (x[i] - 1.5, y[i] - 0.2))
            else:
                plt.annotate(label, (x[i] - 1.5, y[i] + 0.7))

        #####################################################################

        plt.show()

    finally:
        pass


if __name__ == "__main__":
    try:
        main()
    finally:
        print("Done.")
