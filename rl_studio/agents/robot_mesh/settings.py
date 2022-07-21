###########################
# Global variables file
###########################
from pydantic import BaseModel


class QLearnConfig(BaseModel):
    actions: int = 3
    debug_level: int = 0
    telemetry: bool = False
    telemetry_mask: bool = True
    plotter_graphic: bool = False
    my_board: bool = True
    save_positions: bool = False
    save_model: bool = True
    load_model: bool = False
    output_dir = "./logs/qlearn_models/qlearn_camera_solved/"
    poi = 1  # The original pixel row is: 250, 300, 350, 400 and 450 but we work only with the half of the image
    actions_set = "simple"  # test, simple, medium, hard
    max_distance = 0.5

qlearn = QLearnConfig()
