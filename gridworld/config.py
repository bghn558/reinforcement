from rllib2.config import config as PkgConfig
from rllib2.common.const import RunMode

# 1=solver, 2=record, 3=pre_train, 4=comparison, 5=statistics
# 6=record and pre_train, 7=quadtree a-star 8=quadtree dijkstra record, 9=quadtree train
task_type = 9

if PkgConfig.mode == RunMode.PLAY and task_type == 1:
    render = True
else:
    render = False
# render = True
# render duration
render_duration = 0.001
step_duration = 0.1
# state type
# state type = 1~7
state_type = 7

# 1=Euclidean, 2=Manhattan, 3=QValue, 4=Value
heuristic_type = 4
#
train_step = 0
# train_step = 33001

check_disappeared_obstacle = True
#
n_processors = 1
sub_trainers = 2

# 1=AStarSolver, 2=DLSolver, 3=HybridNavigationSolver
solver_type = 3

# config for GVIN model
Train = True
