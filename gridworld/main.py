import sys, os
os.system('pwd')
sys.path.append('/output/workspace')
print(sys.path)

from rllib2.game.gridworld.solver import Solver
from rllib2.game.gridworld.record import Record
from rllib2.game.gridworld.pre_train import PreTrain
from rllib2.game.gridworld.compare import Comparison
from rllib2.game.gridworld.cost_statistics import Statistics
from rllib2.game.gridworld.config import task_type
from rllib2.game.gridworld.record_and_train import Manager
from rllib2.game.gridworld.quadtree import QuadTreeAStarSolver, DijkstraRecord, QuadTreeTrainer
import time

if task_type == 1:
    while True:
        Solver().run()
        time.sleep(1)
    # Solver().check_succ_only_on_dl()
elif task_type == 2:
    record = Record()
    record.run()
elif task_type == 3:
    pre_train = PreTrain()
    pre_train.run()
elif task_type == 4:
    comparison = Comparison()
    comparison.run()
elif task_type == 5:
    statistics = Statistics()
    statistics.run()
elif task_type == 6:
    manager = Manager()
    manager.run()
elif task_type == 7:
    quadtree = QuadTreeAStarSolver()
    quadtree.run()
elif task_type == 8:
    dijkstra_record = DijkstraRecord()
    dijkstra_record.run()
elif task_type == 9:
    quadtree_trainer = QuadTreeTrainer()
    quadtree_trainer.run()
