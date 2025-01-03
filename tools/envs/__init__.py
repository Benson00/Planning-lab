"""
OpenAI Gym environments designed for the AI lab course
"""

from gym.envs.registration import register
from envs.maze_env import *
from envs.lava_env import *
from envs.ppd import *
from envs.advanced_ppd import *

register(
    id='SmallMaze-v0',
    entry_point='envs:SmallMazeEnv')
register(
    id='GrdMaze-v0',
    entry_point='envs:GrdMazeEnv')
register(
    id='BlockedMaze-v0',
    entry_point='envs:BlockedMazeEnv')

register(
    id='LavaFloor-v0',
    entry_point='envs:LavaFloorEnv')
register(
    id='VeryBadLavaFloor-v0',
    entry_point='envs:VeryBadLavaFloorEnv')
register(
    id='NiceLavaFloor-v0',
    entry_point='envs:NiceLavaFloorEnv')
register(
    id='BiggerLavaFloor-v0',
    entry_point='envs:BiggerLavaFloorEnv')
register(
    id='HugeLavaFloor-v0',
    entry_point='envs:HugeLavaFloorEnv')
register(
    id='ProbabilisticPlanningDomain-v0',
    entry_point='envs:ProbabilisticPlanningDomain'
)
register(
    id='ProbabilisticPlanningDomain-v1',
    entry_point='envs:ProbabilisticPlanningDomainV1'
)
