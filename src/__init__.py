from gymnasium.envs.registration import register
from const import ACTIONS

register(
     id="Segment-v0",
     entry_point="envs:SegmentEnv",
     max_episode_steps=100,
)

register(
     id="Graph-v0",
     entry_point="envs:GraphEnv",
     max_episode_steps=100
)

register(
     id="Heat-v0",
     entry_point="envs:HeatEnv",
     max_episode_steps=100
)