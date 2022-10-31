REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .episode_sequential_runner import EpisodeSequentialRunner
REGISTRY["sequential"] = EpisodeSequentialRunner