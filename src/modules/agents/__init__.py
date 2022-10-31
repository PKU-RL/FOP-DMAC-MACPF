REGISTRY = {}

from .rnn_agent import RNNAgent
from .dfop_agent import DFOPAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["dfop"] = DFOPAgent
