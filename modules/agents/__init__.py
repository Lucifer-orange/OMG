REGISTRY = {}

from .rnn_agent import RNNAgent
from .idv_rnn_agent import IDV_RNNAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["idv_rnn"] = IDV_RNNAgent