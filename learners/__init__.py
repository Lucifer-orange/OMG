from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .am_learner import AMLearner
from .omg_learner import OMGLearner
from .q_learner_4am import QLearner as QLearner4AM

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["am_learner"] = AMLearner
REGISTRY["omg_learner"] = OMGLearner
REGISTRY["q_learner_4am"] = QLearner4AM