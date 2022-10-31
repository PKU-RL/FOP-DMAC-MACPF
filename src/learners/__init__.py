from .fop_learner import FOP_Learner
from .dfop_learner import DFOP_Learner
from .dmac_learner import DMAC_Learner
REGISTRY = {}

REGISTRY["fop_learner"] = FOP_Learner
REGISTRY["dfop_learner"] = DFOP_Learner
REGISTRY["dmac_learner"] = DMAC_Learner