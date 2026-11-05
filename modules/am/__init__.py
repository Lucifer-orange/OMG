REGISTRY = {}

from .none_am import None_AM
from .base_am import Base_AM
from .omg_am import OMG_AM

REGISTRY["none_am"] = None_AM
REGISTRY["base_am"] = Base_AM
REGISTRY["omg_am"] = OMG_AM