REGISTRY = {}

from .basic_controller import BasicMAC
from .om_controller import Multi_Module_MAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["multi_module_mac"] = Multi_Module_MAC