REGISTRY = {}

from .basic_controller import BasicMAC
from .dfop_controller import DFOPMAC
REGISTRY["basic_mac"] = BasicMAC
REGISTRY["dfop_mac"] = DFOPMAC