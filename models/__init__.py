from .dkt import DKT
from .dkt_pebg import DKTPEBG
from .dkt_plus import DKTPlus
from .dkvmn import DKVMN
from .akt import AKT
from .sakt import SAKT
from .qikt import QIKTNet
from .gkt import GKT
from .dgekt import DGEKT
from .kqn import KQN
from .atkt import ATKT
from .iekt import IEKT
from .hawkes import HawkesKT
from .lpkt import LPKT
from .hdkt import HDKT
from .deep_irt import DeepIRT
from .saint import SAINT
from .saint_plus import SAINTp
from .simplekt import SimpleKT
from .ukt import UKT
from .keenkt import KeenKT
from .dkt_forget import DKTForget
from .skvmn import SKVMN
from .dimkt import DIMKT
from .atdkt import ATDKT
from .stablekt import StableKT
from .sparsekt import SparseKT
from .dtransformer import DTransformerModel
from .robustkt import RobustKT
from .rekt import ReKT
from .lefokt_akt import LEFOKT_AKT
from .hqaf import HQAFKT

# Backbone+plugin compositions (hd_dkt, hd_akt, hd_simplekt). Must come
# last: register_plugged looks its backbone up in MODEL_REGISTRY, so every
# backbone above has to be registered first. Imported from here rather
# than from each entry point so that `import models` stays the one thing
# a caller needs to populate the model registry.
import plugins  # noqa: F401

# Keep the public export aligned with the import-time model registration.

__all__ = ["DKT", "DKTPEBG", "DKVMN", "AKT", "DKTPlus", "SAKT", "QIKTNet", "GKT", "DGEKT", "KQN", "ATKT", "IEKT", "HawkesKT", "LPKT", "HDKT", "DeepIRT", "SAINT", "SAINTp", "SimpleKT", "UKT", "KeenKT", "DKTForget", "SKVMN", "DIMKT", "ATDKT", "StableKT", "SparseKT", "DTransformerModel", "RobustKT", "ReKT", "LEFOKT_AKT", "HQAFKT"]
