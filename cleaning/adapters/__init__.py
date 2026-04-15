from .assist2009 import run as assist2009
from .assist2017 import run as assist2017
from .algebra2005 import run as algebra2005
from .bridge2algebra2006 import run as bridge2algebra2006
from .nips_task34 import run as nips_task34
from .junyi2015 import run as junyi2015

ADAPTERS = {
    "assist2009": assist2009,
    "assist2017": assist2017,
    "algebra2005": algebra2005,
    "bridge2algebra2006": bridge2algebra2006,
    "nips_task34": nips_task34,
    "junyi2015": junyi2015,
}