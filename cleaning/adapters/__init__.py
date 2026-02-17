from .assist2009 import run as assist2009
from .assist2017 import run as assist2017
from .algebra2005 import run as algebra2005

ADAPTERS = {
    "assist2009": assist2009,
    "assist2017": assist2017,
    "algebra2005": algebra2005,
}