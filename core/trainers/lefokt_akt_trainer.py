from core.registry import TRAINER_REGISTRY
from .akt_trainer import AKTTrainer


@TRAINER_REGISTRY.register("lefokt")
@TRAINER_REGISTRY.register("lefokt_akt")
class LEFOKTAKTTrainer(AKTTrainer):
    pass
