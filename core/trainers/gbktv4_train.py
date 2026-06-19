from core.registry import TRAINER_REGISTRY
from core.trainers.gbktv3_train import GBKTV3Trainer


@TRAINER_REGISTRY.register("gbkt_final")
@TRAINER_REGISTRY.register("gbktv4")
class GBKTV4Trainer(GBKTV3Trainer):
    """Trainer for GBKTV4.

    GBKTV4 keeps the GBKTV3 ball/concept diagnostics but uses the cleaned loss
    found by ablation: final prediction + ball auxiliary + concept auxiliary +
    theta auxiliary. Radius and confidence auxiliary losses are disabled for
    GBKTV4 because they did not improve ASSIST2009 fold-0 performance.
    """

    use_radius_loss = False
    use_conf_loss = False
