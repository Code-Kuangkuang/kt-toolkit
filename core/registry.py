class Registry:
    def __init__(self, name):
        self._name = name
        self._map  = {}

    def register(self, key):
        def deco(obj):
            if key in self._map:
                raise KeyError(f"{key} is already registered in {self._name}")
            self._map[key] = obj
            return obj
        return deco

    def get(self, key):
        return self._map[key]

    def get_all(self):
        return list(self._map.keys())

MODEL_REGISTRY = Registry("model")
DATASET_REGISTRY = Registry("dataset")
TRAINER_REGISTRY = Registry("trainer")
