class Registry:
    """A simple registry to map strings to classes."""
    def __init__(self, name):
        self.name = name
        self._registry = {}

    def register(self, cls):
        """Register a class with the registry."""
        self._registry[cls.__name__] = cls
        return cls  # Return class itself to use as a decorator

    def get(self, name):
        """Retrieve a class from the registry."""
        return self._registry.get(name, None)

    def create(self, cfg:dict, **kwargs):
        """Instantiate an object from the registry."""
        cfg = cfg.copy()
        cls_name = cfg.pop("type")
        cls_args = cfg.pop("args", {})
        cls_args.update(kwargs)
        cls = self.get(cls_name)
        if cls is None:
            raise ValueError(f"Class '{cls_name}' is not registered in {self.name} registry.")
        return cls(**cls_args)
SAMPREDICTOR_REGISTRY = Registry("SAMPREDICTOR")