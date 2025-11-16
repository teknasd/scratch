class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x