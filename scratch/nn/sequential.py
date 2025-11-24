from loguru import logger

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

    def info(self):
        """
        Print descriptive information about the model layers, shapes, and total parameters.
        """
        total_params = 0
        logger.info("Model Architecture:")
        logger.info("=" * 50)
        
        for i, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__
            params = layer.parameters() if hasattr(layer, "parameters") else []
            layer_params = sum(p.data.size for p in params) if params else 0
            total_params += layer_params
            
            # Get shape information if available
            shape_info = ""
            if hasattr(layer, "in_features") and hasattr(layer, "out_features"):
                shape_info = f"({layer.in_features} → {layer.out_features})"
            elif hasattr(layer, "kernel_size"):
                if hasattr(layer, "in_channels") and hasattr(layer, "out_channels"):
                    kernel_size = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
                    shape_info = f"({layer.in_channels}x{layer.out_channels}, {kernel_size[0]}x{kernel_size[1]})"
            
            logger.info(f"[{i}] {layer_name} {shape_info} - Parameters: {layer_params:,}")
        
        logger.info("=" * 50)
        logger.info(f"Total parameters: {total_params:,}")