
import torch
import torch.nn
from layers.layer_with_visualization import LayerWithVisualization
from typing import Dict, Any
import framework


class NDRBase(LayerWithVisualization):
    def __init__(self):
        super().__init__()

        self.plot_cache = []

    def plot(self, options: Dict[str, Any]) -> Dict[str, Any]:
        r = {}
        if self.visualization_enabled:
            n_steps = options.get("n_steps")
            r["gate"] = framework.visualize.plot.AnimatedHeatmap(
                        torch.stack(self.plot_cache, 0)[:, :n_steps].transpose(1,2),
                        ylabel="dest", xlabel="src", textval=False, x_marks=options.get("steplabel"))
            self.plot_cache.clear()

        return r
