"""
The sparse sketch of "what an event looked like" (thesis eq. 8). The content head finds pages with it.

A fixed random matrix G, with M rows and independent N(0, 1) entries, expands the input features y into
M units. Only the top fraction f of the units stay on:

    k = 1_{A(y)} / sqrt(f M),        A(y) = the f*M largest coordinates of G y

Every sketch has exactly f*M units on and has length 1. The overlap <k, k'> of two sketches is therefore
the share of on-units they have in common, a number between 0 and 1.

Why sparse? Two very similar inputs end up sharing fewer on-units than their similarity suggests, because
small differences decide which units make the cut. That widens the gap between near-duplicates
(Theorem 12): at f = 5%, two inputs with cosine 0.9 share only about 64% of their on-units.

Nothing here is learned, and no gradient flows through a sketch.
"""
import torch
import torch.nn as nn


class SparseSketch(nn.Module):

    def __init__(self, input_size: int, n_units: int = 2048, on_fraction: float = 0.05, seed: int = 0):
        super().__init__()
        self.n_units = n_units                          # M
        self.n_on = int(on_fraction * n_units)          # f*M: 102 units on at the design point

        # G gets its own random generator, so making a sketch never changes any other random draw.
        generator = torch.Generator().manual_seed(seed)
        expansion = torch.randn(n_units, input_size, generator=generator)       # G: (M, p)
        self.register_buffer("expansion", expansion)

    @torch.no_grad()
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (B, p) the input features y
        returns:  (B, M) the sketches k: n_on entries equal to 1 / sqrt(n_on), all others 0
        """
        expanded = features @ self.expansion.T                          # G y: (B, M)
        on_units = torch.topk(expanded, self.n_on, dim=-1).indices      # A(y): (B, n_on)

        sketch = torch.zeros_like(expanded)
        sketch.scatter_(-1, on_units, 1.0 / self.n_on ** 0.5)
        return sketch
