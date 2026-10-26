"""
The clock that stamps every page of the diary (thesis Section 3.2).

Picture K clock hands that turn at different speeds. From one page to the next, hand k turns by omega_k
radians, so the positions of the hands on page j form the page's stamp:

    Phi(j) = ( cos(omega_1 j), ..., cos(omega_K j),   sin(omega_1 j), ..., sin(omega_K j) )

Here the K cosines come first and the K sines after them. (The thesis interleaves them: cos, sin, cos,
sin, ... The order changes nothing.)

Two stamps are compared hand by hand, and each hand's opinion is weighted by how much we trust it:

    agreement( Phi(i), Phi(j) ) = sum_k  w_k * cos( omega_k * (i - j) )  =  kappa(i - j)

The agreement is exactly 1 for the same page and smaller for every other page. It depends only on the
gap i - j, never on where in the diary the two pages are (Theorem 11).

The trust weights w are designed once, by a linear program, so that every wrong page within the horizon
(a gap of 1 to H pages) agrees at most 1 - m_H with the right one. Then they are frozen. They are not
learned, because training only "hears" the clock at the gaps it practises (Theorem 1).

Design point: K = 16 hands and a horizon of H = 49 give the margin m_H = 0.325.
Nothing in this file is learned.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linprog


def hand_speeds(n_hands: int) -> np.ndarray:
    """
    The speed of every clock hand, in radians per page (thesis eq. 3):

        omega_k = 10000 ** ( -2 (k - 1) / d ),     k = 1, ..., K,     d = 2K

    This is the frequency formula of the upstream index embedding. The fastest hand (omega_1 = 1) goes
    round once every 6.3 pages. With 16 hands, the slowest goes round about once every 35,000 pages.
    """
    stamp_size = 2 * n_hands                    # d
    k_minus_one = np.arange(n_hands)             # 0, 1, ..., K - 1
    exponents = -2.0 * k_minus_one / stamp_size
    return 10000.0 ** exponents


def design_trust_weights(speeds: np.ndarray, horizon: int) -> tuple[np.ndarray, float]:
    """
    The trust weights with the largest margin over the horizon (thesis eq. 5):

        maximise    m
        subject to  1 - kappa_w(gap) >= m        for every gap = 1, 2, ..., horizon
                    w_k >= 0  and  sum_k w_k = 1

    kappa_w is linear in w, so this is a linear program. Returns the weights w* and their margin m_H.
    """
    n_hands = len(speeds)
    gaps = np.arange(1, horizon + 1)

    # mismatch[g, k] = 1 - cos(omega_k * gap_g): how differently hand k points on two pages gap_g apart.
    # Because the weights sum to one, 1 - kappa_w(gap_g) = sum_k w_k * mismatch[g, k].
    angles = np.outer(gaps, speeds)                                    # (H, K)
    mismatch = 1.0 - np.cos(angles)                                    # (H, K)

    # The unknowns are x = (w_1, ..., w_K, m). linprog minimises, so we ask it to minimise -m.
    cost = np.zeros(n_hands + 1)
    cost[-1] = -1.0

    # One inequality per gap, in linprog's "A x <= b" form:   m - sum_k w_k * mismatch[g, k] <= 0
    margin_column = np.ones((horizon, 1))
    inequality_matrix = np.hstack([-mismatch, margin_column])          # (H, K + 1)
    inequality_bound = np.zeros(horizon)

    # The weights sum to one. The margin m takes no part in this equation.
    equality_matrix = np.ones((1, n_hands + 1))
    equality_matrix[0, -1] = 0.0
    equality_bound = np.ones(1)

    weight_bounds = [(0.0, None)] * n_hands      # every weight is non-negative
    margin_bound = [(None, None)]                # the margin may be any number

    solution = linprog(
        cost,
        A_ub=inequality_matrix,
        b_ub=inequality_bound,
        A_eq=equality_matrix,
        b_eq=equality_bound,
        bounds=weight_bounds + margin_bound,
    )
    if not solution.success:
        raise RuntimeError(f"designing the clock failed: {solution.message}")

    # Remove the solver's round-off (about 1e-9), so the weights are exactly non-negative and sum to one.
    # That makes the agreement of a page with itself exactly 1.
    weights = np.clip(solution.x[:n_hands], 0.0, None)
    weights = weights / weights.sum()

    # The margin these weights achieve: how far below a perfect match the most look-alike wrong page stays.
    margin = float((mismatch @ weights).min())
    return weights, margin


class Clock(nn.Module):
    """
    The designed clock. It stamps pages, compares stamps, and turns stamps forward or back.

    The hand speeds and the trust weights are buffers, not parameters: they move with the model to a GPU
    and are saved with it, but they are never trained.
    """

    def __init__(self, n_hands: int = 16, horizon: int = 49):
        super().__init__()
        speeds = hand_speeds(n_hands)
        weights, margin = design_trust_weights(speeds, horizon)

        self.n_hands = n_hands          # K
        self.horizon = horizon          # H: the clock is designed for gaps of up to H pages
        self.margin = margin            # m_H: 0.325 for 16 hands and a horizon of 49

        self.register_buffer("speeds", torch.tensor(speeds, dtype=torch.float32))     # omega: (K,)
        self.register_buffer("trust", torch.tensor(weights, dtype=torch.float32))     # w*:    (K,)

    def stamp(self, page_numbers: torch.Tensor) -> torch.Tensor:
        """
        The stamp Phi(j) of every page number j.

        page_numbers: integer page numbers, any shape S (negative numbers are allowed)
        returns:      shape S + (2K,): the K cosines, then the K sines
        """
        pages = page_numbers.to(self.speeds.dtype).unsqueeze(-1)       # S + (1,)
        angles = pages * self.speeds                                    # S + (K,): where each hand points
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

    def agreement(self, query_stamps: torch.Tensor, page_stamps: torch.Tensor) -> torch.Tensor:
        """
        How much each query agrees with each page (thesis eq. 4):

            agreement = sum_k w_k * ( cos_query_k * cos_page_k  +  sin_query_k * sin_page_k )

        For two exact stamps this is sum_k w_k cos(omega_k (i - j)) = kappa(i - j). A query may also be a
        blend of stamps (a pointer); the formula stays the same.

        query_stamps: (B, 2K)
        page_stamps:  (n, 2K), the stamps of pages 0, 1, ..., n - 1
        returns:      (B, n)
        """
        # Hand k counts twice, through its cosine and through its sine, both times with the same trust w_k.
        trust_per_number = torch.cat([self.trust, self.trust])          # (2K,)
        weighted_query = query_stamps * trust_per_number                 # (B, 2K)
        return weighted_query @ page_stamps.T                            # (B, n)

    def turn(self, stamps: torch.Tensor, pages: int) -> torch.Tensor:
        """
        Turn every hand on by `pages` pages, or back if `pages` is negative (R_delta in thesis eq. 7).

        By the angle-addition formulas, turning an exact stamp gives the exact stamp of the new page:
        turn(Phi(j), delta) = Phi(j + delta) (Theorem 11). It works the same way on a blend of stamps.

        stamps:  (B, 2K)
        returns: (B, 2K)
        """
        cos_now = stamps[..., :self.n_hands]
        sin_now = stamps[..., self.n_hands:]

        turn_angles = self.speeds * pages                                # (K,): how far each hand turns
        cos_turn = torch.cos(turn_angles)
        sin_turn = torch.sin(turn_angles)

        # cos(a + b) = cos a cos b - sin a sin b         sin(a + b) = sin a cos b + cos a sin b
        cos_new = cos_now * cos_turn - sin_now * sin_turn
        sin_new = sin_now * cos_turn + cos_now * sin_turn
        return torch.cat([cos_new, sin_new], dim=-1)
