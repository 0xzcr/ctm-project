"""
The diary: the Clocked Event Memory (CEM) of the thesis (Section 3), in one file.

The whole memory on one page:

    write    at the end of event e    v_e = LN(W_v u_e),   stamp Phi(e),   sketch k_e
    find     every tick, every head   a_j = softmax_j( c log(1 + n) * score_j )
    scores   by number                kappa(i - j)                          i = the page the question names
             next / previous page     agreement( turn(pointer, +1 or -1), Phi(j) )
             content                  <cue, k_j>                            cue = sketch of what is seen now
             latest page (optional)   the last page, exactly (not in the thesis)
    read                              m_h = sum_j a_j v_j,    r_t = W_o [m_number; m_next; m_previous; m_content] + b_o
    edit     the three clock heads    v_j <- LN( v_j + a_j eps_t W_e z_t )
    fixed                             the clock (hand speeds and trust weights), c = 12.5, the sketch matrix G, f
    learned                           W_v, W_o, b_o, W_e, eps_t;     W_o = b_o = W_e = 0 at the start

Words (thesis Table 2):
    page       one memory entry, written at the end of an event. Pages are numbered 0, 1, 2, ... here, because
               the upstream index tokens count from 0 (the thesis counts from 1).
    stamp      the clock's time for a page, Phi(j); fixed, never learned
    spotlight  attention over the pages; its light always adds up to one
    focus      how sharply the spotlight concentrates on the best page: lambda(n) = c log(1 + n)

Every guarantee in the thesis assumes at most H + 1 = 50 pages per episode (assumption A3). The diary still works
with more pages, but without the guarantees.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linprog

# ---- The design point (thesis Table 6). Every guarantee in the thesis is proved for these values. ----
N_HANDS = 16                  # K: clock hands, so a stamp has d = 2K = 32 numbers
HORIZON = 49                  # H: the largest gap between pages the clock is designed for (so <= 50 pages)
FOCUS = 12.5                  # c in lambda(n) = c log(1 + n); must be >= c_flip(0.95) = 12.253 (Theorem 9)
VALUE_SIZE = 64               # d_v: the length of what is written on a page
SKETCH_UNITS = 2048           # M: the number of units in a content sketch
SKETCH_ON_FRACTION = 0.05     # f: the share of sketch units that are on

# The heads, by the names probe.py silences them by. The latest-page head is optional.
HEADS = ("by number", "next page", "previous page", "content", "latest page")

# Layer norm's eps. PyTorch's default, 1e-5, makes a page about 0.004% shorter than sqrt(d_v), so the next
# re-normalisation would move every page a little, edited or not. With 1e-12 every page has length sqrt(d_v) to
# float precision, as the thesis assumes, and eps still guards against dividing 0 by 0.
LAYER_NORM_EPS = 1e-12


# -------------------------------------------------------------------------------------------------------
# The clock (thesis Section 3.2). Nothing here is learned.
# -------------------------------------------------------------------------------------------------------

def design_clock(n_hands: int, horizon: int) -> tuple[np.ndarray, np.ndarray, float]:
    """
    The clock's hand speeds, its trust weights, and their margin m_H.

    Hand k turns by omega_k = 10000 ** (-2 (k - 1) / 2K) radians per page (eq. 3), the frequencies of the upstream
    index embedding. The trust weights w are the ones with the largest margin over the horizon (eq. 5), found by a
    linear program:

        maximise  m   subject to   1 - kappa_w(gap) >= m  for every gap = 1, ..., H,   w_k >= 0,   sum_k w_k = 1

    They are designed, not learned, because training only "hears" the clock at the gaps it practises (Theorem 1).
    """
    k_minus_one = np.arange(n_hands)
    exponents = -2.0 * k_minus_one / (2 * n_hands)
    speeds = 10000.0 ** exponents

    # mismatch[g, k] = 1 - cos(omega_k * gap_g). Because the weights sum to one, 1 - kappa_w(gap_g) = mismatch[g] @ w.
    gaps = np.arange(1, horizon + 1)
    mismatch = 1.0 - np.cos(np.outer(gaps, speeds))                     # (H, K)

    # The unknowns are x = (w_1, ..., w_K, m). linprog minimises, so it minimises -m, with one inequality per gap
    # in its "A x <= b" form: m - mismatch[g] @ w <= 0.
    cost = np.zeros(n_hands + 1)
    cost[-1] = -1.0
    inequality_matrix = np.hstack([-mismatch, np.ones((horizon, 1))])   # (H, K + 1)
    equality_matrix = np.ones((1, n_hands + 1))                         # the weights sum to one; m takes no part
    equality_matrix[0, -1] = 0.0
    solution = linprog(
        cost,
        A_ub=inequality_matrix,
        b_ub=np.zeros(horizon),
        A_eq=equality_matrix,
        b_eq=np.ones(1),
        bounds=[(0.0, None)] * n_hands + [(None, None)],
    )
    if not solution.success:
        raise RuntimeError(f"designing the clock failed: {solution.message}")

    # Remove the solver's round-off (about 1e-9), so the weights are exactly non-negative and sum to one. That makes
    # the agreement of a page with itself exactly 1.
    weights = np.clip(solution.x[:n_hands], 0.0, None)
    weights = weights / weights.sum()
    margin = float((mismatch @ weights).min())       # how far below a perfect match the most look-alike page stays
    return speeds, weights, margin


class Clock(nn.Module):
    """
    Stamps pages, compares stamps, and turns stamps forward or back. A stamp holds the positions of K hands:

        Phi(j) = ( cos(omega_1 j), ..., cos(omega_K j),   sin(omega_1 j), ..., sin(omega_K j) )

    (the thesis interleaves the cosines and sines; the order changes nothing). Design point: 16 hands and a horizon
    of 49 give the margin m_H = 0.325. The speeds and trust weights are buffers: saved with the model, never trained.
    """

    def __init__(self, n_hands: int = N_HANDS, horizon: int = HORIZON):
        super().__init__()
        speeds, weights, margin = design_clock(n_hands, horizon)
        self.n_hands = n_hands
        self.margin = margin
        self.register_buffer("speeds", torch.tensor(speeds, dtype=torch.float32))     # omega: (K,)
        self.register_buffer("trust", torch.tensor(weights, dtype=torch.float32))     # w*:    (K,)

    def stamp(self, page_numbers: torch.Tensor) -> torch.Tensor:
        """Phi(j) of every page number j: any shape S of integers -> S + (2K,)."""
        pages = page_numbers.to(self.speeds.dtype).unsqueeze(-1)       # S + (1,)
        angles = pages * self.speeds                                    # S + (K,): where each hand points
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

    def agreement(self, query_stamps: torch.Tensor, page_stamps: torch.Tensor) -> torch.Tensor:
        """
        How much each query agrees with each page (eq. 4): sum_k w_k (cos_query_k cos_page_k + sin_query_k sin_page_k).
        For two exact stamps this is kappa(i - j) = sum_k w_k cos(omega_k (i - j)): exactly 1 for the same page, at
        most 1 - m_H for every other page within the horizon, and it depends only on the gap (Theorem 11).

        query_stamps (B, 2K), page_stamps (n, 2K)  ->  (B, n)
        """
        trust_per_number = torch.cat([self.trust, self.trust])     # hand k counts through its cosine and its sine
        return (query_stamps * trust_per_number) @ page_stamps.T

    def turn(self, stamps: torch.Tensor, pages: int) -> torch.Tensor:
        """
        Turn every hand on by `pages` pages, or back if negative (eq. 7). By the angle-addition formulas,
        turn(Phi(j), delta) = Phi(j + delta) (Theorem 11), and it works the same way on a blend of stamps.
        """
        cos_now = stamps[..., :self.n_hands]
        sin_now = stamps[..., self.n_hands:]
        turn_angles = self.speeds * pages
        cos_turn = torch.cos(turn_angles)
        sin_turn = torch.sin(turn_angles)
        cos_new = cos_now * cos_turn - sin_now * sin_turn                  # cos(a + b)
        sin_new = sin_now * cos_turn + cos_now * sin_turn                  # sin(a + b)
        return torch.cat([cos_new, sin_new], dim=-1)


# -------------------------------------------------------------------------------------------------------
# The sparse sketch (thesis eq. 8). Nothing here is learned, and no gradient flows through a sketch.
# -------------------------------------------------------------------------------------------------------

class SparseSketch(nn.Module):
    """
    k = 1_{A(y)} / sqrt(f M), where A(y) are the f*M largest coordinates of G y and G is a fixed N(0, 1) matrix.
    Every sketch has exactly f*M units on and length 1, so the overlap of two sketches is the share of on-units they
    have in common. Sparseness widens the gap between near-duplicates (Theorem 12): at f = 5%, two inputs with
    cosine 0.9 share only about 64% of their on-units.
    """

    def __init__(self, input_size: int, n_units: int = SKETCH_UNITS, on_fraction: float = SKETCH_ON_FRACTION,
                 seed: int = 0):
        super().__init__()
        self.n_on = int(on_fraction * n_units)          # f*M: 102 units on at the design point
        # G has its own random generator, so making a sketch never changes any other random draw.
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("expansion", torch.randn(n_units, input_size, generator=generator))     # G: (M, p)

    @torch.no_grad()
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features (B, p) -> sketches (B, M): n_on entries equal to 1 / sqrt(n_on), all others 0."""
        expanded = features @ self.expansion.T
        on_units = torch.topk(expanded, self.n_on, dim=-1).indices
        sketch = torch.zeros_like(expanded)
        sketch.scatter_(-1, on_units, 1.0 / self.n_on ** 0.5)
        return sketch


# -------------------------------------------------------------------------------------------------------
# What a page holds (thesis eq. 2 and Theorem 15)
# -------------------------------------------------------------------------------------------------------

def event_sync(alpha_now: torch.Tensor, beta_now: torch.Tensor, alpha_before: torch.Tensor,
               beta_before: torch.Tensor, fade: torch.Tensor, n_ticks: int) -> torch.Tensor:
    """
    What the neurons did together during one event, and nothing else. The CTM keeps a fading tally for every neuron
    pair, alpha_t = r alpha_{t-1} + z_i z_j and beta_t = r beta_{t-1} + 1. Subtracting the tallies from just before
    the event, faded over its R ticks, leaves exactly the event's own ticks:

        u_e = ( alpha_f - r^R alpha_{s-1} ) / sqrt( beta_f - r^R beta_{s-1} )        all (B, P)
    """
    fade_over_event = fade ** n_ticks
    alpha_of_event = alpha_now - fade_over_event * alpha_before
    beta_of_event = beta_now - fade_over_event * beta_before        # at least 1: the last tick counts fully
    return alpha_of_event / torch.sqrt(beta_of_event)


# -------------------------------------------------------------------------------------------------------
# The diary
# -------------------------------------------------------------------------------------------------------

class Diary(nn.Module):
    """
    One page per event, four heads with one job each, and a read that is added to the CTM's input. How the CTM uses
    it on every tick (cem/model.py):

        open_at(i)       the input shows an index token i: the by-number head holds page i, and both pointers are
                         set to page i's exact stamp (assumption A2)
        read(seen)       before the synapses; the read r_t is added to what the CTM saw in its input
        cross_off(z)     after the neurons; the clock heads may edit the pages they are looking at
        write(u, seen)   at the last tick of every event; adds one page

    n_pairs:  P, the neuron pairs of the CTM's output synchronisation (pages are made from them)
    d_model:  the number of CTM neurons; their outputs z_t drive the edits
    d_input:  the size of the CTM's view of its input, o_t; the read r_t has the same size
    seed:     the diary's own random numbers. It never draws from the global generator, so the CTM starts from the
              same weights with or without it (paired comparisons).
    latest_head:  adds the latest-page head, so W_o reads 5 d_v numbers instead of 4 d_v
    """

    def __init__(self, n_pairs: int, d_model: int, d_input: int, seed: int = 0, latest_head: bool = False):
        super().__init__()
        self.clock = Clock()
        self.sketch = SparseSketch(d_input, seed=seed)
        self.latest_head = latest_head
        n_heads = 5 if latest_head else 4

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.write_value = nn.Linear(n_pairs, VALUE_SIZE, bias=False)       # W_v
            self.edit_direction = nn.Linear(d_model, VALUE_SIZE, bias=False)    # W_e
            self.edit_strength = nn.Linear(d_model, 1)                          # eps_t = sigmoid(this)
            self.output = nn.Linear(n_heads * VALUE_SIZE, d_input)              # W_o and b_o

        # Silent at the start (Theorem 3). With W_o = 0 and b_o = 0 every read is exactly zero, so the CTM with the
        # diary computes exactly what the CTM without it computes. With W_e = 0 no page is edited. Learning still
        # starts (Proposition 4): W_o gets a gradient at once, W_v and W_e from the first update that moves W_o.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.edit_direction.weight)

        self.silenced = set()       # probe.py silences heads by name (HEADS): they read nothing and shine no light
        self.recording = None       # probe.py sets {"pages": [], "by_number": []} to record writes and by-number reads
        self.reset(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)

    def reset(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """An empty diary for each of the `batch_size` episodes."""
        self.values = torch.zeros(batch_size, 0, VALUE_SIZE, device=device, dtype=dtype)       # v_j: (B, n, d_v)
        self.sketches = torch.zeros(batch_size, 0, SKETCH_UNITS, device=device, dtype=dtype)   # k_j: (B, n, M)
        # Until the input first names a page, the clock heads have nothing to look for:
        self.number_page = None         # (B,) the page the by-number head holds
        self.pointers = None            # {+1 or -1: (B, 2K)} the stamp the next / previous head has just read
        self.pointer_pages = None       # {+1 or -1: (B,)} the page number that head is looking for
        self.edit_light = None          # (B, n) the clock heads' light this tick; cross_off() edits through it

    @property
    def n_pages(self) -> int:
        """n: the number of pages written so far (the same for every episode in the batch)."""
        return self.values.size(1)

    def open_at(self, page_numbers: torch.Tensor):
        """
        The input names page i (B,). The by-number head holds it until the next index token, like a finger kept on
        the page. Both pointers get page i's exact stamp, so the next head looks at page i + 1 next, and the previous
        head at page i - 1.
        """
        exact_stamp = self.clock.stamp(page_numbers)
        self.number_page = page_numbers
        self.pointers = {+1: exact_stamp, -1: exact_stamp}
        self.pointer_pages = {+1: page_numbers, -1: page_numbers}

    # ---- Read: every tick, before the synapses ------------------------------------------------------------

    def read(self, seen: torch.Tensor | None) -> torch.Tensor:
        """
        r_t = W_o [m_number; m_next; m_previous; m_content (; m_latest)] + b_o   (eq. 9). Every head reads a weighted
        average of pages of length sqrt(d_v), so ||r_t|| <= ||W_o|| sqrt(4 d_v) + ||b_o|| however many pages there
        are (Theorem 3).

        seen: (B, d_input), the CTM's view of its input this tick, or None on a tick with no input  ->  (B, d_input)
        """
        if self.n_pages == 0:
            # An empty diary reads nothing, and its heads do not move.
            self.edit_light = None
            nothing_read = self.values.new_zeros(self.values.size(0), self.output.in_features)
            return self.output(nothing_read)

        page_stamps = self.clock.stamp(torch.arange(self.n_pages, device=self.values.device))   # (n, 2K)
        number_read, number_light = self.read_by_number(page_stamps)
        next_read, next_light = self.flip(+1, page_stamps)
        previous_read, previous_light = self.flip(-1, page_stamps)
        content_read, _ = self.read_by_content(seen)

        # Only the heads that hold a page by its stamp may cross it off (Section 3.5). open_at() sets all three.
        self.edit_light = None
        if self.number_page is not None:
            self.edit_light = number_light + next_light + previous_light

        if self.recording is not None and self.number_page is not None:
            self.recording["by_number"].append((number_read, number_light))

        reads = [number_read, next_read, previous_read, content_read]
        if self.latest_head:
            latest_read, _ = self.read_latest()
            reads.append(latest_read)
        return self.output(torch.cat(reads, dim=-1))

    # The heads. Each returns (what it read (B, d_v), its light on every page (B, n)). A head with no page to look
    # for, or a silenced head, reads nothing and shines no light.

    def read_by_number(self, page_stamps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Open the diary at the page the question named: score_j = kappa(i - j). Every wrong page scores at least
        m_H = 0.325 below the right one, so the right page gets at least 0.9887 of the light in every diary of 2 to
        50 pages (Theorem 5).
        """
        if self.number_page is None or "by number" in self.silenced:
            return self.nothing()
        query = self.clock.stamp(self.number_page)                         # Phi(i): (B, 2K)
        scores = self.clock.agreement(query, page_stamps)                  # kappa(i - j): (B, n)
        light = self.spotlight(scores) * self.is_written(self.number_page)
        return self.look_up(light), light

    def flip(self, direction: int, page_stamps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The next-page head (direction +1) or the previous-page head (-1): turn every hand one page on or back from
        the pointer and look the result up, score_j = agreement(turn(pointer, direction), Phi(j)) (eq. 7). The new
        pointer is the stamp just read, a blend that is mostly the right page, so every flip snaps back onto a single
        page and small slips never pile up: each flip keeps at least 95% of the light on the right page (Theorem 9).
        A head that runs off either end stays silent until the input names a page again.
        """
        name = "next page"
        if direction < 0:
            name = "previous page"
        if self.pointers is None or name in self.silenced:
            return self.nothing()

        query = self.clock.turn(self.pointers[direction], direction)       # R_{+1 or -1} pointer: (B, 2K)
        self.pointer_pages[direction] = self.pointer_pages[direction] + direction
        scores = self.clock.agreement(query, page_stamps)
        light = self.spotlight(scores)
        self.pointers[direction] = light @ page_stamps                     # the stamp just read: (B, 2K)
        light = light * self.is_written(self.pointer_pages[direction])
        return self.look_up(light), light

    def read_by_content(self, seen: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Leaf through the diary for a page that looked like what is seen now: score_j = <cue, k_j>, the share of sketch
        units the cue and the page have in common (Theorem 12). With no input, no cue. This head never edits.
        """
        if seen is None or "content" in self.silenced:
            return self.nothing()
        cue = self.sketch(seen)
        scores = torch.einsum("bm,bnm->bn", cue, self.sketches)
        light = self.spotlight(scores)
        return self.look_up(light), light

    def read_latest(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The optional latest-page head (not in the thesis): the page of the event that has just ended, read exactly.
        It gives the CTM a place to carry its running result from one question part to the next. It never edits.
        """
        if "latest page" in self.silenced:
            return self.nothing()
        light = self.values.new_zeros(self.values.size(0), self.n_pages)
        light[:, -1] = 1.0
        return self.values[:, -1], light

    def spotlight(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Light that adds up to one (eq. 6): a_j = softmax_j(lambda(n) score_j), lambda(n) = c log(1 + n). The focus
        grows like log n because more pages compete for the light; a fixed focus loses its grip (Proposition 6).
        """
        focus = FOCUS * math.log(1 + self.n_pages)
        return torch.softmax(focus * scores, dim=-1)

    def look_up(self, light: torch.Tensor) -> torch.Tensor:
        """What a spotlight reads: m = sum_j a_j v_j.     (B, n) -> (B, d_v)"""
        return torch.einsum("bn,bnv->bv", light, self.values)

    def is_written(self, page_numbers: torch.Tensor) -> torch.Tensor:
        """1.0 for every episode whose page `page_numbers` has been written, else 0.0.     (B,) -> (B, 1)"""
        written = (page_numbers >= 0) & (page_numbers < self.n_pages)
        return written.to(self.values.dtype).unsqueeze(-1)

    def nothing(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The read and the light of a head with no page to look for: all zeros."""
        batch_size = self.values.size(0)
        return self.values.new_zeros(batch_size, VALUE_SIZE), self.values.new_zeros(batch_size, self.n_pages)

    # ---- Cross off: every tick, after the neurons ---------------------------------------------------------

    def cross_off(self, neurons: torch.Tensor):
        """
        Edit the pages the clock heads are looking at (eq. 10): v_j <- LN( v_j + a_j eps_t W_e z_t ), through this
        tick's light. Stamps and sketches are never edited, so every finding guarantee still holds after any number of
        edits, and an edit made through the by-number head moves any other page by at most 2.3% of its size
        (Theorem 14). neurons: (B, d_model), the CTM's neuron outputs z_t.
        """
        if self.edit_light is None:
            return
        strength = torch.sigmoid(self.edit_strength(neurons))            # eps_t in [0, 1]: (B, 1)
        edit = strength * self.edit_direction(neurons)                    # e_t: (B, d_v)
        change = self.edit_light.unsqueeze(-1) * edit.unsqueeze(1)        # a_j e_t on every page: (B, n, d_v)
        self.values = F.layer_norm(self.values + change, (VALUE_SIZE,), eps=LAYER_NORM_EPS)
        self.edit_light = None                                            # never edit through the same light twice

    # ---- Write: at the last tick of every event -----------------------------------------------------------

    def write(self, event_synchronisation: torch.Tensor, seen: torch.Tensor):
        """
        Add one page for the event that has just ended (eq. 2): v_e = LN(W_v u_e), stamp Phi(e), sketch k_e. LN has
        no learned scale, so every page has length sqrt(d_v). The stamp is not stored: page e's stamp is always Phi(e).
        Every page is written at full strength; nothing decides how important an event was (Section 3.7).

        event_synchronisation: (B, P) u_e from event_sync();   seen: (B, d_input) the view at the event's last tick
        """
        raw_value = self.write_value(event_synchronisation)                               # W_v u_e: (B, d_v)
        value = F.layer_norm(raw_value, (VALUE_SIZE,), eps=LAYER_NORM_EPS)                # v_e: (B, d_v)
        sketch = self.sketch(seen)                                                        # k_e: (B, M)
        self.values = torch.cat([self.values, value.unsqueeze(1)], dim=1)
        self.sketches = torch.cat([self.sketches, sketch.unsqueeze(1)], dim=1)
        if self.recording is not None:
            self.recording["pages"].append(value)
