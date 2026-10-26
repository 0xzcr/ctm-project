"""
The Clocked Event Memory (CEM): a diary for the Continuous Thought Machine (thesis Section 3).

The whole memory on one page (the grey box at the end of thesis Section 3):

    write    at the end of event e    v_e = LN(W_v u_e),   stamp Phi(e),   sketch k_e
    find     every tick, every head   a_j = softmax_j( c log(1 + n) * score_j )
    scores   by number                kappa(i - j)                          i = the page the question names
             next / previous          agreement( turn(pointer, +1 or -1), Phi(j) )
             by content               <cue, k_j>                            cue = sketch of what is seen now
    read                              m_h = sum_j a_j v_j,    r_t = W_o [m_number; m_next; m_previous; m_content] + b_o
    edit     the three clock heads    v_j <- LN( v_j + a_j eps_t W_e z_t )
    fixed                             the clock (hand speeds and trust weights), c = 12.5, the sketch matrix G, f
    optional                          a fifth head that reads the latest page exactly (not in the thesis; see
                                      read_latest)
    learned                           W_v, W_o, b_o, W_e, eps_t;     W_o = b_o = W_e = 0 at the start

How the CTM uses it on every tick (see cem/model.py):

    open_at(i)       the input shows an index token i: the by-number head holds page i, and both
                     pointers are set to page i's exact stamp
    read(seen)       before the synapses; the read r_t is added to what the CTM saw in its input
    cross_off(z)     after the neurons; the clock heads may edit the pages they are looking at
    write(u, seen)   at the last tick of every event; adds one page

Words (thesis Table 2):
    page       one memory entry, written at the end of an event. Pages are numbered 0, 1, 2, ... here,
               because the upstream index tokens count from 0 (the thesis counts from 1).
    stamp      the clock's time for a page, Phi(j); fixed, never learned
    spotlight  attention over the pages; its light always adds up to one
    focus      how sharply the spotlight concentrates on the best page: lambda(n) = c log(1 + n)

Every guarantee in the thesis assumes at most H + 1 = 50 pages per episode (assumption A3). The diary
still works with more pages, but without the guarantees.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clock import Clock
from .sketch import SparseSketch

# ---- The design point (thesis Table 6). Every guarantee in the thesis is proved for these values. ----
N_HANDS = 16                  # K: clock hands, so a stamp has d = 2K = 32 numbers
HORIZON = 49                  # H: the largest gap between pages the clock is designed for (so <= 50 pages)
FOCUS = 12.5                  # c in lambda(n) = c log(1 + n); must be >= c_flip(0.95) = 12.253 (Theorem 9)
VALUE_SIZE = 64               # d_v: the length of what is written on a page
SKETCH_UNITS = 2048           # M: the number of units in a content sketch
SKETCH_ON_FRACTION = 0.05     # f: the share of sketch units that are on

N_HEADS = 4                   # by number, next, previous, content

# Layer norm's eps. PyTorch's default, 1e-5, makes a page about 0.004% shorter than sqrt(d_v), so the next
# re-normalisation would move every page a little, edited or not. With 1e-12 every page has length sqrt(d_v) to
# float precision, as the thesis assumes, and eps still guards against dividing 0 by 0.
LAYER_NORM_EPS = 1e-12


def event_sync(alpha_now: torch.Tensor,
               beta_now: torch.Tensor,
               alpha_before: torch.Tensor,
               beta_before: torch.Tensor,
               fade: torch.Tensor,
               n_ticks: int) -> torch.Tensor:
    """
    What the neurons did together during one event, and nothing else (thesis eq. 2 and Theorem 15).

    The CTM keeps a fading tally for every neuron pair:  alpha_t = r alpha_{t-1} + z_i z_j  and
    beta_t = r beta_{t-1} + 1. Subtracting the tallies from just before the event, faded over the event's
    length, leaves exactly the event's own ticks:

        u_e = ( alpha_f - r^R alpha_{s-1} ) / sqrt( beta_f - r^R beta_{s-1} )

    alpha_now, beta_now:        the tallies at the event's last tick f             (B, P)
    alpha_before, beta_before:  the tallies just before the event's first tick s   (B, P)
    fade:                       r, the fading factor of every pair                 (B, P)
    n_ticks:                    R, the number of ticks the event lasted
    returns:                    u_e                                                (B, P)
    """
    fade_over_event = fade ** n_ticks
    alpha_of_event = alpha_now - fade_over_event * alpha_before
    beta_of_event = beta_now - fade_over_event * beta_before        # at least 1: the last tick counts fully
    return alpha_of_event / torch.sqrt(beta_of_event)


class ClockedEventMemory(nn.Module):
    """
    The diary: one page per event, four heads with one job each, and a read that is added to the CTM's input.

    n_pairs:  P, the number of neuron pairs in the CTM's output synchronisation (pages are made from them)
    d_model:  the number of CTM neurons; their outputs z_t drive the edits
    d_input:  the size of the CTM's view of its input, o_t; the read r_t has the same size
    seed:     seeds the diary's own random numbers. The diary never draws from the global random generator,
              so the CTM starts from the same weights with or without it (paired comparisons).
    latest_head:  adds the latest-page head (read_latest), so that W_o reads 5 d_v numbers instead of 4 d_v
    """

    def __init__(self, n_pairs: int, d_model: int, d_input: int, seed: int = 0, latest_head: bool = False):
        super().__init__()
        self.clock = Clock(N_HANDS, HORIZON)
        self.sketch = SparseSketch(d_input, SKETCH_UNITS, SKETCH_ON_FRACTION, seed=seed)
        self.latest_head = latest_head
        self.n_heads = N_HEADS + 1 if latest_head else N_HEADS

        # The learned parts, drawn from a forked random generator (see `seed` above).
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.write_value = nn.Linear(n_pairs, VALUE_SIZE, bias=False)       # W_v
            self.edit_direction = nn.Linear(d_model, VALUE_SIZE, bias=False)    # W_e
            self.edit_strength = nn.Linear(d_model, 1)                          # eps_t = sigmoid(this)
            self.output = nn.Linear(self.n_heads * VALUE_SIZE, d_input)        # W_o and b_o

        # Silent at the start (Theorem 3). With W_o = 0 and b_o = 0 every read is exactly zero, so the CTM
        # with the diary computes exactly what the CTM without it computes. With W_e = 0 no page is edited.
        # Learning still starts (Proposition 4): W_o gets a gradient at once, and W_v and W_e start
        # learning from the first update that makes W_o non-zero.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.edit_direction.weight)

        self.reset(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)

    # ---------------------------------------------------------------------------------------------------
    # A new episode
    # ---------------------------------------------------------------------------------------------------

    def reset(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """An empty diary for each of the `batch_size` episodes."""
        self.values = torch.zeros(batch_size, 0, VALUE_SIZE, device=device, dtype=dtype)       # v_j: (B, n, d_v)
        self.sketches = torch.zeros(batch_size, 0, SKETCH_UNITS, device=device, dtype=dtype)   # k_j: (B, n, M)

        # The by-number head: the page it holds, one per episode. None until the input first names a page.
        self.number_page = None                         # (B,) integers

        # The next and previous heads: the stamp each one has just read (its pointer) and the page number
        # it is looking for. None until the input first names a page.
        self.next_pointer = None                        # (B, 2K)
        self.next_page = None                           # (B,) integers
        self.previous_pointer = None                    # (B, 2K)
        self.previous_page = None                       # (B,) integers

        # The clock heads' light on every page from this tick's read. cross_off() edits through it.
        self.edit_light = None                          # (B, n)

    @property
    def n_pages(self) -> int:
        """n: the number of pages written so far (the same for every episode in the batch)."""
        return self.values.size(1)

    # ---------------------------------------------------------------------------------------------------
    # The input names a page
    # ---------------------------------------------------------------------------------------------------

    def open_at(self, page_numbers: torch.Tensor):
        """
        The input shows an index token i, "digit i", which names page i (assumption A2).

        The by-number head holds page i until the next index token, like a finger kept on the page. Both
        pointers are set to page i's exact stamp, so on its next read the next head looks at page i + 1
        and the previous head looks at page i - 1.

        page_numbers: (B,) one page number per episode
        """
        exact_stamp = self.clock.stamp(page_numbers)             # Phi(i): (B, 2K)

        self.number_page = page_numbers

        self.next_pointer = exact_stamp
        self.next_page = page_numbers
        self.previous_pointer = exact_stamp
        self.previous_page = page_numbers

    # ---------------------------------------------------------------------------------------------------
    # Read: every tick, before the synapses
    # ---------------------------------------------------------------------------------------------------

    def read(self, seen: torch.Tensor | None) -> torch.Tensor:
        """
        Read the diary through all four heads (thesis eq. 9), and the latest-page head if there is one:

            r_t = W_o [m_number; m_next; m_previous; m_content (; m_latest)] + b_o

        The CTM adds r_t to what it saw in its input, so reading the diary is like looking at the input
        again. Every head's read is a weighted average of pages that all have length sqrt(d_v), so
        ||r_t|| <= ||W_o|| sqrt(4 d_v) + ||b_o|| however many pages there are (Theorem 3; sqrt(5 d_v) with
        the latest-page head).

        seen:    (B, d_input) the CTM's view of its input this tick (o_t), or None on a tick with no input
        returns: (B, d_input)
        """
        if self.n_pages == 0:
            # An empty diary reads nothing, and its heads do not move.
            self.edit_light = None
            nothing_read = self.values.new_zeros(self.values.size(0), self.n_heads * VALUE_SIZE)
            return self.output(nothing_read)

        page_numbers = torch.arange(self.n_pages, device=self.values.device)
        page_stamps = self.clock.stamp(page_numbers)                       # Phi(0), ..., Phi(n-1): (n, 2K)

        number_read, number_light = self.read_by_number(page_stamps)
        next_read, next_light = self.read_next(page_stamps)
        previous_read, previous_light = self.read_previous(page_stamps)
        content_read, _ = self.read_by_content(seen)

        # Only the heads that hold a page by its stamp may cross it off (thesis Section 3.5). open_at() sets
        # all three of them at once, so before the input first names a page there is nothing to edit.
        if self.number_page is None:
            self.edit_light = None
        else:
            self.edit_light = number_light + next_light + previous_light     # (B, n)

        reads = [number_read, next_read, previous_read, content_read]
        if self.latest_head:
            latest_read, _ = self.read_latest()
            reads.append(latest_read)

        all_reads = torch.cat(reads, dim=-1)                               # (B, 4 d_v), or (B, 5 d_v)
        return self.output(all_reads)

    # ---------------------------------------------------------------------------------------------------
    # The four heads. Each one returns (what it read (B, d_v), its light on every page (B, n)).
    # A head with no page to look for reads nothing and shines no light.
    # ---------------------------------------------------------------------------------------------------

    def read_by_number(self, page_stamps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Open the diary at the page the question named:  score_j = kappa(i - j).

        Every wrong page scores at least m_H = 0.325 below the right one, by design, so the right page gets
        at least 0.9887 of the light in every diary of 2 to 50 pages (Theorem 5).
        """
        if self.number_page is None:
            return self.nothing()

        query = self.clock.stamp(self.number_page)                         # Phi(i): (B, 2K)
        scores = self.clock.agreement(query, page_stamps)                  # kappa(i - j): (B, n)
        light = self.spotlight(scores) * self.is_written(self.number_page)
        return self.look_up(light), light

    def read_next(self, page_stamps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The next-page head: one page on from the page it has just read."""
        if self.next_pointer is None:
            return self.nothing()

        read, light, pointer, page = self.flip(self.next_pointer, self.next_page, +1, page_stamps)
        self.next_pointer = pointer
        self.next_page = page
        return read, light

    def read_previous(self, page_stamps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The previous-page head: one page back from the page it has just read."""
        if self.previous_pointer is None:
            return self.nothing()

        read, light, pointer, page = self.flip(self.previous_pointer, self.previous_page, -1, page_stamps)
        self.previous_pointer = pointer
        self.previous_page = page
        return read, light

    def flip(self, pointer: torch.Tensor, page: torch.Tensor, direction: int, page_stamps: torch.Tensor):
        """
        Turn every hand one page on (direction +1) or one page back (direction -1) from the pointer, and
        look the result up (thesis eq. 7):    score_j = agreement( turn(pointer, direction), Phi(j) ).

        The pointer is the stamp of what was just read, pi = sum_j a_j Phi(j): a blend that is mostly the
        right page. Every flip snaps back onto a single page, so small slips never pile up. Starting from
        an exact stamp, every flip keeps at least 95% of the light on the right page (Theorem 9).

        `page` is the number of the page the head is looking for. It is used only to stay silent when that
        page does not exist. Once a head runs off either end of the diary it stays silent until the input
        names a page again: it moves one page per tick, and the diary grows by at most one page per tick.

        Returns (what it read, its light, the new pointer, the page it now looks for).
        """
        query = self.clock.turn(pointer, direction)                        # R_{+1 or -1} pi: (B, 2K)
        page = page + direction
        scores = self.clock.agreement(query, page_stamps)                  # (B, n)
        light = self.spotlight(scores)
        new_pointer = light @ page_stamps                                  # the stamp just read: (B, 2K)

        light = light * self.is_written(page)
        return self.look_up(light), light, new_pointer, page

    def read_by_content(self, seen: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Leaf through the diary for a page that looked like what is seen now:  score_j = <cue, k_j>, the
        share of sketch units the cue and the page have in common (Theorem 12). With no input, no cue.
        """
        if seen is None:
            return self.nothing()

        cue = self.sketch(seen)                                            # q_t: (B, M)
        scores = torch.einsum("bm,bnm->bn", cue, self.sketches)            # overlaps in [0, 1]: (B, n)
        light = self.spotlight(scores)
        return self.look_up(light), light

    def read_latest(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The latest-page head (optional, not in the thesis): the page of the event that has just ended, read
        exactly. It gives the CTM a place to carry its running result from one question part to the next, so it
        tests whether the errors that remain come from having nowhere to keep that result. Like the content
        head, it never edits.
        """
        light = self.values.new_zeros(self.values.size(0), self.n_pages)
        light[:, -1] = 1.0
        return self.values[:, -1], light

    # ---------------------------------------------------------------------------------------------------
    # Tools the heads share
    # ---------------------------------------------------------------------------------------------------

    def spotlight(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Turn scores into light that adds up to one (thesis eq. 6):

            a_j = exp( lambda(n) score_j ) / sum_l exp( lambda(n) score_l ),       lambda(n) = c log(1 + n)

        The focus grows like log n because more pages compete for the light. A fixed focus loses its grip
        as pages pile up (Proposition 6).

        scores:  (B, n)
        returns: (B, n)
        """
        focus = FOCUS * math.log(1 + self.n_pages)
        return torch.softmax(focus * scores, dim=-1)

    def look_up(self, light: torch.Tensor) -> torch.Tensor:
        """What a spotlight reads: m = sum_j a_j v_j.     light: (B, n)  ->  (B, d_v)"""
        return torch.einsum("bn,bnv->bv", light, self.values)

    def is_written(self, page_numbers: torch.Tensor) -> torch.Tensor:
        """1.0 for every episode whose page `page_numbers` has been written, else 0.0.     (B,)  ->  (B, 1)"""
        written = (page_numbers >= 0) & (page_numbers < self.n_pages)
        return written.to(self.values.dtype).unsqueeze(-1)

    def nothing(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The read and the light of a head with no page to look for: all zeros."""
        batch_size = self.values.size(0)
        no_read = self.values.new_zeros(batch_size, VALUE_SIZE)
        no_light = self.values.new_zeros(batch_size, self.n_pages)
        return no_read, no_light

    # ---------------------------------------------------------------------------------------------------
    # Cross off: every tick, after the neurons
    # ---------------------------------------------------------------------------------------------------

    def cross_off(self, neurons: torch.Tensor):
        """
        Edit the pages the clock heads are looking at (thesis eq. 10):

            v_j <- LN( v_j + a_j * eps_t * W_e z_t ),     a_j = this tick's light on page j from the clock heads

        Edits change only what is written. Stamps and sketches are never edited, so every finding guarantee
        still holds after any number of edits (Theorem 14). The spotlight that edits a page is the one that
        found it, so an edit made through the by-number head moves any other page by at most 2.3% of the
        edit's size (Theorem 14; the worst case is a diary of 2 pages).

        neurons: (B, d_model) the CTM's neuron outputs z_t at this tick
        """
        if self.edit_light is None:
            return

        strength = torch.sigmoid(self.edit_strength(neurons))            # eps_t in [0, 1]: (B, 1)
        edit = strength * self.edit_direction(neurons)                    # e_t: (B, d_v)

        change = self.edit_light.unsqueeze(-1) * edit.unsqueeze(1)        # a_j e_t on every page: (B, n, d_v)
        self.values = F.layer_norm(self.values + change, (VALUE_SIZE,), eps=LAYER_NORM_EPS)

        # This light belongs to this tick's read. Never edit through it twice.
        self.edit_light = None

    # ---------------------------------------------------------------------------------------------------
    # Write: at the last tick of every event
    # ---------------------------------------------------------------------------------------------------

    def write(self, event_synchronisation: torch.Tensor, seen: torch.Tensor):
        """
        Add one page for the event that has just ended (thesis eq. 2):

            v_e = LN(W_v u_e),     stamp Phi(e),     sketch k_e of what the event looked like

        LN has no learned scale or shift, so every page has the same length, sqrt(d_v). The stamp is not
        stored: page e's stamp is always Phi(e), and it is recomputed when needed. Every page is written at
        full strength; nothing decides how important an event was (thesis Section 3.7).

        event_synchronisation: (B, P) u_e, from event_sync()
        seen:                  (B, d_input) the CTM's view of the event's input at its last tick
        """
        raw_value = self.write_value(event_synchronisation)                               # W_v u_e: (B, d_v)
        value = F.layer_norm(raw_value, (VALUE_SIZE,), eps=LAYER_NORM_EPS)                # v_e: (B, d_v)
        sketch = self.sketch(seen)                                                        # k_e: (B, M)

        self.values = torch.cat([self.values, value.unsqueeze(1)], dim=1)
        self.sketches = torch.cat([self.sketches, sketch.unsqueeze(1)], dim=1)
