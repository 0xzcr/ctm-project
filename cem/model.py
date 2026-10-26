"""
The upstream Q&A-MNIST CTM with the diary attached (thesis Figure 1).

Every tick does six things, in this order:

    1. LOOK       attention over the input, with a query from the synchronisation              (upstream)
    2. READ       the four heads read the diary, and the read r_t is added to what was seen
    3. THINK      the synapses mix what was seen with the neurons' latest outputs, then every
                  neuron passes its private history through its own tiny network               (upstream)
    4. CROSS OFF  the clock heads may edit the pages they are looking at
    5. WRITE      at the last tick of every event, one new page
    6. ANSWER     a prediction from the output synchronisation                                 (upstream)

The steps marked (upstream) are the forward pass of the pinned upstream code, unchanged. With
use_diary=False this class is exactly the upstream model, with one-hot digits instead of images (or, with
images=True, with the upstream image backbone and real MNIST images).
"""
import numpy as np
import torch
import torch.nn as nn

from . import upstream  # noqa: F401  (puts the upstream code on the import path)
from .memory import ClockedEventMemory, event_sync
from models.ctm_qamnist import ContinuousThoughtMachineQAMNIST  # noqa: E402  (upstream)


class CTMWithDiary(ContinuousThoughtMachineQAMNIST):

    def __init__(self, *args, use_diary: bool = True, diary_seed: int = 0, latest_head: bool = False,
                 images: bool = False, n_digit_classes: int = 10, **kwargs):
        super().__init__(*args, **kwargs)

        # Digits arrive as one-hot vectors instead of images, which separates memory from perception
        # (thesis Section 2.2). A 1x1 convolution turns each one-hot vector into the CTM's input features.
        # With images=True the digits are real MNIST images, and the upstream image backbone stays.
        if not images:
            self.backbone_digit = nn.Conv2d(n_digit_classes, self.d_input, kernel_size=1)
        self.backbone_frozen = False

        if use_diary:
            self.diary = ClockedEventMemory(
                n_pairs=self.synch_representation_size_out,
                d_model=self.d_model,
                d_input=self.d_input,
                seed=diary_seed,
                latest_head=latest_head,
            )
        else:
            self.diary = None

    # ---------------------------------------------------------------------------------------------------
    # A pretrained image backbone can be frozen (train.py --pretrain-backbone)
    # ---------------------------------------------------------------------------------------------------

    def freeze_backbone(self):
        """Stop the image backbone from learning: its weights and its batch-norm statistics stay as they are."""
        self.backbone_digit.requires_grad_(False)
        self.backbone_frozen = True
        self.backbone_digit.eval()

    def train(self, mode: bool = True):
        """nn.Module.train(), except that a frozen backbone stays in eval mode, so its batch norm stops learning."""
        super().train(mode)
        if self.backbone_frozen:
            self.backbone_digit.eval()
        return self

    # ---------------------------------------------------------------------------------------------------
    # Where the events are. The input marks them (assumption A1): every digit is shown for
    # iterations_per_digit ticks and every part of the question for iterations_per_question_part ticks.
    # ---------------------------------------------------------------------------------------------------

    def is_last_tick_of_event(self, tick: int, digit_ticks: int, question_ticks: int) -> bool:
        """True at the last tick of a digit or of a question part. The answer ticks show nothing: no event."""
        if tick < digit_ticks:
            ticks_into_digits = tick + 1
            return ticks_into_digits % self.iterations_per_digit == 0

        if tick < digit_ticks + question_ticks:
            ticks_into_question = tick - digit_ticks + 1
            return ticks_into_question % self.iterations_per_question_part == 0

        return False

    def event_length(self, tick: int, digit_ticks: int) -> int:
        """R: how many ticks the event at this tick lasts."""
        if tick < digit_ticks:
            return self.iterations_per_digit
        return self.iterations_per_question_part

    def page_named_by_input(self, tick: int, digit_ticks: int, question_ticks: int, z: torch.Tensor):
        """
        The page an index token names at this tick (assumption A2), or None when no index token is shown.
        The digits are the first events, so index token i ("digit i") names page i.
        """
        _, is_question, _ = self.determine_step_type(digit_ticks, question_ticks, tick)
        if not is_question:
            return None

        is_index, _ = self.determine_index_operator_step_type(digit_ticks, tick)
        if not is_index:
            return None

        return z[:, tick - digit_ticks]                                   # (B,) page numbers

    # ---------------------------------------------------------------------------------------------------
    # One episode
    # ---------------------------------------------------------------------------------------------------

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        """
        x: (B, digit_ticks, 10, 1, 1)  one-hot digits, each repeated for the ticks it is shown
                                       (or (B, digit_ticks, 1, 28, 28) images, with images=True)
        z: (B, question_ticks)         question tokens: an index (0, 1, ...) or an operator (-1 plus, -2 minus)

        returns predictions (B, 10, ticks), certainties (B, 2, ticks), and the last output synchronisation
        """
        batch_size = x.size(0)
        device = x.device
        digit_ticks = x.size(1)
        question_ticks = z.size(1)
        total_ticks = digit_ticks + question_ticks + self.iterations_for_answering

        # --- The CTM's starting state (upstream) ---
        state_trace = self.start_trace.unsqueeze(0).expand(batch_size, -1, -1)              # (B, neurons, history)
        activated_state = self.start_activated_state.unsqueeze(0).expand(batch_size, -1)    # z: (B, neurons)
        predictions = torch.empty(batch_size, self.out_dims, total_ticks, device=device, dtype=x.dtype)
        certainties = torch.empty(batch_size, 2, total_ticks, device=device, dtype=x.dtype)

        # --- The synchronisation tallies (upstream) ---
        self.decay_params_action.data = torch.clamp(self.decay_params_action, 0, 15)
        self.decay_params_out.data = torch.clamp(self.decay_params_out, 0, 15)
        fade_action = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(batch_size, 1)   # r: (B, pairs)
        fade_out = torch.exp(-self.decay_params_out).unsqueeze(0).repeat(batch_size, 1)
        alpha_action, beta_action = None, None
        _, alpha_out, beta_out = self.compute_synchronisation(activated_state, None, None, fade_out, synch_type="out")

        # --- The diary starts empty. Pages are made from the output tallies, the ones the CTM reads its
        #     answers from, so remember them as they stand just before the first event begins. ---
        if self.diary is not None:
            self.diary.reset(batch_size, device, x.dtype)
        alpha_before_event = alpha_out
        beta_before_event = beta_out

        previous_input, previous_kv = None, None
        for tick in range(total_ticks):
            is_digit, is_question, is_answer = self.determine_step_type(digit_ticks, question_ticks, tick)
            kv, previous_input = self.get_kv_for_step(digit_ticks, question_ticks, tick, x, z, previous_input, previous_kv)
            previous_kv = kv

            sync_action, alpha_action, beta_action = self.compute_synchronisation(
                activated_state, alpha_action, beta_action, fade_action, synch_type="action")

            # 1. LOOK at the input (upstream)
            if is_digit:
                query = self.q_proj(sync_action).unsqueeze(1)
                seen, _ = self.attention(query, kv, kv, average_attn_weights=False, need_weights=True)
                seen = seen.squeeze(1)                                                       # o_t: (B, d_input)
            else:
                seen = kv.squeeze(1)

            # 2. READ the diary, and add what was read to what was seen
            view = seen
            if self.diary is not None:
                named_page = self.page_named_by_input(tick, digit_ticks, question_ticks, z)
                if named_page is not None:
                    self.diary.open_at(named_page)

                # The answer ticks show no input, so the content head has no cue on them.
                cue = None if is_answer else seen
                view = seen + self.diary.read(cue)

            # 3. THINK (upstream)
            synapse_input = torch.concatenate((view, activated_state), dim=-1)
            state = self.synapses(synapse_input)
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)
            activated_state = self.trace_processor(state_trace)
            sync_out, alpha_out, beta_out = self.compute_synchronisation(
                activated_state, alpha_out, beta_out, fade_out, synch_type="out")

            if self.diary is not None:
                # 4. CROSS OFF, through the light of this tick's read
                self.diary.cross_off(activated_state)

                # 5. WRITE one page at the last tick of every event
                if self.is_last_tick_of_event(tick, digit_ticks, question_ticks):
                    event_length = self.event_length(tick, digit_ticks)
                    u = event_sync(alpha_out, beta_out, alpha_before_event, beta_before_event, fade_out, event_length)
                    self.diary.write(u, seen)

                    # The next event starts on the next tick.
                    alpha_before_event = alpha_out
                    beta_before_event = beta_out

            # 6. ANSWER (upstream)
            prediction = self.output_projector(sync_out)
            predictions[..., tick] = prediction
            certainties[..., tick] = self.compute_certainty(prediction)

        return predictions, certainties, sync_out


def build_model(use_diary: bool = True, seed: int = 0, ticks_per_item: int = 3, d_model: int = 64,
                latest_head: bool = False, images: bool = False) -> CTMWithDiary:
    """
    The Q&A-MNIST CTM of our earlier experiments (the "CTM" row of thesis Box 1), with or without the diary.

    `seed` fixes the CTM's initial weights and its choice of synchronised neuron pairs. The diary draws its
    own random numbers, so with the same seed both variants start from exactly the same CTM.
    `d_model` sizes the plain CTM up to the diary's number of parameters (107 neurons: 51,164 against 50,925).
    `latest_head` gives the diary its optional fifth head, and `images` shows real MNIST images.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)            # upstream picks the synchronised neuron pairs with numpy's global generator

    model = CTMWithDiary(
        iterations=1,                                   # not used by the Q&A-MNIST model
        d_model=d_model,                                # neurons
        d_input=32,                                     # size of the CTM's view of its input
        heads=2,                                        # attention heads for looking at the input
        n_synch_out=16,                                 # 16 neurons -> 136 pairs for the answer (and the pages)
        n_synch_action=16,                              # 16 neurons -> 136 pairs for the attention query
        synapse_depth=1,
        memory_length=3,                                # each neuron's private history, in ticks
        deep_nlms=True,
        memory_hidden_dims=8,
        do_layernorm_nlm=False,
        out_dims=10,                                    # the answer is a digit
        iterations_per_digit=ticks_per_item,
        iterations_per_question_part=ticks_per_item,
        iterations_for_answering=ticks_per_item,
        prediction_reshaper=[-1],
        neuron_select_type="random",
        use_diary=use_diary,
        diary_seed=seed + 1_000_000,
        latest_head=latest_head,
        images=images,
    )
    create_lazy_weights(model, ticks_per_item, images)
    return model


def create_lazy_weights(model: CTMWithDiary, ticks_per_item: int, images: bool = False):
    """
    The upstream CTM creates some of its layers lazily, on its first forward pass. Run one tiny episode
    (the digit 0, or a blank image, then the question "index 0") so that every weight exists before an
    optimiser is made.
    """
    if images:
        one_digit = torch.zeros(1, ticks_per_item, 1, 28, 28)
    else:
        one_digit = torch.zeros(1, ticks_per_item, 10, 1, 1)
        one_digit[:, :, 0] = 1.0
    question = torch.zeros(1, ticks_per_item, dtype=torch.long)

    # In eval mode, so that the image backbone's batch norm does not learn from this made-up episode.
    model.eval()
    with torch.no_grad():
        model(one_digit, question)
    model.train()
