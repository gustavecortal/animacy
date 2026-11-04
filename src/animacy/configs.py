"""The three models of the page and how each was made."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    name: str
    steps: int
    lr: float  # peak learning rate
    mean_weight: float  # weight of the squared error of the mixture's mean step, beside the likelihood
    neighbours: bool  # trained on the scenes, seeing up to 16 nearest neighbours, or on the corpus alone


# Every model is trained in three stages, each starting from the last one's weights: alone on the corpus,
# then with its neighbours, then 2,000 more steps with the same settings from a fresh warmup.
STAGES = (
    Stage("solo", 3000, 2e-3, 0.0, False),
    Stage("group", 2000, 5e-4, 10.0, True),
    Stage("group-more", 2000, 5e-4, 10.0, True),
)
BATCH = 512  # windows of LENGTH tokens per step
LENGTH = 256  # tokens per window, and each layer's attention window
NEIGHBOURS = 16  # most neighbours an animal sees
KEEP = (0, 1, 2, 4, 8, 16)  # nearest neighbours a training batch shows, drawn uniformly per batch
# Each stage's learning rate rises linearly over its first WARMUP_FRACTION of steps, then follows a cosine down
# to FINAL_LR_FRACTION of its peak. AdamW's weight decay applies to the weight matrices only.
WARMUP_FRACTION = 0.1
FINAL_LR_FRACTION = 0.1
WEIGHT_DECAY = 0.1


@dataclass(frozen=True)
class Recipe:
    families: tuple[str, ...]
    body_lengths: bool  # positions rescaled so every session's animals are one length (data.in_body_lengths)
    # Per layer, per head, the tokens each head attends to in the exported model: the smallest of 16, 32, 64
    # and 256 steps that holds 90% of that head's attention on validation windows.
    head_windows: tuple[tuple[int, ...], ...]
    # The page's default sampling temperature: below 1 where free-running animals would move faster than
    # recorded ones.
    temperature: float


MODELS = {
    "fish": Recipe(("fish",), False, ((256, 256, 256, 256), (16, 16, 16, 16), (16, 16, 16, 16), (16, 16, 256, 64)), 1.0),
    "flies": Recipe(("flies",), True, ((256, 256, 256, 256), (16, 16, 16, 32), (16, 16, 32, 256), (256, 16, 16, 16)), 0.75),
    "both": Recipe(("fish", "flies"), True, ((256, 256, 256, 16), (16, 16, 64, 16), (16, 16, 16, 16), (256, 16, 256, 16)), 0.85),
}
