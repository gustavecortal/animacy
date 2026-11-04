"""The model: a small causal transformer over displacement tokens with a Gaussian mixture head.

A token is one 30 Hz displacement divided by the corpus scale. The model reads tokens d[0..n-1] and, at every
position i, predicts the distribution of d[i+1] as a mixture of bivariate Gaussians. An animal that sees its
neighbours also gets, added to each token's embedding, a pooled description of its nearest neighbours.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

SEED = 20261014
SIGMA_FLOOR = 0.001  # token units
RHO_LIMIT = 0.999


@dataclass(frozen=True)
class Config:
    layers: int = 4
    width: int = 128
    heads: int = 4
    ff_width: int = 512
    window: int = 256  # each attention sees the current token and at most window - 1 before it
    components: int = 5
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    neighbours: int = 0  # most neighbours an animal sees, 0 for the model that sees none
    social_width: int = 64
    # Per layer, per head, how many of the last tokens that head attends to, at most `window`, or None for
    # `window` everywhere. Set at export only: a trained model whose heads use only their last few steps
    # runs more cheaply in the browser with them cut there.
    head_windows: tuple | None = None


def rope_frequencies(head_dim: int, base: float, device=None, dtype=None) -> torch.Tensor:
    """One rotation frequency per adjacent pair of head dimensions."""
    return base ** (-torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim)


def rope_angles(n: int, head_dim: int, base: float, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin for positions 0..n-1."""
    angles = torch.arange(n, device=like.device, dtype=like.dtype)[:, None] * rope_frequencies(head_dim, base, like.device, like.dtype)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent pairs (x[2k], x[2k+1]) of the last dimension by each position's angle."""
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).flatten(-2)


def local_causal_mask(n: int, window: int, device) -> torch.Tensor:
    """True where query i may attend key j: i - window < j <= i."""
    i = torch.arange(n, device=device)[:, None]
    j = torch.arange(n, device=device)[None, :]
    return (j <= i) & (j > i - window)


class Block(nn.Module):
    def __init__(self, c: Config, layer: int = 0):
        super().__init__()
        self.heads = c.heads
        self.window = c.window
        self.head_windows = tuple(c.head_windows[layer]) if c.head_windows else None
        self.attn_norm = nn.RMSNorm(c.width, eps=c.norm_eps)
        self.qkv = nn.Linear(c.width, 3 * c.width, bias=False)
        self.out = nn.Linear(c.width, c.width, bias=False)
        self.q_norm = nn.RMSNorm(c.width // c.heads, eps=c.norm_eps)  # QK-norm bounds attention logits
        self.k_norm = nn.RMSNorm(c.width // c.heads, eps=c.norm_eps)
        self.ff_norm = nn.RMSNorm(c.width, eps=c.norm_eps)
        self.up = nn.Linear(c.width, c.ff_width, bias=False)
        self.down = nn.Linear(c.ff_width, c.width, bias=False)
        # Each residual branch starts at zero, so the untrained network is the identity plus the head.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.down.weight)

    def forward(self, x, cos, sin, mask):
        b, n, w = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).view(b, n, 3, self.heads, w // self.heads).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(self.q_norm(q), cos, sin), apply_rope(self.k_norm(k), cos, sin)
        # Up to `window` tokens the local mask is plain causal, which runs the fused kernel.
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
        return self.feed(x + self.out(a.transpose(1, 2).reshape(b, n, w)))

    def feed(self, x):
        return x + self.down(F.gelu(self.up(self.ff_norm(x))))

    def step(self, x, cos, sin, keys, values, slot: int, count: int):
        """One token per row of x [b, width], its key and value written to cache `slot`, attending to the
        first `count` slots of keys and values [b, heads, window, head_dim]. Keys are stored after RoPE at
        their own positions, so scores depend on distance, as in forward."""
        b, w = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).view(b, 3, self.heads, w // self.heads).unbind(1)
        q, k = apply_rope(self.q_norm(q), cos, sin), apply_rope(self.k_norm(k), cos, sin)
        keys[:, :, slot], values[:, :, slot] = k, v
        mask = None
        if self.head_windows:
            # A slot's age is how many tokens ago it was written, the ring's slots wrapping round.
            age = (slot - torch.arange(count, device=x.device)) % self.window
            mask = age[None, None, :] < torch.tensor(self.head_windows, device=x.device)[:, None, None]
        a = F.scaled_dot_product_attention(q[:, :, None], keys[:, :, :count], values[:, :, :count], attn_mask=mask)[:, :, 0]
        return self.feed(x + self.out(a.reshape(b, w)))


class Neighbours(nn.Module):
    """What an animal sees of its neighbours, pooled into one vector added to its token embedding.

    Each neighbour is described by the animal's own token, the direction and log distance to the neighbour,
    and the neighbour's last displacement, all in token units. One small network encodes every neighbour, and
    a softmax over the visible ones weights them, so any number from 0 works. The output projection starts at
    zero, so an untrained pathway leaves the model that sees no neighbours unchanged.
    """

    def __init__(self, c: Config):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(7, c.social_width), nn.GELU(), nn.Linear(c.social_width, c.social_width), nn.GELU())
        self.score = nn.Linear(c.social_width, 1)
        self.out = nn.Linear(c.social_width, c.width, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, tokens, relative, displacement, visible):
        """tokens [b, n, 2], relative position and displacement [b, n, k, 2], visible [b, n, k]."""
        distance = relative.norm(dim=-1, keepdim=True).clamp_min(0.1)
        features = torch.cat([tokens[..., None, :].expand_as(relative), relative / distance, distance.log(), displacement], -1)
        h = self.encode(features)
        weights = self.score(h)[..., 0].masked_fill(~visible, -1e4).softmax(-1) * visible
        return self.out((weights[..., None] * h).sum(-2))


def neighbour_inputs(seen: torch.Tensor, step: torch.Tensor, me: torch.Tensor, k: int, scale: float) -> tuple:
    """What each animal sees of its k nearest neighbours at one instant, as Neighbours takes it.

    seen and step [..., animals, 2] are every animal's latest observed position and last step, NaN where
    unobserved, and me [...] is the animal's own index. Returns relative positions and displacements
    [..., k, 2] in token units, nearest first and zero where nothing is visible, and visible [..., k]."""
    own = seen.gather(-2, me[..., None, None].expand(*me.shape, 1, 2))
    relative, displacement = (seen - own) / scale, step / scale
    others = torch.arange(seen.shape[-2], device=me.device) != me[..., None]
    usable = others & relative.isfinite().all(-1) & displacement.isfinite().all(-1)
    # A stable sort keeps equally distant animals in index order, as the browser's sort does.
    distance, order = relative.norm(dim=-1).masked_fill(~usable, torch.inf).sort(dim=-1, stable=True)
    distance, order = distance[..., :k], order[..., :k]
    visible = distance.isfinite()
    pick = lambda v: v.gather(-2, order[..., None].expand(*order.shape, 2)).nan_to_num() * visible[..., None]
    return pick(relative), pick(displacement), visible


class Model(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.config = c
        self.embed = nn.Linear(2, c.width, bias=False)
        self.blocks = nn.ModuleList(Block(c, layer) for layer in range(c.layers))
        self.norm = nn.RMSNorm(c.width, eps=c.norm_eps)
        self.head = nn.Linear(c.width, 6 * c.components, bias=False)
        self.social = Neighbours(c) if c.neighbours else None

    def forward(self, tokens: torch.Tensor, neighbours: tuple | None = None) -> torch.Tensor:
        """Raw head output [batch, n, 6 * components] for tokens [batch, n, 2], with `neighbours`
        (relative, displacement, visible) as Neighbours takes them, or None for none seen.

        Token i is the step to the animal's position at row i + 1, and neighbour input i is what was
        observed by that row's instant, so the prediction of the next step never sees a later one."""
        n = tokens.shape[1]
        c = self.config
        cos, sin = rope_angles(n, c.width // c.heads, c.rope_base, tokens)
        mask = local_causal_mask(n, c.window, tokens.device) if n > c.window else None
        masks = [mask] * c.layers
        if c.head_windows:  # one mask per head, [heads, n, n]
            masks = [torch.stack([local_causal_mask(n, w, tokens.device) for w in row]) for row in c.head_windows]
        x = self.embed(tokens)
        if neighbours is not None:
            x = x + self.social(tokens, *neighbours)
        for block, mask in zip(self.blocks, masks):
            x = block(x, cos, sin, mask)
        # The mixture's means and sigmas need finer steps than bfloat16's, so the head runs in float32.
        with torch.autocast(tokens.device.type, enabled=False):
            return self.head(self.norm(x.to(self.head.weight.dtype)))

    def cache(self, batch: int) -> dict:
        """An empty cache for `step`: each layer's last `window` keys and values, as the browser keeps."""
        c, p = self.config, self.head.weight
        shape = (c.layers, 2, batch, c.heads, c.window, c.width // c.heads)
        # RoPE frequencies in float64, so a long run's angles lose no precision before the cast.
        return {"kv": torch.zeros(shape, device=p.device, dtype=p.dtype), "position": 0,
                "frequencies": rope_frequencies(c.width // c.heads, c.rope_base, dtype=torch.float64)}  # fmt: skip

    def step(self, token: torch.Tensor, cache: dict, neighbours: tuple | None = None) -> torch.Tensor:
        """Consume one token [batch, 2], with what each animal sees of its neighbours at that instant, and
        return the raw head output [batch, 6 * components] for the next. Equal to forward's last output
        over every token consumed since `cache` was made."""
        c = self.config
        position = cache["position"]
        angles = position * cache["frequencies"]
        cos, sin = (f(angles).to(token.device, token.dtype) for f in (torch.cos, torch.sin))
        x = self.embed(token)
        if neighbours is not None:
            x = x + self.social(token[:, None], *(n[:, None] for n in neighbours))[:, 0]
        slot, count = position % c.window, min(position + 1, c.window)
        for block, (keys, values) in zip(self.blocks, cache["kv"]):
            x = block.step(x, cos, sin, keys, values, slot, count)
        cache["position"] = position + 1
        return self.head(self.norm(x))


def load_checkpoint(path, neighbours: int | None = None) -> tuple[Model, dict]:
    """A checkpoint's model on the CPU, and the checkpoint. With `neighbours`, the model that sees that
    many, its pathway new where the checkpoint has none."""
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    config = {k: v for k, v in ckpt["config"].items() if k != "walls"}  # checkpoints of the site's research code name an unused wall input
    if ckpt["config"].get("walls"):
        raise ValueError(f"{path} senses walls, which this model does not")
    if neighbours is not None:
        config["neighbours"] = neighbours
    model = Model(Config(**config))
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if unexpected or any(not (neighbours and k.startswith("social.")) for k in missing):
        raise ValueError(f"{path} does not fit this model: missing {missing}, unexpected {unexpected}")
    return model, ckpt


def decode(raw: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Mixture log-weights, means, sigmas and correlations from raw head output, each [..., components]."""
    raw = raw.unflatten(-1, (-1, 6))
    log_w = raw[..., 0].log_softmax(-1)
    sigma = F.softplus(raw[..., 3:5]) + SIGMA_FLOOR
    rho = RHO_LIMIT * torch.tanh(raw[..., 5])
    return log_w, raw[..., 1], raw[..., 2], sigma[..., 0], sigma[..., 1], rho


def nll(raw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of target displacements [..., 2] under the mixture, per position."""
    log_w, mx, my, sx, sy, rho = decode(raw)
    zx = (target[..., :1] - mx) / sx
    zy = (target[..., 1:] - my) / sy
    one_minus = 1 - rho * rho
    log_p = -math.log(2 * math.pi) - sx.log() - sy.log() - 0.5 * one_minus.log() - (zx * zx - 2 * rho * zx * zy + zy * zy) / (2 * one_minus)
    return -torch.logsumexp(log_w + log_p, dim=-1)


def mixture_mean(raw: torch.Tensor) -> torch.Tensor:
    """The mixture's expected displacement [..., 2]."""
    log_w, mx, my, *_ = decode(raw)
    w = log_w.exp()
    return torch.stack([(w * mx).sum(-1), (w * my).sum(-1)], -1)


def sample(raw: torch.Tensor, generator: torch.Generator, temperature: float = 1.0) -> torch.Tensor:
    """One displacement token per row of raw head output [batch, 6 * components], on the CPU. The
    temperature divides the mixture's logits only, leaving each component's shape alone, as on the page."""
    log_w, mx, my, sx, sy, rho = (t.double().cpu() for t in decode(raw))
    k = torch.multinomial((log_w / temperature).softmax(-1), 1, generator=generator)
    pick = lambda t: t.gather(-1, k)[:, 0]
    z = torch.randn(len(k), 2, generator=generator, dtype=torch.float64)
    x = pick(mx) + pick(sx) * z[:, 0]
    y = pick(my) + pick(sy) * (pick(rho) * z[:, 0] + (1 - pick(rho) ** 2).sqrt() * z[:, 1])
    return torch.stack([x, y], dim=-1)


def fold(p: torch.Tensor, lo, hi) -> torch.Tensor:
    """Reflect coordinates into the box [lo, hi], handling any number of crossings."""
    width = hi - lo
    q = torch.remainder(p - lo, 2 * width)
    return lo + width - (q - width).abs()
