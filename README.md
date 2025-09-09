# Evolutionary-Gradient-Descent

Want a crisp way to say that? Here are a few options you can drop into a README, paper, or slide.

# Short tagline

**Evolutionary Gradient Descent (EGD):** SGD that periodically explores via evolutionary mutations—CPU-paged so memory stays small.

# One-liner

We introduce **Evolutionary Gradient Descent (EGD)**, an optimizer that runs standard SGD most steps and, at intervals, spawns a small CPU-paged population of mutated parameter sets, selects the fittest by loss, recombines them (e.g., mean), and resumes SGD—delivering exploration without blowing up VRAM.

# 3–4 sentence abstract

**Evolutionary Gradient Descent (EGD)** combines first-order updates with intermittent evolutionary search. During training, we interleave ordinary SGD steps with short “evolution cycles” that evaluate a population of mutated models, select elites by validation loss, and recombine them to form a new parameter state. To keep memory footprint modest, population candidates and elites are **offloaded to pinned CPU memory** and only the current candidate is copied to GPU for evaluation. This design adds global exploration while preserving the speed and simplicity of SGD.

# Key bullets

* **Exploit + explore:** SGD every step; periodic evolutionary mutation, selection, and recombination.
* **CPU offload:** Population and elites live on pinned CPU; only the active candidate is moved to GPU.
* **Lean memory:** Streaming top-k elites avoids storing the whole population.
* **Drop-in:** Works with momentum/Nesterov/weight decay; closure is forward-only.
* **Configurable:** Population size, elite fraction, σ schedule, mean or softmax recombination; DDP-friendly loss reduction.

# README snippet

```markdown
### Evolutionary Gradient Descent (EGD)

EGD is SGD with periodic evolutionary exploration. Every *N* steps, it
(1) snapshots params, (2) creates a small population of Gaussian-mutated candidates,
(3) evaluates their loss, (4) keeps the elites, (5) recombines (mean or softmax),
and (6) resumes SGD from the recombined params.

To keep VRAM small, candidates and elites are **CPU-paged** (pinned memory). Only
the current candidate is copied to GPU for a forward pass, then evicted.

**Why EGD?** It escapes poor local minima/plateaus while preserving SGD’s speed.
```

# Tweet-length

We built **Evolutionary Gradient Descent**: SGD + periodic evolutionary search, with population offloaded to CPU so memory stays tiny. Explore globally, train locally. 🧬⚙️

If you want this adapted into a paper title + subtitle, or a 30-sec elevator pitch for a talk, say how formal you want it and I’ll tailor it.

# PSEUDOCODE

awesome — here’s a **drop-in EvoSGD** with an `offload_population_to_cpu` toggle (plus streaming elites, optional softmax recombine, DDP-friendly `loss_reduce`, deterministic `rng`, and AMP-safe touches). It defaults to **aggressive CPU offload**.

```python
import math
import torch
from torch.optim import Optimizer

class EvoSGD(Optimizer):
    """
    SGD with momentum/Nesterov/weight_decay + periodic evolutionary step.

    Key knobs:
      - evo_interval: every N steps, run evo cycle
      - pop_size: population size (includes current params as one)
      - elite_frac: fraction of best candidates to keep (recombine)
      - sigma: mutation std (relative to param RMS); decays by sigma_decay
      - recombine: "mean" or "softmax" (weights by exp(-loss))
      - loss_reduce: callable(loss_tensor) -> float (e.g., DDP all-reduce)
      - rng: optional torch.Generator for determinism
      - offload_population_to_cpu: keep base & elites on pinned CPU; copy candidates to GPU only to eval
      - mutate_filter: optional callable(param) -> bool to choose which params to mutate (e.g., only head)

    Notes:
      - Evo cycle uses a forward-only closure (no backward, wrapped in no_grad).
      - Normal SGD uses gradients computed by you before calling .step().
      - For BatchNorm/Dropout, make your closure set model.eval() and restore mode.
    """
    def __init__(
        self,
        params,
        lr=1e-2,
        momentum=0.0,
        nesterov=False,
        weight_decay=0.0,
        # evolution
        evo_interval=200,
        pop_size=8,
        elite_frac=0.25,
        sigma=0.02,
        sigma_decay=0.98,
        recombine="mean",  # "mean" | "softmax"
        reset_momentum_on_evo=True,
        loss_reduce=None,       # e.g., lambda t: allreduce_mean(t)
        rng: torch.Generator | None = None,
        sigma_schedule=None,    # optional: lambda step, sigma -> new_sigma
        offload_population_to_cpu=True,
        mutate_filter=None,     # optional: lambda p -> bool
    ):
        if lr <= 0: raise ValueError("Invalid lr")
        if momentum < 0: raise ValueError("Invalid momentum")
        if nesterov and momentum <= 0: raise ValueError("Nesterov requires momentum > 0")
        if pop_size < 2: raise ValueError("pop_size must be >= 2")
        if not (0 < elite_frac <= 0.5): raise ValueError("elite_frac in (0, 0.5]")
        if sigma <= 0: raise ValueError("sigma must be > 0")
        if recombine not in ("mean", "softmax"): raise ValueError("Unsupported recombine")

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self._t = 0
        self.evo_interval = evo_interval
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.sigma = float(sigma)
        self.sigma_decay = float(sigma_decay)
        self.recombine = recombine
        self.reset_momentum_on_evo = reset_momentum_on_evo
        self.loss_reduce = loss_reduce
        self.rng = rng
        self.sigma_schedule = sigma_schedule
        self.offload_population_to_cpu = offload_population_to_cpu
        self.mutate_filter = mutate_filter

    @torch.no_grad()
    def step(self, closure=None):
        """Run one SGD step; every evo_interval steps, do evolutionary search.

        closure: callable returning current loss (forward-only; no backward).
                 Required on evo steps.
        """
        # --- 1) Normal SGD (uses precomputed grads) ---
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if wd != 0:
                    d_p = d_p.add(p, alpha=wd)

                state = self.state[p]
                if mom != 0:
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.zeros_like(p)
                    buf.mul_(mom).add_(d_p)
                    state["momentum_buffer"] = buf
                    d = d_p.add(buf, alpha=mom) if nesterov else buf
                else:
                    d = d_p

                p.add_(d, alpha=-lr)

        self._t += 1

        # sigma scheduling hook
        if self.sigma_schedule is not None:
            self.sigma = float(self.sigma_schedule(self._t, self.sigma))

        # --- 2) Evolutionary step gate ---
        if self.evo_interval <= 0 or (self._t % self.evo_interval != 0):
            return None
        if closure is None:
            raise RuntimeError("Evo step requires a forward-only closure returning loss.")

        # gather trainable leaf params (keep order stable)
        flat_params = [p for g in self.param_groups for p in g["params"] if p.requires_grad]
        if not flat_params:
            return None

        dev = flat_params[0].device
        use_pin = (dev.type == "cuda")
        mutate_mask = None
        if self.mutate_filter is not None:
            mutate_mask = [bool(self.mutate_filter(p)) for p in flat_params]
        else:
            mutate_mask = [True] * len(flat_params)

        def param_scale(p: torch.Tensor):
            if p.numel() == 0:
                return torch.tensor(0., device=p.device, dtype=p.dtype)
            s = torch.sqrt(torch.mean(p.pow(2)))
            return torch.clamp(s, min=torch.tensor(1e-8, device=p.device, dtype=p.dtype))

        def randn_like(p, g=None):
            return torch.randn_like(p, generator=g) if g is not None else torch.randn_like(p)

        # --- CPU offload helpers ---
        def cpu_clone_like(p):
            # pinned CPU clone for fast H2D
            return (torch.empty_like(p, device="cpu", pin_memory=use_pin)
                        .copy_(p, non_blocking=use_pin))

        def snapshot_base():
            if self.offload_population_to_cpu:
                return [cpu_clone_like(p) for p in flat_params]
            else:
                return [p.detach().clone() for p in flat_params]

        def load_params(src_list):
            # Copy src (CPU or GPU) into model params (likely on GPU)
            for p, s in zip(flat_params, src_list):
                if s.data_ptr() == p.data_ptr():
                    continue
                if s.device.type == "cpu":
                    p.copy_(s, non_blocking=use_pin)
                else:
                    p.copy_(s)

        def make_mutant_from(base_list):
            # Create mutated params next to where base_list lives (CPU if offloading)
            out = []
            for use, p0 in zip(mutate_mask, base_list):
                if not use:
                    out.append(p0)  # unchanged tensor reference OK; will be copied on load
                    continue
                scale = param_scale(p0) * self.sigma
                # keep noise dtype same as p0; generate where p0 lives
                noise = randn_like(p0, self.rng) * scale.to(dtype=p0.dtype, device=p0.device)
                out.append(p0 + noise)
            return out

        # --- Evaluate base ---
        base = snapshot_base()          # CPU (pinned) if offloading, else GPU clone
        load_params(base)               # ensure model has base params
        base_loss_t = closure().detach()
        base_loss = base_loss_t if isinstance(base_loss_t, torch.Tensor) else torch.tensor(float(base_loss_t))
        if self.loss_reduce is not None:
            base_loss = self.loss_reduce(base_loss)
        base_loss_f = float(base_loss)

        # top-k container: list of (loss_float, param_list_on_cpu_or_gpu)
        k = max(1, int(math.ceil(self.elite_frac * self.pop_size)))
        elites = [(base_loss_f, base)]

        # --- Evaluate population (streaming elites, offloaded) ---
        for _ in range(self.pop_size - 1):
            cand = make_mutant_from(base)   # lives where base lives (CPU if offloading)
            load_params(cand)               # copy to GPU for evaluation
            cand_loss_t = closure().detach()
            cand_loss = cand_loss_t if isinstance(cand_loss_t, torch.Tensor) else torch.tensor(float(cand_loss_t))
            if self.loss_reduce is not None:
                cand_loss = self.loss_reduce(cand_loss)
            cand_loss_f = float(cand_loss)

            if len(elites) < k:
                # store elite; keep on CPU if offloading to save VRAM
                if self.offload_population_to_cpu and cand[0].device.type != "cpu":
                    cand = [cpu_clone_like(t) for t in cand]
                else:
                    cand = [t.detach().clone() for t in cand]
                elites.append((cand_loss_f, cand))
            else:
                worst_i = max(range(len(elites)), key=lambda i: elites[i][0])
                if cand_loss_f < elites[worst_i][0]:
                    if self.offload_population_to_cpu and cand[0].device.type != "cpu":
                        cand = [cpu_clone_like(t) for t in cand]
                    else:
                        cand = [t.detach().clone() for t in cand]
                    elites[worst_i] = (cand_loss_f, cand)

        # --- Recombine elites ---
        elites.sort(key=lambda x: x[0])
        elite_params = [e[1] for e in elites]
        elite_losses = torch.tensor([e[0] for e in elites], dtype=torch.float32)

        def weighted_mix(parts, weights=None):
            stacked = torch.stack(parts, dim=0)  # on CPU if offloaded
            if weights is None:
                return stacked.mean(dim=0)
            w = weights.view(-1, *([1] * (stacked.dim() - 1)))
            return (stacked * w).sum(dim=0)

        mixed = []
        if self.recombine == "mean" or len(elite_params) == 1:
            for parts in zip(*elite_params):
                mixed.append(weighted_mix(list(parts)))
        else:
            # softmax over -loss (lower loss -> higher weight)
            w = torch.softmax(-elite_losses, dim=0)
            for parts in zip(*elite_params):
                mixed.append(weighted_mix(list(parts), weights=w))

        # single H2D sweep to set final params
        load_params(mixed)

        # optional: reset momentum after jump
        if self.reset_momentum_on_evo:
            for p in flat_params:
                st = self.state[p]
                if "momentum_buffer" in st:
                    st["momentum_buffer"].zero_()

        # decay sigma
        self.sigma *= self.sigma_decay

        # report current loss after recombination
        cur = closure().detach()
        cur = cur if isinstance(cur, torch.Tensor) else torch.tensor(float(cur))
        if self.loss_reduce is not None:
            cur = self.loss_reduce(cur)
        return float(cur)
```

### How to use (CPU offload on, safe eval closure)

```python
import torch, torch.nn as nn, torch.nn.functional as F

model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1)).cuda()
opt = EvoSGD(
    model.parameters(),
    lr=1e-2, momentum=0.9, nesterov=True, weight_decay=1e-4,
    evo_interval=100, pop_size=8, elite_frac=0.25, sigma=0.05, sigma_decay=0.97,
    recombine="mean",
    offload_population_to_cpu=True,     # <= key toggle
    mutate_filter=None,                 # e.g., lambda p: p.ndim == 2 to mutate only linear weights
)

X = torch.randn(512, 10, device="cuda")
true_w = torch.randn(10, 1, device="cuda")
y = X @ true_w + 0.1 * torch.randn(512, 1, device="cuda")

def loss_fn():
    pred = model(X)
    return F.mse_loss(pred, y)

def closure_eval_only():
    was_training = model.training
    model.eval()
    with torch.no_grad():
        val = loss_fn()
    if was_training:
        model.train()
    return val

for step in range(400):
    opt.zero_grad(set_to_none=True)
    loss = loss_fn()
    loss.backward()
    evo_loss = opt.step(closure=closure_eval_only)
    if step % 50 == 0:
        print(f"step {step:4d} | loss {float(loss):.4f}" +
              (f" | evo_loss {evo_loss:.4f}" if evo_loss is not None else ""))
```

### Tips

* If VRAM is super tight, set `mutate_filter` to only mutate the head or last block → much smaller H2D copies.
* With DDP, pass a `loss_reduce` that all-reduces the scalar loss across ranks (mean).
* For determinism, pass `rng=torch.Generator(device='cpu').manual_seed(123)` (or on GPU).
* AMP: keep optimizer/master params FP32; the code mutates in the param’s dtype (safe as long as params are FP32).

