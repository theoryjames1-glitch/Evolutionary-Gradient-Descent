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
