---
layout: kv-research
title: "KV Cache Gradient Leakage in Autoregressive Video Training"
date: 2026-11-17 09:00:00 +0800
description: "A KV cache gradient leak inherited from Self Forcing and a two-line fix that reduces DMD training memory by 22%."
categories: research
tags: video-generation pytorch training
related_posts: false
---

<div class="kv-note" markdown="1">
**Self Forcing is designed to block gradients across chunks, but its training code unintentionally allows them to leak through the KV cache.** This changes the intended training gradients and increases GPU memory usage. Downstream implementations that inherit the same cache-update pattern also inherit the bug.
</div>

## How a fresh cache keeps a stale graph

Self-forcing training generates a video one chunk at a time. Within each chunk, the forward pass at a selected denoising step—the **exit-step forward pass**—runs with gradients enabled. Its output contributes to the training loss. The model then performs a **context refresh** under `torch.no_grad()`, updating the cache with K/V computed from the generated chunk.

Both forward passes write to the **same persistent cache buffers**. The exit-step forward pass records an autograd connection through its in-place write, typically via `CopySlices`. The subsequent no-grad refresh overwrites the values without removing that connection. When the next chunk reads the cache, its gradients can flow back into the earlier exit-step forward pass.

<figure class="kv-mechanism">
  <div class="kv-mechanism-scroll"><img src="{{ '/assets/img/kv-cache-gradient-leakage/cache-autograd-trace.svg' | relative_url }}" width="1000" height="380" alt="With w equal to 1, writing 2w stores [2, 2] and records derivative [2, 2]. Overwriting with 5 under no_grad stores [5, 5] but retains derivative [2, 2]. Detach disconnects that path. A later loss with q equal to 3 therefore gives an erroneous historical gradient of 12 without detach, and no connected gradient with detach."></div>
  <figcaption><span>Figure 1. The overwrite changes the stored values, but autograd retains the derivative of the old write.</span><a href="{{ '/assets/img/kv-cache-gradient-leakage/cache-autograd-trace.svg' | relative_url }}" download>Download SVG</a></figcaption>
</figure>

PyTorch's [`no_grad()`](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html) disables recording of new operations; it does not retroactively detach an existing tensor. [`detach()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html) explicitly disconnects a tensor from its autograd history. These connections keep earlier computation graphs alive across chunks, increasing GPU memory usage.

### A minimal reproduction

```python
import torch


def probe(detach_cache=False):
    w = torch.ones(1, requires_grad=True)
    cache = torch.zeros(4)

    # Earlier chunk: a differentiable, in-place cache write.
    cache[:2] = w * 2

    # Refresh: replace the values without recording a new graph.
    with torch.no_grad():
        cache[:2] = 5.0

    if detach_cache:
        cache = cache.detach()

    # A later chunk reads the refreshed context.
    q = torch.tensor(3.0, requires_grad=True)
    loss = (cache[:2] * q).sum()
    grad_w, grad_q = torch.autograd.grad(loss, (w, q), allow_unused=True)
    return loss.item(), None if grad_w is None else grad_w.item(), grad_q.item()


print(probe(False))  # (30.0, 12.0, 10.0)
print(probe(True))   # (30.0, None, 10.0)
```

Each tuple contains the loss, the gradient with respect to `w`, and the gradient with respect to `q`. `None` means that the loss is no longer connected to `w`; the gradient with respect to `q` is unchanged.

[Download the CPU reproduction]({{ '/assets/code/kv_cache_gradient_leak.py' | relative_url }}) and run `python kv_cache_gradient_leak.py` with PyTorch installed.

## The fix: detach the cache at the boundary

After refreshing the context, detach **both K and V in every layer** before generating the next chunk:

```python
# Preserve the current chunk's connection to the training loss.
outputs.append(denoised_pred)

with torch.no_grad():
    refresh_context(denoised_pred, kv_cache)

# Stop gradients through persistent context, not through the output above.
for layer_cache in kv_cache:
    layer_cache["k"] = layer_cache["k"].detach()
    layer_cache["v"] = layer_cache["v"].detach()
```

Apply `.detach()` at **every chunk boundary**, after the cache-update forward pass represented by `refresh_context`. Detachment shares the existing storage and preserves cached values. The current chunk’s output remains connected to the training loss.

**Recomputing context offers another way to avoid this leak.** [Self Gradient Forcing (SGF)](https://arxiv.org/html/2607.20368v2#S3.SS3) performs a no-grad rollout, then re-encodes the recorded context latents in a separate gradient-enabled forward pass. This gives the backward pass a fresh graph corresponding to the context computation used in that forward pass, avoiding the stale cache connections described above. Detachment restores Self Forcing's zero cross-chunk gradient boundary; SGF instead enables gradients through recomputed context K/V while keeping the generated context latents detached.

## Results

We compare **DMD training with and without KV cache detachment** using two few-step initializations: **Causal Forcing's ODE checkpoint** ([`causal_ode.pt`](https://huggingface.co/zhuhz22/Causal-Forcing/blob/main/chunkwise/causal_ode.pt)) and **Causal Forcing++'s CD checkpoint** ([`causal_cd.pt`](https://huggingface.co/zhuhz22/Causal-Forcing/blob/main/chunkwise/causal_cd.pt)).

### Memory and gradient norms

<figure class="kv-results" aria-label="GPU memory in the original DMD comparison: 86.99 GB without detachment and 67.89 GB with detachment">
  <section class="kv-result-panel" aria-labelledby="kv-memory-title">
    <div class="kv-result-heading"><h3 id="kv-memory-title">Training GPU memory</h3><strong>−19.10 GB <span>· −22.0%</span></strong></div>
    <div class="kv-result-bar-row"><span>Without KV detachment</span><div class="kv-result-track"><div class="kv-result-bar kv-result-original" style="width: 100%">86.99 GB</div></div></div>
    <div class="kv-result-bar-row"><span>With KV detachment</span><div class="kv-result-track"><div class="kv-result-bar kv-result-detached" style="width: 78.0435%">67.89 GB</div></div></div>
  </section>
  <figcaption class="kv-small">Memory measured in the original DMD comparison.</figcaption>
</figure>

Blocking unintended cross-chunk gradients also lowers the generator gradient norm during training. Across the logged steps, the mean falls from **0.1483 to 0.0921 (−37.9%)** with ODE initialization and from **0.1307 to 0.0830 (−36.5%)** with CD initialization.

<figure class="kv-trend-figure" aria-label="Generator gradient norm during DMD training with ODE and CD initialization">
  <div class="kv-trend-legend"><span><i class="kv-trend-key kv-trend-without" aria-hidden="true"></i>Without KV detachment</span><span><i class="kv-trend-key kv-trend-with" aria-hidden="true"></i>With KV detachment</span></div>
  <div class="kv-trend-grid">
    <div>
      <img class="kv-chart-light" src="{{ '/assets/img/kv-cache-gradient-leakage/gradient-norm-ode.svg' | relative_url }}" width="500" height="375" loading="lazy" alt="ODE initialization: generator gradient norms are lower on average with KV cache detachment during DMD training.">
      <img class="kv-chart-dark" src="{{ '/assets/img/kv-cache-gradient-leakage/gradient-norm-ode-dark.svg' | relative_url }}" width="500" height="375" loading="lazy" alt="ODE initialization: generator gradient norms are lower on average with KV cache detachment during DMD training.">
    </div>
    <div>
      <img class="kv-chart-light" src="{{ '/assets/img/kv-cache-gradient-leakage/gradient-norm-cd.svg' | relative_url }}" width="500" height="375" loading="lazy" alt="CD initialization: generator gradient norms are lower on average with KV cache detachment during DMD training.">
      <img class="kv-chart-dark" src="{{ '/assets/img/kv-cache-gradient-leakage/gradient-norm-cd-dark.svg' | relative_url }}" width="500" height="375" loading="lazy" alt="CD initialization: generator gradient norms are lower on average with KV cache detachment during DMD training.">
    </div>
  </div>
  <figcaption>Figure 2. Generator gradient norm during DMD training. Faint lines show all 300 logged values per run (one every five iterations); bold lines show an 11-point centered moving average, with shorter windows at the endpoints. Means use the unsmoothed values. <a href="{{ '/assets/data/kv-cache-gradient-norm.csv' | relative_url }}" download>Download scores</a>.</figcaption>
</figure>

### VBench after 750 DMD iterations

Detachment improves video dynamics with both initializations, reflected in higher Dynamic Degree scores. The overall VBench score is nearly unchanged with ODE initialization and lower with CD initialization.

<figure class="kv-results kv-quality-results" aria-label="Overall VBench scores after 750 DMD iterations: ODE initialization, 84.06 without detachment and 84.01 with detachment; CD initialization, 84.38 without detachment and 83.20 with detachment">
  <div class="kv-results-main">
    <section class="kv-result-panel" aria-labelledby="kv-ode-title">
      <div class="kv-result-heading"><h4 id="kv-ode-title">Causal Forcing · ODE initialization</h4><strong>−0.05 points</strong></div>
      <div class="kv-score-change">
        <div><span>Overall · Without detachment</span><strong>84.06</strong></div>
        <span class="kv-score-arrow" aria-hidden="true">→</span>
        <div><span>Overall · With detachment</span><strong>84.01</strong></div>
      </div>
      <p class="kv-dynamic-summary">Dynamic Degree <strong>60.00 → 71.94</strong></p>
    </section>
    <section class="kv-result-panel" aria-labelledby="kv-cd-title">
      <div class="kv-result-heading"><h4 id="kv-cd-title">Causal Forcing++ · CD initialization</h4><strong>−1.18 points</strong></div>
      <div class="kv-score-change">
        <div><span>Overall · Without detachment</span><strong>84.38</strong></div>
        <span class="kv-score-arrow" aria-hidden="true">→</span>
        <div><span>Overall · With detachment</span><strong>83.20</strong></div>
      </div>
      <p class="kv-dynamic-summary">Dynamic Degree <strong>80.56 → 84.72</strong></p>
    </section>
  </div>
  <figcaption class="kv-small">EMA checkpoints; scores ×100. All four runs reach their highest overall score among the evaluated checkpoints at 750 iterations.</figcaption>
</figure>

### Dynamic Degree during training

With CD initialization, detachment yields higher Dynamic Degree at every evaluated checkpoint. With ODE initialization, the initial advantage diminishes during training and reverses at 1500 iterations.

<figure class="kv-trend-figure" aria-label="Dynamic Degree from 750 to 1500 DMD iterations, shown separately for ODE and CD initialization">
  <div class="kv-trend-legend"><span><i class="kv-trend-key kv-trend-without" aria-hidden="true"></i>Without KV detachment</span><span><i class="kv-trend-key kv-trend-with" aria-hidden="true"></i>With KV detachment</span></div>
  <div class="kv-trend-grid">
    <div>
      <img class="kv-chart-light" src="{{ '/assets/img/kv-cache-gradient-leakage/dynamic-degree-ode.svg' | relative_url }}?v=20261117-axis50" width="500" height="375" loading="lazy" alt="ODE initialization: without detachment, Dynamic Degree is 60.00, 55.28, 50.28, 51.94 at 750, 1000, 1250, 1500 iterations. With detachment: 71.94, 65.28, 56.94, 50.56.">
      <img class="kv-chart-dark" src="{{ '/assets/img/kv-cache-gradient-leakage/dynamic-degree-ode-dark.svg' | relative_url }}?v=20261117-axis50" width="500" height="375" loading="lazy" alt="ODE initialization: without detachment, Dynamic Degree is 60.00, 55.28, 50.28, 51.94 at 750, 1000, 1250, 1500 iterations. With detachment: 71.94, 65.28, 56.94, 50.56.">
    </div>
    <div>
      <img class="kv-chart-light" src="{{ '/assets/img/kv-cache-gradient-leakage/dynamic-degree-cd.svg' | relative_url }}?v=20261117-axis50" width="500" height="375" loading="lazy" alt="CD initialization: without detachment, Dynamic Degree is 80.56, 80.56, 71.39, 67.78 at 750, 1000, 1250, 1500 iterations. With detachment: 84.72, 93.89, 90.00, 81.11.">
      <img class="kv-chart-dark" src="{{ '/assets/img/kv-cache-gradient-leakage/dynamic-degree-cd-dark.svg' | relative_url }}?v=20261117-axis50" width="500" height="375" loading="lazy" alt="CD initialization: without detachment, Dynamic Degree is 80.56, 80.56, 71.39, 67.78 at 750, 1000, 1250, 1500 iterations. With detachment: 84.72, 93.89, 90.00, 81.11.">
    </div>
  </div>
  <figcaption>Figure 3. Dynamic Degree during DMD training, evaluated on EMA checkpoints. Both panels use a 50–100 scale. Each curve represents one training run. <a href="{{ '/assets/data/kv-cache-emafix-vbench.csv' | relative_url }}" download>Download scores</a>.</figcaption>
</figure>

<h2 id="ode-initialization-explore-the-paired-videos">Video comparisons after DMD training</h2>

The videos use **EMA checkpoints after 750 DMD iterations**, with **CD or ODE initialization**. For each initialization, the two variants use the **same training seed and initial generation noise** across **21 matched prompts**.

{% include kv-cache-comparison.liquid %}

## Where the pattern appears in official code

The table below links the papers and official code, with code evidence for each status.

<div class="kv-impact-legend" aria-label="Affected status legend">
  <span class="kv-impact"><span class="kv-impact-icon" aria-hidden="true">✓</span> Likely affected</span>
  <span class="kv-impact"><span class="kv-impact-icon kv-half-check" aria-hidden="true">✓</span> Possibly affected</span>
  <span class="kv-impact"><span class="kv-impact-icon" aria-hidden="true">✗</span> Not affected in the audited path</span>
</div>

Statuses apply to the linked training paths. A half-filled check marks configuration-dependent risk or incomplete training code.

<div class="kv-audit-table kv-code-audit">
<table>
<thead><tr><th scope="col">Affected?</th><th scope="col">Paper</th><th scope="col">Official code</th><th scope="col">Code evidence and scope</th></tr></thead>
<tbody>
<tr><td><span class="kv-impact"><span class="kv-impact-icon" aria-hidden="true">✓</span> Likely</span></td><td><a href="https://arxiv.org/abs/2506.08009">Self Forcing</a></td><td><a href="https://github.com/guandeh17/Self-Forcing/tree/33593df3e81fa3ec10239271dd2c100facac6de1">Self-Forcing · <code>33593df</code></a></td><td><details class="kv-evidence"><summary>The refresh retains the exit graph</summary><div>A <a href="https://github.com/guandeh17/Self-Forcing/blob/33593df3e81fa3ec10239271dd2c100facac6de1/pipeline/self_forcing_training.py#L186-L217">gradient-enabled exit-step forward pass and no-grad refresh</a> share persistent buffers with <a href="https://github.com/guandeh17/Self-Forcing/blob/33593df3e81fa3ec10239271dd2c100facac6de1/wan/modules/causal_model.py#L221-L233">in-place K/V writes and cache reads</a>. The cache remains connected to the exit-step forward pass.</div></details></td></tr>
<tr><td><span class="kv-impact"><span class="kv-impact-icon" aria-hidden="true">✓</span> Likely</span></td><td><a href="https://arxiv.org/abs/2602.02214">Causal Forcing</a></td><td><a href="https://github.com/thu-ml/Causal-Forcing/tree/eb8705b2f00b7c361193b25aa3aae9e7edc94667">Causal-Forcing · <code>eb8705b</code></a></td><td><details class="kv-evidence"><summary>Shared cache across exit and refresh</summary><div>The <a href="https://github.com/thu-ml/Causal-Forcing/blob/eb8705b2f00b7c361193b25aa3aae9e7edc94667/pipeline/self_forcing_training.py#L219-L227">gradient-enabled exit-step forward pass</a> writes through <a href="https://github.com/thu-ml/Causal-Forcing/blob/eb8705b2f00b7c361193b25aa3aae9e7edc94667/wan/modules/causal_model.py#L225-L237">causal attention's persistent K/V buffers</a>; <a href="https://github.com/thu-ml/Causal-Forcing/blob/eb8705b2f00b7c361193b25aa3aae9e7edc94667/pipeline/self_forcing_training.py#L233-L250">refresh uses <code>no_grad()</code> without detaching the cache</a>.</div></details></td></tr>
<tr><td><span class="kv-impact"><span class="kv-impact-icon" aria-hidden="true">✓</span> Likely</span></td><td><a href="https://arxiv.org/abs/2605.15141">Causal Forcing++</a></td><td><a href="https://github.com/thu-ml/Causal-Forcing/tree/eb8705b2f00b7c361193b25aa3aae9e7edc94667">Causal-Forcing · <code>eb8705b</code></a></td><td><details class="kv-evidence"><summary>The same DMD training loop</summary><div>The shared repository uses the same <a href="https://github.com/thu-ml/Causal-Forcing/blob/eb8705b2f00b7c361193b25aa3aae9e7edc94667/pipeline/self_forcing_training.py#L219-L250">self-forcing DMD training loop</a>.</div></details></td></tr>
<tr><td><span class="kv-impact"><span class="kv-impact-icon" aria-hidden="true">✓</span> Likely</span></td><td><a href="https://arxiv.org/abs/2509.22622">LongLive 1.0</a></td><td><a href="https://github.com/NVlabs/LongLive/tree/31c70d9071cd7851d82939806dc9fac645fcabe5/LongLive1.0">LongLive1.0 · <code>31c70d9</code></a></td><td><details class="kv-evidence"><summary>Deferred writes retain autograd history</summary><div><a href="https://github.com/NVlabs/LongLive/blob/31c70d9071cd7851d82939806dc9fac645fcabe5/LongLive1.0/wan/modules/causal_model.py#L837-L889">Deferred K/V writes</a> are <a href="https://github.com/NVlabs/LongLive/blob/31c70d9071cd7851d82939806dc9fac645fcabe5/LongLive1.0/wan/modules/causal_model.py#L1043-L1044">applied within the same forward pass</a>; <a href="https://github.com/NVlabs/LongLive/blob/31c70d9071cd7851d82939806dc9fac645fcabe5/LongLive1.0/pipeline/self_forcing_training.py#L234-L271">refresh runs under <code>no_grad()</code></a>. Its <a href="https://github.com/NVlabs/LongLive/blob/31c70d9071cd7851d82939806dc9fac645fcabe5/LongLive1.0/model/streaming_training.py#L601-L658">cache-clearing helper</a> does not detach the cache between every pair of blocks within a training chunk.</div></details></td></tr>
<tr><td><span class="kv-impact"><span class="kv-impact-icon" aria-hidden="true">✓</span> Likely</span></td><td><a href="https://arxiv.org/abs/2605.18739">LongLive 2.0</a></td><td><a href="https://github.com/NVlabs/LongLive/tree/31c70d9071cd7851d82939806dc9fac645fcabe5/LongLive2.0">LongLive2.0 · <code>31c70d9</code></a></td><td><details class="kv-evidence"><summary>Non-quantized cache path</summary><div>The <strong>non-quantized path</strong> <a href="https://github.com/NVlabs/LongLive/blob/31c70d9071cd7851d82939806dc9fac645fcabe5/LongLive2.0/wan_5b/modules/causal_model.py#L657-L670">reads historical cache through <code>cat</code> or <code>clone</code></a>, <a href="https://github.com/NVlabs/LongLive/blob/31c70d9071cd7851d82939806dc9fac645fcabe5/LongLive2.0/wan_5b/modules/causal_model.py#L1431-L1481">writes undetached K/V</a>, and <a href="https://github.com/NVlabs/LongLive/blob/31c70d9071cd7851d82939806dc9fac645fcabe5/LongLive2.0/pipeline/self_forcing_training.py#L232-L266">refreshes under <code>no_grad()</code></a>.</div></details></td></tr>
<tr><td><span class="kv-impact"><span class="kv-impact-icon kv-half-check" aria-hidden="true">✓</span> Possible</span></td><td><a href="https://arxiv.org/abs/2509.25161">Rolling Forcing</a></td><td><a href="https://github.com/TencentARC/RollingForcing/tree/a1477d09e85dc759a6a6728f55f77f59342ce388">RollingForcing · <code>a1477d0</code></a></td><td><details class="kv-evidence"><summary>Rolling-window training</summary><div>A <a href="https://github.com/TencentARC/RollingForcing/blob/a1477d09e85dc759a6a6728f55f77f59342ce388/pipeline/rolling_forcing_training.py#L186-L248">gradient-enabled window forward pass and no-grad refresh</a> share a cache with <a href="https://github.com/TencentARC/RollingForcing/blob/a1477d09e85dc759a6a6728f55f77f59342ce388/wan/modules/causal_model.py#L225-L234">in-place K/V writes</a>. The rolling-window gradient behavior remains unverified.</div></details></td></tr>
<tr><td><span class="kv-impact"><span class="kv-impact-icon kv-half-check" aria-hidden="true">✓</span> Possible</span></td><td><a href="https://arxiv.org/abs/2607.26811">DistillAlign</a></td><td><a href="https://github.com/LiJiaxing0213/DistillAlign/tree/00eac44b8412e0991d1763bc15e50994b4b94b66">DistillAlign · <code>00eac44</code></a></td><td><details class="kv-evidence"><summary>Training rollout not released</summary><div>The released attention implementation contains the same <a href="https://github.com/LiJiaxing0213/DistillAlign/blob/00eac44b8412e0991d1763bc15e50994b4b94b66/wan/modules/causal_model.py#L221-L233">in-place K/V write and cache read</a>. The training rollout is not included in this release.</div></details></td></tr>
<tr><td><span class="kv-impact"><span class="kv-impact-icon kv-half-check" aria-hidden="true">✓</span> Possible</span></td><td><a href="https://arxiv.org/abs/2607.08766">OPSD-V</a></td><td><a href="https://github.com/MeiGen-AI/OPSD-V/tree/0f4b9b6b426f67cba14891e3a374a8df0b6170f2">OPSD-V · <code>0f4b9b6</code></a></td><td><details class="kv-evidence"><summary>Default settings detach; an optional mode does not</summary><div>The released <a href="https://github.com/MeiGen-AI/OPSD-V/blob/0f4b9b6b426f67cba14891e3a374a8df0b6170f2/configs/train_self_forcing_lora.yaml#L42-L54">Self-Forcing</a> and <a href="https://github.com/MeiGen-AI/OPSD-V/blob/0f4b9b6b426f67cba14891e3a374a8df0b6170f2/configs/train_longlive_lora.yaml#L42-L54">LongLive</a> training configurations set <code>opsd_backward_per_chunk=true</code>, which <a href="https://github.com/MeiGen-AI/OPSD-V/blob/0f4b9b6b426f67cba14891e3a374a8df0b6170f2/model/opsd_streaming_training.py#L810-L823">detaches caches after each backward pass</a>, as specified in <a href="https://arxiv.org/html/2607.08766v1#S4.SS5">Algorithm 1</a>. Setting it to <code>false</code> skips detachment: <a href="https://github.com/MeiGen-AI/OPSD-V/blob/0f4b9b6b426f67cba14891e3a374a8df0b6170f2/wan/modules/causal_model.py#L981-L1027">persistent K/V writes retain their graph</a> across the <a href="https://github.com/MeiGen-AI/OPSD-V/blob/0f4b9b6b426f67cba14891e3a374a8df0b6170f2/model/opsd_streaming_training.py#L855-L874">no-grad refresh</a>, and <a href="https://github.com/MeiGen-AI/OPSD-V/blob/0f4b9b6b426f67cba14891e3a374a8df0b6170f2/wan/modules/causal_model.py#L354-L370">later cache reads</a> can propagate gradients into earlier chunks.</div></details></td></tr>
<tr><td><span class="kv-impact"><span class="kv-impact-icon" aria-hidden="true">✗</span> No</span></td><td><a href="https://arxiv.org/abs/2607.20368">Self Gradient Forcing</a></td><td><a href="https://github.com/zhuang2002/Self_Gradient_Forcing/tree/79be9acbd874c9be050944a6b1de6693e988a171">Self_Gradient_Forcing · <code>79be9ac</code></a></td><td><details class="kv-evidence"><summary>Recomputed context gradients</summary><div>The <a href="https://github.com/zhuang2002/Self_Gradient_Forcing/blob/79be9acbd874c9be050944a6b1de6693e988a171/pipeline/self_gradient_forcing_training.py#L154-L214">entire serial rollout and refresh run without gradients</a>; <a href="https://github.com/zhuang2002/Self_Gradient_Forcing/blob/79be9acbd874c9be050944a6b1de6693e988a171/pipeline/self_gradient_forcing_training.py#L216-L230">gradient-enabled replay recomputes context from detached latents</a>. Backward differentiates the replay computation, avoiding stale rollout-cache connections.</div></details></td></tr>
<tr><td><span class="kv-impact"><span class="kv-impact-icon" aria-hidden="true">✗</span> No</span></td><td><a href="https://arxiv.org/abs/2606.25473">Causal-rCM</a></td><td><a href="https://github.com/NVlabs/rcm/tree/ed3cb14dd936f92cdc9f9381af7369991509b41f">rcm · <code>ed3cb14</code></a></td><td><details class="kv-evidence"><summary>Detached writes; read-only cache access</summary><div><a href="https://github.com/NVlabs/rcm/blob/ed3cb14dd936f92cdc9f9381af7369991509b41f/rcm/utils/kv_cache.py#L43-L53">Persistent K/V writes explicitly detach</a>, while the <a href="https://github.com/NVlabs/rcm/blob/ed3cb14dd936f92cdc9f9381af7369991509b41f/rcm/models/t2v_model_causal.py#L703-L722">gradient-enabled exit-step forward pass uses <code>READONLY</code></a>. The current chunk remains differentiable; cached context is detached.</div></details></td></tr>
</tbody>
</table>
</div>

## References

<div class="kv-references" markdown="1">

1. Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. **Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion.** [NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/f4823f831af67a3ef15e41a85434422a-Abstract-Conference.html). [arXiv:2506.08009](https://arxiv.org/abs/2506.08009).

2. Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. **Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation.** [ICML 2026](https://proceedings.mlr.press/v306/zhu26bf.html). [arXiv:2602.02214](https://arxiv.org/abs/2602.02214).

3. Min Zhao, Hongzhou Zhu, Kaiwen Zheng, et al. **Causal Forcing++: Scalable Few-Step Autoregressive Diffusion Distillation for Real-Time Interactive Video Generation.** arXiv preprint, 2026. [arXiv:2605.15141](https://arxiv.org/abs/2605.15141).

4. Shuai Yang, Wei Huang, Ruihang Chu, et al. **LongLive: Real-time Interactive Long Video Generation.** [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/91a1610c6ed9e02d33f826b46f472b92-Abstract-Conference.html). [arXiv:2509.22622](https://arxiv.org/abs/2509.22622).

5. Yukang Chen, Luozhou Wang, Wei Huang, et al. **LongLive-2.0: An NVFP4 Parallel Infrastructure for Long Video Generation.** arXiv preprint, 2026. [arXiv:2605.18739](https://arxiv.org/abs/2605.18739).

6. Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. **Rolling Forcing: Autoregressive Long Video Diffusion in Real Time.** [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/935151cc6cb5d8b6816133b75233775a-Abstract-Conference.html). [arXiv:2509.25161](https://arxiv.org/abs/2509.25161).

7. Jiaxing Li, Kai Zou, Cindy Zhou, et al. **DistillAlign: Coordinating Mode Covering and Mode Seeking in Autoregressive Video Distillation.** arXiv preprint, 2026. [arXiv:2607.26811](https://arxiv.org/abs/2607.26811).

8. Junhao Zhuang, Shiyi Zhang, Yuxuan Bian, et al. **Self Gradient Forcing: Native Long Video Extrapolation.** arXiv preprint, 2026. [arXiv:2607.20368](https://arxiv.org/abs/2607.20368).

9. Kaiwen Zheng, Guande He, Min Zhao, et al. **Causal-rCM: A Unified Teacher-Forcing and Self-Forcing Open Recipe for Autoregressive Diffusion Distillation in Streaming Video Generation and Interactive World Models.** arXiv preprint, 2026. [arXiv:2606.25473](https://arxiv.org/abs/2606.25473).

10. Hongyu Liu, Chun Wang, Feng Gao, et al. **OPSD-V: On-Policy Self-Distillation for Post-Training Few-Step Autoregressive Video Generators.** arXiv preprint, 2026. [arXiv:2607.08766](https://arxiv.org/abs/2607.08766).

</div>

<details id="cite-this-post" class="kv-citation" markdown="1">
<summary>Cite this post</summary>

```bibtex
@misc{yang2026kvcache,
  author = {Yang, Donglin},
  title  = {KV Cache Gradient Leakage in Autoregressive Video Training},
  year   = {2026},
  month  = nov,
  url    = {https://linydthu.github.io/blog/2026/kv-cache-gradient-leakage/},
  note   = {Blog post}
}
```

</details>
