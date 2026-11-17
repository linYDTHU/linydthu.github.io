"""Reproduce stale autograd history after a no-grad cache refresh.

Requires PyTorch. Run on CPU with:
    python kv_cache_gradient_leak.py
"""

import torch


def probe(detach_cache=False):
    w = torch.ones(1, requires_grad=True)
    cache = torch.zeros(4)

    # Earlier chunk: a differentiable, in-place cache write.
    cache[:2] = w * 2

    # Refresh the values; no new graph is recorded.
    with torch.no_grad():
        cache[:2] = 5.0

    if detach_cache:
        cache = cache.detach()

    # A later chunk reads the refreshed context.
    q = torch.tensor(3.0, requires_grad=True)
    loss = (cache[:2] * q).sum()
    grad_w, grad_q = torch.autograd.grad(loss, (w, q), allow_unused=True)
    return loss.item(), None if grad_w is None else grad_w.item(), grad_q.item()


if __name__ == "__main__":
    original = probe(False)
    detached = probe(True)
    print(f"PyTorch {torch.__version__} | CPU")
    print("                       Forward   Earlier w   Current q")
    for label, result in (("Without detach", original), ("With detach", detached)):
        value, grad_w, grad_q = result
        print(f"{label:<22} {value:>7.1f} {str(grad_w):>11} {grad_q:>11.1f}")
    assert original == (30.0, 12.0, 10.0), original
    assert detached == (30.0, None, 10.0), detached
    print("Verified: equal forward values; historical path removed; current gradient preserved.")
