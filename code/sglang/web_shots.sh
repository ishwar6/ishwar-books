#!/usr/bin/env bash
# Screenshots of engineering blogs and docs used in Part 5 (run from the repository root).
# Each crop is one figure or one passage; the article box names the source and section.
set -e
O=public/img/sglang; S="node code/agents/blogshot.mjs"
$S https://www.lmsys.org/blog/2024-01-17-sglang/ $O/blog-sharing-patterns.png 'img[src*="sharing_wide"]' --pad 12
$S https://www.lmsys.org/blog/2024-02-05-compressed-fsm/ $O/blog-jump-forward.png 'img[src*="compare.png"]' --pad 12
$S https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/ $O/blog-cache-aware-router.png 'img[src*="cache_aware"]' --pad 12
$S https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/ $O/blog-overlap-scheduler.png 'img[src*="scheduler.jpg"]' --pad 12
$S https://docs.vllm.ai/en/latest/design/prefix_caching/ $O/vllm-hash-blocks.png "text=A gentle breeze stirred" --pad 16
$S https://docs.vllm.ai/en/latest/design/prefix_caching/ $O/vllm-example-time3.png 'img[alt="Example Time 3"]' --pad 12
$S https://docs.vllm.ai/en/latest/design/prefix_caching/ $O/vllm-free-reverse.png "text=added to the tail of the free queue" --pad 16
$S https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/ $O/vllm-apc-limits.png "text=APC in general does not reduce" --pad 16
D="node code/sglang/docshot.mjs"   # prose inside <span>/<div>: crop from a start phrase to an end phrase
$D https://docs.sglang.io/docs/advanced_features/radix_eviction_policy $O/sglang-eviction-doc.png "How eviction picks a victim" "eviction walks a branch from its tip"
$D https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning $O/sglang-lpm-doc.png "stands for longest prefix match" "stands for longest prefix match"
