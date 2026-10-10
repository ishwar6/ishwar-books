#!/usr/bin/env bash
# Screenshots of docs and engineering pages used in Part 7 (run from the repository root).
# docshot.mjs (from Part 5) crops from the block holding a start phrase to the block holding an end phrase.
set -e
O=public/img/multigpu; D="node code/sglang/docshot.mjs"; S="node code/agents/blogshot.mjs"
$D https://docs.vllm.ai/en/latest/serving/parallelism_scaling/ $O/vllm-parallelism-doc.png "To choose a distributed inference strategy for a single-model replica" "for higher throughput and lower communication overhead"
$D https://docs.vllm.ai/en/latest/features/disagg_prefill/ $O/vllm-disagg-doc.png "Why disaggregated prefilling?" "Disaggregated prefill DOES NOT improve throughput"
$D https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/ $O/vllm-ep-doc.png "vLLM provides multiple communication backends for EP" "Multi-node decode"
$D https://docs.sglang.io/docs/advanced_features/pd_disaggregation $O/sglang-pd-doc.png "Large Language Model (LLM) inference comprises two distinct phases" "Currently, we support Mooncake and NIXL as the transfer engine"
$D https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md $O/deepseek-day6-units.png "Each deployment unit spans 4 nodes" "Each deployment unit spans 18 nodes"
$D https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md $O/deepseek-day6-stats.png "Total input tokens: 608B" "Each H800 node delivers an average throughput"
