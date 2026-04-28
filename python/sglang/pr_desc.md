<!-- Thank you for your contribution! Please follow these guidelines to enhance your pull request. If anything is unclear, submit your PR and reach out to maintainers for assistance. Join our Slack community at https://slack.sglang.io to discuss further. -->

## Motivation

Fix the Nemotron-3-Nano FP8 failure exposed after enabling the JIT grouped-topk path.

Nemotron now selects the JIT `grouped_topk` path because its router config matches the kernel constraints: one expert group, `topk_group=1`, 128 routed experts, `topk=6`, and correction bias. That exposed two separate issues.

First, the grouped-topk kernel did not order negative choice scores correctly. Nemotron's correction bias can make `sigmoid(score) + bias` negative, so the float packing used for max reduction must preserve ordering across the full float range, not just positive values.

Second, Nemotron's PCG Mamba split op receives buffers padded to the captured graph size, but Mamba only computes the actual token rows. Before this PR, the padded rows in the split-op output were left uninitialized. Once execution continued into the grouped-topk/FP8 MoE path under graph replay, downstream kernels could see those padded rows as garbage/non-finite hidden states. Zero-filling the padded tail makes that region deterministic and harmless.

## Modifications

- Fix `grouped_topk` float packing so `sigmoid(score) + correction_bias` is ordered correctly even when the result is negative.
- Zero-fill padded Nemotron Mamba outputs in piecewise CUDA graph mode.
- Add a regression test for Nemotron-like `E=128, topk=6` routing with negative choice scores.

## Accuracy Tests

`pytest python/sglang/jit_kernel/tests/test_grouped_topk.py` and `pytest test/registered/models/test_nvidia_nemotron_3_nano.py` all passed locally

## Speed Tests and Profiling
N/A

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [x] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed).
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).

## Review and Merge Process

1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process).
2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers.
3. Trigger CI tests with [comments](https://docs.sglang.io/developer_guide/contribution_guide.html#how-to-trigger-ci-tests) or contact authorized users to do so.
   - Common commands include `/tag-and-rerun-ci`, `/tag-run-ci-label`, `/rerun-failed-ci`
4. After green CI and required approvals, ask Merge Oncalls or people with Write permission to merge the PR.
