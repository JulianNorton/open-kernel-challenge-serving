# Open Kernel Challenge, serving track: 1.977x

A patch to SGLang v0.5.23 that serves Muse Spark Open (816B, NVFP4, DFlash) on 8×H200 at 1.977x the throughput of stock
SGLang at stock's per-user speed. This repository holds exactly what produced that score.

| File | What it is |
|---|---|
| `patch.diff` | The submitted patch: a `git diff` against SGLang commit `297e54fd`, 21 files, all under `python/sglang/` |
| `flags.json` | The launch flags submitted with it (`null` means the judge's flag is dropped) |
| `SHA256SUMS` | Checksums of both. The judge recorded the same SHA-256 for the patch |

## The run

Passed correctness and later-turn checks; GPU memory
3.4 GB per GPU below stock; server start 341 s.

| Users | Ratio | Ours, tok/s per GPU at stock's per-user speed | Stock, tok/s per GPU |
|---|---|---|---|
| 1 | 1.864 | 12,066 | 6,473 |
| 4 | 1.944 | 18,976 | 9,761 |
| 16 | 2.082 | 25,553 | 12,273 |
| 32 | 2.025 | 25,553 | 12,620 |

The score is the geometric mean of the four ratios. Stock ran on the same node with the same conversations.

## Reproduce

Hardware and image: one node with 8×H200, in the judge's image
`lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f`.

1. Check the files and build the patched SGLang.

   ```bash
   sha256sum -c SHA256SUMS
   git clone https://github.com/sgl-project/sglang && cd sglang
   git checkout 297e54fd105e974cdf401e5b4d20253a22c16f90
   git apply ../patch.diff
   pip install -e python --no-deps
   ```

2. Start the server. This is the judge's launch command with `flags.json` applied: `--enable-hierarchical-cache` is
   dropped and the last three lines are added.

   ```bash
   SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 SGLANG_TIMEOUT_KEEP_ALIVE=1800 python -m sglang.launch_server \
     --model-path meta-models/Muse-Spark-Open-816B-A42B-NVFP4 \
     --speculative-draft-model-path RadixArk/Muse-Spark-Open-DFlash2 --tp-size 8 --ep-size 8 \
     --quantization modelopt_fp4 --moe-runner-backend marlin --mem-fraction-static 0.85 \
     --speculative-algorithm DFLASH --speculative-num-draft-tokens 8 \
     --speculative-draft-attention-backend fa4 --mm-feature-transport cpu \
     --hicache-size 160 --chunked-prefill-size 16384 \
     --context-length 131072 --enable-cache-report --incremental-streaming-output \
     --random-seed 1234 --max-running-requests 32 --cuda-graph-max-bs-decode 32 \
     --moe-proj-shard-min-tokens 2048 --enable-moe-pre-proj-gather --moe-proj-gather-max-tokens 8 \
     --swa-full-tokens-ratio 0.3 --muse-spark-kernel-moe-large --muse-spark-kernel-moe-large-min-rows 512 \
     --muse-spark-kernel-moe-marlin-layout --muse-spark-kernel-moe-marlin-layout-table 32:11,256:12 \
     --enable-layernorm-sp --draft-proj-shard-min-tokens 2048 --muse-spark-kernel-attn-verify \
     --scheduler-recv-interval 4
   ```

   The first start compiles the patch's CUDA extensions and checks each new code path against the stock path on the
   real weights. The judged run took 341 s to start; later starts reuse the compiled extensions.

3. Measure. The score came from the organizers' judge, which is not part of this repository. It drives multi-turn
   coding conversations with greedy decoding at 1, 4, 16 and 32 concurrent users: 13 turns per conversation, a first turn
   of about 74K tokens, 753 new tokens per later turn, about 220 output tokens per turn. Each rung's ratio is our
   throughput at stock's end-to-end per-user speed, read from our own curve across the rungs, divided by stock's
   throughput on the same node. To reproduce the number, run the same load against stock SGLang at the same commit with
   the judge's unmodified flags.

## What to expect

- **Run-to-run spread.** Two earlier runs with the same kernels on different nodes scored within 0.4% of each other,
  while single rungs moved by up to 6%. Compare whole scores, not single rungs.
- **Hopper only.** The fast kernel variants need H100 or H200. On other GPUs they are refused at load, the stock paths
  run, and the score is not reproduced.
- **Hierarchical cache off.** On this workload nothing is evicted, so the host cache is not needed. With more than about
  108K tokens per conversation at 32 users, evicted prefixes are recomputed instead of reloaded. Keeping
  `--enable-hierarchical-cache` and setting `--hicache-size 120` avoids that; an earlier version of this patch scored
  1.92 that way.

## License

* Apache-2.0
* The patch includes portions © vjeux (github.com/vjeux/okc-2026, commit f35a41e, which is also Apache-2.0, submitted by
alygg77) and decodes weights as Marlin does (Elias Frantar, IST-DASLab/marlin, again - Apache-2.0). The notices are in the
patched files.
