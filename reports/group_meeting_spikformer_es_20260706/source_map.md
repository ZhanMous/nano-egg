# Source Map

Report: `spikformer_es_group_meeting_20260706.pdf`

Primary local source:

- `docs/spikformer_es_smoke.md`

Local EggRoll reproduction logs:

- `/home/zhan_shaoji/code/Replay_EGGROLL/HyperscaleES/experiments/paper_repro/countdownn_0.1B.log`
- `/home/zhan_shaoji/code/Replay_EGGROLL/HyperscaleES/experiments/paper_repro/gsm8k_0.1B.log`
- Parsed summary: `data/eggroll_repro_summary.tsv`

Note: the dated `paper_repro/*_202606*` run directories are empty in this checkout; the usable evidence is in the two `.log` files above.

Remote result source, copied from `10.28.2.47:/home/zhanshaoji/code/nano-egg-snn-es`:

- `runs/spikformer4_1m_calibrated_top5/results.json`
- `runs/spikformer4_head_rank_1m/results.json`
- `runs/spikformer4_adapter_only_1m_b64_e15_b1024_top5/results.json`
- `runs/spikformer4_adapter_b64_b1024_stage2_top1/results.json`
- `runs/spikformer4_adapter_b64_b1024_stage3_sigma025_top1/results.json`
- `runs/spikformer4_adapter_b64_b1024_stage4_sigma025_top05/results.json`
- `runs/spikformer4_adapter_b64_b1024_stage5_sigma025_top05_c128/results.json`
- `runs/spikformer4_adapter_b64_b1024_stage6_sigma025_top05_c128/results.json`
- `runs/spikformer4_adapter_b64_b1024_stage7_trainhead_sigma025_top01_c128/results.json`
- `runs/spikformer4_adapter_b64_b1024_stage8_trainhead_sigma025_top01_c128/results.json`

Current best artifact:

- `10.28.2.47:/home/zhanshaoji/code/nano-egg-snn-es/runs/spikformer4_adapter_b64_b1024_stage8_trainhead_sigma025_top01_c128/best_state.npz`

Scope note:

- The report is a group-meeting progress report, not a paper-ready claim.
- The current result proves a repeatable ES signal on a Spikformer-style CIFAR-10 setup, but does not yet satisfy the final goal of approaching or exceeding surrogate-gradient Spikformer training.
- The EggRoll reproduction slide is used as negative diagnostic evidence: under the local 0.1B / 32-generation setting, candidate fitness variance and parameter updates were effectively absent.
