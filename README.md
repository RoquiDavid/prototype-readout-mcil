# Prototype readouts over a frozen fusion encoder

Code for the article *Prototype Readouts over a Frozen Fusion Encoder for Replay-Free Multimodal Class-Incremental Learning with Missing Modalities* (D. Roqui, N. Grozavu, A. Cormier, A. Bourges).

Apart from `stats_article.py` and `fig_article.py`, the scripts are the files that produced the results of the article. We only replaced their comments and docstrings with a one-line header and reworded ten messages of `validsel.py` and `prep_cfgours.py`. No line that computes a result was changed. `stats_article.py` and `fig_article.py` were written after the runs. They read the saved results and give the same p-values and the same figures as the article. Many messages printed by the scripts are in French.

## Environment

Python 3.12.3 and the packages of `requirements.txt`:

```bash
pip install -r requirements.txt
```

The runs used one GPU.

## Data

- **MELD**: pre-extracted features of the official MELD repository (`text_emotion.pkl`, 600 dimensions, and `audio_emotion.pkl`, 300 dimensions), with the official training, validation and test split. Expected in `/data/roqui/meld_data/features/features/`.
- **IEMOCAP**: features of DialogueRNN (`IEMOCAP_features_raw.pkl`). Expected in `/data/roqui/iemocap_data/`.

## Paths

The scripts read and write under `/data/roqui`, the folder used on our server. To run them elsewhere, create this folder or replace this path in the scripts.

## Which script gives which result

| Result in the article | Scripts |
|---|---|
| Task split (Table 1) | `meld_dataset.py`, `iemocap_dataset.py` |
| Main results (Table 2) | `final_10seeds.py`, `final_10seeds_iemocap.py` (method), `baselines_sealed.py` (baselines), `grid_baselines_ci.py` (hyperparameters of the baselines, MELD validation set) |
| Weight of the routing loss of CL-MoE | `validsel.py` (jobs `ch5`), `baselines_vs.py`, `meld_baselines_vs.py` |
| ER and DER++ with three experts of rank 16 | `prep_cfgours.py`, `baselines_sealed_cfgours.py` |
| Missing modalities (Table 3, Fig. 2) | `runs_nuit.py` (run A), `baselines_drop.py` |
| Feature spaces (Table 4) | `run_untrained.py`, `run_randproj.py` |
| Classes of the first task (Table 5) | `run_t0classes.py` |
| Barlow Twins objective on the first task | `run_bt_t0.py` |
| Experts for the decision, single modality | `runs_nuit.py` (runs E and C) |
| Readouts (Table 6) | `readout_ablation.py`, `grid_shrink.py` (preliminary grid on λ) |
| Larger replay buffers (Table 7, Fig. 3) | `baselines_buffer.py` |
| Settings of the method (validation set) | `grid_archi.py`, `grid_combo.py`, `grid_space.py`, `grid_barlow.py`, `grid_latents.py` |
| Paired Wilcoxon tests (Tables 2 and 7) | `stats_article.py` (reads the saved results, no training) |
| Figures 2 and 3 | `fig_article.py` (values of Tables 3 and 7) |

Shared modules: `models_meld.py` (encoder), `continual_learning_v3_clffix.py` (training of the method), `fecam_classifier.py` and `ncm_classifier.py` (readouts), `meld_baselines.py` (baselines).

`validsel.py` also holds jobs from other chapters of the thesis. For this article, only the `ch5` jobs are used, and `baselines_vs.py` and `meld_baselines_vs.py` are the files it generated for them.

## Seeds

All results use the ten seeds 10, 20, ..., 100, and all seeds are reported.

## License

MIT (see `LICENSE`).
