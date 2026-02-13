---
title: "ProSkill: Segment-Level Skill Assessment in Procedural Videos"
date: 2026-02-13T10:00:00+01:00
bibtex: ["@inproceedings{Mazzamuto2026ProSkill,
  year = { 2026 },
  booktitle = { IEEE Winter Conference on Application of Computer Vision (WACV) },
  title = { ProSkill: Segment-Level Skill Assessment in Procedural Videos },
  author = { Michele Mazzamuto and Daniele Di Mauro and Gianpiero Francesca and Giovanni Maria Farinella and Antonino Furnari },
  pdf = {https://arxiv.org/pdf/2601.20661},
  url = {https://fpv-iplab.github.io/ProSkill/}
}"]
teaser: "proskill.png"
video_teaser: false
---

Skill assessment in procedural videos is crucial for the objective evaluation of human performance in settings such as manufacturing and procedural daily tasks. Current research on skill assessment has predominantly focused on sports and lacks large-scale datasets for complex procedural activities. Existing studies typically involve only a limited number of actions, focusing either on pairwise assessments (e.g., A is better than B) or on binary labels (e.g., good execution vs needs improvement). In response to these shortcomings, we introduce PROSKILL, the first benchmark dataset for action-level skill assessment in procedural tasks. PROSKILL provides absolute skill assessment annotations, along with pairwise ones. This is enabled by a novel and scalable annotation protocol that allows for the creation of an absolute skill assessment ranking starting from pairwise assessments. This protocol leverages a Swiss Tournament scheme for efficient pairwise comparisons, which are then aggregated into consistent, continuous global scores using an ELO-based rating system. We use our dataset to benchmark the main state-of-the-art skill assessment algorithms, including both ranking-based and pairwise paradigms. The suboptimal results achieved by the current state-of-the-art highlight the challenges and thus the value of PROSKILL in the context of skill assessment for procedural videos