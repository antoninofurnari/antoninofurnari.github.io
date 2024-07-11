---
title: "Rolling-Unrolling LSTMs for Egocentric Action Anticipation"
date: 2019-01-01
draft: false
bibtex: ["@article{furnari2020rulstm,
  author = {Antonino Furnari and Giovanni Maria Farinella},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI)},
  title = {Rolling-Unrolling LSTMs for Action Anticipation from First-Person Video},
  url = {https://iplab.dmi.unict.it/rulstm},
  pdf = {https://arxiv.org/pdf/2005.02190.pdf},
  year = {2020},
  doi = {10.1109/TPAMI.2020.2992889}
}", "@inproceedings{furnari2019rulstm, 
  title = { What Would You Expect? Anticipating Egocentric Actions with Rolling-Unrolling LSTMs and Modality Attention }, 
  author = { Antonino Furnari and Giovanni Maria Farinella },
  year = { 2019 },
  booktitle = { International Conference on Computer Vision },
  pdf = {https://arxiv.org/pdf/1905.09035.pdf},
  url = {http://iplab.dmi.unict.it/rulstm}
}"]
teaser: "https://www.youtube.com/embed/EjjUdG2EYDo"
video_teaser: true
---

<!-- 
<div class='pull-left'>
<div class="video-container">
<iframe src="https://www.youtube.com/embed/buIEKFHTVIg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
</div> -->

Egocentric action anticipation consists in understanding which objects the camera wearer will interact with in the near future and which actions they will perform. We tackle the problem proposing an architecture able to anticipate actions at multiple temporal scales using two LSTMs to 1) summarize the past, and 2) formulate predictions about the future. The input video is processed considering three complimentary modalities: appearance (RGB), motion (optical
flow) and objects (object-based features). Modality-specific predictions are fused using a novel Modality ATTention (MATT) mechanism which learns to weigh modalities in an adaptive fashion. Extensive evaluations on three large-scale benchmark datasets show that our method outperforms prior art by up to +7% on the challenging EPIC-Kitchens dataset including more than 2500 actions, and generalizes to EGTEA Gaze+ and Activitynet. Our approach is also shown to generalize to the tasks of early action recognition and action recognition. Our method was ranked first in the public leaderboard of the EPIC-Kitchens egocentric action anticipation challenge 2019. [Web Page](http://iplab.dmi.unict.it/rulstm) - [Code](https://github.com/fpv-iplab/rulstm).