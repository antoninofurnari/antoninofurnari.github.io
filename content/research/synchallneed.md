---
title: "Synchronization is All You Need: Exocentric-to-Egocentric Transfer for Temporal Action Segmentation with Unlabeled Synchronized Video Pairs"
date: 2024-07-06T21:53:00+02:00
draft: false
bibtex: ["@inproceedings{quattrocchi2024synchronization,
  pdf = { https://arxiv.org/pdf/2312.02638.pdf },
  year = { 2024 },
  booktitle = { European Conference on Computer Vision (ECCV) },
  title = { Synchronization is All You Need: Exocentric-to-Egocentric Transfer for Temporal Action Segmentation with Unlabeled Synchronized Video Pairs },
  author = { Camillo Quattrocchi and Antonino Furnari and Daniele Di Mauro and Mario Valerio Giuffrida and Giovanni Maria Farinella },
  url = {https://github.com/fpv-iplab/synchronization-is-all-you-need}
}","@article{quattrocchi2024synchronization,
  url = { https://github.com/fpv-iplab/synchronization-is-all-you-need },
  author = { Camillo Quattrocchi and Antonino Furnari and Daniele Di Mauro and Mario Valerio Giuffrida and Giovanni Maria Farinella },
  title = {  Exocentric-to-Egocentric Adaptation for Temporal Action Segmentation with Unlabeled Synchronized Video Pairs},
  journal = {  International Journal on Computer Vision (IJCV)  },
  year = { 2025 }
}"]
teaser: syncallneed.png
---


We consider the problem of transferring a temporal action segmentation system initially designed for exocentric (fixed) cameras to an egocentric scenario, where wearable cameras capture video data. The conventional supervised approach requires the collection and labeling of a new set of egocentric videos to adapt the model, which is costly and time-consuming. Instead, we propose a novel methodology which performs the adaptation leveraging existing labeled exocentric videos and a new set of unlabeled, synchronized exocentric-egocentric video pairs, for which temporal action segmentation annotations do not need to be collected. We implement the proposed methodology with an approach based on knowledge distillation, which we investigate both at the feature and Temporal Action Segmentation model level. Experiments on Assembly101 and EgoExo4D demonstrate the effectiveness of the proposed method against classic unsupervised domain adaptation and temporal alignment approaches. Without bells and whistles, our best model performs on par with supervised approaches trained on labeled egocentric data, without ever seeing a single egocentric label, achieving a +15.99 improvement in the edit score (28.59 vs 12.60) on the Assembly101 dataset compared to a baseline model trained solely on exocentric data. In similar settings, our method also improves edit score by +3.32 on the challenging EgoExo4D benchmark. <a href="https://github.com/fpv-iplab/synchronization-is-all-you-need">Web page</a>