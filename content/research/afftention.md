---
title: "AFF-ttention! Affordances and Attention models for Short-Term Object Interaction Anticipation"
date: 2024-07-06T21:56:50+02:00
bibtex: ["@inproceedings{mur-labadia2024AFF-ttention,\n
  pdf = { https://arxiv.org/pdf/2406.01194.pdf },\n
  year = { 2024 },\n
  booktitle = { European Conference on Computer Vision (ECCV) },\n
  title = { AFF-ttention! Affordances and Attention models for Short-Term Object Interaction Anticipation },\n
  author = { Lorenzo Mur-Labadia and Ruben Martinez-Cantin and Josechu Guerrero and Giovanni Maria Farinella and Antonino Furnari },\n
  url = {https://github.com/lmur98/AFFttention}\n
}",
"@ARTICLE{mur2026integrating,
  author={Mur-Labadia, Lorenzo and Martinez-Cantin, Ruben and Guerrero, Jose J. and Farinella, Giovanni Maria and Furnari, Antonino},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Integrating Affordances and Attention models for Short-Term Object Interaction Anticipation}, 
  year={2026},
  volume={},
  number={},
  pages={1-17},
  keywords={Affordances;Videos;Transformers;Object detection;Predictive models;Hands;Forecasting;Trajectory;Training;Head;Short-term forecasting;Affordances;Egocentric video understanding},
  doi={10.1109/TPAMI.2026.3652831}}"]
teaser: afftention.png
video_teaser: false
---

Short-Term object-interaction Anticipation consists of detecting the location of the next-active objects, the noun and verb categories of the interaction, and the time to contact from the observation of egocentric video. This ability is fundamental for wearable assistants or human robot interaction to understand the user goals, but there is still room for improvement to perform STA in a precise and reliable way. In this work, we improve the performance of STA predictions with two contributions: 1. We propose STAformer, a novel attention-based architecture integrating frame guided temporal pooling, dual image-video attention, and multiscale feature fusion to support STA predictions from an image-input video pair. 2. We introduce two novel modules to ground STA predictions on human behavior by modeling affordances.First, we integrate an environment affordance model which acts as a persistent memory of interactions that can take place in a given physical scene. Second, we predict interaction hotspots from the observation of hands and object trajectories, increasing confidence in STA predictions localized around the hotspot. Our results show significant relative Overall Top-5 mAP improvements of up to +45% on Ego4D and +42% on a novel set of curated EPIC-Kitchens STA labels. We will release the code, annotations, and pre extracted affordances on Ego4D and EPIC- Kitchens to encourage future research in this area. 
<!--<a href="https://github.com/lmur98/AFFttention">Web page</a>-->