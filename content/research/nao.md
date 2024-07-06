---
title: "Next-Active-Object-Prediction from Egocentric Video"
date: 2017-01-01
draft: false
---

<table id="bibtexify-12" class="display"></table>
<pre id="bibtex-12" class="raw-bibtex js-hidden">
@article{furnari2017next,
  title = {  Next-active-object prediction from egocentric videos  },
  journal = {  Journal of Visual Communication and Image Representation  },
  volume = {  49  },
  number = {  Supplement C  },
  pages = {  401 - 411  },
  year = {  2017  },
  issn = {  1047-3203  },
  doi = {  https://doi.org/10.1016/j.jvcir.2017.10.004  },
  url = {  http://iplab.dmi.unict.it/NextActiveObjectPrediction/  },
  pdf = {https://www.antoninofurnari.it/downloads/publications/furnari2017next.pdf},
  author = { Antonino Furnari and Sebastiano Battiato and Kristen Grauman and Giovanni Maria Farinella },
}
</pre>

<div class='pull-left'>
<div class="video-container">
<iframe width="560" height="315" src="https://www.youtube.com/embed/P_7dyRQFgZw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
</div>

Although First Person Vision systems can sense the environment from the user's perspective, they are generally unable to predict his intentions and goals. Since human activities can be decomposed in terms of atomic actions and interactions with objects, intelligent wearable systems would benefit from the ability to anticipate user-object interactions. Even if this task is not trivial, the First Person Vision paradigm can provide important cues useful to address this challenge. Specifically, we propose to exploit the dynamics of the scene to recognize next-active-objects before an object interaction actually begins. We train a classifier to discriminate trajectories leading to an object activation from all others and perform next-active-object prediction using a sliding window. Next-active-object prediction is performed by analyzing fixed-length trajectory segments within a sliding window. We investigate what properties of egocentric object motion are most discriminative for the task and evaluate the temporal support with respect to which such motion should be considered. The proposed method compares favorably with respect to several baselines on the ADL egocentric dataset which has been acquired by 20 subjects and contains 10 hours of video of unconstrained interactions with several objects. [Web Page](http://iplab.dmi.unict.it/NextActiveObjectPrediction/)