---
title: "PREGO: online mistake detection in PRocedural EGOcentric videos"
date: 2024-03-03
draft: false
---

<table id="bibtexify-24" class="display"></table>
<pre id="bibtex-24" class="raw-bibtex js-hidden">
@inproceedings{flaborea2024PREGO,
  year = { 2024 },
  booktitle = {  Conference on Computer Vision and Pattern Recognition (CVPR)  },
  title = {  PREGO: online mistake detection in PRocedural EGOcentric videos  },
  author = { Alessandro Flaborea and Guido D'Amely and Leonardo Plini and Luca Scofano and Edoardo De Matteis and Antonino Furnari and Giovanni Maria Farinella and Fabio Galasso },
  pdf={https://arxiv.org/pdf/2404.01933}
}
</pre>

<img src="prego.png" class='pull-left' width=500>
Promptly identifying procedural errors from egocentric videos in an online setting is highly challenging and valuable for detecting mistakes as soon as they happen. This capability has a wide range of applications across various fields, such as manufacturing and healthcare. The nature of procedural mistakes is open-set since novel types of failures might occur, which calls for one-class classifiers trained on correctly executed procedures. However, no technique can currently detect open-set procedural mistakes online. We propose PREGO, the first online one-class classification model for mistake detection in PRocedural EGOcentric videos. PREGO is based on an online action recognition component to model the current action, and a symbolic reasoning module to predict the next actions. Mistake detection is performed by comparing the recognized current action with the expected future one. We evaluate PREGO on two procedural egocentric video datasets, Assembly101 and Epic-tent, which we adapt for online benchmarking of procedural mistake detection to establish suitable benchmarks, thus defining the Assembly101-O and Epic-tent-O datasets, respectively.
<a href="https://github.com/aleflabo/PREGO">Web Page</a>