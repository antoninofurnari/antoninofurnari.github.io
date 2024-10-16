---
title: "Differentiable Task Graph Learning: Procedural Activity Representation and Online Mistake Detection from Egocentric Videos"
date: 2024-10-16T14:56:27+02:00
draft: false
bibtex: ["@inproceedings{seminara2024differentiable,
 author = {Seminara, Luigi and Farinella, Giovanni Maria and Furnari, Antonino},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {Differentiable Task Graph Learning: Procedural Activity Representation and Online Mistake Detection from Egocentric Videos},
 pdf = {https://arxiv.org/pdf/2406.01486.pdf},
 url = {https://github.com/fpv-iplab/Differentiable-Task-Graph-Learning},
 year = {2024}
}"]
teaser: dtg.png
video_teaser: false
---

Procedural activities are sequences of key-steps aimed at achieving specific goals.
They are crucial to build intelligent agents able to assist users effectively. In this
context, task graphs have emerged as a human-understandable representation of
procedural activities, encoding a partial ordering over the key-steps. While previous
works generally relied on hand-crafted procedures to extract task graphs from
videos, in this paper, we propose an approach based on direct maximum likelihood
optimization of edges’ weights, which allows gradient-based learning of task graphs
and can be naturally plugged into neural network architectures. Experiments on the
CaptainCook4D dataset demonstrate the ability of our approach to predict accurate
task graphs from the observation of action sequences, with an improvement of
+16.7% over previous approaches. Owing to the differentiability of the proposed
framework, we also introduce a feature-based approach, aiming to predict task
graphs from key-step textual or video embeddings, for which we observe emerging
video understanding abilities. Task graphs learned with our approach are also
shown to significantly enhance online mistake detection in procedural egocentric
videos, achieving notable gains of +19.8% and +7.5% on the Assembly101 and
EPIC-Tent datasets. Code for replicating experiments will be publicly released.