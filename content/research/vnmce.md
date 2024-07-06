---
title: "Verb-Noun Marginal Cross Entropy Loss for Egocentric Action Anticipation"
date: 2018-01-01
draft: false
---

<table id="bibtexify-8" class="display"></table>
<pre id="bibtex-8" class="raw-bibtex js-hidden">
@inproceedings{furnari2018Leveraging,
  author = { A. Furnari and S. Battiato and G. M. Farinella },
  title = {  Leveraging Uncertainty to Rethink Loss Functions and Evaluation Measures for Egocentric Action Anticipation  },
  booktitle = {  International Workshop on Egocentric Perception, Interaction and Computing (EPIC) in conjunction with ECCV  },
  pdf = { ../publications/furnari2018Leveraging.pdf },
  url = {https://github.com/fpv-iplab/action-anticipation-losses/},
  year = { 2018 },
}
</pre>
<div class='pull-left'>
<div class="video-container">
<iframe src="https://www.youtube.com/embed/w_3FiIcnUlc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
</div>

Current action anticipation approaches often neglect the in-trinsic uncertainty of future predictions when loss functions or evalua-tion  measures  are  designed.  The  uncertainty  of  future  observations  isespecially  relevant  in  the  context  of  egocentric  visual  data,  which  isnaturally  exposed  to  a  great  deal  of  variability.  Considering  the  prob-lem of egocentric action anticipation, we investigate how loss functionsand evaluation measures can be designed to explicitly take into accountthe  natural  multi-modality  of  future  events.  In  particular,  we  discusssuitable measures to evaluate egocentric action anticipation and studyhow  loss  functions  can  be  defined  to  incorporate  the  uncertainty  aris-ing from the prediction of future events. Experiments performed on theEPIC-KITCHENS dataset show that the proposed loss function allowsimproving the results of both egocentric action anticipation and recog-nition methods. [Code](https://github.com/fpv-iplab/action-anticipation-losses/)