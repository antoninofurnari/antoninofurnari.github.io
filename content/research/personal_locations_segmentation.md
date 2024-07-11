---
title: "Location-Based Temporal Segmentation of Egocentric Videos"
date: 2016-01-02
draft: false
bibtex: ["@article{furnari2018personal,
  pages = { 1-12 },
  volume = { 52 },
  doi = { https://doi.org/10.1016/j.jvcir.2018.01.019 },
  issn = { 1047-3203 },
  author = { Antonino Furnari and Sebastiano Battiato and Giovanni Maria Farinella },
  url = { http://iplab.dmi.unict.it/PersonalLocationSegmentation/ },
  pdf = { ../publications/furnari2018personal.pdf },
  year = { 2018 },
  journal = { Journal of Visual Communication and Image Representation },
  title = { Personal-Location-Based Temporal Segmentation of Egocentric Video for Lifelogging Applications },
}","@inproceedings{furnari2016temporal,
  url = { http://iplab.dmi.unict.it/PersonalLocationSegmentation/ },
  pdf = { ../publications/furnari2016temporal.pdf },
  year = { 2016 },
  publisher = { Springer Lecture Notes in Computer Science },
  series = { Lecture Notes in Computer Science },
  volume = { 9913 },
  pages = { 474--489 },
  booktitle = { International Workshop on Egocentric Perception, Interaction and Computing (EPIC) in conjunction with ECCV, The Netherlands, Amsterdam, October 9 },
  title = { Temporal Segmentation of Egocentric Videos to Highlight Personal Locations of Interest },
  author = { Antonino Furnari and Giovanni Maria Farinella and Sebastiano Battiato },
}"]
teaser: "https://www.youtube.com/embed/URM0EdYuKEw"
video_teaser: true
---


Temporal video segmentation can be useful to improve the exploitation of long egocentric videos. Previous work has focused on general purpose methods designed to work on data acquired by different users. In contrast, egocentric data tends to be very personal and meaningful for the user who acquires it. In particular, being able to extract information related to personal locations can be very useful for life-logging related applications such as indexing long egocentric videos, detecting semantically meaningful video segments for later retrieval or summarization, and estimating the amount of time spent at a given location. In this paper, we propose a method to segment egocentric videos on the basis of the locations visited by user. The method is aimed at providing a personalized output and hence it allows the user to specify which locations he wants to keep track of. To account for negative locations (i.e., locations not specified by the user), we propose an effective negative rejection methods which leverages the continuous nature of egocentric videos and does not require any negative sample at training time. To perform experimental analysis, we collected a dataset of egocentric videos containing 10 personal locations of interest. Results show that the method is accurate and compares favorably with the state of the art. [Web Page](http://iplab.dmi.unict.it/PersonalLocationSegmentation/)