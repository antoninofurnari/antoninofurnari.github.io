---
title: "Recognizing Personal Locations from Egocentric Videos"
date: 2015-12-01
draft: false
---



<table id="bibtexify-pl1" class="display"></table>
<pre id="bibtex-pl1" class="raw-bibtex js-hidden">
@article{furnari2016recognizing,
    author={Furnari, Antonino and Farinella, Giovanni Maria and Battiato, Sebastiano}, 
    journal={IEEE Transactions on Human-Machine Systems}, 
    title={Recognizing Personal Locations From Egocentric Videos}, 
    year={2016},
    doi={10.1109/THMS.2016.2612002}, 
    ISSN={2168-2291},
    url={http://iplab.dmi.unict.it/PersonalLocations/},
    pdf={../publications/furnari2016recognizing.pdf}
}
</pre>


<table id="bibtexify-pl2" class="display"></table>
<pre id="bibtex-pl2" class="raw-bibtex js-hidden">
@inproceedings{furnari2015recognizing,
  url = { http://iplab.dmi.unict.it/PersonalLocations/ },
  pdf = { ../publications/furnari2015recognizing.pdf },
  year = { 2015 },
  booktitle = { Workshop on Assistive Computer Vision and Robotics (ACVR) in conjunction with ICCV, Santiago, Chile, December 12 },
  page = { 393--401 },
  title = { Recognizing Personal Contexts from Egocentric Images },
  author = { Antonino Furnari and Giovanni Maria Farinella and Sebastiano Battiato },
}
</pre>

<div class='pull-left'>
<div class="video-container">
<iframe src="https://www.youtube.com/embed/5o1vxCZoI3Y" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
</div>

Contextual awareness in wearable computing allows for construction of intelligent systems which are able to interact with the user in a more natural way. In this paper, we study how personal locations arising from the user’s daily activities can be recognized from egocentric videos. We assume that few training samples are available for learning purposes. Considering the diversity of the devices available on the market, we introduce a benchmark dataset containing egocentric videos of 8 personal locations acquired by a user with 4 different wearable cameras. To make our analysis useful in real-world scenarios, we propose a method to reject negative locations, i.e., those not belonging to any of the categories of interest for the end-user. We assess the performances of the main state-of-the-art representations for scene and object classification on the considered task, as well as the influence of device-specific factors such as the Field of View (FOV) and the wearing modality. Concerning the different device-specific factors, experiments revealed that the best results are obtained using a head-mounted, wide-angular device. Our analysis shows the effectiveness of using representations based on Convolutional Neural Networks (CNN), employing basic transfer learning techniques and an entropy-based rejection algorithm. [Web Page](http://iplab.dmi.unict.it/PersonalLocations/)