---
title: "The Ego-Exo4D Dataset"
date: 2024-03-03
draft: false
---

<table id="bibtexify-22" class="display"></table>
<pre id="bibtex-22" class="raw-bibtex js-hidden">
@inproceedings{grauman2023egoexo4d,
  primaryclass = { cs.CV },
  archiveprefix = { arXiv },
  eprint = { 2311.18259 },
  pdf = { https://arxiv.org/pdf/2311.18259.pdf },
  url = { https://ego-exo4d-data.org/ },
  year = { 2024 },
  booktitle = {  Conference on Computer Vision and Pattern Recognition (CVPR)  },
  title = { Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person Perspectives },
  author = { Kristen Grauman and Andrew Westbury and Lorenzo Torresani and Kris Kitani and Jitendra Malik and Triantafyllos Afouras and Kumar Ashutosh and Vijay Baiyya and Siddhant Bansal and Bikram Boote and Eugene Byrne and Zach Chavis and Joya Chen and Feng Cheng and Fu-Jen Chu and Sean Crane and Avijit Dasgupta and Jing Dong and Maria Escobar and Cristhian Forigua and Abrham Gebreselasie and Sanjay Haresh and Jing Huang and Md Mohaiminul Islam and Suyog Jain and Rawal Khirodkar and Devansh Kukreja and Kevin J Liang and Jia-Wei Liu and Sagnik Majumder and Yongsen Mao and Miguel Martin and Effrosyni Mavroudi and Tushar Nagarajan and Francesco Ragusa and Santhosh Kumar Ramakrishnan and Luigi Seminara and Arjun Somayazulu and Yale Song and Shan Su and Zihui Xue and Edward Zhang and Jinxu Zhang and Angela Castillo and Changan Chen and Xinzhu Fu and Ryosuke Furuta and Cristina Gonzalez and Prince Gupta and Jiabo Hu and Yifei Huang and Yiming Huang and Weslie Khoo and Anush Kumar and Robert Kuo and Sach Lakhavani and Miao Liu and Mi Luo and Zhengyi Luo and Brighid Meredith and Austin Miller and Oluwatumininu Oguntola and Xiaqing Pan and Penny Peng and Shraman Pramanick and Merey Ramazanova and Fiona Ryan and Wei Shan and Kiran Somasundaram and Chenan Song and Audrey Southerland and Masatoshi Tateno and Huiyu Wang and Yuchen Wang and Takuma Yagi and Mingfei Yan and Xitong Yang and Zecheng Yu and Shengxin Cindy Zha and Chen Zhao and Ziwei Zhao and Zhifan Zhu and Jeff Zhuo and Pablo Arbelaez and Gedas Bertasius and David Crandall and Dima Damen and Jakob Engel and Giovanni Maria Farinella and Antonino Furnari and Bernard Ghanem and Judy Hoffman and C. V. Jawahar and Richard Newcombe and Hyun Soo Park and James M. Rehg and Yoichi Sato and Manolis Savva and Jianbo Shi and Mike Zheng Shou and Michael Wray },
}
</pre>

<img src="egoexo4d.jpg" class='pull-left' width=500>
Ego-Exo4D presents three meticulously synchronized natural language datasets paired with videos. (1) expert commentary, revealing nuanced skills. (2) participant-provided narrate-and-act descriptions in a tutorial style. (3) one-sentence atomic action descriptions to support browsing, mining the dataset, and addressing challenges in video-language learning. Our goal is to capture simultaneous ego and multiple exo videos, together with multiple egocentric sensing modalities. Our camera configuration features Aria glasses for ego capture, including an 8 MP RGB camera and two SLAM cameras. The ego camera is calibrated and time-synchronized with 4-5 (stationary) GoPros as the exo capture devices. The number and placement of the exocentric cameras is determined per scenario in order to allow maximal coverage of useful viewpoints without obstructing the participants’ activity. Apart from multiple views, we also capture multiple modalities. Along with the dataset, we introduce four benchmarks. The recognition benchmark aims to recognize individual keysteps and infer their relation in the execution of procedural activities. The proficiency estimation benchmark aims to estimate the camera wearer's skills. The relation benchmark focuses on methods to establish spatial relationships between synchronized multi-view frames. The pose estimation benchmarks concerns the estimation of the camera pose of the camera wearer.
<a href="http://ego-exo4d-data.org">Web Page</a>