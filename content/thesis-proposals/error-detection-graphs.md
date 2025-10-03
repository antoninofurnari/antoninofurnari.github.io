---
title: "Detection di errori da video mediante rappresentazioni a grafo e Progress-Aware Online Action Prediction"
date: 2024-08-25T17:33:11+02:00
draft: false
teaser: progaware.png
---

Individuare gli errori commessi da un utente in attività di tipo procedurali da video acquisiti mediante dispositivi indossabili ha diverse applicazioni quale ad esempio quella di fornire assistenza all'utente mediante realtà aumentata.

Tra i vari lavori che hanno affrontato questo problema, alcuni hanno recentemente esplorato la possibilità di utilizzare delle strutture a grafo estratte dalle annotazioni di ground truth dei video. Mentra questi sistemi funzionano bene quando testati su sequenze di azioni di ground truth, le loro performance sono limitate quando le azioni vengono predette da video per via della inerente incertezza di tali predizioni.

Alcuni lavori recenti hanno indagato la possibilità di predire il livello di progresso delle azioni come forma di stima di tale incertezza. Lo scopo della tesi è quello di integrare queste tecniche con tecniche esistenti di individuazione degli errori basate su grafi.

Letture di riferimento:
* [PREGO: Online Mistake Detection in PRocedural EGOcentric Videos. Alessandro Flaborea, Guido Maria D'Amely di Melendugno, Leonardo Plini, Luca Scofano, Edoardo De Matteis, Antonino Furnari, Giovanni Maria Farinella, Fabio Galasso; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 18483-18492](https://openaccess.thecvf.com/content/CVPR2024/papers/Flaborea_PREGO_Online_Mistake_Detection_in_PRocedural_EGOcentric_Videos_CVPR_2024_paper.pdf)
* [Seminara, L., Farinella, G. M., & Furnari, A. (2024). Differentiable Task Graph Learning: Procedural Activity Representation and Online Mistake Detection from Egocentric Videos. NeurIPS 2024.](https://arxiv.org/pdf/2406.01486)
* [Shen, Yuhan, and Ehsan Elhamifar. "Progress-aware online action segmentation for egocentric procedural task videos." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.](https://openaccess.thecvf.com/content/CVPR2024/papers/Shen_Progress-Aware_Online_Action_Segmentation_for_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.pdf)
