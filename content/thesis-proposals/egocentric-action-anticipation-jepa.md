---
title: "Egocentric Action Anticipation con Architetture JEPA"
date: 2024-08-25T17:33:11+02:00
draft: false
teaser: jepa.png
---

Il problema della egocentric action anticipation consiste nel predire la prossima azione da un video acquisito mediante dispositivi indossabili. 

I modelli attuali di action anticipation non gestiscono esplicitamente la inerente incertezza delle predizioni del futuro, trattando il problema di egocentric action anticipation come un problema di classificazione deterministico.

L'architettura JEPA è stata recentemente ipotizzata come un paradigma di learning capace di gestire queste tipo di ambiguità utilizzando modelli di minimizzazione dell'energia.

L'obiettivo della tesi è quello di sviluppare modelli di egocentric action anticipation facendo uso di metodi di representation learning basati sul paradigma JEPA.

Letture di riferimento:
* [Antonino Furnari, Giovanni Maria Farinella (2021). Rolling-Unrolling LSTMs for Action Anticipation from First-Person Video. IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 43(11), pp. 4021-4036.](https://arxiv.org/pdf/2005.02190.pdf)
* [LeCun, Yann. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27.Open Review 62.1 (2022): 1-62.](https://openreview.net/pdf?id=BZ5a1r-kVsf)
