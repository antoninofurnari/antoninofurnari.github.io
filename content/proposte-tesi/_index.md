---
title: "Proposte di Tesi"
date: 2024-08-25T17:33:11+02:00
disable_share: true
type: "page"
draft: false
---

Sono disponibile per la supervisione di un numero limitato di tesi in contemporanea durante l'anno accademico. Si considerino le seguenti indicazioni di carattere logistico:
* Se interessati a svolgere un percorso di tesi, è ideale contattare il docente circa 6 mesi prima della data di laurea prevista e idealmente dopo aver seguito una parte di uno dei corsi insegati;
* La tesi verte generalmente su tematiche affini a quelle dei <a href="https://antoninofurnari.github.io/teaching/">corsi insegnati</a> e delle <a href="https://antoninofurnari.github.io/research/">tematiche di ricerca trattate</a>;
* Gli studenti possono proporre un argomento di tesi di loro interesse (ma coerente con quanto specificato sopra) al docente.

Le seguenti proposte possono costituire delle tesi da affrontare insieme dal docente o degli spunti per la ricerca di un tema di tesi di interesse per lo studente e il docente.

## Egocentric Action Anticipation con architetture Mamba
Il problema della egocentric action anticipation consiste nel predire la prossima azione da un video acquisito mediante dispositivi indossabili. 

Gli approcci più recenti dello stato dell'arte di egocentric action anticipation hanno affrontato questo problema mediante l'uso di reti ricorrenti dapprima e Transformer successivamente. Questi modelli hanno però capacità limitate di gestire sequenze molto lunghe. Gli state space models, recententemente proposti, e i modelli di tipo Mamba, hanno dimostrato di essere in grado di gestire lunghe sequenze in maniera efficiente in fase di runtime. 

L'obiettivo della tesi è quello di sviluppare modelli di egocentric action anticipation facendo uso di modelli di tipo Mamba e opzionalmente di Transformer.

Letture di riferimento:
* [Antonino Furnari, Giovanni Maria Farinella (2021). Rolling-Unrolling LSTMs for Action Anticipation from First-Person Video. IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 43(11), pp. 4021-4036.](https://arxiv.org/pdf/2005.02190.pdf)
* [Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752.](https://arxiv.org/pdf/2312.00752)
* [Vaswani, A. et al (2017). Attention is all you need. Advances in Neural Information Processing Systems.](https://arxiv.org/pdf/1706.03762)
* [Chen, G., Huang, Y., Xu, J., Pei, B., Chen, Z., Li, Z., ... & Wang, L. (2024). Video mamba suite: State space model as a versatile alternative for video understanding. arXiv preprint arXiv:2403.09626.](https://arxiv.org/pdf/2403.09626)

## Egocentric Action Anticipation con Architetture JEPA
Il problema della egocentric action anticipation consiste nel predire la prossima azione da un video acquisito mediante dispositivi indossabili. 

I modelli attuali di action anticipation non gestiscono esplicitamente la inerente incertezza delle predizioni del futuro, trattando il problema di egocentric action anticipation come un problema di classificazione deterministico.

L'architettura JEPA è stata recentemente ipotizzata come un paradigma di learning capace di gestire queste tipo di ambiguità utilizzando modelli di minimizzazione dell'energia.

L'obiettivo della tesi è quello di sviluppare modelli di egocentric action anticipation facendo uso di metodi di representation learning basati sul paradigma JEPA.

Letture di riferimento:
* [Antonino Furnari, Giovanni Maria Farinella (2021). Rolling-Unrolling LSTMs for Action Anticipation from First-Person Video. IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 43(11), pp. 4021-4036.](https://arxiv.org/pdf/2005.02190.pdf)
* [LeCun, Yann. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27.Open Review 62.1 (2022): 1-62.](https://openreview.net/pdf?id=BZ5a1r-kVsf)

## Detection di errori da video mediante rappresentazioni a grafo e Progress-Aware Online Action Prediction
Individuare gli errori commessi da un utente in attività di tipo procedurali da video acquisiti mediante dispositivi indossabili ha diverse applicazioni quale ad esempio quella di fornire assistenza all'utente mediante realtà aumentata.

Tra i vari lavori che hanno affrontato questo problema, alcuni hanno recentemente esplorato la possibilità di utilizzare delle strutture a grafo estratte dalle annotazioni di ground truth dei video. Mentra questi sistemi funzionano bene quando testati su sequenze di azioni di ground truth, le loro performance sono limitate quando le azioni vengono predette da video per via della inerente incertezza di tali predizioni.

Alcuni lavori recenti hanno indagato la possibilità di predire il livello di progresso delle azioni come forma di stima di tale incertezza. Lo scopo della tesi è quello di integrare queste tecniche con tecniche esistenti di individuazione degli errori basate su grafi.

Letture di riferimento:
* [PREGO: Online Mistake Detection in PRocedural EGOcentric Videos. Alessandro Flaborea, Guido Maria D'Amely di Melendugno, Leonardo Plini, Luca Scofano, Edoardo De Matteis, Antonino Furnari, Giovanni Maria Farinella, Fabio Galasso; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 18483-18492](https://openaccess.thecvf.com/content/CVPR2024/papers/Flaborea_PREGO_Online_Mistake_Detection_in_PRocedural_EGOcentric_Videos_CVPR_2024_paper.pdf)
* [Seminara, L., Farinella, G. M., & Furnari, A. (2024). Differentiable Task Graph Learning: Procedural Activity Representation and Online Mistake Detection from Egocentric Videos. NeurIPS 2024.](https://arxiv.org/pdf/2406.01486)
* [Shen, Yuhan, and Ehsan Elhamifar. "Progress-aware online action segmentation for egocentric procedural task videos." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.](https://openaccess.thecvf.com/content/CVPR2024/papers/Shen_Progress-Aware_Online_Action_Segmentation_for_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.pdf)

## Action anticipation da video mediante rappresentazioni a grafo e Large Language Models
Il problema della egocentric action anticipation consiste nel predire le prossime azioni da un video acquisito mediante dispositivi indossabili. 

I large language models sono stati recentemente utilizzati con successo per la predizione di azioni future, ma la loro capacità di allucinazione ne limita le performance in diversi casi. I grafi procedurali sono stati recentemente utilizzati come un modo di condificare la conoscenza di una procedura da video. Altri lavori hanno esplorato metodologie per integrare le informazioni fornite dai grafi all'interno di modelli di linguaggio.

Lo scopo della tesi è quello di integrare la conoscenza fornita da un grafo all'interno di un modello LLM per la predizione di azioni future.

Letture di riferimento:
* [Zhao, Q., Wang, S., Zhang, C., Fu, C., Do, M. Q., Agarwal, N., ... & Sun, C. (2023). Antgpt: Can large language models help long-term action anticipation from videos?. arXiv preprint arXiv:2307.16368.](https://arxiv.org/pdf/2307.16368)
* [Seminara, L., Farinella, G. M., & Furnari, A. (2024). Differentiable Task Graph Learning: Procedural Activity Representation and Online Mistake Detection from Egocentric Videos. NeurIPS 2024.](https://arxiv.org/pdf/2406.01486)
* [Fatemi, Bahare, Jonathan Halcrow, and Bryan Perozzi. "Talk like a graph: Encoding graphs for large language models." arXiv preprint arXiv:2310.04560 (2023).](https://arxiv.org/pdf/2310.04560)
* [Perozzi, B., Fatemi, B., Zelle, D., Tsitsulin, A., Kazemi, M., Al-Rfou, R., & Halcrow, J. (2024). Let your graph do the talking: Encoding structured data for llms. arXiv preprint arXiv:2402.05862.](https://arxiv.org/pdf/2402.05862)

## Procedure planning da video mediante rappresentazioni a grafo e Large Language Models
Il problema del planning delle procedure consiste nel determinare la corretta sequenze di azioni che permettano di portare dallo stato corrente (generalmente indicato come una immagine - es. "l'immagine del pane sul tavolo") a uno stato finale (indicato anch'esso come immagine - es. "bruschette in un piatto").

I large language models sono stati recentemente utilizzati con successo per la predizione di piani procedurali. I grafi procedurali sono stati recentemente utilizzati come un modo di condificare la conoscenza di una procedura da video. Altri lavori hanno esplorato metodologie per integrare le informazioni fornite dai grafi all'interno di modelli di linguaggio.

Lo scopo della tesi è quello di integrare la conoscenza fornita da un grafo all'interno di un modello LLM per effettuare procedure planning.

Letture di riferimento:
* [Islam, M. M., Nagarajan, T., Wang, H., Chu, F. J., Kitani, K., Bertasius, G., & Yang, X. (2024). Propose, Assess, Search: Harnessing LLMs for Goal-Oriented Planning in Instructional Videos. arXiv preprint arXiv:2409.20557.](https://arxiv.org/pdf/2409.20557?)
* [Seminara, L., Farinella, G. M., & Furnari, A. (2024). Differentiable Task Graph Learning: Procedural Activity Representation and Online Mistake Detection from Egocentric Videos. NeurIPS 2024.](https://arxiv.org/pdf/2406.01486)
* [Fatemi, Bahare, Jonathan Halcrow, and Bryan Perozzi. "Talk like a graph: Encoding graphs for large language models." arXiv preprint arXiv:2310.04560 (2023).](https://arxiv.org/pdf/2310.04560)
* [Perozzi, B., Fatemi, B., Zelle, D., Tsitsulin, A., Kazemi, M., Al-Rfou, R., & Halcrow, J. (2024). Let your graph do the talking: Encoding structured data for llms. arXiv preprint arXiv:2402.05862.](https://arxiv.org/pdf/2402.05862)

