---
title: "Procedure planning da video mediante rappresentazioni a grafo e Large Language Models"
date: 2024-08-25T17:33:11+02:00
draft: false
teaser: graphllm.png
---

Il problema del planning delle procedure consiste nel determinare la corretta sequenze di azioni che permettano di portare dallo stato corrente (generalmente indicato come una immagine - es. "l'immagine del pane sul tavolo") a uno stato finale (indicato anch'esso come immagine - es. "bruschette in un piatto").

I large language models sono stati recentemente utilizzati con successo per la predizione di piani procedurali. I grafi procedurali sono stati recentemente utilizzati come un modo di condificare la conoscenza di una procedura da video. Altri lavori hanno esplorato metodologie per integrare le informazioni fornite dai grafi all'interno di modelli di linguaggio.

Lo scopo della tesi è quello di integrare la conoscenza fornita da un grafo all'interno di un modello LLM per effettuare procedure planning.

Letture di riferimento:
* [Islam, M. M., Nagarajan, T., Wang, H., Chu, F. J., Kitani, K., Bertasius, G., & Yang, X. (2024). Propose, Assess, Search: Harnessing LLMs for Goal-Oriented Planning in Instructional Videos. arXiv preprint arXiv:2409.20557.](https://arxiv.org/pdf/2409.20557?)
* [Seminara, L., Farinella, G. M., & Furnari, A. (2024). Differentiable Task Graph Learning: Procedural Activity Representation and Online Mistake Detection from Egocentric Videos. NeurIPS 2024.](https://arxiv.org/pdf/2406.01486)
* [Fatemi, Bahare, Jonathan Halcrow, and Bryan Perozzi. "Talk like a graph: Encoding graphs for large language models." arXiv preprint arXiv:2310.04560 (2023).](https://arxiv.org/pdf/2310.04560)
* [Perozzi, B., Fatemi, B., Zelle, D., Tsitsulin, A., Kazemi, M., Al-Rfou, R., & Halcrow, J. (2024). Let your graph do the talking: Encoding structured data for llms. arXiv preprint arXiv:2402.05862.](https://arxiv.org/pdf/2402.05862)
