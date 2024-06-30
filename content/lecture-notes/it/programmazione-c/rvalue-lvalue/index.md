---
title: "Lvalue e Rvalue"
disable_share: true
toc: true
note:
    lang: ita
---

<!--
>  Note del corso di <a target="_blank" href="http://web.dmi.unict.it/corsi/l-31/insegnamenti?seuid=52B6DAFA-58EB-4BF7-AD16-8238324A6855">Laboratorio di Programmazione 1 F-N 2022/2023</a> <br>
> Corso di Laurea in Informatica, Università di Catania <br>
> Note a cura di Antonino Furnari - <a href="mailto:antonino.furnari@unict.it">antonino.furnari@unict.it</a>
-->

In C, esistono due tipi di entità, `lvalue` e `rvalue`, che entrano in gioco nelle operazioni di assegnamento. Questi termini appaiono a volte nei messaggi di errore dati dal compilatore, per cui è utile comprenderli. Vediamoli nelle prossime sezioni.

## Lvalue
Lvalue sta per "left value" (poiché può trovarsi a sinistra dell'operatore di assegnamento) ed è una entità che ha una locazione di memoria ben definita:

 * In ogni operazioni di assegnamento, un `lvalue` deve possedere la capacità di memorizzare il risultato dell'operazione;
 * Non possono essere `lvalue` espressioni (es. `x + y` o constanti letterali (es. `3`).
 
Esempi:

```c
int x;

x = 1; // x è un valido l-value, possiamo scrivere in x

int y = x + 1; // anche y è un valido l-value
// nell'espressione sopra, x appare a destra dell'operatore
// ciò è possibile perché gli r-value possono apparire
// anche a destra degli operatori. 

5 = x; // non consentito, 5 non è un lvalue (non possiamo modificarne il valore!)
```

## Rvalue

Un `rvalue` è una entità che non è associata a una ben definita locazione di memoria.
 * Qualsiasi cosa che possa restituire una espressione costante o un valore è un lvalue;
 * Espressioni del tipo `x + y` sono `lvalue`.
 
Esempi:

```c
int x = 1;
int y = 0;

x + y = 2; // restituisce un errore, l'espressione
// x + y non è un valido lvalue

// sia f una funzione che restituisce un valore intero
f(0) = 9; // restituisce un errore, f ritorna un valore
// che non può essere modificato. Questo sarà più chiaro quando vedremo le funzioni
```
