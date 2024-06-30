---
title: "Espressioni"
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

Le espressioni in C sono delle formule che coinvolgono degli operandi legati tra di loro mediante degli operatori per calcolare un valore. Esistono quattro tipi di espressioni in C:

 * Espressioni aritmetiche;
 * Espressioni relazionali;
 * Espressioni logiche;
 * Espressioni condizionali.

## Espressioni aritmetiche
Le espressioni aritmetiche includono valori numerici e uno o più operandi e uno o più operatori aritmetici. Il risultato è generalmente un valore numerico. Esempi:

```c
5 * 2;
2 + 3;
3 + (2 * 3);
1.0/2 + 8 * 2 - 10;
```

## Espressioni relazionali
Sono espressioni usate per comparare due operandi. Il risultato è un valore nullo (false) o non nullo (true). Esempi:

```c
3 > 2;
x == 2;
z > 8;
a != h;
x-y == a*h;
z >= 2;
```

## Espressioni logiche
Una espresione logica include uno o più operandi e degli operatori logici. Il risultato è un valore nullo (false) o non nullo (true). Esempi:

```c
(x > 6) && (y == 2);
a || b;
!x && (y+2>3);
!(a || b);
```

## Espressioni condizionali
Una espressione condizionale restituisce 1 se la condizione è vera, zero altrimenti. Queste espressioni fanno uso degli operatori condizionali:

```c
int res = (x > 3) ? 12 : 8;
```
