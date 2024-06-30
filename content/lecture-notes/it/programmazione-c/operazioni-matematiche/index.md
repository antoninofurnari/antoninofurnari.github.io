---
title: "Operazioni matematiche"
disable_share: true
toc: false
note:
    lang: ita
---

<!--
>  Note del corso di <a target="_blank" href="http://web.dmi.unict.it/corsi/l-31/insegnamenti?seuid=52B6DAFA-58EB-4BF7-AD16-8238324A6855">Laboratorio di Programmazione 1 F-N 2022/2023</a> <br>
> Corso di Laurea in Informatica, Università di Catania <br>
> Note a cura di Antonino Furnari - <a href="mailto:antonino.furnari@unict.it">antonino.furnari@unict.it</a>
-->

Oltre agli operatori di base, la libreria standard di C mette a disposizione altre operazioni matematiche quali le radici quadrate (`sqrt`), gli elevamenti a potenza (`pow`), i seni (`sin`) e i coseni (`cos`), oltre alle principali costanti matematiche. Per accedere a queste funzioni, dobbiamo importare l'header `<math.h>`. Vediamo qualche esempio:


```c
#include<stdio.h>
#include<math.h>

int main() {
    printf("sqrt(2)=%f\n", sqrt(2)); // radice quadrata di due
    printf("cbrt(2)=%f\n", cbrt(3)); // radice cubica di tre
    printf("pow(2,3)=%f\n", pow(2,3)); // 2 elevato 3
    printf("sin(M_PI)=%f\n", sin(M_PI)); // seno di Pi greca
    printf("cos(2*M_PI)=%f\n", cos(2*M_PI)); // coseno di 2*Pi greca
}
```

    sqrt(2)=1.414214
    cbrt(2)=1.442250
    pow(2,3)=8.000000
    sin(M_PI)=0.000000
    cos(2*M_PI)=1.000000


Altre operazioni matematiche utili sono `round`, `ceil` e `floor`. Tutte e tre ci permettono di convertire un numero in virgola mobile in un numero intero. Le tre funzioni effettuano però delle conversioni in maniera differente:

 * `round` arrotonda il numero in virgola mobile come siamo abituati a fare (es. $2.3$ -> $3$, $2.8$ -> $3$);
 * `floor` restituisce il numero intero più grande che sia inferiore al numero in input (es. $2.3$ -> $2$, $2.8$ -> $2$);
 * `ceil` restituisce il numero intero più piccolo che sia superiore al numero in input (es. $2.3$ -> $3$, $2.8$ -> $3$);

Vediamo degli esempi:


```c
#include<stdio.h>
#include<math.h>

int main() {
    printf("round(2.2)=%f\n", round(2.2));
    printf("round(2.6)=%f\n", round(2.6));
    
    printf("floor(2.2)=%f\n", floor(2.2));
    printf("floor(2.6)=%f\n", floor(2.6));
    
    printf("ceil(2.2)=%f\n", ceil(2.2));
    printf("ceil(2.6)=%f\n", ceil(2.6));
}
```

    round(2.2)=2.000000
    round(2.6)=3.000000
    floor(2.2)=2.000000
    floor(2.6)=2.000000
    ceil(2.2)=3.000000
    ceil(2.6)=3.000000


Si noti che il risultato di queste operazioni resta comunque un `double`.


<table class="question">
<tr>
<td><img style="float: left; margin-right: 15px; border:none; height:150px;" src="img/qmark.jpg"></td>
<td>

**Domanda**

Quale valore della variabile `x` è tale che il seguente seguente codice assegni lo stesso valore a `a` e `b` e `c`?

```c++
auto a = floor(x);
auto b = ceil(x);
auto c = round(x);
```
 
 </td>
</tr>
</table>


Possiamo trovare maggiori informazioni su `math` qui: https://devdocs.io/c/numeric/math.

## Esercizi

<table class="question">
<tr>
<td><img src="img/code.png" style="width:150px; margin-right:30px; float:left"></td>
<td>

**Esercizio**

Si scriva un programma che:

 * Definisce la variabile float `r` che rappresenta il raggio di un cerchio;
 * Assegna a `r` il valore 3.8;
 * Calcola e stampa a schermo l'area e la circonferenza del cerchio.
 </td>
</tr>
</table>


<table class="question">
<tr>
<td><img src="img/code.png" style="width:150px; margin-right:30px; float:left"></td>
<td>

**Esercizio**
Si scriva un programma che:

 * Definisce la variabile float x e la pone uguale a 9.2;
 * Calcola e stampa schermo il seno e il coseno di x;
 * Assumendo che x sia il valore di un angolo in radianti, lo converta in gradi.

 </td>
</tr>
</table>
