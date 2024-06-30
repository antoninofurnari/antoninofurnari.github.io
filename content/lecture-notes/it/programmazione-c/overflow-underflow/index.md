---
title: "Overflow, Underflow, NaN e infinito"
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

Abbiamo visto che i tipi hanno dei limiti ben precisi. E' quindi lecito chiedersi cosa succede quando superiamo questi limiti. Abbiamo visto che importando l'header `<limits.h>`, possiamo utilizzare la costante `INT_MAX`, che indica l'intero di dimensione massima. Cosa succede se aggiungiamo una unità a `INT_MAX`? Consideriamo questo programma:


```c
#include<stdio.h>
#include<limits.h>

int main() {
    printf("INT_MAX-1:\t%d\n",INT_MAX-1);
    printf("INT_MAX:\t%d\n",INT_MAX);
    printf("INT_MAX+1:\t%d",INT_MAX+1);
}
```

    /var/folders/cs/p62_d78d49n3ddj0xlfh1h7r0000gn/T/tmpp_1f709j.c:7:36: warning: overflow in expression; result is -2147483648 with type 'int' [-Winteger-overflow]
        printf("INT_MAX+1:\t%d",INT_MAX+1);
                                       ^
    1 warning generated.


    INT_MAX-1:	2147483646
    INT_MAX:	2147483647
    INT_MAX+1:	-2147483648

Come mostrato sopra, la compilazione del programma restituisce un warning. L'espressione ha scatenato un **overflow**: in pratica non è possibile rappresentare il numero richiesto con un `int`. Il risultato sarà un numero negativo (siamo tornati all'estremo inferiore dei limiti di `int`):

Possiamo evitare l'underflow promuovendo il risultato della somma ad un `long`. Per fare ciò usiamo il letterale `1L` che indica che `1` è un `long`, per cui la somma con `INT_MAX` sarà un `long`:


```c
#include<stdio.h>
#include<limits.h>

int main() {
    printf("INT_MAX+1:\t%li",INT_MAX+1L);
}
```

    INT_MAX+1:	2147483648

Notiamo che abbiamo dovuto specificare il formato `%li` per permettere la stampa del risultato come long int.1e-350;

Allo stesso modo, il seguente codice genererà un **underflow** (non è possibile rappresentare un numero così piccolo in doppia precisione):


```c
#include<stdio.h>
#include<limits.h>

int main() {
    printf("%e",1e-350);
}
```

    /var/folders/cs/p62_d78d49n3ddj0xlfh1h7r0000gn/T/tmpe9hilh2g.c:5:17: warning: magnitude of floating-point constant too small for type 'double'; minimum is 4.9406564584124654E-324 [-Wliteral-range]
        printf("%e",1e-350);
                    ^
    1 warning generated.


    0.000000e+00

Come possiamo vedere, il numero è stato "arrotondato" a zero, con conseguente perdita di informazione. Per evitare l'underflow, possiamo transformare il letterale in un long double e sistemare il formato di stampa:


```c
#include<stdio.h>
#include<limits.h>

int main() {
    printf("%Le",1e-350L);
}
```

    1.000000e-350

Attenzione, il risultato sopra dipende dall'implementazione del compilatore. Alcuni compilatori rappresentano i long double in maniera identica ai double, nel quale caso anche l'istruzione sopra genererebbe un underflow.

Possiamo ottenere un overflow anche con i numeri in virgola mobile:


```c
#include<stdio.h>
#include<limits.h>

int main() {
    float res = 10E10*10E30;
    printf("%f", res);
}
```

    inf

In questo caso però non otteniamo un warning, ma il risultato sarà `inf` ovvero "infinito".

## Not A Number (NAN) e Infinito

L'header `<cmath>` mette a disposizione alcuni simboli speciali, quali NAN, +infinito e -infinito:


```c
#include<stdio.h>
#include<math.h>

int main() {
    printf("%f\n", NAN);
    printf("%f\n", +INFINITY);
    printf("%f\n", -INFINITY);
}
```

    nan
    inf
    -inf


Va notato che questi tre valori sono tutti considerati come numeri in virgola mobile (e si stampano con il formato `%f`). Il significato di questi tre valori è il seguente:

 * `nan`: "Not A Number" - è il risultato di una operazione aritmetica indefinita (come una divisione per zero);
 * `inf`: infinito;
 * `-inf`: meno infinito.
 
Possiamo ottenere `nan` e `infinity` anche con delle operazioni matematiche **tra numeri in virgola mobile**:


```c
#include<stdio.h>
#include<math.h>

int main() {
    printf("%f\n", 0.0/0);
    printf("%f\n", 1.0/0);
    printf("%f\n", (-1.0)/0);
    printf("%f\n", 1/INFINITY);
}
```

    nan
    inf
    -inf
    0.000000

<table class="question">
<tr>
<td><img style="float: left; margin-right: 15px; border:none; height:150px;" src="img/qmark.jpg"></td>
<td>

**Domanda**

Qual è il risultato della seguente riga di codice?

```c
printf("%f", 0/0);
```

Si compari il risultato con quello della seguente:

```c
printf("%f", 0/0.0);
```

Si discutano le eventuali differenze tra i due risultati.
 
 </td>
</tr>
</table>

## Esercizi

<table class="question">
<tr>
<td><img src="img/code.png" style="width:150px; margin-right:30px; float:left"></td>
<td>

**Esercizio**

Si scriva un programma che:

 * Dichiara le variabili intere a, b, c;
 * Le inizializza rispettivamente con i valori 12, 18, 21;
 * Stampa a schermo la somma delle tre variabili;
 * Stampa a schermo il prodotto delle tre variabili;
 * Stampa il risultato di (a+b)/c;

 </td>
</tr>
</table>

<table class="question">
<tr>
<td><img src="img/code.png" style="width:150px; margin-right:30px; float:left"></td>
<td>

**Esercizio**

Si scriva un programma che:
 
 * Dichiara la variabile intera x e la pone pari a 2;
 * Dichiara la variabile intera y e la pone pari a 7;
 * Dichiara la variabile float z e le assegna il risultato in virgola mobile dell'espressione y/x;
 * Dichiara una variabile intera h e le assegna il valore arrotondato di z.
 </td>
</tr>
</table>

<table class="question">
<tr>
<td><img src="img/code.png" style="width:150px; margin-right:30px; float:left"></td>
<td>

**Esercizio**

Si scriva un programma che:

 * Dichiara la variabile double x e le assegna il valore 3.2;
 * Divide il valore di x per due utilizzando un operatore di assegnamento composto;
 * Incrementa x di una unità utilizzando un operatore di incremento;
 * Dichiara una variabile booleana b e le assegna il valore di verità dell'espressione `x>1`;
 </td>
</tr>
</table>