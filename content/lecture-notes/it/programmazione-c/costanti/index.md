---
title: "Costanti"
disable_share: true
toc: true
note:
    lang: ita
---
<!--<pre>
Note del corso di Laboratorio di Programmazione 1 F-N 2022/2023
Corso di Studi in Informatica, Università di Catania
Note a cura di Antonino Furnari - antonino[dot]furnari[at]unict[dot]it
</pre>-->

<!--
>  Note del corso di <a target="_blank" href="http://web.dmi.unict.it/corsi/l-31/insegnamenti?seuid=52B6DAFA-58EB-4BF7-AD16-8238324A6855">Laboratorio di Programmazione 1 F-N 2022/2023</a> <br>
> Corso di Laurea in Informatica, Università di Catania <br>
> Note a cura di Antonino Furnari - <a href="mailto:antonino.furnari@unict.it">antonino.furnari@unict.it</a>
-->

Le costanti in C++ sono delle espressioni con un valore fisso. Esistono diversi tipi di constanti. Li vediamo nelle sezioni successive.

## Letterali
I valori letterali sono il tipo più semplice di costante. Vengono utilizzati per specificare dei valori all'interno dei programmi. Ad esempio, nell'espressione:

```c
int x = 5;
```

"5" è una costante letterale. Così come le variabili, anche i letterali sono associati a un tipo. In generale possiamo avere letterali interi, a virgola mobile, caratteri, stringhe (ovvero sequenze di caratteri - le approfondiremo in seguito), puntatori e letterali definiti dall'utente. Un letterale intero è di default un `int`, mentre un letterale in virgola mobile è di default un `double`. E' tuttavia possibile specificare un tipo diverso aggiungendo dei suffissi ai letterali:

```c
75         // int
75u        // unsigned int
75l        // long
75ul       // unsigned long 
75lu       // unsigned long
```

Nel caso dei numeri in virgola mobile:
```c
3.14159L   // long double
2.50101    // float
6.02e23f   // float  
```

I letterali possono essere anche di tipo carattere e stringa:

```c
'c'        // character
"hello"    // string
```

Non abbiamo ancora formalmente parlato delle stringhe, ma abbiamo usato delle stringhe letterali negli esempi passati (ad esempio per il programma "Hello World"). Vederemo meglio e in termini più formali cosa sono le stringhe nelle prossime lezioni. Per adesso, consideriamole come delle sequenze di caratteri che servono a rappresentare una o più righe di testo.

## Espressioni costanti con un nome

In certi casi, può essere utile assegnare dei nomi alle costanti. Questo può essere fatto definendo una variabile come costante mediante la keyword `const`:

```c
const float pi = 3.14;
const int x = 0;
```

Non è possibile cambiare il valore di una costante:

```c
const float pi = 3.14;
p = 3.1415; //questa istruzione darà un errore
```

Le costanti definite possono essere utilizzate al posto dei relativi letterali. Ad esempio:

```c
const float pi = 3.14;
float r = 2;
float circle = 2 * pi * r;
```

Da notare che in C non si può separare dichiarazione e inizializzazione delle costanti. Ad esempio:

```c
const int c; //errore, variabile non inizializzata
c = 0; //errore, non si può assegnare un valore alle variabili
```

## Definizioni del preprocessore (#define)

Un altro modo di definire delle costanti è usare la direttiva al preprocessore `#define`. La definizione va fatta seguendo il formato:

```c
#define identifier replacement
```

Dopo questa definizione, ogni occorrenza di `identifier` verrà rimpiazzata con `replacement` in tutto il codice dal pre-processore del compilatore. Per convenzione, i nomi di questo tipo di costanti vengono definiti usando solo lettere maiuscole. Ad esempio:


```c
#include<stdio.h>
#define NUM 4

int main() {
    printf("%d", NUM);
}
```

    4

La definizione appena vista viene detta MACRO. Questa definizione è più efficiente di utilizzare le costante definite mediante `const`, in quanto nel primo caso sarà il preprocessore a modificare il codice e non verrà introdotta alcuna nuova variabile (con risparmio di memoria in alcuni casi). In pratica, il codice scritto sopra diventa come segue dopo che il pre-processore l'ha elaborato:


```c
#include<stdio.h>

int main() {
    printf("%d", 4);
}
```

    4

Va notato però che il pre-processore effettua una sostituzione senza prestare attenzione alla semantica, il che potrebbe portare a poca leggibilità o errori di semantica. Ad esempio, il seguente codice è del tutto valido, anche se la sua interpretazione non è chiara:


```c
#include<stdio.h>
#define NUM 3>5

int main() {
    printf("%d", NUM);
}
```

    0

## Esercizi
<table class="question">
<tr>
<td><img src="img/code.png" style="width:150px; margin-right:30px; float:left"></td>
<td>

**Esercizio**

Si definisca una costante `g` e si ponga il suo valore pari a 9.8. Si usi un tipo opportuno e si motivi la scelta. Si stampi il valore di `g`. Si producano diverse versioni che definiscono la costante in diversi modi.

 </td>
</tr>
</table>