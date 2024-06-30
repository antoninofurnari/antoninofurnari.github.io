---
title: "Operatori in C"
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

Una volta definite delle variabili, possiamo utilizzarle per effettuare della computazione mediante gli operatori.

## Assegnamento (=)

L'operatore assegnamento ha l'effetto di cambiare il valore contenuto all'interno di una variabile:

```c
int x; // valore indefinito
x = 2; // il valore di x è adesso 2
x = 4; // il valore di x è adesso 4
```

Le operazioni di assegnamento sono delle espressioni che restituiscono il valore dell'assegnamento. Ad esempio, l'espressione `x=5` restituisce il numero 5. Esempi:


```c
#include<stdio.h>

int main() {
    int x; //dichiara x
    int y = 2 + (x = 3); //assegna 3 a x e poi calcola y = 2 + 3
    printf("%d\n",x);
    printf("%d\n",y);
}
```

    3
    5


Anche espressioni del genere sono valide in C:


```c
#include<stdio.h>

int main() {
    int x, y, z;
    x = y = z = 2;
    
    printf("%d %d %d\n", x, y, z);
}
```

    2 2 2


<table class="question">
<tr>
<td><img style="float: left; margin-right: 15px; border:none; height:150px;" src="img/qmark.jpg"></td>
<td>

**Domanda**

Si scriva un codice che:
 
 * Definisce una nuova variabile intera `x`;
 * Assegna alla variabile `x` il valore $8$;
 * Somma alla variabile `x` il valore $3$;
 * Dichiara le variabili intere `z` e `h`;
 * Assegna a `z` e `h` il valore di `x'.
 </td>
</tr>
</table>


## Operatori aritmetici ( +, -, *, /, % )

C supporta $5$ operazioni fondamentali, associate ad altrettanti operatori:

<table class="boxed">
<tbody><tr><th>operator</th><th>description</th></tr>
<tr><td><code>+</code></td><td>addition</td></tr>
<tr><td><code>-</code></td><td>subtraction</td></tr>
<tr><td><code>*</code></td><td>multiplication</td></tr>
<tr><td><code>/</code></td><td>division</td></tr>
<tr><td><code>%</code></td><td>modulo</td></tr>
</tbody></table>

Esempi di utilizzo degli operatori:

```c
int x = 2;
int y = 3;
int somma = x+y; //somma
int media = (x+y)/2; //media
int prodotto = x*y;
int quozioente = x/y;
```

L'operatore modulo `%` restituisce invece il resto della divisione tra due numeri interi. Ad esempio:


```c
#include<stdio.h>

int main() {
    int x = 9;
    int y = 1;
    int z = x % y;

    printf("%d\n",z);
}
```

    0


Quando si usa l'operatore di divisione (`/`), bisogna stare particolarmente attenti ai **tipi degli operandi**. Se entrambi gli operandi sono interi, allora il risultato dell'operazione sarà un numero intero (il quoziente della divisione). Ad esempio:


```c
#include<stdio.h>

int main() {
    printf("%d\n", 2/2); //divisione intera
    printf("%d\n", 2/3); //divisione intera
    printf("%d\n", 5/2); //divisione intera
}
```

    1
    0
    2


Lo stesso discorso vale usando variabili di tipo int al posto dei letterali:


```c
#include<stdio.h>

int main() {
    int x = 2;
    int y = 3;
    printf("%d\n", x/y); //divisione intera
}
```

    0


Se vogliamo una divisione in virgola mobile invece che intera, dobbiamo forzare almeno uno degli operandi ad essere un numero in virgola mobile:


```c
#include<stdio.h>

int main() {
    int x = 2;
    int y = 3;
    printf("%f\n", 2.0/3);
    printf("%f\n", y/2.0);
    printf("%f\n", (x+0.0)/y);
}
```

    0.666667
    1.500000
    0.666667


<table class="question">
<tr>
<td><img style="float: left; margin-right: 15px; border:none; height:150px;" src="img/qmark.jpg"></td>
<td>

**Domanda**

Si scriva un codice che:
 
 * Dichiari le variabili `x` e `y` e le inizializzi con `2` e `3`;
 * Calcoli l'espressione $\frac{x*y}{x-y} + \frac{x}{y+2} \cdot 3$.
 </td>
</tr>
</table>

## Assegnamento composto (+=, -=, *=, /=)

Quando la variabile di cui cambiamo il valore appare anche nel lato destro dell'espressione, l'operazione è anche detta di aggiornamento. Ad esempio:

```c
float x = 1;
x = x+1;
x = x*5;
x = x-3;
x = x/2;
```

Per queste operazioni esiste anche una notazione abbreviata:

```c
float x = 1;
x+=1; //equivalente a x = x+1
x*=5; //equivalente a x = x*5
x-=3; //equivalente a x = x-3
x/=2; //equivalente a x = x/2
```

## Incremento e decremento
Le operazioni di incremento e decremento di una unità sono molto comuni e quindi esistono delle "notazioni abbreviate" per esse. Queste espressioni restituiscono inoltre un valore di ritorno:

```c
int x = 1;
++x; //incrementa x di 1 e restituisce il valore x+1
// Ad esempio:
int y = ++x; //porrà y=2
```

```c
int x = 1;
x++; //incrementa x di 1 ma restituisce il valore x
//Ad esempio:
int y = x++; //porrà y=1
```

Analoghe espressioni esistono per il decremento:

```c
int x = 1;
int y = --x; //pone y=0
```

```c
int x = 1;
int y = x--; //pone y=1
```

<table class="question">
<tr>
<td><img style="float: left; margin-right: 15px; border:none; height:150px;" src="img/qmark.jpg"></td>
<td>

**Domanda**

Qual è il valore stampato da questo codice?

```c
int x = 3;
printf("%d",x);
```

 </td>
</tr>
</table>

## Relazioni e comparazioni (==, !=, >, <, >=, <=)

Due espressioni possono essere comparate utilizzando degli appositi operatori, ad esempio per vedere se il valore contenuto in una variabile è maggiore o minore di un dato valore. Il risultato di queste operazioni di comparazione è un booleano. Gli operatori relazionali in C sono i seguenti:

<table class="boxed">
<tbody><tr><th>operator</th><th>description</th></tr>
<tr><td><code>==</code></td><td>Equal to</td></tr>
<tr><td><code>!=</code></td><td>Not equal to</td></tr>
<tr><td><code>&lt;</code></td><td>Less than</td></tr>
<tr><td><code>&gt;</code></td><td>Greater than</td></tr>
<tr><td><code>&lt;=</code></td><td>Less than or equal to</td></tr>
<tr><td><code>&gt;=</code></td><td>Greater than or equal to</td></tr>
</tbody></table>

Ecco alcuni esempi di comparazioni:

```c
(7 == 5)     // false
(5 > 4)      // true
(3 != 2)     // true
(6 >= 6)     // true
(5 < 5)      // false
```

Chiaramente è possibile comparare anche variabili al di là di costanti numeriche (letterali), Ad esempio:

```c
int a = 2;
int b = 3;
int c = 6;
(a == 5)     //false, a non è uguale a 5
(a*b >= c)   // true, 2*3 >= 6
(b+4 > a*c)  // false, 3+4 < 2*6
((b=2) == a) // true. Questa istruzione assegna 2 a b e poi controlla che 2 sia uguale al valore di a (2)
```

Bisogna fare attenzione a non confondere l'operatore di comparazione `==` con quello di assegnamento `=`. Ad esempio, nell'espressione scritta sopra `b=2` assegna `2` a `b`, non controlla che `b` sia uguale a `2`.

Il risultato di una comparazione può essere assegnato a una variabile booleana:

```c
int a = 2;
int b = 3;

bool condition = a==b; //false
```

Un utilizzo pratico dell'**operatore modulo** consiste nel verificare se un numero è multiplo di un altro numero. Infatti, in tal caso, il resto della divisione del primo numero per il secondo sarà pari a zero. Esempi:


```c
#include<stdio.h>

int main() {
    printf("%d\n",3%2==0); //false, 3 non è multiplo di 2
    printf("%d\n",9%3==0); //true, 9 è multiplo di 3
    printf("%d\n",11%5==0); //false, 11 non è multiplo di 5
}
```

    0
    1
    0


<table class="question">
<tr>
<td><img style="float: left; margin-right: 15px; border:none; height:150px;" src="img/qmark.jpg"></td>
<td>

**Domanda**

Cosa stampa il seguente codice?

```c++
int x = 3;
float z = 8.2;

cout << (x>z);
```

 </td>
</tr>
</table>

## Operatori logici

Gli operatori logici in C ci permettono di implementare l'algebra booleana. In C, l'operatore `!` rappresenta il `not` logico. Gli altri operatori logici sono `&&`, che rappresenta l'`and` logico e `||`, che rappresenta l'`or` logico. Questi operatori seguono le seguenti regole:

<table class="boxed">
<tbody><tr><th colspan="3">&amp;&amp; OPERATOR (and)</th></tr>
<tr><th><code>a</code></th><th><code>b</code></th><th><code>a &amp;&amp; b</code></th></tr>
<tr><td><code>true</code></td><td><code>true</code></td><td><code>true</code></td></tr>
<tr><td><code>true</code></td><td><code>false</code></td><td><code>false</code></td></tr>
<tr><td><code>false</code></td><td><code>true</code></td><td><code>false</code></td></tr>
<tr><td><code>false</code></td><td><code>false</code></td><td><code>false</code></td></tr>
</tbody></table>

<table class="boxed">
<tbody><tr><th colspan="3">|| OPERATOR (or)</th></tr>
<tr><th><code>a</code></th><th><code>b</code></th><th><code>a || b</code></th></tr>
<tr><td><code>true</code></td><td><code>true</code></td><td><code>true</code></td></tr>
<tr><td><code>true</code></td><td><code>false</code></td><td><code>true</code></td></tr>
<tr><td><code>false</code></td><td><code>true</code></td><td><code>true</code></td></tr>
<tr><td><code>false</code></td><td><code>false</code></td><td><code>false</code></td></tr>
</tbody></table>

Ad esempio:

```c
( (5 == 5) && (3 > 6) )  // valuta a false ( true && false )
( (5 == 5) || (3 > 6) )  // valuta a true ( true || false )
```

Quando si usano gli operatori logici, C valuta solo ciò che è necessario valutare da sinistra a destra, ignorando il resto. Ad esempio, quando viene valutata l'espressione `((5==5)||(3>6))`, C valuterà se `5==5`. Dato che il risultato è `true` e le espressioni `true || false` e `true || true` valutano entrambe a `true`, C non valuterà mai l'espressione `3>6` in quanto ciò non è necessario per trovare il valore di verità dell'espressione `((5==5)||(3>6))`. Questo tipo di valutazione è detta anche **valutazione a corto circuito** o **short-circuit evaluation** e funziona come segue:

<table class="boxed">
<tbody><tr><th>operator</th><th>short-circuit</th></tr>
<tr><td><code>&amp;&amp;</code></td><td>if the left-hand side expression is <code>false</code>, the combined result is <code>false</code> (the right-hand side expression is never evaluated).</td></tr>
<tr><td><code>||</code></td><td>if the left-hand side expression is <code>true</code>, the combined result is <code>true</code> (the right-hand side expression is never evaluated).</td></tr>
</tbody></table>


<table class="question">
<tr>
<td><img style="float: left; margin-right: 15px; border:none; height:150px;" src="img/qmark.jpg"></td>
<td>

**Domanda**

Cosa stampa il seguente codice?

```c
#include<stdio.h>
#include<stdbool.h>

int main() {
    bool a = true;
    bool b = false;

    printf("%d", (a && b) || (b && a));
}
```

 </td>
</tr>
</table>

## Operatore condizionale ternario (?)

Questo operatore valuta una espressione e restituisce un valore se l'espressione è vera, mentre ne restituisce un altro se questa è falsa. La sua sintassi è `condizione ? risultato_true : risultato_false`, dove `risultato_true` è il valore restituito se `condizione` è vera, mentre `risultato_falso` è il valore restituito se `condizione` è falsa. Vediamo qualche esempio:


```c
#include<stdio.h>

int main() {
    int a = 5;
    int b = 7;

    printf("%s\n", (a==b) ? "a==b" : "a!=b");
    printf("%s\n", (a>b) ? "a>b" : "a<=b");
    
    int max = a>b ? a : b;
    
    printf("max(a,b)=%d\n", max);
}
```

    a!=b
    a<=b
    max(a,b)=7


<table class="question">
<tr>
<td><img style="float: left; margin-right: 15px; border:none; height:150px;" src="img/qmark.jpg"></td>
<td>

**Domanda**

Cosa stampa il seguente codice?

```c
int x = 8;
int y = 3;
char z = y<x ? 'c' : 'm';

printf("%c", z);
```

 </td>
</tr>
</table>


## Operatori di casting di tipo esplicito

Gli operatori di casting di tipo permettono di convertire esplicitamente una variabile da un tipo a un altro:


```c
#include<stdio.h>

int main() {
    float x = 2.2;
    int y = (int) x;
    printf("%d\n", y);
}
```

    2


Il casting può essere utile quando si devono fare divisioni tra interi:


```c
#include<stdio.h>

int main() {
    int x = 1;
    int y = 3;
    printf("%f\n", x / (float) y);
}
```

    0.333333


<table class="question">
<tr>
<td><img style="float: left; margin-right: 15px; border:none; height:150px;" src="img/qmark.jpg"></td>
<td>

**Domanda**

Cosa stampa il seguente codice?

```c++
int x = 8;
cout << (x/3);
```


 </td>
</tr>
</table>

### Arrotondamento
Va notato che la conversione da numero in virgola mobile a int non effettua nessun arrotondamento. Ad esempio:

```c
(int)(2.8); //2, non 3
(int)(0.3); //0
(int)(8.9); //8, non 9
```

Un modo semplice per arrotondare correttamente, consiste nel sommare $0.5$ al numero da arrotondare:


```c
(int)(2.8 + 0.5); //3
(int)(0.3 + 0.5); //0
(int)(8.9 + 0.5); //9
```

In generale quindi:


```c
#include<stdio.h>
int main() {
    float x = 8.9;
    int arrotondato = (int)(x+0.5);
    printf("%d\n", arrotondato);
}
```

    9


## sizeof

L'operatore sizeof permette di conoscere la dimensione in byte di un tipo o di una variabile. Ad esempio, un intero di solito occupa 4 byte:


```c
#include<stdio.h>
int main() {
    printf("%lu\n", sizeof(int));
    
    int x = 2;
    printf("%lu\n", sizeof(x));
    
    printf("%lu\n", sizeof(2L));
}
```

    4
    4
    8


L'operatore `sizeof` restituisce un `long int`, quindi dobbiamo usare il formato `%lu`.


<table class="question">
<tr>
<td><img src="img/code.png" style="width:150px; margin-right:30px; float:left"></td>
<td>

**Esercizio**

Si scriva un programma che:

 * Dichiara le variabili float a e b;
 * Assegna ad a il valore `10E30` e a b il valore `-10E40`;
 * Dichiara la variabile float c e le assegna il risultato dell'espressione `a*b`;
 * Stampa a schermo il valore di c. Qual è il valore di c? Perché?

 </td>
</tr>
</table>

<table class="question">
<tr>
<td><img src="img/code.png" style="width:150px; margin-right:30px; float:left"></td>
<td>

**Esercizio**

Si scriva un programma che:

 * Definisca le variabili float a, b e c;
 * Inserisca dentro le tre variabili i valori `nan`, `inf`, `-inf` e `0` ottenuti mediante opportune operazioni aritmetiche.
 </td>
</tr>
</table>