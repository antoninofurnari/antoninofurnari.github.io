---
title: "Variabili e Tipi"
disable_share: true
toc: true
note:
    lang: ita
---

<!--  Note del corso di <a target="_blank" href="http://web.dmi.unict.it/corsi/l-31/insegnamenti?seuid=52B6DAFA-58EB-4BF7-AD16-8238324A6855">Laboratorio di Programmazione 1 F-N 2022/2023</a> <br>
> Corso di Laurea in Informatica, Università di Catania <br>
> Note a cura di Antonino Furnari - <a href="mailto:antonino.furnari@unict.it">antonino.furnari@unict.it</a>-->


In termini astratti, possiamo vedere una variabile come un contenitore all'interno del quale possiamo inserire dei dati. In termini più tecnici, le variabili sono dei riferimenti simbolici a delle aree fisiche della memoria del computer che possono contenere dei dati. Ad esempio, il seguente pseudo-codice mostra come usare due variabili per calcolarne la somma (attenzione, quello che segue non è codice C valido!):

```c
a = 3
b = 2
s = a+b //s conterrà il valore 5
```

<br>
Nelle sezioni seguenti vedremo come utilizzare le variabili in C.

## Nomi delle variabili

Un nome valido di variabile contiene lettere numeri e underscore. Altri simboli o spazio non possono essere inclusi nei nomi di variabili. I normi delle variabili devono obbligatoriamente iniziare con una lettera o un underscore. Esempi di nomi validi sono:

 * `var_123`;
 * `_var2`;
 * `_23_var`.
 
Esempi di nomi non validi sono:

 * `2var`;
 * `2_var`;
 * `var-2`.
 
I nomi delle variabili inoltre non possono essere uguali a delle parole chiave che sono "riservate" in C++. Queste sono:

`auto`	`double`	`int`	`struct`
`break`	`else`	`long`	`switch`
`case`	`enum`	`register`	`typedef`
`char`	`extern`	`return`	`union`
`continue`	`for`	`signed`	`void`
`do`	`if`	`static`	`while`
`default`	`goto`	`sizeof`	`volatile`
`const`	`float`	`short`	`unsigned`

Va inoltre notato che C è un linguaggio **case sensitive**, il che significa che una lettera maiuscola e una minuscola vengono considerate come diverse. Per cui i simboli `var` e `Var` sono di fatto considerati simboli diversi (es nomi di varibaili diversi).


<table class="question">
<tr>
<td><img style="float: left; margin-right: 15px; border:none; height:150px;" src="img/qmark.jpg"></td>
<td>

**Domanda**

Quali dei seguenti sono nomi validi di variabili?

 * `star_123`;
 * `123_star`;
 * `star123`;
 * `star star`;
 * `_123star`;
 * `_star`;
 * `__star`;
 * `:star`;
 * `_:star_123`.
 
 </td>
</tr>
</table>

Quando il nome di una variabile è composto da più parole, è comune seprare le diverse parole con un underscore:

```c
int variabile_composta_da_diverse_parole;
```

Questo schema è noto come **snake case**. Alternativamente, è possibile usare il **camel case**, che non separa le parole, ma rende maiuscola la prima lettera di ciascuna parola:

```c
int variabileCompostaDaDiverseParole
```

## Tipi di variabili

Qualsiasi variabile in C è associata a un tipo, che specifica la natura dei dati che essa può contenere. Ciò è necessario perché, a seconda del tipo, sarà necessaria una diversa quantità di memoria per memorizzare il dato associato alla variabile e perché dati di viersi tipi vengono rappresentati in maniera diversa in memoria. Dato che C è più uno standard che un linguaggio implementato in maniera uniforme in diverse piattaforme, in generale i tipi non hanno una dimensione ben specifica. 

I tipi fondamentali del C sono:

 * `char`: codificano caratteri quali `c`, `;` o `%`;
 * `int`: rappresentano numeri interi quali 2 o -777, che possono essere sia con segno (e dunque positivi e negativi) che senza segno;
 * `float`: numeri con la virgola a singola precisione, quali 2.2 o -7.8;
 * `double`: numeri con la virgola a doppia precisione (si possono rappresentare più cifre dopo la virgola).

É possibile inoltre applicare i modificatori `unsigned`, `short` e `long` ad alcuni tipi sopra per modificare alcune proprietà della rappresentazione dei tipo fondamentali. In particolare:

 * `unsigned` viene usato per rappresentare numeri senza segno. Può essere applicato a `char` e `int`. Ad esempio `unsigned int` permette di rappresentare interi positivi (senza senzo). Dato che viene risparmiato il bit necessario alla memorizzazione del segno, i tipi unsigned permettono di rappresentare un numero più elevato di valori delle rispettive parti con segno. Esiste anche il modificatore `signed`, che viene però applicato di default (`signed int` e `int` sono la stessa cosa). Va inoltre notato che è possible abbreviare `unsigned int` in `unsigned`;
 * `short` viene usato con `int` per indicare un tipo di intero che occupa meno spazio in memoria e che quindi può rappresentare meno numeri. Il tipo `short int` è utile quando si sa che la variabile usata rappresenterà numeri piccoli. Il tipo `short int` può essere abbreviato in `short`;
 * `long` viene usato con `int` per indicare un tipo di intero che occupa più spazio in memoria e che quindi può rappresentare più numeri. Il tipo `long int` è utile quando si sa che una variabile usata prappresenterà numeri molto grandi. Il tipo `long int` può essere abbreviato in `long`.
 
I modificatori `unsigned` e `short` o `long` possono essere combinati. Ad esempio `unsigned short int` o `unsigned short` indica un intero piccolo senza segno.

Il numero di byte necessario per rappresentare ciascun tipo varia da compilatore a compilatore. Lo standard stabilisce solo alcune direttive (ad esempio il tipo `long int` deve usare almeno tanti byte quanto `int`). Il `char` occupa generalmente un byte (8 bit), mentre `int` occupa generalemtne 4 bytes, così come `float`. `double` e `long int` occupano generalmente 8 byte, mentre `long double` occupa generalmente 16 byte.

### Formati di stampa
Prima di procedere, dobbiamo fare una piccola parentesi sui formati di stampa che si possono utilizzare mediante `printf`. Finora abbiamo visto il formato `%d` che ci permette di stampare un intero. Ogni tipo richiede però il suo specificatore di formato in `printf`. I principali sono:

 * `%c`: caratteri;
 * `%d`: interi;
 * `%f`: numeri in virgola mobile;
 * `%e`: numeri in notazione scientifica (es. 2e-5);
 * `%u`: unsigned int;
 * `%li`: long int;
 * `%lu`: unsigned long int;
 * `%Le`: long double in notazione scientifica;
 * `%s`: stringhe.
 
Torneremo a parlare in termini più accurati formati di stampa nelle prossime lezioni.

Possiamo visualizzare le dimensioni in byte e i limiti di ciascun tipo mediante il seguente programma:


```c
#include <stdio.h>
#include <limits.h>
#include <float.h>
 
int main()
{
    printf("TYPE\t\t\tSIZE\tMIN\t\t\tMAX\n");
    printf("----------------------------------------------------------------------------\n");

    printf("char\t\t\t%lu\t%d\t\t\t%d\n", sizeof(char), CHAR_MIN, CHAR_MAX);
    printf("unsigned char\t\t%lu\t%d\t\t\t%d\n", sizeof(unsigned char), 0, UCHAR_MAX);
    printf("----------------------------------------------------------------------------\n");
    printf("short int\t\t%lu\t%d\t\t\t%u\n", sizeof(short int), SHRT_MIN, SHRT_MAX);
    printf("unsigned short int\t%lu\t%d\t\t\t%u\n", sizeof(unsigned short int), 0, USHRT_MAX);
    printf("int\t\t\t%lu\t%d\t\t%d\n", sizeof(int), INT_MIN, INT_MAX);
    printf("unsigned int\t\t%lu\t%d\t\t\t%u\n", sizeof(unsigned int), 0, UINT_MAX);
    printf("long int\t\t%lu\t%ld\t%ld\n", sizeof(long int), LONG_MIN, LONG_MAX);
    printf("unsigned long int\t%lu\t%d\t\t\t%lu\n", sizeof(unsigned long int), 0, ULONG_MAX);
    printf("----------------------------------------------------------------------------\n");
    printf("float\t\t\t%lu\t%e\t\t%e\n", sizeof(float), FLT_MIN, FLT_MAX);
    printf("double\t\t\t%lu\t%e\t\t%e\n", sizeof(double), DBL_MIN, DBL_MAX);
    printf("long double\t\t%lu\t%Le\t\t%Le\n", sizeof(long double), LDBL_MIN, LDBL_MAX);
    printf("----------------------------------------------------------------------------\n");
}

```

    TYPE			SIZE	MIN			MAX
    ----------------------------------------------------------------------------
    char			1	-128			127
    unsigned char		1	0			255
    ----------------------------------------------------------------------------
    short int		2	-32768			32767
    unsigned short int	2	0			65535
    int			4	-2147483648		2147483647
    unsigned int		4	0			4294967295
    long int		8	-9223372036854775808	9223372036854775807
    unsigned long int	8	0			18446744073709551615
    ----------------------------------------------------------------------------
    float			4	1.175494e-38		3.402823e+38
    double			8	2.225074e-308		1.797693e+308
    long double		16	3.362103e-4932		1.189731e+4932
    ----------------------------------------------------------------------------


Si noti che nel programma sopra abbiamo importato gli header `limits.h` e `float.h` che mettono a disposizioni le costanti quali `INT_MIN` e `DBL_MAX` che contengono i limiti dei tipi numerici.

### Il tipo booleano

Molti linguaggi di programmazione mettono a disposizione il tipo booleano, che permette di rappresentare solo due valori (true e false). Dallo standard C99 in poi, viene messo a disposizione il tipo `_Bool` per assolvere a questo compito. Una variabile di tipo `_Bool` può contenere solo due valori `0` e `1`, che vengono assimilati ai valori `true` e `false`. Vediamo qualche esempio:



```c
#include<stdio.h>

int main() {
    _Bool b1 = 0;
    _Bool b2 = 1;    //assegnare un valore diverso da zero a
    _Bool b3 = -100; //una variabile di tipo _Bool
    _Bool b4 = 100;  //ha l'effetto di impostare quella variabile a 1 (true)
    
    printf("%d\n%d\n%d\n%d\n",b1,b2,b3,b4);
    
    // le variabili booleane possono essere utilizzate per effettuare controlli
    if(b2) {
        printf("true");
    }
}
```

    0
    1
    1
    1
    true

<br>
Il C mette a disposizione anche una libreria `stdbool` che definisce le macro `bool`, `true` e `false` che semplificano il codice quando si lavora con le variabili booleane:


```c
#include<stdio.h>
#include<stdbool.h> //senza questa direttiva, otterremmo un errore di compilazione

int main() {
    bool x = true;
    
    if(x) {
        printf("true\n");
    }
}
```

    true

<br>
Va notato che la dimensione di un `bool` è comunque di un byte (come char), anche se di fatto è necessario solo un bit per rappresentare i valori booleani:


```c
#include<stdio.h>
#include<stdbool.h> //senza questa direttiva, otterremmo un errore di compilazione

int main() {
    printf("Size of bool: %lu byte", sizeof(bool));
}
```

    Size of bool: 1 byte

<br>

<table class="question">
<tr>
</tr>
<tr>
<td><img style="float: left; margin-right: 15px; border:none; height:150px;" src="img/qmark.jpg"></td>
<td>

**Domanda**

Qual è il tipo più adeguato per i seguenti valori?

 * 12.8;
 * 12365;
 * 12;
 * 214748364799;
 * 'c';
 * true.
 
 </td>
</tr>
</table>

## Dichiarazione

Qualsiasi variabile in C è associata a un tipo, che ne specifica la natura. Quando dichiariamo una variabile, dobbiamo dichiararne sempre il tipo. Dichiarare una variabile serve a chiedere al compilatore di riservare in memoria lo spazio necessario per la variabile. Dichiarare una variabile in C è molto semplice. Basta scrivere il tipo seguito dal nome della variabile. Esempi di dichiarazione di variabili sono:

```c
int x;
float y;
bool z;
long int h;
char c;
uchar u;
```

E' possibile dichiarare più di una variabile dello stesso tipo alla volta facendole seguire da delle virgole nella stessa dichiarazione:

```c
int a, b, c;
```

Che è del tutto equivalente a:

```c
int a;
int b;
int c;
```

## Inizializzazione

Una volta inizializzata una variabile, è possibile assegnare dei valori alle variabili mediante l'operatore di assegnamento `=`. Ad esempio:


```c
#include<stdio.h>

int main() {
    int a, b, c;

    a = 1;
    b = 8;
    c = a+b;

    printf("%d",c);
}
```

    9

<br>
Il primo assegnamento che effettuiamo con una variabile viene detto "inizializzazione". Esempio:

```c
int x;
x = 0; // è una inizializzazione
x = 2; // è corretto (cambia il valore di x), ma non è una inizializzazione
```

Una volta dichiarata una variabile, essa contiene un valore indeterminato finché non viene inizializzata mediante l'operatore di assegnamento `=`. Dato che è buona norma inizializzare sempre le variabili, il linguaggio C permette di inizializzarle contestualmente alla loro definizione. Ad esempio:


```c
#include<stdio.h>

int main() {
    int x = 1;
    float y = 2.2;
    float z = x+y;

    printf("x+y=%.2f",z);
}
```

    x+y=3.20

## Conversione di tipo implicita

C converte i tipi implicitamente all'occorrenza. Vediamo qualche esempio:


```c
#include<stdio.h>
#include<stdbool.h>
int main() {
    //conversione implicita da double a int
    int x = 2.0;
    printf("%d\n",x);
    //conversione implicita da intero a float
    float y = 12;
    printf("%0.2f\n",y);
    //conversione implicita da intero a bool
    bool b = 5;
    printf("%d\n",b);
}
```

    2
    12.00
    1


## Esercizi


<table class="question">
<tr>
<td><img src="img/code.png" style="width:150px; margin-right:30px; float:left"></td>
<td>

**Esercizio**

Si scriva un programma che verifichi i limiti numerici del tipo `long double`.

 </td>
</tr>
</table>

<table class="question">
<tr>
<td><img src="img/code.png" style="width:150px; margin-right:30px; float:left"></td>
<td>

**Esercizio**

Si dichiari una variabile con un tipo opportuno per contenere il numero `1234566712345667`.

 </td>
</tr>
</table>