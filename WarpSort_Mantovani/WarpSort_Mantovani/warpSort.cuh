#include <stdio.h>
#include <math.h>
#include <string.h>
#include "gpu_utility.cuh"

#ifndef _WARPSORT_H
#define _WARPSORT_H

/*
	Questa costante indica il numero minimo di righe in cui la fase 2 deve portare, attrverso dei merge, l'array iniziale
	Questa costante, per avere una buona parallelizzazione del programma, dovrebbe essere maggiore al numero di SM della GPU.
*/
#define L_2 4
/*
	Questa costante, moltiplicata per S_3, indica la dimensione del campione estratto durante la fase 0.
	Inolte indica serve per estrarre dai campioni i valori dei separatori (splitters) usati nella fase 3.
	Questo valore deve essere una potenza di 2.
	La moltiplicazione di BL_4 e K_3 deve essere maggiore di 128.
	Più è grande K_3, meglio avviene la selezione dei separatori, anche se ciò implica un maggiore consumo di tempo.
*/
#define K_3 32
/*
	Questa costane serve a determinare il valore di S_3
	(che moltiplicata per K_3, indica la dimensione del campione estratto durante la fase 0).
	Questo valore deve essere una potenza di 2.
	La moltiplicazione di BL_4 e K_3 deve essere maggiore di 128.
*/
#define BL_4 64

/*
	Questa funzione, preso in input un array, la sua dimensione e il valore di S_3, restituisce un array ordinato per ordine crescente.

	Array in deve essere salvato in memoria pinned (attraverso le API cudaMallocHost() o cudaHostAlloc()).
	La dimensione array_size deve essere un multiplo di una potenza di 2.
	Il valore S_3 deve essere una potenza di 2.

	Queste condizioni non vengono controllate dalla funzione e l'immissione di dati non coerenti può portare a risultati indeterminati
	(quali la creazione di un array non ordinato o il sollevamento di errori a runtime).

	Alla fine di questa funzione la memoria in cui risiede l'array viene liberata.

	L'array di ritornato ha dimensione array_size ed è salvato su memoria host non pinned.
*/
float* warpSort(float* in, const int array_size, const int S_3);

#endif _WARPSORT_H