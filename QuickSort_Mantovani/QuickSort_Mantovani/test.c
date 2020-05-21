#include "test.h"

/*
	Questa funzione compara due float.
	Ritorna:
		+1 se il primo è maggiore del secondo
		-1 se il primo è minore del secondo
		0  se sono uguali
*/
int comp(const void* elem1, const void* elem2) {
	float f = *((float*)elem1);
	float s = *((float*)elem2);
	if (f > s) {
		return  1;
	}
	if (f < s) {
		return -1;
	}
	return 0;
}

/*
	Questa funzione controlla che l'array sia ordinato.
	Se il parametro reverse è false allora controlla che sia ordinato in modo crescente, altrimenti in modo decrescente.
	Alla fine questa funzione ritorna true se l'array è ordinato nel modo corretto, altriementi false.
*/
bool checkOrder(const float* array_ord, const int N, const bool reverse) {
	//Scorrendo l'intero array vengono confrontati gli elementi consecutivi.
	for (int j = 0; j < N - 1; ++j) {
		//Se l'ordine deve essere crescente e l'elemento successivo è maggiore dell'attuale si ritorna false
		if (!reverse && array_ord[j] > array_ord[j + 1]) {
			return false;
		}
		//Se l'ordine deve essere decrescente e l'elemento successivo è minore dell'attuale si ritorna false
		else if (reverse && array_ord[j] < array_ord[j + 1]) {
			return false;
		}
	}
	return true;
}

/*
	Questa funzione controlla che l'array iniaziale e quello finale contengano gli stessi elementi
	Se il parametro reverse è false allora ordina l'array iniziale in modo crescente, altrimenti in modo decrescente.
	Alla fine questa funzione ritorna true se gli array sono uguali, altriementi false.
	L'array iniziale viene modificato (viene ordinato).
*/
bool checkAll(float* array_in, const float* array_ord, const int N, const bool reverse) {

	//Si ordina con un quick sort su CPU l'array in input
	qsort(array_in, N, sizeof(float), comp);

	//Scorrendo l'intero array vengono confrontati gli elementi tra i due array.
	for (int j = 0; j < N; ++j) {
		//Se l'ordine deve essere crescente e gli elementi in egual posizione sono diversi si ritorna false
		if (!reverse && array_ord[j] != array_in[j]) {
			return false;
		}
		//Se l'ordine deve essere decrescente e gli elementi con egual offset dall'inizio e dalla fine dei due array sono diversi si ritorna false
		else if (reverse && array_ord[N - 1 - j] != array_in[j]) {
			return false;
		}
	}
	return true;
}

void do_tests(float* array_in, const float* array_ord, const int N, const bool reverse) {
	//Controllo dell'ordine dell'array ordinato
	printf("Inizio controllo ordine... ");
	if (checkOrder(array_ord, N, reverse)) {
		printf("Fatto! Array e' ordinato! Test passato!\n");
	}
	else {
		printf("Fatto! Array non e' ordinato! Test non passato!\n");
	}
	//Controllo l'array iniaziale e quello finale contengano gli stessi elementi
	printf("Inizio controllo correttezza... ");
	if (checkAll(array_in, array_ord, N, reverse)) {
		printf("Fatto! Array contiene tutti e soli gli elementi dell'array di partenza! Test passato!\n");
	}
	else {
		printf("Fatto! Array non contiene gli elementi corretti! Test non passato!\n");
	}
}