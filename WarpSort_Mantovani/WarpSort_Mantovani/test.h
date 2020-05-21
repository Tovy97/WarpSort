#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#ifndef _TEST_H
#define _TEST_H

#ifdef __cplusplus
extern "C" {
	/*
		Questa funzione esegue i controlli di correttezza dell'array ordinato prodotto.
		In particolare controlla che l'array sia ordinato in modo crescente (o decrescente se reverse è true) e se contiene tutti e
		soli gli elementi dell'array iniziale.
		Alla fine di questa funzione array iniziale è stato modificato (viene ordinato dalla CPU con un qsort per velocizzarne il
		confronto con l'array ordinato dalla GPU con il warpsort).
		I risultati del controllo vengono stampati su schermo.
	*/
	void do_tests(float* array_in, const float* array_ord, const int N, const bool reverse);
}
#endif

#endif _TEST_H