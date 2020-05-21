#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include "test.h"
#include "warpSort.cuh"
#include "gpu_utility.cuh"

/*
	Questa fuzione viene invocata nel caso in cui i parametri passati da console sono errati.
	Viene stampato a schermo che c'è stato un errore e la giusta sintassi con cui lanciare il porgramma.
	Infine viene terminato il programma con il codice EXIT_FAILURE.
*/
void wrongParameter() {
	printf("Errore: parametri errari!\n");
	printf("Lanciare: WarpSort_Mantovani.exe{ -f \"path\"| -t N | -i N | -r N} [-g] [-p S] [-c] [-o \"path\" | -s]\n");
	printf("\t-f path \t se l'array da ordinare e' nel file \"path\"\n");
	printf("\t-t N    \t se l'array da ordinare deve essere di dimensione N (con N > 0) e deve essere generato randomicamente\n");
	printf("\t-i N    \t se l'array da ordinare deve essere di dimensione N (con N > 0) e deve essere letto da standard input\n");
	printf("\t-r N    \t se l'array da ordinare deve essere di dimensione N (con N > 0) e deve essere generato randomicamente (i numeri generati verranno stampati su standard output)\n");
	printf("\t-g      \t se si vogliono effettuare i controlli sull'ordine e sulla presenza di tutti e soli gli elementi dell'array iniziale nell'array ordinato (queste operazioni possono essere molto lunghe)\n");
	printf("\t-p S    \t se si vuole usare il seme S per i generatori pseudo-randomici\n");
	printf("\t-c      \t se l'ordinamento deve essere decrescente. Se questa opzione manca l'ordinamento e' crescente\n");
	printf("\t-o path \t se si vuole salvare l'array ordinato sul file \"path\" (non avvengono controlli: questa opzione puo' portare alla sovrascrittura di file esistenti)\n");
	printf("\t-s      \t se si vuole stampare l'array ordinato su standard output\n");
	exit(EXIT_FAILURE);
}

/*
	Main del programma.
	Viene lanciato con i seguenti parametri:
		WarpSort_Mantovani.exe {-f "path"| -t N | -i N | -r N} [-g] [-p S] [-c] [-o "path" | -s]

	dove:
		-f path		se l'array da ordinare e' nel file path
		-t N		se l'array da ordinare deve essere di dimensione N (con N > 0) e deve essere generato randomicamente
		-i N		se l'array da ordinare deve essere di dimensione N (con N > 0) e deve essere letto da standard input
		-r N		se l'array da ordinare deve essere di dimensione N (con N > 0) e deve essere generato randomicamente (i numeri generati verranno stampati su standard output)
		-g			se si vogliono effettuare i controlli sull'ordine e sulla presenza di tutti e soli gli elementi dell'array iniziale nell'array ordinato (queste operazioni possono essere molto lunghe)
		-p S		se si vuole usare il seme S per i generatori pseudo-randomici
		-c			se l'ordinamento deve essere decrescente. Se questa opzione manca l'ordinamento e' crescente
		-o path		se si vuole salvare l'array ordinato sul file path (non avvengono controlli: questa opzione puo' portare alla sovrascrittura di file esistenti)
		-s			se si vuole stampare l'array ordinato su standard output
*/
int main(int argc, char* argv[]) {
	//Inizializzo i parametri necessari a gestire l'input letto da console
	bool f = false, t = false, i = false, r = false, c = false, o = false, s = false, g = false;
	char* path_input = NULL, * path_output = NULL;
	int N = 0;
	long S = (long)time(NULL);

	//Lettura dei parametri passati da console
	for (int j = 1; j < argc; ++j) {
		if (strcmp(argv[j], "-f") == 0) {
			++j;
			if (j < argc) {
				f = true;
				path_input = argv[j];
			}
		}
		else if (strcmp(argv[j], "-t") == 0) {
			++j;
			if (j < argc) {
				t = true;
				N = atoi(argv[j]);
			}
		}
		else if (strcmp(argv[j], "-i") == 0) {
			++j;
			if (j < argc) {
				i = true;
				N = atoi(argv[j]);
			}
		}
		else if (strcmp(argv[j], "-r") == 0) {
			++j;
			if (j < argc) {
				r = true;
				N = atoi(argv[j]);
			}
		}
		else if (strcmp(argv[j], "-g") == 0) {
			g = true;
		}
		else if (strcmp(argv[j], "-p") == 0) {
			++j;
			if (j < argc) {
				S = atoi(argv[j]);
			}
		}
		else if (strcmp(argv[j], "-c") == 0) {
			c = true;
		}
		else if (strcmp(argv[j], "-o") == 0) {
			++j;
			if (j < argc) {
				o = true;
				path_output = argv[j];
			}
		}
		else if (strcmp(argv[j], "-s") == 0) {
			s = true;
		}
		else {
			wrongParameter();
		}
	}

	//Controllo correttezza dei paremetri letti da console
	if (!((f && !t && !i && !r) || (!f && t && !i && !r) || (!f && !t && i && !r) || (!f && !t && !i && r))) {
		wrongParameter();
	}
	if (o && s) {
		wrongParameter();
	}
	if (f) {
		FILE* file = fopen(path_input, "r");
		if (!file) {
			wrongParameter();
		}
		N = 0;
		while (!feof(file)) {
			float temp;
			fscanf(file, "%f\n", &temp);
			N++;
		}
		fclose(file);
	}
	if (N <= 0 || S < 0) {
		wrongParameter();
	}

	//Creazione puntatori per array in input e per array copia (serve nel caso si vogliano effettuare i test di correttezza con il parametro -g)
	float* array_in;
	float* array_in_check;

	//Inizializazzione dei parametri necessari per riempire l'array
	int MULT = (int)pow(2, 3 * (floor(log10(N)) + 1));
	int S_3 = BL_4 * (int)pow(2, ceil(log2((floor(log10(N)) + 1))));
	MULT = (MULT < S_3 * K_3) ? S_3 * K_3 : MULT;
	int newN = (N % MULT == 0) ? N : N + MULT - (N % MULT);

	//Allocazione della memoria host pinned per l'array di input
	CHECK(cudaHostAlloc(&array_in, sizeof(float) * newN, cudaHostAllocDefault));

	//Allocazione della memoria host per l'array di input copia (se presente l'opzione -g)
	if (g) {
		array_in_check = (float*)malloc(sizeof(float) * newN);
	}

	printf("Inizio riempimento array... ");

	//Inizializzazione dell'array di input a seconda del parametro letto da console
	int j = 0;
	if (f) {
		FILE* file = fopen(path_input, "r");
		while (!feof(file)) {
			fscanf(file, "%f\n", &array_in[j]);
			j++;
		}
		fclose(file);
	}
	else if (i) {
		for (j = 0; j < N; ++j) {
			printf("Numero %d: ", j);
			scanf("%f", &array_in[j]);
		}
	}
	else {
		//Generazione attraverso cuRAND dell'array di input randomico
		curandGenerator_t gen;
		CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
		CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, S));
		CHECK_CURAND(curandGenerateUniform(gen, array_in, N));
		CHECK(cudaDeviceSynchronize());
		CHECK_CURAND(curandDestroyGenerator(gen));
		//Stampa dei valori generati (se presente parametro -r)
		if (r) {
			for (j = 0; j < N; ++j) {
				printf("%f\n", array_in[j]);
			}
		}
	}

	//Inserimento del padding alla dine dell'array per raggiungere una lunghezza multiplo di una potenza di 2. Il padding è il valore infinito
	for (int j = N; j < newN; ++j) {
		array_in[j] = INFINITY;
	}

	//Copia dell'array in input nell'array copia (se presente parametro -g)
	if (g) {
		memcpy(array_in_check, array_in, sizeof(float) * newN);
	}

	printf("Fatto!\nInizio ordinamento... ");

	//Creazione eventi su stream 0 per calcolare il tempo impiegato dal warpsort
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cudaEventSynchronize(start);

	//Chiamata al warpsort (che poi chiamerà dei kernel GPU). Il risultato viene salvato nell'array di output. 
	//L'ordine è sempre crescente e presenta padding
	float* array_out = warpSort(array_in, newN, S_3);

	//Registrazione evento di fine ordinamento e calcolo del tempo impiegato
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("Fatto!\n\nGPU warpsort ha ordinato %d [%d] elementi (%.2f [%.2f] GB) in %.2f secondi\n\n", N, newN, ((sizeof(float) * N) / (1024.0 * 1024.0 * 1024.0)), ((sizeof(float) * newN) / (1024.0 * 1024.0 * 1024.0)), elapsedTime / 1000);

	//Allocazione dell'array ordinato
	float* array_ord = (float*)malloc(N * sizeof(float));

	//Copia dall'array di putput all'array ordinato togliendo il padding. 
	//Se viene passato il parametro -c allora la copia avviene al contrario in modo ra dovesciare gli elementi dell'array di output e ritrovarsi un array ordinato in modo decrescente
	if (c) {
		for (int j = 0; j < N; ++j) {
			array_ord[N - 1 - j] = array_out[j];
		}
	}
	else {
		for (int j = 0; j < N; ++j) {
			array_ord[j] = array_out[j];
		}
	}

	//Rilascio aarray di output
	free(array_out);

	//Esecuzione dei test di correttezza (se presente il parametro -g)
	if (g) {
		do_tests(array_in_check, array_ord, N, c);
		free(array_in_check);
	}

	//Stampa dell'array ordinato (su console se presente il parametro -s o su file se presente il parametro -o)
	if (s) {
		printf("\nArray ordinato:\n");
		for (int j = 0; j < N; ++j) {
			printf("%f\n", array_ord[j]);
		}
	}
	else if (o) {
		FILE* file = fopen(path_output, "w");
		for (int j = 0; j < N; ++j) {
			fprintf(file, "%f\n", array_ord[j]);
		}
		fclose(file);
	}

	//Rilascio memeoria array ordinato
	free(array_ord);

	//Reset della GPU
	CHECK(cudaDeviceReset());

	return EXIT_SUCCESS;
}
