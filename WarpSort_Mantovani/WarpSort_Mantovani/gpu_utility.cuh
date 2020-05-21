#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef _GPU_UTILITY_H
#define _GPU_UTILITY_H

/*
	Questa macro permette di gestire gli eventuali errori ritornati da funzioni appartenenti alle CUDA API.
	In caso di errori vengono stampati:
		- il file       in cui l'errore si è verificato;
		- la riga       in cui l'errore si è verificato;
		- il codice     dell'errore;
		- la ragione    (fornita dalla funzione cudaGetErrorString) dell'errore.
	Inoltre, dopo la stampa, viene terminato il programma con il codice EXIT_FAILURE.
 */
#define CHECK(call)	{																		            \
    const cudaError_t error = call;														            	\
    if (error != cudaSuccess) {																            \
        fprintf(stderr, "Errore: %s:%d, ", __FILE__, __LINE__);								            \
        fprintf(stderr, "codice errore: %d, ragione: %s\n", error, cudaGetErrorString(error));	    	\
        exit(EXIT_FAILURE);																			    \
    }																						            \
}

 /*
	 Questa macro permette di gestire gli eventuali errori ritornati da funzioni appartenenti a cuRAND.
	 In caso di errori vengono stampati:
		 - il file       in cui l'errore si è verificato;
		 - la riga       in cui l'errore si è verificato;
		 - il codice     dell'errore;
	 Inoltre, dopo la stampa, viene terminato il programma con il codice EXIT_FAILURE.
  */
#define CHECK_CURAND(call) {                                                                            \
    curandStatus_t err = call;                                                                          \
    if (err != CURAND_STATUS_SUCCESS) {                                                                 \
        fprintf(stderr, "Errore CURAND %s:%d, ", __FILE__, __LINE__);								\
        fprintf(stderr, "codice errore: %d\n",  err);                                                   \
        exit(EXIT_FAILURE);                                                                             \
    }                                                                                                   \
}

#endif _GPU_UTILITY_H