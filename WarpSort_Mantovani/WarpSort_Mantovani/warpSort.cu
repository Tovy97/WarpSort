#include "warpSort.cuh"

/*
	Questo kernel divide l'array di input in blocchi da 128 e ordina in modo crescente i singoli blocchi.
	Su ogni blocco deve lavorare un warp.
	Per ordinare i blocchi viene usato un bitonic sort.
	Il kernel lavora in place.
	Il kernel fa uso di shared memory e garantisce accessi alla global memory coalesced e allineati.
	Alla fine del kernel nell'array in input si trovano i blocchi ordinati.
*/
__global__ void bitonic_sort(float* in) {
	//Inizializzazione del id the thread e dell'offset rispetto all'array in input su cui il warp deve lavorare
	int tid = threadIdx.x;
	int offset = blockDim.x * blockIdx.x * 4;

	//Creazione di un tile di 128 elementi nella shared memory (SMEM)
	__shared__ float tile[128];

	//Copia dei dati del blocco dall'array di input che si trova in global memory al tile che si trova in SMEM attraverso un unrolling (ogni thread del warp carica 4 dati)
	//Vengono garantiti accessi coalesced e allineati alla global memory
	tile[tid] = in[tid + offset];
	tile[tid + 32] = in[tid + offset + 32];
	tile[tid + 64] = in[tid + offset + 64];
	tile[tid + 96] = in[tid + offset + 96];

	//Svolgimento delle prime log(n) - 1 fasi del bitonic sort. Alla fine di questa procedura gli elementi del blocco formano una bitonic sequence
	for (int i = 2, group_witdh = 1, group = tid % 32; i < 128; i *= 2, group_witdh *= 2, group /= 2) {
		int index = (tid % group_witdh) + group * i * 2;
		for (int j = i / 2, k = group_witdh; j > 0; j /= 2, k /= 2) {
			if (tile[index] > tile[index + j]) {
				float temp = tile[index];
				tile[index] = tile[index + j];
				tile[index + j] = temp;
			}
			int index2 = 127 - index;
			if (tile[index2] > tile[index2 - j]) {
				float temp = tile[index2];
				tile[index2] = tile[index2 - j];
				tile[index2 - j] = temp;
			}
			if ((tid % k) >= k / 2) {
				index = index + k / 2;
			}
		}
	}

	//Svolgimento dell'ultima fase del bitonic sort (ovvero il bitonic merge). Alla fine di questa procedura gli elementi del blocco sono ordinati
	int index = tid;
	for (int j = 64, k = 32; j > 0; j /= 2, k /= 2) {
		if (tile[index] > tile[index + j]) {
			float temp = tile[index];
			tile[index] = tile[index + j];
			tile[index + j] = temp;
		}
		int index2 = index + k;
		if (tile[index2] > tile[index2 + j]) {
			float temp = tile[index2];
			tile[index2] = tile[index2 + j];
			tile[index2 + j] = temp;
		}

		if (k == 1) {
			index = tid * 4;
			k = 4;
		}
		else if ((tid % k) >= k / 2) {
			index = index + k + k / 2;
		}
	}

	//Copia dei dati del blocco dal tile che si trova in SMEM all'array di input che si trova in global memory attraverso un unrolling (ogni thread del warp carica 4 dati)
	//Vengono garantiti accessi coalesced e allineati alla global memory
	in[tid + offset] = tile[tid];
	in[tid + offset + 32] = tile[tid + 32];
	in[tid + offset + 64] = tile[tid + 64];
	in[tid + offset + 96] = tile[tid + 96];
}

/*
	Questo kernel fonde blocchi ordinati di dimensione size a due a due garantendo che i blocchi risultato della fusione rimangano ordinati.
	Su ogni coppia di blocchi deve lavorare un warp.
	Per fondere i blocchi viene usato un bitonic merge.
	Il kernel lavora out of place.
	Il kernel fa uso di shared memory e garantisce accessi alla global memory coalesced e allineati.
	Alla fine del kernel nell'array in output si trovano i blocchi fusi e ordinati.
*/
__global__ void bitonic_merge(const float* in, float* out, const int size) {
	//Inizializzazione del id the thread e dell'offset rispetto all'array in input su cui il warp deve lavorare
	int tid = threadIdx.x;
	int offset = blockDim.x * blockIdx.x * (size / 32);

	//Creazione di un tile di 64 elementi nella shared memory (SMEM)
	__shared__ float tile[64];

	//Inizializzazione dei puntatori ai due blocchi da fondere nella global memory
	const float* A = &in[offset];
	const float* B = &in[offset + (size / 2)];

	//Copia dei dati del blocco dall'array di input che si trova in global memory al tile che si trova in SMEM (ogni thread del warp carica 2 dati)
	//Vengono garantiti accessi coalesced e allineati alla global memory
	tile[31 - tid] = A[tid];
	tile[32 + tid] = B[tid];

	//Inizializazzione dei parametri necessari alla fusione
	float maxA = tile[0];
	float maxB = tile[63];
	int indexA = 32;
	int indexB = 32;
	int indexOut = 0;

	while (true) {
		//Svolgimento del bitonic merge. Alla fine di questa procedura gli elementi del tile sono fusi e ordinati
		int index = tid;
		for (int j = 32, k = 32; j > 0; j /= 2, k /= 2) {
			if (tile[index] > tile[index + j]) {
				float temp = tile[index];
				tile[index] = tile[index + j];
				tile[index + j] = temp;
			}
			if ((tid % k) >= k / 2) {
				index = index + k / 2;
			}
		}
		//Copia dei dati del blocco dal tile che si trova in SMEM all'array di output che si trova in global memory (ogni thread del warp carica 1 dato)
		//Vengono garantiti accessi coalesced e allineati alla global memory
		out[offset + indexOut + tid] = tile[tid];

		//Aggiornamento dei parametri necessari alla fusione.
		indexOut += 32;
		if (indexOut >= size - 32) {
			break;
		}
		//Copia dei dati del blocco dall'array di input che si trova in global memory al tile che si trova in SMEM (ogni thread del warp carica 1 dato)
		//Vengono garantiti accessi coalesced e allineati alla global memory
		//Inoltre avviene l'aggiornamento dei parametri necessari alla fusione.
		if ((maxA < maxB && indexA < (size / 2)) || (indexB >= (size / 2))) {
			tile[31 - tid] = A[indexA + tid];
			maxA = tile[0];
			indexA += 32;
		}
		else {
			tile[31 - tid] = B[indexB + tid];
			maxB = tile[0];
			indexB += 32;
		}
	}

	//Copia dei dati del blocco dal tile che si trova in SMEM all'array di output che si trova in global memory (ogni thread del warp carica 1 dato)
	//Vengono garantiti accessi coalesced e allineati alla global memory
	out[offset + indexOut + tid] = tile[tid + 32];
}

/*
	Questo kernel fonde a due a due blocchi ordinati i cui valori sono compresi tra due splitter garantendo che i blocchi risultato della fusione rimangano ordinati.
	Su ogni coppia di blocchi deve lavorare un warp.
	Per fondere i blocchi viene usato un bitonic merge.
	Il kernel lavora out of place.
	Il kernel fa uso di shared memory e garantisce accessi alla global memory coalesced.
	Alla fine del kernel nell'array in output si trovano i blocchi fusi e ordinati.
*/
__global__ void bitonic_merge_2(const float* in, float* out, const int* count_total, const int n, int i, const int S_3) {
	//Inizializzazione del id the thread e del numero di blocco si cui il warp deve lavorare
	int sub_array = blockIdx.x;
	int tid = threadIdx.x;

	//Inizializazzione degli offset dei blocchi rispetto all'arry in input
	int index1 = 0;
	int index2 = 0;
	for (int j = 0; j < sub_array; ++j) {
		index1 += count_total[i * (S_3 + 1) + j];
		index2 += count_total[(i - 1) * (S_3 + 1) + j];
	}

	//Inizializzazione dei puntatori ai due blocchi da fondere nella global memory
	const float* A = &in[i * n + index1];
	const float* B = &in[index2];

	//Inizializazzione del numero di elementi di ogni blocco
	const int numElA = count_total[i * (S_3 + 1) + sub_array];
	const int numElB = count_total[(i - 1) * (S_3 + 1) + sub_array];

	//Creazione di un tile di 64 elementi nella shared memory (SMEM)
	__shared__ float tile[64];

	//Copia dei dati del blocco dall'array di input che si trova in global memory al tile che si trova in SMEM (ogni thread del warp carica 2 dati)
	//Vengono garantiti accessi coalesced alla global memory. C'è aggiunta di padding se necessario.
	if (tid < numElA) {
		tile[31 - tid] = A[tid];
	}
	else {
		tile[31 - tid] = INFINITY;
	}
	if (tid < numElB) {
		tile[32 + tid] = B[tid];
	}
	else {
		tile[32 + tid] = INFINITY;
	}

	//Inizializazzione dei parametri necessari alla fusione
	float maxA = tile[0];
	float maxB = tile[63];
	int indexA = 32;
	int indexB = 32;
	int indexOut = 0;

	//Inizializazzione dell'offset sull'array di output
	const int offset = index1 + index2;

	//Creazione di un array di 32 elementi nella shared memory (SMEM)
	__shared__ bool enter[32];

	while (true) {

		//Svolgimento del bitonic merge. Alla fine di questa procedura gli elementi del tile sono fusi e ordinati
		int index = tid;
		for (int j = 32, k = 32; j > 0; j /= 2, k /= 2) {
			if (tile[index] > tile[index + j]) {
				float temp = tile[index];
				tile[index] = tile[index + j];
				tile[index + j] = temp;
			}
			if ((tid % k) >= k / 2) {
				index = index + k / 2;
			}
		}

		//Copia dei dati del blocco dal tile che si trova in SMEM all'array di output che si trova in global memory (ogni thread del warp carica 1 dato)
		//Vengono garantiti accessi coalesced alla global memory
		if (indexOut + tid < numElA + numElB) {
			out[indexOut + offset + tid] = tile[tid];
			enter[tid] = true;
		}
		else {
			enter[tid] = false;
		}

		//Aggiornamento dell'indice dell'array di output attraverso un unrolling
		indexOut = (enter[0]) ? indexOut + 1 : indexOut;
		indexOut = (enter[1]) ? indexOut + 1 : indexOut;
		indexOut = (enter[2]) ? indexOut + 1 : indexOut;
		indexOut = (enter[3]) ? indexOut + 1 : indexOut;
		indexOut = (enter[4]) ? indexOut + 1 : indexOut;
		indexOut = (enter[5]) ? indexOut + 1 : indexOut;
		indexOut = (enter[6]) ? indexOut + 1 : indexOut;
		indexOut = (enter[7]) ? indexOut + 1 : indexOut;
		indexOut = (enter[8]) ? indexOut + 1 : indexOut;
		indexOut = (enter[9]) ? indexOut + 1 : indexOut;
		indexOut = (enter[10]) ? indexOut + 1 : indexOut;
		indexOut = (enter[11]) ? indexOut + 1 : indexOut;
		indexOut = (enter[12]) ? indexOut + 1 : indexOut;
		indexOut = (enter[13]) ? indexOut + 1 : indexOut;
		indexOut = (enter[14]) ? indexOut + 1 : indexOut;
		indexOut = (enter[15]) ? indexOut + 1 : indexOut;
		indexOut = (enter[16]) ? indexOut + 1 : indexOut;
		indexOut = (enter[17]) ? indexOut + 1 : indexOut;
		indexOut = (enter[18]) ? indexOut + 1 : indexOut;
		indexOut = (enter[19]) ? indexOut + 1 : indexOut;
		indexOut = (enter[20]) ? indexOut + 1 : indexOut;
		indexOut = (enter[21]) ? indexOut + 1 : indexOut;
		indexOut = (enter[22]) ? indexOut + 1 : indexOut;
		indexOut = (enter[23]) ? indexOut + 1 : indexOut;
		indexOut = (enter[24]) ? indexOut + 1 : indexOut;
		indexOut = (enter[25]) ? indexOut + 1 : indexOut;
		indexOut = (enter[26]) ? indexOut + 1 : indexOut;
		indexOut = (enter[27]) ? indexOut + 1 : indexOut;
		indexOut = (enter[28]) ? indexOut + 1 : indexOut;
		indexOut = (enter[29]) ? indexOut + 1 : indexOut;
		indexOut = (enter[30]) ? indexOut + 1 : indexOut;
		indexOut = (enter[31]) ? indexOut + 1 : indexOut;
		
		if (indexOut >= numElA + numElB) {
			break;
		}

		//Copia dei dati del blocco dall'array di input che si trova in global memory al tile che si trova in SMEM (ogni thread del warp carica 1 dato)
		//Vengono garantiti accessi coalesced alla global memory. C'è aggiunta di padding se necessario.
		//Inoltre avviene l'aggiornamento dei parametri necessari alla fusione.
		if ((maxA < maxB && indexA < numElA) || (indexB >= numElB)) {
			if (indexA + tid < numElA) {
				tile[31 - tid] = A[indexA + tid];
			}
			else {
				tile[31 - tid] = INFINITY;
			}
			maxA = tile[0];
			indexA += 32;
		}
		else {
			if (indexB + tid < numElB) {
				tile[31 - tid] = B[indexB + tid];
			}
			else {
				tile[31 - tid] = INFINITY;
			}
			maxB = tile[0];
			indexB += 32;
		}
	}
}

float* warpSort(float* in, const int array_size, const int S_3) {
	/************************************************** Init Fase 0 **************************************************/

	//Creazione di uno stream dedicato alla fase 0	
	cudaStream_t streamFase0;
	CHECK(cudaStreamCreate(&streamFase0));

	//Allocazione della memoria host e device per il campionamento	
	float* d_sample, * d_sample_out;
	float* sample = (float*)malloc(sizeof(float) * S_3 * K_3);
	float* sample_out = (float*)malloc(sizeof(float) * S_3 * K_3);
	CHECK(cudaMalloc(&d_sample, sizeof(float) * S_3 * K_3));
	CHECK(cudaMalloc(&d_sample_out, sizeof(float) * S_3 * K_3));

	//Campionamento dell'array di input su memoria host pinned
	for (int i = 0, j = 0; i < S_3 * K_3; ++i, j += (array_size / (S_3 * K_3))) {
		sample[i] = in[j % (array_size - 1)];
	}

	/************************************************** Fase 1 **************************************************/

	//Allocazione della memoria device per l'array in input	
	float* array_in;
	CHECK(cudaMalloc(&array_in, sizeof(float) * array_size));

	//Creazione dello stream per le fasi principali 
	cudaStream_t streamFasiPrincipali;
	CHECK(cudaStreamCreate(&streamFasiPrincipali));

	//Copia asincrona dalla memoria host pinned dell'array in input a quella device attraverso lo stream delle fasi principali	
	CHECK(cudaMemcpyAsync(array_in, in, sizeof(float) * array_size, cudaMemcpyDeviceToDevice, streamFasiPrincipali));

	//Lancio dei kernel della fase 1 (che eseguono un bitonin_sort) sull'array di input attraverso lo stream delle fasi principali	
	bitonic_sort <<< array_size / 128, 32, 0, streamFasiPrincipali >>> (array_in);

	/************************************************** Fase 2 **************************************************/

	//Creazione dei parametri necessari per lo svolgimento della fase 2
	int n = 256;
	int l = array_size / 256;
	bool enter = false;

	//Allocazione della memoria device per l'array in output
	float* array_out;
	CHECK(cudaMalloc(&array_out, sizeof(float) * array_size));

	//Svolgimetno della fase 2
	while (l >= L_2 && array_size % n == 0) {

		//Lancio del kernel della fase 2 (che esegue un bitonin_merge) sull'array di input attraverso lo stream delle fasi principali
		bitonic_merge <<< l, 32, 0, streamFasiPrincipali >>> (array_in, array_out, n);

		//Copia asincrona dalla memoria device dell'array di output a quella dell'array di input attraverso lo stream delle fasi principali
		CHECK(cudaMemcpyAsync(array_in, array_out, sizeof(float) * array_size, cudaMemcpyDeviceToDevice, streamFasiPrincipali));

		//Aggiornamento dei parametri necessari per lo svolgimento della fase 2
		n *= 2;
		l = array_size / n;
		enter = true;
	}

	//Aggiornamento dei parametri necessari per lo svolgimento della fase 3 e 4
	n = (enter) ? n / 2 : 128;
	l = array_size / n;

	/************************************************** Fase 0 **************************************************/

	//Copia asincrona dalla memoria host pinned del campione alla memoria device attraverso lo stream della fase 0
	CHECK(cudaMemcpyAsync(d_sample, sample, sizeof(float) * S_3 * K_3, cudaMemcpyHostToDevice, streamFase0));

	//Lancio dei kernel della fase 0 (che eseguono un bitonin_sort) sul campione attraverso lo stream della fase 0
	bitonic_sort <<< ((S_3 * K_3) / 128), 32, 0, streamFase0 >>> (d_sample);

	//Creazione dei parametri necessari per lo svolgimento della seconda parte della fase 0
	int temp_n = 256;
	int temp_l = (S_3 * K_3) / 256;

	//Svolgimetno della seconda parte della fase 0
	while (temp_l != 1) {

		//Lancio del kernel della seconda parte della fase 0 (che esegue un bitonin_merge) sul campione attraverso lo stream della fase 0
		bitonic_merge <<< temp_l, 32, 0, streamFase0 >>> (d_sample, d_sample_out, temp_n);

		//Copia asincrona dalla memoria device del campione di output a quella del campione di input attraverso lo stream della fase 0
		CHECK(cudaMemcpyAsync(d_sample, d_sample_out, sizeof(float) * S_3 * K_3, cudaMemcpyDeviceToDevice, streamFase0));

		//Aggiornamento dei parametri necessari per lo svolgimento della seconda parte della fase 0
		temp_n *= 2;
		temp_l = (S_3 * K_3) / temp_n;
	}

	//Allocazione della memoria host per gli splitters
	float* splitters = (float*)malloc(sizeof(float) * S_3);

	//Copia asincrona dalla memoria pinned host del campione di output a quella device attraverso lo stream della fase 0
	CHECK(cudaMemcpyAsync(sample_out, d_sample_out, sizeof(float) * S_3 * K_3, cudaMemcpyDeviceToHost, streamFase0));

	//Attesa della conclusione del lavoro dello stream della fase 0 e disruzione dello stream della fase 0
	CHECK(cudaStreamSynchronize(streamFase0));
	CHECK(cudaStreamDestroy(streamFase0));

	//Estrazione degli splitters dall'array campione 
	for (int i = 0; i < S_3; ++i) {
		splitters[i] = sample_out[i * K_3];
	}

	//Rilascio della memoria host e device in cui si trova l'array campione
	CHECK(cudaFree(d_sample));
	CHECK(cudaFree(d_sample_out));
	free(sample);
	free(sample_out);

	/************************************************** Fase 3 **************************************************/

	//Copia asincrona dalla memoria device dell'array di input a quella host attraverso lo stream della fase 2
	CHECK(cudaMemcpyAsync(in, array_in, sizeof(float) * array_size, cudaMemcpyDeviceToDevice, streamFasiPrincipali));

	//Allocazione della memoria host e device per gli contenere la matrice contatore degli elementi tra due splitter consecutivi
	int* local_count = (int*)malloc(l * (S_3 + 1) * sizeof(int));
	int* d_local_count;
	CHECK(cudaMalloc(&d_local_count, l * (S_3 + 1) * sizeof(int)));

	//Inizializzazione a zero della matrice contatore sulla memoria host
	memset(local_count, 0, sizeof(int) * l * (S_3 + 1));

	//Attesa della conclusione del lavoro dello stream della fase 2 e distruzione dello stream della fase 2
	CHECK(cudaStreamSynchronize(streamFasiPrincipali));

	//Svolgimetno della fase 3 in cui si popola la matrice host contatore riga per riga con il numero di elementi compresi tra due splitter consecutivi
	for (int i = 0; i < l; ++i) {
		int pos = 0;
		int c = 0;
		for (int j = 0; j < n; ++j) {
			if (in[i * n + j] > splitters[pos]) {
				local_count[i * (S_3 + 1) + pos] = c;
				c = 0;
				++pos;
				if (pos >= S_3) {
					c = n - j;
					break;
				}
				--j;
			}
			else {
				++c;
			}
		}
		local_count[i * (S_3 + 1) + pos] = c;
	}

	//Rilascio della memoria host in cui si trova l'array degli splitters
	free(splitters);

	//Rilascio della memoria pinned in cui si trova l'array in input
	CHECK(cudaFreeHost(in));

	/************************************************** Fase 4 **************************************************/

	//Copia asincrona dalla memoria host della matrice contatori a quella device attraverso lo stream delle fasi principali
	CHECK(cudaMemcpyAsync(d_local_count, local_count, l * (S_3 + 1) * sizeof(int), cudaMemcpyHostToDevice, streamFasiPrincipali));

	//Svolgimento della fase 4
	for (int i = 0; i < l - 1; ++i) {

		//Lancio del kernel della fase 4 (che esegue un bitonin_merge) sull'array attraverso lo stream delle fasi principali
		bitonic_merge_2 <<< (S_3 + 1), 32, 0, streamFasiPrincipali >>> (array_in, array_out, d_local_count, n, i + 1, S_3);

		//Attesa della conclusione del lavoro dello stream delle fasi principali
		CHECK(cudaStreamSynchronize(streamFasiPrincipali));

		//Aggiornamento della matrice su memoria host dei contatori
		for (int j = 0; j <= S_3; ++j) {
			local_count[j + (1 + i) * (S_3 + 1)] += local_count[j + i * (S_3 + 1)];
		}

		//Copia asincrona dalla memoria host della matrice contatori a quella device attraverso lo stream delle fasi principali
		CHECK(cudaMemcpyAsync(d_local_count, local_count, (2 + i) * (S_3 + 1) * sizeof(int), cudaMemcpyHostToDevice, streamFasiPrincipali));

		//Copia asincrona dalla memoria device dell'array di output a quella device dell'array di input attraverso lo stream delle fasi principali
		CHECK(cudaMemcpyAsync(array_in, array_out, (i + 2) * n * sizeof(float), cudaMemcpyDeviceToDevice, streamFasiPrincipali));
	}

	//Allocazione della memoria host per l'array di output
	float* out = (float*)malloc(sizeof(float) * array_size);

	//Copia asincrona dalla memoria device dell'array di output a quella host dell'array di output attraverso lo stream delle fasi principali
	CHECK(cudaMemcpyAsync(out, array_in, array_size * sizeof(float), cudaMemcpyDeviceToHost, streamFasiPrincipali));

	//Rilascio della memoria host e device delle matrici di contati, dell'array di output 
	free(local_count);
	CHECK(cudaFree(d_local_count));
	CHECK(cudaFree(array_out));

	//Attesa della conclusione del lavoro dello stream delle fasi principali
	CHECK(cudaStreamSynchronize(streamFasiPrincipali));

	//Rilascio della memoria device dell'array di input
	CHECK(cudaFree(array_in));

	//Distruzione dello stream delle fasi principali
	CHECK(cudaStreamDestroy(streamFasiPrincipali));

	return out;
}