#include "quicksort.h"

void swap(float* a, float* b)
{
	float t = *a;
	*a = *b;
	*b = t;
}

int partition(float arr[], int l, int h) {
	float x = arr[h];
	int i = (l - 1);

	for (int j = l; j <= h - 1; j++) {
		if (arr[j] <= x) {
			i++;
			swap(&arr[i], &arr[j]);
		}
	}
	swap(&arr[i + 1], &arr[h]);
	return (i + 1);
}

void sort(float arr[], int l, int h) {
	const unsigned int N = h - l + 1;
	int*stack = (int*)malloc(sizeof(int) * N);
	int top = -1;
	stack[++top] = l;
	stack[++top] = h;

	while (top >= 0) {
		h = stack[top--];
		l = stack[top--];

		int p = partition(arr, l, h);

		if (p - 1 > l) {
			stack[++top] = l;
			stack[++top] = p - 1;
		}

		if (p + 1 < h) {
			stack[++top] = p + 1;
			stack[++top] = h;
		}
	}
}

void quickSort(float in[], const int N) {
	sort(in, 0, N - 1);
}