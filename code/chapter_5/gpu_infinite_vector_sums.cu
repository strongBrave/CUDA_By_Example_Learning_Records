#include "../common/book.h"

#define N (1024 * 1024 * 512)

__global__ void add(int *a, int *b, int *c)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x; // handle the data at this index
    while (tid < N)
    {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main(void)
{
    // int a[N], b[N], c[N];

    // Declare arrays on the host. Instead of using stack memory (which is limited),
    // allocate memory for arrays 'a', 'b', and 'c' on the heap to handle large sizes.
    int *a = (int *)malloc(N * sizeof(int));
    int *b = (int *)malloc(N * sizeof(int));
    int *c = (int *)malloc(N * sizeof(int));

    // Check if memory allocation on the host was successful
    if (!a || !b || !c) 
    {
        printf("Failed to allocate memory on host\n");
        return -1;
    }

    int *dev_a, *dev_b, *dev_c;

    // allocate the memory on the GPU
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

    // fill the arrays "a" and "b" on the CPU
    for (int i = 0; i < N; i ++ )
    {
        a[i] = -i;
        b[i] = i * i;
    }

    // copy the arrays "a" and "b" to the GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

    add<<<1024, 1024>>>(dev_a, dev_b, dev_c);

    // copy the array "c" back from GPU to the CPU
    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    // vecrify that the GPU did the work we requested
    bool success = true;
    for (int i = 0; i < N; i ++ )
    {
        if ((a[i] + b[i] != c[i]))
        {
            printf("Error: %d + %d != %d\n", a[i], b[i], c[i]);
            success = false;
        }
    }
    if (success) printf("We did it\n");

    // Free the host memory allocated on the heap
    free(a);
    free(b);
    free(c);

    // free the memory allocated on the GPU
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;

}