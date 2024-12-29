#include "../common/book.h"
#include <time.h>  // 引入 time.h 头文件

#define SIZE (100 * 1024 * 1024)

int main(void)
{
    unsigned char *buffer = (unsigned char*)big_random_block(SIZE);
    
    clock_t start = clock();  // 开始时间

    unsigned int histo[256];
    for (int i = 0; i < 256; i++)
        histo[i] = 0;

    for (int i = 0; i < SIZE; i++)
        histo[buffer[i]]++;

    long histoCount = 0;
    for (int i = 0; i < 256; i++)
    {
        histoCount += histo[i];
    }

    printf("Histogram Sum: %ld\n", histoCount);

    clock_t end = clock();  // 结束时间
    double elapsed_time = double(end - start) / CLOCKS_PER_SEC;  // 计算秒数
    printf("Elapsed Time: %.6f seconds\n", elapsed_time);
    
    free(buffer);

    return 0;
}
