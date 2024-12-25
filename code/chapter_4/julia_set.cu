// note: Complie command "nvcc -o res/julia_set julia_set.cu -lglut -lGL -lGLU"
#include "../common/book.h"
#include "../common/cpu_bitmap_save.h"

#define DIM 1000

struct cuComplex
{
    float r;
    float i;
    __device__ cuComplex(float a, float b): r(a), i(b) {}
    __device__ float magnitude2(void)
    {
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& a)
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

    __device__ cuComplex operator+(const cuComplex& a)
    {
        return cuComplex(r + a.r, i + a.i);
    }
};

__device__ int julia(int x, int y)
{
    const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (i = 0; i < 200; i++)
    {
        a = a * a + c;
        if (a.magnitude2() > 1000) return i; // 返回迭代次数
    }
    return 200; // 最大迭代次数
}

// 将迭代次数映射到颜色的函数
__device__ void color_map(int value, unsigned char &r, unsigned char &g, unsigned char &b)
{
    // 创建一个渐变色，基于 value 映射到 [0, 1]
    float t = (float)value / 200.0f;

    // 使用一种平滑的颜色映射算法
    r = (unsigned char)(9 * (1 - t) * t * t * t * 255);  // 渐变的红色分量
    g = (unsigned char)(15 * (1 - t) * (1 - t) * t * t * 255); // 渐变的绿色分量
    b = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255); // 渐变的蓝色分量
}

__global__ void kernel(unsigned char *ptr)
{
    // map from threadIdx/BlockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    // now calculate the value at that position
    int juliaValue = julia(x, y);

    // 根据 Julia 集值映射颜色
    unsigned char r, g, b;
    color_map(juliaValue, r, g, b);

    ptr[offset * 4 + 0] = r; // R channel
    ptr[offset * 4 + 1] = g; // G channel
    ptr[offset * 4 + 2] = b; // B channel 
    ptr[offset * 4 + 3] = 255; // Alpha channel (opacity)
}

int main(void)
{
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;

    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));

    dim3 grid(DIM, DIM);
    kernel<<<grid, 1>>>(dev_bitmap);

    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

    bitmap.display_and_exit();

    HANDLE_ERROR(cudaFree(dev_bitmap));
}
