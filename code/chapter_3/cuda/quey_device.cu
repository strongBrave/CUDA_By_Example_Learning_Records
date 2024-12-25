// #include "../book.h"

// int main(void) 
// {
//     cudaDeviceProp prop;

//     int count;
//     HANDLE_ERROR( cudaGetDeviceCount( &count));
//     for (int i = 0; i < count; i ++)
//     {
//         HANDLE_ERROR( cudaGetDeviceProperties(&prop, i));

//         // Do something with our device's properties.
//         printf("  --- General Information for device %d --- \n", i);
//         printf("Name: %s\n", prop.name);
//         printf("Compute capability: %d.%d\n", prop.major, prop.minor);
//         printf("Clock rate: %d\n", prop.clockRate);
//         printf("Device copy overlap: ");
//         if (prop.deviceOverlap)
//             printf("Enabled\n");
//         else printf("Disbaled\n");
//         printf("Kernel exection timeout: ");
//         if (prop.kernelExecTimeoutEnabled) printf("Enabled\n");
//         else printf("Disabled\n");

//         printf("  --- Memory Information for device %d --- \n", i);
//         printf("Total global mem: %zu\n", prop.totalGlobalMem);
//         printf("Total constant Mem: %zu\n", prop.totalConstMem);
//         printf("Max mem pitch: %zu\n", prop.memPitch);
//         printf("Texture Alignment: %zu\n", prop.textureAlignment);

//         printf("  --- MP Information for device %d ---\n", i);
//         printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
//         printf("Shared mem per mp: %zu\n", prop.sharedMemPerBlock);
//         printf("Registers per mp: %d\n", prop.regsPerBlock);
//         printf("Threads in warp: %d\n", prop.warpSize);
//         printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
//         printf("Max thread dimensions: (%d, %d, %d)\n", 
//                 prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
//         printf("Max grid dimensions: (%d, %d, %d)\n",
//                 prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
//         printf("\n");
//     }
//     return 0;
// }


# include "../book.h"

int main(void)
{
    cudaDeviceProp prop;
    int dev;

    HANDLE_ERROR(cudaGetDevice(&dev));
    printf("ID of current CUDA device: %d\n", dev);

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 3;
    HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
    printf("ID of CUDA device closest to revision 1,3: %d\n", dev);
    HANDLE_ERROR(cudaSetDevice(dev));

    return 0;
}