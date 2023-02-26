#include <stdlib.h>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define COLOR_CHANNELS 1

#define GRAYLEVELS 256
#define BLOCK_SIZE 32 // NOTE: BLOCK_SIZE * BLOCK_SIZE should always be at least as big as GRAY_LEVELS
#define DESIRED_NCHANNELS 1

typedef char i8;
typedef short i16;
typedef int i32;
typedef long long i64;

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

__device__ unsigned char scale(unsigned long cdf, unsigned long cdfmin, unsigned long imageSize) {

    float scale;
    scale = (float) (cdf - cdfmin) / (float) (imageSize - cdfmin);
    scale = round(scale * (float) (GRAYLEVELS - 1));
    return (int) scale;
}

__global__ void CalculateHistogram(const unsigned char *image, int width, int heigth, u64 *hist) {
    __shared__ u32 sharedHist[GRAYLEVELS];
    u32 x = blockIdx.x * blockDim.x + threadIdx.x;
    u32 y = blockIdx.y * blockDim.y + threadIdx.y;
    u32 tID = threadIdx.y * blockDim.x + threadIdx.x;

    //Clear histogram
    if (tID < GRAYLEVELS) sharedHist[tID] = 0;
    __syncthreads();

    //Calculate histogram

    if (x < width && y < heigth) {
        atomicAdd(sharedHist + image[y * width + x], 1);
    }


    // Write to global memory
    __syncthreads();
    //if (tID < GRAYLEVELS) hist[tID] = sharedHist[tID];
    if (tID < GRAYLEVELS) atomicAdd(hist + tID, sharedHist[tID]);
}

// Interleaved addressing
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
__global__ void CalculateCDF(const u64 *hist, u64 *cdf) {
    __shared__ u64 sharedCDF[GRAYLEVELS + 1];

    // Init CDF
    u32 tID = threadIdx.x;
    sharedCDF[tID] = hist[tID];

    // Up sweep
    u32 stride = 1;
    while (stride < GRAYLEVELS) {
        __syncthreads();
        // if (tID % (stride << 1) == 0) slow...
        u32 idx = 2 * stride * tID;
        if (idx < GRAYLEVELS) {
            u32 lIdx = (GRAYLEVELS - idx - 1) - stride;
            u32 rIdx = (GRAYLEVELS - idx - 1);

            sharedCDF[rIdx] += sharedCDF[lIdx];
        }
        stride <<= 1;
    }


    if (tID == 0) {
        sharedCDF[GRAYLEVELS] = sharedCDF[GRAYLEVELS - 1];
        sharedCDF[GRAYLEVELS - 1] = 0;
    }

    // Down sweep
    stride >>= 1;
    while (stride) {
        __syncthreads();
        //if (tID % (stride << 1) == 0) {
        u32 idx = 2 * stride * tID;
        if (idx < GRAYLEVELS) {
            u32 lIdx = (GRAYLEVELS - idx - 1) - stride;
            u32 rIdx = (GRAYLEVELS - idx - 1);

            u64 tmp = sharedCDF[lIdx];
            sharedCDF[lIdx] = sharedCDF[rIdx];
            sharedCDF[rIdx] += tmp;
        }
        stride >>= 1;
    }
    __syncthreads();

    // Write to global memory
    cdf[tID] = sharedCDF[tID + 1];
}

__global__ void FindMin(const u64 *hist, u64 *min) {
    __shared__ u32 minShared;
    if (threadIdx.x == 0) minShared = GRAYLEVELS - 1;
    __syncthreads();

    if (hist[threadIdx.x]) atomicMin(&minShared, threadIdx.x);
    __syncthreads();

    if (threadIdx.x == 0) *min = hist[minShared];
}

__global__ void Equalize(unsigned char *inImage, unsigned char *outImage, u64 width, u64 height, u64 imageSize, u64 *min, u64 *cdf) {

    u64 cdfmin = *min;

    u32 x = blockIdx.x * blockDim.x + threadIdx.x;
    u32 y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        outImage[y * width + x] = scale(cdf[inImage[y * width + x]], cdfmin, imageSize);
    }
}

int main(int argc, char **argv) {
    if (argc <= 2) {
        fprintf(stderr, "Error. Usage: %s inputImage outputImage\n", argv[0]);
        return 1;
    }

    // Read image from file
    int width, height, cpp;
    // read only DESIRED_NCHANNELS channels from the input image:
    unsigned char *imageIn = stbi_load(argv[1], &width, &height, &cpp, DESIRED_NCHANNELS);
    if (imageIn == NULL) {
        printf("Error in loading the image\n");
        return 1;
    }
    printf("Loaded image W= %d, H = %d, actual cpp = %d \n", width, height, cpp);


    //Allocate memory for raw output image data, histogram, and CDF
    unsigned char *imageOut = (unsigned char *) malloc(height * width * sizeof(unsigned long));
    u64 imageSize = width * height * COLOR_CHANNELS;

    u64 *d_histogram;
    cudaMalloc(&d_histogram, GRAYLEVELS * sizeof(*d_histogram));
    cudaMemset(d_histogram, 0, GRAYLEVELS * sizeof(*d_histogram));

    u64 *d_CDF;
    cudaMalloc(&d_CDF, GRAYLEVELS * sizeof(*d_CDF));
    cudaMemset(d_CDF, 0, GRAYLEVELS * sizeof(*d_CDF));

    unsigned char *d_image;
    //unsigned char *d_out;
    cudaMalloc(&d_image, imageSize);
    //cudaMalloc(&d_out, height * width * cpp);

    u64 *d_min;
    cudaMalloc(&d_min, sizeof(*d_min));

    cudaDeviceSynchronize();
    clock_t startTime = clock();
    cudaMemcpy(d_image, imageIn, imageSize, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    // For algorithm to correctly work for very small images grid size must be at least 16x16
    if (grid.x < 16) grid.x = 16;
    if (grid.y < 16) grid.y = 16;
    //dim3 grid(256, 256);
    // Histogram equalization steps:
    // 1. Create the histogram for the input grayscale image.
    CalculateHistogram<<<grid, block>>>(d_image, width, height, d_histogram);

    // 2. Calculate the cumulative distribution histogram.
    CalculateCDF<<<1, GRAYLEVELS>>>(d_histogram, d_CDF);

    // 3. Calculate the new gray-level values through the general histogram equalization formula
    //    and assign new pixel values
    FindMin<<<1, GRAYLEVELS>>>(d_histogram, d_min);
    Equalize<<<grid, block>>>(d_image, d_image, width, height, width * height, d_min, d_CDF);

    cudaMemcpy(imageOut, d_image, imageSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    clock_t endTime = clock();

    cudaFree(d_histogram);
    cudaFree(d_CDF);
    cudaFree(d_min);
    cudaFree(imageIn);
    if (imageIn != imageOut) cudaFree(imageOut);

    cudaDeviceSynchronize();

    float diff = ((float) (endTime - startTime) / 1.0E6f) * 1000;
    printf("[totalTime]: %f\n", diff);

    cudaDeviceReset();

    // write output image:
    size_t len = strlen(argv[2]);
    char *extension = argv[2] + len - 3;
    if (strcmp("png", extension) == 0)
        stbi_write_png(argv[2], width, height, DESIRED_NCHANNELS, imageOut, width * DESIRED_NCHANNELS);
    else if (strcmp("jpg", extension) == 0)
        stbi_write_jpg(argv[2], width, height, DESIRED_NCHANNELS, imageOut, 100);

    //Free memory
    free(imageIn);
    free(imageOut);

    return 0;
}
