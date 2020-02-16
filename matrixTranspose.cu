#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void intialData(float *input, int n)
{
    for (int i=0;i<n;i++)
    {
        input[i] = (float)rand()/10.f;
    }
    return;
}

void transposeHost(float *out, const float *in , const int nx, const int ny)
{
    for (int j = 0;j<ny;j++)
    {
        for (int i = 0;i<nx;i++)
        {
            out[i*ny+j] = in[j*nx+i];
        }
    }
    return ;
}
__global__ void warmup(float *in, float *out, const int nx, const int ny)
{
    unsigned int i = threadIdx.x+blockDim.x*blockIdx.x;
    unsigned int j = threadIdx.y+blockDim.y*blockIdx.y;

    if (i<nx && j<ny)
    {
        out[j*nx+i] = in[j*nx+i];
    }
}
__global__ void copyGlobalRow(float *in, float *out, const int nx, const int ny)
{
    unsigned int i = threadIdx.x+blockDim.x*blockIdx.x;
    unsigned int j = threadIdx.y+blockDim.y*blockIdx.y;

    if (i<nx && j<ny)
    {
        out[j*nx+i] = in[j*nx+i];
    }
}

__global__ void copyGlobalCol(float *out, float *in, const int nx, const int ny)
{
    unsigned int i = threadIdx.x+blockDim.x*blockIdx.x;
    unsigned int j = threadIdx.y+blockDim.y*blockIdx.y;

    if (i<nx && j<ny)
    {
        out[i*ny+j] = in[i*ny+j];
    }
}

__global__ void transposeGlobalRow(float *in, float *out, const int nx, const int ny)
{
    unsigned int i = threadIdx.x+blockDim.x*blockIdx.x;
    unsigned int j = threadIdx.y+blockDim.y*blockIdx.y;

    if (i<nx && j<ny)
    {
        out[i*ny+j] = in[j*nx+i];
    }
}

__global__ void transposeGlobalCol(float *in, float *out, const int nx, const int ny)
{
    unsigned int i = threadIdx.x+blockDim.x*blockIdx.x;
    unsigned int j = threadIdx.y+blockDim.y*blockIdx.y;

    if (i<nx && j<ny)
    {
        out[j*nx+i] = in[i*ny+j];
    }
}

void verify(const float *host, const float *gpu , const int nx, const int ny)
{
    for (int j = 0;j<ny;j++)
    {
        for (int i = 0;i<nx;i++)
        {
            if (host[j*nx + i] != gpu[j*nx + i]){
                printf("Error\n");
                break;
            }
        }
    }
}

int main(){
    int nx = 1 <<11;
    int ny = 1 <<11;
    size_t nBytes = (nx*ny)*sizeof(float);

    int nRep = 1;
    float ms = 0;
    float bw = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block (16, 16);
    dim3 grid  ((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    float *a_h = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef = (float *)malloc(nBytes);

    printf("Matrix %d nx %d ny\n", nx,ny);

    intialData(a_h, nx*ny);
    transposeHost(hostRef, a_h, nx, ny);

    //allocate device memory
    float *a_d, *c_d;
    cudaMalloc((float **)&a_d,nBytes);
    cudaMalloc((float **)&c_d,nBytes);

    //copy data
    cudaMemcpy(a_d,a_h, nBytes, cudaMemcpyHostToDevice);
    warmup<<<grid,block>>>(a_d,c_d,nx,ny);

    //copyGlobalRow
    cudaEventRecord(start,0);
    for (int k=0;k<nRep;k++)
    {
        copyGlobalRow<<<grid,block>>>(a_d,c_d,nx,ny);
    }
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    bw = 2 * nx * ny * sizeof(float)/ms/1e6;
    printf("copyGlobalRow: %f ms, effective bandwidth %f GB/s\n",ms/((float)nRep),bw/((float)nRep));

    //copyGlobalCol
    cudaEventRecord(start);
    for (int k=0;k<nRep;k++)
    {
        copyGlobalCol<<<grid,block>>>(a_d,c_d,nx,ny);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    bw = 2 * nx * ny * sizeof(float)/ms/1e6;
    printf("copyGlobalCol: %f ms, effective bandwidth %f GB/s\n",ms/((float)nRep),bw/((float)nRep));

    //transposeGlobalRow
    cudaEventRecord(start,0);
    for (int k=0;k<nRep;k++)
    {
        transposeGlobalRow<<<grid,block>>>(a_d,c_d,nx,ny);
    }
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    bw = 2 * nx * ny * sizeof(float)/ms*1e-6;
    printf("transposeGlobalRow: %f ms, effective bandwidth %f GB/s\n",ms/((float)nRep),bw/((float)nRep));
    cudaMemcpy(gpuRef,c_d, nBytes, cudaMemcpyDeviceToHost);
    verify(hostRef,gpuRef,nx,ny);

    //transposeGlobalCol
    cudaEventRecord(start,0);
    for (int k=0;k<nRep;k++)
    {
        transposeGlobalCol<<<grid,block>>>(a_d,c_d,nx,ny);
    }
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    bw = 2 * nx * ny * sizeof(float)/ms*1e-6;
    printf("transposeGlobalCol: %f ms, effective bandwidth %f GB/s\n",ms/((float)nRep),bw/((float)nRep));
    cudaMemcpy(gpuRef,c_d, nBytes, cudaMemcpyDeviceToHost);
    verify(hostRef,gpuRef,nx,ny);


    cudaFree(a_d);
    cudaFree(c_d);
    free(hostRef);
    free(gpuRef);
    free(a_h);
    cudaDeviceReset();

    return 0;
}
