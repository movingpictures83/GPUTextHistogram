#include "GPUTextHistogramPlugin.h"

void histogram(const char *input, unsigned int *bins,
               unsigned int num_elements, unsigned int num_bins) {

  // zero out bins
  CUDA_CHECK(cudaMemset(bins, 0, num_bins * sizeof(unsigned int)));
  // Launch histogram kernel on the bins
  {
    dim3 blockDim(256), gridDim(30);
    histogram_kernel<<<gridDim, blockDim,
                       num_bins * sizeof(unsigned int)>>>(
        input, bins, num_elements, num_bins);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}


void GPUTextHistogramPlugin::input(std::string infile) {
  readParameterFile(infile);
}

void GPUTextHistogramPlugin::run() {}

void GPUTextHistogramPlugin::output(std::string outfile) {

  int inputLength;
  char *hostInput;
  unsigned int *hostBins;
  char *deviceInput;
  unsigned int *deviceBins;

  inputLength = atoi(myParameters["N"].c_str());
  hostInput = (char*) malloc (inputLength*sizeof(char));
   std::ifstream myinput((std::string(PluginManager::prefix())+myParameters["data"]).c_str(), std::ios::in);
 int i;
 for (i = 0; i < inputLength; ++i) {
        int k;
        myinput >> k;
        hostInput[i] = k;
 }



  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));

  //@@ Allocate GPU memory here
  cudaMalloc((void **)&deviceInput, inputLength);
  cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(unsigned int));
  cudaDeviceSynchronize();
  cudaMemcpy(deviceInput, hostInput, inputLength,
                        cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  // @@ Insert code here
  histogram(deviceInput, deviceBins, inputLength, NUM_BINS);
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins,
                        NUM_BINS * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

          std::ofstream outsfile(outfile.c_str(), std::ios::out);
        int j;
        for (i = 0; i < NUM_BINS; ++i){
                outsfile << hostBins[i];//std::setprecision(0) << a[i*N+j];
                outsfile << "\n";
        }

  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);


  free(hostBins);
  free(hostInput);
}

PluginProxy<GPUTextHistogramPlugin> GPUTextHistogramPluginProxy = PluginProxy<GPUTextHistogramPlugin>("GPUTextHistogram", PluginManager::getInstance());

