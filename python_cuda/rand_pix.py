import numpy as np
import pycuda.autoinit as _ 
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import cv2
import pycuda.gpuarray as gpuarray
from pycuda.curandom import rand as curand
import os
os.environ['PYCUDA_DEFAULT_NVCC_FLAGS'] = '-w' # suppress all warning messages

# Load the image using OpenCV
img = cv2.imread('./out/merged.png', cv2.IMREAD_COLOR)

# Allocate memory on the device and copy the image to the device
img_gpu = gpuarray.to_gpu(np.float32(img))

# Generate random numbers on the GPU
rand_gpu = curand((img.shape[0], img.shape[1]))

# Define the CUDA kernel
mod = SourceModule("""
    __global__ void randomize_pixels(float *img, float *rand, int width, int height) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int idy = threadIdx.y + blockIdx.y * blockDim.y;

        if (idx < width && idy < height) {
            // Generate a random destination position
            int dst_x = rand[idy * width + idx] * width;
            int dst_y = rand[idy * width + idx] * height;

            // Swap the pixel at (idx, idy) with the pixel at (dst_x, dst_y)
            float3 temp = ((float3*)img)[idy * width + idx];
            ((float3*)img)[idy * width + idx] = ((float3*)img)[dst_y * width + dst_x];
            ((float3*)img)[dst_y * width + dst_x] = temp;
        }
    }
""")

# Get the kernel function
func = mod.get_function("randomize_pixels")

# Create CUDA events for timing
start = cuda.Event()
end = cuda.Event()

start.record()

# Call the kernel function
func(img_gpu, rand_gpu, np.int32(img.shape[1]), np.int32(img.shape[0]), block=(32, 32, 1), grid=(int((img.shape[1] + 31) / 32), int((img.shape[0] + 31) / 32), 1))

end.record()

# Wait for the end event to complete
end.synchronize()

print(f"Elapsed time: {start.time_till(end)} ms")

# Copy the result back to the host
new_img = np.uint8(img_gpu.get())

# Save the new image
cv2.imwrite('./out/randomised.jpg', new_img)
