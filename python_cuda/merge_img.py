import numpy as np
import pycuda.autoinit as _
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import cv2
import os
os.environ['PYCUDA_DEFAULT_NVCC_FLAGS'] = '-w' # suppress all warning messages

# Load the images using OpenCV
img1 = cv2.imread('./in/Image1.png', cv2.IMREAD_COLOR)
img2 = cv2.imread('./in/Image2.png', cv2.IMREAD_COLOR)

# Print the total number of pixels in each image
print(f"Pixels in image 1: {img1.size}")
print(f"Pixels in image 2: {img2.size}")

# Create a new image that can hold both images side by side
new_img = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)

# Allocate memory on the device and copy the images to the device
img1_gpu = cuda.mem_alloc(img1.nbytes)
cuda.memcpy_htod(img1_gpu, img1)
img2_gpu = cuda.mem_alloc(img2.nbytes)
cuda.memcpy_htod(img2_gpu, img2)
new_img_gpu = cuda.mem_alloc(new_img.nbytes)

# Define the CUDA kernel
mod = SourceModule("""
    __global__ void merge_images(unsigned char *img1, unsigned char *img2, unsigned char *new_img, int width1, int width2, int height)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int idy = threadIdx.y + blockIdx.y * blockDim.y;

        if (idx < width1 && idy < height) {
            new_img[idy * (width1 + width2) * 3 + idx * 3] = img1[idy * width1 * 3 + idx * 3];
            new_img[idy * (width1 + width2) * 3 + idx * 3 + 1] = img1[idy * width1 * 3 + idx * 3 + 1];
            new_img[idy * (width1 + width2) * 3 + idx * 3 + 2] = img1[idy * width1 * 3 + idx * 3 + 2];
        }

        if (idx < width2 && idy < height) {
            new_img[idy * (width1 + width2) * 3 + (idx + width1) * 3] = img2[idy * width2 * 3 + idx * 3];
            new_img[idy * (width1 + width2) * 3 + (idx + width1) * 3 + 1] = img2[idy * width2 * 3 + idx * 3 + 1];
            new_img[idy * (width1 + width2) * 3 + (idx + width1) * 3 + 2] = img2[idy * width2 * 3 + idx * 3 + 2];
        }
    }
""")

# Get the kernel function
func = mod.get_function("merge_images")

# Call the kernel function
func(img1_gpu, img2_gpu, new_img_gpu, np.int32(img1.shape[1]), np.int32(img2.shape[1]), np.int32(max(img1.shape[0], img2.shape[0])), block=(32, 32, 1), grid=(int((new_img.shape[1] + 31) / 32), int((new_img.shape[0] + 31) / 32), 1))

# Copy the result back to the host
cuda.memcpy_dtoh(new_img, new_img_gpu)

# Print the total number of pixels in the merged image
print(f"Pixels in merged image: {new_img.size}")

# Save the merged image
cv2.imwrite('./out/merged.png', new_img)
