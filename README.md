# Image Convolution Based on AVX2 & FMA
Image Convolution Based on AVX2 &amp; FMA Assembly.  
AVX2 and FMA3 are SIMD instructions.They allow CPU to process 8 32-bit number or 4 64-bit number at same time with 256-bit YMM registers.
## Introduction
These are three demos use different frameworks.They are placed in root folders .  

    MPI
    pthread
    OpenMP

pthread and OpenMP is shared-memory architecture and maybe faster for OS to schedule tasks in single computer.  
Meanwhile, MPI uses distribute memory architecture, which means it can run in multple computers that connected by networks.
## Envirement Requirement
### Hardware
Obviously you need a x86-64(AMD64) CPU. Besides, these two extension are required:

    AVX2
    FMA3

### Software
#### MPI
You can try MSMPI on Windows or MPICH on Linux.
#### pthread
You should run the demo under OS which supports POSIX thread standard. You can try Linux or macOS.
#### OpenMP
You just need a compiler that support OpenMP.

## Compile Command(Linux / macOS)
### MPI
```
mpicc -g -Wall -o mpiConv mpi.c 
```
### pthread
```
gcc -g -Wall -o pthreadConv pthread.c -lpthread
```
### OpenMP
```
gcc -Wall -o OpenMPConv openmp.c -fopenmp
```
## Run demos
### Step 1.  
Rename image as 1.bmp, and put it in the same directory as executable file. The image cannot be compressed, which means, if the image is w * h * 3, its size should be (54 + w * h * 3) byte.  

### Step 2.
Run program with passing thread(pthread OpenMP) / process(MPI) number. Here is a example that use 4 threads:  
```
./pthreadConv 4
```
### Step 3.
Program will generate a BMP file in the directory. It would be named as mpi.bmp / pthread.bmp / openmp.bmp.
## Notice
The filter is a 5 * 5 Gaussian filter.You can find and change it directly in source code.  
The padding mode is SAME, which means when 1.bmp is a w * h * 3 image, the result would be w * h * 3 image. 1.bmp will be round by 2 black pixel when convoluting.  
mpi.c may not work properly under macOS Big Sur.(Segment fault)
## Performance
These data was measured under following environment:  
CPU: Intel Core i7 9700  
RAM: 16 GB DDR4-2666  
OS: Ubuntu 20.01 LTS
Compiler: MPICC / GCC 10.2
| Arch | 2 Thread / s | 4 Thread / s | 8 Thread / s |
| ---- | -------- | -------- | -------- |
| MPI(Win10 MSMPI) | 0.107567 | 0.088506 | 0.080368 |
| pthread | 0.131346 | 0.081878 | 0.074302 |
| OpenMP | 0.119463 | 0.086379 | 0.064605 |

\* Convolute a 4096 * 2304 * 3 bmp file   
\* the whole program running time, including loading image from disk and storing image to disk
## Acknowledgement
My Course teacher: Profeccor Zhang  
Course TA: Mr. Zhuang  
Me: Mr. Han  
And my PC.
## Postscripst
You can contact me by email hml0814@163.com   
In fact, the code in this repository is the third of the four assignments in the course "principle and implementation of parallel programming" in the spring semester of 2020-2021 which implemented by myself. Because it took a lot of time to optimize the code, it's a pity not to share it.(Doge)  
If you want to use the code here for academic using(Include but not limited to papers, presentation, , course design, course project), please declare the reference in your paper and to your mentor.   
The use in commercial work is granted if you inform me by email in advance. You DO NOT need to wait for my permission if you send the email to me.  
Pull requestion is welcomed if you have performance optimization on this.    