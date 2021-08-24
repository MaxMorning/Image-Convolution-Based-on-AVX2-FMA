#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "omp.h"


// 文件信息头结构体
typedef struct
{
    unsigned int   bfSize;        // 文件大小 以字节为单位(2-5字节)
    unsigned short bfReserved1;   // 保留，必须设置为0 (6-7字节)
    unsigned short bfReserved2;   // 保留，必须设置为0 (8-9字节)
    unsigned int   bfOffBits;     // 从文件头到像素数据的偏移  (10-13字节)
} BITMAPFILEHEADER;

//图像信息头结构体
typedef struct
{
    unsigned int    biSize;          // 此结构体的大小 (14-17字节)
    int             biWidth;         // 图像的宽  (18-21字节)
    int             biHeight;        // 图像的高  (22-25字节)
    unsigned short  biPlanes;        // 表示bmp图片的平面属，显然显示器只有一个平面，所以恒等于1 (26-27字节)
    unsigned short  biBitCount;      // 一像素所占的位数，一般为24   (28-29字节)
    unsigned int    biCompression;   // 说明图象数据压缩的类型，0为不压缩。 (30-33字节)
    unsigned int    biSizeImage;     // 像素数据所占大小, 这个值应该等于上面文件头结构中bfSize-bfOffBits (34-37字节)
    int             biXPelsPerMeter; // 说明水平分辨率，用象素/米表示。一般为0 (38-41字节)
    int             biYPelsPerMeter; // 说明垂直分辨率，用象素/米表示。一般为0 (42-45字节)
    unsigned int    biClrUsed;       // 说明位图实际使用的彩色表中的颜色索引数（设为0的话，则说明使用所有调色板项）。 (46-49字节)
    unsigned int    biClrImportant;  // 说明对图象显示有重要影响的颜色索引的数目，如果是0，表示都重要。(50-53字节)
} BITMAPINFOHEADER;


long thread_cnt = 4;

float kernel[5][5][3] = {
        0.01441881, 0.01441881, 0.01441881, 0.02808402, 0.02808402, 0.02808402, 0.0350727, 0.0350727, 0.0350727, 0.02808402, 0.02808402, 0.02808402, 0.01441881, 0.01441881, 0.01441881,
        0.02808402, 0.02808402, 0.02808402, 0.0547002, 0.0547002, 0.0547002, 0.06831229, 0.06831229, 0.06831229, 0.0547002, 0.0547002, 0.0547002, 0.02808402, 0.02808402, 0.02808402,
        0.0350727, 0.0350727, 0.0350727, 0.06831229, 0.06831229, 0.06831229, 0.08531173, 0.08531173, 0.08531173, 0.06831229, 0.06831229, 0.06831229, 0.0350727, 0.0350727, 0.0350727,
        0.02808402, 0.02808402, 0.02808402, 0.0547002, 0.0547002, 0.0547002, 0.06831229, 0.06831229, 0.06831229, 0.0547002, 0.0547002, 0.0547002, 0.02808402, 0.02808402, 0.02808402,
        0.01441881, 0.01441881, 0.01441881, 0.02808402, 0.02808402, 0.02808402, 0.0350727, 0.0350727, 0.0350727, 0.02808402, 0.02808402, 0.02808402, 0.01441881, 0.01441881, 0.01441881};
;
int width, height;

unsigned char* ori_byte_img = NULL;
float* red_result_array = NULL;
float* green_result_array = NULL;
float* blue_result_array = NULL;

float* red_float_array = NULL;
float* green_float_array = NULL;
float* blue_float_array = NULL;


int* thread_row_offset = NULL;
int* ori_row_offset = NULL;

int write_mask_array[8] = {0};
int target_col;

// boarder process
int rest_pix;
int rest_bias;

unsigned char* target_write_array = NULL;

int load_mask[8] = {-1, -1, -1, -1, -1, -1, 0, 0};

unsigned char red_idx[32] = {2, 5, 8, 11, 14, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 1, 4, 7, 255, 255, 255, 255, 255, 255, 255, 255};

unsigned char green_idx[32] = {1, 4, 7, 10, 13, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 3, 6, 255, 255, 255, 255, 255, 255, 255, 255};

unsigned char blue_idx[32] = {0, 3, 6, 9, 12, 15, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 2, 5, 255, 255, 255, 255, 255, 255, 255, 255};

void load_image(FILE* file_ptr)
{
    double start, end;
    start = omp_get_wtime();

    // skip headers
    BITMAPFILEHEADER fileHeader;
    BITMAPINFOHEADER infoHeader;

    unsigned short  fileType;
    fread(&fileType,1,sizeof (unsigned short), file_ptr);
    if (fileType == 0x4d42)
    {
        fread(&fileHeader, sizeof(BITMAPFILEHEADER), 1, file_ptr);
        fread(&infoHeader, sizeof(BITMAPINFOHEADER), 1, file_ptr);
    }

    width = infoHeader.biWidth;
    height = infoHeader.biHeight;

    int image_size = height * width * 3;
    ori_byte_img = (unsigned char*)malloc(image_size);

    fread(ori_byte_img, 1, image_size, file_ptr);

    end = omp_get_wtime();
    printf("LOAD Time cost : %f s\n", end - start);
}

void broadcast_convert_asm(int index)
{
//    double start, end;
//    start = omp_get_wtime();

    int float_offset = index * (width + 4);
    red_float_array[float_offset] = 0;
    green_float_array[float_offset] = 0;
    blue_float_array[float_offset] = 0;

    ++float_offset;

    red_float_array[float_offset] = 0;
    green_float_array[float_offset] = 0;
    blue_float_array[float_offset] = 0;

    ++float_offset;
    int float_target = float_offset + width;
    int ori_offset = (index - 2) * width * 3;
//    int float_offset = (ori_row_offset[index] + 2) * width;
//    int ori_offset = float_offset * 3;
//    int float_target = (ori_row_offset[index + 1] + 2) * width;

    __asm__(
    // ymm0 as load mask
    // ymm1 as red idx
    // ymm2 as green idx
    // ymm3 as blue idx
    "vmovdqu %[load_mask], %%ymm0\t\n"
    "vmovdqu %[red_idx], %%ymm1\t\n"
    "vmovdqu %[green_idx], %%ymm2\t\n"
    "vmovdqu %[blue_idx], %%ymm3\t\n"
    "BROAD_CONV_COMP:\t\n"
    "cmpq %[target_8], %[float_offset]\t\n"
    "jge BROAD_CONV_END\t\n"

        "vpmaskmovd (%[ori_byte_img], %[ori_offset]), %%ymm0, %%ymm4\t\n"

        "vpshufb %%ymm1, %%ymm4, %%ymm5\t\n"
        "vperm2i128 $241, %%ymm5, %%ymm5, %%ymm6\t\n"
        "vpor %%ymm6, %%ymm5, %%ymm5\t\n"
        "vpmovzxbd %%xmm5, %%ymm5\t\n"
        "vcvtdq2ps %%ymm5, %%ymm5\t\n"
        "vmovups %%ymm5, (%[red_float_array], %[float_offset], 4)\t\n"

        "vpshufb %%ymm2, %%ymm4, %%ymm5\t\n"
        "vperm2i128 $241, %%ymm5, %%ymm5, %%ymm6\t\n"
        "vpor %%ymm6, %%ymm5, %%ymm5\t\n"
        "vpmovzxbd %%xmm5, %%ymm5\t\n"
        "vcvtdq2ps %%ymm5, %%ymm5\t\n"
        "vmovups %%ymm5, (%[green_float_array], %[float_offset], 4)\t\n"

        "vpshufb %%ymm3, %%ymm4, %%ymm5\t\n"
        "vperm2i128 $241, %%ymm5, %%ymm5, %%ymm6\t\n"
        "vpor %%ymm6, %%ymm5, %%ymm5\t\n"
        "vpmovzxbd %%xmm5, %%ymm5\t\n"
        "vcvtdq2ps %%ymm5, %%ymm5\t\n"
        "vmovups %%ymm5, (%[blue_float_array], %[float_offset], 4)\t\n"

    "BROAD_CONV_INC:\t\n"
    "addq $8, %[float_offset]\t\n"
    "addq $24, %[ori_offset]\t\n"
    "jmp BROAD_CONV_COMP\t\n"
    "BROAD_CONV_END:\t\n"
    :
    :   [red_idx]"m"(red_idx),
    [green_idx]"m"(green_idx),
    [blue_idx]"m"(blue_idx),
    [load_mask]"m"(load_mask),
    [float_offset]"r"((long long)float_offset),
    [ori_offset]"r"((long long)ori_offset),
    [red_float_array]"r"(red_float_array),
    [green_float_array]"r"(green_float_array),
    [blue_float_array]"r"(blue_float_array),
    [target_8]"r"((long long)(float_target - 8)),
    [ori_byte_img]"r"(ori_byte_img)
    : "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6"
    );

    float_offset = float_target - width % 8 - 8;
//    ori_offset = (index * width + width - width % 8 - 8) * 3;
    ori_offset = ((index - 2 + 1) * width - width % 8 - 8) * 3;
//    float_offset += 2;

    for (; float_offset < float_target; ++float_offset) {
        blue_float_array[float_offset] = (float)ori_byte_img[ori_offset];
        ++ori_offset;

        green_float_array[float_offset] = (float)ori_byte_img[ori_offset];
        ++ori_offset;

        red_float_array[float_offset] = (float)ori_byte_img[ori_offset];
        ++ori_offset;
    }

    red_float_array[float_offset] = 0;
    green_float_array[float_offset] = 0;
    blue_float_array[float_offset] = 0;

    ++float_offset;

    red_float_array[float_offset] = 0;
    green_float_array[float_offset] = 0;
    blue_float_array[float_offset] = 0;

//    end = omp_get_wtime();
//    printf("BROADCAST & CONVERT AVX @ %lld Time cost : %f s\n", index, end - start);
}
void conv_all_fma_asm(int row_idx);
inline void conv_all_fma_asm(int row_idx)
{
//    double start, end;
//    start = omp_get_wtime();

    __asm__(
    // r10 is the base addr of filter
    // r9 as r offset
    // r8 as c
    // r12 as green_float_array
    // r15 as blue_float_array
    "movl %[row_index], %%r9d\t\n"

    // ymm5 as writemask
    "vmovdqu %[write_mask], %%ymm5\t\n"

        "COL_INIT:\t\n"
        "xorq %%r8, %%r8\t\n"

        "COL_CMP:\t\n"
        "cmpq %[target_col_8], %%r8\t\n"
        "jg PROC_B\t\n"

            // ymm0 as redConvResult
            "vxorps %%ymm0, %%ymm0, %%ymm0\t\n"
            // ymm1 as greenConvResult
            "vxorps %%ymm1, %%ymm1, %%ymm1\t\n"
            // ymm2 as blueConvResult
            "vxorps %%ymm2, %%ymm2, %%ymm2\t\n"
            // r11 / rax as bias
            "movq %[ori_width], %%rax\t\n"
            "imulq %%r9, %%rax\t\n"
            "addq %%r8, %%rax\t\n"
            //                    "imul rax, 4\t\n"
            // rax == bias now
            //                    "movq %%rax, %%r11\t\n"

                // reuse rcx as ker_r
                // r13 as ker_c
                // r14 as ker_bias
                "xorq %%r14, %%r14\t\n"
                "xorq %%rcx, %%rcx\t\n"
                "KER_R_CMP:\t\n"
                "cmpq $5, %%rcx\t\n"
                "jge KER_END\t\n"

                "xorq %%r13, %%r13\t\n"
                "KER_C_CMP:\t\n"
                "cmpq $5, %%r13\t\n"
                "jge INC_KER_R\t\n"

                    "vbroadcastss (%[filter_addr], %%r14), %%ymm3\t\n"
                    "vmovups (%[red_float_array], %%rax, 4), %%ymm4\t\n"
                    "vfmadd231ps %%ymm3, %%ymm4, %%ymm0\t\n"
                    "addq $4, %%r14\t\n"

                    "vbroadcastss (%[filter_addr], %%r14), %%ymm3\t\n"
                    "vmovups (%[green_float_array], %%rax,4), %%ymm4\t\n"
                    "vfmadd231ps %%ymm3, %%ymm4, %%ymm1\t\n"
                    "addq $4, %%r14\t\n"

                    "vbroadcastss (%[filter_addr], %%r14), %%ymm3\t\n"
                    "vmovups (%[blue_float_array], %%rax,4), %%ymm4\t\n"
                    "vfmadd231ps %%ymm3, %%ymm4, %%ymm2\t\n"
                    "addq $4, %%r14\t\n"

                "INC_KER_C:\t\n"
                "addq $1, %%r13\t\n"
                "addq $1, %%rax\t\n"
                "jmp KER_C_CMP\t\n"

            "INC_KER_R:\t\n"
            "addq $1, %%rcx\t\n"
            "addq %[ori_width], %%rax\t\n"
            "subq $5, %%rax\t\n"
            "jmp KER_R_CMP\t\n"
            "KER_END:\t\n"

        // rcx as result offset
        "movq %[ori_width], %%rcx\t\n"
        "subq $4, %%rcx\t\n"
        "imulq %%r9, %%rcx\t\n"
        // rax no longer = bias
        "addq %%r8, %%rcx\t\n"
        //                    "imulq rax, 4\t\n"
        // rax == result offset now
        //                    "movq %%rax, %%rcx\t\n"
        "movq %[red_result_array], %%rdx\t\n"
        "vmovups %%ymm0, (%%rdx ,%%rcx, 4)\t\n"
        "movq %[green_result_array], %%rdx\t\n"
        "vmovups %%ymm1, (%%rdx ,%%rcx, 4)\t\n"
        "movq %[blue_result_array], %%rdx\t\n"
        "vmovups %%ymm2, (%%rdx ,%%rcx, 4)\t\n"


    "INC_C:\t\n"
    "addq $8, %%r8\t\n"
    // recovery rcx
    //                "popq %%rcx\t\n"
    "jmp COL_CMP\t\n"

    // boarder process
    "PROC_B:\t\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\t\n"
    "vxorps %%ymm1, %%ymm1, %%ymm1\t\n"
    "vxorps %%ymm2, %%ymm2, %%ymm2\t\n"

    "movq %[ori_width], %%rax\t\n"
    "imul %%r9, %%rax\t\n"
    "addl %[rest_bias], %%eax\t\n"

    // rcx as ker_r
    // r13 as ker_c
    // r14 as ker_bias
    "xorq %%r14, %%r14\t\n"
    "xorq %%rcx, %%rcx\t\n"
    "B_KER_R_CMP:\t\n"
    "cmpq $5, %%rcx\t\n"
    "jge B_KER_END\t\n"

        "xorq %%r13, %%r13\t\n"
        "B_KER_C_CMP:\t\n"
        "cmpq $5, %%r13\t\n"
        "jge B_INC_KER_R\t\n"

            "vbroadcastss (%[filter_addr], %%r14), %%ymm3\t\n"
            "vmaskmovps (%[red_float_array], %%rax, 4), %%ymm5, %%ymm4\t\n"
            "vfmadd231ps %%ymm3, %%ymm4, %%ymm0\t\n"
            "addq $4, %%r14\t\n"

            "vbroadcastss (%[filter_addr], %%r14), %%ymm3\t\n"
            "vmaskmovps (%[green_float_array], %%rax,4), %%ymm5, %%ymm4\t\n"
            "vfmadd231ps %%ymm3, %%ymm4, %%ymm1\t\n"
            "addq $4, %%r14\t\n"

            "vbroadcastss (%[filter_addr], %%r14), %%ymm3\t\n"
            "vmaskmovps (%[blue_float_array], %%rax,4), %%ymm5, %%ymm4\t\n"
            "vfmadd231ps %%ymm3, %%ymm4, %%ymm2\t\n"
            "addq $4, %%r14\t\n"

        "B_INC_KER_C:\t\n"
        "addq $1, %%r13\t\n"
        "addq $1, %%rax\t\n"
        "jmp B_KER_C_CMP\t\n"

    "B_INC_KER_R:\t\n"
    "addq $1, %%rcx\t\n"
    "addq %[ori_width], %%rax\t\n"
    "subq $5, %%rax\t\n"
    "jmp B_KER_R_CMP\t\n"
    "B_KER_END:\t\n"

    // rax as r * target_col + rest_bias
    "movq $8, %%rax\t\n"
    "addq %[target_col_8], %%rax\t\n"
    "imulq %%r9, %%rax\t\n"
    "addl %[rest_bias], %%eax\t\n"

    "movq %[red_result_array], %%rdx\t\n"
    "vmaskmovps %%ymm0, %%ymm5, (%%rdx, %%rax, 4)\t\n"
    "movq %[green_result_array], %%rdx\t\n"
    "vmaskmovps %%ymm1, %%ymm5, (%%rdx, %%rax, 4)\t\n"
    "movq %[blue_result_array], %%rdx\t\n"
    "vmaskmovps %%ymm2, %%ymm5, (%%rdx, %%rax, 4)\t\n"
    :
    :   [red_float_array]"r"(red_float_array),
    [green_float_array]"r"(green_float_array),
    [blue_float_array]"r"(blue_float_array),
    [red_result_array]"m"(red_result_array),
    [green_result_array]"m"(green_result_array),
    [blue_result_array]"m"(blue_result_array),
    [filter_addr]"r"(kernel),
    [ori_width]"r"((long long)(width + 4)),
    [row_index]"r"(row_idx),
    [target_col_8]"r"((long long)(width - 8)),
    [rest_bias]"m"(rest_bias),
    [write_mask]"m"(write_mask_array)
    :   "memory", "rax", "rdx", "rcx", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "r8", "r9", "r13", "r14"
    );
//    end = omp_get_wtime();
//    printf("FMA ASM @ %d Time cost : %f s\n", row_idx, end - start);
}

void convert_test(int row_idx) {
    int result_offset = row_idx * width;
    int result_target = result_offset + width;
    int ori_offset = (row_idx + 2) * (width + 4) + 2;
    for (; result_offset < result_target; ++result_offset, ++ori_offset) {
        red_result_array[result_offset] = red_float_array[ori_offset];
        blue_result_array[result_offset] = blue_float_array[ori_offset];
        green_result_array[result_offset] = green_float_array[ori_offset];
    }
}

void convert_to_uc_gather_asm(int index)
{
    double start, end;
    start = omp_get_wtime();

    int offset = thread_row_offset[index] * (width);
    int target = thread_row_offset[index + 1] * (width);

    int idx_array[8] = {0, 4, 2, 3, 4, 5, 6, 7};

    unsigned char red_idx_lo[16] = {255, 255, 0, 255, 255, 1, 255, 255, 2, 255, 255, 3, 255, 255, 4, 255};
    unsigned char red_idx_hi[16] = {255, 5, 255, 255, 6, 255, 255, 7, 255, 255, 255, 255, 255, 255, 255, 255};

    unsigned char green_idx_lo[16] = {255, 0, 255, 255, 1, 255, 255, 2, 255, 255, 3, 255, 255, 4, 255, 255};
    unsigned char green_idx_hi[16] = {5, 255, 255, 6, 255, 255, 7, 255, 255, 255, 255, 255, 255, 255, 255, 255};

    unsigned char blue_idx_lo[16] = {0, 255, 255, 1, 255, 255, 2, 255, 255, 3, 255, 255, 4, 255, 255, 5};
    unsigned char blue_idx_hi[16] = {255, 255, 6, 255, 255, 7, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};

    int write_array[8] = {-1, -1, -1, -1, -1, -1, 0, 0};

    __asm__(
    // ymm0 as zero reg
    // ymm1 as index reg
    // ymm2 as write mask
    // xmm6 as red idx lo
    // xmm7 as red idx hi
    // xmm8 as green idx lo
    // xmm9 as green idx hi
    // xmm10 as blue idx lo
    // xmm11 as blue idx hi
    // xmm12 as gather result lo
    // xmm13 as gather result hi
    // ymm4 as gather result
    // xmm5 as temp
    "vpxor %%ymm0, %%ymm0, %%ymm0\t\n"
    "vmovdqu %[idx_array], %%ymm1\t\n"
    "vmovdqu %[write_array], %%ymm2\t\n"
    "vmovdqa %[red_idx_lo], %%xmm6\t\n"
    "vmovdqa %[red_idx_hi], %%xmm7\t\n"
    "vmovdqa %[green_idx_lo], %%xmm8\t\n"
    "vmovdqa %[green_idx_hi], %%xmm9\t\n"
    "vmovdqa %[blue_idx_lo], %%xmm10\t\n"
    "vmovdqa %[blue_idx_hi], %%xmm11\t\n"

    "movq $3, %%rax\t\n"
    "imulq %[offset], %%rax\t\n"
    "CONV_UC_GATHER_CMP_OFFSET_DIRECT:\t\n"
    "cmpq %[target_8], %[offset]\t\n"
    "jge CONV_UC_GATHER_DIRECT_END\t\n"

        "vmovups (%[red_result_array], %[offset], 4), %%ymm3\t\n"
        "vcvtps2dq %%ymm3, %%ymm3\t\n"
        "vpackusdw %%ymm0, %%ymm3, %%ymm3\t\n"
        "vpackuswb %%ymm0, %%ymm3, %%ymm3\t\n"
        "vpermd %%ymm3, %%ymm1, %%ymm3\t\n"

        "vpshufb %%xmm6, %%xmm3, %%xmm12\t\n"
        "vpshufb %%xmm7, %%xmm3, %%xmm13\t\n"

        "vmovups (%[green_result_array], %[offset], 4), %%ymm3\t\n"
        "vcvtps2dq %%ymm3, %%ymm3\t\n"
        "vpackusdw %%ymm0, %%ymm3, %%ymm3\t\n"
        "vpackuswb %%ymm0, %%ymm3, %%ymm3\t\n"
        "vpermd %%ymm3, %%ymm1, %%ymm3\t\n"

        "vpshufb %%xmm8, %%xmm3, %%xmm5\t\n"
        "por %%xmm5, %%xmm12\t\n"
        "vpshufb %%xmm9, %%xmm3, %%xmm5\t\n"
        "por %%xmm5, %%xmm13\t\n"

        "vmovups (%[blue_result_array], %[offset], 4), %%ymm3\t\n"
        "vcvtps2dq %%ymm3, %%ymm3\t\n"
        "vpackusdw %%ymm0, %%ymm3, %%ymm3\t\n"
        "vpackuswb %%ymm0, %%ymm3, %%ymm3\t\n"
        "vpermd %%ymm3, %%ymm1, %%ymm3\t\n"

        "vpshufb %%xmm10, %%xmm3, %%xmm5\t\n"
        "por %%xmm5, %%xmm12\t\n"
        "vpshufb %%xmm11, %%xmm3, %%xmm5\t\n"
        "por %%xmm5, %%xmm13\t\n"

        "vinsertf128 $1, %%xmm13, %%ymm12, %%ymm4\t\n"
        "vpmaskmovd %%ymm4, %%ymm2, (%[target_write_array], %%rax)\t\n"


    "GATHER_INC_OFF_DIRECT:\t\n"
    "addq $8, %[offset]\t\n"
    "addq $24, %%rax\t\n"
    "jmp CONV_UC_GATHER_CMP_OFFSET_DIRECT\t\n"
    "CONV_UC_GATHER_DIRECT_END:\t\n"
    :
    :   [idx_array]"m"(idx_array),
    [write_array]"m"(write_array),
    [offset]"r"((long long)offset),
    [target_8]"r"((long long)(target - 8)),
    [red_result_array]"r"(red_result_array),
    [green_result_array]"r"(green_result_array),
    [blue_result_array]"r"(blue_result_array),
    [red_idx_lo]"m"(red_idx_lo),
    [red_idx_hi]"m"(red_idx_hi),
    [green_idx_lo]"m"(green_idx_lo),
    [green_idx_hi]"m"(green_idx_hi),
    [blue_idx_lo]"m"(blue_idx_lo),
    [blue_idx_hi]"m"(blue_idx_hi),
    [target_write_array]"r"(target_write_array)
    :   "memory", "rax", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13"
    );

    int rest_pix = (target - thread_row_offset[index] * width) % 8;
    int rest_offset = target - rest_pix - 8;
    int rest_offset_x3 = rest_offset * 3;
    rest_pix += 8;

    for (int i = 0; i < rest_pix; ++i, ++rest_offset, rest_offset_x3 += 3) {
        int temp = (int)red_result_array[rest_offset];
        target_write_array[rest_offset_x3 + 2] = temp > 255 ? 255 : (temp < 0 ? 0 : temp);
        temp = (int)green_result_array[rest_offset];
        target_write_array[rest_offset_x3 + 1] = temp > 255 ? 255 : (temp < 0 ? 0 : temp);
        temp = (int)blue_result_array[rest_offset];
        target_write_array[rest_offset_x3] = temp > 255 ? 255 : (temp < 0 ? 0 : temp);
    }

    end = omp_get_wtime();
    printf("CONVERT UC GATHER @ %d Time cost : %f s\n", index, end - start);
}

void save_image(FILE* file_ptr)
{
    double start, end;
    start = omp_get_wtime();

    unsigned short fileType = 0x4d42;
    BITMAPFILEHEADER fileHeader;
    BITMAPINFOHEADER infoHeader;

    fileHeader.bfSize = (width) * (height) * 3 + 54;
    fileHeader.bfReserved1 = 0;
    fileHeader.bfReserved2 = 0;
    fileHeader.bfOffBits = 54;

    infoHeader.biSize = 40;
    infoHeader.biWidth = width;
    infoHeader.biHeight = height;
    infoHeader.biPlanes = 1;
    infoHeader.biBitCount = 24;
    infoHeader.biCompression = 0;
    infoHeader.biSizeImage = (width) * (height) * 3;
    infoHeader.biXPelsPerMeter = 0;
    infoHeader.biYPelsPerMeter = 0;
    infoHeader.biClrUsed = 0;
    infoHeader.biClrImportant = 0;

    fwrite(&fileType, sizeof(unsigned short), 1, file_ptr);
    fwrite(&fileHeader, sizeof(BITMAPFILEHEADER), 1, file_ptr);
    fwrite(&infoHeader, sizeof(BITMAPINFOHEADER), 1, file_ptr);

    fwrite(target_write_array, sizeof(unsigned char), (height) * (width) * 3, file_ptr);

    end = omp_get_wtime();
    printf("SAVE Time cost : %f s\n", end - start);
}

int main(int argc, char* argv[]) {
    thread_cnt = strtol(argv[1], NULL, 10);
    

    double start, end;
    start = omp_get_wtime();

    FILE* file_ptr = fopen("1.bmp", "rb");
    if (file_ptr == NULL) {
        printf("Image don't exist.\n");
        fflush(stdout);
        return 0;
    }

    load_image(file_ptr);

    fclose(file_ptr);

    int target_row = height;
    int result_size = width * height;
    int ori_size = (width + 4) * (height + 4);
    red_result_array = (float*)malloc(result_size * sizeof(float));
    green_result_array = (float*)malloc(result_size * sizeof(float));
    blue_result_array = (float*)malloc(result_size * sizeof(float));

    red_float_array = (float*)malloc(ori_size * sizeof(float));
    green_float_array = (float*)malloc(ori_size * sizeof(float));
    blue_float_array = (float*)malloc(ori_size * sizeof(float));


    thread_row_offset = (int*)malloc((thread_cnt + 1) * sizeof(int));
    ori_row_offset = (int*)malloc((thread_cnt + 1) * sizeof(int));

    target_col = width - 4;

    // boarder process
    rest_pix = width % 8;
    rest_bias = width - rest_pix;

    for (int i = 0; i < rest_pix; ++i) {
        write_mask_array[i] = -1;
    }

    thread_row_offset[0] = 0;
    ori_row_offset[0] = 0;

    int less_cnt = thread_cnt - target_row % thread_cnt;
    int less_raw_cnt = target_row / thread_cnt;
    for (int i = 0; i < less_cnt; ++i) {
        thread_row_offset[i + 1] = thread_row_offset[i] + less_raw_cnt;
    }

    for (int i = less_cnt; i < thread_cnt; ++i) {
        thread_row_offset[i + 1] = thread_row_offset[i] + less_raw_cnt + 1;
    }

    less_cnt = thread_cnt - (height + 4) % thread_cnt;
    less_raw_cnt = (height + 4) / thread_cnt;

    for (int i = 0; i < less_cnt; ++i) {
        ori_row_offset[i + 1] = ori_row_offset[i] + less_raw_cnt;
    }

    for (int i = less_cnt; i < thread_cnt; ++i) {
        ori_row_offset[i + 1] = ori_row_offset[i] + less_raw_cnt + 1;
    }

    printf("Parallel Start.\n");

    double broadcast_start, broadcast_end;
    broadcast_start = omp_get_wtime();

    memset(red_float_array, 0, (width + 4) * 2);
    memset(green_float_array, 0, (width + 4) * 2);
    memset(blue_float_array, 0, (width + 4) * 2);
# pragma omp parallel for num_threads(thread_cnt) default(none) shared(height) schedule(guided)
    for (int i = 2; i < height + 2; ++i) {
        broadcast_convert_asm(i);
    }

# pragma omp barrier

    memset(red_float_array + (width + 4) * (height + 2), 0, (width + 4) * 2);
    memset(green_float_array + (width + 4) * (height + 2), 0, (width + 4) * 2);
    memset(blue_float_array + (width + 4) * (height + 2), 0, (width + 4) * 2);

    broadcast_end = omp_get_wtime();
    printf("BROADCAST Time cost : %f s\n", broadcast_end - broadcast_start);

    double conv_start, conv_end;
    conv_start = omp_get_wtime();

# pragma omp parallel for num_threads(thread_cnt) default(none) shared(height) schedule(guided)
    for (int i = 0; i < height; ++i) {
        conv_all_fma_asm(i);
    }

# pragma omp barrier
    conv_end = omp_get_wtime();
    printf("CONV Time cost : %f s\n", conv_end - conv_start);


    target_write_array = (unsigned char*)malloc(result_size * sizeof(unsigned char) * 3);

//    double gather_start, gather_end;
//    gather_start = omp_get_wtime();
# pragma omp parallel for num_threads(thread_cnt) default(none) shared(thread_cnt)
    for (int i = 0; i < thread_cnt; ++i) {
        convert_to_uc_gather_asm(i);
    }

# pragma omp barrier
//    gather_end = omp_get_wtime();
//    printf("CONV Time cost : %f s\n", gather_end - gather_start);
    FILE* savePtr = fopen("openmp.bmp", "wb+");

    save_image(savePtr);

#ifdef STORE_RESULT
    save_txt_file(red_result_array, (height - 4) * (width - 4), "red_result.txt");
    save_txt_file(red_result_array, (height - 4) * (width - 4), "green_result.txt");
    save_txt_file(red_result_array, (height - 4) * (width - 4), "blue_result.txt");
    save_txt_file((float*)kernel, 5 * 5 * 3, "filter.txt");
#endif
    printf("Parallel End.\n");

    // free memory

    double free_start = omp_get_wtime();

    free(ori_row_offset);
    free(thread_row_offset);
    free(blue_float_array);
    free(green_float_array);
    free(red_float_array);
    free(blue_result_array);
    free(green_result_array);
    free(red_result_array);
//    free(blue_array);
//    free(green_array);
//    free(red_array);
    free(target_write_array);
    free(ori_byte_img);

    end = omp_get_wtime();
    printf("Free Time cost : %f s\n", end - free_start);

    printf("Total Time cost : %f s\n", end - start);
    return 0;
}
