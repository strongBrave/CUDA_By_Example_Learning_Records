#ifndef __CPU_BITMAP_H__
#define __CPU_BITMAP_H__

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" // 用于保存图片的库

#include <iostream>
#include <string>

struct CPUBitmap {
    unsigned char *pixels;
    int x, y;
    void *dataBlock;
    void (*bitmapExit)(void *);

    CPUBitmap(int width, int height, void *d = NULL) {
        pixels = new unsigned char[width * height * 4];
        x = width;
        y = height;
        dataBlock = d;
    }

    ~CPUBitmap() {
        delete[] pixels;
    }

    unsigned char *get_ptr(void) const { return pixels; }
    long image_size(void) const { return x * y * 4; }

    // 保存图片并调用退出函数
    void display_and_exit(void (*e)(void *) = NULL) {
        bitmapExit = e;

        // 保存图像到文件
        save_to_file("output_image.png");

        // 如果有退出函数，执行退出清理逻辑
        if (bitmapExit != NULL) {
            bitmapExit(dataBlock);
        }
    }

    // 保存图片为 PNG 文件
    void save_to_file(const std::string &filename) {
        if (!stbi_write_png(filename.c_str(), x, y, 4, pixels, x * 4)) {
            std::cerr << "Failed to save image: " << filename << std::endl;
        } else {
            std::cout << "Image saved to: " << filename << std::endl;
        }
    }
};

#endif // __CPU_BITMAP_H__
