#ifndef __CPU_ANIM_H__
#define __CPU_ANIM_H__
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <string>
#include "stb_image_write.h" // 使用 stb_image_write 保存图片

struct CPUAnimBitmap {
    unsigned char *pixels;
    int width, height;
    void *dataBlock;
    void (*fAnim)(void *, int);  // 动画生成函数
    void (*animExit)(void *);    // 退出清理函数

    CPUAnimBitmap(int w, int h, void *d = NULL) {
        width = w;
        height = h;
        pixels = new unsigned char[width * height * 4];
        dataBlock = d;
    }

    ~CPUAnimBitmap() {
        delete[] pixels;
    }

    unsigned char *get_ptr(void) const { return pixels; }
    long image_size(void) const { return width * height * 4; }

    // 保存当前帧为 PNG 图片
    void save_to_file(const std::string &filename) {
        if (!stbi_write_png(filename.c_str(), width, height, 4, pixels, width * 4)) {
            std::cerr << "Failed to save image: " << filename << std::endl;
        } else {
            std::cout << "Image saved to: " << filename << std::endl;
        }
    }

    // 替代 anim_and_exit 的实现，仅保存图片，不播放动画
    void anim_and_exit(void (*f)(void *, int), void (*e)(void *)) {
        fAnim = f;
        animExit = e;

        int total_frames = 1800; // 设定保存的总帧数
        for (int i = 0; i < total_frames; ++i) {
            fAnim(dataBlock, i); // 调用生成帧的函数

            // 保存当前帧为图片
            std::string filename = "/nas/home/yujunhao/cuda_by_example/code/chapter_7/imgs/frame_" + std::to_string(i) + ".png";
            save_to_file(filename);
        }

        animExit(dataBlock); // 调用清理函数
    }
};

#endif // __CPU_ANIM_H__
