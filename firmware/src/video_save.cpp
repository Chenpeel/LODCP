#include "video_save.h"
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "esp_camera.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

// 构造函数
VideoSave::VideoSave() : 
    baseDir(BASE_SAVE_PATH),
    aviFile(nullptr),
    frameWidth(0),
    frameHeight(0),
    frameRate(DEFAULT_FPS),
    totalFrames(0),
    moviListPos(0),
    aviStartPos(0)
{
    // 初始化保存目录
    initSaveDirectory();
}

// 析构函数
VideoSave::~VideoSave() {
    // 确保文件被正确关闭
    finishVideoCapture();
}

// 初始化保存目录
bool VideoSave::initSaveDirectory() {
    ESP_LOGI(TAG, "正在初始化保存目录: %s", baseDir.c_str());
    
    // 创建基本目录
    struct stat st;
    if (stat(baseDir.c_str(), &st) != 0) {
        // 目录不存在，尝试创建
        if (mkdir(baseDir.c_str(), 0755) != 0) {
            ESP_LOGE(TAG, "无法创建基本保存目录: %s", baseDir.c_str());
            return false;
        }
    }
    
    // 测试目录可写性
    std::string testFile = baseDir + "/test_write.tmp";
    FILE* fp = fopen(testFile.c_str(), "w");
    if (fp == nullptr) {
        ESP_LOGE(TAG, "无法写入保存目录: %s", baseDir.c_str());
        return false;
    }
    
    // 写入测试内容
    fprintf(fp, "Test write");
    fclose(fp);
    
    // 删除测试文件
    remove(testFile.c_str());
    
    ESP_LOGI(TAG, "保存目录初始化成功");
    return true;
}

// 生成时间戳
std::string VideoSave::generateTimestamp() {
    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return ss.str();
}

// 开始新的保存会话
bool VideoSave::startSession() {
    // 生成新的会话目录名（基于时间戳）
    std::string timestamp = generateTimestamp();
    currentSessionDir = baseDir + "/" + timestamp;
    
    // 创建会话目录
    if (mkdir(currentSessionDir.c_str(), 0755) != 0) {
        ESP_LOGE(TAG, "无法创建会话目录: %s", currentSessionDir.c_str());
        return false;
    }
    
    ESP_LOGI(TAG, "创建新的保存会话: %s", currentSessionDir.c_str());
    return true;
}

// 结束当前会话
void VideoSave::endSession() {
    // 确保视频捕获已结束
    finishVideoCapture();
    
    // 清空会话信息
    currentSessionDir = "";
    ESP_LOGI(TAG, "会话结束");
}

// 转换为JPEG格式（使用ESP32相机库）
bool VideoSave::convertToJPEG(uint8_t* input, int width, int height, int channels, 
                           uint8_t** output, size_t* outSize) {
    if (input == nullptr || output == nullptr || outSize == nullptr) {
        return false;
    }
    
    // 为JPEG编码分配内存
    camera_fb_t fb = {0};
    fb.width = width;
    fb.height = height;
    fb.format = PIXFORMAT_RGB888; // 假设输入是RGB888格式
    fb.buf = input;
    fb.len = width * height * channels;
    
    *outSize = 0;
    *output = nullptr;
    
    // 根据输入格式选择转换方式
    if (channels == 3) {
        // 处理RGB888格式
        size_t jpgSize = 0;
        bool ret = frame2jpg(&fb, QUALITY, output, &jpgSize);
        if (ret) {
            *outSize = jpgSize;
            return true;
        } else {
            ESP_LOGE(TAG, "RGB888转JPEG失败");
            return false;
        }
    } else if (channels == 1) {
        // 已经是JPEG格式的数据，直接复制
        *output = (uint8_t*)malloc(fb.len);
        if (*output == nullptr) {
            ESP_LOGE(TAG, "内存分配失败");
            return false;
        }
        memcpy(*output, input, fb.len);
        *outSize = fb.len;
        return true;
    } else {
        ESP_LOGE(TAG, "不支持的图像格式: %d 通道", channels);
        return false;
    }
}

// 在图像上可视化检测结果
void VideoSave::visualizeDetections(uint8_t* imageData, int width, int height, int channels,
                                 const std::vector<Detection>& detections) {
    if (imageData == nullptr || channels != 3 || detections.empty()) {
        return;
    }
    
    // 定义类别颜色 (RGB)
    const uint8_t colors[2][3] = {
        {255, 0, 0},    // Active (红色)
        {0, 255, 0}     // Traffic Sign (绿色)
    };
    
    // 类别名称
    const char* classNames[2] = {
        "Active",
        "Traffic Sign"
    };
    
    // 遍历所有检测结果
    for (const auto& det : detections) {
        // 将归一化坐标转换为图像坐标
        int x = static_cast<int>(det.x * width);
        int y = static_cast<int>(det.y * height);
        int w = static_cast<int>(det.width * width);
        int h = static_cast<int>(det.height * height);
        
        // 确保边界在图像内
        x = std::max(0, std::min(x, width - 1));
        y = std::max(0, std::min(y, height - 1));
        w = std::min(w, width - x - 1);
        h = std::min(h, height - y - 1);
        
        // 获取颜色
        uint8_t r, g, b;
        if (det.classId < 2) {
            r = colors[det.classId][0];
            g = colors[det.classId][1];
            b = colors[det.classId][2];
        } else {
            // 默认颜色
            r = 255;
            g = 255;
            b = 0; // 黄色
        }
        
        // 绘制边界框
        int lineWidth = 2;
        
        // 上边框
        for (int i = std::max(0, x - lineWidth); i < std::min(width, x + w + lineWidth); i++) {
            for (int j = std::max(0, y - lineWidth); j < std::min(height, y + lineWidth); j++) {
                if (i >= 0 && i < width && j >= 0 && j < height) {
                    imageData[(j * width + i) * channels] = r;
                    imageData[(j * width + i) * channels + 1] = g;
                    imageData[(j * width + i) * channels + 2] = b;
                }
            }
        }
        
        // 下边框
        for (int i = std::max(0, x - lineWidth); i < std::min(width, x + w + lineWidth); i++) {
            for (int j = std::max(0, y + h - lineWidth); j < std::min(height, y + h + lineWidth); j++) {
                if (i >= 0 && i < width && j >= 0 && j < height) {
                    imageData[(j * width + i) * channels] = r;
                    imageData[(j * width + i) * channels + 1] = g;
                    imageData[(j * width + i) * channels + 2] = b;
                }
            }
        }
        
        // 左边框
        for (int i = std::max(0, x - lineWidth); i < std::min(width, x + lineWidth); i++) {
            for (int j = std::max(0, y - lineWidth); j < std::min(height, y + h + lineWidth); j++) {
                if (i >= 0 && i < width && j >= 0 && j < height) {
                    imageData[(j * width + i) * channels] = r;
                    imageData[(j * width + i) * channels + 1] = g;
                    imageData[(j * width + i) * channels + 2] = b;
                }
            }
        }
        
        // 右边框
        for (int i = std::max(0, x + w - lineWidth); i < std::min(width, x + w + lineWidth); i++) {
            for (int j = std::max(0, y - lineWidth); j < std::min(height, y + h + lineWidth); j++) {
                if (i >= 0 && i < width && j >= 0 && j < height) {
                    imageData[(j * width + i) * channels] = r;
                    imageData[(j * width + i) * channels + 1] = g;
                    imageData[(j * width + i) * channels + 2] = b;
                }
            }
        }
        
        // 添加标签
        std::string label = std::string(det.classId < 2 ? classNames[det.classId] : "Unknown") + 
                          " " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";
        
        // 简单文本渲染（仅英文字符支持）
        int textX = x;
        int textY = std::max(0, y - 10);
        int fontSize = 1;
        int charWidth = 6 * fontSize;
        
        for (size_t c = 0; c < label.size(); c++) {
            int tx = textX + c * charWidth;
            if (tx + charWidth >= width) break;
            
            // 绘制字符背景（黑色矩形）
            for (int i = tx; i < tx + charWidth; i++) {
                for (int j = textY; j < textY + 10; j++) {
                    if (i >= 0 && i < width && j >= 0 && j < height) {
                        imageData[(j * width + i) * channels] = 0;
                        imageData[(j * width + i) * channels + 1] = 0;
                        imageData[(j * width + i) * channels + 2] = 0;
                    }
                }
            }
            
            // 绘制简单的字符（仅绘制一个点表示文本位置，实际项目中可以使用更复杂的字体渲染）
            for (int i = tx + 1; i < tx + charWidth - 1; i++) {
                for (int j = textY + 1; j < textY + 9; j++) {
                    if (i >= 0 && i < width && j >= 0 && j < height) {
                        imageData[(j * width + i) * channels] = 255;
                        imageData[(j * width + i) * channels + 1] = 255;
                        imageData[(j * width + i) * channels + 2] = 255;
                    }
                }
            }
        }
    }
}

// 保存单帧图像
bool VideoSave::saveFrame(const std::shared_ptr<FrameData>& frameData, bool visualize) {
    if (frameData == nullptr || frameData->imageData == nullptr || 
        currentSessionDir.empty() || frameData->width <= 0 || frameData->height <= 0) {
        ESP_LOGE(TAG, "无效的帧数据或未初始化会话");
        return false;
    }
    
    // 生成文件名
    std::string timestamp = generateTimestamp();
    std::string filename = currentSessionDir + "/frame_" + 
                         std::to_string(frameData->frameId) + "_" + 
                         timestamp + ".jpg";
    
    // 准备图像数据
    uint8_t* imageToSave = nullptr;
    bool needToFreeImage = false;
    
    if (visualize && !frameData->detections.empty()) {
        // 复制原始图像以便修改
        size_t imgSize = frameData->width * frameData->height * frameData->channels;
        imageToSave = (uint8_t*)malloc(imgSize);
        if (imageToSave == nullptr) {
            ESP_LOGE(TAG, "内存分配失败");
            return false;
        }
        memcpy(imageToSave, frameData->imageData, imgSize);
        
        // 在图像上绘制检测结果
        visualizeDetections(imageToSave, frameData->width, frameData->height, 
                          frameData->channels, frameData->detections);
        
        needToFreeImage = true;
    } else {
        // 使用原始图像
        imageToSave = frameData->imageData;
    }
    
    // 将图像转换为JPEG
    uint8_t* jpegData = nullptr;
    size_t jpegSize = 0;
    bool conversionResult = convertToJPEG(imageToSave, frameData->width, frameData->height, 
                                        frameData->channels, &jpegData, &jpegSize);
    
    // 如果复制了图像，需要释放
    if (needToFreeImage) {
        free(imageToSave);
    }
    
    if (!conversionResult || jpegData == nullptr) {
        ESP_LOGE(TAG, "图像转换为JPEG失败");
        return false;
    }
    
    // 保存JPEG文件
    FILE* fp = fopen(filename.c_str(), "wb");
    if (fp == nullptr) {
        ESP_LOGE(TAG, "无法创建文件: %s", filename.c_str());
        free(jpegData);
        return false;
    }
    
    size_t written = fwrite(jpegData, 1, jpegSize, fp);
    fclose(fp);
    free(jpegData);
    
    if (written != jpegSize) {
        ESP_LOGE(TAG, "写入文件失败，expected %d bytes, wrote %d bytes", jpegSize, written);
        return false;
    }
    
    ESP_LOGI(TAG, "保存帧 %d 到 %s，大小: %d bytes", frameData->frameId, filename.c_str(), jpegSize);
    return true;
}

// 开始视频捕获
bool VideoSave::startVideoCapture(int width, int height, int fps) {
    if (width <= 0 || height <= 0 || fps <= 0 || currentSessionDir.empty()) {
        ESP_LOGE(TAG, "无效的参数或未初始化会话");
        return false;
    }
    
    // 如果已经有打开的文件，先关闭
    finishVideoCapture();
    
    // 生成文件名
    std::string timestamp = generateTimestamp();
    currentAviFile = currentSessionDir + "/video_" + timestamp + ".avi";
    
    // 打开文件
    aviFile = fopen(currentAviFile.c_str(), "wb");
    if (aviFile == nullptr) {
        ESP_LOGE(TAG, "无法创建视频文件: %s", currentAviFile.c_str());
        return false;
    }
    
    // 保存参数
    frameWidth = width;
    frameHeight = height;
    frameRate = fps;
    totalFrames = 0;
    
    // 清空索引列表
    indexEntries.clear();
    
    // 写入AVI头部
    if (!writeAVIHeader(width, height, fps)) {
        ESP_LOGE(TAG, "写入AVI头部失败");
        fclose(aviFile);
        aviFile = nullptr;
        return false;
    }
    
    ESP_LOGI(TAG, "开始视频捕获: %s, 分辨率: %dx%d, 帧率: %d", 
             currentAviFile.c_str(), width, height, fps);
    
    return true;
}

// 写入AVI文件头
bool VideoSave::writeAVIHeader(int width, int height, int fps) {
    if (aviFile == nullptr) {
        return false;
    }
    
    // 保存文件起始位置
    aviStartPos = ftell(aviFile);
    
    // 1. RIFF AVI Header
    fwrite("RIFF", 1, 4, aviFile);
    uint32_t riffSize = 0; // 将在最后更新
    fwrite(&riffSize, 1, 4, aviFile);
    fwrite("AVI ", 1, 4, aviFile);
    
    // 2. LIST hdrl
    fwrite("LIST", 1, 4, aviFile);
    uint32_t hdrlSize = 4 + 8 + sizeof(AVIHeader) + 8 + sizeof(AVIStreamHeader) + 8 + sizeof(BITMAPINFOHEADER);
    fwrite(&hdrlSize, 1, 4, aviFile);
    fwrite("hdrl", 1, 4, aviFile);
    
    // 3. avih
    fwrite("avih", 1, 4, aviFile);
    uint32_t avihSize = sizeof(AVIHeader);
    fwrite(&avihSize, 1, 4, aviFile);
    
    AVIHeader avih = {0};
    avih.microSecPerFrame = static_cast<uint32_t>(1000000 / fps);
    avih.maxBytesPerSec = width * height * 3 * fps; // 估计最大数据率
    avih.paddingGranularity = 0;
    avih.flags = 0x10; // AVIF_HASINDEX
    avih.totalFrames = 0; // 将在最后更新
    avih.initialFrames = 0;
    avih.streams = 1; // 只有视频流
    avih.suggestedBufferSize = width * height * 3;
    avih.width = width;
    avih.height = height;
    // reserved[4]都为0
    
    fwrite(&avih, 1, sizeof(avih), aviFile);
    
    // 4. LIST strl
    fwrite("LIST", 1, 4, aviFile);
    uint32_t strlSize = 4 + 8 + sizeof(AVIStreamHeader) + 8 + sizeof(BITMAPINFOHEADER);
    fwrite(&strlSize, 1, 4, aviFile);
    fwrite("strl", 1, 4, aviFile);
    
    // 5. strh
    fwrite("strh", 1, 4, aviFile);
    uint32_t strhSize = sizeof(AVIStreamHeader);
    fwrite(&strhSize, 1, 4, aviFile);
    
    AVIStreamHeader strh = {0};
    strh.fccType = AVIConstants::VIDS_FOURCC;
    strh.fccHandler = AVIConstants::MJPG_FOURCC;
    strh.flags = 0;
    strh.priority = 0;
    strh.language = 0;
    strh.initialFrames = 0;
    strh.scale = 1;
    strh.rate = fps;
    strh.start = 0;
    strh.length = 0; // 将在最后更新
    strh.suggestedBufferSize = width * height * 3;
    strh.quality = 10000; // 最高质量
    strh.sampleSize = 0;
    // rcFrame默认为0
    
    fwrite(&strh, 1, sizeof(strh), aviFile);
    
    // 6. strf (BITMAPINFOHEADER)
    fwrite("strf", 1, 4, aviFile);
    uint32_t strfSize = sizeof(BITMAPINFOHEADER);
    fwrite(&strfSize, 1, 4, aviFile);
    
    BITMAPINFOHEADER bih = {0};
    bih.biSize = sizeof(BITMAPINFOHEADER);
    bih.biWidth = width;
    bih.biHeight = height;
    bih.biPlanes = 1;
    bih.biBitCount = 24; // 24位RGB
    bih.biCompression = AVIConstants::MJPG_FOURCC;
    bih.biSizeImage = width * height * 3;
    bih.biXPelsPerMeter = 0;
    bih.biYPelsPerMeter = 0;
    bih.biClrUsed = 0;
    bih.biClrImportant = 0;
    
    fwrite(&bih, 1, sizeof(bih), aviFile);
    
    // 7. LIST movi
    fwrite("LIST", 1, 4, aviFile);
    uint32_t moviSize = 4; // 仅包括'movi'标志，实际大小将在最后更新
    moviListPos = ftell(aviFile); // 保存movi列表大小位置
    fwrite(&moviSize, 1, 4, aviFile);
    fwrite("movi", 1, 4, aviFile);
    
    fflush(aviFile);
    
    return true;
}

// 添加帧到视频
bool VideoSave::addFrameToVideo(const std::shared_ptr<FrameData>& frameData, bool visualize) {
    if (aviFile == nullptr || frameData == nullptr || frameData->imageData == nullptr) {
        ESP_LOGE(TAG, "无效的参数或视频捕获未初始化");
        return false;
    }
    
    // 准备图像数据
    uint8_t* imageToSave = nullptr;
    bool needToFreeImage = false;
    
    if (visualize && !frameData->detections.empty()) {
        // 复制原始图像以便修改
        size_t imgSize = frameData->width * frameData->height * frameData->channels;
        imageToSave = (uint8_t*)malloc(imgSize);
        if (imageToSave == nullptr) {
            ESP_LOGE(TAG, "内存分配失败");
            return false;
        }
        memcpy(imageToSave, frameData->imageData, imgSize);
        
        // 在图像上绘制检测结果
        visualizeDetections(imageToSave, frameData->width, frameData->height, 
                          frameData->channels, frameData->detections);
        
        needToFreeImage = true;
    } else {
        // 使用原始图像
        imageToSave = frameData->imageData;
    }
    
    // 将图像转换为JPEG
    uint8_t* jpegData = nullptr;
    size_t jpegSize = 0;
    bool conversionResult = convertToJPEG(imageToSave, frameData->width, frameData->height, 
                                        frameData->channels, &jpegData, &jpegSize);
    
    // 如果复制了图像，需要释放
    if (needToFreeImage) {
        free(imageToSave);
    }
    
    if (!conversionResult || jpegData == nullptr) {
        ESP_LOGE(TAG, "图像转换为JPEG失败");
        return false;
    }
    
    // 写入帧数据
    uint32_t chunkOffset = ftell(aviFile) - aviStartPos - 8; // 帧在文件中的偏移
    
    // 写入00dc标记和数据大小
    fwrite("00dc", 1, 4, aviFile);
    fwrite(&jpegSize, 1, 4, aviFile);
    
    // 写入JPEG数据
    fwrite(jpegData, 1, jpegSize, aviFile);
    
    // 如果大小为奇数，添加padding
    if (jpegSize % 2 != 0) {
        uint8_t padding = 0;
        fwrite(&padding, 1, 1, aviFile);
    }
    
    // 添加到索引
    AVIINDEXENTRY indexEntry;
    indexEntry.ckid = AVIConstants::DC_FOURCC;
    indexEntry.dwFlags = 0x10; // AVIIF_KEYFRAME
    indexEntry.dwChunkOffset = chunkOffset;
    indexEntry.dwChunkLength = jpegSize;
    indexEntries.push_back(indexEntry);
    
    // 释放JPEG数据
    free(jpegData);
    
    totalFrames++;
    
    // 每10帧输出一次进度日志
    if (totalFrames % 10 == 0) {
        ESP_LOGI(TAG, "已添加 %d 帧到视频", totalFrames);
    }
    
    return true;
}

// 完成视频捕获
bool VideoSave::finishVideoCapture() {
    if (aviFile == nullptr) {
        return false; // 没有打开的视频文件
    }
    
    bool result = finalizeAVI();
    fclose(aviFile);
    aviFile = nullptr;
    
    ESP_LOGI(TAG, "完成视频捕获，总共 %d 帧，文件: %s", 
             totalFrames, currentAviFile.c_str());
    
    // 清空变量
    currentAviFile = "";
    totalFrames = 0;
    
    return result;
}

// 完成AVI文件写入
bool VideoSave::finalizeAVI() {
    if (aviFile == nullptr) {
        return false;
    }
    
    // 计算movi列表大小
    uint64_t endPos = ftell(aviFile);
    uint32_t moviSize = endPos - moviListPos - 8; // 减去LIST和大小字段
    
    // 写入idx1块
    fwrite("idx1", 1, 4, aviFile);
    uint32_t idxSize = indexEntries.size() * sizeof(AVIINDEXENTRY);
    fwrite(&idxSize, 1, 4, aviFile);
    
    // 写入索引数据
    for (const auto& entry : indexEntries) {
        fwrite(&entry, 1, sizeof(AVIINDEXENTRY), aviFile);
    }
    
    // 更新文件大小
    uint64_t fileEndPos = ftell(aviFile);
    uint32_t fileSize = fileEndPos - aviStartPos - 8; // 减去RIFF和大小字段
    
    // 更新RIFF大小
    fseek(aviFile, aviStartPos + 4, SEEK_SET);
    fwrite(&fileSize, 1, 4, aviFile);
    
    // 更新movi列表大小
    fseek(aviFile, moviListPos, SEEK_SET);
    fwrite(&moviSize, 1, 4, aviFile);
    
    // 更新总帧数
    fseek(aviFile, aviStartPos + 48, SEEK_SET);
    fwrite(&totalFrames, 1, 4, aviFile);
    
    // 更新视频流长度
    fseek(aviFile, aviStartPos + 128, SEEK_SET);
    fwrite(&totalFrames, 1, 4, aviFile);
    
    fflush(aviFile);
    return true;
}