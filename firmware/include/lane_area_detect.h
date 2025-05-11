#ifndef __LANE_AREA_DETECT_H__
#define __LANE_AREA_DETECT_H__

#include "config.h"
#include "esp_camera.h"
#include "common_types.h"
#include "esp_tflite.h"
#include <vector>

// 为避免重复定义ROI_RATIO，如果它在config.h中没有定义，则在这里定义
#ifndef ROI_RATIO
#define ROI_RATIO 0.4f
#endif

class LaneAreaDetect
{
public:
    // 构造和析构函数
    LaneAreaDetect();
    ~LaneAreaDetect();

    // 统一对外接口函数
    std::vector<LaneInfo> detect(uint8_t *imageData, int width, int height, int channels);
    void setUseTraditionalMethod(bool useTraditional)
    {
        m_useTraditionalMethod = useTraditional;
        // ESP_LOGI(TAG, "车道线检测设置为%s模式",
        //          useTraditional ? "传统" : "直接segmentation");
    }
    bool getUseTraditionalMethod() const
    {
        return m_useTraditionalMethod;
    }

private:
    bool m_useTraditionalMethod = true; // 默认使用传统方法

    // 传统图像处理函数 - 原始接口
    esp_err_t color_regmentation(camera_fb_t *fb, processed_image_t *output); // step1
    esp_err_t color_2_gray(camera_fb_t *fb, processed_image_t *output);       // step2
    esp_err_t multi_scale_DoG(camera_fb_t *fb, processed_image_t *output);    // step3
    esp_err_t enhance_edges(camera_fb_t *fb, processed_image_t *output);      // step4
    esp_err_t ROI_mask(camera_fb_t *fb, processed_image_t *output);           // step5
    esp_err_t detect_lines(camera_fb_t *fb, processed_image_t *output);       // step6
    esp_err_t fit_lane_lines(camera_fb_t *fb, processed_image_t *output);     // step7

    // 模型处理 - 原始接口
    esp_err_t fastSCNN_load();                                            // step8
    esp_err_t model_fit_area(camera_fb_t *fb, processed_image_t *output); // step9

    // 最后车道线扩展 - 向左右两侧各扩展100像素的红色掩膜区域
    // 并分三级区域，车道线内侧为0级区域，车道线联合红色掩膜为1级区域，最外侧为2级区域
    esp_err_t lane_line_expand(camera_fb_t *fb, processed_image_t *output); // step10

    // 辅助绘图函数
    void bresenham_line(processed_image_t *img, int x1, int y1, int x2, int y2,
                        uint8_t r, uint8_t g, uint8_t b);

    void bresenham_line(uint8_t *data, int width, int height, int channels,
                        int x1, int y1, int x2, int y2, uint8_t r, uint8_t g, uint8_t b);

private:
    // 形态学处理内核
    uint8_t kernel_large[15][15]; // 用于形态学操作的大内核

    // FastSCNN模型相关
    bool model_loaded = false;
    tflite::MicroInterpreter *interpreter = nullptr;
    int model_input_width = 0;
    int model_input_height = 0;
    uint8_t *model_data = nullptr;
    const tflite::Model *tflite_model = nullptr;
    uint8_t *tensor_arena = nullptr;
    static bool resolver_initialized;

    // 车道线段存储
    line_segment_t *left_lines = nullptr;
    line_segment_t *right_lines = nullptr;
    int left_line_count = 0;
    int right_line_count = 0;

    // 拟合结果
    poly_fit_t left_lane_fit;
    poly_fit_t right_lane_fit;
};

#endif
