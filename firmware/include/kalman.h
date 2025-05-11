#ifndef __KALMAN_H__
#define __KALMAN_H__

#include <vector>
#include <cmath>

// 简化版卡尔曼滤波器
class SimpleKalmanFilter {
public:
    // 构造函数，初始化状态和噪声参数
    SimpleKalmanFilter(int stateSize, int measureSize);
    
    // 默认构造函数，需要后续配置
    SimpleKalmanFilter();
    
    // 析构函数
    ~SimpleKalmanFilter();
    
    // 初始化函数
    void init(int stateSize, int measureSize);
    
    // 预测下一状态
    void predict();
    
    // 更新状态
    void correct(const std::vector<float>& measurement);
    
    // 获取当前状态
    std::vector<float> getState() const;
    
    // 获取状态元素
    float getStateElement(int index) const;
    
    // 设置状态转移矩阵
    void setTransitionMatrix(const std::vector<std::vector<float>>& matrix);
    
    // 设置测量矩阵
    void setMeasurementMatrix(const std::vector<std::vector<float>>& matrix);
    
    // 设置过程噪声协方差
    void setProcessNoiseCov(const std::vector<std::vector<float>>& matrix);
    
    // 设置测量噪声协方差
    void setMeasurementNoiseCov(const std::vector<std::vector<float>>& matrix);
    
    // 设置后验误差协方差
    void setErrorCovPost(const std::vector<std::vector<float>>& matrix);
    
    // 设置初始状态
    void setStatePost(const std::vector<float>& state);

private:
    // 状态向量和协方差矩阵
    std::vector<float> statePost;              // 后验状态
    std::vector<float> statePre;               // 先验状态
    std::vector<std::vector<float>> transitionMatrix;    // 状态转移矩阵
    std::vector<std::vector<float>> measurementMatrix;   // 测量矩阵
    std::vector<std::vector<float>> processNoiseCov;     // 过程噪声协方差
    std::vector<std::vector<float>> measurementNoiseCov; // 测量噪声协方差
    std::vector<std::vector<float>> errorCovPost;        // 后验误差协方差
    std::vector<std::vector<float>> errorCovPre;         // 先验误差协方差
    
    int stateSize;   // 状态向量维度
    int measureSize; // 测量向量维度
    bool initialized; // 是否已初始化
    
    // 矩阵运算辅助函数
    std::vector<std::vector<float>> multiplyMatrix(
        const std::vector<std::vector<float>>& A, 
        const std::vector<std::vector<float>>& B);
        
    std::vector<float> multiplyMatrixVector(
        const std::vector<std::vector<float>>& A, 
        const std::vector<float>& v);
        
    std::vector<std::vector<float>> transposeMatrix(
        const std::vector<std::vector<float>>& A);
        
    std::vector<std::vector<float>> addMatrix(
        const std::vector<std::vector<float>>& A, 
        const std::vector<std::vector<float>>& B);
        
    std::vector<std::vector<float>> subtractMatrix(
        const std::vector<std::vector<float>>& A, 
        const std::vector<std::vector<float>>& B);
        
    std::vector<std::vector<float>> inverseMatrix(
        const std::vector<std::vector<float>>& A);
};

#endif // __KALMAN_H__