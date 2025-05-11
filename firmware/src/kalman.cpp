#include "kalman.h"
#include <cmath>
#include <stdexcept>
#include "esp_log.h"

static const char* TAG = "KalmanFilter";

// 默认构造函数
SimpleKalmanFilter::SimpleKalmanFilter() 
    : stateSize(0), measureSize(0), initialized(false) {
}

// 完整构造函数
SimpleKalmanFilter::SimpleKalmanFilter(int stateSize, int measureSize) 
    : stateSize(stateSize), measureSize(measureSize), initialized(true) {
    
    init(stateSize, measureSize);
}

// 析构函数
SimpleKalmanFilter::~SimpleKalmanFilter() {
    // 不需要特殊清理
}

// 初始化函数
void SimpleKalmanFilter::init(int stateSize, int measureSize) {
    this->stateSize = stateSize;
    this->measureSize = measureSize;
    
    // 初始化所有矩阵
    statePost.resize(stateSize, 0.0f);
    statePre.resize(stateSize, 0.0f);
    
    // 创建矩阵并初始化为0
    transitionMatrix.resize(stateSize, std::vector<float>(stateSize, 0.0f));
    measurementMatrix.resize(measureSize, std::vector<float>(stateSize, 0.0f));
    processNoiseCov.resize(stateSize, std::vector<float>(stateSize, 0.0f));
    measurementNoiseCov.resize(measureSize, std::vector<float>(measureSize, 0.0f));
    errorCovPost.resize(stateSize, std::vector<float>(stateSize, 0.0f));
    errorCovPre.resize(stateSize, std::vector<float>(stateSize, 0.0f));
    
    // 设置为单位矩阵
    for (int i = 0; i < stateSize; i++) {
        transitionMatrix[i][i] = 1.0f;
        errorCovPost[i][i] = 1.0f;
    }
    
    for (int i = 0; i < measureSize; i++) {
        if (i < stateSize) {
            measurementMatrix[i][i] = 1.0f;
        }
        measurementNoiseCov[i][i] = 1.0f;
    }
    
    initialized = true;
}

// 预测下一状态
void SimpleKalmanFilter::predict() {
    if (!initialized) {
        ESP_LOGE(TAG, "Kalman filter not initialized!");
        return;
    }
    
    // 预测状态 x'(k) = F * x(k-1)
    statePre = multiplyMatrixVector(transitionMatrix, statePost);
    
    // 预测误差协方差 P'(k) = F * P(k-1) * F' + Q
    auto Ft = transposeMatrix(transitionMatrix);
    auto temp = multiplyMatrix(transitionMatrix, errorCovPost);
    temp = multiplyMatrix(temp, Ft);
    errorCovPre = addMatrix(temp, processNoiseCov);
}

// 更新状态
void SimpleKalmanFilter::correct(const std::vector<float>& measurement) {
    if (!initialized) {
        ESP_LOGE(TAG, "Kalman filter not initialized!");
        return;
    }
    
    if (measurement.size() != measureSize) {
        ESP_LOGE(TAG, "Measurement size mismatch! Expected: %d, Got: %d", 
                measureSize, (int)measurement.size());
        return;
    }
    
    // 计算卡尔曼增益 K(k) = P'(k) * H' * [H * P'(k) * H' + R]^-1
    auto Ht = transposeMatrix(measurementMatrix);
    auto PHt = multiplyMatrix(errorCovPre, Ht);
    auto HPHt = multiplyMatrix(measurementMatrix, PHt);
    auto S = addMatrix(HPHt, measurementNoiseCov);
    auto Sinv = inverseMatrix(S);
    auto K = multiplyMatrix(PHt, Sinv);
    
    // 更新状态 x(k) = x'(k) + K(k) * [z(k) - H * x'(k)]
    auto Hx = multiplyMatrixVector(measurementMatrix, statePre);
    
    // 计算残差 [z(k) - H * x'(k)]
    std::vector<float> residual(measureSize);
    for (int i = 0; i < measureSize; i++) {
        residual[i] = measurement[i] - Hx[i];
    }
    
    // 计算 K(k) * residual
    std::vector<float> Kres(stateSize, 0.0f);
    for (int i = 0; i < stateSize; i++) {
        for (int j = 0; j < measureSize; j++) {
            Kres[i] += K[i][j] * residual[j];
        }
    }
    
    // 更新状态
    for (int i = 0; i < stateSize; i++) {
        statePost[i] = statePre[i] + Kres[i];
    }
    
    // 更新状态协方差 P(k) = (I - K(k) * H) * P'(k)
    std::vector<std::vector<float>> I(stateSize, std::vector<float>(stateSize, 0.0f));
    for (int i = 0; i < stateSize; i++) {
        I[i][i] = 1.0f;
    }
    
    auto KH = multiplyMatrix(K, measurementMatrix);
    auto IKH = subtractMatrix(I, KH);
    errorCovPost = multiplyMatrix(IKH, errorCovPre);
}

// 获取当前状态
std::vector<float> SimpleKalmanFilter::getState() const {
    return statePost;
}

// 获取状态的特定元素
float SimpleKalmanFilter::getStateElement(int index) const {
    if (index >= 0 && index < (int)statePost.size()) {
        return statePost[index];
    }
    return 0.0f;
}

// 设置状态转移矩阵
void SimpleKalmanFilter::setTransitionMatrix(const std::vector<std::vector<float>>& matrix) {
    if (!initialized || matrix.size() != stateSize || matrix[0].size() != stateSize) {
        ESP_LOGE(TAG, "Invalid transition matrix size!");
        return;
    }
    transitionMatrix = matrix;
}

// 设置测量矩阵
void SimpleKalmanFilter::setMeasurementMatrix(const std::vector<std::vector<float>>& matrix) {
    if (!initialized || matrix.size() != measureSize || matrix[0].size() != stateSize) {
        ESP_LOGE(TAG, "Invalid measurement matrix size!");
        return;
    }
    measurementMatrix = matrix;
}

// 设置过程噪声协方差
void SimpleKalmanFilter::setProcessNoiseCov(const std::vector<std::vector<float>>& matrix) {
    if (!initialized || matrix.size() != stateSize || matrix[0].size() != stateSize) {
        ESP_LOGE(TAG, "Invalid process noise covariance matrix size!");
        return;
    }
    processNoiseCov = matrix;
}

// 设置测量噪声协方差
void SimpleKalmanFilter::setMeasurementNoiseCov(const std::vector<std::vector<float>>& matrix) {
    if (!initialized || matrix.size() != measureSize || matrix[0].size() != measureSize) {
        ESP_LOGE(TAG, "Invalid measurement noise covariance matrix size!");
        return;
    }
    measurementNoiseCov = matrix;
}

// 设置后验误差协方差
void SimpleKalmanFilter::setErrorCovPost(const std::vector<std::vector<float>>& matrix) {
    if (!initialized || matrix.size() != stateSize || matrix[0].size() != stateSize) {
        ESP_LOGE(TAG, "Invalid error covariance matrix size!");
        return;
    }
    errorCovPost = matrix;
}

// 设置初始状态
void SimpleKalmanFilter::setStatePost(const std::vector<float>& state) {
    if (!initialized || state.size() != stateSize) {
        ESP_LOGE(TAG, "Invalid state vector size!");
        return;
    }
    statePost = state;
}

// 矩阵乘法
std::vector<std::vector<float>> SimpleKalmanFilter::multiplyMatrix(
    const std::vector<std::vector<float>>& A, 
    const std::vector<std::vector<float>>& B) {
    
    if (A.empty() || B.empty() || A[0].size() != B.size()) {
        ESP_LOGE(TAG, "Invalid matrix dimensions for multiplication!");
        return std::vector<std::vector<float>>();
    }
    
    int m = A.size();
    int n = B[0].size();
    int p = A[0].size();
    
    std::vector<std::vector<float>> C(m, std::vector<float>(n, 0.0f));
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < p; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    return C;
}

// 矩阵向量乘法
std::vector<float> SimpleKalmanFilter::multiplyMatrixVector(
    const std::vector<std::vector<float>>& A, 
    const std::vector<float>& v) {
    
    if (A.empty() || v.empty() || A[0].size() != v.size()) {
        ESP_LOGE(TAG, "Invalid dimensions for matrix-vector multiplication!");
        return std::vector<float>();
    }
    
    int m = A.size();
    int n = v.size();
    
    std::vector<float> result(m, 0.0f);
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            result[i] += A[i][j] * v[j];
        }
    }
    
    return result;
}

// 矩阵转置
std::vector<std::vector<float>> SimpleKalmanFilter::transposeMatrix(
    const std::vector<std::vector<float>>& A) {
    
    if (A.empty()) {
        ESP_LOGE(TAG, "Cannot transpose empty matrix!");
        return std::vector<std::vector<float>>();
    }
    
    int m = A.size();
    int n = A[0].size();
    
    std::vector<std::vector<float>> AT(n, std::vector<float>(m, 0.0f));
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            AT[j][i] = A[i][j];
        }
    }
    
    return AT;
}

// 矩阵加法
std::vector<std::vector<float>> SimpleKalmanFilter::addMatrix(
    const std::vector<std::vector<float>>& A, 
    const std::vector<std::vector<float>>& B) {
    
    if (A.empty() || B.empty() || A.size() != B.size() || A[0].size() != B[0].size()) {
        ESP_LOGE(TAG, "Invalid matrix dimensions for addition!");
        return std::vector<std::vector<float>>();
    }
    
    int m = A.size();
    int n = A[0].size();
    
    std::vector<std::vector<float>> C(m, std::vector<float>(n, 0.0f));
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    
    return C;
}

// 矩阵减法
std::vector<std::vector<float>> SimpleKalmanFilter::subtractMatrix(
    const std::vector<std::vector<float>>& A, 
    const std::vector<std::vector<float>>& B) {
    
    if (A.empty() || B.empty() || A.size() != B.size() || A[0].size() != B[0].size()) {
        ESP_LOGE(TAG, "Invalid matrix dimensions for subtraction!");
        return std::vector<std::vector<float>>();
    }
    
    int m = A.size();
    int n = A[0].size();
    
    std::vector<std::vector<float>> C(m, std::vector<float>(n, 0.0f));
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    
    return C;
}
 
// 矩阵求逆
std::vector<std::vector<float>> SimpleKalmanFilter::inverseMatrix(
    const std::vector<std::vector<float>>& A) {
    
    if (A.empty() || A.size() != A[0].size()) {
        ESP_LOGE(TAG, "Matrix must be square for inversion!");
        return std::vector<std::vector<float>>();
    }
    
    int n = A.size();
    
    // 1x1矩阵求逆 - 标量的倒数
    if (n == 1) {
        if (fabs(A[0][0]) < 1e-10) {
            ESP_LOGE(TAG, "Matrix is singular, cannot invert!");
            return std::vector<std::vector<float>>(1, std::vector<float>(1, 0.0f));
        }
        std::vector<std::vector<float>> inv(1, std::vector<float>(1, 1.0f / A[0][0]));
        return inv;
    }
    
    // 2x2矩阵求逆 - 使用解析公式
    else if (n == 2) {
        float det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
        if (fabs(det) < 1e-10) {
            ESP_LOGE(TAG, "Matrix is singular, cannot invert!");
            return std::vector<std::vector<float>>(n, std::vector<float>(n, 0.0f));
        }
        
        std::vector<std::vector<float>> inv(2, std::vector<float>(2, 0.0f));
        inv[0][0] = A[1][1] / det;
        inv[0][1] = -A[0][1] / det;
        inv[1][0] = -A[1][0] / det;
        inv[1][1] = A[0][0] / det;
        
        return inv;
    }
    
    // 3x3矩阵求逆 - 伴随矩阵法
    else if (n == 3) {
        float det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
                  - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                  + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
                  
        if (fabs(det) < 1e-10) {
            ESP_LOGE(TAG, "Matrix is singular, cannot invert!");
            return std::vector<std::vector<float>>(n, std::vector<float>(n, 0.0f));
        }
        
        std::vector<std::vector<float>> inv(3, std::vector<float>(3, 0.0f));
        
        inv[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) / det;
        inv[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) / det;
        inv[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) / det;
        inv[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) / det;
        inv[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) / det;
        inv[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) / det;
        inv[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) / det;
        inv[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) / det;
        inv[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) / det;
        
        return inv;
    }
    
    // 4x4矩阵求逆 - 完整实现
    else if (n == 4) {
        ESP_LOGI(TAG, "Computing 4x4 matrix inverse");
        
        // 计算全部16个代数余子式
        float C00 = A[1][1]*(A[2][2]*A[3][3]-A[2][3]*A[3][2]) - 
                    A[1][2]*(A[2][1]*A[3][3]-A[2][3]*A[3][1]) + 
                    A[1][3]*(A[2][1]*A[3][2]-A[2][2]*A[3][1]);
                    
        float C01 = -(A[1][0]*(A[2][2]*A[3][3]-A[2][3]*A[3][2]) - 
                     A[1][2]*(A[2][0]*A[3][3]-A[2][3]*A[3][0]) + 
                     A[1][3]*(A[2][0]*A[3][2]-A[2][2]*A[3][0]));
                     
        float C02 = A[1][0]*(A[2][1]*A[3][3]-A[2][3]*A[3][1]) - 
                    A[1][1]*(A[2][0]*A[3][3]-A[2][3]*A[3][0]) + 
                    A[1][3]*(A[2][0]*A[3][1]-A[2][1]*A[3][0]);
                    
        float C03 = -(A[1][0]*(A[2][1]*A[3][2]-A[2][2]*A[3][1]) - 
                     A[1][1]*(A[2][0]*A[3][2]-A[2][2]*A[3][0]) + 
                     A[1][2]*(A[2][0]*A[3][1]-A[2][1]*A[3][0]));
                     
        float C10 = -(A[0][1]*(A[2][2]*A[3][3]-A[2][3]*A[3][2]) - 
                     A[0][2]*(A[2][1]*A[3][3]-A[2][3]*A[3][1]) + 
                     A[0][3]*(A[2][1]*A[3][2]-A[2][2]*A[3][1]));
                     
        float C11 = A[0][0]*(A[2][2]*A[3][3]-A[2][3]*A[3][2]) - 
                    A[0][2]*(A[2][0]*A[3][3]-A[2][3]*A[3][0]) + 
                    A[0][3]*(A[2][0]*A[3][2]-A[2][2]*A[3][0]);
                    
        float C12 = -(A[0][0]*(A[2][1]*A[3][3]-A[2][3]*A[3][1]) - 
                     A[0][1]*(A[2][0]*A[3][3]-A[2][3]*A[3][0]) + 
                     A[0][3]*(A[2][0]*A[3][1]-A[2][1]*A[3][0]));
                     
        float C13 = A[0][0]*(A[2][1]*A[3][2]-A[2][2]*A[3][1]) - 
                    A[0][1]*(A[2][0]*A[3][2]-A[2][2]*A[3][0]) + 
                    A[0][2]*(A[2][0]*A[3][1]-A[2][1]*A[3][0]);
                    
        float C20 = A[0][1]*(A[1][2]*A[3][3]-A[1][3]*A[3][2]) - 
                    A[0][2]*(A[1][1]*A[3][3]-A[1][3]*A[3][1]) + 
                    A[0][3]*(A[1][1]*A[3][2]-A[1][2]*A[3][1]);
                    
        float C21 = -(A[0][0]*(A[1][2]*A[3][3]-A[1][3]*A[3][2]) - 
                     A[0][2]*(A[1][0]*A[3][3]-A[1][3]*A[3][0]) + 
                     A[0][3]*(A[1][0]*A[3][2]-A[1][2]*A[3][0]));
                     
        float C22 = A[0][0]*(A[1][1]*A[3][3]-A[1][3]*A[3][1]) - 
                    A[0][1]*(A[1][0]*A[3][3]-A[1][3]*A[3][0]) + 
                    A[0][3]*(A[1][0]*A[3][1]-A[1][1]*A[3][0]);
                    
        float C23 = -(A[0][0]*(A[1][1]*A[3][2]-A[1][2]*A[3][1]) - 
                     A[0][1]*(A[1][0]*A[3][2]-A[1][2]*A[3][0]) + 
                     A[0][2]*(A[1][0]*A[3][1]-A[1][1]*A[3][0]));
                     
        float C30 = -(A[0][1]*(A[1][2]*A[2][3]-A[1][3]*A[2][2]) - 
                     A[0][2]*(A[1][1]*A[2][3]-A[1][3]*A[2][1]) + 
                     A[0][3]*(A[1][1]*A[2][2]-A[1][2]*A[2][1]));
                     
        float C31 = A[0][0]*(A[1][2]*A[2][3]-A[1][3]*A[2][2]) - 
                    A[0][2]*(A[1][0]*A[2][3]-A[1][3]*A[2][0]) + 
                    A[0][3]*(A[1][0]*A[2][2]-A[1][2]*A[2][0]);
                    
        float C32 = -(A[0][0]*(A[1][1]*A[2][3]-A[1][3]*A[2][1]) - 
                     A[0][1]*(A[1][0]*A[2][3]-A[1][3]*A[2][0]) + 
                     A[0][3]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]));
                     
        float C33 = A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1]) - 
                    A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0]) + 
                    A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
        
        // 计算行列式
        float det = A[0][0]*C00 + A[0][1]*C01 + A[0][2]*C02 + A[0][3]*C03;
        
        if (fabs(det) < 1e-10) {
            ESP_LOGE(TAG, "4x4 Matrix is singular, cannot invert! Det=%f", det);
            return std::vector<std::vector<float>>(n, std::vector<float>(n, 0.0f));
        }
        
        // 创建结果矩阵
        std::vector<std::vector<float>> inv(4, std::vector<float>(4, 0.0f));
        
        // 将伴随矩阵除以行列式得到逆矩阵
        float invDet = 1.0f / det;
        
        // 注意：根据伴随矩阵的定义，我们需要转置代数余子式矩阵
        inv[0][0] = C00 * invDet;
        inv[0][1] = C10 * invDet;
        inv[0][2] = C20 * invDet;
        inv[0][3] = C30 * invDet;
        
        inv[1][0] = C01 * invDet;
        inv[1][1] = C11 * invDet;
        inv[1][2] = C21 * invDet;
        inv[1][3] = C31 * invDet;
        
        inv[2][0] = C02 * invDet;
        inv[2][1] = C12 * invDet;
        inv[2][2] = C22 * invDet;
        inv[2][3] = C32 * invDet;
        
        inv[3][0] = C03 * invDet;
        inv[3][1] = C13 * invDet;
        inv[3][2] = C23 * invDet;
        inv[3][3] = C33 * invDet;
        
        return inv;
    }
    
    // 对于更大的矩阵，使用高斯-约旦消元法
    else {
        ESP_LOGI(TAG, "Computing %dx%d matrix inverse using Gauss-Jordan elimination", n, n);
        
        // 创建增广矩阵 [A|I]
        std::vector<std::vector<float>> augmented(n, std::vector<float>(2 * n, 0.0f));
        
        // 填充增广矩阵的左半部分为原始矩阵A
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                augmented[i][j] = A[i][j];
            }
        }
        
        // 填充增广矩阵的右半部分为单位矩阵I
        for (int i = 0; i < n; i++) {
            augmented[i][i + n] = 1.0f;
        }
        
        // 高斯消元法将左半部分变成上三角矩阵
        for (int i = 0; i < n; i++) {
            // 找到当前列中绝对值最大的元素的行索引
            int maxRow = i;
            float maxVal = fabs(augmented[i][i]);
            
            for (int k = i + 1; k < n; k++) {
                if (fabs(augmented[k][i]) > maxVal) {
                    maxVal = fabs(augmented[k][i]);
                    maxRow = k;
                }
            }
            
            // 如果最大值接近零，矩阵奇异
            if (maxVal < 1e-10) {
                ESP_LOGE(TAG, "Matrix is singular, cannot invert!");
                return std::vector<std::vector<float>>(n, std::vector<float>(n, 0.0f));
            }
            
            // 如果需要，交换行
            if (maxRow != i) {
                for (int j = 0; j < 2 * n; j++) {
                    std::swap(augmented[i][j], augmented[maxRow][j]);
                }
            }
            
            // 将主元归一化
            float pivot = augmented[i][i];
            for (int j = 0; j < 2 * n; j++) {
                augmented[i][j] /= pivot;
            }
            
            // 消元，将其他行的当前列元素变为0
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    float factor = augmented[k][i];
                    for (int j = 0; j < 2 * n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }
        
        // 从增广矩阵的右半部分提取逆矩阵
        std::vector<std::vector<float>> inv(n, std::vector<float>(n, 0.0f));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inv[i][j] = augmented[i][j + n];
            }
        }
        
        return inv;
    }
}