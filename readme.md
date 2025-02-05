# BiliCoin 目标检测

## 项目概述

本项目是一个基于ONNX Runtime的轻量级目标检测模型，专门用于检测Bilibili视频页面中的交互按钮状态。可识别包括点赞/未点赞、投币/未投币、收藏/未收藏等8种状态。模型采用YOLO11架构实现，兼顾检测精度与推理速度。

## 功能特性

- **多状态检测**：支持8种B站核心交互状态的识别
- **跨平台支持**：兼容CPU/GPU推理，适配多种硬件环境
- **可视化输出**：自动生成带置信度的检测框和类别标签
- **灵活配置**：支持置信度阈值和IoU阈值动态调整
- **非极大抑制**：有效处理重叠检测框，提升检测准确性

## 环境要求


### 软件依赖

```bash
# 安装依赖
pip install onnxruntime opencv-python numpy
```

## 模型文件

| 文件名称          | 大小    | 
|------------------|---------|
| bili_coin.onnx   | 10.0 MB | 
| bili_coin.pt     | 5.22 MB | 


[模型下载地址](https://github.com/danel-phang/BiliCoin/releases/tag/models)


## 快速开始

```python
from bilicoin import BiliCoin

# 初始化检测器
model = BiliCoin(
    onnx_model="models\\bili_coin.onnx",
    cpu=True  # CPU推理
)

# 执行检测
result_img, class_info = model.detect(
    input_image="images\\test.png",
    confidence_thres=0.5,  # 置信度阈值
    iou_thres=0.45         # IoU阈值
)

# 保存结果
cv2.imwrite("images\\output.jpg", result_img)
print("检测结果:", class_info)
```

### 参数说明

#### 初始化BiliCoin类
| 参数        | 类型   | 默认值 | 说明                          |
|-------------|--------|--------|-------------------------------|
| onnx_model  | str    | 必填   | ONNX模型文件路径              |
| cpu         | bool   | True   | 强制使用CPU推理               |

#### detect方法
| 参数             | 类型   | 默认值 | 有效范围    | 说明                     |
|------------------|--------|--------|-------------|--------------------------|
| input_image      | str    | 必填   | -           | 输入图像路径             |
| confidence_thres | float  | 0.5    | [0.01, 0.99]| 置信度过滤阈值           |
| iou_thres        | float  | 0.45   | [0.1, 0.9]  | 非极大抑制IoU阈值        |

### 输出说明

#### 可视化图像
- 输出示例：

<p align="center">
  <img src="https://tvax3.sinaimg.cn/large/008A9mE2gy1hy9v7596loj31fn0t07ag.jpg" alt="output">
</p>

#### 结构化数据
返回字典格式的检测结果：
```python
{
    "like": (0.92, [x, y, w, h]),
    "coin": (0.88, [x, y, w, h]),
    "bookmarke": (0.95, [x, y, w, h])
}
```
- 键：类别名称
- 值：元组(置信度, 边界框坐标)
- 边界框格式：[左上角x, 左上角y, 宽度, 高度]

### 核心指标

| 名称       | 数值    | 说明                          |
|----------------|---------|-------------------------------|
| mAP@0.5        | 0.98   | 平均精度（IoU阈值0.5）        |
| mAP@0.5:0.95   | 0.68   | 平均精度（IoU阈值0.5-0.95）   |
| Precision      | 0.97   | 查准率                        |
| Recall         | 0.97   | 查全率                        |

<p align="center">
  <img src="https://tvax1.sinaimg.cn/large/008A9mE2gy1hy9v6rgw5tj31uo0xcqlt.jpg" alt="核心指标">
</p>

---
