深度学习框架目标检测算法如何使用YOLOV8模型训练工业传送带异物及问题检测数据集 检测传送带中出现的['堵塞', '裂缝', '异物', '孔洞

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/253bc9de42c843ab826561640a88910e.png)

工业传送带问题检测数据集
4类: ['堵塞', '裂缝', '异物', '孔洞']
共计图片数量:
训练集 1860张,
验证集 318张,
测试集 167张
标注文件格式: YOLO的txt格式




---

## 📊 传送带问题检测数据集概览

| 项目 | 说明 |
| :--- | :--- |
| **任务类型** | 工业缺陷检测（目标检测） |
| **图像总数** | 2,345 张（训练 1860 + 验证 318 + 测试 167） |
| **缺陷类别** | 4 类：<br>• `堵塞`（blockage）<br>• `裂缝`（crack）<br>• `异物`（foreign_object）<br>• `孔洞`（hole） |
| **标注格式** | YOLO `.txt` 格式（每图一个文件，每行：`class_id x_center y_center w h`，归一化） |
| **图像格式** | `.jpg` / `.png`（建议统一为 `.jpg`） |
| **适用场景** | 智能工厂、自动化产线、设备巡检、工业AI质检 |
| **推荐模型** | **YOLOv8n / YOLOv8s**（轻量级，适合实时检测） |

---

## 📂 数据集目录结构

```bash
conveyor_dataset/
│
├── images/
│   ├── train/          # 1860 张
│   ├── val/            # 318 张
│   └── test/           # 167 张
│
├── labels/
│   ├── train/          # 对应 YOLO .txt 标注文件
│   ├── val/
│   └── test/
│
├── data.yaml           # YOLO 配置文件
└── README.md
```

> ✅ 确保 `images` 和 `labels` 中的文件名一一对应（如 `img_001.jpg` ↔ `img_001.txt`）。

---

## 📄 `data.yaml` 配置文件

```yaml
# data.yaml
train: ./conveyor_dataset/images/train
val: ./conveyor_dataset/images/val
test: ./conveyor_dataset/images/test

# 类别数量
nc: 4

# 类别名称（顺序必须与标注一致）
names: ['blockage', 'crack', 'foreign_object', 'hole']
```

> 💡 建议使用英文命名，便于模型训练与日志输出。

---

## 🚀 YOLOv8 详细训练代码

```python
# train_conveyor.py
from ultralytics import YOLO
import torch
import os

# ==================== 配置参数 ====================
MODEL_TYPE = 'yolov8s.pt'           # 可选: yolov8n.pt (更快), yolov8m.pt (更准)
DATA_CONFIG = 'conveyor_dataset/data.yaml'
EPOCHS = 150
BATCH_SIZE = 32                     # 根据 GPU 显存调整（16/32）
IMG_SIZE = 640
PROJECT_NAME = 'conveyor_defect_detection'
EXPERIMENT_NAME = 'yolov8s_640_conveyor'

# 创建项目目录
os.makedirs(f"runs/train/{EXPERIMENT_NAME}", exist_ok=True)

# ==================== 设备检查 ====================
DEVICE = '0' if torch.cuda.is_available() else 'cpu'
print(f"✅ 使用设备: {DEVICE}")

# ==================== 加载模型 ====================
model = YOLO(MODEL_TYPE)  # 自动下载预训练权重

# ==================== 开始训练 ====================
results = model.train(
    data=DATA_CONFIG,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    project=PROJECT_NAME,
    name=EXPERIMENT_NAME,
    save=True,
    save_period=10,           # 每10轮保存一次
    device=DEVICE,
    workers=4,                # 数据加载线程数
    optimizer='AdamW',        # 推荐 AdamW 提升稳定性
    lr0=1e-3,                 # 初始学习率
    lrf=1e-2,                 # 最终学习率 = lr0 * lrf
    momentum=0.937,
    weight_decay=1e-4,
    warmup_epochs=3.0,        # 学习率预热
    box=7.5,                  # 检测框损失权重
    cls=0.5,                  # 分类损失权重
    dfl=1.5,                  # 分布焦点损失
    close_mosaic=10,          # 最后10轮关闭 Mosaic 增强
    augment=True,             # 启用数据增强
    hsv_h=0.015,              # 色调扰动
    hsv_s=0.7,                # 饱和度扰动
    hsv_v=0.4,                # 明度扰动
    degrees=10.0,             # 随机旋转
    translate=0.2,            # 平移
    scale=0.5,                # 缩放
    shear=2.0,                # 剪切
    flipud=0.0,               # 上下翻转
    fliplr=0.5,               # 左右翻转
    mosaic=1.0,               # Mosaic 增强
    mixup=0.1,                # MixUp 增强
    copy_paste=0.1,           # Copy-Paste 增强（对小缺陷如“孔洞”有效）
    patience=30,              # 早停：30轮无提升则停止
)

print("🎉 训练完成！")
print(f"📌 模型保存路径: runs/train/{EXPERIMENT_NAME}")
print(f"📈 最佳 mAP@0.5: {results.best_box_map:.4f}")
```

---

## 🔍 模型验证与评估

```python
# validate.py
from ultralytics import YOLO

# 加载最佳模型
model = YOLO(f'runs/train/{EXPERIMENT_NAME}/weights/best.pt')

# 在验证集上评估
metrics = model.val(
    data=DATA_CONFIG,
    split='val',
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    save_json=True,           # 生成 COCO 风格 results.json
    conf=0.001,               # 低置信度用于完整评估
    iou=0.5,
)

# 输出关键指标
print(f"📊 mAP@0.5: {metrics.box.map:.4f}")
print(f"📊 mAP@0.5:0.95: {metrics.box.map50:.4f}")
print(f"📊 Precision: {metrics.box.p:.4f}")
print(f"📊 Recall: {metrics.box.r:.4f}")

# 按类别输出 AP
class_names = ['blockage', 'crack', 'foreign_object', 'hole']
for i, name in enumerate(class_names):
    print(f"  ├─ {name}: AP@0.5 = {metrics.box.ap[i]:.4f}")
```

---

## 🔦 推理检测代码（支持图片、视频、摄像头）

```python
# predict.py
from ultralytics import YOLO
import cv2

# ==================== 配置 ====================
MODEL_PATH = f'runs/train/{EXPERIMENT_NAME}/weights/best.pt'
SOURCE = 'conveyor_dataset/images/test'  # 可为: 图片路径、视频、目录、摄像头ID (0)
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
SHOW = True          # 是否实时显示（视频/摄像头）
SAVE = True          # 是否保存结果
PROJECT = 'runs/predict/conveyor_result'

# ==================== 加载模型 ====================
model = YOLO(MODEL_PATH)

# ==================== 推理 ====================
results = model.predict(
    source=SOURCE,
    imgsz=IMG_SIZE,
    conf=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    device=DEVICE,
    show=SHOW,
    save=SAVE,
    project=PROJECT,
    name='result',
    exist_ok=True,
    classes=None,     # 检测所有类别
    line_width=2,
    verbose=True,
    max_det=300,      # 单图最大检测数
)

# ==================== 遍历结果并打印 ====================
for r in results:
    boxes = r.boxes
    print(f"🖼️ 图像: {r.path}")
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        print(f"  ├─ {label}: {conf:.2f}")

print("✅ 推理完成！结果已保存。")
```

---

## 🎥 视频推理示例（单独脚本）

```python
# predict_video.py
from ultralytics import YOLO
import cv2

model = YOLO('runs/train/yolov8s_640_conveyor/weights/best.pt')
video_path = 'test_conveyor.mp4'  # 或 0 表示摄像头
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 推理
    results = model(frame, imgsz=640, conf=0.3)

    # 绘制结果
    annotated_frame = results[0].plot()

    # 显示
    cv2.imshow('Conveyor Defect Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## ✅ 训练优化建议

| 技巧 | 说明 |
|------|------|
| **Copy-Paste Augmentation** | 将“异物”、“孔洞”粘贴到正常区域，增强小目标检测能力 |
| **高分辨率训练（1280）** | 若缺陷较小，可尝试 `imgsz=1280` 提升检出率 |
| **Focal Loss** | 若类别不平衡（如“堵塞”样本少），可在 `model.yaml` 中修改损失函数 |
| **TTA（Test-Time Augmentation）** | 推理时启用多尺度增强提升召回率 |
| **模型蒸馏** | 训练 `yolov8m` 作为教师模型，蒸馏到 `yolov8n` 用于边缘部署 |

---


