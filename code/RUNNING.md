# Assignment 2 实现与运行说明

## 当前状态

基础作业的模型代码已实现，尚未完成正式数据集训练和评估。

- `cls_libs/modeling.py`：保留 SimpleNet 基线；新增 CompactCNN；完成 Attention、TransformerBlock 和 SimpleViT。
- `det_libs/modeling.py`：完成 FCOS 的目标匹配、三个损失、解码和分类别 NMS。
- `tests/test_models.py`：13 项合成输入数值测试。
- 自设计 CNN、ViT 均使用随机初始化。ResNet-18 微调和检测 backbone 使用 starter 允许的 ImageNet 权重。

配置是待验证的初始实验设置，不能把作业中的参考准确率写成自己的实验结果。

## 环境与测试

在云端先准备匹配的 CUDA、PyTorch 和 torchvision 环境，再安装 `requirements.txt` 中其余依赖。
安装命令从 `code/` 运行：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

只测试核心模型时，可以不加载数据集和 COCO 评估依赖：

```bash
python tests/run_core_tests.py
```

后一个入口跳过两个包的 `__init__`，加载真实模型和辅助函数；没有替换注意力、损失或 NMS。
它不能证明训练入口、数据加载和 COCO 评估已经可用。

默认配置使用 `cuda:0`。小规模 CPU 调试可复制一份 YAML，加入：

```yaml
devices: [cpu]
dtype: fp32
loader:
  batch_size: 2
  num_workers: 0
```

## 数据准备

在 release 根目录执行提供的下载脚本（需要 `gdown` 和 `wget`）：

```bash
sh download_cls_dataset.sh
sh download_det_dataset.sh
```

MiniPlaces 需要 `data/images/`、`data/train.txt` 和 `data/val.txt`。
VOC 需要 `data/VOCdevkit/VOC2007/JPEGImages/`，release 已附 `trainval.json`、`test.json`。
截至本次检查，本地尚无这两个数据集的图片。

## 分类实验

以下命令从 `code/` 执行。在云端用 tmux 保持训练会话。

```bash
python train_cls.py configs/cls_miniplaces_baseline.yaml
python train_cls.py configs/cls_miniplaces_cnn.yaml
python train_cls.py configs/cls_miniplaces_resnet18.yaml
python train_cls.py configs/cls_miniplaces_vit.yaml
```

| 配置 | 网络 | 初始训练安排（含 warmup） |
|---|---|---|
| baseline | 原始 SimpleNet | 默认 60 + 1 epochs |
| cnn | 四阶段残差 CNN，通道 32/64/128/256，每阶段两个 block | SGD，batch 128，lr 0.05，60 + 1 epochs |
| resnet18 | ImageNet 预训练 ResNet-18 微调 | 原始配置，20 + 1 epochs |
| vit | patch 8，维度 192，6 层、6 heads；4 层局部注意力、2 层全局注意力 | AdamW，batch 128，lr 0.0005，80 + 5 epochs |

CNN 约 282 万参数，ViT 约 278 万参数。精度、耗时及是否需要调参，均待真实实验验证。
现在 `cls_miniplaces_cnn.yaml` 对应自设计 CNN，原始基线使用新增的 `cls_miniplaces_baseline.yaml`。

训练后用对应配置评估，例如将 `<experiment>` 替换为真实实验文件夹：

```bash
python eval_cls.py configs/cls_miniplaces_cnn.yaml ../logs/<experiment>/
tensorboard --logdir=../logs
```

最终评估默认启用水平翻转平均；训练期间验证未启用。报告应注明这种区别。
原始训练引擎只在打印间隔采样训练指标，验证指标覆盖整个验证集。

## 检测实验

先验证提供的预训练模型。实际文件名与 PDF 示例不同，使用以下命令：

```bash
python eval_det.py ../pretrained/det_voc_res18.yaml ../pretrained/det_voc_res18.pth.tar -v
```

作业参考 mAP@0.5 约为 60.9%，本地尚未运行这一完整评估。
确认推理结果后，再训练自己的模型：

```bash
python train_det.py configs/det_voc_fcos.yaml
python eval_det.py configs/det_voc_fcos.yaml ../logs/<experiment>/ -v
```

重要实现约定：PointGenerator 存储 `(y, x)`；框使用 `(x1, y1, x2, y2)`；训练类别从 0 开始、推理输出加 1。
匹配使用中心采样、层级回归范围和最小面积优先。分类、GIoU、centerness 损失使用 starter 的 EMA 前景数量归一化。
推理分数为 `sqrt(sigmoid(class) * sigmoid(centerness))`。

`-v` 使用 starter 的可视化代码，环境需要能找到 `FreeMono.ttf`。
检测配置的默认训练安排为 10 + 1 个 warmup epoch；PDF 的时间参考不是本机测量值。

## 报告与提交前待办

1. 运行上述 4 个分类实验，保存曲线及 Top-1/Top-5，比较三个 CNN 方案。
2. 验证预训练 FCOS 的 mAP、推理时间和检测示例。
3. 训练自己的 FCOS，记录损失曲线、mAP、耗时和检测示例。
4. 在 `writeup/` 写 PDF 报告，补充实际成员姓名和分工。
5. 将自设计 CNN、ViT、FCOS 的 YAML、训练日志和最终 checkpoint 复制到各自的 `results/` 子目录；不复制中间 checkpoint。
6. 在 release 根目录运行 `python zip_submission.py`，检查 ZIP 只有要求的 `code/`、`writeup/`、`results/` 内容后提交。

目前未生成最终报告、正式结果或提交 ZIP。Swin 和 COCO bonus 尚未实现。
