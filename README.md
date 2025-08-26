# PyTorch 复现 TensorFlow Notebook (O'Shea 2016 CNN2)

本项目将 RadioML2016.10a 的 **TensorFlow / Keras Notebook** 转换为 **模块化 PyTorch 实现**，在数据准备、模型结构、训练与评估流程上保持对齐，可复现经典 CNN2 结果。

## 特性概览
* 相同数据构建：从 pickle 字典生成 `X`, `lbl`, `mods`, `snrs`
* 固定随机种子 2016，复现实验 80/20 训练 / 测试划分
* CNN2 模型结构对应：
  * ZeroPadding2D((0,2)) → Conv2D(256,(1,3)) → Dropout(0.5)
  * ZeroPadding2D((0,2)) → Conv2D(80,(2,3))  → Dropout(0.5)
  * Flatten → Dense(1024) → Dense(256) → Dropout(0.5) → Dense(num_classes)
* 使用 Adam + CrossEntropy，基于验证集 loss 的 EarlyStopping（patience=5）
* 训练保存最佳模型 `conv.pt`，绘制训练/验证曲线
* 输出整体混淆矩阵与逐 SNR 混淆矩阵
* 统计各 SNR 精度并保存到 `results_cnn.pkl`：内容为 `( "CNN2", dropout, acc_dict )`

## 快速开始
```bash
# 创建虚拟环境 (Linux / macOS)
python -m venv .venv && source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt

# 训练（现在无需写任何参数，使用默认配置）
python -m src.train
# 如需临时覆盖，可用环境变量：
# $Env:EPOCHS=50; $Env:BATCH_SIZE=512; python -m src.train

# 评估（仍需指定模型路径与想要的 dropout，保持与训练一致）
python -m src.eval --ckpt_path runs/conv.pt --batch_size 1024
```

### 自定义训练配置方式
优先级：环境变量 > 配置文件 > 内置默认
1. 配置文件：创建 `config/train_config.json`（与 `src` 同级）例如：
```json
{ "epochs": 120, "batch_size": 512, "lr": 0.0005 }
```
2. 环境变量（PowerShell 示例）：
```powershell
$Env:EPOCHS=80; $Env:BATCH_SIZE=256; $Env:LR=0.0007; python -m src.train
```
3. 内置默认：epochs=100, batch_size=1024, dropout=0.5, lr=1e-3, patience=5, out_dir=runs

## 目录说明
| 文件/目录 | 说明 |
|-----------|------|
| `src/dataio/radioml_loader.py` | 加载 pickle 数据并复现标签与 80/20 划分（内置默认路径） |
| `src/models/cnn2.py` | PyTorch 模型，结构与 TF 版本对应 |
| `src/train.py` | 训练循环、早停、曲线绘制 |
| `src/eval.py` | 评估：整体/逐 SNR 混淆矩阵、SNR 精度统计与结果保存 |
| `src/utils/plots.py` | 绘图工具函数 |
| `scripts/run_train.sh` / `scripts/run_eval.sh` | 示例脚本 |

## 重要说明
* 输入形状处理为 `[B, 1, 2, 128]`（channels-first）。Keras `ZeroPadding2D((0,2))` 对应 PyTorch 中 `nn.ZeroPad2d((2,2,0,0))`。
* 默认不做归一化（与原 Notebook 对齐）。
* CrossEntropyLoss 自带 softmax，因此训练直接用 logits；评估需要概率时再 softmax。

## 常见问题 FAQ
1. 数据文件在哪里？ 位于 `src/dataio/RML2016.10a_dict.pkl`。如需替换，直接替换该文件或在代码里 `load_radioml2016_dict(path=新路径)`。
2. 想恢复命令行传 `--data_path`？ 在 `train.py` / `eval.py` 的 ArgumentParser 中重新添加参数并传递给 `load_radioml2016_dict` 即可。
3. 精度与原结果略有差异？ GPU / 随机性 / 依赖版本可能带来 ±0.x% 波动，可多次运行取平均。

## License
仅供科研与教学使用；原始数据与模型思想版权归各自作者所有。
