# 网站肾脏教学模型

此目录包含 `02_main` 原有教学 SmallUNet 权重和历史训练/评估记录，GitHub 分发版明确保留这两个文件。它不是 `backend/kidney` 中研究用的 ResNet18/U-Net。

- 模型：`kidney_unet_teaching_v1`，单通道 128×128，背景/肾脏/肿瘤三类。
- 12 个 epoch，验证集选择 epoch 10。
- 历史测试患者均值：肾脏 Dice 0.8543，肿瘤 Dice 0.4006。
- 权重 SHA-256：`a312727471e2f3359e2be4df7672477ac4d3468124df87c4463d46311bd8ef39`。
- 本次保留权重原字节，未重新训练；API 预测实际运行此权重。

网页训练页读取记录；离线新实验可使用 `backend/train_kidney_model.py --output artifacts/kidney/experiment-new.pt`。新实验默认不进入 Git，不应覆盖这个基线。

CT 数据许可与来源见 `data/kidney/LICENSE_DATA.md`。本模型仅限课堂演示，非临床工具。
