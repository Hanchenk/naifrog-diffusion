# 从零训练实验：论文与实现核对

核对日期：2026-10-16。这里说明本次真实训练代码借用了哪些公式，以及哪些部分属于本项目的工程选择。本文没有把小范围合成数据实验等同于复现论文完整结果。

## 数据、网络和训练范围

训练图像是用户提供的奶蛙三维资产的二维 RGB 渲染，大小 64×64；文字覆盖 3 个视角 × 2 种构图 × 4 种背景，共 24 种条件组合。U-Net、中文字符 embedding、位置 embedding、2 层文本 Transformer、交叉注意力均随机初始化并在本次训练中更新。没有加载 CLIP、T5、Stable Diffusion 或其他预训练参数。

网络输出的是每个 RGB 位置的 3 通道高斯噪声预测。它没有三维输出、物体零件追踪或把真实噪声点搬到某个三维位置的机制；上一版视频中的点云是教学动画，而本实验实际训练的是二维像素空间条件扩散模型。

文本词表由训练 captions 中出现的 Unicode 字符建立，另加 PAD/BOS/EOS/UNK。PAD 不参与文本 self-attention 或图文 cross-attention 的 key/value。虽然输入是文字而不是手写类别 ID，本实验中的文字分布有限，不能据此宣称模型理解通用中文、开放类别或训练分布外提示词。

## 1. 加噪与噪声预测目标：DDPM

Jonathan Ho, Ajay Jain, Pieter Abbeel. *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.

- [原始论文页面](https://arxiv.org/abs/2006.11239) · [PDF](https://arxiv.org/pdf/2006.11239)
- PDF 第 2 页 Eq. (4)：`q(x_t|x_0)=N(sqrt(alpha_bar_t)*x_0,(1-alpha_bar_t)I)`。
- PDF 第 4 页 Algorithm 1；第 5 页 Eq. (14)：简化的 epsilon 预测平方误差目标。

本代码 `DiffusionSchedule.add_noise()` 实现

`x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon`，`epsilon ~ N(0,I)`。

`train.py` 随机抽图片、时间步、Gaussian noise，计算

`L = mean[(epsilon - epsilon_theta(x_t,t,c))²]`。

这里的 mean 包括 batch、通道与像素；它与论文按向量求平方和的表示只差固定归一化。所有网络参数通过反向传播更新。文字 `c` 是本项目增加的条件输入；不学习反向过程方差，不使用论文完整变分下界损失，也不声称复现原论文网络规模、数据量或指标。预览采用下述 DDIM，而不是 DDPM Algorithm 2 的完整 1000 步随机采样。

## 2. 余弦加噪日程：Improved DDPM

Alexander Quinn Nichol, Prafulla Dhariwal. *Improved Denoising Diffusion Probabilistic Models*. ICML 2021, PMLR 139:8162–8171.

- [PMLR 页面](https://proceedings.mlr.press/v139/nichol21a.html) · [PDF](https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf)
- PDF 第 4 页 §3.2、Eq. (16)：`alpha_bar_t=f(t)/f(0)`，`f(t)=cos²(((t/T+s)/(1+s))*pi/2)`；相邻 alpha_bar 比值转换为 beta。
- 同页文字说明：beta 上限 0.999，offset `s=0.008`。

代码用 float64 生成 1001 个边界，再形成 1000 个 beta，截断到 0.999 后重新累乘，并以 float32 保存。训练索引 `i=0..999` 对应论文扩散时间 `t=1..1000`；不是在真实干净图片时刻添加了第一份噪声。

由于 beta 上限，末端 `alpha_bar` 约为 `2.43e-9`，是接近零的正值而非严格为零。这样避免除零；采样从标准高斯开始近似末端分布。极低 SNR 时，epsilon 预测的小误差会在除以 `sqrt(alpha_bar_t)` 时放大，因此采样用 float32 做 alpha/x0 运算并裁剪 x0。裁剪是工程稳定性选择，不能视作消除了预测误差。

只借用该文的 cosine schedule；本实验没有复现其学习反向方差、hybrid likelihood objective 或完整模型配置。

## 3. 条件丢弃与 CFG

Jonathan Ho, Tim Salimans. *Classifier-Free Diffusion Guidance*. arXiv:2207.12598, 2022.

- [论文页面](https://arxiv.org/abs/2207.12598) · [PDF](https://arxiv.org/pdf/2207.12598)
- PDF 第 4 页 Algorithm 1 和 §3.2：训练时随机丢弃条件，同一网络学习有条件与无条件预测。
- PDF 第 5 页 Algorithm 2、Eq. (6)：`epsilon_tilde=(1+w)*epsilon_cond-w*epsilon_uncond`。

本次训练以 15% 概率把文字替换为空串的 BOS/EOS 编码；其 embedding 同样从零学习。采样代码使用等价记号

`epsilon_hat = epsilon_uncond + s*(epsilon_cond-epsilon_uncond)`。

因此本代码 `s=1+w`。`guidance=1` 是普通有条件预测，`guidance=0` 是无条件预测，默认 `2.5` 对应论文的 `w=1.5`。不是把文字内容直接相加到噪声图片上，也不额外训练分类器。尺度 2.5 是本实验可调超参数；不能把它称作论文证明的最优值。视觉质量与条件遵循度需检查实际生成图。

## 4. 确定性快速采样：DDIM

Jiaming Song, Chenlin Meng, Stefano Ermon. *Denoising Diffusion Implicit Models*. ICLR 2021.

- [论文页面](https://arxiv.org/abs/2010.02502) · [PDF](https://arxiv.org/pdf/2010.02502)
- PDF 第 5 页 Eq. (12)：由预测 x0、epsilon 方向项和随机项组成的采样更新；其后讨论 `sigma_t=0` 的确定性过程。
- PDF 第 6 页 Eq. (16)：子序列采样中的 `sigma_tau(eta)`，`eta=0` 为 DDIM。

原 DDIM 文中 `alpha_t` 表示本项目沿用 DDPM 记号中的累计量 `alpha_bar_t`。本实验用 50 个近似等间隔整数时间索引倒序采样，`eta=0`，因此没有每一步新增的随机噪声：

`x0_raw = (x_t - sqrt(1-alpha_bar_t)*epsilon_hat)/sqrt(alpha_bar_t)`

`x0_hat = clamp(x0_raw,-1,1)`

`x_prev = sqrt(alpha_bar_prev)*x0_hat + sqrt(1-alpha_bar_prev)*epsilon_hat`。

最后一步直接输出 `x0_hat`，相当于转到干净时刻 `alpha_bar=1`。初始随机噪声由固定 seed 生成；在同一运行设备/计算设置下，后续 DDIM 步骤是确定性的。跨不同 GPU、PyTorch/CUDA 版本不承诺逐位一致。

**与原 Eq. (12) 的区别：** 本代码裁剪预测 x0 到图像训练范围 [-1,1]，并保留原始 epsilon 方向项；没有根据裁剪后的 x0 重新推导 epsilon。因此应称“带 x0 裁剪的 eta=0 DDIM 实现”，不能宣称逐行复现未裁剪 Eq. (12)。CFG 也将原噪声预测替换为引导后的 epsilon。它们是明确记录的采样选择，不改变训练 MSE 的定义。

## 5. 文本交叉注意力：LDM 中的条件机制

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer. *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR 2022:10684–10695.

- [CVF 页面](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html) · [PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)
- PDF 第 4 页（印刷页 10687）§3.3 中未编号的 attention 公式：`softmax(QK^T/sqrt(d))*V`，图像特征提供 Q，条件编码提供 K/V。
- PDF 第 5 页（印刷页 10688）Eq. (3)：有条件噪声预测损失，并说明条件编码器和去噪网络可以共同优化。

本项目在 U-Net 的 16×16 与 8×8 特征层使用这种交叉注意力；小型中文字符 Transformer 提供文本 token 特征。特征是二维 RGB 像素网络的中间特征，不是预训练 autoencoder 的潜变量。**本项目没有 VAE，没有 CLIP，没有预训练文本编码器，因此它是像素空间的文本条件 DDPM，不是 LDM 或 Stable Diffusion。** 借用交叉注意力机制也不等于复现 LDM 论文的文本模型与全部训练配置。

## 工程选择与验证的含义

AdamW、梯度裁剪、FP16 AMP、GPU uint8 数据缓存、EMA 的早期 decay ramp、学习率 warmup/余弦衰减、固定种子采样和断点文件均为本项目工程配置；没有把它们冒称为上述每篇论文规定的唯一配置。

`verify.py` 通过独立的数值检查验证：加噪可逆公式、固定样本噪声目标能下降、文本参数获得梯度、padding 不泄漏、已知 epsilon 的 oracle DDIM 能恢复目标、checkpoint 保存加载等价、固定 seed 重复生成、CFG=0 时结果与提示词无关。小规模数学/代码检查通过，只能支持实现正确性，不能替代真实模型的图像质量和条件遵循度评价。

固定验证集的 `validation_ema_mse` 使用与训练独立渲染的图像以及固定噪声/时间步，衡量噪声预测目标的验证表现。它不是 FID、CLIP score、人工偏好，也不是所有提示词都能成功的保证。对同一个角色的合成视图学习，仍可能接近记忆模板，应结合新 seed 的实际生成图片、成对提示词控制测试评估结果。

## 原始材料与版权

本文公式位置按照上面链接的一手 PDF 核对。开源仓库保留精确链接，不重新分发论文 PDF。
