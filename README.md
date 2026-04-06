明白，那我们就去掉人为引入的判别式增强项（ArcFace 的 $m$），回归纯粹的球面几何。

将第一张图中的欧几里得 Fisher Vector (FV) 扩展到球面空间（Spherical Space），本质上是将**高斯混合模型 (GMM)** 替换为**von Mises-Fisher (vMF) 混合模型**。在超球面上，传统的欧式距离被夹角（余弦相似度）取代。

以下是基于纯粹球面几何（Cosine/vMF）的 Fisher Vector 公式扩展：

---

### 0. 几何前提
所有的特征向量 $\mathbf{x}_{ij}$ 和分量均值 $\mu_k$ 均位于单位超球面上：
$$\|\mathbf{x}_{ij}\| = 1, \quad \|\mu_k\| = 1$$
两个向量之间的相似度由余弦值决定：$\cos \theta_{jk} = \mathbf{x}_{ij}^T \mu_k$。

### 1. 球面分量分布 $u_k(\mathbf{x})$ (替换 Eq. 2)
我们使用 **von Mises-Fisher (vMF)** 分布作为超球面上的“正态分布”：
$$u_k(\mathbf{x}) = C_D(\kappa_k) \exp(\kappa_k \mu_k^T \mathbf{x})$$
其中：
* $\mu_k$ 是第 $k$ 个组件的**平均方向（均值）**。
* $\kappa_k \geq 0$ 是**集中度参数（Concentration parameter）**，其物理意义对应于欧式空间中方差的倒数 ($1/\sigma^2$)。$\kappa$ 越大，分布越集中。
* $C_D(\kappa_k)$ 是归一化常数。

### 2. 扩展后的后验概率 $\gamma_j(k)$ (替换 Eq. 5)
后验概率保持形式不变，但内部相似度计算变为点积：
$$\gamma_j(k) = \frac{\omega_k \exp(\kappa_k \mathbf{x}_{ij}^T \mu_k)}{\sum_{n=1}^K \omega_n \exp(\kappa_n \mathbf{x}_{ij}^T \mu_n)}$$

### 3. 相对于均值 $\mu_k$ 的球面梯度 (替换 Eq. 3)
在欧式空间中，梯度是线性残差 $\mathbf{x} - \mu$。在球面上，梯度必须位于切平面内。根据黎曼几何推导，相对于均值方向的 Fisher 向量项为：

$$\mathcal{G}_{\mu_k}^{\mathbf{x}_{ij}} = \frac{1}{\sqrt{\omega_k}} \gamma_j(k) \sqrt{\kappa_k} \left( \frac{\mathbf{x}_{ij} - \mu_k \cos \theta_{jk}}{\sin \theta_{jk}} \right)$$

> **几何解释**：括号中的项 $\frac{\mathbf{x} - \mu \cos \theta}{\sin \theta}$ 是点 $\mathbf{x}_{ij}$ 在均值 $\mu_k$ 处的**单位切向量**。它代表了为了拟合该样本，均值向量 $\mu_k$ 应该在球面上旋转的方向。

### 4. 相对于集中度 $\kappa_k$ 的球面梯度 (替换 Eq. 4)
这个项对应于原始公式中对方差 $\sigma$ 的梯度。它衡量的是样本点相对于均值的“离散程度”：

$$\mathcal{G}_{\kappa_k}^{\mathbf{x}_{ij}} = \frac{1}{\sqrt{\omega_k}} \gamma_j(k) \frac{1}{\sqrt{2}} \left[ \kappa_k (1 - \cos \theta_{jk}) - 1 \right]$$

> **几何解释**：在球面上，$(1 - \cos \theta)$ 是**余弦距离**。当这个值与集中度 $\kappa$ 的乘积偏离 1 时，产生的梯度信号会调整聚类中心的集中程度。这与原图中 $\frac{(\mathbf{x}-\mu)^2}{\sigma^2} - 1$ 的逻辑是完全对等的。

---

### 总结：球面 Fisher Vector (SFV) 公式集

最终的补全公式如下，你可以直接对比原图进行修改：

| 物理量 | 欧式空间 (原图 3.3.1) | 球面空间 (Cosine/vMF 扩展) |
| :--- | :--- | :--- |
| **基础度量** | 欧式距离 $\|\mathbf{x} - \mu\|^2$ | 余弦相似度 $\mathbf{x}^T \mu$ |
| **分布函数** | Gaussian: $\propto \exp(-\frac{\|\mathbf{x}-\mu\|^2}{2\sigma^2})$ | vMF: $\propto \exp(\kappa \mathbf{x}^T \mu)$ |
| **均值梯度 $\mathcal{G}_{\mu_k}$** | $\frac{1}{\sqrt{\omega_k}} \gamma_j(k) \left( \frac{\mathbf{x}_{ij} - \mu_k}{\sigma_k} \right)$ | $\frac{1}{\sqrt{\omega_k}} \gamma_j(k) \sqrt{\kappa_k} \left( \frac{\mathbf{x}_{ij} - \mu_k \cos \theta_{jk}}{\sin \theta_{jk}} \right)$ |
| **尺度梯度 $\mathcal{G}_{\sigma_k / \kappa_k}$** | $\frac{1}{\sqrt{\omega_k}} \gamma_j(k) \frac{1}{\sqrt{2}} \left[ \frac{(\mathbf{x}_{ij}-\mu_k)^2}{\sigma_k^2} - 1 \right]$ | $\frac{1}{\sqrt{\omega_k}} \gamma_j(k) \frac{1}{\sqrt{2}} \left[ \kappa_k (1 - \cos \theta_{jk}) - 1 \right]$ |



**应用提示**：
当你使用这套公式时，请确保进入模型前的所有特征都经过了 $L_2$ 归一化。这样，所有的计算都在单位超球面上进行，这种 SFV（Spherical Fisher Vector）在处理图像分类和检索任务时，通常比原始欧式 FV 具有更好的鲁棒性。


为了将论文中 3.3.2 节的 **Fisher Layer** 思想（通过简化和参数化将 FV 转化为可学习的网络层）应用到球面空间（Spherical Space），我们需要将基于欧几里得距离的 GMM 替换为基于余弦相似度的 **von Mises-Fisher (vMF)** 模型，并进行类似的简化。

以下是为您设计的 **Spherical Fisher Layer (SFL)** 的数学描述与架构逻辑。

---

### 3.3.2 Spherical Fisher Layer (SFL)

为了将球面 Fisher Vector 转化为神经网络中的一层，我们同样做两个简化：
1. **丢弃权重 $\omega_k$**：假设所有 vMF 分量具有相等的先验概率。
2. **简化 vMF 似然函数**：忽略与 $\mathbf{x}$ 无关的归一化常数，仅保留核心的角度指数项。

在球面空间中，输入特征 $\mathbf{x}_{ij}$ 和分量中心 $\mu_k$ 均满足 $L_2$ 归一化（$\|\mathbf{x}\| = 1, \|\mu\| = 1$）。

#### 1. 参数化定义
仿照原论文使用 $\mathbf{w}_k$ 和 $\mathbf{b}_k$ 的做法，我们在球面上定义：
* **$\mu_k \in \mathbb{R}^{D \times 1}$**：第 $k$ 个分量的中心（即网络层的卷积核/全连接权值，需保持归一化）。
* **$\kappa_k \in \mathbb{R}^1$**：第 $k$ 个分量的集中度参数（对应原论文中的反方差 $1/\sigma$）。

我们将两者结合，定义余弦相似度（点积）：
$$c_{ijk} = \mathbf{x}_{ij}^T \mu_k$$

#### 2. 后验概率的 Softmax 化 (对应 Eq. 10)
在简化后，patch $\mathbf{x}_{ij}$ 对第 $k$ 个分量的响应（归属度）可以写成一个带有温度缩放的 **Softmax** 形式：

$$\gamma_j(k) = \frac{\exp(\kappa_k c_{ijk})}{\sum_{n=1}^K \exp(\kappa_n c_{ijn})}$$

> **注**：这在神经网络中非常容易实现，只需将输入 $x$ 与权值矩阵 $\mu$ 相乘，乘以缩放因子 $\kappa$，再过一遍 Softmax。

#### 3. 梯度项的神经网络表达 (对应 Eq. 8 & 9)

为了提取一阶和二阶统计量，我们将上一轮推导的球面梯度进行“层化”处理。

**一阶均值位移项（First-order Term）：**
这一项捕捉的是样本相对于聚类中心的**切向切片**信息。
$$\mathcal{G}_{\mu_k}^{\mathbf{x}_{ij}} = \gamma_j(k) \sqrt{\kappa_k} \left( \frac{\mathbf{x}_{ij} - c_{ijk}\mu_k}{\sqrt{1 - c_{ijk}^2 + \epsilon}} \right)$$
* 这里 $\epsilon$ 是为了数值稳定性加入的极小值。
* 括号内的部分实际上是 $\mathbf{x}_{ij}$ 在 $\mu_k$ 处的**单位切向量**。

**二阶集中度偏差项（Second-order Term）：**
这一项捕捉的是样本相对于聚类中心的**径向分布**信息（紧凑程度）。
$$\mathcal{G}_{\kappa_k}^{\mathbf{x}_{ij}} = \gamma_j(k) \frac{1}{\sqrt{2}} [\kappa_k (1 - c_{ijk}) - 1]$$
* 这里 $(1 - c_{ijk})$ 是标准的余弦距离。

#### 4. 最终层结构输出 (对应 Eq. 11)
将所有 patch 的梯度进行均值池化（Mean-pooling），得到最终的球面 Fisher 层表示：
$$\phi_{SFL}(X_i) = \frac{1}{m_i} \sum_{j=1}^{m_i} [\mathcal{G}_{\mu_1}, \dots, \mathcal{G}_{\mu_K}, \mathcal{G}_{\kappa_1}, \dots, \mathcal{G}_{\kappa_K}]^T$$

---

### 架构图描述 (类似 Fig. 3)

如果您要画出对应的网络架构图，流程如下：

1.  **Input**: 归一化特征 $\mathbf{x}_{ij}$。
2.  **Dot Product Layer**: 计算 $\mathbf{x}_{ij}^T \mu_k$，得到余弦相似度矩阵 $C$。
3.  **Softmax Branch**: 将 $C$ 乘以 $\kappa_k$，通过 Softmax 得到权重 $\gamma_j(k)$。
4.  **Tangent Projection (First-order)**: 
    * 计算残差向量：$\mathbf{v} = \mathbf{x}_{ij} - c_{ijk} \cdot \mu_k$。
    * 归一化：$\mathbf{n} = \mathbf{v} / \sqrt{1 - c^2}$。
    * 加权：$\gamma_j(k) \sqrt{\kappa_k} \cdot \mathbf{n}$。
5.  **Concentration Deviation (Second-order)**:
    * 计算：$\gamma_j(k) \cdot (\kappa_k(1-c) - 1)$。
6.  **Pooling**: 对所有 $j$ 进行 Mean Pooling。
7.  **Output**: 拼接所有 $k$ 的向量，得到最终 FV。

---

### 为什么这样改进更好？

1.  **消除数值不稳定性**：原论文中的 $1/\sigma$ 在 $\sigma$ 接近 0 时会爆炸。在球面上，$\kappa$ 是相乘关系，且余弦值被严格限制在 $[-1, 1]$，训练更稳定。
2.  **更强的判别力**：球面坐标系天然适合深度学习中的特征表示（尤其是 Face Recognition 或细粒度分类），因为它只关注“角度”信息，过滤掉了幅度噪声。
3.  **计算简化**：原论文中需要计算 $(x - \mu)^2$ 的平方项，而球面层核心运算是 **矩阵点积**，这在现代 GPU 上有极高的运算效率。

**建议：** 在实际实现时，可以参考论文中的做法，将 $\kappa_k$ 初始化为一个较大的正数（如 10~30），将 $\mu_k$ 初始化为通过球面 $k$-means 得到的聚类中心。





python train_backnone.py --pretrained_imagenet


python train_fisher.py --backbone_ckpt checkpoints/backbone.pth


python SVM_test.py --fishernet_ckpt checkpoints/fishernet.pth

这套推导和架构设计非常出色，逻辑严密。将欧式空间的度量替换为球面空间的度量，不仅在数学上与 von Mises-Fisher (vMF) 分布完美契合，而且在深度学习的实际应用中（尤其是在特征表示和度量学习中），引入余弦相似度和特征归一化通常能显著提升模型对类内差异的鲁棒性。

你的推导完全合理，特别是一阶和二阶梯度项的几何解释，精准地捕捉到了球面流形（Spherical Manifold）上的切向（方向）和径向（集中度）信息。

针对你在工程实现和理论转换上的疑问，以下是详细的解答：

### 1. 哪些是反向过程优化的参数？
在你的 Spherical Fisher Layer (SFL) 中，可以通过反向传播（Backpropagation）优化的核心可学习参数有两个：
* **聚类中心方向 $\mu_k$**（维度为 $K \times D$）：代表 $K$ 个 vMF 分布的均值向量。
* **集中度参数 $\kappa_k$**（维度为 $K \times 1$）：代表每个聚类的紧凑程度。

此外，由于这是一个端到端（End-to-End）的网络，SFL 之前的卷积层或全连接层（用于提取特征 $\mathbf{x}_{ij}$ 的层）的权重，同样会接收到来自 SFL 的梯度并进行更新。

---

### 2. $\mu$ 是参数，训练过程中如何一直保持归一化？
在欧式空间中，参数可以自由更新。但在球面空间中，$\mu_k$ 必须严格约束在单位超球面上（即 $\|\mu_k\|_2 = 1$）。在深度学习框架（如 PyTorch 或 TensorFlow）中，通常有以下三种标准方案来实现这一点：

#### 方案 A：参数重参数化 (Reparameterization) **[最推荐]**
不在网络中直接将 $\mu_k$ 定义为严格的单位向量，而是定义一个无约束的自由参数变量 $\tilde{\mu}_k \in \mathbb{R}^D$。
在每次**前向传播（Forward Pass）**时，强制进行 L2 归一化：
$$\mu_k = \frac{\tilde{\mu}_k}{\|\tilde{\mu}_k\|_2}$$
利用深度学习框架的自动求导机制，反向传播会自动计算经过归一化操作后的梯度，并更新底层的自由变量 $\tilde{\mu}_k$。这种方法最简单，且不需要修改优化器。

#### 方案 B：投影梯度下降 (Projected Gradient Descent, PGD)
将 $\mu_k$ 定义为模型参数。在正常的优化器（如 SGD 或 Adam）执行完一步参数更新（`optimizer.step()`）之后，手动将参数“拉回”到球面上：
$$\mu_k \leftarrow \frac{\mu_k}{\|\mu_k\|_2}$$
这种方法在流形优化中很常见，可以直接控制最终的权重。

#### 方案 C：黎曼优化 (Riemannian Optimization)
如果你追求极致的数学严谨性，可以使用专门的黎曼优化器（如 GeoSGD）。在计算出 $\mu_k$ 的欧式梯度后，将其投影到 $\mu_k$ 的切平面上，并沿着超球面上的测地线（Geodesic curve）更新参数。对于常规的计算机视觉任务，这种方法计算成本较高，通常方案 A 就足够了。

---

### 3. $\mu$ 和 $\kappa$ 的初始化问题
你的思路非常正确。既然我们将 GMM 替换为了 vMF 混合模型，初始化策略也必须相应地从欧式聚类切换到球面聚类。

* **$\mu_k$ 的初始化**：
    你可以使用 **Spherical K-Means (spkmeans)** 算法对预训练网络提取的特征 $\mathbf{x}_{ij}$ 进行聚类。Spherical K-Means 使用余弦相似度进行聚类分配，并在更新中心时对其进行 L2 归一化，其输出的聚类中心直接就是极其优秀的 $\mu_k$ 初始值。
* **$\kappa_k$ 的初始化**：
    在使用 Spherical K-Means（或者 vMF-EM 算法）得到聚类后，你可以计算分配到第 $k$ 个聚类的所有样本特征与中心 $\mu_k$ 的平均余弦相似度，记为均值合量长度（Mean Resultant Length）$\bar{R}$：
    $$\bar{R}_k = \frac{1}{N_k} \sum_{\mathbf{x} \in C_k} \mathbf{x}^T \mu_k$$
    利用方向统计学中的近似公式，可以通过 $\bar{R}_k$ 估算 $\kappa_k$ 的初始值（其中 $D$ 是特征维度）：
    $$\kappa_k \approx \frac{\bar{R}_k D - \bar{R}_k^3}{1 - \bar{R}_k^2}$$
    将估算出的 $\kappa_k$ 直接赋给网络中的可学习标量即可。

你的这套改进方案不仅在理论上自洽，在解决类别边界重叠问题时通常会有奇效。在实现一阶均值位移项时，公式分母中的极小值 $\epsilon$ 有时会带来数值不稳定的梯度（尤其当样本点 $\mathbf{x}$ 与聚类中心 $\mu$ 几乎完全重合时，切向量接近零向量），你是否有考虑过在代码层面如何优雅地处理这种几何奇异点？