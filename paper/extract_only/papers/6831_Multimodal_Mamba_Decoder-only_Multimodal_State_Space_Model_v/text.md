## Multimodal Mamba: Decoder-only Multimodal State Space Model via Quadratic to Linear Distillation

# arXiv:2502.13145v2[cs.CV]18Mar2025

#### Bencheng Liao*12 Hongyuan Tao*2 Qian Zhang3 Tianheng Cheng2 Yingyue Li2 Haoran Yin3 Wenyu Liu2 Xinggang Wang2

POPE

Our mmMamba-linear-2.7B

Image tokens Text tokens Quadratic to linear distillation

MMB

86.7

(Linear, Decoder-only)

85.2

63.7 57.2

###### Transformer (quadratic)

Transformer (quadratic)

Mamba-2 (linear/hybrid)

Our mmMamba-hybrid-2.7B

TQA

(Hybrid, Decoder-only)

55.1 47.7

VL-Mamba-3B

(Linear, Encoder-based)

1371.1

Connector

MME

N/A

1303.5

EVE-7B

(Quadratic, Decoder-only)

Vision Encoder

79.2

SQA-I 86.9

MobileVLM-3B

(Quadratic, Encoder-based)

31.1

57.4

59.3

GQA

HoVLE-2.6B

36.9

Encoder-based VLM Decoder-only VLM Our mmMamba

(Quadratic, Decoder-only)

MM-Vet

(a) Architecture Comparison (b) Performance comparison

OOM OOM

142.65

21.69

-60.2%

###### -75.9%

###### 13.5x

###### 20.6x

8.64

10.54

5.22

6.94

(c) Speed Comparison (d) Memory Comparison

Figure 1: Comprehensive comparison of mmMamba. (a) Our mmMamba can build linear-complexity and hybrid decoder-only VLM by distilling the knowledge in Transformer to Mamba-2. (b) By distilling from the quadratic-complexity decoder-only VLM HoVLE, our mmMamba-linear achieves competitive performance against existing linear and quadraticcomplexity VLMs with fewer parameters (e.g., 2× fewer than EVE-7B), while mmMamba-hybrid surpasses them across all benchmarks and approaches the teacher model HoVLE’s performance. (c)-(d) We compare the speed and memory of mmMamba-linear and mmMamba-hybrid with the teacher model HoVLE on the same single NVIDIA 4090 GPU. mmMamba-linear maintains consistently low latency and memory usage, while mmMamba-hybrid’s resource consumption scales significantly better than HoVLE. At 103K tokens, mmMamba-linear demonstrates 20.6× speedup compared to HoVLE and saves 75.8% GPU memory, while mmMamba-hybrid achieves 13.5× speedup and saves 60.2% GPU memory.

### Abstract

tional resources. Our approach enables the direct conversion of trained decoder-only MLLMs to linear-complexity architectures without requiring pre-trained RNN-based LLM or vision encoders. We propose an seeding strategy to carve Mamba from trained Transformer and a three-stage distillation recipe, which can effectively transfer the knowledge from Transformer to Mamba while preserving multimodal capabilities. Our method also supports flexible hybrid architectures that combine Transformer and Mamba layers for customizable efficiency-performance trade-offs. Distilled from the Transformer-based decoderonly HoVLE, mmMamba-linear achieves competitive performance against existing linear and quadratic-complexity VLMs, while mmMambahybrid further improves performance significantly,

Recent Multimodal Large Language Models (MLLMs) have achieved remarkable performance but face deployment challenges due to their quadratic computational complexity, growing Key-Value cache requirements, and reliance on separate vision encoders. We propose mmMamba, a framework for developing linearcomplexity native multimodal state space models through progressive distillation from existing MLLMs using moderate academic computa-

*Equal contribution 1Institute of Artificial Intelligence, Huazhong University of Science & Technology 2School of EIC, Huazhong University of Science & Technology 3Horizon Robotics. Correspondence to: Xinggang Wang <xgwang@hust.edu.cn>.

approaching HoVLE’s capabilities. At 103K tokens, mmMamba-linear demonstrates 20.6× speedup and 75.8% GPU memory reduction compared to HoVLE, while mmMamba-hybrid achieves 13.5× speedup and 60.2% memory savings. Code and models are released at https: //github.com/hustvl/mmMamba

### 1. Introduction

Recent advances in Large Language Models (LLMs) (Brown et al., 2020; Achiam et al., 2023; Touvron et al., 2023a;b; Dubey et al., 2024; Bai et al.,

- 2023a; Jiang et al., 2023; Bi et al., 2024; Javaheripi

- et al., 2023) have catalyzed significant research interest in expanding their capabilities beyond text to encompass multimodal understanding, particularly in processing both visual and textual information simultaneously. This expansion has given rise to Multimodal Large Language Models (MLLMs), with Vision Language Models (VLMs) emerging as a prominent subset. Notable examples such as LLaVA (Liu et al., 2024a), BLIP (Li et al., 2022), Qwen-VL (Bai et al., 2023b), InternVL (Chen et al., 2024b), and Monkey (Li et al., 2024b) have demonstrated remarkable success in enhancing LLMs’ visual comprehension capabilities through the integration of pre-trained vision encoders and specialized connectors that bridge the modality gap between vision and language.

While these encoder-based compositional VLMs have achieved state-of-the-art (SOTA) performance and established themselves as the de-facto paradigm, they face two critical limitations. First, processing long contexts becomes prohibitively expensive due to the quadratic computational complexity and linearly growing Key-Value (KV) cache with respect to sequence length. This limitation becomes particularly problematic given the increasing demand for long chain-of-thought reasoning (Muennighoff et al., 2025; DeepSeek-AI et al., 2025; Team et al., 2025; Xu et al., 2024) and high-resolution image/video understanding (Chen et al., 2023; Li et al., 2024a; Zhu et al.; Liao et al., 2024; Zhu et al.,

- 2024a). Second, their heterogeneous architecture heavily relies on pre-trained vision encoders, introducing significant complexity in both training procedures and deployment scenarios (Chen et al., 2024a).

Current research efforts to address these limitations have followed two distinct paths. One approach focuses on developing linear-complexity VLMs by adhering to the conventional encoder-based recipe, which requires both pre-trained vision encoders and pre-trained linear-complexity language models (Hou et al., 2024; Qiao et al., 2024). The alternative approach aims to enhance decoder-only VLMs through increased model scale and expanded training datasets, achiev-

ing performance competitive with encoder-based counterparts (Bavishi et al., 2023; Diao et al., 2024; Tao et al., 2024; Team, 2024; Wang et al., 2024b).

Despite these advances, the development of linearcomplexity decoder-only MLLMs remains an understudied yet critical challenge. Addressing this gap holds substantial value for three key reasons: (1) Unified multimodal understanding: Such models could seamlessly integrate multimodal reasoning within a single architecture, eliminating the need for heterogeneous, modality-specific frameworks. (2) Practical efficiency: Linear-complexity models inherently reduce computational demands during both training and inference, lowering costs and enabling deployment on resource-constrained edge devices. (3) Untapped Potential: While recent linear-time models like Mamba-2 demonstrate high text-processing capabilities, their ability to handle multimodal tasks—particularly in cross-modal alignment and reasoning—remains largely unexplored. The research of linear-complexity decoder-only MLLMs could unlock scalable, cost-effective multimodal systems without sacrificing performance.

A straightforward solution is to synergize the recipe of decoder-only VLMs and linear-complexity encoder-based VLMs. This integration requires a pre-trained linearcomplexity LLM and performs image-caption alignment pre-training (PT) and supervised fine-tuning (SFT) using text instructions and image prompts. However, this integrated recipe faces two significant challenges: (1) It demands the curation of different large-scale multimodal datasets for different purposes (i.e., PT and SFT) and requires substantial computational resources. (2) The overall performance is inherently limited by the capabilities of pre-trained linear-complexity LLMs, which consistently underperform mainstream SOTA Transformer-based LLMs in language understanding tasks.

In this paper, we propose a novel distillation-based recipe to develop linear-complexity decoder-only VLMs, which requires only moderate academic resources while circumventing the limitations of pre-trained linear-complexity LLMs. Our method leverages the fundamental similarity between the Transformer attention mechanism and the Mamba-2 state space model (SSM) mechanism. We introduce an initialization scheme that enables direct parameter transfer from Transformer to Mamba-2 layers, effectively converting the attention mechanism into the SSM function while carefully initializing SSM-specific parameters to mimic attention behavior. This approach enables the direct transformation of pre-trained Transformer-based VLMs into linearcomplexity Mamba-2-based VLMs without relying on underperforming pre-trained linear-complexity LLMs. While this parameter inheritance and initialization strategy provides a promising starting point, the transformed Mamba-

2-based VLM requires further distillation to recover robust multimodal conversation capabilities. To enhance alignment with the Transformer-based teacher VLM, we develop a three-stage progressive distillation strategy: (1) Stage-1: we first train the SSM-specific parameters while freezing inherited parameters, and align layer-wise behavior using MSE distillation loss; (2) Stage-2: we then optimize complete Mamba-2 layer behavior by enabling the training of inherited Transformer parameters; (3) Stage-3: we finally perform complete model alignment using KL-divergence loss on final output logits to recover the teacher model’s multimodal understanding capabilities through end-to-end distillation.

The proposed distillation recipe enables two distinct architectural variants: mmMamba-linear, which converts all Transformer layers into Mamba-2 layers, achieving full linear complexity, and mmMamba-hybrid, which strategically transforms fixed intervals of Transformer layers into Mamba-2 layers. The hybrid design systematically preserves Transformer layers at critical feature hierarchies while leveraging Mamba-2’s linear complexity for the majority of computations, striking a balance between efficiency and capability. During the final end-to-end distillation stage, we can flexibly adjust the number of interleaved Transformer layers, enabling precise control over the computation-performance trade-off. This architectural flexibility makes our approach highly adaptable to diverse deployment scenarios, allowing optimization for specific computational constraints while maintaining desired performance.

By distilling from the recent Transformer-based decoderonly VLM, HoVLE, we demonstrate that mmMamba achieves competitive performance across multiple visionlanguage benchmarks while significantly improving computational efficiency. Our pure Mamba-2-based linearcomplexity variant, mmMamba-linear, achieves comparable performance to existing quadratic/linear-complexity VLMs like Mobile-VLM-3B (Chu et al., 2023), VisualRWKV3B (Hou et al., 2024), VL-Mamba-3B (Qiao et al., 2024) while eliminating the need for separate vision encoders. mmMamba-pure also matches the performance of the previous SOTA Transformer-based decoder-only EVE-7B with 2× fewer parameters. The hybrid variant, mmMambahybrid, significantly improves performance on all benchmarks compared to mmMamba-pure, approaching the teacher model HoVLE. Notably, at the context length of 103K tokens, mmMamba-linear demonstrates 20.6× speedup compared to HoVLE and saves 75.8% GPU memory, while mmMamba-hybrid achieves 13.5× speedup and saves 60.2% GPU memory. These results and extensive ablation studies validate the effectiveness of our distillation recipe and highlight the potential for practical applications.

Our main contributions can be summarized as follows:

- • We present a novel three-stage progressive distillation recipe for building native multimodal state space models without the reliance on underperforming pre-trained linear-complexity LLMs, enabling effective knowledge transfer from quadratic to linear architectures.
- • With the proposed distillation recipe, we propose the first decoder-only multimodal state space models that include two distinct architectural variants: mmMambalinear with purely linear complexity and mmMambahybrid offering flexible performance-efficiency tradeoffs.
- • Extensive experimental results demonstrate competitive performance with significantly improved computational efficiency across various vision-language tasks, achieving up to 20.6× speedup and 4.2× memory reduction for long sequence modeling on NVIDIA 4090 GPU.

### 2. Related Work

Decoder-only VLM. The remarkable success of Large Language Models (LLMs) has inspired the research community to extend their capabilities to multi-modal Vision-Language Models (VLMs). While compositional encoder-based architectures (Liu et al., 2024a; Chen et al., 2024b; Li et al., 2024b; 2022), leveraging pre-trained foundation vision encoders (Fang et al., 2023; 2024; Sun et al., 2023; Zhai et al., 2023) and additional connectors, have dominated the field. Recently, a pioneering work Fuyu-8B (Bavishi et al., 2023) demonstrated that a single unified decoder-only Transformer can achieve competitive performance against encoder-based VLMs, offering an appealing alternative due to its architectural simplicity and deployment efficiency. This breakthrough has sparked researchers’ interest in decoder-only VLM. SOLO (Chen et al., 2024a) proposed a systematic training recipe tailored for decoder-only VLM by adapting pre-trained LLMs to vision-language tasks. EVE (Diao et al., 2024) advanced this approach by introducing visionlanguage pre-alignment and auxiliary visual representation supervision during fine-tuning to enhance the performance of decoder-only VLM. To better preserve the inherited LLM’s language capabilities, HoVLE (Tao et al., 2024) introduces an extra Transformer-based decoder-only holistic embedding module that aligns language and vision modalities before LLM processing multi-modal input tokens. Despite these advances, existing decoder-only VLMs remain constrained by the quadratic computational complexity of Transformer architectures, resulting in substantial training and deployment costs. In contrast, our proposed mmMamba addresses these limitations by converting Transformer layers to linear-complexity Mamba-2 layers through progressive

distillation, enabling both pure linear and hybrid architectural variants.

Linear-complexity VLM. The development of linearcomplexity RNN-based LLMs (e.g., Mamba (Gu & Dao, 2023), Mamba-2 (Dao & Gu, 2024), RWKV (Peng et al., 2023)) has inspired increasing interest in addressing the quadratic complexity limitations of Transformer-based VLMs. VL-Mamba (Qiao et al., 2024) follows the recipe of LLaVA by incorporating a Vision Selective Scan connector with the pre-trained Mamba LLM. Similarly, Cobra (Zhao et al., 2024) enhances pre-trained Mamba LLM’s visual capabilities by integrating DINOv2 (Oquab et al.,

- 2023) and SigLIP (Zhai et al., 2023) vision encoders. MLMamba (Huang et al., 2024) introduces a Mamba-2 Scan Connector to process visual tokens between the pre-trained vision encoder and the pre-trained Mamba-2 LLM. Instead of relying on Mamba, VisualRWKV (Hou et al., 2024) leverages CLIP ViT-L/14 (Radford et al., 2021) as the vision encoder and a pre-trained RWKV LLM (Peng et al., 2023;
- 2024) with a 2D image scanning mechanism for visual sequence processing. However, the above works remain constrained by their reliance on pre-trained RNN-based LLMs and vision encoders, following the compositional encoderbased paradigm. In contrast, our proposed mmMamba eliminates the dependency on pre-trained RNN-based LLMs and vision encoders, and enables training a flexible hybrid architecture that interleaves Mamba with Transformer layers with minimal training cost. This capability enables customizable trade-offs between performance and efficiency, making it adaptable to diverse practical applications.

Transformer to RNN distillation. Instead of training RNNbased LLMs from scratch, recent studies propose to linearize the pre-trained Transformer-based LLMs into RNNbased LLMs through distillation, which can significantly reduce the training cost for building RNN-based LLMs. Kasai et al. (2021) pioneered this approach by using linear attention and initializing linear attention parameters using pre-trained LLM weights, exploiting the inherent similarities with Transformer’s softmax attention. Zhang et al. (2024b) propose to add loss for matching softmax attention to approximate more closely the base transformer. Mercat et al. (2024) advanced the field by replacing softmax attention with a linear RNN kernel coupled with a novel normalization strategy. Building upon these foundations, Bick et al. (2024), Wang et al. (2024a), and Zhang et al. (2024a) developed multi-stage distillation approaches for more effective Transformer to RNN distillation. Inspired by these advances, we extend this distillation paradigm to VLMs through the proposed novel multi-stage distillation strategy. Our approach first aligns the newly added parameters of the linearized LLM at each layer, followed by layer-wise distillation, and concludes with end-to-end distillation. This progressive pipeline ensures efficient trans-

fer from quadratic knowledge to linear knowledge while maintaining performance.

### 3. Preliminary

We firstly give a brief background on quadratic-complexity sequence modeling Transformer and linear-complexity sequence modeling Mamba-2. Given an input sequence

- X = [x1,...,xT]⊤ ∈ RT×d, where T is the sequence length and d is the hidden dimension. The above two sequence modeling layers will compute the output sequence
- Y = [y1,...,yT]⊤ ∈ RT×d.

Transformer The standard autoregressive Transformer used in LLM employs attention mechanism (Vaswani, 2017) by interacting with all historical positions in the sequence, which is defined as:

qt,kt,vt = xtWQ,xtWK,xtWV , yt =

t i=1 exp(qtk⊤i )vi

,

t i=1 exp(qtk⊤i )

(1)

where WQ,WK,WV ∈ Rd×d are the learnable parameters. The current output token ot is computed by performing attention over the growing sequence of historical keys {ki}ti=1 and values {vi}ti=1.

Mamba-2 Instead of interacting with all historical positions, Mamba-2 (Dao & Gu, 2024) compresses the historical information into a fixed-size matrix-shaped hidden state, which is defined as:

qt,kt,vt = xtWQ,xtWK,xtWV , γt = exp(−softplus(xtWγ)exp(a)), St = γtSt−1 + vtk⊤t , yt = Stqt,

(2)

where WQ,WK,WV ∈ Rd×d, Wγ ∈ Rd×1 and a ∈ R are the learnable parameters. St is the fixed-size matrixshaped hidden state, γt is the data-dependent gating term to control the information flow by dynamically decaying the historical information St−1.

### 4. Method

Our method consists of three key components. First, we detail the seeding strategy, which carves the Mamba-2 architecture from a pre-trained Transformer by inheriting parameters and carefully initializing the newly introduced SSMspecific parameters in Sec. 4.1. Building upon this seeding strategy, we present the proposed progressive distillation pipeline in Sec. 4.2, Sec. 4.3 and Sec. 4.4 to effectively transfer knowledge from Transformer to Mamba-2. With the designed distillation training recipe, we then instantiate

𝐎

Inherited Param Extra Param

𝐎

| |
|---|

| |
|---|

#### 4.2. Stage-1: Layerwise Distillation for the Newly Introduced SSM Parameters

|𝑾𝑶|
|---|

|𝑾𝑶|
|---|

We first perform layerwise distillation for the introduced extra parameters to align the proposed Mamba-2 layer with the original trained Transformer layer. Specifically, we instantiate the trained Transformer-based VLM as teacher model, and the transferred Mamba-2 VLM model as student model. The only difference lies in the sequence mixer layer. We feed the multimodal sequence into the teacher model.

⨀ 𝐆

| | |
|---|---|
|𝑾𝑮| |

𝐘

𝐘

Initialize

Attention

Mamba-2

| | |
|---|---|
|𝑾𝑲| |

| | |
|---|---|
|𝑾𝑸| |

| | |
|---|---|
|𝑾𝑽| |

| | |
|---|---|
|𝒂| |

| | |
|---|---|
|𝑾𝜸| |

Causal Conv: 𝑾𝐜𝐨𝐧

|𝑾𝑸|
|---|

|𝑾𝑽|
|---|

|𝑾𝑲|
|---|

𝐗

𝐗

To keep the layerwise alignment and diminish the accumulated error of the cascading layers, we input the i-th Mamba-2 layer with the output of the i − 1-th Transformer layer, i.e., i-th Mamba-2 layer and i-th Transformer layer have the same input. And we align the layerwise behavior by applying the MSE distillation loss between the output of the i-th Mamba-2 layer and the output of the i-th Transformer layer:

Trained Transformer Layer Initialized Mamba-2 Layer

- Figure 2: Initialize Mamba-2 from Transformer. By comparing the mechanism similarity in Sec. 3, we directly in-

herit WQ, WK, WV , WO parameters (blue) from trained Transformer layer and carefully initialize the extra parameters (orange) including a, Wγ, Wconv, and WG in Mamba-

- 2 to initially mimic the Transformer’s behavior, providing a strong foundation for subsequent distillation.

ϕistage1 = {ai,Wiγ,Wiconv,WiG}, min {ϕistage1}Li=1

(3)

L

LMSE(Attn(Xi),Mamba-2ϕi

(Xi)),

two model variants in Sec. 4.5: mmMamba-linear using only Mamba-2 layers, and mmMamba-hybrid incorporating interleaved Transformer and Mamba-2 layers.

stage1

i=1

where ϕistage1 is the trainable parameters of the i-th Mamba-2 layer, which only includes the introduced ex-

#### 4.1. Seeding: Initialize Mamba-2 from Transformer

tra parameters ai,Wiγ,Wiconv,WiG. Xi is the input sequence to the i-th Mamba-2 layer and Transformer layer, Attn(Xi) is the output of the i-th teacher Transformer layer, Mamba-2ϕi

To transfer as much knowledge as possible from quadratic Transformer to linear Mamba-2, we initialize Mamba-2 from Transformer at each layer. By comparing Eq. 1 and Eq. 2, we can find that Mamba-2 shares the similarity with Transformer, which means we can directly inherit WQ,WK,WV and WO projection parameters at each layer instead of building from scratch. Furthermore, we need to introduce extra parameters Wγ and a for state space modeling, replacing the attention mechanism. For better replacement and ease the training difficulty (Trockman

(Xi) is the output of the i-th student Mamba2 layer.

stage1

#### 4.3. Stage-2: Layerwise Distillation for the Whole Mamba-2 Parameters

After the stage-1 distillation, we have obtained good initialization of the introduced extra parameters, and we further train all the Mamba-2 parameters to better align the layerwise behavior of the student Mamba-2 with the teacher Transformer. The only difference between the stage-1 and stage-2 is that we further include the parameters of WQ,WK,WV for optimizing the distillation loss:

- et al., 2024), we initialize Wγ and a to make the gating term γt close to 1 at the beginning of training, which means we begin by memorizing all historical information without selectivity.

Beyond the core SSM mechanism, we also introduce extra causal convolution and output gating for enhanced positional awareness and expressiveness. To eliminate the initial impact of causal convolution, we initialize the weights and biases to make it function as an identity layer (i.e., the output of causal convolution is the same as the input) without affecting the original function of SSM at the beginning of training.

ϕiStage2 = {ai,Wiγ,Wiconv,WiG,WiQ,WiK,WiV }, min

(4)

L

LMSE(Attn(Xi),Mamba-2ϕi

(Xi)),

Stage2

{ϕiStage2}Li=1

i=1

#### 4.4. Stage-3: End-to-End Distillation

Beyond the layerwise alignment, the final stage-3 distillation aims to align the end-to-end behavior of the student Mamba2 with the teacher Transformer. Specifically, we input the same multi-modal sequence to both the teacher Transformer and the student Mamba-2 without sharing the intermediate

The other parts of the model such as the MLP layers and text and image patch embedding layers are directly inherited from the original Transformer-based VLM and kept as frozen.

|Stage-3: End-to-End Distillation<br><br>𝐗<br><br>ℒ 𝐏 𝐏<br><br>Mamba-2<br><br>MLP<br><br>𝐿×<br><br>[Figure 1]<br><br>[Figure 2]<br><br>Attention<br><br>MLP<br><br>𝐿×<br><br>[Figure 3]<br><br>[Figure 4]<br><br>Decode Decode<br><br>[Figure 5]<br><br>[Figure 6]|
|---|

ℒ

ℒ

𝐎 𝐎

𝐎 𝐎

|[Figure 7]|
|---|

| | | |
|---|---|---|
| | | |
| | | |
| | | |

[Figure 8]

Patch Embedding

Mamba-2

Mamba-2

Attention

Attention

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

|𝑊|
|---|

|𝑊|
|---|

|𝑊|
|---|

|𝑊|
|---|

|𝑊|
|---|

|𝑊|
|---|

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

|𝑊|
|---|

|𝑊|
|---|

|𝑊|
|---|

|𝑊|
|---|

|𝑊|
|---|

|𝑊|
|---|

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

|𝑊|
|---|

|𝑊|
|---|

|𝑊|
|---|

|𝑊|
|---|

|𝑊|
|---|

|𝑊|
|---|

|Text Question & Answer|
|---|

[Figure 27]

Text Embedding

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

|𝑎|
|---|

|𝑊|
|---|

|𝑎|
|---|

|𝑊|
|---|

|𝑊|
|---|

|𝑊|
|---|

𝐗

𝐗

𝐗 Update Frozen Image tokens Text tokens

Stage-1: Distill the New

Stage-2: Distill the Whole

… …

| |
|---|

| |
|---|

Inherited Params

Extra Params Forward Distill loss

[Figure 34]

- Figure 3: Progressive distillation pipeline of our mmMamba. We keep MLP layers, text and image patch embedding layers and freeze them in subsequent distillation training stages. Stage-1: Train the newly-introduced SSM-specific parameters while freezing inherited Transformer parameters in a layer-wise manner. Stage-2: Train all parameters to align Mamba’s state representation with Transformer in a layer-wise manner. Stage-3: Train all the Mamba layers of the model to align the end-to-end behavior with the teacher Transformer-based VLM.

output. For the output of the teacher model and the student model, we apply the word-level KL-Divergence loss, in other words, they are used as soft labels, we enforce the output logits of the student model to be close to the output logits of the teacher model:

various deployment scenarios with varied requirements. In this paper, we set the interval as 4, building mmMambahybrid with 8 Transformer layers and 24 Mamba-2 layers in total.

### 5. Experiment

ϕStage-3 = {ai,Wiγ,Wiconv,WiG,WiQ,WiK,WiV }Li=1, min

LKL(Teacher-(X0),Studentϕ

##### (X0)),

#### 5.1. Implementation Detail

Stage3

ϕStage-3

(5)

Training. All models are trained using 8 NVIDIA A800 80GB GPUs with BF16 precision and DeepSpeed ZeRO2 (Rajbhandari et al., 2020; Rasley et al., 2020). The distillation process utilizes SOLO’s (Chen et al., 2024a) supervised fine-tuning dataset, comprising 1.7M samples across both language-only and image-text paired instances. We employ the AdamW (Loshchilov, 2017) optimizer with β = (0.9,0.999), gradient clipping at 5.0, and a WarmupStable-Decay (WSD) scheduler with 10% warmup and 10% decay periods. For stages-1 and stage-2 distillation, we use a batch size of 128, train for 20K steps, and set weight decay to 0.05, with learning rates of 1 × 10−3 and 5 × 10−4 respectively. Stage-3 distillation employs a reduced batch size of 64, continues for 20K steps with weight decay at 0.05, and uses a learning rate of 5 × 10−5.

where X0 is the same multi-modal input sequence to the teacher model and the student model, ϕStage3 is the trainable parameters of the student model.

#### 4.5. Architecture

Our mmMamba builds upon HoVLE, a decoder-only VLM that consists of 32 Transformer layers. For mmMambalinear, we convert each Transformer layer into a Mamba2 layer while preserving the MLP layers, resulting in a linear-complexity decoder-only VLM. To enhance model expressiveness, we adopt a multi-head design in our Mamba-

- 2 layers by partitioning the SSM into multiple groups and implementing shared queries across groups, consistent with the grouped query attention used in HoVLE.

Evaluation benchmarks. We evaluate our model on 9 diverse public benchmarks, encompassing 6 general VLM benchmarks and 3 visual question answering tasks. The general VLM benchmarks include: MME (Yin et al., 2023), which evaluates visual perception and reasoning through true/false questions; MMBench (Liu et al., 2024b), which assesses model robustness through multiple-choice questions; POPE (Li et al., 2023b), which evaluates object hallucination; SEED (Li et al., 2023a), which gauges open-world multi-modal understanding; MMMU (Yue et al., 2024),

For mmMamba-hybrid, we introduce a systematic layer conversion scheme. Specifically, within every fixed number of consecutive layers, we preserve the first layer as Transformer and convert the remaining layers to Mamba-2. This hybrid scheme maintains the Transformer’s modeling capacity at critical feature hierarchies while leveraging Mamba2’s linear complexity for the majority of computation. Such design enables an effective and flexible trade-off between computational efficiency and model capability, suitable for

Method Recipe Complexity # P. # T.P. MME MMB POPE SEED MMMU MM-Vet TQA SQA-I GQA Encoder-based VLMs OpenFlamingo (Awadalla et al., 2023) PT, SFT Quadratic 9B 96.6% - 4.6 - - - - 33.6 - MiniGPT-4 (Zhu et al., 2023) PT, SFT Quadratic 13B 94.8% 581.7 23.0 - - - 22.1 - - 32.2 Qwen-VL (Bai et al., 2023b) PT, SFT Quadratic 7B 100.0% - 38.2 - 56.3 - - 63.8 67.1 59.3 LLaVA-Phi (Zhu et al., 2024b) PT, SFT Quadratic 3B 90.0% 1335.1 59.8 85.0 - - 28.9 48.6 68.4 MobileVLM-3B (Chu et al., 2023) PT, SFT Quadratic 3B 90.0% 1288.9 59.6 84.9 - - - 47.5 61.0 59.0 VisualRWKV (Hou et al., 2024) PT, SFT Linear 3B 90.0% 1369.2 59.5 83.1 - - - 48.7 65.3 59.6 VL-Mamba (Qiao et al., 2024) PT, SFT Linear 3B 90.0% 1369.6 57.0 84.4 - - 32.6 48.9 65.4 56.2 Cobra (Zhao et al., 2024) PT, SFT Linear 3.5B 82.6% - - 88.4 - - - 58.2 - 62.3 Decoder-only VLMs Fuyu-8B (HD) (Bavishi et al., 2023) PT, SFT Quadratic 8B 100.0% 728.6 10.7 74.1 - - 21.4 - - SOLO (Chen et al., 2024a) PT, SFT Quadratic 7B 100.0% 1001.3 - - 64.4 - - - 73.3 Chameleon-7B (Team, 2024) PT, SFT Quadratic 7B 100.0% 170 31.1 - 30.6 25.4 8.3 4.8 47.2 EVE-7B (Diao et al., 2024) PT, SFT Quadratic 7B 100.0% 1217.3 49.5 83.6 61.3 32.3 25.6 51.9 63.0 60.8 Emu3 (Wang et al., 2024b) PT, SFT Quadratic 8B 100.0% - 58.5 85.2 68.2 31.6 37.2 64.7 89.2 60.3 HoVLE (Tao et al., 2024) DT, PT, SFT Quadratic 2.6B 100.0% 1433.5 71.9 87.6 70.7 33.7 44.3 66.0 94.8 60.9 mmMamba DT Linear 2.7B 14.7% 1303.5 57.2 85.2 62.9 30.7 31.1 47.7 79.2 57.4 mmMamba DT Hybrid 2.7B 11.2% 1371.1 63.7 86.7 66.3 32.3 36.9 55.1 86.9 59.3

- Table 1: Comparison with existing VLMs on general VLM benchmarks. “Recipe” denotes the adopted training recipe. “PT”, “SFT”, and “DT” denote the pre-training, supervised fine-tuning, and distillation training, respectively. “Complexity” denotes the model computation complexity with respect to the number of tokens. “# P.” denotes the number of total parameters. “# T.P.” denotes the percentage of trainable parameters (trainabletotal parametersparamters). The best performance is highlighted in bold and the second-best result is underlined.

Model LLM Backbone Vision Encoder Total Params Visual Tokens Output Tokens Speed (tokens/s) Total (s) LLaVA-Phi Phi2-2.7B CLIP ViT-L/14 3.1B 576 256 26.92 9.51

MobileVLM-3B LLaMA-2.7B CLIP ViT-L/14 3.1B 144 256 35.26 7.26 HoVLE 32-layer Transformer 2.6B 768 256 33.03 7.75

Cobra-3.5B Mamba-2.8B DINOv2 + SigLIP ViT-SO 3.5B 729 256 99.22 2.58 VisualRWKV-3B RWKV6-3B CLIP ViT-L/14 3.4B 577 256 41.34 6.19 mmMamba-linear 32-layer Mamba2 2.7B 768 256 132.43 1.93 mmMamba-hybrid 24-layer Mamba2 + 8-layer Transformer 2.7B 768 256 134.77 1.90

- Table 2: Inference efficiency comparison under same multimodal prompt and fixed decode length. We compare with VLMs of the similar parameter scale (3B) across encoder-based, decoder-only, quadratic-complexity, and linear-complexity. The results highlight the speed advantage of mmMamba-linear/hybrid. The benchmark recipe directly follows Cobra, and we report the results on the same single NVIDIA RTX 4090 GPU. Note that “Total Time” includes the time of both prefilling and decoding, and “Speed” = “Output Tokens” / “Total Time”.

which scrutinizes models with college-level multi-discipline reasoning tasks; and MM-Vet (Yu et al., 2023), which evaluates the model on 16 emergent tasks from core visual and linguistic capabilities. The visual question answering benchmarks comprise: TextVQA (Singh et al., 2019), which evaluates optical character recognition (OCR) capabilities and text-based reasoning; ScienceQA (Lu et al., 2022), which tests scientific image comprehension; and GQA (Hudson & Manning, 2019), which assesses real-world visual reasoning and compositional question answering.

For the specific score in the comparison, we report the MMEperception score as the MME score, MMB score is calculated on the MMBench-EN split, and the POPE score is calculated by averaging across its three categories.

#### 5.2. Main Comparison

In Tab. 1, we compare mmMamba with previous encoderbased and decoder-only VLMs on 9 popular benchmarks. We highlight the following findings:

- • mmMamba only performs distillation as the training recipe, which requires much lower training cost in two aspects: (1) dataset collection–unlike other methods require separate curated datasets for pre-training (PT) and supervised fine-tuning (SFT), our distillation recipe only needs a single SFT dataset; (2) trainable parameters–our method only updates 14.7% parameters for mmMamba-linear and 11.2% parameters for mmMamba-hybrid during training, while other methods require training most of parameters.
- • mmMamba-linear surpasses previous SOTA Transformer-based decoder-only VLM EVE-7B on 6/9 benchmarks (i.e., MME, MMB, POPE,

ID Stage1 Stage2 Stage3 MME POPE TextVQA SQA-I

Init Strategy MME POPE TextVQA SQA-I from scratch 1214.0 83.1 40.0 67.4

- 1 NAN NAN NAN NAN

- 2 ✓ 969.8 70.6 13.47 40.8

- 3 ✓ 1007.1 72.9 25.5 52.1

- 4 ✓ 1188.4 83.0 40.0 63.4

- 5 ✓ ✓ 1108.9 75.3 28.0 59.3

- 6 ✓ ✓ 1263.1 84.0 42.5 77.1

- 7 ✓ ✓ 1255.5 83.5 41.1 72.1

- 8 ✓ ✓ ✓ 1303.5 85.2 47.7 79.2 Table 3: Ablation for training stages.

inherit WQ,K,V 1222.6 84.0 41.9 73.3 inherit WQ,K,V + mimic 1303.5 85.2 47.7 79.2

#### Table 4: Ablation for parameter initialization.

Attention Layers MME POPE TextVQA SQA-I

- 0 1303.5 85.2 47.7 79.2

- 1 1304.3 85.5 48.0 79.3

- 2 1318.4 86.3 48.4 79.9 4 1329.1 86.8 51.5 82.8 8 1371.1 86.7 55.1 86.9

SEED, MM-Vet, ScienceQA), while matching the performance on the left 3 benchmarks with 2× fewer parameters. Even compared with encoder-based VLMs (e.g., MobileVLM-3B, LLaVA-phi), mmMamba-linear still demonstrates a comparable performance, while the computation complexity is reduced to linear complexity.

32 1433.5 87.6 66.0 94.8

Table 5: Ablation for the number of interleaved attention layers. “0” denotes mmMamba-pure, “8” denotes mmMamba-hybrid, “32” denotes the full Transformer model HoVLE.

- • mmMamba-linear matches the performance of recent encoder-based linear-complexity VLMs (VisualRWKV-3B and VL-Mamba-3B), while significantly outperforming them on the ScienceQA benchmark.
- • By interleaving with the Transformer layers, mmMamba-hybrid achieves improved performance on all benchmarks over mmMamba-linear, significantly narrowing the gap with the teacher Transformer-based HoVLE and outperforming the linear complexity encoder-based VLMs (VisualRWKV-3B and VL-Mamba-3B).

#### 5.3. Efficiency Analysis

Fixed prompt and fixed decode length. We directly follow the benchmark recipe of Cobra in Tab. 2, where we prompt the VLM model with the same example image and question “Describe the image specifically”, and set the number of output tokens to 256. We record the total time of the VLM model, which includes the image/text prompt prefilling time and the decoding time. The speed (tokens/s) is calculated by the number of output tokens (i.e., 256) divided by the total time. We compare our method with 3 transformer-based VLMs and 2 linear-complexity encoder-based VLMs of similar parameter scale. All the evaluations are conducted on the same single NVIDIA RTX 4090 GPU.

Thanks to the fixed hidden state rooted in linear-complexity modeling, mmMamba-linear/hybrid achieve nearly 4× faster inference speed than all the Transformer-based VLMs. mmMamba-linear/hybrid also outperforms the linear complexity encoder-based VLMs (Cobra-3.5B and VisualRWKV-3B) by a large margin (about 30 tokens/s and

- 3× faster, respectively) due to the simple decoder-only architecture.

Increasing context length. Long context processing has emerged as a crucial capability in modern VLMs, becoming increasingly important for high-resolution image/video understanding (Chen et al., 2023; Li et al., 2024a) and long chain-of-thought multimodal reasoning (Xu et al., 2024; Lightman et al., 2023; DeepSeek-AI et al., 2025; Muennighoff et al., 2025; Team et al., 2025), which often require processing sequences of thousands of tokens. To showcase the efficiency of the proposed mmMamba in this application, we compare our model with Transformer-based HoVLE in the same single NVIDIA RTX 4090 GPU. We report the GPU memory usage and the latency of the model for decoding the next token.

As shown in Fig. 1, thanks to the efficient implementation of FlashAttention2, HoVLE demonstrates stable and low latency under 4K token length. As the context token length reaches 8K and beyond, the latency and memory of HoVLE increase linearly with the token length due to the growing Key-Value cache, when the token length reaches 128K, HoVLE squeezes out of the GPU memory and fails to decode. On the contrary, mmMamba-linear exhibits low and stable latency and memory usage with increasing token length, and the inference cost of mmMambahybrid increases much slower than HoVLE, which can still decode at the 128K token length. Specifically, at the 103K token length, mmMamba-linear demonstrates 20.6× speedup compared to HoVLE and saves 75.8% GPU memory, while mmMamba-hybrid achieves 13.5× speedup and saves 60.2% GPU memory.

#### 5.4. Ablation Study

Stage importance. As shown in Tab. 3, direct weight transfer from Transformer to Mamba-2 without distillation (Sec. 4.1) lost the multi-modal understanding ability. By

### 6. Conclusion

Hybrid strategy MME POPE TextVQA SQA-I

Tail-stacked 1305.5 85.9 53.7 79.4 Head-stacked 1329.4 85.9 55.0 80.8

We presented mmMamba, a novel framework for building linear-complexity decoder-only VLMs with only moderate academic resources through the proposed distillation recipe, eliminating the need for pre-trained linear-complexity LLMs and vision encoders. Our recipe enables both pure linear and hybrid architectures, achieving competitive performance while significantly reducing computational costs. Experimental results demonstrate that mmMamba-linear matches or exceeds the performance of existing linear-complexity and quadratic-complexity VLMs, while mmMamba-hybrid further improves performance through flexible efficiencyperformance trade-offs with interleaved Transformer layers. At 103K tokens, mmMamba-linear achieves up to 20.6× speedup and 75.8% memory reduction compared to Transformer-based teacher HoVLE, while mmMambahybrid achieves 13.5× speedup and saves 60.2% GPU memory. These results validate the effectiveness of our distillation recipe for building linear-complexity decoder-only VLMs suitable for long-context applications.

Tail-interleaved 1308.3 86.1 55.0 86.5 Head-interleaved 1371.1 86.7 55.1 86.9

#### Table 6: Ablation for hybrid strategy.

progressively adding the designed distillation stages, the model’s performance is increasingly improved. When comparing ID-7 and ID-8, we can see that the proposed extra parameter distillation stage-1 decouples the optimization and eases the training, leading to a better alignment, with consistent improvements across all metrics (48 in MME, 1.7 in POPE, 6.6 in TextVQA, 7.1 in ScienceQA).

Parameter initialization. In Tab. 4, We compare with the “from scratch” strategy used in Phi-Mamba (Bick et al., 2024), which replace the trained Transformer layer with directly initialized Mamba-2 layer without inheriting the parameters, and the “inherit WQ,K,V ” strategy used in LoLCATs (Zhang et al., 2024a) and Mamba in the LLaMA (Wang et al., 2024a), which exploit the similarity and only inherit the parameters of WQ,K,V,O from Transformer to Mamba-2. The results validate the superiority of our proposed parameter initialization strategy, which should not only inherit the trained parameters but also initialize the extra introduced parameters by mimicking the original attention mechanism.

### References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Awadalla, A., Gao, I., Gardner, J., Hessel, J., Hanafy, Y., Zhu, W., Marathe, K., Bitton, Y., Gadre, S., Sagawa, S., et al. Openflamingo: An open-source framework for training large autoregressive vision-language models. arXiv preprint arXiv:2308.01390, 2023.

Hybrid architecture. The proposed distillation recipe is more flexible than the previous training recipe used in building linear-complexity encoder-based VLM, which requires the trained linear-complexity LLM and can not modify the architecture. As shown in Tab. 5, we can build a hybrid architecture with varied interleaved Transformer layers, enabling the flexible trade-off between performance and efficiency. By increasing the number of Transformer layers, the performance is gradually improved. The hybrid architecture with 24 Mamba-2 layers and 8 Transformer layers can achieve comparable performance with a minor decrease compared to the full Transformer model HoVLE.

Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023a.

Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., and Zhou, J. Qwen-vl: A frontier large visionlanguage model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023b.

Bavishi, R., Elsen, E., Hawthorne, C., Nye, M., Odena, A., Somani, A., and Ta¸sırlar, S. Introducing our multimodal models, 2023. URL https://www.adept. ai/blog/fuyu-8b.

Hybrid strategy. In Tab. 6, we explore specific hybrid strategies while fixing the number of interleaved Transformer layers to 8. We study 4 interleaving strategies: (1) Tail-stacked: stacking all 8 Transformer layers at the top of the network. (2) Head-stacked: stacking all 8 Transformer layers at the bottom of the network; (3) Tail-interleaved: interleaving a Transformer layer at the tail of every 4-layer block; (4) Head-interleaved: interleaving a Transformer layer at the head of every 4-layer block; The results demonstrate that the Head-interleaved strategy is the most effective, achieving the best performance across all metrics

Bi, X., Chen, D., Chen, G., Chen, S., Dai, D., Deng, C., Ding, H., Dong, K., Du, Q., Fu, Z., et al. Deepseek llm: Scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954, 2024.

Bick, A., Li, K. Y., Xing, E. P., Kolter, J. Z., and Gu, A. Transformers to ssms: Distilling quadratic knowledge to

subquadratic models. arXiv preprint arXiv:2408.10189, 2024.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in neural information processing systems, 33: 1877–1901, 2020.

Chen, G., Zheng, Y.-D., Wang, J., Xu, J., Huang, Y., Pan, J., Wang, Y., Wang, Y., Qiao, Y., Lu, T., et al. Videollm: Modeling video sequence with large language models. arXiv preprint arXiv:2305.13292, 2023.

- Chen, Y., Wang, X., Peng, H., and Ji, H. A single transformer for scalable vision-language modeling. arXiv preprint arXiv:2407.06438, 2024a.
- Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q., Zhu, X., Lu, L., et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24185–24198, 2024b.

Chu, X., Qiao, L., Lin, X., Xu, S., Yang, Y., Hu, Y., Wei, F., Zhang, X., Zhang, B., Wei, X., et al. Mobilevlm: A fast, reproducible and strong vision language assistant for mobile devices. arXiv preprint arXiv:2312.16886, 2023.

Dao, T. and Gu, A. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. arXiv preprint arXiv:2405.21060, 2024.

DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, J.-M., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., Zhang, X., Yu, X., Wu, Y., Wu, Z. F., Gou, Z., Shao, Z., Li, Z., Gao, Z., Liu, A., Xue, B., Wang, B.-L., Wu, B., Feng, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., Dai, D., Chen, D., Ji, D.-L., Li, E., Lin, F., Dai, F., Luo, F., Hao, G., Chen, G., Li, G., Zhang, H., Bao, H., Xu, H., Wang, H., Ding, H., Xin, H., Gao, H., Qu, H., Li, H., Guo, J., Li, J., Wang, J., Chen, J., Yuan, J., Qiu, J., Li, J., Cai, J., Ni, J., Liang, J., Chen, J., Dong, K., Hu, K., Gao, K., Guan, K., Huang, K., Yu, K., Wang, L., Zhang, L., Zhao, L., Wang, L., Zhang, L., Xu, L., Xia, L., Zhang, M., Zhang, M., Tang, M., Li, M., Wang, M., Li, M., Tian, N., Huang, P., Zhang, P., Wang, Q., Chen,

- Q., Du, Q., Ge, R., Zhang, R., Pan, R., Wang, R., Chen,
- R. J., Jin, R. L., Chen, R., Lu, S., Zhou, S., Chen, S., Ye,
- S., Wang, S., Yu, S., Zhou, S., Pan, S., Li, S. S., Zhou, S., Wu, S.-K., Yun, T., Pei, T., Sun, T., Wang, T., Zeng,

- W., Zhao, W., Liu, W., Liang, W., Gao, W., Yu, W.-X., Zhang, W., Xiao, W. L., An, W., Liu, X., Wang, X., Chen,
- X., Nie, X., Cheng, X., Liu, X., Xie, X., Liu, X., Yang, X., Li, X., Su, X., Lin, X., Li, X. Q., Jin, X., Shen, X.-C., Chen, X., Sun, X., Wang, X., Song, X., Zhou, X., Wang,

- X., Shan, X., Li, Y. K., Wang, Y. Q., Wei, Y. X., Zhang,
- Y., Xu, Y., Li, Y., Zhao, Y., Sun, Y., Wang, Y., Yu, Y., Zhang, Y., Shi, Y., Xiong, Y., He, Y., Piao, Y., Wang, Y., Tan, Y., Ma, Y., Liu, Y., Guo, Y., Ou, Y., Wang, Y., Gong,

- Y., Zou, Y.-J., He, Y., Xiong, Y., Luo, Y.-W., mei You, Y., Liu, Y., Zhou, Y., Zhu, Y. X., Huang, Y., Li, Y., Zheng,

- Y., Zhu, Y., Ma, Y., Tang, Y., Zha, Y., Yan, Y., Ren, Z., Ren, Z., Sha, Z., Fu, Z., Xu, Z., Xie, Z., guo Zhang, Z., Hao, Z., Ma, Z., Yan, Z., Wu, Z., Gu, Z., Zhu, Z., Liu,
- Z., Li, Z.-A., Xie, Z., Song, Z., Pan, Z., Huang, Z., Xu,

- Z., Zhang, Z., and Zhang, Z. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Diao, H., Cui, Y., Li, X., Wang, Y., Lu, H., and Wang, X. Unveiling encoder-free vision-language models. arXiv preprint arXiv:2406.11832, 2024.

Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Fang, Y., Wang, W., Xie, B., Sun, Q., Wu, L., Wang, X., Huang, T., Wang, X., and Cao, Y. Eva: Exploring the limits of masked visual representation learning at scale. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19358–19369, 2023.

Fang, Y., Sun, Q., Wang, X., Huang, T., Wang, X., and Cao, Y. Eva-02: A visual representation for neon genesis. Image and Vision Computing, 149:105171, 2024.

Gu, A. and Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

Hou, H., Zeng, P., Ma, F., and Yu, F. R. Visualrwkv: Exploring recurrent neural networks for visual language models. arXiv preprint arXiv:2406.13362, 2024.

Huang, W., Pan, J., Tang, J., Ding, Y., Xing, Y., Wang, Y., Wang, Z., and Hu, J. Ml-mamba: Efficient multi-modal large language model utilizing mamba-2. arXiv preprint arXiv:2407.19832, 2024.

Hudson, D. A. and Manning, C. D. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6700– 6709, 2019.

Javaheripi, M., Bubeck, S., Abdin, M., Aneja, J., Bubeck, S., Mendes, C. C. T., Chen, W., Del Giorno, A., Eldan, R., Gopi, S., et al. Phi-2: The surprising power of small language models. Microsoft Research Blog, 1(3):3, 2023.

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. d. l., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

Kasai, J., Peng, H., Zhang, Y., Yogatama, D., Ilharco, G., Pappas, N., Mao, Y., Chen, W., and Smith, N. A. Finetuning pretrained transformers into rnns. arXiv preprint arXiv:2103.13076, 2021.

Li, B., Wang, R., Wang, G., Ge, Y., Ge, Y., and Shan, Y. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023a.

Li, J., Li, D., Xiong, C., and Hoi, S. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International conference on machine learning, pp. 12888–12900. PMLR, 2022.

Li, J., Chen, D., Cai, T., Chen, P., Hong, Y., Chen, Z., Shen, Y., and Gan, C. Flexattention for efficient high-resolution vision-language models. In European Conference on Computer Vision, pp. 286–302. Springer, 2024a.

- Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W. X., and Wen, J.-R. Evaluating object hallucination in large visionlanguage models. arXiv preprint arXiv:2305.10355,

- 2023b.

Li, Z., Yang, B., Liu, Q., Ma, Z., Zhang, S., Yang, J., Sun, Y., Liu, Y., and Bai, X. Monkey: Image resolution and text label are important things for large multi-modal models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26763–26773,

- 2024b.

Liao, B., Wang, X., Zhu, L., Zhang, Q., and Huang, C. Vig: Linear-complexity visual sequence learning with gated linear attention. arXiv preprint arXiv:2405.18425, 2024.

Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. Advances in neural information processing systems, 36, 2024a.

Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pp. 216–233. Springer, 2024b.

Loshchilov, I. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P., and Kalyan, A. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022.

Mercat, J., Vasiljevic, I., Keh, S., Arora, K., Dave, A., Gaidon, A., and Kollar, T. Linearizing large language models. arXiv preprint arXiv:2405.06640, 2024.

Muennighoff, N., Yang, Z., Shi, W., Li, X. L., Fei-Fei, L., Hajishirzi, H., Zettlemoyer, L., Liang, P., Cand`es, E., and Hashimoto, T. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393, 2025.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., ElNouby, A., et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Peng, B., Alcaide, E., Anthony, Q., Albalak, A., Arcadinho, S., Biderman, S., Cao, H., Cheng, X., Chung, M., Grella, M., et al. Rwkv: Reinventing rnns for the transformer era. arXiv preprint arXiv:2305.13048, 2023.

Peng, B., Goldstein, D., Anthony, Q., Albalak, A., Alcaide, E., Biderman, S., Cheah, E., Du, X., Ferdinan, T., Hou, H., et al. Eagle and finch: Rwkv with matrixvalued states and dynamic recurrence. arXiv preprint arXiv:2404.05892, 2024.

Qiao, Y., Yu, Z., Guo, L., Chen, S., Zhao, Z., Sun, M., Wu, Q., and Liu, J. Vl-mamba: Exploring state space models for multimodal learning. arXiv preprint arXiv:2403.13600, 2024.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Rajbhandari, S., Rasley, J., Ruwase, O., and He, Y. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1–16. IEEE, 2020.

Rasley, J., Rajbhandari, S., Ruwase, O., and He, Y. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 3505–3506, 2020.

Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., and Rohrbach, M. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8317–8326, 2019.

Sun, Q., Fang, Y., Wu, L., Wang, X., and Cao, Y. Evaclip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023.

Tao, C., Su, S., Zhu, X., Zhang, C., Chen, Z., Liu, J.,

- Wang, W., Lu, L., Huang, G., Qiao, Y., et al. Hovle: Unleashing the power of monolithic vision-language models with holistic vision-language embedding. arXiv preprint arXiv:2412.16158, 2024.

Team, C. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

Team, K., Du, A., Gao, B., Xing, B., Jiang, C., Chen, C., Li, C., Xiao, C., Du, C., Liao, C., et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozi`ere, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. arXiv preprint arXiv:2307.09288, 2023b.

Trockman, A., Harutyunyan, H., Kolter, J. Z., Kumar, S., and Bhojanapalli, S. Mimetic initialization helps state space models learn to recall. arXiv preprint arXiv:2410.11135, 2024.

Vaswani, A. Attention is all you need. Advances in Neural Information Processing Systems, 2017.

Wang, J., Paliotta, D., May, A., Rush, A. M., and Dao, T. The mamba in the llama: Distilling and accelerating hybrid models. arXiv preprint arXiv:2408.15237, 2024a.

- Wang, X., Zhang, X., Luo, Z., Sun, Q., Cui, Y., Wang, J., Zhang, F., Wang, Y., Li, Z., Yu, Q., et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024b.

Xu, G., Jin, P., Hao, L., Song, Y., Sun, L., and Yuan, L. Llava-cot: Let vision language models reason step-bystep. arXiv preprint arXiv:2411.10440, 2024.

Yin, S., Fu, C., Zhao, S., Li, K., Sun, X., Xu, T., and Chen, E. A survey on multimodal large language models. arXiv preprint arXiv:2306.13549, 2023.

Yu, W., Yang, Z., Li, L., Wang, J., Lin, K., Liu, Z., Wang, X., and Wang, L. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.

Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9556–9567, 2024.

Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 11975–11986, 2023.

Zhang, M., Arora, S., Chalamala, R., Wu, A., Spector, B., Singhal, A., Ramesh, K., and R´e, C. Lolcats: On lowrank linearizing of large language models. arXiv preprint arXiv:2410.10254, 2024a.

Zhang, M., Bhatia, K., Kumbong, H., and R´e, C. The hedgehog & the porcupine: Expressive linear attentions with softmax mimicry. arXiv preprint arXiv:2402.04347, 2024b.

Zhao, H., Zhang, M., Zhao, W., Ding, P., Huang, S., and Wang, D. Cobra: Extending mamba to multi-modal large language model for efficient inference. arXiv preprint arXiv:2403.14520, 2024.

Zhu, D., Chen, J., Shen, X., Li, X., and Elhoseiny, M. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.

Zhu, L., Liao, B., Zhang, Q., Wang, X., Liu, W., and Wang, X. Vision mamba: Efficient visual representation learning with bidirectional state space model. In Forty-first International Conference on Machine Learning.

Zhu, L., Huang, Z., Liao, B., Liew, J. H., Yan, H., Feng, J., and Wang, X. Dig: Scalable and efficient diffusion models with gated linear attention. 2024a.

Zhu, Y., Zhu, M., Liu, N., Xu, Z., and Peng, Y. Llavaphi: Efficient multi-modal assistant with small language model. In Proceedings of the 1st International Workshop on Efficient Multimedia Computing under Limited, pp. 18–22, 2024b.

