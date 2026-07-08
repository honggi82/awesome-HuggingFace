# arXiv:2504.00502v2[cs.CV]3Nov2025

## ShortV: Efficient Multimodal Large Language Models by Freezing Visual Tokens in Ineffective Layers

Qianhao Yuan1,2, Qingyu Zhang1,2, Yanjiang Liu1,2, Jiawei Chen1,2, Yaojie Lu1,*, Hongyu Lin1, Jia Zheng1,∗, Xianpei Han1, Le Sun1 1Institute of Software, Chinese Academy of Sciences 2University of Chinese Academy of Sciences {yuanqianhao2024,zhangqingyu2024,liuyanjiang2021,chenjiawei2024, luyaojie,hongyu,zhengjia,xianpei,sunle}@iscas.ac.cn

### Abstract

Text Tokens Visual Tokens

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

ShortVLayer

ShortVLayer

ShortVLayer

DenseLayer

DenseLayer

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Multimodal Large Language Models (MLLMs) suffer from high computational costs due to their massive size and the large number of visual tokens. In this paper, we investigate layer-wise redundancy in MLLMs by introducing a novel metric, Layer Contribution (LC), which quantifies the impact of a layer’s transformations on visual and text tokens, respectively. The calculation of LC involves measuring the divergence in model output that results from removing the layer’s transformations on the specified tokens. Our pilot experiment reveals that many layers of MLLMs exhibit minimal contribution during the processing of visual tokens. Motivated by this observation, we propose ShortV, a training-free method that leverages LC to identify ineffective layers, and freezes visual token updates in these layers. Experiments show that ShortV can freeze visual token in approximately 60% of the MLLM layers, thereby dramatically reducing computational costs related to updating visual tokens. For example, it achieves a 50% reduction in FLOPs on LLaVA-NeXT-13B while maintaining superior performance. The code is publicly available at https://github.com/icip-cas/ShortV.

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

(a) ShortV

60.0

AveragePerformance

57.5

55.0

52.5

Model

50.0

LLaVA-1.5-13B (40 layers)

LLaVA-1.5-7B (32 layers)

47.5

0 5 10 15 20 25 30 35

# ShortV Layers

(b) Performance vs. the Number of ShortV Layers

Figure 1. (a) Illustration of ShortV. We identify ineffective layers for visual tokens and replace these layers with sparse ShortV layers. In ShortV layers, we freeze visual tokens, and eliminate computations related to updating them. ShortV improves MLLM efficiency in a training-free manner and involves no parameter updates. Notably, ShortV is compatible with token pruning methods, e.g. FastV. (b) Performance vs. the number of ShortV layers. Average Performance means a normalized average score on multiple benchmarks. ShortV can freeze visual tokens in approximately 60% of the MLLM layers with nearly no performance degradation.

### 1. Introduction

Large language models (LLMs) have achieved remarkable performance in natural language tasks [1, 12, 37, 47, 49]. Building upon LLMs, Multimodal Large Language Models (MLLMs) [31, 32, 38, 39] take a significant step towards understanding the real physical world by incorporating visual information into their processes. Typically, an MLLM consists of a visual encoder, a projector, and an LLM backbone. Most of them preprocess visual information through

a visual encoder, e.g. a CLIP-ViT [14, 41], and project the patch-level visual features into visual tokens through a projector. Then they concatenate visual and text tokens and feed them into the LLM backbone.

However, MLLMs face a substantial increase in compu-

*Corresponding authors.

tational overhead. This burden primarily stems from the large scale of the LLM backbones and the significantly extended length of the concatenated visual-text token sequences. To address this issue, Chen et al. [10] discovers significant token-wise redundancy in MLLMs. Based on this redundancy, they propose FastV, which identifies and prunes unimportant visual tokens in MLLMs to improve their efficiency.

Other than token-wise redundancy, in this paper, we reveal that MLLMs also exhibit significant layer-wise redundancy in processing visual tokens. Specifically, we propose Layer Contribution (LC), a metric that quantifies how much a layer’s transformations on certain tokens contribute to the model’s output. In the LC calculation of a layer, we freeze certain tokens in this layer, i.e. keep the hidden states of the tokens unchanged, and then compute the Kullback-Leibler (KL) divergence between the resulting model’s output logits and those of the original model. This metric provides a direct measure of a layer’s importance for certain tokens. By comparing LC scores on visual tokens and those on text tokens, we discover that MLLM layers are ineffective for visual tokens, and their transformations on visual tokens contribute minimal to the model’s output.

This phenomenon inspires us to propose ShortV, a simple but effective method to improve the efficiency of MLLMs. In ShortV, we first utilize the LC metric to identify layers least effective at transforming visual tokens, and then replace these layers with sparse ShortV layers. Within these sparse layers, visual tokens remain frozen, and the corresponding computations for updating them are eliminated, as shown in Figure 1a.

To validate the effectiveness of ShortV, we conduct evaluations across multiple benchmarks, including MME [15], MMBench [33], MMMU [55], MMStar [9], SEEDBench [25], GQA [19], and Flickr30K [40]. Figure 1b illustrates the correlation between the normalized average performance on these benchmarks and the number of replaced layers. As observed, ShortV can replace approximately 60% of MLLM layers without performance degradation. Unlike FastV and other token pruning methods, ShortV reduces computations of per visual token rather than reducing the number of visual tokens. Therefore, ShortV and token pruning methods are orthogonal and compatible. Furthermore, we demonstrate that combining ShortV and FastV can further enhance MLLM efficiency.

We summarize our contribution as follows.

- • We propose Layer Contribution (LC), a metric to quantify how much a layer’s transformations on specific tokens contribute to the model’s output.
- • Leveraging LC, we reveal significant redundancy in MLLM layers for visual tokens. Transformations on visual tokens in many layers contribute minimally and are thus ineffective.

• Based on the observation above, we propose ShortV, which improves MLLM efficiency by freezeing visual tokens in ineffective layers. ShortV can freeze visual tokens in approximately 60% of MLLM layers without performance degradation. Extensive experiments and ablation studies demonstrate ShortV’s effectiveness.

### 2. Layer Redundancy in MLLMs

In this section, we first introduce the background of layer redundancy in text-only LLMs, and then propose a metric to measure MLLM layer redundancy for different types of tokens. Next, we conduct a pilot experiment to investigate layer redundancy in MLLMs.

#### 2.1. Background

Typically, MLLMs are built on text-only LLMs. MLLMs employ pre-trained visual encoders, such as CLIP-ViT [14, 41], to convert images into visual features, and then use projectors to project them into visual tokens in the text token embedding space. The visual tokens are concatenated with text tokens and fed into the LLM backbones.

For text-only LLMs, Men et al. [36] identify notable redundancy across their layer. Some layers’ transformations on the hidden states of text token contribute minimally to the overall model functionality. Consequently, these layers are considered ineffective. Removing these transformations in approximately 25% of LLM layers has minimal impact on model outputs. Such redundancy mainly occurs in middle-to-deeper layers, whereas initial layers and the last layer remain critical to the model functionality. However, this pattern may not hold for MLLMs. Huang et al. [18] demonstrate a clear modality gap in the embedding space of current MLLMs, where visual and text tokens exhibit a uniform distribution within each modality but a significant distribution gap between modalities. Such a modality gap implies that MLLMs might adopt distinct computational patterns or strategies for processing visual and text tokens, potentially affecting how redundancy is distributed across layers. This raises several key questions: Are MLLM layers as ineffective for visual tokens as LLM layers are for text tokens? To what extent does layer redundancy exist in MLLMs? How is this redundancy distributed across different MLLM layers?

#### 2.2. Layer Contribution Metric

To investigate layer redundancy for certain tokens, we freeze these tokens within the investigated layer, i.e. keep hidden states of these tokens unchanged. To achieve this, we introduce sparse layers shown in Figure 2 for visual and text tokens, respectively. Based on these designs, we propose the Layer Contribution (LC) metric, which evaluates how much a layer’s transformations on certain tokens contribute to the model’s overall functionality. In the calcu-

[Figure 56]

[Figure 57]

layer with similar designs. Based on these, we define the ith layer’s Layer Contribution (LC) score for certain tokens X as the KL divergence between the output logits of the original model M and those of the model MXi where X are frozen in the i-th layer:

Text Tokens Visual Tokens

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

##### Transformer Layer

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

LCiX = KL(logits(M),logits(MXi )), (2)

Attention Mask

- (a) Sparse layer where visual tokens are frozen, used to investigate layer redundancy for visual tokens. Only text tokens function as queries and are passed through the FFN. We also denote this layer as ShortV layer.

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

Attention Mask

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

Transformer Layer

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

- (b) Sparse layer where text tokens are frozen, used to investigate layer redundancy for text tokens. Only visual tokens function as queries and are passed through the FFN.

here KL(·) denotes KL divergence. A lower LC score implies that the layer’s transformations on the tokens exhibit minimal contribution to the model’s output, suggesting that these transformations are ineffective.

Discussion: Why not use perplexity or cosine similarity as the metric to measure the importance of layers? Some work in text-only LLMs utilizes perplexity [23, 46] or cosine similarity [11, 17, 36] as metrics to measure the importance of each layer. For the former metric, instead of KL divergence, they measure the change in perplexity of the the models, and the layers causing minimal perplexity changes are deemed ineffective. This metric, however, is inadequate when measuring layer redundancy for visual tokens. We find that even if we do not feed visual tokens into the MLLMs, they can still generate reasonable responses, and the changes in perplexity of the MLLMs is relatively low. Nevertheless, they face significant performance degradation in vision language tasks when the visual information is absent. Thus, perplexity is not a reliable measure when evaluating layer redundancy for visual tokens.

- Figure 2. Sparse layers used to investigate layer redundancy for different tokens. To investigate layer redundancy for certain tokens, we freeze these tokens within the layer, i.e. keep hidden states of these tokens unchanged, and measure the divergence between the model’s output logits and those of the original model. We gray out the attention that does not need calculation.

lation of LC, we replace the investigated dense layer with the corresponding sparse layer, and compute the KullbackLeibler (KL) divergence between the resulting model’s output logits and those of the original dense model.

For the latter metric, the cosine similarity between the input and output of a certain layer is calculated. The hypothesis here is that the ineffective layers have less transformations on the hidden states of tokens, and therefore their inputs and outputs demonstrate higher similarities. However, in our evaluation on LLaVA-1.5 [31] models, the cosine similarity metric and LC differ in their measurement of the redundancy distribution across different layers. Compared with the LC metric, which directly measures the logits divergence between model outputs, cosine similarity consistently overestimates the redundancy of the shallow layers and underestimates the redundancy of the deep layers. We believe the reason behind this difference is that cosine similarity neglects the position of a layer within the model. Specifically, minor transformations of hidden states in the shallow layers can influence all subsequent layers, whereas transformations with similar extent in deeper layers tend to have less impact on overall model functionality.

Specifically, We assume the LLM backbone M has L layers, and each of them consists of a self-attention block and a feed-forward network (FFN). The input of the i-th layer at the j-th token position is Hij, and the corresponding output is Hij+1. The output of the last layer, i.e. the L-th layer, at the same position is HLj+1. The model M utilizes HL−+11 at the last token position to compute logits for nexttoken prediction through the language model head LMhead:

logits(M) = LMhead(HL−+11 ). (1)

To investigate how much the i-th layer’s transformations on certain tokens X contribute to the model functionality, we replace the i-th layer with a sparse layer where X are frozen, i.e. X’s hidden states remain unchanged in this sparse layer. The resulting model is denoted as MXi . In practice, X can be visual tokens V or text tokens T. As shown in Figure 2, we introduce sparse layers where V and T are frozen, respectively. In Figure 2a, we freeze the visual tokens in the sparse layer. Within the self-attention block of this layer, the visual tokens do not attend to other tokens, and only the text tokens function as queries. For the FFN of this layer, we simply do not pass the visual tokens through it. In Figure 2b, we freeze the text tokens in another sparse

#### 2.3. Ineffective MLLM Layers for Visual Tokens

We conduct a pilot experiment to investigate layer redundancy in LLaVA-1.5-7B and LLaVA-1.5-13B [31]. We first randomly sample 2,000 cases from a combination of two major vision language tasks, including caption (Flickr30K [40]) and visual question answering

- 100

- 101

Visual Tokens

Visual Tokens

- 100

- 101

Text Tokens

Text Tokens

10 1

LCscore

LCscore

10 1

10 2

10 2

10 3

10 3

10 4

0 5 10 15 20 25 30

0 5 10 15 20 25 30 35 40

Layer id

Layer id

(a) LLaVA-1.5-7B

(b) LLaVA-1.5-13B

- Figure 3. The Layer Contribution (LC) scores of LLaVA-1.5-7B and LLaVA-1.5-13B. A lower LC score implies that the layer’s transformations on the specified tokens are more ineffective. Layers are more ineffective for visual tokens than for text tokens, and freezing visual tokens in ineffective layers results in minimal output divergence from the original model.

(GQA [19]). Then we utilize these samples to calculate each MLLM layer’s average LC score for visual and text tokens, respectively. Figure 3 shows the results. We summarize our findings as follows.

[Figure 205]

ShortV Layer

First, for text tokens, middle to deeper layers are more ineffective, while the initial and last layers make more contributions to the MLLM functionality. These observations align with the layer redundancy distribution of text-only LLMs found in Men et al. [36], indicating that visual instruction tuning [30, 31] does not significantly alter the manner LLMs process text tokens.

Text

Attention Operation

&

Visual + Text Text

Second, for visual tokens, the initial and the deep layers, including the last one, exhibit higher redundancy than other layers, which is different from the distribution for text token. Notably, since the last layer’s transformations on visual tokens do not contribute to the model’s output, its LC score for visual tokens is always 0.

Figure 4. Details of ShortV layer. In this layer, only text tokens pass through the WQ and WO matrices and the FFN. The attention mask is same as that in Figure 2a, where visual tokens do not attend to other tokens, and only text tokens function as queries.

Third, layer redundancy shows an imbalance between visual and text tokens. Each layer’s LC score on visual tokens are lower than that on text tokens, which means that many layers’ transformations on visual tokens are ineffective, and freezing visual token in these ineffective layers results in minimal impact on the models’ output.

the model functionality. Based on this observation, we propose a direct method to enhance MLLM efficiency in a training-free manner: freezing visual tokens in ineffective layers. We denote this method as ShortV.

In ShortV, we replace the ineffective dense layers with sparse ShortV layers, where visual tokens are frozen. The ShortV layers are same as the sparse layer shown in 2a, and we illustrate their detailed architecture in Figure 4. In each ShortV layer, only text tokens are passed through the FFN and the WQ and WO matrices of the attention block, and the attention mask is same as that in Figure 2a, where visual tokens do not attend to other tokens, and only text tokens function as queries.

We attribute the different layer redundancy patterns for different modalities to the modality gap. The clear distribution gap of visual and text tokens result in the difference in how MLLMs process them. We hope these findings can provide insights into how MLLMs process visual and text tokens in different layers.

### 3. ShortV

ShortV has one parameter: the number of replaced layers, which we denote as N. First, we construct a tiny dataset, which contains a small number of samples from vision language tasks. Then we use this dataset to calculate each layer’s average LC score for visual tokens. Next, we sort the layers in ascending order according to LC scores,

#### 3.1. Freezing Visual Tokens in Ineffective Layers

As demonstrated in the previous section, we identify significant layer redundancy for visual tokens in MLLMs. Most layers’ transformations on visual tokens are ineffective for

FLOPs

Flickr30K Ratio Bench (val) EN CIDEr

SEED- MMMU

MMBench

Method TFLOPs

VQAv2 GQA

MME

MMStar

LLaVA-1.5-7B

Vanilla 8.5 100% 76.5 61.9 66.1 36.3 1510.7 64.1 33.7 74.9 FastV (K=2, R=50%) 4.9 58% 73.5 60.2 65.4 35.8 1475.6 64.3 32.4 67.5 VTW (K=16) 4.7 55% 66.3 55.1 66.2 36.1 1497.0 64.0 32.8 44.5 ShortV (Ours, N=19) 4.7 55% 75.7 60.9 66.2 36.2 1503.1 64.8 33.3 71.3

LLaVA-1.5-13B

Vanilla 16.6 100% 78.0 63.3 68.2 35.4 1531.3 68.9 36.1 79.6 FastV (K=2, R=50%) 9.4 57% 76.7 59.4 67.8 34.6 1506.6 68.3 35.9 73.4 VTW (K=20) 9.1 55% 75.3 60.6 68.2 34.9 1533.0 68.5 36.1 65.9 ShortV (Ours, N=24) 9.1 55% 77.2 62.0 68.0 35.8 1535.9 68.6 37.1 76.4

LLaVA-NeXT-7B

Vanilla 42.7 100% 80.0 64.1 70.2 36.4 1519.0 67.1 37.1 69.7 FastV (K=2, R=50%) 22.0 52% 79.5 63.0 69.6 35.1 1482.0 66.3 36.5 66.7 VTW (K=16) 21.8 51% 75.6 55.8 70.2 35.7 1518.2 67.1 37.6 38.4 ShortV (Ours, N=19) 21.6 51% 78.8 63.4 70.4 36.0 1525.1 67.2 37.8 65.7

LLaVA-NeXT-13B

Vanilla 81.8 100% 80.9 65.7 71.9 35.9 1570.0 69.3 39.9 66.7 FastV (K=2, R=50%) 42.1 51% 76.8 62.9 71.5 35.9 1546.4 68.5 39.6 66.0 VTW (K=20) 41.7 51% 77.0 61.5 71.8 34.8 1569.4 69.1 39.8 56.6 ShortV (Ours, N=24) 41.0 50% 79.7 63.6 71.8 36.2 1553.0 70.2 39.9 67.5

- Table 1. Comparison of various training-free methods for MLLM efficiency. FLOPs Ratio denotes the proportion of FLOPs retained after applying the corresponding method to improve MLLM efficiency, compared with the vanilla model.

and the layers with lower LC scores are more ineffective for visual tokens. Finally, we freeze visual tokens in the N layers with the lowest LC scores by replacing them with sparse ShortV layers, while keeping their original parameters.

ShortV is training-free and involves no parameter updates. It can be applied to various MLLMs for different vision language tasks. Notably, ShortV is orthogonal to and compatible with popular visual token pruning methods, e.g. FastV [10]. Visual token pruning directly reduces the number of visual tokens, while ShortV mitigates the computational overhead related to each visual token. This means that we can apply ShortV and token pruning at the same time to further improve MLLM efficiency.

#### 3.2. Computational Cost

We consider the computations of self-attention blocks and feed-forward networks (FFNs) in layers of the LLM backbone. Assume t is the number of text tokens, v is the number of visual tokens, h is the hidden state size, m is the intermediate size of the FFNs, the total FLOPs of one dense Transformer layer can be estimated as:

FLOPs = 2(t + v)(4h + 3m)h + 4(t + v)2h. (3) For one ShortV layer, the FLOPs can be calculated as:

FLOPs∗ = 2t(4h + 3m)h + 4vh2 + 4t(t + v)h. (4)

For the whole model, assume the LLM has L layers in total, ShortV selects N ineffective dense layers and replaces them with ShortV layers. The FLOPs of the original dense model are L × FLOPs, and the FLOPs of ShortV are calculated as (L − N) × FLOPs + N × FLOPs∗. The FLOPs ratio of ShortV and the original model is computed as:

(L − N) × FLOPs + N × FLOPs∗ L × FLOPs

. (5)

r =

### 4. Experiments

#### 4.1. Experimental Setups

Models. To validate the effectiveness of ShortV, we conduct experiments on popular open-source MLLMs, such as LLaVA-1.5-7B [31], LLaVA-1.5-13B, LLaVA-NeXT7B [32] and LLaVA-NeXT-13B. LLaVA-1.5 models process images with a 336×336 resolution and treat each image as 576 tokens. LLaVA-NeXT splits high-resolution images into subimages, and encode the subimages and down-sampled original images independently. This allows the models to scale the input to any arbitrary resolution, without performing positional embedding interpolation for ViTs [14]. LLaVA-NeXT scales the input image resolution to 4× and visual token number up to 5× compared with LLaVA-1.5, i.e. 2880 tokens for each image.

# ShortV

FLOPs

MMBench MMMU

SEED-

Flickr30K

Avg. Per. Layers (N) Ratio EN (val) Bench CIDEr

TFLOPs

MME

MMStar

GQA

LLaVA-1.5-7B (32 layers) 0 8.5 100% 1510.7 64.1 36.3 33.7 66.1 61.9 74.9 58.9 100.0 8 6.9 81% 1508.6 64.3 36.0 33.8 66.2 61.4 74.5 58.8 99.8 16 5.3 62% 1487.0 64.9 36.1 33.3 65.7 61.0 72.8 58.3 99.0 19 4.7 55% 1503.1 64.8 36.2 33.3 66.2 60.9 71.3 58.3 99.0 24 3.7 44% 1341.7 60.7 34.1 33.4 62.5 58.3 64.2 54.3 92.2

LLaVA-1.5-13B (40 layers) 0 16.6 100% 1531.3 68.9 35.4 36.1 68.2 63.3 79.6 61.2 100.0 8 14.1 85% 1521.9 68.6 35.6 36.0 68.2 63.0 79.0 60.9 99.5 16 11.6 70% 1534.9 68.6 36.3 36.2 68.0 62.9 78.5 61.0 99.7 24 9.1 55% 1535.9 68.6 35.8 37.1 68.0 62.0 76.4 60.7 99.2 32 6.6 40% 1298.8 64.5 33.6 36.0 63.2 59.3 68.4 55.7 91.0

LLaVA-NeXT-7B (32 layers)

0 42.7 100% 1519.0 67.1 36.4 37.1 70.2 64.1 69.7 60.1 100.0 8 33.8 79% 1515.1 67.2 36.6 36.9 70.2 64.1 70.0 60.1 100.0

16 24.9 58% 1476.8 67.2 36.2 37.3 70.2 63.5 67.8 59.4 98.8 19 21.6 51% 1525.1 67.2 36.0 37.8 70.4 63.4 65.7 59.5 99.0 24 16.0 37% 1504.1 65.4 36.4 36.0 68.1 60.5 64.9 58.1 96.7

LLaVA-NeXT-13B (40 layers) 0 81.8 100% 1570.0 70.5 35.9 39.9 71.9 65.7 66.7 61.3 100.0 8 68.2 83% 1552.4 70.6 35.0 39.6 71.9 65.1 66.9 61.0 99.5 16 54.6 67% 1561.0 70.1 35.0 39.7 71.9 64.8 66.9 60.9 99.3 24 41.0 50% 1553.0 70.2 36.2 39.9 71.8 63.6 67.5 61.0 99.5 32 27.5 34% 1468.4 65.8 35.2 38.9 69.3 60.5 58.5 57.4 93.6

- Table 2. Performance vs. Efficiency Balance of ShortV under different configurations. # ShortV Layers (N): the number of ShorV layers, Avg.: a normalized average score on the benchmarks, Per.: the relative performance retention compared with the vanilla models.

Baselines. To evaluate the effectiveness of ShortV, which improves MLLM efficiency in a training-free manner, we compare it with popular training-free methods for MLLM efficiency, such as FastV [10] and VTW [29]. FastV drops visual tokens by a percentage of R after the K-th layer in the forward process of input tokens. It computes the average attention score one token received from all other tokens as the importance criterion to select pruned tokens. VTW drops all visual tokens after the K-th layer, enabling only text tokens to engage in the subsequent layers. We use the default settings for the baselines as in their original papers. Specifically, for FastV, K=2 and R=50%. For VTW, K=16 for 7B models and K=20 for 13B models.

#### 4.2. Main Results

In this section, we conduct experiments to compare ShortV with the baselines. The results are shown in Table 1. We perform evaluation on multiple popular vision language benchmarks, including MME [15], MMBench [33], MMMU (val) [55], MMStar [9], SEED-Bench [25], VQAv2 [16], and GQA [19]. We manually choose the number of ShortV layers N to maintain a similar or lower FLOPs ratio compared with the baselines. The FLOPs ratio of ShortV is calculated according to Equation 5, where we set the number of text tokens to 64. Specifically, we

choose N=19 for the 7B models, and N=24 for the 13B models. As shown in Table 1, our ShortV achieves comparable or superior performance across multiple benchmarks compared with the baselines.

#### 4.3. Balance between Efficiency and Performance

In this section, we conduct an experiment to investigate the impact of ShortV’s parameter N, which denotes the number of ShortV layers. The experimental results are presented in Table 2. To facilitate intuitive comprehension, we plot the correlation between the normalized average score on benchmarks and the number of ShortV layers in Figure 1b.

We observe that ShortV can freeze visual tokens in approximately 60% of the MLLM layers while preserving superior performance. As N continues to increase, both 7B and 13B models can maintain more than 90% performance when the hidden states of visual tokens remain unchanged in about 80% of the layers. These results are significantly different from those on text-only LLMs. For LLMs, Men et al. [36] remove transformations on text tokens in approximately 25% of the LLM layers, and this results in about 10% performance degradation on language benchmarks. As the number of layers increases, the performance of LLMs rapidly declines. These differences align with our observation in Section 2.3 that layers are more ineffective for visual

# ShortV Layers (N) 0 8 16 19 24

LLaVA-1.5-7B 1.00× 1.13× 1.23× 1.30× 1.40× LLaVA-NeXT-7B 1.00× 1.15× 1.35× 1.44× 1.64×

- Table 3. Inference speedups over the vanilla models, based on the 7B models. We conduct this test on a single GPU.

# ShortV Layers (N) 0 8 16 24 32

LLaVA-1.5-13B 1.00× 1.13× 1.24× 1.39× 1.50× LLaVA-NeXT-13B 1.00× 1.13× 1.30× 1.52× 1.84×

- Table 4. Inference speedups over the vanilla models, based on the 13B models. We conduct this test on a single GPU.

Method

FLOP MMBench MMMU SEEDRatio EN (val) Bench GQA

Vanilla 100% 64.0 36.3 66.1 61.9 FastV 58% 64.3 35.8 65.4 60.2 ShortV 55% 64.8 36.2 66.2 60.9 ShortV+FastV 29% 64.2 37.1 65.1 59.3

- Table 5. ShortV is compatible with FastV, and applying both at the same time can further enhance MLLM efficiency. This experiment is based on LLaVA-1.5-7B.

tokens than for text tokens.

In addition to the theoretical FLOPs ratios, we provide the speedups on real hardware using different settings, as shown in Table 3 and Table 4. To get rid of the influence of different output sequence lengths, we use the first token latency to calculate the speedups. We utilize the MMMU dataset for the latency test. For comparison, we note that FastV [10] with its default setting, i.e. K=2 and R=50%, achieves a 1.31× speedup over the vanilla LLaVA-1.5-13B model. In contrast, our ShortV with its default parameter, i.e. N=24, achieves a greater speedup of 1.39×.

#### 4.4. Orthogonal to Token Pruning

In this section, we demonstrate that ShortV is orthogonal to and compatible with visual token pruning, e.g. FastV [10]. FastV identifies R% unimportant visual tokens and drops them after the K-th layer in the forward process of input tokens. We apply FastV to ShortV, which already replaces N ineffective layers for visual tokens with ShortV layers. We use the default settings for FastV and ShortV in this experiment, i.e. K=2 and R=50% for FastV, and N=19 for ShortV. We employ LLaVA-1.5-7B as the vanilla model. The experimental results in Table 5 demonstrate that ShortV is compatible with FastV and that the application of both can further improve MLLM efficiency.

FLOP MMBench MMMU SEEDRatio EN (val) Bench GQA

Strategy

Vanilla 100% 64.0 36.3 66.1 61.9 Random 55% 58.4 33.6 60.5 56.1 Cosine Sim. 55% 60.8 34.2 62.7 59.5 LC (Ours) 55% 64.8 36.2 66.2 60.9

Table 6. Ablation on strategies to select replaced layers, based on LLaVA-1.5-7B. “Random” denotes randomly selecting 19 layers and freezing visual tokens in them. “Cosine Sim.” denotes using cosine similarity to select ineffective layers for visual tokens.

#### 4.5. Ablation Studies

Ablation on strategies to select replaced layers. In this paragraph, we perform an ablation experiment on LLaVA1.5-7B to investigate the impact of strategies for selecting which layers to replace. ShortV selects ineffective layers for visual tokens, and replace them with ShortV layers. To identify which layers are ineffective, we utilize the LC metric introduced in Section 2. In contrast, previous work [36] on text-only LLMs uses a metric based on cosine similarity. It calculates the average cosine similarity between the inputs and outputs of each layer. The layers with higher cosine similarities are deemed more ineffective. To make a comparison between this cosine similarity metric and our LC metric, we calculate each layer’s cosine similarity between the input hidden states and output hidden states of visual tokens, and select the same number of layers with the highest cosine similarities. We show the comparison in Table 6. We also include the results of another baseline, ShortV (Random), where visual tokens are frozen in the same number of randomly selected layers. These results clearly demonstrate that our LC metric performs better than cosine similarity in identifying ineffective MLLM layers for visual tokens, and ShortV based on the LC metric achieves performance comparable to the vanilla model. In contrast, ShortV based on the cosine similarity metric cannot match the performance of the vanilla model, although it outperforms the baseline with randomly selected layers.

Ablation on frozen tokens. In this paragraph, we conduct an ablation study on LLaVA-1.5-7B to investigate the impact of freezing different types of tokens. In Section 2, we demonstrate that MLLM layers are ineffective for visual tokens, as measured by the LC metric. Motivated by this observation, ShortV freezes visual tokens in ineffective layers. In Table 7, we compare our method with the strategies of freezing other tokens. In the experiment detailed in line (a), we utilize the LC metric to identify 19 ineffective layers for text tokens, and freeze text tokens in them. Despite having fewer frozen tokens, we can observe that this strategy results in significant performance declines compared with our method, which freezes visual tokens rather than text to-

MMBench MMMU SEED-

Frozen Tokens

EN (val) Bench GQA None (Vanilla) 64.0 36.3 66.1 61.9

- (a) Text 2.1 23.7 8.9 2.9
- (b) Text+Visual 1.3 26.6 0.8 0.0
- (c) Random 1.5 22.9 5.5 2.3
- (d) Visual (Ours) 64.8 36.2 66.2 60.9

Table 7. Ablation on frozen tokens, based on LLaVA-1.5-7B. (a) identifying 19 ineffective layers for text tokens and freezing text tokens in them. In lines (b) and (c), we select ineffective layers for all tokens. line (b) involves freezing all input tokens in them, whereas line (c) denotes randomly freezing the same number of tokens as the visual tokens.

kens. These experimental results align with our findings in Section 2 that MLLM layers are more ineffective for visual tokens than for text tokens. In lines (b) and (c), we first calculate each layer’s average LC score for all tokens, including visual and text tokens, and then select 19 ineffective layers. Next, in the experiment corresponding to line (b), we freeze all input tokens in these layers. In line (c), we freeze random input tokens, and the number of frozen tokens matches that of the visual tokens. As a result, the computational overhead associated with line (c) is the same as that of our method in line (d). We can find that freezing tokens other than visual tokens leads to substantial performance degradation in vision-language tasks. These ablations demonstrate the effectiveness of our ShortV.

### 5. Related Work

#### 5.1. Multimodal Large Language Models

Built upon Large Language Models (LLMs) [1, 12, 37, 47, 49], Multimodal Large Language Models (MLLMs) [4, 7, 35, 48, 52, 57] have made significant progress in processing and understanding the visual world. Typically, they use a decoder-only architecture. Specifically, they utilize visual encoders [14, 21, 34, 41, 56] to convert input visual information into visual features and then use projectors to project these visual features into visual tokens. These visual tokens are then concatenated with text tokens and fed into the LLM backbones. Current MLLMs use hundreds to thousands of visual tokens to represent a single image, significantly increasing the length of the token sequences. For instance, the LLaVA-1.5 models [31] transform each image with 336×336 resolution into 576 tokens. For images with higher resolutions, the LLaVA-NeXT models [32] process images into up to 2,880 visual tokens, and SPHINX2k [28] divides one image into nine subimages, resulting in 2,890 visual tokens. Applying LLMs to such large numbers of visual tokens incurs substantial computational costs. In this paper, we introduce ShortV to enhance the efficiency of MLLMs by reducing the computational overhead associ-

ated with visual tokens.

#### 5.2. Efficient LLMs and MLLMs

For LLMs, previous studies [36, 46] find that layers in LLMs are ineffective for text tokens. They remove computations in about 25% of the layers, while preserving approximately 90% of the performance. LaCo [50] utilizes layer merging for efficient LLMs.

To address the computational inefficiency of MLLMs, previous methods [10, 29, 42, 44, 45, 54] have primarily focused on two aspects: efficient model architecture and visual token compression. Among efficient model architectures, cross-attention-based models [2, 3, 8, 20] insert gated cross-attention layers within LLM layers for visual perception, but previous studies [13, 24] demonstrate that this architecture performs worse than the decoder-only architecture in the same settings. Instead of inserting cross-attention layers, mPLUG-Owl3 [51] and Vamba [42] introduce crossattention operations in parallel with self-attention. In contrast, SAISA [54] introduces NAAViT self-attention blocks, which incorporate multimodal cross-attention into the original self-attention operations of the LLMs, and reuse the parameters of self-attention blocks. The design of ShortV layers is inspired by NAAViT. Differently, in ShortV layers, visual tokens also skip their FFNs.

Visual token compression methods improve MLLM efficiency in both training-based [5, 6, 22, 26, 27] and trainingfree [10, 43, 53] manners. FastV [10] reveals token-wise redundancy, and it removes unimportant tokens during inference. In this paper, we reveal layer-wise redundancy in MLLMs. Layers in MLLMs are much more ineffective for visual tokens than for text token. Therefore, we can freeze visual tokens in approximately 60% of the MLLM layers with minimal performance degradation. Unlike previous methods for MLLM efficiency, ShortV does not reduce the number of visual tokens but instead decreases the computational costs of processing each token. ShortV is trainingfree and orthogonal to token compression. We demonstrate that ShortV is compatible with FastV, allowing for simultaneous application to further enhance MLLM efficiency.

### 6. Conclusion

In this paper, we explore the layer-wise redundancy in MLLMs. We discover that layers in MLLMs are more ineffective for visual tokens than for text tokens. MLLM layers’ transformations on visual tokens have a minimal impact on the MLLM output. Motivated by this observation, we propose ShortV, a training-free method to enhance MLLM efficiency. ShortV utilizes our proposed LC metric to select ineffective layers for visual tokens, and freezes visual tokens in these layers. It can freeze visual tokens in about 60% of the layers while preserving superior performance.

### 7. Acknowledgment

We sincerely thank the reviewers for their insightful comments and valuable suggestions. This work was supported by Beijing Natural Science Foundation (L243006), Beijing Municipal Science and Technology Project (Nos. Z231100010323002), the Natural Science Foundation of China (No. 62306303, 62476265) and the Basic Research Program of ISCAS (Grant No. ISCAS-ZD-202401).

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 1, 8

- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736,

2022. 8

- [3] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. Openflamingo: An opensource framework for training large autoregressive visionlanguage models. arXiv preprint arXiv:2308.01390, 2023. 8
- [4] Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, and Sa˘gnak Ta¸sırlar. Introducing our multimodal models, 2023. 8
- [5] Junbum Cha, Wooyoung Kang, Jonghwan Mun, and Byungseok Roh. Honeybee: Locality-enhanced projector for multimodal llm. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13817–13827, 2024. 8
- [6] Wenhao Chai, Enxin Song, Yilun Du, Chenlin Meng, Vashisht Madhavan, Omer Bar-Tal, Jeng-Neng Hwang, Saining Xie, and Christopher D Manning. Auroracap: Efficient, performant video detailed captioning and a new benchmark. arXiv preprint arXiv:2410.03051, 2024. 8
- [7] Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny. Minigpt-v2: large language model as a unified interface for vision-language multi-task learning. arXiv preprint arXiv:2310.09478, 2023. 8
- [8] Kaibing Chen, Dong Shen, Hanwen Zhong, Huasong Zhong, Kui Xia, Di Xu, Wei Yuan, Yifei Hu, Bin Wen, Tianke Zhang, et al. Evlm: An efficient vision-language model for visual understanding. arXiv preprint arXiv:2407.14177,

2024. 8

- [9] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330,

2024. 2, 6

- [10] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In European Conference on Computer Vision, pages 19–35. Springer, 2024. 2, 5, 6, 7, 8
- [11] Xiaodong Chen, Yuxuan Hu, Jing Zhang, Yanling Wang, Cuiping Li, and Hong Chen. Streamlining redundant layers to compress large language models. arXiv preprint arXiv:2403.19135, 2024. 3, 12
- [12] Wei-Lin Chiang, Zhuohan Li, Ziqing Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2(3):6, 2023. 1, 8
- [13] Wenliang Dai, Nayeon Lee, Boxin Wang, Zhuolin Yang, Zihan Liu, Jon Barker, Tuomas Rintamaki, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nvlm: Open frontier-class multimodal llms. arXiv preprint arXiv:2409.11402, 2024. 8
- [14] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 1, 2, 5, 8
- [15] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models, 2024. 2, 6
- [16] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913, 2017. 6
- [17] Shwai He, Guoheng Sun, Zheyu Shen, and Ang Li. What matters in transformers? not all attention is needed. arXiv preprint arXiv:2406.15786, 2024. 3, 12
- [18] Qidong Huang, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Jiaqi Wang, Dahua Lin, Weiming Zhang, and Nenghai Yu. Deciphering cross-modal alignment in large vision-language models with modality integration rate. arXiv preprint arXiv:2410.07167, 2024. 2
- [19] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019. 2, 4, 6
- [20] IDEFICS. Introducing idefics: An open reproduction of state-of-the-art visual language model. https:// huggingface.co/blog/idefics, 2023. 8
- [21] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, 2021. 8

- [22] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In International conference on machine learning, pages 4651–4664. PMLR, 2021. 8
- [23] Bo-Kyeong Kim, Geonmin Kim, Tae-Ho Kim, Thibault Castells, Shinkook Choi, Junho Shin, and Hyoung-Kyu Song. Shortened llama: Depth pruning for large language models with comparison of retraining methods. arXiv preprint arXiv:2402.02834, 2024. 3
- [24] Hugo Lauren¸con, L´eo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models? Advances in Neural Information Processing Systems, 37:87874–87907, 2025. 8
- [25] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 2, 6
- [26] Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song Wang, Jie Qin, Jianke Zhu, and Lei Zhang. Tokenpacker: Efficient visual projector for multimodal llm. arXiv preprint arXiv:2407.02392, 2024. 8
- [27] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In European Conference on Computer Vision, pages 323–340. Springer, 2024. 8
- [28] Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, et al. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575, 2023. 8
- [29] Zhihang Lin, Mingbao Lin, Luxi Lin, and Rongrong Ji. Boosting multimodal large language models with visual tokens withdrawal for rapid inference. arXiv preprint arXiv:2405.05803, 2024. 6, 8
- [30] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 4
- [31] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024. 1, 3, 4, 5, 8
- [32] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 1, 5, 8
- [33] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024. 2, 6
- [34] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11976–11986,

2022. 8

- [35] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, et al. Deepseek-vl: towards real-world visionlanguage understanding. arXiv preprint arXiv:2403.05525,

2024. 8

- [36] Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, and Weipeng Chen. Shortgpt: Layers in large language models are more redundant than you expect, 2024. URL https://arxiv. org/abs/2403.03853, 2024. 2, 3, 4, 6, 7, 8
- [37] AI Meta. Introducing meta llama 3: The most capable openly available llm to date. Meta AI, 2(5):6, 2024. 1, 8
- [38] OpenAI. Gpt-4v(ision) system card. https://cdn. openai.com/papers/GPTV_System_Card.pdf,

2023. 1

- [39] OpenAI. Introducing gpt-4o: our fastest and most affordable flagship model. https://platform.openai.com/ docs/guides/vision, 2024. Accessed: 2024-05-26. 1
- [40] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In Proceedings of the IEEE international conference on computer vision, pages 2641–2649, 2015. 2, 3
- [41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 1, 2, 8
- [42] Weiming Ren, Wentao Ma, Huan Yang, Cong Wei, Ge Zhang, and Wenhu Chen. Vamba: Understanding hourlong videos with hybrid mamba-transformers. arXiv preprint arXiv:2503.11579, 2025. 8
- [43] Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. arXiv preprint arXiv:2403.15388,

2024. 8

- [44] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18221–18232, 2024. 8
- [45] Enxin Song, Wenhao Chai, Tian Ye, Jenq-Neng Hwang, Xi Li, and Gaoang Wang. Moviechat+: Question-aware sparse memory for long video question answering. arXiv preprint arXiv:2404.17176, 2024. 8
- [46] Jiwon Song, Kyungseok Oh, Taesu Kim, Hyungjun Kim, Yulhwa Kim, and Jae-Joon Kim. Sleb: Streamlining llms through redundancy verification and elimination of transformer blocks. arXiv preprint arXiv:2402.09025, 2024. 3, 8
- [47] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 1, 8

- [48] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Song XiXuan, et al. Cogvlm: Visual expert for pretrained language models. Advances in Neural Information Processing Systems, 37:121475–121499, 2025. 8
- [49] Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, et al. Baichuan 2: Open large-scale language models. arXiv preprint arXiv:2309.10305, 2023. 1, 8
- [50] Yifei Yang, Zouying Cao, and Hai Zhao. Laco: Large language model pruning via layer collapse. arXiv preprint arXiv:2402.11187, 2024. 8
- [51] Jiabo Ye, Haiyang Xu, Haowei Liu, Anwen Hu, Ming Yan, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplugowl3: Towards long image-sequence understanding in multimodal large language models. In The Thirteenth International Conference on Learning Representations, 2024. 8
- [52] Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, and Fei Huang. mplugowl2: Revolutionizing multi-modal large language model with modality collaboration. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition, pages 13040–13051, 2024. 8
- [53] Xubing Ye, Yukang Gan, Xiaoke Huang, Yixiao Ge, Ying Shan, and Yansong Tang. Voco-llama: Towards vision compression with large language models. arXiv preprint arXiv:2406.12275, 2024. 8
- [54] Qianhao Yuan, Yanjiang Liu, Yaojie Lu, Hongyu Lin, Ben He, Xianpei Han, and Le Sun. Saisa: Towards multimodal large language models with both training and inference efficiency. arXiv preprint arXiv:2502.02458, 2025. 8
- [55] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556– 9567, 2024. 2, 6
- [56] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023. 8
- [57] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 8

### A. Limitations and Future Work

Despite the effectiveness of ShortV, It remains a coarsegrained method, and there are several directions to improve it. First, ShortV treats each layer as a whole, whereas LLM layers have a more fine-grained structure, including attention blocks and FFNs, and He et al. [17] reveal that they exhibit different levels of redundancy in text-only LLMs. Freezing visual tokens in different proportions of attention blocks and FFNs could achieve a more favorable balance between performance and efficiency. Second, Chen et al. [11] uses a small network to update tokens in ineffective layers of LLMs, which is also a promising path to improve the performance of ShortV.

### B. Replaced Layers

For the LC metric calculation to select the replaced layers, we randomly sample 40 cases from GQA and Flickr30K, with 20 from each of them. Layers are replaced with ShortV layers in ascending order based on their LC values, starting from the lowest and moving to the highest. In Table 8, we list the layer ids of replaced layers in default ShortV.

Model Replaced Layers LLaVA-1.5-7B 31, 29, 30, 28, 0, 26, 27, 25, 24, 22,

- 23, 21, 2, 3, 20, 18, 17, 12, 19

LLaVA-1.5-13B 39, 32, 28, 36, 27, 37, 29, 30, 1, 38, 25, 31, 2, 26, 23, 34, 0, 33, 35, 22,

- 24, 21, 20, 17

LLaVA-NeXT-7B 31, 29, 30, 28, 26, 27, 22, 24, 21, 23,

- 25, 20, 19, 17, 18, 15, 12, 0, 2

LLaVA-NeXT-13B 39, 32, 29, 36, 27, 30, 37, 23, 25, 31,

- 26, 2, 28, 22, 33, 35, 34, 24, 38, 21, 20, 18, 1, 17

Table 8. Replaced layers for different MLLMs.

### C. Qualitative Results

In this section, we use the qualitative examples in the original LLaVA-1.5 paper to compare the responses of LLaVA1.5-13B before and after applying ShortV. We reference the responses of the vanilla model from the original LLaVA1.5 paper. We utilize the default parameter, i.e. N=24, for the model with ShortV. In Table 9 and 10, we can see that the model with ShortV generates responses similar to the vanilla model, despite freezing visual tokens in 24 MLLM layers. These examples validate the capability of ShortV in enhancing MLLM efficiency while preserving performance.

[Figure 206]

User Here’s a formula for a Stable Diffusion image prompt: a cartoon-style image of [adjective] [subject] [haircut] [clothing] [doing action], [color scheme], detailed, trending on artstation. Generate the prompt for this image.

w/o ShortV A cartoon-style image of a young girl with a bob haircut, wearing a white kimono, holding a stuffed animal, and posing for a picture, with a color scheme of pink and white, detailed, and trending on artstation.

w/ ShortV A cartoon-style image of a young girl with a bob haircut, wearing a pink and white kimono, holding a stuffed animal, and posing for a picture, with a color scheme of pink and white, detailed, and trending on artstation.

- Table 9. Responses comparison between the vanilla LLaVA-1.513B model and the LLaVA-1.5-13B model with ShortV (N=24), on constrained Stable Diffusion prompt generation.

[Figure 207]

User If there are factual errors in the questions, point it out; if not, proceed to answering the question. What’s happening in the desert?

w/o ShortV There are no deserts in the image. The image features a beach with palm trees, a city skyline, and a large body of water.

w/ ShortV There are no deserts in the image. The image features a beach with a large body of water, a city skyline, and palm trees.

- Table 10. Responses comparison between the vanilla LLaVA-1.513B model and the LLaVA-1.5-13B model with ShortV (N=24), on a tricky question.

### D. ShortV under Different Settings

In this section, we provide comprehensive ShortV performance under different settings, i.e. different numbers of replaced layers. The results are shown in Table 11.

# ShortV

FLOPs

MMBench MMMU

SEED-

Flickr30K

Avg. Per. Layers (N) Ratio EN (val) Bench CIDEr

TFLOPs

MME

MMStar

GQA

LLaVA-1.5-7B (32 layers)

0 8.5 100% 1510.7 64.1 36.3 33.7 66.1 61.9 74.9 58.9 100.0 4 7.7 91% 1507.5 64.1 36.6 33.5 66.2 61.9 74.7 58.9 100.0 8 6.9 81% 1508.6 64.3 36.0 33.8 66.2 61.4 74.5 58.8 99.8

12 6.1 72% 1495.2 64.2 36.2 34.0 66.2 61.2 74.1 58.7 99.7 16 5.3 62% 1487.0 64.9 36.1 33.3 65.7 61.0 72.8 58.3 99.0

- 19 4.7 55% 1503.1 64.8 36.2 33.3 66.2 60.9 71.3 58.3 99.0
- 20 4.5 53% 1466.8 63.4 35.3 34.7 65.2 60.4 70.7 57.6 97.8 24 3.7 44% 1341.7 60.7 34.1 33.4 62.5 58.3 64.2 54.3 92.2 28 2.9 34% 1079.0 57.9 31.0 30.2 56.0 52.0 53.6 47.8 81.2

LLaVA-1.5-13B (40 layers) 0 16.6 100% 1531.3 68.9 35.4 36.1 68.2 63.3 79.6 61.2 100.0 4 15.3 92% 1521.6 68.6 35.8 36.5 68.2 63.3 79.4 61.1 99.8 8 14.1 85% 1521.9 68.6 35.6 36.0 68.2 63.0 79.0 60.9 99.5

12 12.8 77% 1521.9 68.6 35.9 36.2 68.1 62.9 78.9 61.0 99.7 16 11.6 70% 1534.9 68.6 36.3 36.2 68.0 62.9 78.5 61.0 99.7 20 10.3 62% 1533.0 68.6 36.1 36.8 68.0 62.4 77.5 60.9 99.5 24 9.1 55% 1535.9 68.6 35.8 37.1 68.0 62.0 76.4 60.7 99.2 28 7.8 47% 1417.6 65.5 34.6 35.9 65.4 60.8 74.9 58.3 95.3 32 6.6 40% 1298.8 64.5 33.6 36.0 63.2 59.3 68.4 55.7 91.0 36 5.3 32% 1259.6 62.9 33.2 34.9 62.5 58.7 62.8 54.0 88.2

LLaVA-NeXT-7B (32 layers)

0 42.7 100% 1519.0 67.1 36.4 37.1 70.2 64.1 69.7 60.1 100.0 4 38.3 90% 1519.3 67.2 36.8 36.8 70.7 64.1 69.3 60.1 100.0 8 33.8 79% 1515.1 67.2 36.6 36.9 70.2 64.1 70.0 60.1 100.0

12 29.4 69% 1476.8 67.1 36.6 37.4 70.2 63.4 70.3 59.8 99.5 16 24.9 58% 1476.8 67.2 36.2 37.3 70.2 63.5 67.8 59.4 98.8

- 19 21.6 51% 1525.1 67.2 36.0 37.8 70.4 63.4 65.7 59.5 99.0
- 20 20.5 48% 1505.6 66.7 36.3 37.3 70.0 63.0 65.5 59.2 98.5 24 16.0 37% 1504.1 65.4 36.4 36.0 68.1 60.5 64.9 58.1 96.7

LLaVA-NeXT-13B (40 layers) 0 81.8 100% 1570.0 70.5 35.9 39.9 71.9 65.7 66.7 61.3 100.0 4 75.0 92% 1574.8 70.6 34.8 39.7 71.9 65.4 66.5 61.1 99.7 8 68.2 83% 1552.4 70.6 35.0 39.6 71.9 65.1 66.9 61.0 99.5

12 61.4 75% 1568.5 70.1 34.8 39.8 71.9 65.0 66.7 61.0 99.5 16 54.6 67% 1561.0 70.1 35.0 39.7 71.9 64.8 66.9 60.9 99.3 20 47.8 58% 1565.8 70.0 35.8 40.2 71.8 64.1 68.3 61.2 99.8 24 41.0 50% 1553.0 70.2 36.2 39.9 71.8 63.6 67.5 61.0 99.5 28 34.3 42% 1536.1 69.3 35.1 39.4 71.0 62.8 66.3 60.1 98.0 32 27.5 34% 1468.4 65.8 35.2 38.9 69.3 60.5 58.5 57.4 93.6

Table 11. Performance vs. Efficiency Balance of ShortV under different configurations. # ShortV Layers (N): the number of ShortV layers, Avg.: a normalized average score on benchmarks, Per.: the relative performance retention compared with the vanilla models.

