## HyperLLaVA: Dynamic Visual and Language Expert Tuning for Multimodal Large Language Models

###### Wenqiao Zhang 1♠ Tianwei Lin 2♠ Jiang Liu 3♠ Fangxun Shu 4 Haoyuan Li 1 Lei Zhang 4 He Wanggui 4 Hao Zhou 5 Zheqi Lv 1 Hao Jiang 4♣ Juncheng Li 1♣ Siliang Tang 1 Yueting Zhuang 1♣

1 Zhejiang University , 2 ShanghaiTech University , 3 Chongqing University , 4 Alibaba Group , 5 Harbin Institute of Technology {wenqiaozhang, lihaoyuan, zl.leizhang, zheqilv, junchengli, siliang, yzhuang}@zju.edu.cn linjiawei@shanghaitech.edu.cn,jiangliu@stu.cqu.edu.cn, {shufangxun.sfx, aoshu.jh}@alibaba-inc.com

###### Abstract

et al., 2023c). MLLMs are crucial for the development of flexible, general-purpose assistants, as everyday interactions encompass information from various modalities (e.g., videos, audio, 3D environments, point clouds) in addition to text.

Recent advancements indicate that scaling up Multimodal Large Language Models (MLLMs) effectively enhances performance on downstream multimodal tasks. The prevailing MLLM paradigm, e.g., LLaVA, transforms visual features into text-like tokens using a static vision-language mapper, thereby enabling static LLMs to develop the capability to comprehend visual information through visual instruction tuning. Although promising, the static tuning strategy 1 that shares the same parameters may constrain performance across different downstream multimodal tasks. In light of this, we introduce HyperLLaVA, which involves adaptive tuning of the projector and LLM parameters, in conjunction with a dynamic visual expert and language expert, respectively. These experts are derived from HyperNetworks, which generates adaptive parameter shifts through visual and language guidance, enabling dynamic projector and LLM modeling in two-stage training. Our experiments demonstrate that our solution significantly surpasses LLaVA on existing MLLM benchmarks, including MME, MMBench, SEED-Bench, and LLaVA-Bench. 2.

# arXiv:2403.13447v1[cs.AI]20Mar2024

Contemporary MLLMs (e.g., LLaVA (Liu et al., 2023b,a)) typically adhere to a two-stage training protocol: (i) Vision-Language Alignment: A static projector is trained by leveraging imagetext pairs to synchronize visual features with the language model’s word embedding space. The projector with static parameters connects the vision and language modalities by translating visual features into visual tokens, enabling the LLM to understand the visual content. The quality of conveyed visual tokens directly influences the MLLM’s performance (Zhou et al., 2023). (ii) Multimodal Insturction Tuning. Following vision-language alignment, multimodal instruction data are used to refine the LLM, enabling it to respond to users’ varied requests involving visual content. This step is crucial for augmenting the capabilities and controllability of MLLM to address different downstream multimodal tasks.

Despite two-stages’ critical importance, the projector’s structure and LLM tuning strategy have been relatively underexplored, most of the pieces of literature focus on scaling up the pretraining data (Bai et al., 2023a; Dai et al., 2023), instruction-following data (Li et al., 2023a; Zhang et al., 2023b; Zhao et al., 2023), visual encoders (Bai et al., 2023a) or language models (Lu et al., 2023) to facilitate vision-language understanding. What’s more, further quantitative analyses show that the learned model with static parameters may limit their potential for multidownstream tasks (Mahabadi et al., 2021; Zhang et al., 2023a). Based on the aforementioned insights, our investigation focuses on the twostage training process, transitioning from static to dynamic tuning—that is, tuning both the pro-

###### 1 Introduction

The landscape of Large Language Models (LLMs) (Devlin et al., 2018; Radford et al., 2018; Ouyang et al., 2022) has undergone significant evolution, highlighting their exceptional versatility in managing a wide variety of language-centric applications. To extend the capabilities of LLMs to a wider array of modal inputs, Multimodal Large Language Models (MLLMs) have garnered increasing attention (Radford et al., 2021; Li et al., 2022; Huang et al., 2023; Achiam et al., 2023; Li

- 1The static tuning refers to the trained model with static parameters.
- 2Our project is available on the link https://github.com/DCDmllm/HyperLLaVA

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

#### a Expert Module b HyperLLaVA Framework c Comparison

[Figure 14]

𝑟𝑟1 𝑟𝑟2 𝑟𝑟3 𝑟𝑟4 𝑟𝑟5 𝑟𝑟6 𝑟𝑟7 𝑟𝑟8 𝑟𝑟9 … 𝑟𝑟𝑇𝑇

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

###### HyperLLaVA

Visual Expert

[Figure 19]

[Figure 20]

[Figure 21]

Language

Large Language Model

[Figure 22]

[Figure 23]

Expert 𝑬𝑬𝑳𝑳 𝑣𝑣3 𝑣𝑣4 𝑣𝑣2

𝑭𝑭𝑳𝑳

Vicuna v1.5

[Figure 24]

Visual Guidance Language Guidance

…

𝑡𝑡1 𝑡𝑡2 … 𝑡𝑡𝑇𝑇

[Figure 25]

[Figure 26]

𝑣𝑣1 𝑣𝑣2 𝑣𝑣𝑇𝑇

Projector LLaVA

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Word Embedding

Visual Expert 𝑬𝑬𝑽𝑽

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

𝑭𝑭𝑾𝑾

LLM

Projector𝑭𝑭𝑷𝑷 Visual Encoder 𝑭𝑭𝑽𝑽

HyperNetwork

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Language Expert

[Figure 40]

Visual Instruction

[Figure 41]

[Figure 42]

Visual Tokens Textual Tokens Textual Response Tokens Trainable Parameters Fixed Parameters

- Figure 1: (a) is the overview of LLaVA. (b) describes the simplified version of our HyperLLaVA. (c) shows that compared to LLaVA, our method achieves superior performance across different MLLM benchmarks.

jector and LLM with dynamic parameters to provide flexible design choices that bolster the MLLM’s reasoning abilities across diverse multimodal tasks.

In this paper, we propose HyperLLaVA (Figure 1(b)), its dynamic characterization benefits from a carefully designed expert module, which is derived from HyperNetworks (Ha et al., 2017) to generate the dynamic parameters based on input information. Our bootstrapping philosophy is to dynamically generate strongly correlated features according to visual and language guidance, thereby dynamically modeling the projector and LLM layers, respectively. In detail, HyperLLaVA is learned following the two steps: (i) In visionlanguage alignment, we divide the projector into static layers (original MLP in LLaVA (Liu et al., 2023a)) and dynamic layers (visual expert), where the parameters of static layers are fixed, while the parameters of dynamic layers are dynamically generated based on visual input. The visual expert leverages the Hypernetwork to assist the static projector learn a visual-specific projector that adaptively models the visual features according to the visual guidance. By doing so, the projector can deliver the adaptative visual tokens to the language semantic space. (2) In the multimodal instruction tuning stage, we equip the LLM with a language expert, modeling dynamic parameters for LLM blocks. We regard the intermediate output of LLM as language guidance that guides the language expert to provide an improved instruction-specific comprehension of the user’s request. By doing so, the MLLM increases the flexibility by instead generating unique parameters for every input, allowing the MLLM to make use of similarities between samples across datasets and avoid potential inter-

ference between samples within the same dataset. Notably, the proposed language expert serves as a parameter-efficient fine-tuning approach for the MLLMs, yielding a comparable performance according to the original LLaVA.

In summary, our contributions are three-fold as follows:

- • We study the under-explored dynamic tuning strategy for MLLMs and introduce HyperLLaVA, leveraging the visual and languageguided dynamic tuning for projector and LLM;
- • The proposed visual and language expert serves as a parameter-efficient method of multitask fine-tuning;
- • We conducted comprehensive and detailed experiments across multiple MLLM benchmarks. The rich experimental results prove the effectiveness and universality of our proposed method.

###### 2 Related Work

Large Language Model. The proliferation of Large Language Models (LLMs) has dramatically reshaped the landscape of natural language processing. Pioneering models such as encodercentric model BERT (Devlin et al., 2018) and decoder-centric model GPT (Radford et al., 2018) have led this charge, showcasing that enhancements in model size and the expansiveness of training datasets can result in unprecedented improvements in performance. Building on the achievements of their predecessors, subsequent models have brought forth substantial innovations that further advance the prowess of LLMs.

PaLM (Chowdhery et al., 2023) highlighted the benefits of increasing model parameters for enhanced language comprehension. Meanwhile, InstructGPT (Ouyang et al., 2022) and ChatGPT utilized fine-tuning and reinforcement learning strategies to refine their performance in conversational interaction (Chen et al., 2023b). However, the reliance on textual data as the sole source of learning has been a limiting factor, as it constrains the models’ ability to engage with the richly interconnected world of multimodal information.

Multimodal Large Language Model. In recent years, the development of deep learning has brought prosperity to the field of multimodal intelligence (Baltrušaitis et al., 2018; Li et al., 2023d; Zhang et al., 2022b, 2019b, 2022a). Multimodal Large Language Models (MLLMs) leverage the power of LLMs, mitigating extra computational cost and enhancing the efficacy of multimodal pre-training (Zhang et al., 2024), to bridge the gap between textual and multimodal data(e.g., images, videos, and audios). A prominent attempt is CLIP (Radford et al., 2021), demonstrating the alignment of visual and textual modalities via contrastive learning across a broad dataset of imagetext pairs. (Li et al., 2022) and (Li et al., 2023e) follow this trend, proposing BLIP and BLIP-2 improved upon CLIP, and gain remarkable performance in basic visual tasks. Flamingo (Alayrac

- et al., 2022) led the way in merging vision and language models by utilizing vast amounts of intertwined image-text dataset, revealing unparalleled zero-shot capabilities in processing imagetext content within conversational contexts for the first time. LLaVA (Liu et al., 2023b) distinctively incorporates short captions annotated by humans and bounding boxes into the GPT4 language model. In the realm of audio processing, there are also some brilliant works, such as SpeechT5 (Ao et al., 2021), MMS (Pratap et al., 2023), PandaGPT (Su et al., 2023), etc. Hypernetworks. The original HyperNetwork (Ha et al., 2017) is designed to reduce the number of parameters, i.e, a small neural network generates parameters for another big neural network, thereby obtaining the model compression for different tasks. Subsequently, HyperNetwork is developed to various domain tasks, including few-shot learning (Brock et al., 2018), graph modeling (Zhang et al., 2019a), domain adaptation (Zhang et al., 2023a), device-cloud collaboration (Lv et al., 2023b,a), etc.

###### 3 Methodology

This section describes the proposed MLLM framework HyperLLaVA. We shall present each module and its training strategy.

###### 3.1 Problem Formulation

The primary goal is to effectively leverage the capabilities of both the pre-trained LLM and visual model. The network architecture is illustrated in Figure 2. Given an RGB image x ∈ RH×W×3, where H and W are the origin resolution. The vision encoder processes input images to obtain a visual token sequence V = [v1,v2,··· ,vNv], where Nv represents the sequence length of text tokens. Subsequently, we concatenate the visual tokens and text tokens T = [t1,t2,··· ,tNt], together and feed them into a LLM Mllm, then generate the language response R = [r1,r2,··· ,tNr], where Nt and Nr indicate the length of text tokens and textual response. In general, MLLM model M(·) consists of two functions as below:

M(·)

: Mp((T |V);Θp)

→ Ml((R|V,T );Θl)

,

MLLM

LLM

Projector

(1)

where Mp(·;Θp) is the projector and Mt(·;Θl) LLM tuning with multi-modal instructions with

parameters Θp and Θl, respectively.

###### 3.2 Preliminaries

LLaVA. LLaVA (Liu et al., 2023b) is trained following two steps: (i) First, a two-layer MLP is employed as vision-language projector Mp(·) to convert visual features into visual tokens V , which have the same dimensionality as the word embedding space in the language model. (ii) Then LLaVA performs instruction-tuning with visual tokens V and language tokens T for the LLM (Llama) Ml(·), generating response tokens R by optimizing its auto-regressive training objective.

HyperNetwork. Hypernetwork (Ha et al., 2016) is a neural network that generates the weights for another neural network. Specifically, HyperNetwork treats the parameters of the multi-layer perception (MLP) as a matrix K(n) ∈ RNin×Nout, where Nin and Nout represent the number of input and output neurons of the nth layer of MLP, respectively. Nin and Nout portray the structure of the MLP layers together. The generation process of K(n) can be regarded as a matrix factorization:

K(n) = ξ(z(n);Θp),∀n = 1,··· ,Nl . (2)

In the training procedure, z(n) and ξ(·) are randomly initialized. The gradients are backpropagated to z(n) and ξ(·), which can help to update them. z(n) and ξ(·) will be saved instead of K(n).

###### 3.3 Vision-language Guided Expert Module

Original LLaVA’s projector and LLM are trained with static parameters. We argue that the static tuning paradigm may limit the flexible visual token delivery and appropriate expression for different downstream multi-modal tasks. Thus, we propose to equip the original’s LLaVA projector and LLM with a visual expert EV and a language expert EL: (i) the visual expert adaptively fits the projector’s output according to the specific visual guidance (e.g, visual features); (ii) the language expert dynamically modeling the posterior blocks of LLM through anterior LLM’s block output.

The expert module is derived from Hypernetorks, which is a neural network that generates its parameters for another neural network. As HyperNetwork dynamically generates a network conditioned on the input embeddings, i.e., the “dynamic characterization” can be modeled by HyperNetwork. However, directly utilizing the HyperNetwork may not satisfactorily model dynamic learning for two key reasons:

- • Weak Correlation. The original HyperNetwork learns the latent vector to generate another model’s parameters. This lacks a strong correlation between parameter generation and input guidance.
- • Unstable Optimization. Using HyperNetwork generate the parameters of the projector

or LLM block is large (Dx×Nin×Nout), i.e., it is hard to optimize the such the numerous parameters, the optimization process is intuitively unstable.

To this end, we carefully tailor the HyperNetwork with the following adjustments:

Input Prior Guidance. We first propose to model the dynamic layers by replacing the learned latent vector z with specific input. Specifically, given the feature fx(i) extracted from backbone of sample x(i), we first develop a layer-specific encoder En(·) that encode the fx(i) as e(n). This vector represents the nth layer parameters.

e(n) = En(fx(i)),∀n = 1,··· ,Nl , (3) where Nl is the number of the modeled layers.

Then the HyperNetwork is used to convert the embedding e(n) into parameters, i.e., we input e(n) into the following two MLP layers to generate parameters of dynamic layers.

w(n) = (W1e(n) + B1)W2 + B2, K(n) = w(n) + b(n),

(4)

where K(n) denotes the nth layer parameters of dynamic layers. Two MLP layers’s weight are denoted by W1 and W2, respectively. b(n), B1 and B2 are the biases.

HyperNetwork-aware Adapter. Adapters are sub-networks with small parameters that are inserted after every attention and feed-forward layer in a model (Houlsby et al., 2019). The original adapter is a parameter-efficient learning approach that learns downstream tasks by updating only a small number of parameters. The adapters consist of a pair of downsampling and upsampling layers, and a residual connection. We found that using downsampling and upsampling strategies, the HyperNetwork-generated parameters can be substantially reduced.

Given the visual guidance xV and language guidance xL, the vision-language guided expert can be defined as:

EM(xM) = WMu (SwiGLU(WMd (xM))) WMu ,WMd = HM(xM),whereM ∈ V,L

(5)

where M indicate the modality, WMu ,WMd respectively denote the weights for upsampling and

downsampling. SwiGLU (Ramachandran et al., 2017) is the activation function, Gaussian Error Linear Unit. HM is the HyperNetwork.

###### 3.4 Visual Expert-Assisted Projector

In this stage, our objective is to adapt the image tokens to LLM, allowing the LLM to comprehend the instances in the images. As shown in Figure 2, we divide the projector as static layers and dynamic layers. Following LLaVA1.5 (Liu et al., 2023a), we employ an two-layer MLP as the static layers. To empower the projector’s expression, we develop a visual expert that learning the projector shift to model the dynamic text tokens. Specifically, given the visual feature fV extracted from visual encoder, the visual expert will adaptively convert fV to dynamic visual embeddings. We show three alternatives for the dynamic visionlanguage alignment, the visual tokens V can be

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

### a VisualExpert-assistedProjector

[Figure 51]

b LanguageExpert-integratedTuning

[Figure 52]

[Figure 53]

[Figure 54]

The image shows a red pencil with a looped or coiled shape, which is unusual because pencils typically have a straight shaft.

##### …

[Figure 55]

|[Figure 56]|
|---|

𝑟𝑟1 𝑟𝑟2 𝑟𝑟3 𝑟𝑟4 𝑟𝑟5 𝑟𝑟𝑁𝑁

[Figure 57]

𝑟𝑟

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

RMS Norm

Softmax Linear

RMS Norm

Downsample

[Figure 79]

[Figure 80]

RMS Norm

Parameter Generation

[Figure 81]

Input Image

###### ½ Layers

[Figure 82]

[Figure 83]

[Figure 84]

SwiGLU

[Figure 85]

[Figure 86]

Language Guidance

[Figure 87]

Language Expert

[Figure 88]

FeedForward

[Figure 89]

[Figure 90]

FeedForward SwiGLU

Parameter Generation

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

SwiGLU

SwiGLU

Upsample

[Figure 100]

[Figure 101]

[Figure 102]

Visual Encoder

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

RMS Norm

RMS Norm

RMS Norm

| |
|---|

| |
|---|

Visual Guidance

HyperNetwork

HyperNetwork

[Figure 109]

[Figure 110]

Upsample

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Self-Attention

Self-Attention

Self-Attention

[Figure 119]

[Figure 120]

1st-layer MLP

Visual Expert

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Q K V

Q K V

Parameter Generation

Parameter Generation

Q K V

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

SwiGLU

SwiGLU

[Figure 125]

[Figure 126]

2nd-layer MLP Visual Expert

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

RMS Norm

RMS Norm

𝑣𝑣7 … 𝑣𝑣𝑁𝑁𝑣𝑣 𝑡𝑡1 𝑡𝑡2 𝑡𝑡3 𝑡𝑡4 𝑡𝑡5 𝑡𝑡6 𝑡𝑡7 … 𝑡𝑡𝑁𝑁𝑡𝑡

RMS Norm

[Figure 137]

Downsample

𝑣𝑣1 𝑣𝑣2 𝑣𝑣3 𝑣𝑣4 𝑣𝑣5 𝑣𝑣6

Parameter Generation

Parameter Generation

| |
|---|

[Figure 138]

Word Embedding Layer

User:What is unusual about this image?

- Figure 2: Overview of proposed HyperLLaVA. (a) describes how the proposed visual expert assists the static projector that dynamically converts the image features to adaptive visual tokens, yielding an augmented visual expression for subsequent instruction tuning. (b) is the proposed language expert-integrated tuning, which uses the output of the intermediate layer as language guidance to generate dynamic instruction-specific feature, increasing the flexibility for processing different multimodal tasks.

calculated as:



L2(L1(fV ) + EV1(fV )) Use 1st Visual Expert



L2(L1(fV )) + EV2(L1(fV )) Use 2nd Visual Expert

V =

L2(L1(fV ) + EV1(fV )) + EV2(L1(fV )) Use 1st&2nd Visual Expert



(6) where L1 and L2 denotes two MLPs, EV1 and EV2 are the visual expert for first and second MLPs. We give a details comparison in the Sec. experiments.

Such visual experts learn the projector shift to model the dynamic text tokens, and thus empower the projector’s expression for downstream tasks.

###### 3.5 Language Expert-integrated Tuning

In this stage, LLM is adjusted to become an LVLM with multi-modal understanding. We use more complex instructions, including tasks such as image logical reasoning and text recognition, which require the model to have a stronger multimodal understanding. Different Previous studies have shown that features provided by the inter-

mediate layer may suffice to preliminarily understand the given input samples (Xin et al., 2020)and can serve as guidance hints to improve training (Romero et al., 2014). Thus, generating guidance in the intermediate LLM layer allows the model to form a preliminary understanding of the given instruction. Therefore, we regard the output of the intermediate LLM layer as language guidance that generates adaptive instruction-specific features that enhance the generation accuracy. As shown in Figure 2, given the language guidance fL, the adapter’s parameters {WLu,WLd} are generated by HL(fL). By doing so, the instructionspecific features can be calculated as below:

###### xˆL = EL(xL) + xL + FFN(SwiGLU(xl)) (7)

where xL is the features generated from RMS normalization and self-attention in LLM’s block.

###### 4 Experiments

We verify HyperLLaVA’s effectiveness on multiple datasets and then discuss HyperLLaVA’s properties with controlled studies.

- Table 1: Comparison with SoTA methods on 12 benchmarks. Res, PT, IT indicate input image resolution, the number of samples in the pretraining and instruction tuning stage, respectively. We color each row as the best and second best . Improv. ↑ indicates performance improvement compared with LLaVA.

|Method LLM Res. PT IT<br><br>|VQA Datasets|Benchmark Toolkits|
|---|---|---|
| |VQAv2 GQA VizWiz SQAI VQAT<br><br>|POPE MME MMB MMBCN SEED LLaVAW MM-Vet|
|BLIP-2 (Li et al., 2023e) Vicuna-13B 224 129M InstructBLIP (Dai et al., 2023) Vicuna-7B 224 129M 1.2M InstructBLIP (Dai et al., 2023) Vicuna-13B 224 129M 1.2M<br><br>Shikra (Chen et al., 2023a) Vicuna-13B 224 600K 5.5M IDEFICS-9B (Laurençon et al., 2023) LLama-7B 224 353M 1M<br><br>IDEFICS-80B (Laurençon et al., 2023) LLama-65B 224 353M 1M<br><br>Qwen-VL (Bai et al., 2023b) Qwen-7B 448 1.4B 50M Qwen-VL-Chat (Bai et al., 2023b) Qwen-7B 448 1.4B 50M<br><br>|41.0 41 19.6 61 42.5<br><br>- 49.2 34.5 60.5 50.1<br>- 49.5 33.4 63.1 50.7<br><br><br>77.4 - - - 50.9 38.4 35.5 - 25.9 60.0 45.2 36.0 - 30.9<br>78.8 59.3 35.2 67.1 63.8 78.2 57.5 38.9 68.2 61.5<br><br><br>|85.3 1293.8 - - 46.4 38.1 22.4 - - 36 23.7 53.4 60.9 26.2 78.9 1212.8 - - 58.2 - 25.6<br><br>- 58.8 - - - - -<br><br>- - 48.2 25.2 - - -<br><br>- - 54.5 38.1 - - -<br><br>- - 38.2 7.4 56.3 - -<br><br>- 1487.5 60.6 56.7 58.2 - -<br><br><br>|
|HyperLLaVA w/o EV (Ours) Vicuna-7B 336 558K 665K HyperLLaVA w/o EL (Ours) Vicuna-7B 336 558K 665K<br><br>|79.0 62.5 50.3 70.4 58.1 78.8 61.9 52.1 70.7 57.5<br><br>|85.9 1486.0 65.9 59.7 61.0 63.7 32.8 85.6 1492.0 66.7 58.6 60.8 62.6 30.9<br><br>|
|LLaVA-1.5 (Liu et al., 2023a) Vicuna-7B 336 558K 665K HyperLLaVA (Ours) Vicuna-7B 336 558K 665K<br><br>Improv. ↑ - - - -<br><br>|78.5 62.0 50.0 66.8 58.2<br><br>79.1 62.7 51.9 70.4 58.5 0.6 +0.7 +1.9 +3.6 +0.3<br><br><br>|85.9 1510.7 64.3 58.3 58.6 63.4 30.5<br><br>86.3 1481.2 65.9 60.6 61.4 64.0 31.0<br><br><br>+0.4 - +1.6 +2.3 +2.8 +0.6 +0.5<br><br>|

- Table 2: Three Alternatives for Dynamic Vision-language

Table 3: Analysis of Language Expert Integration for Different LLM Layers.

Alignment. EV1 and EV2 denote visual expert for first and second MLP layer.

|Method<br><br>|VQA Datasets|Benchmark Toolkits|
|---|---|---|
| |GQA SQA-I VQA-T<br><br>|POPE MME|
|w/o EL Anterior 16 Blocks All 32 Blocks|61.9 70.7 57.5<br><br>62.5 69.4 58.5<br><br><br>62.7 69.5 58.6|85.6 1492.0<br><br>85.9 1481.4<br><br>86.0 1460.3<br>|
|Posterior 16 Blocks|62.7 70.4 58.5<br><br>|86.3 1481.2<br><br>|

|Methods|VQA Datasets|Benchmark Toolkits|
|---|---|---|
| |GQA SQA-I VQA-T<br><br>|POPE MME|
|w/o EV EV2 EV1&EV2<br><br>|62.5 70.4 58.1 62.0 69.8 58.0 60.1 69.5 54.4<br><br>|85.9 1486.0<br>86.4 1442.6 86.1 1449.8<br>|
|EV1|62.7 70.4 58.5<br><br>|86.3 1481.2<br><br>|

egy that effectively addresses the unique demands of each training phase.

###### 4.1 Dataset and Setting

Besides, We train our model following the same training process as LLaVA-1.5. The process includes two stages: (1) feature alignment stage: use 558K subset of the LAION-CC-SBU dataset to connect a frozen pretrained vision encoder to a frozen LLM; (2) visual instruction tuning stage: use 150K GPT-generated multimodal instructionfollowing data, plus around 515K VQA data from academic-oriented tasks, to teach the model to follow multimodal instructions.

Benchmark Datasets. We evaluate our proposed HyperLLaVA on five VQA datasets: VQAv2 (Goyal et al., 2017); GQA (Hudson and Manning, 2019); VizWiz (Gurari et al., 2018); SQAI: ScienceQA-IMG (Lu et al., 2022); VQAT (Singh et al., 2019): TextVQA and seven Benchmark Toolkits: POPE (Li et al., 2023f); MME (Fu

- et al., 2023); MMB: MMBench (Liu et al., 2023c); MMBCN: MMBench-Chinese (Liu et al., 2023c); SEED: SEED-Bench (Li et al., 2023b); LLaVAW: LLaVA-Bench(In-the-Wild) (Liu et al., 2023b); MM-Vet (Yu et al., 2023). Implementation Details. The model was trained on an 8-A100 machine in one day. The implementation details refer to the Appendix. In the training of the HyperLLaVA, we utilize the ADAMW (Loshchilov and Hutter, 2017) optimizer, adapting hyperparameters to cater to the specific requirements of each phase. For the feature alignment stage, parameters are set as B = 32, Lr = 0.001, while for visual instruction tuning stage, we adjust the parameters to B = 16, Lr = 0.00002. The configuration for the ADAMW optimizer incorporates the following settings: β = (0.9,0.999), ε = 1 × 10−8, and Wd = 0.0, ensuring a bespoke optimization strat-

Comparison of Methods. For quantifying the efficacy of the proposed framework, we compare HyperLLaVA with previous SOTA approaches. We choose BLIP-2(Li et al., 2023e), InstructBLIP(Dai et al., 2023) based on Vicuna-7B, InstructBLIP(Dai et al., 2023) based on Vicuna13B, Shikra (Chen et al., 2023a), IDEFICS9B(Laurençon et al., 2023), IDEFICS-80B (Laurençon et al., 2023), Qwen-VL (Bai et al., 2023b), Qwen-VL-Chat (Bai et al., 2023b) and LLaVA1.5 (Liu et al., 2023a). More details of baselines are in the Appendix.

###### 4.2 Overall Performance

We benchmark HyperLLaVA on a wide range of academic VQA benchmarks and recent bench-

Table 4: Zero-shot object hallucination evaluation results on POPE dataset. "Yes" indicates the proportion of positive responses to the given question.

|Method<br><br>|LLM Activated<br><br>|Adersarial|Popular<br><br>|Random|
|---|---|---|---|---|
| | |Acc F1-Score Yes|Acc F1-Score Yes|Acc F1-Score Yes|
|mPLUG-Owl (Ye et al., 2023) MM-GPT (Gong et al., 2023) LLaVA-1.5 (Liu et al., 2023a)<br><br>|LLaMA-7B 6.7B LLaMA-7B 6.7B Vicuna-7B 7B<br><br>|82.4 81.6 45.2 50.0 66.7 100.0 85.1 84.2 44.0|85.5 84.3 42.1<br><br>50.0 66.7 100.0<br><br>87.2 86.1 41.9|86.3 85.3 42.3<br><br>50.0 66.7 100.0<br><br>50.3 45.9 41.9|
|HyperLLaVA|Vicuna-7B 7B<br><br>|85.6 84.7 44.1<br><br>|87.3 86.2 42.4<br><br>|50.7 46.5 42.1<br><br>|

Table 5: Deep analysis of expert structure.

Table 6: Comparsion of parameter-efficient learning.

|Method<br><br>|VQA Datasets|Benchmark Toolkits|
|---|---|---|
| |GQA SQA-I VQA-T<br><br>|POPE MME|
|Adapter (Houlsby et al., 2019) Hypernetwork+MLP Hypernetwork+Adapter (Mahabadi et al., 2021)|57.7 69.4 53.5 60.2 68.8 52.9 62.1 69.9 57.0<br><br>|83.5 1371.8<br>84.6 1460.7<br>85.4 1494.4<br>|
|Ours|62.7 70.4 58.5<br><br>|86.3 1481.2<br><br>|

|Method<br><br>|VQA Datasets<br><br>|Benchmark Toolkits|
|---|---|---|
| |GQA SQA-I VQA-T<br><br>|POPE MME|
|LoRa (Hu et al., 2021) Adapter (Houlsby et al., 2019) Hypernetwork+Adapter (Mahabadi et al., 2021)|63.0 68.4 58.2 42.6 61.2 41.0 49.0 63.6 48.3<br><br>|86.4 1496.9<br><br>81.2 874.4<br><br>84.6 1140.0|
|Language Expert|62.5 70.4 58.1<br><br>|85.9 1486.0<br><br>|

marks specifically proposed for instructionfollowing LMMs, totaling 12 benchmarks. Table 1 summarizes the quantitative results of our framework and baselines on five VQA datasets and five Benchmark Toolkits. We make the following observations: 1) In general, irrespective of the different scenarios, compared to LLaVA, HyperLLaVA achieves the best performance on almost all the multimodal scenarios across both datasets (Except for the MME benchmark), which strongly demonstrates the generalizability of the proposed HyperLLaVA. 2) HyperLLaVA (both 7B and 13B) outperforms bigger MLLMs with billions of trainable parameters for cross-modal connection (e.g., 80B IDEFICS (Laurençon et al., 2023)). This further indicates the effectiveness of the proposed MLLM structure. 3) Compared with the original LLaVA, we show that HyperLLaVA achieves the best performance across 11 out of 12 benchmarks. Such results benefit from the carefully designed lightweight visual and language expert, which empowers the static projector and LLM to facilitate different multimodal tasks.

###### 4.3 Ablation Study

Effectiveness of Each Component. Table 1 also illustrate the effectiveness of each component, i.e., visual expert EV and language expert EL. Comparing HyperLLaVA and HyperLLaVA(-EV ) (Row 11 v.s Row 13), the EV contributes 2.61% improvement on mean accuracy. Meanwhile, Row 11 indicates that it suffers from 0.94%, a noticeable performance degradation without the EL. To sum up, we can observe that the improvement of using each module alone is distinguishable. Combining all the components, our HyperLLaVA exhibits steady improvement over the baselines.

###### 4.4 In-Depth Analysis

We validate the effectiveness of the proposed two modules through the experiments on GQA, SQAI, VQA-T, POPE and MME benchmarks.

Three Alternatives for Vision-language Alignment. To build insights on the visual expertassisted projector in HyperLLaVA, we perform an in-depth analysis of three alternatives for dynamic vision-language alignment. Table 2 exhibits the three results. According to our observation, using one visual expert to access the dynamic projection yields the best results. Besides, the other two plans also obtained comparable results, indicating the effectiveness of dynamic projection.

Analysis of Language Expert Integration for Different Blocks. To deeply analyze the effectiveness of language experts, we study the language expert integration for different blocks in Table 3, including anterior 16 blocks (before 1/2 LLM layers), all 32 blocks (all LLM layers) and posterior 16 blocks (after 1/2 LMM layers). Generally speaking, leveraging the language expert integration for the posterior 16 blocks obtained almost the best performance. Besides, Row 2 and Row 3 utilize the initial language input as language guidance, obtaining suboptimal results compared with language expert integration for the posterior 16 blocks. Our intuition is that the language guidance might not have gathered sufficient contextual information for subsequent dynamic LLM layer modeling.

Analysis on the Inserted Blocks for Language Guidance. We investigate the impact of inserting language guidance into different layers of LLMs. We report the evaluation score of GQA and POPE datasets in Figure 4. We observe that the performance is low when we insert language guid-

[Figure 139]

[Figure 140]

(a) POPE Performance (b) GQA Performance

Figure 3: Selected blocks for language guidance.

ance too early (i.e., 4, 8) as the model might not have gathered sufficient contextual information to generate effective guidance. Meanwhile, inserting language guidance too late (i.e., 24, 28) degenerates the performance. We speculate this is due to the generated guidance being too concentrated and there not being enough layers to integrate the language-aware details.

Analysis of Expert’s Structure. We systematically present the explicit benefits from the carefully designed expert’s structure in Table 5. The adapter-based structure surpasses MLP-based structure across all datasets, mainly due to the generated MLP is no longer a lightweight network to optimize, producing unstable performance. Compared with HyperNetwork+Adapter (Row 3 vs Row 4), our proposed vision-language guided expert structure obtained the best performance. The results correspond with our assumption of the original HyperNetworks, which lacks a strong correlation between input and parameter generation. Our method, allows the model to make use of similarities between samples across datasets and avoid potential interference between samples within the same dataset.

Effect of Dimension of Expert Input and Downsampling. Figure 4 empirically provides an appropriate dimension of input and downsampling, i.e, 64 and 16, respectively, either increasing or decreasing this value results in a performance decay. According to our analysis, a bigger dimension may result in an unstable HyperNetwork optimization and a smaller value contains less languageguided information for dynamic learning, and thus yielding performance decay.

Parameter-efficient Fine-tuning. Our proposed language expert also can serve as a parameterefficient fine-tuning function. The structure is similar to the HyperNetwork+Adapter. However, original hypernetwork-based approaches generally condition their parameters on a learned latent

[Figure 141]

[Figure 142]

(a) Effect of Expert’s Input Dimension (b) Effect of Expert’s Downsampling Dimension

Figure 4: Performance with respect to the different input and downsampling dimension in expert.

embedding, implying the model is the same for every example, yield performance decay. Summing up, the proposed language expert is an effective and parameter-efficient way to share information across multiple adapters to enable positive transfer to low-resource and related tasks.

Object Hallucination Evaluation. We adopt the evaluation pipeline of POPE (Li et al., 2023f), a polling-based query method, to evaluate object hallucination in HyperLLaVA. The results are presented in Table 4, where HyperLLaVA exhibits the best performance, indicating that HyperLLaVA tends to generate objects consistent with the given image. Additionally, we observe that the “yes” ratio of HyperLLaVA remains relatively balanced, indicating that our model is capable of providing accurate feedback based on the questions.

###### 5 Conclusion

Building upon HyperLLaVA’s innovative dynamic tuning strategy, our work paves the way for groundbreaking advancements in multimodal learning systems. By adaptively tuning both projector and LLM parameters, and integrating dynamical visual and language experts, we not only surpass the performance benchmarks set by LLaVA but also introduce a parameter-efficient methodology. This approach offers a new horizon for enhancing multimodal task performances through personalized, dynamic adjustments. Future research could further explore the scalability of dynamic tuning mechanisms, potentially unlocking new avenues for understanding and integrating multimodal information more seamlessly.

###### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. 2022. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736.

Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, et al. 2021. Speecht5: Unified-modal encoder-decoder pre-training for spoken language processing. arXiv preprint arXiv:2110.07205.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023a. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023b. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966.

Tadas Baltrušaitis, Chaitanya Ahuja, and LouisPhilippe Morency. 2018. Multimodal machine learning: A survey and taxonomy. IEEE transactions on pattern analysis and machine intelligence, 41(2):423–443.

Andrew Brock, Theodore Lim, James M. Ritchie, and Nick Weston. 2018. SMASH: one-shot model architecture search through hypernetworks. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net.

Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. 2023a. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195.

Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. 2023b. Sharegpt4v: Improving large multimodal models with better captions. arXiv preprint arXiv:2311.12793.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2023. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113.

W Dai, J Li, D Li, AMH Tiong, J Zhao, W Wang, B Li, P Fung, and S Hoi. 2023. Instructblip: towards general-purpose vision-language models with instruction tuning. arxiv. Preprint posted online on June, 15:2023.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. 2023. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394.

Tao Gong, Chengqi Lyu, Shilong Zhang, Yudong Wang, Miao Zheng, Qian Zhao, Kuikun Liu, Wenwei Zhang, Ping Luo, and Kai Chen. 2023. Multimodal-gpt: A vision and language model for dialogue with humans. arXiv preprint arXiv:2305.04790.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. 2017. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913.

Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. 2018. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3608–3617.

David Ha, Andrew Dai, and Quoc V Le. 2016. Hypernetworks. arXiv preprint arXiv:1609.09106.

David Ha, Andrew M. Dai, and Quoc V. Le. 2017. Hypernetworks. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net.

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019. Parameter-efficient transfer learning for nlp. In International Conference on Machine Learning, pages 2790–2799. PMLR.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Qiang Liu, et al. 2023. Language is not all you need: Aligning perception with language models. arXiv preprint arXiv:2302.14045.

Drew A Hudson and Christopher D Manning. 2019. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709.

Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, and Victor Sanh. 2023. Obelics: An open web-scale filtered dataset of interleaved image-text documents.

Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Fanyi Pu, Jingkang Yang, Chunyuan Li, and Ziwei Liu. 2023a. Mimic-it: Multi-modal in-context instruction tuning. arXiv preprint arXiv:2306.05425.

Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. 2023b. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125.

Juncheng Li, Kaihang Pan, Zhiqi Ge, Minghe Gao, Hanwang Zhang, Wei Ji, Wenqiao Zhang, Tat-Seng Chua, Siliang Tang, and Yueting Zhuang. 2023c. Fine-tuning multimodal llms to follow zero-shot demonstrative instructions. In The Twelfth International Conference on Learning Representations.

Juncheng Li, Siliang Tang, Linchao Zhu, Wenqiao Zhang, Yi Yang, Tat-Seng Chua, and Fei Wu. 2023d. Variational cross-graph reasoning and adaptive structured semantics learning for compositional temporal grounding. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023e. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, pages 12888–12900. PMLR.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. 2023f. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2023a. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023b. Visual instruction tuning. arXiv preprint arXiv:2304.08485.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. 2023c. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281.

Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, KaiWei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521.

Yadong Lu, Chunyuan Li, Haotian Liu, Jianwei Yang, Jianfeng Gao, and Yelong Shen. 2023. An empirical study of scaling instruct-tuned large multimodal models. arXiv preprint arXiv:2309.09958.

Zheqi Lv, Zhengyu Chen, Shengyu Zhang, Kun Kuang, Wenqiao Zhang, Mengze Li, Beng Chin Ooi, and Fei Wu. 2023a. Ideal: Toward high-efficiency devicecloud collaborative and dynamic recommendation system. arXiv preprint arXiv:2302.07335.

Zheqi Lv, Wenqiao Zhang, Shengyu Zhang, Kun Kuang, Feng Wang, Yongwei Wang, Zhengyu Chen, Tao Shen, Hongxia Yang, Beng Chin Ooi, and Fei Wu. 2023b. Duet: A tuning-free device-cloud collaborative parameters generation framework for efficient device model generalization. In Proceedings of the ACM Web Conference 2023.

Rabeeh Karimi Mahabadi, Sebastian Ruder, Mostafa Dehghani, and James Henderson. 2021. Parameterefficient multi-task fine-tuning for transformers via shared hypernetworks. arXiv preprint arXiv:2106.04489.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback, 2022. URL https://arxiv. org/abs/2203.02155, 13.

Vineel Pratap, Andros Tjandra, Bowen Shi, Paden Tomasello, Arun Babu, Sayani Kundu, Ali Elkahky, Zhaoheng Ni, Apoorv Vyas, Maryam FazelZarandi, et al. 2023. Scaling speech technology to 1,000+ languages. arXiv preprint arXiv:2305.13516.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748– 8763. PMLR.

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. 2018. Improving language understanding by generative pre-training.

Prajit Ramachandran, Barret Zoph, and Quoc V Le.

2017. Searching for activation functions. arXiv preprint arXiv:1710.05941.

Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, and Yoshua Bengio. 2014. Fitnets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550.

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326.

Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai. 2023. Pandagpt: One model to instruction-follow them all. arXiv preprint arXiv:2305.16355.

Ji Xin, Raphael Tang, Jaejun Lee, Yaoliang Yu, and Jimmy Lin. 2020. Deebert: Dynamic early exiting for accelerating bert inference. arXiv preprint arXiv:2004.12993.

Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. 2023. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. 2023. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490.

Chris Zhang, Mengye Ren, and Raquel Urtasun. 2019a. Graph hypernetworks for neural architecture search. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net.

Duzhen Zhang, Yahan Yu, Chenxing Li, Jiahua Dong, Dan Su, Chenhui Chu, and Dong Yu. 2024. Mmllms: Recent advances in multimodal large language models. arXiv preprint arXiv:2401.13601.

Wenqiao Zhang, Zheqi Lv, Hao Zhou, Jia-Wei Liu, Juncheng Li, Mengze Li, Siliang Tang, and Yueting Zhuang. 2023a. Revisiting the domain shift and sample uncertainty in multi-source active domain transfer. arXiv preprint arXiv:2311.12905.

Wenqiao Zhang, Haochen Shi, Jiannan Guo, Shengyu Zhang, Qingpeng Cai, Juncheng Li, Sihui Luo, and Yueting Zhuang. 2022a. Magic: Multimodal relational graph adversarial inference for diverse and unpaired text-based image captioning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 3335–3343.

Wenqiao Zhang, Siliang Tang, Yanpeng Cao, Shiliang Pu, Fei Wu, and Yueting Zhuang. 2019b. Frame augmented alternating attention network for video question answering. IEEE Transactions on Multimedia, 22(4):1032–1041.

Wenqiao Zhang, Lei Zhu, James Hallinan, Shengyu Zhang, Andrew Makmur, Qingpeng Cai, and Beng Chin Ooi. 2022b. Boostmis: Boosting medical image semi-supervised learning with adaptive pseudo labeling and informative active annotation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20666– 20676.

Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou, Nedim Lipka, Diyi Yang, and Tong Sun. 2023b. Enhanced visual instruction tuning for text-rich image understanding. In NeurIPS 2023 Workshop on Instruction Tuning and Instruction Following.

Bo Zhao, Boya Wu, and Tiejun Huang. 2023. Svit: Scaling up visual instruction tuning. arXiv preprint arXiv:2307.04087.

Qiang Zhou, Zhibin Wang, Wei Chu, Yinghui Xu, Hao Li, and Yuan Qi. 2023. Infmllm: A unified framework for visual-language tasks. arXiv preprint arXiv:2311.06791.

