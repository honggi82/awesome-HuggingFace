# arXiv:2511.12609v2[cs.CL]23Nov2025

[Figure 1]

Lychee

## Uni-MoE-2.0-Omni: Scaling Language-Centric Omnimodal Large Model with Advanced MoE, Training and Data

Yunxin Li∗, Xinyu Chen, Shenyuan Jiang, Haoyuan Shi, Zhenyu Liu, Xuanyu Zhang, Nanhao Deng, Zhenran Xu, Yicheng Ma, Meishan Zhang, Baotian Hu‡, Min Zhang‡ Research Institute of Computing and Intelligence Harbin Institute of Technology, Shenzhen

[Figure 2]

[Figure 3]

[Figure 4]

Models Codes Website

### Abstract

We present Uni-MoE 2.0 from the Lychee family. As a fully open-source omnimodal large model (OLM), it substantially advances Lychee’s Uni-MoE series in language-centric multimodal understanding, reasoning, and generating. Based on the dense LLM, we build Uni-MoE-2.0-Omni from scratch through three core contributions: dynamic-capacity Mixture-of-Experts (MoE) design, a progressive training strategy enhanced with an iterative reinforcement strategy, and a carefully curated multimodal data matching technique. It is capable of omnimodal understanding, as well as generating images, text, and speech. Architecturally, our new MoE framework balances computational efficiency and capability for 10 cross-modal inputs using shared, routed, and null experts, while our Omni-Modality 3D RoPE ensures spatio-temporal cross-modality alignment in the self-attention layer. For training, following cross-modal pretraining, we use a progressive supervised finetuning strategy that activates modality-specific experts and is enhanced by balanced data composition and an iterative GSPO-DPO method to stabilise RL training and improve reasoning. Data-wise, the base model, trained on approximately 75B tokens of open-source multimodal data, is equipped with special speech and image generation tokens, allowing it to learn these generative tasks by conditioning its outputs on linguistic cues. Extensive evaluation across 85 benchmarks demonstrates that our model achieves state-of-the-art or highly competitive performance against leading OLMs, surpassing Qwen2.5-Omni (trained with 1.2T tokens) on over 50 of 76 benchmarks. Key strengths include video understanding (+7% avg. of 8), omnimodallity understanding (+7% avg. of 4), and audiovisual reasoning (+4%). It also advances long-form speech processing (WER reduced by up to 4.2%) and leads in low-level image processing and controllable generation across 5 metrics.

[Figure 5]

Figure 1: The performance of Uni-MoE-2.0-Omni and previous SOTA omnimodal large models.

∗Contributions are shown in Sec. 8. ‡ indicates the corresponding author and project leader.

CONTENTS

### Contents

- 1 Introduction 3
- 2 Uni-MoE-2.0-Omni 4

- 2.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 2.2 Uni-Perception . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 2.3 Main Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 2.3.1 Omni-Modality 3D RoPE . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 2.3.2 Dynamic-Capacity MoE . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 2.4 Uni-Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 3 Training and Data Recipes 11

- 3.1 Training Recipe: From LLMs to OLMs . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 3.2 Data Recipe . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

- 3.2.1 Audio-centric Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 3.2.2 Vision-centric Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 3.2.3 Text-only Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

- 4 Experiment 14

- 4.1 Vision-Language Understanding . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

- 4.1.1 Image Understanding . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 4.1.2 Video Understanding . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- 4.1.3 Language Capability . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- 4.2 Audio Understanding and Speech Generation . . . . . . . . . . . . . . . . . . . . . . . . . 16

- 4.2.1 Audio X → Text . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- 4.2.2 Text → Speech . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- 4.2.3 Speech → Speech/Text . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- 4.2.4 Vision + Speech → Speech/Text . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- 4.3 Omnimodality Understanding . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- 4.4 Image Generation and Edition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- 4.5 MoE Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- 4.6 Thinking vs. No-Thinking . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24

- 4.6.1 Visual Reasoning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- 4.6.2 Visual Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25

- 5 Discussion and Future Work 25
- 6 Conclusion 26
- 7 Acknowledgment 26
- 8 Contributors 27

- 1. Introduction

#### A Appendix 41

- A.1 Training Data List . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
- A.2 Evaluation Data List . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
- A.3 Thinking Prompt . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
- A.4 Gradient Estimation Formalization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45

### 1 Introduction

The pursuit of Artificial Intelligence (AI) has long been guided by a vision of creating systems that can perceive, reason, and interact with the world with human-like breadth and fluency. Recently, the field of artificial intelligence has been rapidly advancing towards omnimodal systems, e.g., Qwen-Omni (Xu et al., 2025), Ming-Omni (AI et al., 2025), Gemini (Comanici et al., 2025), and GPT-4o (Hurst et al., 2024), which aim to seamlessly integrate and process a diverse array of data types, including text, images, audio, and beyond, within a single, unified large model. These omnimodal models represent a pivotal step towards more general and versatile AI, promising models capable of richer understanding and more natural interaction with the complex, multi-sensory world (Li et al., 2025e; Zhou et al., 2025). This is not only for pushing the frontiers of AI research but also for enabling a wide range of real-world applications, from advanced human-computer interfaces (Xie et al., 2024b; Qin et al., 2025) and content creation tools (Schick et al., 2023; Hou et al., 2025) to super-intelligence AI assistants that can comprehend and reason across all modalities.

Initial forays, predominantly from well-resourced industrial labs (Xu et al., 2025; AI et al., 2025), have demonstrated impressive capabilities. However, the architectural and algorithmic path to a genuinely comprehensive omnimodal large model is fraught with complexity. A central challenge lies in the inherent tension between two core competencies: deep context understanding and high-fidelity content generation. Many existing systems are architected with a bias, excelling either as powerful context understanding and reasoning (e.g., audio-only Qwen-Omni (Xu et al., 2025), Baichuan-Omni (Li et al., 2025b)) that lack generative faculties or as generative powerhouses (e.g., OmniGen (Wu et al., 2025), BAGEL (Deng et al., 2025), Janus-Pro (Chen et al., 2025a), Show-o (Xie et al., 2025)) confined to a narrow set of modalities. Underlying this issue is the inefficient scaling of model capacity, where simply expanding dense transformers (Zhang et al., 2023c; Li et al., 2025d) proves computationally prohibitive and difficult to balance dozens of task types with multiple modal interactions. The challenge shows two critical shortcomings: a lack of robust and efficient compositional reasoning architecture across multiple modalities, and significant instability when training on heterogeneous data at scale. Hence, the field still lacks a computationally efficient and unified architecture that achieves a synthesis of comprehensive understanding and versatile generation for all modalities.

This work is anchored in a language-centric perspective to develop an efficient omnimodal large model capable of unified multimodal understanding, reasoning, and generation. Our approach is predicated on the view that language, serving as a structured representation of the world and a mediator between modalities, provides the cornerstone for effective multimodal integration and transition. This architectural choice is motivated by established advantages of LLMs-based multimodal models (Yin et al., 2023; Lopez-Avila & Du, 2025; Cui

- et al., 2024), including great training stability and straightforward scalability to new scenarios.

Specifically, we build Uni-MoE-2.0-Omni, a fully open-source, diverse MoE-based Omnimodal Large Model (OLM). Evolved from the dense Qwen2.5-7B language model, this work demonstrates an efficient pathway to scale a LLM into a powerful and comprehensive OLM. Through progressive architectural evolution and optimized training strategies, we have successfully transformed the base model into an omnimodal model, achieving the crucial transition from mere multimodal understanding to seamless integration of both understanding and generation. Our work is founded on three key architectural contributions (Sec. 2):

- • Unified Modality Encoding & Generation: We design a unified speech encoder that maps diverse audio inputs, including environmental sound, speech, and music, into a shared representation space. For output, a Context-Aware MoE-based TTS module supports dynamic speech synthesis (especially for long speech) and interaction. On the visual side, we employ pre-trained visual encoders to process images and videos, and build a Task-Aware Diffusion Transformer for instruction-guided image generation and editing.
- • Deep Cross-Modal Alignment: To enable deep and efficient fusion of any modality, we introduce an OmniModality 3D RoPE mechanism in the self-attention layers. It encodes the temporal-spatial dimensions of speech, image, text, and video tokens, ensuring seamless alignment and interaction across all input types.

- 2. Uni-MoE-2.0-Omni

• MoE-Driven Cross-Modal Fusion: We strategically extend the standard MLP layers to MoE layers. This new MoE architecture incorporates three expert types: null experts for inference-time computation skipping, modality-specific routed experts for storing modality knowledge and processing cross-modal information, and small-size shared experts to facilitate universal information exchange. This design enables efficient computation, specialized modality handling, and effective and stable multimodal fusion.

#### At training and data recipes, we introduce the following training recipe with the data matching (Sec. 3) :

- • Progressive Omnimodal Training Optimization: To mitigate the instability often encountered when training MoE-based omnimodal large models, we designed a progressive training strategy. This approach sequentially advances through: cross-modal alignment → expert warm-up → MoE fine-tuning and reinforcement learning → generative training. This process efficiently scales dense LLMs into MoEbased omnimodal large models with minimal data requirements, while ensuring convergence stability during iterative reinforcement training (GSPO-DPO) in omnimodal data environments.
- • Language-Centric Hybrid Training for Multimodal Understanding and Generation: To bridge the gap between multimodal understanding and generation tasks—often treated separately during training—we propose a hybrid training approach anchored in language generation tasks. By unifying tasks such as image editing, image generation, and speech synthesis within a language generation framework, we break down the inherent barriers between understanding and generation, enabling synergistic enhancement and bidirectional empowerment of both capabilities.

The comprehensive evaluation of Uni-MoE-2.0-Omni across 85 multimodal benchmarks reveals that Uni-MoE

- 2.0 further pushes the boundaries of omnimodal intelligence. In comparison to leading omnimodal models, e.g., Qwen2.5-Omni (trained with 1.2T Tokens), Ming-Lite-Omni-1.0/1.5, Baichuan-Omni-1.5, MiniCPM-

- o 2.6, and other task-specific models, our model mainly achieves superior results in video understanding and reasoning (averaging + 4% than Ming-Lite-Omni-1.5 on 8 benchmarks), omnimodality comprehension (averaging + 7% than Qwen2.5-Omni on 4 benchmarks, including popular OmniVideoBench and WorldSense), long speech understanding (↓ 3.5% lower WER than Qwen2.5-Omni on LibriSpeech-clean/other-long) and generation (↓ 1% lower WER on TinyStories-en), and AudioVisual tasks (averaging + 4% than Qwen2.5-Omni
- on Speech-Image QA (Reasoning)). Furthermore, Uni-MoE-2.0-Omni exhibits competitive performance in most image generation and editing tasks, outperforming strong generative models (e.g., OmniGen and BAGEL) in image editing, controllable generation and low-level image processing tasks, e.g., + 0.5% than Ming-Lite-Omni on GEdit-Bench and suppressing previous SOTA Qwen-Image and PixWizard on 6 metrics. We open-source all training and inference codes (GitHub), five model checkpoints (Hugging Face), and data lists (Appendix) to support reproducibility and further research.

### 2 Uni-MoE-2.0-Omni

#### 2.1 Overview

Our model processes multimodal inputs—including audio, images, text, and video—through a unified tokenization scheme, as illustrated in Figure 2. As detailed in Sec. 2.2, audio is segmented into 30-second clips, with each clip represented by a sequence of 200 tokens (20 tokens/3s). High-resolution images are encoded using a sliding window method, where each 384×384 patch is independently tokenized. To enhance the model’s comprehension of multimodal data (Sec. 2.3), we introduce Omni-Modality 3D RoPE, a mechanism that temporally aligns inputs from speech, text, video, and images. Within the MoE layer, we employ diverse experts to enable seamless switching between specialized modules. This is stabilized by null experts to realize a token skipping layer, modal-specific routed experts that ensure balanced training, and enhanced by small-size shared experts (only 1/8 of routed experts) that facilitate cross-modal knowledge transfer. For generation tasks (Sec. 2.4), the model utilizes dedicated control tokens. Audio Generation tokens govern language and role attributes, which are leveraged by context-aware MoE-TTS modules. Similarly, Image Generation tokens, composed of task and content instructions, bridge the foundation model to a task-aware diffusion transformer, enabling end-to-end image generation and editing.

#### 2.2 Uni-Perception

Vision Understanding The visual understanding module of Uni-MoE-2.0-Omni employs a unified encoding strategy for both images and videos. The overall architecture consists of two major components: a visual encoder and a mapping network (MLP).

[Figure 10]

- Figure 2: The Uni-MoE-2.0-Omni architecture processes multimodal data through a unified tokenization strategy. Audio is tokenized in 30-second clips, augmented with generation tokens for voice control in the Context-Aware MoE-TTS module, while images are encoded using a sliding window technique. Image Generation Tokens bridge the model to a Task-Aware Diffusion Transformer for end-to-end generation tasks. The model’s comprehension is powered by Omni-Modality 3D RoPE, which aligns inputs across time, and a dynamic-capacity MoE layer. This MoE layer dynamically routes information using diverse experts, with stability ensured by null experts (for token skipping) and modality-specific routed experts (A, V, T indicate audio, visual, and textual expert pretrained on corresponding data). In contrast, compact shared experts (only 1/8 size of routed experts) enable efficient cross-modal knowledge transfer.

- • Visual Encoder. The visual encoder is initialized with SigLIP (Zhai et al., 2023) vision transformer,

which transforms the input image or video frames V into visual features ZV . We adopt the output of the last layer of the transformer as the visual features.

- • Mapping Network. The mapping network is composed of a two-layer MLP and a 2D average pooling layer. Specifically, the MLP takes the visual features as input and projects them into the representation space

of the language model, thereby producing a one-dimensional sequence of visual features HV = p(ZV ). Subsequently, the average pooling layer performs length compression along both spatial dimensions, enabling more efficient training.

Through these visual understanding modules, we can uniformly encode Single-Image, Multi-Image, and Video inputs, thereby transforming different visual input paradigms into a unified one-dimensional sequence of visual representations. The specific encoding strategies for each paradigm are described as follows:

- • Single-Image. For a given single image with arbitrary resolution, we preprocess the input while preserving its aspect ratio. Suppose the original resolution of the input image is (h,w). We traverse a set of candidate resolutions and select a target resolution (h′,w′) that is closest to the original resolution while requiring minimal padding. The image is then resized so that either its height or width matches the target resolution, while the other dimension is padded with blank pixels to reach the target resolution. Each candidate resolution is constrained such that both its height and width are integer multiples of the vision encoder

#### Module Architecture Params

Audio Encoder Whisper-Large-v3 637M Vision Encoder SigLIP-So400M 398M MoE-LLM MoE Transformer 26B MoE-TTS MoE Transformer 1.2BA0.7B Task-DiT Dense Transformer 1.5B Codec Decoder WavTokenizer-large-600-24k-4096 442M VAE Decoder SD-XL 49M

Shared Expert MLP 712M Routed Expert MLP 5.7B Activated Expert 2 Shared Expert + 0~3 Routed Expert Min: 1.5B; Max: 18B

#### Table 1: The architectural design of Uni-MoE-2.0-Omni.

patch size p, ensuring compatibility with the vision encoder input specification. After this preprocessing, the image can be partitioned into a × b visual patches, where a = h′/p and b = w′/p, with each patch of size p×p. Assuming there are T tokens each visual patch, the total number of visual tokens is (a×b)×T.

- • Multi-Image. The encoding of multiple images follows a similar procedure to that of a single image. For each image, we independently search the most suitable target resolution, resize it accordingly, and then convert it into the corresponding set of visual patches. Assuming there are n images are provided as input,

the total number of visual tokens is ni=1((ai × bi) × T), where ai and bi denote the number of patches along the height and width of the i-th image.

- • Video. For video data, we adopt the minimum resolution accepted by the vision encoder (p,p) as the target resolution. Each frame of the video is directly resized to this resolution. Regarding frame selection, we uniformly sample the original video at a rate of one frame per second, yielding fs sampled frames. If the number of sampled frames is less than a predefined lower bound fl or greater than an upper bound fu, we uniformly resample the video to satisfy these constraints. Consequently, the final number of frames is determined as fn = min(max(fs,fl),fu), and the total number of visual tokens is fn × T.

This unified encoding of Single-Image, Multi-Image, and Video inputs into a one-dimensional sequence of visual representations enables the transfer of the model’s capability in high-resolution image understanding to the video domain, thereby facilitating faster convergence and improved performance in video understanding.

Speech Understanding Speech understanding plays a crucial and indispensable role in speech dialogue models. However, for the speech input of most speech dialogue models, they merely conduct analysis and compilation on semantic information, thus paying relatively less attention to prosodic features such as intonation and timbre information. A model has the potential to comprehensively understand multiple types of audio information in a unified manner through a larger-scale and powerful encoding structure in terms of comprehension ability. This architecture is comprises of two main components:

- • Audio Encoder. We adopt the Whisper-Large-v3 encoder (Radford et al., 2023) as our speech encoding module and conduct training on a diverse set of audio datasets. Specifically, input audio signals A first undergo resampling to a unified sample rate of 16000 Hz; subsequently, these resampled audio signals

are fed into the Whisper encoder, which generates audio features ZA. Notably, the audio features output by this encoder not only encapsulate rich intrinsic information from the input audio but also integrate the generalized speech representation capabilities acquired by the pre-trained Whisper encoder during its prior training phase.

- • Audio-Language Mapping.Through comparison, it is found that by using the decoder module of WhisperLarge-v3 as the Qformer and adopting a mapping method of 400 tokens per minute, not only can speech information be efficiently extracted, but also a great deal of paralinguistic semantic information, such as timbre, intonation, and emotion, can be obtained. We utilize all decoder layers of Whisper-Large-v3, with the number of query tokens in the Qformer set to 200 (20 tokens/3s). These types of information are ultimately mapped to the textual dimension through the projection layer, enabling the model to effectively utilise this information for understanding and reasoning.

Specifically, the workflow of speech understanding is described as follows:

XQA = [h1Q,...,hLQ], hAW = WhisperEncoder(A), hAS = MSA(LN(XQA)) + XQA, hAC = MCA(hAW,LN(hAS)) + hAS, hAl = MLP(hAC),

(1)

where hAW is the last hidden states of the pre-trained audio encoder adopted from Whisper-large-v3. MSA and hMCA denote the multihead self-attention and cross-attention operations, and LN is the layer norm function.

By leveraging the fixed-length query vectors XQA, each 30 seconds of audio encoded by the Whisper encoder is mapped to L audio tokens. hAC represents the output of the cross-attention module, which is used to distill the main content of the input audio. After going through all layers of the Whisper decoder, we apply a learnable linear layer for projecting the last output into the representation space of LLM.

To process audio longer than Whisper-Large-v3’s 30-second limit, we employ a chunking mechanism. Long audio is segmented into consecutive 30-second clips, which are batched and processed by the audio understanding module. The resulting features are then concatenated along the time dimensions, enabling the understanding of audio of arbitrary length.

- 2.3 Main Architecture

- 2.3.1 Omni-Modality 3D RoPE

Inspired by the M-RoPE design in Qwen2-VL (Wang et al., 2024b), we propose Omni-Modality 3D RoPE to enable efficient positional modeling across text, audio, image, and video modalities. Similar to M-RoPE, we decompose the original rotary embedding into three components corresponding to the temporal, height, and width dimensions. For textual inputs, these components share the same position IDs, rendering Omni-Modality

- 3D RoPE functionally equivalent to the standard 1D-RoPE. For audio inputs, we align the temporal position IDs with absolute time. Specifically, we define 20 tokens as the minimum temporal unit, which corresponds to a duration of 3 seconds. Instead of adopting the original single-step increment of position IDs, we replace it with the variation rate of absolute time. For image inputs, the temporal IDs of visual tokens remain fixed, while the height and width IDs are assigned according to the spatial locations of tokens within the original image. Unlike M-RoPE, the height and width dimensions are traversed in a patch-wise manner, such that all tokens within a single vision patch are enumerated before proceeding to the next patch. For video inputs, the temporal IDs of each frame are incremented according to absolute time, while the height and width components follow the same ID assignment pattern as images.

As an illustrative example, consider a video input of length 120 seconds accompanied by audio. We place the visual sequence before the audio sequence and sample the video at a rate of one frame every two seconds. Suppose the RoPE ID of the last text token preceding the visual sequence is (x − 1,x − 1,x − 1). Then, the RoPE IDs of the first video frame start from (x,x,x) and increment row by row and column by column up to (x,x + p,x + p), where p denotes the number of tokens along both the height and width of a video frame. The second frame begins with (x + 2θ,x,x) and increases to (x + 2θ,x + p,x + p), where θ is a specific scaling factor for absolute time. Following this procedure, the final frame in the video sequence has an ID of (x + 118θ,x + p,x + p). Similarly, assume the RoPE ID of the last token before the audio sequence is (y −1,y −1,y −1). The first audio segment is then assigned the ID (y,y,y), which is repeated 20 times to represent the minimum audio unit. The next audio segment receives the ID (y + 3θ,y + 3θ,y + 3θ). Continuing this process, the final audio segment is assigned the ID (y + 117θ,y + 117θ,y + 117θ). In this manner, temporal alignment between the visual and audio sequences can also be achieved.

#### 2.3.2 Dynamic-Capacity MoE

Previous studies have demonstrated that factual and procedural knowledge is predominantly stored in the feed-forward network (FFN) modules (Sato & Takagi, 2025). The Mixture-of-Experts architecture enables adaptive knowledge retrieval by dynamically activating different FFN modules, which are determined by a router network. Particularly, given n expert parameters {w0,...,wn−1}, the output of a vanilla MoE for inference is:

n−1

Gating(z)i · TopK(z)i · Expert(x,wi), (2)

yi =

i=0

where z = Wrx, as we use linear network as the router, Gating(∗) is a gating function (usually softmax), and Expert(∗) is a FNN. TopK(z) is the Top-K function, i.e., TopK(z)i := 1 if zi is among the TopK coordinates of z and TopK(z)i := 0 otherwise.

Although achieved notable success, vanilla MoE designs suffer from three key limitations:

- • Discrete expert selection hinders gradient backpropagation. Specifically, the non-differentiable nature of

the Top-K function prevents gradients from being effectively propagated from oi back to xi in the routing equation, leading to biased optimization directions.

- • Homogeneous expert types fail to distinguish between domain-specific and general knowledge, and cannot support operations such as selectively forgetting outdated knowledge.
- • The number of activated experts is fixed, which limits the ability to dynamically adjust the amount of retrieved parametric knowledge according to the varying demands of different tokens.

These limitations prevent MoE from fully exploiting its adaptive activation capability, ultimately resulting in suboptimal performance. To address these challenges, we propose the Dynamic-Capacity MoE architecture, which incorporates (i) routing gradient estimation to enable differentiable expert selection, (ii) explicit expert role specialization to separately model domain-specific and general knowledge, as well as facilitate knowledge deletion, and (iii) dynamic expert number allocation to adaptively control the amount of parametric knowledge.

Routing Gradient Estimation A central difficulty in training vanilla MoE models arises from the nondifferentiable Top-K operation used in expert selection, which obstructs gradient propagation to the router and leads to biased optimization. To address this, we migrate the gradient estimation strategy proposed in Grin-MoE (Liu et al., 2024a), which integrates straight-through gradient estimators under the framework of numerical methods for ordinary differential equations (ODEs). This enables end-to-end optimization of both router and experts under sparse activation constraints, improving the stability of router training and allowing more precise token-to-expert assignments. A formal description is given in Appendix A.4.

Expert Role Specialization To further enhance the adaptability of MoE and address the limitation of homogeneous expert types, we explicitly categorize experts into three distinct roles:

- • Routed Experts: These are task-specific experts responsible for modeling domain-specific knowledge. They are dynamically activated according to the proposed dynamic capacity routing strategy.
- • Shared Experts: These experts capture general, domain-independent knowledge. Unlike routed experts, shared experts are persistently activated for all tokens, ensuring that common knowledge is always available during inference.
- • Null Experts: These are “empty” experts whose output is identically zero. They serve as a mechanism for selective forgetting, effectively removing outdated or irrelevant knowledge from the model’s output. Null experts are also dynamically activated via the dynamic capacity routing strategy.

This role specialization enables the model to (i) allocate computational resources according to token-specific knowledge demands, (ii) maintain a persistent general-knowledge backbone via shared experts, and (iii) selectively forget outdated or irrelevant knowledge through null experts, thereby improving both adaptability and controllability of MoE-based architectures.

Dynamic Capacity Routing Vanilla MoE applies a fixed number of experts to every token, ignoring variations in token complexity and knowledge demand. We address this limitation by introducing a dynamic capacity routing strategy, which determines the number of routed experts for each token based on a Top-P sampling. Formally, let the router produce a probability vector over routed experts for token i:

Nr

##### p(i) = [p(1i),p(2i),...,p(Ni)

p(ji) = 1,

],

r

j=1

where Nr is the number of routed experts. We sort experts in descending order of p(ji), obtaining a permutation π(i) such that:

p(πi)

(i)(1) ≥ p(πi)

(i)(2) ≥ ··· ≥ p(πi)

(i)(Nr).

Algorithm 1: Dynamic-Capacity MoE Training Procedure Input: Router Output z, Input Hidden State x, Routed Expert Count Nr, Shared Expert Count Ns,

Routed Expert Set Er, Shared Expert Set Es Output: Final MoE output y

m ← ActivateExperts(z) /* Binary mask of selected experts per token */ n(tok) ← sum(m) − Ns /* Number of non-shared experts per token */ w ← MaskedSoftmax(z,m) /* Routing weights over selected experts */

- for k ← 1 to Nr do

I ← {i | n(itok) ≥ k} /* Tokens still requiring the k-th routed expert */ x(sel) ← Gather(x,I) /* Hidden states of selected tokens */ z(sel) ← Gather(z,I) /* Router logits of selected tokens */ w(sel) ← Gather(w,I) /* Routing weights of selected tokens */ e⋆ ← Top1(z(sel)) /* Highest-probability expert index */ w⋆ ← w(sel)[e⋆] /* Routing weight for chosen expert */

- o(sel) ← w⋆ · FFN(x(sel),Er[e⋆]) /* Process tokens with chosen expert */
- oest ← GradientEstimation(o(sel),z(sel)) /* Full gradient propagation using gradient estimation */

y ← y + Scatter(oest,I) /* Add weighted outputs to final result */

- for k ← 1 to Ns do w(s) ← wshared[k] /* Routing weight for shared expert */ e(ks) ← Es[k] /* k-th shared expert index */

o(s) ← w(s) · FFN(x,e(ks)) /* Process all tokens with shared expert */ y ← y + o(s) /* Add weighted shared outputs */

#### return y

The set of activated routed experts for token i is then:

 

Ri = π(i)(1),...,π(i)(ki) , ki = min

k



where P is the cumulative probability threshold (e.g., P = 0.7).

 

k

p(πi)

(i)(j) ≥ P

,



j=1

Algorithm To provide a holistic view of our method, Algorithm 1 presents the complete training procedure in pseudocode form. For simplicity of presentation, we treat null experts as a special case of routed experts in the pseudocode. The algorithm integrates the three proposed components into a unified workflow: First, ActivateExperts function determines expert activation, combining the Top-P sampling strategy for routed experts with the persistent activation constraint for shared experts; Second, gradient estimation is applied to routed expert outputs to enable end-to-end optimization under sparse activation; Finally, after computing routed expert contributions, shared expert outputs are computed and incorporated into the MoE representation, injecting general knowledge into the final output.

#### 2.4 Uni-Generation

Speech Generation: Context-Aware MoE-TTS Our approach to speech processing employs a dual-strategy architecture to handle input and output efficiently. For speech understanding, we use a continuous encoding method that significantly reduces token consumption (20 tokens/3 seconds). For speech generation, the base model produces both text and control signals. To empower the model with rich audio generation capabilities, including ambient sounds, music, and speech with diverse timbres, emotions, and tones, we integrate the Wavtokenizer (Ji et al., 2024a) and design a context-aware MoE-TTS module for synthetic voice with three styles.

As shown in Figure 3, our MoE-TTS module is built upon a Qwen2.5-0.5B model initialized with a MoE structure. This autoregressive module takes input instructions and text tokens to generate the corresponding speech encodings, which are subsequently decoded into waveform audio by the Wavtokenizer decoder. The

[Figure 16]

- Figure 3: The illustration of Context-Aware MoE-TTS. This figure uses different colored blocks to represent distinct token types, illustrating our long-context streaming decoding method. Furthermore, the Uni-MoE-TTS module will be released separately, featuring three unique and controllable voice styles.

training process is divided into two key stages: First, we pre-train three separate dense models, each on a single-speaker TTS dataset to capture a unique vocal style. Second, this model is transformed into a MoE architecture by replicating its FFN layers to create four experts, which are then fine-tuned on a diverse, multi-style, and multi-speaker dataset. The overall workflow of MoE-TTS is outlined as follows:

s0 = [P1,...,Pn;T1,...,Tm;], SlS = MSA(LN(sl−1)) + sl−1, SlM = MoE(LN(Xls)) + Xls,

sl = LN(SlM),

(3)

where we denote the prompt text, target text content representations to (P1,...,Pn) and (T1,...,Tm) respectively. “MSA”, “MoE” and “LN” refer to the multi-head self-attention, mixture of experts and layer

normalization. Xl shows the output of l th block.

To enhance the performance of MoE-TTS in voice timbre control, we adopt a text-context prompt-guided approach to direct the model to synthesize audio with the specified language and timbre. When the model receives an instruction to generate an audio response, the Uni-MoE base model first produces a special token (i.e. <speech start>) indicating the start of speech, . Subsequently, the model generates commands that specify the voice timbre for audio synthesis: fixed timbres include those of Brain, Jenny, and Xiaoxiao, while supported languages cover Chinese and English. Beyond fixed options, the model also enables users to describe desired timbres via natural language; the MoE-TTS module then generates audio with the corresponding timbre based on these descriptive instructions. Upon completion of the timbre prompt, a special token <speech prompt> is generated to signal the end of the prompt. This is followed by the text content that the model needs to transcribe into speech, which is terminated with the special token <speech end>. Through this combination of text-based timbre prompts and target text content, MoE-TTS is guided to perform diverse timbre synthesis.

To synthesize long-form speech, we employ a sentence-splitting strategy during both training and inference. In training, lengthy texts are segmented into short phrases at punctuation marks, which are then batched for efficient processing by MoE-TTS. During inference, we ensure contextual coherence by using the previously generated speech segment to guide the synthesis of the next. This split-and-guide approach enables our model to produce coherent and fluent audio clips exceeding two minutes in duration2.

2Context-Aware MoE-TTS are available: https://huggingface.co/HIT-TMG/Uni-MoE-TTS

- 3. Training and Data Recipes

[Figure 18]

- Figure 4: The overview of the Task-aware Diffusion Transformer (Task-DiT). The role of the projection modules is to map external, task-conditioning features into the latent space of the Diffusion Transformer, where they are utilized as context in cross-attention blocks to guide the image generation.

Image Generation: Task-Aware Diffusion Transformer We introduce a Task-Aware Diffusion Transformer (Task-DiT) that bridges the gap between image understanding and generation. Unlike unified models that suffer from performance trade-offs, our framework preserves the strengths of specialized pre-trained models by connecting them via a lightweight, task-aware bridge. At its core, Task-DiT employs a frozen image generator from PixWizard (Lin et al., 2024b) to safeguard its high-fidelity synthesis capabilities. To steer this generator, we introduce a query-based conditioning mechanism. Two distinct sets of learnable tokens are processed by a powerful understanding module:

- • Task Tokens (<TASK[i]>): Encode high-level commands (e.g., text-to-image, editing, low-level image processing) to specify the generative mode.
- • Image Tokens (<IMG[i]>): Capture the rich semantic essence of the desired output from Uni-MoE-2.0Omni, forming a compressed scene representation.

The enriched features from these tokens are then translated for the generator by dedicated, lightweight projectors. A Task Projector modulates the DiT’s denoising process based on the command, while a content projector transforms the <IMG[i]> features into a dense conditioning sequence for cross-attention. For imageguided tasks, a Visual Projector aligns source image features encoded by ViT with the DiT’s conditioning space. This design creates a versatile and efficient channel for task-aware instruction, enabling high-quality, multi-modal image generation without the catastrophic interference typical of end-to-end fine-tuned models.

### 3 Training and Data Recipes

#### 3.1 Training Recipe: From LLMs to OLMs

Alignment with Pretraining As shown in the leftmost panel of the Figure 5, the goal of pre-training is to enable the LLM to comprehend multimodal data, such as images, videos, and speech. This is achieved by mapping the representations of these modalities into the LLM’s linguistic space, using paired data of multimodal inputs and their corresponding text description.

Supervised Fine-tuning In this stage, we collected a large-scale multimodal instruction-following dataset to enable the model to understand and process any cross-modal information. The training comprises two phases:

• Expert Warmup with Dense Model: We first pre-trained three specialized expert models focusing on mainstream modalities: speech comprehension, speech generation, and visual comprehension. This stage aims to build the model-specific experts for the following stable MoE-based fine-tuning.

- 3.1 Training Recipe: From LLMs to OLMs

[Figure 20]

Figure 5: The training recipe for adapting an LLM into an omnimodal large model.

• Mixed-Data Fine-tuning for MoE-based Model: We then fine-tuned the model on a mixture of all data types using our proposed MoE architecture. The experts in this MoE layer were initialized from the pre-trained experts above, with a final configuration comprising two speech experts, one visual expert, and one null expert. This setup also integrates the inherent language knowledge of the base LLM and two small, shared experts.

For generative tasks, the model employs specialized decoding strategies. In speech generation, the Dense LLM outputs both dialogue content and acoustic control signals, which are then rendered into audio by a Dense TTS model. For image tasks, the model first verbalizes its reasoning before generating the final output tokens for tasks like captioning or editing. This unified approach allows multimodal understanding and generation tasks to be jointly leveraged during instruction tuning, leading to further performance gains.

Annealing Empirical results indicated that the initial modality-specific expert warmup was insufficient to ensure uniform capability across the diverse spectrum of cross-modal tasks after mixed-data fine-tuning. To further calibrate the model and bridge this performance gap, we conducted a subsequent annealing training phase using a balanced mixture of all data types. Crucially, this annealing process was not confined to the omnimodal foundation model; it was also deployed to fine-tune the specialized sub-modules responsible for image editing/generation and text-to-speech (TTS), promoting stability and proficiency throughout the entire architecture.

Omnimodal Reinforcement Learning To develop the Thinking variant of Uni-MoE 2.0, we adopted a training strategy combining cold-start initialization with online reinforcement learning (GSPO) (Zheng

- et al., 2025) and Direct Preference Optimization (DPO) (Rafailov et al., 2024). The cold-start phase aims to stimulate the model’s foundational reasoning capabilities across multimodal inputs—including text, images, and video—while online reinforcement learning enhances the model’s autonomous exploration and further refines its reasoning under sparse reward signals. However, during training, we observed limited improvement in the accuracy reward, particularly when processing full-modality data. To address this, we introduced a DPO stage to specifically strengthen the model’s reasoning ability. In this stage, we reuse rollout samples from the online RL phase under an “LLM as a Judge” mechanism, selecting those with a logical reasoning process and accurate outcomes as positive examples. Additionally, for samples that yielded zero accuracy during online training, we leverage closed-source commercial models (Gemini-2.5-Flash) as a teacher to generate high-quality reasoning demonstrations, thereby improving the student model, Uni-MoE 2.0. The GSPO-DPO pipeline supports iterative refinement for progressively stronger reasoning performance. A comprehensive treatment of this iterative optimization strategy and associated experimental analysis is provided in VIPO-R13 (Li et al., 2025c). The specific datasets are given in Table 20 of the Appendix.

3Iterative Policy Optimization: https://github.com/HITsz-TMG/VerIPO

- 3.2 Data Recipe

Modality Pretraining

Expert Warmup

Omni-Modal Fine-Tuning

Omni-Modal Simulated Annealing

Stage

Image(5M) Video(5M) Audio(5M) Text(5M) Tokens 13B(i&v) & 16B(a) 19B(i) & 9B(v) & 5B(a) 22B(i) & 19B(v) & 8B(a) & 1B(t) 5B(i) & 21B(v) & 6B(a) & 4B(t) Training Components MLP&Qformer MLP&Qformer& MLP of LLM MLP&Qformer&MoE ViT&MLP&Qformer&MoE

Image(17M) Video(0.1M) and Audio(26M)

Image(11M) Video(2M) and Audio(3M)

Image(15M) Video(4M) Audio(9M) Text(4M)

Data

- Table 2: Overview of multi-stage training pipeline, detailing data composition, token volume, and trainable components for each stage. The total training volume is about 75B tokens, with datasets shared across stages.

Generative Training The final stage integrates comprehensive omnimodal capabilities by adding speech and image generation. In this phase, the foundational model remains frozen, while the context-aware MoE-TTS and Task-Aware DiT modules are fine-tuned using the corresponding data from the preceding training stages. This approach preserves the model’s core omnimodal understanding while efficiently scaling its generative abilities to new modalities. Because the foundation model was pre-trained on data related to image and speech generation, this final training stage converges rapidly, and the new generative modules align effectively with the existing model architecture.

- 3.2 Data Recipe

- 3.2.1 Audio-centric Data

Audio Understanding In the pre-training phase of the audio understanding modality, the audio data primarily comprises automatic speech recognition (ASR), audio-caption, and music-caption datasets. Among all pretraining data, ASR datasets constitute the majority, encompassing approximately 15B tokens. In contrast, audio-caption and music-caption datasets are more challenging to acquire than ASR datasets, with a combined total of merely 1B tokens.

During the supervised fine-tuning phase, we reduced the volume of ASR data to 1 billion tokens. To expand the model’s recognition capabilities—originally focused on speech, environmental sounds, and music—toward tasks including speech dialogue, environmental sound question answering, music question answering, and emotion recognition, we collected and constructed a set of relevant datasets for the fine-tuning phase. Furthermore, to activate the model’s capability to generate special tokens required for speech generation, we incorporated data from speech generation tasks, including TTS tasks and speech dialogue tasks, into the training data. In this stage, the total volume of audio data amounts to 5B tokens.

In the annealing stage, we performed sample-level balancing to ensure that the data from all modalities were controlled to be of comparable scale. For the abundant image data, we conducted quality filtering and selected a subset of high-quality samples. For video data, we selected a subset of samples with clear and semantically meaningful audio tracks, which were then expanded into audio–visual unified understanding data. Finally, the image training data amount to approximately 5 billion tokens, while the video training data are expanded to 21 billion tokens.

Speech Generation The training of the MoE-TTS module is divided into three stages. In the first stage, the pre-training activation stage, data with consistent timbre (covering both Chinese and English) is used; this phase primarily aims to activate the capability of the dense model, initialized with the Qwen-0.5B language model, to convert text content of codec tokens into audio content, with a total data volume of 2B tokens. The second stage incorporates a large amount of data with diverse timbres, including two types: data for same-timbre synthesis (where three timbre categories dominate in quantity) and data for stylized custom timbre synthesis (where timbre is controlled via natural language), with a data volume of 5B tokens. In the final stage, training for MoE model activation is conducted using data of the same quantity as the previous stage, resulting in the final version of the MoE-TTS model.

- 3.2.2 Vision-centric Data

Vision-Language Understanding During the vision modality pre-training stage, the visual data primarily comprised image–caption and video–caption datasets. The image–caption corpus contained approximately 17 million samples (13 billion tokens), whereas the video–caption data represented a smaller portion, with around 0.1 million samples (0.2 billion tokens).

- 4. Experiment

In the supervised instruction-tuning stage, we curated a large collection of open-source instruction-tuning datasets encompassing a wide range of tasks, including general image understanding, STEM reasoning, document understanding, visual grounding, and video description. At this stage, the image instruction-tuning data consisted of roughly 22 billion tokens, while the video instruction-tuning data accounted for about 19 billion tokens.

During the annealing stage, we applied sample-level balancing to ensure that data from all modalities were maintained at comparable scales. For the abundant image data, we performed quality filtering to retain only a high-quality subset. For the video data, we selected samples with clear and semantically meaningful audio tracks, which were further extended into audio–visual unified understanding datasets. Ultimately, the image training data comprised approximately 5 billion tokens, while the video training data were expanded to 21 billion tokens.

Image Generating and Editing In the task projector alignment stage and the visual projector alignment stage, we utilized data encompassing a variety of task types. This included image generation, image editing, low-level computer vision tasks such as deraining and denoising, and conditional generation like Canny-toimage. In total, this data amounted to approximately 2M samples (1.5B tokens).

In the caption projector alignment stage, we used high-quality image-caption pairs sourced from a diverse range of data types. This collection included factual, human-annotated descriptions of everyday scenes; detailed, context-rich prompts for complex image generation; and elaborate narratives describing the compositional and spatial relationships between elements. This data accounted for a total of 4.2M samples (3.2B tokens).

In the supervised instruction-tuning stage, we curated and filtered a large-scale dataset from multiple opensource dataset collections, amounting to approximately 10.5M samples (8.1B tokens). This stage aimed to unify the model’s capabilities under a consistent instruction-following framework. The training data was formatted as instruction-response pairs covering the same core task categories mentioned previously: image generation, image editing, low-level CV enhancements, and conditional image synthesis.

In the annealing stage, we utilized a curated, high-quality dataset of selected samples across various task types. This stage placed particular emphasis on dense captions for image generation, generation processes that incorporate reasoning steps, and various long-tail image editing tasks, such as changing human emotion. This dataset consisted of approximately 3M samples (2.8B tokens).

#### 3.2.3 Text-only Data

In the Omni-Modal Fine-tuning stage, we introduce pure textual data to further enhance the model’s understanding and reasoning capabilities. For this stage, the textual data primarily consists of text-only instructional data and mathematical question–answer pairs, totalling approximately 1 billion tokens. In the subsequent Omni-Modal Annealing stage, STEM-related data (e.g., codes, math) are incorporated into the training data to further improve model performance, resulting in approximately 4 billion tokens of textual data at this stage.

### 4 Experiment

To comprehensively evaluate the capabilities of Uni-MoE-2.0-Omni, we conducted experiments on 85 benchmarks, including multimodal or omnimodal understanding (image, text, video, cross/tri-modal), audio and music understanding, speech generation, image generation and editing, and low-level image processing tasks. “Uni-MoE-2.0” refers to the Uni-MoE-2.0-Omni model without annealing training. “Uni-MoE-2.0Base/thinking” represents the omnimodal understanding and thinking model without speech and image generation abilities.

- 4.1 Vision-Language Understanding

- 4.1.1 Image Understanding

As shown in Table 3, Uni-MoE-2.0-Omni achieves performance comparable to other Omnimodal Understanding models on both general image understanding and STEM-related image understanding tasks. In particular, Uni-MoE-2.0-Omni attains the highest performance on the GQA (test-dev) and MathVision (Test) benchmarks. For Doc & OCR tasks, Uni-MoE-2.0-Omni exhibits a certain performance gap compared to the strongest existing Omnimodal Understanding models. We attribute this to the limited availability of Doc & OCR data during pre-training and the consistently low proportion of such data in subsequent training stages.

- 4.1 Vision-Language Understanding

Benchmark Uni-MoE-2.0 Uni-MoE-2.0-Omni Qwen2.5-Omni∗ MiniCPM-o 2.6∗ Baichuan-Omni-1.5∗ Ming-Lite-Omni∗ Ming-Lite-Omni-1.5∗ General

MMBench-EN (Dev) 80.76 80.50 75.42 82.39 82.82 82.65 82.56 MMBench-CN (Dev) 78.69 79.90 68.65 80.50 68.73 78.78 80.07 MMStar 59.72 59.38 59.94 59.59 60.77 63.30 63.80 RealWorldQA 62.22 63.12 64.71 65.23 66.93 64.18 66.01 GQA (test-dev) 61.95 62.18 49.55 59.43 58.86 61.68 61.88 MME-RealWorld 53.73 53.67 49.96 46.94 52.70 58.13 58.55 CV-Bench 75.70 76.46 75.82 75.02 76.72 79.49 79.15

###### STEM

AI2D 80.93 81.35 78.79 82.55 79.24 80.80 82.67 MMMU (Val) 42.67 46.67 44.44 47.33 47.11 51.78 53.44 MMMU-Pro (Standard) 29.48 29.65 36.42 30.69 33.93 33.30 32.49 MMMU-Pro (Vision) 14.91 14.51 13.33 11.62 25.90 12.66 15.90 MathVista (Testmini) 60.80 61.30 56.20 66.20 59.50 69.50 69.00 MathVision (Test) 40.76 36.61 17.14 14.21 21.13 13.29 25.20 LogicVista 31.47 32.81 33.93 37.05 32.59 39.51 37.05

###### Doc & OCR

DocVQA (Test) 79.75 79.53 88.11 82.09 89.94 92.50 93.55 ChartQA 71.64 73.04 74.80 82.80 83.28 85.56 85.80 CharXiv (DQ) 48.80 47.68 59.69 51.32 41.42 67.40 68.15 CharXiv (RQ) 23.20 24.10 25.83 28.60 30.70 26.00 25.83 SEED-Bench-2-Plus 64.21 64.38 68.07 65.26 66.23 68.47 68.25

###### Video

Video-MME (w/o sub) 64.85 66.41 59.78 60.78 59.85 62.04 62.56 LongVideoBench (Val) 56.62 55.35 53.83 51.83 54.00 54.90 55.20 MVBench 69.33 70.53 61.23 58.90 61.12 66.92 68.40 VSI-Bench 53.87 55.97 19.32 25.58 33.89 36.30 37.77 Charades-Sta 27.73 30.62 29.24 16.41 20.41 8.81 10.88

- Table 3: Comparison of Uni-MoE-2.0-Omni and variants with other omnimodal models across General, STEM, Doc & OCR, and Video benchmarks. All results presented in this table are evaluated using the lmms-eval (Zhang et al., 2024a) to ensure consistency and reproducibility. Bold indicates the highest score, and underline indicates the second-highest score for each benchmark.

Model Video-MME VideoMMMU LongVideoBench MVBench VSI-Bench Charades-Sta TOMATO EgoSchema Avg. Vision-Language Models LLaVA-OneVision-7B 58.2 33.9 50.5 56.7 - - - 60.1 LLaVA-Video-7B 63.3 36.1 58.2 58.6 - - - 57.3 NVILA-8B 64.2 20.9 57.7 68.1 - - - 54.3 VideoLLaMA3-7B 66.2 47.0 59.8 69.7 - - - 63.3 InternVL2.5-8B* 64.1 46.0 58.9 71.9 34.6 9.5 28.0 51.2 45.5 Qwen2.5-VL-7B* 63.0 49.6 57.6 66.3 37.7 40.3 22.6 58.4 49.4 Omni Models

Qwen2.5-Omni* 59.8 43.6 55.1 61.2 19.3 29.2 25.5 53.8 43.5 MiniCPM-o 2.6* 60.8 37.6 51.8 58.9 25.6 16.4 25.0 43.2 39.9 Baichuan-Omni-1.5* 59.9 43.5 54.0 61.1 33.9 20.4 25.3 57.5 44.4 Ming-Lite-Omni* 62.0 48.8 54.9 66.9 36.3 8.8 28.2 57.0 45.4 Ming-Lite-Omni-1.5* 62.6 49.3 55.2 68.4 37.8 10.9 34.2 54.5 46.6

Our Models

Uni-MoE-2.0 64.9 39.1 56.6 69.3 53.9 27.7 27.0 52.2 46.3 Uni-MoE-2.0-Omni 66.4 43.6 55.4 70.5 56.0 30.6 27.8 54.3 50.6

- Table 4: Comparison of Uni-MoE-2.0-Omni and variants with other MLLMs across 8 Video benchmarks.

* denotes the reproduced results. When evaluating Video-MME, the subtitles are not used. Bold indicates the highest score, and underline indicates the second-highest score for each benchmark.

This observation also, to some extent, highlights the scarcity of publicly available Doc & OCR datasets in the current open-source ecosystem.

#### 4.1.2 Video Understanding

The evaluation results presented in Table 3 demonstrate that the Uni-MoE-2.0-Omni model achieves SOTA performance across multiple video understanding benchmarks. Specifically, Uni-MoE-2.0-Omni surpasses

Benchmark Uni-MoE-2.0 Uni-MoE-2.0-Omni Qwen2.5-Omni* MiniCPM-o 2.6* Baichuan-Omni 1.5* Ming-Lite-Omni* Ming-Lite-Omni-1.5*

|GPQA Diamond GPQA Main GPQA Extended MMLU-Pro|30.30 32.83 32.37 33.48 32.05 33.15 36.00 38.76<br><br>|27.27 30.81 26.77 24.24 31.31 23.66 29.30 25.46 32.59 31.70 22.16 24.78 25.00 34.70 34.43 32.23 29.80 42.43 42.72 44.75<br><br>|
|---|---|---|
|Avg.<br><br>|32.68 34.56<br><br>|26.33 28.67 29.92 33.56 35.55|

- Table 5: Comparison of Uni-MoE-2.0-Omni and other multimodal models on reasoning and general knowledge benchmarks. All models are tested with direct answers in a zero-shot setting. Bold indicates the highest score, and underline indicates the second-highest score for each benchmark.

the previous SOTA model Ming-Lite-Omni-1.5 by approximately 3.85% on the long video understanding benchmark Video-MME, and by about 18.20% on the spatial reasoning benchmark VSI-Bench.

Furthermore, as shown in Table 4, Uni-MoE-2.0-Omni demonstrates competitive performance across a diverse set of video understanding and reasoning benchmarks. Notably, our model achieves state-of-the-art results on Video-MME (66.4) and VSI-Bench (56.0), while maintaining strong competitiveness on MVBench (70.5). Compared to other omnimodal models, Uni-MoE-2.0-Omni shows particularly outstanding performance in visual-spatial reasoning (VSI-Bench) and comprehensive video QA (Video-MME), surpassing Qwen2.5-Omni by significant margins of 36.7% and 6.6% respectively. These results collectively validate that Uni-MoE-2.0Omni delivers superior video comprehension capabilities, highlighting its robust generalization and reasoning proficiency in multimodal video understanding.

#### 4.1.3 Language Capability

Although the training corpus contains a relatively small amount of text data, as shown in Table 5, our UniMoE-2.0-Omni series models demonstrate strong language capabilities on complex reasoning and knowledge benchmarks such as GPQA and MMLU-Pro. Their average performance (32.68–34.56) significantly surpasses that of a comparable competitor, Qwen2.5-Onmi 7B (27.27) and Ming-Lite-Omni-1.5 (31.31), while maintaining competitive results on challenging scientific QA (GPQA Diamond) and comprehensive academic reasoning (MMLU-Pro). These results indicate that Uni-MoE-2.0-Omni not only performs reliably in specialized domain knowledge but also reflects the effectiveness and scalability of its architecture in complex language understanding and reasoning tasks.

#### Takeaways: Vision and Language

Experiments demonstrate Uni-MoE-2.0-Omni’s strong multimodal performance. It achieves balanced image understanding, leading on GQA and MathVision, though document OCR lags due to limited data. The model excels particularly in video understanding, setting new SOTA on long-video and spatial reasoning tasks, highlighting its generalization ability. Despite scarce text training data, it outperforms peers on complex reasoning and knowledge benchmarks like GPQA Diamond, validating its scalable and effective architecture for challenging tasks.

- 4.2 Audio Understanding and Speech Generation

- 4.2.1 Audio X → Text

As shown in Table 6 and 7, Uni-MoE-2.0-Omni demonstrates competitive performance against other omnimodal models and audio large models across Chinese and English ASR, Speech Understanding, Audio Understanding, and Music Understanding benchmarks.

Uni-MoE-2.0-Omni demonstrates leading performance across diverse audio tasks. It sets a new standard in English Automatic Speech Recognition (ASR), achieving remarkably low Word Error Rates (WER) of 1.73 on LibriSpeech-clean, 3.26 on LibriSpeech-other, and 5.46 on mls-en. The model also delivers strong results in Mandarin ASR, with WERs of 3.69 on Aishell1 and 4.84 on Aishell2. It is worth emphasising that our model possesses strong long-speech understanding capabilities (with an average duration of 3.6 minutes). Specifically, it achieves a low Word Error Rate (WER) of 2.04 and 4.2 on the LibriSpeech-clean-long and LibriSpeech-otherlong test sets, two long-audio benchmarks constructed from short-audio synthesis. Beyond transcription, it excels in semantic understanding, attaining high accuracy on spoken question-answering (RACE-audio: 90.32

Benchmark Uni-MoE 2.0 Uni-MoE-2.0-Omni Qwen2-Audio Qwen2.5-Omni∗ Ming-Lite-Omni∗ Speech Understanding

RACE-audio-middle 90.32 89.69 28.27 92.76 88.30 RACE-audio-high 87.62 87.19 26.95 87.79 80.38 EHSL-short 88.00 90.00 24.00 86.00 82.00 EHSL-long 85.33 87.33 15.33 90.66 83.33 MELD 40.93 40.4 37.92 16.76 39.35 MMAU-speech 64.69 65.00 - 70.97 60.88 MMBench-hint-Speech 97.98 100 - 81.06 95.34 Avg. 79.27 79.94 - 75.14 75.65

Audio Understanding ClothoAQA 61.76 61.83 43.45 62.29 53.43 ClothoV1 37.9 33.4 28.9 21.2 6.3 ClothoV2 38.1 33.4 29.1 30.1 6.3 AudioCaps 33.8 33.6 40.9 37.1 18.5 MMAU-Sound 67.17 68.06 - 71.97 59.3 Avg. 47.74 46.05 - 44.53 28.77

Music Understanding MusicCaps 23.9 62.4 21.8 4.00 0.5 MMAU-Music 59.3 56.4 - 65.33 52.23

- Table 6: Comparison of Uni-MoE-2.0-Omni and variants with other omnimodal models across Speech/Audio/Music Understanding benchmarks. The accuracy (ACC) metric is employed to assess results of speech understanding, AQA and MMAU; the CIDER is used to evaluate results of all captioning tasks. We found that Ming-Lite-Omni-1.5 often fails to follow instructions and generates off-topic content, making it difficult to evaluate its performance accurately.

middle, 87.82 high) and competitive scores on audio captioning tasks (Uni-MoE-2.0-Omni vs. Qwen2.5-Omni: 46.1 vs. 44.5).

However, a performance gap is observed in specialized domains such as Music Understanding when compared to certain benchmarks. We hypothesize that this stems from the relatively lower proportion of high-quality, musically annotated data during pre-training and instruction tuning. This observation not only highlights a specific area for future improvement for our model but also reflects a broader challenge within the multimodal research community: the scarcity of large-scale, open-source music understanding datasets. In summary, Uni-MoE-2.0-Omni proves to be a powerful and versatile model for a wide spectrum of audio-centric tasks, with particularly strong results in speech recognition and general audio comprehension, while its performance in niche domains like music is contingent on the availability of relevant training data.

#### 4.2.2 Text → Speech

As summarised in Table 8, we evaluate the text-to-speech (TTS) performance of our model against several omnimodal models on standardized benchmarks. The evaluation methodology involves generating audio from text and then transcribing it back to calculate the Word Error Rate (WER); we employ Whisper for English and Paraformer for Chinese outputs.

The results demonstrate the competitive capability of Uni-MoE-2.0-Omni in Chinese and English speech generation. On the LibriTTS-clean benchmark, it achieves a WER of 5.85, significantly outperforming models like Ming-lite-Omni (11.15) and approaching the performance of the state-of-the-art Qwen2.5-Omni-7B. Furthermore, on the challenging SEED-hard benchmark, Uni-MoE-TTS attains a WER of 2.67, surpassing both Dense-TTS and Ming-Lite-Omni, which underscores its robustness in complex synthesis scenarios.

For long-form speech synthesis, tests on the TinyStory validation sets reveal a nuanced strength profile: UniMoE-2.0-Omni excels in English, demonstrating superior prosodic consistency, timbre stability, and linguistic fluency, while Ming-Lite-Omni shows a comparative advantage in Chinese. Overall, our model shows the best comprehensive performance in Chinese and English compared to Ming-Lite-Omni and Qwen2.5-Omni-7B.

In conclusion, Uni-MoE-2.0-Omni proves to be a balanced and effective model for omnimodal TTS, exhibiting particular strength in English synthesis and robust performance across diverse and challenging tasks.

Benchmark Uni-MoE-2.0 Uni-MoE-2.0-Omni Qwen2.5-Omni∗ Ming-Lite-Omni∗ Ming-Lite-Omni-1.5∗ Qwen2-Audio ASR-EN ↓

LibriSpeech-clean 1.73 1.66 3.57 5.36 1.34 1.60 LibriSpeech-other 3.26 3.42 7.03 9.89 2.79 3.60 fleurs-en 7.78 7.72 9.74 10.16 8.07 6.90 mls-en 5.46 5.39 6.85 9.66 4.04 5.40 CV15-en 3.63 4.13 12.25 13.75 7.04 8.60 voxpopuli 10.35 9.43 9.6 10.01 7.13 6.84 LibriSpeech-clean-long 3.55 2.04 7.73 43.82 61.86 11.2 LibriSpeech-other-long 6.12 4.2 7.98 32.2 61.49 10.3 Avg. 5.24 4.75 8.09 16.85 19.22 6.81

###### ASR-ZH↓

Aishell1 3.69 3.23 2.63 7.83 1.33 1.53 Aishell2-test-ios 4.84 4.94 23.74 8.05 2.45 2.92 Aishell2-test-android 4.84 4.84 25.14 6.19 2.46 2.92 Fleurs-zh 11.27 9.58 8.87 9.73 8.36 7.50 CV15-zh 3.37 2.97 11.01 19.08 5.96 6.90 Avg. 5.60 5.11 14.28 10.18 4.12 4.35

- Table 7: Comparison of Uni-MoE-2.0-Omni and variants with other omnimodal models across ASR benchmarks. The ASR results presented in this table are evaluated using the word error rate (WER). The LibriSpeech-clean/other-long datasets contain speech samples longer than 3 minutes.

Model LibriTTS-clean LibriTTS-other SEED-en SEED-zh SEED-hard TinyStories-en TinyStories-zh

FireRedTTS - - 1.51 3.82 17.45 - MaskGCT - - 2.27 2.62 10.27 - E2 TTS - - 1.97 2.19 - - F5-TTS - - 1.56 1.83 8.67 - Llasa - - 1.59 2.97 11.09 - CosyVoice 3.17 - 3.39 3.10 11.75 - CosyVoice 2 - - 1.45 2.57 6.83 - GLM-4-Voice 5.64 - 2.91 2.10 - - Qwen2.5-Omni-7B∗ 5.20 6.68 1.73 1.68 2.15 6.20 8.51 Ming-Lite-Omni∗ 11.15 11.33 2.92 2.68 5.52 15.07 4.74

Dense-TTS 6.51 6.84 3.03 3.41 3.1 - Uni-MoE-2.0-Omni 5.85 7.13 2.72 3.10 2.67 5.02 7.02

- Table 8: Comparison of Uni-MoE-2.0-Omni and variants with other omnimodal models across TTS benchmarks. For English TTS audio results, they are transcribed into text using Whisper. For Chinese TTS audio results, Paraformer is utilized to obtain the transcriptions. Subsequently, these transcriptions are compared with the original text using the WER metric. Bold indicates the highest score, and underline indicates the second-highest score for each benchmark.

#### 4.2.3 Speech → Speech/Text

- As shown in Table 9, Uni-MoE-2.0-Omni demonstrates competitive performance against other omnimodal models across Speech QA benchmarks, including LlamaQA, WebQA, BigBench Audio, and MultiChallenge Audio. Notably, Uni-MoE-2.0-Omni achieves remarkable results in several key tasks: in the LlamaQA (s→s) subtask, it attains an accuracy of 75.33, ranking second only to the top-performing model, which underscores its strong ability to handle speech-to-speech reasoning in this scenario. In the BigBench Audio (s→s) task, it achieves 44.7, showcasing its robustness in audio-related QA. For some challenging benchmarks, such as MultiChallenge Audio, Uni-MoE-2.0-Omni still maintains a competitive position, indicating its versatility across diverse speech QA scenarios. While there is a performance gap compared to the strongest models like Ming-Lite-Omni on certain tasks, we attribute this to the relatively limited specialized Knowledge QA data during pre-training and the low proportion of such data in subsequent training stages.

LlamaQA WebQA BigBench Audio MultiChallenge Audio

Model

s → t s → s ∆t−s s → t s → s ∆t−s s → t s → s ∆t−s s → t s → s ∆t−s

Spectron-1B 21.9 - - 6.1 - - - - - - - SpeechGPT-7B 21.6 - - 6.5 - - - - - - - Freeze-Omni-7B 72 - - 44.73 - - - - - - - Moshi-7B 62.3 21.0 41.3 26.6 9.2 17.4 - - - - - LLaMA-Omni-7B 67.7 49.0 18.7 33.4 23.7 9.7 - - - - - LLaMA-Omni2-7B 70.3 60.7 9.6 34.5 31.3 3.2 - - - - - VITA-1.5-7B 76.7 - - 42.7 - - - - - - - Stream-Omni-8B 76.3 65.0 11.3 44.2 27.5 16.7 - - - - - OpenOmni-7B 74.6 67.2 7.4 44.5 28.9 15.6 - - - - - NExT-Omni-7B 78.4 66.4 12 45.6 28.3 17.3 - - - - - Step-Audio2-mini-7B - - - - - - 50.90 47.50 3.40 13.64 8.08 5.56 Kimi-Audio-7B - - - - - - 59.40 51.00 8.40 7.07 1.01 6.06 GLM-4-Voice-8B 74.33 65.67 8.66 45.90 43.20 2.70 44.80 42.70 2.10 9.09 6.06 3.03 Qwen2.5-Omni-7B∗ 77.33 77.33 0.00 48.28 48.28 0.00 58.1 53.8 4.3 13.13 10.61 2.52 Ming-Lite-Omni∗ 80.33 63.66 16.67 53.79 44.19 9.60 53.3 26.6 26.7 27.27 23.23 4.04

Uni-MoE-2.0-Omni 75.33 75.33 0.00 45.13 43.95 1.18 49.2 44.7 4.5 10.61 9.6 1.01

- Table 9: Comparison of Uni-MoE-2.0-Omni and variants with other omnimodal models across Speech QA benchmarks. We evaluate the model’s performance by employing existence matching to assess the ACC of the answers. Some results (models w/o ∗) are from the MOSS speech and GLM-4-voice, where a lower ∆t−s value indicates better performance.

Model

Speech-Image QA Speech-Video QA

Avg.

A-OK-VQA Speech (Reasoning) VQAv2 Speech ActivityNet Speech s, i → t s, i → s s, i → t s, i → s s, v → t s, v → s

Qwen2.5-Omni-7B* 58.55 51.65 76.84 71.01 60.00 58.91 62.83 Ming-Lite-Omni* 65.08 42.69 81.21 30.43 58.43 0.03 55.57

Uni-MoE-2.0-Omni 65.73 52.58 78.03 71.15 60.16 57.94 64.27

- Table 10: Comparison of Uni-MoE-2.0-Omni and variants with other omnimodal models across SpeechImage/Video QA benchmarks. For Speech-Image QA tasks, we evaluate the model’s performance using the ACC metric. For Speech-Video QA tasks, we adopt GPT-based evaluation to assess their performance.

#### 4.2.4 Vision + Speech → Speech/Text

- As shown in Table 10, Uni-MoE-2.0-Omni demonstrates strong and balanced performance across multimodal question-answering tasks, establishing a compelling advantage over comparable models.

In the Speech-Image QA domain, our model achieves top performance on the challenging A-OK-VQA Speech benchmark, leading all competitors in both text (65.73) and speech (52.58) output modalities. This dual dominance highlights Uni-MoE-2.0-Omni’s superior capability in integrating visual and auditory information for complex reasoning.

This strength extends to the Speech-Video QA realm. On the ActivityNet benchmark, Uni-MoE-2.0-Omni achieves the highest score (60.16) for text-based answers (s→t), demonstrating excellent comprehension of temporal and visual narratives. More importantly, it maintains a very strong performance for direct speech answers (s→s, 57.94), a task where other models like Ming-Lite-Omni fail catastrophically. This result underscores a key advantage of our model: its unique ability to generate high-quality, contextually accurate responses directly in the speech modality without sacrificing performance, a capability where most other omnimodal models show significant weakness.

Overall, while other models may excel in a single metric, Uni-MoE-2.0-Omni distinguishes itself through its robust and consistent performance across both text and speech output tasks, making it a more versatile and reliable solution for real-world multimodal applications.

- 4.3 Omnimodality Understanding

Method WorldSense StreamingBench (Omni) OmniVideoBench OmniBench Avg. Vision-Language Models

LLaVA-OneVision-7B 37.7 40.8 - - LLaVA-Video-7B 40.2 41.7 - - Qwen2.5-VL-7B 38.3 45.0 29.8 - -

Omni Models Unified-IO-2 XL 24.7 - - 38.0 Unified-IO-2 XXL 25.9 - - 34.0 VideoLLaMA 2 25.4 35.9 29.2 - Qwen2.5-Omni-7B* 43.1 47.1 29.8 26.2 36.6 MiniCPM-o 2.6* 43.2 51.0 34.7 36.7 41.4 Baichuan-Omni-1.5* 42.5 47.1 35.0 42.9 41.9 Ming-Lite-Omni* 42.2 38.8 33.3 43.5 39.4 Ming-Lite-Omni-1.5* 43.5 40.4 32.1 47.7 40.9

###### Our Models

Uni-MoE-2.0 42.8 39.8 34.3 46.5 40.8 Uni-MoE-2.0-Omni 44.7 48.1 35.1 47.1 43.7

- Table 11: Comparison of Uni-MoE-2.0-Omni and other MLLMs across 4 Omnimodal understanding benchmarks. * denotes the reproduced results. Bold indicates the highest score, and underline indicates the second-highest score for each benchmark.

Takeaways: Audio + X →Speech/Text

Uni-MoE-2.0-Omni demonstrates comprehensive and competitive capabilities across diverse audio and speech tasks, establishing balanced performance in recognition, understanding, and generation while showing particular strength in multimodal integration.

#### 4.3 Omnimodality Understanding

We present in the Table 11 the performance of the Uni-MoE-2.0-Omni model on four omnimodal benchmarks that require simultaneous understanding of visual and audio information. On WorldSense and OmniVideoBench, which evaluate long video omnimodal comprehension, Uni-MoE-2.0-Omni achieves SOTA performance. On StreamingBench (Omni), which focuses on shorter videos, and OmniBench, which assesses joint image-audio understanding, our model ranks second. Across all evaluation metrics, Uni-MoE-2.0-Omni attains a SOTA overall score of 43.7%, outperforming the second-best model Baichuan-Omni-1.5 (41.9%), by approximately 2%. These results demonstrate that Uni-MoE-2.0-Omni possesses a strong omni-modal understanding capability.

#### 4.4 Image Generation and Edition

As shown in Table 12 and Table 13, our Uni-MoE-2.0-Omni models demonstrate strong and versatile performance. They particularly excel in controllable generation and low-level image restoration, significantly outperforming other omni-models. While specialized generators may lead in pure generation, our models remain highly competitive in complex editing and conditional tasks, showcasing their versatility.

Image Generation In pure image generation, our models are competitive. Notably, Uni-MoE-2.0-Omni (0.44) surpasses the original PixWizard (0.43) on the Wise benchmark. While specialized models like Qwen-Image lead on the Wise, our FID score (18.04) is highly competitive, outperforming several omnimodalmodels. For instance, it is 9.0% lower (better) than JanusPro-7B (19.82) and 29.2% lower than Bagel (25.47).

Image Edition Our models show particularly robust performance in image editing. On GEdit-Bench, Uni-MoE2.0-Omni achieves a score of 6.02, which is 88.1% higher (better) than the original PixWizard (3.20). On the Emu Edit benchmark, our model’s score of 0.076 is also 94.8% higher than PixWizard (0.039), demonstrating strong performance in instruction-following edits. While state-of-the-art models like Qwen-Image (7.42 GEdiT, 0.127 Emu) still lead, our model is highly competitive. Furthermore, the Uni-MoE-2.0-Image model (MoE and DiT training version) demonstrates even stronger capabilities on complex tasks, especially on the

- 4.4 Image Generation and Edition

Image generation Image edition Wise ↑ FID ↓ GEdit-Bench ↑ Emu Edit ↑

Model

MagicBrush CLIPimg ↑ DINOimg ↑ CLIPout ↑

Image Generation Models SDXL 0.48 13.63 - - - - PixWizard 0.43 11.99 3.20 0.039 0.907 0.811 0.298 Qwen-Image 0.63 25.37 7.42 0.127 0.920 0.800 0.313

Omni Models JanusPro-7B 0.41 19.82 - - - - Bagel 0.49 25.47 6.52 0.124 0.921 0.844 0.310 OmniGen 0.40 29.32 5.61 0.091 0.907 0.802 0.298 OmniGen2 0.45 31.65 6.10 0.103 0.899 0.794 0.307 Ming-Lite-Omni 0.54 16.86 5.55 0.052 0.845 0.683 0.299 Ming-Lite-Omni-1.5 0.52 32.39 6.09 0.094 0.910 0.822 0.309

###### Our Models

Uni-MoE-2.0-Omni 0.44 18.04 6.02 0.076 0.789 0.590 0.288 Uni-MoE-2.0-Image 0.46 18.95 6.00 0.080 0.854 0.714 0.293

#### Table 12: Comparison of models across Image Generation and Image Edition benchmarks.

Controllable Generation Low-Level Image Restoration Canny-to-Image Depth-to-Image Derain Denoise

Model

F1-Score ↑ FID ↓ CLIP-S ↑ RMSE ↓ FID ↓ CLIP-S ↑ PSNR ↑ SSIM ↑ PSNR ↑ SSIM ↑ Image Generation Models

PixWizard 0.24 18.32 28.88 42.61 23.41 27.59 24.62 0.77 27.75 0.81 Qwen-Image 0.47 37.59 27.45 51.23 27.54 24.81 26.37 0.80 22.19 0.46

###### Omni Models

Bagel 0.17 130.44 24.36 64.85 39.40 27.53 17.14 0.59 18.14 0.25 OmniGen 0.42 32.15 29.25 32.09 16.98 29.81 13.57 0.22 17.05 0.19 OmniGen2 0.16 45.67 27.35 59.57 52.30 26.47 22.22 0.77 16.78 0.41 Ming-Lite-Omni 0.17 154.95 21.83 85.71 126.31 20.33 11.63 0.21 17.61 0.51 Ming-Lite-Omni-1.5 0.20 187.42 21.67 87.53 134.01 20.78 16.01 0.38 14.35 0.13

###### Our Models

Uni-MoE-2.0-Omni 0.24 20.23 28.58 42.41 27.45 27.00 25.41 0.82 25.70 0.48 Uni-MoE-2.0-Image 0.24 18.23 29.25 44.23 21.91 27.60 25.69 0.82 26.01 0.47

#### Table 13: Comparison of models across Controllable Generation and Low-Level Image Restoration benchmarks.

MagicBrush benchmark, where it significantly improves upon the base model’s CLIPImg score (0.854 vs. 0.789).

Controllable Generation This is a particularly strong area for our models. For Canny-to-Image tasks, our Uni-MoE-2.0-Image model achieves a dramatically better FID score (18.23) compared to both the specialized Qwen-Image (37.59) and the omni-model OmniGen2 (45.67). While its F1-Score (0.24) is lower than QwenImage (0.47), it surpasses OmniGen2 (0.16). For Depth-to-Image tasks, our model (27.45 FID, 42.41 RMSE) outperforms both Qwen-Image (27.54 FID, 51.23 RMSE) and OmniGen2 (52.30 FID, 59.57 RMSE) on the key metrics of FID and RMSE, demonstrating superior controllable generation.

Low-Level Image Restoration In low-level image restoration tasks, our Uni-MoE-2.0-Omni model shows superior performance. For Derain tasks, its PSNR (25.41) is highly competitive with the specialized QwenImage (26.37) and significantly better than OmniGen2 (22.22). Notably, its SSIM (0.82) is superior to both Qwen-Image (0.80) and OmniGen2 (0.77). For Denoise tasks, our model (25.70) outperforms Qwen-Image (22.19) by 15.8% and OmniGen2 (16.78) by 53.1% in terms of PSNR.

[Figure 31]

- Figure 6: The cases of Image Generation, Image Edition, Controllable Generation, and Low-Level Image Restoration.

#### Takeaways: Image Generating and Editing

While highly competitive in pure image generation, our model’s primary strength lies in its versatility and control. It demonstrates state-of-the-art performance in conditional tasks like depth-to-image generation and excels in low-level restoration like image denoising, outperforming strong specialized models and establishing a new benchmark for omnimodal capabilities.

#### 4.5 MoE Analysis

Expert Activation To better understand the operational dynamics of our Dynamic-Capacity MoE, we visualise expert routing probabilities across layers for diverse tasks (Figure 7). The results reveal that initial layers exhibit nearly identical routing across tasks, indicating common set of experts for low-level feature extraction. In deeper layers, routing patterns diverge, reflecting task-specific specialization, while the probability of selecting the null expert (Expert 5) decreases sharply, suggesting that computation is increasingly concentrated in semantically critical stages. The activation profiles align with the intended expert roles: Expert 1 dominates vision-centric tasks (Image, Video and omnimodal Understanding), Experts 2 and

- 3 are prominent in audio-related tasks (Audio and omnimodal Understanding), and Expert 4 consistently contributes across tasks as a general-purpose semantic expert. Notably, understanding tasks show dynamic, layer-dependent routing, implying adaptive allocation of computational resources from concrete to abstract representations. In contrast, generation tasks exhibit stable, uniform routing across layers, consistent with

1.0

1.0

1.0

1.0

0.8

0.8

0.8

0.8

0.6

0.6

0.6

0.6

0.4

0.4

0.4

0.4

0.2

0.2

0.2

0.2

Expert 1 Expert 2 Expert 3 Expert 4 Expert 5

Expert 1 Expert 2 Expert 3 Expert 4 Expert 5

Expert 1 Expert 2 Expert 3 Expert 4 Expert 5

Expert 1 Expert 2 Expert 3 Expert 4 Expert 5

0.0

0.0

0.0

0.0

1 3 5 7 9 11 13 15 17 19 21 23 25 27

1 3 5 7 9 11 13 15 17 19 21 23 25 27

1 3 5 7 9 11 13 15 17 19 21 23 25 27

1 3 5 7 9 11 13 15 17 19 21 23 25 27

Layer

Layer

Layer

Layer

(a) Overall

(b) Image Understanding

(c) Video Understanding

(d) Audio Understanding

1.0

1.0

1.0

1.0

0.8

0.8

0.8

0.8

0.6

0.6

0.6

0.6

0.4

0.4

0.4

0.4

0.2

0.2

0.2

0.2

Expert 1 Expert 2 Expert 3 Expert 4 Expert 5

Expert 1 Expert 2 Expert 3 Expert 4 Expert 5

Expert 1 Expert 2 Expert 3 Expert 4 Expert 5

Expert 1 Expert 2 Expert 3 Expert 4 Expert 5

0.0

0.0

0.0

0.0

1 3 5 7 9 11 13 15 17 19 21 23 25 27

1 3 5 7 9 11 13 15 17 19 21 23 25 27

1 3 5 7 9 11 13 15 17 19 21 23 25 27

1 3 5 7 9 11 13 15 17 19 21 23 25 27

Layer

Layer

Layer

Layer

(e) Omni Understanding

(f) Image Generation.

(g) Image Edit

(h) Audio Generation.

- Figure 7: Analysis of expert activation proportion of Uni-MoE-2.0-Omni across different subtasks. The five experts shown in this figure are four routed experts (E1-E4, colored) and the null expert (E5, green). The vertical axis represents the proportion of tokens assigned to each expert at layer i for the current task.

1 3 5 7 9 11 13 15 17 19 21 23 25 27

Layer

0.0

0.2

0.4

0.6

0.8

1.0

Expert 1 Expert 2 Expert 3 Expert 4

(a) Overall

1 3 5 7 9 11 13 15 17 19 21 23 25 27

Layer

0.0

0.2

0.4

0.6

0.8

1.0

Expert 1 Expert 2 Expert 3 Expert 4

(b) Image Understanding

1 3 5 7 9 11 13 15 17 19 21 23 25 27

Layer

0.0

0.2

0.4

0.6

0.8

1.0

Expert 1 Expert 2 Expert 3 Expert 4

(c) Video Understanding

1 3 5 7 9 11 13 15 17 19 21 23 25 27

Layer

0.0

0.2

0.4

0.6

0.8

1.0

Expert 1 Expert 2 Expert 3 Expert 4

(d) Audio Understanding

1 3 5 7 9 11 13 15 17 19 21 23 25 27

Layer

0.0

0.2

0.4

0.6

0.8

1.0

Expert 1 Expert 2 Expert 3 Expert 4

(e) Omni Understanding

1 3 5 7 9 11 13 15 17 19 21 23 25 27

Layer

0.0

0.2

0.4

0.6

0.8

1.0

Expert 1 Expert 2 Expert 3 Expert 4

(f) Image Generation

1 3 5 7 9 11 13 15 17 19 21 23 25 27

Layer

0.0

0.2

0.4

0.6

0.8

1.0

Expert 1 Expert 2 Expert 3 Expert 4

(g) Image Edit

1 3 5 7 9 11 13 15 17 19 21 23 25 27

Layer

0.0

0.2

0.4

0.6

0.8

1.0

Expert 1 Expert 2 Expert 3 Expert 4

(h) Audio Generation

- Figure 8: Visualization of the dynamic computational budget allocated by our Top-P routing mechanism. The figure illustrates the proportion of tokens activating a varying number of experts at each layer, revealing a “peak-trough-peak-fall” pattern where more computational resources are adaptively assigned to the middle layers. In this figure, "Expert 2" represents tokens activating either 2 experts.

a homogeneous refinement process. These observations confirm that our MoE design effectively promotes functional specialization and learns efficient, task-dependent computation pathways.

Computational Cost Analysis of the Top-P routing mechanism (Figure 8) shows a structured allocation of computational budget across layers, measured by the proportion of tokens activating different numbers of experts. The distribution follows a distinct peak–trough–peak–fall pattern: high load in early layers (1–3) for general-purpose feature extraction, a brief reduction in layers 4–6, a primary peak in middle-to-deep layers (7–21) corresponding to complex reasoning and feature integration, and a final decline in layers 21–27 as processing converges to output. While consistent across tasks, modality-specific differences emerge: temporal inputs such as Video and Audio exhibit a stronger initial peak than static Image, with more tokens activating multiple experts, indicating greater parallel resource needs for spatiotemporal processing. These results suggest that the routing mechanism learns both a global computational structure and fine-grained, modality-aware resource allocation.

Routing Dynamics We tracked expert activation at four representative layers (Figure 9). Expert usage remains stable in the shallowest (Layer 0) and deepest (Layer 27) layers, while middle layers (Layers 9 and 18) show clear changes during training. This indicates that most refinement of expert specialization happens in the middle of the network, where both routing patterns and expert parameters are actively optimized. A key trend in these middle layers is the steady increase in activation of the null expert (Expert 5), meaning the model learns to skip tokens that no longer need processing at intermediate stages. This behavior confirms the

- 4.6 Thinking vs. No-Thinking

Expert 1 Expert 2 Expert 3 Expert 4 Expert 5

Expert 1 Expert 2 Expert 3 Expert 4 Expert 5

0.19

0.20

0.18

0.17

0.18

0.16

0.15

0.16

0.14

0.14

0.13

0.12

0.12

0.11

0 50000 100000 150000 200000

0 50000 100000 150000 200000

Training Step

Training Step

(a) Layer0

(b) Layer9

- 0.200

Expert 1 Expert 2 Expert 3 Expert 4 Expert 5

(c) Layer18

0 50000 100000 150000 200000

Training Step

0.00

0.05

0.10

0.15

0.20

0.25

0.30

Expert 1 Expert 2 Expert 3 Expert 4 Expert 5

(d) Layer27

Figure 9: Analysis of expert activation proportion of Uni-MoE 2.0 during annealing training steps. The five experts shown in this figure are four routed experts (E1-E4, colored) and the null expert (E5, green).

Method MathVista (testmini) MathVerse (vision) LogicVista (testmini) MMMU (val) Avg. Non-Thinking

Uni-MoE-2.0 60.80 17.26 31.47 42.67 38.05 Uni-MoE-2.0-Base 61.30 18.15 32.81 46.67 39.73

Thinking

Uni-MoE-2.0-ColdStart 55.50 19.54 28.35 39.67 35.77 Uni-MoE-2.0-GSPO 58.90 21.19 33.71 47.11 40.23 Uni-MoE-2.0-DPO 63.90 22.97 34.82 45.78 41.87

Table 14: Comparison of Uni-MoE-2.0-Thinking and Base (non-thinking) version. Bold indicates the highest score, and underline indicates the second-highest score for each benchmark.

value of the null expert for efficient computation and shows that our training strategy effectively improves inference efficiency.

4.6 Thinking vs. No-Thinking

- 4.6.1 Visual Reasoning

In Table 14, we present the performance of Uni-MoE-2.0 models at various reinforcement learning stages, as well as their comparison with the original direct-answer models. It can be observed that a significant performance drop in the Uni-MoE-2.0-Cold Start model. We attribute this to the limited amount of Cold-Start data used, which contained complex and diverse tasks that differed substantially from the main evaluation focus on mathematical and scientific reasoning, resulting in a noticeable decline in generalization capability.

After continuing GSPO training based on the Cold-Start model, we find that Uni-MoE-2.0-GSPO shows performance recovery across all evaluation metrics, even surpassing the original Uni-MoE-2.0 model in MathVerse, LogicVista, and MMMU. In particular, MathVerse performance increased by 3.04%.

Finally, after further applying DPO training on top of Uni-MoE-2.0-GSPO, the model exhibits a clear upward trend in performance, especially achieving a 5% improvement on MathVista (testmini) and an average gain of

- 1.64% across all evaluation metrics. This demonstrates the effectiveness of the DPO stage, showing that using

- 0.175

0.150

0.125

0.100

0.075

0.050

0.025

0 50000 100000 150000 200000

Training Step

- 5. Discussion and Future Work

[Figure 35]

- Figure 10: Comparison of image generation results with and without thinking guidance. The “w.o. Thinking” column shows images generated directly from prompts, while the “w. Thinking” column illustrates results produced after incorporating step-by-step reasoning. The middle column presents the structured thinking process guiding the model toward more accurate and contextually faithful image synthesis.

a small amount of DPO data annotated by powerful commercial models can significantly enhance the model’s reasoning ability.

#### 4.6.2 Visual Generation

Figure 10 demonstrates that integrating a structured thinking process markedly improves the faithfulness and coherence of generated images. The baseline model, lacking explicit reasoning, frequently generates semantically inconsistent visuals (e.g., a rodent in an arid habitat or apple trees bearing fruit in winter). Conversely, when guided by a step-by-step reasoning chain, the model successfully parses prompts into constituent visual elements—such as texture, environment, and seasonal context—resulting in outputs that are precisely aligned with the prompt’s semantic intent. Unlike Qwen2.5-Omni and Ming-Lite-Omni, our model incorporates a thinking chain for image generation, similar to the approach used in BAGEL.

#### Takeaways: Thinking

Our method of integrating a structured “thinking” process significantly enhances model performance. It boosts complex reasoning in multimodal understanding tasks with the GSPO-DPO training strategy and ensures faithful, coherent outputs in text-to-image generation tasks.

### 5 Discussion and Future Work

Audio Understanding and Generating Our investigation reveals two key principles for token-based TTS: a dual-rate token strategy and strategic architectural scaling. We find that while a compact 20 tokens per second suffices for audio representation, generation requires 40 tokens per second to capture fine-grained acoustic details. Furthermore, smaller (0.5B) autoregressive models struggle with style conversion, a limitation we address by adopting a MoE architecture. For these models, training on token embeddings from a pre-trained base model proves more effective for capturing vocal style than end-to-end training. Together, these approaches form an effective framework for building efficient and expressive neural spoken systems. Building on these insights, our future work will follow Uni-MoE-Audio, using a single tokenizer for both understanding and generation tasks. This foundation will enable cross-modal training with text and video, facilitating context-

- 6. Conclusion

aware and voice-preserved applications. Concurrently, we will optimize our MoE architecture with conditional routing for more efficient and controllable multi-speaker synthesis.

Image Generating and Editing To facilitate collaborative development, we decoupled the image generation module from the base model. This architectural choice not only simplifies joint training but also enables flexible image generation control through natural language and specialized tokens. While the current text-to-image performance in automated metrics remains limited—due to the constrained version of our external diffusion model and the scarcity of text–image paired data—we observe notable improvements in image editing and low-level image processing tasks. These gains suggest the viability of a language-centric training strategy for integrating diverse image processing capabilities. In future work, we will further refine the architecture for image editing and generation, allowing the base model to perceive and interpret such tasks during pre-training. Nevertheless, the iterative diffusion process for image synthesis and editing will remain externalized. This design reflects our view that multi-step temporal modelling (as in diffusion) operates at a fundamentally different temporal granularity compared to single-step next-token prediction.

Omnimodality Understanding We identify that the model’s enhanced omnimodality understanding capability stems primarily from its audio-text-vision joint coding training on large-scale video data. Future work will focus on scaling this video data and introducing new multimodal positional encoding methods to advance the model’s comprehension abilities further.

Model Architecture Our analysis indicates that a dense model can be directly extended to an MoE architecture. This approach enables the model to achieve more comprehensive capabilities even with limited additional data (75B tokens). Building on this, our future work will focus on optimizing expert specialization and leveraging large-scale dense model distillation. This strategy will allow us to efficiently construct a unified, fine-grained MoE-based model for omnimodal understanding and generation.

Training and Data Recipe Our training approach, which builds upon the progressive strategy from Uni-MoE

- 1.0, was augmented in version 2.0 with annealing experiments and reinforcement learning. Comparative results demonstrate that the omnimodal large model’s training is highly sensitive to data recipe, yet our progressive training strategy ensured remarkable stability during the RL phase for MoE-based models. Moving forward, we will refine this iterative RL strategy by introducing distinct RL methods at various training stages to enhance model capabilities periodically.

### 6 Conclusion

In this paper, Uni-MoE-2.0-Omni represents a significant advancement in the development of open-source, omnimodal large models. By building upon the dense Qwen2.5-7B architecture and introducing key innovations—a dynamic-capacity MoE design, a progressive training strategy enhanced with iterative reinforcement learning, and a curated multimodal data matching technique—the model achieves robust capabilities in understanding, reasoning, and generating across text, image, and speech modalities. Notably, we explored a progressive model architecture evolution and training strategy optimization, which can extend dense large language models into efficient MoE-based omnimodal large models, achieving a leap from multimodal understanding to both understanding and generation. Extensive evaluations across 85 benchmarks confirm that Uni-MoE-2.0-Omni sets a new state-of-the-art or is highly competitive with leading omnimodal large models, demonstrating particular strengths in video understanding, omnimodal comprehension, long speech understanding and generating, audio-visual, controllable image generation, and low-level image restoration tasks. The commitment to open-sourcing the model’s code, checkpoints, and data ensures transparency and fosters further innovation in the field of multimodal artificial intelligence.

### 7 Acknowledgment

We would like to express our sincere gratitude to Qi Wang, Qixun Teng, Feifan Wen, and Zitao Li for their contributions to data collection and demo recording. We are thankful to Jinchao Li and Tongshu Bian for their assistance in paper proofreading and video production. We also thank Longyue Wang and Wenhan Luo for their valuable suggestions on paper writing.

- 8. Contributors

### 8 Contributors

Project Leaders Min Zhang, Baotian Hu Contributors (∗ Core Contributions)

Yunxin Li∗, Xinyu Chen∗, Shenyuan Jiang∗, Haoyuan Shi∗, Zhenyu Liu∗, Xuanyu Zhang, Nanhao Deng, Zhenran Xu, Yicheng Ma, Meishan Zhang

Corresponding Authors Baotian Hu Harbin Institute of Technology, Shenzhen Email: hubaotian@hit.edu.cn Min Zhang Harbin Institute of Technology, Shenzhen Email: zhangmin2021@hit.edu.cn

### References

Abdelrahman Abdelhamed, Stephen Lin, and Michael S Brown. A high-quality denoising dataset for smartphone cameras. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1692–1700, 2018.

Abdullah Abuolaim and Michael S Brown. Defocus deblurring using dual-pixel data. In European conference on computer vision, pp. 111–126. Springer, 2020.

Adaeze Adigwe, Noé Tits, Kevin El Haddad, Sarah Ostadabbas, and Thierry Dutoit. The emotional voices database: Towards controlling the emotion dimension in voice generation systems. CoRR, abs/1806.09514,

2018. URL http://arxiv.org/abs/1806.09514.

Eirikur Agustsson and Radu Timofte. Ntire 2017 challenge on single image super-resolution: Dataset and study. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pp. 126–135, 2017.

Inclusion AI, Biao Gong, Cheng Zou, Chuanyang Zheng, Chunluan Zhou, Canxiang Yan, Chunxiang Jin, Chunjie Shen, Dandan Zheng, Fudong Wang, Furong Xu, Guangming Yao, Jun Zhou, Jingdong Chen, Jianxin Sun, and et al. Ming-omni: A unified multimodal model for perception and generation. CoRR, abs/2506.09344, 2025. doi: 10.48550/ARXIV.2506.09344. URL https://doi.org/10.48550/ arXiv.2506.09344.

Philip Anastassiou, Jiawei Chen, Jitong Chen, Yuanzhe Chen, Zhuo Chen, Ziyi Chen, Jian Cong, Lelai Deng, Chuang Ding, Lu Gao, Mingqing Gong, Peisong Huang, Qingqing Huang, Zhiying Huang, Yuanyuan Huo, Dongya Jia, and et al. Seed-tts: A family of high-quality versatile speech generation models. CoRR, abs/2406.02430, 2024. doi: 10.48550/ARXIV.2406.02430. URL https://doi.org/10.48550/ arXiv.2406.02430.

Codruta O Ancuti, Cosmin Ancuti, Mateu Sbert, and Radu Timofte. Dense-haze: A benchmark for image dehazing with dense-haze and haze-free images. In 2019 IEEE international conference on image processing (ICIP), pp. 1014–1018. IEEE, 2019.

Codruta O Ancuti, Cosmin Ancuti, and Radu Timofte. Nh-haze: An image dehazing benchmark with nonhomogeneous hazy and haze-free images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pp. 444–445, 2020.

Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi Parikh. VQA: visual question answering. In 2015 IEEE International Conference on Computer Vision, ICCV 2015, Santiago, Chile, December 7-13, 2015, pp. 2425–2433. IEEE Computer Society, 2015. doi: 10.1109/ICCV.2015.279. URL https://doi.org/10.1109/ICCV.2015.279.

- R. Ardila, M. Branson, K. Davis, M. Henretty, M. Kohler, J. Meyer, R. Morais, L. Saunders, F. M. Tyers, and G. Weber. Common voice: A massively-multilingual speech corpus. In Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020), pp. 4211–4215, 2020.

Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. Semantic parsing on freebase from question-answer pairs. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, EMNLP 2013, 18-21 October 2013, Grand Hyatt Seattle, Seattle, Washington, USA, A meeting of SIGDAT, a Special Interest Group of the ACL, pp. 1533–1544. ACL, 2013. doi: 10.18653/V1/D13-1160. URL https://doi.org/10.18653/v1/d13-1160.

Akhiad Bercovich, Itay Levy, Izik Golan, Mohammad Dabbah, Ran El-Yaniv, Omri Puny, Ido Galil, Zach Moshe, Tomer Ronen, Najeeb Nabwani, Ido Shahaf, Oren Tropp, and et al. Llama-nemotron: Efficient reasoning models, 2025. URL https://arxiv.org/abs/2505.00949.

Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 18392–18402, 2023.

Hui Bu, Jiayu Du, Xingyu Na, Bengu Wu, and Hao Zheng. AISHELL-1: an open-source mandarin speech corpus and a speech recognition baseline. In 20th Conference of the Oriental Chapter of the International Coordinating Committee on Speech Databases and Speech I/O Systems and Assessment, O-COCOSDA 2017, Seoul, South Korea, November 1-3, 2017, pp. 1–5. IEEE, 2017. doi: 10.1109/ICSDA.2017.8384449. URL https://doi.org/10.1109/ICSDA.2017.8384449.

Jianrui Cai, Hui Zeng, Hongwei Yong, Zisheng Cao, and Lei Zhang. Toward real-world single image superresolution: A new benchmark and a new model. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 3086–3095, 2019.

Houwei Cao, David G. Cooper, Michael K. Keutmann, Ruben C. Gur, Ani Nenkova, and Ragini Verma. CREMA-D: crowd-sourced emotional multimodal actors dataset. IEEE Trans. Affect. Comput., 5(4):377– 390, 2014. doi: 10.1109/TAFFC.2014.2336244. URL https://doi.org/10.1109/TAFFC.2014. 2336244.

Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3558–3568, 2021.

Guoguo Chen, Shuzhou Chai, Guanbo Wang, Jiayu Du, Wei-Qiang Zhang, Chao Weng, Dan Su, Daniel Povey, Jan Trmal, Junbo Zhang, Mingjie Jin, Sanjeev Khudanpur, Shinji Watanabe, Shuaijiang Zhao, Wei Zou, Xiangang Li, Xuchen Yao, Yongqing Wang, Yujun Wang, Zhao You, and Zhiyong Yan. Gigaspeech: An evolving, multi-domain asr corpus with 10,000 hours of transcribed audio. In Proc. Interspeech 2021, 2021.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024a.

Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, Li Yuan, Yu Qiao, Dahua Lin, Feng Zhao, and Jiaqi Wang. Sharegpt4video: Improving video understanding and generation with better captions. arXiv preprint arXiv:2406.04325, 2024b.

Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. CoRR, abs/2501.17811, 2025a.

Xinyu Chen, Yunxin Li, Haoyuan Shi, Baotian Hu, Wenhan Luo, Yaowei Wang, and Min Zhang. Videovistaculturallingo: 360° horizons-bridging cultures, languages, and domains in video comprehension. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 27102–27128, 2025b.

Zhihong Chen, Xuehai Bai, Yang Shi, Chaoyou Fu, Huanyu Zhang, Haotian Wang, Xiaoyan Sun, Zhang Zhang, Liang Wang, Yuanxing Zhang, et al. Opengpt-4o-image: A comprehensive dataset for advanced image generation and editing. arXiv preprint arXiv:2509.24900, 2025c.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit S. Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, Luke Marris, Sam Petulla, Colin Gaffney, Asaf Aharoni, Nathan Lintz, Tiago Cardal Pais, Henrik Jacobsson, Idan Szpektor, Nan-Jiang Jiang, Krishna Haridasan, and et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. CoRR, abs/2507.06261, 2025. doi: 10.48550/ARXIV.2507.06261. URL https://doi.org/10.48550/arXiv.2507.06261.

Can Cui, Yunsheng Ma, Xu Cao, Wenqian Ye, Yang Zhou, Kaizhao Liang, Jintai Chen, Juanwu Lu, Zichong Yang, Kuei-Da Liao, Tianren Gao, Erlong Li, Kun Tang, Zhipeng Cao, Tong Zhou, Ao Liu, Xinrui Yan, Shuqi Mei, Jianguo Cao, Ziran Wang, and Chao Zheng. A survey on multimodal large language models for autonomous driving. In WACV (Workshops), pp. 958–979. IEEE, 2024.

Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, Jiasen Lu, Taira Anderson, Erin Bransom, Kiana Ehsani, Huong Ngo, Yen-Sung Chen, and et al. Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pp. 91–104. Computer Vision Foundation / IEEE, 2025. doi: 10.1109/CVPR52734.2025.00018. URL https:

//openaccess.thecvf.com/content/CVPR2025/html/Deitke_Molmo_and_PixMo_ Open_Weights_and_Open_Data_for_State-of-the-Art_CVPR_2025_paper.html.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

Zihao Deng, Yinghao Ma, Yudong Liu, Rongchen Guo, Ge Zhang, Wenhu Chen, Wenhao Huang, and Emmanouil Benetos. Musilingo: Bridging music and text with pre-trained language models for music captioning and query response. In Kevin Duh, Helena Gómez-Adorno, and Steven Bethard (eds.), Findings of the Association for Computational Linguistics: NAACL 2024, Mexico City, Mexico, June 16-21, 2024, pp. 3643–3655. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.FINDINGS-NAACL. 231. URL https://doi.org/10.18653/v1/2024.findings-naacl.231.

Kaustubh Deshpande, Ved Sirdeshmukh, Johannes Baptist Mols, Lifeng Jin, Ed-Yeremai Hernandez-Cardona, Dean Lee, Jeremy Kritz, Willow E. Primack, Summer Yue, and Chen Xing. Multichallenge: A realistic multiturn conversation evaluation benchmark challenging to frontier llms. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Findings of the Association for Computational Linguistics, ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pp. 18632–18702. Association for Computational Linguistics, 2025. URL https://aclanthology.org/2025.findings-acl.

958/.

Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pp. 3029–3051. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.EMNLP-MAIN.183. URL https://doi.

org/10.18653/v1/2023.emnlp-main.183.

Seungheon Doh, Keunwoo Choi, Jongpil Lee, and Juhan Nam. Lp-musiccaps: Llm-based pseudo music captioning. In Augusto Sarti, Fabio Antonacci, Mark Sandler, Paolo Bestagini, Simon Dixon, Beici Liang, Gaël Richard, and Johan Pauwels (eds.), Proceedings of the 24th International Society for Music Information Retrieval Conference, ISMIR 2023, Milan, Italy, November 5-9, 2023, pp. 409–416, 2023. doi: 10.5281/ZENODO.10265311. URL https://doi.org/10.5281/zenodo.10265311.

Konstantinos Drossos, Samuel Lipping, and Tuomas Virtanen. Clotho: an audio captioning dataset. In 2020 IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2020, Barcelona, Spain, May 4-8, 2020, pp. 736–740. IEEE, 2020. doi: 10.1109/ICASSP40776.2020.9052990. URL https://doi.org/10.1109/ICASSP40776.2020.9052990.

Ronen Eldan and Yuanzhi Li. Tinystories: How small can language models be and still speak coherent english? CoRR, abs/2305.07759, 2023. doi: 10.48550/ARXIV.2305.07759. URL https://doi.org/ 10.48550/arXiv.2305.07759.

Bernard Ghanem Fabian Caba Heilbron, Victor Escorcia and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 961–970, 2015.

Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025. URL https://github. com/huggingface/open-r1.

Miquel Farré, Andi Marafioti, Lewis Tunstall, Leandro Von Werra, and Thomas Wolf. Finevideo. https: //huggingface.co/datasets/HuggingFaceFV/finevideo, 2024.

Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 24108–24118, 2025.

Xueyang Fu, Jiabin Huang, Delu Zeng, Yue Huang, Xinghao Ding, and John Paisley. Removing rain from single images via a deep detail network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3855–3863, 2017.

Daniel Galvez, Greg Diamos, Juan Torres, Keith Achorn, Juan Felipe Cerón, Anjali Gopi, David Kanter, Max Lam, Mark Mazumder, and Vijay Janapa Reddi. The people’s speech: A largescale diverse english speech recognition dataset for commercial usage. In Joaquin Vanschoren and Sai-Kit Yeung (eds.), Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual, 2021. URL https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/ 202cb962ac59075b964b07152d234b70-Abstract-round1.html.

Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. Tall: Temporal activity localization via language query. In Proceedings of the IEEE international conference on computer vision, pp. 5267–5275, 2017.

Ridouane Ghermi, Xi Wang, Vicky Kalogeiton, and Ivan Laptev. Long story short: Story-level video understanding from 20k short films. arXiv preprint arXiv:2406.10221, 2025.

Jack Hong, Shilin Yan, Jiayin Cai, Xiaolong Jiang, Yao Hu, and Weidi Xie. Worldsense: Evaluating real-world omnimodal understanding for multimodal llms. arXiv preprint arXiv:2502.04326, 2025.

Xinyi Hou, Yanjie Zhao, Shenao Wang, and Haoyu Wang. Model context protocol (MCP): landscape, security threats, and future research directions. CoRR, abs/2503.23278, 2025.

Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. Videommmu: Evaluating knowledge acquisition from multi-discipline professional videos. arXiv preprint arXiv:2501.13826, 2025.

Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.

Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6700–6709, 2019.

Mude Hui, Siwei Yang, Bingchen Zhao, Yichun Shi, Heng Wang, Peng Wang, Yuyin Zhou, and Cihang Xie. Hq-edit: A high-quality dataset for instruction-based image editing. arXiv preprint arXiv:2404.09990, 2024.

Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, Aleksander Madry, Alex Baker-Whitcomb, Alex Beutel, Alex Borzunov, Alex Carney, Alex Chow, Alex Kirillov, Alex Nichol, Alex Paino, Alex Renzin, and et al. Gpt-4o system card. CoRR, abs/2410.21276, 2024. doi: 10.48550/ARXIV.2410.21276. URL https://doi.org/10.48550/arXiv.2410.21276.

Jesin James, Li Tian, and Catherine Inez Watson. An open source emotional speech corpus for human robot interaction applications. In B. Yegnanarayana (ed.), 19th Annual Conference of the International Speech Communication Association, Interspeech 2018, Hyderabad, India, September 2-6, 2018, pp. 2768– 2772. ISCA, 2018. doi: 10.21437/INTERSPEECH.2018-1349. URL https://doi.org/10.21437/ Interspeech.2018-1349.

Shengpeng Ji, Ziyue Jiang, Wen Wang, Yifu Chen, Minghui Fang, Jialong Zuo, Qian Yang, Xize Cheng, Zehan Wang, Ruiqi Li, et al. Wavtokenizer: an efficient acoustic discrete codec tokenizer for audio language modeling. arXiv preprint arXiv:2408.16532, 2024a.

Shengpeng Ji, Jialong Zuo, Minghui Fang, Siqi Zheng, Qian Chen, Wen Wang, Ziyue Jiang, Hai Huang, Xize Cheng, Rongjie Huang, and Zhou Zhao. Controlspeech: Towards simultaneous zero-shot speaker cloning and zero-shot language style control with decoupled codec. CoRR, abs/2406.01205, 2024b. doi: 10.48550/ARXIV.2406.01205. URL https://doi.org/10.48550/arXiv.2406.01205.

Yunjie Ji, Yong Deng, Yan Gong, Yiping Peng, Qiang Niu, Baochang Ma, and Xiangang Li. Belle: Be everyone’s large language model engine, 2023.

Baoxiong Jia, Ting Lei, Song-Chun Zhu, and Siyuan Huang. Egotaskqa: Understanding human tasks in egocentric videos. In The 36th Conference on Neural Information Processing Systems (NeurIPS 2022) Track on Datasets and Benchmarks, 2022.

Zeyu Jin, Jia Jia, Qixin Wang, Kehan Li, Shuoyi Zhou, Songtao Zhou, Xiaoyu Qin, and Zhiyong Wu. Speechcraft: A fine-grained expressive speech dataset with natural language description. In Jianfei Cai, Mohan S. Kankanhalli, Balakrishnan Prabhakaran, Susanne Boll, Ramanathan Subramanian, Liang Zheng, Vivek K. Singh, Pablo César, Lexing Xie, and Dong Xu (eds.), Proceedings of the 32nd ACM International Conference on Multimedia, MM 2024, Melbourne, VIC, Australia, 28 October 2024 - 1 November 2024, pp. 1255–1264. ACM, 2024. doi: 10.1145/3664647.3681674. URL https://doi.org/10.1145/ 3664647.3681674.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pp. 235–251. Springer, 2016.

Chris Dongjoo Kim, Byeongchang Kim, Hyunmin Lee, and Gunhee Kim. Audiocaps: Generating captions for audios in the wild. In Jill Burstein, Christy Doran, and Thamar Solorio (eds.), Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pp. 119–132. Association for Computational Linguistics, 2019. doi: 10.18653/V1/N19-1011. URL https://doi.org/10.18653/v1/n19-1011.

Maksim Kuprashevich, Grigorii Alekseenko, Irina Tolstykh, Georgii Fedorov, Bulat Suleimanov, Vladimir Dokholyan, and Aleksandr Gordeev. Nohumansrequired: Autonomous high-quality image editing triplet mining. arXiv preprint arXiv:2507.14119, 2025.

Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard H. Hovy. RACE: large-scale reading comprehension dataset from examinations. In Martha Palmer, Rebecca Hwa, and Sebastian Riedel (eds.), Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, EMNLP 2017, Copenhagen, Denmark, September 9-11, 2017, pp. 785–794. Association for Computational Linguistics, 2017. doi: 10.18653/V1/D17-1082. URL https://doi.org/10.18653/v1/d17-1082.

Dejoli Landry, Qianhua He, Haikang Yan, and Yanxiong Li. Asvp-esd: A dataset and its benchmark for emotion recognition using both speech and non-speech utterances. Global Scientific Journals, 8:1793–1798, 2020.

Hugo Laurençon, Andrés Marafioti, Victor Sanh, and Léo Tronchon. Building and better understanding vision-language models: insights and future directions., 2024.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326,

- 2024a.

Bohao Li, Yuying Ge, Yi Chen, Yixiao Ge, Ruimao Zhang, and Ying Shan. Seed-bench-2-plus: Benchmarking multimodal large language models with text-rich visual comprehension. arXiv preprint arXiv:2404.16790,

- 2024b.

Boyi Li, Wenqi Ren, Dengpan Fu, Dacheng Tao, Dan Feng, Wenjun Zeng, and Zhangyang Wang. Benchmarking single-image dehazing and beyond. IEEE transactions on image processing, 28(1):492–505, 2018.

Caorui Li, Yu Chen, Yiyan Ji, Jin Xu, Zhenyu Cui, Shihao Li, Yuanxing Zhang, Jiafu Tang, Zhenghao Song, Dingling Zhang, et al. Omnivideobench: Towards audio-visual understanding evaluation for omni mllms. arXiv preprint arXiv:2510.10689, 2025a.

Guangyao Li, Yake Wei, Yapeng Tian, Chenliang Xu, Ji-Rong Wen, and Di Hu. Learning to answer questions in dynamic audio-visual scenarios. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pp. 19086–19096. IEEE, 2022. doi: 10.1109/ CVPR52688.2022.01852. URL https://doi.org/10.1109/CVPR52688.2022.01852.

Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In CVPR, 2024c.

Ming Li, Taojiannan Yang, Huafeng Kuang, Jie Wu, Zhaoning Wang, Xuefeng Xiao, and Chen Chen. Controlnet++: Improving conditional controls with efficient consistency feedback: Project page: liming-ai. github. io/controlnet_plus_plus. In European Conference on Computer Vision, pp. 129–147. Springer, 2024d.

Ruoteng Li, Loong-Fah Cheong, and Robby T Tan. Heavy rain image restoration: Integrating physics model and conditional adversarial learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1633–1642, 2019.

Yadong Li, Jun Liu, Tao Zhang, Song Chen, Tianpeng Li, Zehuan Li, Lijun Liu, Lingfeng Ming, Guosheng Dong, Da Pan, Chong Li, Yuanbo Fang, Dongdong Kuang, Mingrui Wang, Chenglin Zhu, Youwei Zhang, Hongyu Guo, Fengyu Zhang, Yuran Wang, Bowen Ding, Wei Song, Xu Li, and et al. Baichuan-omni-1.5 technical report. CoRR, abs/2501.15368, 2025b.

Yizhi Li, Ge Zhang, Yinghao Ma, Ruibin Yuan, Kang Zhu, Hangyu Guo, Yiming Liang, Jiaheng Liu, Zekun Wang, Jian Yang, et al. Omnibench: Towards the future of universal omni-language models. arXiv preprint arXiv:2409.15272, 2024e.

Yunxin Li, Xinyu Chen, Baotian Hu, Longyue Wang, Haoyuan Shi, and Min Zhang. Videovista: A versatile benchmark for video understanding and reasoning, 2024f.

Yunxin Li, Xinyu Chen, Zitao Li, Zhenyu Liu, Longyue Wang, Wenhan Luo, Baotian Hu, and Min Zhang. Veripo: Cultivating long reasoning in video-llms via verifier-gudied iterative policy optimization. arXiv preprint arXiv:2505.19000, 2025c.

Yunxin Li, Shenyuan Jiang, Baotian Hu, Longyue Wang, Wanqi Zhong, Wenhan Luo, Lin Ma, and Min Zhang. Uni-moe: Scaling unified multimodal llms with mixture of experts. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(5):3424–3439, 2025d. doi: 10.1109/TPAMI.2025.3532688.

Yunxin Li, Zhenyu Liu, Zitao Li, Xuanyu Zhang, Zhenran Xu, Xinyu Chen, Haoyuan Shi, Shenyuan Jiang, Xintong Wang, Jifang Wang, Shouzheng Huang, Xinping Zhao, Borui Jiang, Lanqing Hong, Longyue Wang, Zhuotao Tian, Baoxing Huai, Wenhan Luo, Weihua Luo, Zheng Zhang, Baotian Hu, and Min Zhang. Perception, reason, think, and plan: A survey on large multimodal reasoning models. CoRR, abs/2505.04921, 2025e.

Wing Lian, Bleys Goodson, Eugene Pentland, Austin Cook, Chanvichet Vong, and "Teknium". Openorca: An open dataset of gpt augmented flan reasoning traces. https://https://huggingface.co/ datasets/Open-Orca/OpenOrca, 2023.

Junming Lin, Zheng Fang, Chi Chen, Zihao Wan, Fuwen Luo, Peng Li, Yang Liu, and Maosong Sun. Streamingbench: Assessing the gap for mllms to achieve streaming video understanding. arXiv preprint arXiv:2411.03628, 2024a.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pp. 740–755. Springer, 2014.

Weifeng Lin, Xinyu Wei, Renrui Zhang, Le Zhuo, Shitian Zhao, Siyuan Huang, Huan Teng, Junlin Xie, Yu Qiao, Peng Gao, et al. Pixwizard: Versatile image-to-image visual assistant with open-language instructions. arXiv preprint arXiv:2409.15278, 2024b.

Samuel Lipping, Parthasaarathy Sudarsanam, Konstantinos Drossos, and Tuomas Virtanen. Clotho-aqa: A crowdsourced dataset for audio question answering. In 30th European Signal Processing Conference, EUSIPCO 2022, Belgrade, Serbia, August 29 - Sept. 2, 2022, pp. 1140–1144. IEEE, 2022. URL https: //ieeexplore.ieee.org/document/9909680.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Liyuan Liu, Young Jin Kim, Shuohang Wang, Chen Liang, Yelong Shen, Hao Cheng, Xiaodong Liu, Masahiro Tanaka, Xiaoxia Wu, Wenxiang Hu, Vishrav Chaudhary, Zeqi Lin, Chenruidong Zhang, Jilong Xue, Hany Awadalla, Jianfeng Gao, and Weizhu Chen. Grin: Gradient-informed moe, 2024a. URL https: //arxiv.org/abs/2409.12136.

Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.

Yang Liu, Zhen Zhu, and Xiang Bai. Wdnet: Watermark-decomposition network for visible watermark removal. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 3685–3693, 2021.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pp. 216–233. Springer, 2024b.

Yun-Fu Liu, Da-Wei Jaw, Shih-Chia Huang, and Jenq-Neng Hwang. Desnownet: Context-aware deep network for snow removal. IEEE Transactions on Image Processing, 27(6):3064–3073, 2018.

Steven R Livingstone and Frank A Russo. The ryerson audio-visual database of emotional speech and song (ravdess): A dynamic, multimodal set of facial and vocal expressions in north american english. PloS one, 13(5):e0196391, 2018.

Alejo Lopez-Avila and Jinhua Du. A survey on large language models in multimodal recommender systems. CoRR, abs/2505.09777, 2025.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.

Ruipu Luo, Ziwang Zhao, Min Yang, Junwei Dong, Minghui Qiu, Pengcheng Lu, Tao Wang, and Zhongyu Wei. Valley: Video assistant with large language model enhanced ability, 2023.

Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Videogpt+: Integrating image and video encoders for enhanced video understanding. arxiv, 2024. URL https://arxiv.org/abs/ 2406.09418.

Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. In NeurIPS, 2023.

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022.

Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 2200–2209, 2021.

Xinhao Mei, Chutong Meng, Haohe Liu, Qiuqiang Kong, Tom Ko, Chengqi Zhao, Mark D. Plumbley, Yuexian Zou, and Wenwu Wang. Wavcaps: A chatgpt-assisted weakly-labelled audio captioning dataset for audiolanguage multimodal research. IEEE ACM Trans. Audio Speech Lang. Process., 32:3339–3354, 2024. doi: 10.1109/TASLP.2024.3419446. URL https://doi.org/10.1109/TASLP.2024.3419446.

Jan Melechovský, Zixun Guo, Deepanway Ghosal, Navonil Majumder, Dorien Herremans, and Soujanya Poria. Mustango: Toward controllable text-to-music generation. In Kevin Duh, Helena Gómez-Adorno, and Steven Bethard (eds.), Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), NAACL 2024, Mexico City, Mexico, June 16-21, 2024, pp. 8293–8316. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.NAACL-LONG.459. URL https://doi.org/10.18653/ v1/2024.naacl-long.459.

Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, et al. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365, 2025.

Eliya Nachmani, Alon Levkovitch, Roy Hirsch, Julian Salazar, Chulayuth Asawaroengchai, Soroosh Mariooryad, Ehud Rivlin, R. J. Skerry-Ryan, and Michelle Tadmor Ramanovich. Spoken question answering and speech continuation using spectrogram-powered LLM. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=izrOLJov5y.

Arsha Nagrani, Mingda Zhang, Ramin Mehran, Rachel Hornung, Nitesh Bharadwaj Gundavarapu, Nilpa Jha, Austin Myers, Xingyi Zhou, Boqing Gong, Cordelia Schmid, Mikhail Sirotenko, Yukun Zhu, and Tobias Weyand. Neptune: The long orbit to benchmarking long video understanding. arXiv preprint arXiv:2412.09582, 2024.

Seungjun Nah, Tae Hyun Kim, and Kyoung Mu Lee. Deep multi-scale convolutional neural network for dynamic scene deblurring. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3883–3891, 2017.

Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, et al. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.

Kun Ouyang, Yuanxin Liu, Haoning Wu, Yi Liu, Hao Zhou, Jie Zhou, Fandong Meng, and Xu Sun. Spacer: Reinforcing mllms in video spatial reasoning. arXiv preprint arXiv:2504.01805, 2025.

Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: An ASR corpus based on public domain audio books. In 2015 IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2015, South Brisbane, Queensland, Australia, April 19-24, 2015, pp. 5206–5210. IEEE, 2015. doi: 10.1109/ICASSP.2015.7178964. URL https://doi.org/10.1109/ICASSP.

2015.7178964. Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023.

Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In Proceedings of the IEEE international conference on computer vision, pp. 2641–2649, 2015.

Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea. MELD: A multimodal multi-party dataset for emotion recognition in conversations. In Anna Korhonen, David R. Traum, and Lluís Màrquez (eds.), Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pp. 527–536. Association for Computational Linguistics, 2019. doi: 10.18653/V1/P19-1050. URL https://doi.org/10.18653/v1/p19-1050.

Vineel Pratap, Qiantong Xu, Anuroop Sriram, Gabriel Synnaeve, and Ronan Collobert. Mls: A large-scale multilingual dataset for speech research. arXiv preprint arXiv:2012.03411, 2020.

Rui Qian, Robby T Tan, Wenhan Yang, Jiajun Su, and Jiaying Liu. Attentive generative adversarial network for raindrop removal from a single image. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2482–2491, 2018.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, Wanjun Zhong, Kuanye Li, Jiale Yang, Yu Miao, Woyu Lin, Longxiang Liu, Xu Jiang, Qianli Ma, Jingyu Li, Xiaojun Xiao, Kai Cai, Chuang Li, Yaowei Zheng, Chaolin Jin, Chen Li, Xiao Zhou, Minchao Wang, Haoli Chen, Zhaojian Li, Haihua Yang, Haifeng Liu, Feng Lin, Tao Peng, Xin Liu, and Guang Shi. UI-TARS: pioneering automated GUI interaction with native agents. CoRR, abs/2501.12326, 2025.

Ruijie Quan, Xin Yu, Yuanzhi Liang, and Yi Yang. Removing raindrops and rain streaks in one go. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9147–9156, 2021.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pp. 28492–28518. PMLR, 2023. URL https://proceedings.mlr.press/ v202/radford23a.html.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model, 2024. URL https: //arxiv.org/abs/2305.18290.

Ruchit Rawal, Khalid Saifullah, Ronen Basri, David Jacobs, Gowthami Somepalli, and Tom Goldstein. Cinepile: A long video question answering dataset and benchmark. arXiv preprint arXiv:2405.08813, 2024.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

Jaesung Rim, Haeyun Lee, Jucheol Won, and Sunghyun Cho. Real-world blur dataset for learning and benchmarking deblurring algorithms. In European conference on computer vision, pp. 184–201. Springer, 2020.

- S. Sakshi, Utkarsh Tyagi, Sonal Kumar, Ashish Seth, Ramaneswaran Selvakumar, Oriol Nieto, Ramani Duraiswami, Sreyan Ghosh, and Dinesh Manocha. MMAU: A massive multi-task audio understanding and reasoning benchmark. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/ forum?id=TeVAZXr3yv.

Yugen Sato and Tomohiro Takagi. Identifying multi-modal knowledge neurons in pretrained transformers via two-stage filtering. CoRR, abs/2503.22941, 2025. doi: 10.48550/ARXIV.2503.22941. URL https: //doi.org/10.48550/arXiv.2503.22941.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In NeurIPS, 2023.

Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. AOKVQA: A benchmark for visual question answering using world knowledge. In Shai Avidan, Gabriel J. Brostow, Moustapha Cissé, Giovanni Maria Farinella, and Tal Hassner (eds.), Computer Vision - ECCV 2022 - 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part VIII, volume 13668 of Lecture Notes in Computer Science, pp. 146–162. Springer, 2022. doi: 10.1007/978-3-031-20074-8\_9. URL https://doi.org/10.1007/978-3-031-20074-8_9.

Ziyao Shangguan, Chuhan Li, Yuxuan Ding, Yanan Zheng, Yilun Zhao, Tesca Fitzgerald, and Arman Cohan. Tomato: Assessing visual temporal reasoning capabilities in multimodal foundation models, 2024. URL https://arxiv.org/abs/2410.23266.

Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8871–8879, 2024.

Yao Shi, Hui Bu, Xin Xu, Shaoji Zhang, and Ming Li. AISHELL-3: A multi-speaker mandarin TTS corpus and the baselines. CoRR, abs/2010.11567, 2020. URL https://arxiv.org/abs/2010.11567.

Vasu Singla, Kaiyu Yue, Sukriti Paul, Reza Shirkavand, Mayuka Jayawardhana, Alireza Ganjdanesh, Heng Huang, Abhinav Bhatele, Gowthami Somepalli, and Tom Goldstein. From pixels to prose: A large dataset of dense image captions. arXiv preprint arXiv:2406.10328, 2024.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, Agnieszka Kluska, Aitor Lewkowycz, Akshat Agarwal, Alethea Power, Alex Ray, Alex Warstadt, Alexander W. Kocurek, and et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. CoRR, abs/2206.04615, 2022. doi: 10.48550/ARXIV.2206.04615. URL https://doi.org/10.48550/arXiv.2206.

04615.

Keqiang Sun, Junting Pan, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, et al. Journeydb: A benchmark for generative image understanding. Advances in neural information processing systems, 36:49659–49678, 2023.

Tianxiang Sun, Xiaotian Zhang, Zhengfu He, Peng Li, Qinyuan Cheng, Xiangyang Liu, Hang Yan, Yunfan Shao, Qiong Tang, Shiduo Zhang, Xingjian Zhao, Ke Chen, Yining Zheng, Zhejian Zhou, Ruixiao Li, Jun Zhan, Yunhua Zhou, Linyang Li, Xiaogui Yang, Lingling Wu, Zhangyue Yin, Xuanjing Huang, Yu-Gang Jiang, and Xipeng Qiu. MOSS: an open conversational large language model. Mach. Intell. Res., 21(5):888–905, 2024. doi: 10.1007/S11633-024-1502-8. URL https://doi.org/10.1007/ s11633-024-1502-8.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github. com/tatsu-lab/stanford_alpaca, 2023.

Teknium. Openhermes 2.5: An open dataset of synthetic data for generalist llm assistants, 2023. URL https://huggingface.co/datasets/teknium/OpenHermes-2.5.

Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri IYER, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, visioncentric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310– 87356, 2024a.

Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms, 2024b.

Alex Jinpeng Wang, Dongxing Mao, Jiawei Zhang, Weiming Han, Zhuobai Dong, Linjie Li, Yiqi Lin, Zhengyuan Yang, Libo Qin, Fuwei Zhang, et al. Textatlas5m: A large-scale dataset for dense text image generation. arXiv preprint arXiv:2502.07870, 2025a.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024a.

Longguang Wang, Yulan Guo, Yingqian Wang, Juncheng Li, Shuhang Gu, Radu Timofte, Ming Cheng, Haoyu Ma, Qiufang Ma, Xiaopeng Sun, et al. Ntire 2023 challenge on stereo image super-resolution: Methods and results. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1346–1372, 2023.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024b.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025b.

Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Jilan Xu, Zun Wang, et al. Internvideo2: Scaling video foundation models for multimodal video understanding. arXiv preprint arXiv:2403.15377, 2024c.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. arXiv preprint arXiv:2406.01574, 2024d.

Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, et al. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. Advances in Neural Information Processing Systems, 37:113569–113697, 2024e.

Lai Wei, Yuting Li, Kaipeng Zheng, Chen Wang, Yue Wang, Linghe Kong, Lichao Sun, and Weiran Huang. Advancing multimodal reasoning via reinforcement learning with cold start. arXiv preprint arXiv:2505.22334, 2025.

Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, and Zheng Liu. Omnigen2: Exploration to advanced multimodal generation. CoRR, abs/2506.18871, 2025.

Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. Advances in Neural Information Processing Systems, 37:28828– 28857, 2024.

Penghao Wu and Saining Xie. V?: Guided visual search as a core mechanism in multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13084–13094,

- 2024.

X. Real world qa benchmark. https://huggingface.co/datasets/xai-org/RealworldQA,

- 2025.

Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts, 2024. URL https://arxiv.org/abs/2407.04973.

Binzhu Xie, Sicheng Zhang, Zitang Zhou, Bo Li, Yuanhan Zhang, Jack Hessel, Jingkang Yang, and Ziwei Liu. Funqa: Towards surprising video comprehension. In European Conference on Computer Vision (ECCV), 2024a. URL https://www.ecva.net/papers/eccv_2024/papers_ECCV/ papers/00010.pdf.

Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. In ICLR. OpenReview.net, 2025.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. In NeurIPS, 2024b.

Zhifei Xie and Changqiao Wu. Mini-omni: Language models can hear, talk while thinking in streaming. CoRR, abs/2408.16725, 2024. doi: 10.48550/ARXIV.2408.16725. URL https://doi.org/10.48550/ arXiv.2408.16725.

Long Xing, Qidong Huang, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Jinsong Li, Shuangrui Ding, Weiming Zhang, Nenghai Yu, et al. Scalecap: Inference-time scalable image captioning via dual-modality debiasing. arXiv preprint arXiv:2506.19848, 2025.

Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and Junyang Lin. Qwen2.5-omni technical report. CoRR, abs/2503.20215, 2025. doi: 10.48550/ARXIV.2503.20215. URL https://doi.org/10.48550/ arXiv.2503.20215.

Dongjie Yang, Suyuan Huang, Chengqiang Lu, Xiaodong Han, Haoxin Zhang, Yan Gao, Yao Hu, and Hai Zhao. Vript: A video is worth thousands of words, 2024a.

Jihan Yang, Shusheng Yang, Anjali Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in Space: How Multimodal Large Language Models See, Remember and Recall Spaces. arXiv preprint arXiv:2412.14171, 2024b.

Pinci Yang, Xin Wang, Xuguang Duan, Hong Chen, Runze Hou, Cong Jin, and Wenwu Zhu. AVQA: A dataset for audio-visual question answering on videos. In João Magalhães, Alberto Del Bimbo, Shin’ichi Satoh, Nicu Sebe, Xavier Alameda-Pineda, Qin Jin, Vincent Oria, and Laura Toni (eds.), MM ’22: The 30th ACM International Conference on Multimedia, Lisboa, Portugal, October 10 - 14, 2022, pp. 3480–3491. ACM,

- 2022. doi: 10.1145/3503161.3548291. URL https://doi.org/10.1145/3503161.3548291.

Wenhan Yang, Robby T Tan, Jiashi Feng, Jiaying Liu, Zongming Guo, and Shuicheng Yan. Deep joint rain detection and removal from a single image. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1357–1366, 2017.

Wenhan Yang, Wenjing Wang, Haofeng Huang, Shiqi Wang, and Jiaying Liu. Sparse gradient regularized deep retinex network for robust low-light image enhancement. IEEE Transactions on Image Processing, 30: 2072–2086, 2021.

Linli Yao, Yicheng Li, Yuancheng Wei, Lei Li, Shuhuai Ren, Yuanxin Liu, Kun Ouyang, Lean Wang, Shicheng Li, Sida Li, Lingpeng Kong, Qi Liu, Yuanxing Zhang, and Xu Sun. Timechat-online: 80URL https://arxiv.org/abs/2504.17343.

Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. CoRR, abs/2306.13549, 2023.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Liping Yuan, Jiawei Wang, Haomiao Sun, Yuchen Zhang, and Yuan Lin. Tarsier2: Advancing large visionlanguage models from detailed video description to comprehensive video understanding, 2025. URL https://arxiv.org/abs/2501.07888.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9556–9567, 2024a.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024b.

Heiga Zen, Viet Dang, Rob Clark, Yu Zhang, Ron J. Weiss, Ye Jia, Zhifeng Chen, and Yonghui Wu. Libritts: A corpus derived from librispeech for text-to-speech. In Gernot Kubin and Zdravko Kacic (eds.), 20th Annual Conference of the International Speech Communication Association, Interspeech 2019, Graz, Austria, September 15-19, 2019, pp. 1526–1530. ISCA, 2019. doi: 10.21437/INTERSPEECH.2019-2441. URL https://doi.org/10.21437/Interspeech.2019-2441.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pretraining. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 11975–11986, 2023.

Howard Zhang, Yunhao Ba, Ethan Yang, Varan Mehra, Blake Gella, Akira Suzuki, Arnold Pfahnl, Chethan Chinder Chandrappa, Alex Wong, and Achuta Kadambi. Weatherstream: Light transport automation of single image deweathering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 13499–13509, 2023a.

Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems, 36:31428–31449,

- 2023b.

Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmms-eval: Reality check on the evaluation of large multimodal models, 2024a. URL https://arxiv.org/abs/2407.12772.

Shaolei Zhang, Shoutao Guo, Qingkai Fang, Yan Zhou, and Yang Feng. Stream-omni: Simultaneous multimodal interactions with large language-vision-speech model. CoRR, abs/2506.13642, 2025a. doi: 10.48550/ARXIV.2506.13642. URL https://doi.org/10.48550/arXiv.2506.13642.

Yi-Fan Zhang, Huanyu Zhang, Haochen Tian, Chaoyou Fu, Shuangqing Zhang, Junfei Wu, Feng Li, Kun Wang, Qingsong Wen, Zhang Zhang, et al. Mme-realworld: Could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans? arXiv preprint arXiv:2408.13257, 2024b.

Yiyuan Zhang, Kaixiong Gong, Kaipeng Zhang, Hongsheng Li, Yu Qiao, Wanli Ouyang, and Xiangyu Yue. Meta-transformer: A unified framework for multimodal learning. CoRR, abs/2307.10802, 2023c.

Yu Zhang, Yunqi Li, Yifan Yang, Rui Wang, Yuqing Yang, Dai Qi, Jianmin Bao, Dongdong Chen, Chong Luo, and Lili Qiu. Reasongen-r1: Cot for autoregressive image generation models through sft and rl. arXiv preprint arXiv:2505.24875, 2025b.

Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data, 2024c. URL https://arxiv.org/abs/2410.02713.

Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Llava-video: Video instruction tuning with synthetic data. Trans. Mach. Learn. Res., 2025, 2025c. URL https: //openreview.net/forum?id=EElFGvt39K.

Haozhe Zhao, Xiaojian Shawn Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. Advances in Neural Information Processing Systems, 37:3058–3093, 2024.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization, 2025. URL https://arxiv.org/abs/2507.18071.

Guanghao Zhou, Panjia Qiu, Cen Chen, Jie Wang, Zheming Yang, Jian Xu, and Minghui Qiu. Reinforced MLLM: A survey on rl-based reasoning in multimodal large language models. CoRR, abs/2504.21277, 2025.

Kun Zhou, Berrak Sisman, Rui Liu, and Haizhou Li. Emotional voice conversion: Theory, databases and ESD. Speech Commun., 137:1–18, 2022. doi: 10.1016/J.SPECOM.2021.11.006. URL https: //doi.org/10.1016/j.specom.2021.11.006.

Yurui Zhu, Tianyu Wang, Xueyang Fu, Xuanyu Yang, Xin Guo, Jifeng Dai, Yu Qiao, and Xiaowei Hu. Learning weather-general and weather-specific features for image restoration under multiple adverse weather conditions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 21747–21758, 2023.

A. Appendix

Stage Dataset

PixelProse-RedCaps (Singla et al., 2024) PixelProse-CommonPool (Singla et al., 2024) GRIT (Peng et al., 2023) CC3M (Changpinyo et al., 2021) LLaVA-Pretrain (Liu et al., 2023)

Pretrain

Cambrian-10M (Tong et al., 2024a) LLaVA-OneVision (Li et al., 2024a) Docmatix (Laurençon et al., 2024) Pixmo-Docx (Deitke et al., 2025) Pixmo-Points (Deitke et al., 2025) Pixmo-Point-Explanations (Deitke et al., 2025) Pixmo-Ask-Model-Anything (Deitke et al., 2025) Latex-OCR Latex-Formulas Arxiv-OCR-v0.2 MMK12 (Meng et al., 2025) V* (Wu & Xie, 2024) Vision-R1-cold (Huang et al., 2025) Multimodal-Cold-Start (Wei et al., 2025)

SFT & Annealing

Table 15: The list of image data used during our training.

Stage Dataset

Valley-Pretrain (Luo et al., 2023) ShareGPT4Video (Chen et al., 2024b) VideoVista-Event (Li et al., 2024f)

Pretrain

VideoChat2-IT* (Li et al., 2024c) LLaVA-Video-178K (Zhang et al., 2024c) VideoVista-Train (Li et al., 2024f) VideoGPT-Plus (Maaz et al., 2024) fineVideo (Farré et al., 2024) TimeChat-Online (Yao et al., 2025) Charades-STA (Gao et al., 2017) CinePile (Rawal et al., 2024) SF20K (Ghermi et al., 2025) Neptune (Nagrani et al., 2024) EgoTaskQA (Jia et al., 2022) FunQA (Xie et al., 2024a) Vript (Yang et al., 2024a) Tarsier2-Recap (Yuan et al., 2025) InternVideo2-Vid-Text* (Wang et al., 2024c) SR-91K* (Ouyang et al., 2025) VideoVista2-CoT (Li et al., 2024f)

SFT & Annealing

Table 16: The list of video data used during our training. * indicates that only a subset of the dataset was employed.

### A Appendix

#### A.1 Training Data List

All open-source training data are presented in Tables 15-18.

Multilingual-LibriSpeech-English (Pratap et al., 2020) GigaSpeech-L (Chen et al., 2021) CommonVoice-English (Ardila et al., 2020) WavCaps (Mei et al., 2024) ClothoV1 (Drossos et al., 2020) MELD (Poria et al., 2019) MusicBench (Melechovský et al., 2024) LP-MusicCaps (Doh et al., 2023)

Pretrain

CommonVoice-English (Ardila et al., 2020) WavCaps (Mei et al., 2024) ClothoV1 (Drossos et al., 2020) MELD (Poria et al., 2019) MusicBench (Melechovský et al., 2024) LP-MusicCaps (Doh et al., 2023) ClothoAQA (Lipping et al., 2022) AudioCaps (Kim et al., 2019) MELD (Poria et al., 2019) ASVP-ESD (Landry et al., 2020) CREMA-D (Cao et al., 2014) EMOV (Adigwe et al., 2018) ESD (Zhou et al., 2022) JL-Coprus (James et al., 2018) RAVDESS (Livingstone & Russo, 2018) MusicInstruct (Deng et al., 2024) LibriSpeech-Long (Panayotov et al., 2015) RACE-Audio (Lai et al., 2017) Aishell1 (Bu et al., 2017) Aishell3 (Shi et al., 2020) Mozilla-CommonVoice17 (Ardila et al., 2020) Peoples-Speech (Galvez et al., 2021) GigaSpeech-M (Chen et al., 2021) LibriSpeech (Panayotov et al., 2015) AVQA (Yang et al., 2022) Music-AVQA (Li et al., 2022) ClothoV2 (Drossos et al., 2020) Ultra-Chat-Audio (Ding et al., 2023) Belle-Audio (Ji et al., 2023) Openhermes-Audio (Teknium, 2023) moss-Audio (Sun et al., 2024) alpaca-Audio (Taori et al., 2023) Llava-150k-Audio (Liu et al., 2023) Pixmo-Audio (Deitke et al., 2025) Llava-video-180k-Audio (Zhang et al., 2025c) TinyStories-en-Audio (Eldan & Li, 2023) TinyStories-zh-Audio (Eldan & Li, 2023) VCCM (Ji et al., 2024b) SpeechCraft (Jin et al., 2024) Stream-Omni-Instruct-Audio (Zhang et al., 2025a) Voice-Assistant-Audio (Xie & Wu, 2024)

SFT & Annealing

Table 17: The list of audio data used during our training.

#### A.2 Evaluation Data List

Image Understanding We evaluate our models across three categories of capabilities of image understanding: General Visual Understanding, STEM Image Understanding (focusing mainly on Science and Mathematics), and OCR & Document Understanding.

OpenOrca-GPT4 (Lian et al., 2023) OpenOrca-Chinese-GPT4 (Lian et al., 2023) DAPO-Math (Yu et al., 2025) Nemotron-Post-Training-Dataset-v1 (Bercovich et al., 2025) Mixture-of-Thoughts (Face, 2025)

SFT & Annealing

Table 18: The list of video data used during our training.

Task Dataset

Flickr30k (Plummer et al., 2015) Coco* (Lin et al., 2014) LLaVA-Pretrain (Liu et al., 2023) JourneyDB* (Sun et al., 2023) ReasonGen (Zhang et al., 2025b) OpenGPT-4o-Image (Chen et al., 2025c) ScaleCap (Xing et al., 2025) TextAtlas5M* (Wang et al., 2025a)

Image Generation (4.81M)

InstructPix2pix* (Brooks et al., 2023) MagicBrush (Zhang et al., 2023b) HQ-Edit (Hui et al., 2024) UltraEdit* (Zhao et al., 2024) OpenGPT-4o-Image (Chen et al., 2025c) NHR-Edit (Kuprashevich et al., 2025) Controllable Generation (0.50M)

Image Edition (5.68M)

MultiGen-20M* (Li et al., 2024d)

DenseHaze (Ancuti et al., 2019) NH-HAZE (Ancuti et al., 2020) Reside-6K (Li et al., 2018) Weather-Stream (Zhang et al., 2023a) Outdoor-Rain (Li et al., 2019) Rain1400 (Fu et al., 2017) RainDS (Quan et al., 2021) RainDrop (Qian et al., 2018) RealSnow (Zhu et al., 2023) Snow100K (Liu et al., 2018) LOL-v2 (Yang et al., 2021) CLWD (Liu et al., 2021) DIV2K (Agustsson & Timofte, 2017) Flickr2K (Wang et al., 2023) GoPro (Nah et al., 2017) RealBlur (Rim et al., 2020) RealSR (Cai et al., 2019) DPDD (Abuolaim & Brown, 2020) SIDD (Abdelhamed et al., 2018)

Image Restoration (0.36M)

Table 19: The list of visual generation data used during our training. An asterisk (*) indicates that only a subset of the dataset was employed.

General Visual Understanding. MMBench (EN/CN) (Liu et al., 2024b), MMStar (Chen et al., 2024a), RealWorldQA (X, 2025), GQA (Hudson & Manning, 2019), MME-RealWorld (Zhang et al., 2024b), and CV-Bench (Tong et al., 2024b).

STEM Image Reasoning. AI2D (Kembhavi et al., 2016), MMMU (Yue et al., 2024a) and MMMU-Pro (Yue et al., 2024b) for science reasoning. MathVista (Lu et al., 2023), MathVision (Wang et al., 2024a) and LogicVista (Xiao et al., 2024) for mathematics reasoning.

Mixture-of-Thoughts (Face, 2025) Vision-R1-cold (Huang et al., 2025) Multimodal-Cold-Start (Wei et al., 2025) VideoVista-2-LongCoT (Chen et al., 2025b) GSPO MMPR-Tiny* (Wang et al., 2025b)

Cold Start

Table 20: The list of video data used during our training. * indicates that only a subset of the dataset was employed.

OCR & Document Understanding. DocVQA (Mathew et al., 2021), ChartQA (Masry et al., 2022), CharXiv (DQ/RQ) (Wang et al., 2024e), and SEED-Bench-2-Plus (Li et al., 2024b).

Video Understanding We evaluate our models across four categories of capabilities of video understanding: Short Video Understanding, Long Video Understanding, Video Reasoning and Video Temporal Localization.

Short Video Understanding. MVBench (Li et al., 2024c). Long Video Understanding. Video-MME (Fu et al., 2025), LongVideoBench (Wu et al., 2024) and EgoSchema (Mangalam et al., 2023). Video Reasoning. VideoMMMU (Hu et al., 2025) for video knowledge reasoning, VSI-Bench (Yang et al.,

- 2024b) for video spatial reasoning and TOMATO (Shangguan et al., 2024) for video temporal reasoning. Video Temporal Localization. Charades-STA (Gao et al., 2017).

Language Capability We evaluate our models on two benchmarks focused on complex reasoning and knowledge tasks, including the three versions of GPQA (Rein et al., 2024) and the MMLU-Pro (Wang et al., 2024d) dataset.

Omni Understanding We evaluate our models on two categories of omni benchmarks that require simultaneous understanding of visual and audio information: Video&Audio and Image&Audio.

Video&Audio. WorldSense (Hong et al., 2025), OmniVideoBench (Li et al., 2025a) and StreamingBench (Lin et al., 2024a).

Image&Audio. OmniBench (Li et al., 2024e)

Visual Generation We evaluate our models across four categories of visual generation tasks: Image

Generation, Image Edition, Controllable Generation and Image Restoration. Image Generation. Wise (Niu et al., 2025) and Coco30K (Lin et al., 2014). Image Edition. GEdit-Bench-EN (Liu et al., 2025), Emu Edit Test (Sheynin et al., 2024) and Mag-

icBrush (Zhang et al., 2023b). Controllable Generation. MultiGen (Li et al., 2024d). Low-Level Image Restoration. Rain100L (Yang et al., 2017) and SIDD (Abdelhamed et al., 2018).

Audio Understanding and Speech Generation We evaluate our models across four categories of capabilities of audio understanding and speech generation: Audio understanding with text reply, text to speech, voice conversation(speech to speech or speech to text), and voice conversation with visual Information(speech and image/video to speech or speech and image/video to text.

Audio X → Text RACE-audio(middle/high) (Lai et al., 2017), EHSL (Li et al., 2025d), MELD (Poria et al., 2019), MMAU (Sakshi et al., 2025), ClothoAQA (Lipping et al., 2022), ClothoV1 (Drossos et al., 2020), ClothoV2 (Drossos et al., 2020), AudioCaps (Kim et al., 2019), and MusicCaps (Doh et al., 2023).

Text → Speech LibriTTS (Zen et al., 2019), SEED (Anastassiou et al., 2024), and TinyStories (Eldan & Li,

- 2023).

- A.3 Thinking Prompt

Speech → Speech/Text LlamaQA (Nachmani et al., 2024), WebQA (Berant et al., 2013), BigBench Audio (Srivastava et al., 2022), and MultiChallenge Audio (Deshpande et al., 2025).

Vision + Speech → Speech/Text A-OK-VQA (Schwenk et al., 2022), VQAv2 (Antol et al., 2015), and ActivityNet (Fabian Caba Heilbron & Niebles, 2015).

#### A.3 Thinking Prompt

Prompt For Visual Understanding

SYSTEM: You are Uni-MoE-v2, a helpful multi-modal model. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analyzing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. Now, try to solve the following question through the above guidelines.

USER: Question: {question} Prompt For Visual Generation

SYSTEM: You should first think step by step about how to construct the image, including background, objects, colors, lighting, and style. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think>1. Position a man seated indoors, capturing him from the upper chest to just above the chin, focusing on ... </think> <answer> Here is the image: [TASK0][TASK1][TASK2][IMG0][IMG1]......[IMG31] </answer>, which means your output should start with <think> and end with </answer>.

USER: Image generation: {prompt} Table 21: The prompt setting for thinking models.

#### A.4 Gradient Estimation Formalization

For clarity of exposition, we restrict our discussion to the Top-1 MoE setting, and later describe how the approach can be extended to our Dynamic-Capacity MoE. We first consider the Top-1 MoE layer whose output is given by:

n−1

Softmax(z)i · Di · Expert(x,wi), where D ∼ Softmax(z). (4)

i=0

Here, z denotes the router logits, Softmax(z)i is the gating probability for expert i, Di is a binary mask indicating whether expert i is selected. Let f(·) denote the remainder of the network, including the loss function. The training objective can then be expressed as:

n−1

Softmax(z)i · Di · Expert(x,wi)

L = ED∼Softmax(z) f

i=0

n−1

f Softmax(z)i · Expert(x,wi) · Softmax(z)i. (5)

=

i=0

Equation 5 rewrites the expectation over D as a weighted sum over experts, where each term is the loss contribution from a single expert multiplied by its routing probability. For notational simplicity, we denote p = Softmax(z). The gradient of L with respect to z can be written as:

n−1

∂f pi · Expert(x,wi) ∂z

∂pi ∂z

∇z =

pi ·

+ f pi · Expert(x,wi) ·

i=0

n−1

∂f pi · Expert(x,wi) ∂z

∂pi ∂z

. (6)

pi ·

+ f pi · Expert(x,wi) − f(0) ·

=

i=0

The second equality in Equation 6 is usually known as baseline subtraction. In the ODE literature, the term f pi · Expert(x,wi) − f(0) can be approximated in various ways. We focus on two common numerical schemes:

- • Euler’s method: a first-order ODE solver that approximates f pi · Expert(x,wi) − f(0) as f′ pi · Expert(x,wi) · pi · Expert(x,wi).
- • Heun’s third-order method: a third-order ODE solver that approximates f pi·Expert(x,wi) −f(0) as 14 · f′ pi · Expert(x,wi) + 34 · f′(pi·Expert3 (x,wi)) · pi · Expert(x,wi).

We next present two alternative approximations of ∇z based on two numerical schemes. First-order (Euler) approximation. Applying Euler’s method, the gradient can be expressed as:

n−1

∂f pi · Expert(x,wi) ∂z

∂pi ∂z

+ f′ piExpert(x,wi) · piExpert(x,wi) ·

∇1st z =

pi ·

i=0

n−1

∂f pi · Expert(x,wi) ∂z

∂f piExpert(x,wi) ∂piExpert(x,wi)

∂piExpert(x,wi) ∂pi

∂pi ∂z

pi ·

+ pi ·

=

i=0

n−1

∂f pi · Expert(x,wi) ∂z

2 · pi ·

=

i=0

∂f pD · Expert(x,wD) ∂z

= ED∼Softmax(z) 2 ·

.

Third-order (Heun) approximation. Using Heun’s method, which combines multiple derivative evaluations for higher accuracy, we obtain:

n−1

∇3rd z =

i=0

- 3

- 4 · f′

∂f pi · Expert(x,wi) ∂z

pi ·

+

1 4 · f′ pi · Expert(x,wi) +

pi · Expert(x,wi)

∂pi ∂z

3 · pi · Expert(x,wi) ·

n−1

i·Expert(x,wi)

∂f(p

∂f pi · Expert(x,wi) ∂z

3 ) ∂z

9 4 · pi ·

5 4 · pi ·

=

+

i=0

∂f 1+23B · pD · Expert(x,wD) ∂z

8) (6 − 4B) ·

=ED∼Softmax(z),B∼Bernoulli(5

.

Hybrid gradient estimator. To balance the stability of router training with the diversity of expert learning, we combine the two estimators above: the first-order gradient is used when the selected expert corresponds to arg max(z), while the third-order gradient is applied otherwise. Let δD denote the indicator δ(D = arg max(z)). The combined estimator can be written as:

∇z = ED∼Softmax(z)[∇Dz],

where ∇Dz

∂f( 1+23 B · pD · Expert(x, wD)) ∂z

) (1 − δD) · (6 − 4B) ·

+ δD · 2 ·

=EB∼Bernoulli( 5 8

∂f 1+2·max(3 B,δD) · pD · Expert(x, wD) ∂z

) 6 − 4 · max(B, δD)

=EB∼Bernoulli( 5 8

∂f(pD · Expert(x, wD)) ∂z

) 2 · f′

=EB∼Bernoulli( 5 8

∂pD · Expert(x, wD) ∂z

1 + 2 max(B, δD) 3 · pD · Expert(x, wD)

. (7)

Gradient Estimation Function. Following the hybrid gradient estimation in Equation 7, the computation can be described as follows. First, we compute the forward output of the sampled expert D weighted by its routing probability:

#### o = Expert(x,wD) · pD.

We then define an indicator variable to check whether D is the highest-probability expert:

δD =

1, if D = arg max(z), 0, otherwise.

Next, we sample a Bernoulli random variable:

5 8

B ∼ Bernoulli

, Finally, the hybrid scaling factor applied to h is:

1 + 2B 3 · o − 2 · o ,

oest = 2 · o + detach max δD,

where detach(·) returns a copy of its argument without gradient flow. When δD = 1, the scaling factor is fixed at 2 (first-order approximation); otherwise, it is given by 1+23 B (third-order approximation).

To extend it to our Top-P strategy, we apply the same computation sequentially to each activated expert, sampling without replacement from the routing distribution until the cumulative probability exceeds P. In this way, the gradient estimation naturally generalizes to multiple experts while remaining consistent with the selection process in Algorithm 1, ensuring that the hybrid scaling mechanism is fully compatible with our dynamic-capacity MoE framework.

