# arXiv:2503.11579v2[cs.CV]16Jul2025

## VAMBA: Understanding Hour-Long Videos with Hybrid Mamba-Transformers

Weiming Ren1,4, Wentao Ma2, Huan Yang3, Cong Wei1,4, Ge Zhang1,5, Wenhu Chen1,4 1University of Waterloo, 2University of Toronto, 3Kuaishou Technology, 4Vector Institute, 5M-A-P

{w2ren,wenhuchen}@uwaterloo.ca, hyang@fastmail.com

https://tiger-ai-lab.github.io/Vamba/

### Abstract

State-of-the-art transformer-based large multimodal models (LMMs) struggle to handle hour-long video inputs due to the quadratic complexity of the causal self-attention operations, leading to high computational costs during training and inference. Existing token compression-based methods reduce the number of video tokens but often incur information loss and remain inefficient for extremely long sequences. In this paper, we explore an orthogonal direction to build a hybrid Mamba-Transformer model (VAMBA) that employs Mamba-2 blocks to encode video tokens with linear complexity. Without any token reduction, VAMBA can encode more than 1024 frames (640×360) on a single GPU, while transformer-based models can only encode 256 frames. On long video input, VAMBA achieves at least 50% reduction in GPU memory usage during training and inference, and nearly doubles the speed per training step compared to transformer-based LMMs. Our experimental results demonstrate that VAMBA improves accuracy by 4.3% on the challenging hour-long video understanding benchmark LVBench over prior efficient video LMMs, and maintains strong performance on a broad spectrum of long and short video understanding tasks.

### 1. Introduction

The field of large multimodal models (LMMs) has seen tremendous progress in recent years. The seminal work of LLaVA [45] successfully transferred the power of autoregressive next-token prediction from large language models (LLMs) [2, 5, 29, 50, 61] to the multimodal domain, establishing a new standard for visual understanding by autoregressively modelling visual and language tokens through causal transformer layers. Since then, transformer-based LMMs have been widely studied and enhanced to tackle a diverse range of tasks, such as high-resolution image understanding [34, 44, 63] and interleaved image-text reasoning [30, 35, 36]. Nevertheless, the quadratic complexity inherent in causal self-attention operations presents significant challenges for long-context inputs, making it difficult for

| | | | |
|---|---|---|---|
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

Figure 1. VAMBA achieves strong long video understanding performance (42.1% on LVBench [64]) while being more computationally efficient compared to transformer-based LMMs.

transformer-based LMMs to effectively handle the task of understanding extremely long videos.

A key challenge for advanced LMMs when processing long video inputs is that they tend to encode each frame into a large number of vision tokens, which leads to substantial computational and memory costs during both training and inference. For example, Qwen2-VL [63] can only process 256 frames (360p) on a single GPU, which is far from sufficient for hour-long video understanding. To reduce the computational/memory cost, previous efforts have primarily focused on reducing the vision tokens in the input sequence. Several approaches [20, 38, 40, 72] utilize a Q-Former [37] to compress vision tokens. More recent methods, such as LongVU [55] and Video-XL [56], partition vision token sequences into chunks and employ adaptive token compression mechanisms to decrease the token count. Another line of work, such as InternVideo2.5 [66] and Orxy-1.5 [48], employs various mechanisms to evaluate the vision tokens’ importance, and thus dropping or merging those less significant tokens [6] during training

- Table 1. Theoretical time and memory complexity for transformerbased LMMs and VAMBA in pre-filling. M and N represent the number of vision and text tokens. d denotes the hidden dimension.

Model Time Complexity Memory Complexity Transformer O(d(M + N)2) O((M + N)2) VAMBA O(dMN + d2M) O(MN)

and inference. Despite these advancements, key challenges in LMMs for long video understanding still persist. First, aggressive token reduction can lead to critical information loss, particularly when processing extremely long videos. Second, these methods still suffer from quadratic computational complexity as the number of input frames increases. Third, token reduction-based methods require additional overhead, which might increase the actual wall-clock time.

In this study, we investigate an orthogonal direction to previous approaches: instead of compressing the video tokens, we seek to develop an alternative model architecture that improves the efficiency of processing video tokens during training and pre-filling stage of inference. We propose VAMBA, a hybrid Mamba-Transformer model for efficient hour-long video understanding. The key insight of our method is that we can design efficient modules to approximate the causal self-attention operation for both text and video tokens in transformer-based LMMs. In particular, we propose to (1) utilize cross-attentions to update text tokens based on video tokens, which is affordable due to the short length of text tokens, and (2) adopt Mamba-2 [17] to process the massive video tokens with linear complexity.

Assuming a combined input sequence of M + N tokens, where M is the number of video tokens and N is the number of text tokens, we find that M could be at least 100 times larger than N on many long video tasks (M ≫ N). Our model can reduce the training/prefilling computational complexity from O(d(M + N)2) to O(dMN + d2M), where d is the hidden dimension, according to Table 1. In practice, this theoretical improvement may not be fully realized due to the hardware underoptimization for Mamba [24]. Nevertheless, we still observe a more than 50% reduction in GPU memory usage and FLOPs/runtime during training and inference for long video inputs, as shown in Figure 1. VAMBA can be efficiently trained using 8×A800 GPUs, whereas other efficient video LMMs such as LongVU [55] and LongLLaVA [65] require 64 and 24 GPUs for training, respectively. By performing a two-stage training, our VAMBA achieves a 4.3% improvement over the best efficient video LMMs on the challenging hour-scale video understanding benchmark LVBench [64]. On other long video understanding datasets like Video-MME [21], MLVU [76] and LongVideoBench [67], VAMBA also achieves top-notch performance.

Our contributions can be summarized as follows:

- 1. We propose VAMBA, a hybrid Mamba-Transformer model for hour-long video understanding. VAMBA’s design features efficient modules such as Mamba-2 blocks and cross-attention layers, effectively reducing the computational overhead of transformer-based LMMs.
- 2. We conduct a comprehensive ablation study and show that employing Mamba-2 blocks and initializing crossattention weights from pretrained self-attention layers are crucial for achieving high performance in VAMBA.
- 3. Our extensive evaluations demonstrate that VAMBA achieves strong video understanding capabilities. Specifically, VAMBA achieves 4.3% improvement over stateof-the-art efficient video LMMs on the challenging hourlong benchmark LVBench.

### 2. Preliminaries

#### 2.1. State Space Models and Mamba

State space models (SSMs) are linear time-invariant systems for modelling continuous signals. A continuous SSM can be expressed by ordinary differential equations (ODEs):

h′(t) = Ah(t) + Bx(t), y(t) = Ch(t),

here, x(t) ∈ R is the input signal, h(t) ∈ RN is the N-dimensional hidden state and y(t) ∈ R is the output signal. A ∈ RN×N is the state transition matrix and B ∈ RN×1,C ∈ R1×N are projection matrices.

Mamba [24], and its predecessor S4 [26], are discretized SSMs for discrete sequence modelling. The discretization of the SSM is done by the zero-order hold (ZOH) technique:

A = exp(∆A), B = (∆A)−1(exp(∆A) − I) · ∆B,

where A,B are discretized state variables and ∆ is the step size. The discretized SSM can thus be rewritten as:

ht = Aht−1 + Bxt, yt = Cht.

Mamba proposes the selective scan algorithm that dynamically determines B,C,∆ based on the sequence inputs. Following S4, the A matrix is initialized as the diagonalized HiPPO matrix [25]. These design choices are proven to be efficient and effective for modelling extremely long sequences. Mamba-2 [17] further simplifies the formulation of A to a scalar times identity structure. Mamba-2 supports multi-head SSM and allows a much larger state dimension N than Mamba, while being faster during training.

#### 2.2. Transformer-based LMMs

We study how text and video tokens are being updated in transformers. As shown in Figure 2, we assume a standard case where video tokens are placed before text tokens and ignore the chat template tokens for simplicity. In each LMM decoder layer, the video and text tokens are updated as:

L

L

Mamba Block

MLP

MLP

Mamba Layer Layer Norm

Layer Norm

Layer Norm

Output Text Tokens

Query Key/Value

Self-Attention

Self-Attention

Layer Norm

Cross-Attention Layer Norm

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

| | | | |
|---|---|---|---|
| | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

| | | | |
|---|---|---|---|
| | | | |

Vision Encoder

Tokenizer Video Input Text Input

Vision Encoder

Tokenizer Video Input Text Input

(a) Transformer-based Video LMMs (b) Our Vamba Model Architecture

- Figure 2. Overview of our VAMBA model architecture. Compared to transformer-based LMMs (left), we replace the costly causal selfattention operations with the more efficient cross-attention layers and Mamba blocks to achieve better efficiency.

- • All video tokens get updated by an input layer normalization (LN) layer, a self-attention layer, a post LN layer and a final MLP layer. There are also residual connections

for self-attention and MLP layers. The ith video token vi computes self-attention based on the first i video tokens:

ov

i

= (σ(

qv

i

K⊤[v

√ 1:vi] d

)V[v

1:vi])Wo, (1) where σ(·) denotes the softmax operation, qv

i

is the query vector for vi and K[v

1:vi], V[v

1:vi] are key and value matrices of the first i video tokens. d denotes the dimension of the query and key vectors and Wo is the out projection matrix. ov

i

represents the final output for token vi.

- • The text tokens share a similar update route to the video tokens. The self-attention updates are slightly different:

for the jth text token tj, it computes self-attention based on all video tokens and the first j text tokens:

1:tj]]⊤ √

qt

[Kv,K[t

1:tj]])Wo, (2) where [Kv,K[t

j

)[Vv,V[t

ot

= (σ(

j

d

1:tj]] are the key and value matrices of the combination of all video tokens and text tokens up to tj.

1:tj]] and [Vv,V[t

The main computation overhead in the transformerbased LMMs comes from the quadratic complexity of the self-attention in the video tokens. To overcome this issue, we design a hybrid Mamba Transformer architecture to process text and video tokens differently. Our detailed model architecture designs are listed in the sections below.

### 3. Our Method: VAMBA

Our goal is to devise a model Θ′ that preserves the performance of transformer Θ, while being more efficient. We approach this problem by approximating the causal transformer layers in the pretrained video LMMs.

#### 3.1. Updating Text Tokens via Cross-Attentions

The key idea of our method is to split the expensive selfattention operation over the entire video and text token sequence into two more efficient components. Since video tokens typically dominate the sequence while text tokens remain few, we maintain the self-attention mechanism exclusively for the text tokens and eliminate it for the video tokens. Instead, we add cross-attention layers that use text tokens as queries and video tokens as keys and values. As shown in Figure 2, this design updates the text tokens during the attention layers in our model as follows:

K⊤v √

qt

)Vv)Woc (Cross-Attention)

j

=(1 − α)(σ(

ot

j

d

K⊤[t

qt

√ 1:tj] d

j

1:tj]])Wos, (Self-Attention)

+ α(σ(

)V[t

(3) where Woc and Wos are the output projection matrices for the cross- and self-attention layers. α ∈ [0,1] is a learnable weighting parameter that balances the cross- and self-attention outputs. Assuming a total of M video tokens and N text tokens, the self-attention operations in the

transformer-based video LMMs have a (pre-filling) computational complexity of O(d(M + N)2). Our self+crossattention design has the complexity of O(dN2) (selfattention) and O(dMN) (cross-attention), effectively reducing the pre-filling complexity to O(dN2 + dMN) ∼ O(dMN), given that M ≫ N ≈ d.

Our attention formulation can be viewed as an approximation of the causal self-attention operation (Equation 2), as each text token qt

can still attend to all video tokens and the first j text tokens. As the video and text tokens share the same channel dimension, the projection matrices in the cross-attention layer will have identical dimensions as in the self-attention layers. This allows us to consider two model design choices: we can either randomly initialize the weights in the cross-attention layers, or we can initialize the cross-attention layers using the self-attention layer weights (i.e. Woc = Wos, same applies to the query, key and value projection matrices Wqc,Wkc,Wvc). We add these two design choices to the VAMBA design space.

j

#### 3.2. Updating Video Tokens with Mamba Blocks

While the self+cross-attention design significantly reduces the model’s complexity, relying solely on cross-attention layers can compromise the model’s representational power. Specifically, rather than updating video tokens through selfattention and MLP layers (i.e., through the causal transformer block, c.f. Section 2.2), the video tokens remain unchanged after the cross-attention layers. To address this limitation, we seek an efficient architecture to approximate the effects of the transformer blocks. Motivated by the success of the Mamba [17, 24] architectures in image and video modelling [39, 47, 77], we propose to employ Mamba blocks to effectively process the video tokens. As shown in Figure 2, the video token updates can be formulated as:

ov

= Mamba(LN(vi),hv

i−1

i

##### ,A,B,C), (4)

where LN(·) is the layer normalization operator and hv

i−1

is the context (hidden states) at token vi−1. As a recurrent model, Mamba operates similarly to causal self-attention layers, enabling each video token to update itself based on all preceding tokens. Crucially, Mamba reduces the complexity of updating video tokens from O(dM2) in causal self-attentions to O(d2M). Combining the Mamba blocks and the self+cross-attention design, our model achieves an overall complexity of O(dN2 + dMN + d2M) ∼ O(dMN + d2M), which is substantially lower than the quadratic complexity O(d(M + N)2) in the transformerbased video LMMs. In Section 3.3, we further detail how the Mamba layers are trained to approximate Equation 1.

We consider employing Mamba or Mamba-2 blocks in Equation 4 in the VAMBA design space.

#### 3.3. Training Paradigm

We utilize a two-stage training strategy to optimize VAMBA, including a pretraining stage and an instruction-tuning stage. In the pretraining stage, the model is initialized with pretrained transformer-based LMM weights in all modules except for the newly introduced cross-attention and Mamba layers. We freeze the pretrained components and train only the new layers using image captioning data to restore the model’s visual understanding capabilities. We explore two types of training objectives during pretraining: (1) the standard language modelling loss LLM = −T1 Tt=1 log p(xt|x<t), which is the cross-entropy loss for next-token (xt) prediction; (2) the distillation loss LDistill, where we extract the top 100 logits PΘ with the highest values from the transformer-based model Θ and compute a KL-divergence loss with the logit outputs PΘ′ of our model Θ′ at the same indices, such that LDistill = DKL(PΘ||PΘ′) (DKL(·||·) denotes KL-divergence). The final loss is formulated as L = LLM + λLDistill, where we use λ to balance the weights of the two losses. We include λ = 0,0.001,0.1,0.5,1,2 to the VAMBA design space.

In the instruction-tuning stage, we leverage both image and video instruction-following data to fully finetune VAMBA, thereby enhancing its instruction-following capability. When GPU memory limitations prevent full finetuning, we freeze the vision encoder and finetune only the LMM decoder. We only employ the language modelling loss during the instruct-tuning stage to ensure that the teacher model does not restrict the student’s performance.

The full design space for VAMBA includes whether to initialize cross-attention weights from self-attention; whether to use Mamba or Mamba-2 blocks to update video tokens, and the choice of λ in the pretraining stage.

### 4. Experiments

In this section, we first ablate the VAMBA design choices and then adopt the best setup for full-scale training.

#### 4.1. Ablation Study

We explore the VAMBA design space to determine the effectiveness of our model components. We initialize all our models based on a strong pretrained transformer-based LMM Qwen2-VL-7B [63]. For all ablation experiments, we randomly select one million images from CC12M [9] and employ the PixelProse [57] captions for pretraining. We use the LLaVA-Video-178K [75] dataset with a total of 1.3M video-instruction pairs for instruction-tuning. Detailed hyperparameter settings can be found in Appendix 7.

Evaluation Metrics For pretraining (stage 1) evaluation, we sample 500 images from the COCO-2017 [14] dataset and have the models generate captions for each image. We then apply the reference-free version of G-VEval [60], a

- Table 2. Ablation study results for the VAMBA design space explorations. “CA from SA?” denotes whether to initialize crossattention weights from self-attention layers. “VidMME” represents the Video-MME benchmark.

Model ID

CA from SA?

Mamba Block Type

Stage 1 Stage 2 G-VEval LVBench VidMME MVBench ref. free test w/o sub. test

- A ✗ N/A 75.7 23.7 47.6 40.9
- B ✓ N/A 81.0 34.2 51.7 51.8
- C ✓ Mamba 81.8 34.2 53.4 53.5
- D ✓ Mamba-2 82.2 35.3 54.1 53.5

- Table 3. Ablation study results on pretraining VAMBA models with a transformer-based LMM teacher and a distillation loss.

Model D

λ value in Stage 1 Training

+ Distill

0 0.001 0.01 0.5 1 2 G-VEval 82.19 81.05 80.68 73.69 63.65 47.61

metric based on GPT-4o [51] to evaluate the quality of the captions. For instruction-tuning (stage 2), we evaluate model performances across three benchmarks: LVBench [64] for hour-long videos, Video-MME [21] for medium-tolong videos, and MVBench [40] for short videos. Ablation study results are shown in Table 2 and Table 3.

Cross-Attention Layer Initialization Strategy According to Table 2, we find that it is crucial to initialize the cross-attention layer weights using the corresponding selfattention layer from the same decoder level. Compared to Model A, the dramatic performance boost observed in

- Model B across all metrics underscores the critical impact of this weight initialization strategy. We believe this observation stems from the fact that initializing the crossattention layer weights from the self-attention layers allows

- our self+cross-attention operation (Equation 3) to more closely approximate the pretrained causal self-attention op-

eration (Equation 2): once Wqc,Wkc,Wvc and Woc are set equal to Wqs,Wks,Wvs and Wos, the only discrepancy between the two equations becomes their attention score matrices. This difference can be mitigated by our two-stage training process, making it easier for Model B to recover its multimodal understanding capabilities.

Mamba Block Design Choices Comparing Model B with

- Model C and D in Table 2, we observe that both Mamba and Mamba-2 blocks improve the model performances across all metrics, indicating that Mamba blocks bring extra representation power to the model’s visual understanding capabilities. Furthermore, when comparing Model C and Model
- D, we find that Mamba-2 demonstrates superior image and video modelling capabilities. Despite having a simplified

Table 4. Comparisons between baseline models and VAMBA on hour-long video understanding benchmarks. Bold: best results among efficient video LMMs. Underline: second-best.

Hour-Long Video Understanding Models Size LVBench HourVideo HourEval

test dev test Transformer-based LMMs

Gemini-1.5-Pro [59] - 33.1 37.3 GPT-4o [51] - 34.7 19.6 Qwen2-VL [63] 7B 42.0 33.8 53.0

Efficient LMMs

LLaVA-Mini [74] 7B 17.6 20.6 24.2 LongLLaVA [65] 9B 31.2 27.7 39.1 LongVU [55] 7B 37.8 30.8 46.8 Video-XL [56] 7B 36.8 33.0 47.1

###### VAMBA 10B 42.1 33.6 50.7

A matrix structure, Mamba-2 accommodates a larger state dimension than Mamba (64 versus 16 in our case), which likely contributes to its improved performance.

Pretraining Distillation Loss We employ the pretrained Qwen2-VL-7B model as the teacher model and perform a series of pretraining experiments with the additional distillation loss based on our best model setting (Model D in Table 2). The results are shown in Table 3. In contrast to previous findings from cross-attention-based LLMs (e.g. CEPE [70]), where distillation loss is reported to be beneficial, we observe that incorporating an additional distillation loss does not further improve our model’s performance, as the G-VEval score decreases for all λ > 0. Therefore, we exclude the distillation loss and only employ the language modelling loss in both stages in our final training setting.

#### 4.2. Main Evaluation Results

Implementation Details Based on our ablation study (Table 2), we adopt Model D as our final design and initialize VAMBA from Qwen2-VL-7B. We source ∼3M image caption data for pretraining and ∼7M image and video QA data for instruction-tuning. The full implementation details can be found in Appendix 7.

Hour-Long Video Understanding We evaluate our model’s ability to handle extremely long videos using two public benchmarks: LVBench [64] and HourVideo [8] (development set). We further compose a new benchmark called HourEval by selecting all questions related to videos longer than 30 minutes from Video-MME [21], MLVU [76] development set, and LongVideoBench [67] validation set. The average video lengths for LVBench, HourVideo, and HourEval are 68.4, 47.2, and 54.7 minutes, respectively. We compare our model against four efficient video LMMs:

Table 5. Comparisons between baseline models and VAMBA on medium-length or short video understanding benchmarks. Bold: best results among each section. Underline: second-best. * indicates the results based on our evaluation scripts.

Medium-Length Understanding Short Video Understanding Video Captioning Models Size Video-MME MLVU LongVideoBench MVBench NExT-QA DREAM-1K

w/o subtitles m-avg val test mc F1 Proprietary Models

GPT-4V [2] - 59.9 49.2 59.1 43.5 70.4 34.4 GPT-4o [51] - 71.9 64.6 66.7 - 76.0 39.2 Gemini-1.5-Pro [59] - 75.0 62.9 64.0 54.2 76.4 36.2

Open-source Transformer-based LMMs VideoChat2 [40] 7B 39.5 47.9 39.3 51.9 78.6 26.6 ShareGPT4Video [12] 7B 39.9 46.4 39.7 51.2 - 19.5 LongVA [73] 7B 52.4 56.3 51.8 49.2 68.3 Video-CCAM [20] 9B 50.3 58.5 - 64.6 - Kangaroo [46] 8B 56.0 61.0 54.8 61.1 - InternVL2 [15] 8B 54.0 48.1* 51.8* 66.4 - 26.9 LLaVA-OneVision [35] 7B 58.2 62.6* 56.4 56.7 79.4 31.7 Qwen2-VL [63] 7B 63.3 64.2* 52.4* 67.0 - 29.6 Phi-4-Mini [1] 5.6B 55.0 60.1 46.7 60.4 - -

Open-source Efficient LMMs

LLaVA-Mini [74] 7B 40.3 44.3 19.3* 44.5 47.6* 22.9 LongLLaVA [65] 9B 51.6 53.3* 42.1* 54.6 72.2* 24.6 LongVU [55] 7B 55.3* 65.4 53.5* 66.9 78.0* 28.1 Video-XL [56] 7B 55.5 64.9 50.7 55.3 77.5* 23.5

###### VAMBA 10B 57.8 65.9 55.9 60.4 78.1 28.1

LLaVA-Mini [74], LongLLaVA [65], Video-XL [56] and LongVU [55]. We also include the results from Qwen2-VL7B (our baseline transformer-based LMM) as a reference.

Experimental results are shown in Table 4. Our VAMBA consistently outperforms all efficient video LMMs across the three hour-long video benchmarks, highlighting its exceptional ability to understand and reason over hour-scale videos. Notably, our model surpasses the baseline Qwen2VL-7B on the LVBench benchmark, and its performance on HourVideo is also very close to Qwen2-VL-7B. These results underscore that our VAMBA is competitive with the best open-sourced transformer-based LMMs, while being significantly more efficient during training and inference.

Medium-Length or Short Video Understanding To further demonstrate our model’s generalizability across vari-

- ous video durations, we test VAMBA on several mediumlength or short video understanding benchmarks. Specifically, we report multiple-choice question performance on Video-MME, MLVU, LongVideoBench, NExT-QA [68] and MVBench [40]. We also evaluate VAMBA on DREAM1K [62] for video captioning assessment.

According to Table 5, our VAMBA demonstrates superior performance across three medium-length video understanding benchmarks (with average video durations between 10–20 minutes), ranking first among efficient video LMMs

on all metrics. Despite the newly integrated cross-attention and Mamba layers in VAMBA have only been trained on relatively smaller datasets, VAMBA’s performance remains competitive with large-scale pretrained transformer-based LMMs. Furthermore, all our evaluations can be conducted on a single 80G GPU, while some transformer-based LMMs, such as Qwen2-VL-7B, require additional inference strategies like sliding window attention or ring attention over multiple GPUs to achieve optimal results.

For short video understanding and video captioning benchmarks, VAMBA also achieves competitive performances, ranking first on NExT-QA and DREAM-1K and second on MVBench among efficient LMMs. Overall, our model delivers the best results on medium-length and long video benchmarks, demonstrating its strong ability to handle long-context video-language inputs.

#### 4.3. Runtime Efficiency Analysis

To understand our model’s runtime efficiency gains over the baseline transformer-based LMM (Qwen2-VL-7B), we conduct an efficiency analysis for both training and inference. For model training, we ensure a consistent training environment by using 8 NVIDIA A800 80G GPUs for both models. All input videos are resized to a resolution of 640 × 360, and a batch size of 1 is maintained

(a) Training GPU Memory Usage Comparison

80

TrainingGPUMemory(GB)

Qwen2-VL-7B OOM

Vamba

70

60

50

40

8 16 32 64 128 256 512 Number of Frames

(b) Training Runtime Comparison (time for 1 training step)

OOM Qwen2-VL-7B

15

TrainingRuntime(s)

Vamba

10

5

0

8 16 32 64 128 256 512 Number of Frames

- Figure 3. Comparison of training GPU memory usage and runtime per training step between Qwen2-VL-7B and VAMBA.

(a) Pre-filling GPU Memory Usage Comparison

80

OOM Qwen2-VL-7B

PrefillGPUMemory(GB)

Vamba

60

40

20

0

8 16 32 64 128 256 512 1024 Number of Frames

(b) Pre-filling FLOPs Comparison (measured using calflops)

1600

OOM Qwen2-VL-7B

Vamba

1200

FLOPs(T)

800

400

0

8 16 32 64 128 256 512 1024 Number of Frames

- Figure 4. Comparison of GPU memory usage and FLOPs between Qwen2-VL-7B and VAMBA during inference.

- 4.4. Case Study

In this section, we perform a case study for Video-XL and VAMBA using different video inputs and show the results in Figure 5. The “cookie” example focuses on object property identification in a long video (36 minutes). Video-XL incorrectly identifies the cookie colors as green and white, possibly due to the video also containing scenes for making other desserts (e.g. white ice cream in the frame highlighted in red). On the other hand, VAMBA more accurately describes them as green and brown. In the “magic trick” example, Video-XL misinterprets the rubber band as a green ring, whereas VAMBA correctly describes the performer manipulating the rubber band around their fingers, making it appear to pass through. These comparisons highlight how VAMBA is better at retrieving relevant events in long videos and capturing fine-grained visual details and actions.

- 5. Related Work

on each GPU. We apply standard optimization methods such as Flash-Attention 2 [16], DeepSpeed ZeRO-3 [52] and gradient checkpointing [13] for both models. Based on different numbers of input frames, we measure the average GPU memory across 8 GPUs and the runtime per training step during training. As shown in Figure 3a, although VAMBA and Qwen2-VL-7B share the same vision encoder design and generate vision tokens of identical sequence lengths, our model requires over 50% less training memory when processing videos with more than 16 frames. This efficiency gain allows us to handle a larger number of video frames during training (512 vs. 128). Furthermore, as depicted in Figure 3b, our efficient design also accelerates model training, achieving nearly twice the speedup per training step when working with more than 64 frames.

For model inference, we focus on the pre-fill stage and analyze GPU memory usage and FLOPs for both models. All measurements are conducted on an input video with a resolution of 640 × 360. As shown in Figure 4a, VAMBA requires slightly more GPU memory than Qwen2-VL-7B when the number of input frames is low (fewer than 32 frames), due to the additional ∼3B parameters in its crossattention and Mamba layers. However, VAMBA’s memory usage increases more slowly as the frame increases, allowing it to handle four times as many frames on a single NVIDIA A800 80G GPU compared to Qwen2-VL-7B (256 vs. 1024). Regarding computational cost, VAMBA reduces FLOPs by 30% to 50% during inference (Figure 4b), demonstrating a significantly lower complexity than its transformer-based LMM counterparts.

Large Multimodal Models (LMMs) Large Multimodal Models (LMMs) have rapidly evolved in recent years. Recent research has focused on improving LMM’s instructionfollowing ability through better data curation [11, 35, 40, 44, 53, 75], building better vision encoders that can process images at higher resolutions [34, 44, 63], and extending LMMs to interleaved image [30, 34, 36] or video [42, 49, 73] understanding. To enhance instruction-following capabilities, LLaVA-1.5 [43] and LLaVA-NeXT [44] leverage GPT-4-generated multimodal data for knowledge-enhanced tuning. To optimize visual feature extraction, Idefics2 [34] leverages the NaViT [18] strategy and the SigLIP [71] vision encoder to support image inputs with native resolu-

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

|[Figure 6]|
|---|

|[Figure 7]|
|---|

|[Figure 8]|
|---|

[Figure 9]

[Figure 10]

Question: What are the colors of the cookies that the person is making?

Video-XL: The cookies have different colored fillings, with at least one being green and another one being white.

Vamba: The person is making cookies in two different colors, green and brown.

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Question: Describe the magic trick.

Video-XL: The magic trick involves a person holding a green ring and a piece of string... The person's fingers are spread out, and they seem to be guiding the ring around the string.

Vamba: ...The person demonstrates a magic trick by manipulating the rubber band around their fingers, making it appear to pass through their hand.

Figure 5. Qualitative comparison between VAMBA and efficient LMM baselines. Red text denotes incorrect responses, while green text highlights the correct responses by our VAMBA.

tion. Qwen2-VL [63] applied Multimodal RoPE (M-RoPE) to optimize feature extraction for varying input sizes.

Despite significant advances, current state-of-the-art LMMs remain inefficient at handling long-context multimodal inputs due to the quadratic complexity of causal selfattention operations. To mitigate this issue, methods like Flamingo [3], OpenFlamingo [4], Idefics [33] and LLaMA 3.2 [19] interleave gated cross-attention layers with selfattention layers for vision-text modelling. mPlug-Owl3 [69] develops the hyper-attention layer that integrates selfattention with cross-attention to support multimodal understanding. However, these models generally underperform compared to LLaVA-like transformer-based LMMs. In this study, we identify that one possible reason is the lack of vision token updates for cross-attention-based models. Our VAMBA addresses this limitation by incorporating additional Mamba layers to update vision tokens.

Efficient LLMs/LMMs To improve the efficiency of LMM reasoning, recent work [37, 38, 40, 55, 56, 72, 74] develops techniques to reduce the length of the vision token sequence. Video-LLaMA [72] and VideoChat [38, 40] use a Q-Former [37] to compress dense visual tokens into a more compact sequence. LLaVA-Mini [74] further reduces each image to only one token. Video-XL [56] introduces a Visual Summarization Token to summarize the visual information, while LongVU [55] designs a spatiotemporal adaptive compression mechanism to reduce the number of video tokens.

Another research avenue, exemplified by the Mamba [17, 24] model series, focuses on developing linear-time sequence models to reduce complexity. Mamba-based [78] and Mamba-Transformer hybrid models [22, 41] have achieved notable success in the realm of pure language modelling. However, these efficient architectures have received little attention in the multimodal understanding domain. LongLLaVA [65] took an initial approach by incorporating the hybrid Jamba [41] model as the LMM decoder, yet its performance in long video understanding remains suboptimal. This shortfall is likely due to the absence of a strong hybrid LLM as the base model and the lack of largescale training on multimodal instruction-following data.

### 6. Conclusion

We presented VAMBA, a hybrid Mamba-Transformer model for efficient hour-long video understanding. By integrating cross-attention for text tokens and Mamba-2 blocks for video token updates, our approach reduces computational complexity and GPU memory usage while achieving competitive performance across long, medium, and short video benchmarks. Extensive evaluations on datasets such as LVBench demonstrate VAMBA ’s superiority over existing efficient video LMMs. For future work, since our model design is orthogonal to token reduction-based methods, we will focus on combining the strengths of both approaches to develop more efficient VAMBA variants.

### References

- [1] Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, Dong Chen, Dongdong Chen, Junkun Chen, Weizhu Chen, YenChun Chen, Yi ling Chen, Qi Dai, Xiyang Dai, Ruchao Fan, Mei Gao, Min Gao, Amit Garg, Abhishek Goswami, Junheng Hao, Amr Hendy, Yuxuan Hu, Xin Jin, Mahmoud Khademi, Dongwoo Kim, Young Jin Kim, Gina Lee, Jinyu Li, Yunsheng Li, Chen Liang, Xihui Lin, Zeqi Lin, Mengchen Liu, Yang Liu, Gilsinia Lopez, Chong Luo, Piyush Madan, Vadim Mazalov, Ali Mousavi, Anh Nguyen, Jing Pan, Daniel Perez-Becker, Jacob Platin, Thomas Portet, Kai Qiu, Bo Ren, Liliang Ren, Sambuddha Roy, Ning Shang, Yelong Shen, Saksham Singhal, Subhojit Som, Xia Song, Tetyana Sych, Praneetha Vaddamanu, Shuohang Wang, Yiming Wang, Zhenghao Wang, Haibin Wu, Haoran Xu, Weijian Xu, Yifan Yang, Ziyi Yang, Donghan Yu, Ishmam Zabir, Jianwen Zhang, Li Lyna Zhang, Yunan Zhang, and Xiren Zhou. Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras, 2025. 6
- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 1, 6

- [3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736,

2022. 8

- [4] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. Openflamingo: An opensource framework for training large autoregressive visionlanguage models. arXiv preprint arXiv:2308.01390, 2023. 8
- [5] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 1
- [6] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. In ICLR, 2023. 1
- [7] Joao Carreira, Eric Noland, Chloe Hillier, and Andrew Zisserman. A short note on the kinetics-700 human action dataset. arXiv preprint arXiv:1907.06987, 2019. 1
- [8] Keshigeyan Chandrasegaran, Agrim Gupta, Lea M Hadzic, Taran Kota, Jimming He, Crist´obal Eyzaguirre, Zane Durante, Manling Li, Jiajun Wu, and Fei-Fei Li. Hourvideo:

- 1-hour video-language understanding. Advances in Neural Information Processing Systems, 37:53168–53197, 2025. 5,
- 2

- [9] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pretraining to recognize long-tail visual concepts. In Proceed-

- ings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3558–3568, 2021. 4
- [10] Guiming Hardy Chen, Shunian Chen, Ruifei Zhang, Junying Chen, Xiangbo Wu, Zhiyi Zhang, Zhihong Chen, Jianquan Li, Xiang Wan, and Benyou Wang. Allava: Harnessing gpt4v-synthesized data for lite vision-language models. arXiv preprint arXiv:2402.11684, 2024. 1
- [11] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. In European Conference on Computer Vision, pages 370–387. Springer, 2024. 7, 1
- [12] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Zhenyu Tang, Li Yuan, et al. Sharegpt4video: Improving video understanding and generation with better captions. Advances in Neural Information Processing Systems, 37:19472–19495, 2025. 6, 1
- [13] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174, 2016. 7, 1
- [14] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015. 4, 1
- [15] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024. 6
- [16] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023. 7, 1
- [17] Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. arXiv preprint arXiv:2405.21060, 2024. 2, 4, 8
- [18] Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M Alabdulmohsin, et al. Patch n’pack: Navit, a vision transformer for any aspect ratio and resolution. Advances in Neural Information Processing Systems, 36:2252–2274, 2023. 7
- [19] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 8

- [20] Jiajun Fei, Dian Li, Zhidong Deng, Zekun Wang, Gang Liu, and Hui Wang. Video-ccam: Enhancing video-language understanding with causal cross-attention masks for short and long videos. arXiv preprint arXiv:2408.14023, 2024. 1, 6
- [21] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 2, 5

- [22] Paolo Glorioso, Quentin Anthony, Yury Tokpanov, James Whittington, Jonathan Pilault, Adam Ibrahim, and Beren Millidge. Zamba: A compact 7b ssm hybrid model. arXiv preprint arXiv:2405.16712, 2024. 8
- [23] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18995–19012, 2022. 2
- [24] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023. 2, 4, 8
- [25] Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, and Christopher R´e. Hippo: Recurrent memory with optimal polynomial projections. Advances in neural information processing systems, 33:1474–1487, 2020. 2
- [26] Albert Gu, Karan Goel, and Christopher R´e. Efficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396, 2021. 2
- [27] Muyang He, Yexin Liu, Boya Wu, Jianhao Yuan, Yueze Wang, Tiejun Huang, and Bo Zhao. Efficient multimodal learning from data-centric perspective. arXiv preprint arXiv:2402.11530, 2024. 1
- [28] Md Mohaiminul Islam, Tushar Nagarajan, Huiyu Wang, Gedas Bertasius, and Lorenzo Torresani. Bimba: Selectivescan compression for long-range video question answering. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29096–29107, 2025. 3
- [29] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023. 1
- [30] Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max Ku, Qian Liu, and Wenhu Chen. Mantis: Interleaved multi-image instruction tuning. arXiv preprint arXiv:2405.01483, 2024. 1, 7
- [31] Jindong Jiang, Xiuyu Li, Zhijian Liu, Muyang Li, Guo Chen, Zhiqi Li, De-An Huang, Guilin Liu, Zhiding Yu, Kurt Keutzer, et al. Token-efficient long video understanding for multimodal llms. arXiv preprint arXiv:2503.04130, 2025. 3
- [32] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123:32–73, 2017. 1
- [33] Hugo Laurenc¸on, Lucile Saulnier, L´eo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander Rush, Douwe Kiela, et al. Obelics: An open web-scale filtered dataset of interleaved image-text documents. Advances in Neural Information Processing Systems, 36:71683–71702, 2023. 8
- [34] Hugo Lauren¸con, L´eo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models? arXiv preprint arXiv:2405.02246, 2024. 1, 7

- [35] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 1, 6, 7
- [36] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024. 1, 7
- [37] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–

19742. PMLR, 2023. 1, 8

- [38] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 1, 8
- [39] Kunchang Li, Xinhao Li, Yi Wang, Yinan He, Yali Wang, Limin Wang, and Yu Qiao. Videomamba: State space model for efficient video understanding. In European Conference on Computer Vision, pages 237–255. Springer, 2024. 4
- [40] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195– 22206, 2024. 1, 5, 6, 7, 8, 2
- [41] Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen, Jhonathan Osin, Itay Dalmedigos, Erez Safahi, Shaked Meirom, Yonatan Belinkov, Shai Shalev-Shwartz, et al. Jamba: A hybrid transformer-mamba language model. arXiv preprint arXiv:2403.19887, 2024. 8
- [42] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023. 7
- [43] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024. 7, 1
- [44] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next. https: //llava-vl.github.io/blog/2024-01-30llava-next/, 2024. Accessed: 2025-02-14. 1, 7
- [45] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024. 1
- [46] Jiajun Liu, Yibing Wang, Hanghang Ma, Xiaoping Wu, Xiaoqi Ma, Xiaoming Wei, Jianbin Jiao, Enhua Wu, and Jie Hu. Kangaroo: A powerful video-language model supporting long-context video input. arXiv preprint arXiv:2408.15542,

2024. 6

- [47] Yue Liu, Yunjie Tian, Yuzhong Zhao, Hongtian Yu, Lingxi Xie, Yaowei Wang, Qixiang Ye, and Yunfan Liu. Vmamba: Visual state space model 2024. arXiv preprint arXiv:2401.10166, 2024. 4
- [48] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Oryx mllm: On-demand

- spatial-temporal understanding at arbitrary resolution. arXiv preprint arXiv:2409.12961, 2024. 1
- [49] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023. 7, 1
- [50] OpenAI. Chatgpt. https://openai.com/index/ chatgpt/, 2023. 1
- [51] OpenAI. Gpt-4o. https://openai.com/index/ hello-gpt-4o/, 2024. 5, 6
- [52] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–16. IEEE, 2020. 7, 1
- [53] Weiming Ren, Huan Yang, Jie Min, Cong Wei, and Wenhu Chen. Vista: Enhancing long-duration and high-resolution video understanding by video spatiotemporal augmentation. arXiv preprint arXiv:2412.00927, 2024. 7
- [54] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 1
- [55] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, et al. Longvu: Spatiotemporal adaptive compression for long video-language understanding. arXiv preprint arXiv:2410.17434, 2024. 1, 2, 5, 6, 8
- [56] Yan Shu, Peitian Zhang, Zheng Liu, Minghao Qin, Junjie Zhou, Tiejun Huang, and Bo Zhao. Video-xl: Extra-long vision language model for hour-scale video understanding. arXiv preprint arXiv:2409.14485, 2024. 1, 5, 6, 8
- [57] Vasu Singla, Kaiyu Yue, Sukriti Paul, Reza Shirkavand, Mayuka Jayawardhana, Alireza Ganjdanesh, Heng Huang, Abhinav Bhatele, Gowthami Somepalli, and Tom Goldstein. From pixels to prose: A large dataset of dense image captions. arXiv preprint arXiv:2406.10328, 2024. 4
- [58] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18221–18232, 2024. 1
- [59] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 5, 6
- [60] Tony Cheng Tong, Sirui He, Zhiwen Shao, and Dit-Yan Yeung. G-veval: A versatile metric for evaluating image and video captions using gpt-4o. arXiv preprint arXiv:2412.13647, 2024. 4
- [61] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al.

- Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1
- [62] Jiawei Wang, Liping Yuan, Yuchen Zhang, and Haomiao Sun. Tarsier: Recipes for training and evaluating large video description models. arXiv preprint arXiv:2407.00634, 2024. 6, 2
- [63] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1, 4, 5, 6, 7, 8
- [64] Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Xiaotao Gu, Shiyu Huang, Bin Xu, Yuxiao Dong, et al. Lvbench: An extreme long video understanding benchmark. arXiv preprint arXiv:2406.08035, 2024. 1, 2, 5
- [65] Xidong Wang, Dingjie Song, Shunian Chen, Chen Zhang, and Benyou Wang. Longllava: Scaling multi-modal llms to 1000 images efficiently via a hybrid architecture. arXiv preprint arXiv:2409.02889, 2024. 2, 5, 6, 8, 1
- [66] Yi Wang, Xinhao Li, Ziang Yan, Yinan He, Jiashuo Yu, Xiangyu Zeng, Chenting Wang, Changlian Ma, Haian Huang, Jianfei Gao, et al. Internvideo2. 5: Empowering video mllms with long and rich context modeling. arXiv preprint arXiv:2501.12386, 2025. 1
- [67] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. Advances in Neural Information Processing Systems, 37:28828–28857, 2025. 2, 5
- [68] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9777–9786, 2021. 6, 2
- [69] Jiabo Ye, Haiyang Xu, Haowei Liu, Anwen Hu, Ming Yan, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplugowl3: Towards long image-sequence understanding in multimodal large language models. In The Thirteenth International Conference on Learning Representations, 2024. 8
- [70] Howard Yen. Long-context language modeling with parallel context encoding. Master’s thesis, Princeton University,

2024. 5

- [71] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023. 7
- [72] Hang Zhang, Xin Li, and Lidong Bing. Video-LLaMA: An instruction-tuned audio-visual language model for video understanding. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 543–553, Singapore, 2023. Association for Computational Linguistics. 1, 8
- [73] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024. 6, 7

- [74] Shaolei Zhang, Qingkai Fang, Zhe Yang, and Yang Feng. Llava-mini: Efficient image and video large multimodal models with one vision token. arXiv preprint arXiv:2501.03895, 2025. 5, 6, 8, 1
- [75] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 4, 7
- [76] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv preprint arXiv:2406.04264,

2024. 2, 5

- [77] Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, and Xinggang Wang. Vision mamba: Efficient visual representation learning with bidirectional state space model. arXiv preprint arXiv:2401.09417, 2024. 4
- [78] Jingwei Zuo, Maksim Velikanov, Dhia Eddine Rhaiem, Ilyas Chahed, Younes Belkada, Guillaume Kunsch, and Hakim Hacid. Falcon mamba: The first competitive attention-free

- 7b language model. arXiv preprint arXiv:2410.05355, 2024.
- 8

## VAMBA: Understanding Hour-Long Videos with Hybrid Mamba-Transformers Supplementary Material

### 7. Additional Implementation Details

We use 8 NVIDIA A800 80G GPUs to train our models for both ablation study and full-scale training. For ablation studies, the learning rate is set to 1e-5 for pretraining and 1e-7 for instruction-tuning. We further conduct a hyperparameter search and find that setting the learning rate to 5e-6 during instruction-tuning works the best for VAMBA across multiple benchmarks. We therefore set the learning rate to 1e-5 for pretraining and 5e-6 for instruction tuning for fullscale training. We employ a cosine learning rate schedule for all training stages in both ablation studies and full-scale training. The training batch size is set to 128. We employ training optimization methods such as Flash-Attention 2 [16], DeepSpeed ZeRO-3 [52] and gradient checkpointing [13] to reduce the training cost, and apply sequence parallelism to pack multiple samples into one sequence during training in both stages.

### 8. Model Evaluation Details

In this section, we provide more details for benchmarking VAMBA and our selected baseline models.

#### 8.1. Baseline Models

Qwen2-VL [63] is an LMM that uses the Qwen2 LLM as its backbone and a DFN-derived Vision Transformer with 2D RoPE positional embedding. It is pretrained on a vast 1.4Ttoken multimodal corpus composed of image-text pairs, OCR text (images of text), interleaved image–text web articles, visual QA data, video dialogues, and image-based knowledge datasets. The pre-training is staged: 600B tokens for vision-language alignment followed by 800B tokens mixing richer image–text content and VQA/multitask data, alongside continued pure text to maintain language skills. Finally, Qwen2-VL is instruction-tuned via ChatMLformat dialogs that span multiple modalities, e.g. document parsing, comparisons of two images, long video understanding, and even agent-oriented visual tasks.

LLaVA-Mini [74] is a compact multimodal model built on a 7–8B Vicuna LLM with a CLIP ViT-based vision encoder. It uses the same training data as LLaVA-1.5 [43]: about 558K image–caption pairs for initial vision–language pre-training and 665K image-grounded instruction examples for fine-tuning. The pre-training stage aligns visual features to text using caption datasets like COCO [14] and VisualGenome-based [32] captions, while the instructiontuning stage uses multimodal dialogues. An enhanced variant of LLaVA-Mini further incorporates 100K video-based instruction samples from Video-ChatGPT [49] and other

open sources, combined with the original 665K image instructions (total 3M training instances) to extend its capability to video understanding.

LongLLaVA [65] extends LLaVA [43] to handle very long visual contexts by using a hybrid Transformer–Mamba architecture with a Jamba-9B backbone for language. It follows a three-stage training process, including a single-image feature alignment on Allava-Caption [10] and ShareGPT4V [11], a single-image instruction finetuning on LLava-1.5 and Mantis-Single [30], and multiimage instruction fine-tuning on VideoChat2 [38] and ShareGPT4Video [12]. By progressively increasing the number of images per sample, LongLLaVA learns temporal and spatial dependencies and can efficiently handle input sequences up to around 1000 images.

LongVU [55] is a multimodal model geared toward long video understanding. It first learns from 3.2 million image–text pairs via a single-image training stage using the LLaVA-OneVision dataset [35]. It then leverages a subset of VideoChat2-IT [38] that contains around 0.55M videos, 1K video-classification clips from Kinetics-710 [7], and about 85K multimodal video instruction dialogues from the ShareGPT4Video [12]. Additionally, the MovieChat longvideo dialogue data [58] is used to provide hour-length conversational examples. This rich training mix enables LongVU to handle extended videos by adaptively compressing frames while preserving essential visual details.

Video-XL [56] employs an LLaMA-based 7B language model and a CLIP ViT-L vision encoder, and it is trained entirely on image-based data despite targeting long videos. Its two-stage training first performs projection-layer pretraining on 2M image–text pairs from Laion-2M [54] to align visual features with the text space. It then undergoes visual instruction tuning on roughly 695K image-grounded instruction samples from Bunny-695k [27], where the model learns to follow image-based instructions. The training approach lets Video-XL handle hour-long videos in context by compressing visual tokens, achieving strong results on benchmarks for long video comprehension.

#### 8.2. Evaluation Benchmarks

LVBench [64] is a benchmark designed to test the ability of video LMMs to comprehend extremely long videos. It contains 1,549 question-answer pairs, with an average video length of 4,101 seconds. The evaluation focuses on six fundamental aspects: temporal grounding, which involves identifying specific moments in a video; video summarization, which assesses the model’s ability to condense key information; video reasoning, which tests logical infer-

ence from video content; entity recognition, which identifies people, objects, or places; event understanding, which captures the sequence and significance of events; and key information retrieval, which ensures the model extracts crucial details. The full test set is used for evaluation.

HourVideo [8] is a benchmark dataset for long-form videolanguage understanding, focusing on videos up to one hour in length. It consists of 500 carefully selected firstperson videos sourced from the Ego4D [23] dataset, with each video ranging from 20 to 120 minutes in duration. The dataset includes 12,976 human-annotated multiplechoice questions covering four major task categories: summarization, perception, visual reasoning, and navigation. HourVideo is designed to challenge models in long-context reasoning and multimodal comprehension across extended video timelines. Benchmark results reveal that existing multimodal models, such as GPT-4 and LLaVA-NeXT, perform only marginally better than random chance, while human experts achieve an accuracy of 85.0%. This highlights the dataset’s difficulty and the current gap in long-video understanding capabilities.

Video-MME [21] is a benchmark specifically designed to evaluate how well LMMs can analyze video content. It features a dataset of 900 videos and 2700 questions, covering six different visual domains. The questions are grouped based on video length into short, medium, and long categories, with median durations of 26 seconds, 164.7 seconds, and 890.7 seconds, respectively. The benchmark supports two evaluation methods: (1) the “w/ subtitle” setting, where both subtitles and questions are provided as text inputs, and (2) the “w/o subtitle” setting, which relies only on raw video inputs alongside the questions. Our study primarily focuses on the “w/o subtitle” format to enhance long video comprehension by leveraging video-based augmentation rather than textual cues from subtitles.

MLVU [76] is a benchmark designed to assess long video understanding across various tasks and video genres. It includes two types of questions: multiple-choice and freeform generation. The evaluation framework measures LMM performance in three key aspects: (1) holistic video understanding, which requires comprehending the entire video for global context; (2) single-detail video understanding, which focuses on recognizing key moments or short segments; and (3) multi-detail video understanding, which involves drawing connections between multiple short clips within the video. Our paper specifically reports accuracy scores for multiple-choice questions from the MLVU development set.

LongVideoBench [67] is a question-answering benchmark designed for interleaved long video-text input. It includes 3,763 videos and 6,678 human-annotated multiple-choice questions covering 17 fine-grained categories. The benchmark supports two evaluation formats: (1) the standard for-

mat, where video tokens are processed first, followed by the question descriptions, and (2) the interleaved video-text format, where subtitles are inserted between video frames. We evaluate all baseline models and our VAMBA using the standard input format. The reported results are based on the validation split.

NExT-QA [68] is a video question-answering benchmark designed to evaluate reasoning-based video understanding. It consists of 5,440 videos and approximately 52,000 human-annotated question-answer pairs, covering a diverse range of real-world activities. The dataset includes two types of question formats: multiple-choice questions and open-ended free-form questions. NExT-QA emphasizes causal and temporal reasoning, requiring models to understand event sequences, cause-effect relationships, and interactions within videos. The dataset is divided into three categories: causal, temporal, and descriptive questions. The dataset is split into training (3,870 videos), validation (570 videos), and test (1,000 videos), ensuring standardized benchmarking.

MVBench [40] is a comprehensive multimodal video understanding benchmark. The dataset introduces a novel static-to-dynamic task transformation, converting existing static image tasks into video-based challenges, assessing a model’s ability to perform both low-level perception and high-level cognitive reasoning over time. MVBench automatically converts annotations from 11 publicly available video datasets into unified multiple-choice question-answer pairs, covering diverse scenarios ranging from first-person to third-person perspectives and indoor to outdoor environments. Each question presents five answer choices, ensuring standardized evaluation through human-verified ground truth responses.

DREAM-1K [62] is a video description dataset designed for fine-grained event and motion understanding. It contains 1,000 short videos, each averaging 9 seconds, and covers a diverse set of real-world and cinematic scenarios. Unlike question-answering datasets, DREAM-1K requires models to generate detailed multi-sentence descriptions that capture all key actions, interactions, and temporal sequences within each video. The dataset includes videos from five different sources—live-action movies, animated films, stock footage, long YouTube videos, and shortform social media clips—ensuring broad coverage of visual styles. DREAM-1K prioritizes event-based reasoning, expecting models to understand sequential actions, motion cues, and interactions rather than just static descriptions. Evaluation is conducted using AutoDQ (Automatic Description Quality), which measures how well generated descriptions align with reference descriptions by comparing extracted events.

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Question: Briefly describe the video.

Vamba: The video features a small bird with brown and white feathers perched on a ledge, set against a backdrop of a cityscape with tall buildings under a clear blue sky with scattered clouds. The bird is positioned next to a yellow bucket attached to the ledge by a chain. As the scene progresses, a small gray rodent with a pink nose and white fur appears and starts to climb into the yellow bucket. The bird remains perched on the ledge, observing the rodent's actions. The rodent continues to climb into the bucket, eventually disappearing inside. The bird then looks around, seemingly curious about the rodent's disappearance. The video concludes with the bird still perched on the ledge, gazing out over the cityscape.

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Question: Briefly describe the video.

Vamba: The video features two children sitting on a beige couch in a cozy living room, engaging in a playful and interactive activity. The child on the left is dressed in a blue shirt with a colorful design and a blue headband, while the child on the right wears a blue shirt with a Nike logo. They are holding a large brown teddy bear and appear to be playing a game of rock-paper-

scissors. The background includes a staircase with wooden steps and a white door, adding to the homely atmosphere. The

children are using hand gestures and facial expressions to emphasize their points, and their interaction is lively and engaging. The scene continues with the children still deeply involved in their game, maintaining their energetic and playful demeanor. The video captures the essence of their playful interaction, highlighting their enthusiasm and engagement in the activity.

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Question: Briefly describe the video.

Vamba: The man is struggling to climb out of the icy pool, using his hands and a pickaxe to grip the icy surface.

Figure 6. Additional qualitative results for VAMBA.

### 9. Comparison with Contemporary Work

Several contemporary works also investigate hybrid Mamba-Transformer models for long video understanding. For example, STORM [31] and BIMBA [28] utilize Mamba blocks between the vision encoder and LMM decoder as additional processing and compression modules for video tokens, achieving high performance in long video understanding. However, different from VAMBA, the overall architecture of the LMM remains unchanged in these methods, with the decoder still relying on full causal self-attention layers for both text and video tokens. As a result, the model architectures proposed in these methods offer limited gains in training and inference efficiency, with any speedup in video processing still primarily attributed to token reduction. In comparison, VAMBA directly employs Mamba-2 layers in the LMM decoder and bypasses the self-attention updates

for video tokens. This design enables highly efficient video processing even without reducing the number of tokens.

Table 6. Quantitative results for VAMBA with token reduction.

Models GPU Mem (MB) LVBench VideoMME MLVU

VAMBA 45791 42.4 57.4 65.9 VAMBA-TR 33847 41.6 56.9 66.5

### 10. Combining VAMBA and Token Reduction

As mentioned in the paper, we expect VAMBA to be compatible with token reduction, and combining VAMBA and token reduction can potentially result in similar performance and even higher efficiency. We provide some preliminary results for combining VAMBA and token reduction in this section. As shown in Table 6, we can simply uniformly drop 50% of

the video tokens during inference (denoted as VAMBA-TR) and achieve little performance drop across multiple benchmarks with better efficiency. We believe finetuning VAMBA with token reduction can further preserve its capacity and leave this as a future work.

### 11. Additional Qualitative Results

In this section, we showcase more qualitative results from our VAMBA for detailed video captioning and video event understanding. The results are shown in Figure 6.

