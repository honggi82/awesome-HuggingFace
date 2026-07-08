# arXiv:2504.10068v2[cs.CV]27Nov2025

## Mavors: Multi-granularity Video Representation for Multimodal Large Language Model

Yang Shi1,2*† Jiaheng Liu3* Yushuo Guan2* Zhenhua Wu2 Yuanxing Zhang2‡ Zihao Wang2 Weihong Lin2 Jingyun Hua2 Zekun Wang2 Xinlong Chen4 Bohan Zeng1 Wentao Zhang1 Fuzheng Zhang2 Wenjing Yang Di Zhang2 1Peking University 2Kling Team 3Nanjing University 4CASIA

https://mavors-mllm.github.io/

### Abstract

Long-context video understanding in multimodal large language models (MLLMs) faces a critical challenge: balancing computational efficiency with the retention of finegrained spatio-temporal patterns. Existing approaches (e.g., sparse sampling, dense sampling with low resolution, and token compression) suffer from significant information loss in temporal dynamics, spatial details, or subtle interactions, particularly in videos with complex motion or varying resolutions. To address this, we propose Mavors, a novel framework that introduces Multi-granularity video representation for holistic long-video modeling. Specifically, Mavors directly encodes raw video content into latent representations through two core components: 1) an Intra-chunk Vision Encoder (IVE) that preserves highresolution spatial features via 3D convolutions and Vision Transformers, and 2) an Inter-chunk Feature Aggregator (IFA) that establishes temporal coherence across chunks using transformer-based dependency modeling with chunklevel rotary position encodings. Moreover, the framework unifies image and video understanding by treating images as single-frame videos via sub-image decomposition. Experiments across diverse benchmarks demonstrate Mavors’ superiority in maintaining both spatial fidelity and temporal continuity, significantly outperforming existing methods in tasks requiring fine-grained spatio-temporal reasoning.

### 1. Introduction

Long-context video modeling stands as one of the most crucial capabilities within MLLMs [6, 47, 67, 116]. This capability empowers MLLMs to proficiently manage hours-long

*Equal contribution. †Work done during an internship at Kling Team. ‡Corresponding author.

movies, documentaries, and online video streams, all of which demand sophisticated long video processing. Recent advances in MLLMs perform well in short video understanding. However, it remains challenging to build MLLMs for processing extremely long videos (lasting for hours or even longer). The difficulty lies in how to enable MLLMs to efficiently understand the extremely long video context brought by long videos.

As shown in Figure 1, we have compared three mainstream types of video MLLMs with our method, and provided the video caption results of different methods for better illustration. Specifically, in Figure 1(a), these methods (e.g., LLaVA-Video [124], InternVL 2.5 [14]) usually employ the sparse sampling strategy to decrease the number of frames and reduce the computation costs. However, these methods have a significant limitation, where many temporal contexts are lost as many frames are not sampled. Thus, the performance results of video-related tasks, which require detailed temporal contexts from many frames, are degraded a lot for these methods. When compared to methods in Figure 1(a), some methods (e.g., Oryx [60], Qwen2VL [98]) have introduced the strategy of dense sampling with low-resolution input in Figure 1(b). However, for these methods, many spatial contexts are lost as only the low-resolution frames are given, which also significantly degrade the results of video-related tasks requiring detailed spatial contexts, e.g., video captioning. Recently, in Figure 1(c), several works (e.g., VideoLLaMA 3 [116], VideoChat-Flash [47]) have proposed token compression strategies (e.g., token merge or token dropping), which reduces tokens based on vector or pixel similarity and effectively preserves spatial-temporal features of large visual elements. However, token compression inevitably leads to the loss of information regarding small spatial objects, subtle temporal motions, and interactions among multiple objects, thereby posing challenges for understanding complex

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

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

… …

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

…

(c) Dense Sampling with Token Compression

(d) Dense Sampling with Chunk Modeling

(b) Dense Sampling with Low Resolution

(a) Sparse Sampling

###### LLaVA-Video-7B Qwen2.5-VL-7B VideoChat-Flash-7B Mavors-7B (Ours)

Caption: Two men push up the glass outside the red house and then high-five each other. A woman appears inside the house and smashes the window glass with a hammer. The two men outside the house become surprised.

Caption: A man climbs up a ladder outside the Red House and looks in the window. A woman appears and breaks the glass with a chandelier. Then another man appears outside the house and starts talking to the previous man.

Caption: Two people are standing next to a red house, the one on the left is pointing at the one on the right. A man walks into the shot, punches the window, and then runs away.

Caption: Two male characters are talking outside a window. A woman can be seen inside the window. The wind ow glass then shatters.

Video-MME score: 65.1 Video-MME score: 65.3 Video-MME score: 65.0

Video-MME score: 63.5

Figure 1. (a) Sparse sampling, which remains the high resolution but loses many details in the unsampled frames; (b) Dense sampling with low resolution, which understands the videos from a large number of frames but would confuse on the low-resolution content; (c) Dense sampling with token compression, which keeps the key tokens on the main characters but suffers from hallucinations owing to the missing of visual tokens; (d) Our Mavors, balancing the demands of resolution and number of frames. Though all these approaches could perform similarly on Video-MME, Mavors significantly improves the caption capability on complex scenes. Note that the words in red and green denote incorrect and correct details, respectively.

scenes.

which includes the modality alignment, temporal understanding enhancement, instruction tuning and DPO training stages.

Therefore, the fundamental problem of video understanding is that existing methods often rely on sparse sampling or token compression strategies and struggle to balance computational efficiency with the retention of fine-grained spatio-temporal patterns, particularly in videos with variable motion, aspect ratios, or resolutions.

The contributions of Mavors are shown as follows:

- • We propose the Mavors by utilizing the Multigranularity video representation for multimodal large language model, which aims to better preserve the spatiotemporal contexts based on dense sampling with chunk modeling.
- • Mavors includes two modules: Intra-chunk Vision Encoder (IVE) and Inter-chunk Feature Aggregator (IFA). IFA encodes consecutive video chunks into latent representation based on 3D convolutions and ViT, and IFA builds the temporal coherence based on the temporal transformer and chunk-level rotary-encoding strategies.
- • Comprehensive experimental results and detailed analysis show the effectiveness and efficiency of Mavors.

To address this problem, as shown in Figure 1(d), we introduce the Mavors method to extract the Multi-granularity video representation for MLLMs. which is designed to process raw video content holistically while preserving both spatial fidelity and temporal coherence. Specifically, Mavors eliminates the information loss inherent in conventional frame sampling or token compression methods by directly encoding consecutive video chunks into latent representations. This approach leverages a two-tier architecture: an Intra-chunk Vision Encoder (IVE) extracts highresolution spatial features from localized video segments using 3D convolutions and Vision Transformer (ViT) layers, while an Inter-chunk Feature Aggregator (IFA) employs temporal transformer and chunk-level rotary position embeddings (C-RoPE) to model temporal dependencies across chunks. Besides, Mavors further unifies image and video understanding by treating images as single-frame videos by employing a sub-image divide-and-conquer approach for image processing. Moreover, following the common training strategy, we also adopt a multi-stage training paradigm,

### 2. Related Works

#### 2.1. MLLM Architecture

Current MLLMs employ two architectural strategies for visual processing. The first paradigm is based on crossattention approach, which maintains frozen model parameters while establishing dynamic visual-language interactions through attention mechanisms [2] . Alternatively, the second paradigm processes visual content through pretrained encoders (CLIP [76], SigLIP [115]) before con-

catenating image tokens with text embeddings for unified language model processing [43, 51, 53–55]. The second paradigm can be readily extensible to video analysis through sequential frame processing [45, 116], and many architectural innovations for temporal modeling have been proposed [34, 56, 103].

#### 2.2. MLLM for Video Understanding

Existing MLLMs have revealed divergent capabilities in temporal comprehension across different video durations. While existing systems demonstrate proficiency in minutescale video analysis [45, 47, 50], emerging efforts targeting hour-level sequences [23, 101] face fundamental challenges. To address the challenges of long video modeling, current approaches primarily pursue two optimization directions: (1) context window expansion for large language models [23, 101, 108, 120] and (2) efficient token compression via spatial-temporal feature distillation [20, 49, 85, 86, 90, 104]. For the first strategy, though theoretically enabling long-sequence processing, suffers from impractical computational overhead, which bring significant challenges for practical applications. In contrast, recent token compression methods like LLaMA-VID [49] achieve compression rates at the cost of discarding subtle details, which results in performance degradation on standard video understanding benchmarks. When compared to the existing works, our Mavors can directly process the raw videos to maintain spatial and temporal details well with acceptable computation costs.

### 3. Method 3.1. Preliminaries

Necessity of Dense Sampling with High Resolution. As shown in Figure 2 and Figure 3, we have compared the results of two popular video MLLMs (i.e., Qwen2.5VL-7B [4] and Oryx-1.5-7B [60]) on two representative benchmarks (i.e., Video-MME [22] and DREAM-1K [96]). Specifically, the Video-MME focuses on multiple-choice question answering based on video content and requires a better understanding of the temporal relations between different frames. DREAM-1K involves open-ended video captioning, where models must generate detailed descriptions of the main events in the video. Thus, both the spatial and temporal fine-grained details are important. In Figure 2, we observe that performance increases a lot when increasing the number of frames, which shows the necessity of dense sampling with more frames. In Figure 3, performance results on Video-MME are relatively stable for both MLLMs. For this phenomenon, we assume that understanding fine spatial details is not vital for Video-MME. In contrast, the results on DREAM-1K increase a lot, which demonstrates the necessity of high resolution.

65

33

Qwen2.5-VL-7B

Oryx-1.5-7B

62

31

Score

Score

59

29

56

Qwen2.5-VL-7B

Oryx-1.5-7B

27

53

8 16 32 64

4 8 16 32

Number of frames

Number of frames

(a) Video-MME

(b) Dream1K

Figure 2. The impact of the number of frames (720P).

65

- 29

- 30

- 31

- 32

- 33

Qwen2.5-VL-7B

Oryx-1.5-7B

62

Score

Score

59

56

Qwen2.5-VL-7B

Oryx-1.5-7B

53

224 336 448 560

224 336 448 560

Resolution of frames

Resolution of frames

(a) Video-MME

(b) Dream1K

Figure 3. The impact of the resolution of frames (64 frames).

In summary, as real-world video understanding tasks usually rely on understanding the fine-grained spatiotemporal contexts well, it is important to design video MLLMs by sampling dense and high-resolution frames and maintaining efficiency.

#### 3.2. Overview of Mavors

In Figure 4, the key objective of Mavors is to enhance the video understanding capability by introducing an efficient video encoding strategy based on dense sampling with high resolution strategy.

Specifically, Mavors employs a video encoder that directly processes pixel information from video chunks, converting them into latent representations. Figure 4 illustrates the overview of Mavors when dealing with video content and images. We consider an input video SV ∈ RW

I×HI×3, where WV,HV and WI,HI denote the respective widths and heights, and TV denotes the total number of video frames. Mavors follows the auto-regressive architecture to generate a textual response based on a given textual instruction. Specifically, in Mavors, we first perform the preprocessing on the raw videos or images to obtain the model input. Then, we employ an intra-chunk vision encoder and an inter-chunk feature aggregator to fully comprehend videos, so that the spatial and temporal details would be remained. Following the mainstream architecture of MLLMs, the temporally integrated features are passed through an MLP projector for modality alignment before being input to the LLM.

V×HV×3×TV or an image SI ∈ RW

#### 3.3. Intra-chunk Vision Encoder

Mavors partitions the video frames into cV = ⌈T

F ⌉ video chunks, where each chunk contains F consecutive frames describing the dynamic scenes and temporal events, i.e.,

V

…

LLM

…

MLP Projector

###### Inter-chunk Feature Aggregator

Temporal Transformer

Pooling

ViT

###### Intra-chunk Video Encoder

C-RoPE

3D Conv

… … … … …

Video Chunks

Image Chunks

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

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

…

0.0s 0.5s 5.0s 0.0s 0.0s

Visual Tokens Textual Tokens Output Tokens

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

Thumbnail and Sub-images

Raw Video Raw Image

Video Preprocessing Image Preprocessing

Figure 4. The architecture of Mavors.

= Partition(SV). Intra-chunk vision encoder is designed to represent the vision features of the video content. It begins with 3D convolutions applied to individual video chunks, and we would obtain the visual feature Fi for the i-th chunk as follows:

C1,...,c

V

V×dV,i = 1,...,cV, (1)

Fi = Conv(Ci)/F ∈ Rn

where nV indicates the number of visual features per video chunk, and dV denotes the dimension of the visual features. We then adopt a standard ViT with parameter θViT to capture high-level spatial-temporal features, denoted as Hˆi, within the i-th chunk. To manage the computational load and complexity for the downstream LLM module arising from a large number of tokens, we apply a 2x2 pooling layer on Hˆi to obtain Hi ∈ Rn

V/4×dV.

We initialize θViT by SigLIP weights. Specifically, the 2D convolutional kernels from SigLIP are replicated F times along the temporal dimension to form the 3D kernels. As the resulting visual features are divided by F in Eqn. (1), the spatial absolute position embedding is added to the feature vectors towards the corresponding pixel patches. This ensures that the model’s initial behavior precisely matches its capability for single image-text understanding.

#### 3.4. Inter-chunk Feature Aggregator

The intra-chunk vision encoder mainly captures the highlevel visual features within video chunks. Mavors leverages the the inter-chunk feature aggregator, to integrate temporal information across the multiple video chunks of the com-

plete video. First, we concatenate the high-level visual features to form the original feature sequence as follows:

χ(0) = Concat(H1,...,cV

). (2)

Inter-chunk feature aggregator consists of Linter Transformer layers with Causal Attention. To identify the sequential order of the visual features, we propose chunklevel Rotary Encoding (C-RoPE) to the Transformer layers, so that the temporal information can be correctly retained. Specifically, the causal scaled dot product (SDP) attention in the j-th Transformer layer would be calculated by

Q(Interj) ,KInter(j) ,VInter(j) = Linear(χ(j−1)), (3)

4ι′ nV ⌉)

4ι nV ⌉,⌈

SDP(qι(j),kι(′j)) = C-RoPE(qι(j),kι(′j);⌈

⌋kι(′j)⊺, ∀qι(j) ∈ Q(Interj) ,kι(′j) ∈ KInter(j)

(4)

= qι(j)R⌊ 4ι

⌋−⌊ 4nι′

nV

V

Here, R represents the rotation matrix. In practice, we would transcode the video into fixed FPS, so that the index of the video chunk can be identified from the actual timestamp of the first frame of the chunk. In the remaining process of the Transformer layer, we follow

µj = softmax(SDP(Q(Interj) ,KInter(j) )), (5)

χ(j) = µjVInter(j) . (6) We then feed χ(L

Inter) to the MLP projector to obtain the visual tokens, where the feature dimension of these visual

Image Preprocessing. As shown in Figure 5, Mavors first partitions the raw image into several sub-images, and then leverages the thumbnail of the original image and all subimages into the vision encoder. Besides, Mavors incorporates a special design in the feature aggregator to accommodate the joint training of videos and images. The details are as follows.

##### tokens is the same as the feature dimension of textual tokens in LLM.

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

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

4. Duplicate sub-images to image chunks.

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

First, as image understanding tasks often require spatial details, we follow the image partition method in [110] and support dynamic resolution for processing high-resolution images, where the raw image will be partitioned into multiple sub-images and the size of these sub-images is supposed to match the number of pixels in the ViT’s pretraining. Specifically, we first determine the ideal number of sub-images Ns = (WI × HI)/Rv2 , where (WI,HI) is the resolution of the original raw image and (Rv,Rv) is the resolution of the ViT’s pretraining images. Next, we identify potential partition configurations by finding pairs of integers (m,n), representing the number of columns and rows, respectively, such that their product equals the target number of slices Ns. These pairs form the set CNs

1. Thumbnail with raw aspect ratio.

3. Flatten the sub-images.

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

2. Best partition (𝑚∗,𝑛∗)

Partitions: (𝑚∗ = 3,𝑛∗ = 2)

Raw image (shape: [𝑊 , 𝐻 ])

Figure 5. The dynamic resolution strategy in Mavors.

#### 3.5. Preprocessing

Video Preprocessing. The video processing strategy of Mavors varies based on the video length. Specifically, videos with short lengths are directly processed into chunks. To accommodate long videos, we employ an initial step of accelerated playback achieved through frame dropping, thereby reducing the total frame count to be compatible with Mavors processing limits. Specifically, the position IDs utilized by C-RoPE correspond to timestamps derived from the original, non-accelerated video timeline. This mechanism informs the model that the processed frames are not temporally contiguous. While alternative strategies for very long video comprehension exist, e.g., in-video RetrievalAugmented Generation (RAG) [65], they represent an orthogonal direction to Mavors.

= {(m,n)|m × n = Ns,m,n ∈ Z}. Then, we select the best configuration (m∗,n∗) from C˜ = CNs−1 ∪ CNs ∪ CNs+1 based on the following criteria:

- m

- n

WI HI − log

(m∗,n∗) = arg min

. (8)

log

(m,n)∈C˜

We will leverage the thumbnail of the original raw image I0 and all sub-images I1,...,Im∗×n∗ as the input of the vision encoder. Before feeding into the vision encoder, we will rescale the original image and the sub-images, which have more pixels than the ViT’s pretraining images. We use the same dynamic resolution strategy as video processing.

Second, when compared to video processing, the feature aggregator operates on the features extracted from each subimage independently, thus avoiding redundant temporal relationships. Furthermore, given that the model must process both images and videos, the representation of an image (treated as a single frame) is replicated across all temporal positions within the input sequence. Placing the image representation at only a single temporal position would cause the model parameters to become biased towards that static position, ultimately hindering the model’s capacity to perceive temporal information effectively in video sequences.

Meanwhile, Mavors could process videos with arbitrary resolutions and aspect ratios. Specifically, Mavors employs a dynamic resolution strategy to maintain the original aspect ratio of the video frames, avoiding distortion artifacts that can arise from fixed-shape resizing. The resized video frames roughly keep the original aspect ratio and match the number of pixels in the ViT’s pretraining images. For example, given the frames with the (WV,HV) resolution and the ViT’s pretrained image resolution (Rv,Rv), Mavors will rescale the frames into the resolution of (Rv ∗

WV/HV,Rv ∗ HV/WV). We also resize the positional embedding of patches, following SigLIP [115]. Specifically, the positional embedding of the video chunk in the (x,y) position, denoted as E(x,y), will be formulated as:

### 4. Training Paradigm

In Figure 6, multi-stage training is adopted, serving to improve the collaboration of the video encoder and LLM and the performance of multimodal tasks. Given SigLIP’s robust image understanding performance, we forgo an independent CLIP training phase to avoid redundancy. Instead, we adopt a tailored initialization strategy to ensure compatibility with both video and image inputs, where the 2D

E(x,y) = Ev(x ∗ Pv/PW,y ∗ (Pv/PH)), (7)

where (PW,PH) is the number of patches in the video chunk. Pv and Ev(x,y) are the number of patches and the positional embedding in the ViT’s pretraining images, respectively.

[Figure 123]

Figure 6. Training paradigm of different stages.

convolutional kernels from SigLIP are replicated F times along the temporal dimension to form the 3D kernels. Then, we leverage multiple training stages to progressively build a vision encoder that maintains image understanding while effectively encoding spatio-temporal information of videos. The data used for training Mavors is detailed in Appendix A.

- Stage 1: Modality Alignment. As SigLIP’s training involved alignment with the T5 model [78], the first stage aims to align the semantic space of the vision encoder with the LLM’s semantic space. In this stage, we train the inter-chunk feature aggregator and the MLP projector, while keeping the LLM and the intra-chunk vision encoder frozen. Although the model exhibits only coarse video comprehension at this stage, the principal aim is to achieve modality alignment and instill basic temporal understanding. Therefore, we prioritize diverse, generalconcept image-text pairs and short video-text pairs with low complexity (e.g., LAION [81] and PANDA-70M[12]), thereby avoiding excessively difficult data that could impede the development of foundational abilities. Stage 1.5: Temporal Understanding Enhancement. Subsequent to Stage 1, we implement Stage 1.5, which focuses on enhancing the video encoder’s capacity for genuine video comprehension. Based on the modality alignment from Stage 1, parameter updates are performed on all components excluding the LLM. For data selection in this stage, we augment the initial dataset with standard computer vision (CV) tasks applied to images and short video chunks, such as captioning, classification, OCR, interleaved image-text, and perception QA.
- Stage 2: Multitask Instruction Tuning. In Stage 2, the primary objective is to adapt the model for a range of multimodal tasks, leveraging data formats including text-only, single-image, multi-images, and complex video. Beyond standard CV tasks, we incorporate grounding tasks and temporal grounding tasks to enhance the model’s perception of spatio-temporal details. Similar to the practice in Qwen2.5VL [4], we find that representing bounding boxes

using plain text coordinates yields performance comparable to using special tokens; consequently, we adopt the plain text representation. This stage also activates the sub-image partitioning paradigm to enhance the model’s image understanding capabilities. All model parameters are unfrozen and trained on a large dataset, allowing for extensive selfadjustment. Upon completion, the model possesses significant world knowledge, semantic understanding, and logical reasoning abilities, though its application is initially limited by the specific tasks and query formats encountered. Therefore, towards the end of this stage, we introduce more diverse data types, covering a broader spectrum of real-world task scenarios and textual query formulations.

Stage 3: DPO Training. Our empirical evaluations reveal that while the previously described training procedure yields strong leaderboard performance, the resulting model exhibits distinct patterns. Specifically, for QA tasks, the model tends to generate overly concise responses, likely due to extensive training on multiple-choice or short-answer datasets. Conversely, for descriptive tasks, the model fails to terminate generation appropriately. To mitigate these issues, we incorporate a Direct Preference Optimization (DPO) [77] stage following Stage 2. The preference dataset mainly covers three domains: open-ended QA, image captioning, and video captioning. More details can be found in Appendix A.

Loss Function. We employ the next-token-prediction (NTP) training methodology in all training stages except the DPO stage. During DPO training, we employ the standard DPO loss.

### 5. Experiments

#### 5.1. Experimental Setup

Implementation Details. The Mavors model utilizes Qwen2.5-7B as its language model module, with the intrachunk vision encoder initialized using SigLIP weights. To balance effectiveness and efficiency, the frame count per video chunk, F, is set to 16. The inter-chunk feature ag-

###### Model Size MMWorld PerceptionTest Video-MME MLVU MVBench EventHallusion TempCompass VinoGround DREAM-1K

GPT-4o-20240806 - 62.5 - 71.9 64.6 64.6 92.0 73.8 38.9 39.2 Gemini-1.5-Pro - - - 75.0 - 60.5 80.3 67.1 22.9 36.2

LLaVA-OneVision 7B 59.2 56.9 58.9 64.8 56.7 64.3 61.4 26.2 31.9 InternVL 2.5 8B 62.2 65.0 64.3 67.0 72.0 64.1 71.4 24.0 29.7 NVILA 8B 55.2 55.5 64.2 70.1 68.1 69.9 66.5 20.2 26.9 LLaVA-Video 7B 60.1 67.5 63.6 67.2 58.6 70.7 65.7 26.9 33.3 Oryx-1.5 7B 58.8 70.3 59.0 63.8 67.5 61.3 60.2 22.3 32.5 Qwen2.5-VL 7B 61.3 66.2 65.1 70.2 69.6 66.5 71.4 34.6 32.6 VideoLLaMA3 7B 56.4 72.8 66.2 73.0 69.7 63.4 68.1 31.3 30.5 VideoChat-Flash 7B 57.9 74.7 65.3 74.7 74.0 66.4 70.0 33.3 29.5 Slow-fast MLLM 7B 58.2 69.7 60.2 60.4 68.9 67.4 69.9 27.1 33.2

Qwen2.5-VL 72B 73.1 73.2 73.3 76.6 70.4 76.3 79.1 58.6 35.1 InternVL 2.5 78B 77.2 73.5 72.1 76.6 76.4 67.7 75.5 38.7 30.3

Mavors (Ours) 7B 68.1 70.3 65.0 69.8 68.0 73.5 77.4 36.9 39.4

Table 1. Performance on video benchmarks. Most of the scores are from their original studies. The others are reproduced following the official benchmark recommendation.

gregator consists of LInter=3 layers. The training is conducted on 416 GPUs. Given the model’s moderate size, we employed DeepSpeed with ZeRO stage 2 optimization. As mentioned in Section 4, the pre-training proceeded in three stages: Stage 1 used approximately 127 million samples with a global batch size of 6,656, taking 71 hours; Stage 1.5 used 52 million samples with a global batch size of 3,328, taking 177 hours; and Stage 2 used 19 million samples with a global batch size of 1,664, requiring 28 hours. The learning rates for the LLM and projector are set to 1e-5 in both Stage 1 and Stage 1.5, with a constant learning rate schedule applied during these phases. In Stage 2 and DPO, the learning rate was initialized at the same value (1e-5) as the preceding stages but followed a cosine decay schedule, gradually reducing to 1/10th of its initial value. Meanwhile, the learning rates for the inter-chunk feature aggregator and intra-chunk vision encoder remained fixed at 1/10th of the LLM’s learning rate across all training stages.

For inference, Mavors is adapted using the vLLM framework [38]. Since Mavors requires comprehensive video encoding and frame preprocessing occurs on the CPU, the CPU processor can thus become a bottleneck. Recognizing that the intra-chunk vision encoder’s computation is a one-time GPU operation per video, with results stored in the LLM’s KV cache, we overlaps the pipeline. Specifically, the intra-chunk vision encoder and inter-chunk feature aggregator execute directly on the GPU, while the language model component leverages vLLM. This separation can effectively balance CPU-bound preprocessing, compute-intensive visual encoding (Intra/Inter), and language model inference. More details of the inference efficiency can be found in Appendix B.

Baseline Models. We select several representative video models for performance comparison. We include GPT-4o20240806 [32] and Gemini-1.5-Pro-002 [23] as the closedsource APIs baselines. Standard auto-regressive models using resolution-preserving frame sampling are represented by LLaVA-OneVision [43] and InternVL 2.5 [14].

For video understanding tasks, we add models based on: (a) high-performing sparse frame sampling (NVILA [61], LLaVA-Video [124]); (b) dense sampling with lower resolution (Qwen2.5-VL [4], Oryx-1.5 [60]); (c) dense sampling with token compression (VideoChat-Flash [47], VideoLLaMA3 [116]); and (d) slow-fast architecture, a special frame sampling strategy (Slow-fast MLLM [84]). Regarding image tasks, as some video-centric models either lack image input (e.g., VideoChat-Flash) or are not SOTA on image tasks, we include four strong models on QA/Caption benchmarks: GLM-4V [99], Qwen2.5-VL, DeepSeek-VL2 [105] and CogVLM2 [29]. Crucially, aside from prompt modifications, no benchmark-specific hyperparameters (e.g., frame sampling, resolution) were tuned during evaluation for any model, including Mavors.

Benchmarks. Video understanding capabilities are assessed across general knowledge QA (MMWorld [28], PerceptionTest [74]), long-video QA (Video-MME [22], MLVU [126]), event understanding QA (MVBench [46], EventHallusion [117]), temporal understanding QA (TempCompass [58], VinoGround [118]), and captioning (DREAM-1K [96]). Image understanding evaluation includes comprehensive capabilities (MMMU [114]), cognitive understanding (MathVista [62], AI2D [37]), and captioning (CapsBench [52]). More experiment details can be found in Appendix C.

#### 5.2. Main Results

Video Understanding. Table 1 presents a performance comparison of Mavors against baseline models on various video benchmarks. Approaches employing dense frame sampling with lower resolution demonstrate strong performance on long video QA by incorporating extensive temporal information, but exhibit limitations in understanding spatial details for knowledge-intensive and captioning tasks. token compression strategies show a similar pattern, yielding excellent scores on long video QA due to abundant temporal cues, but their merging of non-primary

###### Model Size MMMU MathVista AI2D CapsBench

GPT-4o-20240806 - 69.9 62.9 84.7 67.3 Gemini-1.5-Pro - 60.6 58.3 79.1 71.2

CogVLM2 8B 42.6 38.7 73.4 50.9 GLM-4V 9B 46.9 52.2 71.2 61.0 LLaVA-OneVision 7B 47.9 62.6 82.4 57.4 InternVL 2.5 8B 56.2 64.5 84.6 66.5 Qwen2.5-VL 7B 58.0 68.1 84.3 64.9

DeepSeek-VL2 27B 54.0 63.9 83.8 61.3 Qwen2.5-VL 72B 68.2 74.2 88.5 70.1 InternVL 2.5 78B 70.0 70.6 89.1 68.5

Mavors (Ours) 7B 53.2 69.2 84.3 75.2

Table 2. Performance on image benchmarks.

tokens compromises the comprehension of environmental context, resulting in marked deficiencies, especially in captioning. In contrast, sparse frame sampling approaches, which inherently lose temporal detail and consequently perform less effectively on event understanding QA. Mavors’s multi-granularity video understanding framework successfully balances these trade-offs. Leveraging efficient visual information compression, Mavors delivers performance on long video QA nearly on par with dense sampling and token compression techniques, while preserving robust capabilities for knowledge-based and temporal reasoning tasks, eliminating the need for dataset-specific hyperparameter tuning. The substantial gains observed for Mavors in captioning highlight the effectiveness in achieving accurate and comprehensive understanding of entire video events.

Image Understanding. Table 2 compares Mavors’s performance against baseline models on image benchmarks. Mavors achieves performance on par with similarly-sized image understanding models in Image QA. Its captioning performance is particularly strong, surpassing even 72B models. This effectiveness is partly due to Mavors’s architecture: images and videos offer complementary visual perception within the intra-chunk vision encoder, yet are processed without mutual interference by the inter-chunk feature aggregator.

#### 5.3. Ablation Studies

We conduct a series of ablation studies to validate our model design. Given the extensive training time required for the full training paradigm, these ablations utilize standard compositive datasets and train various versions up to the completion of Stage 2. Specifically, Stage 1 employs LLaVA-Pretrain-558K [53] and LLaVAHound-Pretrain [122]; Stage 1.5 uses M4-Instruct [44] and ShareGPT4o [16]; and Stage 2 utilizes LLaVA-OneVision and LLaVA-Video. This approach reduces the duration of a full training cycle to under 24 hours with 64 GPUs. Performance is subsequently monitored using MMMU, MathVista, and CapsBench for image understanding capabilities, and Video-MME, Vinoground, and DREAM-1K for video

LInter MMMU MathVista CapsBench Video-MME VinoGround DREAM-1K

- 0 50.3 63.0 51.4 61.0 27.9 30.2
- 1 51.5 63.3 50.6 60.9 30.6 32.4 3 52.0 62.6 50.6 61.1 31.1 33.8 5 49.8 61.9 50.3 61.1 31.2 33.6

Table 3. Ablation on layers of Transformers in IFA.

RoPE MMMU MathVista CapsBench Video-MME VinoGround DREAM-1K

Standard 51.9 62.6 50.7 61.0 30.3 32.9 C-RoPE 52.0 62.6 50.6 61.1 31.1 33.8

(+0.1) (+0.0) (-0.1) (+0.1) (+0.8) (+0.9)

Table 4. Ablation on C-RoPE.

understanding capabilities.

Effect of the Number of Frames in a Video Chunk. We conduct experiments with four settings, varying a parameter F with values of 4, 8, 16, and 32. Upon the preliminary study evaluating video captioning performance on the validation set of KVQ [63], we observe that configurations with F = 8 or F = 16 yield more accurate and comprehensive captions. To ensure exposure to richer visual information, we finalize the F = 16 setting. We further evaluate these four model variants on six benchmark datasets in Figure 7. On image-based tasks, we observe a marginal improvement in performance metrics with increasing F. We hypothesize that this improvement stems from the model’s increased exposure to individual frames during video processing when F is larger, thereby enhancing its image understanding capabilities. Conversely, for video understanding tasks, performance degrades significantly for F = 4 due to insufficient temporal information and for F = 32, likely due to excessive information compression.

Effect of the IFA Module. We establish two baseline models for comparison in Table 3. The first baseline completely removes the inter-chunk feature aggregator

- (LInter=0), where the output from the IVE module is passed directly through a projector and then concatenated with the LLM’s input sequence. In this setup, the integration of temporal and spatial information relies solely on the LLM. The second baseline utilizes only a single Transformer layer
- (LInter=1) for the aggregator, thereby reducing its computational complexity. In Table 3, on image evaluation tasks,

removing the Transformer (LInter=0) shows a slight advantage, potentially due to the lower parameter count facilitating faster convergence on static perception tasks. However, for video evaluation, we observe that a deeper inter-chunk feature aggregator (LInter=3) enhances the model’s understanding, leading to better scores, although with diminishing marginal returns. Considering model complexity and convergence difficulty, LInter=3 should be an efficient configuration of Mavors.

Effect of C-RoPE. To assess the performance of C-RoPE, we replace it with the standard RoPE implementation and monitor changes in the Mavors model’s visual understand-

60

F=4 F=8 F=16 F=32

| |
|---|

| |
|---|

Score

40

| |
|---|

20

0

MMMU MathVista CapsBenchVideo-MMEVinoGroundDREAM-1K

Benchmarks

Figure 7. Performance with different numbers of frames in a video chunk.

65

Video-MME

Score

50

MLVU

DREAM-1K

35

20

0% 15% 30% 45% 60% 75%

Compression ratio

Figure 8. Performance with different token compression ratios.

Stage1

8

Start Stage1.5 (19.1k steps)

Start Stage2 (34.7k steps)

- Stage1.5

- Stage2

Lossvalue

6

4

2

0

0k 4k 8k 12k 16k 20k 24k 28k 32k 36k 40k 44k

Training steps

Figure 9. The dynamic of training losses across different stages for Mavors.

ing performance. Table 4 shows the performance across six metrics. For image understanding, given that the IFA architecture processes sub-images independently, both RoPE variants perform comparably. Conversely, for video understanding, C-RoPE outperforms standard RoPE by an average of 0.6 points. It indicates that standard RoPE suffers from differentiating intra-chunk from inter-chunk tokens and may hinder temporal sequence modeling. These findings demonstrate the efficacy and importance of C-RoPE within the IFA architecture.

#### 5.4. Further Analysis

Analysis on the Ratios of Token Compression. We apply token compression techniques within Mavors to decrease the number of tokens on each video chunk. Specifically, prior to the inter-chunk feature aggregator, we compute similarity between features at corresponding indices in adjacent chunks. Tokens exceeding a predefined similarity threshold are merged via averaging, retaining the positional ID from the earlier chunk. We vary thresholds to achieve different token reduction ratios, summarized in Figure 8. Results indicate that Mavors’ performance on video QA remains largely unaffected with token reductions up to 60%. Conversely, a significant performance degradation is observed for video captioning. This suggests that token compression on Mavors can be a feasible strategy for reducing inference costs in long-video QA applications. We provide two representative cases in Appendix F.

Stage MMMU CapsBench Video-MME DREAM-1K

- Stage 1 36.3 54.8 48.4 23.6 Stage 1.5 47.3 62.5 53.9 26.3
- Stage 2 53.0 73.4 65.0 38.9 DPO 53.2 75.2 65.0 39.2

Table 5. Results of different training stages.

Analysis on the Training Dynamics. Table 5 shows the results on the image QA dataset (MMMU), image caption dataset (CapsBench), video QA dataset (Video-MME) and video caption dataset (DREAM-1K) at different stages. The performance on all four datasets improves consistently across the three training stages (Stage 1, Stage 1.5, and

Stage 2), indicating that each stage contributes positively to the model’s ability to handle different tasks and modalities. The DPO stage provides further improvements. Note that we also provide the training loss curve of Mavors in

- Figure 9. Visualization. We pick a complex video cut from DREAM1K and present the captions generated by Qwen2.5VL-7B and Mavors-7B in Figure 10. Despite processing densely sampled frames, Qwen2.5VL-7B fails to capture many details (e.g., omitting the mention of a cow driving), leading to flawed inferences (words in red). In contrast, we observe that Mavors-7B predicts fine-grained and correct details (words in greed), which show the effect of our Mavors7B.

[Figure 124]

[Figure 125]

[Figure 126]

Qwen2.5 -VL-7B

|The video depicts a nighttime scene with a rural setting, featuring a dirt road winding through a landscape dotted with trees and wooden fences. the sky is dark, filled with stars, indicating it‘s late at night. In the foreground, there are two police cars with flashing blue and red lights, suggesting they are in pursuit or responding to an emergency. The vehicles are moving quickly along the road, kicking up dust as they drive.<br><br>In the background, another vehicle, possibly a pickup truck, is also on the road, but its details are less clear due to the distance and motion blur. The environment appears calm despite the urgency suggested by the police activity, with no visible signs of other people or additional vehicles.<br><br>The overall atmosphere is tense and dynamic, with the focus on the movement of the vehicles and the rural backdrop adding a sense of isolation and urgency to the scene.<br><br>Qwen2.5VL-7B: ... (36 trivial words) In the foreground, there are two police cars with flashing blue and red lights, suggesting they are in pursuit or responding to an emergency. The vehicles are moving<br><br>along the road, kicking up dust as they drive.|
|---|

Ours

|The video begins with a nighttime scene on a rural dirt road, illuminated by a spotlight from a police car. Several police cars with flashing lights are seen driving along the road, which is flanked by wooden fences and trees under a starry sky. The police cars continue to chase a blue car that is speeding ahead, kicking up dust as they go. The chase intensifies as the police cars maneuver around the blue car, which is trying to evade capture. The scene then transitions to the interior of the blue car, revealing a cow character driving. The cow appears to be talking or reacting to the chase,<br><br>with a determined expression on its face. The video captures the urgency and action of the police chase, culminating in a close-up of the cow driver, highlighting its role in the unfolding events.<br><br>In the background, another vehicle, possibly a pickup truck, is also on the road, but its details are less clear due to the distance and motion blur. The environment appears calm despite the urgency suggested by|
|---|

[Figure 127]

Q p p quickly I t b the police activity, with no visible signs of other people or additional vehicles. … (30 trivial words)

Mavors-7B: The video begins with a nighttime scene on a rural dirt road, illuminated by a spotlight from a police car. Several police cars with flashing lights are seen driving along the road, which is flanked

by wooden fences and trees under a starry sky. The police cars continue to chase a blue car that is speeding ahead, kicking up dust as they go. The chase intensifies as the police cars maneuver around the

blue car, which is trying to evade capture. The scene then transitions to the interior of the blue car, revealing a cow character driving. The cow appears to be talking or reacting to the chase, with a determined

expression on its face. The video captures the urgency and action of the police chase, culminating in a close-up of the cow driver, highlighting its role in the unfolding events.

- Figure 10. Comparison of generated video captions from Qwen2.5-VL-7B and Mavors-7B.

### 6. Conclusion

In this work, we present Mavors, a novel framework for holistic long-context video understanding in MLLMs.

Mavors introduces multi-granularity video representation based on Intra-chunk Vision Encoder (IVE) and Inter-chunk Feature Aggregator (IFA) to preserve both spatial details and temporal dynamics and maintain high efficiency. Extensive experiments on multiple benchmarks demonstrate the effectiveness and efficiency of our Mavors.

### References

- [1] Laion coco: 600m synthetic captions from laion2b-en. https://laion.ai/blog/laion-coco/. 1
- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L. Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Kar´en Simonyan. Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. 2
- [3] Anas Awadalla, Le Xue, Manli Shu, An Yan, Jun Wang, Senthil Purushwalkam, Sheng Shen, Hannah Lee, Oscar Lo, Jae Sung Park, et al. Blip3-kale: Knowledge augmented large-scale dense captions. arXiv preprint arXiv:2411.07461, 2024. 1
- [4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 3, 6, 7, 2
- [5] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1728–1738,

2021. 1

- [6] Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, and Sa˘gnak Ta¸sırlar. Introducing our multimodal models, 2023. 1
- [7] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/ kakaobrain/coyo-dataset, 2022. 1
- [8] Joao Carreira, Eric Noland, Chloe Hillier, and Andrew Zisserman. A short note on the kinetics-700 human action dataset. arXiv preprint arXiv:1907.06987, 2019. 1
- [9] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In IEEE Conference on Computer Vision and Pattern Recognition,

CVPR 2021, virtual, June 19-25, 2021, pages 3558–3568,

2021. 1

- [10] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions,

2023. 1

- [11] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, Li Yuan, Yu Qiao, Dahua Lin, Feng Zhao, and Jiaqi Wang. Sharegpt4video: Improving video understanding and generation with better captions. ArXiv preprint, abs/2406.04325, 2024. 1
- [12] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13320–13331, 2024. 6, 1
- [13] Xiaohui Chen, Satya Narayan Shukla, Mahmoud Azab, Aashu Singh, Qifan Wang, David Yang, ShengYun Peng, Hanchao Yu, Shen Yan, Xuewen Zhang, et al. Compcap: Improving multimodal large language models with composite captions. arXiv preprint arXiv:2412.05243, 2024.

- 1

[14] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and testtime scaling. arXiv preprint arXiv:2412.05271, 2024. 1, 7,

- 2

- [15] Chee Kheng Chng, Yuliang Liu, Yipeng Sun, Chun Chet Ng, Canjie Luo, Zihan Ni, ChuanMing Fang, Shuaitao Zhang, Junyu Han, Errui Ding, et al. Icdar2019 robust reading challenge on arbitrary-shaped text-rrc-art. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1571–1576. IEEE, 2019. 1
- [16] Erfei Cui, Yinan He, Zheng Ma, Zhe Chen, Hao Tian, Weiyun Wang, Kunchang Li, Yi Wang, Wenhai Wang, Xizhou Zhu, Lewei Lu, Tong Lu, Yali Wang, Limin Wang, Yu Qiao, and Jifeng Dai. Sharegpt-4o: Comprehensive multimodal annotations with gpt-4o, 2024. 8
- [17] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. 1
- [18] Debidatta Dwibedi, Yusuf Aytar, Jonathan Tompson, Pierre Sermanet, and Andrew Zisserman. Counting out time: Class agnostic video repetition counting in the wild. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 1
- [19] Dave Epstein, Boyuan Chen, and Carl. Vondrick. Oops! predicting unintentional action in video. arXiv preprint arXiv:1911.11206, 2019. 1
- [20] Jiajun Fei, Dian Li, Zhidong Deng, Zekun Wang, Gang Liu, and Hui Wang. Video-ccam: Enhancing video-language understanding with causal cross-attention masks for short and long videos. arXiv preprint arXiv:2408.14023, 2024. 3
- [21] Wikimedia Foundation. Wikimedia downloads. 1

- [22] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. ArXiv preprint, abs/2405.21075, 2024. 3, 7
- [23] Gemini Team. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. ArXiv preprint, abs/2403.05530, 2024. 3, 7, 2
- [24] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fr¨und, Peter Yianilos, Moritz Mueller-Freitag, Florian Hoppe, Christian Thurau, Ingo Bax, and Roland Memisevic. The ”something something” video database for learning and evaluating visual common sense. In IEEE International Conference on Computer Vision, ICCV 2017, Venice, Italy, October 22-29, 2017, pages 5843–5851, 2017. 1
- [25] Jiaxi Gu, Xiaojun Meng, Guansong Lu, Lu Hou, Niu Minzhe, Xiaodan Liang, Lewei Yao, Runhui Huang, Wei Zhang, Xin Jiang, et al. Wukong: A 100 million large-scale chinese cross-modal pre-training benchmark. Advances in Neural Information Processing Systems, 35:26418–26431,

2022. 1

- [26] Shuhao Gu, Jialing Zhang, Siyuan Zhou, Kevin Yu, Zhaohu Xing, Liangdong Wang, Zhou Cao, Jintao Jia, Zhuoyi Zhang, Yixuan Wang, et al. Infinity-mm: Scaling multimodal performance with large-scale and high-quality instruction data. arXiv preprint arXiv:2410.18558, 2024. 1
- [27] Mingfei Han, Linjie Yang, Xiaojun Chang, and Heng Wang. Shot2story20k: A new benchmark for comprehensive understanding of multi-shot videos, 2023. 1
- [28] Xuehai He, Weixi Feng, Kaizhi Zheng, Yujie Lu, Wanrong Zhu, Jiachen Li, Yue Fan, Jianfeng Wang, Linjie Li, Zhengyuan Yang, et al. Mmworld: Towards multidiscipline multi-faceted world model evaluation in videos. In The Thirteenth International Conference on Learning Representations. 7, 3
- [29] Wenyi Hong, Weihan Wang, Ming Ding, Wenmeng Yu, Qingsong Lv, Yan Wang, Yean Cheng, Shiyu Huang, Junhui Ji, Zhao Xue, et al. Cogvlm2: Visual language models for image and video understanding. arXiv preprint arXiv:2408.16500, 2024. 7, 2
- [30] Yu-Guan Hsieh, Cheng-Yu Hsieh, Shih-Ying Yeh, Louis B´ethune, Hadi Pour Ansari, Pavan Kumar Anasosalu Vasu, Chun-Liang Li, Ranjay Krishna, Oncel Tuzel, and Marco Cuturi. Graph-based captioning: Enhancing visual descriptions by interconnecting region captions. arXiv preprint arXiv:2407.06723, 2024. 1
- [31] Huazhang Hu, Sixun Dong, Yiqun Zhao, Dongze Lian, Zhengxin Li, and Shenghua Gao. Transrac: Encoding multi-scale temporal correlation with transformers for repetitive action counting. arXiv preprint arXiv:2204.01018, 2022. 1
- [32] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 7, 2

- [33] Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max W.F. Ku, Qian Liu, and Wenhu Chen. Mantis: Interleaved multiimage instruction tuning. Transactions on Machine Learning Research, 2024, 2024. 1
- [34] Jindong Jiang, Xiuyu Li, Zhijian Liu, Muyang Li, Guo Chen, Zhiqi Li, De-An Huang, Guilin Liu, Zhiding Yu, Kurt Keutzer, Sungjin Ahn, Jan Kautz, Hongxu Yin, Yao Lu, Song Han, and Wonmin Byeon. Token-efficient long video understanding for multimodal llms. 2025. 3
- [35] Xuan Ju, Yiming Gao, Zhaoyang Zhang, Ziyang Yuan, Xintao Wang, Ailing Zeng, Yu Xiong, Qiang Xu, and Ying Shan. Miradata: A large-scale video dataset with long durations and structured captions. Advances in Neural Information Processing Systems, 37:48955–48970, 2024. 1
- [36] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. ReferItGame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 787–798, 2014. 1
- [37] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In European conference on computer vision, pages 235–251. Springer, 2016. 7, 3
- [38] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023. 7
- [39] OMEGA Lab. Omega labs bittensor subnet: Multimodal dataset for agi research. 1
- [40] Hugo Laurenc¸on, Lucile Saulnier, L´eo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, and Victor Sanh. OBELICS: an open webscale filtered dataset of interleaved image-text documents. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. 1
- [41] Hugo Lauren¸con, Andr´es Marafioti, Victor Sanh, and L´eo Tronchon. Building and better understanding visionlanguage models: insights and future directions. In Workshop on Responsibly Building the Next Generation of Multimodal Foundational Models, 2024. 1
- [42] Hugo Laurenc¸on, L´eo Tronchon, and Victor Sanh. Unlocking the conversion of web screenshots into html code with the websight dataset, 2024. 1
- [43] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. ArXiv preprint, abs/2408.03326, 2024. 3, 7, 1, 2
- [44] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024. 8
- [45] Kunchang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu

- Qiao. Videochat: Chat-centric video understanding. ArXiv preprint, abs/2305.06355, 2023. 3, 1
- [46] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206, 2024. 7, 3
- [47] Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, et al. Videochat-flash: Hierarchical compression for long-context video modeling. arXiv preprint arXiv:2501.00574, 2024. 1, 3, 7, 2
- [48] Xiaotong Li, Fan Zhang, Haiwen Diao, Yueze Wang, Xinlong Wang, and LINGYU DUAN. Densefusion-1m: Merging vision experts for comprehensive multimodal perception. Advances in Neural Information Processing Systems, 37:18535–18556, 2024. 1
- [49] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. 2024. 3
- [50] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. ArXiv preprint, abs/2311.10122, 2023. 3
- [51] Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models, 2023. 3
- [52] Bingchen Liu, Ehsan Akhgari, Alexander Visheratin, Aleks Kamko, Linmiao Xu, Shivam Shrirao, Chase Lambert, Joao Souza, Suhail Doshi, and Daiqing Li. Playground v3: Improving text-to-image alignment with deep-fusion large language models. arXiv preprint arXiv:2409.10695, 2024.

- 7, 2, 3

[53] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023. 3,

- 8

- [54] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.
- [55] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 3
- [56] Jiajun Liu, Yibing Wang, Hanghang Ma, Xiaoping Wu, Xiaoqi Ma, xiaoming Wei, Jianbin Jiao, Enhua Wu, and Jie Hu. Kangaroo: A powerful video-language model supporting long-context video input. arXiv preprint arXiv:2408.15542, 2024. 3
- [57] Yangzhou Liu, Yue Cao, Zhangwei Gao, Weiyun Wang, Zhe Chen, Wenhai Wang, Hao Tian, Lewei Lu, Xizhou Zhu, Tong Lu, et al. Mminstruct: A high-quality multi-modal instruction tuning dataset with extensive diversity. Science China Information Sciences, 67(12):1–16, 2024. 1
- [58] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. TempCompass: Do video LLMs really understand videos? In

- Findings of the Association for Computational Linguistics ACL 2024, pages 8731–8772, 2024. 7, 3
- [59] Ziyu Liu, Tao Chu, Yuhang Zang, Xilin Wei, Xiaoyi Dong, Pan Zhang, Zijian Liang, Yuanjun Xiong, Yu Qiao, Dahua Lin, et al. Mmdu: A multi-turn multi-image dialog understanding benchmark and instruction-tuning dataset for lvlms. arXiv preprint arXiv:2406.11833, 2024. 1
- [60] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Oryx mllm: On-demand spatialtemporal understanding at arbitrary resolution. arXiv preprint arXiv:2409.12961, 2024. 1, 3, 7, 2
- [61] Zhijian Liu, Ligeng Zhu, Baifeng Shi, Zhuoyang Zhang, Yuming Lou, Shang Yang, Haocheng Xi, Shiyi Cao, Yuxian Gu, Dacheng Li, et al. Nvila: Efficient frontier visual language models. arXiv preprint arXiv:2412.04468, 2024. 7, 2
- [62] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating math reasoning in visual contexts with gpt-4v, bard, and other large multimodal models. ArXiv preprint, abs/2310.02255, 2023. 7, 2, 3
- [63] Yiting Lu, Xin Li, Yajing Pei, Kun Yuan, Qizhi Xie, Yunpeng Qu, Ming Sun, Chao Zhou, and Zhibo Chen. Kvq: Kwai video quality assessment for short-form videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 25963–25973, 2024. 8
- [64] Run Luo, Haonan Zhang, Longze Chen, Ting-En Lin, Xiong Liu, Yuchuan Wu, Min Yang, Minzheng Wang, Pengpeng Zeng, Lianli Gao, et al. Mmevol: Empowering multimodal large language models with evol-instruct. arXiv preprint arXiv:2409.05840, 2024. 1
- [65] Yongdong Luo, Xiawu Zheng, Xiao Yang, Guilin Li, Haojia Lin, Jinfa Huang, Jiayi Ji, Fei Chao, Jiebo Luo, and Rongrong Ji. Video-rag: Visually-aligned retrievalaugmented long video comprehension. arXiv preprint arXiv:2411.13093, 2024. 5
- [66] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Khan. Videogpt+: Integrating image and video encoders for enhanced video understanding. arXiv preprint arXiv:2406.09418, 2024. 1
- [67] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024), 2024. 1
- [68] Jordan Meyer, Nick Padgett, Cullen Miller, and Laura Exline. Public domain 12m: A highly aesthetic image-text dataset with novel governance mechanisms. arXiv preprint arXiv:2410.23144, 2024. 1
- [69] Mathew Monfort, SouYoung Jin, Alexander Liu, David Harwath, Rogerio Feris, James Glass, and Aude Oliva. Spoken moments: Learning joint audio-visual representations from video descriptions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14871–14881, 2021. 1

- [70] Mathew Monfort, Bowen Pan, Kandan Ramakrishnan, Alex Andonian, Barry A McNamara, Alex Lascelles, Quanfu Fan, Dan Gutfreund, Rog´erio Schmidt Feris, and Aude Oliva. Multi-moments in time: Learning and interpreting models for multi-action video understanding. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44

(12):9434–9445, 2021. 1

- [71] Zach Nagengast, Eduardo Pach, Seva Maltsev, and Ben Egan. Dataset card for laion dall·e 3 discord dataset. 1
- [72] Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-tovideo generation. arXiv preprint arXiv:2407.02371, 2024. 1
- [73] Vicente Ordonez, Girish Kulkarni, and Tamara Berg. Im2text: Describing images using 1 million captioned photographs. Advances in neural information processing systems, 24, 2011. 1
- [74] Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Mateusz Malinowski, Yi Yang, Carl Doersch, et al. Perception test: A diagnostic benchmark for multimodal video models. Advances in Neural Information Processing Systems, 36:42748–42761, 2023. 7, 3
- [75] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. ArXiv, abs/2306.14824, 2023. 1
- [76] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, pages 8748– 8763, 2021. 2
- [77] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. ArXiv, abs/2305.18290, 2023. 6
- [78] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020. 6
- [79] Ruchit Rawal, Khalid Saifullah, Miquel Farr´e, Ronen Basri, David Jacobs, Gowthami Somepalli, and Tom Goldstein. Cinepile: A long video question answering dataset and benchmark. arXiv preprint arXiv:2405.08813, 2024. 1
- [80] Anna Rohrbach, Atousa Torabi, Marcus Rohrbach, Niket Tandon, Christopher Pal, Hugo Larochelle, Aaron Courville, and Bernt Schiele. Movie description. International Journal of Computer Vision, 123:94–120, 2017. 1
- [81] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion400m: Open dataset of clip-filtered 400 million image-text pairs. ArXiv preprint, abs/2111.02114, 2021. 6, 1

- [82] Share. Sharegemini: Scaling up video caption data for multimodal large language models, 2024. 1
- [83] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, 2018. 1
- [84] Min Shi, Shihao Wang, Chieh-Yun Chen, Jitesh Jain, Kai Wang, Junjun Xiong, Guilin Liu, Zhiding Yu, and Humphrey Shi. Slow-fast architecture for video multi-modal large language models. arXiv preprint arXiv:2504.01328, 2025. 7, 3
- [85] Yan Shu, Peitian Zhang, Zheng Liu, Minghao Qin, Junjie Zhou, Tiejun Huang, and Bo Zhao. Video-xl: Extra-long vision language model for hour-scale video understanding. arXiv preprint arXiv:2409.14485, 2024. 3
- [86] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Xun Guo, Tian Ye, Yan Lu, Jenq-Neng Hwang, et al. Moviechat: From dense token to sparse memory for long video understanding. arXiv preprint arXiv:2307.16449, 2023. 3
- [87] Krishna Srinivasan, Karthik Raman, Jiecao Chen, Michael Bendersky, and Marc Najork. Wit: Wikipedia-based image text dataset for multimodal multilingual machine learning. arXiv preprint arXiv:2103.01913, 2021. 1
- [88] Yipeng Sun, Zihan Ni, Chee-Kheng Chng, Yuliang Liu, Canjie Luo, Chun Chet Ng, Junyu Han, Errui Ding, Jingtuo Liu, Dimosthenis Karatzas, et al. Icdar 2019 competition on large-scale street view text with partial labeling-rrc-lsvt. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1557–1562. IEEE, 2019. 1
- [89] Mohammad Reza Taesiri and Cor-Paul Bezemer. Videogamebunny: Towards vision assistants for video games. arXiv preprint arXiv:2407.15295, 2024. 1
- [90] Reuben Tan, Ximeng Sun, Ping Hu, Jui hsien Wang, Hanieh Deilamsalehy, Bryan A. Plummer, Bryan Russell, and Kate Saenko. Koala: Key frame-conditioned long video-llm.

2024. 3

- [91] Zhiyu Tan, Xiaomeng Yang, Luozheng Qin, and Hao Li. Vidgen-1m: A large-scale dataset for text-to-video generation. arXiv preprint arXiv:2408.02629, 2024. 1
- [92] Shaun Toh, Adriel Kuek, Wen-Haw Chong, and Roy KaWei Lee. Mermaid: A dataset and framework for multimodal meme semantic understanding. In 2023 IEEE International Conference on Big Data (BigData), pages 433–

442. IEEE, 2023. 1

- [93] Shubham Toshniwal, Wei Du, Ivan Moshkov, Branislav Kisacanin, Alexan Ayrapetyan, and Igor Gitman. Openmathinstruct-2: Accelerating ai for math with massive open-source instruction data. arXiv preprint arXiv:2410.01560, 2024. 1
- [94] Yuxiang Tuo, Wangmeng Xiang, Jun-Yan He, Yifeng Geng, and Xuansong Xie. Anytext: Multilingual visual text generation and editing. arXiv preprint arXiv:2311.03054, 2023. 1
- [95] Haibo Wang, Zhiyang Xu, Yu Cheng, Shizhe Diao, Yufan Zhou, Yixin Cao, Qifan Wang, Weifeng Ge, and Lifu

- Huang. Grounded-videollm: Sharpening fine-grained temporal grounding in video large language models. arXiv preprint arXiv:2410.03290, 2024. 1
- [96] Jiawei Wang, Liping Yuan, Yuchen Zhang, and Haomiao Sun. Tarsier: Recipes for training and evaluating large video description models. arXiv preprint arXiv:2407.00634, 2024. 3, 7, 1, 2
- [97] Junjie Wang, Yin Zhang, Yatai Ji, Yuxiang Zhang, Chunyang Jiang, Yubo Wang, Kang Zhu, Zekun Wang, Tiezhen Wang, Wenhao Huang, et al. Pin: A knowledge-intensive dataset for paired and interleaved multimodal documents. arXiv preprint arXiv:2406.13923, 2024. 1
- [98] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1
- [99] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Song XiXuan, et al. Cogvlm: Visual expert for pretrained language models. Advances in Neural Information Processing Systems, 37:121475–121499, 2024. 7
- [100] Weiyun Wang, Yiming Ren, Haowen Luo, Tiantong Li, Chenxiang Yan, Zhe Chen, Wenhai Wang, Qingyun Li, Lewei Lu, Xizhou Zhu, et al. The all-seeing project v2: Towards general relation comprehension of the open world. In European Conference on Computer Vision, pages 471–

490. Springer, 2024. 1

- [101] Xidong Wang, Dingjie Song, Shunian Chen, Chen Zhang, and Benyou Wang. Longllava: Scaling multi-modal llms to 1000 images efficiently via a hybrid architecture. arXiv preprint arXiv:2409.02889, 2024. 3
- [102] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023. 1
- [103] Yi Wang, Xinhao Li, Ziang Yan, Yinan He, Jiashuo Yu, Xiangyu Zeng, Chenting Wang, Changlian Ma, Haian Huang, Jianfei Gao, et al. Internvideo2. 5: Empowering video mllms with long and rich context modeling. arXiv preprint arXiv:2501.12386, 2025. 3
- [104] Yuetian Weng, Mingfei Han, Haoyu He, Xiaojun Chang, and Bohan Zhuang. Longvlm: Efficient long video understanding via large language models. In European Conference on Computer Vision, 2024. 3
- [105] Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, et al. Deepseek-vl2: Mixture-ofexperts vision-language models for advanced multimodal understanding. arXiv preprint arXiv:2412.10302, 2024. 7, 2
- [106] Tianwei Xiong, Yuqing Wang, Daquan Zhou, Zhijie Lin, Jiashi Feng, and Xihui Liu. Lvd-2m: A long-take video dataset with temporally dense captions. arXiv preprint arXiv:2410.10816, 2024. 1

- [107] Bo Xu, Tingting Li, Junzhe Zheng, Mehdi Naseriparsa, Zhehuan Zhao, Hongfei Lin, and Feng Xia. Met-meme: A multimodal meme dataset rich in metaphors. In Proceedings of the 45th international ACM SIGIR conference on research and development in information retrieval, pages 2887–2899, 2022. 1
- [108] Fuzhao Xue, Yukang Chen, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, et al. Longvila: Scaling long-context visual language models for long videos. arXiv preprint arXiv:2408.10188, 2024. 3
- [109] Dongjie Yang, Suyuan Huang, Chengqiang Lu, Xiaodong Han, Haoxin Zhang, Yan Gao, Yao Hu, and Hai Zhao. Vript: A video is worth thousands of words. Advances in Neural Information Processing Systems, 37:57240–57261,

2024. 1

- [110] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, Qianyu Chen, Huarong Zhou, Zhensheng Zou, Haoye Zhang, Shengding Hu, Zhi Zheng, Jie Zhou, Jie Cai, Xu Han, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. Minicpm-v: A gpt-4v level mllm on your phone, 2024. 5
- [111] Jianhui Yu, Hao Zhu, Liming Jiang, Chen Change Loy, Weidong Cai, and Wayne Wu. Celebv-text: A large-scale facial text-video dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14805–14814, 2023. 1
- [112] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284, 2023. 1
- [113] Shenghai Yuan, Jinfa Huang, Yongqi Xu, Yaoyang Liu, Shaofeng Zhang, Yujun Shi, Rui-Jie Zhu, Xinhua Cheng, Jiebo Luo, and Li Yuan. Chronomagic-bench: A benchmark for metamorphic evaluation of text-to-time-lapse video generation. Advances in Neural Information Processing Systems, 37:21236–21270, 2024. 1
- [114] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of CVPR, 2024. 7, 2, 3
- [115] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023. 2, 5
- [116] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, et al. Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106, 2025. 1, 3, 7, 2

- [117] Jiacheng Zhang, Yang Jiao, Shaoxiang Chen, Na Zhao, and Jingjing Chen. Eventhallusion: Diagnosing event hallucinations in video llms. arXiv preprint arXiv:2409.16597,

2024. 7, 3

- [118] Jianrui Zhang, Cai Mu, and Yong Jae Lee. Vinoground: Scrutinizing lmms over dense temporal reasoning with short videos. arXiv preprint arXiv:2410.02763, 2024. 7, 3
- [119] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, et al. Lmms-eval: Reality check on the evaluation of large multimodal models. arXiv preprint arXiv:2407.12772, 2024. 2
- [120] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. ArXiv, abs/2406.16852, 2024. 3
- [121] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. ArXiv preprint, abs/2406.16852, 2024. 4
- [122] Ruohong Zhang, Liangke Gui, Zhiqing Sun, Yihao Feng, Keyang Xu, Yuanhan Zhang, Di Fu, Chunyuan Li, Alexander Hauptmann, Yonatan Bisk, and Yiming Yang. Direct preference optimization of video large multimodal models from language model reward. ArXiv preprint, abs/2404.01258, 2024. 8, 1
- [123] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data, 2024. 2
- [124] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 1, 7
- [125] Hang Zhao, Antonio Torralba, Lorenzo Torresani, and Zhicheng Yan. Hacs: Human action clips and segments dataset for recognition and temporal localization. In Proceedings of the IEEE International Conference on Computer Vision, pages 8668–8678, 2019. 1
- [126] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. ArXiv preprint, abs/2406.04264, 2024. 7, 3
- [127] Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In Proceedings of the AAAI conference on artificial intelligence, 2018. 1

## Appendix

|Task<br><br>|Dataset|
|---|---|
|Stage 1 Datasets| |
|Image Caption<br><br>|LAION (EN 6.7M, ZH 3.2M) [81], Conceptual Captions (7.3M) [83], SBU (0.8M) [73], COYO (11M) [7], WuKong (2.9M) [25], LAION COCO (16M) [1], OMEGA Image Caption (79M) [39]|
|Video Caption<br><br>|InternVid-10M-FLT (1.6M) [102], Panda-70M (0.9M) [12], OMEGA Video Caption (4M) [39]|
|Stage 1.5 Datasets| |
|Image Caption<br><br>|Met-meme [107], PD12M [68], dalle3 [71], GBC10M [30], DenseFusion-1M [48], GameBunny [89], MERMAID [92], CC12M (1M) [9], BLIP3 [3], AllSeeingV2 [100]|
|Video Caption<br><br>|ChronoMagic [113], VideoChatGPT [67], YouCook2 [127], CelebV [111], SthSthV2 [24], MiraData [35], Hacs [125], OpenVid-1M [72], Kinetics 700 [8], ShareGPT4Video [11], Vript [109], Shot2Story [27], ShareGemini [82]<br><br>|
|Question Answering<br><br>|MMDU [59], MMiT [70]|
|Knowledge|Wikipedia [21], Wikimedia [21], WIT [87]|
|Code<br><br>|WebSight [42]|
|OCR|LSVT [88], ArT [15], DocMatix [41]|
|Interleaved|OBELICS [40], PIN [97]|
|Mixed-Task Dataset<br><br>|MMInstruct [57], LVD-2M [106], MMEvol [64]|
|Stage 2 Datasets| |
|Instruction<br><br>|Countix [18], VideoChat [45], Videogpt+ [66], Openmathinstruct-2 (2M) [93], RepCountA [31], Vidgen1m [91], CompCap [13], Metamath [112], Llava-Onevision [43], Anytext (0.3M) [94], Llava-Video [124], S-MiT [69], LSMDC [80], Infinity-MM [26], Mantis [33], ShareGPT4V [10], CinePile [79], LLaVA-Hound [122]|
|Grounding<br><br>|GRIT [75], RefCOCO [36]|
|Temporal Grounding<br><br>|GroundedVideoLLM [95]|
|Stage 3 (DPO) Datasets| |
|Open-ended QA<br><br>|Llava-Video [124] (10K)|
|Image Caption<br><br>|Llava-Onevision [43] (10K), DenseFusion-1M [48] (10K)|
|Video Caption<br><br>|WebVid [5] (8K), Kinetics 700 [8] (8K), OOPS [19] (4K)<br><br>|

Table 6. Summary of the training datasets of different stages.

### A. Training Datasets

The datasets used for training our model at different stages are shown in Table 6. For a number of large-scale datasets, we have randomly selected a specific number of samples. The count of these samples is also indicated in Table 6.

We have also curated two datasets from the OMEGA project [39], the OMEGA Image Caption (containing 79M samples) and OMEGA Video Caption (containing 4M samples), by sampling videos and images along with their corresponding titles and captions. These two datasets are utilized in the first stage of our model training.

For certain datasets that either lack captions or only possess low-quality ones, for example, CC12M [9], CelebV [111], Hacs [125], and Kinetics 700 [8], we carefully designed a pipeline to generate high-quality captions. Initially, we utilized Qwen2VL-72B [98], InternVL2.5-78BMPO [14] and Tarsier-34B [96] (video only) to describe these samples in detail. Subsequently, we used DeepSeekR1-Distill-Llama-70B [17] to amalgamate captions generated by different models while attempting to resolve all inconsistencies using its COT capabilities. The captions produced by this process generally demonstrated superior qual-

Qwen2.5VL-7B Mavors-7B Images

Prefilling (ms) 397 392 Decoding (token/s) 23 30

Prefilling (ms) 1,225 448 Decoding (token/s) 22 30

Videos

Table 7. Inference efficiency between Qwen2.5VL-7B and Mavors-7B. Model is better when Prefilling (ms) is lower and Decoding (token/s) is larger.

ity and comprehensibility.

We observed that many composite datasets incorporate content from established standalone datasets, leading to potential data redundancy. To address this, we implemented a deduplication process for identical samples (images or videos). Specifically, we calculated the Perplexity (PPL) of the associated text using the Qwen2VL-72B [98] model, distinguishing between QA and Captioning tasks. For duplicate visual content within QA tasks, we retained the two samples exhibiting the lowest text PPL scores. For Captioning tasks, one sample was randomly selected from the two with the lowest PPL for inclusion in our training set.

For the data in the DPO stage, we selected a specific number of samples from the corresponding datasets. The preference datasets were then generated in accordance with the following methods:

- 1. Open-ended QA: Positive examples are generated by prompting the model with diverse inputs to produce responses that are correct, of appropriate length, and properly terminated. Negative examples are derived from the same inputs by adjusting the sampling temperature to elicit incorrect or overly brief answers.
- 2. Image Captioning: Multiple candidate captions are generated per image using the model under high temperatures. These candidates are then ranked according to a predefined scoring strategy, forming positive (higherranked) and negative (lower-ranked) pairs for DPO training.
- 3. Video Captioning: Captions generated from the original video serve as positive examples. Negative examples are created by captioning the video after segmenting it into four equal parts and shuffling their temporal order.

### B. Analysis on the Inference Costs

We evaluate the inference performance of Qwen2.5VL7B and Mavors-7B using an GPU. Initially, we measure the execution time of the model.generate function via the standard HuggingFace implementation (with FlashAttention-2 enabled) to capture the core model execution time, excluding video preprocessing. Table 7 summa-

rizes the inference times for both models on the DREAM1K and CapsBench video captioning tasks. The results show that Mavors’ more efficient video representation reduces both the ViT computations and the language model’s context window requirements. Consequently, Mavors-7B demonstrates significant speed improvements on video understanding tasks, achieving 2.7x faster prefill and 1.4x faster decoding compared to Qwen2.5VL-7B. Furthermore, integrating the vLLM inference framework with overlapping vision preprocessing enables 2.5s per image in CapsBench and 3.7s per video in DREAK-1K, reducing from about 13s per image and 20s per video respectively. These findings indicate that Mavors provides an economical solution for scenarios requiring frequent or high-volume multimodal model inference.

### C. Details of Experiments

Evaluation Setup. To ensure a standardized and reproducible evaluation, we conduct experiments on both opensource and closed-source models using consistent protocols. For open-source models, we adopt the lmms-eval framework [119], which offers a unified pipeline tailored for benchmarking MLLMs. All open-source models are evaluated using the officially released checkpoints to preserve the integrity of reported results. To maintain experimental stability, we fix the decoding strategy to greedy decoding, set the maximum number of generated tokens to 1024. Image and video resolution, along with other preprocessing settings, follow the default configurations provided by the lmms-evak framework or the respective model implementations. For closed-source models, including Gemini1.5-Pro-002 [23] and GPT-4o-20240806 [32], we access them through their official APIs. Due to the restricted controllability over decoding parameters, we adopt the default generation settings provided by each platform. For benchmarks requiring GPT-based automatic scoring, such as those involving instruction-following or open-ended generation tasks, we follow the evaluation protocol described in the original benchmark papers or apply the default settings specified by the lmms-eval framework to select the judge model. Specifically, for MathVista [62], we use GPT4-Turbo (1106) as the judge model. For CapsBench [52] and MMMU [114], we adopt GPT-4o (20240806), while for DREAM-1K [96], we follow the original benchmark and employ GPT-3.5-Turbo (0125) to perform automatic scoring. These choices align with the evaluation protocols used in the respective benchmark papers, ensuring fair and comparable results across models.

Baseline Models. To comprehensively evaluate the performance of our proposed Mavors-7B, we select a diverse set of baseline models tailored to the specific characteristics of both image and video benchmarks.

For image benchmarks, we compare against two leading

proprietary models, GPT-4o [32] and Gemini-1.5-Pro [23]. GPT-4o, developed by OpenAI, is capable of processing text, images, and audio in a unified manner and has demonstrated strong performance in visual reasoning tasks. Gemini, developed by Google DeepMind, similarly integrates multimodal capabilities and excels in scenarios requiring complex cross-modal understanding. We also include a range of high-performing open-source MLLMs in our comparison. These include CogVLM2 [29], a model optimized for visual-language understanding in dynamic contexts; GLM-4V [29], which extends the GLM architecture with strong visual recognition capabilities; LLaVA-OneVision [43], a widely recognized open-source MLLM that integrates a collection of high-quality multimodal datasets, advanced training strategies, and refined model designs to achieve strong performance across imagebased benchmarks; InternVL2.5 [14], which is an advanced MLLM series developed by Shanghai Artificial Intelligence Laboratory. Building upon the architecture of InternVL2, it introduces significant enhancements in training strategies and data quality; DeepSeek-VL2 [105], an MoE-based model balancing scalability and accuracy; and Qwen2.5VL [4], a model that significantly enhance general image recognition capabilities, expanding to a vast array of categories, including plants, animals, landmarks, and various products. It also excels in precise object localization, advanced text recognition, and document parsing.

For video benchmarks, we select four representative categories of baseline models, each exemplifying distinct video processing strategies. The first category includes models that employ sparse frame sampling with high performance, such as NVILA [61] and LLaVA-Video [123], which focus on selecting key frames to reduce computational overhead while maintaining contextual understanding. NVILA, developed by NVIDIA, utilizes a “scale-thencompress” paradigm that first increases spatial and temporal resolutions and then compresses visual tokens, enabling efficient processing of high-resolution images and long videos. LLaVA-Video improves video understanding by introducing a high-quality synthetic dataset, LLaVA-Video178K [123], specifically designed for video instructionfollowing tasks. Models like Qwen2.5-VL [4] and Oryx1.5 [60] adopt dense frame sampling at lower resolutions to achieve a trade-off between information richness and efficiency (we set at most 768 frames in our experiments). Oryx-1.5 is a unified MLLM designed to flexibly and efficiently handle visual inputs with varying spatial scales and temporal lengths, making it well-suited for processing both high-resolution images and extended video sequences. In addition, we include models such as VideoChat-Flash [47] and VideoLLaMA3 [116], which apply dense sampling combined with token compression to handle long video sequences effectively (up to 1000 frames in our experi-

ments). VideoChat-Flash leverages this strategy to mitigate the computational overhead introduced by dense sampling, enabling effective handling of long-duration inputs without sacrificing performance. Similarly, VideoLLaMA3 integrates token compression with dense sampling to reduce input redundancy, thereby enhancing the model’s ability to understand and reason over extended video content. Finally, we include Slow-fast MLLM [84], which employs a specialized dual-pathway sampling mechanism to capture temporal dynamics at multiple granularities. By processing visual inputs through both slow and fast pathways, the model effectively models temporal variations across different timescales.

Benchmarks. It is crucial to comprehensively and objectively assess a model’s capabilities across various aspects and dimensions. To this end, we include a broad range of representative image and video benchmarks in our evaluation.

We adopt MMMU [114], MathVista [62], AI2D [37], and CapsBench [52] as representative image benchmarks, covering a broad range of visual understanding and reasoning tasks.

- • MMMU targets expert-level multimodal reasoning across diverse academic domains, featuring varied visual inputs such as charts, diagrams, and tables.
- • MathVista focuses on complex mathematical problem solving that integrates textual and visual information.
- • AI2D evaluates the ability to interpret scientific diagrams commonly used in elementary science education.
- • CapsBench emphasizes compositional reasoning by requiring models to generate comprehensive, detailed, and accurate descriptions of visual scenes. It challenges models to precisely capture a wide range of visual information, including object attributes, spatial relationships, and inter-object interactions.

Together, these benchmarks offer a comprehensive assessment of image-based multimodal capabilities.

We conduct evaluations on a diverse set of video benchmarks, including MMWorld [28], PerceptionTest [74], Video-MME [22], MLVU [126], MVBench [46], EventHallusion [117], TempCompass [58], VinoGround [118], and DREAM-1K [96].

- • MMWorld evaluates MLLMs’ ability to reason about real-world dynamics across diverse disciplines and tasks. It includes 1,910 videos and 6,627 QA pairs covering explanation, counterfactual reasoning, and future prediction.
- • PerceptionTest evaluates the perceptual and reasoning skills of MLLMs across video, audio, and text modalities. It includes 11.6K real-world videos and focuses on cognitive skills and reasoning types—such as memory, abstraction, and counterfactual thinking—beyond traditional classification or detection tasks. We use the vali-

- dation set in the experiments.
- • Video-MME is a comprehensive benchmark for evaluating MLLMs across diverse video types, temporal lengths, and multimodal inputs including subtitles and audio. It features 900 manually annotated videos spanning 254 hours and 2,700 QA pairs, offering a rigorous test of models’ generalization and contextual understanding. We evaluate Video-MME without subtitles in our experiments.
- • MLVU is a benchmark designed for comprehensive evaluation of long video understanding, featuring extended video durations and diverse genres such as movies, surveillance, and egocentric videos. It includes a variety of tasks to assess MLLMs’ abilities in handling complex temporal dependencies and multi-scene reasoning across long-form content.
- • MVBench is a diagnostic benchmark designed to evaluate the temporal understanding capabilities of MLLMs through 20 challenging video tasks that go beyond static image reasoning. By systematically transforming static tasks into dynamic ones, it covers a wide range of temporal skills and ensures fair evaluation using ground-truth annotations converted into multiple-choice questions.
- • EventHallusion is a benchmark designed to evaluate hallucination in MLLMs, specifically focusing on eventlevel understanding—a core aspect of video analysis. It probes models’ susceptibility to language priors and vision-language biases, providing a targeted assessment of their reliability in temporal event reasoning.
- • TempCompass is a benchmark designed to evaluate the fine-grained temporal perception abilities of MLLMs across diverse task types. By introducing videos with controlled temporal variations and minimizing static or linguistic bias, it enables precise assessment of model performance on aspects such as speed, direction, and sequence understanding.
- • VinoGround is a benchmark that evaluates temporal counterfactual reasoning in short videos through 1,000 natural video-caption pairs.
- • DREAM-1K is a challenging benchmark for detailed video description, featuring 1,000 clips from diverse sources such as films, stock footage, and short-form videos. Each video is paired with fine-grained humanannotated descriptions, and evaluated using AutoDQ, a metric better suited for assessing rich, multi-event narratives than traditional captioning scores.

These benchmarks collectively cover a wide range of video understanding challenges, such as temporal reasoning, event prediction, visual grounding, perception under uncertainty, and multi-turn video-based instruction following, enabling a comprehensive assessment of the model’s performance across different video-centric tasks.

### D. Needle in a Haystack Test

1

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

15

Totalnumberofchunks

30

45

60

15th

30th

45th

60th

1st

Index of chunk with needle

1.0

[Figure 128]

0.8

0.6

0.4

0.2

0.0

- Figure 11. Results of NIAH of Mavors with at most 60 video chunks.

Inspired by the design in LongVA [121], we conduct Needle-in-a-Haystack (NIAH) test. We adopt a chunk-level NIAH evaluation scheme, which primarily focuses on evaluating the model’s comprehension accuracy when a target frame is inserted into different video chunks. We utilize 10 short-duration and 10 medium-duration videos from the Video-MME benchmark. We examine the model’s performance across video lengths ranging from 1 to 60 chunks. Recall that 60 chunks correspond to 960 frames. For a given number of chunks cV, we performed 50*cV trials. In each trial, we randomly select a video, an image (the ‘needle’), a specific chunk within the video, and a frame position within that chunk. The selected image then replaces the original frame at the chosen position. Notably, after selecting a video, we first apply accelerating playback (temporal subsampling) to ensure the video frames precisely fit into cV chunks. Figure 11 illustrates the accuracy results. As observed, perfect accuracy is achieved within the model’s training window length (32 chunks). Moreover, Mavors maintains strong accuracy even as the number of chunks increases beyond this window. This experiment indicates that Mavors can provide reliable understanding for videos of reasonable duration, provided essential frame information is not lost during the accelerating playback process.

### E. Showcases of Mavors in Image Captioning

We present a few examples of Mavors’ performance on the CapsBench benchmark in Figure 12. Mavors demonstrates a strong ability to generate accurate and detailed image captions, and it could recognize more proper nouns and under-

stand human interactions and emotions more precisely than Qwen2.5-VL-7B.

### F. Showcases of Mavors with Token Compression

Our analysis indicates that as token compression increases up to 60%, there is negligible impact on Video QA performance, while performance on Captioning tasks degrades progressively. We present two case studies to illustrate the specific effects of captioning. In the first case (Figure 13), we observe that despite an imperfect initial caption, higher compression rates lead to increased model hallucinations. The model appears to perceive visual changes from the retained tokens but resorts to speculation, resulting in inaccurate descriptions. In the second case (Figure 14), increased compression causes the model to generate sparser descriptions, omitting critical details and introducing hallucinations. These findings suggest that token compression can pose performance risks, particularly for complex scene captioning tasks.

[Figure 129]

[Figure 130]

QWen2.5-VL-7B: The image depicts a scene set in what appears to be an underground train station, characterized by its dim lighting and blue-toned ambiance. A train is stationed on the left side of the frame, its windows reflecting the subdued light from above. The platform is lined with posters and advertisements, some of which are partially visible, adding to the urban atmosphere. A statue of a lion, positioned near the center-right of the image, stands as a focal point amidst the otherwise industrial setting. In the foreground, a man with long dark hair and a beard is seated, facing slightly towards the left side of the frame. He is dressed in a dark suit, suggesting a formal or professional demeanor. His posture is relaxed yet attentive, and his gaze is directed off-screen, indicating he might be observing something or someone outside the frame. The man‘s expression is serious, hinting at a moment of contemplation or anticipation. The background features a series of arches and columns, typical of a subway station design, contributing to the sense of depth and scale. The lighting is low, creating a moody and somewhat mysterious atmosphere. The overall color palette is dominated by cool tones, with shades of blue and gray dominating the scene, enhancing the cinematic quality of the image. The image appears to be a still from a film, given the high level of detail and the dramatic composition. The focus is sharp on the man and the immediate surroundings, while the background is softly blurred, drawing attention to the subject. There are no visible texts or logos within the image itself, but the overall aesthetic suggests a narrativedriven scene, possibly from a thriller or action genre. The image type is a photograph, capturing a moment that feels both intimate and expansive, blending human presence with the vastness of the urban environment.

QWen2.5-VL-7B: The image is an animated scene featuring two

characters in what appears to be a hospital room. The character on the right is an older man with gray hair, wearing large black-framed glasses and a brown sweater over a white collared shirt with a black bow tie. His expression is

serious, and he seems to be engaged in conversation with the other character. The character on the left has short gray hair and is wearing dark-rimmed glasses, though only the side of his face is visible. The background includes a pink wall with a window covered by a striped curtain, and two IV bags hanging from a pole, suggesting a medical setting. The lighting is soft and even, typical of indoor scenes, and the overall color palette is muted with pastel tones. The background elements, such as the IV bags and the curtain, indicate a clinical environment, likely a hospital ward. The characters‘ attire and the setting suggest a formal or professional interaction, possibly a doctorpatient discussion. The image does not contain any text or logos that are discernible. The style of the image is characteristic of modern computergenerated animation, with smooth textures and detailed rendering of the characters and environment. The color grading is balanced, with natural skin tones and a subdued color scheme that enhances the realism of the scene. There are no noticeable artifacts such as film grain, lens flare, or vignetting effects, indicating a high-quality render. The image type is clearly an animation, as evidenced by the stylized features and the absence of photographic details like depth of field or natural light variations.

Mavors-7B: This image is a photograph, likely a still from a film, depicting a man standing on a dimly lit subway platform. Positioned on the right side of the frame, the man, identified as Keanu Reeves, is shown in profile, facing towards the left. He has long, dark, somewhat messy hair and a beard, and his expression appears contemplative or perhaps weary as he looks off into the distance. He is wearing a dark, possibly black, collared shirt or jacket. To his left, a subway train is visible, occupying the left third of the image. The train is mostly out of focus, rendered in cool blue and teal tones, with its windows reflecting the ambient light. The platform itself is dark, with a wet-looking surface suggesting recent rain or cleaning, reflecting the overhead lights. In the middle ground, slightly behind the man, stands a blurred statue, possibly of a lion or similar creature, near the edge of the platform. The background features the arched ceiling of the subway station, lined with bright, evenly spaced lights that create a tunnel effect leading into the darkness further down the platform. The wall on the right side is covered in a collage of indistinct posters and advertisements, adding texture and color with patches of red, blue, and white. The overall environment is an underground subway station, likely at night or in a very dimly lit area. The image style features a cool color palette dominated by blues, teals, and dark shadows, contrasted by the warmer yellow and white lights. There is a shallow depth of field, keeping the man relatively sharp while blurring the train, statue, and background elements, creating a bokeh effect with the lights. The lighting is low-key, contributing to a moody, atmospheric feel. There are no visible texts or logos that are clear or legible.

Mavors-7B: This image is a still frame from the 3D animation film \"Up\". It features two characters, presumably Carl and Ellie, in a close-up shot. On the right side of the frame, occupying the majority of the space, is Carl, an older man with grey hair, a large round nose, and blue eyes. He wears thick, black, rectangular-framed glasses and is dressed in a brown tweed jacket over a white shirt and a black bowtie. His facial expression is stern and slightly annoyed, with furrowed brows and a downturned mouth, as he looks down and slightly to his left towards the other character. On the left side, partially visible from the back and side, is Ellie, also an older woman with grey hair. She wears thin, brown-framed glasses and is looking up at Carl. Only the back of her head, her right ear, and the side of her face are visible. The background suggests an indoor setting, possibly a room in a house or a hospital room, indicated by the presence of two clear glass IV bags hanging from metal stands in the upper left corner. Behind Carl, a textured, light-colored curtain hangs, and the wall appears to be a muted pink or reddish-brown color. The lighting is soft and warm, casting gentle shadows, particularly on Carl's face and the background elements. The image style is characteristic of modern 3D animation, with smooth textures, detailed rendering of the characters' features and clothing, and a warm color palette dominated by browns, greys, and pinks. There is a shallow depth of field, keeping Carl in sharp focus while slightly blurring the background elements like the IV bags and the curtain. The image type is a digital animation still.

- Figure 12. Comparison of the generated image captions from Qwen2.5-VL-7B and Mavors-7B. The text in red contains wrong content, and the text in green marks the detailed descriptions only appear in Mavors.

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Mavors: The video begins with a close-up of a large, orange, humanoid figure with a square head and two red, circular eyes, kneeling on one knee in a dimly lit, industrialenvironment filled with smoke and small fires. The figure has a label '04114' on its chest and two green buttons below it. As the figure raises its right hand, more similar orange figures, each with a different label such as '04114', '04115', '04116', and '04117', emerge from the smoke and walk towards the kneeling figure. The background reveals a chaotic scene with debris and small fires, indicating a recent disturbance or battle. The kneeling figure continues to gesture with its hands, seemingly addressing the approaching group. The camera then shifts to show a group of smaller, green, furry creatures, including one with a single large eye (Mike) and another large, green, furry creature (Sulley), who appear to be waking up or becoming aware of the situation. The orange figures continue to approach the green creatures, who seem to be interacting with each other, possibly discussing the situation. The video concludes with the orange figures standing around the green creatures, who are now more alert and engaged in conversation. The environment remains consistent throughout, with a dimly lit, industrial setting filled with smoke and small fires, suggesting a recent conflict or disturbance.

Drop 45%: The video begins with a close-up of a large, orange, humanoid figure with a square head and two red, circular eyes, kneeling on the ground amidsta smoky, industrial environment. The figure, marked with the number '04114' and two circular emblems on its chest, raises its right hand as if signaling or greeting. As the camera zooms out, more identical orange figures, each with a unique number and emblem, become visible in the background, also kneeling and facing the same direction. The scene is filled with smoke and small fires, suggesting a recent disturbance or ongoing activity. The camera continues to pull back, revealing a green, furry monster with blue spots, labeled 'GOD,' lying on the ground in the center, looking up at the orange figures with a slightly concerned expression. To the left of 'GOD,' a smaller, green, furry creature with large eyes, labeled 'MILFORD,' stands looking towards the orange figures. The background remains consistent with industrial elements like pipes and machinery, and the lighting is dim with a yellowish hue, casting shadows on the floor. The orange figures maintain their kneeling positions, with '04114' still raising its hand, while 'GOD' and 'MILFORD' continue to observe the scene. The video wraps up with the orange figures and the green monsters maintaining their positions, creating a tense yet intriguing atmosphere.

Drop 75%:The video begins with a close-up of a large, orange, humanoid robotwith a square head, kneeling on the ground with one armraised and the other on the ground. The background is filled with smoke and debris, suggesting a recent battle or destruction. As the camera angle shifts slightly, more robots of similar design, but in various colors including green and blue, come into view, all appearing to be in a state of distress or damage. The environment is dimly lit, with a yellowish light source illuminating the scene from the front. The robots have numbers and letters on their bodies, indicating they might be part of a larger group or team. The scene continues to reveal more robots, all in similar poses, with some appearing to be injured or inactive. The background remains consistent with smoke and debris, and the lighting continues to highlight the robots' distressed state. The video wraps up with a wider view of the group, showing at least five distinct robots, all in a state of apparent defeat or exhaustion, surrounded by the aftermath of a chaotic event.

Figure 13. Example of captioning task with token compression: higher compression ratio leads to the missing of critical details.

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

Mavors: The video begins with a man lying in bed, wearing a white tank top, interacting with a brown and white dog thatis standing on the bed. The man holds the dog's paw and appears to be talking to it, showing a range of emotions from concern to surprise. The scene then transitions to a different setting inside a tent-like room with blue walls and wooden

furniture. Three men are present: one with red hair sitting on a bed, another with dark hair sitting on a bed in the background, and a third with blonde hair sitting at a wooden table. The dog enters the room and approaches the red-haired man, who looks at it with a neutral expression. The dark-haired man stands up and stretches, while the blonde-haired man adjusts his hair. The red-haired man turns his head to look at the dark-haired man, who then stands with his hands on his hips, looking around the room. The dog sniffs the floor near the table, and the blonde-haired man continues to adjust his hair. The video wraps up with the red-haired man looking towards the dark-haired man, who is standing with his hands on his hips, and the blonde-haired man sitting at the table, still adjusting his hair. The dog remains near the table, and the scene is illuminated by a warm light coming from the entrance of the tent, suggesting it is either early morning or late afternoon.

Drop 45%: The video begins with a person lyingin bed, wearing a white tank top, interacting with a brown andwhite dogthat is standing onthe bed. The persongentlyholds the dog's paw and appears to be talking to it. The scene then transitions to a wider view of a tent-like room with a slanted ceiling, where three people are present. One person with red hair is sitting on a bed, looking towards the other two individuals.The person with blonde hair, who is sitting at a wooden table, gets up and walks towards the center of the room,where the dog is now standing. The blonde-haired person then sits down at the table, while the red-haired person continues to observe. The room is illuminated by soft, warm light coming through the fabric walls, suggesting either early morning or late afternoon. The red-haired person appears to be talking or reacting to the blonde-haired person, who is now sitting at the table. The dog remains in the center of the room, occasionally looking around. The video wraps up with the red-haired person still sitting on the bed, the blonde-haired person sitting at the table, and the dog standing in the middle, all within the same tent-like room.

Drop 75%:A man is lying in bed, looking up ata brown dog thatis standing on the bed and interacting with him. The scene then transitions to a wider view of a roomwith a tent-like ceiling, where three men are present. One man with red hair is sitting on a bed, looking towards the other two men. The man in the middle stands with his hands on his hips, while the man on the right is sitting at a wooden table, holding his head in his hands. The room is furnished with beds, a table, and benches, and the background shows a cloudy sky outside the tent.

Figure 14. Example of captioning task with token compression: higher compression ratio leads to the missing of critical details.

