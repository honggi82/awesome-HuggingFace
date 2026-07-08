arXiv:2410.16267v2[cs.CV]9Jun2025

# xGen-MM-Vid (BLIP-3-Video): You Only Need 32 Tokens to Represent a Video Even in VLMs

Michael S. Ryoo1, Honglu Zhou1, Shrikant Kendre1, Can Qin1, Le Xue1, Manli Shu1, Jongwoo Park2, Kanchana Ranasinghe2, Silvio Savarese1, Ran Xu1, Caiming Xiong1, Juan Carlos Niebles1 1Salesforce AI Research 2Stony Brook University mryoo@salesforce.com

## Abstract

We present xGen-MM-Vid (BLIP-3-Video): a multimodal language model for videos, particularly designed to efficiently capture temporal information over multiple frames. BLIP-3-Video takes advantage of the ‘temporal encoder’ in addition to the conventional visual tokenizer, which maps a sequence of tokens over multiple frames into a compact set of visual tokens. This enables BLIP-3Video to use much fewer visual tokens than its competing models (e.g., 32 vs. 4608 tokens). We explore different types of temporal encoders, including learnable spatio-temporal pooling as well as sequential models like Token Turing Machines. We experimentally confirm that BLIP-3-Video obtains video question-answering accuracies comparable to much larger state-of-the-art models (e.g., 34B), while being much smaller (i.e., 4B) and more efficient by using fewer visual tokens.

## 1 Introduction

Large Vision-Language Models (VLMs), benefiting from large-scale image-text training, have been dominating the field of computer vision. Recently, open-source VLMs are obtaining strong results, despite having much smaller size than the commercial models (e.g., 4B vs. Trillions).

In addition to VLMs trained with images, VLMs for videos are becoming increasingly popular. The key component in a VLM for videos is the temporal abstraction of tokens over multiple frames. Models like Video-ChatGPT [31] and PLLaVA [49] rely on a simple spatial/temporal pooling on top of image framelevel tokens to represent the entire video. Some models rely on a separate video encoder to capture temporal information in videos [28]. Similarly, some models use additional convolutional layers (or Transformer layers) over frames to reduce their representation size (e.g., Video-LLaMA [56], Kangaroo [30]). Approaches that simply collect all the visual tokens from all the frames (e.g., MiniGPT4-video [3], LLaVANeXT [23], Tarsier [43] and LLaVA-OneVision [43]) also have been very popular recently, as they allow capturing all the details from the frame-level tokens.

Tarsier-34B

78

BLIP-3-Video-4B-32t

###### MSVD-QAaccuracy(%)

BLIP-3-Video-4B-16t

76

Tarsier-7B LLaVA-NeXT-V-32B

74

LLaVA-OneVision-7B

72

Video-LLaVA

LLaMA-VID-13B

70

LLaMA-VID-7B Chat-UniVi

102 103 104

Number of tokens

Figure 1: SOTA video VLM model comparison: Number of visual tokens vs. video-QA accuracy.

However, this often makes the number of tokens for video to be very huge (e.g., thousands even for 8

38th Conference on Neural Information Processing Systems (NeurIPS 2024).

[Figure 1] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile1.png>)

(pre-trained) LLM

M=32

[Figure 2] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile2.png>)

Temporal encoder

N=128

[Figure 3] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile3.png>)

[Figure 4] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile4.png>)

encoderimage encoderimage encoderimage encoderimage … encoder encoderimage

image

Text encoder

[Figure 5] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile5.png>)

[Figure 6] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile6.png>)

[Figure 7] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile7.png>)

[Figure 8] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile8.png>)

[Figure 9] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile9.png>)

How does the subject's interaction with the surrounding tools and objects contribute to the primary goal of the video?

- Figure 2: BLIP-3-Video model architecture. It has an explicit temporal encoder inserted to BLIP-3.

frames). Such large number of video tokens could be critical for longer videos as the LLM computation is quadratic to the number of total tokens.

In this paper, we introduce BLIP-3-Video, which is an efficient compact vision-language model with an explicit temporal encoder, designed particularly for videos. BLIP-3-Video particularly focuses on incorporating a learnable ‘temporal encoder’ within it. We explore different types of temporal encoder, and demonstrate that the model can abstract each video into much fewer visual tokens (e.g., 16) while being successful in open-ended question-answering and captioning tasks. We include a space-time attentional pooling as well as a sequential model as our temporal encoder, relying on token operations to iteratively abstract a series of frame-level tokens into a learnable memory.

There has been prior work investigating the role of pooling [16], convolutions, and cross attention layers [56, 30, 27], but study on full space-time attentional pooling or sequential model to this extent has been limited in the past. Our objective in this paper is to provide a fundamental alternative to more brute-force way of collecting all the visual tokens which have been increasing popular recently. We experimentally confirm that 16 ∼ 32 video tokens abstracted by the temporal encoder is often sufficient to represent the entire video for question-answering (Figure 1).

## 2 BLIP-3-Video

### 2.1 Model architecture

We build BLIP-3-Video based on the image-based vision-language model (VLM), BLIP-3 [51]. The model architecture is composed of the following four components: (1) the vision encoder (ViT) taking each frame input, (2) the frame-level tokenizer to reduce the number of tokens, (3) the temporal encoder to build video-level token representations, and (4) the autoregressive LLM generating output text captions based on such video tokens and text prompt tokens. Figure 2 shows an overview.

First, we apply a pretrained SigLIP as the vision encoder, designed to take one single image frame at a time. Perceiver-Resampler is then applied to map such visual tokens into N = 128 visual tokens per frame, independently. Once the model has such visual tokens over time (i.e., over multiple frames in the video), they are provided to an explicit ‘temporal encoder’. The role of the temporal encoder is to build a video-level token representation from such sequence of image-level tokens, serving as a mapping function between a set of N · T image tokens to M video tokens where T is the number of frames and M is a constant number of tokens. We explore various forms of the temporal encoder, including temporal pooling as well as sequential models, which we discuss further in the following subsection. The resulting tokens are given to the LLM together with the encoded text tokens in a prefix manner, as in many standard VLMs.

For computational efficiency, the model takes uniformly sampled 8 frames per video. As a result, in our model, ViT first maps a video into 8 ∗ 729 visual tokens, which is then mapped to 8 ∗ 128 visual tokens using Perceiver-Resampler, and then to 16 ∼ 128 video tokens using the temporal encoder.

We use Phi-3 [1] as our LLM backbone taking such video tokens in addition to the text prompt tokens. This enables the model to take text+video as an input and generate text sentences as an output.

Time

Mean pool

x1 x2 x3

|v(1,1)| |
|---|---|
| | |

- x1

- x2

- x3

| | | | |
|---|---|---|---|
|Transformer| | | |

Ntokens

| | |
|---|---|
| | |

|v(3,4)| |
|---|---|
| | |

v(1,1) … v(3,4)

(a) Temporal pooling (b) Transformer-based

Learnable soft space-time selection

- x1

- x2

|h(1,4) h(1,2) h(2,1) h(2,3) h(3,2) h(3,4)<br><br>Sequential Model| |
|---|---|
| | |

| |
|---|

- x1

- x2

- x3

x3

- v(1,1)

- v(2,1)

- v(3,1)

- v(1,2)

- v(2,2)

- v(3,2)

- v(1,3)

- v(2,3)

- v(3,3)

- v(1,4)

- v(2,4)

- v(3,4)

(c) Attentional pooling (TokenLearner) (d) Sequential model (grouped)

- Figure 3: Visually comparing different types of temporal encoders we explored in our model architecture. (c) and (d) are particularly effective, as we discuss further in the experiments.

### 2.2 Temporal encoders

A temporal encoder is a function of tokens, taking N ·T tokens as an input and returning M tokens as an output: x1,...,M = f(v(1,1),...,(N,T)). We explore different types of encoders as part of our model. The simplest form of the temporal encoder will be temporal pooling, e.g., summating per-frame

tokens over time: x1,...,M = t(v(j,t)) Mj=1 where M is restricted to be identical to N, which was also used in [31]. Another possible implementation would be the use of a temporal Transformer, modeling the entire token sequence and selecting the last m tokens as in Mirasol3B [33]:

x1,...,M = {Transformer(v)}NN··TT−M+1 (1)

In addition to these temporal encoders, we explore two important temporal encoders considering space-time nature of tokens: spatio-temporal attentional pooling and sequential models (Figure 3).

Spatio-temporal attentional pooling: Attentional pooling allows learnable ‘soft selection’ of multiple tokens given a larger set of tokens. It was developed for Transformers (e.g., Perceiver [13] and TokenLearner [37]), and also used in earlier foundation models (e.g., CoCa [52]) for images. In our model, we use TokenLearner [37], which explicitly serves as our space-time aware temporal encoder. Unlike previous per-frame use of pooling where spatial pooling and temporal pooling are applied separately [31], our temporal encoder directly takes all N ·T tokens and ‘learns’ to soft-select M informative tokens spatio-temporally. Here, N tokens could be viewed as spatial representations of a frame and we have T of them, suggesting it is a spatio-temporal representation selection.

Our attentional pooling in its simplest form is expressed as:

xi = A(V ) · V = softmax α(V T) · V (2)

where V is a matrix formed by concatenating input tokens v(1,1),...,(N,T). The function A(·) computes the summation weights for V , performing soft selection of tokens. This is further decomposed to the softmax and the function α(·). In Perceiver, a matrix multiplication with a latent query tokens (i.e., cross attention where |Q| = m) have been used to implement this: α(V ) = Q · V T/c. TokenLearner uses a convolution/MLP on top of V : α(V ) = MLPm(V T), which we use in our model. This allows selecting a smaller number of tokens (e.g., M = 32 tokens).

We experimentally confirm that such learnable spatio-temporal attentional pooling has advantages over the conventional approach of non-learnable spatial pooling and temporal pooling, in Section 3.2.

### 2.3 Grouped tokens sequential model

Motivated by the success of sequential models in representing series of tokens such as Mamba [12] and Token Turing Machines [38], we design a new class of sequential models for videos and take advantage of it as our temporal encoder. The sequential models are capable of taking any number of inputs to generate a fixed number of output tokens, making it suitable to build a video-level token representation (e.g., M = 32 regardless the number of frames).

The main distinction between our new sequential models and the conventional sequential models is that our sequential model maintains a grouped memory, separately processing and maintaining different visual features. It is in a way analogous to the operational difference in the standard convolution vs. the grouped convolution, but it is done not over channels but over tokens and they are processed with multiple time steps.

Let F be the main model function in a standard sequential model: hi = F(hi−1,vi) where hi is the memory token and the index i ranges from 1 to N · T (i.e., the total number of tokens) in the entire video. In the new grouped formulation, we instead maintain the set of tokens at time step t as:

Ht = F(h(j,t−1),v(j,t)) Nj=1 (3)

where j is the visual token index within each frame. That is, we enforce the sequential model to focus on its group (specified with j) and maintain a ‘set’ of memory tokens for every time step t: Ht.

We tried both the grouped version of Mamba and TTM in our implementation as temporal encoder architectures, finding that our grouped tokens sequential model based on TTM functions better. In its instantiation, we first extend TTM by adding time-stamped positional encodings to embed the frame index of each token in the latent space. This enables the tokens in the ‘memory’ of TTM to preserve the temporal ordering information, which we found to be crucial when representing complicated or long video scenes.

Next, we implement Equation 3, while using the Read and Write operations of the original TTM. This follows our new grouped sequential model formulation, and the TTM now maintains a separate memory of size G = 4 for each of N = 128 tokens over time. That is, it maintains a ‘grouped’ memory, which we found to better preserve scene details. The memory is maintained to have the size of N · G, and the final output from the sequence model is attentionally pooled from the final memory to give M tokens.

In our experiments, we confirm that the proposed class of sequential models perform significantly better than the conventional sequential models like TTMs.

### 2.4 Training recipe

BLIP-3-Video follows a three-stage curriculum learning: (1) image caption pretraining, (2) video caption pretraining, and (3) video instruction tuning. In all its training we freeze the vision encoder, only training the model parameters after the vision encoder. First, we directly use the pretrained weights from BLIP-3 [51]. BLIP-3 is for images and it does not contain weights for the temporal encoder, so we randomly initialize those weights.

As its 2nd stage, the model is then trained on LLaVA-Hound-DPO’s video caption data [57], featuring over 900k video captions. Instead of directly using the text captions provided in LLaVA-Hound-DPO, we used GPT-4 to rephrase such text captions so that they become more GPT-style captions.

Finally, we tuned the model using a mix of video question-answering datasets, including VideoChat-

- GPT’s 99k-sample video instruction tuning data [31], along with the training splits of the MSVDQA [48], MSRVTT-QA [48], ActivityNet-QA [53], TGIF-QA [14], and NExT-QA [47] datasets, which contain 30k, 149k, 32k, 71k, and 34k samples, respectively. For TGIF-QA, we only used the training data associated with the Repeating Action and State Transition tasks. In our video instruction tuning recipe, we employ both open-ended and multiple-choice video QA formats for TGIF-QA and NExT-QA. For the open-ended video QA training data sourced from the MSVD-QA, MSRVTT-QA, TGIF-QA, and NExT-QA training sets, we used GPT-3.5 to rephrase the original single-word or single-phrase answer into a natural language sentence, providing the question in the LLM prompt context. For open-ended TGIF-QA and NExT-QA, we also double the sample size by using both the original short-phrase answers and the rephrased sentence-based answers. In addition, we added a

Table 1: Comparison against reported numbers of other models on open-ended question answering evaluation. The number of visual tokens are also reported. The numbers after ‘/’ are answer quality scores. ∗ indicates our evaluation using the checkpoint and inference code provided by the author, with the identical videos used in our model (8 frames of 384×384 resolution).

|Method<br><br>|Size|#tokens<br><br>|MSVD-QA|MSRVTT-QA|ActivityNet-QA<br><br>|TGIF-QA|
|---|---|---|---|---|---|---|
|VideoChat [25] Video-LLaMA [56] Video-ChatGPT [31] Chat-UniVi [16] LLaMA-VID [27] LLaMA-VID [27] Video-LLaVA [28] MiniGPT4-Video [3] PLLaVA [49] SlowFast-LLaVA [50] LLaVA-Hound-DPO [57] LLaVA-OneVision∗ [43] Tarsier [43] Tarsier ∗ [43]|7B 7B 7B 7B 7B<br><br>13B 7B 7B 7B 7B 7B 7B 7B 7B<br><br>|32 32<br><br>264+<br><br>112 32 32<br><br>2048<br><br>2880+ 576+ 3680 2048 1568<br><br>4608+ 4608|56.3 / 2.8 51.6 / 2.5 64.9 / 3.3 69.3 / 3.7<br><br>69.7 / 3.7<br><br>70.0 / 3.7<br><br>71.8 / 3.9<br><br>73.9 / 4.1<br><br>76.6 / 4.1<br><br>79.1 / 4.1<br><br>80.7 / 4.1<br><br><br>72.9 / 3.9<br><br>77.0 / 4.1<br><br><br>74.4 / 4.0<br><br><br><br><br>|45.0 / 2.5 29.6 / 1.8 49.3 / 2.8 55.0 / 3.1<br><br>57.7 / 3.2<br>58.9 / 3.3<br>59.2 / 3.5 59.7 / 3.3 62.0 / 3.5 65.8 / 3.6 70.2 / 3.7 57.8 / 3.4 62.0 / 3.5 59.1 / 3.4<br>|- / 2.2 12.4 / 1.1 34.2 / 2.8<br><br>46.1 / 3.3<br><br>47.4 / 3.3<br><br><br>47.5 / 3.3<br><br>45.3 / 3.3<br><br>46.3 / 3.4<br><br><br>56.3 / 3.5 56.3 / 3.4<br><br>- / 55.3 / 3.6 59.5 / 3.6 54.3 / 3.5<br><br>|34.4 / 2.3 - / 51.4 / 3.0<br><br>69.0 / 3.8<br><br>-<br><br>70.0 / 4.0 72.2 / 4.1<br><br><br>77.5 / 4.1<br>78.7 / 4.2 61.4 / 3.5 41.1 / 3.1<br>79.2 / 4.2<br><br><br>- / -|
|PLLaVA [49] LLaVA-NeXT-Video∗ [23] Tarsier [43] Tarsier ∗ [43]<br><br>|34B 32B 34B 34B|576+ 1152<br><br>4608+ 4608<br><br>|79.9 / 4.2 73.6 / 4.0<br>80.3 / 4.2 79.3 / 4.1<br>|68.7 / 3.8 56.8 / 3.4 66.4 / 3.7 62.2 / 3.5<br><br>|60.9 / 3.7<br><br>58.4 / 3.6<br><br>61.6 / 3.7<br><br><br>61.5 / 3.7<br><br>|80.6 / 4.3 73.5 / 4.1 82.5 / 4.4<br><br>- / -|
|BLIP-3-Video BLIP-3-Video<br><br>|4B 4B<br><br>|32 128|77.7 / 4.2 77.9 / 4.3<br><br>|60.0 / 3.6 59.7 / 3.6<br><br>|55.7 / 3.5<br>56.9 / 3.6<br>|76.5 / 4.3<br><br>77.1 / 4.3<br>|

filtered version of the Mira caption dataset [17] for our video instruction tuning. That is, we are using both video question-answering and video captioning for our final training. We excluded captions for Mira videos longer than one minute, totaling 935k video caption samples.

We trained our model with 8 × H100 GPUs. For the video caption pretraining, we use the batch size of 16 per GPU, 500 warmup steps, and the learning rate of 2e-5 with the cosine decay. We trained the model for 1 epoch. The video QA sft (i.e., instruction tuning) was done with the batch size of 4 per gpu, 500 warmup steps, and the learning rate of 1e-5 with the cosine decay. We trained the model for 1 epoch in this case as well. The entire training (combining both video pretraining and the sft) takes around 12 hours, confirming the efficiency of our model.

## 3 Experiments and Results

Model implementation details. We share the model details with BLIP-3 (4B), except that BLIP-3Video has the new temporal encoder component in its architecture. This model takes the video with the input resolution of 384×384, using SigLIP encoder to map it to 729 tokens per frame with the channel size 1152. Perceiver-Resampler is implemented with multiple cross-attention layers with the same channel dim, which is then given to the temporal encoder.

TokenLearner serving as the spatio-temporal attentional pooling was implemented using a MLP as the attention function. The size of its inner dim was the number of target tokens * 2. The grouped sequential model temporal encoder was implemented using 4 Transformer layers (with the channel dim of 1152) as the processor module while using TokenLearners for read/write modules. Memory size was set to N ∗ 4 = 512 tokens total. The resulting 16 ∼ 128 tokens are mapped to the channel dim of 3072, before given to the LLM (Phi-3).

### 3.1 Public benchmarks

We conducted experiments measuring video question-answering accuracies on multiple public datasets. This includes open-ended answer generation tasks like MSVD-QA, as well as multiple choice questions like NExT-QA and MVBench [26]. We follow their standard settings in all cases.

- Table 1 compare open-ended question answering accuracies of BLIP-3-Video against reported numbers of other models. We use four commonly used public datasets, MSVD-QA, MSRVTT-QA, ActivityNet-QA, and TGIF-QA, following standard VideoLLM evaluation settings. Note that our MSVD-QA and MSRVTT-QA accuracy was measured by training our model with a subset (i.e.,

- Table 2: Comparison against reported numbers of other models on NExT-QA.

|Method|Size<br><br>|#tokens|Acc.|
|---|---|---|---|
|LangRepo [18] LangRepo [18] Tarsier [43]|7B 12B 7B<br><br>|3136+ 3136+ 4608+|54.6 60.9 71.6<br><br>|
|LLoVi [55] IG-VLM [19] VideoAgent [44] VideoTree [45] Tarsier [43]<br><br>|157B<br><br>34B GPT-4 GPT-4<br><br>34B|1000s 1536+ 2091+ 3978+ 4608+<br><br>|67.7 70.9 71.3 73.5 79.2|
|BLIP-3-Video BLIP-3-Video|4B 4B<br><br>|32 128|76.4 77.1<br><br>|

Table 3: Comparison on MVBench. ‘VC-IT’ indicates whether the model was trained on the MVBenchprovided training dataset. ‘∼Y’ means the model’s training recipe includes a major subset of VideoChat2-IT (e.g., CLEVRER, Kinetics-710, SthSthV2, WebVid, ...).

|Method (Size)<br><br>|#tokens<br><br>|VC-IT|Acc.<br><br>|
|---|---|---|---|
|PLLaVA (7B) VideoLLaMA2 (7B) ST-LLM (7B) PPLLaVA (7B) VideoChat2-Mistral (7B) Kangaroo (8B) Tarsier (7B)|576+ 1152<br><br>256 1024 96 ∼10000 4608+<br><br>|Y Y<br><br>∼Y ∼Y<br><br>Y Y<br><br>∼Y|46.6 54.6 54.9 59.2 60.4 61.1 62.6<br><br>|
|VideoChatGPT (7B) VideoLLaMA (7B) VideoChat (7B) LLaMA-VID (7B) Video-LLaVA (7B) mPLUG-Owl3 (8B) LLaVA-OneVision (7B)<br><br>|264+ 32 32 32<br><br>2048<br><br>n/a 3136<br><br>|N N N N N N N|32.7 34.1 35.5 41.4 43.5 54.5 56.7<br><br>|
|BLIP-3-Video (4B)|32<br><br>|N|54.9|

Table 4: Ablations comparing different temporal encoders: 128 tokens. ∗A slightly different training recipe using a subset of the entire dataset (without Mira data) was used for the ablations.

|Encoder| |MSVD-QA<br><br>|TGIF-QA<br><br>|ActivityNet-QA| |NExT-QA|
|---|---|---|---|---|---|---|
|1 frame Mean pooling Transformer Perceiver-Resampler Vanilla Token Turing Machine| |71.49 / 4.01 76.75 / 4.17 76.24 / 4.15 76.17 / 4.12 76.42 / 4.15|72.74 / 4.16 77.01 /4.30 76.33 / 4.28 72.46 / 4.13 75.80 / 4.26<br><br>|51.83 / 3.39 55.89 / 3.53 55.59 / 3.50<br>52.61 / 3.38 54.45 / 3.48<br>| |72.79 76.24 76.34 76.44 75.42|
|Ours (Space-time) Ours (Sequential)| |77.49 / 4.18 77.86 / 4.20|76.90 / 4.29<br><br>77.10 / 4.31<br><br><br>|56.94 / 3.56 56.66 / 3.56| |76.27 77.07|

Video-ChatGPT dataset-only) of our training data, as this allows more direct comparison to some of the prior work and enables more stable results due to its data distribution. We are including the model size as well as the number of visual tokens in the table. We are able to observe that, despite its smaller size (i.e., 4B vs. 7B or 34B), our model is obtaining superior or comparable performance.

With the temporal encoder, BLIP-3-Video was able to retain the performance with much fewer tokens, which we discuss more in the following subsection. Our results suggest that not too many visual tokens are really necessary to be successful on these open-ended question answering benchmarks, as long as we have an effective temporal encoder.

In addition, we evaluated BLIP-3-Video’s ability to solve multiple choice questions (MCQ). Table 2 shows the results on NExT-QA, and Table 3 shows the results on MVBench. Due to the nature of its questions requiring understanding of multiple frames, many prior models use quite a bit of tokens. For instance, GPT-4 uses a minimum of 255 tokens per frame. It is interesting that BLIP-3-Video achieves comparable accuracy while representing the entire video with only 32 (or 128) tokens.

### 3.2 Ablations

Temporal encoder: We conducted an ablation comparing different temporal encoders within our model. These include: (1) the base single frame model (i.e., BLIP-3 trained with videos), (2) mean pooling similar to Video-ChatGPT, and (3) transformer temporal encoder similar to Mirasol3B. We also tried (4) the approach of directly extending the spatial Perceiver-Resampler to do spatio-temporal encoding, and (5) vanilla Token Turing Machines, which is not our grouped sequential model.

- Table 4 shows the result, comparing the question-answering accuracies of different types of temporal encoders when abstracting a video into 128 tokens. We are able to observe that our sequential model temporal encoder performs the best overall. In particular, directly extending Perceiver-Resampler performed poorly compared to the others.

- Table 5: Ablations comparing different pooling strategies (32).

Table 6: Ablations comparing different # of tokens. Ours with sequential model as a temporal encoder was used.

|Encoder<br><br>|MSVD-QA|
|---|---|
|Space-time pooling (4*8) Per-frame (4*8)<br><br>|76.04 76.78|
|Ours (Space-time) Ours (Sequential)<br><br>|77.07 77.11|

|# tokens| |MSVD-QA<br><br>|TGIF-QA<br><br>|NExT-QA|
|---|---|---|---|---|
|16 tokens 32 tokens 128 tokens 256 tokens| |76.17 / 4.16<br>77.11 / 4.17 77.86 / 4.20 77.67 / 4.18<br>|76.19 / 4.28<br><br>77.07 / 4.30<br><br><br>77.10 / 4.31 77.35 / 4.31<br><br>|75.8 76.4 77.07 77.06|

Table 7: Ablations on sequential models (128).

|Temporal encoder| |MSVD-QA|ActNet-QA<br><br>|NExT-QA|
|---|---|---|---|---|
|Original TTM TTM + time-stamp TTM + grouped Ours| |76.42 / 4.15<br><br>76.43 / 4.16<br><br><br>76.99 / 4.17<br><br>77.86 / 4.20<br><br><br>|54.45 / 3.48 56.15 / 3.53<br>55.92 / 3.54<br>56.66 / 3.56<br>|75.42 75.96 76.46 77.07<br><br>|

Table 8: Ablations on the number of frames

|# frames<br><br>|# tokens| |NExT-QA<br><br>|ActNet-QA|
|---|---|---|---|---|
|8 frames 8 frames 16 frames 16 frames|32 tokens 128 tokens 32 tokens 128 tokens<br><br>| |76.4 77.1 76.7 77.6<br><br>|55.7 / 3.5<br>56.7 / 3.6 55.9 / 3.5<br>57.3 / 3.6<br>|

Vs. pooling: We compared against different pooling approaches similar to the ones tried in prior works, when they are required to select a much smaller number of tokens (e.g., 32) from a large set of visual tokens. We compare our spatio-temporal attentional pooling as well as the sequential model against its alternatives, including (1) fixed-window (non-learnable) space-time pooling and (2) learnable ‘per-frame’ pooling. In particular, (2) is similar to the approach taken in LLaMA-VID [27], which independently selected a fixed number of tokens (e.g., 2) per frame. Table 5 shows the results.

Number of visual tokens: Table 6 explicitly compares the impact of having smaller visual tokens. We are able to observe that 32 visual tokens or more gives a reasonable video QA accuracy.

Grouped token sequential model: Our grouped token sequential model temporal encoder implementation extends the conventional Token Turing Machines. Table 7 shows the ablations confirming the benefits of the new grouped token sequential model formulation. As we observe, the original TTM performs poorly. Adding the time-stamped positional encoding and memory grouping in the sequential model enables much better results, confirming their importance in the real-world models.

More frames: In this experiment we validate whether BLIP-3-Video is able to scale when trained to take more frames. Table 8 shows the results. Even while maintaining the number of tokens, we are able to observe that providing more frames in the input allows BLIP-3-Video to scale to better performance. We believe this is due to the fact that increasing the number of frames has an effect of increasing the size of the “pool” of tokens the temporal encoder can select from. We believe this trend will continue until it saturates.

Speed: Reducing the number of visual tokens increases the computational efficiency of the models, as the total computation is quadratic to the number of tokens fed to the LLM. We measure the runtime of our models in the training setting for the fair comparison. Here, we report ‘samples per second per

- GPU’. Without the temporal encoder (i.e., directly using 1024 visual tokens), the model processed

- 3.3 samples per second. With 16/32/128 tokens using the temporal encoder, the model was able to process 8.5 / 8.2 / 7.5 samples per second.

### 3.3 Video captioning evaluation

We evaluate our model on the video captioning task by comparing it against state-of-the-art models on the test splits of MSVD-Caption and MSRVTT-Caption, as well as a custom evaluation split from the

Table 9: Video caption evaluation results using 8 frames. We employ VideoChatGPT’s LLM evaluation and report Average Accuracy / Average Score. The ‘captioning-only model’ was trained only using Mira video caption data (without QA data), making it specialized for the captioning task.

|Method<br><br>|Size<br><br>|# tokens| |MSVD-Cap|MSRVTT-Cap<br><br>|Mira-Cap|
|---|---|---|---|---|---|---|
|LLaVA-OneVision Tarsier<br><br>|7B 7B|1152 4608<br><br>| |61.62 / 3.31<br><br>62.26 / 3.37<br><br><br>|38.60 / 2.71 40.27 / 2.77<br><br>|48.83 / 3.10 40.55 / 2.87|
|BLIP-3-Video BLIP-3-Video BLIP-3-Video (captioning-only model)|4B 4B 4B<br><br>|32 128 128<br><br>| |63.59 / 3.38<br>64.17 / 3.41 69.50 / 3.52<br>|42.06 / 2.82<br><br>43.05 / 2.85<br><br><br>50.45 / 2.98<br><br>|80.67 / 3.96<br>81.13 / 3.97 81.76 / 4.00<br>|

[Figure 10] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile10.png>)

[Figure 11] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile11.png>)

[Figure 12] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile12.png>)

[Figure 13] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile13.png>)

[Figure 14] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile14.png>)

Question: Please describe the primary object or subject in the video, capturing their attributes, actions, positions, and movements. GT: A white jellyfish, is seen floating and gently pulsating as it moves through the water. Its bell contracts and expands rhythmically, propelling it forward. The tentacles trail behind, undulating with the water's movement. Throughout the sequence, the jellyfish maintains a steady, graceful motion, drifting from left to right and slightly rotating as it progresses. Prediction: A jellyfish, is seen swimming in the water. It moves in a slow, undulating motion, with its bell-shaped body contracting and expanding to propel itself forward. The jellyfish's tentacles trail behind it, creating a delicate, flowing effect. Throughout the video, the jellyfish maintains a consistent upward and forward movement, occasionally adjusting its direction slightly.

[Figure 15] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile15.png>)

[Figure 16] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile16.png>)

[Figure 17] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile17.png>)

[Figure 18] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile18.png>)

[Figure 19] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile19.png>)

Question: Please provide a dense description of the video, covering the main subject, movements, style, backgrounds, and cameras. GT: The video provides a breathtaking aerial view of a historic castle perched on the edge of a steep cliff. As the drone camera glides over the scene, viewers can appreciate the castle's ancient stone walls and the surrounding lush greenery, which contrasts sharply with the rugged cliff faces. The castle, a relic of medieval architecture, is shown in various states of preservation, with some parts reduced to ruins while others stand robustly. The video captures the castle's strategic location, offering panoramic views of the surrounding landscape and the churning waters below. Prediction: The video provides an aerial view of a historic fortification perched on a rugged cliff overlooking the ocean. The fort, characterized by its robust stone walls and strategic positioning, includes various defensive structures such as cannons and barricades. As the video progresses, the camera captures the changing light of the setting sun, casting long shadows and highlighting the textures of the stone and the surrounding landscape. The fort's layout, including pathways and small buildings, is revealed in greater detail, showcasing its historical significance and architectural complexity.

[Figure 20] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile20.png>)

[Figure 21] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile21.png>)

[Figure 22] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile22.png>)

[Figure 23] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile23.png>)

[Figure 24] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile24.png>)

Question: Please provide a dense description of the video, covering the main subject, movements, style, backgrounds, and cameras. GT: A step-by-step guide on assembling a wooden bed frame, featuring a person methodically working through the construction process. Initially, the person is seen preparing the wooden planks and arranging the necessary tools. As the video progresses, the individual meticulously attaches and secures the slats to the frame, ensuring stability and precision. The final scenes depict the person tightening the last screws and reviewing the completed structure, demonstrating a sense of accomplishment and attention to detail. Prediction: A step-by-step process of assembling a piece of furniture, specifically a bed frame. The person involved is seen working on a wooden bed frame, using various tools and components to construct the bed. The video likely serves as a tutorial, providing viewers with a visual guide on how to assemble the bed frame themselves. The person's actions are deliberate and precise, indicating a level of expertise or familiarity with the task.

[Figure 25] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile25.png>)

[Figure 26] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile26.png>)

[Figure 27] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile27.png>)

[Figure 28] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile28.png>)

[Figure 29] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile29.png>)

Question: Please provide a dense description of the video, covering the main subject, movements, style, backgrounds, and cameras. GT: Two men engaged in the task of loading a barrel onto a small, yellow floatplane near a body of water. The men are dressed in casual attire suitable for outdoor work, with one wearing gloves for handling the barrel. The plane is docked at a wooden structure, possibly a jetty or makeshift dock, and the men coordinate their efforts to lift and maneuver the barrel into the cargo area of the aircraft. The surrounding environment is serene and natural, with a forested backdrop suggesting a remote or wilderness setting. Prediction: A sequence of events where two men are engaged in loading a barrel onto a small, yellow seaplane parked on a wooden dock beside a serene lake. The men, dressed in casual work attire, appear to be coordinating their efforts to secure the barrel, which is large and cylindrical, onto the aircraft. The setting is tranquil, with the calm waters of the lake reflecting the clear blue sky and the lush greenery of the surrounding landscape. The video conveys a sense of teamwork and the meticulous nature of preparing an aircraft for a journey.

- Figure 4: Example video captioning results on Mira dataset, formed in question-answering style.

Mira dataset. For the Mira dataset, we randomly selected 6,000 samples from our full, filtered data to create the evaluation split, with the remainder used for training. We employed Video-ChatGPT’s LLM evaluation, specifically using GPT-3.5 to compare model-predicted captions with ground truth captions. The LLM assesses accuracy by checking if the predicted caption matches the ground truth, and assigns a score on a scale of 0 to 5 for each sample.

Table 9 demonstrates the results. All three models were provided with 8 frames per video, and consistent visual input and prompts were ensured across the models. Our BLIP-3-Video consistently outperforms LLaVA-OneVision-7B and Tarsier-7B across all three video captioning benchmarks, with particularly notable improvements on the Mira video captioning task.

We present qualitative video captioning results for the Mira dataset in Figure 4 (and more in Appendix). BLIP-3-Video generates high-quality, detailed captions.

### 3.4 Longer video datasets

We conducted experiments to confirm BLIP-3-Video’s capability to help long video understanding. We combined BLIP-3-Video’s captioning with a hierarchical text-based video understanding mechanism (i.e., LVNet [32]), and evaluated it on public benchmarks including EgoSchema and VideoMME. We use LVNet [32] as our backbone, relying on its key frame selection by extracting 8 frames centered at those key frames. BLIP-3-Video was applied to such 8-frame segments for each

Table 10: VideoMME long split results.

|Method<br><br>|Accuracy (%)|
|---|---|
|LongVILA VideoChat-T LLaVA-OneVision LLaVA-NexT-Video VideoAgent Frame-Voyager VideoTree<br><br>|38.8 41.9 43.8 44.3 46.4 51.2 53.1|
|LVNet LVNet + BLIP-3-Video|52.4 55.6<br><br>|
|LLaVA-Video InternVL2.5 Qwen2-VL|61.5 62.6 62.2<br><br>|

Table 11: EgoSchema subset results. LVNet* accuracy is obtained by replicating the LVNet results by running its code with GPT-4o as LLM.

|Method<br><br>|Accuracy (%)|
|---|---|
|LLoVi VideoAgent VideoTree|57.6 60.2 66.2<br><br>|
|LVNet* LVNet* + BLIP-3-Video|66.2 67.4<br><br>|
|LifelongMemory Tarsier<br><br>|68.0 68.6|

key frame, serving as the video clip captioning model, and the generated captions were given to the high-level LLM in the LVNet framework.

Tables 10 and 11 compares the results. As we observe, compared to the backbone framework used (LVNet), BLIP-3-Video meaningfully improves its accuracy. It captures video information in the short intervals with LVNet, thereby enabling better question answering with long videos. Note that despite BLIP-3-Video was not trained with long videos, it shows promising results.

## 4 Related Work

Image-Text LLMs. Recent image-text multimodal models [24, 2, 29, 8, 51, 21, 9] typically use a pre-trained image encoder (e.g., ViT [34, 54]) and a language-only LLM [1, 5, 10], connected via a vision-language connector that maps visual embeddings into LLM-compatible tokens. These tokens match the shape of language embeddings, allowing joint training via next-token prediction. Connector designs vary: BLIP-2 [24] uses a Q-Former, Flamingo [2] uses a perceiver resampler and cross-attention, and others simply adopt MLPs [29]. Training is often multi-stage—pre-training, instruction tuning, and post-training (e.g., DPO [35])—on both structured (e.g., captioning, VQA) and unstructured data (e.g., interleaved image-text [20, 4], multi-image VQA [15, 22]).

Video LLMs. Video LLMs extend image-based LLMs to handle video input. [56] uses frozen encoders and LLMs with Video/Audio Q-Formers for modality alignment. [31] extends CLIP with temporal features and fine-tunes on video-instruction pairs collected via tools like BLIP-2 [24] and GRiT [46]. [27] compresses frame-level features into two tokens per frame but lacks temporal recency modeling. Video-LLaVA [28] and LLaVa-OneVision [22] treat videos as multi-image sequences, sacrificing compute efficiency. SlowFast-LLaVA [50] introduces dual slow-fast pathways without extra fine-tuning. LLaVa-hound-DPO [57] explores DPO [35] with GPT-4V-generated preference data. Our BLIP-3-Video explores the orthogonal direction of efficiency via reduced tokens.

Token Pruning. Token pruning reduces redundancy in ViTs and LLMs. [6, 36] merges similar tokens within ViTs. [40] focus on temporal redundancy and progressively merges tokens across neighboring clips. [7] prunes or merges tokens in deeper layers based on attention scores. [39] proposes adaptive token reduction via important token selection and supplementation. In LLMs, KV cache pruning improves efficiency [11], and [42] extends this to VLMs with token merging strategies. LongVU [41] spatially compresses frame tokens conditioned on the first frame, orthogonal to temporal reduction. Our BLIP-3-Video utilizes a novel temporal encoder to represent videos in as few as 16 or 32 tokens.

## 5 Conclusion

We introduce BLIP-3-Video, which is an efficient, compact vision-language model for videos with 4B parameters. BLIP-3-Video incorporates a temporal encoder in its architecture, which allows the model to abstract the entire video with as few as 16 or 32 tokens. In contrast to many state-of-the-art video VLMs taking advantage of thousands of visual tokens to represent a video (e.g., 4608), BLIP-3-Video shows a competitive performance while utilizing much fewer visual tokens (e.g., 32).

## References

- [1] Abdin, M., Jacobs, S. A., Awan, A. A., Aneja, J., Awadallah, A., Awadalla, H., Bach, N., Bahree, A., Bakhtiari, A., Bao, J., Behl, H., Benhaim, A., Bilenko, M., Bjorck, J., Bubeck, S., Cai, Q., Cai, M., Mendes, C. C. T., Chen, W., Chaudhary, V., Chen, D., Chen, D., Chen, Y.-C., Chen, Y.-L., Chopra, P., Dai, X., Giorno, A. D., de Rosa, G., Dixon, M., Eldan, R., Fragoso, V., Iter, D., Gao, M., Gao, M., Gao, J., Garg, A., Goswami, A., Gunasekar, S., Haider, E., Hao, J., Hewett, R. J., Huynh, J., Javaheripi, M., Jin, X., Kauffmann, P., Karampatziakis, N., Kim, D., Khademi, M., Kurilenko, L., Lee, J. R., Lee, Y. T., Li, Y., Li, Y., Liang, C., Liden, L., Liu, C., Liu, M., Liu, W., Lin, E., Lin, Z., Luo, C., Madan, P., Mazzola, M., Mitra, A., Modi, H., Nguyen, A., Norick, B., Patra, B., Perez-Becker, D., Portet, T., Pryzant, R., Qin, H., Radmilac, M., Rosset, C., Roy, S., Ruwase, O., Saarikivi, O., Saied, A., Salim, A., Santacroce, M., Shah, S., Shang, N., Sharma, H., Shukla, S., Song, X., Tanaka, M., Tupini, A., Wang, X., Wang, L., Wang, C., Wang, Y., Ward, R., Wang, G., Witte, P., Wu, H., Wyatt, M., Xiao, B., Xu, C., Xu, J., Xu, W., Yadav, S., Yang, F., Yang, J., Yang, Z., Yang, Y., Yu, D., Yuan, L., Zhang, C., Zhang, C., Zhang, J., Zhang, L. L., Zhang, Y., Zhang, Y., Zhang, Y., and Zhou, X. Phi-3 technical report: A highly capable language model locally on your phone, 2024. URL https://arxiv.org/abs/2404.14219.
- [2] Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., Ring, R., Rutherford, E., Cabi, S., Han, T., Gong, Z., Samangooei, S., Monteiro, M., Menick, J., Borgeaud, S., Brock, A., Nematzadeh, A., Sharifzadeh, S., Binkowski, M., Barreira, R., Vinyals, O., Zisserman, A., and Simonyan, K. Flamingo: a visual language model for few-shot learning. In Advances in neural information processing systems, 2022.
- [3] Ataallah, K., Shen, X., Abdelrahman, E., Sleiman, E., Zhu, D., Ding, J., and Elhoseiny, M. Minigpt4-video: Advancing multimodal llms for video understanding with interleaved visualtextual tokens. arXiv preprint arXiv:2404.03413, 2024.
- [4] Awadalla, A., Xue, L., Lo, O., Shu, M., Lee, H., Guha, E. K., Jordan, M., Shen, S., Awadalla, M., Savarese, S., Xiong, C., Xu, R., Choi, Y., and Schmidt, L. Mint-1t: Scaling opensource multimodal data by 10x: A multimodal dataset with one trillion tokens, 2024. URL https://arxiv.org/abs/2406.11271.
- [5] Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., Hui, B., Ji, L., Li, M., Lin, J., Lin, R., Liu, D., Liu, G., Lu, C., Lu, K., Ma, J., Men, R., Ren, X., Ren,

- X., Tan, C., Tan, S., Tu, J., Wang, P., Wang, S., Wang, W., Wu, S., Xu, B., Xu, J., Yang, A., Yang, H., Yang, J., Yang, S., Yao, Y., Yu, B., Yuan, H., Yuan, Z., Zhang, J., Zhang, X., Zhang,
- Y., Zhang, Z., Zhou, C., Zhou, J., Zhou, X., and Zhu, T. Qwen technical report, 2023. URL https://arxiv.org/abs/2309.16609.

- [6] Bolya, D., Fu, C.-Y., Dai, X., Zhang, P., Feichtenhofer, C., and Hoffman, J. Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461, 2022.
- [7] Chen, L., Zhao, H., Liu, T., Bai, S., Lin, J., Zhou, C., and Chang, B. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. arXiv preprint arXiv:2403.06764, 2024.
- [8] Dai, W., Li, J., Li, D., Tiong, A. M. H., Zhao, J., Wang, W., Li, B., Fung, P., and Hoi, S. C. H. Instructblip: Towards general-purpose vision-language models with instruction tuning. In NeurIPS, 2023.
- [9] Deitke, M., Clark, C., Lee, S., Tripathi, R., Yang, Y., Park, J. S., Salehi, M., Muennighoff, N., Lo, K., Soldaini, L., Lu, J., Anderson, T., Bransom, E., Ehsani, K., Ngo, H., Chen, Y., Patel, A., Yatskar, M., Callison-Burch, C., Head, A., Hendrix, R., Bastani, F., VanderBilt, E., Lambert, N., Chou, Y., Chheda, A., Sparks, J., Skjonsberg, S., Schmitz, M., Sarnat, A., Bischoff, B., Walsh, P., Newell, C., Wolters, P., Gupta, T., Zeng, K.-H., Borchardt, J., Groeneveld, D., Dumas, J., Nam, C., Lebrecht, S., Wittlif, C., Schoenick, C., Michel, O., Krishna, R., Weihs, L., Smith, N. A., Hajishirzi, H., Girshick, R., Farhadi, A., and Kembhavi, A. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models, 2024. URL https://arxiv.org/abs/2409.17146.
- [10] Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., Goyal, A., Hartshorn, A., Yang, A., Mitra, A., Sravankumar, A., et al. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.

- [11] Fu, Q., Cho, M., Merth, T., Mehta, S., Rastegari, M., and Najibi, M. Lazyllm: Dynamic token pruning for efficient long context llm inference. arXiv preprint arXiv:2407.14057, 2024.
- [12] Gu, A. and Dao, T. Mamba: Linear-time sequence modeling with selective state spaces, 2024. URL https://arxiv.org/abs/2312.00752.
- [13] Jaegle, A., Borgeaud, S., Alayrac, J., Doersch, C., Ionescu, C., Ding, D., Koppula, S., Zoran, D., Brock, A., Shelhamer, E., H´enaff, O. J., Botvinick, M. M., Zisserman, A., Vinyals, O., and Carreira, J. Perceiver IO: A general architecture for structured inputs & outputs. In ICLR, 2022.
- [14] Jang, Y., Song, Y., Yu, Y., Kim, Y., and Kim, G. Tgif-qa: Toward spatio-temporal reasoning in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2758–2766, 2017.
- [15] Jiang, D., He, X., Zeng, H., Wei, C., Ku, M. W., Liu, Q., and Chen, W. Mantis: Interleaved multi-image instruction tuning, 2024.
- [16] Jin, P., Takanobu, R., Zhang, W., Cao, X., and Yuan, L. Chat-univi: Unified visual representation empowers large language models with image and video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13700–13710, 2024.
- [17] Ju, X., Gao, Y., Zhang, Z., Yuan, Z., Wang, X., Zeng, A., Xiong, Y., Xu, Q., and Shan, Y. Miradata: A large-scale video dataset with long durations and structured captions, 2024. URL https://arxiv.org/abs/2407.06358.
- [18] Kahatapitiya, K., Ranasinghe, K., Park, J., and Ryoo, M. S. Language repository for long video understanding, 2024. URL https://arxiv.org/abs/2403.14622.
- [19] Kim, W., Choi, C., Lee, W., and Rhee, W. An image grid can be worth a video: Zero-shot video question answering using a vlm, 2024. URL https://arxiv.org/abs/2403.18406.
- [20] Laurenc¸on, H., Saulnier, L., Tronchon, L., Bekman, S., Singh, A., Lozhkov, A., Wang, T., Karamcheti, S., Rush, A. M., Kiela, D., Cord, M., and Sanh, V. OBELICS: an open web-scale filtered dataset of interleaved image-text documents. In NeurIPS, 2023.
- [21] Laurenc¸on, H., Marafioti, A., Sanh, V., and Tronchon, L. Building and better understanding vision-language models: insights and future directions., 2024.
- [22] Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Li, Y., Liu, Z., and Li, C. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [23] Li, F., Zhang, R., Zhang, H., Zhang, Y., Li, B., Li, W., Ma, Z., and Li, C. Llava-next: Tackling multi-image, video, and 3d in large multimodal models, June 2024. URL https: //llava-vl.github.io/blog/2024-06-16-llava-next-interleave/.
- [24] Li, J., Li, D., Savarese, S., and Hoi, S. C. H. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, volume 202 of Proceedings of Machine Learning Research, pp. 19730–19742. PMLR, 2023.
- [25] Li, K., He, Y., Wang, Y., Li, Y., Wang, W., Luo, P., Wang, Y., Wang, L., and Qiao, Y. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023.
- [26] Li, K., Wang, Y., He, Y., Li, Y., Wang, Y., Liu, Y., Wang, Z., Xu, J., Chen, G., Luo, P., Wang, L., and Qiao, Y. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
- [27] Li, Y., Wang, C., and Jia, J. Llama-vid: An image is worth 2 tokens in large language models. European Conference on Computer Vision, 2024.
- [28] Lin, B., Zhu, B., Ye, Y., Ning, M., Jin, P., and Yuan, L. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023.
- [29] Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning, 2023.
- [30] Liu, J., Wang, Y., Ma, H., Wu, X., Ma, X., Wei, X., Jiao, J., Wu, E., and Hu, J. Kangaroo: A powerful video-language model supporting long-context video input, 2024. URL https: //arxiv.org/abs/2408.15542.
- [31] Maaz, M., Rasheed, H., Khan, S., and Khan, F. S. Video-chatgpt: Towards detailed video understanding via large vision and language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024), 2024.

- [32] Park, J. S., Ranasinghe, K., Kahatapitiya, K., Ryoo, W., Kim, D., and Ryoo, M. S. Too many frames, not all useful: Efficient strategies for long-form video qa. ArXiv, abs/2406.09396, 2024. URL https://api.semanticscholar.org/CorpusID:270440923.
- [33] Piergiovanni, A., Noble, I., Kim, D., Ryoo, M. S., Gomes, V., and Angelova, A. Mirasol3b: A multimodal autoregressive model for time-aligned and contextual modalities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
- [34] Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision. In ICML, volume 139 of Proceedings of Machine Learning Research, pp. 8748–8763. PMLR, 2021.
- [35] Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.
- [36] Ren, S., Chen, S., Li, S., Sun, X., and Hou, L. Testa: Temporal-spatial token aggregation for long-form video-language understanding. arXiv preprint arXiv:2310.19060, 2023.
- [37] Ryoo, M., Piergiovanni, A., Arnab, A., Dehghani, M., and Angelova, A. Tokenlearner: Adaptive space-time tokenization for videos. In NeurIPS, volume 34, pp. 12786–12797, 2021.
- [38] Ryoo, M. S., Gopalakrishnan, K., Kahatapitiya, K., Xiao, T., Rao, K., Stone, A., Lu, Y., Ibarz, J., and Arnab, A. Token turing machines. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.
- [39] Shang, Y., Cai, M., Xu, B., Lee, Y. J., and Yan, Y. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. arXiv preprint arXiv:2403.15388, 2024.
- [40] Shen, L., Hao, T., Zhao, S., Zhang, Y., Liu, P., Bao, Y., and Ding, G. Tempme: Video temporal token merging for efficient text-video retrieval. arXiv preprint arXiv:2409.01156, 2024.
- [41] Shen, X., Xiong, Y., Zhao, C., Wu, L., Chen, J., Zhu, C., Liu, Z., Xiao, F., Varadarajan, B., Bordes, F., Liu, Z., Xu, H., Kim, H. J., Soran, B., Krishnamoorthi, R., Elhoseiny, M., and Chandra, V. Longvu: Spatiotemporal adaptive compression for long video-language understanding, 2024. URL https://arxiv.org/abs/2410.17434.
- [42] Wan, Z., Wu, Z., Liu, C., Huang, J., Zhu, Z., Jin, P., Wang, L., and Yuan, L. Look-m: Lookonce optimization in kv cache for efficient multimodal long-context inference. arXiv preprint arXiv:2406.18139, 2024.
- [43] Wang, J., Yuan, L., and Zhang, Y. Tarsier: Recipes for training and evaluating large video description models. arXiv preprint arXiv:2407.00634, 2024.
- [44] Wang, X., Zhang, Y., Zohar, O., and Yeung-Levy, S. Videoagent: Long-form video understanding with large language model as agent, 2024. URL https://arxiv.org/abs/2403.10517.
- [45] Wang, Z., Yu, S., Stengel-Eskin, E., Yoon, J., Cheng, F., Bertasius, G., and Bansal, M. Videotree: Adaptive tree-based video representation for llm reasoning on long videos, 2024. URL https: //arxiv.org/abs/2405.19209.
- [46] Wu, J., Wang, J., Yang, Z., Gan, Z., Liu, Z., Yuan, J., and Wang, L. Grit: A generative region-to-text transformer for object understanding. arXiv preprint arXiv:2212.00280, 2022.
- [47] Xiao, J., Shang, X., Yao, A., and Chua, T.-S. Next-qa: Next phase of question-answering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9777–9786, 2021.
- [48] Xu, D., Zhao, Z., Xiao, J., Wu, F., Zhang, H., He, X., and Zhuang, Y. Video question answering via gradually refined attention over appearance and motion. In ACM Multimedia, 2017.
- [49] Xu, L., Zhao, Y., Zhou, D., Lin, Z., Ng, S. K., and Feng, J. Pllava : Parameter-free llava extension from images to videos for video dense captioning, 2024.
- [50] Xu, M., Gao, M., Gan, Z., Chen, H.-Y., Lai, Z., Gang, H., Kang, K., and Dehghan, A. Slowfast-llava: A strong training-free baseline for video large language models. arXiv preprint arXiv:2407.15841, 2024.
- [51] Xue, L., Shu, M., Awadalla, A., Wang, J., Yan, A., Purushwalkam, S., Zhou, H., Prabhu, V., Dai, Y., Ryoo, M. S., et al. xgen-mm (blip-3): A family of open large multimodal models. arXiv preprint arXiv:2408.08872, 2024.

- [52] Yu, J., Wang, Z., Vasudevan, V., Yeung, L., Seyedhosseini, M., and Wu, Y. Coca: Contrastive captioners are image-text foundation models. arXiv preprint arXiv:2205.01917, 2022.
- [53] Yu, Z., Xu, D., Yu, J., Yu, T., Zhao, Z., Zhuang, Y., and Tao, D. Activitynet-qa: A dataset for understanding complex web videos via question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 9127–9134, 2019.
- [54] Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. Sigmoid loss for language image pretraining. In ICCV, pp. 11941–11952. IEEE, 2023.
- [55] Zhang, C., Lu, T., Islam, M. M., Wang, Z., Yu, S., Bansal, M., and Bertasius, G. A simple llm framework for long-range video question-answering, 2024. URL https://arxiv.org/abs/ 2312.17235.
- [56] Zhang, H., Li, X., and Bing, L. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023.
- [57] Zhang, R., Gui, L., Sun, Z., Feng, Y., Xu, K., Zhang, Y., Fu, D., Li, C., Hauptmann, A., Bisk, Y., and Yang, Y. Direct preference optimization of video large multimodal models from language model reward, 2024.

## A Appendix

### A.1 More qualitative examples

Figure 5 shows BLIP-3-Video’s captioning results on the MSVD and MSRVTT datasets. We are able to observe that it generates reasonable captions despite utilizing a small number of visual tokens, e.g., 32.

Figures 6 and 7 compares the video captioning results of BLIP-3-Video against Tarsier and LLaVAOneVision, in detail. The models take the exact same video input with the same number of frames, for a fair comparison. We observe that BLIP-3-Video generate captioning much more similar to the ground truth. It also is able to provide more holistic understanding of the video (rather than focusing on unnecessary details such as texts in the background) while suffering from less hallucinations.

[Figure 30] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile30.png>)

[Figure 31] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile31.png>)

[Figure 32] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile32.png>)

[Figure 33] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile33.png>)

[Figure 34] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile34.png>)

Question: What’s the video about in brief? Prediction: A person is using a citrus juicer to extract juice from a lemon on a wooden cutting board. The person’s hands are visible, and they are wearing a black sleeve. The lemon is placed on the cutting board, and the person squeezes it to extract the juice, which is collected in a white bowl. The process is methodical and focused, with the person ensuring to extract as much juice as possible.

[Figure 35] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile35.png>)

[Figure 36] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile36.png>)

[Figure 37] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile37.png>)

[Figure 38] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile38.png>)

[Figure 39] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile39.png>)

Question: Can you briefly tell me what's in the video? Prediction: A person is preparing shrimp for cooking. They start by placing the shrimp in a glass bowl and adding a white powdery substance, likely salt or flour. The person then mixes the shrimp and the powder together with their hands, ensuring the shrimp are evenly coated. Finally, the person transfers the shrimp onto a cutting board, ready for the next step in the cooking process.

[Figure 40] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile40.png>)

[Figure 41] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile41.png>)

[Figure 42] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile42.png>)

[Figure 43] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile43.png>)

[Figure 44] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile44.png>)

Question: What’s the main gist of the video? Prediction: The video depicts a dramatic encounter between a scuba diver and an octopus in the ocean. The octopus, with its tentacles spread wide, appears to be approaching the diver, who is equipped with a camera and other diving gear. The octopus's movements are fluid and deliberate, showcasing its ability to maneuver through the water with ease. The diver remains relatively stationary, observing the octopus's behavior.

[Figure 45] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile45.png>)

[Figure 46] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile46.png>)

[Figure 47] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile47.png>)

[Figure 48] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile48.png>)

[Figure 49] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile49.png>)

Question: Can you briefly tell me what's in the video?

Prediction: A wrestling match between two athletes on a red mat. The wrestlers are engaged in a series of competitive maneuvers, attempting to gain control over each other. The sequence shows the wrestlers in various positions, from standing to on the ground, as they grapple and execute moves. The intensity of the match is evident through their physical exertion and strategic positioning.

Figure 5: Example video captioning results on MSVD and MSRVTT caption dataset.

### A.2 More experimental results

We tested our model on additional benchmarks: VideoInstruct (which was used in VideoChatGPT) and TempCompass. Table 12 shows the results on the VideoInstruct benchmark. We believe BLIP3-Video performs very reasonably on all these benchmarks, considering its smaller size and its use of much fewer visual tokens. Similarly, Table 13 shows BLIP-3-Video’s results on ‘Yes/No QA’ and ‘Caption matching’ tasks in TempCompass. They show similar trends to the other experiments. BLIP-3-Video is a compact model giving us a reasonable accuracy.

[Figure 50] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile50.png>)

[Figure 51] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile51.png>)

[Figure 52] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile52.png>)

[Figure 53] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile53.png>)

[Figure 54] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile54.png>)

Question: Please describe the primary object or subject in the video, capturing their attributes, actions, positions, and movements. GT: A white jellyfish, is seen floating and gently pulsating as it moves through the water. Its bell contracts and expands rhythmically, propelling it forward. The tentacles trail behind, undulating with the water's movement. Throughout the sequence, the jellyfish maintains a steady, graceful motion, drifting from left to right and slightly rotating as it progresses. Tarsier-7b: The primary object in the video is a large jellyfish. It pulsates and swims through the water. The jellyfish changes its position and orientation in the water. LLaVA-OneVision-7b: The primary object in the video is a jellyfish, characterized by its translucent body with a white and purple coloration. It has long, thin tentacles that trail behind it as it moves through the water. The jellyfish is seen floating near the ocean floor, which is covered with small rocks and some debris. The water surrounding the jellyfish is clear blue, and there are strands of seaweed visible in the background. BLIP-3-Video: A jellyfish, is seen swimming in the water. It moves in a slow, undulating motion, with its bell-shaped body contracting and expanding to propel itself forward. The jellyfish's tentacles trail behind it, creating a delicate, flowing effect. Throughout the video, the jellyfish maintains a consistent upward and forward movement, occasionally adjusting its direction slightly.

[Figure 55] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile55.png>)

[Figure 56] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile56.png>)

[Figure 57] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile57.png>)

[Figure 58] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile58.png>)

[Figure 59] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile59.png>)

Question: Please provide a dense description of the video, covering the main subject, movements, style, backgrounds, and cameras. GT: The video provides a breathtaking aerial view of a historic castle perched on the edge of a steep cliff. As the drone camera glides over the scene, viewers can appreciate the castle's ancient stone walls and the surrounding lush greenery, which contrasts sharply with the rugged cliff faces. The castle, a relic of medieval architecture, is shown in various states of preservation, with some parts reduced to ruins while others stand robustly. The video captures the castle's strategic location, offering panoramic views of the surrounding landscape and the churning waters below. Tarsier-7b: The video begins with an aerial view of a grassy field with several rectangular structures. The camera pans left, revealing more of the field and structures. A lighthouse and a metal walkway become visible. The camera continues to pan left, showing the coastline and additional structures. The video ends with a black screen. LLaVA-OneVision-7b: The video begins with an aerial view of a historical site, showcasing a series of rectangular stone structures and pathways. The structures are surrounded by lush green grass, and the pathways are lined with metal railings. A small building with a sloped roof is visible in the foreground. The perspective shifts to reveal more of the site, including a steep cliff with a staircase leading up to a platform. The cliffside features several large, rectangular stone formations arranged in a linear pattern. The video continues to focus on these formations, highlighting their arrangement along the cliff edge. The background reveals a body of water, possibly the ocean, adding to the scenic beauty of the location. The video wraps up with a close-up view of the stone formations, emphasizing their historical significance and the meticulous craftsmanship involved in their construction. BLIP-3-Video: The video provides an aerial view of a historic fortification perched on a rugged cliff overlooking the ocean. The fort, characterized by its robust stone walls and strategic positioning, includes various defensive structures such as cannons and barricades. As the video progresses, the camera captures the changing light of the setting sun, casting long shadows and highlighting the textures of the stone and the surrounding landscape. The fort's layout, including pathways and small buildings, is revealed in greater detail, showcasing its historical significance and architectural complexity.

- Figure 6: Example video captioning results on Mira dataset, formed in question-answering style. We compare the outputs of BLIP-3-Video, Tarsier, and LLaVA-OneVision. GT stands for the ground truth. Different colored texts are different parts of ground truth captions and their corresponding sentences in the model outputs. Underlined texts are hallucinations.

Table 12: Results on VideoInstruct benchmark.

|Model<br><br>|Accuracy|
|---|---|
|PLLaVA-34B SlowFast-LLaVA-34B VideoGPT+ ST-LLM-7B BLIP-3-Video VideoChat2 HD mistral LITA-13B LLaMA-VID-13B VideoChat2 LLaMA-VID-7B Video-ChatGPT<br><br>|3.32 3.32 3.28 3.15 3.11 3.10 3.04 2.99 2.98 2.89 2.38|

Table 13: Results on TempCompass benchmark.

|Model| |Yes/No QA|Caption matching|
|---|---|---|---|
|GPT-4o Qwen2-VL-7B-Instruct Gemini-1.5-pro LLaVA-OneVision-Qwen-2-7B LLaVA-NeXT-Video-32B-Qwen InternVL2-8B BLIP-3-Video Llama-3-VILA1.5-8B LongVA-7B LLaVA-NeXT-Video-7B-DPO VideoChat2-vicuna-stage3 LLaVA-1.5-13B Video-LLaVA-7B Video-LLaMA-2-13B LLaMA-VID-7B-short-video| |73.66 72.77 70.32 69.67 69.38 68.24 66.7 63.64 62.13 61.19 58.01 56.38 56.38 53.73 52.96<br><br>|80.84 77.31 77.45 73.79 76.51 77.11 66.5 68.93 65.67 63.01 53.69 64.27 63.34 54.16 56.02|

### A.3 Reproducibility Statement

We build on top of the open-source BLIP-3 (XGen-MM) model and training code hosted on Huggingface and github. All the experiments were conducted with public datasets. The code and the trained model will be released together with the final version of the paper.

[Figure 60] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile60.png>)

[Figure 61] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile61.png>)

[Figure 62] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile62.png>)

[Figure 63] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile63.png>)

[Figure 64] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile64.png>)

Question: Please provide a dense description of the video, covering the main subject, movements, style, backgrounds, and cameras. GT: A step-by-step guide on assembling a wooden bed frame, featuring a person methodically working through the construction process. Initially, the person is seen preparing the wooden planks and arranging the necessary tools. As the video progresses, the individual meticulously attaches and secures the slats to the frame, ensuring stability and precision. The final scenes depict the person tightening the last screws and reviewing the completed structure, demonstrating a sense of accomplishment and attention to detail. Tarsier-7b: The video starts with a person assembling a piece of furniture. The scene then transitions to a website showcasing different types of beds, including a bed with a canopied frame and a minimalist bed with a white mattress. The video concludes with social media and website icons for 'dactylomoovis' and 'dactylomoveis'. LLaVA-OneVision-7b: The video begins with a person in a pink shirt with the number 18 on it, working on assembling a wooden bed frame. The individual is seen using a black brush to apply paint or stain to the wooden parts of the bed frame. The text 'Utilize espacador para colar os estrados' (Use an spacer to glue the slats) appears at the bottom of the screen, providing instructions for the assembly process. The scene transitions to a top-down view of the person continuing to work on the bed frame, with the text 'Parafuse os estrados nas barras' (Screw the slats into the bars) appearing at the bottom, offering further guidance. The background remains consistent throughout, showing a white wall and some tools on the floor. The video then shifts to a promotional website for Dactylo.com.br, featuring images of different bed frames and a brief description of the company's mission to create high-quality furniture using sustainable materials. The website also includes social media icons for Facebook, Twitter, Google+, and YouTube, along with the Dactylo logo and the tagline 'Movimento de Madeira' (Wood Movement). The video concludes with a promotional image for Dactylo.com.br, showcasing a white bed frame with the text 'COMO MANTER A BICAMA MISSION' (How to maintain the Bicama Mission), emphasizing the importance of proper care for the furniture. BLIP-3-Video: A step-by-step process of assembling a piece of furniture, specifically a bed frame. The person involved is seen working on a wooden bed frame, using various tools and components to construct the bed. The video likely serves as a tutorial, providing viewers with a visual guide on how to assemble the bed frame themselves. The person's actions are deliberate and precise, indicating a level of expertise or familiarity with the task.

[Figure 65] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile65.png>)

[Figure 66] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile66.png>)

[Figure 67] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile67.png>)

[Figure 68] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile68.png>)

[Figure 69] You Only Need 32 Tokens to Represent a Video Even in VLMs_images/imageFile69.png>)

Question: Please provide a dense description of the video, covering the main subject, movements, style, backgrounds, and cameras. GT: Two men engaged in the task of loading a barrel onto a small, yellow floatplane near a body of water. The men are dressed in casual attire suitable for outdoor work, with one wearing gloves for handling the barrel. The plane is docked at a wooden structure, possibly a jetty or makeshift dock, and the men coordinate their efforts to lift and maneuver the barrel into the cargo area of the aircraft. The surrounding environment is serene and natural, with a forested backdrop suggesting a remote or wilderness setting. Tarsier-7b: The video shows two men near a seaplane by a lakeside. One man is wearing a red shirt and the other is in a blue shirt. They are handling a barrel. The scene changes to a construction site with heavy machinery and road signs. The camera captures a vehicle moving forward on a dirt road. The scene switches back to the men by the seaplane, now with a view of the lake and mountains in the background. The video includes on-screen text commentary about living in Yukon and the challenges faced by the community. LLaVA-OneVision-7b: The video begins with two individuals standing next to a yellow and brown seaplane on a wooden dock by a body of water. The seaplane has the registration number C-GFPR and is adorned with red and white stripes. One person, wearing a maroon shirt and jeans, is holding a red barrel, while the other, in a blue shirt and shorts, is also handling a barrel. In the background, there are red barrels and a small boat. The scene transitions to a view from inside a vehicle, showing a dirt road under construction with orange cones and construction equipment visible. The road is surrounded by dense green forests and mountains in the distance. The video then returns to the seaplane, where the same two individuals are now standing near the open door of the plane. The text 'Living up YUKON is tough' appears at the bottom of the screen, followed by 'Collecting resources is the biggest issue for the people up here.' The video wraps up with the two individuals standing near the seaplane, with the serene backdrop of the lake and forested mountains still visible. BLIP-3-Video: A sequence of events where two men are engaged in loading a barrel onto a small, yellow seaplane parked on a wooden dock beside a serene lake. The men, dressed in casual work attire, appear to be coordinating their efforts to secure the barrel, which is large and cylindrical, onto the aircraft. The setting is tranquil, with the calm waters of the lake reflecting the clear blue sky and the lush greenery of the surrounding landscape. The video conveys a sense of teamwork and the meticulous nature of preparing an aircraft for a journey.

- Figure 7: Example video captioning results on Mira dataset, formed in question-answering style. We compare the outputs of BLIP-3-Video, Tarsier, and LLaVA-OneVision. GT stands for the ground truth. Different colored texts are different parts of ground truth captions and their corresponding sentences in the model outputs. Underlined texts are hallucinations.

### A.4 Limitations

At this point, all the models have been trained with videos up to a couple of minutes. Although we confirmed their capability to handle longer videos by combining it with a framework like LVNet, the model is expected to perform better with better training dataset curation with longer videos. The model sizes (3B) are also relatively small compared to other models. Despite our model showsing very promising model size-accuracy trade-off, training a bigger model with a larger dataset remains as future work.

### A.5 Societal Impact

Our proposed BLIP-3-Video significantly improves video-language model efficiency by reducing the number of visual tokens needed question answering. This compact design lowers the computational and energy requirements, enabling more sustainable AI deployments and reducing the environmental footprint of large-scale video understanding. Furthermore, the compact architecture

makes high-quality video-language reasoning more accessible to researchers and developers in resource-constrained settings, especially within academia.

As with many powerful AI models, BLIP-3-Video could be misused in ways that raise ethical concerns. The ability to process and understand video content efficiently may be exploited for mass surveillance, privacy-invasive monitoring, or unauthorized profiling. Moreover, if trained on biased or unbalanced data, the temporal encoder may overlook or misrepresent minority-relevant visual cues, potentially reinforcing harmful stereotypes. The model’s compact design could also facilitate easier generation harmful content or misinformation.

