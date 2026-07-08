# arXiv:2504.11455v1[cs.CV]15Apr2025

## SimpleAR: Pushing the Frontier of Autoregressive Visual Generation through Pretraining, SFT, and RL

#### Junke Wang1, Zhi Tian2, Xun Wang2, Xinyu Zhang2, Weilin Huang2, Zuxuan Wu1, Yu-Gang Jiang1 1Fudan University, 2ByteDance Seed Code is available at https://github.com/wdrink/SimpleAR.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Figure 1: Text-to-image generation results by SimpleAR in 1024×1024 resolution.

### Abstract

This work presents SimpleAR, a vanilla autoregressive visual generation framework without complex architecure modifications. Through careful exploration of training and inference optimization, we demonstrate that: 1) with only 0.5B parameters, our model can generate 1024×1024 resolution images with high fidelity, and achieve competitive results on challenging text-to-image benchmarks, e.g., 0.59 on GenEval and 79.66 on DPG; 2) both supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO) training could lead to significant improvements on generation aesthectics and prompt alignment; and 3) when optimized with inference acceleraton techniques like vLLM, the time for SimpleAR to generate an 1024×1024 image could be reduced to around 14 seconds. By sharing these findings and open-sourcing the code, we hope to reveal the potential of autoregressive visual generation and encourage more participation in this research field.

### 1 Introduction

Recent years have seen the rapid advancements of deep generative models [20, 13, 35, 15, 58, 18, 66], which offer innovative approaches for the creation of visual content. Among these models, diffusion models[20, 51] and autoregressive models[55, 75] have emerged as the leading paradigms.

Diffusion models [20, 51] synthesize visual outputs by iteratively refining random noise through a learned denoising process. This manner has been proven highly effective in producing high-fidelity images [36, 14] and videos [5, 27, 74, 37, 9], thus gaining remarkable popularity in the field of multimodal generation.

Another line of work is autoregressive (AR) models [55, 75, 26], which formulate visual generation as a sequential process, i.e., each pixel or token is generated based on the preceding ones. This autoregressive process naturally excels in precise and coherent prediction, making it particularly effective for tasks that demand fine-grained control. Moreover, AR visual generation models are naturally compatible with diverse modalities (e.g., language and audio), which facilitates native multimodal understanding and generation [56, 68, 67, 24].

Despite these strengths, autoregressive visual generation still underperforms diffusion models currently. Several hypotheses have been proposed to explain this gap. One posits that discrete visual tokenizer imposes a fundamental limit on the quality of autoregressive generation, while another points to the considerably longer length of visual sequences compared to text, and long-range dependency modeling leads to significantly more challenges. To mitigate these issues, various variants have been proposed, such as MAR [31] and VAR [58]. Although these approaches have achieved promising results on academic benchmarks [12], they compromise the intrinsic pattern of “next-token prediction” in language models, resulting in a trade-off between performance and simplicity.

In this work, we aim to push the frontier of autoregressive visual generation by preserving the simplicity of the autoregressive framework while carefully optimizing its training paradigm through pretraining, supervised fine-tuning (SFT) and GRPO [48]-based reinforcement learning (RL). With this, we show that a vanilla AR model with only 0.5B parameters is capable of generating 1024×1024 images with superior aesthetics, and achieving competitive results on existing text-to-image benchmarks, e.g., 0.59 on GenEval. When scaled with more computes (i.e., parameters and tokens), our model consistently demonstrates improved generation quality with higher fidelity and more coherent structures. These results highlight the potential of autoregressive visual generation to compete with diffusion models.

In addition, we also investigate the inference acceleration of autoregressive visual generation models using vLLM [29] and speculative sampling [7]. When deployed with vLLM, our model can generate a 1024×1024 image in approximately 14 seconds, showcasing its potential for real-world applications.

### 2 Related Work

#### 2.1 Autoregressive Visual Generation Models

Similar to language models [41, 60], autoregressive visual generation methods formulate image and video generation as a next-token prediction process. These approaches rely on a tokenizer [15, 64, 50, 59, 63] to tokenize visual inputs into discrete tokens, and train an autoregressive transformer to model the sequential dependencies among these tokens using causal attention. Representative work, including DALL-E [44], Parti [75], and LlamaGen [55], exhibit strong instruction-following and high-fidelity image generation capabilities.

#### 2.2 Unified Models for Multimodal Understanding and Generation

Recent advancements in multimodal learning have led to the development of unified models that seamlessly integrate vision and language for both understanding and generation [68, 10, 71, 69, 33]. Chameleon [56] and Emu3 [67] adopt token-based architecture by handling diverse modalities with a unified autoregressive transformer. Transfusion [76] and Show-o [72], on the other hand, integrate next-token prediction for text and diffusion processes for images within a single model, enabling effective handling of both discrete and continuous data. These unified approaches demonstrate the potential of leveraging large-scale transformer architectures to enhance the synergy between vision and language, setting the stage for more versatile and generalizable multimodal AI systems.

### 3 Method

Our goal is to advance autoregressive visual generation by preserving the simplicity of the AR framework while optimizing its training pipeline and inference efficiency. To this end, we propose SimpleAR, which consists of a pretrained visual tokenizer [16] that discretizes images into compact visual tokens, a text tokenizer, and a decoder-only transformer that autoregressively models the joint distribution of text and image tokens. Note that, unlike diffusion models [8] or previous AR models [55] that require an additional text encoder [42], our model integrates text encoding and visual generation within a unified transformer architecture, gaining remarkable advantages in both efficiency and multimodal coherence.

#### 3.1 Autoregressive Visual Generation

Given an input image X ∈ RH×W×3, we first tokenize it into a sequence of discrete visual tokens Z ∈ Rh×w using a learned tokenizer [16], where p = H/p denotes the compression ratio. Each item in Z represents an index of the codebook, mapping a visual patch to a learned discrete representation. Then, these image tokens are flatten into a 1D sequence z in raster scan ordering, following previous work [15, 55]. We also have a text tokenizer to convert text prompts into text tokens t ∈ RN.

After this, we concatenate text and image tokens along the sequence dimension, and input them to a transformer decoder [73] to model their dependency.

#### 3.2 Three-stage Training

To improve the generation quality and training efficiency, we employ a three-stage learning paradigm: large-scale pretraining on diverse visual datasets to capture generalizable patterns, supervised finetuning (SFT) on high-quality data to enhance fidelity and instruction-following, and reinforcement learning (RL) to further refine multimodal alignment [30, 4] and alleviate exposure bias [45, 1].

Pretraining and SFT: during pretraining and supervised-finetuning (SFT), our model is both trained with language modeling loss:

L

logp(zi|z<i,t), (1)

LLM = −

i=1

where the prediction of zi is conditioned on both text tokens t and its preceeding visual tokens z<i. RL with GRPO: recently, reinforcement learning has gained increasing attention in the post-training of large language models (LLMs) [48, 23] to improve reasoning capability and diffusion models [30, 62] for better alignment with human preferences. Among these approaches, Group Relative Policy Optimization (GRPO) [48] has emerged as a particularly promising technique by offering better training efficiency and stability.

Using the checkpoint after SFT, GRPO initializes a trainable policy model πθ and a frozen reference model πref. For a given text prompt t, it samples a group of outputs o1,o2,...,oG from the policy model πθ

, and maximizes the following object to optimize πθ:

old

  

   (2)

G

πθ(oi|t) πθ

πθ(oi|t) πθ

1 G

,1 − ϵ,1 + ϵ Ai − βDKL(πθ||πref)

min

Ai,clip

JGRPO(θ) = Eo

(oi|t)

(oi|t)

i∼πθold

old

old

i=1

where ϵ and β denote clipping hyper-parameter [47] and coefficient controlling the Kullback–Leibler (KL) penalty, and Ai is the advantage computed in one group. We adopt CLIP [40] as the reward and find it surprisingly useful during experiments.

#### 3.3 Inference

During inference, we sample the visual tokens sequentially through the following conditional probability: zˆi = argmaxpθ(zi|z<i,t). The sampled tokens are then input to the decoder of visual tokenizer to generate images. Unless otherwise stated, greedy search is employed with topK = 64000 (codebook size). We also use Classfier-Free Guidance (CFG) [21] to improve the generation quality.

Table 1: Evaluation on the GenEval [17] and DPG [22] benchmark. † result is with prompt rewriting. Here we only classify next-token prediction models as AR methods, and next-scale prediction model Infinity as NAR (non-autoregressive) methods.

GenEval↑ DPG↑

Methods # Params Type

Two Obj. Position Color Attri. Overall Global Relation Overall

SDv1.5 [46] 0.9B Diff. 0.38 0.04 0.06 0.43 74.63 73.49 63.18 PixArt-alpha [8] 0.6B Diff. 0.50 0.08 0.07 0.48 74.97 82.57 71.11 SDv2.1 [46] 0.9B Diff. 0.51 0.07 0.17 0.50 77.67 80.72 68.09 LlamaGen [55] 0.8B AR 0.34 0.07 0.04 0.32 - - 65.16

- Ours 0.5B AR 0.82 0.26 0.38 0.59 86.64 88.51 79.66 LDM [46] 1.4B Diff. 0.29 0.02 0.05 0.37 - - -

- DALL-E 2 [43] 6.5B Diff. 0.66 0.10 0.19 0.52 - - -
- DALL-E 3 [3] - Diff. - - - 0.67† 90.97 90.58 83.50 Show-o [72] 1.3B NAR 0.80 0.31 0.50 0.68 - - 67.48 Infinity [19] 2B NAR 0.85† 0.49† 0.57† 0.73† 93.11 90.76 83.46 Chameleon [56] 7B AR - - - 0.39 - - Janus [68] 1.5B AR 0.68 0.46 0.42 0.61 82.33 85.46 79.68 Emu3 [67] 8.5B AR 0.81† 0.49† 0.45† 0.66† - - 81.60

Ours 1.5B AR 0.90 0.28 0.45 0.63 87.97 88.33 81.97

Since token prediction in autoregressive (AR) models must be performed sequentially, inference latency can be a significant bottleneck. However, various optimizations developed in the LLM community for inference accelerate, such as KV cache [38] and paged attention [29], offer promising solutions to this. In this work, we explore the application of these techniques to accelerate AR-based visual generation, which will be introduced below beriefly.

KV Cache [38] stores previously computed key-value embeddings from the attention layers and, reuse them across autoregressive decoding steps to reduce redundant computation. It is widely used in LLM inference and could decrease the complexity from O(N2) to O(N).

vLLM Serving [29] leverages optimized memory management and efficient attention mechanisms, such as paged attention, to enable high-throughput and low-latency inference for autoregressive models on modern hardware.

Speculative Jacobi Decoding [7, 57] accelerates autoregressive generation by first sampling multiple candidate token sequences from a draft model and then efficiently verifying them using the target model. We use the pretrained AR model to serve as both draft model and target model for acceleration.

- 4 Experiments

#### 4.1 Implementation details

Experimental Setup: we adopt the same architecture as Qwen [2, 73] for our transformer model, and intialize it with LLM weights. While for the visual tokenizer, we use Cosmos-Tokenizer [16], whose codebook size is 64k and downsample ratio is 16.

As mentioned in Sec. 3.2, the training consists of three stages: 1) 512 resolution pretraining, 2) 1024 resolution SFT, and 3) 1024 resolution RL. During pretraining / SFT, the learning rate is set to 1e-4 / 2e-5, and batch size is 256 in total. The RL training is conducted using trl [61] framework, the learning rate is 1e-5, and batch size is 28. We do not use warm up and learning rate decay in all stages. AdamW [32] is employed for optimization. All experiments are conducted on 32 NVIDIA A100 GPUs.

Training data: the pretraining data involves CC3M [49], CC12M [6], OpenImages [28], SAM1B [25], and Megalith-huggingface [34] (around 43M in total). For SFT, we use JourneyDB [54], Synthetic-

dataset-1M [39], and 10M internal data. We adopt a simple data filtering strategy for supervised fine-tuning (SFT) data by removing all images whose short edge is smaller than 1024 pixels. We recaption all the images using Qwen2-VL [65], and randomly choose from long and short prompts during training.

#### 4.2 Comparing with State-of-the-arts

We present a comprehensive comparison between SimpleAR and existing state-of-the-art visual generation models in Table 1. Remarkably, with only 0.5B parameters, our model achieves superior performance on established text-to-image benchmarks, i.e., 0.59 overall score on GenEval [17] and 79.66 on DPG-Bench [22], outperforming all comparable-scale methods (those with fewer than 1B parameters) by significant margins. This includes both diffusion-based approaches (e.g., SDv2.1 [46]) and autoregressive alternatives (e.g., LlamaGen [55]). Notably, diffusion models also require auxiliary text encoders (e.g., Flan-T5-XL [11] with 3B parameters) that effectively double their parameter footprint, but SimpleAR process both modalities using a unified transformer, enjoying more efficient parameter utilization and native support for conditional generation.

Moreover, scaling SimpleAR to 1.5B yields consistent improvements across benchmarks (+0.04 on GenEval, +1.85 on DPG-Bench), demonstrating predictable scaling behavior analogous to large language models. Although our model still falls behind Infinity [19] on GenEval, we believe this gap is primarily due to the disparity in the volume of training data and can be narrowed with data scaling.

#### 4.3 Ablation Studies

In this section, we conduct ablation studies to explore the training and inference of autoregressive visual geneation models, including model initialization, position encodings, etc.

Table 2: Initialization of transformer model.

Method Global Relation Overall

w/o LLM init 76.10 85.61 69.43 w/ LLM init 77.47 85.96 70.52

Table 3: 1D pos embedinggs v.s. 2D.

Method Global Relation Overall

2D pos 73.91 70.58 70.96 1D pos 77.47 85.96 70.52

Effects of LLM initialization: as shown in Table 2, whether or not using LLM intialization does not have remarkable effect on the DPG-Bench performance. We believe this is due to the text prompts in existing benchmarks is simple and lack sufficient linguistic complexity or reasoning requirements. The results may also indicate that training on text-to-image generation data will lead to drastic forgetting of initial language capabilities. Therefore, incorporating text data during the training phase is essential for maintaining the text understanding and generation capabilities of LLMs.

Effects of Position Encodings: it is also interesting to comapre 1D and 2D rotary position encoding [53] for image generation using autoregressive framework. We compare the pretraining results on DPG-Bench, and the results in Table 3 demonstrate that replacing 1D positional encoding in LLM with 2D will not improve visual generation significantly. However, we believe 2D positional encoding is quite necessary for dynamic resolution generation and video generation.

Effects of RL: for GRPO-based reinforcement learning, we compare two different reward modules: CLIP-ViT-H-14 and HPSv2 [70]. Notably, HPSv2 is a fine-tuned version of CLIP-ViT-H-14, trained on a specially constructed human preference dataset. The GenEval performance is quantitatively compared in Table 4, where we observe that using both reward modules results in performance improvements. Specifically, the CLIP reward achieves a greater performance gain, with a +0.6 increase on GenEval for the 0.5B model. The qualitative results, as shown in Figure 2, further demonstrate that CLIP could enhance the text rendering capability, along with improved perception of quantifiers and spatial descriptions.

We also plot the reward value and GenEval performance during training in Figure 4.3- 4.3. As can be seen from the left, the reward values exhibit a gradual increase during training. Interestingly, the GenEval performance demonstrates a positive correlation with the reward progression, suggesting that a simple reward function, i.e., CLIP-ViT-H-14, is effective in providing consistent feedback that aligns well with the desired task performance.

- Table 4: GenEval performance of SFT/RL model, TO: two objects, P: position, CA: color attribute. Param Reward TO P CA Overall

- 0.5B

- 0.73 0.22 0.23 0.53 HPSv2 0.81 0.23 0.32 0.56

CLIP 0.82 0.26 0.38 0.59 (↑ 0.6)

- 1.5B

- 0.87 0.27 0.33 0.61 HPSv2 0.94 0.26 0.38 0.61

CLIP 0.90 0.28 0.45 0.63 (↑ 0.2)

Figure 2: Generation results before/after GRPO.

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

A photo of a stop sign

[Figure 13]

[Figure 14]

A photo of four apples

A photo of a wine glass right of a hot dog

[Figure 15]

[Figure 16]

0 50 100 150 200 250 300 350

Steps

0.24

0.26

0.28

0.30

0.32

Reward

Raw Reward

Smoothed Reward

50 100 150 200 250 300 350 400

Steps

0.52

0.53

0.54

0.55

0.56

0.57

0.58

0.59

0.60

GenEvalPerf.

GenEval overall

Figure 3: Visualizations of the GRPO training: reward and GenEval overall performance.

Inference Speedup: the sequential prediction nature of autoregressive (AR) models could result in high inference latency, making it challenging to deploy them in real-time applications. However, recent advancements in LLM optimization techniques, such as KV cache, paged attention, and speculative decoding, offer opportunities for accelerating inference in AR visual generation models. Therefore, we conduct experiments to apply these approaches to speed up the inference of SimpleAR. The throughput is calculated on an Nvidia A100 node, with CFG enabled.

- Table 5: Inference speed comparison w/ KV Cache and vLLM.

Method Throughput (sec/img)

baseline 227.62 + KV Cache 150.19 + vLLM 13.55

Table 6: Speculative Jacobi Decoding.

Method Avg Steps DPG-overall

baseline 4096 79.66 SJD 2685 80.33 SJD-w16 2199 81.39 SJD-w32 2034 81.05

Table 5 demonstrates that using KV cache can effectivelly save 34% inference time, while serving with vLLM can lead to more sigficant inference acceleration, reducing the time to generate a 1024×1024 image to 13.55 sceconds.

We also try speculative jacobi decoding (SJD), which speculatively decoding multiple tokens in parallel [52, 7] at inference time to reduce the autoregressive generation steps. The results are shown in Table 6, we can see that SJD can lead to around 2× reduction in steps and slightly better performance on DPG. We also compare the sliding-window design proposed by [57]. Although SJD does not practically reduce the testing latency of the autoregressive (AR) model (unable to use KV cache and need to forward the entire sequence each time), it still presents many possibilities for optimizing the AR inference process.

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

The cowboy stands at 6000 mm tall, capturing every facet of his rugged attire from the leather boots to the wide-brimmed hat. In the background, the bokeh effect beautifully blurs the lights, creating a striking contrast with the sharpness of his figure.

A close-up image of a unique four-leaf clover, intricately formed from droplets of water on a smooth, reflective surface. Each leaf of the clover is perfectly shaped, with the water's surface tension creating a delicate appearance.

An intricately detailed oil painting that captures the whimsical essence of a feline super math wizard. The cat, adorned with a wizard's hat and cape, is surrounded by floating mathematical symbols and equations.

A captivating portrait showcasing a young girl with ethereal beauty, gracefully suspended amidst the soft, billowy clouds. Her delicate features are rendered with meticulous attention to detail.

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

On a clear warm day, the sun radiates down on a sandy beach where a close-up of a wafer cone reveals chocolate ice cream beginning to melt down its textured sides.

A large, round orange pumpkin carved with a smiling face, sitting on a wooden table. The pumpkin is surrounded by a scattering of fallen autumn leaves.

Two sleek blue showerheads release a steady stream of water. The water cascades down onto a vivid, crisp green pear that is centrally positioned directly beneath them.

An animated frog with a rebellious punk rock style, clad in a black leather jacket adorned with shiny metal studs, is energetically shouting into a silver microphone.

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

A detailed photograph captures the intricate features of a

A playful monkey with a chestnut coat and bright eyes is clumsily handling a crimson red heartshaped tea pot. The monkey sits in a verdant jungle environment, surrounded by an array of glossy green leaves and suspended vines. The tea pot, with its glossy ceramic finish, reflects the dappled sunlight that filters through the dense canopy overhead.

A digital illustration of a girl features her with vibrant rainbow-colored hair that cascades smoothly down her shoulders. She has two spiraling unicorn horns emerging from her forehead, adding a fantastical element to the portrait. Fresh, vivid colors blend seamlessly in gradients across the composition, showcasing a high level of detail and skill.

In a brightly-lit laundry room, a pair of ripe yellow bananas rest nonchalantly against the sleek surface of a silver washing machine. Sunlight streams through a nearby window, enhancing the vibrancy of the space and casting soft shadows around the cheerful fruits.

pharaoh statue adorned with unconventional accessories. The statue is wearing steampunk glasses that have intricate bronze gears and round, reflective lenses. It is also dressed in a stark white tshirt that contrasts with a dark, textured leather jacket draped over its shoulders.

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

A breathtaking photograph capturing the vibrant hues of a sunset with streaks of pink and orange painting the sky behind the majestic Grand Canyon. The canyon's intricate rock formations are silhouetted against the illuminated backdrop, showcasing the deep crevices and towering spires. In the foreground, the Colorado River can be glimpsed winding its way through the ancient geological marvel.

A majestic granite statue of a wombat warrior, clad in intricately detailed armor, stands proudly in the center of a temple's cella. The statue, gripping a broad sword with both hands, is bathed in a soft beam of light that filters down from an unseen source above, highlighting the textures of its stony surface. It is perched upon an ornate pedestal, which adds to its imposing presence.

A clear night sky where a slender crescent moon hangs delicately between the silhouetted branches of tall trees. The moon's pale light casts a soft glow on the intricate patterns of the branches, creating a contrast against the dark blue of the night sky.

A powerful steam locomotive with a black and red exterior, billowing white steam as it speeds along the tracks through a vast, sandy desert landscape. The locomotive's wheels kick up small clouds of sand, and the clear blue sky stretches endlessly above.

##### Figure 4: Text-to-image generation results by SimpleAR using DPG prompts.

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

a photo of two beds

a photo of two vases a photo of four tvs

a photo of two bears

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

a photo of a white handbag and a purple bed

a photo of a white pizza and a green umbrella

a photo of a white wine glass and a brown giraffe

a photo of an orange microwave and a black spoon

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

a photo of a wine glass above a kite

a photo of a tie right of a baseball bat

a photo of a baseball glove below an umbrella

a photo of a couch below a cup

Figure 5: Text-to-image generation results by SimpleAR using GenEval prompts.

#### 4.4 Visualizations and Failure Cases

We visualize image generation results of SimpleAR in Figure 4 and 5. It can be observed that our model could not only generate high-fidelity, aesthetically pleasing images but also demonstrate strong instruction-following capabilities.

Several failure cases are also shown in Figure 6. The limited data scale and parameter size constrain SimpleAR to generate complex poses, objects, and text. Additionally, our model may synthesize content that does not adhere to physical laws.

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Figure 6: Failure cases of SimpleAR.

### 5 Conclusion and Future Work

This work presented SimpleAR, a vanilla autoregressive framework for visual generation that discards complex architecture modifications. We focus on the optimization of two fundamental components: 1) Training pipeline, through large-scale pretraining, high-quality supervised finetuning, and GRPO

training, SimpleAR achieves competitive performance on existing text-to-image generation benchmarks with only 0.5B parameters. 2) Inference efficiency, we explore various inference acceleration techniques, and show SimpleAR could generate a 1024×1024 image in around 14 seconds when served with vLLM. We hope this work can inspire further exploration into autoregressive visual generation and firmly believe that it is a promising alternative to diffusion models.

Despite the superior results achieved, we admit that there still exist many limitations that are worth deeper improvements or exploring:

Stronger visual tokenizers: the reconstruction performance of Cosmos-Tokenizer [16] is limited, especially in capturing fine-grained visual details, e.g., faces and texts. This leaves room for better visual generation results with improved tokenization methods.

Text-to-video generation: compared to text-to-image generation, text-to-video generation presents significantly more challenges since the model has to generate coherent outputs that are contextually and temporally consistent.

Native multimodal understanding and generation: recently, the native multimodal understanding and generation capabilities of GPT-4o have captured widespread attention. It offers a glimpse into the future of models that can seamlessly integrate vision, text, and even audio without relying on modality-specific encodings. Moving SimpleAR forward, building truly native large multimodal models that can perform end-to-end reasoning across images, text, and other modalities is a promising and crucial research direction.

### References

- [1] D. Bahdanau, P. Brakel, K. Xu, A. Goyal, R. Lowe, J. Pineau, A. Courville, and Y. Bengio. An actor-critic algorithm for sequence prediction. arXiv preprint arXiv:1607.07086, 2016.
- [2] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [3] J. Betker, G. Goh, L. Jing, T. Brooks, J. Wang, L. Li, L. Ouyang, J. Zhuang, J. Lee, Y. Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2023.
- [4] K. Black, M. Janner, Y. Du, I. Kostrikov, and S. Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.
- [5] A. Blattmann, T. Dockhorn, S. Kulal, D. Mendelevitch, M. Kilian, D. Lorenz, Y. Levi, Z. English, V. Voleti, A. Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [6] S. Changpinyo, P. Sharma, N. Ding, and R. Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In CVPR, 2021.
- [7] C. Chen, S. Borgeaud, G. Irving, J.-B. Lespiau, L. Sifre, and J. Jumper. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318, 2023.
- [8] J. Chen, J. Yu, C. Ge, L. Yao, E. Xie, Y. Wu, Z. Wang, J. Kwok, P. Luo, H. Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.
- [9] S. Chen, C. Ge, Y. Zhang, Y. Zhang, F. Zhu, H. Yang, H. Hao, H. Wu, Z. Lai, Y. Hu, et al. Goku: Flow based video generative foundation models. arXiv preprint arXiv:2502.04896, 2025.
- [10] X. Chen, Z. Wu, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, and C. Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.
- [11] H. W. Chung, L. Hou, S. Longpre, B. Zoph, Y. Tay, W. Fedus, Y. Li, X. Wang, M. Dehghani, S. Brahma, et al. Scaling instruction-finetuned language models. JMLR, 2024.
- [12] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009.
- [13] P. Dhariwal and A. Nichol. Diffusion models beat gans on image synthesis. In NeurIPS, 2021.

- [14] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Müller, H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICLR, 2024.
- [15] P. Esser, R. Rombach, and B. Ommer. Taming transformers for high-resolution image synthesis. In CVPR, 2021.
- [16] N. et. al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.
- [17] D. Ghosh, H. Hajishirzi, and L. Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. In NeurIPS, 2024.
- [18] L. Gong, X. Hou, F. Li, L. Li, X. Lian, F. Liu, L. Liu, W. Liu, W. Lu, Y. Shi, et al. Seedream 2.0: A native chinese-english bilingual image generation foundation model. arXiv preprint arXiv:2503.07703, 2025.
- [19] J. Han, J. Liu, Y. Jiang, B. Yan, Y. Zhang, Z. Yuan, B. Peng, and X. Liu. Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis. In CVPR, 2025.
- [20] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020.
- [21] J. Ho and T. Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.
- [22] X. Hu, R. Wang, Y. Fang, B. Fu, P. Cheng, and G. Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.
- [23] W. Huang, B. Jia, Z. Zhai, S. Cao, Z. Ye, F. Zhao, Y. Hu, and S. Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.
- [24] Y. Jiao, H. Qiu, Z. Jie, S. Chen, J. Chen, L. Ma, and Y.-G. Jiang. Unitoken: Harmonizing multimodal understanding and generation through unified visual encoding. arXiv preprint arXiv:2504.04423, 2025.
- [25] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, et al. Segment anything. In ICCV, 2023.
- [26] D. Kondratyuk, L. Yu, X. Gu, J. Lezama, J. Huang, G. Schindler, R. Hornung, V. Birodkar, J. Yan, M.-C. Chiu, et al. Videopoet: A large language model for zero-shot video generation. In ICML, 2024.
- [27] W. Kong, Q. Tian, Z. Zhang, R. Min, Z. Dai, J. Zhou, J. Xiong, X. Li, B. Wu, J. Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [28] A. Kuznetsova, H. Rom, N. Alldrin, J. Uijlings, I. Krasin, J. Pont-Tuset, S. Kamali, S. Popov, M. Malloci, A. Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. IJCV, 2020.
- [29] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and

I. Stoica. Efficient memory management for large language model serving with pagedattention. In SOSP, 2023.

- [30] K. Lee, H. Liu, M. Ryu, O. Watkins, Y. Du, C. Boutilier, P. Abbeel, M. Ghavamzadeh, and S. S. Gu. Aligning text-to-image models using human feedback. arXiv preprint arXiv:2302.12192, 2023.
- [31] T. Li, Y. Tian, H. Li, M. Deng, and K. He. Autoregressive image generation without vector quantization. In NeurIPS, 2024.
- [32] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [33] C. Ma, Y. Jiang, J. Wu, J. Yang, X. Yu, Z. Yuan, B. Peng, and X. Qi. Unitok: A unified tokenizer for visual generation and understanding. arXiv preprint arXiv:2502.20321, 2025.
- [34] madebyollin. Megalith-huggingface. https://huggingface.co/datasets/madebyollin/ megalith-10m, 2024.
- [35] W. Peebles and S. Xie. Scalable diffusion models with transformers. In CVPR, 2023.

- [36] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Müller, J. Penna, and R. Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [37] A. Polyak, A. Zohar, A. Brown, A. Tjandra, A. Sinha, A. Lee, A. Vyas, B. Shi, C.-Y. Ma, C.-Y. Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2025.
- [38] R. Pope, S. Douglas, A. Chowdhery, J. Devlin, J. Bradbury, J. Heek, K. Xiao, S. Agrawal, and J. Dean. Efficiently scaling transformer inference. JMLS, 2023.
- [39] ProGamerGov. synthetic-dataset-1m-dalle3-high-qualitycaptions. https://huggingface.co/datasets/ProGamerGov/ synthetic-dataset-1m-dalle3-high-quality-captions, 2022.
- [40] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021.
- [41] A. Radford, K. Narasimhan, T. Salimans, I. Sutskever, et al. Improving language understanding by generative pre-training. OpenAI Blog, 2018.
- [42] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 2020.
- [43] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.
- [44] A. Ramesh, M. Pavlov, G. Goh, S. Gray, C. Voss, A. Radford, M. Chen, and I. Sutskever. Zero-shot text-to-image generation. In ICML, 2021.
- [45] M. Ranzato, S. Chopra, M. Auli, and W. Zaremba. Sequence level training with recurrent neural networks. arXiv preprint arXiv:1511.06732, 2015.
- [46] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.
- [47] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [48] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [49] P. Sharma, N. Ding, S. Goodman, and R. Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In ACL, 2018.
- [50] F. Shi, Z. Luo, Y. Ge, Y. Yang, Y. Shan, and L. Wang. Taming scalable visual tokenizer for autoregressive image generation. arXiv preprint arXiv:2412.02692, 2024.
- [51] J. Song, C. Meng, and S. Ermon. Denoising diffusion implicit models. In ICLR, 2021.
- [52] Y. Song, C. Meng, R. Liao, and S. Ermon. Accelerating feedforward computation via parallel nonlinear equation solving. In ICML, 2021.
- [53] J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 2024.
- [54] K. Sun, J. Pan, Y. Ge, H. Li, H. Duan, X. Wu, R. Zhang, A. Zhou, Z. Qin, Y. Wang, et al. Journeydb: A benchmark for generative image understanding. In NeurIPS, 2023.
- [55] P. Sun, Y. Jiang, S. Chen, S. Zhang, B. Peng, P. Luo, and Z. Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.
- [56] C. Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.
- [57] Y. Teng, H. Shi, X. Liu, X. Ning, G. Dai, Y. Wang, Z. Li, and X. Liu. Accelerating autoregressive text-to-image generation with training-free speculative jacobi decoding. In ICLR, 2025.
- [58] K. Tian, Y. Jiang, Z. Yuan, B. Peng, and L. Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. arXiv preprint arXiv:2404.02905, 2024.

- [59] R. Tian, Q. Dai, J. Bao, K. Qiu, Y. Yang, C. Luo, Z. Wu, and Y.-G. Jiang. Reducio! generating 1024×1024 video within 16 seconds using extremely compressed motion latents. arXiv preprint arXiv:2411.13552, 2024.
- [60] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [61] L. von Werra, Y. Belkada, L. Tunstall, E. Beeching, T. Thrush, N. Lambert, S. Huang, K. Rasul, and Q. Gallouédec. Trl: Transformer reinforcement learning, 2020.
- [62] B. Wallace, M. Dang, R. Rafailov, L. Zhou, A. Lou, S. Purushwalkam, S. Ermon, C. Xiong, S. Joty, and N. Naik. Diffusion model alignment using direct preference optimization. In CVPR, 2024.
- [63] H. Wang, S. Suri, Y. Ren, H. Chen, and A. Shrivastava. Larp: Tokenizing videos with a learned autoregressive generative prior. In ICLR, 2025.
- [64] J. Wang, Y. Jiang, Z. Yuan, B. Peng, Z. Wu, and Y.-G. Jiang. Omnitokenizer: A joint imagevideo tokenizer for visual generation. In NeurIPS, 2024.
- [65] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [66] S. Wang, Z. Tian, W. Huang, and L. Wang. Ddt: Decoupled diffusion transformer. arXiv preprint arXiv:2504.05741, 2025.
- [67] X. Wang, X. Zhang, Z. Luo, Q. Sun, Y. Cui, J. Wang, F. Zhang, Y. Wang, Z. Li, Q. Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.
- [68] C. Wu, X. Chen, Z. Wu, Y. Ma, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, C. Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. In CVPR, 2024.
- [69] J. Wu, Y. Jiang, C. Ma, Y. Liu, H. Zhao, Z. Yuan, S. Bai, and X. Bai. Liquid: Language models are scalable multi-modal generators. arXiv preprint arXiv:2412.04332, 2024.
- [70] X. Wu, Y. Hao, K. Sun, Y. Chen, F. Zhu, R. Zhao, and H. Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.
- [71] Y. Wu, Z. Zhang, J. Chen, H. Tang, D. Li, Y. Fang, L. Zhu, E. Xie, H. Yin, L. Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. In ICLR, 2025.
- [72] J. Xie, W. Mao, Z. Bai, D. J. Zhang, W. Wang, K. Q. Lin, Y. Gu, Z. Chen, Z. Yang, and M. Z. Shou. Show-o: One single transformer to unify multimodal understanding and generation. In ICLR, 2025.
- [73] A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.
- [74] Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, X. Zhang, G. Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.
- [75] J. Yu, Y. Xu, J. Y. Koh, T. Luong, G. Baid, Z. Wang, V. Vasudevan, A. Ku, Y. Yang, B. K. Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. In ICLR, 2024.
- [76] C. Zhou, L. Yu, A. Babu, K. Tirumala, M. Yasunaga, L. Shamis, J. Kahn, X. Ma, L. Zettlemoyer, and O. Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. In ICLR, 2025.

