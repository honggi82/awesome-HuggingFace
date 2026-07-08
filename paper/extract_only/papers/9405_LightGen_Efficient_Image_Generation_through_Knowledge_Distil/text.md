# arXiv:2503.08619v1[cs.CV]11Mar2025

## LightGen: Efficient Image Generation through Knowledge Distillation and Direct Preference Optimization

Xianfeng Wu1,2* Yajing Bai1,2* Haoze Zheng1,2* Harold Haodong Chen1,2* Yexin Liu1,2* Zihao Wang1,2 Xuran Ma1,2 Wen-Jie Shu1,2 Xianzu Wu1,2 Harry Yang1,2† Ser-Nam Lim2,3†

1The Hong Kong University of Science and Technology, 2Everlyn AI, 3University of Central Florida *Core Contributor, †Corresponding Author

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Generation

[Figure 5]

[Figure 6]

[Figure 7]

| |
|---|

Zero-shot inpainting

12

Imagen

7132

10000

Flux

10000

WebLI

Flux

PixArt-𝛼 788

Laion

SD3

8

5000

5000

SD3

Lumina SDXL -Next Ours

Coyo

700

288

2.6

700

SD1.5

0.7 17.1x 81.0x

Ours

Ours 88

2 5000x

|0 1000 2000 3000 4000 5000 6000 7000|
|---|

|0 2 4 6 8 10 12|
|---|

0 2000 4000 6000 8000 10000

Dataset Size (M)

Model Parameters (B) GPU Hours (h)

Figure 1. Overview of LightGen’s capabilities in image generation, zero-shot inpainting, and resource usage. (First Row) Images generated at multiple resolutions (512 × 512 and 1024 × 1024) illustrate the scalability of LightGen. (Second Row) Zero-shot inpainting results showcasing LightGen’s inherent editing ability. (Third Row) LightGen’s resource consumption with drastically reduced dataset size, model parameters, and GPU hours compared to state-of-the-art models, demonstrates significant cost reductions without sacrificing performance.

#### Abstract

Recent advances in text-to-image generation have primarily relied on extensive datasets and parameter-heavy architectures. These requirements severely limit accessibility for researchers and practitioners who lack substantial computational resources. In this paper, we introduce LightGen, an efficient training paradigm for image generation mod-

els that uses knowledge distillation (KD) and Direct Preference Optimization (DPO). Drawing inspiration from the success of data KD techniques widely adopted in MultiModal Large Language Models (MLLMs), LightGen distills knowledge from state-of-the-art (SOTA) text-to-image models into a compact Masked Autoregressive (MAR) architecture with only 0.7B parameters. Using a compact synthetic dataset of just 2M high-quality images generated

from varied captions, we demonstrate that data diversity significantly outweighs data volume in determining model performance. This strategy dramatically reduces computational demands and reduces pre-training time from potentially thousands of GPU-days to merely 88 GPU-days. Furthermore, to address the inherent shortcomings of synthetic data, particularly poor high-frequency details and spatial inaccuracies, we integrate the DPO technique that refines image fidelity and positional accuracy. Comprehensive experiments confirm that LightGen achieves image generation quality comparable to SOTA models while significantly reducing computational resources and expanding accessibility for resource-constrained environments. Code is available at https://github.com/XianfengWu01/ LightGen

#### 1. Introduction

Text-to-image generation has witnessed significant advancements in recent years, with various generative models [16, 37, 55], including diffusion models [11, 22, 38, 49] and autoregressive models [12, 24], achieving remarkable results [9, 51]. These models have demonstrated the ability to generate high-quality images under diverse input conditions, such as text prompts, sketches, and lighting specifications [11, 22, 34]. However, their reliance on largescale datasets and parameter-heavy architectures imposes substantial costs in terms of data collection, training resources, and computational infrastructure. This makes these models less accessible to researchers and practitioners with limited access to high-performance GPU clusters or massive labeled datasets.

Although recent work has explored efficient pretraining strategies for label-to-image generation models [52], the development of lightweight and efficient training paradigms for text-to-image generation models remains largely underexplored. Text-to-image generation models introduce unique challenges due to embedding captions to visual tokens, it is hard to combine discrete tokens and continuous tokens, which inherently requires balancing model efficiency and performance. Moreover, the SOTA in textto-image generation often demands O(100M) to O(1B) of labeled images and extensive pretraining on highperformance GPU clusters. These requirements are prohibitively expensive and limit the accessibility of AR models for broader research and application.

In this work, we propose LightGen, a novel strategy for efficient pre-training of the MAR-based text-to-image generation model. We find that achieving comparable textto-image generation models does not necessarily require massive datasets and large parameters, and data diversity plays a more vital role than sheer data volume. Drawing on the success of data distillation in the MLLM [17, 43], we

utilize high-quality image data as pre-train data come from SOTA models generate, LightGen rapidly gets a comparable result in the 256px pre-train stage by learning SOTA models how to generate high-quality images. To this end, we propose leveraging synthetic datasets which use understanding caption data as the caption to ensure sufficient caption diversity distribution, then use SOTA text-to-image models to generate images as pre-train data to ensure image data are high-quality. By combining this strategy with a lightweight MAR architecture, LightGen significantly reduces the resource requirements of text-to-image generation models without sacrificing performance. As shown in Fig. 1, LightGen achieves efficiency in three key aspects:

▶ Data Efficiency: We only use 2M image data, compared to the size of pre-train data of SOTA models, we significantly reduce the data scale. (O(100M) → O(1M)). ▶ Parameter Efficiency: Compare with SOTA models using 2.6B, 8B, 12B and even larger parameters, we only use the 0.7B model to achieve a comparable performance with SOTA models.

▶ Training Efficiency: In the last text-to-image pre-traing stage, they often required at least 288 A100 GPU Days or more, even tens of thousands GPU Days, we only require 88 A100 GPU Days.

However, while synthetic data addresses the challenge of data diversity, it also introduces two key limitations: ❶ Poor high-frequency information, as synthetic data often lack fine-grained details present in real-world images; and ❷ Difficulty in position capture, we utilize data augmentation in the pre-train stage, which makes LightGen have misalignment or simplified spatial relationships problem. To overcome these issues, we incorporate a postprocessing technique called DPO (Direct Preference Optimization). DPO refines high-frequency details and enhances positional accuracy, effectively addressing the shortcomings of synthetic data. Its integration into the training pipeline further improves LightGen’s robustness and image quality under data-scale-constrained scenarios. Extensive experiments demonstrate that LightGen achieves performance comparable to SOTA image generation models while significantly reducing resource demands and training time. In summary, our contributions are as follows:

- • Efficient Text-to-Image Training Pipeline: We propose LightGen, a novel training framework for T2I generation models that significantly reduces the dependence on large-scale datasets and computational resources.
- • Synthetic Data Utilization: We find that data diversity, rather than sheer data volume, is critical for the performance of text-to-image models. Using diverse synthetic data generated by SOTA models, we achieve data efficiency while maintaining high-quality training signals.
- • Lightweight Architecture Design: LightGen incorporates a compact architecture, improving parameter effi-

ciency and reducing memory usage, making it suitable for resource-constrained environments.

• Post-Processing with DPO: To address the limitations of synthetic data (e.g., poor high-frequency details and position capture), we introduce Direct Preference Optimization (DPO) as a post-processing technique, enhancing both robustness and image quality.

#### 2. Related Work

##### 2.1. Diffusion Models

Diffusion models have become a fundamental class of generative models with the development of deep learning [4, 5, 8, 20, 40], particularly for tasks like image synthesis [9, 38]. These models operate by progressively adding noise to an image and then reversing this process to generate new samples. Their ability to model complex distributions has made them pivotal in generative modeling.

UNet-based Diffusion Models. UNet-based diffusion models leverage the UNet architecture, which employs a symmetric encoder-decoder structure with skip connections [6]. This architecture is particularly effective in denoising images during the reverse diffusion process, as demonstrated by recent work [19, 38]. However, a key limitation of UNet-based diffusion models is their inability to scale up efficiently. While these models perform well with existing architectures, further increasing the number of model parameters to improve generative capabilities has proven to be challenging.

Diffusion Transformer Models. To address the scalability limitations of UNet-based models, Diffusion Transformer (DiT) models have been introduced [13, 21, 30, 34, 36]. These models integrate transformer architectures into the diffusion process, enabling better scaling of parameters and leading to improved generative quality. DiT models can generate more intricate details and handle more complex inputs, outperforming UNet-based models when computational resources are available.

However, despite their superior scalability and generative quality, DiT models struggle to embed conditional signals—such as captions or specific image attributes—into the diffusion process. To overcome this, diffusion models often rely on external control mechanisms like ControlNet [56] to guide image generation with specific conditions.

##### 2.2. Autoregressive Models

The recent success of autoregressive (AR) models in the language domain [1] has led researchers to explore the possibility of using a unified model that can handle both text and visual generation tasks, so they have recently gained renewed attention in the field. AR models generate images token by token, conditioning each new token on the previ-

ously generated tokens, effectively capturing complex dependencies in natural images.

Pure AR Models. Pure AR models [3, 23, 29, 41, 54], such as Llamagen [41], generate images by modeling image tokens sequentially through constant-scale tokens. VAR [44] improves image quality by pretraining on multiscale tokens. While these models have succeeded in image generation tasks, their scalability remains challenging. As the models grow in size, the complexity of training increases, and improvements in image quality become less pronounced. Consequently, pure AR models struggle to compete with the latest diffusion-based methods, especially in terms of generative performance at larger scales.

AR + Diffusion Models. A promising direction in generative modeling combines the advantages of both AR and diffusion models [50]. By integrating the AR framework for token generation with a diffusion process to refine these generated tokens, these hybrid models aim to improve both the quality and efficiency of image generation. For example, MAR [24] and Fluid [12] combine the precision of AR token generation with refining power of diffusion processes.

These hybrid models have shown significant improvements in generative performance. The AR component provides fine-grained control over token-wise generation, while the diffusion component ensures that the overall image quality is enhanced. This combination allows the models to scale more efficiently, generating high-quality images that outperform both pure AR and pure diffusion models.

#### 3. Preliminary

In this section, we introduce the foundational concepts of diffusion models and AR models in the context of image generation. These models have played critical roles in advancing the state-of-the-art in generative modeling.

##### 3.1. Diffusion Models

Diffusion Models. Diffusion models learn a process of progressively adding noise to an image and then reversing this noising process to generate new samples. The process is typically divided into two phases: a forward process, where noise is gradually added to the image, and a reverse process, where the model learns to recover the original image by denoising it. This approach has proven to be highly effective in generating high-quality images.

Given an image x0, the forward diffusion process defines a Markov chain that progressively adds Gaussian noise to the image, resulting in a sequence of latent variables (x1,x2,··· ,xT), where T is the total number of diffusion steps. The forward process can be mathematically described as:

xt = 1 − βtxt−1 + βtϵt, ϵt ∼ N(0,I), (1)

[Figure 8]

(a) Training Pipeline (b) Inference Pipeline

Figure 2. Overview of LightGen efficient pretraining. (a) Training: Images are encoded into tokens via a pre-trained tokenizer, while text embeddings from a T5 encoder are refined by a trainable aligner. A masked autoencoder uses text tokens as queries/values and image tokens as keys for cross-attention, followed by refinement with a Diffusion MLP (D-MLP). (b) Inference: Tokens are predicted and iteratively refined over N steps, then decoded by the image tokenizer to generate final images.

where βt represents a variance schedule and ϵt is Gaussian noise added at each step.

The reverse diffusion process is trained to predict a clean image x0 from noisy observations xt. The model learns to approximate the reverse diffusion step:

p(xt−1|xt) = N(xt−1;µθ(xt,t),σt2I), (2)

where µθ is the model’s prediction at step t, and σt2 is the noise variance schedule.

##### 3.2. Autoregressive Models

AR Models. AR models generate images by sequentially producing their components (pixels or tokens), conditioning each new element on the previously generated ones. In the context of image generation, the model produces each pixel (or token) in a sequence, where each new element depends on the ones generated before it.

Given an image x = (x1,x2,··· ,xn), an AR model factorizes the joint distribution of the tokens as:

n

p(xi|x1,x2,··· ,xi−1), (3)

p(x) =

i=1

where p(xi|x1,··· ,xi−1) represents the conditional distribution of each token given its predecessors. This factorization allows the model to capture complex dependencies between pixels, resulting in high-quality images.

Random Autoregressive Models. Random autoregressive (RAR) models [53] extend the basic AR approach by generating tokens in random order. By introducing randomness into the token selection process, RAR models are able

to capture global dependencies more effectively compared to standard AR models, as described in Eq. (3). The probability distribution for RAR is given by:

n

p(xj

| xj

,xj

,··· ,xj

), (4)

p(x) =

i−1

i

1

2

i=1

where ji is a random, non-repeating index ranging from 1 to n, Supplementary Material prove that RAR is better than Pure AR in continous tokens’ filed.

Masked Autoregressive Models. Masked autoregressive models (MAR) [12, 24] generalize the RAR framework by predicting a set of tokens during each inference step. This formulation strikes a balance between the speed of inference and the generative capability. The probability distribution in MAR is given by:

n τ

p(xτ(i−1),xτ(i−1)+1,··· ,

p(x) =

(5)

i=1

xτi−1 | xj

,··· ,xτ(i−1)−1),

,xj

1

2

where τ represents the number of tokens predicted in one inference step, Supplementary Material has prove why MAR is better than RAR.

#### 4. Methodology

In this section, we detail the methodology used in this work, focusing on our proposed model architecture, training pipeline, and post-processing techniques. LightGen uses a novel pre-training strategy without modifying the architec-

ture to achieve comparable generative capabilities with few datasets and GPU resources.

##### 4.1. LightGen Pipeline

In this paper, LightGen build upon Fluid’s architecture [12], which serves as the base model for our approach. We also incorporate interpolated positional embeddings inspired by DINO [2, 33] to allow LightGen to generate images at different resolutions. This flexibility is crucial for training on multiple image scales and achieving high-quality image generation.

Fig. 2 (a) illustrates our training pipeline. The ground truth image is first passed through an image VAE encoder to obtain a latent representation of the image. The latent features are then patchified into continuous tokens I = {i1,i2,··· ,in}, which are subsequently masked in a random order at a high ratio. These masked tokens are input into a masked encoder-decoder architecture. The masked decoder generates token predictions based on the masked input, which are used as queries in a cross-attention mechanism.

For conditioning the model, we introduce text as an additional modality. We process the text using a T5 encoder [31] and align it with the image features through a text aligner. The processed text is then converted into discrete tokens T = {t1,t2,··· ,tn}, which serve as the keys and values in the cross-attention mechanism. This enables the model to generate semantic tokens S = {s1,s2,··· ,sn}.

These semantic tokens S are passed through a tiny diffusion MLP, denoted as D(·), which conditions the token generation process and generates high-quality image tokens I = { i1, i2,··· , in}. These generated image tokens are then conditioned on the ground truth latent representation I produced by the image VAE encoder. The model parameters are updated using the diffusion loss function:

n i=1 Eϵ,t ∥ϵ − ϵθ(ii

t|t,si)∥2 n

, (6)

Lθ(I,S) =

where ϵ ∈ Rd is a noise vector sampled from N(0,I). The noise-corrupted vector it is computed via Eq. (1). The variable t denotes the time step of the noise schedule. The noise estimator ϵθ, parameterized by θ, is a small MLP network, which takes it as input and is conditional on both t and si. The notation ϵθ(ii

t|t,si) indicates that the network takes it as input, conditioned on the time step t and the corresponding semantic token si.

Fig. 2 (b) shows the inference pipeline. The process begins by inputting noise tokens Z = {z1,z2,··· ,zn} of the same size into the masked encoder-decoder. The crossattention blocks then combine the text’s discrete tokens T with the visual tokens. The tiny diffusion MLP D(·) iteratively refines the tokens, acting as a denoising function to guide the tokens towards a high-quality image representa-

[Figure 9]

Figure 3. Illustrate of DPO Post-processing of LightGen.

tion. This process is repeated for N steps to further refine the tokens. Finally, the tokens I are reshaped and passed through the image VAE decoder to produce the final image.

##### 4.2. Post-processing

After pre-training on images of size 256 × 256 and finetuning on 512 × 512 images, we observe that the model achieves competitive image generation performance. However, we identify several potential issues with the generator. Specifically, the model was pre-trained on a dataset containing artifacts, and the preprocessing method imitate DIT [34] and ADM [9] will flip image cannot let models learning positional signals. As a result, the model may struggle to capture fine-grained spatial relationships in the generated images.

To address these concerns and improve the performance of our generator, we introduce Direct Preference Optimization (DPO) [35]. DPO is a post-processing technique that fine-tunes the model by optimizing its ability to generate high-quality images while incorporating positional signals. This post-processing stage helps the model better handle spatial relationships and refine the generated images, improving both quality and coherence.

DPO optimizes the model’s outputs by minimizing the difference between the generated and reference images under the learned preference model. Given a generated image xw0 and a reference image xl0, DPO [45] aims to minimize the following loss function:

LDPO(θ) = − E(xw

0 ,xl0)∼D,t∼U(0,T)logσ

− βTw(λt) Lθ(xwt ,t) − Lref(xwt ,t)

(7)

− (Lθ(xlt,t) − Lref(xlt,t)) ,

where σ(·) is sigmoid function. xwt ,xlt can calculate via Eq. (1). λt = α

2 t

σt2 is the signal-to-noise ratio, w(λt) a weighting function. Lδ(xγt ,t) = ∥ϵγ − ϵδ(xγt ,t)∥22,δ ∈ {θ,ref},γ ∈ {w,l},t ∼ U(0,T). This loss encourages ϵθ

Table 1. Performance comparison in 256 × 256 on GenEval [15]. Best results are shown in bold.

GenEval

Model #Params Pre-train Data

Single Obj. Two Obj. Colors Counting Position Color Attri. Overall Diffusion-based

Stable Diffusion v1.5 [38] 0.9B 2B 0.41 0.05 0.30 0.02 0.01 0.01 0.13 Stable Diffusion v2.1 [38] 0.9B 5B 0.11 0.01 0.04 0.01 0.00 0.00 0.02 Stable Diffusion XL [38] 2.6B - 0.05 0.03 0.30 0.01 0.00 0.00 0.01 Stable Diffusion 3 [11] 8B - 0.89 0.62 0.70 0.30 0.17 0.33 0.50 Flux [22] - 12B 0.96 0.58 0.77 0.48 0.13 0.33 0.54 Autoregressive

Llamagen [41] 0.7B 50M 0.69 0.34 0.55 0.19 0.06 0.02 0.31 LightGen w/o DPO (80k steps) 0.7B 2M 0.98 0.44 0.85 0.36 0.08 0.24 0.49 LightGen w/o DPO 0.7B 2M 0.99 0.53 0.85 0.40 0.13 0.25 0.53

Table 2. Performance comparison in 512 × 512 on GenEval [15].

GenEval

Model #Params Pre-train Data

Single Obj. Two Obj. Colors Counting Position Color Attri. Overall Diffusion-based

Stable Diffusion v1.5 [38] 0.9B 2B 0.96 0.38 0.77 0.37 0.03 0.05 0.42 Stable Diffusion v2.1 [38] 0.9B 2B 0.91 0.24 0.69 0.14 0.03 0.06 0.34 Stable Diffusion XL [38] 2.6B 5B 0.63 0.23 0.51 0.12 0.04 0.05 0.26 Stable Diffusion 3 [11] 8B - 0.99 0.82 0.80 0.51 0.27 0.52 0.65 Flux [22] 12B - 0.97 0.71 0.68 0.76 0.15 0.43 0.61 Autoregressive

Llamagen [41] 0.7B 50M 0.19 0.16 0.10 0.03 0.09 0.01 0.10 LightGen w/o DPO 0.7B 2M 0.98 0.58 0.86 0.37 0.14 0.28 0.53 LightGen 0.7B 2M 0.99 0.65 0.87 0.60 0.22 0.43 0.62

to improve more at denoising xwt than xlt. The DPO optimization process iteratively refines the generated image by updating the model parameters to align the generated output more closely with the reference image in Fig. 3. This results in improved generalization, allowing the model to generate higher-quality images across a wider range of tasks.

By applying DPO after the pre-training and fine-tuning stages, we refine the model to generate better quality image, and it can handle positional signals and spatial relationships.

##### 4.3. Data Distillation

In contrast to SOTA image generative models use large image data and employ complex data pipelines to get large data to pretrain, LightGen leverages an artifact dataset [57] use understanding dataset [10, 25, 26] to ensure that caption set T = {t1,t2,··· ,tn} has enough diversity. Subsequently, SOTA image generators (e.g., DALLE3 [32], Flux [22]) transform semantic richness caption into a visually diverse dataset D = {d1,d2,··· ,dn}.

During both the pre-training and high-resolution stages, our target is to learn the feature representations (logits) of the artifact dataset like offline knowledge distillation [18]. Instead of relying on hard targets, our model learns from the soft targets provided by SOTA models. Formally, we define our objective as:

n

1 n

t | t 2 , (8) where fi = Enc(di) denotes the feature representation (log-

Eϵ,t ϵ − ϵθ fi

Lθ(D) =

i=1

its) extracted from the image di (generate by SOTA models) by a image VAE encoder. ϵθ is our student model parameterized by θ.

Under assumptions, the gradient of the loss with respect to the parameters θ is given by:

n

2 n

Eϵ,t [(ϵθ(fi

∇θLθ(D) =

t | t) − ϵ)∇θϵθ(fi

t | t)].

i=1

(9) If ϵθ is sufficiently expressive, then with enough diverse samples the gradient becomes an unbiased estimator (it will prove in Supplementary Material) of the true gradient of the expected loss. Hence, this method ensuring that the student model converges towards approximating the teacher’s behavior.

#### 5. Experiment

In this section, we present our experimental setup and results. We first describe the datasets, precomputation of image and text features, and the training/inference procedures. We then compare our method to previous systems on standard benchmarks, followed by an ablation study examining the key design choices in our approach.

##### 5.1. Experimental Setup

Datasets. We conduct experiments on a samll highquality artifact dataset [57], which use SOTA MLLM like GPT-4o [1] and Qwen2VL [46] to generate high-quality

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

[Figure 31]

Figure 4. Visualization Results. Sample outputs generated using LightGen, showcasing high-quality images at multiple resolutions (256 × 256, 512 × 512, 1024 × 1024) and across diverse styles (realistic, animated, virtual, etc.), which demonstrate the versatility and scalability of our approach.

[Figure 32]

Figure 5. Image inpainting demonstrations.

caption, then use SOTA image generation models like Flux [22] and DALLE3 [32] to generate high-quality image datas. To maintain consistent input dimensions, we center-crop and resize images to 256 × 256 for the initial pre-training stage, then fine-tune at 512 × 512. In addition, we evaluate on GenEval [15] and FID to measure text alignment and generation quality. For the DPO training stage, we use LAION-aesthethic [39] as the DPO dataset and use Qwen2.5VL to filter out some low quality data.

Precomputation of Image and Text Features. To reduce redundant encoding costs and stabilize training:

- • Image Features: We use a high-quality VAE [11] to obtain latent representations of the images. Each image is encoded into a set of continuous latent tokens.
- • Text Features: We employ a T5-XXL encoder [7] to convert text prompts into high-dimensional embeddings. These embeddings are then tokenized or projected to match the dimensions expected by our model.

Training & Post-processing. Unless otherwise specified, we train our model using the AdamW optimizer [27] with β1 = 0.9, β2 = 0.99, and a weight decay of 0.02. We use a linear warm-up schedule for the first few epochs, then use constant learning rate in 1e − 4. The initial pre-training is conducted for 100k steps at 256×256 resolution using a total batch size of 2048. For fine-tuning at 512×512, we train for an additional few steps, using a reduced learning rate of 1e − 5 (0.1x the base learning rate). We using same training setting in DPO training stage, and we use very samll constant learning rate 1e − 8 and β in Eq. (7) is 5000.

Inference. During inference, we follow the pipeline described in Section 4. We use the precomputed T5-XXL text embeddings as conditioning and sample noise vectors for the image latents. We perform 64 iterative refinement steps using our masked autoregressive + diffusion approach. The final latent representations are then decoded back to the pixel space via the VAE decoder.

Evaluation. We report the performance on standard benchmarks GenEval [15] for image quality evaluation.

##### 5.2. Main Results

We compare our method against several SOTA text-toimage models, including Diffusion-based model(e.g. Stable Diffusion [11, 38] and Flux [22]), Autoregressive models (e.g. Llamagen [41]).

Quantitative Results. Tab. 1 and Tab. 2 illustrate the performance of LightGen on the GenEval benchmark in resolutions of 256 × 256 and 512 × 512. At 256 × 256 resolution, LightGen significantly outperforms diffusion-based and autoregressive models in the task Single Object, Two Objects, and Colors, achieving overall performance scores of 0.49 (80k steps without DPO) and 0.53 (without DPO).

|(a)|
|---|

| |
|---|
| |

|[Figure 33]<br><br>|
|---|

|Overall Single Obj. Two Obj. Counting Colors Position Color Attr.|
|---|

(b)

|[Figure 34]<br><br>|
|---|

|Overall Single Obj. Two Obj. Counting Colors Position Color Attr.|
|---|

Figure 6. Ablation studies. (a) Pre-train in different data scales, we find it achieves the limitation when pre-train in O(1M) data scale. (b) demonstrate different iteration results.

At higher 512×512 resolution, LightGen achieves a comparable result with an overall score of 0.62, almost surpassing all SOTA models. In particular, the integration of DPO consistently enhances performance in positional accuracy and high-frequency details, underscoring its effectiveness in addressing the limitations of synthetic data.

Qualitative Results. Fig. 4 and Fig. 5 visually demonstrate the capabilities and versatility of LightGen. Figure 4 shows sample outputs at multiple resolutions (256 × 256, 512 × 512, 1024 × 1024) and in various artistic styles, including realistic, animated, and virtual images. Fig. 5 further illustrates the strength of LightGen through image inpainting demonstrations.

##### 5.3. Ablation Study

Fig. 6 (a) analyzes the impact of varying the pre-training data scale. The results indicate that performance reaches a bottleneck when the dataset scale approaches approximately 1M images, suggesting that increasing beyond this scale yields diminishing returns. Consequently, we select 2M images as our optimal pre-training dataset size. Fig. 6 (b) explores the effects of different training iterations on GenEval performance at resolutions of 256px and 512px. Notably, we observe that at the 256px stage, reasonable performance can be achieved after just 80k training iterations, highlighting the efficiency of data distillation.

#### 6. Conclusion

In this paper, we propose LightGen, a novel and efficient pipeline to accelerate text-to-image generation training through KD and DPO. Our results demonstrate that efficient image generation can be realized without sacrificing quality by focusing on data diversity, compact architectures, and pre-train strategic. The proposed methodology opens avenues for exploring similar efficiency-focused training paradigms in related generative tasks such as video generation. We hope that our work encourages further research in developing accessible and efficient generative models.

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 3, 6

- [2] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 9650–9660, 2021. 5
- [3] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. Maskgit: Masked generative image transformer. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 3
- [4] Haodong Chen, Haojian Huang, Junhao Dong, Mingzhe Zheng, and Dian Shao. Finecliper: Multi-modal fine-grained clip for dynamic facial expression recognition with adapters. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 2301–2310, 2024. 3
- [5] Haodong Chen, Yongle Huang, Haojian Huang, Xiangsheng Ge, and Dian Shao. Gaussianvton: 3d human virtual tryon via multi-stage gaussian splatting editing with image prompting. arXiv preprint arXiv:2405.07472, 2024. 3
- [6] Haodong Chen, Lan Wang, Harry Yang, and Ser-Nam Lim. Omnicreator: Self-supervised unified generation with universal editing. arXiv preprint arXiv:2412.02114, 2024. 3
- [7] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70):1–53, 2024. 8
- [8] Haoyu Deng, Zijing Xu, Yule Duan, Xiao Wu, Wenjie Shu, and Liang-Jian Deng. Exploring the low-pass filtering behavior in image super-resolution. arXiv preprint arXiv:2405.07919, 2024. 3
- [9] Prafulla Dhariwal and Alexander Quinn Nichol. Diffusion

- models beat GANs on image synthesis. In Advances in Neural Information Processing Systems, 2021. 2, 3, 5
- [10] Ben Egan, Alex Redden, XWAVE, and SilentAntagonist. Dalle3 1 Million+ High Quality Captions, 2024. 6
- [11] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In Proceedings of the 41st International Conference on Machine Learning. JMLR.org, 2024. 2, 6, 8, 13
- [12] Lijie Fan, Tianhong Li, Siyang Qin, Yuanzhen Li, Chen Sun, Michael Rubinstein, Deqing Sun, Kaiming He, and Yonglong Tian. Scaling autoregressive text-to-image generative models with continuous tokens. In The Thirteenth International Conference on Learning Representations, 2025. 2, 3, 4, 5, 14
- [13] Shanghua Gao, Pan Zhou, Ming-Ming Cheng, and Shuicheng Yan. Masked diffusion transformer is a strong image synthesizer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 23164–23173,

2023. 3

- [14] Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation, 2025. 13
- [15] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating textto-image alignment. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023. 6, 8, 13
- [16] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks, 2014. 2
- [17] Muyang He, Yexin Liu, Boya Wu, Jianhao Yuan, Yueze Wang, Tiejun Huang, and Bo Zhao. Efficient multimodal learning from data-centric perspective. arXiv preprint arXiv:2402.11530, 2024. 2
- [18] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015. 6
- [19] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. Simple diffusion: end-to-end diffusion for high resolution images. In Proceedings of the 40th International Conference on Machine Learning. JMLR.org, 2023. 3
- [20] Yongle Huang, Haodong Chen, Zhenbang Xu, Zihan Jia, Haozhou Sun, and Dian Shao. Sefar: Semi-supervised fine-grained action recognition with temporal perturbation and learning stabilization. arXiv preprint arXiv:2501.01245,

2025. 3

- [21] Liya Ji, Zhefan Rao, Sinno Jialin Pan, Chenyang Lei, and Qifeng Chen. A diffusion model with state estimation for degradation-blind inverse imaging. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2471– 2479, 2024. 3
- [22] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 2, 6, 8, 13

- [23] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11523–11532, 2022. 3
- [24] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. In Advances in Neural Information Processing Systems, pages 56424–56445. Curran Associates, Inc.,

2024. 2, 3, 4, 14

- [25] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 6
- [26] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 6
- [27] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. 8
- [28] Tianyu Luan, Yuanhao Zhai, Jingjing Meng, Zhong Li, Zhang Chen, Yi Xu, and Junsong Yuan. High fidelity 3d hand shape reconstruction via scalable graph frequency decomposition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16795– 16804, 2023. 12
- [29] Zhuoyan Luo, Fengyuan Shi, Yixiao Ge, Yujiu Yang, Limin Wang, and Ying Shan. Open-magvit2: An open-source project toward democratizing auto-regressive visual generation, 2025. 3
- [30] Nanye Ma, Mark Goldstein, Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In Computer Vision – ECCV 2024, pages 23–40, Cham, 2024. Springer Nature Switzerland. 3
- [31] Jianmo Ni, Gustavo Hernandez Abrego, Noah Constant, Ji Ma, Keith Hall, Daniel Cer, and Yinfei Yang. Sentencet5: Scalable sentence encoders from pre-trained text-to-text models. In Findings of the Association for Computational Linguistics: ACL 2022, pages 1864–1874, Dublin, Ireland,

2022. Association for Computational Linguistics. 5

- [32] OpenAI. Dall·e 3 system card, 2023. 6, 8, 13
- [33] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. Featured Certification. 5
- [34] William Peebles and Saining Xie. Scalable diffusion models with transformers. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 4172–4182, 2023. 2, 3, 5
- [35] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct

- preference optimization: Your language model is secretly a reward model. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 5
- [36] Zhefan Rao, Liya Ji, Yazhou Xing, Runtao Liu, Zhaoyang Liu, Jiaxin Xie, Ziqiao Peng, Yingqing He, and Qifeng Chen. Modelgrow: Continual text-to-video pre-training with model expansion and language understanding enhancement. arXiv preprint arXiv:2412.18966, 2024. 3
- [37] Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In Proceedings of the 32nd International Conference on Machine Learning, pages 1530–1538, Lille, France, 2015. PMLR. 2
- [38] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022. 2, 3, 6, 8, 13
- [39] C. Schuhmann. Laion-aesthetics, 2022. 8
- [40] Wen-Jie Shu, Hong-Xia Dou, Rui Wen, Xiao Wu, and LiangJian Deng. Cmt: Cross modulation transformer with hybrid loss for pansharpening. IEEE Geoscience and Remote Sensing Letters, 21:1–5, 2024. 3
- [41] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation, 2024. 3, 6, 8, 13
- [42] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models, 2024. 13
- [43] DeepSeek-AI Team. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. 2
- [44] Keyu Tian, Yi Jiang, Zehuan Yuan, BINGYUE PENG, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. In The Thirtyeighth Annual Conference on Neural Information Processing Systems, 2024. 3
- [45] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8228–8238, 2024. 5
- [46] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution, 2024. 6
- [47] Xianfeng Wu, Xinyi Liu, Junfei Wang, Zhongyuan Lai, Jing Zhou, and Xia Liu. Point cloud classification based on transformer. Computers and Electrical Engineering, 104:108413,

2022. 12

- [48] Xianzu Wu, Xianfeng Wu, Tianyu Luan, Yajing Bai, Zhongyuan Lai, and Junsong Yuan. Fsc: Few-point shape completion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26077– 26087, 2024. 12

- [49] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340, 2024. 2
- [50] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. In The Thirteenth International Conference on Learning Representations, 2025. 3, 13
- [51] Chao Yang, Xin Lu, Zhe Lin, Eli Shechtman, Oliver Wang, and Hao Li. High-resolution image inpainting using multiscale neural patch synthesis. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 2
- [52] Jingfeng Yao, Cheng Wang, Wenyu Liu, and Xinggang Wang. Fasterdit: Towards faster diffusion transformers training without architecture modification. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 2
- [53] Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, and LiangChieh Chen. Randomized autoregressive visual generation. arXiv preprint arXiv:2411.00776, 2024. 4
- [54] Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, and LiangChieh Chen. Randomized autoregressive visual generation. arxiv, 2024. 3, 12
- [55] Yuanhao Zhai, Tianyu Luan, David Doermann, and Junsong Yuan. Towards generic image manipulation detection with weakly-supervised self-consistency learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22390–22400, 2023. 2
- [56] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 3836–3847, 2023. 3
- [57] Kai Zhou. text-to-image-2m, 2024. 6

## LightGen: Efficient Image Generation through Knowledge Distillation and Direct Preference Optimization

### Supplementary Material

[Figure 35]

Figure 7. Illustrate three different paradigms of image generative models.

#### A. RAR is better than Pure AR

In the traditional deep learning field [28, 47, 48], especially MLLM, Pure AR achieves great success. The probability distribution for RAR models [54], as illustrated in Fig. 7 (b), is defined as:

n

p(xj

| xj

,xj

,··· ,xj

). (10)

p(x) =

i−1

i

1

2

i=1

During training, the distribution in RAR can be approximated by:

n

p(xi | xS\i), (11)

(·) ≈

pR

θ

i=1

where S = [1,n] represents the set of all token indices, and xS\i denotes all tokens except the current token xi. In this formulation, each token xi is inferred using the global context provided by all other tokens. This global condition-

ing enables RAR models to capture richer inter-token dependencies compared to traditional AR models, which rely solely on a sequential left-to-right generation process.

The continuous representations utilized in DIT have demonstrated exceptional generative capabilities in the visual generation domain, suggesting the inherent preference for continuous and global context in visual content generation, such as images and videos. RAR employs a bidirectional inference approach analogous to continuous representation in diffusion models, thereby leveraging global contextual information. This bidirectional conditioning enables RAR models to better approximate the superior generative performance of diffusion-based methods compared to pure AR models, leading to improved image fidelity and consistency.

Table 3. Performance comparison with other work’s paper demonstrate on GenEval [15].

GenEval

Model #Params Pre-train Data

Single Obj. Two Obj. Colors Counting Position Color Attri. Overall Diffusion-based

Stable Diffusion v1.5 [38] 0.9B 2B 0.97 0.38 0.76 0.35 0.04 0.06 0.43 Stable Diffusion v2.1 [38] 0.9B 2B 0.98 0.51 0.85 0.44 0.07 0.17 0.50 Stable Diffusion XL [38] 2.6B 5B 0.98 0.74 0.85 0.39 0.15 0.23 0.55 DALLE 3 [32] - - 0.96 0.87 0.83 0.47 0.43 0.45 0.67 Stable Diffusion 3 [11] 8B - 0.98 0.84 0.74 0.66 0.40 0.43 0.68 Flux [22] 12B - 0.98 0.81 0.79 0.74 0.22 0.45 0.66 Autoregressive

Llamagen [41] 0.7B 50M 0.19 0.16 0.10 0.03 0.09 0.01 0.10 Chameleon [42] 7B - - - - - - - 0.39 SEED-X [14] 17B - 0.96 0.65 0.80 0.31 0.18 0.14 0.51 Show-o [50] 1.3B - 0.95 0.52 0.82 0.49 0.11 0.28 0.53 LightGen 0.7B 2M 0.99 0.65 0.87 0.60 0.22 0.43 0.62

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

Figure 8. Additional generation result.

#### B. Why MAR is better?

The probability distribution in MAR [12, 24] in Fig. 7 (c) is given by:

n τ

p(xτ(i−1),xτ(i−1)+1,··· ,

p(x) =

(12)

i=1

,··· ,xτ(i−1)−1),

xτi−1 | xj

,xj

2

1

where τ represents the number of tokens predicted in one inference step

When τ = 1, MAR reduces to the RAR model. By increasing τ, MAR offers a trade-off between:

- • Performance: Smaller values of τ capture fine-grained dependencies between tokens, improving the quality of image generation.
- • Efficiency: Larger values of τ speed up inference by enabling parallelization of token predictions, reducing computation time.

This balance allows MAR models to achieve better performance and efficiency compared to both standard AR and RAR models.

#### C. Unbiased Estimator To see why the gradient

n

2 n

Eϵ,t [(ϵθ(fi

t | t) − ϵ)∇θϵθ(fi

t | t)],

∇θLθ(D) =

i=1

(13) can be regarded as an unbiased estimator, note the following:

t | t). (14) Then the gradient can be expressed as:

t | t) − ϵ)∇θϵθ(fi

g(θ,ϵ,t) = (ϵθ(fi

n

2 n

Eϵ,t[g(θ,ϵ,t)]. (15)

∇θLθ(D) =

i=1

When approximating the expectation using Monte Carlo samples {(ϵj,tj)}Mj=1, the estimator becomes:

n

M

2 n

1 M

g(θ,ϵj,tj). (16)

∇θLθ(D) =

i=1

j=1

###### Due to the linearity of expectation, we have:

n

2 n

E ∇θLθ(D) =

Eϵ,t[g(θ,ϵ,t)] = ∇θLθ(D).

i=1

(17)

Under appropriate regularity conditions (such as differentiability and integrability), the gradient operator can be interchanged with the expectation operator:

∇θEϵ,t[g(θ,ϵ,t)] = Eϵ,t[∇θg(θ,ϵ,t)]. (18)

Therefore, the sample-based gradient computed in Eq. (13) is an unbiased estimator of the true gradient.

