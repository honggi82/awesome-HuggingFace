## TextCraftor: Your Text Encoder Can be Image Quality Controller

Yanyu Li1,2 Xian Liu1 Anil Kag1 Ju Hu1 Yerlan Idelbayev1 Dhritiman Sagar1 Yanzhi Wang2 Sergey Tulyakov1 Jian Ren1

1Snap Inc. 2Northeastern University

# arXiv:2403.18978v1[cs.CV]27Mar2024

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

a plate with white rice topped by cooked vegetables a thumbnail image of a gingerbread man a bowl with a cartoon dinosaur on it

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

a girl with long curly blonde hair and sunglasses world's best brother t-shirt a frustrated child

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

a cartoon of a house on a mountain a cartoon of a boy playing with a tiger an owl standing on a telephone wire

Figure 1. Example generated images. For each prompt, we show images generated from three different models, which are SDv1.5, TextCraftor, TextCraftor + UNet, listed from left to right. The random seed is fixed for all generation results.

### Abstract

Diffusion-based text-to-image generative models, e.g., Stable Diffusion, have revolutionized the field of content generation, enabling significant advancements in areas like image editing and video synthesis. Despite their formidable capabilities, these models are not without their limitations. It is still challenging to synthesize an image that aligns well with the input text, and multiple runs with carefully crafted prompts are required to achieve satisfactory results. To mitigate these limitations, numerous studies have endeavored to fine-tune the pre-trained diffusion models, i.e., UNet, utilizing various technologies. Yet, amidst these efforts, a pivotal question of text-to-image diffusion model training has remained largely unexplored: Is it possible and feasible to fine-tune the text encoder to improve the performance of text-to-image diffusion models? Our findings reveal that, instead of replacing the CLIP text encoder used in Stable Diffusion with other large language models, we can enhance it through our proposed fine-tuning approach, TextCraftor, leading to substantial improvements in quantitative benchmarks and human assessments. Interestingly, our technique also empowers controllable image generation through the interpolation of different text encoders

fine-tuned with various rewards. We also demonstrate that TextCraftor is orthogonal to UNet finetuning, and can be combined to further improve generative quality.

### 1. Introduction

Recent breakthroughs in text-to-image diffusion models have brought about a revolution in content generation [10, 18, 28, 41, 52]. Among these models, the open-sourced Stable Diffusion (SD) has emerged as the de facto choice for a wide range of applications, including image editing, super-resolution, and video synthesis [4, 19, 26, 30, 32, 43, 45, 48, 61]. Though trained on large-scale datasets, SD still holds two major challenges. First, it often produces images that do not align well with the provided prompts [5, 58]. Second, generating visually pleasing images frequently requires multiple runs with different random seeds and manual prompt engineering [13, 54]. To address the first challenge, prior studies explore the substitution of the CLIP text encoder [37] used in SD with other large language models like T5 [7, 44]. Nevertheless, the large T5 model has an order of magnitude more parameters than CLIP, resulting in additional storage and computation overhead. In tack-

ling the second challenge, existing works fine-tune the pretrained UNet from SD on paired image-caption datasets with reward functions [8, 35, 57]. Nonetheless, models trained on constrained datasets may still struggle to generate high-quality images for unseen prompts.

Stepping back and considering the pipeline of text-toimage generation, the text encoder and UNet should both significantly influence the quality of the synthesized images. Despite substantial progress in enhancing the UNet model [15, 47], limited attention has been paid to improving the text encoder. This work aims to answer a pivotal question: Can fine-tuning a pre-trained text encoder used in the generative model enhance performance, resulting in better image quality and improved text-image alignment?

To address this challenge, we propose TextCraftor, an end-to-end fine-tuning technique to enhance the pre-trained text encoder. Instead of relying on paired text-image datasets, we demonstrate that reward functions (e.g., models trained to automatically assess the image quality like aesthetics model [1], or text-image alignment assessment models [24, 55]) can be used to improve text-encoder in a differentiable manner. By only necessitating text prompts during training, TextCraftor enables the on-the-fly synthesis of training images and alleviates the burden of storing and loading large-scale image datasets. We summarize our findings and contributions as follows:

- • We demonstrate that for a well-trained text-to-image diffusion model, fine-tuning text encoder is a buried gem, and can lead to significant improvements in image quality and text-image alignment (as in Fig. 1 & 3). Compared with using larger text encoders, e.g., SDXL, TextCraftor does not introduce extra computation and storage overhead. Compared with prompt engineering, TextCraftor reduces the risks of generating irrelevant content.
- • We introduce an effective and stable text encoder finetuning pipeline supervised by public reward functions. The proposed alignment constraint preserves the capability and generality of the large-scale CLIP-pretrained text encoder, making TextCraftor the first generic reward fine-tuning paradigm among concurrent arts. Comprehensive evaluations on public benchmarks and human assessments demonstrate the superiority of TextCraftor.
- • We show that the textual embedding from different finetuned and original text encoders can be interpolated to achieve more diverse and controllable style generation. Additionally, TextCraftor is orthogonal to UNet finetuning. We further show quality improvements by subsequently fine-tuning UNet with the improved text encoder.

### 2. Related Works

Text-to-Image Diffusion Models. Recent efforts in the synthesis of high-quality, high-resolution images from natural language inputs have showcased substantial progress [2,

41]. Diverse investigations have been conducted to improve model performance by employing various network architectures and training pipelines, such as GAN-based approaches [21], auto-regressive models [31, 59], and diffusion models [18, 22, 49, 51, 52]. Since the introduction of the Stable Diffusion models and their state-of-the-art performance in image generation and editing tasks, they have emerged as the predominant choice [41]. Nevertheless, they exhibit certain limitations. For instance, the generated images may not align well with the provided text prompts [58]. Furthermore, achieving high-quality images may necessitate extensive prompt engineering and multiple runs with different random seeds [13, 54]. To address these challenges, one potential improvement involves replacing the pre-trained CLIP text-encoder [37] in the Stable Diffusion model with T5 [7] and fine-tuning the model using highquality paired data [9, 44]. However, it is crucial to note that such an approach incurs a substantial training cost. Training the Stable Diffusion model alone from scratch demands considerable resources, equivalent to 6,250 A100 GPUs days [5]. This work improves pre-trained text-to-image models while significantly reducing computation costs.

Automated Performance Assessment of Text-to-Image Models. Assessing the performance of text-to-image models has been a challenging problem. Early methods use automatic metrics like FID to gauge image quality and CLIP scores to assess text-image alignment [38, 39]. However, subsequent studies have indicated that these scores exhibit limited correlation with human perception [34]. To address such discrepancies, recent research has delved into training models specifically designed for evaluating image quality for text-to-image models. Examples include ImageReward [57], PickScore [24], and human preference scores [55, 56], which leverage human-annotated images to train the quality estimation models. In our work, we leverage these models, along with an image aesthetics model [1], as reward functions for enhancing visual quality and textimage alignment for the text-to-image diffusion models.

Fine-tuning Diffusion Models with Rewards. In response to the inherent limitations of pre-trained diffusion models, various strategies have been proposed to elevate generation quality, focusing on aspects like image color, composition, and background [11, 25]. One direction utilizes reinforcement learning to fine-tune the diffusion model [3, 12]. Another area fine-tunes the diffusion models with reward function in a differentiable manner [57]. Following this trend, later studies extend the pipeline to trainable LoRA weights [20] with the text-to-image models [8, 35]. In our work, we delve into the novel exploration of fine-tuning the text-encoder using reward functions in a differentiable manner, a dimension that has not been previously explored.

Improving Textual Representation. Another avenue of research focuses on enhancing user-provided text to gen-

erate images of enhanced quality. Researchers use large language models, such as LLAMA [53], to refine or optimize text prompts [14, 36, 62]. By improving the quality of prompts, the text-to-image model can synthesize higherquality images. However, the utilization of additional language models introduces increased computational and storage demands. This study demonstrates that by fine-tuning the text encoder, the model can gain a more nuanced understanding of the given text prompts, obviating the need for additional language models and their associated overhead.

### 3. Method

#### 3.1. Preliminaries of Latent Diffusion Models

Latent Diffusion Models. Diffusion models convert the real data distribution e.g., images, into a noisy distribution, e.g., Gaussian distribution, and can reverse such a process to for randomly sampling [49]. To reduce the computation cost, e.g., the number of denoising steps, latent diffusion model (LDM) proposes to conduct the denoising process in the latent space [41] using a UNet [18, 42], where real data is encoded through variational autoencoder (VAE) [23, 40]. The latent is then decoded into an image during inference time. LDM demonstrates promising results for text-conditioned image generation. Trained with large-scale text-image paired datasets [46], a series of LDM models, namely, Stable Diffusion [41], are obtained. The text prompts are processed by a pre-trained text encoder, which is the one from CLIP [37] used by Stable Diffusion, to obtain textual embedding as the condition for image generation. In this work, we use the Stable Diffusion as the baseline model to conduct most of our experiments, as it is widely adopted in the community for various tasks.

Formally, let (x, p) be the real-image and prompt data pair (for notation simplicity, x also represents the data encoded by VAE) drawn from the distribution pdata(x,p), ϵˆθ(·) be the diffusion model with parameters θ, Tφ(·) be the text encoder parameterized by φ, training the text-toimage LDM under the objective of noise prediction can be formulated as follows [18, 49, 52]:

Et∼U[0,1],(x,p)∼pdata(x,p),ϵ∼N(0,I) ||ϵˆθ(t, zt, c)−ϵ||22, (1)

min

θ

where ϵ is the ground-truth noise; t is the time step; zt = αtx+σtϵ is the noised sample with αt represents the signal and σt represents the noise, that both decided by the scheduler; and c is the textual embedding such that c = Tφ(p).

During the training of SD models, the weights of text encoder T are fixed. However, the text encoder from CLIP model is optimized through the contrastive objective between text and images. Therefore, it does not necessarily learn the semantic meaning of the prompt, resulting the generated image might not align well with the given prompt using such a text encoder. In Sec. 3.2, we introduce the

technique of improving the text encoder without using the text and image contrastive pre-training in CLIP [37].

Denoising Scheduler – DDIM. After a text-to-image diffusion model is trained, we can sample Gaussian noises for the same text prompt using numerous samplers, such as DDIM [50], that iteratively samples from t to its previous step t′ with the following denoising process, until t becomes 0:

zt − σtϵˆθ(t, zt, c) αt

+ σt′ϵˆθ(t, zt, c). (2)

zt′ = αt′

Classifier-Free Guidance. One effective approach to improving the generation quality during the sampling stage is the classifier-free guidance (CFG) [17]. By adjusting the guidance scale w in CFG, we can further balance the trade-off between the fidelity and the text-image alignment of the synthesized image. Specifically, for the process of text-conditioned image generation, by letting ∅ denote the null text input, classifier-free guidance can be defined as follows:

ϵˆ = wϵˆθ(t, zt, c) − (w − 1)ϵˆθ(t, zt, ∅). (3)

#### 3.2. Text Encoder Fine-tuning with Reward Propagation

We introduce and experiment with two techniques for finetuning the text encoder by reward guidance.

##### 3.2.1 Directly Fine-tuning with Reward

Recall that for a normal training process of diffusion models, we sample from real data and random noise to perform forward diffusion: zt = αtx + σtϵ, upon which the denoising UNet, ϵˆθ(·), makes its (noise) prediction. Therefore, instead of calculating zt′ as in Eqn. 2, we can alternatively predict the original data as follows [50],

zt − σtϵˆθ(t, zt, Tφ(p)) αt

, (4)

ˆx =

where xˆ is the estimated real sample, which is an image for the text-to-image diffusion model. Our formulation works for both pixel-space and latent-space diffusion models, where in latent diffusion, xˆ is actually post-processed by the VAE decoder before feeding into reward models. Since the decoding process is also differentiable, for simplicity, we omit this process in formulations and simply refer xˆ as the predicted image. With xˆ in hand, we are able to utilize public reward models, denoted as R, to assess the quality of the generated image. Therefore, to improve the text encoder used in the diffusion model, we can optimize its weights, i.e., φ in T , with the learning objective as maximizing the quality scores predicted by reward models.

More specifically, we employ both image-based reward model R(xˆ), i.e., Aesthetic score predictor [1], and

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

- Figure 2. Overview of TextCraftor, an end-to-end text encoder fine-tuning paradigm based on prompt data and reward functions. The text embedding is forwarded into the DDIM denoising chain to obtain the output image and compute the reward loss, then we backward to update the parameters of the text encoder (and optionally UNet) by maximizing the reward.

text-image alignment-based reward models R(xˆ,p), i.e., HPSV2 [55] and PickScore [24]. Consequently, the loss function for maximizing the reward scores can be defined as follows,

L(φ) = −R(xˆ, ·/p)

zt − σtϵθ(t, zt, Tφ(p)) αt

= −R(

, ·/p).

(5)

Note that when optimizing Eqn. 5, the weights for all reward models and the UNet model are fixed, while only the weights in the CLIP text encoder are modified.

Discussion. Clearly, directly fine-tuning shares a similar training regime with regular training of diffusion models, where we are ready to employ text-image paired data (x,p) and predict reward by converting predicted noise into the predicted real data xˆ. However, considering the very beginning (noisy) timesteps, the estimated xˆ can be inaccurate and less reliable, making the predicted reward less meaningful. Instead of utilizing xˆ, Liu et al. [27] propose to finetune the reward models to enable a noisy latent (zt) aware score prediction, which is out of the scope of this work. For the best flexibility and sustainability of our method, we only investigate publicly available reward models, thus we directly employ xˆ prediction. We discuss the performance of direct finetuning in Section. 4.

##### 3.2.2 Prompt-Based Fine-tuning

As an alternative way to overcome the problem of the inaccurate xˆ prediction, given a specific text prompt p and an initial noise zT, we can iteratively solve the denoising process in Eqn. 2 to get xˆ = z0, which can then be substituted to Eqn. 5 to compute the reward scores. Consequently, we are able to precisely predict xˆ, and also eliminate the need for paired text-image data and perform the reward fine-tuning with only prompts and a pre-defined denoising schedule, i.e., 25-steps DDIM in our experiments. Since each timestep in the training process is differentiable, the gradient to update φ in T can be calculated through

Algorithm 1 Prompt-Based Reward Finetuning

Require: Pretrained UNet ϵˆθ; pretrained text encoder Tφ; prompt

set: P{p}.

Ensure: Tφ (optionally ϵˆθ if fine-tuning UNet) converges and

maximizes Ltotal.

→ Perform text encoder fine-tuning. Freeze UNet ϵˆθ and reward models Ri, activate Tφ. while Ltotal not converged do

Sample p from P; t = T while t > 0 do

zt−σtϵˆθ(t,zt,Tφ(p))

zt−1 ← αt′

αt + σt′ϵˆθ(t, zt, Tφ(p))

end while xˆ ← z0 Ltotal ← − i γiRi(xˆ, ·/p). Backward Ltotal and update Tφ for last K steps.

###### end while

→ Perform UNet finetuning. Freeze Tφ and reward models Ri, activate UNet ϵˆθ. Repeat the above reward training until converge.

chain rule as follows,

zt−σtϵˆθ(t,zt,Tφ(p)) αt

t′ ϵˆθ(t, zt, Tφ(p))] ∂Tφ(p)

t

∂[α

+ σ

t′

∂L ∂φ

∂R ∂xˆ

∂Tφ(p) ∂φ

= −

·

·

.

t=0

(6)

It is notable that solving Eqn. 6 is memory infeasible for early (noisy) timesteps, i.e., t = {T,T − 1,...}, as the computation graph accumulates in the backward chain. We apply gradient checkpointing [6] to trade memory with computation. Intuitively, the intermediate results are recalculated on the fly, thus the training can be viewed as solving one step at a time. Though with gradient checkpointing, we can technically train the text encoder with respect to each timestep, early steps still suffer from gradient explosion and vanishing problems in the long-lasting accumulation [8]. We provide a detailed analysis of step selection in Section. 4.2. The proposed prompt-based reward finetuning is further illustrated in Fig. 2 and Alg. 1.

#### 3.3. Loss Function

We investigate and report the results of using multiple reward functions, where the reward losses Ltotal can be weighted by γ and linearly combined as follows,

Ltotal =

i

Li = −

i

γiRi(xˆ, ·/p). (7)

Intuitively, we can arbitrarily combine different reward functions with various weights. However, as shown in Fig. 6, some reward functions are by nature limited in terms of their capability and training scale. As a result, fine-tuning with only one reward can result in catastrophic forgetting and mode collapse.

To address this issue, recent works [3, 57] mostly rely on careful tuning, including focusing on a specific subdomain, e.g., human and animals [35], and early stopping [8]. Unfortunately, this is not a valid approach in the generic and large-scale scope. In this work, we aim at enhancing generic models and eliminating human expertise and surveillance.

To achieve this, we set CLIP space similarity as an always-online constraint as follows,

RCLIP = cosine-sim(I(xˆ), Tφ(p)), (8)

and ensure γCLIP > 0 in Eqn. 7. Specifically, we maximize the cosine similarity between the textual embeddings and image embeddings. The textual embedding is obtained in forward propagation, while the image embedding is calculated by sending the predicted image xˆ to the image encoder I of CLIP. The original text encoder Tφ is pre-trained in large-scale contrastive learning paired with the image encoder I [37]. As a result, the CLIP constraint preserves the coherence of the fine-tuned text embedding and the original image domain, ensuring capability and generalization.

#### 3.4. UNet Fine-tuning with Reward Propagation

The proposed fine-tuning approach for text encoder is orthogonal to UNet reward fine-tuning [8, 35], meaning that the text-encoder and UNet can be optimized under similar learning objectives to further improve performance. Note that our fine-tuned text encoder can seamlessly fit the pretrained UNet in Stable Diffusion, and can be used for other downstream tasks besides text-to-image generation. To preserve this characteristic and avoid domain shifting, we finetune the UNet by freezing the finetuned text encoder Tφ. The learning objective for UNet is similar as Eqn. 6, where we optimize parameters θ of ϵˆθ, instead of φ.

### 4. Experiments

Reward Functions. We use image-based aesthetic predictor [1], text-image alignment-based CLIP predictors, (i.e., Human Preference Score v2 (HPSv2) [55] and

PickScore [24]), and CLIP model [37]. We adopt the improved (v2) version of the aesthetic predictor that is trained on 176,000 image-rating pairs. The predictor estimates a quality score ranging from 1 to 10, where larger scores indicate higher quality. HPSv2 is a preference prediction model trained on a large-scale well-annotated dataset of human choices, with 798K preference annotations and 420K images. Similarly, PickScore [24] is a popular human preference predictor trained with over half a million samples.

Training Datasets. We perform training on OpenPrompt1 dataset, which includes more than 10M high quality prompts for text-to-image generation. For direct finetuning, we use the public LAION-2B dataset with conventional pre-processing, i.e., filter out NSFW data, resize and crop images to 5122px, and use Aesthetics> 5.0 images.

Experimental Settings. We conduct experiments with the latest PyTorch [33] and HuggingFace Diffusers2. We choose Stable Diffusion v1.5 (SDv1.5) [41] as the baseline model, as it performs well in real-world human assessments with appealing model size and computation than other large diffusion models [34]. We fine-tune the ViT-L text encoder of SDv1.5, which takes 77 tokens as input and outputs an embedding with dimension 768. The fine-tuning is done on 8 NVIDIA A100 nodes with 8 GPUs per node, using AdamW optimizer [29] and a learning rate of 10−6. We set CFG scale to 7.5 in all the experiments.

Comparison Approaches. We compare our method with the following approaches.

- • Pre-trained text-to-image models that include SDv1.5, SDv2.0, SDXL Base 0.9, and DeepFloyd-XL.
- • Direct fine-tuning that is described in Sec. 3.2.1.
- • Reinforcement learning-based approach that optimize the diffusion model using reward functions [3].
- • Prompt engineering. From the scope of the enhancement of text information, prompt engineering [13, 54] can be considered as a counterpart of our approach. By extending and enriching the input prompt with more detailed instructions, e.g., using words like 4K, photorealistic, ultra sharp, etc., the output image quality could be greatly boosted. However, prompt engineering requires case-bycase human tuning, which is not appealing in real-world applications. Automatic engineering method3 employs text generation models to enhance the prompt, while the semantic coherence might not be guaranteed. We experiment and compare with automatic prompt engineering on both quantitative and qualitative evaluations.

Quantitative Results. We report the results with different training settings (the CLIP constraint is utilized under all the settings of our approach) on two datasets:

• We report zero-shot evaluation results for the score of

- 1https://github.com/krea-ai/open-prompts
- 2https://github.com/huggingface/diffusers
- 3https://huggingface.co/daspartho/prompt-extend

[Figure 35]

[Figure 36]

A tiger wearing a train conductor's hat and holding a skateboard decorated with a yin-yang symbol.

A punk rock squirrel in a studded leather jacket shouting into a microphone while standing on a stump and holding a beer on dark stage.

[Figure 37]

[Figure 38]

A cyan silver the hedgehog with black tipped quills wearing green-tinted sunglasses, a purple and green cape, and shoes.

A solitary figure shrouded in mists peers up from the cobble stone street at the imposing and dark gothic buildings surrounding it. an old-fashioned lamp shines nearby. oil painting.

[Figure 39]

[Figure 40]

A VTuber model concept art of a beautiful girl in a black and yellow hoodie looking on a smartphone in her hand, with blue eyes, long hair, and a futuristic city background.

the Eiffel Tower in winter

[Figure 41]

[Figure 42]

A girl with white hair and a school uniform, depicted in an illustration with warm clothes and a cold background.

an old-fashioned cocktail

[Figure 43]

[Figure 44]

Colorful scifi shanty town with metal rooftops and wooden and concrete walls in the style of Studio Ghibli and other anime influences.

Downtown Seattle at sunrise. detailed ink wash.

[Figure 45]

[Figure 46]

A horse and an astronaut appear in the same image.

A submarine

[Figure 47]

[Figure 48]

Soldier with plasma rifle walking through a portal to

the International Space Station

another dimension, art by Emmanuel Shiu.

- Figure 3. Qualitative visualizations. Left: generated images on Parti-Prompts, in the order of SDv1.5, prompt engineering, DDPO, and TextCraftor. Right: examples from HPSv2, ordered as SDv1.5, prompt engineering, and TextCraftor.

three rewards on Parti-Prompts [59], which contains 1632 prompts with various categories, in Tab. 1. We show experiments using a single reward function, e.g., Aesthetics, and the combination of all reward functions, i.e., denoted as All. We also fine-tune the UNet by freezing the finetuned text-encoder (TextCraftor + UNet in Tab. 1). We evaluate different methods by forwarding the generated images (or the given prompt) to reward functions to obtain scores, where higher scores indicate better performance.

• We report zero-shot results on the HPSv2 benchmark set, which contains 4 subdomains of animation, concept art, painting, and photo, with 800 prompts per category. In addition to the zero-shot model trained with combined rewards (denote as All in Tab. 2), we train the model solely with HPSv2 reward to report the best possible scores TextCraftor can achieve on the HPSv2 benchmark.

From the results, we can draw the following observations:

- • Compared to the pre-trained text-to-image models, i.e., SDv1.5 and SDv2.0, our TextCraftor achieves significantly higher scores, i.e., Aesthetics, PickScore, and HPSv2, compared to the baseline SDv1.5. More interestingly, TextCraftor outperforms SDXL Base 0.9. and DeepFloyd-XL, which have much larger UNet and text encoder.
- • Direct fine-tuning (described in Sec. 3.2.1) can not provide reliable performance improvements.
- • Compared to prompt engineering, TextCraftor obtains better performance, without necessitating human effort and ambiguity. We notice that the incurred additional information in the text prompt leads to lower alignment scores.
- • Compared to previous state-of-the-art DDPO [3] that performs reward fine-tuning on UNet, we show that TextCraftor + UNet obtains better metrics by a large margin. It is notable that DDPO is fine-tuned on subdomains, e.g., animals and humans, with early stopping, limiting its capability to generalize for unseen prompts. The proposed TextCraftor is currently the first large-scale and generic reward-finetuned model.
- • Lastly, fine-tuning the UNet can further improve the performance, proving that TextCraftor is orthogonal to UNet fine-tuning and can be combined to achieve significantly better performance.

Qualitative Results. We demonstrate the generative quality of TextCraftor in Fig. 1 and 3. Images are generated with the same noise seed for direct and fair comparisons. We show that with TextCraftor, the generation quality is greatly boosted compared to SDv1.5. Additionally, compared to prompt engineering, TextCraftor exhibits more reliable text-image alignment and rarely generates additional or irrelevant objects. Compared to DDPO [3], the proposed TextCraftor resolves the problem of mode collapse and catastrophic forgetting by employing text-image sim-

- Table 1. Comparison results on Parti-Prompts [59]. We perform TextCraftor fine-tuning on individual reward functions, including Aesthetics, PisckScore, and HPSv2, and the combination of all rewards to form a more generic model.

Parti-1632 Reward Aesthetics PickScore HPSv2 SDXL Base 0.9 - 5.7144 20.466 0.2783 SDv2.0 - 5.1675 18.893 0.2723 SDv1.5 - 5.2634 18.834 0.2703 DDPO [3] Aesthetic 5.1424 18.790 0.2641 DDPO [3] Alignment 5.2620 18.707 0.2676 Prompt Engineering - 5.7062 17.311 0.2599 Direct Fine-tune (Sec. 3.2.1) All 5.2880 18.750 0.2701 TextCraftor Aesthetics 5.5212 18.956 0.2670 TextCraftor PickScore 5.2662 19.023 0.2641 TextCraftor HPSv2 5.4506 18.922 0.2800 TextCraftor (Text) All 5.8800 19.157 0.2805 TextCraftor (UNet) All 6.0062 19.281 0.2867 TextCraftor (Text+UNet) All 6.4166 19.479 0.2900

- Table 2. Comparison results on HPSv2 benchmark [55]. In addition to the generic model, we report TextCraftor fine-tuned solely on HPSv2 reward, denoted as TextCraftor (HPSv2).

HPS-v2 Animation Concept Art Painting Photo Average DeepFloyd-XL 0.2764 0.2683 0.2686 0.2775 0.2727 SDXL Base 0.9 0.2842 0.2763 0.2760 0.2729 0.2773 SDv2.0 0.2748 0.2689 0.2686 0.2746 0.2717 SDv1.5 0.2721 0.2653 0.2653 0.2723 0.2688 TextCraftor (HPSv2) 0.2938 0.2919 0.2930 0.2851 0.2910 TextCraftor + UNet (HPSv2) 0.3026 0.2995 0.3005 0.2907 0.2983 TextCraftor (All) 0.2829 0.2800 0.2797 0.2801 0.2807 TextCraftor + UNet (All) 0.2885 0.2845 0.2851 0.2807 0.2847

- Table 3. Human evaluation on 1632 Parti-Prompts [59]. Human annotators are given two images generated from different approaches and asked to choose the one that has better image quality and text-image alignment. Our approach obtains better human preference over all compared methods.

Comparison Methods SDv1.5 SDv2.0 SDXL Base 0.9 Prompt Eng. DDPO Align. DDPO Aes. Our Win Rate 71.7% 81.7% 59.7% 81.3% 56.7% 66.2%

ilarity as the constraint reward. We also show that finetuning the UNet models upon the TextCraftor enhanced text encoder can further boost the generation quality in the figure in the Appendix. From the visualizations, we observe that the reward fine-tuned models tend to generate more artistic, sharp, and colorful styles, which results from the preference of the reward models. When stronger and better reward predictors emerge in the future, TextCraftor can be seamlessly extended to obtain even better performance. Lastly, we provide a comprehensive human evaluation in Tab. 3, proving the users prefer the images synthesized by TextCraftor.

- 4.1. Controllable Generation

Instead of adjusting reward weights γi in Eqn. 7, we can alternatively train dedicated text encoders optimized for each reward, and mix-and-match them in the inference phase for flexible and controllable generation.

Interpolation. We demonstrate that, besides quality en-

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

a cream colored labradoodle next to a white cat with black-tipped ears

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

a portrait of a statue of a pharaoh wearing steampunk glasses, white t-shirt and leather jacket. dslr photograph

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

a wooden toy horse with a mane made of rope

weight = 0.0 weight = 0.25 weight = 0.5 weight = 0.75 weight = 1.0

- Figure 4. Interpolation between original text embedding (weight 0.0) and the one from TextCraftor (weight 1.0) , demonstrating controllable generation. From top to bottom row: TextCraftor using HPSv2, PickScore, and Aesthetics as reward models.

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

[1, 0, 0, 0] [0.1, 0.3, 0.3, 0.3] [0.1, 0.7, 0.1, 0.1] [0.1, 0.1, 0.7, 0.1] [0.1, 0.1, 0.1, 0.7]

Three-quarters front view of a yellow 2017 Corvette coming around a curve in a mountain road and looking over a green valley on a cloudy day.

Four deer surrounding a moose.

Siberian husky playing the piano.

- Figure 5. Style mixing. Text encoders fine-tuned from different reward models can collaborate and serve as style mixing. The weights listed at the bottom are used for combining text embedding from {origin, Aesthetics, PickScore, HPSv2}, respectively.

hancements, TextCraftor can be weighted and interpolated with original text embeddings to control the generative strength. As in Fig. 4, with the increasing weights of enhanced text embeddings, the generated image gradually transforms into the reward-enhanced style. Style Mixing. We also show that different reward-finetuned models can collaborate together to form style mixing, as in Fig. 5.

#### 4.2. Ablation Analysis

Rewards and CLIP Constraint. We observe that simply relying on some reward functions might cause mode collapse problems. As in Fig. 6, training solely on Aesthetics score or PickScore obtains exceptional rewards, but the model loses its generality and tends to generate a specific

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

a photo of an astronaut riding a horse on mars

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

A small cabin on top of a snowy mountain in the style of Disney, artstation

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

portrait of Emma Watson as Hermione Granger sitting next to a window reading a book

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

decorated modern country house interior

Aesthetics=9.35 Aesthetics=5.82 PickScore=23.42 PickScore=18.85 HPSv2=0.3176 HPSv2=0.2855

Figure 6. Ablation on reward models and the effect of CLIP constraint. The leftmost column shows original images. Their averaged Aesthetics, PickScore, and HPSv2 scores are 5.49, 18.19, and 0.2672, respectively. For the following columns, we show the synthesized images without and with CLIP constraint using different reward models. The reward scores are listed at the bottom.

image that the reward model prefers. To conclude the root cause, not all reward models are pre-trained with large-scale fine-labeled data, thus lacking the capability to justify various prompts and scenarios. We see that HPSv2 shows better generality. Nevertheless, the CLIP constraint prevents the model from collapsing in all three reward regimes, while with reliable improvements in the corresponding scores.

Training and Testing Steps. As discussed in Section 3.2, the reward fine-tuning introduces a long chain for gradient propagation. With the danger of gradient explosion and vanishing, it is not necessarily optimal to fine-tune over all timesteps. In Tab. 4, we perform the analysis on the training and evaluation steps for TextCraftor. We find that training with 5 gradient steps and evaluating the fine-tuned text encoder with 15 out of the total 25 steps gives the most balanced performance. In all of our experiments and reported results, we employ this configuration.

Table 4. Analysis of training and evaluation steps for fine-tuned text encoder. We report results on Parti-Prompts [59].

Train Test Aes PickScore HPSv2

SDv1.5 25 5.2634 18.834 0.2703 5 5 6.0688 19.195 0.2835 5 10 6.3871 19.336 0.2847 5 15 6.5295 19.355 0.2828 5 25 6.5758 19.071 0.2722

10 10 5.8680 19.158 0.2799 15 15 5.3533 18.919 0.2735

We include more ablation studies on denoising scheduler and steps and reward weights in the supplementary material.

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

a snake curled around a wooden post

an ornate gold harp a satellite image...mountain to the north...cloud covering...

a laptop with a maze sticker on it

a stop sign with 'ALL WAY' written below it

- Figure 7. Applying the fine-tuned SDv1.5 text encoder (ViT-L) under TextCraftor to SDXL can improve the generation quality, e.g., better text-image alignment. For each pair of images, the left one is generated using SDXL and the right one is from SDXL+TextCraftor.

4.3. Discussion on Training Cost and Data

TextCraftor is trained on 64 NVIDIA A100 80G GPUs, with batch size 4 per GPU. We report all empirical results of TextCraftor by training 10K iterations, and the UNet fine-tuning (TextCraftor+UNet) with another 10K iterations. Consequently, TextCraftor observes 2.56 million data samples. TextCraftor overcomes the collapsing issue thus eliminating the need for tricks like early stopping. The estimated GPU hour for TextCraftor is about 2300. Finetuning larger diffusion models can lead to increased training costs. However, TextCraftor has a strong generalization capability. As in Fig 7, the fine-tuned SDv1.5 text encoder (ViT-L) can be directly applied to SDXL [34] to generate better results (for each pair, left: SDXL, right: SDXL + TextCraftor-ViT-L). Note that SDXL employs two text encoders and we only replace the ViT-L one. Therefore, to reduce the training cost on larger diffusion models, one interesting future direction is to fine-tune their text encoder within a smaller diffusion pipeline, and then do inference directly with the larger model.

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

Pose2image. Prompt: An old man cooking in kitchen, vintage.

Scrible2image. Prompt: Backpack for iron man.

- Figure 8. Applying the fine-tuned SDv1.5 text encoder (ViT-L) under TextCraftor to ControlNet improves the generation quality. From left to right: input condition, SDv1.5, TextCraftor + SDv1.5 UNet.

[Figure 118]

Initial Image Mask SDv1.5 TextCraftor Prompt: concept art digital painting of an elven castle, inspired by lord of the rings

Figure 9. Applying TextCraftor on inpainting task can improve the generation quality. The prompt in the example is concept art digital painting of an elven castle, inspired by lord of the rings.

#### 4.4. Applications

We apply TextCraftor on ControlNet [60] and image inpainting for zero-shot evaluation (i.e., the pre-trained text encoder from TextCraftor is directly applied on these tasks), as in Fig. 8 and Fig. 9. We can see that TextCraftor can readily generalize to downstream tasks (with the same pretrained baseline model, i.e., SDv1.5), and achieves better generative quality.

### 5. Conclusion

In this work, we propose TextCraftor, a stable and powerful framework to fine-tune the pre-trained text encoder to improve the text-to-image generation. With only prompt dataset and pre-defined reward functions, TextCraftor can significantly enhance the generative quality compared to the pre-trained text-to-image models, reinforcement learningbased approach, and prompt engineering. To stabilize the reward fine-tuning process and avoid mode collapse, we introduce a novel similarity-constrained paradigm. We demonstrate the superior advantages of TextCraftor in different datasets, automatic metrics, and human evaluation. Moreover, we can fine-tune the UNet model in our reward pipeline to further improve synthesized images. Given the superiority of our approach, an interesting future direction is to explore encoding the style from reward functions into specific tokens of the text encoder.

### References

- [1] Clip+mlp aesthetic score predictor. https : / / github.com/christophschuhmann/improvedaesthetic- predictor#clipmlp- aestheticscore-predictor, 2022. 2, 3, 5
- [2] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, et al. ediffi: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. 2
- [3] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023. 2, 5, 7
- [4] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704, 2023. 1
- [5] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis,

2023. 1, 2

- [6] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174, 2016. 4
- [7] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416, 2022. 1, 2
- [8] Kevin Clark, Paul Vicol, Kevin Swersky, and David J Fleet. Directly fine-tuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400, 2023. 2, 4, 5
- [9] Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon Vandenhende, Xiaofang Wang, Abhimanyu Dubey, et al. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807, 2023. 2
- [10] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems, 34:8780–8794, 2021. 1
- [11] Hanze Dong, Wei Xiong, Deepanshu Goyal, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. Raft: Reward ranked finetuning for generative foundation model alignment. arXiv preprint arXiv:2304.06767, 2023. 2
- [12] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. arXiv preprint arXiv:2305.16381, 2023. 2
- [13] Jindong Gu, Zhen Han, Shuo Chen, Ahmad Beirami, Bailan He, Gengyuan Zhang, Ruotong Liao, Yao Qin, Volker Tresp, and Philip Torr. A systematic survey of prompt engineering on vision-language foundation models. arXiv preprint arXiv:2307.12980, 2023. 1, 2, 5

- [14] Yaru Hao, Zewen Chi, Li Dong, and Furu Wei. Optimizing prompts for text-to-image generation. arXiv preprint arXiv:2212.09611, 2022. 3
- [15] Yingqing He, Shaoshu Yang, Haoxin Chen, Xiaodong Cun, Menghan Xia, Yong Zhang, Xintao Wang, Ran He, Qifeng Chen, and Ying Shan. Scalecrafter: Tuning-free higherresolution visual generation with diffusion models. arXiv preprint arXiv:2310.07702, 2023. 2
- [16] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 13
- [17] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 3
- [18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020. 1, 2, 3
- [19] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 1
- [20] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 2
- [21] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10124–10134, 2023. 2
- [22] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. arXiv preprint arXiv:2206.00364, 2022. 2
- [23] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 3
- [24] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. arXiv preprint arXiv:2305.01569, 2023. 2, 4, 5
- [25] Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu. Aligning textto-image models using human feedback. arXiv preprint arXiv:2302.12192, 2023. 2
- [26] Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey Tulyakov, and Jian Ren. Snapfusion: Text-to-image diffusion model on mobile devices within two seconds. arXiv preprint arXiv:2306.00980, 2023. 1
- [27] Xihui Liu, Dong Huk Park, Samaneh Azadi, Gong Zhang, Arman Chopikyan, Yuxiao Hu, Humphrey Shi, Anna Rohrbach, and Trevor Darrell. More control for free! image synthesis with semantic diffusion guidance. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 289–299, 2023. 4
- [28] Xian Liu, Jian Ren, Aliaksandr Siarohin, Ivan Skorokhodov, Yanyu Li, Dahua Lin, Xihui Liu, Ziwei Liu, and Sergey

- Tulyakov. Hyperhuman: Hyper-realistic human generation with latent structural diffusion. arXiv preprint arXiv:2310.08579, 2023. 1
- [29] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 5
- [30] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11461–11471, 2022. 1
- [31] Jacob Menick and Nal Kalchbrenner. Generating high fidelity images with subscale pixel networks and multidimensional upscaling. arXiv preprint arXiv:1812.01608, 2018. 2
- [32] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 1
- [33] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems 32, pages 8024–8035. Curran Associates, Inc., 2019. 5
- [34] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2, 5, 9
- [35] Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, and Katerina Fragkiadaki. Aligning text-to-image diffusion models with reward backpropagation. arXiv preprint arXiv:2310.03739, 2023. 2, 5
- [36] Reid Pryzant, Dan Iter, Jerry Li, Yin Tat Lee, Chenguang Zhu, and Michael Zeng. Automatic prompt optimization with” gradient descent” and beam search. arXiv preprint arXiv:2305.03495, 2023. 3
- [37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 1, 2, 3, 5
- [38] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 8821–8831. PMLR, 2021. 2
- [39] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125,

2022. 2

- [40] Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference

- in deep generative models. In International conference on machine learning, pages 1278–1286. PMLR, 2014. 3
- [41] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 1, 2, 3, 5
- [42] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In MICAI, 2015. 3
- [43] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In ACM SIGGRAPH 2022 Conference Proceedings, pages 1– 10, 2022. 1
- [44] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 1, 2
- [45] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image superresolution via iterative refinement. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022. 1
- [46] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. arXiv preprint arXiv:2210.08402, 2022. 3
- [47] Chenyang Si, Ziqi Huang, Yuming Jiang, and Ziwei Liu. Freeu: Free lunch in diffusion u-net. arXiv preprint arXiv:2309.11497, 2023. 2
- [48] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

2022. 1

- [49] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pages 2256–2265. PMLR, 2015. 2, 3
- [50] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 3
- [51] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019. 2
- [52] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 1, 2, 3
- [53] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al.

- Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 3
- [54] Sam Witteveen and Martin Andrews. Investigating prompt engineering in diffusion models. arXiv preprint arXiv:2211.15462, 2022. 1, 2, 5
- [55] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341,

2023. 2, 4, 5, 7

- [56] Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score: Better aligning textto-image models with human preference. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2096–2105, 2023. 2
- [57] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for textto-image generation, 2023. 2, 5
- [58] Zeyue Xue, Guanglu Song, Qiushan Guo, Boxiao Liu, Zhuofan Zong, Yu Liu, and Ping Luo. Raphael: Text-to-image generation via large mixture of diffusion paths. arXiv preprint arXiv:2305.18295, 2023. 1, 2
- [59] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2022. 2, 7, 8
- [60] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 9
- [61] Zhixing Zhang, Ligong Han, Arnab Ghosh, Dimitris N Metaxas, and Jian Ren. Sine: Single image editing with textto-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6027–6037, 2023. 1
- [62] Shanshan Zhong, Zhongzhan Huang, Weushao Wen, Jinghui Qin, and Liang Lin. Sur-adapter: Enhancing text-to-image pre-trained diffusion models with large language models. In Proceedings of the 31st ACM International Conference on Multimedia, pages 567–578, 2023. 3

### A. Ablation on Scheduler and Denoising Steps

The main paper uses a 25-step DDIM scheduler as the default configuration. We provide an additional study on the choice of scheduler type and denoising steps in Tab. 5. We find that the widely adopted DDIM scheduler already yields satisfactory performance, which is comparable to or even better than second-order counterparts such as DPM. We also find that 25 denoising steps are good enough for generative quality, while increasing the inference steps to 50 has minimal impact on performance.

Table 5. Ablation study on denoising scheduler and steps. We choose TextCraftor on all rewards as the baseline model.

Scheduler Steps Aesthetic PickScore HPSv2 DDIM 25 5.8800 19.157 0.2805 DDIM 50 6.0178 19.121 0.2769 PNDM 25 5.0991 18.479 0.2632 PNDM 50 5.9355 19.026 0.2748

DPM 25 5.8564 19.145 0.2803 Euler 25 5.9098 19.151 0.2804

### B. Weight of Reward Functions

With TextCraftor, it is free to choose different reward functions and different weights as the optimization objective. For simplicity, in the main paper, we scale all the rewards to the same magnitude. We report empirical results by setting the weight of CLIP constraint to 100, Aesthetic reward as 1, PickScore as 1, and HPSv2 as 100. In Tab. 6, we provide an additional ablation study on different reward weights. Specifically, we train TextCraftor with emphasis on CLIP regularization, Aesthetic score, PickScore, and HPSv2 respectively. We can observe that assigning a higher weight to a specific reward simply results in better scores. TextCraftor is flexible and readily applicable to different user scenarios and preferences. We observe the issue of repeated objects in Fig. 12, which is introduced along with UNet fine-tuning. Thus, we ablate TextCraftor+UNet fine-tuning with differ-

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

Figure 10. Adjusting reward weights can further reduce artifacts (repeated objects.)

ent weights of rewards. We find that HPSv2 is the major source of repeated objects. We show in Fig. 10 that we can remove the repeated sloth and chimpanzee by using a smaller weight of HPSv2 reward.

### C. Interpretability

We further demonstrate the enhanced semantic understanding ability of TextCraftor in Fig. 11. Similar to Prompt to Prompt [16], we visualize the cross-attention heatmap which determines the spatial layout and geometry of the generated image. We discuss two failure cases of the baseline model in Fig. 11. The first is misaligned semantics, as the purple hat of the corgi. We can see that the hat in pixel space correctly attends to the word purple, but in fact, the color is wrong (red). Prompt engineering does not resolve this issue. While in TextCraftor, color purple is correctly reflected in the image. The second failure case is missing elements. SDv1.5 sometimes fails to generate desired objects, i.e., Eiffel Tower or desert, where there is hardly any attention energy upon the corresponding words. Prompt engineering introduces many irrelevant features and styles, but can not address the issue of the missing desert. While with TextCraftor, both Eiffel Tower and desert are correctly understood and reflected in the image. We show that (i) Finetuning the text encoder improves its capability and has the potential to correct some inaccurate semantic understandings. (ii) Finetuning text encoder helps to emphasize the core object, reducing the possibility of missing core elements in the generated image, thus improving text-image alignment, as well as benchmark scores.

### D. More Qualitative Results

We provide more qualitative visualizations in Fig. 12 to demonstrate the performance and generalization of TextCraftor.

Table 6. Ablation study on different reward weights. The reported results are TextCraftor for text encoder only.

|Weight<br><br>|Score|
|---|---|
|CLIP Aesthetic PickScore HPSv2|CLIP Aesthetic PickScore HPSv2<br><br>|
|200 3 1 100 100 6 1 100 100 3 2 100 100 3 1 200<br><br>|0.2952 6.0900 19.123 0.2757 0.2385 7.1680 19.435 0.2730 0.2615 6.6831 19.494 0.2798 0.2711 6.4020 19.306 0.2850|

[Figure 123]

[Figure 124]

SDv1.5: a corgi wearing a red bowtie and a purple party hat.

[Figure 125]

[Figure 126]

Prompt Engineering: a corgi wearing a red bowtie and a purple party hat , Graceful body structure,cute,Symmetrical face,highly

detailed,elegant,Marc Simonetti and Caspar David Friedrich, Trending on artstation,depicted as a scifi scene,highly detailed matte painting.

[Figure 127]

[Figure 128]

TextCraftor: a corgi wearing a red bowtie and a purple party hat.

[Figure 129]

[Figure 130]

SDv1.5: the Eiffel Tower in a desert

[Figure 131]

[Figure 132]

Prompt Engineering: the Eiffel Tower in a desert, anime fantasy illustration by tomoyuki yamasaki, kyoto studio, madhouse, ufotable,

comixwave films, trending on artstation pixiv, background car up in road, low angle view, epic sky, cinematic lighting, sun shaft, clouds

[Figure 133]

[Figure 134]

TextCraftor: the Eiffel Tower in a desert

Figure 11. Illustration of the enhanced semantic understanding in TextCraftor, visualized by the cross-attention heatmap.

[Figure 135]

[Figure 136]

a chimpanzee wearing a bowtie and playing a piano A white-haired girl in a pink sweater looks out a window in her bedroom.

[Figure 137]

[Figure 138]

A tiger wearing a train conductor's hat and holding a skateboard decorated with a yin-yang symbol.

A dignified beaver wearing glasses, a vest, and colorful neck tie. He stands

next to a tall stack of books in a library.

[Figure 139]

[Figure 140]

A product still of metallic black and white Nike shoes with a red glowing swoosh, styled after Darth Vader.

A smiling sloth wearing a bowtie and holding a quarterstaff and a big book. A shiny VW van parked on grass.

[Figure 141]

[Figure 142]

A fox wearing a yellow dress.

A map of the United States made out of sushi on the table.

[Figure 143]

[Figure 144]

A toast with black sunglasses and a blue flower on the top right corner.

an old-fashioned cocktail

[Figure 145]

[Figure 146]

a photograph of the mona lisa drinking coffee as she has her breakfast. her plate has an omelette and croissant.

there is a chocolate cake and ice cream on a plate

Figure 12. Additional visualizations. Left: generated images on Parti-Prompts, in the order of SDv1.5, prompt engineering, DDPO, TextCraftor, and TextCraftor + UNet. Right: examples from HPSv2, ordered as SDv1.5, prompt engineering, TextCraftor, and TextCraftor + UNet.

