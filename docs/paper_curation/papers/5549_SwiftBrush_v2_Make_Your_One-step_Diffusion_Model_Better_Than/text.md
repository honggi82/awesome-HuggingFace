# arXiv:2408.14176v2[cs.CV]27Aug2024

## SwiftBrush v2: Make Your One-step Diffusion Model Better Than Its Teacher

Trung Dao1 Thuan Hoang Nguyen1,∗ Thanh Le1,∗ Duc Vu1,∗ Khoi Nguyen1 Cuong Pham1,2 Anh Tran1

1VinAI Research 2Posts & Telecommunications Inst. of Tech.

Abstract. In this paper, we aim to enhance the performance of SwiftBrush, a prominent one-step text-to-image diffusion model, to be competitive with its multi-step Stable Diffusion counterpart. Initially, we explore the quality-diversity trade-off between SwiftBrush and SD Turbo: the former excels in image diversity, while the latter excels in image quality. This observation motivates our proposed modifications in the training methodology, including better weight initialization and efficient LoRA training. Moreover, our introduction of a novel clamped CLIP loss enhances image-text alignment and results in improved image quality. Remarkably, by combining the weights of models trained with efficient LoRA and full training, we achieve a new state-of-the-art one-step diffusion model, achieving an FID of 8.14 and surpassing all GAN-based and multi-step Stable Diffusion models. The project page is available at: https://swiftbrushv2.github.io/

Keywords: One-step Diffusion models · Text-to-image synthesis

### 1 Introduction

Text-to-image generation has experienced tremendous growth in recent years, allowing users to create high-quality images from simple descriptions. State-ofthe-art models [3, 33, 40, 42] could surpass humans in art competition [43] or produce synthetic images nearly indistinguishable from real ones [7]. Among popular text-to-image networks, Stable Diffusion (SD) models [41, 42] are the most widely used due to their open-source accessibility. However, most SD models are designed as multi-step diffusion models, which require multiple forwarding steps to produce an output image. Such a slow and computationally expensive mechanism hinders the use of these models in real-time or on-device applications.

Recently, many works have tried to reduce the denoising steps required in text-to-image diffusion models. Notably, few recent studies have successfully developed one-step diffusion models, thus significantly speed up the image generation. While early attempts [13,32] produce blurry and malformed photos, subsequent methods produce sharp and high-quality outputs. These methods mainly distill knowledge from a pre-trained multi-step SD model (referred to as

* Equal Contribution.

SD Turbo SDv2.1 SwiftBrush Ours

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

[Figure 31]

[Figure 32]

Quality Diversity Speed

Quality Diversity Speed Quality Diversity Speed Quality Diversity Speed

- Fig. 1: Our one-step diffusion model achieves an impressive FID of 8.14, generating high-quality and diverse results with a single UNet forwarding. The example images generated from the “A laughing cute grey rabbit with white stripe on the head, piles of gold coins in background, colorful, Disney Picture render, photorealistic” (first two rows) and “Portrait of a woman looking at the camera” (last two rows) prompts demonstrate our model’s ability to create fast, visually appealing, and varied outputs.

the teacher model) to a one-step student model. InstaFlow [30] employs Rectified Flows [29] in a multi-stage and computation-expensive training procedure. DMD [54] combines a reconstruction and a distribution matching loss as the training objectives, requiring massive pre-generated images from the teacher. SD Turbo [46] incorporates adversarial training alongside a score distillation loss, achieving photorealistic generation. However, it heavily relies on a large-scale image-text pair training dataset and, as later discussed, has a poor diversity. SwiftBrush [34] utilizes Variational Score Distillation (VSD) to transfer knowledge from the teacher network to the one-step student through a LoRA [17] intermediate teacher model. Notably, training SwiftBrush is simple, fast, and image-free, making it an intriguing method.

Despite these promising achievements, one-step text-to-image diffusion models still fall short of multi-step models in terms of the FID metric. On the standard COCO 2014 benchmark [28], SDv2.1 can achieve the lowest FID-30K score of 9.64 with classifier-free guidance (cfg) scale of 2, while the best-reported score from the one-step models of equivalent parameters scale is 11.49 [54]. The gap is expected since directly predicting a clean image from noise in a single step is much more challenging than via a multi-step scheme. Hence, one may believe that one-step text-to-image models could only approach or reach a similar performance as the teacher model but never exceeding it.

In this paper, we challenge this belief by seeking a one-step model that can surpass its multi-step teacher model quantitatively and qualitatively. Our so-

lution drew inspiration from SwiftBrush, with its image-free training, enabling an effective, scalable, and flexible distillation. We examine its current state and compare it with SD Turbo. The comparison in Sec. 4.1 highlights a qualitydiversity trade-off: SwiftBrush offers more diverse outputs due to its image-free and dynamic training, while SD Turbo yields high-quality but mode-collapselike outputs due to its adversarial training. This insight drives us to initialize SwiftBrush training with SD Turbo, enhancing one-step student models significantly. Moreover, our extra clamped CLIP loss combined with SwiftBrush’s flexible training mechanism empowers the student model to surpass the teacher. Lastly, we train the student on a larger text prompt dataset for better knowledge transfer between teacher and student models.

Given limited resources and the goal of offering effective and affordable model training solutions, we restrict the training on A100 40GB GPUs with affordable GPU hours. Such a restricted condition prevents us from employing efficiently the clamped CLIP loss and fully finetuning the student model. Hence, we propose two training schemes, one supporting full student model training without the extra loss and one employing LoRA-based model training associated with the mentioned auxiliary loss. Both training schemes produce high-quality output models that surpass all previous one-step diffusion-based approaches in most metrics. Especially when merging these two models using a simple weight linear interpolation, we obtain an one-step model with FID-30K of 8.77 on the COCO 2014 benchmark. Such a student model is the first to surpass its multi-step teacher model, breaking the common belief. It even exceeds all GAN-based textto-image approaches [22, 45] while also offering near real-time speed. With an extra regularization [54] on minimal real data, our merged model gets enhanced further to achieve the FID score of 8.14, setting a new standard for efficient and high-quality text-to-image models.

In summary, our contributions include (1) an analysis of representative existing diffusion-based text-to-image models to reveal the quality-diversity trade-off, (2) a simple but effective integration of SwiftBrush and SD Turbo to combine the advantages of both, (3) an extra clamped CLIP loss proposed to boost the image-text alignment of the student network and surpass the teacher model, (4) two resource-efficient training strategies to utilize the mentioned proposals, and (5) a fused one-step student model that is superior to its multi-step text-to-image teacher in all metrics and sets a new standard in this field.

### 2 Related Work

#### 2.1 Text-to-Image Generation

Text-to-image generation involves synthesizing high-quality images based on input text prompts. This task has evolved over decades, transitioning from constrained domains like CUB [48] and COCO [28] to general domains such

- as LAION-5B [47]. This evolution is driven by the emergence of large visionlanguage models (VLMs) like CLIP [39] and ALIGN [20]. Leveraging these

models and datasets, various approaches have been introduced, including autoregressive models like DALL-E [40], CogView [11], and Parti [55]; mask-based transformers such as MUSE [5] and MaskGIT [6]; GAN-based models such as StyleGAN-T [45], GigaGAN [22]; and diffusion models like GLIDE [35], Imagen [44], Stable Diffusion (SD) [41], DALL-E2 [40], DALL-E3 [3], and eDiff-I [2]. Among these, diffusion models are popular due to their ability to generate highquality images. However, they typically require many-step sampling to generate high-quality images, limiting their real-time and on-device applications.

#### 2.2 Accelerating Text-to-Image Diffusion Models

Efforts to accelerate diffusion model sampling include faster samplers and distillation techniques. Early methods reduce sampling steps to as few as 4-8 steps by incorporating Latent Consistency Models to distill latent diffusion models [9,32]. Recent studies have achieved one-step text-to-image generation by training a student model distilled from a pretrained multi-step diffusion model, employing various techniques such as Rectified Flows [30], reconstruction and distribution matching losses [54], and adversarial objectives [27,46,52,57]. However, the output images often exhibit blurriness and artifacts, and one-step methods still underperform compared to multi-step models while requiring large-scale textimage pairs for training.

Differentiating itself from the rest, SwiftBrush [34] proposed a one-step distillation technique that required only training on prompt inputs. The method gradually transfers knowledge from the teacher to the one-step student through an intermediate LoRA multi-step teacher. SwiftBrush’s image-free training procedure offers a simple way to scale training data and extend the student model capability via auxiliary losses, which are not constrained by limited-size imagery training data. Therefore, while SwiftBrush also falls short in quality compared to the teacher model, we find its high potential for further development to produce a one-step student that even beat the multi-step teacher at its own game.

### 3 Background

Diffusion Models are generative models that transform a noise distribution into a target data distribution by simulating the diffusion process. This transformation involves gradually adding noise ϵ ∼ N(0,I) to clean image x0 over a series of T steps (forward process) and then learning to reverse this process (reverse process). The forward process can be formulated as:

xt = αtx0 + σtϵ ∀t ∈ 0,T (1)

where xt is the data at time step t and {(αt,σt)}Tt=1 is the noise schedule such that (αT,σT) = (0,1) and (α0,σ0) = (1,0). On the other hand, the reverse process aims to reconstruct the original data from noise. Training involves minimizing the difference between predicted output from model ϵϕ parameterized by

ϕ and the actual added noise: min

Et∼U(0,T),ϵ∼N(0,I)∥ϵϕ(xt,t) − ϵ∥22 (2)

ϕ

Text-to-Image Diffusion Models generate images from textual descriptions by integrating text embeddings within the inference process, aiming to align textual descriptions with visual outputs. A key challenge is the lack of mechanisms to ensure textual relevance and image fidelity. To deal with this, large-scale textto-image diffusion models usually utilize classifier-free guidance which enhances text-image alignment without a separate classifier. This is achieved by interpolating the model’s output with and without the conditioning text, controlled by a guidance scale γ. The model’s final output with guidance is given by:

ϵˆϕ(xt,t,y) = ϵϕ(xt,t,y) + γ · (ϵϕ(xt,t,y) − ϵϕ(xt,t)), (3)

where ϵϕ(xt,t,y) is the predicted output conditioned on text embedding y and ϵϕ(xt,t) is the unconditional prediction, i.e., using null text. This formula showcases how classifier-free guidance directly influences the generation process, leading to more precise and relevant image outputs.

SwiftBrush [34] is a one-step text-to-image generative model, employing an image-free distillation technique inspired by text-to-3D synthesis methods. At the heart of this approach is re-purposing Variational Score Distillation (VSD) [49], a novel loss that tackles the challenges of over-smoothing and diversity reduction often seen in early text-to-3D work [38]. Specifically, SwiftBrush uses two teachers, one frozen teacher ϵϕ and one LoRA [17] teacher ϵψ, to guide the one-step student fθ. Here, the LoRA teacher is to bridge the gap between the frozen teacher and the student. On one hand, the loss for training the student model is formalized as:

∂fθ(z,y) ∂θ

∇θLV SD = Et,y,z w(t)(ˆϵϕ(xt,t,y) − ϵˆψ(xt,t,y))

, (4)

where z ∼ N(0,I) is the input noise to the student network, xt = αtxˆ0 + σtϵ is the noise-added version at a time step t of the student’s output xˆ0 = fθ(z,y), and w(t) is the weighting of the loss. On the other hand, the LoRA teacher is trained using diffusion loss ∥Et,ϵ,y[ϵψ(xt,t,y) − ϵ]∥22. SwiftBrush alternates between the training of the student model and the LoRA teacher until convergence.

### 4 Proposed Methods

In this section, we begin by conducting an in-depth analysis of the qualitydiversity trade-off in representative diffusion-based text-to-image models (Sec. 4.1). Subsequently, we discuss our strategy for incorporating the strengths of SwiftBrush and SD Turbo (Sec. 4.2). Lastly, we explore various approaches to enhance the distillation process and post-training procedures (Sec. 4.3 to 4.5). An overview of our methodologies is presented in Fig. 2.

- Table 1: Comparison between the multi-step teacher (SDv2.1), SD Turbo, and SwiftBrush on the zero-shot MS COCO-2014 30K benchmark. The best scores are in bold, while the better scores among one-step models are underlined.

##### Model Name NFE FID↓ CLIP↑ Precision↑ Recall↑

SD 2.1 (cfg = 2) 25 9.64 0.31 0.57 0.53 SD 2.1 (cfg = 4.5) 25 12.26 0.33 0.61 0.41 SD 2.1 (cfg = 7.5) 25 15.93 0.33 0.59 0.36

SD Turbo 1 16.10 0.33 0.65 0.35 SwiftBrush 1 15.46 0.30 0.47 0.46

#### 4.1 Quality-Diversity Trade-off in Existing Models

We first analyze the properties of the teacher model, SDv2.1, and existing onestep diffusion-based text-to-image models. For the teacher model, we assess its performance across different guidance scales. Here, we select SwiftBrush and SD Turbo due to their quality and distinct training procedures. SwiftBrush relies solely on score distillation from the teacher in its image-free training, while SD Turbo trains on real images with adversarial and distillation loss. We conduct our analysis on the COCO 2014 benchmark and report relevant metrics in Tab. 1.

When assessing the multi-step teacher’s performance, the classifier-free guidance scale (cfg) plays a crucial role. A low cfg (e.g., cfg = 2) yields a low FID score of 9.64, driven by high output diversity (recall = 0.53). However, this setting results in weak alignment between images and prompts (CLIP score = 0.30) and lower image quality (low precision). Conversely, a large cfg (e.g., cfg = 7.5) markedly improves text-image alignment (CLIP score = 0.33) but restricts diversity (recall = 0.36), resulting in a poor FID score of 15.93. Moderate cfg values (e.g., cfg = 4.5) strike a better balance, offering the highest precision score.

When evaluating the one-step students, we notice distinct behaviors. SD Turbo, benefiting from adversarial training on real images, yields highly naturalistic outputs with an exceptionally high precision score, surpassing even those of the multi-step teacher. However, this results in poor diversity, reflected in a low recall of 0.35. Conversely, SwiftBrush adopts an image-free training approach, allowing flexible combinations of random-noise latents and input prompts. Such a relaxed supervision enables the student model to generate more diverse outputs but at the expense of quality (Tab. 1). We further verify this finding by a qualitative evaluation, illustrated in Fig. 1. When given identical input prompts, SD Turbo generates realistic yet similar outputs. In contrast, SwiftBrush produces a wider range of outcomes, albeit with greatly distorted artifacts. Regardless, both one-step models exhibit FID scores around 15-16, significantly higher than the teacher’s best score. Observing a quality-diversity trade-off in existing one-step diffusion model such as SD Turbo and SwiftBrush, we aim to combine these two to leverage the strengths of both.

#### 4.2 SwiftBrush and SD Turbo Integration

In this section, we explore strategies for effectively merging SwiftBrush and SD Turbo to enhance the quality-diversity trade-off. A direct approach is unifying their training procedure, i.e., combining adversarial training from SD Turbo and Variational Score Distillation from SwiftBrush. However, this simplistic approach proves challenging due to computational demands and potential failure. While SwiftBrush’s image-free procedure is easier to implement, reproducing SD Turbo’s training process is complex and resource-intensive. The presence of the discriminator complicates the training, necessitating significant VRAM and dataset requirements. Additionally, SD Turbo’s intense supervision may constrain SwiftBrush’s loose guidance, limiting output diversity.

Based on our discussions, we opt not to utilize SD Turbo’s adversarial training. Instead, we leverage its pretrained weights to initialize the student network within SwiftBrush’s training framework. This straightforward approach proves highly effective. As can be seen in the comparison between the second and the third row in Tab. 2, the resulting model has improved FID and recall. By employing SD Turbo’s pretrained weights, we provide a solid foundation for the training model to maintain high-quality outputs, while SwiftBrush’s image-free training process gradually enhances generation diversity.

#### 4.3 In-training Improvements

Besides data efficiency and diversity promotion, SwiftBrush’s image-free training still has room for improvement. First, it allows an easy means to scale up training data by collecting more prompt inputs. This task is simple, given the abundance of textual datasets and the availability of large language models, unlike the costly and labor-intensive task of collecting image-text pair data commonly required. Second, by not forcing the output of the student model to be the same as that of the teacher, SwiftBrush allows the student to go even beyond the quality and capability of the teacher. We can advocate it to happen by adding extra auxiliary loss functions in SwiftBrush training. In this section, we will discuss the implementation of those ideas for improvement.

Implications of Dataset Size. SwiftBrush’s image-free approach allows for scalable training datasets without limitations. To explore the dataset’s impact on SwiftBrush performance, we conducted supplementary experiments by augmenting the dataset with an additional 2M prompts from the LAION dataset [47] to the original 1.5M deduplicated prompts from the JourneyDB dataset [36]. Analysis (Tab. 2) reveals improved performance with the expanded dataset. Specifically, this leads to a significant improvement in terms of FID and precision, suggesting a positive correlation between dataset size and the quality of the generated outputs. However, a slight degradation in recall was observed, indicating a potential trade-off between image diversity and overall quality. Furthermore, despite an increase in CLIP score compared to the previous version, there remains room for improvements in terms of text alignment.

- Table 2: The impact of SD Turbo initialization and dataset size on SwiftBrush’s performance, compared to SD Turbo on the zero-shot COCO-2014 benchmark.

##### Model Name Training size FID↓ CLIP↑ Precison↑ Recall↑

SD Turbo - 16.10 0.33 0.65 0.35 SwiftBrush 1.3M 15.46 0.30 0.47 0.46 SwiftBrush (Turbo init.) 1.3M 14.59 0.30 0.43 0.55 SwiftBrush (Turbo init.) 3.3M 11.27 0.31 0.48 0.54

|Student<br><br>[Figure 33]<br><br>CLIP Loss<br><br>VSD Loss<br><br>Sampled noise Latent<br><br>|Prompt<br><br>|
|---|
<br><br>TinyVAE Decoder<br><br>| | |
|---|---|
| | |
<br><br>Prompt : A realistic photo of a cat sleeping on the tabletop<br><br>Generated image<br><br>[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>UNet<br><br>LoRA<br><br>[Figure 38]<br><br>Resource-efficient advanced training|
|---|

|Student<br><br>[Figure 39]<br><br>VSD Loss<br><br>Sampled noise Latent<br><br>Prompt<br><br>| | |
|---|---|
| | |
<br><br>: A realistic photo of a cat sleeping on the tabletop<br><br>[Figure 40]<br><br>UNet<br><br>[Figure 41]<br><br>Simple full model training|
|---|

###### Our final student

- Fig. 2: SwiftBrush v2 overview: two versions of the student model: a fully finetuned model trained with the Variational Score Distillation (VSD) loss, and a LoRA finetuned model trained with both VSD and CLIP loss. The final model is obtained by merging the two student models, leveraging the strengths of both training schemes.

Tackling the text-alignment problem. To refine the coherence between textual prompts and visual outputs, we integrate an additional CLIP loss within the distillation process. However, naively employing such loss between the student model’s predictions and the original textual prompts poses challenges , as over-optimizing for the CLIP score potentially degrade image quality. We observed issues such as blurriness, increased color saturation, and the emergence of textual artifacts within the generated images.

To address this, we propose clamping the CLIP value during training with ReLU activation. This aims to balance text alignment with preserving image quality, ensuring the model maintains visual integrity. Additionally, we introduce dynamic scheduling to control the influence of CLIP loss, gradually reducing its weight to zero by the end of distillation. This balanced approach integrates visual-textual alignment and image fidelity effectively. Our clamped CLIP loss is formulated as:

LCLIP = max(0,τ − ⟨Eimage (D (fθ(z,y))),Etext(y)⟩), (5)

where Eimage and Etext represent the CLIP image and text encoders, respectively. D is the VAE decoder used to map the latent back to the image. The term τ introduces a threshold on the desired cosine similarity ⟨·,·⟩ between the image and text embeddings, preventing the model from overemphasizing textual alignment

- at the expense of image quality.

#### 4.4 Resource-efficient training schemes

While our CLIP loss is highly beneficial, it comes with memory and computation costs. Particularly, the CLIP image encoder can only work on image space, requiring decoding the predicted latent to image via the image decoder D as can be seen in Eq. (5). We find incorporating CLIP loss into SwiftBrush’s full-model distillation significantly slows down training speed, particularly on GPUs with moderate VRAM. This urges us to design a resource-efficient training scheme to fully exploit the proposed CLIP loss in constrained setting.

It is possible to significantly reduce memory requirements during fine-tuning with the LoRA framework [17], where only a set of small-rank parameters are trained. Also, to compute the CLIP loss, the predicted latent goes through a large VAE decoder, increasing training length and memory consumption. To address this, we integrate TinyVAE [4], a compact variant of Stable Diffusion’s VAE. TinyVAE sacrifices some fine detail in images but preserves overall structure and object identity comparable to the original VAE. This approach maintains training efficiency close to those of the original fully fine-tuned model, as shown in Tab. 6 and Sec. 5.3.

#### 4.5 Post-training improvements

Recent literature [25] has shown a growing interest in model fusion techniques, aiming to integrate models performing distinct subtasks into a unified multitask model [21, 53] or combining fine-tuned iterations to create an enhanced version [10,19,50]. Our research focuses on the latter, particularly in the context of one-step text-to-image diffusion models. These models, although designed for the same task, differ in their training objectives, providing each with unique advantages. By merging these models, we aim to create a new model that captures the strength of each model without increasing model size or inference costs. Given two one-step diffusion models with weights θA and θB and an interpolation weight λ, we merge them using a simple linear interpolation of the weights:

θmerged = λθA + (1 − λ)θB. (6)

We empirically demonstrate the benefit of such interpolation scheme with SD Turbo, known for its precision and strong text alignment, and the original SwiftBrush, which excels in diversity. In our empirical analysis (refer to Fig. 3), we observe that by interpolating from one model to the other, all evaluated metrics (except for the CLIP score) show improvement at some optimal point. This indicates that the fused model could potentially outperform the original models. These findings underscore the potential of model fusion techniques in enhancing model efficacy, as evidenced by the metric analysis.

As discussed in Sec. 4.4, our proposal suggests two training schemes. We can either train the student model with LoRA and TinyVAE utilizing VSD and CLIP losses or fully finetune the student model employing only VSD loss. These two training schemes lead to two resulting one-step models with different behaviors, making them ideal ingredients for merging. By merging these models, we obtain the final model output of our proposed SwiftBrush v2 framework.

- 15
- 16
- 17

CLIPScore

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

FIDScore

Precision

0.45

0.33

Recall

0.6

0.32

0.40

0.5

0.31

0.35

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Interpolation Weight

Interpolation Weight

- Fig. 3: The effect of weight interpolation upon FID, CLIP score, precision, and recall calculated on the zero-shot MS COCO-2014 benchmark. 0.0 indicates SD Turbo, and 1.0 indicates the original SwiftBrush.

### 5 Experiments

#### 5.1 Experimental Setup

Evaluation metrics. Our text-to-image model is evaluated using the “zeroshot” setting, i.e., trained on some datasets and tested on another dataset, undergoing comprehensive evaluation across three key aspects: image quality, diversity, and textual fidelity. We use the Fréchet Inception Distance (FID) [15] on resized 256 × 256 images as our primary metric for evaluating image quality, consistent with prior text-to-image research [23]. In addition, we employ precision [24] as a complementary metric to FID. For evaluating diversity, we rely on the recall metric [24]. Textual alignment is measured using the CLIP [39] score and the Human Preference Score v2 [51] (HPSv2).

Datasets. We utilize two training datasets: (1) 1.3M prompts from JourneyDB [36], and (2) an expanded dataset incorporating 2M prompts from LAION [47], totaling 3.3M prompts. Additionally, a human-feedback dataset, comprising 200K pairs from LAION-Aesthetic-6.25+ [47], can be optionally used for further image regularization [54], which is around 5% of the total training data. We use the MS COCO-2014 validation set as the standard zero-shot text-to-image benchmark, consistent with established practices in the field [13,23,30,32,45,46]. Samples are generated from the first 30k prompts, with the entire dataset serving as the reference for obtaining metrics. For HPSv2, we adopt the evaluation protocol from [51].

Training details. Our method is built on top of SwiftBrush [34] with our proposed modifications. We conduct all our training on four NVIDIA A100 40GB GPUs, with training durations of one or three days depending on the dataset (JourneyDB alone or combined with LAION prompts). The batch size is 16 per GPU, and we use learning rates of 1e−6 and 1e−3 for the student and LoRA teacher, respectively, with the AdamW optimizer [31]. Our approach utilizes Stable Diffusion 2.1 [41] with cfg = 4.5 as the frozen teacher and LoRA teacher initialized with rank r = 64 and scaling γ = 128. As for the LoRA student, we set r = 256 and γ = 512 to enhance its learning capacity. In addition, we introduce the clamped CLIP loss with a margin of τ = 0.35, starting with a weight of 0.1 and gradually reducing to zero. We use ViT-B/32 [18] as the backbone for CLIP image and text feature extraction. Finally, we merge two final models with λ = 0.5, further details are available in the Appendix.

- Table 3: Quantitative comparisons between our method and others on zero-shot MS COCO-2014 benchmark. For multi-step SD models, we report each with the cfg that returns the best FID, e.g., cfg = 3 for SDv1.5 and cfg = 2 for SDv2.1. We also report performance of the teacher model (SDv2.1 with cfg = 4.5). † denotes reported numbers, ‡ denotes our rerun based on provided GitHubs. ‘-’ denotes unreported data. Ours* indicates training with additional image regularization.

##### Method NFEs FID↓ CLIP↑ Precision↑ Recall↑

StyleGAN-T [45]† 1 13.90 - - GigaGAN [22]† 1 9.09 - - -

SDv1.5 [41] (cfg = 3)† 25 8.78 0.30 0.59 0.53 SDv2.1 [41] (cfg = 2)‡ 25 9.64 0.31 0.57 0.53 SDv2.1 [41] (cfg = 4.5)‡ 25 12.26 0.33 0.61 0.41

SD Turbo [46]‡ 1 16.10 0.33 0.65 0.35 UFOGen [52]† 1 12.78 - - MD-UFOGen [57]† 1 11.67 - - HiPA [56]† 1 13.91 0.31 - InstaFlow-0.9B [30]‡ 1 13.33 0.30 0.53 0.45 DMD [54]† 1 11.49 0.32 - SwiftBrush 1 15.46 0.30 0.47 0.46 Ours 1 8.77 0.32 0.55 0.53 Ours* 1 8.14 0.32 0.57 0.52

#### 5.2 Comparison with Prior Approaches

Quantitative results. Tab. 3 presents a comprehensive quantitative comparison between our approach and prior text-to-image models. This encompasses GAN-based models (group 1), multi-step diffusion models (group 2), and a variety of distillation techniques (group 3), both with and without image supervision. Our approach outperforms all competitors, notably achieving superior results even without direct image regularization. Remarkably, our distilled student models exceed their teacher model, SDv2.1, in FID scores by a significant margin while maintaining equivalent model size and inference times comparable to SD Turbo or SwiftBrush. Our model effectively addresses previous text alignment issues observed in SwiftBrush, being close to the CLIP scores of SD Turbo and multi-step models. Precision metrics show high-realism akin to the reference dataset, enhanced solely with a text-driven training dataset. Notably, ours exhibits significant recall improvements due to its image-free nature. Image-based regularization further improves student quality with a small reduction in recall.

In terms of HPSv2 (Tab. 4), our approach achieves competitive scores compared to the multi-step teacher model SDv2.1 and other distillation methods. Particularly, our model with image regularization achieves the highest HPSv2 scores for photos and remains close to the top performers in other categories.

Qualitative results. We provide a qualitative comparison in Fig. 5. Our model produces higher quality compared to its teachers and one-step counterparts. In

- Table 4: HPSv2 comparisons between our method and previous work. † denotes reported numbers, ‡ denotes our rerun based on provided GitHubs. Ours* indicates training with additional image regularization. Bold indicates the best, while underline indicates the second best.

##### Method Anime Photo Concept Art Painting

SDv2.1† 27.48 26.89 26.86 27.46 SD Turbo‡ 27.98 27.59 27.16 27.19 InstaFlow† 25.98 26.32 25.79 25.93 BOOT† 25.29 25.16 24.40 24.61 SwiftBrush† 26.91 27.21 26.32 26.37 Ours 27.13 27.56 26.69 26.76 Ours* 27.25 27.62 26.86 26.77

the first row, despite correctly illustrating the jumping action, other models yield deformed shapes of cats. This issue also reappears in the third row. In contrast, our model realistically represents the action and the cat’s figure. SDv2.1 encounters out-of-frame issues in the fourth row, while others generate facial deformities in the last two rows. Meanwhile, our model produces artifact-free faces that are aligned with the text description regarding hairstyle and expression.

Among the counterparts, SD Turbo shows the best image quality but also the worst diversity. We reaffirm its weakness when comparing it to our method in Fig. 6. SD Turbo tends to generate similar images, as evidenced by the similar details and colors for different samples in each prompt. In contrast, our model can generate diverse views and colors in the first prompt as well as different environments in the second prompt.

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

Diversity

Win

Tie

Quality

Lose

0 20 40 60 80 100 Percentage

Fig. 4: User survey. We asked participants to compare the quality and diversity of images generated by our method and its teacher model across 20 random text prompts.

We also surveyed 250 participants to compare our distilled model with its multi-step teacher, SDv2.1. Participants evaluated images generated by each model for 20 random prompts, and the results (Fig. 4) show that our method consistently matched or exceeded the teacher in quality and diversity.

#### 5.3 Ablation Studies

Effect of each proposed component is summarized in Tab. 5. We compare two student training schemes: (1) full model training and (2) efficient model training. Initializing the model from SD Turbo [46] significantly improves the FID in both schemes. Adding prompts from LAION [47] leads to notable FID

SDv2.1 SwiftBrush SD Turbo Ours

InstaFlow

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

“A DSLR photo of a cat jumping over a fence in high resolution”

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

“A hyperrealistic photo of fox astronaut, perfect face, artstation”

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

“A stunning sports photo of a cat snowboarding and backﬂipping taken by ﬁsheye lens”

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

“A child ﬂying a kite on a grassy hill”

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

“Portrait of a woman with freckles and a necklace on her neck lightly smiling at the camera”

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

“Librarian with purple wavy hair holding a book, pixar animated”

- Fig. 5: Exemplified images generated by SD Turbo, SwiftBrush, SDv2.1 with 50 sampling steps, InstaFlow-0.9B and Ours. Images in the same row are sampled from the same text prompt, while images in the same column are from the same model.

SD Turbo Ours

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

“A vibrant ﬁeld of tulips in full bloom”

“A farmer working in a lush green rice ﬁeld”

- Fig. 6: Exemplified images generated by SD Turbo and Ours to demonstrate our method’s high diversity.

improvements in both cases, reaching 11.27 in (1) and 11.02 in (2). Also, efficient training allows for the use of the Clamped CLIP loss, resulting in a considerable FID improvement. Notably, combining models from both schemes further reduces the FID to 8.77, surpassing their SDv2.1 teacher. Finally, with extra image regularization, we achieve a further boost, reaching an FID of 8.14.

Detailed study in clamped CLIP loss is shown in Tab. 6. Naively applying CLIP loss [39] worsens performance, increasing the FID by 5 points. In contrast, our proposed clamped CLIP loss (Eq. (5)) is highly effective, reducing the FID

- Table 5: Ablation of our methods upon zero-shot MS COCO-2014 30K. Bold indicates the best, while underline indicates the second best.

Label SD Turbo CLIP LAION FID↓ CLIP↑ Precision↑ Recall↑ SwiftBrush 15.46 0.30 0.47 0.46

Fully training

✓ 14.59 0.29 0.43 0.55

- A ✓ ✓ 11.27 0.31 0.48 0.54

Efficient training

✓ 13.21 0.32 0.61 0.38 ✓ ✓ 11.70 0.33 0.63 0.42

- B ✓ ✓ ✓ 11.02 0.32 0.51 0.52

Ours Merge A and B 8.77 0.32 0.55 0.53 Ours* Merge A and B w/ regularization 8.14 0.32 0.57 0.52

- Table 6: Ablation of our enhanced CLIP loss and resource-efficient training scheme.

CLIP Clamped Scheduler Efficient FID↓ CLIP↑ GPU days

14.59 0.297 4.1

✓ 21.32 0.337 7.8 ✓ ✓ 13.19 0.319 7.8 ✓ ✓ ✓ ✓ 11.70 0.330 4.3

to 13.19. Moreover, employing a weight scheduler for the clamped CLIP loss further enhances performance, resulting in an FID of 11.70.

### 6 Discussion and Conclusion

Limitations: Despite the promising results, our distilled model still inherits some limitations from the teacher model, such as compositional problems. To address these limitations, future work could explore the integration of auxiliary losses that focus on cross-attention mechanisms during the distillation process.

Societal Impact: Our advancements improve high-quality image synthesis speed

and accessibility. However, misuse of our advancements could spread misinformation and manipulate public perception. Thus, responsible use and safeguards are crucial to ensure that the benefits outweigh the risks.

Conclusion: This paper proposes a novel method to enhance SwiftBrush, a onestep text-to-image diffusion models. We address the quality-diversity trade-off by initializing the SwiftBrush student model with SD Turbo’s pretrained weights and incorporating efficient in-training techniques as well as margin CLIP loss and large-scale dataset training. Also, by weight merging and optional image regularization, we achieve an outstanding FID score of 8.14, surpassing existing approaches in both GAN-based and one-step diffusion-based text-to-image generation while maintaining near real-time inference speed.

### References

- 1. Agarwal, A., Karanam, S., Joseph, K., Saxena, A., Goswami, K., Srinivasan, B.V.: A-star: Test-time attention segregation and retention for text-to-image synthesis. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 2283–2293 (2023)
- 2. Balaji, Y., Nah, S., Huang, X., Vahdat, A., Song, J., Zhang, Q., Kreis, K., Aittala, M., Aila, T., Laine, S., et al.: ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324 (2022)
- 3. Betker, J., Goh, G., Jing, L., Brooks, T., Wang, J., Li, L., Ouyang, L., Zhuang, J., Lee, J., Guo, Y., et al.: Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf 2(3), 8 (2023)
- 4. Bohan, O.B.: Madebyollin/taesd (Mar 2024), https://github.com/madebyollin/ taesd
- 5. Chang, H., Zhang, H., Barber, J., Maschinot, A., Lezama, J., Jiang, L., Yang, M.H., Murphy, K., Freeman, W.T., Rubinstein, M., et al.: Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704

(2023)

- 6. Chang, H., Zhang, H., Jiang, L., Liu, C., Freeman, W.T.: Maskgit: Masked generative image transformer. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11315–11325 (2022)
- 7. Check, F.: Images appearing to show Donald Trump arrest created by AI. Reuters (Mar 2023), https://www.reuters.com/article/idUSL1N35T2TU
- 8. Chefer, H., Alaluf, Y., Vinker, Y., Wolf, L., Cohen-Or, D.: Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM Transactions on Graphics (TOG) 42(4), 1–10 (2023)
- 9. Chen, J., Wu, Y., Luo, S., Xie, E., Paul, S., Luo, P., Zhao, H., Li, Z.: Pixart-δ: Fast and controllable image generation with latent consistency models (2024)
- 10. Choshen, L., Venezian, E., Slonim, N., Katz, Y.: Fusing finetuned models for better pretraining (2022)
- 11. Ding, M., Yang, Z., Hong, W., Zheng, W., Zhou, C., Yin, D., Lin, J., Zou, X., Shao, Z., Yang, H., et al.: Cogview: Mastering text-to-image generation via transformers. Advances in Neural Information Processing Systems 34, 19822–19835 (2021)
- 12. Epstein, D., Jabri, A., Poole, B., Efros, A., Holynski, A.: Diffusion self-guidance for controllable image generation. Advances in Neural Information Processing Systems 36 (2024)
- 13. Gu, J., Zhai, S., Zhang, Y., Liu, L., Susskind, J.: BOOT: Data-free Distillation of Denoising Diffusion Models with Bootstrapping. arXiv preprint arXiv:2306.05544

(2023)

- 14. He, Y., Yang, S., Chen, H., Cun, X., Xia, M., Zhang, Y., Wang, X., He, R., Chen, Q., Shan, Y.: Scalecrafter: Tuning-free higher-resolution visual generation with diffusion models. In: The Twelfth International Conference on Learning Representations (2023)
- 15. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30 (2017)
- 16. Hong, S., Lee, G., Jang, W., Kim, S.: Improving sample quality of diffusion models using self-attention guidance. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 7462–7471 (2023)

- 17. Hu, E.J., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. In: Int. Conf. Learn. Represent. (2021)
- 18. Ilharco, G., Wortsman, M., Wightman, R., Gordon, C., Carlini, N., Taori, R., Dave, A., Shankar, V., Namkoong, H., Miller, J., Hajishirzi, H., Farhadi, A., Schmidt, L.: Openclip (Jul 2021). https://doi.org/10.5281/zenodo.5143773, https://doi. org/10.5281/zenodo.5143773, if you use this software, please cite it as below.
- 19. Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., Wilson, A.G.: Averaging weights leads to wider optima and better generalization. arXiv preprint arXiv:1803.05407 (2018)
- 20. Jia, C., Yang, Y., Xia, Y., Chen, Y.T., Parekh, Z., Pham, H., Le, Q., Sung, Y.H., Li, Z., Duerig, T.: Scaling up visual and vision-language representation learning with noisy text supervision. In: International conference on machine learning. pp. 4904–4916. PMLR (2021)
- 21. Jin, X., Ren, X., Preotiuc-Pietro, D., Cheng, P.: Dataless knowledge fusion by merging weights of language models. arXiv preprint arXiv:2212.09849 (2022)
- 22. Kang, M., Zhu, J.Y., Zhang, R., Park, J., Shechtman, E., Paris, S., Park, T.: Scaling up gans for text-to-image synthesis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10124–10134 (2023)
- 23. Kang, M., Zhu, J.Y., Zhang, R., Park, J., Shechtman, E., Paris, S., Park, T.: Scaling up gans for text-to-image synthesis. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2023)
- 24. Kynkäänniemi, T., Karras, T., Laine, S., Lehtinen, J., Aila, T.: Improved precision and recall metric for assessing generative models. Advances in Neural Information Processing Systems 32 (2019)
- 25. Li, W., Peng, Y., Zhang, M., Ding, L., Hu, H., Shen, L.: Deep model fusion: A survey. arXiv preprint arXiv:2309.15698 (2023)
- 26. Li, Y., Keuper, M., Zhang, D., Khoreva, A.: Divide & bind your attention for improved generative semantic nursing. In: British Machine Vision Conference (2023)
- 27. Lin, S., Wang, A., Yang, X.: Sdxl-lightning: Progressive adversarial diffusion distillation (2024)
- 28. Lin, T., Maire, M., Belongie, S.J., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft COCO: Common Objects in Context. Eur. Conf. Comput. Vis. (2014)
- 29. Liu, X., Gong, C., Liu, Q.: Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. Int. Conf. Learn. Represent. (2023)
- 30. Liu, X., Zhang, X., Ma, J., Peng, J., Liu, Q.: InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation. arXiv preprint arXiv:2309.06380 (2023)
- 31. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017)
- 32. Luo, S., Tan, Y., Huang, L., Li, J., Zhao, H.: Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378 (2023)
- 33. Midjourney: Midjourney. https://www.midjourney.com
- 34. Nguyen, T.H., Tran, A.: Swiftbrush: One-step text-to-image diffusion model with variational score distillation. IEEE Conf. Comput. Vis. Pattern Recog. (2024)
- 35. Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I., Chen, M.: Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In: International Conference on Machine Learning (2021)

- 36. Pan, J., Sun, K., Ge, Y., Li, H., Duan, H., Wu, X., Zhang, R., Zhou, A., Qin, Z., Wang, Y., Dai, J., Qiao, Y., Li, H.: JourneyDB: A Benchmark for Generative Image Understanding. Adv. Neural Inform. Process. Syst. (2023)
- 37. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023)
- 38. Poole, B., Jain, A., Barron, J.T., Mildenhall, B.: Dreamfusion: Text-to-3d using 2d diffusion. In: Int. Conf. Learn. Represent. (2022)
- 39. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: Int. Conf. Learn. Represent. pp. 8748–8763. PMLR (2021)
- 40. Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M.: Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125 1(2), 3 (2022)
- 41. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 10674–10685 (2021)
- 42. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 10684–10695 (2022)
- 43. Roose, K.: AI-Generated Art Won a Prize. Artists Aren’t Happy. N.Y. Times (Sep 2022), https://www.nytimes.com/2022/09/02/technology/ai-artificialintelligence-artists.html
- 44. Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E.L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al.: Photorealistic textto-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems 35, 36479–36494 (2022)
- 45. Sauer, A., Karras, T., Laine, S., Geiger, A., Aila, T.: Stylegan-t: Unlocking the power of gans for fast large-scale text-to-image synthesis. arXiv preprint arXiv:2301.09515 (2023)
- 46. Sauer, A., Lorenz, D., Blattmann, A., Rombach, R.: Adversarial diffusion distillation. arXiv preprint arXiv:2311.17042 (2023)
- 47. Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., Schramowski, P., Kundurthy, S., Crowson, K., Schmidt, L., Kaczmarczyk, R., Jitsev, J.: LAION-5B: An open large-scale dataset for training next generation image-text models. Adv. Neural Inform. Process. Syst. (2022)
- 48. Wah, C., Branson, S., Welinder, P., Perona, P., Belongie, S.: The caltech-ucsd birds-200-2011 dataset (2011)
- 49. Wang, Z., Lu, C., Wang, Y., Bao, F., Li, C., Su, H., Zhu, J.: ProlificDreamer: HighFidelity and Diverse Text-to-3D Generation with Variational Score Distillation. Adv. Neural Inform. Process. Syst. (2023)
- 50. Wortsman, M., Ilharco, G., Gadre, S.Y., Roelofs, R., Gontijo-Lopes, R., Morcos, A.S., Namkoong, H., Farhadi, A., Carmon, Y., Kornblith, S., et al.: Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In: International Conference on Machine Learning. pp. 23965–23998. PMLR (2022)
- 51. Wu, X., Hao, Y., Sun, K., Chen, Y., Zhu, F., Zhao, R., Li, H.: Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis. arXiv preprint arXiv:2306.09341 (2023)

- 52. Xu, Y., Zhao, Y., Xiao, Z., Hou, T.: Ufogen: You forward once large scale text-toimage generation via diffusion gans (2023)
- 53. Yadav, P., Tam, D., Choshen, L., Raffel, C., Bansal, M.: Resolving interference when merging models. arXiv preprint arXiv:2306.01708 (2023)
- 54. Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W.T., Park, T.: One-step diffusion with distribution matching distillation. IEEE Conf. Comput. Vis. Pattern Recog. (2024)
- 55. Yu, J., Xu, Y., Koh, J.Y., Luong, T., Baid, G., Wang, Z., Vasudevan, V., Ku, A., Yang, Y., Ayan, B.K., et al.: Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789 (2022)
- 56. Zhang, Y., Hooi, B.: Hipa: Enabling one-step text-to-image diffusion models via high-frequency-promoting adaptation. arXiv preprint arXiv:2311.18158 (2023)
- 57. Zhao, Y., Xu, Y., Xiao, Z., Hou, T.: Mobilediffusion: Subsecond text-to-image generation on mobile devices (2023)

## SwiftBrush v2: Make Your One-step Diffusion Model Better Than Its Teacher – Supplementary Materials –

### 7 Additional details

#### 7.1 Weight Interpolation

In this section, we first provide quantitative analyses on the model merging process conducted on the fully finetuned (Model A) and the resource-efficient trained model (Model B) to form our final model. We run a comprehensive evaluation by reporting essential metrics, including FID, CLIP score, Precision, and Recall, upon the zero-shot MS-COCO 2014 across different interpolation weights, following the same protocol as in the main paper. We provide the plots in both scenarios when the regularization term is applied (Fig. 7.b) or not (Fig. 7.a). In either case, we observe that the CLIP and the precision scores change monotonically from one model to another, while both the FID and recall scores get enhanced when fusing the two models. This analysis provides a data-driven justification for the selected interpolation weight used in the final model, ensuring it achieves the best combination of visual quality, semantic coherence, and diversity. Specifically, for both cases, we pick the weight to optimize for the FID metrics, while not trading off too much with other metrics, hence the interpolation weight λ = 0.5 serve well with our purposes.

Furthermore, we delve into the visual analysis of model interpolation (Fig. 8), exploring the effects on the generated output as the interpolation weight is varied between two trained one-step text-to-image models. We provide qualitative figures for both the interpolation between SwiftBrush and SD Turbo (analyzed

0.33

0.55

0.55

CLIPScore

- 9

- 10

- 11

FIDScore

Precision

Recall

0.54

0.32

0.50

0.53

0.31

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Interpolation Weight

Interpolation Weight

###### (a) Model Ours.

0.55

CLIPScore

- 9

- 10

- 11

0.320

FIDScore

Precision

0.55

Recall

0.315

0.50

0.50

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Interpolation Weight

Interpolation Weight

(b) Model Ours*.

- Fig. 7: The effect of weight interpolation upon FID, CLIP score, precision, and recall calculated on the zero-shot MS COCO-2014 benchmark when combining models trained with our proposed mechanisms (Model A and B) to form the final model.

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

0.0 0.1 0.2 0.3 0.4 0.5

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

0.6 0.7 0.8 0.9 1.0

(a) Interpolation of original SwiftBrush and SD Turbo. Prompt: “A chihuahua at the beach”

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

0.0 0.1 0.2 0.3 0.4 0.5

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

0.6 0.7 0.8 0.9 1.0

(b) Model Ours* (combined between model A - fully fine-tuned and model B - efficient fine-tuned with LoRA and CLIP loss). Prompt: “cat, airbrush style, soft lighting, detailed face, snowy, concept art, digital painting, epic.”.

Fig. 8: The effect of weight interpolation upon the generated samples.

in the main paper) and the interpolation between Model A and B mentioned above. By gradually adjusting the interpolation weight, we can observe how the visual characteristics of the generated samples evolve, providing insights into the learned representations of each model and their contributions to the final output. This interpolation study allows us to understand the interplay between the two models and how their combined representation affects the generated results.

#### 7.2 CLIP loss

Fig. 9 presents a qualitative comparison between the naive approach of integrating the CLIP loss and our proposed method. The output of the distilled model using the naive approach suffers from poor quality, with issues such as over-saturation, over-smoothing, and the appearance of textual artifacts on the image that reflect the conditioned prompt. These artifacts can be visually dis-

###### Naive CLIP Loss Ours

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

Fig. 9: Effect of the CLIP loss design. We provide a qualitative comparison between naively adding CLIP loss and our approach. The output of the first approach is over-smooth, over-saturated, and has textual artifacts on the image. On the other hand, our approach shows both good diversity and quality. The prompt for the first row is: “A raccoon wearing formal clothes, wearing a tophat. Oil painting in the style of Rembrandt”, while the second’s is: “a beautiful dark forest, a witches house in the trees, creepy, dark wooded”.

tracting and detract from the overall aesthetic of the generated images. In contrast, when the loss is applied properly using our method, the generated images exhibit significant improvements in both quality and diversity. The images are more visually appealing, with better color balance, sharper details, and a greater range of content that accurately captures the essence of the input prompt. By carefully integrating the CLIP loss into the distillation process, our approach effectively leverages the semantic understanding of the CLIP model while preserving the generative capabilities of the underlying model. This results in highquality, diverse images that closely align with the desired output specified by the prompt.

#### 7.3 Dependency on the existing one-step diffusion

Our work aims to enhance the one-step diffusion models’ performance. When no pretrained one-step model is available, we can still run the SwiftBrush (SB) training procedure on a small prompt dataset to build that initial model. To validate SBv2’s effectiveness in that scenario, we re-train our method but using SB pretrained weights for initialization and report the results in Tab. 7. As shown, our final merged model obtains the FID score of 11.69 without image regularization and 11.17 with image regularization. Note that DMD achieves FID of 11.49 with regularization on real images too. This result demonstrates that SBv2 still can reach SoTA one-step performance w/o the help of existing one-step models, though the gain is not as significant. Even when counting SB training time, our pipeline in this setting is still much more efficient than other one-step distillation methods.

- Table 7: Comparison between SwiftBrush (SB) and our method when using SB as initialization weight.

Approach FID Pre Rec SB 15.46 0.47 0.46 Fully ft 12.20 0.51 0.49 LoRA 13.54 0.49 0.47 Merged 11.69 0.50 0.49 Merged* 11.17 0.51 0.49

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Fig. 10: Mode collapse w/ SD Turbo training strategy.

#### 7.4 Compare to SD Turbo training with SB initialization.

Three reasons make this training scheme less favorable than our approach: (1) SD Turbo training needs a large text-image pair dataset, which is costly in storage and computation, while SB training is image-free, making it more efficient and scalable. (2) SD Turbo’s training code isn’t publicly available, making reproduction difficult, whereas SB’s training scheme much easier to reimplement and use. (3) SD Turbo’s adversarial training is prone to mode collapse, requiring careful monitoring. Our attempts to reimplement SD Turbo faced mode collapse issues in early epochs (Fig. 10).

#### 7.5 Training Cost and Inference Speed

Setup. All of the self-measurements are taken on NVIDIA A100 40GB GPUs. However, most of the reported numbers about training time are taken directly from the corresponding papers, which did not specify whether A100 40GB or A100 80GB GPUs were used during training, except for SwiftBrush families. Even though both of these GPUs have equivalent computational speed, the 80GB version is capable of larger training batch size, which can drastically improve the training time. For inference time, we re-run every method except for GigaGAN due to its public model unavailability. We follow a standardized procedure to ensure fair comparisons for inference times. First, we warm up the model by running 5 inference passes. Then, we perform inference 50 times, repeating this process 10 times and taking the average of the results. The inference flow for distilled one-step diffusion models consists of three main steps: text encoding, UNet feedforwarding, and VAE decoding, without the denoising step since all models are one-step. To optimize memory usage and computational efficiency, all input data and model parameters are stored in float16 format throughout the inference process.

Analysis. We sum up the training and inference time of the existing methods, including GAN-based, multi-step diffusion models, and one-step diffusion models in Tab. 8. Notice that all of the one-step diffusion models are in some form

- Table 8: Comparison of inference and training time between our method against other works upon the zero-shot benchmark on MS COCO-2014. †means that we obtain the numbers from the reports. ‡means that we obtain the numbers by ourselves. The inference time in float16 precision for all methods was reproduced using a single NVIDIA A100 40GB to ensure a fair comparison. The units for training time were also calculated using NVIDIA A100. Model A is the fully-finetuned SwiftBrush longer and with more data (as described in the main paper), whereas Model B is trained with auxiliary loss by utilizing the resources-efficient training scheme. Note that in the case of HiPA, the model is trained with COCO-2017, hence not a zero-shot result; and for methods using SDv1.5 as the teacher [30,52,54,56], the inference time gap with ours mostly comes from the used text encoder, which is smaller than SDv2.1 based teacher.

Inference Time (s) ↓

Training Time (GPU days) ↓

Method NFE Distill

FID↓

StyleGAN-T† [45] 1 ✗ 0.10 1792.0 13.90 GigaGAN† [23] 1 ✗ 0.13 6250.0 9.09 SDv1.5† [41] 25 ✗ 1.74 4783.0 8.78 SDv2.1(cfg=4.5)† [41] 25 ✗ 1.77 - 12.26 InstaFlow-0.9B† [30] 1 ✓ 0.12 108.0 13.10 UFOGen† [52] 1 ✓ 0.12 - 12.78 DMD† [54] 1 ✓ 0.12 108.0 11.49 HiPA† [56] 1 ✓ - 3.8 13.91 SD Turbo† [46] 1 ✓ 0.13 - 16.10 SwiftBrush‡ [34] 1 ✓ 0.13 4.1 15.46

- Model A‡ 1 ✓ 0.13 12.1 11.27
- Model B‡ 1 ✓ 0.13 12.0 11.02 Ours (A+B)‡ 1 ✓ 0.13 24.1 8.77 Ours w/ regularizer‡ 1 ✓ 0.13 24.1 8.14

of distillation from the teacher model, hence the training time gap. Among the distillation-based methods, SwiftBrush and our proposed approach stand out for their exceptional performance. SwiftBrush achieves an impressive inference time of 0.13s while maintaining a competitive FID score of 15.46. Our method, while maintaining the same inference speed, further improves the FID metrics, surpassing even the computationally intensive StyleGAN-T and GigaGAN models and the multi-step teacher. Although our method requires a slightly longer training time compared to SwiftBrush, the superior quality of the generated images, as evidenced by the significantly lower FID and other analyses in the main paper, justifies the additional training effort. These results demonstrate the effectiveness of our approach in achieving state-of-the-art image generation quality while maintaining near real-time inference speeds. Note that for methods using SDv1.5 as the teacher [30,52,54,56], the inference time gap mostly comes from the used text encoder, which is smaller than SDv2.1 based teacher.

###### CLIP Loss CLIP + HPS Loss CLIP Loss CLIP + HPS Loss

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

Fig. 11: Qualitative comparison between samples training SwiftBrush with auxiliary losses: CLIP vs CLIP+HPS.

- Table 9: Comparison of metrics between teacher, our method and our method with auxiliary loss HPS upon the zero-shot benchmark on MS COCO-2014. Bold means the best result while underline means the second-best result.

Human Preference Score v2↑ Anime Photo Concept Art Painting

Method FID↓ CLIP↑

SDv2.1(cfg=4.5) 12.26 0.32 27.55 27.80 26.85 26.73 Model B 11.02 0.32 26.55 26.71 26.17 26.11 Model B + HPSv2 11.15 0.32 28.22 28.00 27.35 27.42

### 8 Analysis and further applications

#### 8.1 About the robustness of the training scheme

We further demonstrate the robustness and flexibility of our distillation scheme, as stated in the main paper, by incorporating additional loss functions. Observing that the student can still improve upon the human preference aspect in the HPS benchmark, we integrate an HPSv2 loss, similar to the integration of the CLIP loss: We calculate the HPS score using decoded student predictions and apply a clamped ReLU loss with a weighting scheme for fine-grained control into the current resource-efficient distillation process (which already embeds the CLIP loss).

The results presented in Tab. 9 show that by additionally integrating the HPSv2 loss, we successfully improve the model’s performance on the HPSv2 benchmark under its zero-shot protocol while maintaining other metrics. This showcases the robustness of the SwiftBrush distillation process, enabling us to in-

corporate additional guidance to surpass the performance of the teacher model.

- Fig. 11 illustrates that by including HPS loss, the overall color of the generated images is more soothing for human eyes, enforcing the quantitative result improvement in the HPSv2 benchmark. However, upon closer examination of Tab. 9, we observe slight trade-offs in FID, suggesting the need for further investigation to enhance the synergy between these loss functions for it to surpass our current proposed solution. Therefore, we leave it as a direction for future research.

#### 8.2 Composition

In Fig. 12, similar to other diffusion models, our distilled model still demonstrates low compositional ability and text-image misalignment when tasked with prompts that require generating multiple objects associated with attributes. Our model can generate the purple frog; however, it fails to generate the ball as well as its color. There have been a number of works aimed at solving this issue by running attention guidance [12, 16] or enhancing attention masks [1, 8, 26]. In

- our experiments, we chose to apply the Divide-and-Bind [26] approach since this method enhances both the generation of the objects and their corresponding attributes. Since the model predicts the final image in only one step, we choose to iteratively update the latent 100 times while keeping the same scale size of 20 as the original paper. As illustrated in the final row of Fig. 12, our model demonstrates the ability to generate following the prompt accurately. The technique, initially developed for multi-step diffusion models, has been successfully adapted to enhance the output quality of our one-step model with minimal modifications. This achievement highlights the remarkable versatility and compatibility of our model with existing techniques commonly associated with text-to-image diffusion models. Although the original optimization process was designed for a multi-step approach, resulting in slower running times, our model demonstrates faster performance compared to the original multi-step teacher model. Moving forward, we believe that exploring novel latent optimization methods tailored explicitly for one-step diffusion models presents a promising new research direction, and we hope to draw attention to this area in the future.

#### 8.3 Latent Interpolation

By interpolating between two random input noises via spherical linear interpolation (slerp) in Fig. 13, our model seamlessly captures the gradual transformations in visual features without compromising adherence to the given prompt. This not only showcases the robustness of our model but also highlights the versatile image synthesis capabilities stemming from a well-trained conditional text-to-image model.

#### 8.4 Prompt Interpolation

We showcase our model’s capability when interpolating two input prompt embeddings in Fig. 14 using same template but with only one word different. Our

model effectively captures and blends the characteristics of both prompts, creating visually compelling and coherent intermediate representations. This feature demonstrates our model’s capacity to understand and manipulate the semantic relationships between different textual inputs, providing a powerful tool for creative exploration and generating diverse images that seamlessly bridge different concepts or styles.

#### 8.5 Arbitrary size and aspect ratio generation with ScaleCrafter

Our models inherit the architecture of SDv2.1, which limits the ability to generate images with varying sizes or aspect ratios, unlike other works such as SD-XL [27, 37]. To address this limitation, we apply ScaleCrafter [14], a technique that adjusts the convolution layers of the U-Net model during inference through re-dilation. Fig. 15 illustrates the synthesized images generated in vari-

- ous ratios and resolutions using this method. This once again demonstrates that our one-step model can effectively incorporate techniques from the multi-step diffusion model domain to enhance performance, similar to the application of Divide-and-Bind for improving composition, as discussed in Sec. 8.2.

### 9 More qualitative results

We provide additional comparisons of our model with SD Turbo, SDv2.1, and SwiftBrush in Fig. 16. Fig. 17 illustrates the diversity of our model when generating images with the same input prompts. Finally, Fig. 18 shows more uncurated samples synthesized by our model.

SDSDv2.1Turbo “a photo of a purple frog and a blue ball”

[Figure 137]

[Figure 138]

[Figure 139]

SwiftBrush v1

[Figure 140]

SwiftBrush v2

[Figure 141]

SwiftBrush v2 + Divide & Bind

- Fig. 12: Cross-attention visualization for each object and attribute tokens between our models and the multi-step teacher SDv2.1, SD Turbo and SwiftBrush, generated with the same random seeds for the following prompt: “A photo of a purple frog and a blue ball”.

z1 z2

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

Dog

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

Cat

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

Owl

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

Racoon

- Fig. 13: Results of interpolating the noise input. The prompts used here are selected from a standard template: “A photo of a {animal} reading a book”. Here, ‘animal’ is dog, cat, owl or racoon and we interpolate the noise input using spherical linear interpolation (Slerp). Same text input y is used for images at each row.

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

Dog

Cat

Owl

Racoon

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

Cat

Owl

Racoon

Dog

- Fig. 14: Results of interpolating the input prompt. The prompts used here are selected from a standard template: “A photo of a {animal} reading a book”. Here, the text embedding is interpolated using spherical linear interpolation (Slerp). We interpolate two prompts using the same template but two different ‘animal’ (from left to right) and same noise input z is used for images at each row.

[Figure 222]

[Figure 223]

- A

- B

###### C

1024x512

[Figure 224]

2944x256 512x1024

- Fig. 15: Qualitative results of our model combined with ScaleCrafter. Here, the caption for A is “A dog sitting on a beautiful beach, with palm trees behind, bokeh”, that for B is “A river running through a forest” and that for C is “Photo of a cat holding a pineapple”.

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

SD Turbo SDv2.1 SwiftBrush Ours

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

Prompt: A painting of a cat

Prompt: digital art of a little cat traveling around forest, wearing a scarf around the neck, carrying a tiny backpack on his back

Prompt: Elon Musk as Fremen, Dune

- Fig. 16: Qualitative comparison between our models and the multi-step teacher SDv2.1, SD Turbo and SwiftBrush.

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

Prompt: a architectural drawing of a new town square for Cambridge England, big traditional museum with columns, fountain in middle, classical design, traditional design, trees

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

Prompt: A burger falling in pieces juicy, tasty, hot, promotional photo, intricate details, hdr, cinematic, adobe lightroom, highly detailed

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

Prompt: a cute kitty, (extremely detailed CG unity 8k wallpaper), professional majestic impressionism oil painting

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

Prompt: A celestial temple in a distant galaxy with cosmic elements and ancient architecture, drawing inspiration from the concept art of Craig Mullins, featuring ethereal lighting

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

Prompt: A futuristic cyberpunk street scene, echoing the stylistic choices of Yukito Kishiro, featuring detailed urban environments

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

Prompt: Snow Princess, smooth soft skin, soft lighting, detailed face, concept art, digital painting, looking into camera

###### Fig. 17: Diversity. Our models generate realistic, diverse images spanning various object categories, styles, and scenes.

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

###### Fig. 18: Additional qualitative images generated by our model.

