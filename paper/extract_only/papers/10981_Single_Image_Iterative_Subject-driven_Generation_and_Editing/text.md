## Single Image Iterative Subject-driven Generation and Editing

Yair Shpitzer Bar-Ilan University

Gal Chechik Bar-Ilan University, NVIDIA

Idan Schwartz Bar-Ilan University

# arXiv:2503.16025v1[cs.CV]20Mar2025

Edit Image Editing Edit Image Generation

Subject Image

|[Figure 1]|
|---|

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

"... eating a salad"

"... celebrate b-day"

[Figure 8]

[Figure 9]

|[Figure 10]|
|---|

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

"A boy is holding...""... drinking coffee"

Figure 1. SISO is an inference-time optimization method to personalize images from a single subject image without training . SISO can personalize the subject of a given image or generate new images with the personal subject.

#### Abstract

set of personal subjects, and demonstrate significant improvements over existing methods in image quality, subject fidelity, and background preservation.

Personalized image generation and image editing from an image of a specific subject is at the research frontier. It becomes particularly challenging when one only has a few images of the subject, or even a single image. A common approach to personalization is concept learning, which can integrate the subject into existing models relatively quickly but produces images whose quality tends to deteriorate quickly when the number of subject images is small. Quality can be improved by pre-training an encoder, but training restricts generation to the training distribution, and is time consuming. It is still a difficult and open challenge to personalize image generation and editing from a single image without training. Here, we present SISO, a new, training-free approach based on optimizing a similarity score with an input subject image. More specifically, SISO iteratively generates images and optimizes the model based on loss of similarity to the given subject image until a satisfactory level of similarity is achieved, allowing plug-and-play optimization to any image generator. We evaluated SISO in two tasks, image editing and image generation, using a diverse data

#### 1. Introduction

Subject-driven text-conditioned image generation and editing combines the ease-of-use of prompt-conditioning with the superior visual control provided when creating visual content using personalized elements. It is crucial for creative expression, from advertising to digital art, but remains a challenging task when only few images of the personal element are available.

The most common approach to personalization is concept learning, where a pre-trained model is fine-tuned on a few images of a specific concept [10, 47]. While effective when multiple training samples are available, these methods struggle when given only a single image, failing to generalize and often overfitting to the specific details of the input. This leads to style leakage and structural distortions rather than accurate personalization. encoder-based meth-

ods [11, 31, 67] adapt better to a single image by training on a diverse set of concepts. However, this training requires significant computational resources and datasetspecific tuning, delaying their public availability. As a result, subject-driven generation and editing remain largely inaccessible for newer models, leaving the challenge of efficient single-image personalization unsolved.

To address these challenges, we describe a new method called SISO (Single Image Subject Optimization) . During generation, SISO directly optimizes a subject similarity score between the generated image and a single image. Specifically, we show that by using a similarity score based on DINO [37] and IR features [51]SISO excels at capturing identity features and filtering out the background even with a single image. By optimizing this score, our method focuses on preserving the identity of the concept rather than other elements of the scene.

Employing pre-trained score models for fine-tuning a diffusion model presents significant challenges. Current approaches [8, 47] continue the standard optimization of a diffusion process, which can be viewed as predicting the noise of a given latent. They do not work with pixel-level input because they operate on the latent space. In contrast with these previous methods, our optimization process iteratively takes as input decoded generated images during inference. We generate an image at each step, compute a similarity loss, and update the parameters. After each step, we generate a new image and repeat the process until a satisfactory level is achieved.

Our method steers the model at inference time by backpropagating through the diffusion process. With the rise of distilled diffusion models that require as few as one denoising step [30, 42], our approach becomes significantly more practical. We further describe how SISO can be efficiently applied to standard diffusion processes like Sana [64], which can be computationally expensive. We describe a two-stage training simplification: first, training in an efficient setup with a low number of denoising steps and simple prompts; then, at inference, applying the optimized model with more denoising steps and varied prompts to enhance output quality.

Fig. 1 demonstrates the effectiveness of SISO, personalizing with a single subject image. SISO allows for highly natural edits, such as accurately replacing the cat while keeping the original cat’s stance. For the plush images, we successfully replaced the subject without altering the background, maintaining a natural pose on the tree. Additionally, our image-generation variant showcases the subject’s versatility in various complex prompts.

Beyond improving accuracy and image quality, the testtime optimization approach presented here offers two benefits: (i) it is plug-and-play, meaning both the similarity loss and the generative model can easily be replaced, making

it very suitable for the high-paced release cycles of image generators; and (ii) the optimization generates an image at each step, making the optimization process visible and able to stop at each point, enhancing user control.

We ran SISO with a single subject image for both generating and editing images on the ImageHub benchmark, demonstrating significant improvements in image naturalness while maintaining high fidelity in identity and background preservation. Our human evaluations support these results, showing better prompt alignment and naturalness in image generation, as well as enhanced background preservation and naturalness in image editing. We also provide qualitative results illustrating the significant improvements.

This work has the following contributions:

- (i) We propose SISO, a novel inference-time iterative optimization technique that alters the subject of a vanilla image generator using only one reference subject image.
- (ii) We show that SISO can be applied to two popular tasks: subject-driven image generation and editing, with minor adaptations to the regularization of penalties.
- (iii) Our results demonstrate significant improvements in single-image subject-driven personalization, opening up a new thread for research in image personalization that, to our knowledge, has not been explored yet.

#### 2. Related Work

Concept Learning. Concepts are typically trained using a small set of up to 20 images. Various fine-tuning techniques have been proposed. Initial attempts used prompt tuning, i.e., learning a token representation [10], and learning negative prompts as well [9]. The following approach updates the entire model [47]. Newer variations learn style and content separately [50] or consider multiple concepts [17]. However, these methods often leak style or fail to learn complex objects needed for subject-driven generation, especially with a limited training image set.

Encoder Learning. Early methods trained an encoder to generate an initial subject embedding or to adjust network weights and then fine-tuning during inference for highquality personalization. However, these methods were often restricted to specific concepts [11, 31, 48]. Recent approaches studied how to bypass inference-time optimization [4, 26, 34, 52, 62, 67]. Significant efforts have focused on personalizing human faces, utilizing identity recognition networks or incorporating them as auxiliary losses to enhance identity preservation [12, 16, 41, 61, 63, 68]. Some recent studies have explored adding cross-attention layers [12, 16, 67]. However, methods that encode subjects into existing cross-attention layers tend to preserve the original content more effectively [1, 34, 40, 57, 60, 63]. Despite their advancements, training such encoders still requires substantial computational resources. Recent state-

### 1) Iterative Optimization

###### 2) Generation

Subject Image

"an oil painting of..."

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

IR DINO

|[Figure 19]|
|---|

|[Figure 20]|
|---|

|[Figure 21]|
|---|

[Figure 22]

"...chasing a ball"

"... eating lunch"

Initial 15th 25th 35th

VAE

VAE

LDM

LDM

"Image of a dog"

[Figure 23]

- Figure 2. SISO workflow for image generation. SISO generates images by iteratively optimizing based on pre-trained identity metrics IR and DINO. The added LoRA parameters are updated at each step, while the rest of the models remain frozen. The left panel shows the progress of subject-driven optimization for the prompt ”image of a dog” by displaying the initial image, followed by the 15th, 25th, and 35th iteration steps. Similarity to the subject image (top) increases during optimization. We find that optimizing with a simple prompt is effective, since the optimized model generates novel images of the subject without further optimization, even with complex prompts, as shown on the right.

of-the-art encoder solutions, such as the ones proposed for Flux [30], require large-scale datasets and extended training times [3, 56]. Furthermore, to the best of our knowledge, no encoder solution currently exists for Sana [64]. In contrast, our proposed method is plug-and-play, allowing for rapid adaptation to a variety of generative models.

Subject-driven Image Editing. Initial methods train an adapter to align image encodings with text encodings [54, 66]. These methods fail on novel concepts. Later methods replaced semantic representations with identity features [5]. Other works add more control via camera parameters or text prompts [38, 65, 69, 70]. Following, identity preservation was improved with part-level composition [6]. Recent works leverage rectified flow models and tailored diffusion noise schedules to enable fast, zero-shot inversion and highquality semantic image editing [7, 35, 45].

Another thread explored concept learning from a set of images instead of training an adapter [14, 15, 33]. These approaches are closely related to ours; however, learning a concept requires up to 20 images, while SISO uses a single image. Another recent method is training-free, creating a collage of the reference on the background image [32]. However, this approach is better suited for insertion rather than subject replacement. Instead, we leverage a subject similarity score that modifies image subjects.

Training Free Image Editing. Refers to methods with no separate learning phase, commonly used in image editing tasks. Style-transfer methods employ an inversion technique and transfer attention key-and-value representations from a reference style image [18, 23, 25, 53] or use an encoder [59]. A recent method fuses content and style without inversion [46]. While training-free methods may enable a single image reference for edits, they mostly focus on style. Our approach, which steers the model at inference time, can learn subjects from input reference images.

#### 3. Method

We introduce SISO (Single Image Subject Optimization), a subject-driven conditioning method operating with a single subject image. SISO operates by fine-tuning the diffusion model at inference time, using a loss function computed over the generated image. Specifically, since SISO operates over images in pixel space, we can use high-quality pretrained models that measure object similarity and encourage the model to produce images similar to the desired subject. This approach is different from existing approaches that operate by predicting the noise, as done during the training of the diffusion model.

###### 3.1. Preliminaries: Conditioned Latent Diffusion

A conditioned latent diffusion model (LDM) generates an image x ∼ p(x|y), where y is the conditioning term, such as text. Training the model is typically achieved by adding

[Figure 24]

1st

IR

DINO Subject Image

MSE

Iterative Optimization

|[Figure 25]|
|---|

|[Figure 26]|
|---|

[Figure 27]

[Figure 28]

Mask

[Figure 29]

6th 12th

1st

Mask

VAE

[Figure 30]

|[Figure 31]<br><br>[Figure 32]|
|---|

LDM

inversion

[Figure 33]

Input Image

- Figure 3. SISO workflow for image editing. The main differences from generation (Fig. 2) are: (1) Use diffusion inversion to map the input image into a latent begins (bottom); and (2) it adds a background preservation regularization term (Eq. 3)

noise to an image and learning to predict the added noise: minθ ||ϵˆθ(zt,t,y) − ϵt||22 . Here, zt is an intermediate noisy latent, ϵt is the added noise up to step t, and θ represents the learnable weights. In many personalization approaches, fine-tuning the model is achieved using the same objective followed during training, namely, reconstruction loss over the latents. In personalization tasks, one is given a set of images of a specific subject that one wishes the model to learn. Here, we assume that only a single subject image is given and denote it by xs.

###### 3.2. SISO: Single-Image Subject Optimization

SISO optimizes the image generation model during inference using the generated images to compute the loss. By defining the loss in pixel space, we enable using highquality pre-trained models to measure the similarity between the subject in the generated image and in the input.

SISO operates iteratively (Fig. 2 left). We start with randomly initializing low-rank adaptation parameters θLoRA and adding them to a diffusion model following LoRA [21]. We also fix a specific seed for the noise latent zT and use a deterministic sampler [53]. Then, at step i of the iterative process, we generate an image xˆi using the diffusion model. The generated image is the output of a differentiable and deterministic LDM, hence any differentiable loss L(θLoRA,xˆi) computed over the image can be used for propagating gradients back to the model parameters θLoRA.

To preserve subject identity, we set L to be a subject similarity loss that takes the generated image xˆi and a reference subject image xs as input and computes the similarity of subjects across images. We then update the parameters with a gradient descent step:

θLoRA ←− θLoRA + α∇θLoRAL(ˆxi,xs) |L(ˆxi,xs)|2

. (1)

This update rule is simplified for brevity. In practice, we use Adam optimizer [27]. After updating the model parameters, we repeat this process iteratively. Since this iterative process involves generating well-formed images, rather than noisy latent, it can be used in an interactive manner. Users can observe and stop the optimization process based on the optimized image displayed at each step, or it can stop automatically using standard early stopping strategies (see Appendix C).

By default, backpropagation through LDM is performed through the entire diffusion process, significantly increasing memory requirements. Our approach is particularly well suited for efficient distilled turbo variants that require only a single diffusion step [42]. To support non-distilled models and reduce computational costs, we stop backpropagation after several denoising steps. For instance, with Sana, we backpropagated through the last three denoising steps, which we find sufficient for personalization. This is probably because the final diffusion steps primarily refine local appearance details [20].

We now discuss in detail how SISO can be used for (i) image generation and (ii) image editing.

###### 3.3. Subject-driven Image Generation

To use SISO for generation, we expect two inputs: a conditioning prompt and a single reference image of the subject. We define the similarity loss as

Lsim(ˆxi,xs) = a · δDINO(ˆxi,xs) + b · δIR(ˆxi,xs), (2)

where xˆi is the generated image at optimization step i, δDINO and δIR are distances in DINO [37] and IR [51] embedding spaces, and a,b ∈ R are calibration hyper-parameters. IR and DINO are suited for assessing the identity distance of objects independent of background influences. Using two metrics in our loss function serves two purposes. (i) They enhance performance thanks to an “ensemble” effect; and (ii) they serve as a form of penalty regularization, mitigating the risk of mode collapse that might occur when optimizing based on a single metric.

Training Simplification. To enhance training stability, we find generating simple images using a simple prompt beneficial, as similarity metrics often struggle in complex scenes. Additionally, we observe that training with a low number of denoising steps, even a single step, is sufficient for efficiency.

Notably, the optimized LoRA weights, even when trained with a simple prompt and minimal denoising steps, can be used for inference with different prompts and more denoising steps to enhance quality.

This insight inspired a two-stage approach for handling detailed scenes: (1) first, optimize with a simple prompt and a low number of denoising steps, then (2) use the fine-tuned

- Table 1. Comparison of two baselines for subject-driven image generation using a single reference image per subject. We evaluate identity preservation (DINO, IR), prompt adherence (CLIP-T), and naturalness (FID, KID, CMMD).

Identity Preservation Prompt Adherence Naturalness

DINO↑ IR↑ CLIP-T↑ FID↓ KID↓ CMMD↓

AttnDreamBooth 0.47 0.51 0.29 164.4 0.004 0.41 ClassDiffusion 0.50 0.59 0.29 166.6 0.003 0.18

SISO (ours) 0.48 0.53 0.31 149.2 0.002 0.18

- Table 2. Comparing SISOwith Dreambooth using three backbone models: SDXL-Turbo, Flux Schnell and Sana. for subject-driven image generation using a single reference image. SISO improves prompt adherence while maintaining image fidelity.

Backbone Identity Preservation Prompt Adherence Naturalness Diversity

DINO↑ IR↑ CLIP-T↑ FID↓ KID↓ CMMD↓ MSE↑

DreamBooth SDXL-Turbo 0.58 0.67 0.28 177.69 0.010 0.85 0.05 SISO (ours) SDXL-Turbo 0.48 0.53 0.31 149.2 0.002 0.18 0.11

DreamBooth FLUX Schnell 0.33 0.45 0.25 227.1 0.023 1.09 0.05 SISO (ours) FLUX Schnell 0.51 0.56 0.31 149.5 0.002 0.14 0.12

DreamBooth Sana 0.45 0.46 0.29 149.5 0.003 0.23 0.16 SISO (ours) Sana 0.46 0.51 0.29 149.9 0.003 0.29 0.19

model to generate images with more complex prompts and additional denoising steps. As shown in Fig. 2 (right), after optimizing LoRA weights for the prompt “image of a dog,” the learned subject can be generated for various prompts without further optimization.

- 3.4. Subject-driven Image Editing

#### 4. Experiments

Benchmark Dataset and evaluation protocol. We use the benchmark dataset and the experimental protocol from ImagenHub [29]. For subject-driven image editing, their setup consists of 154 samples, each featuring one of 22 unique subjects from various categories. These include as animals (cat, dog) and day-to-day objects like a backpack, sunglasses, or a teapot. Subject images were taken from DreamBooth [47]. For subject-driven image generation, the setup comprises of 150 prompts with 29 unique sample subjects with similar categories.

In subject-driven image editing, the model swaps the subject of a given image x˜0 with reference image xs while crucially preserving the background, unlike in image generation, where background coherence with the prompt suffices. Additionally, editing an image requires converting it into the domain of the diffusion model (see Fig. 3).

Implementation details. For image generation, we used SDXL-Turbo [49], the distilled version of SDXL [42]. For image editing, we used SD-Turbo1, a distilled version of Stable Diffusion 2.1 [44]. We set the loss calibration hyperparameters to a = 1,b = 1,c = 10, and the learning rate to α = 3e−4. The resolution in all our experiments is 512 × 512.

We begin with inversion using ReNoise inversion [13], which yields faithful inversions (more details in section A of the Appendix). Let xˆ0 = Inversion(˜x0) be the inverted image of x˜0. To preserve the background, we first generate a subject mask Ms by classifying the image xs and employing object detection with Grounding DINO to identify objects of the same class [36]. We then extract a segmentation mask from the detected bounding box using SAM [28]. The background loss is defined as follows:

Baselines. Since our task is to efficiently adapt a pretrained image generator using a single image of a reference subject, we compare SISO against baselines that can operate without requiring to train an encoder learning. For image generation, we compared with AttnDreamBooth [39]. It improves over DreamBooth [47] with a three-stage process, optimizing textual embedding, cross-attention layers, and the U-Net. We also compared with ClassDiffusion, which uses a semantic preservation loss [22]. For image

Lbg(¯xi,xs,xˆ0) = MSE(M¯s(¯xi),M¯s(ˆx0)), (3) where M¯s is the inverse subject mask, i.e., the subject’s background. Intuitively, this loss acts as a penalty for maintaining the background of the original image x˜0. Overall, the loss for subject-driven image editing is:

L(¯xi,xs,xˆ0) = Lsim(¯xi,xs) + c · Lbg(¯xi,xs,xˆ0), (4)

where c is a hyperparameter. We optimize the loss with our iterative inference-time optimization technique.

1https://huggingface.co/stabilityai/sd-turbo

- Table 3. Subject-driven image editing. All experiments used a single reference image per subject. We report identity preservation (DINO, IR, CLIP-I), background preservation (LPIPS), and naturalness (FID, KID, CMMD).

Identity Preservation Background Naturalness

Preservation DINO ↑ IR↑ CLIP-I LPIPS ↓ FID ↓ KID ↓ CMMD ↓

TIGIC 0.51 0.58 0.77 0.22 143.26 0.0066 0.759 SwapAnything 0.45 0.60 0.74 0.11 185.74 0.0277 1.101 SISO (ours) 0.55 0.75 0.80 0.14 114.83 0.0031 0.475

- Table 4. Ablation for image generation. We report identity preservation (DINO, IR) and prompt adherence (CLIP-T)

Identity Preservation Prompt Adherence DINO ↑ IR↑ CLIP-T ↑

SISO (ours) 0.48 0.53 0.31 Ours - w/o Prompt Simpl. 0.52 0.62 0.29 Ours - w/o DINO 0.44 0.50 0.31 Ours - w/o IR 0.49 0.50 0.31

- Table 5. Ablation for image editing. We report identity preservation (DINO, IR, CLIP-I) and background preservation (LPIPS)

Identity Preservation BG Preservation DINO ↑ IR↑ CLIP-I↑ LPIPS ↓

SISO (ours) 0.55 0.75 0.80 0.14 Ours - w/o BG Pres. 0.55 0.76 0.80 0.18 Ours - w/o DINO 0.49 0.74 0.78 0.13 Ours - w/o IR 0.54 0.56 0.78 0.12

editing, we used SwapAnything, which employs masked latent blending and appearance adaptation [15]. All the methods above use concept learning to depict the subject and typically require up to 20 subject images for accurate performance. However, here, we use them with a single image. We also compared with TIGIC, a training-free technique that uses an attention-blending strategy during denoising [32].

###### 4.1. Evaluation Metrics

Identity Preservation. To evaluate subject similarity, we crop the subject using Grounding DINO [36] and compare it using: (i) DINO distance for instance similarity, particularly for animals, (ii) IR features, effective in item similarity [51], and (iii) CLIP-I, which measures class-level similarity [43].

Naturalness. To assess image realism, we compare generated images against a reference set: vanilla Stable Diffusion outputs for generation and input images for editing. We compute three metrics: FID [19], KID [2], which has been shown to be more stable in small datasets, and CMMD [24] for semantically richer CLIP-based evaluation.

Table 6. User study for image editing (left) and generation (right). values are the win rate of our method (fraction of preferred cases) against the leading baseline. ± denotes the standard error of the mean (SEM) based on a binomial distribution.

TIGIC ClassDiffusion (Editing) (Generation)

Identity Preservation 0.45 ± 0.05 0.47 ± 0.05 Naturalness 0.58 ± 0.05 0.65 ± 0.05 Background Preservation 0.60 ± 0.05 Prompt Adherence - 0.69 ± 0.05

Prompt adherence. In image generation, we also measure alignment with the input prompt using CLIP-T, the CLIP score between the generated image and the input prompt.

Diversity. Single-image concept learning often leads to overfitting, limiting diversity in generated images due to reconstruction loss. To quantify this, we compute the mean squared error (MSE) between generated and subject images.

Background Preservation. For image editing, maintaining the background while altering the subject is crucial. We assess this using LPIPS [71], where lower scores indicate higher similarity. To exclude the edited region, we mask the subject using Grounding DINO and SAM [28] before computing LPIPS.

###### 4.2. Quantitative Results

Image Generation. Table 1 shows results for image generation, comparing SISO to two subject-driven baselines that typically learn from multiple subject images but are tested here with a single reference. SISO significantly improves naturalness metrics, suggesting that baselines degrade image quality due to overfitting. Additionally, SISO enhances prompt adherence while maintaining subject identity. This suggests that aligning the image directly, rather than splitting the process into separate optimization and generation stages, improves identity preservation—albeit with a slight trade-off in naturalness or prompt accuracy.

Next, in Table 2, we further evaluate the adaptability of different models for subject-driven generation using a single image. To our knowledge, DreamBooth is the only baseline that can be easily adapted across models, as others are tailored specifically for Stable Diffusion 2.1. Our results show that our method outperforms DreamBooth in identity preservation for FLUX and Sana. Although DreamBooth achieves better identity preservation with SDXL-Turbo, this is mainly due to overfitting, as indicated by the diversity metrics (0.05 vs. 0.11).

Subject Image Ours ClassDiffusion AttnDreamBooth DreamBooth TI

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

...inParis

###### →

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

woodendeck

...ona

###### →

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

...singingkaraoke

###### →

- Figure 4. Qualitative results for subject-driven image generation using a single subject image. The subject image is shown on the left, followed by the given prompt and the generated results from our method and various baselines.

Input Image Subject Image Ours TIGIC SwapAnything DreamBooth TI

[Figure 51]

[Figure 52]

→

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

→

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

→

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

- Figure 5. Qualitative results for subject-driven image editing using a single subject image. Each row shows an original input image to be edited, a reference subject image, and results generated by our method SISO and four baselines.

Image Editing. Table 3 compares our approach against subject-driven image editing baselines. TIGIC blends the subject into the image during diffusion, often resulting in background corruption (0.22 vs. 0.14). SwapAnything learns the subject concept, but when only a single subject image is used, its identity preservation significantly declines (0.55 vs. 0.80 on DINO). Additionally, naturalness metrics are low, with an FID score of 185.7, suggesting that fewer input subject images can substantially drop image quality.

Ablation. In Table 4, we examine prompt simplification. We observe a trade-off: Simplifying the prompt improves adherence, while direct optimization with the full prompt better preserves subject identity.

Table 5 evaluates the impact of background preservation loss (Eq. 3) on editing. Adding this loss improves background consistency (LPIPS: 0.14 vs. 0.18) without compromising identity preservation. We also assess using DINO and IR in an ensemble, which enhances identity preser-

#### SDXL Turbo FLUX Schnell Sana

Subject Image Ours DreamBooth Ours DreamBooth Ours DreamBooth

wearingpinkglasses

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

...readingabook

##### →

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

inarestaurant

two...

##### →

Figure 6. Subject-driven image generation using three backbone models (single reference image)

vation with only slightly reduced background consistency (LPIPS: 0.12 vs. 0.14).

User study. In addition to automated metrics, we conducted a user study to measure identity preservation, background preservation, prompt adherence, and naturalness. We used Amazon MTurk for 100 images, with five raters per image. See full details in the appendix (Sec. B). Two user studies were conducted separately, one for editing and one for generation, comparing our method against the best available baseline of each task.

The results of the user study are given in Table 6. For editing, TIGIC better preserves subject identity because it often acts almost as a copy-paste of the subject into the given input image. This is reflected in SISO obtaining higher scores for both naturalness and background preservation, with win rates of 58% and 60%, respectively. In the generation task, we see a slight improvement for the baseline in subject preservation (47% win rate). However, SISO produces significantly more natural images (65% win rate) and shows prompt adherence (69% win rate).

###### 4.3. Qualitative Results

We begin by showing the results of our generative model compared to popular baselines (Fig. 4). We evaluate subject-driven image generation on three subjects: a plush toy, glasses and a dog. Only our method correctly places the plush in Paris, while others overfit to the input image. Textual Inversion (TI) avoids this but fails to capture identity. Similar issues arise with the glasses, where most methods retain background elements, except ours and TI, though TI lacks detail. Our method preserves subject identity while generating diverse backgrounds. In the final row, the baselines fail to depict the subject and follow the prompt.

In Fig. 5, we compare baselines for image editing by

learning subject concepts, inverting, and regenerating images. In the first row, our method accurately preserves the wolf plush, while baselines either blend unnaturally (TIGIC), leak background details (SwapAnything), or distort both subject and background (DreamBooth, TI). In the second row, our method correctly replaces the black cat, though with a slight eye color mismatch, while baselines fail entirely. In the third row, all methods perform better, but ours best preserves the background.

In Fig. 6, we present subject-driven image generation using a single reference image with SDXL Turbo, FLUX Schnell, and Sana models. DreamBooth, the only baseline adaptable across models, shows several limitations when trained on a single image: (i) low diversity, with generations closely resembling the subject (e.g., the dog generated by SDXL and FLUX, the cat by FLUX), (ii) artifacts and unnatural attributes (e.g., the cats generated by SDXL and FLUX), and (iii) poor identity preservation (e.g., the dog in Sana).

We also assess the stability of our method using various seeds (see Figures 12 and 13 in the appendix).

#### 5. Conclusion

We present SISO, a novel optimization technique that employs a single subject image and enables subject-driven image generation and subject-driven image editing by leveraging pre-trained image similarity score models. We show that in all previous baselines, enabling such capability with a single image in an existing diffusion model is far from being solved. While our method still has room for improvement in subject identity preservation, it opens up a new research thread that may make the personalization of image generators as simple as possible with the use of only a single image.

- 6. Acknowledgments This work was supported by a Vatat datascience grant. References

- [1] Yuval Alaluf, Elad Richardson, Gal Metzer, and Daniel Cohen-Or. A neural space-time representation for text-to-image personalization, 2023. 2
- [2] Mikołaj Bi´nkowski, Dougal J. Sutherland, Michael Arbel, and Arthur Gretton. Demystifying MMD GANs. In International Conference on Learning Representations, 2018. 6
- [3] Shengqu Cai, Eric Chan, Yunzhi Zhang, Leonidas Guibas, Jiajun Wu, and Gordon Wetzstein. Diffusion self-distillation for zero-shot customized image generation. arXiv preprint arXiv:2411.18616, 2024. 3
- [4] Wenhu Chen, Hexiang Hu, Yandong Li, Nataniel Ruiz, Xuhui Jia, Ming-Wei Chang, and William W. Cohen. Subject-driven text-to-image generation via apprenticeship learning. arXiv preprint arXiv:2304.00186, 2023. 2
- [5] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6593–6602, 2023. 3
- [6] Xi Chen, Yutong Feng, Mengting Chen, Yiyang Wang, Shilong Zhang, Yu Liu, Yujun Shen, and Hengshuang Zhao. Zero-shot image editing with reference imitation. ArXiv, abs/2406.07547, 2024. 3
- [7] Zhi Deng, Yibo He, Yulun Zhang, Yunfu Zhang, Zhen Li, Sifei Liu, Zhangyang Wang, Xiaolong Wang, and Yulun Wang. Fireflow: Fast inversion of rectified flow for image semantic editing. arXiv preprint arXiv:2412.07517, 2024. 3
- [8] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems, 34:8780– 8794, 2021. 2
- [9] Ziyi Dong, Pengxu Wei, and Liang Lin. Dreamartist: Towards controllable one-shot text-to-image generation via positive-negative prompt-tuning. arXiv preprint arXiv:2211.11337, 2022. 2
- [10] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In The Eleventh International Conference on Learning Representations, 2023. 1, 2
- [11] Rinon Gal, Moab Arar, Yuval Atzmon, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Encoder-based domain tuning for fast personaliza-

- tion of text-to-image models. ACM Transactions on Graphics (TOG), 42(4):1–13, 2023. 2
- [12] Rinon Gal, Or Lichter, Elad Richardson, Or Patashnik, Amit H. Bermano, Gal Chechik, and Daniel CohenOr. Lcm-lookahead: Encoder-based text-to-image personalization. arXiv preprint arXiv:2401.12345,

2024. 2

- [13] Daniel Garibi, Or Patashnik, Andrey Voynov, Hadar Averbuch-Elor, and Daniel Cohen-Or. Renoise: Real image inversion through iterative noising. arXiv preprint arXiv:2403.14602, 2024. 5
- [14] Jing Gu, Yilin Wang, Nanxuan Zhao, Tsu-Jui Fu, Wei Xiong, Qing Liu, Zhifei Zhang, He Zhang, Jianming Zhang, HyunJoon Jung, and Xin Eric Wang. Photoswap: Personalized subject swapping in images, 2023. 3
- [15] Jing Gu, Nanxuan Zhao, Wei Xiong, Qing Liu, Zhifei Zhang, He Zhang, Jianming Zhang, HyunJoon Jung, Yilin Wang, and Xin Eric Wang. Swapanything: Enabling arbitrary object swapping in personalized image editing. ECCV, 2024. 3, 6
- [16] Zinan Guo, Yanze Wu, Zhuowei Chen, Lang Chen, and Qian He. Pulid: Pure and lightning id customization via contrastive alignment. arXiv preprint arXiv:2404.16022, 2024. 2
- [17] Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, and Feng Yang. Svdiff: Compact parameter space for diffusion fine-tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7323–7334, 2023. 2
- [18] Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or. Style aligned image generation via shared attention.(2023). 2023. 3
- [19] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 6
- [20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020. 4
- [21] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685,

2021. 4

- [22] Jiannan Huang, Jun Hao Liew, Hanshu Yan, Yuyang Yin, Yao Zhao, and Yunchao Wei. Classdiffusion: More aligned personalization tuning with explicit class guidance. arXiv preprint arXiv:2405.17532,

2024. 5

- [23] Xun Huang and Serge Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. In Proceedings of the IEEE international conference on computer vision, pages 1501–1510, 2017. 3
- [24] Sadeep Jayasumana, Srikumar Ramalingam, Andreas Veit, Daniel Glasner, Ayan Chakrabarti, and Sanjiv Kumar. Rethinking fid: Towards a better evaluation metric for image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9307–9315, 2024. 6
- [25] Jaeseok Jeong, Junho Kim, Yunjey Choi, Gayoung Lee, and Youngjung Uh. Visual style prompting with swapping self-attention. arXiv preprint arXiv:2402.12974, 2024. 3
- [26] Xuhui Jia, Yang Zhao, Kelvin C.K. Chan, Yandong Li, Han Zhang, Boqing Gong, Tingbo Hou, Huisheng Wang, and Yu-Chuan Su. Taming encoder for zero fine-tuning image customization with text-to-image diffusion models. arXiv preprint arXiv:2304.02642,

2023. 2

- [27] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 4
- [28] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023. 5, 6
- [29] Max Ku, Tianle Li, Kai Zhang, Yujie Lu, Xingyu Fu, Wenwen Zhuang, and Wenhu Chen. Imagenhub: Standardizing the evaluation of conditional image generation models. arXiv preprint arXiv:2310.01596,

2023. 5

- [30] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 2, 3
- [31] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597,

2023. 2

- [32] Pengzhi Li, Qiang Nie, Ying Chen, Xi Jiang, Kai Wu, Yuhuan Lin, Yong Liu, Jinlong Peng, Chengjie Wang, and Feng Zheng. Tuning-free image customization with image and text guidance. In European Conference on Computer Vision, 2024. 3, 6
- [33] Tianle Li, Max Ku, Cong Wei, and Wenhu Chen. Dreamedit: Subject-driven image editing, 2023. 3
- [34] Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, Ming-Ming Cheng, and Ying Shan. Photomaker: Customizing realistic human photos via stacked id embed-

ding. arXiv preprint arXiv:2312.04461, 2023. 2

- [35] Haonan Lin, Yan Chen, Jiahao Wang, Wenbin An, Mengmeng Wang, Feng Tian, Yong Liu, Guang Dai, Jingdong Wang, and Qianying Wang. Schedule your edit: A simple yet effective diffusion noise schedule for image editing. arXiv preprint arXiv:2410.18756,

2024. 3

- [36] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European Conference on Computer Vision, pages 38–55. Springer, 2025. 5, 6
- [37] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 2, 4
- [38] Yulin Pan, Chaojie Mao, Zeyinzi Jiang, Zhen Han, and Jingfeng Zhang. Locate, assign, refine: Taming customized image inpainting with text-subject guidance. arXiv preprint arXiv:2403.19534, 2024. 3
- [39] Lianyu Pang, Jian Yin, Baoquan Zhao, Feize Wu, Fu Lee Wang, Qing Li, and Xudong Mao. Attndreambooth: Towards text-aligned personalized text-toimage generation. Advances in Neural Information Processing Systems, 37:39869–39900, 2025. 5
- [40] Or Patashnik, Rinon Gal, Daniil Ostashev, Sergey Tulyakov, Kfir Aberman, and Daniel Cohen-Or. Nested attention: Semantic-aware attention values for concept personalization. arXiv preprint arXiv:2501.01407, 2025. 2
- [41] Xu Peng, Junwei Zhu, Boyuan Jiang, Ying Tai, Donghao Luo, Jiangning Zhang, Wei Lin, Taisong Jin, Chengjie Wang, and Rongrong Ji. Portraitbooth: A versatile portrait model for fast identity-preserved personalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27080–27090, 2024. 2
- [42] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2, 4, 5
- [43] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning Transferable Visual Models From Natural Language Supervision. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, pages 8748–8763. PMLR,

2021. 6

- [44] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022. 5
- [45] Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Semantic image inversion and editing using rectified stochastic differential equations. arXiv preprint arXiv:2410.10792, 2024. 3
- [46] Litu Rout, Yujia Chen, Nataniel Ruiz, Abhishek Kumar, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Rb-modulation: Training-free personalization of diffusion models using stochastic optimal control. arXiv preprint arXiv:2405.17401, 2024. 3
- [47] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. 1, 2, 5
- [48] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models. arXiv preprint arXiv:2307.06949, 2023. 2
- [49] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pages 87–

103. Springer, 2025. 5

- [50] Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, and Varun Jampani. Ziplora: Any subject in any style by effectively merging loras. In European Conference on Computer Vision, pages 422–438. Springer, 2025. 2
- [51] Shihao Shao and Qinghua Cui. 1st place solution in google universal images embedding, 2022. 2, 4, 6
- [52] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without test-time finetuning. arXiv preprint arXiv:2304.03411, 2023. 2
- [53] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2020. 3, 4
- [54] Yizhi Song, Zhifei Zhang, Zhe Lin, Scott Cohen, Brian Price, Jianming Zhang, Soo Ye Kim, and Daniel Aliaga. Objectstitch: Generative object compositing. arXiv preprint arXiv:2212.00932, 2022. 3
- [55] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich.

- Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1–9, 2015. 14
- [56] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098, 3, 2024. 3
- [57] Yotam Tewel, Omer Sadik, Amit H. Bermano, and Daniel Cohen-Or. Key-locked rank one editing for text-to-image personalization. arXiv preprint arXiv:2305.01644, 2023. 2
- [58] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, Dhruv Nair, Sayak Paul, William Berman, Yiyi Xu, Steven Liu, and Thomas Wolf. Diffusers: State-of-the-art diffusion models. https: //github.com/huggingface/diffusers,

2022. 13

- [59] Haofan Wang, Matteo Spinelli, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards style-preserving in text-to-image generation. arXiv preprint arXiv:2404.02733, 2024. 3
- [60] Kuan-Chieh Wang, Daniil Ostashev, Yuwei Fang, Sergey Tulyakov, and Kfir Aberman. Moa: Mixtureof-attention for subject-context disentanglement in personalized image generation. arXiv preprint arXiv:2404.11565, 2024. 2
- [61] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, Anthony Chen, Huaxia Li, Xu Tang, and Yao Hu. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024. 2
- [62] Chen Wei, Karttikeya Mangalam, Po-Yao Huang, Yanghao Li, Haoqi Fan, Hu Xu, Huiyu Wang, Cihang Xie, Alan Yuille, and Christoph Feichtenhofer. Diffusion models as masked autoencoders. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16284–16294, 2023. 2
- [63] Guangxuan Xiao, Tianwei Yin, William T. Freeman, Fr´edo Durand, and Song Han. Fastcomposer: Tuningfree multi-subject image generation with localized attention. International Journal of Computer Vision,

2024. 2

- [64] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. Sana: Efficient highresolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629, 2024. 2, 3
- [65] Shaoan Xie, Yang Zhao, Zhisheng Xiao, Kelvin CK Chan, Yandong Li, Yanwu Xu, Kun Zhang, and Tingbo Hou. Dreaminpainter: Text-guided subjectdriven image inpainting with diffusion models. arXiv preprint arXiv:2312.03771, 2023. 3

- [66] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18381–18391, 2023. 3
- [67] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023. 2
- [68] Ge Yuan, Xiaodong Cun, Yong Zhang, Maomao Li, Chenyang Qi, Xintao Wang, Ying Shan, and Huicheng Zheng. Inserting anybody in diffusion models via celeb basis. arXiv preprint arXiv:2306.00926, 2023. 2
- [69] Ziyang Yuan, Mingdeng Cao, Xintao Wang, Zhongang Qi, Chun Yuan, and Ying Shan. Customnet: Zero-shot object customization with variableviewpoints in text-to-image diffusion models. arXiv preprint arXiv:2310.19784, 2023. 3
- [70] Bo Zhang, Yuxuan Duan, Jun Lan, Yan Hong, Huijia Zhu, Weiqiang Wang, and Li Niu. Controlcom: Controllable image composition using diffusion model. arXiv preprint arXiv:2308.10040, 2023. 3
- [71] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 6

In this supplementary material, we present additional experiment results. The supplementary comprises the following subsections:

- 1. Sec. A, details the inversion method we used for image editing.
- 2. Sec. B, details about the user study.
- 3. Sec. C, details about early-stopping method used in our experiments.
- 4. Sec. D, details about the implementation of the baselines.
- 5. Sec. E, details about adaptation of SISO to various bacbone models.
- 6. Sec. F, details about the attempt to use SISO for subjectdriven face swapping.

#### A. Diffusion Inversion

We employ ReNoise for diffusion inversion in our image editing solution. ReNoise hyperparameters include strength, calibrating noise addition, balancing reconstruction, and editability. High values harm reconstruction while improving the ability to edit, and low values hinder object changes but improve reconstruction. We tuned the default setting from 1 to 0.75 in all experiments. Although this setting slightly reduces editing potential, subject-driven editing demands changes to the subject, not the background. Thus, this value empirically proved optimal for both reconstruction and subject editing without altering the background.

#### B. User Study

According to the task, workers in Amazon MTurk were presented with a subject image, a condition (a prompt or an input image), and two generated images - one from SISO and the other from the baseline. The study was conducted on 100 images from the benchmark, with five workers rated each image. The method used for the study was Twoalternative forced choice, where raters must choose the preferred output between two options. In our case, the workers were presented three questions per image. Each question requested the worker to choose between two generated images (the order between the generated images was randomly picked). For subject-driven image generation, the questions tested the following criteria: (i) object similarity (what we refer in the paper as identity preservation), (ii) prompt alignment (what we refer as prompt adherence) and (iii) naturalness. See Fig. 7 for illustration of the user study interface.

#### C. Early Stopping

SISO generates a well-formed image at each iteration, rather than noisy latent. This enables using the method in an interactive manner. One option is to display images from

[Figure 85]

Figure 7. Illustration of the user study interface for Subject-driven image generation task.

all iterations and stop the optimization process when a satisfactory result is obtained. To achieve a fully automated process, we used a simple early-stopping strategy, where the process ends if the loss has not improved by x percent on the last n iterations. Specifically, we set x = 3 and n = 7 in all of our experiments, both for generation and editing.

#### D. Baselines

Here, we describe how we implemented the baselines used in the paper.

Subject-driven image generation. We compared our method against three baselines: (i) DreamBooth, which fine-tunes the diffusion model parameters according to a set of reference images. We used the code given in Diffusers [58] library for all different base models (SDXL, FLUX, and Sana). (ii) AttnDreamBooth, which improves on DreamBooth with a three-stage process, optimizing a textual embedding, cross-attention layers, and the U-Net. (iii) ClassDiffusion, which utilizes a semantic preservation loss. For both AttnDreamBooth and ClassDiffusion we used the official implementation published by the authors, using their default hyper-parameters.

Subject-driven image editing. We compared our method with two baselines: (i) SwapAnything, which employs masked latent blending and appearance adaptation. (ii)

TIGIC, a training-free technique that uses an attentionblending strategy during the denoising process. TIGIC was initially designed for a subject insertion, where the user wants to insert the subject to an empty area in the input image. To adapt to the subject replacement task, we used a state-of-the-art inpainting model (LaMa2) to remove the original object and then applied TIGIC. For both methods, we used the official implementation published by the authors, using their default hyper-parameters.

#### E. Adaptation to Various Backbone Models

A key advantage of SISO is its ability to be used with different backbone models with limited adaptation. In this section, we will describe the main differences in implementation between the different backbones we used (SDXLTurbo, FLUX schnell, Sana). First, SDXL-Turbo and FLUX schnell are distilled versions, meaning that they generate images using a small number of steps (1-4). Sana, on the other hand, does not have a distilled version and requires 20 steps to generate a high-quality image. We found that when using distilled versions, backpropogating through the final denoising step is sufficient. However, when using a non-distilled version, like Sana, it may be beneficial to backpropagate through more than one denoising step. Specifically, we set the number of steps to backpropogate through to 3. Also, even when using a distilled version, the number of denoising steps used in each iteration may be important, and different models behave differently in this context. We will denote this number as t. SDXL-Turbo is less noisy to different values of t, but FLUX schnell showed a significant difference when using various values of t. More specifically, setting t >= 2 resulted in low-quality generated images, even when trying to backpropogate through more denoising steps (see Fig. 8). However, FLUX schnell generates blurred images when used with one denoising step. A naive approach to overcome the blurriness is to use a model trained for up-scaling resolution. But this requires loading another model and may complicate the process. We solved the issue using the training simplification (Sec. 3.3 in the paper). Although the weights were optimized using t = 1, they can be used in inference with different values of t, thus producing high-quality images.

#### F. Subject-driven Face Swapping

A natural use-case for SISO is subject-driven face swapping. We tried to adapt our method to this task by using a different feature extractor more suitable for face recognition. Specifically, we employed InceptionResnet [55], using the implementation from pytorch-facenet library3). While

- 2https://github.com/advimman/lama
- 3https://github.com/timesler/facenet-pytorch

Subject Image Output 1 Output 2

[Figure 86]

[Figure 87]

[Figure 88]

→

[Figure 89]

[Figure 90]

[Figure 91]

→

Figure 8. Optimizing on FLUX schnell using four denoising steps results in low quality images.

this direction has a potential, it did not show satisfactory results (see Fig. 9).

Subject Image Input Image Output

[Figure 92]

[Figure 93]

[Figure 94]

→

[Figure 95]

[Figure 96]

[Figure 97]

→

Figure 9. Results for subject-driven face swapping.

Subject Image

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

... on a wooden counter ... on a finger ... in a box .. on a glass table

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

... in the beach ... as a pirate ... in a gondolla ... as a dj

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

... in the bedroom ... in the kitchen ... in the living room ... in a store

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

... on an armchair ... on a shelf ... on a couch ... on a fence

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

... in Coachella ... in Fuji mountain ... in the beach ... in Paris

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

... on a desk ... in the kitchen ... on a nightstand ... on a book Figure 10. More Qualitative results on Subject Driven Image Generation

Subject Image Input Image Output Input Image Output

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

→

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

→

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

→

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

→

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

→

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

→

Figure 11. More qualitative results on Subject Driven Image Editing.

Subject Image

...inthebeach

##### →

[Figure 158]

...sleeping

##### →

[Figure 159]

glasstable

...ona

##### →

[Figure 160]

...onabed

##### →

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

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

10 20 30 35 42 50 100 120 seed value

- Figure 12. We show the stability of our method across eight seeds for Subject Driven Image Generation.

Subject Image Input Image

[Figure 193]

[Figure 194]

→

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

→

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

→

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

→

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

10 20 30 35 42 50 100 120 seed value

- Figure 13. We show the stability of our method across eight seeds for Subject Driven Image Editing.

