## BootPIG: Bootstrapping Zero-shot Personalized Image Generation Capabilities in Pretrained Diffusion Models

# arXiv:2401.13974v1[cs.CV]25Jan2024

Senthil Purushwalkam* Akash Gokul* Shafiq Joty Nikhil Naik Salesforce AI Research

{spurushwalkam,agokul,sjoty,nnaik}@salesforce.com

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

| |
|---|
| |
| |
| |
| |

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Figure 1. Existing text-to-image models demonstrate exceptional image synthesis capabilities. However, they fail to "personalize" generations according to a specific subject. BootPIG (Ours) enables zero-shot subject-driven generation through a bootstrapped training process that uses images synthesized by the text-to-image model (Bottom). BootPIG trained text-to-image models can synthesize novel scenes containing the input subject without test-time finetuning while maintaining high fidelity to the prompt and subject.

### Abstract

Recent text-to-image generation models have demonstrated incredible success in generating images that faithfully follow input prompts. However, the requirement of using words to describe a desired concept provides limited control over the appearance of the generated concepts. In this work, we address this shortcoming by proposing an approach to enable personalization capabilities in existing text-to-image diffusion models. We propose a novel architecture (BootPIG) that allows a user to provide reference images of an object in order to guide the appearance of a concept in the generated images.

The proposed BootPIG architecture makes minimal modifications to a pretrained text-to-image diffusion model and

utilizes a separate UNet model to steer the generations toward the desired appearance. We introduce a training procedure that allows us to bootstrap personalization capabilities in the BootPIG architecture using data generated from pretrained text-to-image models, LLM chat agents, and image segmentation models. In contrast to existing methods that require several days of pretraining, the BootPIG architecture can be trained in approximately 1 hour. Experiments on the DreamBooth dataset demonstrate that BootPIG outperforms existing zero-shot methods while being comparable with test-time finetuning approaches. Through a user study, we validate the preference for BootPIG generations over existing methods both in maintaining fidelity to the reference object’s appearance and aligning with textual prompts.

### 1. Introduction

Over the last few years, research in generative modeling has pushed the boundaries of technological innovation and enabled new avenues of creative expression. In computer vision, text-to-image generation models [4, 5, 16, 30– 32, 49, 50] have showcased remarkable proficiency in generating high-fidelity images corresponding to novel captions. However, the ability to generate arbitrary images depicting a chosen caption has limited applications. Recent advancements in research have expanded upon these text-toimage models, leveraging their capabilities to solve more widely applicable tasks such as image inpainting[46], image editing[25, 44], style transfer[52], and even generation of 3D models from text[27] and images[24, 28, 47]. One such task with numerous applications is the problem of Personalized Image Generation.

Personalized Image Generation (also know as subjectdriven generation) is the ability to generate images of specific personal objects in various user-specified contexts. For instance, one may want to visualize what their pet would look like if it wore cowboy boots. Beyond such personal visualization experiments, this capability has the potential to serve as a versatile tool with applications ranging from personalized storytelling to interactive design.

More concretely, the Personalized Image Generation problem can be formulated as follows: given a few reference images depicting an object and a target caption, the goal is to generate an image that corresponds to the caption while faithfully capturing the object’s appearance. In recent research, several methods have been developed to leverage pretrained text-to-image models to accomplish this task. These approaches can be divided into two categories– Test-time Finetuning and Zero-Shot Inference methods. The former category involves methods that finetune the text-to-image model parameters or learn additional parameters to learn the subject’s appearancee. These methods often take a few minutes to a few hours to update parameters before personalized images are generated. The latter category includes methods that rely on pretraining the text-to-image models and do not involve test-time updates. While Zero-Shot Inference-based methods are more user-friendly and efficient for deployment, they fall short compared to Test-time Finetuning methods in terms of faithfulness to the reference object and textual prompt following capabilities.

In this work, we propose a novel solution to the Personalized Image Generation problem that provides the efficiency of Zero-Shot Inference methods while outperforming existing Test-time Finetuning methods. We introduce a novel architecture, named BootPIG, designed to enable personalized image generation capabilities in a pretrained text-to-image generation model. The BootPIG architecture comprises two replicas of the pretrained text-to-image model — one dedicated to extracting visual features from reference images

and another for the actual image generation process.

In order to train our proposed model, we present a novel procedure that does not directly utilize any human curated dataset. Instead, we bootstrap the personalization capability by learning from data synthetically generated using pretrained text-to-image generation models, state-of-the-art chat agents and image segmentation models. Furthermore, unlike existing Zero-Shot Inference methods that require several days of compute for pretraining, our proposed model can be trained in approximately 1 hour on 16 A100 GPUs.

Experiments on the DreamBooth dataset show that BootPIG generations outperform existing zero-shot methods while performing comparably to test-time finetuned approaches based on existing metrics. After conducting a user study, which asks users to compare generations from two approaches based on subject fidelity and prompt fidelity, we find that users generally prefer BootPIG generations over existing zero-shot and test-time finetuned methods.

The contributions of our work are summarized as follows:

- • We propose a novel architecture that enables zero-shot subject-driven generation while minimally modifying the architecture of the pretrained text-to-image model and requiring only 1 hour to train.
- • We demonstrate an effective bootstrapped learning procedure which does not require human-curated data and allows a pretrained text-to-image to use its own features to learn subject-driven generation.
- • BootPIG excels in zero-shot personalized image generation outperforming existing zero-shot method and test-time finetuned methods based on quantitative evaluations and user studies.

### 2. Related Work

#### 2.1. Text-to-Image Synthesis

Progress in generative models has led to breakthroughs in text-to-image synthesis. Existing text-to-image models [4, 5, 16, 30–32, 49, 50] are capable of generating highquality images in accordance with a given text prompt. These models fall into one of the following categories: diffusion models [31, 32, 36], autoregressive image models [30, 49, 50], non-autoregressive image models [3][4], and generative adversarial networks (GANs) [8][16]. While these models demonstrate exceptional image generation and fidelity to an input prompt, their outputs are constrained by the limitations of the text interface. This hinders their ability to generate images with a specific subject or follow additional controls from signals such as images. In this work, we extend the capability of pretrained text-to-image diffusion models to enable zero-shot subject-driven generation by introducing subject images as an additional source of control.

#### 2.2. Subject-Driven Text-to-Image Synthesis via Test-time Finetuning

Following the breakthroughs in text-to-image generation, many works extended these models to enable subject-driven generation. Subject-driven generation (personalized image generation) focuses on synthesizing an input subject in novel scenes. The seminal works of Textual Inversion [6] and DreamBooth [34] introduced test-time finetuning approaches that customized a text-to-image generative model given a few images of a subject. Textual Inversion enables subjectdriven generation by learning a unique text token for each subject. In contrast, DreamBooth finetunes the entire text-toimage backbone and demonstrates superior results in fidelity to the subject. Subsequent works [1, 7, 9, 10, 19, 35, 38, 43] extended these approaches by developing improved finetuning methods. In particular, CustomDiffusion [19] and ViCo [10] improve subject-driven generation by combining ideas from Textual Inversion and DreamBooth, by learning a specific text embedding for each concept while finetuning the cross-attention layers of the text-to-image model. Unlike these works, our method enables zero-shot subject-driven generation and avoids the need to train unique models for each subject.

#### 2.3. Zero-Shot Subject-Driven Text-to-Image Synthesis

Subject-driven generation approaches, such as DreamBooth and Textual Inversion, require hundreds, sometimes even thousands, of steps to learn the user-provided subject. Recent works have sought to avoid this tedious finetuning process by developing zero-shot methods for subject-driven generation. These methods typically pretrain image encoders across large datasets to learn image features that improve the generative model’s ability to render the subject. InstantBooth [37] introduces adapter layers in the text-to-image model and pretrains an adapter for each concept, e.g. cats. In comparison, our method is not restricted to learning single categories and, instead, can generalize to novel concepts. The authors of Jia et al. [15] propose to use frozen CLIP [29] image encoders and continue training the generative backbone to use CLIP image features. Similar to this work, our approach continues training the generative backbone. However, our approach does not introduce additional image encoders and does not require real data. Along the lines of pretraining image encoders, BLIP-Diffusion [21] pretrains a Q-Former [22] that learns image features which are aligned with text embeddings. Similarly, ELITE [45] pretrains two image encoders– one that learns image features that align with token embeddings, and a second encoder that learns image features that align with the text encoder’s features. In comparison, our approach avoids introducing a bottleneck in the form of text-aligned features and, instead, uses the features from the generative backbone to guide generation.

#### 2.4. Multimodal Controls for Text-to-Image Models

Concurrent to the work in subject-driven generation, many works have explored extending the conditional controls in text-to-image models. These methods introduce learnable components, such as modality-specific adapters [14, 17, 23, 26, 48, 53], to enable additional control via inputs such as depth maps or color palettes. In particular, ControlNet [51] demonstrated exceptional results in controllable generation by training a learnable copy of the generative model’s backbone to accept new input conditions. Our architecture shares similarities with the ControlNet architecture, as both works use a copy of the generative backbone. However unlike ControlNet, our work focuses on zero-shot subject-driven generation.

### 3. Method

In this section, we present the details of our proposed approach for enabling personalized image generation capabilities in existing latent diffusion models. Given a caption and a set of reference images that depict an object, the goal is to generate an image that follows the caption while ensuring that the appearance of the object matches that of the reference images. We accomplish this by introducing a novel architecture (BootPIG) that builds on top of existing diffusion models for text-to-image generation.

#### 3.1. Preliminary: Diffusion Models

Diffusion models [13, 39, 40] learn a data distribution by iteratively denoising samples from a Gaussian distribution. Given a data sample x, a noised version xt := αtx + σtϵ, is generated by applying noise, ϵ ∼ N(0,1) according to a timestep t ∈ {1,...,T}, where T is the total number of timesteps, and αt, σt control the noise schedule. The model, ϵθ, is trained using the mean squared error objective between the noise ϵ and the predicted noise ϵθ(xt,t,c), where c refers to a conditioning vector e.g. a text prompt (Equation 1).

Ex,c,ϵ,t∼U([0,T]) ∥ϵ − ϵθ(xt,t,c)∥22 (1)

In this work, we use pretrained text-to-image diffusion models and do not alter the diffusion objective during our training. Instead, we introduce additional image features into the architecture which are trained using Equation 1.

#### 3.2. BootPIG Model Architecture

We present an overview of the BootPIG architecture in Figure 2. The key idea of our proposed architecture is to inject the appearance of a reference object into the features of a pretrained text conditioned image diffusion model, such that the generated images imitate the reference object. In this work, we use Stable Diffusion [32] as our pretrained text-to-image diffusion model. Stable Diffusion is a Latent Diffusion Model [32] that uses a U-Net [33] architecture

consisting of Transformer[41] and Residual[11] blocks. Our proposed BootPIG architecture modifies the information processed by the self-attention layers in the Transformer blocks in order to control the appearance of the generated objects. Let the Stable Diffusion U-Net model be denoted by Uθ(x,c,t) where xt are noisy input latents, c is an input textual prompt and t is the timestep in the diffusion process. Injecting Reference Features A self-attention (SA) layer that receives a latent feature f ∈ Rn×d, performs the following operation:

##### SA(f) = Wo softmax q(f)k(f)T v(f) (2)

where q,k,v are linear mappings known as the query, key and value functions with parameters Wq,Wk,Wv ∈ Rd×d

′

respectively that project the features to a chosen dimension d′. Wo ∈ Rd

′xd projects the output back to the original dimension d. We propose to replace all the Self-Attention (SA) layers with an operation that we refer to as Reference Self-Attention (RSA), that allows us to inject reference features. The RSA operator takes as input the latent features f ∈ Rn×d and reference features of the same dimension fref ∈ Rn

ref×d, and performs the following operation:

RSA(f,fref) = Wo softmax q(f)

T v(f) v(fref)

k(f) k(fref)

(3)

where [:] indicates concatenation along the first dimension. Intuitively, the RSA operator facilitates the injection of reference features, allowing the diffusion model to “attend” to them in the computation of the output latent feature. Let us denote this new U-Net, referred to as Base U-Net, by

UθRSA(x,c,t,{fref(1),fref(2),...,fref(L)}) containing L RSA layers. For simplicity, we use the notation Fref to denote the set of L reference features. Note that the RSA operator does not introduce any new parameters and reuses the weight parameters Wo,Wq,Wk,Wv.

Extracting Reference Features Given a reference image Iref, we need to extract the appropriate features fref that can be passed to each RSA layer. We propose to extract features using a separate U-Net Uϕ(x,c,t), referred to as Reference U-Net, that follows the same architecture as the Base U-Net and is initialized with the same parameters (ϕ = θ). For a given t, we perform the forward diffusion process on the reference image Iref to compute the noised reference latents x′reft. We pass x′reft as input along with the textual prompt, and extract the features before the L− SA layers as Fref. This ensures that the extracted reference features have the appropriate dimensions and are compatible with the weights of the RSA layers.

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

| | |
|---|---|
| | |

| |
|---|

| |
|---|

[Figure 23]

[Figure 24]

| |
|---|

Figure 2. Model Architecture: We propose a novel architecture, that we refer to as BootPIG, for personalized image generation. The model comprises of two replicas of a latent diffusion model Reference UNet and Base UNet. The Reference UNet processes reference images to collect the features before each Self-Attention (SA) layer. The Base UNet’s SA layers are modified to Reference Self-Attention (RSA) layers that allow conditioning on extra features. Using the collected reference features as input, the Base UNet equipped with the RSA layers estimates the noise in the input to guide the image generation towards the reference objects.

#### 3.3. Training

The BootPIG architecture allows us to pass features of a reference image to the RSA layers. However, since the original diffusion model Uθ was not trained with RSA layers, we observe that the generated images are corrupted (see supplementary material) and do not correctly follow the input prompts. In order to rectify this, we propose to finetune the parameters of the Reference U-Net ϕ in order to extract better reference features and the parameters of the RSA layers (Wo,Wq,Wk,Wv) to better utilize the reference features.

Given a dataset of triplets containing a reference image, a textual prompt and a target image following the textual prompt while accurately depicting the reference object, we finetune the BootPIG architecture using the same objective as the original latent diffusion model (see Section 3.1). The Reference U-Net takes as input noisy VAE latents (noised according to timestep t) corresponding to the reference image as input, along with the timestep t and the target caption. The Base U-Net receives as input noisy VAE latents corresponding to the target image (similarly noised), the timestep t, the target caption and the reference features collected from the Reference U-Net. The parameters of the Reference UNet and the RSA layers are updated to accurately estimate the noise in the input latents (see Eq 1). In order to preserve the Base U-Net’s prompt following capabilities, we randomly drop the reference image’s features (with probability 0.15), thereby reverting the Base U-Net model back

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

- Figure 3. Synthetic Training Data: We propose an automated data generation pipeline to generate (reference image, target image, target caption) triplets for training BootPIG. The pipeline uses ChatGPT to generate captions, Stable Diffusion to generate images and the Segment Anything Model to segment the foreground which serves as the reference image.

to the SA based architecture. We provide a more detailed description of the training pipeline along with pseudocode in the supplementary material.

#### 3.4. Bootstrapping Training Data

Collecting a large curated source of training data for optimizing the BootPIG architecture is an expensive if not infeasible process. This would involve gathering paired reference and target images depicting the same object instance in different contexts. Instead, we propose a synthetic data generation pipeline that leverages the capabilities of existing pretrained Computer Vision and NLP models.

- In Figure 3, we provide an overview of the data genera-

tion pipeline. First, we utilize ChatGPT[2], a state-of-the-art conversational agent, to generate captions for potential target images. For each caption, we generate an image using Stable Diffusion[32]. We then use Segment Anything Model (SAM)[18], a state-of-the-art segmentation model, to extract a foreground mask corresponding to the main object in the caption. We treat the Stable Diffusion generated image as the target image, the foreground object pasted on a white background as the reference image and the ChatGPT generated caption as the textual prompt. While the reference image does not depict the object in an entirely different context, we observe empirically that this synthetic data is sufficient to learn personalization capabilities.

#### 3.5. Inference

During inference, at each timestep, we extract reference features Fref by passing a noised reference image, the target caption c and the timestep t to the Reference UNet. We use a classifier free guidance [12] strategy to estimate the noise in the noisy generation latents x′t using the Base UNet as:

ϵ = UθSA(x′t, t) + ηim(UθRSA(x′t, t, fref) − UθSA(x′t, t)) +ηtext+im ∗ (UθRSA(x′t, t, fref, c) − UθRSA(x′t, t, fref, c))

(4)

where the first U-Net term estimates the noise without any conditioning information, the second U-Net term estimates noise with just image information and the fourth U-Net term uses both images and captions.

The BootPIG architecture described so far takes as input a single reference image. This allows us to train the parameters of the model with synthesized (reference, target) image pairs. A common scenario with several applications is the case where multiple reference images are available to us. In order to handle this scenario, we now propose an inference procedure to leverage a trained BootPIG model and utilize appearance information from multiple reference images.

Let frefi be the reference feature for image i at a specific layer. At each RSA layer, we first compute the outputs without any reference feature (fref = f) and using each reference feature:

o = RSA(f,f) (5) oi = RSA(f,frefi) ∀i ∈ {1,2,...,k} (6)

For the output generated by each reference oi ∈ Rn×d, we compute the pixelwise norm of difference to the referenceless output o. We denote this by ni = ||oi − o|| ∈ Rn×1. The pixelwise softmax of these norms is then used to weight the contribution of each reference feature. Specifically, we compute the final output as:

pi = softmax(n1,n2,...,nk)[i] ∈ Rn×1 (7) omultiref = o +

pi ∗ (oi − o) (8)

i

Intuitively, at each pixel location, we want to use the reference features that make the largest contribution compared to the reference-less output.

### 4. Experiments

In this section, we describe the experimental setup and present results of our proposed model, BootPIG, for personalized image generation.

#### 4.1. Experimental Setup

Implementation Details The dataset synthesis pipeline discussed in 3.3 was used to synthesize 200000 (reference

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

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

- Figure 4. Qualitative Comparision: We provide visual comparisions of subject-driven generations from related methods such as BLIPDiffusion, ELITE, and DreamBooth. BootPIG exhibits high subject and prompt fidelity, outperforming related methods while avoiding test-time finetuning.

image, target image, target caption) triplets as training data. We provide the ChatGPT prompt used to generate these images in the Supplementary. We use the VAE and text encoder weights from the publicly available Stable Diffusion

version 2.1 [32]. The Base U-Net and Reference U-Net are both initialized from the pretrained Stable Diffusion U-Net weights. During training, we augment the reference images by randomly resizing the foreground, horizontally flipping

and moving the foreground to random locations. We train the model using 16 A100 GPUs, with a total batch size of 64 for 10k updates. We use a learning rate of 5e-6 for the Reference U-Net and 1e-6 for the RSA layers of the Base U-Net. The entire training takes 70 minutes, whereas BLIP-Diffusion requires 6 days with the same amount of compute. During inference, we use the UniPC Scheduler[53] and generate images using 50 inference steps. For classifier free guidance (as described in Section 3.5), we use ηim =

- 5.0 and ηtext+im = 7.5 in all the experiments.

Evaluation We evaluate our method on the DreamBooth dataset. The dataset contains 30 personalization subjects, each with multiple 3-6 reference images, and 25 novel prompts per subject. Unless specified, we use all available reference images for each generation. We evaluate the zeroshot performance of our method on this dataset using the metrics introduced by the authors of DreamBooth: CLIP-T, CLIP-I, and DINO. CLIP-T is used to measure the alignment between the generated image and the textual prompt. CLIP-I and DINO are used to evaluate the faithfulness of the generated image to the appearance of the reference object.

Additionally, we perform a user-study to compare our method to existing work. We perform a subject fidelity

- user study and a text alignment user study following the procedure outlined in ELITE [45]. In the subject fidelity study, users are provided the original subject image, and two generated images of that subject, generated by different methods (both methods use the same prompt). Users are asked to select which image better represents the original subject. In the text alignment study, users are provided the caption and two generated images, generated by different methods, for that caption and are asked to select which image better represents the caption.

For comparisons with Textual Inversion and DreamBooth, we use the corresponding implementations in the diffusers package [42] to train on each of the 30 subjects. We report CustomDiffusion’s evaluation metrics using the results reproduced in Wei et al. [45]. For zero-shot methods, such as BLIP-Diffusion and ELITE, we use their official pretrained models to generate images for each subject, following their respective inference procedures.

#### 4.2. Qualitative Comparisons

- In Figure 4, we provide qualitative comparisons between BootPIG and several personalized image generation methods. In comparison to zero-shot methods such as BLIP-Diffusion and ELITE, BootPIG exhibits significantly higher fidelity to the original subject. This improved fidelity to the subject is seen at both the object-level and at the level of fine-grained details as seen in examples such as the dog (row 3) or the can (row 5). We attribute this improvement in subject alignment to our reference feature extraction method which, unlike

BLIP-Diffusion and ELITE, does not introduce a bottleneck by forcing image features to be aligned with the text features. Compared to DreamBooth, a test-time finetuned approach, we also see improvements in subject fidelity. For example, BootPIG better preserves the markings and colors of the cat’s fur (row 4) and the facial features, e.g. eyes, of the dog (row 3).

#### 4.3. Quantitative Comparisons

In Table 1, we present the main quantitative comparison of BootPIG against existing methods on the DreamBooth dataset. BootPIG outperforms all existing zero-shot methods in prompt fidelity (+1.1 CLIP-T) and subject fidelity (+0.8 CLIP-I, +2.4 DINO) metrics. Compared to test-time finetuned methods, BootPIG exhibits state-of-the-art performance in prompt fidelity (+0.6 CLIP-T) while performing comparably in subject fidelity.

Prompt Matching Subject Matching

100

|Rate<br><br>75| |
|---|---|
| | |
| | |
| | |
| | |
| | |
|Win<br><br>25<br><br>50| |
| | |
| | |
| | |
| | |
| | |

0

vs. BLIP-Diffusion vs. ELITE vs. Dreambooth

Figure 5. User Study: We report the win rate (% of users who favored BootPIG generations) against existing methods. We perform two studies per head-to-head comparision, one evaluating prompt fidelity and the other evaluating subject fidelity.

Table 1. Quantitative Comparisons: We report average metrics for subject fidelity (CLIP-I, DINO) and prompt fidelity (CLIP-T) on the DreamBooth dataset.

Method Zero-shot CLIP-T CLIP-I DINO

BLIP Diffusion [21] ✓ 30.0 77.9 59.4 ELITE [45] ✓ 25.5 76.2 65.2 ELITE (reproduced) ✓ 29.6 78.8 61.4 Ours ✓ 31.1 79.7 67.4

Textual Inversion [6] ✗ 25.5 78.0 56.9 DreamBooth [34] ✗ 30.5 80.3 66.8 CustomDiffusion [19] ✗ 24.5 80.1 69.5 BLIP Diffusion + FT ✗ 30.2 80.5 67.0

Reference Images - - 88.5 77.4

From the results of our user study, we find that users consistently prefer BootPIG generations over both zero-shot

[Figure 70]

[Figure 71]

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

[Figure 88]

[Figure 89]

Figure 6. Effect of Inference Strategy: We compare different strategies for pooling information from multiple reference images in the BootPIG architecture. Refer to Section 4.4 for more details.

- Table 2. Effect of training different components: We perform a quantitative study on the effect of updating or fixing different components of the BootPIG architecture during training. We observe that the optimal strategy is to train the entire Reference U-Net and only train the RSA layers of the Base U-Net. Refer to Section 4.4 for more details.

Trainable Parameters

Use Aug. CLIP-T CLIP-I DINO Base UNet Ref. UNet

All All ✓ OOM None All ✓ 30.6 72.8 43.8

All None ✓ 31.6 78.8 66.4 RSA None ✓ 30.9 78.2 64.9 RSA All ✗ 30.2 78.8 67.0

(Ours) RSA All ✓ 31.1 79.7 67.4

and test-time finetuned methods. Human evaluators find BootPIG generations to have significantly greater subject fidelity (69% win rate versus ELITE 65% win rate versus BLIP-Diffusion, and 62% win rate versus DreamBooth). Additionally, the user study demonstrates that generations from BootPIG exhibit higher fidelity to the caption than existing methods (60% win rate versus ELITE, 54% win rate versus BLIP-Diffusion, and 52% win rate versus DreamBooth). These results underscore the efficacy of our training method.

#### 4.4. Ablative Studies

In order to understand the effect of different implementation choices, we further conduct ablative studies using the BootPIG architecture. First, we study the effect of training/freezing different components of the BootPIG architecture during the training process. In Table 2, we present the results on the three evaluation metrics. As discussed in Section 3.3, for our best model, we train all the parameters of the Reference U-Net and only train the RSA layers of the Base U-Net. We

CLIP-T CLIP-Image DINO-Image

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

%RelativeChange

- 0

2

4

6

8

- 1 2 3 4 5

-2

Number of Reference Images

Figure 7. Effect of multiple references: We quantify the effect of using one or more reference images during inference. Increasing the number of reference images significantly improves subject fidelity (+7.23 DINO, +1.74 CLIP-I) while slightly worsening prompt fidelity (−1.03 CLIP-T)

find that the following a ControlNet style training approach, where only the Reference U-Net is trainable, leads to poor performance. We attribute this to the fact that the attention layers in the Base U-Net are not trained to handle features from multiple images. We also find that only training the RSA layers of the Base U-Net also leads to poor subject fidelity and text fidelity. This result highlight the importance of training the Reference U-Net to extract better reference features. Lastly, we find that finetuning both networks, is extremely memory intensive and hits our available memory limits even when training with small batch sizes. We also present results obtained after training our model without randomly augmenting the reference images (2nd last row). While we only see a minor drop in the subject fidelity metrics (CLIP-I, DINO), we observe in qualitative visualizations that the generated images are extremely corrupted (see supplementary material).

Next, we study the effect of the number of references used as input to BootPIG. Since each reference image provides more information about the subject’s appearance, the subject fidelity of an ideal personalized generation model should improve as the number of references increase. In Figure 7, we present the relative change in the three metrics compared to outputs generated using only one reference image. We observe that the our subject fidelity metrics (CLIP-Image, DINO) consistently increase as the number of references increases. This highlights the efficacy of our proposed inference strategy that accounts for multiple reference images (introduced in Section 3.5).

Finally, we study different the efficacy of different inference strategies to handle multiple reference images with the BootPIG architecture. In Figure 6, we present personalized generations on a few subjects using different inference strategies. “Concat” refers to concatenating the features from all reference images into one sequence before passing to the RSA layer. “Average” refers to averaging the outputs of an

RSA layer after passing each reference image’s feature in individually i.e. K1 Ki=1 RSA(f,frefi). We observe that “Concat” often leads to generations where the object is repeated multiple times. This is expected since the model is receiving multiple copies of the same feature as input. On the other hand, the “Average” produces good quality images but smoothens out all the details in the objects. Our proposed inference strategy avoids these issues and generally produces higher quality personalized generations.

### 5. Limitations

BootPIG possesses many of the same failures as related methods. Specifically, in many instances, BootPIG generation may fail to render fine-grained features of the subject and struggle to accurately adhere to the prompt. Some of these failure cases highlight the need to learn stronger fine-grained features, while other failure cases, such as prompt fidelity in certain conditions and text rendering (see Fig 4 row 5), are limitations inherited from the underlying text-to-image model. We provide further illustrations of failure cases in the Supplementary Material. Additionally, the weaknesses and the biases, including harmful stereotypes, of the underlying generative model will be perpetuated by BootPIG generations. Subject-driven generation also opens the possibility of generating unwanted images of individuals without their consent. We urge that users of this technology are mindful of these implications and recommend that they use such technology responsibly.

### 6. Discussion

In this paper, we presented a bootstrapped training procedure that enables a text-to-image model to synthesize userprovided subjects in novel scenes without subject-specific finetuning. Our method, BootPIG, does not require real data or additional pretrained image encoders. Instead, it

- uses images generated by the text-to-image models, and utilizes a copy of the generative backbone to capture the appearance of user-provided subjects. The proposed model can be trained in approximately 1 hour, outperforms similar zero-shot inference methods and performs comparably to test-time finetuning methods. We believe that bootstrapped training of pretrained text-to-image models can be a promising paradigm for learning new capabilities and unlocking other modalities of controlled image generation. References

- [1] Moab Arar, Rinon Gal, Yuval Atzmon, Gal Chechik, Daniel Cohen-Or, Ariel Shamir, and Amit H Bermano. Domainagnostic tuning-encoder for fast personalization of text-toimage models. arXiv preprint arXiv:2307.06925, 2023. 3
- [2] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan,

- Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, T. J. Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeff Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. ArXiv, abs/2005.14165, 2020. 5
- [3] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11315–11325, 2022. 2
- [4] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-toimage generation via masked generative transformers. arXiv preprint arXiv:2301.00704, 2023. 2
- [5] Ming Ding, Wendi Zheng, Wenyi Hong, and Jie Tang. Cogview2: Faster and better text-to-image generation via hierarchical transformers. Advances in Neural Information Processing Systems, 35:16890–16902, 2022. 2
- [6] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618,

2022. 3, 7

- [7] Rinon Gal, Moab Arar, Yuval Atzmon, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Encoder-based domain tuning for fast personalization of text-to-image models. ACM Transactions on Graphics (TOG), 42(4):1–13, 2023. 3
- [8] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020. 2
- [9] Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, et al. Mix-of-show: Decentralized lowrank adaptation for multi-concept customization of diffusion models. arXiv preprint arXiv:2305.18292, 2023. 3
- [10] Shaozhe Hao, Kai Han, Shihao Zhao, and Kwan-Yee K Wong. Vico: Detail-preserving visual condition for personalized textto-image generation. arXiv preprint arXiv:2306.00971, 2023. 3, 14, 15
- [11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 4
- [12] Jonathan Ho. Classifier-free diffusion guidance. ArXiv, abs/2207.12598, 2022. 5
- [13] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3
- [14] Minghui Hu, Jianbin Zheng, Daqing Liu, Chuanxia Zheng, Chaoyue Wang, Dacheng Tao, and Tat-Jen Cham. Cocktail: Mixing multi-modality control for text-conditional image generation. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 3

- [15] Xuhui Jia, Yang Zhao, Kelvin CK Chan, Yandong Li, Han Zhang, Boqing Gong, Tingbo Hou, Huisheng Wang, and Yu-Chuan Su. Taming encoder for zero fine-tuning image customization with text-to-image diffusion models. arXiv preprint arXiv:2304.02642, 2023. 3
- [16] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10124–10134, 2023. 2
- [17] Sungnyun Kim, Junsoo Lee, Kibeom Hong, Daesik Kim, and Namhyuk Ahn. Diffblender: Scalable and composable multimodal text-to-image diffusion models. arXiv preprint arXiv:2305.15194, 2023. 3
- [18] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross B. Girshick. Segment anything. ArXiv, abs/2304.02643,

2023. 5, 12

- [19] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of textto-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1931–1941, 2023. 3, 7, 14, 15
- [20] Min Seok Lee, WooSeok Shin, and Sung Won Han. Tracer: Extreme attention guided salient object tracing network (student abstract). In Proceedings of the AAAI Conference on Artificial Intelligence, pages 12993–12994, 2022. 13
- [21] Dongxu Li, Junnan Li, and Steven CH Hoi. Blipdiffusion: Pre-trained subject representation for controllable text-to-image generation and editing. arXiv preprint arXiv:2305.14720, 2023. 3, 7
- [22] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. 3
- [23] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22511–22521, 2023. 3
- [24] Luke Melas-Kyriazi, Iro Laina, Christian Rupprecht, and Andrea Vedaldi. Realfusion: 360deg reconstruction of any object from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8446–8455, 2023. 2
- [25] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 2
- [26] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023. 3
- [27] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 2

- [28] Senthil Purushwalkam and Nikhil Naik. Conrad: Image constrained radiance fields for 3d generation from a single image. arXiv preprint arXiv:2311.05230, 2023. 2
- [29] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 3
- [30] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 8821–8831. PMLR, 2021.

- 2

[31] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):

- 3, 2022. 2

- [32] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 3, 5, 6, 12
- [33] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pages 234–241. Springer, 2015. 3
- [34] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500–22510,

2023. 3, 7, 14, 15

- [35] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models. arXiv preprint arXiv:2307.06949, 2023. 3
- [36] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 2
- [37] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without test-time finetuning. arXiv preprint arXiv:2304.03411, 2023. 3
- [38] James Seale Smith, Yen-Chang Hsu, Lingyu Zhang, Ting Hua, Zsolt Kira, Yilin Shen, and Hongxia Jin. Continual diffusion: Continual customization of text-to-image diffusion with c-lora. arXiv preprint arXiv:2304.06027, 2023. 3
- [39] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International confer-

- ence on machine learning, pages 2256–2265. PMLR, 2015. 3
- [40] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019. 3
- [41] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 4
- [42] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, and Thomas Wolf. Diffusers: State-of-the-art diffusion models. 7, 14
- [43] Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, and Kfir Aberman. p+: Extended textual conditioning in text-to-image generation. arXiv preprint arXiv:2303.09522, 2023. 3
- [44] Bram Wallace, Akash Gokul, and Nikhil Naik. Edict: Exact diffusion inversion via coupled transformations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22532–22541, 2023. 2
- [45] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. arXiv preprint arXiv:2302.13848, 2023. 3, 7
- [46] Shaoan Xie, Zhifei Zhang, Zhe Lin, Tobias Hinz, and Kun Zhang. Smartbrush: Text and shape guided object inpainting with diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22428–22437, 2023. 2
- [47] Dejia Xu, Yifan Jiang, Peihao Wang, Zhiwen Fan, Yi Wang, and Zhangyang Wang. Neurallift-360: Lifting an in-the-wild 2d photo to a 3d object with 360deg views. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4479–4489, 2023. 2
- [48] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 3

- [49] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022. 2
- [50] Lili Yu, Bowen Shi, Ramakanth Pasunuru, Benjamin Muller, Olga Golovneva, Tianlu Wang, Arun Babu, Binh Tang, Brian Karrer, Shelly Sheynin, et al. Scaling autoregressive multimodal models: Pretraining and instruction tuning. arXiv preprint arXiv:2309.02591, 2023. 2
- [51] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 3
- [52] Yuxin Zhang, Nisha Huang, Fan Tang, Haibin Huang, Chongyang Ma, Weiming Dong, and Changsheng Xu. Inversion-based style transfer with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10146–10156, 2023. 2

[53] Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, and Kwan-Yee K Wong. Unicontrolnet: All-in-one control to text-to-image diffusion models. arXiv preprint arXiv:2305.16322, 2023. 3, 7

### A. Additional Details: Data Generation Pipeline

As described in the main text, we utilize state-of-the-art pretrained models to synthesize training data. First, we use ChatGPT to generate captions for potential images. We use the following prompt for generating captions:

Generate a caption for an image where there is one main object and possibly other secondary objects. The object needs to be finite and solid. The main object should not be things like lakes, fields, sky, etc.

You should only respond in the following format: Caption: A photo of a [object], [describe object positioning in scene] [describe scene]

Some examples of the generated captions are:

- • A photo of a red rose, nestled alone in a glass vase upon a lace table runner.
- • A photo of a vintage typewriter, resting on a weathered desk.
- • A photo of a faded Polaroid camera, lying on a sun-warmed picnic blanket.
- • A photo of a sewing machine, buzzing with creativity amidst colorful fabric swatches in a designer’s studio.
- • A photo of a delicate porcelain teacup, delicately placed on a lace doily.

For each of these captions, we generate an image using Stable Diffusion 2.1[32] which is used as the target image. The formatting of the captions allows us to automatically extract the object categories by parsing the [object] part of the caption. For example, for the above captions, the corresponding objects categories are: red rose, vintage typewriter,

faded Polaroid camera, sewing machine, delicate porcelain teacup. These object categories are used as input the Segment Anything Model[18] to extract foreground images. These foreground images pasted on a white background are used as reference images.

### B. Additional Details: Training and Inference

In this section, we present additional implementation details for training and inference with the BootPIG architecture. The source code to reproduce our experiments will be released with the next update to the paper.

Training In Algorithm 1, we present a pseudocode describing the training pipeline. For optimizing the parameters, we use the AdamW algorithm with learning rate 5e-6, betas (0.9, 0.999), weight decay 1e-2 and epsilon 1e-8. The norm of the gradients is clipped to 1.0.

Algorithm 1: Training algorithm for BootPIG.

- 1 Initialize:

- 2 Uθ, Uϕ = Stable Diffusion UNet

- 3 VAE = Stable Diffusion VAE

- 4 noise_scheduler = Stable Diffusion Noise Scheduler

- 5 UθRSA = Replace SA with RSA in Uθ

- 6 dataset = synthetic dataset containing 200k reference, target images and captions

- 7 optimizer = AdamW( (RSA layers of Uθ) + (All layers of Uϕ) )

- 8

- 9 def collect_features(model, latents, timestep, caption):

- 10 Perform forward pass model(latents, timestep, caption)

- 11 features = Collected features before each SA layer

- 12 return features

- 13

- 14 count = 0

- 15 for reference, target, caption in dataset:

- 16 ref = VAE(reference)

- 17 tar = VAE(target)

- 18 t = random.randint(0, 1000)

- 19

- 20 noise = randn_like(ref)

- 21 noisedref = noise_sceduler.add_noise(ref, noise, t)

- 22 noise = randn_like(tar)

- 23 noisedtar = noise_sceduler.add_noise(tar, noise, t)

- 24

- 25 Fr = collect_features(Uϕ, noisedref, t, caption)

- 26 if random.rand()<0.15:

- 27 caption = ""

- 28

- 29 predtar = UθRSA(noisedtar, t, caption, Fr) # Use corresponding features from F_r in RSA layers

- 30

- 31 loss = ((noisedtar - predtar)**2).mean()

- 32 loss.backward()

- 33 count += 1

- 34 if count %

- 35 optimizer.step()

- 36 optimizer.zero_grad()

- 37

Inference During inference, we first perform foreground segmentation on each reference image using the TRACER[20] model implemented in the carvekit python package. Then we use the Reference UNet in the BootPIG architecture to collect reference features for each reference image. We then perform denoising using the Base UNet with the pooling strategy described in Section 3.5.

### C. Additional Comparisons to Test-Time Finetuning Methods

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

[Figure 124]

Figure 8. Qualitative Comparision: We provide additional visual comparisons of Custom Diffusion (reproduced), ViCo (reproduced), and BootPIG. In comparision, BootPIG generations demonstrate higher prompt fidelity (rows 2-4) and higher subject fidelity (rows 2, 3, 5).

We compare our proposed method, BootPIG, with two recent test-time finetuning approaches: CustomDiffusion [19] and ViCo [10]. For both approaches, we train subject-specific models using each method on the DreamBooth [34] dataset. We use the diffusers [42] implementation of CustomDiffusion and the official code provided by the authors of ViCo. We find

that quantitatively (Table 3), BootPIG outperforms CustomDiffusion (+2.2 CLIP-T, +1.9 DINO) and ViCo (+3.1 CLIP-T, and +2.0 CLIP-I & +8.9 DINO) despite being zero-shot. These results highlight the benefits of BootPIG as the method outperforms existing methods in prompt and subject fidelity, both quantitatively (CLIP-T, DINO) and qualitatively (Figure 8).

We present additional qualitative comparisions with CustomDiffusion and ViCo on the DreamBooth dataset. Qualitative results, provided in Figure 8, demonstrate the improvements in subject fidelity and prompt fidelity when using BootPIG. In comparision to CustomDiffusion, we find that BootPIG provides exhibits greater fidelity to the prompt. Examples of this improvement include: in row 2, CustomDiffusion’s generation is missing the white rug; in row 3, CustomDiffusion adds an unnecessary pink bowtie to the subject; and in row 5 CustomDiffusion fails to place the stuffed animal on a dirt road. We also find noticeable improvements in subject fidelity when comparing BootPIG to CustomDiffusion (shape of the duck in row 1, the length of the fringes on the boot in row 2, and color and thickness of dog’s fur in row 3). Similarly, BootPIG visually outperforms ViCo (e.g. fails to match the details of the subject in rows 2, 3, 5 and does not follow the prompt in row 4).

Method Zero-shot CLIP-T CLIP-I DINO

Ours ✓ 31.1 79.7 67.4 DreamBooth [34] ✗ 30.5 80.3 66.8 CustomDiffusion[19] (reproduced) ✗ 28.9 80.6 65.5 ViCo [10] (reproduced) ✗ 28.0 77.7 58.5 Reference Images - - 88.5 77.4

- Table 3. Quantitative Comparisons: We provide average metrics for subject fidelity (CLIP-I, DINO) and prompt fidelity (CLIP-T) on the DreamBooth dataset.

### D. Additional Qualitative Results

- In Figure 9, we present additional visualizations using our proposed method on several DreamBooth dataset images. We observe that across different subjects, our proposed method is able to successfully maintain subject fidelity while accurately following input prompts.

[Figure 125]

[Figure 126]

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

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

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

[Figure 162]

[Figure 163]

[Figure 164]

Figure 9. Additional Visualizations: We present additional qualitative results on the DreamBooth dataset using our proposed method.

### E. Additional Qualitative Ablation

- In Figure 10, we present additional qualitative results to ablate the effect of training the BootPIG architecture and the importance of augmenting the reference images during training. At initialization, we observe that the BootPIG architecture can already copy some aspects of the subject’s appearance. However, both the subject and prompt fidelity is limited, and the

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

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

Figure 10. Qualitative Ablation:

overall quality of generated images is poor. In Section 4.4 of the main text, we quantitatively show that synthetically augmenting the reference images leads to improvements in final performance. Specifically, we random horizontal flipping, randomly resize the foreground in the images and place the foreground at random locations in the white background images. We observed that the subject fidelity metrics (CLIP-I, DINO-I) demonstrate minor improvements when BootPIG is trained with such augmented images. However, in practice, we observed that training without augmentations led to significantly worse subject preservation and overall image quality. In Figure 10 Column 2 & 3, we demonstrate this by qualitatively comparing the effect of the synthetic augmentations.

### F. Subject-Driven Inpainting

We additionally qualiatively explore the zero-shot capabilities of BootPIG for subject-driven inpainting. Given an input image, prompt, and binary segmentation mask, we use BootPIG to inpaint a given subject into the masked region of the image. We provide examples of BootPIG inpainted images in Figure 11. Similar to the results in the text-to-image experiments, we find that BootPIG can preserve the details of the target subject in the new scene (e.g. face and fur of the dog in row 1, the dog face on the backpack in row 2, and the face and wing of the duck in row 3). We note that BootPIG was not trained using an inpainting objective, thus limiting its capabilities as an image editing model. As a result, we find that BootPIG can struggle to inpaint target subjects when they are drastically different, in terms of size and/or shape, to the existing subject in the scene.

[Figure 200]

Figure 11. Zero-shot Subject-Driven Inpainting: We provide visual comparisons of BootPIG inpainted images according to a given subject. Despite not being trained using an inpainting objective, we find that BootPIG is able to accurately render the subject within the original scene.

### G. Failures Cases

The BootPIG architecture does occasionally fail on certain (subject, caption) pairs. In Figure 12, we present examples of some failed generations. We observe that a commonly occurring failure is in the case where the prompt attempts to modify the appearance of the subject. For example, prompts of the form "a cube shaped [object]" attempts to modify the shape of reference [object] to a cube. Since the BootPIG architecture was trained on synthetic data where the target and reference appearance accurately matches, it fails to capture such modifications to the appearance during inference. We also observed that the BootPIG architecture occasionally generates images depicting multiple instances of the reference object. Furthermore, as with any existing personalization method, our proposed model also occasionally fails at accurately capturing the appearance of the object (see Figure 12 row 3, column 2). As a guideline for practitioners, we find that regenerating the image with a different sample of Gaussian noise often resolves the issue.

[Figure 201]

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

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

Figure 12. Failure Cases: We present examples of a few failure modes demonstrated by the BootPIG architecture.

