# arXiv:2407.08737v1[cs.CV]11Jul2024

## Video Diffusion Alignment via Reward Gradients

#### Mihir Prabhudesai∗ Russell Mendonca∗ Zheyang Qin∗ Katerina Fragkiadaki Deepak Pathak

###### Carnegie Mellon University

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

“A shark playing chess.”

“A raccoon drumming on bongos under a starry night sky.”

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

“A child painting in an art class, using watercolors and a brush on paper.”

“A fairy tends to enchanted, glowing flowers.”

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

“A joyful dog playing in the snow, leaving paw prints and trying to catch snowflakes on its nose.”

“A snow princess stands on the balcony of her ice castle, her hair adorned with delicate snowflakes, overlooking her serene realm.”

Figure 1: Generations from video diffusion models after adaptation with VADER using reward functions for aesthetics and text-image alignment. More visualization results are available at https: //vader-vid.github.io

### Abstract

We have made significant progress towards building foundational video diffusion models. As these models are trained using large-scale unsupervised data, it has become crucial to adapt these models to specific downstream tasks. Adapting these models via supervised fine-tuning requires collecting target datasets of videos, which is challenging and tedious. In this work, we utilize pre-trained reward models that are learned via preferences on top of powerful vision discriminative models to adapt video diffusion models. These models contain dense gradient information with respect to generated RGB pixels, which is critical to efficient learning in complex search spaces, such as videos. We show that backpropagating gradients from these reward models to a video diffusion model can allow for compute and sample efficient alignment. We show results across a variety of reward models and video diffusion models, demonstrating that our approach can learn much more efficiently in terms of reward queries and computation than prior gradient-free approaches. Our code, model weights, and more visualization are available at https://vader-vid.github.io.

∗Equal Contribution

Preprint. Under review.

### 1 Introduction

We would like to build systems capable of generating videos for a wide array of applications, ranging from movie production, creative story-boarding, on-demand entertainment, AR/VR content generation, and planning for robotics. The most common current approach involves training foundational video diffusion models on extensive web-scale datasets. However, this strategy, while crucial, mainly produces videos that resemble typical online content, featuring dull colors, suboptimal camera angles, and inadequate alignment between text and video content.

Contrast this with the needs of an animator who wishes to bring a storyboard to life based on a script and a few preliminary sketches. Such creators are looking for output that not only adheres closely to the provided text but also maintains temporal consistency and showcases desirable camera perspectives. Relying on general-purpose generative models may not suffice to meet these specific requirements. This discrepancy stems from the fact that large-scale diffusion models are generally trained on a broad spectrum of internet videos, which does not guarantee their efficacy for particular applications. Training these models to maximize likelihood across a vast dataset does not necessarily translate to high-quality performance for specialized tasks. Moreover, the internet is a mixed bag when it comes to content quality, and models trained to maximize likelihood might inadvertently replicate lower-quality aspects of the data. This leads us to the question: How can we tailor diffusion models to produce videos that excel in task-specific objectives, ensuring they are well-aligned with the desired outcomes?

The conventional approach to aligning generative models in the language and image domains begins with supervised fine-tuning [22, 4]. This involves collecting a target dataset that contains expected behaviors, followed by fine-tuning the generative model on this dataset. Applying this strategy to video generation, however, presents a significantly greater challenge. It requires obtaining a dataset of target videos, a task that is not only more costly and laborious than similar endeavors in language or image domains, but also significantly more complex. Furthermore, even if we were able to collect a video target dataset, the process would have to be repeated for every new video task, making it prohibitively expensive. Is there a different source of signal we can use for aligning video diffusion, instead of trying to collect a target dataset of desired videos?

Reward models play a crucial role [24, 32, 17] in aligning image and text generations. These models are generally built on top of powerful image or text discriminative models such as CLIP or BERT [21, 1, 29]. To use them as reward models, people either fine-tune them via small amounts of human preferences data [24] or use them directly without any fine-tuning; for instance, CLIP can be used to improve image-text alignment or object detectors can be used to remove or add objects in the images [20].

This begs the question, how should reward models be used to adapt the generation pipeline of diffusion models? There are two broad categories of approaches, those that utilize reward gradients [20, 6, 33], and others that use the reward only as a scalar feedback and instead rely on estimated policy gradients [2, 18]. It has been previously found that utilizing the reward gradient directly to update the model can be much more efficient in terms of the number of reward queries, since the reward gradient contains rich information of how the reward function is affected by the diffusion generation [20, 6]. However, in text-to-image generation space, reward gradient-free approaches are still dominant [23], since these methods can be easily trained within 24 hours and the efficiency gains of leveraging reward gradients are not significant.

In this work, we find that as we increase the dimensionality of generation i.e transition from image to video, the gap between the reward gradient and policy gradient based approaches increases. This is because of the additional amount and increased specificity of feedback that is backpropagated to the model. For reward gradient based approaches, the feedback gradients linearly scale with respect to the generated resolution, as it yields distinct scalar feedback for each spatial and temporal dimension. In contrast, policy gradient methods receive a single scalar feedback for the entire video output. We test this hypothesis in Figure 4, where we find that the gap between reward gradient and policy gradient approaches increases as we increase the generated video resolution. We believe this makes it crucial to backpropagate reward gradient information for video diffusion alignment.

We propose VADER, an approach to adapt foundational video diffusion models using the gradients of reward models. VADER aligns various video diffusion models using a broad range of pre-trained vision models. Specifically, we show results of aligning text-to-video (VideoCrafter, OpenSora, and

ModelScope) and image-to-video (Stable Video Diffusion) diffusion models, while using reward models that were trained on tasks such as image aesthetics, image-text alignment, object detection, video-action-classification, and video masked autoencoding. Further, we suggest various tricks to improve memory usage which allow us to train VADER on a single GPU with 16GB of VRAM. We include qualitative visualizations that show VADER significantly improves upon the base model generations across various tasks. We also show that VADER achieves much higher performance than alternative alignment methods that do not utilize reward gradients, such as DPO or DDPO. Finally, we show that alignment using VADER can easily generalize to prompts that were not seen during training. Our code is available at https://vader-vid.github.io.

### 2 Related Work

Denoising diffusion models [26, 11] have made significant progress in generative capabilities across various modalities such as images, videos and 3D shapes [10, 12, 19]. These models are trained using large-scale unsupervised or weakly supervised datasets. This form of training results in them having capabilities that are very general; however, most end use-cases of these models have specific requirements, such as high-fidelity generation [24] or better text alignment [32].

To be suitable for these use-cases, models are often fine-tuned using likelihood [3, 4] or reward-based objectives [2, 20, 6, 33, 18, 7, 9]. Likelihood objectives are often difficult to scale, as they require access to the preferred behaviour datasets. Reward or preference based datasets on the other hand are much easier to collect as they require a human to simply provide preference or reward for the data generated by the generative model. Further, widely available pre-trained vision models can also be used as reward models, thus making it much easier to do reward fine-tuning [2, 20]. The standard approach for reward or preference based fine-tuning is to do reinforcement learning via policy gradients [2, 30]. For instance, the work of [18] does reward-weighted likelihood and the work of [2] applies PPO [25]. Recent works of [20, 6], find that instead of using policy gradients, directly backpropagating gradients from the reward model to diffusion process helps significantly with sample efficiency.

A recent method, DPO [22, 30], does not train an explicit reward model but instead directly optimizes on the human preference data. While this makes the pipeline much simpler, it doesn’t solve the sample inefficiency issue of policy gradient methods, as it still backpropagates a single scalar feedback for the entire video output.

While we have made significant progress in aligning image diffusion models, this has remained challenging for video diffusion models [3, 31]. In this work, we take up this challenging task. We find that naively applying prior techniques of image alignment [20, 6] to video diffusion can result in significant memory overheads. Further, we demonstrate how widely available image or video discriminative models can be used to align video diffusion models. Concurrent to our work, InstructVideo [34] also aligns video diffusion models via human preference; however, this method requires access to a dataset of videos. Such a dataset is difficult to obtain for each different task, and becomes difficult to scale especially to large numbers of tasks. In this work, we show that one can easily align video diffusion models using pre-trained reward models while not assuming access to any video dataset.

### 3 Background

Diffusion models have emerged as a powerful paradigm in the field of generative modeling. These models operate by modeling a data distribution through a sequential process of adding and removing noise.

The forward diffusion process transforms a data sample x into a completely noised state over a series of steps T. This process is defined by the following equation:

##### xt = √α¯tx + √1 − α¯tϵ, ϵ ∼ N(0,1), (1)

where ϵ represents noise drawn from a standard Gaussian distribution. Here, α¯t = ti=1 αi denotes the cumulative product of αi = 1 − βi, which indicates the proportion of the original data’s signal retained at each timestep t.

The reverse diffusion process reconstructs the original data sample from its noised version by progressively denoising it through a learned model. This model is represented by ϵθ(xt;t) and estimates the noise ϵ added at each timestep t.

Diffusion models can easily be extended for conditional generation. This is achieved by adding c as an input to the denoising model:

||ϵθ(√α¯txi + √1 − α¯tϵ,ci,t) − ϵ||2, (2)

1 |D′|

Ldiff(θ;D′) =

xi,ci∈D′

where D′ denotes a dataset consisting of image-conditiong pairs. This loss function minimizes the distance between the estimated noise and the actual noise, and aligns with the variational lower bound for log p(x|c).

To sample from the learned distribution pθ(x|c), one starts with a noise sample xT ∼ N(0,1) and iteratively applies the reverse diffusion process:

1 √αt

βt √1 − α¯t

ϵθ(xt,t,c) + σtz, z ∼ N(0,1), (3)

xt −

xt−1 =

The above formulation captures the essence of diffusion models, which highlights their ability to generate structured data from random noise.

### 4 VADER: Video Diffusion via Reward Gradients

Reverse Diffusion Send Gradients

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

[Figure 33]

[Figure 34]

Reward Model

Loss

∇θLoss

xT xt xt−1 x0

- Figure 2: VADER aligns various pre-trained video diffusion models by backpropagating gradients from the reward model, to efficiently adapt to specific tasks.

We present our approach for adapting video diffusion models to perform a specific task specified via a reward function R(.).

Given a video diffusion model pθ(.), dataset of contexts Dc, and a reward function R(.), we seek to maximize the following objective:

J(θ) = Ec∼D

c,x0∼pθ(x0|c)[R(x0,c)]

To learn efficiently, both in terms of the number of reward queries and compute time, we seek to utilize the gradient structure of the reward function, with respect to the weights θ of the diffusion model. This is applicable to all reward functions that are differentiable in nature. We compute the gradient ∇θR(x0,c) of these differentiable rewards, and use it to update the diffusion model weights θ. The gradient is given by :

T

∂R(x0,c) ∂xt ·

∂xt ∂θ

∇θR(x0,c) =

.

t=0

Algorithm 1 VADER Require: Diffusion Model weights θ Require: Reward function R(.) Require: Denoising Scheduler f

(eg - DDIM[27], EDM[14])

Require: Gradient cutoff step K

- 1: while training do
- 2: for t = T,..1 do
- 3: pred = ϵθ(xt,c,t)
- 4: if t > K then
- 5: pred = stop_grad(pred)
- 6: end if
- 7: xt−1 = f.step(pred, t, xt)
- 8: end for
- 9: g = ∇θR(x0,c)
- 10: θ ← θ − η ∗ g
- 11: end while

VADER is flexible in terms of the denoising schedule, we demonstrate results with DDIM [27] and EDM solver [14]. To prevent overoptimization, we utilize truncated backpropagation [28, 20, 6], where the gradient is back propagated only for K steps, where K < T, where T is the total diffusion timesteps. Using a smaller value of K also reduces the memory burden of having to backpropagate gradients, making training more feasible. We provide the pseudocode of the full training process in Algorithm 1. Next, we discuss the type of reward functions we consider for aligning video models.

Reward Models: Consider a diffusion model that takes conditioning vector c as input and generates a video x0 of length N, consisting of a series of images ik, for each timestep k from 0 to N. Then the objective function we maximize is as follows:

Jθ = Ec,i

0:N

[R([i0,i1...ik...iN−1],c)]

We use a broad range of reward functions for aligning video diffusion models. Below we list down the distinct types of reward functions we consider.

Image-Text Similarity Reward - The generations from the diffusion model correspond to the text provided by the user as input. To ensure that the video is aligned with the text provided, we can define a reward that measures the similarity between the generated video and the provided text. To take advantage of popular, large-scale image-text models such as CLIP[21], we can take the following approach. For the entire video to be well aligned, each of the individual frames of the video likely need to have high similarity with the context c. Given an image-context similarity model gimg, we have:

R([i0,i1...ik...iN−1],c) =

R(ik,c) =

gimg(ik,c)

k

k

Then, we have Jθ = Ek∈[0,N] [gimg(ik,c)], using linearity of expectation as in the image-alignment case. We conduct experiments using the HPS v2 [32] and PickScore [16] reward models for imagetext alignment. As the above objective only sits on individual images, it could potentially result in a collapse, where the predicted images are the exact same or temporally incoherent. However, we don’t find this to happen empirically, we think the initial pre-training sufficiently regularizes the fine-tuning process to prevent such cases.

Video-Text Similarity Reward - Instead of using per image similarity model gimg, it could be beneficial to evaluate the similarity between the whole video and the text. This would allow the model to generate videos where certain frames deviate from the context, allowing for richer, more diverse expressive generations. This also allows generating videos with more motion and movement, which is better captured by multiple frames. Given a video-text similarity model gvid we have Jθ = E[gvid([i0,i1...ik...iN−1],c)]. In our experiments, we use a VideoMAE[29] fine-tuned on action classification, as gvid, which can classify an input video into one of a set of action text descriptions. We provide the target class text as input to the text-to-video diffusion model, and use the predicted probability of the ground truth class from VideoMAE as the reward.

Image Generation Objective - While text similarity is a strong signal to optimize, some use cases might be better addressed by reward models that only sit on the generated image. There is a prevalence of powerful image-based discriminative models such as Object Detectors and Depth Predictors. These models utilize the image as input to produce various useful metrics of the image, which can be used

- as a reward. The generated video is likely to be better aligned with the task if the reward obtained on each of the generated frames is high. Hence we define the reward in this case to be the mean

of the rewards evaluated on each of the individual frames, i.e R([i0,i1...ik...iN−1],c) = k R(ik). Note that given the generated frames, this is independent of the text input c. Hence we have, Jθ =

Ek∈[0,N] [R(ik)] = Ek∈[0,N] [Mϕ(ik)] via linearity of expectation, where Mϕ is a discriminative model that takes an image as input to produce a metric, that can be used to define a reward. We use the Aesthetic Reward model [24] and Object Detector [8] reward model for our experiments.

Video Generation Objective - With access to an external model that takes in multiple image frames, we can directly optimize for desired qualities of the generated video. Given a video metric model Nϕ, the corresponding reward is Jθ = E[Nϕ([i0,i1,..ik...iN−1])].

VideoCrafter VADER (Ours)

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

“The raccoon is wearing a red coat and holding a snowball.”

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

“The fox is wearing a red hat and playing with leaves.”

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

“A dog playing a slide guitar on a porch during a gentle rainstorm.”

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

“A strong lion and a graceful lioness resting together in the shade of a big tree on a wide grassland.”

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

“A peaceful deer eating grass in a thick forest, with sunlight filtering through the trees.”

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

“A dog strumming an acoustic guitar by a lakeside campfire under the stars.”

- Figure 3: Text-to-video generation results for VideoCrafter and VADER. We show results for VideoCrafter Text-to-Video model on the left and results for VADER on the right, where we use VideoCrafter as our base model. The reward models applied are a combination of HPSV2.1 and Aesthetic model in the first three rows, and PickScore in the last three rows. The videos in the third and last rows are generated based on prompts that are not encountered during training.

Long-horizon consistent generation: In our experiments, we adopt this formulation to enable a feature that is quite challenging for many open-source video diffusion models - that of generating clips that are longer in length. For this task, we use Stable Video Diffusion [3], which is an image-to-video diffusion model. We increase the context length of Stable Video Diffusion by 3x by making it autoregressive. Specifically, we pass the last generated frame by the model as input for generating the next video sequence. However, we find this to not work well, as the model was never trained over its own generations thus resulting in a distribution shift. In order to improve the generations, we use a video metric model Nϕ (V-JEPA [1]) that given a set of frames, produces a score about how predictive the frames are from one another. We apply this model on the autoregressive generations,

to encourage these to remain consistent with the earlier frames. Training the model in this manner allows us to make the video clips temporally and spatially coherent.

Reducing Memory Overhead: Training video diffusion models is very memory intensive, as the amount of memory linearly scales with respect to the number of generated frames. While VADER significantly improves the sample efficiency of fine-tuning these models, it comes

- at the cost of increased memory. This is because the differentiable reward is computed on the generated frame, which is a result of sequential de-noising steps.

- (i) Standard Tricks: To reduce the memory usage we use LoRA [13] that only updates a subset of the model parameters, further we use mixed precision that stores non-trainable parameters in fp16. To reduce memory usage during backpropagation we use gradient checkpointing and for the long horizon tasks, offload the storage of the backward computation graph from the GPU memory to the CPU memory.
- (ii) Truncated Backprop: Additionally, In our experiments we only backpropagate through the diffusion model for one timestep, instead of backpropagating through multiple timesteps [20], and have found this approach to obtain competitive results while requiring much less memory.
- (iii) Subsampling Frames: Since all the video diffusion models we consider are latent diffusion models, we further reduce memory usage by not decoding all the frames to RGB pixels. Instead, we randomly subsample the frames and only decode and apply loss on the subsampled ones.

Figure 4: Reward gradient vs policy gradient approaches as output resolution increases: We train DDPO and VADER, with increasing resolution of the generated video. In the above curve, we report the reward achieved after 100 steps of optimization, we find that as the resolution of the generation increases, the performance gap between both approaches significantly increases.

We conduct our experiments on 2 A6000 GPUS (48GB VRAM), and our model takes an average of

- 12 hours to train. However, our codebase supports training on a single GPU with 16GB VRAM.

Before VADER (Ours)

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

“A book and a cup of tea on a blanket in a sunflower field.”

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

“A book and a cup of hot chocolate on a windowsill with a snowy view.”

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

“A book and a cup of coffee on a rustic wooden table in a cabin.”

- Figure 5: Object removal (remove book) using VADER. The left column displays results from the base model (VideoCrafter), while the right column shows results from VADER after fine-tuning the base model to not display books by using an object detector as a reward model. As can be seen, VADER effectively removes book and replaces it with some other object.

### 5 Results

In this work, we focus on fine-tuning various conditional video diffusion models, including VideoCrafter [5] , Open-Sora [35] , Stable Video Diffusion [3] and ModelScope [31], through a comprehensive set of reward models tailored for images and videos. These include the Aesthetic model for images [24], HPSv2 [32] and PickScore [16] for image-text alignment, YOLOS [8] for object removal, VideoMAE for action classification [29], and V-JEPA [1] self-supervised loss for temporal consistency. Our experiments aim to answer the following questions:

- • How does VADER compare against gradient-free techniques such as DDPO or DPO regarding sample efficiency and computational demand?
- • To what extent can the model generalize to prompts that are not seen during training?
- • How do the fine-tuned models compare against one another, as judged by human evaluators?
- • How does VADER perform across a variety of image and video reward models?

This evaluation framework assesses the effectiveness of VADER in creating high-quality, aligned video content from a range of input conditioning.

Baselines. We compare VADER against the following methods:

- • VideoCrafter [5], Open-Sora 1.2 [35], and ModelScope [31] are current state-of-the-art (publicly available) text-to-video diffusion models. We serve them as base models for fine-tuning and comparison in our experiments in text-to-video space.
- • Stable Video Diffusion [3] is the current state-of-art (publicly available) image-to-video diffusion model. In all our experiments in image-to-video space, we use their base model for fine-tuning and comparison.
- • DDPO [2] is a recent image diffusion alignment method that uses policy gradients to adapt diffusion model weights. Specifically, it applies PPO algorithm [25] to the diffusion denoising process. We extend their code for adapting video diffusion models.
- • Diffusion-DPO [30] extends the recent development of Direct Preference Optimization (DPO) [22] in the LLM space to image diffusion models. They show that directly modeling the likelihood using the preference data can alleviate the need for a reward model. We extend their implementation to aligning video diffusion models, where we use the reward model to obtain the required preference data.

Reward models. We use the following reward models to fine-tune the video diffusion model.

- • Aesthetic Reward Model: We use the LAION aesthetic predictor V2 [24], which takes an image as input and outputs its aesthetic score in the range of 1-10. The model is trained on top of CLIP image embeddings, for which it uses a dataset of 176,000 image ratings provided by humans ranging from 1 to 10, where images rated as 10 are classified as art pieces.
- • Human Preference Reward Models: We use HPSv2 [32] and PickScore [16], which take as input an image-text pair and predict human preference for the generated image. HPSv2 is trained by fine-tuning CLIP model with a vast dataset that includes 798,090 instances of human preference rankings among 433,760 image pairs, while PickScore [16] is trained by fine-tuning CLIP model with 584,000 examples of human preferences. These datasets are among the most extensive in the field, offering a solid foundation for enhancing image-text alignment.
- • Object Removal: We design a reward model based on YOLOS [8], a Vision Transformer based object detection model trained on 118,000 annotated images. Our reward is one minus the confidence score of the target object category, from which video models learns to remove the target object category from the video.
- • Video Action Classification: While the above reward models sit on individual images, we employ a reward model that takes in the whole video as input. This can help with getting gradients for the temporal aspect of video generation. Specifically, we consider VideoMAE

- [29], which is fine-tuned for the task of action classification on Kinetics dataset [15]. Our reward is the probability predicted by the action classifier for the desired behavior.
- • Temporal Consistency via V-JEPA: While action classification models are limited to a fixed set of action labels, here we consider a more general reward function. Specifically, we use self-supervised masked prediction objective as a reward function to improve temporal consistency. Specifically, we use V-JEPA [1] as our reward model, where the reward is the negative of the masked autoencoding loss in the V-JEPA feature space. Note that we employ exactly the same loss objective that V-JEPA uses in their training procedure.

Before VADER (Ours)

Open-Sora

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

“a man in a trendy suit taking a selfie in a city square, surrounded by modern buildings and a fountain.”

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

“A bear enjoying a slice of cake at a picnic.”

ModelScope

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

“A shark riding a bike.”

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

“A bear playing chess.”

- Figure 6: Aligning Open-Sora 1.2 and ModelScope with VADER. The left column shows results from the base models, while results from VADER are demonstrated on the right. The first two rows use Open-Sora as the base model, and the last two rows use ModelScope. The reward models applied are PickScore in the first row, HPSv2.1 in the second row, HPSv2 in the third row, and the Aesthetic reward model in the last row.

Prompts. We consider the following set of prompt datasets for reward fine-tuning of text-to-video and image-to-video diffusion models.

- • Activity Prompts (Text): We consider the activity prompts from the DDPO [2]. Each prompt is structured as "a(n) [animal] [activity]," using a collection of 45 familiar animals. The activity for each prompt is selected from a trio of options: "riding a bike", "playing chess", and "washing dishes".
- • HPSv2 Action Prompts (Text): Here we filter out 50 prompts from a set of prompts introduced in the HPS v2 dataset for text-image alignment. We filter prompts such that they contain action or motion information in them.
- • ChatGPT Created Prompts (Text): We prompt ChatGPT to generate some vivid and creatively designed text descriptions for various scenarios, such as books placed beside cups, animals dressed in clothing, and animals playing musical instruments.
- • ImageNet Dog Category (Image): For image-to-video diffusion model, we consider the images in the Labrador retriever and Maltese category of ImageNet as our set of prompts.

- • Stable Diffusion Images (Image): Here we consider all 25 images from Stable Diffusion online demo webpage.

#### 5.1 Sample and Computational Efficiency

Training of large-scale video diffusion models is done by a small set of entities with access to a large amount of computing; however, fine-tuning of these models is done by a large set of entities with access to a small amount of computing. Thus, it becomes imperative to have fine-tuning approaches that boost both sample and computational efficiency.

In this section, we compare VADER’s sample and computational efficiency with other reinforcement learning approaches such as DDPO and DPO. In Figure 7, we visualize the reward curves during training, where the x-axis in the upper half of the figure is the number of reward queries and the one in the bottom half is the GPU-hours. As can be seen, VADER is significantly more efficient in terms of sample and computation than DDPO or DPO. This is mainly due to the fact that we send dense gradients from the reward model to the diffusion weights, while the baselines only backpropagate scalar feedback.

Aesthetics

Text Alignment

Action Prediction

0.32

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

6.5

AestheticScore

0.8

ClassAccuracy

HPSv2Score

0.30

6.0

0.6

5.5

0.28

0.4

5.0

0.26

0.2

4.5

0 500 1000 1500 Reward Queries

0 500 1000 Reward Queries

0 1000 2000 3000 Reward Queries

0.32

6.5

AestheticScore

0.8

ClassAccuracy

HPSv2Score

0.30

6.0

0.6

5.5

0.28

0.4

5.0

0.26

4.5

0.2

0 2 4 6 8 Compute (GPU-hours)

0 5 10 15 20 Compute (GPU-hours)

0 10 20 30 40 Compute (GPU-hours)

VADER (Ours) DPO DDPO

Figure 7: Training efficiency comparison: Top: Sample efficiency comparison with DPO and DDPO. Bottom: Computational efficiency comparison with DPO and DDPO. It can be seen that VADER starts converging within at most 12 GPU-hours of training, while DPO or DDPO do not show much improvement.

#### 5.2 Generalization Ability

Aes (T2V) HPS (T2V) ActP Aes (I2V) Train. Test. Train. Test. Train. Train. Test. Base 4.61 4.49 0.25 0.24 0.14 4.91 4.96 DDPO 4.63 4.52 0.24 0.23 0.21 N/A N/A DPO 4.71 4.41 0.25 0.24 0.23 N/A N/A Ours 7.31 7.12 0.33 0.32 0.79 7.83 7.64

Method

Table 1: Reward on Prompts in train & test. We split the prompts into train and test sets, such that the prompts in the test set do not have any overlap with the ones for training. We find that VADER achieves the best on both metrics.

A desired property of fine-tuning is generalization, i.e. the model fine-tuned on a limited set of prompts has the ability to generalize to unseen prompts. In this section, we extensively evaluate this property across multiple reward models and baselines. While training text-to-video (T2V) models, we use HPSv2 Action Prompts in our training set, whereas we use Activity Prompts in our test set. We use Labrador dog category in our training set for training image-to-video (I2V) models, while Maltese category forms our test set. Table 1 showcases VADER’s generalization ability.

#### 5.3 Human Evaluation

We carried out a study to evaluate human preferences via Amazon Mechanical Turk. The test consisted of a side-by-side comparison between VADER and ModelScope. To test how well the videos sampled from both the models aligned with their text prompts, we showed participants two videos generated by both VADER and a baseline method, asking them to identify which video better matched the given text. For evaluating video quality, we asked participants to compare two videos generated in response to the same prompt, one from VADER and one from a baseline, and decide which video’s quality seemed higher. We gathered 100 responses for each comparison. The results, illustrated in Table 2, show a preference for VADER over the baseline methods.

Method Fidelity Text Align

ModelScope 21.0% 39.0% VADER (Ours) 79.0% 61.0%

Table 2: Human Evaluation results for HPS reward model, where the task is image-text alignment.

#### 5.4 Qualitative Visualization

In this section, we visualize the generated videos for VADER and the respective baseline. We conduct extensive visualizations across all the considered reward functions on various base models.

HPS Reward Model: In Figure 3, we visualize the results before and after fine-tuning VideoCrafter using both HPSv2.1 and Aesthetic reward function together in the top three rows. Before fine-tuning, the raccoon does not hold a snowball, and the fox wears no hat, which is not aligned with the text description; however, the videos generated from VADER does not result in these inconsistencies. Further, VADER successfully generalizes to unseen prompts as shown in the third row of Figure 3, where the dog’s paw is less like a human hand than the video on the left. Similar improvements can be observed in videos generated from Open-Sora V1.2 and ModelScope as shown in the second and third rows of Figure 6.

Aesthetic Reward Model: In Figure 3, in the top three rows we visualize the results before and after fine-tuning ModelScope using a combination of Aesthetic reward function and HPSv2.1 model. Also, we fine-tune ModelScope via Aesthetic Reward function and demonstrate its generated video in the last row in Figure 6. We observe that Aesthetic fine-tuning makes the generated videos more artistic.

PickScore Model: In the bottom three rows of Figure 3, we showcase videos generated by PickScore fine-tuned VideoCrafter. VADER shows improved text-video alignment than the base model. In the last row, we test both models using a prompt that is not seen during training time. Further, video generated from PickScore fine-tuned Open-Sora is displayed in the first row of Figure 6.

Object Removal: Figure 5 displays the videos generated by VideoCrafter after fine-tuning using the YOLOS-based objection removal reward function. In this example, books are the target objects for removal. These videos demonstrate the successful replacement of books with alternative objects, like a blanket or bread.

Video Action Classification: In Figure 8, we visualize the video generation of ModelScope and VADER. In this case, we fine-tune VADER using the action classification objective, for the action specified in the prompt. For the prompt, "A person eating donuts", we find that VADER makes the human face more evident along with adding sprinkles to the donut. Earlier generations are often misclassified as baking cookies, which is a different action class in the kinetics dataset. The addition

ModelScope VADER (Ours)

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

“A person playing Piano”

of colors and sprinkles to the donut makes it more distinguishable from cookies leading to a higher reward.

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

“A person eating Donuts”

- Figure 8: Video action classifiers as reward model. We use VideoMAE action classification model as a reward function to fine-tune ModelScope’s Text-to-Video Model. We see that after fine-tuning, VADER generates videos that correspond better to the actions.

V-JEPA reward model: In Figure 9, we show results for increasing the length of the video generated by Stable Video Diffusion (SVD). For generating long-range videos on SVD, we use autoregressive inference, where the last frame generated by SVD is given as conditioning input for generating the next set of images. We perform three steps of inference, thus expanding the context length of SVD by three times. However, as one can see in the images bordered in red, after one step of inference, SVD starts accumulating errors in its predictions. This results in deforming the teddy bear, or affecting the rocket in motion. VADER uses V-JEPA objective of masked encoding to enforce self-consistency in the generated video. This manages to resolve the temporal and spatial discrepancy in the generations as shown in Figure 9.

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

Stable Video Diffusion

VADER (Ours)

Stable Video Diffusion

VADER (Ours)

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

|[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]|
|---|

[Figure 152]

[Figure 153]

[Figure 154]

|[Figure 155]<br><br>[Figure 156]<br><br>[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]<br><br>[Figure 161]<br><br>[Figure 162]|
|---|

[Figure 163]

- Figure 9: Improving temporal and spatial consistency of Stable Video Diffusion (SVD) Imageto-Video Model. Given the leftmost frame as input, we use autoregressive inference to generate 3*N frames in the future, where N is the context length of SVD. However, this suffers from error accumulation, resulting in corrupted frames, as highlighted in the red border. We find that VADER can improve the spatio-temporal consistency of SVD by using V-JEPA’s masked encoding loss as its reward function.

### 6 Conclusion

We presented VADER, which is a sample and compute efficient framework for fine-tuning pre-trained video diffusion models via reward gradients. We utilized various reward functions evaluated on images or videos to fine-tune the video diffusion model. We further showcased that our framework is agnostic to conditioning and can work on both text-to-video and image-to-video diffusion models. We hope our work creates more interest towards adapting video diffusion models.

### References

- [1] Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., Assran, M., Ballas, N.: V-jepa: Latent video prediction for visual representation learning (2023)
- [2] Black, K., Janner, M., Du, Y., Kostrikov, I., Levine, S.: Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301 (2023)
- [3] Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M., Lorenz, D., Levi, Y., English, Z., Voleti, V., Letts, A., et al.: Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127 (2023)
- [4] Brooks, T., Holynski, A., Efros, A.A.: Instructpix2pix: Learning to follow image editing instructions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18392–18402 (2023)
- [5] Chen, H., Zhang, Y., Cun, X., Xia, M., Wang, X., Weng, C., Shan, Y.: Videocrafter2: Overcoming data limitations for high-quality video diffusion models (2024)
- [6] Clark, K., Vicol, P., Swersky, K., Fleet, D.J.: Directly fine-tuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400 (2023)
- [7] Dong, H., Xiong, W., Goyal, D., Zhang, Y., Chow, W., Pan, R., Diao, S., Zhang, J., Shum, K., Zhang, T.: Raft: Reward ranked finetuning for generative foundation model alignment (2023)
- [8] Fang, Y., Liao, B., Wang, X., Fang, J., Qi, J., Wu, R., Niu, J., Liu, W.: You only look at one sequence: Rethinking transformer in vision through object detection. CoRR abs/2106.00666

(2021), https://arxiv.org/abs/2106.00666

- [9] Feng, W., He, X., Fu, T.J., Jampani, V., Akula, A., Narayana, P., Basu, S., Wang, X.E., Wang, W.Y.: Training-free structured diffusion guidance for compositional text-to-image synthesis

(2023)

- [10] Ho, J., Chan, W., Saharia, C., Whang, J., Gao, R., Gritsenko, A., Kingma, D.P., Poole, B., Norouzi, M., Fleet, D.J., Salimans, T.: Imagen video: High definition video generation with diffusion models (2022)
- [11] Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. CoRR abs/2006.11239

(2020), https://arxiv.org/abs/2006.11239

- [12] Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., Fleet, D.J.: Video diffusion models

(2022)

- [13] Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: Lora: Low-rank adaptation of large language models (2021)
- [14] Karras, T., Aittala, M., Aila, T., Laine, S.: Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems 35, 26565–26577

(2022)

- [15] Kay, W., Carreira, J., Simonyan, K., Zhang, B., Hillier, C., Vijayanarasimhan, S., Viola, F., Green, T., Back, T., Natsev, P., et al.: The kinetics human action video dataset. arXiv preprint arXiv:1705.06950 (2017)
- [16] Kirstain, Y., Polyak, A., Singer, U., Matiana, S., Penna, J., Levy, O.: Pick-a-pic: An open dataset of user preferences for text-to-image generation (2023)
- [17] Lambert, N., Pyatkin, V., Morrison, J., Miranda, L., Lin, B.Y., Chandu, K., Dziri, N., Kumar, S., Zick, T., Choi, Y., et al.: Rewardbench: Evaluating reward models for language modeling. arXiv preprint arXiv:2403.13787 (2024)
- [18] Lee, K., Liu, H., Ryu, M., Watkins, O., Du, Y., Boutilier, C., Abbeel, P., Ghavamzadeh, M., Gu, S.S.: Aligning text-to-image models using human feedback (2023)

- [19] Liu, R., Wu, R., Van Hoorick, B., Tokmakov, P., Zakharov, S., Vondrick, C.: Zero-1-to-3: Zero-shot one image to 3d object. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 9298–9309 (2023)
- [20] Prabhudesai, M., Goyal, A., Pathak, D., Fragkiadaki, K.: Aligning text-to-image diffusion models with reward backpropagation. arXiv preprint arXiv:2310.03739 (2023)
- [21] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PMLR (2021)
- [22] Rafailov, R., Sharma, A., Mitchell, E., Manning, C.D., Ermon, S., Finn, C.: Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems 36 (2024)
- [23] Sauer, A., Boesel, F., Dockhorn, T., Blattmann, A., Esser, P., Rombach, R.: Fast high-resolution image synthesis with latent adversarial diffusion distillation. arXiv preprint arXiv:2403.12015

(2024)

- [24] Schuhmann, C.: Laoin aesthetic predictor (2022), https://laion.ai/blog/ laion-aesthetics/
- [25] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O.: Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 (2017)
- [26] Sohl-Dickstein, J., Weiss, E.A., Maheswaranathan, N., Ganguli, S.: Deep unsupervised learning using nonequilibrium thermodynamics (2015)
- [27] Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models (2022)
- [28] Tallec, C., Ollivier, Y.: Unbiasing truncated backpropagation through time. arXiv preprint arXiv:1705.08209 (2017)
- [29] Tong, Z., Song, Y., Wang, J., Wang, L.: Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. Advances in neural information processing systems 35, 10078–10093 (2022)
- [30] Wallace, B., Dang, M., Rafailov, R., Zhou, L., Lou, A., Purushwalkam, S., Ermon, S., Xiong, C., Joty, S., Naik, N.: Diffusion model alignment using direct preference optimization. arXiv preprint arXiv:2311.12908 (2023)
- [31] Wang, J., Yuan, H., Chen, D., Zhang, Y., Wang, X., Zhang, S.: Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571 (2023)
- [32] Wu, X., Hao, Y., Sun, K., Chen, Y., Zhu, F., Zhao, R., Li, H.: Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341 (2023)
- [33] Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., Dong, Y.: Imagereward: Learning and evaluating human preferences for text-to-image generation (2023)
- [34] Yuan, H., Zhang, S., Wang, X., Wei, Y., Feng, T., Pan, Y., Zhang, Y., Liu, Z., Albanie, S., Ni, D.: Instructvideo: Instructing video diffusion models with human feedback. arXiv preprint arXiv:2312.12490 (2023)
- [35] Zheng, Z., Peng, X., Yang, T., Shen, C., Li, S., Liu, H., Zhou, Y., Li, T., You, Y.: Opensora: Democratizing efficient video production for all (March 2024), https://github.com/ hpcaitech/Open-Sora

