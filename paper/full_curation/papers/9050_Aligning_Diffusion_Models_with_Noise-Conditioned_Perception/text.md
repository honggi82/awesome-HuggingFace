# arXiv:2406.17636v2[cs.CV]2Dec2025

## ALIGNING TEXT-TO-IMAGE DIFFUSION MODELS WITH NOISE-CONDITIONED PERCEPTION ∗

Alexander Gambashidze Artificial Intelligence Research Institute (AIRI) Skolkovo Institute of Science and Technology Moscow gambashidze@airi.net

Anton Kulikov HSE University Moscow

Ilya Makarov Artificial Intelligence Research Institute (AIRI) Research Center for Trusted Artificial Intelligence, ISP RAS Moscow

Yuriy Sosnin HSE University Moscow

### ABSTRACT

Human preference optimization, originally developed for Language Models, has shown promise in improving text-to-image Diffusion Models by enhancing prompt alignment, visual appeal, and user preference. However, Diffusion Models are typically optimized in pixel or VAE space, which often misaligns with human perception, resulting in slower and less efficient training during fine-tuning and preference optimization. In this work, we demonstrate that using a perceptual objective significantly enhances both training speed and overall model quality. We fine-tune Stable Diffusion 1.5 and XL using Direct Preference Optimization (DPO), Contrastive Preference Optimization (CPO), and supervised fine-tuning (SFT) within this perceptual embedding space. Our approach significantly outperforms standard latent-space implementations across various metrics, including quality and computational cost, when training on the widely-used Pick-a-Pic dataset. For SDXL, our method achieves a 64.6% general preference over the baseline DPO on the PartiPrompts dataset while significantly reducing compute to reach comparable DPO performance. Additionally, we enhance the Pick-a-Pic dataset, making it approximately 10 times smaller while training models that surpass the original published versions in just 12 and 3.5 GPU hours for SDXL and SD1.5, respectively. This paper demonstrates that the overall quality of Diffusion models after fine-tuning can be significantly improved while also being more efficient and requiring far less data.

Keywords Text-to-Image · Diffusion Models, Preference Optimization · Alignment

### 1 Introduction

Recent research has highlighted the benefits of alignment—the process of fine-tuning a generative model to meet specific objectives, particularly human preferences Ziegler et al. [2019], Rafailov et al. [2024]. One of the most prominent methods in this domain is Direct Preference Optimization (DPO), which involves fine-tuning a model on a dataset of ranked sample pairs Rafailov et al. [2024]. DPO has achieved substantial progress in aligning Large Language Models, owing to its supervised nature and simplicity compared to hyperparameter-heavy reinforcement learning methods like RLHF Ziegler et al. [2019]. Recently, DPO has been adapted for Diffusion Models, significantly improving overall user preference, prompt adherence, aesthetics, and image structure Wallace et al. [2023].

Diffusion Models are trained to predict pixel-space noise, and pixel space lacks semantic or perceptual structure Zhang et al. [2018]. Latent Diffusion models attempt to bring the diffusion process into the latent space of a Variational

∗Citation: A. Gambashidze, A. Kulikov, Y. Sosnin and I. Makarov, "Aligning Text-to-Image Diffusion Models With Noise-Conditioned Perception," in IEEE Access, vol. 13, pp. 193745-193753, 2025, doi: 10.1109/ACCESS.2025.3632092.

[Figure 1]

- Figure 1: Noise-Conditioned Perceptual objective for aligning diffusion models significantly improves Direct Preference Optimization.

Autoencoder (VAE) Rombach et al. [2022], but this mainly serves to reduce computational costs, and these latent spaces still retain a lot of pixel-level details.

In contrast, it is well-known that the embedding spaces of pretrained deep vision networks are perceptual and highly informative, capable of capturing high-level properties of images Zhang et al. [2018]. This discrepancy motivates us to explore a perceptual objective for aligning diffusion models more effectively with human preferences.

There have been prior attempts to equip diffusion model training objective with some form of perceptual loss Zhang et al. [2024]. These losses are implemented by denoising from noise level t to the initial sample x0, decoding it with VAE, and then feeding it into a corresponding vision network. However, this approach is challenging to optimize effectively.

Notably, recent advancements have shown that the U-Net backbone used in diffusion models — a deep pretrained vision model itself — possesses an embedding space with perceptual properties Xiang et al. [2023], Zhao et al. [2023], Xu et al. [2024a]. The paper "Diffusion Model with Perceptual Loss" Lin and Yang [2023] proposes utilizing this embedding space for pretraining, showing slightly improved results compared to training without classifier free guidance Ho and Salimans [2022]. Building on this, we propose a noise-conditioned perceptual loss for preference optimization. Specifically, we utilize the pretrained encoder of a denoising U-Net, which operates within a noise-conditioned embedding space. This approach allows us to overcome the limitations of pixel-space optimization and directly align with human perceptual features, significantly accelerating the training process.

Additionally, existing image preference datasets are often much larger than necessary and are challenging to train on consumer devices. For example, while Large Language Models benefit from megabytes of preference data, the most popular image preference dataset, Pick-a-Pic, weighs 330GB and contains million of pairs, yet yields less significant results compared to tuning LLMs. Therefore, we analyze and refine the Pick-a-Pic dataset to enable better model training on consumer-grade hardware.

Our contributions are as follows:

- 1. Noise-Conditioned Perceptual Preference Optimization (NCPPO): We introduce NCPPO, a method that utilizes the U-Net encoder’s embedding space for preference optimization. This approach aligns the optimization process with human perceptual features, rather than the less informative pixel space. It can be seamlessly combined with Direct Preference Optimization (DPO), Contrastive Preference Optimization (CPO), and Supervised Fine-Tuning (SFT), further enhancing their effectiveness.
- 2. Enhanced Training Efficiency: Our method reduces the compute and training time required for preference optimization. For example, NCPPO achieves superior results with 40% less computational time.

[Figure 2]

- Figure 2: Our method adapts much better to human preferences compared to baseline latent / pixel implementations and can produce extremely high visual appeal alignment.

3. Data Enhancement: We demonstrate that the large sizes of current datasets, such as the hundreds of gigabytes required for the Pick-a-Pic dataset Kirstain et al. [2024], are unnecessary. By reducing the size of the Pick-a-Pic dataset by tenfold, we achieve significantly better results.

Embedding the preference optimization process within a noise-conditioned perceptual space provides a more natural and efficient method for aligning diffusion models with human preferences. Our results indicate that this approach not only improves the quality of generated images but also significantly reduces computational costs, making it a promising direction for future research and practical applications.

### 2 Related Works

#### 2.1 Diffusion Model Fine-Tuning

Various approaches have been developed to fine-tune diffusion models for better alignment with specific objectives, such as conditioning or user preferences. One class of methods uses gradients of explicit reward functions. Classifier Guidance Dhariwal and Nichol [2021], Song and Ermon [2019] proposes an inference-time technique to guide generations toward desirable regions using the gradients of a classifier. However, this method requires a noise-conditioned classifier network Ho and Salimans [2022], which makes it impractical in many settings. Unified Guidance Bansal et al. [2023], Ma et al. [2023] instead uses intermediate denoised predictions as inputs for an off-the-shelf classifier. However, these predictions are often out-of-distribution for most vision models due to nonlinearity in sampling trajectories.

Training-time techniques like DRaFT Clark et al. [2023], ReFL Xu et al. [2024b], and AlignProp Prabhudesai et al. [2023] optimize for maximum differentiable reward by backpropagating gradients through the entire sampling process, which demands substantial GPU memory.

Similar to RLHF in language models Ziegler et al. [2019], several works employ Reinforcement Learning (RL) for reward optimization in diffusion models Black et al. [2023], Fan et al. [2024]. These methods frame the generation process as a Markov Decision Process (MDP) with the noise predictor network acting as an agent, using variants of Policy Gradient Schulman et al. [2017] with KL-regularized rewards Christiano et al. [2017]. While these approaches do not require differentiable rewards, they are often unstable and susceptible to reward hacking.

#### 2.2 Direct Preference Optimization

Initially proposed for language models, Direct Preference Optimization (DPO) avoids the instability and complexity of RL methods by reformulating the RLHF objective in terms of an implicit optimal reward Rafailov et al. [2024]. Diffusion-DPO Wallace et al. [2023] extends this approach to diffusion models, employing conditional score matching objectives for efficient training. Significant efforts have been made to improve the DPO procedure, particularly in the context of large language models (LLMs). These efforts include identifying sources of preference data Liu et al.

- [2023], Tajwar et al. [2024], employing listwise instead of pairwise losses Liu et al. [2024], Song et al. [2024], and experimenting with the reference model training, removing, or replacing it Xu et al. [2024c], Hong et al. [2024], Gorbatovski et al. [2024]. Despite these advancements in NLP, relatively little has been proposed specifically for text-to-image diffusion models in this area.

#### 2.3 Diffusion U-Net Properties

Pretrained deep image networks, including U-Nets, have been shown to possess rich, transferable latent spaces that are applicable to various downstream tasks Zhang et al. [2018]. Recent studies indicate that diffusion model backbones, such as U-Net Ronneberger et al. [2015] and DiT Peebles and Xie [2023], are no exception. U-Net backbones trained with a diffusion objective can serve as excellent initializations for fine-tuning on downstream image-to-image tasks, such as semantic segmentation or depth estimation Zhao et al. [2023], Xu et al. [2024a], and can even be used for classification with minimal additional training Xiang et al. [2023]. In fact, noise-conditioned classifiers for Classifier Guidance were initialized with U-Net in the original work Dhariwal and Nichol [2021]. Moreover, features from U-Net cross-attention layers have been extensively manipulated to achieve controlled generation and editing Hertz et al. [2022], Epstein et al. [2023], Kwon et al. [2022].

Building on these insights, our work proposes utilizing the U-Net encoder’s embedding space for preference optimization in diffusion models, offering a more efficient and perceptually aligned training process.

### 3 Preliminaries

Diffusion models are latent-variable generative models that generate data by iteratively denoising a sample from Gaussian noise Ho et al. [2020]. The core idea is to model the data distribution with a two-way Markov chain of gradually noised/denoised latent variables. The diffusion model formulation consists of a fixed forward process, which takes a sample from the data distribution and progressively corrupts it with Gaussian noise, and a parametric reverse process that learns to revert this corruption and effectively recover samples from the data distribution.

In the forward process, a data point x0 is transformed into a noisy version xt over T discrete timesteps. At each timestep t, noise is incrementally added according to a predefined variance schedule {αt}T0 . In terms of samples, this process can be formulated as follows:

xt = √α¯tx0 + √1 − α¯tϵ, ϵ ∼ N(0,I) (1)

The reverse process involves learning a model ϵθ that predicts the noise added at each step, directly recovering an estimate of x0. Transition to previous step sample xt−1 is defined by the DDPM reverse process as follows:

1 √αt

xt−1 =

1 − αt √1 − α¯t

ϵθ(xt,t) + σtz,z ∼ N(0,I) (2)

xt −

Diffusion denoising model ϵθ can be efficiently trained with SGD by minimizing the squared error between the true noise ϵ and predicted noise ϵθ(xt,t). Formally, this loss is defined as follows:

Lθ = ||ϵ − ϵθ(xt,t)||22, L(θ) = Ex∼D [Lθ] (3)

After the denoising model is trained, sampling does not need to involve all T steps and can utilize the SDE or score-based formulation of diffusion models Song and Ermon [2019], Karras et al. [2022].

[Figure 3]

Figure 3: Overall NCPPO pipeline. We optimize preferences inside a Noise-Conditioned embedding space.

#### 3.1 Preference Optimization

Next, we review Direct Preference Optimization (DPO) and Contrastive Preference Optimization (CPO) approaches for diffusion models, establishing the foundation for our proposed Noise-Conditioned Perceptual Preference Optimization (NCPPO).

The task of preference optimization is to fine-tune a generative model such that it produces samples which are more aligned to what humans find preferable. It is assumed that humans express preference according to a latent reward function r∗, and that there is a dataset D = {(c,xw,xl)}, where c is a condition, and xw and xl are the preferred (winner) and dispreferred (loser) generated samples, respectively.

RLHF Christiano et al. [2017] first aims to obtain an explicit parameterized reward function rϕ by fitting Bradley-Terry model on the dataset D with maximum likelihood.

Lr (rϕ,D) = − E(c,xw,xl)∼D log σ rϕ (c,xw) − rϕ c,xl

(4)

With the fitted reward function, it is possible to use Reinforcement Learning to optimize the generative model pθ(x|c) as a policy, and employ a form of Policy Gradient to optimize the model. Following Ziegler et al. [2019], reward is also modified with regularization by adding KL-divergence term:

c,x∼pθ(x|c)[r (c,x)] − βDKL [pθ (x|c)∥pref (x|c)] (5)

Ec∼D

max

pθ

DPO elegantly restates LRLHF without the need for RL Policy Gradient estimators. First, it considers the solution to the given optimization problem, optimal policy p∗(x|c) under reward function r, and rearranges it into an expression in terms of optimal policy:

preference (x|c)exp(r (c,x)/β) x pref (x|c)exp(r (c,x)/β)

p∗θ (x|c) =

Plugging it into the Bradley-Terry model from equation (4), we arrive at the DPO objective:

(6)

LDPO(θ) = − E(c,xw,xl)∼D log σ β log

(7)

pθ(xw|c) pref(xw|c) − β log

pθ(xl|c) pref(xl|c)

Through several approximations and careful rearrangement of expectations using Jensen’s inequality, Wallace et al. Wallace et al. [2023] find a way to rewrite the DPO objective for diffusion models in the following form:

LDiffusionDPO(θ) = − E(c,xw,xl)∼D log σ −βT (Lwθ − Lwref) − (Llθ − Llref)

(8)

where L corresponds to the diffusion loss from (3), calculated over winners, losers, current and reference models, respectively.

Contrastive Policy Optimization (CPO) is derived from the DPO objective by substituting the reference policy pref(x|c) with the optimal policy pw(x|c), defined such that pw(xw|c) = 1 and 0 ≤ pw(xl|c) ≤ 1. By incorporating this substitution into the original DPO objective, Xu et al. Xu et al. [2024c] derive the following formulation:

LCPO(θ) =

− E(c,xw,xl)∼D log σ(β log pθ(xw|c)− β log pθ(xl|c) + β log pref(xl|c))

After performing a series of intermediate calculations, the CPO objective simplifies to:

(9)

LCPO(θ) = E(c,xw,xl)∼D c,xw,xl ∼ D log σ (β log pθ(xw|c)) − β log pθ xl|c

(10)

− λE(c,xw)∼D [β log pθ(xw|c)]

where the second term provides additional guidance for the policy. Applying this formulation to the diffusion problem results in the final objective:

LDiffCPO(θ) = − E(c,xw,xl)∼D [log σ (−βT (Lwθ − Lθl))] + λE(c,xw)∼D [Lwθ ]

### 4 Noise-Conditioned Preference Optimization

(11)

Inspired by the properties of human perception and the successful alignment of large language models (LLMs) within informative embedding spaces, our goal is to perform diffusion preference optimization within a similarly informative perceptual embedding space. As the preference optimization objectives in Equation (8) replace logits with a diffusion squared error objective, we first introduce a diffusion perceptual loss, similar to the approach in Lin and Yang [2023].

Currently, there are no pretrained open-source noise-conditioned encoder networks that operate in the SD1.5 and SDXL VAE latent space, except for the encoder of a pretrained U-Net. A naive way of using an off-the-shelf network like CLIP would require predicting x0 directly from arbitrary timestep, and further decoding it with VAE decoder, introducing distribution shifts along the way. Meanwhile, U-Net is already noise-conditioned, operates in latent space, incorporates text condition, and has been shown to exhibit the same perceptual properties as other pre-trained vision networks Xiang et al. [2023].

Let f(xt′,c,t′) denote the downsampling stack of pre-trained U-Net, evaluated as some timestep t′ - this is our perceptual encoder. We choose to evaluate it at t′ = t−1, previous timestep from the one noise prediction was obtained from, recovered during training by performing DDPM reverse step (2). To obtain ground truth embedding, we perform reverse step on true noise. We subscript xt′ with indications of which noise is used to perform the step: for example, obtaining a sample for winner and optimized model looks like this:

xwθ,t′ = xwθ,t−1 =

1 − αt √1 − α¯t

1 √αt

xwt −

ϵθ(xwt ,t,c) + σtz, z ∼ N(0,I)

This results in the following perceptual diffusion objective:

(12)

PLθ = ||f(xθ,t′,c,t′) − f(xt′,c,t′)||22 (13)

We use this formulation in place of the standard diffusion loss in Preference Optimization objectives. Denote the losses for the winner, loser, winner (reference), and loser (reference) as follows:

PLwθ = ||f(xwθ,t′,c,t′) − f(xwt′,c,t′)||22, (14) PLwref = ||f(xwref,t′,c,t′) − f(xwt′,c,t′)||22 (15) PLlθ = ||f(xlθ,t′,c,t′) − f(xlt′,c,t′)||22, (16) PLwref = ||f(xlref,t′,c,t′) − f(xlt′,c,t′)||22 (17)

DPO target: We use the embeddings in the DiffusionDPO loss, simply replacing the noise terms with the embeddings obtained from the encoder:

LDiffusionDPO(θ) = − E(c,xw,xl)∼D log σ −βT (PLwθ − PLwref) − (PLlθ − PLlref)

(18)

Contrastive Preference Optimization target: Following the original paper and replacing the noise term with perceptual embeddings, we get:

LDiffusionCPO(θ) =

− E(c,xw,xl)∼D log σ −βT PLwθ − PLlθ + λE(c,xw)∼D [PLwθ ]

In the case of CPO the coefficient β is different since we omit the reference model.

(19)

### 5 Enhancing Preference Data

Now, we want to highlight our second contribution, which demonstrates how image preference datasets should be designed. We find that with more carefully curated data, our NCP-DPO method significantly outperforms the baseline DPO even further while the overall quality of both models (DPO, NCP-DPO) increases significantly. To illustrate this, we provide synthetic win rates based on HPSv2 and PickScore, comparing NCP-DPO on both the filtered and original Pick-a-Picv2 datasets. The primary issue with the original dataset lies in the contradictory examples it contains a significant portion of the images serve as both winners and losers in different pairs. For instance, suppose we have three images for a single prompt, x1 < x2 < x3 where < denotes preference. Now, let’s recall the gradient for the Bradley-Terry model:

∇θLDPO(xw,xl) =

− βE(c,xw,xl)∼D σ(rθ(xl) − rθ(xw))× (∇θ log p(xw) − ∇θ log p(xl))

Now let’s write down the sum of two gradients computed on x1,x2 and x2,x3 pairs.

∇θLDPO(x2,x1) + ∇θLDPO(x3,x2) =

− β [σ(rθ(x1) − rθ(x2)) · (∇θ log p(x2) − ∇θ log p(x1))] − β [σ(rθ(x2) − rθ(x3)) · (∇θ log p(x3) − ∇θ log p(x2))]

(20)

(21)

NCP-DPO vs DPO PickScore Winrate HPSv2 Winrate SDXL 54.3 55.7 SDXL (filtered data) 65.8 67.1 SD1.5 56.8 59.1 SD1.5 (filtered data) 64.6 71.3

- Table 1: Synthetic winrates for NCP-DPO models against DPO. Filtering data improves both NCP-DPO and DPO and makes our method even more powerful.

[Figure 4]

- Figure 4: We evaluate training speed using PickScore on a Pick-a-Pic validation set. Our method significantly accelerates the learning process compared to baseline methods like DPO and Supervised Fine-Tuning while also achieving superior quality. Contrastive Preference Optimization is very unstable due to the lack of a reference model, but we demonstrate that our method provides a regularization effect as well.

Notice how terms βσ(rθ(x1) − rθ(x2)) · ∇θ log p(x2) and βσ(rθ(x2) − rθ(x3)) · ∇θ log p(x2) partially cancel each other out, adding noise instead of a clear signal. When excluding contradictory examples, the training process becomes more stable and predictable. Synthetic comparisons for original and filtered dataset are shown in Table 1.

To further analyze the impact of contradictory examples, we created three additional mini versions of the Pick-a-Pic dataset. For each prompt, we selected their absolute winner x (images that have 0 losses or draws and more than 1 win), their latest losers y, and images z that are the latest losers for y. Results from Table 2 show an intuitive fact that winners must be as good as possible and contradictory samples decreases the overall quality.

[Figure 5]

- Figure 5: Side-by-side real human preferences comparison for different SDXL models using PartiPrompts benchmark. We compare NCP-DPO with 1) Our own DPO-SDXL 2) Original published DPO-SDXL 3) Baseline model with no preference optimization. Our method significantly improves Direct Preference Optimization. All models are trained on the same data.
- 6 Experiments

In this section we empirically validate our proposed contributions. As we have two independent contributions (perceptual objective for preferences, data improvement), our goal is to show that each one of them significantly improves the overall quality and training speed in terms of synthetic reward independently. We observe that combining perceptual objective with enhanced data improves the model even further compared to original DPO trained on enhanced data.

Comparison PickScore Median x > y 21.98

- x > y ∪ x > z 22.05
- x > y ∪ y > z 21.96

- Table 2: Synthetic PickScore rewards for different dataset setups. x > y stays for mini dataset with absolute winners across all prompts and their latest losers, x > y ∪ y > z denotes the same dataset with added images z that are losers to y, x > y ∪ x > z denotes the same as previous but instead the winners are always absolute. We train these models for 250 steps keeping the same hyperparameters.

#### 6.1 Setup

We fine-tune Stable Diffusion v1.5 (SD1.5) Rombach et al. [2022] and Stable Diffusion XL (SDXL) Podell et al. [2023] models. We run our largest experiments in terms of compute on a Pick-a-Pic v2 dataset Kirstain et al. [2024], following a setup from Wallace et al. [2023] that consists of 851,293 pairs, with 58,960 unique prompts, obtained from versions of SDXL and DreamLike (a fine-tune of SD1.5). We train both original DPO and NCP-DPO to make the comparison fair, however, we also compare our NCP-DPO with original published SDXL model. This experiment is validated only via side-by-side user study. To enhance the dataset, we leave only absolute winners across all prompts and get a version with 87,687 pairs and 53,701 unique prompts. Here, we score a uniform subset of checkpoints during training to show that Noise-Conditioned Perception works better and quicker adapts to human preferences. Both dataset and best model checkpoint based on reward will be publicly available. All our checkpoints, including baselines, are obtained with rank 64 LoRA Hu et al. [2021] for U-Net weights, contrary to full-finetunes from Wallace et al. [2023], Li et al. [2024]. We also use 8-bit AdamW optimizer Loshchilov and Hutter [2017] with default hyperparameters and half precision for non-trainable weights everywhere. For large experiments we use hyperparameters exactly like in the original Diffusion-DPO paper. We increase the learning rate for smaller experiments in 3 times. All experiments were done using 4x Nvidia H100 GPU and Intel(R) Xeon(R) Platinum 8480+ CPU processor.

#### 6.2 Cheaper and Quicker

To demonstrate the efficiency of our proposed method, we use synthetic rewards Wu et al. [2023], Kirstain et al. [2024] to evaluate the generated images. HPSv2 is considered the most reliable, as it has been shown to best correspond with human judgment. We generated samples for the validation set of the Pick-a-Pic-v2 dataset, which consists of 500 unique prompts, following the exact Diffusion-DPO setup. For each method—Supervised Fine-Tuning, DPO, CPO, and their respective improvements, NCP-SFT, NCP-DPO, and NCP-CPO—we generated images with the same amount of training time. Training speeds are illustrated in Fig. 4. We observed that NCP-DPO-SD1.5 significantly outperformed the original DPO-SD1.5 in terms of PickScore reward, achieving this with only 3.5 GPU hours of training. A similar trend was seen with SDXL: our approach not only outperformed the original SDXL-DPO when trained on the same data but also produced a superior model in just 12 GPU hours when trained on the improved Pick-a-Pic dataset. Additionally, Contrastive Preference Optimization lacks a reference model to stabilize the training process, making the model more prone to overfitting and divergence. However, NCP-CPO effectively mitigates this issue by matching the trainable U-Net’s embeddings with a frozen copy, providing significant regularization.

#### 6.3 Side-by-Side

To validate the method overall improvements, we follow original Diffusion-DPO paper human feedback annotation setup, comparing generations for PartiPrompts Yu et al. [2022] under three different criteria: Q1 General Preference (Which image do you prefer given the prompt?), Q2 Visual Appeal (prompt not considered) (Which image is more visually appealing?) Q3 Prompt Alignment (Which image better fits the text description?). Each generation had 3 independent annotations for each criteria. Image is considered a winner according to a majority votes.

We validate only DPO method with the same dataset as in original paper. Our human annotations show superior results. We compare our largest run with the original DPO-SDXL published weights and a baseline method DPO trained by us with exactly the same hyperparameters as in NCP-DPO version for a fair comparison. Results are shown on Fig.5.

### 7 Conclusion

Optimizing Diffusion models for human preferences using Noise-Conditioning Perception is a natural and powerful way to guide the training process of diffusion models. In this work, we have shown that using U-Net’s own encoder

embedding space is already a strong baseline for improving both the speed of adaptation to human preferences and the overall quality of the model compared to original methods.

### 8 Limitations & Future Work

While Noise-Conditioned Perception works well for diffusion models based on the U-Net architecture, it remains a question for future work whether this approach works that well with Diffusion Transformer architecture Peebles and Xie [2023]. We leave this investigation to future research.

### Acknowledgment

The work of I. Makarov on Section 1 and 2 was supported by a grant, provided by the Ministry of Economic Development of the RF in accordance with the subsidy agreement (agreement identifier 000000C313925P4G0002) and the agreement with the Ivannikov Institute for System Programming of the Russian Academy of Sciences dated June 20, 2025 No. 139-15-2025-011.

### References

Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.

Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. arXiv preprint arXiv:2311.12908, 2023.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

Jiacheng Zhang, Jie Wu, Yuxi Ren, Xin Xia, Huafeng Kuang, Pan Xie, Jiashi Li, Xuefeng Xiao, Weilin Huang, Min Zheng, et al. Unifl: Improve stable diffusion via unified feedback learning. arXiv preprint arXiv:2404.05595, 2024.

Weilai Xiang, Hongyu Yang, Di Huang, and Yunhong Wang. Denoising diffusion autoencoders are unified selfsupervised learners. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 15802–15812, October 2023.

Wenliang Zhao, Yongming Rao, Zuyan Liu, Benlin Liu, Jie Zhou, and Jiwen Lu. Unleashing text-to-image diffusion models for visual perception. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 5729–5739, October 2023.

Guangkai Xu, Yongtao Ge, Mingyu Liu, Chengxiang Fan, Kangyang Xie, Zhiyue Zhao, Hao Chen, and Chunhua Shen. Diffusion models trained with large data are transferable visual models. arXiv preprint arXiv:2403.06090, 2024a. Shanchuan Lin and Xiao Yang. Diffusion model with perceptual loss. arXiv preprint arXiv:2401.00110, 2023. Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open

dataset of user preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36, 2024.

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Universal guidance for diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 843–852, 2023.

Jiajun Ma, Tianyang Hu, Wenjia Wang, and Jiacheng Sun. Elucidating the design space of classifier-guided diffusion generation. In The Twelfth International Conference on Learning Representations, 2023.

Kevin Clark, Paul Vicol, Kevin Swersky, and David J Fleet. Directly fine-tuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400, 2023.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36, 2024b.

Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, and Katerina Fragkiadaki. Aligning text-to-image diffusion models with reward backpropagation. arXiv preprint arXiv:2310.03739, 2023.

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.

Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Reinforcement learning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems, 36, 2024.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Tianqi Liu, Yao Zhao, Rishabh Joshi, Misha Khalman, Mohammad Saleh, Peter J Liu, and Jialu Liu. Statistical rejection sampling improves preference optimization. arXiv preprint arXiv:2309.06657, 2023.

Fahim Tajwar, Anikait Singh, Archit Sharma, Rafael Rafailov, Jeff Schneider, Tengyang Xie, Stefano Ermon, Chelsea Finn, and Aviral Kumar. Preference fine-tuning of llms should leverage suboptimal, on-policy data. arXiv preprint arXiv:2404.14367, 2024.

Tianqi Liu, Zhen Qin, Junru Wu, Jiaming Shen, Misha Khalman, Rishabh Joshi, Yao Zhao, Mohammad Saleh, Simon Baumgartner, Jialu Liu, et al. Lipo: Listwise preference optimization through learning-to-rank. arXiv preprint arXiv:2402.01878, 2024.

Feifan Song, Bowen Yu, Minghao Li, Haiyang Yu, Fei Huang, Yongbin Li, and Houfeng Wang. Preference ranking optimization for human alignment. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 18990–18998, 2024.

Haoran Xu, Amr Sharaf, Yunmo Chen, Weiting Tan, Lingfeng Shen, Benjamin Van Durme, Kenton Murray, and Young Jin Kim. Contrastive preference optimization: Pushing the boundaries of llm performance in machine translation. arXiv preprint arXiv:2401.08417, 2024c.

Jiwoo Hong, Noah Lee, and James Thorne. Reference-free monolithic preference optimization with odds ratio. arXiv preprint arXiv:2403.07691, 2024.

Alexey Gorbatovski, Boris Shaposhnikov, Alexey Malakhov, Nikita Surnachev, Yaroslav Aksenov, Ian Maksimov, Nikita Balagansky, and Daniil Gavrilov. Learn your reference model for real good alignment. arXiv preprint arXiv:2404.09656, 2024.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

Dave Epstein, Allan Jabri, Ben Poole, Alexei Efros, and Aleksander Holynski. Diffusion self-guidance for controllable image generation. Advances in Neural Information Processing Systems, 36:16222–16239, 2023.

Mingi Kwon, Jaeseok Jeong, and Youngjung Uh. Diffusion models already have a semantic latent space. In The Eleventh International Conference on Learning Representations, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems, 35:26565–26577, 2022.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Shufan Li, Konstantinos Kallidromitis, Akash Gokul, Yusuke Kato, and Kazuki Kozuka. Aligning diffusion models by

optimizing human utility. arXiv preprint arXiv:2404.04465, 2024. Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score

v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.

Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022.

