# arXiv:2502.00473v3[cs.LG]24Apr2025

## Weak-to-Strong Diffusion with Reflection

[Figure 1]

##### Weak-to-Strong Generation : Motivation

Lichen Bai1 Masashi Sugiyama23 Zeke Xie1

Can we obtain a ideal model from the combination of a strong model and a weak model? Weak Model Strong Model Ideal Model

### Abstract

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

The goal of diffusion generative models is to align the learned distribution with the real data distribution through gradient score matching. However, inherent limitations in training data quality, modeling strategies, and architectural design lead to inevitable gap between generated outputs and real data. To reduce this gap, we propose Weak-to-Strong Diffusion (W2SD), a novel framework that utilizes the estimated difference between existing weak and strong models (i.e., weak-to-strong difference) to bridge the gap between an ideal model and a strong model. By employing a reflective operation that alternates between denoising and inversion with weak-tostrong difference, we theoretically understand that W2SD steers latent variables along sampling trajectories toward regions of the real data distribution. W2SD is highly flexible and broadly applicable, enabling diverse improvements through the strategic selection of weak-to-strong model pairs (e.g., DreamShaper vs. SD1.5, good experts vs. bad experts in MoE). Extensive experiments demonstrate that W2SD significantly improves human preference, aesthetic quality, and prompt adherence, achieving SOTA performance across various modalities (e.g., image, video), architectures (e.g., UNet-based, DiT-based, MoE), and benchmarks. For example, Juggernaut-XL with W2SD can improve with the HPSv2 winning rate up to 90% over the original results. Moreover, the performance gains achieved by W2SD markedly outweigh its additional computational overhead, while the cumulative improvements from different weak-to-strong difference further solidify its practical utility and deployability. The code is publicly available at github.com/xie-lab-ml/Weakto-Strong-Diffusion-with-Reflection.

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

∆1= weak-to-strong difference ∆2= strong-to-ideal difference

Figure 1. W2SD leverages the gap between weak and strong models to bridge the gap between strong and ideal models.

### 1. Introduction

In probabilistic modeling, the estimated density represents model-predicted distributions while the ground truth density reflects actual data distributions. Diffusion models (Song et al.), known for its powerful generative capabilities and diversity, estimate the gradient of the log probability densities in perturbed data distributions by optimizing a score-based network, which has become a mainstream paradigm in many generative tasks (Podell et al.; Guo et al., 2023).

To enhance the alignment between the learned and real data distributions in diffusion models, extensive research has focused on improving the gradient alignment between estimated and ground truth densities through various strategies (Karras et al., 2022; Song & Ermon, 2020; Ho et al., 2022). However, constrained by practical limitations including architectural design and dataset quality issues, existing diffusion models inevitably exhibit gradient estimation errors, leading to a gap between the learned and real data distributions. Depending on the magnitude of this gap, we can classify them as either “strong” or “weak” diffusion models. Specifically, due to the inaccessibility of the real data distribution, we cannot establish direct methodologies to measure this gap and effectively reduce it.

In this work, we demonstrate that the gap between the strong and weak models can be used to empirically bridge the gap between the ideal and strong models. In Figure 1, we propose to use the estimated weak-to-strong difference ∆1 to bridge the inaccessible strong-to-ideal difference ∆2, thereby enhancing the strong model toward the ideal model.

1xLeaF Lab, The Hong Kong University of Science and Technology (Guangzhou) 2RIKEN AIP 3The University of Tokyo. Correspondence to: Zeke Xie <zekexie@hkust-gz.edu.cn>.

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

W2SD(Ours) StandardModel

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Text: Descriptions: Counting:

Color:

Position:

Object co-occurrence:

A sign that says 'Deep Learning'.

A tiger in a lab coat with a 1980s Miami vibe.

Two cats and two dogs sitting on the grass.

A pink oven and a green motorcycle.

a photo of a sports ball right of an umbrella.

a photo of a skateboard and a cake.

- Figure 2. The qualitative results of W2SD demonstrate the effectiveness of our method in various aspects, such as text rendering, position, color, counting, and object co-occurrence. We present more cases in Appendix C.2.

Further evidence supporting the viability of using ∆1 to bridge ∆2 is provided in Appendix E.1.

We note this weak-to-strong concept has been widely applied in training processes of machine learning, with early instances such as AdaBoost (Schapire, 2013), where the ensembling of weak models enhances the performance of strong models. Also, Burns et al. (2023) have demonstrated that weak models can serve as supervisory signals to assist in the alignment during the training of large LLMs.

To empirically estimate the weak-to-strong difference, we draw upon reflective mechanisms, the process of modifying generated outputs based on prior states, which have been extensively studied in the field of LLMs (Gou et al.; Madaan et al., 2024; Shinn et al., 2024). Specifically, we propose Weak-to-Strong Diffusion (W2SD), a novel framework that leverages weak-to-strong difference to bridge the strongto-ideal difference. By incorporating a reflective operation that alternates between denoising and inversion based on the weak-to-strong difference, we theoretically understand in Section 3.1 that W2SD guides latent variables along sampling trajectories, effectively steering them toward regions of the real data distribution. And we provide qualitative results in Figure 2.

We emphasize that W2SD can be generalized to diverse application scenarios. Notably, in Appendix E.3 we demonstrate existing inference enhancement methods including Re-Sampling (lug, 2022), Z-Sampling (Bai et al., 2024), FreeDom (Yu et al., 2023) and TFG (Ye et al., 2024) can be reinterpreted as specialized instances of W2SD. Users can define “weak-to-strong” model pairs based on their specific needs. Depending on the type of the model pair (e.g., DreamShaper vs. SD1.5, good experts vs. bad experts in MoE), different effects of improvements can be achieved. We provide further application analysis in Section 4 to demonstrate the broad applicability of W2SD.

The contributions can be summarized as follows.

First, we introduce the weak-to-strong concept into the inference enhancement of diffusion models, demonstrating that the gradient difference of estimated log probability densities between weak and strong diffusion models can approximate the difference between strong and ideal models, consequently bridging the gap between the learned and real data distribution.

Second, to estimate the weak-to-strong difference, we propose a novel inference framework called W2SD, with theoretical understanding that implicitly estimates the weakto-strong difference through iterative reflection, effectively steering latent variables along sampling trajectories toward regions corresponding to the real data distribution.

Third, extensive experiments validate the effectiveness and broad applicability of W2SD across various generation tasks (e.g., image, video), architectures (e.g., UNet-based, DiTbased), and evaluation metrics. By defining various weak-tostrong model pairs, W2SD achieves diverse improvements effects, such as human preference, prompt adherence, and personalization. Importantly, these improvements effects can be cumulative in parallel, further enhancing the quality of the generation. In efficiency evaluations, W2SD’s improvements surpass its overhead, maintaining superior quality over the baseline under equal time constraints, suggesting its strong versatility and applicability.

### 2. Preliminaries

In this section, we present preliminaries about denoising and inversion operation in diffusion models (Song et al.; Ho et al., 2020). Due to page limitations, we introduce the related work in Appendix A.

Given the random Gaussian noise zt, we denote the forward

√

∆tzt, where t ∈ [0,1], and σ represents the predefined variance. We denote the ground truth density of real data distribution

process of diffusion models as xt = xt−∆t + σt

gap between existing diffusion models and the ideal model, bringing the learned distribution closer to the real data distribution. In the next subsection, we introduce how to achieve this approximation from a reflective perspective.

- as pgt0 , After noise addition at time t, the resulting density is represented as pgtt . Following Song et al. (2020), we can obtain the denoised

results xt−∆t from noisy data xt through the process of an ordinary differential equation as

xt−∆t = M(xt,t) (1) = xt + σ2tsθ(xt,t)∆t, t ∈ [0,1]. (2)

where sθ(·,·) represents the trained score network, utilized to predict the score at the time t. Similarly, we can invert Equation (2) to transform xt−∆t back to a new x˜t as

x˜t = Minv(xt−∆t,t), (3) = xt−∆t − σ2tsθ(xt,t)∆t, (4) ≈ xt−∆t − σ2tsθ(xt−∆t,t)∆t, t ∈ [0,1]. (5)

In practice, we often approximate the score value predicted

- at time t with time t − ∆t along the inversion process, i.e.,

W2SD Consider a strong model Ms and a weak model Mw, we can optimize the sampling trajectories using the reflection operator Mwinv(Ms(·)), as outlined in Algorithm 1. Through the iterative integration of strong model denoising and weak model inversion, we achieve a step-by-step reflective process during sampling process, refining the latent variable xt into an improved x˜t.

Algorithm 1 W2SD Input: Strong Model Ms, Weak Model Mw, Total Inference Steps: T, optimization steps: λ Output: Clean Data x0 Sample Gaussian noise xT for t = T to 1 do

if t > T − λ then #W2SD with Reflection x˜t = Mwinv(Ms(xt,t),t)

end if xt−1 = Ms(˜xt,t)

sθ(xt,t) ≈ sθ(xt−∆t,t) in Equation (5). Given that the approximation error is negligible and the same socre netowrk sθ, M and Minv can be treated as mutually inverse mappings, thereby satisfying Minv(M(xt,t),t) = xt. In Appendix E.2, we conduct a detailed analysis of the impact caused by this approximation error.

###### end for

Importantly, the choice of Ms and Mw significantly affects the direction of improvements effects. We summarize some promising weak-to-strong model pairs in Table 7 of Appendix B.3. And in Section 4 we present extensive application analyses and experiments, highlighting the powerful capabilities and flexibility of the proposed framework.

### 3. Method

In this section, we discuss the proposed Weak-to-Strong Diffusion and its visual explanation.

In Theorem 1, we demonstrate that W2SD refines latent variable xt toward the direction defined by the estimated weak-to-strong difference ∆1(t). As shown in Figure 3, when the weak-to-strong difference closely approximates the strong-to-ideal difference (i.e., ∆2(t) − ∆1(t) is small), the reflection mechanism of W2SD drives xt closer to the ideal xgtt . The visualization analysis in Section 3.2 validates the correctness of our theory, and we provide a detailed proof in Appendix D.1.

###### 3.1. Weak-to-Strong Diffusion

In this subsection, we integrate the weak-to-strong concept into diffusion model inference, introduce the W2S algorithm, and establish its theoretical understanding.

The Difference of Estimated Density Gradients In diffusion models, the goal is to minimize the gradients difference of log probability densities between estimated results and ground truth. However, due to the real data distribution is inaccessible, this difference cannot be directly quantified. To tackle this issue, we consider models of varying capacities: a strong model (with its corresponding denoising process denoted as Ms and the estimated density as ps) and a weak model (similarly, Mw and pw).

Theorem 1 (Theoretical Understanding of W2SD). Suppose xt is the latent variable at time t, let pst and pwt denote the probability density estimates derived from Ms and Mw. The reflective operator Mwinv(Ms(·)) refines xt to x˜t as

log pwt (xt)), (6) where ∆1(t) = ∇xt

x˜t = xt + σ2t∆t(∇xt

log pst(xt) − ∇xt

log pwt (xt) means the weak-to-strong difference between Ms and Mw at time t.

log pst(xt)−∇xt

As shown in Figure 1, we define the weak-to-strong difference as ∆1 = ∇log ps − ∇log pw and the strong-to-ideal difference as ∆2 = ∇log pgt − ∇log ps. By approximating ∆2 using the estimable ∆1, we indirectly reduce the

It is noted that Karras et al. (2024) proposed Auto-guidance, which employs a simplistic interpolation approach using a

1D Diffusion trajectories via different sampling strategies

###### :ℳ𝑠 (∙) :ℳ𝑖𝑛𝑣𝑤 (∙)

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

Weak Operation

𝑥𝑡−𝛥𝑡

Strong Operation

Start Point

End Point

timesteps(T=10)

|𝑥𝑡𝑔𝑡|
|---|

|𝑥𝑡|
|---|

𝑥෤𝑡

|𝛥1(𝑡)|
|---|

- Figure 3. Visualizing the effectiveness of W2SD. When the weakto-strong difference closely approximates the strong-to-ideal difference (e.g., ∆2(t) − ∆1(t) is small), the refined latent variable x˜t converges to the ideal latent variable xgtt .

4 3 2 1 0 1 2 3 4 Predicted value

Figure 4. Denoising trajectories across different settings (1-D Gauss). The weak model (blue) generates only right-peak data due to missing left-peak training samples, while the strong model (red) produces data between both peaks. W2SD balances the distribution by leveraging the reflective operator Mwinv(Ms(·)). Additional examples are provided in Figure 18.

degraded model version to refine latent variables. While they shares a similar high-level concept, W2SD introduces fundamentally distinct mechanisms, delivering significantly stronger performance and broader applicability. Comprehensive quantitative and qualitative analyses are provided in Appendix E.4.

of W2SD.

###### 3.2. Visualization and Explanation in Various Settings

Real Image Data Furthermore, we investigate the performance of W2SD on real image data. For ease of analysis, we select two classes from CIFAR-10 (Krizhevsky et al., 2009): car and horse. Specifically, we train the weak and strong diffusion models on distinct datasets. For Ms, the dataset consists of 5,000 horse images and 2,500 car images. For Mw, it comprises 5,000 horse images and 500 car images. In this scenario, due to the imbalance in training dataset, both Mw and Ms are more inclined to generate horses. Moreover, since Mw is trained on a limited number of car images, it rarely generates cars.

In this subsection, we validate the theory presented in Section 3.1 using both synthetic Gaussian mixture data and real-world image data, providing intuitive visual evidence.

###### 1-D Gaussian Mixture Data We begin by analyzing the

- 1-D Gaussian data scenario. In Figure 1, the diffusion model is designed to generate data with two distinct peaks at “-4” and “4”. Adjusting the proportion of these two peaks in the training dataset, we obtain Ms and Mw. Although both models effectively generate samples near the right peak, their performance differs significantly for the left peak.

In Figure 4, we visualize the denoising trajectories under three different settings (weak model, strong model, W2SD). Through the reflective operator Mwinv(Ms(·)), the latent variable is progressively drawn closer to the left peak. In contrast, for both strong and weak models, the generated samples are predominantly concentrated around the right peak.

- 2-D Gaussian Mixture Data We also visualize the mechanism of W2SD on 2D scenario. In Figure 5 (first row), we modulate the proportion of the training data to obtain Ms and Mw. W2SD balances the distribution by exploiting the discrepancy between Ms and Mw (the region in the bottom-left corner, denoted as ∆1) to approximate the unattainable strong-to-ideal difference ∆2, and boost the chances of sampling toward the bottom-left region. In Figure 5 (second row), we visualize the denoising trajectories under different settings, further validating the effectiveness

In Figure 6, we note that W2SD can help balance the image categories, enhancing inference process to increase the probability of generating cars. In Figure 7, we also perform a t-SNE dimensionality reduction (Van der Maaten & Hinton, 2008) on the CLIP features of the generated images. W2SD effectively disentangles the representations of “car” and “horse” in the 2D space. Notably, the ratio of “horse” to “car” under W2SD approaches 1:1. In contrast, applying the negative reflection operator Msinv(Mw(·)) worsens the data imbalance, validating the effectiveness of our method.

### 4. Empirical Analysis

In this section, we justify our design choices in Section 3 and illustrate the wide applicability of W2SD across diverse combinations of Ms and Mw.

Due to page limitations, here we validate the effectiveness of W2SD using Pick-a-Pic Dataset (Kirstain et al., 2023). In Appendix B, we provide a detailed description of the

(a) Strong Model (b) Weak Model

Strong Model Weak Model W2SD (Ours)

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

(c) W2SD (d) S2WD

[Figure 31]

[Figure 32]

- Figure 5. Probability contour plot and denoising trajectories across different settings (2-D Gauss). W2SD balances the learned distribution, bringing it closer to the real data distribution

.

[Figure 33]

|ℳ𝑠|
|---|

|ℳ𝑤|
|---|

[Figure 34]

[Figure 35]

W2SD (Ours)

[Figure 36]

S2WD

- Figure 6. Qualitative results of W2SD based on dataset differences (CIFAR-10). Our method enhances the probability of generating “cars” and promote a more balanced generation distribution.

Figure 7. The CLIP feature corresponding to the generated image (32×32×3) is projected into a 2D space. W2SD effectively disentangles the representations of “car” and “horse” in the 2D space. (a) Ms demonstrates the ability to generate cars; (b) Mw can hardly generate cars; (c) W2SD balances the generation distribution, increasing the likelihood of generating cars; (d) S2WD (i.e., Msinv(Mw(·))) exacerbates the imbalance in data generation.

improvements in human preference metrics such as HPS v2 (Wu et al., 2023b) and PickScore (Kirstain et al., 2023).

LoRA Mechanism Furthermore, W2SD is also applicable to efficiently fine-tuned model. we select the personalized models derived through the LoRA mechanism (Hu et al.) as Ms, and employ W2SD to attain a more robust and customized personalization effect. We test 20 LoRA checkpoints across object, person, animal, and style categories, with details in Appendix B.3. Table 2 demonstrates that W2SD results in significant improvements across multiple personalization metrics (e.g., Clip-T and Clip-I). And we present the qualitative results in Figure 8, with more visual cases provided in Appendix C.2.

experimental settings. And in Appendix C, we conduct more extensive quantitative and qualitative results across diverse benchmarks (e.g., Drawbench (Saharia et al., 2022)) and modalities (e.g., video generation), demonstrating the broad applicability of our method.

###### 4.1. Weight Difference

The capability differences between models can be directly captured by their parameter weights. And W2SD leverages this weight difference to empirically estimate the weak-tostrong difference, enabling effective reflective operations.

Full Parameter Fine-tuning We first select the full parameter fine-tuned models (e.g., DreamShaper, JuggernautXL) that align more closely with human preferences as Ms, and the corresponding standard models as Mw (e.g., SD1.5 (Rombach et al., 2022), SDXL (Podell et al.)). We evaluate our method with various metrics, as shown in Table 1, W2SD based on weight difference shows significant

MoE Mechanism We also note that a weak-to-strong difference can be induced by controlling the expert selection strategy within the MoE mechanism. Specifically, we focus on DiT-MoE (Fei et al., 2024), a novel architecture that integrates multiple experts and selects the two highestperforming experts during each denoising step to optimize generation quality. Based on this framework, we use DiTMoE-S as Ms and define Mw as the two lowest-performing experts in each denoising step, thereby establishing a quantifiable weight difference.

In Table 3, we show that W2SD reduces FiD (Seitzer, 2020)

#### Clothing Character

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Style A dog in a batman costume. Detail Bulma is biking, pink dress, blue hair.

[Figure 41]

[Figure 42]

A sandwich styled in the manner of a parchment scroll. mechanical bee flying in nature, electronics, motors.

- Figure 8. Qualitative comparisons with weak model (left), strong model (middle) and W2SD based on weight difference (right). Our method utilizes the differences between chosen strong and weak models (e.g., high-detail LoRA vs. standard model) to deliver improvements in various dimensions, including style, character, clothing, and beyond. We provide more qualitative results in Appendix C.2.

Table 1. Quantitative results of W2SD based on a full parameter fine-tuning strategy. Our method generates results better aligned with human preferences. Datasets: Pick-a-Pic.

Method HPS v2 ↑ AES ↑ PickScore ↑ MPS ↑

SD1.5 24.9558 5.5003 20.1368 DreamShaper 30.1477 6.1155 21.5035 46.8705 W2SD 30.4924 6.2478 21.5727 53.1304 SDXL 29.8701 6.0939 21.6487 Juggernaut-XL 31.6412 5.9790 22.1903 45.7397

W2SD 32.0992 6.0712 22.2434 54.2634

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

- Figure 9. Quantitative Results of W2SD Based on the MoE Mechanism. The first row shows the results for DiT-MoE-S, while the second row presents W2SD. W2SD achieves significant improvements, even with small models featuring 71M activated parameters.

- Table 2. Quantitative results of W2SD based on personalized LoRA model. Here, the weight difference between Mw (SD1.5) and Ms (SD1.5 +LoRA) biases the generated results towards a more personalized direction.

Method DINO ↑ CLIP-I ↑ CLIP-T ↑

SD1.5 27.47 52.08 20.14 Personalized LoRA 48.03 64.37 25.99

W2SD 51.58 68.04 27.66

- Table 3. Quantitative results of W2SD based on MoE Mechanism. Datasets: ImageNet 50K.

Method IS ↑ FiD ↓ AES ↑ HPS v2 ↑ DiT-MoE-S 45.4437 15.1032 4.4755 20.0486

W2SD 55.5341 9.1001 4.5053 22.3225

degradation and distortion (the first row in Figure 9), the application of W2SD significantly enhances the image quality.

- 4.2. Condition Difference

Here we demonstrate that the weak-to-strong difference applies not only across different weights but also within the same diffusion model under the weak-to-strong conditions.

from 15.1032 to 9.1001, aligning the learned distribution closer to the real data distribution. Additionally, as shown in Figure 9, although the limited capacity of MoE-DiT-S (with only 71M active parameters) often results in image

Guidance Scale The classifier-free guidance mechanism (Ho & Salimans, 2021) extrapolates between conditional and unconditional noise predictions, adjusting the

lies solely on the semantic differences between prompts, achieves improvements across all metrics. And in Figure 16 of Appendix C.2 we illustrate that W2SD is capable of more precisely capturing the intricate details based on the differences in prompt pairs.

- Table 4. Quantitative results of W2SD based on guidance difference. Model: SDXL. Datasets: Pick-a-Pic.

Method HPS v2 ↑ AES ↑ PickScore ↑ MPS ↑

SD1.5 24.9558 5.5003 20.1368 42.1101 W2SD 25.5069 5.5073 20.2443 57.8903

SDXL 29.8701 6.0939 21.6487 43.9425 W2SD 31.2020 6.0970 21.7980 56.0608

- Table 5. Quantitative results of W2SD based on semantic differences between prompts. Model: SDXL. Datasets: GenEval.

Prompt Type HPS v2 ↑ AES ↑ PickScore ↑ MPS ↑ Raw Prompt 25.3897 5.4454 20.7144 -

Refined Prompt 28.5698 5.7714 21.6350 45.7719 W2SD 29.4023 5.8812 21.8053 54.2275

- Table 6. The improvements effects from different model differences can be cumulative. Datasets: Pick-a-Pic.

###### 4.3. Sampling Pipeline Difference

Extensive works (Si et al., 2024; Chefer et al., 2023; Zhang et al., 2023) have been devoted to design advanced inference pipelines to improve the quality of diffusion generation results. We demonstrate that W2SD can enhance the inference process by leveraging the difference derived from these powerful existing pipelines. Here we define those enhanced sampling methods as Ms, while Mw represents the standard sampling (Song et al., 2020; Lu et al., 2022).

We first select ControlNet (Zhang et al., 2023) as Ms, which incorporates additional network structures into the pipeline. By utilizing reference images (e.g., edge maps), it facilitates controllable image generation. As shown in Figure 10, the images generated by W2Sd exhibit a stronger alignment with the provided edge maps. We also present visualization cases of other pipelines (e.g., Ip-adapter (Ye et al., 2023)) as strong models in Figure 17 of Appendix C.2.

Weight Difference Condition Difference HPS v2 ↑ Winning Rate ↑

× × 31.6412 × ✓ 32.8217 84% ✓ × 32.0992 76% ✓ ✓ 32.9623 90%

Edge Map ControlNet W2SD (Ours)

[Figure 53]

guidance scale to control the semantic attributes of the generated images. To apply W2SD, diffusion models under high guidance scale can be viewed as Ms, those under low or even negative guidance scale can be treated as Mw. This guidance scale discrepancy constructs a weak-to-strong difference that aligns the learned generated distribution more closely with the semantic conditions of the prompt.

A little blue sprite is laughing.

[Figure 54]

We note that Z-Sampling (Bai et al., 2024) is a specific instance of W2SD under the guidance difference. In Table 4, our method significantly enhances human preference (e.g., HPS v2) and aesthetic characteristics (e.g., AES) in mainstream models such as SDXL and SD1.5.

Two purple oranges.

Semantics of Prompts Aside from adjusting the guidance scale, we demonstrate that the semantic differences within the condition prompts themselves can also be leveraged to enhance generation quality through the reflection process.

Figure 10. Qualitative results of W2SD based on pipeline difference. We set ControlNet as Ms, DDIM as Mw. W2SD improves alignment with reference images.

###### 4.4. Cumulative Effects of Different Model Differences

Specifically, we select a total of 533 text prompts from the GenEval (Ghosh et al., 2024) for Mw, with an average length ranging from 5 to 6 words and minimal contextual complexity. Additionally, we leverage LLM (here we use Qwen-Plus (Yang et al., 2024)) to enrich and refine these prompts with additional details for Ms. resulting in weakto-strong prompt pairs.

Finally, it is important to note that the improvements effects of W2SD, derived from different type of differences, can be even cumulative. Here we apply the weight difference and the condition difference simultaneously, where Ms represents the fine-tuned model Juggernaut-XL with high guidance scale at 5.5, while Mw represents the standard model SDXL with low guidance scale at zero. As shown

We report the quantitative results in Table 5, which re-

34.0

32.0

W2SD

baseline

30

33.5

31.5

33.0

28

31.0

HPSv2

HPSv2

HPSv2

30.5

32.5

26

30.0

32.0

24

29.5

31.5

W2SD

22

29.0

baseline

Standard Sampling

31.0

W2SD

2 0 2 4

0 10 20

28.5

The difference in LoRA Scale

The difference in Guidance Scale

4 6 8 10 12

Time Spent per Image (s)

- Figure 11. The magnitude of weak-to-strong difference is a key factor impacting the effects of improvements. The horizontal axis shows the magnitude of the weak-to-strong difference, while the vertical axis shows the average HPS v2 on the Pick-a-Pic. When Ms is weaker than Mw, W2SD results in negative gains.

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Base model Guidance Difference = -3 Guidance Difference = 0 Guidance Difference = 3

Base model LoRA Difference = -2 LoRA Difference = 0 LoRA Difference = 2

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

- Figure 12. When the weak-to-strong difference is greater than 0, W2SD yields positive gains. When it equals 0, the process degenerates into standard sampling. When it is less than 0, negative gains occurs, resulting in poor image quality.

Figure 13. W2SD outperforms standard sampling with identical time costs. The horizontal axis denotes the average generation time per image, the vertical axis represents the HPS v2 on Pick-a-Pic.

In addition, as shown in Figure 11 (right), we also quantify the weak-to-strong difference based on guidance scale. By fixing the guidance scale of Ms at 5.5 and adjusting that of Mw (e.g., from -10 to 15), we observe similar phenomena. In Figure 12, we present a qualitative analysis showing that the magnitude of the weak-to-strong difference is a key factor that influences the quality of generation results.

###### 5.2. Time Efficiency Comparison

Assuming standard sampling and W2SD require Tstd and Tw2s denoising steps, respectively, the score predictions needed are Tstd and Tw2s +2λ. To ensure a fair comparison, we set Tw2s = ⌊12Tstd⌋ and λ = ⌊21Tw2s⌋, matching the runtime for generating the same image. In this setting, as shown in Figure 13, W2SD consistently outperforms standard sampling, demonstrating that the gains from reflection operations far outweigh the additional computational cost, validating the time efficiency of our method.

in Table 6, the combination of guidance difference and condition difference leads to a substantial improvement in Pick-a-Pic Dataset compared to Ms, achieving a winning rate of up to 90% on HPS v2.

###### 5.3. Appendix Overview

### 5. More Detailed Analysis

In this section, we perform more detailed studies on W2SD, to further validate the effectiveness of our theory.

###### 5.1. The Magnitude of Weak-to-Strong Difference

In this subsection, we explore how the magnitude of weakto-strong difference affects the improvements effects.

To quantify the difference between Ms and Mw, we first consider the LoRA-based W2SD approach. We fix the LoRA scale of Ms at 0.8 and adjust the LoRA scale of Mw (e.g., from -3.5 to 3.5) to observe its impact on the effects of improvements. As shown in Figure 11 (left), when the LoRA scale of Ms exceeds that of Mw, W2SD improves performance. However, when Ms is weaker than Mw (i.e., the LoRA scale difference is negative), the improvements effects diminish or even result in the negative gains.

Due to page limitations, further details are provided in the appendix: Appendix E.2 analysis the impact of inversion approximation errors on performance improvement, Appendix E.3 analyzes the connections between W2SD, ReSampling (lug, 2022), and other inference methods.

### 6. Conclusion

To the best of our knowledge, this work is the first to systematically integrate the weak-to-strong concept into the inference enhancement of diffusion models via a reflection mechanism. We theoretically and empirically understand that the estimated weak-to-strong difference can effectively bridge the strong-to-ideal difference, enhancing the alignment between the learned distributions from existing diffusion models and the real data distribution. Building on this concept, we propose W2SD, which utilizes the estimated difference in density gradients to optimize sampling

trajectories via reflective operations. W2SD demonstrates its effectiveness as a general-purpose framework through its cumulative performance improvements, flexible definition of weak-to-strong model pairs, and efficient performance gains with minimal computational overhead. We hope this work inspires interest in diffusion model scaling laws, advances sampling methods, and deepens understanding of probabilistic modeling in generative models.

### Impact Statement

This work introduces W2SD, a novel framework designed to enhance the inference process of diffusion models. The data and models utilized in our work are released under opensource licenses and sourced from open platforms. While our work may have various societal implications, it does not introduce additional ethnic concerns compared to existing standard sampling methods. As such, none which we feel must be specifically highlighted here.

### References

Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11461– 11471, 2022.

Ahn, D., Kang, J., Lee, S., Min, J., Kim, M., Jang, W., Cho, H., Paul, S., Kim, S., Cha, E., et al. A noise is worth diffusion guidance. arXiv preprint arXiv:2412.03895, 2024.

Bai, L., Shao, S., Zhou, Z., Qi, Z., Xu, Z., Xiong, H., and Xie, Z. Zigzag diffusion sampling: Diffusion models can self-improve via self-reflection. arXiv preprint arXiv:2412.10891, 2024.

Bansal, A., Chu, H.-M., Schwarzschild, A., Sengupta, S., Goldblum, M., Geiping, J., and Goldstein, T. Universal guidance for diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 843–852, 2023.

Burns, C., Izmailov, P., Kirchner, J. H., Baker, B., Gao, L., Aschenbrenner, L., Chen, Y., Ecoffet, A., Joglekar, M., Leike, J., et al. Weak-to-strong generalization: Eliciting strong capabilities with weak supervision. arXiv preprint arXiv:2312.09390, 2023.

Chefer, H., Alaluf, Y., Vinker, Y., Wolf, L., and Cohen-Or, D. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models, 2023.

Chen, J., Ge, C., Xie, E., Wu, Y., Yao, L., Ren, X., Wang, Z., Luo, P., Lu, H., and Li, Z. Pixart-σ: Weak-to-strong training of diffusion transformer for 4k text-to-image

generation. In European Conference on Computer Vision, pp. 74–91. Springer, 2025.

Chen, Z., Deng, Y., Yuan, H., Ji, K., and Gu, Q. Selfplay fine-tuning converts weak language models to strong language models. In Forty-first International Conference on Machine Learning.

Fei, Z., Fan, M., Yu, C., Li, D., and Huang, J. Scaling diffusion transformers to 16 billion parameters. arXiv preprint arXiv:2407.11633, 2024.

Ghosh, D., Hajishirzi, H., and Schmidt, L. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36, 2024.

Gou, Z., Shao, Z., Gong, Y., Yang, Y., Duan, N., Chen, W., et al. Critic: Large language models can self-correct with tool-interactive critiquing. In The Twelfth International Conference on Learning Representations.

Green Larsen, K. and Ritzert, M. Optimal weak to strong learning. Advances in Neural Information Processing Systems, 35:32830–32841, 2022.

Guo, Y., Yang, C., Rao, A., Liang, Z., Wang, Y., Qiao, Y., Agrawala, M., Lin, D., and Dai, B. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. In The Twelfth International Conference on Learning Representations.

Guo, Y., Yang, C., Rao, A., Liang, Z., Wang, Y., Qiao, Y., Agrawala, M., Lin, D., and Dai, B. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.

He, Y., Murata, N., Lai, C.-H., Takida, Y., Uesaka, T., Kim, D., Liao, W.-H., Mitsufuji, Y., Kolter, J. Z., Salakhutdinov, R., et al. Manifold preserving guided diffusion. arXiv preprint arXiv:2311.16424, 2023.

Ho, J. and Salimans, T. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., and Fleet, D. J. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022.

Høgsgaard, M. M., Larsen, K. G., and Ritzert, M. Adaboost is not an optimal weak to strong learner. In International Conference on Machine Learning, pp. 13118– 13140. PMLR, 2023.

Hu, E. J., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., Wang, Y., Chen, X., Wang, L., Lin, D., Qiao, Y., and Liu, Z. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., Tunyasuvunakool, K., Bates, R., Z´ˇıdek,

- A., Potapenko, A., et al. Highly accurate protein structure prediction with alphafold. nature, 596(7873):583–589, 2021.

Karras, T., Aittala, M., Aila, T., and Laine, S. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35: 26565–26577, 2022.

Karras, T., Aittala, M., Kynk¨a¨anniemi, T., Lehtinen, J., Aila, T., and Laine, S. Guiding a diffusion model with a bad version of itself. Advances in Neural Information Processing Systems, 37:52996–53021, 2024.

Kirstain, Y., Polyak, A., Singer, U., Matiana, S., Penna, J., and Levy, O. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:36652–36663, 2023.

Krizhevsky, A., Hinton, G., et al. Learning multiple layers of features from tiny images. 2009.

Liu, B., Shao, S., Li, B., Bai, L., Xu, Z., Xiong, H., Kwok, J., Helal, S., and Xie, Z. Alignment of diffusion models: Fundamentals, challenges, and future. arXiv preprint arXiv:2409.07253, 2024.

Lou, A. and Ermon, S. Reflected diffusion models. In International Conference on Machine Learning, pp. 22675–

22701. PMLR, 2023.

Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., and Zhu, J. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35:5775–5787, 2022.

Ma, N., Tong, S., Jia, H., Hu, H., Su, Y.-C., Zhang, M., Yang, X., Li, Y., Jaakkola, T., Jia, X., et al. Inference-time scaling for diffusion models beyond scaling denoising steps. arXiv preprint arXiv:2501.09732, 2025.

Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S.,

Yang, Y., et al. Self-refine: Iterative refinement with selffeedback. Advances in Neural Information Processing Systems, 36, 2024.

Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., M¨uller, J., Penna, J., and Rombach, R. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Sadat, S., Hilliges, O., and Weber, R. M. Eliminating oversaturation and artifacts of high guidance scales in diffusion models. In The Thirteenth International Conference on Learning Representations, 2024.

Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E. L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35: 36479–36494, 2022.

Schapire, R. E. Explaining adaboost. In Empirical inference: festschrift in honor of vladimir N. Vapnik, pp. 37–52. Springer, 2013.

Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35: 25278–25294, 2022.

Seitzer, M. pytorch-fid: FID Score for PyTorch. https:// github.com/mseitzer/pytorch-fid, August

2020. Version 0.3.0.

Shao, S., Zhou, Z., Bai, L., Xiong, H., and Xie, Z. Iv-mixed sampler: Leveraging image diffusion models for enhanced video synthesis. arXiv preprint arXiv:2410.04171, 2024.

Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., and Yao, S. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36, 2024.

Si, C., Huang, Z., Jiang, Y., and Liu, Z. Freeu: Free lunch in diffusion u-net. In CVPR, 2024.

Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

Song, Y. and Ermon, S. Improved techniques for training score-based generative models. Advances in neural information processing systems, 33:12438–12448, 2020.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations.

Sugiyama, M., Kanamori, T., Suzuki, T., Du Plessis, M. C., Liu, S., and Takeuchi, I. Density-difference estimation. Neural Computation, 25(10):2734–2775, 2013.

Zhang, S., Wang, B., Wu, J., Li, Y., Gao, T., Zhang, D., and Wang, Z. Learning multi-dimensional human preference for text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8018–8027, 2024.

Zhou, Z., Shao, S., Bai, L., Xu, Z., Han, B., and Xie, Z. Golden noise for diffusion models: A learning framework. arXiv preprint arXiv:2411.09502, 2024.

Van der Maaten, L. and Hinton, G. Visualizing data using t-sne. Journal of machine learning research, 9(11), 2008.

Wu, J. Z., Ge, Y., Wang, X., Lei, S. W., Gu, Y., Shi, Y., Hsu, W., Shan, Y., Qie, X., and Shou, M. Z. Tune-avideo: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7623– 7633, 2023a.

Wu, X., Hao, Y., Sun, K., Chen, Y., Zhu, F., Zhao, R., and Li, H. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023b.

Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu,

- B., Li, C., Liu, D., Huang, F., Wei, H., et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.

Ye, H., Lin, H., Han, J., Xu, M., Liu, S., Liang, Y., Ma, J., Zou, J., and Ermon, S. Tfg: Unified training-free guidance for diffusion models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Ye, H., Zhang, J., Liu, S., Han, X., and Yang, W. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721,

- 2023.

Ye, H., Lin, H., Han, J., Xu, M., Liu, S., Liang, Y., Ma, J., Zou, J., and Ermon, S. Tfg: Unified trainingfree guidance for diffusion models. arXiv preprint arXiv:2409.15761, 2024.

Yu, J., Wang, Y., Zhao, C., Ghanem, B., and Zhang, J. Freedom: Training-free energy-guided conditional diffusion model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 23174–23184, 2023.

Zhang, L., Rao, A., and Agrawala, M. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3836–3847, 2023.

- A. Related Work In this section, we review existing works relevant to W2SD.

Weak-to-Strong Mechanism The concept of improving weak models into strong models originates from the AdaBoost (Høgsgaard et al., 2023), which constructs a more accurate classifier by aggregating multiple weak classifiers. Building upon this, Green Larsen & Ritzert (2022); Høgsgaard et al. (2023) introduced a provably optimal weak-to-strong learner, establishing a robust theoretical foundation for this weak-to strong paradigm. In the field of LLMs, several studies (Chen et al.; Burns et al., 2023) have utilized weak models as supervisory signals to facilitate the alignment of LLMs. This paradigm of weak-to-strong generation during training has similarly been investigated in the context of diffusion model training (Chen et al., 2025). And Sugiyama et al. (2013) recommended directly estimating the density difference between weak and strong models instead of separate estimations. And Karras et al. (2024) examines the mechanism of CFG and achieves improved results through interpolation-based perturbation. In contrast, W2SD utilizes the reflection by leveraging the gap between denoising and inversion to steer generation towards user-defined directions. And we note that W2SD generalizes the concept of ideal model, with experiments across human preference, personalization, fidelity and so on, confirming its wide applicability.

Diffusion Inference Enhancement The study of inference scaling laws in diffusion models has recently become a prominent focus within the research community (Ma et al., 2025; Ye et al.; Liu et al., 2024). This line of work can be traced back to Re-Sampling (lug, 2022), which iteratively refines latent variables by injecting random Gaussian noise, effectively reverting the noise level to a previous scale. This iterative paradigm has been utilized in subsequent works, including universal conditional control (Bansal et al., 2023), video generation (Wu et al., 2023a), and protein design (Jumper et al., 2021), to enhance inference performance. However, it has primarily been treated as a heuristic trick, with its underlying mechanisms remaining underexplored. Z-Sampling (Bai et al., 2024) extended this paradigm by replacing random noise injection with inversion operations and identified the guidance difference between denoising and inversion as a critical factor. This phenomenon has also been validated in subsequent studies (Zhou et al., 2024; Shao et al., 2024; Ahn et al., 2024). In our work, we systematically unify these inference enhancement methods, demonstrating that their essence lies in approximating the strong-to-ideal difference via the weak-to-strong difference, and integrate them into a unified reflective framework, W2SD, through theoretical and empirical analysis.

- B. Experiment Settings In this section, we introduce the details of hyperparameters, metrics and datasets used in the experiments.

###### B.1. Hyperparameters

Weight Difference For the W2SD based on weight difference, we consider full-parameter tuning, LoRA-based efficient tuning, and the MoE mechanism.

For the full-parameter tuning, we first set SD1.5 (Rombach et al., 2022) as the weak model and the fine-tuned DreamShaper v8 as the strong model, which demonstrates superior performance in terms of human preference and achieves high-quality generation results. Similarly, we also use SDXL (Podell et al.) as the weak model and Juggernaut-XL as the strong model to further evaluate W2SD’s performance under weight differences.

For the efficient tuning, we distinguish strong and weak models by adjusting the LoRA scale. First, we use the xlMoreArtFullV1 LoRA checkpoint to test W2SD’s ability to enhance overall image quality. Additionally, to validate the ability of W2SD to improve personalization, we select a series of personalized LoRAs, as detailed in Appendix B.3. For the strong model, the LoRA scale is set to 0.8, while for the weak model, it is set to -1.5.

Following the default settings (Bai et al., 2024), we set the denoising steps T = 50, and the guidance scale to 5.5. Reflection operations steps λ = T − 1. Notably, to eliminate influence from guidance differences, the guidance scale of Ms and Mw are both set to 1.0 during reflection, ensuring the guidance difference is zero.

For the MoE (Mixture of Experts) mechanism, we select DiT-MoE-S (Fei et al., 2024) as the strong model, which routes the top 2 optimal experts out of 8 during the inference process. The weak model, in contrast, is configured to route the 2 least optimal experts out of 8. Following the default settings, the denoising steps T = 50, and the guidance scale is set to 1.5.

Condition Difference In the W2SD research based on condition differences, we focus on analyzing the differences caused by two mechanisms: guidance scale and prompt semantics.

By adjusting the guidance scale to distinguish the strong and weak model, we adapt the same settings as Z-Sampling (Bai et al., 2024): the guidance scale of Ms was set to 5.5, and the guidance scale of Mw is set to 0. The diffusion model used is SDXL.

For semantic differences, we set Mw to use the GenEval prompt, which is short (4-5 words), ambiguous, and coarse, often resulting in uncontrolled outputs. In contrast, Ms uses refined prompts enhanced by QWen-Plus (Yang et al., 2024), providing greater detail and semantic richness. During the reflection process, the guidance scale was set to 1.0 to eliminate the influence of guidance differences on the results.

Similar to the weight difference setup, we set T = 50, λ = T − 1, and the denoising guidance scale to 5.5.

Sampling Pipeline Difference We demonstrate that W2SD can also generalize to capability differences across different diffusion pipelines. Specifically, we select ControlNet (Zhang et al., 2023) as Ms, with the control scale set to the default value of 1.0. The standard sampling pipeline (Song et al., 2020) is chosen as the weak model. Consistent with the weight difference setup, we configure T = 50, λ = T − 1, and guidance scale of 5.5. During reflection operation, the guidance scale for both Ms and Mw are set to 0.5 to eliminate the influence of guidance differences.

###### B.2. Metrics

AES. AES (Schuhmann et al., 2022) is an evaluation metric that assesses the visual quality of generated images by analyzing key aesthetic attributes such as contrast, composition, color, and detail, thereby measuring their alignment with human aesthetic standards.

PickScore. PickScore (Kirstain et al., 2023) is a CLIP-based metric model trained on a comprehensive open dataset containing text-to-image prompts and corresponding real user preferences for generated images, specifically designed for predicting human aesthetic preferences.

HPS v2. Building upon the Human Preference Dataset v2 (HPD v2), a comprehensive collection of 798,090 human preference judgments across 433,760 image pairs, (Wu et al., 2023b) developed HPS v2 (Human Preference Score v2) through CLIP fine-tuning, establishing a more accurate predictive model for human preferences in generated images.

MPS. MPS (Zhang et al., 2024) is a metric for text-to-image model evaluation, trained on the MHP dataset containing 918,315 human preference annotations across 607,541 images. This novel metric demonstrates superior performance by effectively capturing human judgments across four critical dimensions: aesthetic quality, semantic alignment, detail fidelity, and overall assessment.

###### B.3. Datasets

Pick-a-Pic. The Pick-a-Pic dataset (Kirstain et al., 2023), collected through user interactions with a dedicated web application for text-to-image generation, systematically records each comparison with a prompt, two generated images, and a preference label (indicating either a preferred image or a tie when no significant preference exists). Following Bai et al. (2024), we utilize the initial 100 prompts as a representative test set, which provides adequate coverage to assess model performance.

Drawbench. DrawBench (Saharia et al., 2022) is a comprehensive evaluation benchmark for text-to-image models, featuring approximately 200 text prompts across 11 distinct categories that assess critical capabilities including color rendering, object counting, and text generation.

GenEval. Geneval (Ghosh et al., 2024) is an object-focused framework that evaluates image composition through object co-occurrence, position, count, and color. Using 553 prompts, it achieves 83% agreement with human judgments on image correctness.

Table 7. Different types of model differences lead to improvements effects in different directions.

Model Difference Ms Mw Results

Finetune Mechanism SDXL/SD1.5 Tables 1 and 8

Weight Difference

LoRA Mechanism SDXL/SD1.5 Tables 2, 10 and 11

Strong Experts (MoE) Weak experts (MoE) Table 3

High CFG Low CFG Tables 4 and 9

Condition Difference

Refined Prompt Raw Prompt Table 5

ControlNet Standard Pipeline (DDIM) Figure 10 IP-Adapter Standard Pipeline (DDIM) Figure 17

Sampling Pipeline Difference

VBench VBench (Huang et al., 2024) is a comprehensive benchmark for video generation models, featuring a hierarchical evaluation framework across multiple quality dimensions. It supports both automatic and human assessment, with VBench++ extending to text-to-video and image-to-video tasks while incorporating trustworthiness evaluation.

Peronalization Dataset To evaluate the performance gains of W2SD in personalized generation, we selected 20 LoRA checkpoints from the Civitai platform, covering a diverse range of categories, including persons (e.g. Anne Hathaway, Scarlett Johansson), animals (e.g., Scottish Fold cat, prehistoric dinosaur), styles (e.g., Disney style, parchment style), anime characters (e.g. Sun Wukong, Bulma) and objects (e.g. cars).

- C. Supplementary experimental results In this section, we present more quantitative and qualitative results of W2SD.

###### C.1. Quantitative Results

Table 8. Quantitative results of W2SD based on a full parameter fine-tuning strategy. Our method generates results better aligned with human preferences. Datasets: Drawbench.

Method HPS v2 ↑ AES ↑ PickScore ↑ MPS ↑

SD1.5 25.3601 5.2023 21.0519 DreamShaper 28.7845 5.7047 21.8522 47.8813 W2SD 28.7901 5.7847 21.9057 52.1192 SDXL 28.5536 5.4946 22.2464 Juggernaut-XL 28.9085 5.3455 22.4906 47.5648

W2SD 29.3246 5.4261 22.5803 52.4358

Table 9. Quantitative results of W2SD based on guidance difference. Model: SDXL. Datasets: DrawBench.

Method HPS v2 ↑ AES ↑ PickScore ↑ MPS ↑

SD1.5 25.3601 5.2023 21.0519 47.7075 W2SD 25.8234 5.2157 21.2079 52.2934

SDXL 28.5536 5.4946 22.2464 40.9590 W2SD 30.1426 5.6600 22.4434 59.0415

Results of W2SD in other benchmarks To further validate the effectiveness of W2SD, we also conducted experiments on Drawbench (Saharia et al., 2022). For clarity, we have systematically organized and summarized these experiments in Table 7. In Drawbench, we report results based on weight difference (see Tables 8 and 11) and guidance difference (see Table 9). Additionally, in Pick-a-Pic, we report results based on weight difference (see Table 10). Notably, W2SD demonstrates consistent improvements across all evaluated metrics.

Results of W2SD in Video Generation Task We also validate the performance of W2SD on video generation task to demonstrate its broad applicability. We randomly select 200 prompts from VBench (Huang et al., 2024) as test prompt cases and focus on analyzing the AnimateDiff (Guo et al.) as video generation model.

For the strong model Ms, we employ AnimateDiff with Juggernaut-XL using guidance scale of 3.0, while the weak model

Table 10. Quantitative results of W2SD based on human preference LoRA model. Our method generates results better aligned with human preferences. Datasets: Pick-a-Pic.

Method HPS v2 ↑ AES ↑ PickScore ↑ MPS ↑ SD1.5 24.9558 5.5003 20.1368 -

Dpo-Lora 25.5678 5.5804 20.3514 44.2889 W2SD 26.0825 5.6567 20.5096 55.7106 SDXL 29.8701 6.0939 21.6487 -

xlMoreArtFullV1 32.8040 6.1176 22.3259 48.2224 W2SD 33.5959 6.2252 22.3644 51.7770

###### Table 11. Quantitative results of W2SD based on human preference LoRA model. Our method generates results better aligned with human preferences. Datasets: DrawBench.

Method HPS v2 ↑ AES ↑ PickScore ↑ MPS ↑ SD1.5 25.3601 5.2023 21.0519 -

Dpo-Lora 25.8896 5.2895 21.2308 49.3617 W2SD 25.9431 5.3553 21.2589 50.6399 SDXL 28.5536 5.4946 22.2464 -

xlMoreArtFullV1 31.2727 5.5487 22.7721 47.0396 W2SD 32.34857 5.7595 22.8301 52.9588

Table 12. Quantitative results of W2SD on the video generation task. Model: AnimateDiff. Datasets: VBench.

Method Subject Consistency ↑ Background Consistency ↑ Motion Smoothness ↑ Dynamic Degree ↑ Aesthetic Quality ↑ Image Quality ↑ AnimateDiff (SDXL) 91.4152% 95.8491% 94.1647% 45.0000% 55.3108% 58.4293%

AnimateDiff (Juggernaut-XL) 96.0820% 97.5463% 96.8834% 13.0653% 57.7097% 64.1674% W2SD 97.1398% 97.9386% 96.8706% 8.0000% 59.3736% 64.6987%

Mw utilizes AnimateDiff with SDXL at guidance scale of 1.0. This setup introduces both weight and guidance differences, meeting the conditions required by W2SD.

- In Table 12, W2SD achieves significant improvements across different dimensions such as Subject Consistency, Background Consistency, Aesthetic Quality and mage Quality, confirming its effectiveness in video generation task. Notably, in the Motion Smoothness dimension, W2SD exhibits a performance degradation due to Juggernaut-XL’s inherent limitations compared to SDXL. This observation further validates our theory in Section 3.1.

###### C.2. Qualitative Results

Weight Difference In Figure 14, we present visualization results of W2SD based on weight difference. Specifically, to enhance human preference for generated images, we select xlMoreArtFullV1 as the strong model and SDXL as the weak model.

Additionally, by setting the LoRA-based personalized model as strong model and the un-finetuned base model as weak model, Figure 15 showcases the improvement of W2SD in personalized generation effectiveness.

Condition Difference In Figure 16, we present the visualization results of W2SD based on condition differences. When the strong model utilizes detailed and semantically rich prompts, while the weak model relies on simple prompts (containing only 4–5 words), W2SD effectively captures fine-grained conditional features. Notably, Z-Sampling (Bai et al., 2024) is a special case of W2SD based on guidance differences, with extensive visual evidence already provided; thus, we omit additional visualizations here.

Sampling Pipeline Difference In Figure 17, we present the visualization results of W2SD based on the differences in the sampling pipeline. By incorporating reference image information through Ip-adapter during the denoising process and employing standard DDIM for inversion, W2SD ensures that the generated image adheres more closely to the given stylistic conditions, resulting in higher-quality results.

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

(WeightDifference) WeakModel

(SDXL) StrongModel

(xlMoreArtFullV1)

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

W2SD

Tiger wearing casual outfit.

Slipper made of potatoes. An little humanlike robot out of scrap.

Gandalf the Grey smoking a blunt.

Gamer playing league of legends at night.

- Figure 14. Qualitative results of W2SD based on weight differences (human preference). Here we select xlMoreArtFullV1 as the strong model and SDXL as the weak model. W2SD can effectively enhance the performance of human preference.

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Reference Image Standard LoRA LoR

W2SD: w/o

weightdifference Reference Image Standard A Chinese Armor

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Messi

Bulma

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

Daenerys Targaryen

W2SD: w/ weight difference

W2SD: w/o weight difference

W2SD: w/ weight difference

- Figure 15. Qualitative results of W2SD based on weight differences (personalization). Here we set LoRA-based personalized model as strong model and the standard model as weak model

###### SDXL

###### SDXL

[Figure 94]

[Figure 95]

S2WD (negative gains)

W2SD (positive gains)

S2WD (negative gains)

W2SD (positive gains)

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

Raw Prompt: a photo of a black teddy bear. Refined Prompt: A heartwarming photograph captures the endearing image of a plush black teddy bear, its soft, velvety fur glistening subtly under the gentle light, with large, soulful eyes that seem to hold a world of stories and dreams.

Raw Prompt: a photo of two clocks.

Refined Prompt: A vintage photograph captures the intricate beauty of two antique clocks, their ornate faces and polished brass casings reflecting the soft, each ticking in a synchronized

dance of time that seems to whisper the secrets of the past.

- Figure 16. Qualitative results of W2SD based on semantic differences between prompts, which refines the generation process by placing greater emphasis on the fine-grained details.

[Figure 100]

[Figure 101]

Reference Image

IP-Adapter W2SD(Ours)

A moon on the sea.

[Figure 102]

[Figure 103]

Reference Image

IP-Adapter W2SD(Ours)

A old man.

- Figure 17. Qualitative results of W2SD based on sampling pipeline difference. When the strong model employs Ip-adapter and the weak model utilizes DDIM , W2SD can enhance the alignment of the generated results with the reference image (e.g., style).

- D. Proofs In this section, we provide the proofs for Theorem 1 presented in this work.

- D.1. Proof of Theorem 1

We first establish the relationship between the latent variable xt and the refined latent variable x˜t through the reflection operation, proving that the core mechanism of W2SD is to optimize xt in the direction of the weak-to-strong difference.

Proof. At a given time t, the W2SD reflection operation applies the operator MwinvMs(·) to the latent variable xt. Specifically, we first denoise xt using the strong model Ms according to Equation (2), obtaining xt−∆t as

xt−∆t = xt + σ2tssθ(xt,t)∆t, (7)

where ssθ denotes the score predicted by Ms, which is equivalent to the gradient of the log probability density log pst, so, Equation (7) can be rewritten as

xt−∆t = xt + σ2t∇xt

log pst(xt)∆t. (8)

After obtaining xt−∆t, we apply the weak model Mw to invert xt−∆t back to the noise level at time t according Equation (5), thereby completing the reflection process as

x˜t ≈ xt−∆t − σ2tswθ (xt−∆t,t)∆t, (9)

where swθ denotes the score predicted by Mw. Since ∆t is typically small, we neglect the approximation error in the diffusion inversion process which implies that swθ (xt−∆t,t) is equivalent to the gradient of the log probability density log pwt . Hence we can reformulate Equation (9) as

x˜t = xt−∆t − σ2t∇xt

log pwt (xt)∆t. (10)

Combining Equation (8) and Equation (10), we obtain x˜t as

x˜t = xt−∆t − σ2t∇xt

log pwt (xt)∆t (11)

= xt + σ2t∇xt

log pst(xt)∆t − σ2t∇xt

log pwt (xt)∆t (12)

= xt + σ2t∆t(∇xt

log pst(xt) − ∇xt

log pwt (xt) weak-to-strong difference∆1(t)

). (13)

From Equation (13), we observe that for the latent variable xt, the reflection operator Mwinv(Ms(·)) in W2SD perturbs xt along the direction of ∆1(t), producing the refined variable x˜t. When the weak-to-strong difference ∆1 closely bridges the unattainable strong-to-ideal difference ∆2(t), the resulting x˜t becomes more aligned with the ground truth ideal distribution.

| |
|---|

- E. Further analytical exploration

- E.1. Deeper Visual Understanding

To provide a more intuitive demonstration of the W2SD effect, we present more 1D visualization examples in Figure 18 to aid understanding. As can be seen, W2SD has a certain probability of “correcting” the denoising trajectory of the model, resulting in more balanced generation outcomes. The settings here are consistent with those in Figure 4.

We also empirically validate that the weak-to-strong difference ∆2 can effectively bridges the strong-to-ideal difference ∆1. We first compute CosineSimilarity(∆1(t),∆2(t)) at each timestep t. We train three models on CIFAR-10: the weak model (100 epochs), the strong model (300 epochs), and the ideal model (fully converged, 500 epochs). Here ∆1(t) = ϵideal(t) − ϵstrong(t) and ∆1(t) = ϵstrong(t) − ϵweak(t). In the table below, the angle between ∆1(t) and ∆2(t) remains below 90o, confirming that weak-to-strong gap can reliably bridge strong-to-ideal gap.

[Figure 104]

- Figure 18. Denoising trajectories across different settings (1-D Gauss). The weak model (blue) generates only right-peak data due to missing left-peak training samples, while the strong model (red) produces data between both peaks. W2SD balances the distribution by leveraging the reflective operator Mwinv(Ms(·)).

[Figure 105]

Figure 19. Positive cosine similarity between ∆1 and ∆2 (cos Θ < 90°).

To further validate this claim that ∆2 can bridge ∆1, we provide additional visualizations. As illustrated in Figure 20, the W2SD trajectory is visualized in 2D space, where its denoising direction (black) demonstrates closer alignment with the ideal direction (green). Moreover, the angles between the strong (red)-to-weak (blue) and ideal (green)-to-strong (red) vectors consistently remain within 90 °.

###### E.2. The Impact of Approximation Error in the Inversion Process

In Section 2, we assume that the inversion and denoising process are reversible, meaning that for the same generative model (including the same guidance scale, etc.), Minv(M(xt,t),t) = xt. However, in practice, the inversion process inevitably introduces errors. Specifically, for xt at time t, the inversion process actually used in the algorithm implementation is as

x˜t = xt−∆t − σ2tsθ(xt−∆t,t)∆t (14)

= xt−∆t − σ2t(sθ(xt−∆t,t) − sθ(xt,t)

###### +sθ(xt,t))∆t, (15)

Inversion Error Et

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

- ∆1( )
- ∆2

[Figure 110]

( )

< 90

Figure 20. 2D denoising path visualization. The weak-to-strong gap can effectively bridge the strong-to-ideal gap.

Table 13. As the magnitude of the approximation error in inversion process increases, the gains from W2SD diminish. Model :xlMoreArtFullV1. Datasets: Pick-a-Pic. The type of W2SD: weight difference.

Method HPS v2 ↑ AES ↑ PickScore ↑ MPS ↑ SDXL 29.8701 6.0939 21.6487 xlMoreArtFullV1 32.8040 6.1176 22.3259 48.2224

W2SD (k=0) 33.5959 6.2252 22.3644 51.7770 W2SD (k=0.005) 32.9341 6.3228 22.3221 46.7130 W2SD (k=0.010) 19.5277 4.3949 17.9267 1.3440

Table 14. By selecting Gaussian noise during the Re-Sampling process, the advanced Algorithm 3 achieves superior performance in the sampling process, demonstrating that Re-Sampling is a specific instance of W2SD. Model: SDXL. Datasets: Pick-a-Pic. The type of W2SD: guidance difference.

Method HPS v2 ↑ AES ↑ PickScore ↑ MPS ↑ SDXL 29.8701 6.0939 21.6487 -

Re-Sampling (sim score<0) 30.2797 6.0744 21.6894 48.4793 Re-Sampling (sim score>0) 30.7844 6.0555 21.8620 51.5210

###### W2SD 31.2020 6.0970 21.7980 56.0608

And the effect of W2S on the latent variable in Theorem 1 can also be expressed as

x˜t = xt + σ2t∆t(∇xt

log pst(xt) − ∇xt

log pwt (xt) − Et), (16)

where Inversion Error Et represents the approximation error in the inversion process at time t. Since ∆t is very small, sθ(xt,t) and sθ(xt−∆t,t) are assumed to be approximately equal by default, i.e., Et ≈ 0.

To further analyze the impact of Et on the W2SD, we simplify the analysis by replacing Et with Gaussian noise of controllable magnitude as

x˜t = xt + σ2t∆t(∇xt

log pst(xt) − ∇xt

log pwt (xt) − kϵ). (17)

By varying the value of k, we adjust the magnitude of the error Et to study its effect on the generated results. We analyze W2SD based on weight differences, with the strong model using xlMoreArt-FullV1 and the weak model using SDXL.

- In Table 13, as k increases, indicating a larger approximation error Et, the gains from W2SD diminish. This finding is consistent with Bai et al. (2024)’s results, demonstrating that this phenomenon occurs in both guidance difference and weight difference scenarios, indicating that minimizing the approximation error in inversion process is crucial.

:ℳ (∙) in  2  

: ( sim_score<0)

: ( sim_score>0)

Good ReNoise (sim_score > 0)

Bad ReNoise (sim_score < 0)

SDXL (base model)

[Figure 111]

[Figure 112]

[Figure 113]

 − 

[Figure 114]

[Figure 115]

[Figure 116]

Figure 21. Re-Sampling, which can be considered a specific instance of W2SD, demonstrates improved performance when the randomly sampled Gaussian noise aligns closely with the perturbation vector introduced by the W2SD reflection mechanism (e.g., cosine similarity > 0)

Figure 22. Qualitative results of advanced Re-Sampling demonstrate that the improvements effects vary depending on the strategy used to select Gaussian noise for Re-Sampling. It can be considered a specific instance of W2SD.

###### E.3. Relationship with Re-Sampling

Re-Sampling (lug, 2022) is the earliest and most classic iterative algorithm of its kind proposed in diffusion models. As illustrated in Algorithm 2, it introduces randomly sampled Gaussian noise into the latent variable xt, followed by repeated denoising processes.

We confirm that Re-Sampling is generally a specific instance of W2SD, which can be interpreted within the framework of our theory in Theorem 1. In Figure 21, we note that Re-Sampling performs better when the randomly sampled Gaussian noise in Re-Sampling is similar to the perturbation vector introduced by the W2SD reflection mechanism (e.g., cosine similarity > 0). Additionally, we propose an advanced version of Re-Sampling (see Algorithm 3), which evaluates whether the similarity score, sim score(ϵw2s,ϵ), exceeds 0 to determine whether the Gaussian noise ϵ should be accepted for the ReNoise operation.

We validate the effectiveness of Algorithm 3 in Table 14. When consistently selecting favorable random noise (i.e., ϵw2s,ϵ) > 0), advanced Re-Sampling demonstrates improved performance. Conversely, when consistently selecting unfavorable random noise (i.e., sim score(ϵw2s,ϵ) < 0), the performance of Re-Sampling deteriorates. We also present the visualization results in Figure 22, further demonstrating that Re-Sampling can be incorporated into the W2SD framework, validating the correctness of our theory in Theorem 1.

Similarly, many subsequent research, such as FreeDoM (Yu et al., 2023), UGD (Bansal et al., 2023), MPGD (He et al., 2023), and TFG (Ye et al., 2024), follow the same approach as Re-Sampling by utilizing random Gaussian noise for iterative optimization. Therefore, these inference enhancement methods can be regarded as specific instances of W2SD. The primary distinction among these methods lies in the specific strong model Ms they employ. For instance, TFG utilizes a more refined parameter search mechanism, resulting in a strong model that exhibits greater robustness and performance compared to algorithms such as FreeDoM and UGD. As a consequence, under the condition of the same weak model (i.e., the addition of random Gaussian noise), TFG demonstrates significantly enhanced performance. However, these studies collectively overlook a fundamental aspect: the weak-to-strong difference constitutes the core principle that fundamentally drives the efficacy of such algorithms.

Algorithm 3 Advanced Re-Sampling Input: Strong Model Ms, Weak Model Mw, Total Inference Steps: T, Optimization Steps: λ Output: Clean Data x0 Sample Gaussian noise xT for t = T to 1 do

Algorithm 2 Vanilla Re-Sampling Input: Strong Model Ms, Total Inference Steps: T, Optimization Steps: λ Output: Clean Data x0 Sample Gaussian noise xT for t = T to 1 do

if t > T − λ then xt−1 = Ms(xt,t) x˜t = Mwinv(xt−1,t) ϵw2s = x˜t − xt Calculate ϵw2s based on Equation (6) #Select Optimal ReNoise Initialize ϵ ∼ N(0,1) while similarity score (ϵw2s,ϵ) < 0 do

if t > T − λ then xt−1 = Ms(xt,t) Initialize ϵ ∼ N(0,1)

xRet = Add Noise(xt,ϵ,t) end if xt−1 = Ms(xRet ,t)

Initialize ϵ ∼ N(0,1) end while xRet = Add Noise(xt,ϵ,t)

end for

end if xt−1 = Ms(xRet ,t)

end for

- E.4. Relationship with Auto-guidance

We note that in the weak-to-strong framework, Auto-guidance (Karras et al., 2024) employs a pre-trained diffusion model along with a corrupted version of it (typically achieved by adding perturbations or reducing training iterations through training from scratch). It directly enhances performance by interpolating in the latent space. And here we clarify the contributions of W2SD in relation to it.

Different mechanisms W2SD employs an reflection mechanism, while Auto-guidance utilizes an interpolation-based method. For comparison with W2SD, we set w=1 in Auto-guidance. When using Strong Model (human preference model) vs Weak Model (SDXL), in Table 15, we show that W2SD achieves notably higher scores on human preference metrics including PickScore and HPS v2. However, in this setting, direct interpolation in Auto-guidance leads to performance degradation in certain metrics, manifesting as oversaturation and artifacts. Actually, the auto-guidance mechanism similarly utilizes the following operations as

xt → xgoodt−1 (18) xt → xbadt−1 (19) xnewt−1 = xgoodt−1 + w · (xgoodt−1 − xbadt−1). (20)

When w is too large, applying Equation (20) roughly refines the distribution of the latent variables, leading to an unnatural shift in the data distribution (Sadat et al., 2024; Lou & Ermon, 2023). When w is too small, it fails to produce sufficient refinement.

In W2SD’s process, xt → xt−1 → xt, the refinement operation (marked in red) is implicitly performed by score network’s internal transformation which avoids common artifacts like distortion and over-saturation, please see Figure 23.

On the other hand, we note W2SD generalizes the concept of weak/strong model pairs—where the ”weak” model is not limited to underperforming variants created through reduced capacity or training strategies (e.g., data corruptions or degradations as in Auto-guidance). We propose that differences in semantic interpretation of prompts or sampling methodologies (e.g., MoE routers, ControlNet adaptations) can equally constitute valid weak-strong pairings. This expanded paradigm demonstrates significantly greater practical utility, as it accommodates real-world deployment scenarios where model capabilities vary along multiple axes.

Table 15. Performance comparison between Auto-guidance’s interpolation and W2SD’s reflection mechanism in latent variable refinement.

Method HPS v2 ↑ AES ↑ PickScore ↑

SDXL 29.8701 6.0939 21.6487 xlMoreArtFullV1 32.8040 6.1176 22.3259

Auto-guidance 32.1650 6.1187 22.0177 W2SD 33.5959 6.2252 22.3644

[Figure 117]

[Figure 118]

[Figure 119]

AutoGuidance

[Figure 120]

[Figure 121]

[Figure 122]

W2SD (Ours)

Figure 23. Compared with Auto-guidance, W2SD achieves superior performance with enhanced robustness, effectively addressing critical issues such as oversaturation and optimization failure.

