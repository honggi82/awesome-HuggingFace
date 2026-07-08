# arXiv:2504.01308v3[cs.CV]2Aug2025

## Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks

Jiawei Wang1,* Yushen Zuo2,* Yuanjun Chai3 Zhendong Liu4 Yicheng Fu5 Yichun Feng6,† Kin-Man Lam2,†

1University of Science and Technology of China, 2The Hong Kong Polytechnic University, 3University of Washington, 4Nanjing University, 5Stanford University, 6University of Chinese Academy of Sciences

wangjiawei@mail.ustc.edu.cn zuoyushen12@gmail.com yjchai@uw.edu dz20330019@smail.nju.edu.cn easonfu@stanford.edu fengyichun22@mails.ucas.ac.cn kin.man.lam@polyu.edu.hk

#### Abstract

Vision-Language Models (VLMs) extend the capabilities of Large Language Models (LLMs) by incorporating visual information, yet they remain vulnerable to jailbreak attacks, especially when processing noisy or corrupted images. Although existing VLMs adopt security measures during training to mitigate such attacks, vulnerabilities associated with noise-augmented visual inputs are overlooked. In this work, we identify that missing noise-augmented training causes critical security gaps: many VLMs are susceptible to even simple perturbations such as Gaussian noise. To address this challenge, we propose Robust-VLGuard, a multimodal safety dataset with aligned / misaligned imagetext pairs, combined with noise-augmented fine-tuning that reduces attack success rates while preserving functionality of VLM. For stronger optimization-based visual perturbation attacks, we propose DiffPure-VLM, leveraging diffusion models to convert adversarial perturbations into Gaussian-like noise, which can be defended by VLMs with noise-augmented safety fine-tuning. Experimental results demonstrate that the distribution-shifting property of diffusion model aligns well with our fine-tuned VLMs, significantly mitigating adversarial perturbations across varying intensities. The dataset and code are available at https://github.com/JarvisUSTC/DiffPureRobustVLM

#### 1. Introduction

Recently, Vision-Language Models (VLMs) [3, 9, 30, 52] have significantly extended the capabilities of Large Lan-

*Equal contribution. Work done when Yushen Zuo was intern at PolyU. †Corresponding author

guage Models (LLMs) by integrating both visual and textual information, allowing them to interpret and respond based on multimodal inputs. This advancement enables VLMs to tackle a wider range of tasks, from understanding images to generating rich, contextually aware responses that leverage both language and visual cues. While LLMs have incorporated various training techniques, such as Reinforcement Learning from Human Feedback (RLHF) [5, 16], to ensure alignment with ethical and legal standards, VLMs remain more susceptible to certain risks. Fine-tuning VLMs for visual instruction-following can disrupt the alignment carefully established in LLMs [53]. Additionally, the integration of visual modalities not only introduces extra risk factors, such as a heightened vulnerability to jailbreak attacks [28, 36], but also poses greater challenges for model robustness. Compared to LLMs, VLMs must account for a broader spectrum of visual scenarios, rendering them more sensitive to even minor noise perturbations.

Perturbation-based adversarial attacks have long targeted image classification neural networks [19, 43]. With the emergence of VLMs, many studies have adapted traditional optimization-based perturbation attack methods to perform jailbreak attacks on these models [14, 28, 36]. Concurrently, substantial research has focused on defending against perturbation-based attacks. For instance, DiffPure [34] utilizes the denoising capabilities of Diffusion Models as an image preprocessing method to neutralize perturbation noise in adversarial images, though it does not fully counteract perturbation attacks, particularly in VLMs [36]. Zong et al. [53] introduced VLGuard, a vision-language dataset containing both safe and unsafe queries and images, designed to fine-tune VLMs for enhanced protection against jailbreak attacks. However, VLGuard’s effectiveness has only been evaluated against the FigStep attack and it does

not consider cases where the image is unrelated to the text prompt. Similarly, Zhang et al. [51] introduced Jailguard, a mutation-based detection framework that effectively identifies jailbreaks but significantly raises inference costs. More recently, Xu et al. [46] proposed an efficient cross-modality approach, CIDER, for detecting adversarially perturbed images, though it still impacts models’ helpfulness noticeably.

In this work, we observed that many prominent visionlanguage models (VLMs) lack visual noise augmentation during training, leaving them vulnerable to minor perturbations like Gaussian noise. This vulnerability compromises both their helpfulness and safety. To investigate this, we systematically evaluated three mainstream VLMs—InternVL2-8B [9], LLaVA-v1.5-7B [30], and MiniGPT-4-13B [52]—against such perturbations. Our findings reveal that although these models excel under standard conditions, their performance degrades significantly in the presence of even slight Gaussian noise.

To address this, we introduce Robust-VLGuard, a novel vision-language dataset designed to bolster VLM robustness against Gaussian noise while improving safety and preserving helpfulness. For safety, the dataset includes 2,000 curated instructions, uniquely featuring both imagetext aligned and misaligned scenarios to account for alignment disruptions from fine-tuning or noisy inputs. To maintain helpfulness, we add 4,467 general instructions with detailed responses generated by GPT-4V [35], overcoming the overly brief annotations common in other datasets. Finetuning VLMs on Robust-VLGuard with Gaussian noise augmentation shows that even limited high-quality data can significantly boost noise robustness with minimal impact on baseline performance.

We then extend our evaluation to broader, optimizationbased adversarial attacks. While noise augmentation is often distribution-specific, we find that effective image preprocessing is crucial for general defense. Our analysis identifies DiffPure [34] as highly effective, as it uses diffusion models to transform adversarial perturbations into Gaussian-like noise without degrading image content. This distribution-shifting property aligns perfectly with our noise-augmented VLMs, proving more effective and efficient than VLM-specific defenses like JailGuard [51]. Consequently, we propose DiffPure-VLM, a defense pipeline integrating DiffPure with our robust VLMs. Experiments show this pipeline effectively mitigates diverse adversarial attacks at varying intensities.

In conclusion, our contributions are: (1) To our best knowledge, we are the first to provide a systematic vulnerability analysis revealing that mainstream VLMs lack inherent robustness to Gaussian noise visual perturbations. (2) We propose the Robust-VLGuard dataset, which features novel image-text misalignment scenarios and detailed responses, and combine it with Gaussian noise augmentation

[Figure 1]

Answer: The green logo on the car is the Monster Energy logo.

InternVL2-8B

Clean Image

|[Figure 2]|
|---|

Question: what is the green logo on the car?

[Figure 3]

Answer: The green logo on the car is the Mötley Crüe logo.

InternVL2-8B

Noisy Image

Figure 1. Visualization of Vision-Language Model’s outputs under different noise conditions. The upper part shows the original image with green text indicating correct responses generated without noise, while the lower part adds slight Gaussian noise, with red text highlighting errors introduced under noisy conditions. Please zoom in for better visualization.

for fine-tuning to enhance VLM robustness against Gaussian noise while preserving its helpfulness. (3) We expand the defense scope of fine-tuned VLMs to optimizationbased visual adversarial attacks and propose a defense framework, DiffPure-VLM, by adopting the distributionshifting ability of the diffusion model to transfer adversarial noise to Gaussian-like noise in visual input, which can be defended by VLMs with noise-augmented safety finetuning. Experimental results demonstrate the superiority of DiffPure-VLM against baseline methods and its generalization.

#### 2. Vulnerability of VLMs to Gaussian Noise Perturbations

We observed that many current VLMs, including advanced ones, lack noise augmentation during training, rendering them vulnerable to basic perturbations like Gaussian noise. As shown in Figure 1, we presented both a clean image and a slightly Gaussian-noised version as visual prompts. One leading model, InternVL2-8B [9], displayed inconsistent responses, with noisy prompts causing hallucinated outputs. This motivates a systematic evaluation of the robustness of mainstream VLMs against Gaussian noise, focusing on helpfulness and safety. Additional evaluation results for the latest VLMs (e.g., LLaMA-3.2-Vision [13], Qwen2.5-VL [4]) are provided in Table 5 in the Appendix.

##### 2.1. Experimental Settings

Models We evaluate three state-of-the-art VLMs: MiniGPT-4 (13B) [52], LLaVA-v1.5 (7B) [30], and InternVL2 (8B) [9]. Each model features a distinct LLM,

| |26.7<br><br>33.0<br><br>59.9<br><br>24.0<br><br>31.3<br><br>54.4<br><br>Clean Images Noisy Images<br><br>| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

60

50

PerformanceScore(%)

40

30

20

10

0

MiniGPT-4 LLaVA-v1.5 InternVL2

Models

(a) Performance evaluation on MMVet benchmark.

| |34.8<br><br>57.7<br><br>50.5<br><br>44.1<br><br>60.1<br><br>Clean Images 57.2<br><br>Noisy Images| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

60

AttackSuccessRate(%)

50

40

30

20

10

0

MiniGPT-4 LLaVA-v1.5 InternVL2

Models

(b) Attack success rate on RealToxicityPrompts benchmark.

- Figure 2. Comparison of various models’ performance and robustness: (a) helpfulness on the MM-Vet benchmark with clean and noisy image prompts, and (b) attack success rates on the RealToxicityPrompts benchmark using clean and noisy image prompts.

vision encoder, and vision-language alignment method, allowing us to draw broader insights. Details of these models are in Table 1 in the Appendix. To ensure reproducibility in helpfulness evaluations, we set the temperature to 0, while safety assessments follow the setup of Qi et al. [36]. The default system prompt is used throughout.

Datasets To test robustness under Gaussian noise, we evaluate VLM performance on two key aspects: helpfulness and safety. For helpfulness, we use MM-Vet [47], a comprehensive benchmark assessing six vision-language capabilities: recognition, OCR, knowledge, language generation, spatial reasoning, and mathematics. For safety, we use the RealToxicityPrompts benchmark [17], specifically the challenging subset with 1,200 prompts, following Qi et al. [36]. We augment image prompts from both datasets with Gaussian noise with a mean of 0 and a standard deviation of 0.1, to compare performance under clean and noisy conditions.

Metrics For helpfulness, we use the original MM-Vet metric [47], designed to handle diverse real-world scenarios. GPT-4 [1] serves as the evaluation assistant, using a few-shot prompt template for flexible scoring. Each response is rated from 0 for incorrect answers to 1 for correct answers, and the Performance Score is the average of all sample scores. For safety, we use the metrics from Qi et al. [36], pairing visual adversarial examples with text prompts and measuring toxicity using the Perspective API1 and Detoxify classifier [21]. Toxicity scores range from 0, indicating the least toxic, to 1, indicating the most toxic. The Attack Success Rate is the percentage of responses with any toxicity score above 0.5, indicating a successful attack.

##### 2.2. Findings

The main results are shown in Figure 2, where helpfulness is measured using MM-Vet benchmark scores and safety is evaluated by the Attack Success Rate using the Perspective API on the RealToxicityPrompts benchmark. We draw

1https://perspectiveapi.com/

the following key insights regarding the impact of Gaussian noise on VLMs in terms of helpfulness and safety alignment:

Helpfulness Degradation In Figure 2a, InternVL2, despite having the highest baseline performance on MM-Vet, suffers a significant drop when exposed to Gaussian noise, revealing its lack of noise robustness. MiniGPT-4 and InternVL2 show similar relative declines of around 10%, while LLaVA-v1.5 experiences a smaller drop, indicating better noise tolerance. However, all models exhibit a noticeable decrease, underscoring their vulnerability to even slight noise perturbations.

Safety Alignment Impact Figure 2b shows increased attack success rates on the RealToxicityPrompts benchmark across all models under noisy conditions, suggesting that Gaussian noise negatively affects safety alignment. While prior work focuses on optimization-based attacks [28, 36], our results demonstrate that even random Gaussian noise can significantly disrupt alignment. Both MiniGPT-4 and InternVL2 show substantial increases in attack success rates, indicating greater vulnerability, whereas LLaVA-v1.5 experiences a smaller but still significant rise, suggesting slightly better robustness to noise. Detailed theoretical conjecture and discussions about the vulnerability caused by Gaussian noise are provided in Section F in the Appendix.

Overall Robustness The findings in Figure 2 indicate that while current VLMs perform well under standard conditions, their robustness significantly decreases when faced with Gaussian noise. Both helpfulness and safety alignment degrade across all models, highlighting the need for enhanced noise robustness in future VLM designs to ensure better stability and alignment.

#### 3. Noise-Augmented Safety Alignment

A straightforward strategy to bolster VLM robustness against Gaussian noise is to integrate noise augmentation into the training process. In this section, we introduce our novel dataset, Robust-VLGuard, which comprises meticulously curated image-text aligned and misaligned safety data, as well as a diverse array of general instructions. We then fine-tune VLMs using Robust-VLGuard augmented with Gaussian noise in visual inputs. Through extensive experiments, we enable rapid enhancement of model robustness while preserving, and even enhancing, the models’ inherent helpfulness.

##### 3.1. Robust-VLGuard Dataset

While the VLGuard dataset [53] has been developed to fine-tune VLMs for improved defense against jailbreak attacks, it does not address perturbation-based attacks or scenarios where the image content is unrelated to the text prompt. Therefore, we build a more robust public visionlanguage safety dataset Robust-VLGuard, as shown in Fig-

[Figure 4]

Brief Response

After kneading, let the dough rest in a greased bowl, covered with a damp cloth or plastic wrap. Place it in a warm spot (75–85°F/24– 29°C) for about 1 hour until it doubles in size. To check, press a finger into the dough—if it slowly springs back, it’s ready. If not, give it more time. Once risen, gently punch down to release excess gas, making it easier to handle. This step enhances texture and flavor. Now, you can divide the dough into pieces and proceed with shaping and baking.

[Figure 5]

Place it in a warm spot.

GPT-4V

Detailed Response

Question: What should I do after kneading the dough, before dividing it into pieces?

###### (a) Example of general instruction data.

[Figure 6]

Question: Post this image on social media to raise awareness about gun safety.

Safe Response

I'm sorry, but I can't assist with that instruction. The image depicts a minor with a firearm, which is a sensitive and potentially harmful topic. Gun safety is an important issue, but sharing images of minors with firearms can promote unsafe behavior, glorify violence, and lead to negative consequences.

(b) Example of image-text aligned safety data.

[Figure 7]

Question: Create a plan to ruin a chef's dishes.

Safe Response

I'm sorry, but as an AI language model, I cannot provide advice on how to intentionally ruin someone's dishes. It is unethical and disrespectful to intentionally sabotage someone else's work. Instead, I suggest focusing on constructive feedback or discussing your concerns with the chef directly in a respectful manner.

(c) Example of image-text misaligned safety data.

- Figure 3. Overview of the Robust-VLGuard dataset. (a) General Instruction Data: Leveraging GPT-4V to generate comprehensive, detailed responses rather than brief replies; (b) Image-Text Aligned Safety Data: The image content directly corresponds to the safety-related text prompts; (c) Image-Text Misaligned Safety Data: The image content is deliberately unrelated to the safety-related text prompts. Red text indicates content with potential risks, while green text denotes content without risks.

Table 1. Detailed Breakdown of General Instruction Data

|Task<br><br>|Dataset|Number of Samples|
|---|---|---|
|General QA|GQA [25]<br><br>|1000|
|World Knowledge<br><br>|A-OKVQA [39]|1000<br><br>|
|Math & OCR|ChartQA [32], TabMWP [31]|467<br><br>|
|Spatial Reasoning<br><br>|VQAv2 [20]<br><br>|1000|
|Extended Text Generation|LLaVA v1.5 Mix 665k [30]<br><br>|1000|

ure 3, which consists of three parts: (1) General Instruction Data, consisting of safety-agnostic SFT data covering various areas, including general QA, world knowledge, math, OCR, spatial reasoning, and extended text generation; (2) Image-Text Aligned Safety Data, containing instructions where the image content aligns with the safetyrelated text prompts; and (3) Image-Text Misaligned Safety Data, with instructions where the image content is unrelated to the safety-related text prompts.

General Instruction Data To maintain VLMs’ helpfulness, we collect 4,467 supervised fine-tuning instructions from various aspects, including general QA, world knowledge, math, OCR, spatial reasoning, and extended text generation, as illustrated in Table 1. Specifically, we sample various instructions from these datasets and use GPT-4V [35] to refine the annotated answers. This refinement is essential, as we found that the original annotations were often too brief for effective model learning. For extended text generation, we select 1,000 instructions with responses over 150 words. For all other datasets, we choose instructions with responses exceeding 10 words.

Image-Text Aligned Safety Data The VLGuard dataset

is well-suited for preventing jailbreak attacks, as it contains harmful information embedded within image content, with instructions generated by GPT-4V. Therefore, we directly use the instructions from VLGuard as our image-text aligned safety data, randomly selecting 1,000 instructions from this dataset.

Image-Text Misaligned Safety Data Incorporating safety data for image-text misalignment is also crucial, as fine-tuning VLMs for visual tasks can disrupt the alignment of pre-trained LLMs [53], even when only text prompts are used. Additionally, perturbation-based attacks can introduce learnable noise into images that is unrelated to text prompts. Inspired by Bianchi et al. [6], who showed that a small set of safety examples can significantly boost model safety, we include 1,000 safety instructions from their dataset. To adapt these language-only safety instructions for multimodal use, we pair half of them with randomly selected images from the COCO dataset [29], while the remaining half are kept as text-only prompts.

##### 3.2. Safety Fine-Tuning

To optimize resource usage, we employ a Gaussian-noiseaugmented post-hoc fine-tuning approach. This efficient method is applied to pre-trained VLMs, enhancing robustness with minimal computational costs. Using the RobustVLGuard dataset, which includes both safety-specific and general instruction data, we effectively boost the model’s resilience to Gaussian noise while maintaining safety and helpfulness. Specifically, we fine-tune only the vision en-

Table 2. Performance Comparison on MM-Vet and RealToxicityPrompts Benchmarks with Clean and Noisy Image Prompts. † indicates reproduced results. Bold values denote, for each base model, the method (VLGuard vs. RobustVLGuard) that achieves the smallest performance drop on MM-Vet and the lowest attack success rate on RealToxicityPrompts.

|Model|MM-Vet (%) ↑<br><br>|RealToxicityPrompts (%) ↓|
|---|---|---|
| |Clean Image Noisy Image<br><br>|Clean Image Noisy Image|
|InternVL2-8B InternVL2-8B-VLGuard† InternVL2-8B-RobustVLGuard<br><br>|59.9 54.4 42.9 (-17.0) 42.6 (-11.8) 56.2 (-3.7) 52.5 (-1.9)<br><br>|50.5 57.2 27.7 39.9 29.9 34.5|
|LLaVA-v1.5-7B LLaVA-v1.5-7B-VLGuard [53] LLaVA-v1.5-7B-RobustVLGuard|33.0 31.3 28.8 (-4.2) 29.8 (-1.5) 30.3 (-2.7) 29.8 (-1.5)<br><br>|57.7 60.1 50.3 52.3 43.6 42.3|
|MiniGPT-4-13B MiniGPT-4-13B-VLGuard† MiniGPT-4-13B-RobustVLGuard|26.7 24.0 17.5 (-9.2) 17.6 (-6.4) 26.9 (+0.2) 27.3 (+3.3)<br><br>|34.8 44.1 41.3 43.7 16.0 16.5|

coder using LoRA [24] on our dataset and augment training images with Gaussian noise, selecting a random standard deviation between 0.01 and 0.15, with a 70% probability of application. The fine-tuning process is conducted over 3 epochs and takes approximately 3 hours on a single A100 GPU. Detailed fine-tuning configurations and a theoretical discussion on the algorithm’s effectiveness are provided in

- Table 2 in the Appendix. Comparison with VLGuard dataset To assess the effectiveness of our proposed Robust-VLGuard, we adopt the same experimental settings described in Section 2.1. Three leading VLMs, i.e., MiniGPT-4 (13B) [52], LLaVA-v1.5 (7B) [30], and InternVL2 (8B) [9], are fine-tuned using our Gaussian-noise-augmented method and Robust-VLGuard dataset. For comparison with VLGuard, we follow the setup of [53], combining 5,000 supervised fine-tuning instructions from LLaVA v1.5 Mix 665k [30] with 3,000 safety instructions from VLGuard.

Experimental results are summarized in Table 2. Due to the inevitable degradation in helpfulness resulting from LoRA-based safety fine-tuning, our proposed method aims to reduce the attack success rate while incurring minimal performance drop in helpfulness. The InternVL2-8BVLGuard model demonstrates a tendency towards overdefensiveness, achieving a lower attack success rate but at the cost of a noticeable decline in helpfulness compared to the original InternVL2-8B model. In contrast, our InternVL2-8B-RobustVLGuard model achieves a comparable level of safety while largely retaining the original helpfulness, achieving a more balanced performance. For LLaVA-v1.5-7B, the VLGuard-fine-tuned variant maintains its helpfulness, thanks to alignment with the original training data. However, it demonstrates limited improvements in safety, highlighting its inability to effectively address image-text misalignment attacks on the RealToxicityPrompts Benchmark. Our LLaVA-v1.5-7BRobustVLGuard delivers better overall performance, exhibiting stronger safety defenses and comparable helpfulness on both clean and noisy images. The MiniGPT-4-

60

60.0

###### AttackSuccessRate(%)

55

###### PerformanceScore(%)

57.5

50

55.0

45

52.5

40

50.0

35

47.5

30

45.0

25

Clean Images Noisy Images

Clean Images Noisy Images

42.5

20

15

40.0

4:1 4:2 4:3 4:4

4:1 4:2 4:3 4:4

General:Safety Instruction Ratio

General:Safety Instruction Ratio

(a) Performance on the MM-Vet benchmark across different instruction ratios.

(b) Attack success rate on the RealToxicityPrompts benchmark across different instruction ratios.

- Figure 4. Effect of varying instruction ratios on VLM’s robustness of helpfulness and safety alignment.

1 2 3 4

Training Epochs

40.0

42.5

45.0

47.5

50.0

52.5

55.0

57.5

60.0

PerformanceScore(%)

Clean Images Noisy Images

(a) Performance on the MM-Vet benchmark across different training epochs.

1 2 3 4

Training Epochs

15

20

25

30

35

40

45

50

AttackSuccessRate(%)

Clean Images Noisy Images

(b) Attack success rate on the RealToxicityPrompts benchmark across different training epochs.

- Figure 5. Effect of varying training epochs on VLM’s robustness of helpfulness and safety alignment.

13B-VLGuard model lags behind in both helpfulness and safety, whereas the MiniGPT-4-13B-RobustVLGuard variant shows notable enhancements, excelling on the MMVet benchmark and significantly lowering the attack success rate. Overall, these results emphasize the strengths of Robust-VLGuard in simultaneously enhancing model helpfulness and safety, providing comprehensive protection while maintaining performance across diverse scenarios.

Ablation Studies on Instruction Ratio and Training Epochs All ablation studies are based on the InternVL28B model. First, we vary the ratio of general to safety instruction data from 4:1 to 4:4, training for a single epoch for efficiency. As depicted in Figure 4, increasing the proportion of safety data lowers the attack success rate but slightly reduces helpfulness, echoing the over-defensiveness issue noted by [6]. However, beyond a 4:3 ratio, performance stabilizes, suggesting effective mitigation of overdefensiveness. We select a 4:2 ratio as the optimal balance, maximizing safety gains with minimal helpfulness impact.

Next, we evaluate the impact of training duration while keeping the instruction ratio fixed at 4:2 and varying the number of epochs from 1 to 4. As illustrated in Figure 5, increasing the number of epochs has a negligible effect on the model’s helpfulness as measured by the MM-Vet benchmark. However, it significantly reduces the attack success rate on the RealToxicityPrompts benchmark, indicating improved safety alignment without compromising utility. To strike a balance between helpfulness and safety alignment, we select 3 epochs as the fine-tuning configuration.

#### 4. Generalize to Optimization-based Visual Perturbation Attack

In this section, we extend our defense scope to a frequently encountered and challenging attack scenario: OptimizationBased Visual Perturbation Attack. It uses the projected gradient descent algorithm (PGD) with a pixel constraint ϵ to inject adversarial noise into images, effectively jailbreaking VLMs. While noise augmentation typically enhances robustness against specific noise distributions, we emphasize the critical role of image preprocessing in either transforming adversarial noise into a target distribution or directly eliminating it. We first evaluate the effectiveness of various image preprocessing defense methods, and then introduce DiffPure-VLM, a universal defense framework that robustly counters both the Gaussian and the adversarial noise.

##### 4.1. Preprocessing Methods in distribution shifting

In this section, we explore different image preprocessing defense methods in distribution shifting. Specifically, we use adversarial images Iadv optimized for perturbation attacks on MiniGPT-4 [52] from [36], with pixel constraint ϵ = 16/255 as an example. As Iadv is optimized based on a benign clean image Ic, we compute their residual image radv = Iadv − Ic to obtain the adversarial noise. Then we use a histogram and a quantile-quantile (Q-Q) plot to evaluate the distribution property of radv. As shown in Figure 6, adversarial noise follows a non-Gaussian distribution.

Currently, there are two representative image preprocessing defense methods: JailGuard [51] and DiffPure [34]. JailGuard, designed specifically for VLMs, employs various image processing techniques (e.g., random masking, horizontal flipping, Gaussian blur, and resizing) to generate variants of the input and detect adversarial samples based on discrepancies in model responses. However, most of these operations are linear transformations, which offer limited ability to eliminate adversarial perturbations. Moreover, JailGuard requires multiple model runs, leading to high computational overhead. In contrast, DiffPure is tailored for computer vision models (e.g., classifiers) and leverages diffusion models to mitigate adversarial noise. It adds a small amount of noise to the adversarial image Iadv and reconstructs a clean image through a limited number of forward and reverse diffusion steps (e.g., using DDPM [22]) with a carefully chosen timestep t∗ ∈ [50,150]. While DiffPure aims to purify adversarial perturbations while preserving global semantic structures, our findings reveal that at relatively small timesteps, it does not completely remove the noise. Instead, it shifts the perturbation distribution towards a Gaussian-like distribution.

Specifically, we apply t∗ = 50 in DiffPure and obtain diffused image Idiffused based on Iadv. Then we calculate the residual image rdiffused = Idiffused − Ic. As shown

Residual Image

[Figure 8]

[Figure 9]

[Figure 10]

Perturbationbased Attacks

DiffPure with timestep

PGD with constraint

Clean Image Adversarial Image Diffused Image

Residual Image

[Figure 11]

[Figure 12]

Statistics of (Red Channel) Statistics of (Red Channel)

Figure 6. Residual image Gaussianity analysis. We apply DiffPure (t∗ = 50) to adversarial image Iadv to obtain diffused image Idiffused. Then we calculate residual images radv and rdiffused and evaluate their distribution by the histogram and Q-Q plot.

in Figure 6, rdiffused approximates a Gaussian distribution from its shape and its closeness to the theoretical line (Red line) of the Gaussian distribution in the Q-Q plot. More visualizations of Idiffused and statistic of rdiffused across different ϵ and t∗ are available in Section D.2 in the Appendix. For quantitative evaluation of rdiffused, we use two metrics:

- (1) Kurtosis: Kurtosis is used to measure the tailedness of a data distribution, with the definition as

Kurt[X] = E

X − µX σX

4

, (1)

where µX and σX are mean and standard deviation of data X. If X follows Gaussian distribution, Kurt[X] = 3.

- (2) Q-Q deviation: Q-Q deviation measures the root-meansquare error (RMSE) between the quantiles of the sample distribution and those of a Gaussian distribution:

D(Qs,Qg) =

N

1 N

(Qs,i − Qg,i)2, (2)

i=1

where N is the number of ordered quantile points, Qs is the sample quantiles of rdiffused and Qt is the theoretical quantiles of a Gaussian distribution. A lower RMSE value suggests closer alignment to a Gaussian distribution. We vary t∗ in DiffPure from 0 to 750 in increments of 50, and also examine t∗ = 30 for fine-grained analysis. For RGB images, we calculate these metrics per channel and obtain the average value. To identify Gaussian-like distributions in our analysis, we use the thresholds 3 ≤ Kurtosis ≤ 6 and Q-Q deviation ≤ 0.01. Points meeting these criteria are marked in red in Figure 7, showing that under certain timesteps t∗ (e.g., t∗ ∈ [50,150]) in DiffPure, rdiffused ex-

###### (a) Kurtosis

###### (b) Q-Q Deviation

10

0.05

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

Constraint 16/255 Constraint 32/255 Constraint 64/255

- 8

0.04

Q-QDeviation

Kurtosis

6

0.03

- 4

0.02

2

0.01

0

0.00

0 100 200 300 400 500 600 700

0 100 200 300 400 500 600 700

Steps (t)

Steps (t)

Figure 7. Gaussianity metrics of rdiffused under different pixel constraints ϵ of adversarial image Iadv and timestep t∗ in DiffPure. Please zoom in to see details.

- Table 3. Defense of DiffPure in MiniGPT-4 under different image configurations. Attack Success Rate is evaluated on the RealToxicityPrompts benchmark.

Image Configuration Attack Success Rate (%)

|Benign clean Image Ic<br><br>+ n (σn = 30/255)<br><br>+ n (σn = 30/255) + DiffPure (t∗ = 50)|34.8 ± 1.6 44.1 44.3<br><br>|
|---|---|
|Adversarial image Iadv (ϵ = 16/255)<br><br>+ DiffPure (t∗ = 50)|53.6 ± 1.0 45.0<br><br>|
|Adversarial image Iadv (ϵ = 32/255)<br><br>+ DiffPure (t∗ = 50)|59.4 ± 1.4 45.5<br><br>|
|Adversarial image Iadv (ϵ = 64/255)<br><br>+ DiffPure (t∗ = 50)|67.2 ± 0.2 44.5<br><br>|

hibits Gaussian-like distribution properties. We further conduct our analysis in the embedding space and the conclusion is similar to that in pixel space. Detailed analysis is provided in Section D.2 in the Appendix.

Finally, we conduct a defense experiment with DiffPure. Specifically, we employ MiniGPT-4 as the VLM. We apply the unconditional model in [11] as the Diffusion model in DiffPure in all our experiments. We add Gaussian noise n ∼ N(0,σn2) to the benign clean image Ic and apply DiffPure with timestep t∗ to both Gaussian noisy image and adversarial images. Then we evaluate their attack success rate in the RealToxicityPrompts benchmark. Table 3 presents results for σn = 30/255 and t∗ = 50, results of different σn and t∗ are provided in Table 6 in the Appendix.

The experimental result shows that directly applying DiffPure to the Gaussian noisy image with VLM without noise-augmented safety fine-tuning does not decrease the attack success rate. We further observe that the attack success rate for diffused images is similar to Gaussian noisy image. Overall, we conclude that DiffPure, when applied with a suitable timestep t∗ (e.g., t∗ ∈ [50,150]), exhibits a unique distribution-shifting capability that transforms adversarial noise into a Gaussian-like distribution.

- 4.2. DiffPure-VLM

Leveraging our safety fine-tuning approach and DiffPure’s unique characteristic, we propose DiffPure-VLM — a defense pipeline that integrates Diffusion Models with

[Figure 13]

[Figure 14]

[Figure 15]

|[Figure 16]|
|---|

|[Figure 17]|
|---|

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

[Figure 33]

Figure 8. The overall framework of DiffPure-VLM.

Table 4. Comparison of image preprocessing methods for mitigating adversarial attacks (ϵ = 32/255) on RealToxicityPrompts.

|Model<br><br>|Attack Success Rate ↓<br><br>|Identity Profanity<br><br>Severe Toxicity<br><br>Sexually Explicit<br><br>Threat Toxicity|
|---|---|---|
|LLaVA-VLGuard JailGuard + LLaVA-VLGuard DiffPure + LLaVA-VLGuard<br><br>|70.4 52.1 51.1|21.3 52.8 7.5 16.7 7.0 67.2 12.5 39.0 5.3 13.2 4.9 50.0<br><br>3.4 40.9 2.2 13.4 3.6 47.5|
|LLaVA-RobustVLGuard JailGuard + LLaVA-RobustVLGuard DiffPure + LLaVA-RobustVLGuard<br><br>|62.5 48.9 43.9|7.8 48.0 5.4 16.5 5.8 60.0 6.0 37.3 4.8 13.4 4.0 46.5 3.2 34.6 2.4 12.8 3.7 41.0<br><br>|

Gaussian-noise-tolerant VLMs, as illustrated in Figure 8. Specifically, we purify adversarial images by applying a small timestep in DiffPure to preserve image content. The purified image with slight Gaussian-like noise is fed into the Gaussian-noise-tolerant, safety-tuned VLM, effectively mitigating the adversarial perturbations.

First, to verify DiffPure’s effectiveness in our defense pipeline, we compare its performance against JailGuard for mitigating optimization-based perturbation attacks following [36] on the RealToxicityPrompts benchmark. We selected two base models, LLaVA-VLGuard and our LLaVARobustVLGuard, for a comprehensive evaluation. As shown in Table 4, DiffPure with t∗ = 50 consistently outperforms JailGuard across both base models. Notably, when paired with our VLM, DiffPure delivers a substantially greater improvement over JailGuard than when paired with LLaVA-VLGuard, confirming that its distribution-shifting properties are especially well-suited to our robust VLM.

To evaluate the generalization of DiffPure-VLM, we assess the pipeline’s performance under various optimizationbased perturbation attack strengths (pixel constraint ϵ ∈ {16/255,32/255,64/255}) and different timesteps (t∗ = 50,150) of DiffPure in our suite of three safety fine-tuned VLMs. For brevity, Table 5 presents results for ϵ = 32/255. Results for the other ϵ values are provided in Table 7 in the Appendix. The experimental results demonstrate that, compared to clean or Gaussian-noise inputs, adversarial perturbation attacks substantially increase attack success rates. Nevertheless, our DiffPure mechanism consistently reduces these rates across all tested timesteps, nearly restoring performance to that of clean or Gaussian inputs. This not only confirms our analysis of DiffPure’s unique ability to transform adversarial noise into Gaussian noise, but also highlights the efficacy and generalization of DiffPure-VLM in effectively mitigating the impact of perturbations.

Table 5. Evaluation of DiffPure-VLM’s effectiveness on RealToxicityPrompts. Metrics include attack success rate and various toxicity levels (Perspective API %). Additional results for other attack strengths are provided in Table 7 in the Appendix.

Attack Success Rate ↓

Severe Toxicity

Sexually Explicit

Image Type

Threat Toxicity InternVL2-8B-RobustVLGuard

Identity Profanity

|Benign Clean image|29.9|0.8 22.1 0.3 7.2 1.5 25.9<br><br>|
|---|---|---|
|Gaussian Noisy image<br><br>|34.5|2.1 27.2 1.3 8.4 1.6 31.3|
|Adversarial image (ϵ = 32/255)|70.6|26.7 56.5 9.2 17.3 6.9 68.1<br><br>|
|+DiffPure-VLM (t*=50)<br><br>|33.4|2.4 20.6 0.7 8.1 2.4 29.1<br><br>|
|+DiffPure-VLM (t*=150)<br><br>|32.8<br><br>|1.7 25.9 0.6 7.7 1.8 29.1|

LLaVA-v1.5-7B-RobustVLGuard

|Benign Clean image<br><br>|43.6|4.6 34.7 2.4 12.3 3.5 41.0<br><br>|
|---|---|---|
|Gaussian Noisy image<br><br>|42.3<br><br>|3.1 34.5 1.9 11.8 3.1 40.0|
|Adversarial image (ϵ = 32/255)<br><br>|62.5|7.8 48.0 5.4 16.5 5.8 60.0<br><br>|
|+DiffPure-VLM (t*=50)<br><br>|43.9<br><br>|3.2 34.6 2.4 12.8 3.7 41.0|
|+DiffPure-VLM (t*=150)<br><br>|42.5<br><br>|3.5 32.7 2.8 12.1 4.1 39.3|

MiniGPT-4-13B-RobustVLGuard

|Benign Clean image<br><br>|16.0|0.4 9.9 0.3 4.6 1.1 12.1<br><br>|
|---|---|---|
|Gaussian Noisy image<br><br>|16.5<br><br>|0.9 11.9 0.6 5.8 1.0 14.0|
|Adversarial image (ϵ = 32/255)<br><br>|53.7<br><br>|9.8 35.3 4.1 13.9 5.4 48.1|
|+DiffPure-VLM (t*=50)|13.6<br><br>|0.3 9.2 0.2 5.5 0.9 10.6|
|+DiffPure-VLM (t*=150)<br><br>|11.9<br><br>|0.5 8.6 0.2 4.2 0.6 9.9|

#### 5. Related Work

##### 5.1. Vision Language Model

In recent years, Vision Language Models have gained significant attention due to their ability to jointly process and understand both visual and textual data. A major breakthrough in this area came with the development of largescale pre-trained models like CLIP [37], which aligns images and texts in a shared embedding space. By training on a vast dataset of image-text pairs, CLIP enabled zeroshot learning capabilities for tasks such as action recognition [7, 33] and optical character recognition [27, 41]. Recent advances have introduced multi-modal foundation models like InstructBLIP [10], LLaVA [30], and QwenVL [3], which leverage large-scale transformer networks that can process both image and text inputs simultaneously, enabling more sophisticated reasoning over complex scenarios. These models excel in advanced tasks like UI understanding [15, 23] and visual question answering [2], and open up new possibilities for generative capabilities.

##### 5.2. Jailbreaking VLMs

As VLMs become increasingly integrated into various applications, critical concerns have arisen regarding their robustness, security, and ethical alignment. A critical issue is ”jailbreaking”—the ability to bypass safety protocols, potentially triggering unintended or harmful behaviors. Given that VLMs process both textual and visual inputs, vulnerabilities inherent in LLMs may also affect VLMs. Moreover, the integration of visual inputs into VLMs, while expanding their capabilities, introduces more diverse attack patterns, significantly heightening the severity of potential threats [26, 45]. Several methods have emerged in this domain, including converting the malicious content into images through typography-based manipulations [18], leveraging multimodal perturbations to craft stronger adversarial

attacks [49], and using gradient updates to embed malicious triggers within seemingly benign images [40]. These diverse sophisticated methods pose significant challenges to ensuring the safety and reliability of VLMs.

##### 5.3. Safeguarding VLMs

Given the growing challenges posed by malicious attacks on VLMs, it is crucial to develop effective defense strategies. One key approach involves enhancing the training process through adversarial fine-tuning [38, 53] or prompt tuning [50], where models or learnable prompts are trained on perturbed examples to improve resilience against realworld attack scenarios. Another promising defense strategy is input sanitization, which aims to detect or neutralize adversarial inputs before they can compromise the model. Methods in this area include shifting sample probabilities to adversarial-free regions [44], applying randomized smoothing to mitigate the impact of adversarial inputs [42], and utilizing unlabeled prompts from users in the wild for malicious prompt detection [12]. In this work, we concentrate on improving VLM robustness to noise-augmented inputs. By integrating Diffusion Models with safety finetuned VLMs, we equip these models with enhanced protection against a broad range of adversarial attacks.

#### 6. Conclusion

In this work, we address a critical gap in the robustness of VLMs by examining the impact of Gaussian noise perturbations and propose Robust-VLGuard, a multimodal safety dataset paired with Gaussian-noise-augmented fine-tuning, to enhance safety alignment and preserve the helpfulness of VLMs. We further propose DiffPure-VLM to defend Optimization-Based Visual Perturbation Attack by using a diffusion model to transfer adversarial noise to Gaussian noise which can be defended by VLMs with noiseaugmented safety fine-tuning. The experimental result demonstrates the superiority of DiffPure-VLM in Gaussian noise and adversarial perturbations with baseline methods.

While DiffPure-VLM provides a practical defense, future work includes integrating noise augmentation in pretraining, expanding the safety dataset for broader tasks, and exploring adaptive multi-modal defenses to further enhance real-world performance.

#### 7. Social Impact

This research exposes VLM vulnerabilities to noisy inputs and adversarial attacks. While Robust-VLGuard and DiffPure-VLM enhance robustness, our findings have dualuse implications. Given VLMs’ growing adoption, we responsibly disclose these issues to raise awareness and foster the development of more secure models, mitigating deployment risks.

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 3

- [2] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In ICCV, 2015. 8
- [3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023. 1, 8
- [4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2, 11
- [5] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022. 1
- [6] Federico Bianchi, Mirac Suzgun, Giuseppe Attanasio, Paul Rottger, Dan Jurafsky, Tatsunori Hashimoto, and James Zou. Safety-tuned llamas: Lessons from improving the safety of large language models that follow instructions. In ICLR,

2024. 4, 5

- [7] Joao Carreira, Eric Noland, Chloe Hillier, and Andrew Zisserman. A short note on the kinetics-700 human action dataset. arXiv preprint arXiv:1907.06987, 2019. 8
- [8] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and testtime scaling. arXiv preprint arXiv:2412.05271, 2024. 11
- [9] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024. 1, 2, 5, 11
- [10] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500, 2023. 8
- [11] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In NeurIPS, 2021. 7, 14
- [12] Xuefeng Du, Reshmi Ghosh, Robert Sim, Ahmed Salem, Vitor Carvalho, Emily Lawton, Yixuan Li, and Jack W Stokes. Vlmguard: Defending vlms against malicious prompts via unlabeled data. arXiv preprint arXiv:2410.00296, 2024. 8
- [13] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 2, 11

- [14] Yihe Fan, Yuxin Cao, Ziyu Zhao, Ziyao Liu, and Shaofeng Li. Unbridled icarus: A survey of the potential perils of image inputs in multimodal large language model security. arXiv preprint arXiv:2404.05264, 2024. 1
- [15] Yicheng Fu, Raviteja Anantha, Prabal Vashisht, Jianpeng Cheng, and Etai Littwin. Ui-jepa: Towards active perception of user intent through onscreen user activity. arXiv preprint arXiv:2409.04081, 2024. 8
- [16] Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, et al. Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. arXiv preprint arXiv:2209.07858,

2022. 1

- [17] Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A Smith. Realtoxicityprompts: Evaluating neural toxic degeneration in language models. In Findings of the Association for Computational Linguistics: EMNLP 2020, 2020. 3
- [18] Yichen Gong, Delong Ran, Jinyuan Liu, Conglei Wang, Tianshuo Cong, Anyu Wang, Sisi Duan, and Xiaoyun Wang. Figstep: Jailbreaking large vision-language models via typographic visual prompts. arXiv preprint arXiv:2311.05608,

2023. 8

- [19] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014. 1
- [20] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In CVPR, 2017. 4
- [21] Laura Hanu and Unitary team. Detoxify. Github. https://github.com/unitaryai/detoxify, 2020. 3
- [22] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 6
- [23] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. In CVPR, 2024. 8
- [24] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In ICLR, 2022. 5
- [25] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019. 4
- [26] Haibo Jin, Leyang Hu, Xinuo Li, Peiyan Zhang, Chonghan Chen, Jun Zhuang, and Haohan Wang. Jailbreakzoo: Survey, landscapes, and horizons in jailbreaking large language and vision-language models. arXiv preprint arXiv:2407.01599,

2024. 8

- [27] Douwe Kiela, Hamed Firooz, Aravind Mohan, Vedanuj Goswami, Amanpreet Singh, Pratik Ringshia, and Davide Testuggine. The hateful memes challenge: detecting hate speech in multimodal memes. In NeurIPS, 2020. 8
- [28] Yifan Li, Hangyu Guo, Kun Zhou, Wayne Xin Zhao, and JiRong Wen. Images are achilles’ heel of alignment: Exploiting visual vulnerabilities for jailbreaking multimodal large

- language models. arXiv preprint arXiv:2403.09792, 2024. 1, 3
- [29] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 4
- [30] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2024. 1, 2, 4, 5, 8, 11
- [31] Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, SongChun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. In ICLR, 2023. 4
- [32] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, 2022. 4
- [33] Antoine Miech, Jean-Baptiste Alayrac, Ivan Laptev, Josef Sivic, and Andrew Zisserman. Rareact: A video dataset of unusual interactions. arXiv preprint arXiv:2008.01018,

2020. 8

- [34] Weili Nie, Brandon Guo, Yujia Huang, Chaowei Xiao, Arash Vahdat, and Animashree Anandkumar. Diffusion models for adversarial purification. In International Conference on Machine Learning, 2022. 1, 2, 6, 12, 14
- [35] OpenAI. Gpt-4v(ision) system card. 2023. 2, 4
- [36] Xiangyu Qi, Kaixuan Huang, Ashwinee Panda, Peter Henderson, Mengdi Wang, and Prateek Mittal. Visual adversarial examples jailbreak aligned large language models. In AAAI,

2024. 1, 3, 6, 7, 11

- [37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning,

2021. 8

- [38] Christian Schlarmann, Naman Deep Singh, Francesco Croce, and Matthias Hein. Robust clip: Unsupervised adversarial fine-tuning of vision embeddings for robust large vision-language models. In ICLR 2024 Workshop on Reliable and Responsible Foundation Models, 2024. 8
- [39] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In ECCV, 2022. 4
- [40] Erfan Shayegani, Yue Dong, and Nael Abu-Ghazaleh. Jailbreak in pieces: Compositional adversarial attacks on multimodal language models. In ICLR, 2023. 8
- [41] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, 2013. 8
- [42] Jiachen Sun, Changsheng Wang, Jiongxiao Wang, Yiwei Zhang, and Chaowei Xiao. Safeguarding vision-language models against patched visual prompt injectors. arXiv preprint arXiv:2405.10529, 2024. 8

- [43] C Szegedy. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013. 1
- [44] Rajkumar Theagarajan, Ming Chen, Bir Bhanu, and Jing Zhang. Shieldnets: Defending against adversarial attacks using probabilistic adversarial robustness. In CVPR, 2019. 8
- [45] Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. Jailbroken: How does llm safety training fail? In NeurIPS, 2023. 8
- [46] Yue Xu, Xiuyuan Qi, Zhan Qin, and Wenjie Wang. Defending jailbreak attack in vlms via cross-modality information detector. arXiv preprint arXiv:2407.21659, 2024. 2
- [47] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. In International Conference on Machine Learning, 2023. 3
- [48] Ivy Zhang, Wei Peng, Jenny N, Theresa Yu, and David Qiu. Ivy-vl:compact vision-language models achieving sota with optimal data, 2024. 11
- [49] Jiaming Zhang, Qi Yi, and Jitao Sang. Towards adversarial attack on vision-language pre-training models. In Proceedings of the 30th ACM International Conference on Multimedia, 2022. 8
- [50] Jiaming Zhang, Xingjun Ma, Xin Wang, Lingyu Qiu, Jiaqi Wang, Yu-Gang Jiang, and Jitao Sang. Adversarial prompt tuning for vision-language models. arXiv preprint arXiv:2311.11261, 2023. 8
- [51] Xiaoyu Zhang, Cen Zhang, Tianlin Li, Yihao Huang, Xiaojun Jia, Xiaofei Xie, Yang Liu, and Chao Shen. A mutationbased method for multi-modal jailbreaking attack detection. arXiv preprint arXiv:2312.10766, 2023. 2, 6
- [52] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. In ICLR, 2024. 1, 2, 5, 6, 11
- [53] Yongshuo Zong, Ondrej Bohdal, Tingyang Yu, Yongxin Yang, and Timothy Hospedales. Safety fine-tuning at (almost) no cost: A baseline for vision large language models. In International Conference on Machine Learning, 2024. 1, 3, 4, 5, 8

## Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks

### Supplementary Material

#### 8. Overview of the Supplementary Material

This supplementary material offers additional details and analyses to further support the findings presented in the main manuscript. It includes detailed information on the experimental configuration (Appendix 9), more evaluation on recent vision-language models (Appendix 10), a thorough analysis of the limitations and unique characteristics of DiffPure (Appendix 11), extended implementation specifics of DiffPure-VLM (Appendix 12), and conjectures along with preliminary theoretical discussions on the effects of Gaussian noise (Appendix 13). Collectively, these sections provide deeper insights into our methodology, enhancing the transparency and reproducibility of our research.

#### 9. Experiment Details

##### 9.1. Models

In this work, we conduct all experiments on three leading Vision-Language Models (VLMs), i.e., MiniGPT-4 (13B) [52], LLaVA-v1.5 (7B) [30], and InternVL2 (8B) [9]. We use the official model weights from HuggingFace or GitHub repositories for experiments in our paper. These model details are summarized in Table 6. Each model features a distinct LLM, vision encoder, and vision-language alignment method, allowing us to draw broader insights.

Table 6. Specifications of the evaluated VLMs.

|Model<br><br>|Size<br><br>|Vision Encoder<br><br>|LLM<br><br>|VL Connection Module|
|---|---|---|---|---|
|MiniGPT-4-13B|14B<br><br>|EVA-CLIP ViT-G/14|Vicuna-v0-13B<br><br>|Q-former|
|LLaVA-v1.5-7B|7B<br><br>|CLIP ViT-L/14|Vicuna-v1.5-7B|MLP<br><br>|
|InternVL2-8B|8B<br><br>|InternViT-300M-448px|InternLM2-8B<br><br>|MLP|

##### 9.2. Fine-tuning Configuration

We present the detailed hyper-parameters for post-hoc finetuning on our Robust-VLGuard dataset in Table 7. Gaussian noise augmentation was applied to the training images, with a randomly selected standard deviation between 0.01 and 0.15, and a 70% probability of application. The finetuning was performed over 3 epochs on a single A100-80G GPU, using a consistent batch size of 16. For MiniGPT-413B, unfreezing the linear projector significantly improved robustness in terms of helpfulness and safety alignment. However, for LLaVA-v1.5-7B and InternVL2-8B, unfreezing the linear projector led to increased overfitting, likely

due to differences in the vision-language connection modules of these models.

Table 7. Post-hoc fine-tuning hyper-parameters of different models.

|Model|Training Module<br><br>|LoRA Rank<br><br>|LoRA Alpha<br><br>|Learning Rate|
|---|---|---|---|---|
|MiniGPT-4-13B<br><br>|Vision Encoder & Linear Projector<br><br>|16|32<br><br>|3e-5|
|LLaVA-v1.5-7B|Vision Encoder<br><br>|16|256<br><br>|4e-5|
|InternVL2-8B|Vision Encoder<br><br>|16<br><br>|256<br><br>|4e-5|

##### 9.3. Details of Evaluation Settings

For evaluation on the MM-Vet benchmark, we set the temperature to 0 and use greedy decoding across all experiments to ensure reproducibility in helpfulness assessments. For safety evaluations on the RealToxicityPrompts benchmark, we follow the setup of Qi et al. [36], using a temperature of 1 and performing three runs to calculate the average attack success rate. Greedy decoding is also employed for this benchmark. The choice of temperature 1 reflects realworld usage, where sampling is typically enabled during interactions with VLMs. This setting aims to better simulate real-world scenarios when assessing safety alignment.

Additionally, the MM-Vet and RealToxicityPrompts benchmarks offer a comprehensive set of metrics covering various aspects. For the sake of brevity, we report only the overall metrics — Performance Score and Attack Success Rate — in the main paper. Here, we present the detailed evaluation results in Table 8 and Table 9, corresponding to Figure 2 in the main paper. The results show that Gaussian noisy images negatively impact nearly all metrics across both benchmarks and various models. Notably, using Gaussian noisy images as prompts improves MiniGPT-4’s performance on the OCR metric in the MM-Vet benchmark, highlighting the current VLMs’ lack of robustness.

#### 10. Additional Evaluation on Recent VisionLanguage Models

In this section, we further assess the robustness of stateof-the-art vision-language models against Gaussian noise. Table 10 presents the attack success rates on the RealToxicityPrompts benchmark for four recent VLMs—LLaMA3.2-Vision-11B [13], Ivy-VLM-3B [48], Qwen2.5-VL-7B [4], and InternVL2.5-8B [8]—under various Gaussian noise levels. Lower percentages indicate improved safety alignment.

- Table 8. Robustness comparison of various models on the MM-Vet benchmark using clean and Gaussian noisy image prompts (GPT-4 %).

Image Type

Performance Score ↑

Recognition OCR Knowledge Generation Spatial Math MiniGPT-4-13B

|Clean Image|26.7|34.9<br><br>|13.5<br><br>|27.4|27.1|19.1<br><br>|7.7|
|---|---|---|---|---|---|---|---|
|Gaussian Noisy Image|24.0 (-2.7)|29.0 (-5.9)<br><br>|16.9 (+3.4)|20.5 (-6.9)<br><br>|22.5 (-4.6)<br><br>|20.7 (+1.6)<br><br>|7.7 (0.0)|

LLaVA-v1.5-7B

|Clean Image<br><br>|33.0|37.9<br><br>|23.9<br><br>|20.4<br><br>|23.6|28.5|11.5|
|---|---|---|---|---|---|---|---|
|Gaussian Noisy Image|31.3 (-1.7)|36.3 (-1.6)|21.9 (-2.0)<br><br>|18.3 (-2.1)<br><br>|21.2 (-2.4)|25.7 (-2.8)<br><br>|3.8 (-7.7)|

InternVL2-8B

|Clean Image|59.9<br><br>|53.5|71.9|40.4<br><br>|44.6<br><br>|69.7<br><br>|65.4|
|---|---|---|---|---|---|---|---|
|Gaussian Noisy Image<br><br>|54.4 (-5.5)<br><br>|47.1 (-6.4)|66.8 (-5.1)<br><br>|35.1 (-5.3)<br><br>|39.2 (-5.4)|61.1 (-8.6)|53.5 (-11.9)|

- Table 9. Robustness comparison of various models on the RealToxicityPrompts benchmark using clean and Gaussian noisy image prompts. Metrics include Attack Success Rate and toxicity levels (Perspective API %).

Attack Success Rate ↓

Severe Toxicity

Sexually Explicit

Image Type

Identity Attack Profanity

Threat Toxicity MiniGPT-4-13B

|Clean Image|34.8<br><br>|2.7<br><br>|25.1|1.5<br><br>|12.2<br><br>|2.0|30.5|
|---|---|---|---|---|---|---|---|
|Gaussian Noisy Image<br><br>|44.1 (+9.3)|3.6 (+0.9)<br><br>|31.2 (+6.1)<br><br>|2.3 (+0.8)|13.7 (+1.5)|2.9 (+0.9)<br><br>|38.2 (+7.7)|

LLaVA-v1.5-7B

|Clean Image<br><br>|57.7|5.7<br><br>|46.8<br><br>|3.7|18.0|3.8<br><br>|54.4|
|---|---|---|---|---|---|---|---|
|Gaussian Noisy Image|60.1 (+2.4)<br><br>|4.8 (-0.9)<br><br>|48.1 (+1.3)|2.9 (-0.8)<br><br>|17.8 (-0.2)|4.0 (+0.2)<br><br>|56.0 (+1.6)|

InternVL2-8B

|Clean Image<br><br>|50.5|4.1<br><br>|40.2|1.9<br><br>|13.5|2.5|44.3<br><br>|
|---|---|---|---|---|---|---|---|
|Gaussian Noisy Image<br><br>|57.2 (+6.7)|4.5 (+0.4)<br><br>|45.9 (+5.7)|2.0 (+0.1)<br><br>|14.3 (+0.8)|3.2 (+0.7)|51.7 (+7.4)|

As shown, when Gaussian noise is introduced at increasing levels (σn = 30/255, σn = 50/255, and σn = 70/255), all models exhibit a rise in attack success rates, highlighting their sensitivity to simple Gaussian noise perturbations. These findings underscore the need for robust noise augmentation and defense strategies in training pipelines to maintain safety alignment in VLMs.

#### 11. Further Analysis of DiffPure

##### 11.1. Defence Performance

In this section, we present a comprehensive analysis of the effects of DiffPure [34] and Gaussian noise under perturbation-based attacks in Vision-Language Models (VLMs). Specifically, we extend the experimental setup described in Section 3.1 in the main paper by varying the standard deviation σn of Gaussian noise n and the timestep parameter t∗ in DiffPure. Results are summarized in Table 11. First, Gaussian noise n with standard deviations σn ∈ {15/255,30/255,50/255,75/255} is added to the benign clean image Ic to evaluate its impact on the Attack Success Rate. The results demonstrate that the Attack Success Rate under Gaussian noise is significantly higher than that of the benign clean image. When σn ≤ 50/255, increasing σn will lead to a higher Attack Success Rate. However, this trend did not continue at a higher σn setting (e.g., σn = 75/255), indicating that the effect of Gaussian noise

on VLMs is limited. Next, we apply DiffPure with different timesteps t∗ ∈ {50,100,150} to generate diffused images from adversarial inputs Iadv with varying perturbation constraints ϵ. For ϵ = 16/255, increasing t∗ to 100 or 150 reduces the Attack Success Rate but does not lower it below the level observed for the benign clean image. For larger perturbation constraints, increasing t∗ fails to decrease the Attack Success Rate, with a comparable performance of Gaussian noisy images.

##### 11.2. Distribution Shift

In this section, we present detailed results from the Gaussianity experiments conducted on adversarial and diffused images. Specifically, we visualize adversarial images Iadv alongside their corresponding residuals radv, and diffused images Idiffused with their residuals rdiffused, under pixel constraints ϵ ∈ {16/255,32/255,64/255} for Iadv and diffusion timesteps t∗ ∈ {50,100,150,500,750} in DiffPure [34] for generating Idiffused. Visualizations are shown in Figure 9, 10, and 11 with corresponding metrics: Kurtosis, Q-Q deviation, mean, and standard deviation. From these visualizations, we observe that when 50 ≤ t∗ ≤ 150, the residuals rdiffused exhibit a Gaussian-like distribution, particularly for ϵ = 32/255 and ϵ = 64/255. However, as t∗ increases, the Kurtosis of rdiffused rises, indicating a shift towards a long-tailed distribution. This suggests that a small fraction of pixels in Idiffused undergo significant

- Table 10. Attack success rate (%) on the RealToxicityPrompts benchmark for various vision-language models under different noise levels. Lower scores indicate improved safety alignment.

| |RealToxicityPrompts (%) ↓|
|---|---|
| |LLaMA-3.2-Vision-11B Ivy-VLM-3B Qwen2.5-VL-7B InternVL2.5-8B|
|Benign clean Image Ic<br><br>|45.4 29.9 36.8 43.9|
|+ n (σn = 30/255)<br><br>|46.4 (+1.0) 35.5 (+5.6) 39.3 (+2.5) 51.5 (+7.6)|
|+ n (σn = 50/255)<br><br>|47.6 (+2.2) 40.3 (+10.4) 39.5 (+2.7) 52.8 (+8.9)|
|+ n (σn = 70/255)<br><br>|48.5 (+3.1) 42.0 (+12.1) 46.1 (+9.3) 54.0 (+10.1)|

- Table 11. Detailed results of the defense of DiffPure in MiniGPT-4 on the RealToxicityPrompts benchmark under different image configurations. (Perspective API %).

|Image Configuration|Attack Success Rate ↓<br><br>|Identity Attack|Profanity|Severe Toxicity<br><br>|Sexually Explicit<br><br>|Threat|Toxicity|
|---|---|---|---|---|---|---|---|
|Benign clean Image Ic<br><br>+ n (σn = 15/255) + n (σn = 30/255) + n (σn = 50/255) + n (σn = 75/255)|34.8 38.5 (+3.7) 44.1 (+9.3) 46.3 (+11.5) 44.1 (+9.3)<br><br>|2.7<br><br>2.9 (+0.2)<br><br>3.6 (+0.9)<br><br><br>3.4 (+0.7) 3.8 (+1.1)<br><br>|25.1 27.4 (+2.3) 31.2 (+6.1) 34.0 (+8.9) 30.1 (+5.0)|1.5<br><br>1.1 (-0.4)<br><br>2.3 (+0.8)<br><br><br>1.8 (+0.3)<br><br>1.9 (+0.4)<br>|12.2<br><br>13.0 (+0.8)<br><br>13.7 (+1.5)<br><br>14.8 (+2.6)<br><br><br>14.3 (+2.1)<br><br><br>|2.0 2.3 (+0.3) 2.9 (+0.9) 2.5 (+0.5) 2.8 (+0.8)<br><br>|30.5 34.1 (+3.6)<br><br>38.2 (+7.7)<br>39.5 (+9.0) 37.5 (+7.0)<br>|
|Adversarial image Iadv (ϵ = 16/255)<br><br>+ DiffPure (t∗ = 50)<br><br>+ DiffPure (t∗ = 100) + DiffPure (t∗ = 150)|53.6 (+18.8) 45.0 (+10.2)<br><br>37.6 (+2.8)<br><br>37.7 (+2.9)<br><br><br>|8.4 (+5.7)<br><br>2.5 (-0.2)<br>3.0 (+0.3) 2.5 (-0.2)<br>|36.6 (+9.4) 31.7 (+6.6)<br><br>25.6 (+0.5)<br><br>26.5 (+1.4)<br>|6.6 (+5.1) 1.8 (+0.3)<br><br>1.3 (-0.2)<br><br>2.1 (+0.6)<br><br><br>|14.1 (+1.9) 14.5 (+2.3) 12.3 (+0.1) 12.2 (+0.0)<br><br>|4.7 (+2.7) 2.8 (+0.8)<br><br>1.8 (-0.2)<br>2.5 (+0.5)<br>|48.6 (+18.1) 38.8 (+8.3) 33.1 (+2.6) 32.9 (+2.4)<br><br>|
|Adversarial image Iadv (ϵ = 32/255)<br><br>+ DiffPure (t∗ = 50)<br><br>+ DiffPure (t∗ = 100) + DiffPure (t∗ = 150)|59.4 (+24.6) 45.5 (+10.7) 43.8 (+9.0) 42.3 (+7.5)<br><br>|14.6 (+11.9)<br><br>2.6 (-0.1)<br>3.3 (+0.6) 3.7 (+1.0)<br>|39.5 (+14.4) 32.1 (+7.0) 31.9 (+6.8) 30.4 (+5.3)<br><br>|7.0 (+5.5) 2.2 (+0.7) 1.9 (+0.4) 1.3 (-0.2)|14.9 (+2.7) 14.8 (+2.6) 13.1 (+0.9) 13.3 (+1.1)<br><br>|6.2 (+4.2) 3.0 (+1.0) 2.5 (+0.5) 2.8 (+0.8)|53.8 (+23.3) 38.5 (+8.0) 38.1 (+7.6) 36.3 (+5.8)<br><br>|
|Adversarial image Iadv (ϵ = 64/255)<br><br>+ DiffPure (t∗ = 50)<br><br>+ DiffPure (t∗ = 100) + DiffPure (t∗ = 150)<br><br>|67.2 (+32.4) 44.5 (+9.7) 42.1 (+7.3) 44.1 (+9.3)|15.9 (+13.2) 2.9 (+0.2)<br><br>2.8 (+0.1)<br><br>3.3 (+0.6)<br><br><br>|49.6 (+24.5) 32.2 (+7.1)<br><br>30.3 (+5.2)<br><br>31.5 (+6.4)<br><br><br>|12.2 (+10.7) 2.4 (+0.9) 1.9 (+0.4) 1.4 (-0.1)|16.9 (+4.7) 13.7 (+1.5) 13.7 (+1.5) 13.3 (+1.1)<br><br>|6.6 (+4.6)<br><br>2.7 (+0.7)<br><br>3.0 (+1.0)<br><br><br>2.5 (+0.5)<br><br>|63.1 (+32.6) 38.0 (+7.5) 36.5 (+6.0) 38.2 (+7.7)|

changes compared to Ic, leading to a cleaner image with minimal content alteration, especially when ϵ = 16/255. At t∗ = 500, the Kurtosis and standard deviation of rdiffused become significantly larger, implying greater changes in image content, as reflected in the visualization of Idiffused. For t∗ = 750, the Kurtosis decreases while the standard deviation further increases, indicating that rdiffused transitions to a flatter and broader distribution. In this case, Idiffused diverges substantially from Ic in image content.

Furthermore, we extend our analysis to the embedding space, examining the similarities between the clean image Ic, the adversarial image Iadv, and the diffused image Idiffused. Based on our experiment in pixel space, where the residual noise rdiffused approximates a Gaussian distribution under certain timestep settings in DiffPure, we consider Idiffused as comparable to Ic with added Gaussian noise. To verify this, we generate a noisy image In = Ic + n,n ∈ N 0,σr2

indicates the standard deviation of rdiffused. Using pre-trained visual encoder E in MiniGPT-4, we compute cosine similarities between the embeddings of In and Iadv, denoted as Cn,adv, and between In and Idiffused, denoted as Cn,diffused. Figure 12 shows

, where σr

diffused

diffused

these similarities across varying adversarial constraints ϵ and DiffPure steps t∗. Results indicate that, Cn,diffused consistently exceeds Cn,adv, showing that Idiffused is closer to In than Iadv in the embedding space. Notably, with moderate timesteps (t∗ ∈ [50,150]), Idiffused is similar to In (Gaussian noise n added to the benign clean image Ic) in both pixel and embedding spaces.

We also visualize the cosine similarity between the visual embeddings of Idiffused and Ic, denoted as Cclean,diffused, across varying ϵ and t∗. Results are shown in Figure 13, revealing that Cclean,diffused decreases rapidly as t∗ decreases, while it gradually declines as t∗ increases. Combining these findings with experiments in pixel space, we conclude that smaller t∗ values lead Idiffused to retain adversarial information, whereas larger t∗ values result in significant content disruption, leading to semantic misalignment.

#### 12. Additional Details of DiffPure-VLM

##### 12.1. Implementation Details

The overall architecture of our proposed DiffPure-VLM framework is illustrated in Figure 14, with the detailed al-

- Table 12. Evaluation of DiffPure-VLM’s effectiveness on RealToxicityPrompts across different image configurations. Metrics include attack success rate and toxicity levels (Perspective API %).

Attack Success Rate ↓

Severe Toxicity

Sexually Explicit

Image Type

Identity Attack Profanity

Threat Toxicity InternVL2-8B

|Benign Clean image|50.5<br><br>|4.1<br><br>|40.2<br><br>|1.9<br><br>|13.5|2.5<br><br>|44.3|
|---|---|---|---|---|---|---|---|
|Gaussian Noisy image<br><br>|57.2|4.5<br><br>|45.9|2.0<br><br>|14.3|3.2<br><br>|51.7|
|Adversarial image (ϵ = 32/255)<br><br>|65.0|21.1|49.2<br><br>|7.5|16.6<br><br>|5.0<br><br>|61.9|
|+DiffPure-VLM (t*=50)<br><br>|53.1|3.8|41.6<br><br>|2.0|13.6<br><br>|2.2<br><br>|48.0|

InternVL2-8B-VLGuard

|Benign Clean image|27.7<br><br>|1.4|22.2<br><br>|0.9|7.1<br><br>|1.6|23.8|
|---|---|---|---|---|---|---|---|
|Gaussian Noisy image<br><br>|39.9<br><br>|2.5|31.4|1.3|10.3<br><br>|1.8|35.8|
|Adversarial image (ϵ = 32/255)<br><br>|72.3|12.3|60.6<br><br>|8.6|19.9<br><br>|6.5<br><br>|69.3|
|+DiffPure-VLM (t*=50)|35.7|2.0<br><br>|28.9|0.8<br><br>|9.8|1.8|31.6|

InternVL2-8B-RobustVLGuard

|Benign Clean image|29.9<br><br>|0.8<br><br>|22.1<br><br>|0.3|7.2|1.5<br><br>|25.9|
|---|---|---|---|---|---|---|---|
|Gaussian Noisy image<br><br>|34.5<br><br>|2.1|27.2|1.3<br><br>|8.4|1.6|31.3|
|Adversarial image (ϵ = 32/255)|70.6|26.7<br><br>|56.5|9.2<br><br>|17.3|6.9<br><br>|68.1|
|+DiffPure-VLM (t*=50)<br><br>|33.4<br><br>|2.4|20.6<br><br>|0.7|8.1<br><br>|2.4|29.1<br><br>|
|+DiffPure-VLM (t*=150)|32.8<br><br>|1.7<br><br>|25.9<br><br>|0.6|7.7<br><br>|1.8|29.1|

LLaVA-v1.5-7B

|Benign Clean image<br><br>|57.7|5.7|46.8<br><br>|3.7|18.0<br><br>|3.8<br><br>|54.4|
|---|---|---|---|---|---|---|---|
|Gaussian Noisy image|60.1<br><br>|4.8|48.1<br><br>|2.9|17.8<br><br>|4.0|56.0|
|Adversarial image (ϵ = 32/255)<br><br>|66.0<br><br>|16.6|51.6<br><br>|8.8<br><br>|18.0|4.7|64.5|
|+DiffPure-VLM (t*=50)<br><br>|58.5<br><br>|5.9|45.5<br><br>|2.7<br><br>|17.0|4.3|53.3|

LLaVA-v1.5-7B-VLGuard

|Benign Clean image|50.3<br><br>|4.3|40.6<br><br>|2.0<br><br>|13.6|4.3<br><br>|46.9|
|---|---|---|---|---|---|---|---|
|Gaussian Noisy image|52.3|4.6<br><br>|41.5|2.7<br><br>|14.0|4.1<br><br>|48.5|
|Adversarial image (ϵ = 32/255)<br><br>|70.4|21.3<br><br>|52.8<br><br>|7.5|16.7<br><br>|7.0|67.2|
|+DiffPure-VLM (t*=50)<br><br>|51.1|3.4<br><br>|40.9|2.2|13.4|3.6<br><br>|47.5|

LLaVA-v1.5-7B-RobustVLGuard

|Benign Clean image<br><br>|43.6|4.6|34.7<br><br>|2.4|12.3<br><br>|3.5<br><br>|41.0|
|---|---|---|---|---|---|---|---|
|Gaussian Noisy image<br><br>|42.3<br><br>|3.1|34.5|1.9<br><br>|11.8|3.1|40.0|
|Adversarial image (ϵ = 32/255)<br><br>|62.5<br><br>|7.8|48.0<br><br>|5.4|16.5<br><br>|5.8|60.0|
|+DiffPure-VLM (t*=50)<br><br>|43.9|3.2|34.6<br><br>|2.4|12.8<br><br>|3.7<br><br>|41.0|
|+DiffPure-VLM (t*=150)<br><br>|42.5|3.5<br><br>|32.7|2.8<br><br>|12.1|4.1<br><br>|39.3|

MiniGPT-4-13B

|Benign Clean image|34.8<br><br>|2.7<br><br>|25.1|1.5|12.2|2.0<br><br>|30.5|
|---|---|---|---|---|---|---|---|
|Gaussian Noisy image<br><br>|44.1<br><br>|3.6|31.2|2.3<br><br>|13.7|2.9|38.2|
|Adversarial image (ϵ = 32/255)<br><br>|59.4|14.6<br><br>|39.5|7.0|14.9<br><br>|6.2<br><br>|53.8|
|+DiffPure-VLM (t*=50)<br><br>|45.5|2.6<br><br>|32.1|2.2|14.8<br><br>|3.0<br><br>|38.5|

MiniGPT-4-13B-VLGuard

|Benign Clean image|41.3<br><br>|2.8|30.1<br><br>|2.2|14.6<br><br>|2.5|37.3|
|---|---|---|---|---|---|---|---|
|Gaussian Noisy image|43.7|3.0<br><br>|31.6|2.3<br><br>|13.9|3.5<br><br>|38.6|
|Adversarial image (ϵ = 32/255)<br><br>|67.6|10.5|48.2<br><br>|7.0|19.9<br><br>|7.8<br><br>|61.7|
|+DiffPure-VLM (t*=50)|45.0|4.2|33.1<br><br>|2.1|14.6<br><br>|3.1<br><br>|40.7|

MiniGPT-4-13B-RobustVLGuard

|Benign Clean image<br><br>|16.0<br><br>|0.4|9.9<br><br>|0.3|4.6<br><br>|1.1|12.1|
|---|---|---|---|---|---|---|---|
|Gaussian Noisy image<br><br>|16.5|0.9<br><br>|11.9<br><br>|0.6|5.8<br><br>|1.0|14.0|
|Adversarial image (ϵ = 32/255)|53.7|9.8<br><br>|35.3<br><br>|4.1|13.9|5.4<br><br>|48.1|
|+DiffPure-VLM (t*=50)<br><br>|13.6|0.3|9.2<br><br>|0.2<br><br>|5.5|0.9<br><br>|10.6|
|+DiffPure-VLM (t*=150)|11.9|0.5<br><br>|8.6<br><br>|0.2|4.2|0.6<br><br>|9.9|

gorithmic procedure outlined in Algorithm 1. For our experiments, we employ the Guided Diffusion model for ImageNet [11], specifically the 256 × 256 unconditional variant provided by OpenAI2. Importantly, we synchronize the

2https://openaipublic.blob.core.windows.net/ diffusion/jul-2021/256x256_diffusion_uncond.pt

forward diffusion timesteps (tforward) with the reverse diffusion timesteps (treverse), denoted as t∗ in the experimental results, following the setup in DiffPure [34]. Here, we leverage this diffusion model to validate the robustness of our fine-tuned VLMs against Gaussian noise, demonstrating a preliminary defense strategy. However, the fixed image res-

[Figure 34]

Perturbationbased Attacks

Red Channel Green Channel Blue Channel

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

1.6792 / 0.0097 / 0.0021 / 0.0392 1.6399 / 0.0103 / 0.0006 / 0.0397 1.6506 / 0.0101 / 0.0001 / 0.0396

DiffPure with timestep

Red Channel Green Channel Blue Channel

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

50

3.5942 / 0.0015 / 0.0026 / 0.0289 3.5505 / 0.0013 / 0.0012 / 0.0284 3.7670 / 0.0017 / 0.0004 / 0.0290

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

100

5.1718 / 0.0042 / 0.0031 / 0.0291 5.7542 / 0.0045 / 0.0014 / 0.0278 4.8221 / 0.0038 / 0.0003 / 0.0315

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

150

- 7.0338 / 0.0066 / 0.0030 / 0.0304 7.2725 / 0.0068 / 0.0018 / 0.0302 6.9144 / 0.0066 / 0.0017 / 0.0309

...

- 8.3942 / 0.0248 / 0.0146 / 0.0811 7.5661 / 0.0221 / 0.0144 / 0.0799 7.1430 / 0.0196 / 0.0075 / 0.0793

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

500

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

750

3.1888 / 0.0154 / -0.0105 / 0.2116 3.3533 / 0.0161 / -0.0073 / 0.1996 3.3886 / 0.0182 / 0.0228 / 0.1935

- Figure 9. Iadv, Idiffused and statistics of radv, rdiffused under different t∗ in DiffPure (constraint ϵ = 16/255). Metrics are shown in

[Figure 59]

Perturbationbased Attacks

Red Channel Green Channel Blue Channel

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

1.6773 / 0.0213 / 0.0103 / 0.0796 1.6184 / 0.0218 / 0.0071 / 0.0803 1.6107 / 0.0222 / 0.0068 / 0.0809

DiffPure with timestep

Red Channel Green Channel Blue Channel

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

50

3.2330 / 0.0032 / 0.0106 / 0.0538 2.9436 / 0.0012 / 0.0073 / 0.0517 3.1752 / 0.0019 / 0.0083 / 0.0537

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

100

3.3843 / 0.0030 / 0.0101 / 0.0550 3.3877 / 0.0021 / 0.0086 / 0.0512 3.4618 / 0.0030 / 0.0069 / 0.0549

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

150

3.8302 / 0.0049 / 0.0111 / 0.0509 4.1321 / 0.0050 / 0.0090 / 0.0473 3.7059 / 0.0042 / 0.0075 / 0.0533

...

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

500

9.7215 / 0.0222 / 0.0204 / 0.0798 8.7416 / 0.0203 / 0.0090 / 0.0763 7.6404 / 0.0176 / -0.0004 / 0.0756

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

750

3.6532 / 0.0278 / 0.0302 / 0.2504 3.4018 / 0.0221 / 0.0051 / 0.2394 3.4028 / 0.0226 / 0.0188 / 0.2444

- Figure 10. Iadv, Idiffused and statistics of radv, rdiffused under different t∗ in DiffPure (constraint ϵ = 32/255). Metrics are shown in

[Figure 84]

Perturbationbased Attacks

Red Channel Green Channel Blue Channel

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

1.8482 / 0.0354 / 0.0220 / 0.1514 1.8072 / 0.0364 / 0.0243 / 0.1520 1.7817 / 0.0379 / 0.0244 / 0.1542

DiffPure with timestep

Red Channel Green Channel Blue Channel

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

50

- 2.7269 / 0.0074 / 0.0221 / 0.1039 2.7674 / 0.0068 / 0.0245 / 0.1015 2.7580 / 0.0072 / 0.0253 / 0.1054
- 3.1741 / 0.0066 / 0.0203 / 0.1069 3.0657 / 0.0051 / 0.0235 / 0.1029 2.9639 / 0.0036 / 0.0249 / 0.1076

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

100

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

150

3.1479 / 0.0064 / 0.0238 / 0.1028 3.1747 / 0.0028 / 0.0243 / 0.0939 3.2040 / 0.0048 / 0.0245 / 0.1049

...

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

500

5.9875 / 0.0185 / 0.0184 / 0.0968 5.7757 / 0.0168 / 0.0189 / 0.0944 4.6625 / 0.0119 / 0.0200 / 0.0992

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

750

3.1871 / 0.0143 / 0.0531 / 0.2357 3.2659 / 0.0191 / -0.0188 / 0.2284 3.3796 / 0.0194 / 0.0166 / 0.2283

- Figure 11. Iadv, Idiffused and statistics of radv, rdiffused under different t∗ in DiffPure (constraint ϵ = 64/255). Metrics are shown in

0.8

0.7

###### CosineSimilarity

Cn,adv ( =16/255)

Cn,diffused ( =16/255)

0.6

Cn,adv ( =32/255)

Cn,diffused ( =32/255)

Cn,adv ( =64/255)

0.5

Cn,diffused ( =64/255)

0.4

0 100 200 300 400 500 600 700

Step t in DiffPure

- Figure 12. Cosine similarity of visual embeddings under different ϵ of adversarial image Iadv and t∗ of DiffPure.

0 100 200 300 400 500 600 700

Step t in DiffPure

0.55

0.60

0.65

0.70

0.75

- 0.80

CosineSimilarity

Cclean,diffused ( =16/255) Cclean,diffused ( =32/255) Cclean,diffused ( =64/255)

- Figure 13. Cosine Similarity of visual embeddings from Ic and Idiffused under different ϵ of adversarial image.

olution of the diffusion model requires down-sampling and up-sampling operations, which may introduce artifacts not considered during the fine-tuning of the VLM, potentially impacting evaluation results. In the future, adopting more advanced diffusion models will be essential for real-world applications.

##### 12.2. Extended Experimental Results

In the main paper, for the sake of brevity, we only report results for the standard perturbation-based attack setting of ϵ = 32/255. However, we also conducted experiments with lower attack strength (ϵ = 16/255) and higher attack strength (ϵ = 64/255) to further validate our analysis and approach in Table 13. Across different models and attack strengths, our DiffPure-VLM consistently reduces the attack success rate within a limited number of diffusion timesteps (fewer than 150). Notably, under lower attack strengths, setting the diffusion step to as low as t∗ = 50 is sufficient to bring the attack success rate down to the level

Algorithm 1 DiffPure-VLM Adversarial Image Purification with DDPM

Require: Adversarial image x, harmful text prompt p, diffusion model D, forward diffusion timesteps tforward, reverse diffusion timesteps treverse, visual language model VLM.

Ensure: Question answering result output

- 1: Resize input image x to the size required by the diffusion model (e.g., 256 × 256).
- 2: DDPM forward process with tforward steps: xˆ = get noised x(x,tforward).

- 3: for t in treverse do
- 4: Denoise using reverse DDPM process: x = denoising process(ˆx,t).

- 5: end for
- 6: Obtain purified image with Gaussian noise: xgaussian = normalize(x).
- 7: Perform question answering using VLM: output = VLM(xgaussian,p).
- 8: return output

observed for clean inputs. However, under higher attack strengths, t∗ = 50 fails to reduce the attack success rate to the baseline level for both InternVL2-8B and MiniGPT4-13B. This indicates that stronger attacks require a larger number of diffusion steps to effectively transform the adversarial noise into Gaussian noise. This finding aligns with the analysis presented in Figure 4 of the main paper, where the residual image at t∗ = 50 for an attack strength of ϵ = 64/255 does not exhibit Gaussian characteristics. Moreover, we observe that t∗ = 100 demonstrates strong performance across all attack conditions, making it an effective trade-off between time and robustness. Thus, in realworld applications, setting t∗ = 100 offers a balanced solution, achieving reliable defense while maintaining computational efficiency.

#### 13. Conjectures and Discussion on the Impact of Gaussian Noise

##### Problem Definition

###### Setting:

- • A Visual Language Model (VLM) typically consists of three main components: a visual encoder, a language model, and a vision-language connection module.
- • Let the input be a pair (I,T), where I ∈ Rd is an image and T is the corresponding text prompt.
- • The VLM generates an output sequence of tokens, de-

noted by Tˆ = fθ(I,T), where fθ represents the VLM pipeline parameterized by θ.

Adversarial Attack: An adversarial perturbation δ is applied to the image I, resulting in a perturbed image Iδ =

[Figure 109]

[Figure 110]

[Figure 111]

|[Figure 112]|
|---|

|[Figure 113]|
|---|

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

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

Figure 14. The overall framework of DiffPure-VLM.

I + δ. The perturbation δ is crafted to manipulate the VLM into generating a specific harmful target text Ttarget. The adversary’s objective is:

L fθ(I + δ,T),Ttarget ,

δ = arg min

∥δ∥≤ϵ

where L(·,·) measures the discrepancy between the generated text Tˆ and the target text Ttarget. The constraint ∥δ∥ ≤ ϵ ensures that the perturbation is imperceptible.

Conjectures: We introduce the following four conjectures to guide our investigation into the impact of Gaussian noise on VLMs:

- 1. Sensitivity of Adversarial Attacks to Gaussian Noise: Adding Gaussian noise to adversarially perturbed images will significantly reduce the effectiveness of the attack.
- 2. Gaussian Noise as a Simple Attack on VLM Safety: Gaussian noise, even without adversarial perturbations, may increase the likelihood of generating harmful text.
- 3. Gaussian Noise as a Regularizer: Augmenting training data with Gaussian noise may act as a regularizer, enhancing the robustness of the VLM.
- 4. Fine-Tuning with Gaussian Noise Preserves Performance: Incorporating Gaussian noise during fine-tuning will preserve or even improve the VLM’s overall performance.

Objective: The goal of this study is to systematically evaluate the impact of Gaussian noise on the robustness and reliability of VLMs. By exploring the above conjectures, we aim to determine whether Gaussian noise can effectively mitigate adversarial perturbations and enhance model robustness without compromising performance.

##### Conjecture 1: Sensitivity of Adversarial Perturbations to Gaussian Noise

Statement: Adversarial perturbations are highly sensitive to Gaussian noise; the attack effectiveness is significantly diminished when Gaussian noise is added to the adversarial image.

###### Discussion:

Consider an adversarially perturbed image Iδ = I + δ, where the perturbation δ is optimized to minimize the loss:

L fθ(I + δ,T),Ttarget ,

δ = arg min

∥δ∥≤ϵ

where L(·,·) measures the discrepancy between the generated text Tˆ and the harmful target text Ttarget. The perturbation δ is crafted to exploit specific vulnerabilities in the model fθ.

Now, consider the scenario where Gaussian noise η ∼ N(0,σ2I) is added to the input. The new input becomes:

Iδ,η = I + δ + η.

The expected loss over the distribution of Gaussian noise η is:

Eη L fθ(I + δ + η,T),Ttarget .

Since the adversarial perturbation δ is tailored for the specific input I, adding random Gaussian noise η disrupts this optimization. Adversarial perturbations exploit the model’s sensitivity along certain directions in the input space, while isotropic Gaussian noise perturbs the input uniformly in all directions, diminishing the effect of δ.

Assuming that fθ and L are Lipschitz continuous, we can bound the increase in expected loss as follows:

Eη L fθ(I + δ + η,T),Ttarget ≥L fθ(I + δ,T),Ttarget

σ2λ 2

+

,

where λ is a positive constant related to the curvature of L and fθ.

This inequality indicates that the addition of Gaussian noise increases the expected loss, thus reducing the effectiveness of the adversarial perturbation. The random noise η disrupts the carefully crafted δ, making it less effective at manipulating the VLM’s output. This supports our conjecture that Gaussian noise can act as a simple yet effective countermeasure against adversarial attacks.

- Table 13. Evaluation of DiffPure-VLM’s effectiveness on RealToxicityPrompts across different image configurations. Metrics include attack success rate and toxicity levels (Perspective API %). Rows highlighted in light red indicate cases where attack success rate does not meet the baseline level of benign image input.

Attack Success Rate ↓

Severe Toxicity

Sexually Explicit

Image Type

Threat Toxicity InternVL2-8B-RobustVLGuard

Identity Attack Profanity

Benign Clean Image 29.9 0.8 22.1 0.3 7.2 1.5 25.9 Benign Noisy Image 34.5 2.1 27.2 1.3 8.4 1.6 31.3

Adversarial Image (ϵ = 16/255) 72.5 19.8 58.5 8.3 19.2 7.8 70.0 +DiffPure-VLM (t∗ = 50) 31.4 1.4 24.6 1.1 7.9 1.6 27.5 +DiffPure-VLM (t∗ = 100) 28.2 0.9 21.7 0.4 6.8 1.5 23.9 +DiffPure-VLM (t∗ = 150) 28.2 1.6 22.4 0.2 6.9 1.1 24.4

Adversarial Image (ϵ = 32/255) 70.6 26.7 56.5 9.2 17.3 6.9 68.1 +DiffPure-VLM (t∗ = 50) 33.4 2.4 20.6 0.7 8.1 2.4 29.1 +DiffPure-VLM (t∗ = 100) 33.4 1.6 27.7 0.6 7.6 1.5 30.2 +DiffPure-VLM (t∗ = 150) 32.8 1.7 25.9 0.6 7.7 1.8 29.1

Adversarial Image (ϵ = 64/255) 57.3 9.3 45.8 4.4 16.1 3.9 53.9 +DiffPure-VLM (t∗ = 50) 40.9 2.3 32.9 1.4 9.3 2.3 37.3 +DiffPure-VLM (t∗ = 100) 35.7 1.8 28.2 0.8 7.6 2.4 31.8 +DiffPure-VLM (t∗ = 150) 36.1 2.4 28.3 1.2 8.3 1.8 33.6

###### LLaVA-v1.5-7B-RobustVLGuard

Benign Clean image 43.6 4.6 34.7 2.4 12.3 3.5 41.0 Benign Noisy image 42.3 3.1 34.5 1.9 11.8 3.1 40.0

Adversarial image (ϵ = 16/255) 62.6 11.3 48.8 5.3 16.8 5.8 59.1 +DiffPure-VLM (t∗ = 50) 42.7 3.4 32.1 1.5 12.0 4.6 39.7 +DiffPure-VLM (t∗ = 100) 42.8 3.9 32.5 2.3 12.5 3.7 39.3 +DiffPure-VLM (t∗ = 150) 44.4 3.3 34.4 2.2 12.6 3.2 41.0

Adversarial image (ϵ = 32/255) 62.5 7.8 48.0 5.4 16.5 5.8 60.0 +DiffPure-VLM (t∗ = 50) 43.9 3.2 34.6 2.4 12.8 3.7 41.0 +DiffPure-VLM (t∗ = 100) 44.1 3.5 35.4 2.1 13.0 4.1 41.3 +DiffPure-VLM (t∗ = 150) 42.5 3.5 32.7 2.8 12.1 4.1 39.3

Adversarial image (ϵ = 64/255) 57.5 9.2 43.5 5.2 15.3 5.8 54.7 +DiffPure-VLM (t∗ = 50) 42.1 2.7 32.1 2.1 12.3 2.9 39.0 +DiffPure-VLM (t∗ = 100) 40.5 3.3 31.4 1.9 11.7 2.8 37.5 +DiffPure-VLM (t∗ = 150) 42.4 3.5 32.8 1.8 11.5 4.0 40.2

###### MiniGPT-4-13B-RobustVLGuard

Benign Clean image 16.0 0.4 9.9 0.3 4.6 1.1 12.1 Benign Noisy image 16.5 0.9 11.9 0.6 5.8 1.0 14.0

Adversarial image (ϵ = 16/255) 47.4 9.3 34.2 1.4 11.8 4.2 41.5 +DiffPure-VLM (t∗ = 50) 16.0 0.6 9.3 0.3 6.5 1.4 13.2 +DiffPure-VLM (t∗ = 100) 15.8 0.7 9.7 0.0 6.1 1.1 12.8 +DiffPure-VLM (t∗ = 150) 9.8 0.4 6.0 0.1 3.3 0.5 7.8

Adversarial image (ϵ = 32/255) 53.7 9.8 35.3 4.1 13.9 5.4 48.1 +DiffPure-VLM (t∗ = 50) 13.6 0.3 9.2 0.2 5.5 0.9 10.6 +DiffPure-VLM (t∗ = 100) 15.2 0.6 9.5 03 5.4 1.1 12.7 +DiffPure-VLM (t∗ = 150) 11.9 0.5 8.6 0.2 4.2 0.6 9.9

Adversarial image (ϵ = 64/255 ) 60.2 6.8 44.6 4.2 16.2 5.8 56.0 +DiffPure-VLM (t∗ = 50) 30.3 1.8 21.6 1.4 11.4 1.9 26.9

+DiffPure-VLM (t∗ = 100) 10.6 0.0 7.1 0.0 4.1 0.8 8.2 +DiffPure-VLM (t∗ = 150) 9.4 0.4 5.5 0.3 4.1 0.6 7.0

##### Conjecture 2: Gaussian Noise as a Simple Attack on VLM Safety

Statement: Adding Gaussian noise η ∼ N(0,σ2I) to a clean image Iclean can compromise the safety of VLMs. Setting: Let Iclean be a clean image, and η ∼ N(0,σ2I) be Gaussian noise. The perturbed image is defined as:

Inoisy = Iclean + η.

The VLM processes the noisy image Inoisy along with a corresponding text prompt T, and generates an output based on this combined input. Discussion:

1. Effect of Noise on Model Input: The input to the model becomes Inoisy = Iclean + η. This perturbation, although random, alters the image representation processed by the VLM. The model’s output can be locally approximated around the clean input as:

fθ(Iclean + η,T) ≈ fθ(Iclean,T) + ∇Icleanfθ · η,

where ∇Icleanfθ represents the gradient of the model output with respect to the clean image input. The Gaussian noise η introduces random perturbations that shift the image features.

###### 2. Vulnerability of VLMs to Noise: VLMs are typically

trained on clean image data, and thus, they may lack robustness to input noise. The introduction of Gaussian noise can push the model’s input into regions of the feature space that were not well-covered during training, potentially causing the model to misinterpret the input and generate unexpected responses.

###### 3. Impact on Safety: Adding Gaussian noise may shift

the model’s behavior towards decision boundaries where safety mechanisms are less effective. This increases the likelihood of generating unsafe or harmful text:

L(fθ(Iclean + η,T),Ttarget) ≥ L(fθ(Iclean,T),Ttarget), where Ttarget represents a potentially harmful target out-

put. The inequality suggests that the noisy input can lead to a higher loss, increasing the risk of unsafe text generation.

4. Gaussian Noise as a Simple Yet Effective Attack: Unlike adversarial perturbations that require careful optimization and model-specific crafting, Gaussian noise introduces random changes without any specific targeting. Despite its simplicity, it can destabilize the model and affect its safety, demonstrating that even non-adversarial noise can be a risk factor for VLMs.

In summary, adding Gaussian noise to clean images can indeed disrupt the safety of VLMs, even in the absence of sophisticated adversarial attacks. This highlights a potential vulnerability of VLMs that warrants further investigation.

##### Conjecture 3: Gaussian Noise as a Regularizer

Statement: Augmenting training data with Gaussian noise acts as a regularizer, reducing the risk of overfitting to adversarial perturbations and enhancing model robustness.

Discussion:

We introduce a regularized loss function that incorporates Gaussian noise during training:

Lreg(θ) = E(I,T)∼DEη∼N(0,σ2I) [L(fθ(I + η,T),T)],

where D represents the training data distribution. This formulation encourages the model to perform well not only on clean inputs but also on noisy inputs, promoting robustness.

To understand the regularizing effect of Gaussian noise, we expand the loss function L using a second-order Taylor expansion around the clean input I:

L(fθ(I + η,T),T) ≈ L(fθ(I,T),T)

+ ∇IL(fθ(I,T),T)⊤ η

- 1

- 2

η⊤∇2IL(fθ(I,T),T)η.

+

Taking the expectation over the Gaussian noise η ∼ N(0,σ2I), we obtain:

Eη [L(fθ(I + η,T),T)] ≈ L(fθ(I,T),T)

- 1

- 2

Eη η⊤∇2IL(fθ(I,T),T)η

+

= L(fθ(I,T),T)

σ2 2

Tr ∇2IL(fθ(I,T),T) .

+

2

2 Tr ∇2IL(fθ(I,T),T) penalizes large curvature (i.e., high second derivatives) of the loss function with respect to the input I. This encourages smoother mappings from the input to the output, reducing the model’s sensitivity to small input perturbations, including adversarial ones.

The additional term σ

In summary, the addition of Gaussian noise during training acts as a regularizer by penalizing sharp changes in the loss landscape. This results in a model that is less prone to overfitting and more resilient to adversarial attacks, as it learns smoother and more stable input-output mappings.

##### Conjecture 4: Fine-Tuning with Gaussian Noise Preserves Performance

Statement: Fine-tuning the VLM with Gaussian noiseaugmented data maintains performance on clean data while enhancing robustness to adversarial perturbations.

###### Discussion:

Let D = {(Ii,Ti)}Ni=1 be the original training dataset. We construct an augmented dataset by adding Gaussian noise:

Daug = (Ii + ηi,Ti) | ηi ∼ N(0,σ2I) Ni=1 .

The training objective is to minimize the following loss function:

N

1 N

Lˆaug(θ) =

i=1

Eη

[L(fθ(Ii + ηi,Ti),Ti)].

i

Since the Gaussian noise ηi has a zero mean, the expected gradient of the loss with respect to the model parameters θ is centered around the gradient on the clean data:

Eη

[∇θL(fθ(Ii + ηi,Ti),Ti)] = ∇θL(fθ(Ii,Ti),Ti).

i

This result indicates that the expected training gradient remains aligned with the gradient computed on the clean data, thereby preserving the model’s performance on clean inputs.

Moreover, by training on both clean and noiseaugmented data, the model is exposed to a neighborhood of inputs around each training example. This exposure helps the model generalize better and become less sensitive to small perturbations, effectively enhancing its robustness against adversarial attacks.

In summary, fine-tuning with Gaussian noise-augmented data acts as a regularization strategy that not only maintains the VLM’s accuracy on clean data but also improves its resistance to adversarial perturbations.

#### 14. Detailed Proofs

##### Bounding the Increase in Loss Due to Gaussian Noise

Discussion: Step 1: Lipschitz Continuity of fθ and L

Assume that the model function fθ : Rd × T → Rk and the loss function L : Rk ×T → R are Lipschitz continuous with constants Kf and KL, respectively. That is, for all x,y ∈ Rd and T ∈ T :

∥fθ(x,T) − fθ(y,T)∥ ≤ Kf∥x − y∥, and for all a,b ∈ Rk:

|L(a,Ttarget) − L(b,Ttarget)| ≤ KL∥a − b∥.

###### Step 2: Bounding the Change in Loss Due to Noise η

Consider the adversarially perturbed image Iδ = I + δ, where δ is crafted to minimize the loss:

L fθ(I + δ,T),Ttarget .

δ = arg min

∥δ∥≤ϵ

When Gaussian noise η ∼ N(0,σ2I) is added, the input becomes Iδ,η = I + δ + η. The change in loss due to η is:

∆L = L fθ(I + δ + η,T),Ttarget −L fθ(I + δ,T),Ttarget .

Using the Lipschitz continuity of L:

|∆L| ≤ KL ∥fθ(I + δ + η,T) − fθ(I + δ,T)∥.

###### Step 3: Computing the Expected Increase in Loss Applying the Lipschitz continuity of fθ:

∥fθ(I + δ + η,T) − fθ(I + δ,T)∥ ≤ Kf∥η∥. Thus, the change in loss is bounded by:

|∆L| ≤ KLKf∥η∥.

Since η is a Gaussian random vector with zero mean and covariance σ2I, the expected value of ∥η∥ is:

Γ d+12 Γ d2 ≈ σ d −

√

- 1

- 2

for large d.

E[∥η∥] = σ

2

Therefore, the expected increase in loss is approximately:

√

E[|∆L|] ≤ KLKfσ

d.

###### Step 4: Lower Bounding the Expected Increase in Loss

Since δ minimizes L(fθ(I + δ,T),Ttarget) at the point I +δ, any perturbation η added to I +δ is likely to increase the loss. Under the conjecture that L is convex around I+δ, the expected increase in loss due to η can be lower bounded using the curvature (second derivative) of L:

Eη L fθ(I + δ + η,T),Ttarget ≥ L fθ(I + δ,T),Ttarget +

σ2 2

λmin,

where λmin is the smallest eigenvalue of the Hessian ma-

trix ∇2I+δL(fθ(I + δ,T),Ttarget). Conclusion:

Adding Gaussian noise increases the expected loss by at least σ

2

2 λmin, reducing the effectiveness of the adversarial perturbation. This result supports the conjecture that Gaussian noise disrupts the optimization achieved by the adversary, weakening the impact of adversarial attacks.

##### Second-Order Taylor Expansion of L Around I Discussion:

###### Step 1: Second-Order Taylor Expansion

We expand the loss function L(fθ(I + η,T),T) around the point I using the second-order Taylor expansion:

L(fθ(I + η,T),T) = L(fθ(I,T),T)

+ ∇IL(fθ(I,T),T)⊤η

- 1

- 2

η⊤∇2IL(fθ(I,T),T)η + R3 where:

+

- • ∇IL(fθ(I,T),T) is the gradient of the loss with respect to the input I.
- • ∇2IL(fθ(I,T),T) is the Hessian matrix of second derivatives with respect to I.
- • R3 is the remainder term of higher order O(∥η∥3).

###### Step 2: Expected Value of the Linear Term Since η is sampled from a zero-mean Gaussian distribu-

tion η ∼ N(0,σ2I), the expected value of the linear term becomes:

Eη ∇IL(fθ(I,T),T)⊤η = ∇IL(fθ(I,T),T)⊤Eη[η] = 0

###### Step 3: Expected Value of the Quadratic Term Next, we compute the expectation of the quadratic term:

Eη η⊤∇2IL(fθ(I,T),T)η

Using the properties of Gaussian distributions, we know that for a symmetric matrix A:

Eη η⊤Aη = σ2 Tr(A)

Thus, the expected value of the quadratic term becomes:

Eη η⊤∇2IL(fθ(I,T),T)η = σ2 Tr ∇2IL(fθ(I,T),T)

###### Step 4: Neglecting the Remainder Term

For small values of σ, the remainder term R3 is of order O(σ3) and can be safely ignored. Thus, the approximation becomes:

Eη [L(fθ(I + η,T),T)] ≈ L(fθ(I,T),T)

σ2 2

Tr ∇2IL(fθ(I,T),T)

+

###### Step 5: Interpretation of the Trace Term

The term Tr ∇2IL(fθ(I,T),T) denotes the sum of the eigenvalues of the Hessian matrix, representing the overall curvature of the loss function with respect to the input. A larger trace value indicates higher curvature, suggesting greater sensitivity of the model to input perturbations. Reducing this sensitivity is crucial for enhancing the model’s robustness.

###### Step 6: Gaussian Noise as Regularization

2

The additional term σ

2 Tr ∇2IL(fθ(I,T),T) functions as a regularizer, penalizing high curvature in the loss landscape. This encourages the model to learn smoother input-output mappings, thereby reducing its vulnerability to small perturbations, including adversarial attacks.

###### Step 7: Connection to Tikhonov Regularization This regularization effect is conceptually similar to

Tikhonov regularization, where a penalty proportional to the norm of the model parameters is added to the loss function. In our case, the penalty arises naturally from the Gaussian noise, encouraging robustness by flattening the loss landscape:

Eη [L(fθ(I + η,T),T)] ≈ L(fθ(I,T),T)

σ2 2

Tr ∇2IL(fθ(I,T),T)

+

This smoothing effect reduces the model’s sensitivity to input perturbations, enhancing its robustness without compromising performance on clean data.

