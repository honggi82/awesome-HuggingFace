# arXiv:2501.13926v2[cs.CV]23Jul2025

## Can We Generate Images with CoT? Let’s Verify and Reinforce Image Generation Step by Step

##### Ziyu Guo 1†, Renrui Zhang 1*†, Chengzhuo Tong3†, Zhizheng Zhao2†, Rui Huang4, Haoquan Zhang3, Manyuan Zhang1, Jiaming Liu2, Shanghang Zhang2, Peng Gao3, Hongsheng Li 1, Pheng-Ann Heng 1

1The Chinese University of Hong Kong, 999077, Hong Kong SAR. 2Peking University, Beijing, China. 3Shanghai AI Laboratory, Shanghai, 200232, China. 4University of Electronic Science and Technology of China, Chengdu, China.

*Corresponding author(s). E-mail(s): 1700012927zrr@gmail.com; Contributing authors: guoziyu86@gmail.com; †These authors contributed equally to this work.

Abstract Chain-of-Thought (CoT) reasoning has been extensively explored in large models to tackle complex understanding tasks. However, it still remains an open question whether such strategies can be applied to verifying and reinforcing image generation scenarios. In this paper, we provide the first comprehensive investigation of the potential of CoT reasoning to enhance autoregressive image generation. We focus on three techniques: scaling test-time computation for verification, aligning model preferences with Direct Preference Optimization (DPO), and integrating these techniques for complementary effects. Our results demonstrate that these approaches can be effectively adapted and combined to significantly improve image generation performance. Furthermore, given the pivotal role of reward models in our findings, we propose the Potential Assessment Reward Model (PARM) and PARM++, specialized for autoregressive image generation. PARM adaptively assesses each generation step through a potential assessment approach, merging the strengths of existing reward models, and PARM++ further introduces a reflection mechanism to self-correct the generated unsatisfactory image, which is the first to incorporate reflection in autoregressive image generation. Using our investigated reasoning strategies, we enhance a baseline model, Show-o, to achieve superior results, with a significant +24% improvement on the GenEval benchmark, surpassing Stable Diffusion 3 by +15%. We hope our study provides unique insights and paves a new path for integrating CoT reasoning with autoregressive image generation. Code and models are released at https://github.com/ZiyuGuo99/Image-Generation-CoT.

Keywords: Autoregressive Image Generation, Chain-of-Thought (CoT) Reasoning, Scaling Test-time Computation, Preference Alignment, Self Reflection

###### Mathematical Problem Solving: Autoregressive Image Generation:

[Figure 1]

[Figure 2]

Unsatisfactory Reasoning Unsatisfactory Generation Process

Textual Prompt:

###### ✓

Question: Step 1:Use the formula: Area = (1/2) *

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Base * Height.

Chain-of-Thought Reasoning

Autoregressive Generation

The fluffy white clouds floated above the jagged mountain peaks

13 13

Find the area of the isosceles triangle in the image.

✗ ✗

- Step 2:Substitute the given values: Area

= (1/2) * 10 * 13, Area = 13 * 7 = 91.

- Step 3:The final answer is 91.

…

10

[Figure 7]

[Figure 8]

Existing Reasoning Strategy: Can Similar Strategy be Applied to Image Generation?

Preference Alignment

Preference Alignment

Test-time Verifier

###### Ranking Data

###### Reflection

Test-time Verifier Ranking Data

###### Reflection

[Figure 9]

[Figure 10]

- Step 1 →
- Step 2 →

ORM PRM 𝑦

𝑦

###### ORM PRM

𝑦

PARM 𝑦

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

… Step N

Text Prompt Text Prompt

Text Prompt

Text Prompt Text Prompt

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

- Step 1 →
- Step 2 → … Step N

- Step 1 →
- Step 2 → … Step N

Image Generation Models

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

≻

≻

Reflect & Correct

Reflect & Correct

LLMs & LMMs

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

- Step 1 →
- Step 2 → … Step N

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

Maximum Likelihood

Maximum Likelihood

[Figure 52]

[Figure 53]

Enhanced Reasoning Enhanced Generation Process

✓ ✓ ✓ ✓ ✓

- Step 1: Use the formula for the area of a triangle, which is ( 1/2 * base * height).

- Step 2: Theperpendicular from the top vertex splits the base into two segments of length 5 each.

- Step 3: Use the Pythagorean theorem for one of these right-angled triangles: h^2 + 5^2 = 13^2, h^2 = 13^2 - 5^2, h^2

= 169 – 25, h^2 = 144. So, h = √144 = 12.

- Step 4: Find the isosceles triangle’s area: Area = (1/2 * base * height) = (1/2 * 10 * 12) = 60.

- Step 5: The area of the isosceles triangle is 60.

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

- Fig. 1: Can We Verify and Reinforce Image Generation with Chain-of-Thought (CoT) Reasoning Strategies? Given the success of mathematical CoT reasoning in LLMs [1, 2] and LMMs [3, 4] (Left), we provide the first investigation to comprehensively explore the potential of applying current reasoning techniques to autoregressive image generation (Right), including test-time verification, preference alignment and reflection, with two newly proposed specialized reward models, termed Potential Assessment Reward Model (PARM) and PARM++.

### 1 Introduction

similar verification and reinforcement techniques. This raises the question: Can we verify and reinforce image generation step-by-step with strategies revealed by OpenAI o1?

Large Language Models (LLMs) [5–7] and Large Multi-modal Models (LMMs) [8–10] have gained remarkable achievements across language [11, 12],

To this end, we conduct a systematic investigation into the potential of CoT reasoning for autoregressive image generation. We adopt Show-

###### 2D image [13, 14], temporal video [15, 16], and 3D point cloud [17, 18]. Building upon general understanding skills, recent efforts have been made toward enhancing LLMs and LMMs with complex Chain-of-Thought (CoT) reasoning capabilities [2, 19–21], e.g., OpenAI o1 [22], contributing to superior performance in mathematics [3, 23], science [24, 25], and coding [26, 27].

- o [29], a latest discrete generative model, as
- our baseline, and evaluate on a challenging textto-image generation benchmark: GenEval [38]. Specifically, we focus on examining two key perspectives: 1) Scaling test-time computation with Outcome/Process Reward Model (ORM/PRM) as verifiers; and 2) Reinforced preference alignment via Direct Preference Optimization (DPO). The specifics are as follows:

Despite the success in multi-modal understanding, it remains under-explored whether multi-step reasoning strategies can be effectively applied to image generation. Considering the discrepancy between two tasks, we observe that, autoregressive image generation [28–31] shares a similar output manner to the nature of LLMs and LMMs. Specifically, they both quantize the target data (language and image) into discrete tokens, and iteratively predict partial content conditioned on previously generated tokens.

• ORM vs PRM as Test-time Verifiers. As the top-1 result may not always be reliable, reward models are employed to score sampled candidates and perform outcome selection, where ORM is instance-level and PRM is process-level. In our settings, the score assesses whether each candidate image is inherently reasonable and aligns with the given textual prompt. We prompt LLaVA-OneVision (7B) [39] as a zero-shot reward model, and then curate text-to-image ranking data for reward fine-tuning. We apply a best-of-N selection approach in the comparison of zero-shot and fine-tuned reward models.

As illustrated in Figure 1, LMMs leverage CoT to break down complex mathematical problems into manageable steps, which enables scaling testtime computation with reward models [1, 32–34] and reinforcement learning for preference alignment [4, 35–37]. Likewise, autoregressive image generation through step-by-step decoding can produce intermediate images, potentially allowing for

Text Prompt: “A red apple was on the oval plate.”

Text Prompt: “The moon rose above the hill and the calm sea.”

Text Prompt: “A boat is sailing on a lake.”

[Figure 61]

[Figure 62]

[Figure 63]

Unsatisfactory Generation Process

Unsatisfactory Generation Process

Unsatisfactory Generation Process

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

Generation Process with Reasoning

Generation Process with Reasoning

Generation Process with Reasoning

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

[Figure 90]

- Fig. 2: Autoregressive Image Generation without (Top) and with (Bottom) Our Reasoning Strategies. We adopt Show-o [29] as the baseline model that produces unsatisfactory text-to-image generation. After using our investigated reasoning strategies (integrating PARM with iterative DPO for both reward model guidance and test-time verification), the generation process is effectively enhanced.

Observation: ORM demonstrates significant boost, while PRM offers minimal benefit.

global-level assessments are unable to capture the nuanced step-wise information, leading to inaccurate reward judgments. For PRM, the early-stage images tend to appear blurry, while later-stage images across different paths often converge to visually similar outputs, severely limiting its discrimination ability.

- • Test-time Verifiers vs Preference Alignment. Exploring the trade-off between inference-time and post-training offers valuable insights into the model’s attainable performance. Preference alignment are adopted to elicit the implicit reasoning capabilities from their widely learned knowledge. In this study, we construct ranking preference data and apply DPO alignment with iterative training [40] to optimize the generation decoding process, comparing its effectiveness to test-time verification.

Observation: DPO alignment with iterative training attains stronger results to the finetuned ORM verifier.

- • Preference Alignment plus Test-time Verifiers. After investigating the individual impact, we integrate the two techniques to highlight their complementary potential in autoregressive image generation. We consider three approaches: 1) DPO with reward model guidance, i.e., integrating DPO’s policy with reward models’ objectives for alignment; 2) Verification after DPO alignment, i.e., applying reward models for best-of-N selection on DPO-aligned models; and 3) Verification after DPO with reward model guidance, which is a combination of 1) and 2).

To alleviate these issues, we propose a specialized reward model for autoregressive image generation, termed Potential Assessment Reward Model (PARM). PARM adaptively verifies the generation process step by step with three delicately designed tasks: 1) Judge which step is clear and convincing enough to be evaluated, given that most early-stage images are typically blurry; 2) Assess whether the current step has the potential to yield a high-quality final image, since later-stage images generally do not change too much; and 3) Score the remaining final paths for selecting the best one, similar to an ORM. In this way, PARM can adaptively conduct assessment at appropriate steps (overcoming PRM’s scoring challenges), while effectively capturing fine-grained step-bystep cues (complementing the coarse ORM). Our experiments showcase that PARM significantly outperforms both ORM and PRM, which finally improves the baseline model by +24% on GenEval, as visualized in Figure 2, surpassing the advanced Stable Diffusion 3 [41] by +15%. To further harness CoT-style reasoning, we introduce a reflection mechanism in PARM++, enabling the model to self-evaluate and refine its outputs in a multi-step, introspective manner.

Observation: All integration methods lead to greater improvements, indicating complementary characteristics.

In addition, we further introduce an enhanced variant, PARM++, equipping PARM with a reflection mechanism that enables autoregressive models to refine its previously generated images. Building upon the three tasks of PARM, PARM++ incorporates a reflection evaluation step to assess whether the generated image aligns with the input text prompt. If any misalignment is detected, PARM++ identifies the issue with

Through our experiments, we demonstrate the promsing potential of applying CoT reasoning strategies to image generation scenarios, uncovering their adaptation methods and compatibility. Furthermore, we identify significant room for improvement in reward models tailored for autoregressive image generation. For ORM, the

detailed descriptions, e.g., discrepancies in visual concepts, and prompts the generative model to iteratively self-correct the image until it passes the reflection evaluation. By refining the text-toimage generation process, PARM++ achieves a +4% improvement over PARM on the baseline model on GenEval.

Our contributions are summarized as follows:

- • We present the first comprehensive empirical study of applying CoT reasoning strategies to autoregressive image generation domains, providing unique insights into the future advancement of this field.
- • We investigate specific adaption methods of techniques, including test-time verification, preference alignment, ranking data curation and reflection, to autoregressive image generation, indicating their performance and complementarity.
- • We further introduce PARM and PARM++, two new reward models tailored for image generation scenarios, which adaptively perform stepwise potential assessment and reflection evaluation for self-correction, significantly enhancing text-to-image generation quality.

### 2 Related Work

###### Scaling Test-time Computation.

Humans often dedicate significant time and effort to solve complex problems. Inspired by this, many efforts have focused on scaling test-time computation for Large Language Models (LLMs) to tackle reasoning tasks such as mathematical problem-solving [30, 34, 42, 43], code synthesis [26, 44, 45], and workflow generation [46– 48]. One line of research adapts the input space to leverage Chain-of-Thought (CoT) capabilities, using approaches like in-context CoT examples [2] or zero-shot CoT prompts [49]. Another branch modifies or integrates reasoning paths within the output space, utilizing strategies such as selfconsistency [50], CoT decoding [50], and verifierbased selection [1, 32, 51]. Among these, testtime verifiers have demonstrated generality and robustness in enhancing reasoning performance. For example, early work [51] trains an Outcome Reward Model (ORM) to evaluate final outputs and select the best-of-N candidates for optimal results. Later, Lightman et al. [1, 33] adopt the Process Reward Model (PRM) to evaluate intermediate reasoning steps, achieving greater effectiveness. Snell et al. [32] further highlights that scaling test-time computation is often more impactful than scaling model parameters during training. Recently, OpenAI o1 [52] has demonstrated exceptional reasoning capabilities across

a variety of complex and challenging scenarios, underscoring the potential of this approach. Building on these advancements in understanding tasks, we conduct a comprehensive investigation into whether verifier-based strategies can also enhance image generation tasks, and propose a new Potential Assessment Reward Model (PARM), specifically designed for this domain.

###### Reinforced Preference Alignment.

After robust pre-training and fine-tuning, LLMs often acquire substantial knowledge. However, a post-training alignment stage is typically required to align their output preferences to meet specific targets, such as human feedback [53–55] or Chain-of-Thought (CoT) reasoning [34, 35, 37]. Traditional approaches [56–59] often leverage reinforcement learning (RL) to address this challenge. These methods usually involve two steps: first, optimizing a neural-network-based reward function within a preference model (e.g., the Bradley-Terry model [60]), and then finetuning the target LLM to maximize this reward using techniques like proximal policy optimization (PPO) [61]. However, RL-based methods often encounter issues related to complexity and instability. To overcome these challenges, Rafailov et al. introduced Direct Preference Optimization (DPO) [62], which parameterizes the reward model to enable the derivation of the optimal policy through a closed-form solution. This approach has been effectively applied to enhance CoT capabilities in mathematical reasoning [34, 63] and code generation [64–66]. Further advancements have extended DPO with step-wise preference data [35, 37] for more granular supervision and multi-modality learning [4, 67] to support visual reasoning. In this study, we apply DPO-based preference alignment to autoregressive image generation, demonstrating its effectiveness in improving image quality during step-by-step decoding.

###### Autoregressive Image Generation.

The transformer architectures with autoregressive output schemes [6, 8, 39, 68–70] have demonstrated a remarkably successful modeling approach in language and multi-modality. Motivated by such progress, a series of work, e.g., DALL-E [71], LlamaGen [72], and Chameleon [73], utilizes such autoregressive modeling with casual attention to learn the dependency within image pixels for image generation tasks, rather than popular diffusion models [41, 74–78]. However, such raster-order autoregression suffers from severe time consumption and performance constraints when synthesizing high-resolution and highfidelity images, attributed to the growing number of discrete tokens compressed by VQ-VQE [79–

[Figure 91]

[Figure 92]

to transform images into discrete tokens, allowing for the cross-entropy loss of Direct Preference Optimization (DPO) [62] in post-training. Additionally, it iteratively predicts one or more tokens at each step, conditioned on prior outputs, thereby creating reasoning paths suitable for step-wise verification with reward models.

TheOverallScoreonGenEval

Baseline

###### Experimental Settings.

We select Show-o [29] as our baseline model for investigation, a latest autoregressive image generation model with advanced capabilities. To comprehensively evaluate different strategies, we assess the text-to-image generation performance on a rigorous benchmark: GenEval [38]. This scenario challenges the model to produce images with not only high visual quality and imagetext alignment, but also accurate object attribute and co-occurrence. In the subsequent sections, we explore three strategies to improve the step-bystep decoding of image generation: test-time verification (Sec. 3.2), preference alignment (Sec. 3.3), and their combination (Sec. 3.4).

[Figure 93]

N in Best-of-N Selection

- Fig. 3: Comparison of Reward Models as Test-time Verifiers. We adopt Show-o [29] as the ‘Baseline’ and evaluate Best-of-N selection on the GenEval [38] benchmark.

82]. To address the challenges, MaskGiT [28] proposes to learn a bidirectional autoregressive transformer with a parallel iterative decoding strategy, benefiting both the generation performance and efficiency. Recently, this approach has been effectively extended, primarily focusing on two aspects: the unification of visual understanding and generation (Show-o [29]) and its integration with diffusion techniques (MAR [83]). Considering that such generation paradigm is quite similar to that of LLMs, representing data with discrete tokens and predicting iteratively conditioned on previous tokens, we explore the potential of applying CoT reasoning techniques within LLMs to autogressive image generation. Through our thorough investigation, we demonstrate its promising effectiveness for enhanced image generation capabilities.

#### 3.2 ORM vs PRM as Test-time Verifiers

Scaling test-time computation [1, 32–34] to enhance reasoning capabilities has emerged as an effective alternative to scaling training costs. Current approaches often employ reward models as test-time verifiers within CoT reasoning paths, typically using two main categories: Outcome Reward Model (ORM) and Process Reward Model (PRM). Drawing inspiration from these methods, we respectively implement and evaluate them within the context of autoregressive image generation, as illustrated in Figure 4.

### 3 Our Investigation

Chain-of-Thought (CoT) reasoning has been widely exploited to solve complex problems for language and multi-modal understanding. In this study, we conduct a systematic investigation aiming to find out, whether we can verify and reinforce image generation step-by-step.

#### 3.1 Overview

###### Task Formulation.

To enable the applicability of current reasoning techniques, we focus on the autoregressive image generation task, demonstrated by models such as MaskGiT [28] and LlamaGen [30]. This task employs a data representation and output paradigm akin to those used in LLMs and LMMs, while achieving comparable performance to continuous diffusion models [75, 76, 84]. Specifically, it leverages quantized autoencoders [82]

###### ORM.

Based on multiple complete reasoning outputs, ORM assigns each candidate a reword score and select the most confident one using a best-of-N strategy. In our study, we adopt ORM solely to evaluate the generated image at the final step, rather than the entire CoT process in mathematical reasoning tasks. Specifically, we begin with a zero-shot ORM, followed by curating a text-toimage ranking dataset to fine-tune the ORM for enhancement, as outlined below:

• Zero-shot ORM: We leverage a pre-trained LLaVA-OneVision (7B) [39], an LMM with superior generalization, as our zero-shot ORM. We input the text prompt along with the generated image into LLaVA-OneVision, and devise

Outcome Reward Model (ORM)

Process Reward Model (PRM)

Potential Assessment Reward Model (PARM)

Text Prompt Text Prompt

Text Prompt

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

Clarity Judgment

Early-stage images are so blurry. I am not confident ….my choice.

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

These pictures are too blurry to tell. I won‘t judge them.

[Figure 115]

…

…

…

…

…

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

…

…

…

…

…

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Two of them are clear enough now. I will assess their potential then.

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

…

…

…

…

…

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

I know which one is the best!

Potential Assessment

Drop lowpotential path. Keep highpotential path.

…

…

…

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

…

[Figure 151]

[Figure 152]

These three are also clear. I will assess their potential.

Later-stage images are too similar. I cannot discriminate them well.

[Figure 153]

Selected Bad Image

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Selected Good Image

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

Final Selected Image

Blurry Image Best-of-N’ Selection

[Figure 167]

[Figure 168]

Verifier’s Monologue

Clear Image Yeah! I made the best pick!

- Fig. 4: Investigation of Reward Models in Autoregressive Image Generation. For test-time verification, we implement Outcome Reward Model (ORM) and Process Reward Model (PRM), and introduce a new Potential Assessment Reward Model (PARM) customized for image generation scenarios, which progressively performs three tasks (highlighted in blue) to enhance the reasoning of generation.

these objects, we apply the six object-centered prompt templates from GenEval, constructing a diverse set of 13K text prompts. We perform a strict filtering to ensure that these prompts do not overlap with the GenEval test samples. Then, using our baseline model, Show-o, we synthesize around 50 images per prompt at a high temperature, and label ‘yes’ or ‘no’ in the response to denote the positive or negative instance, as showcased below:

a prompt template to activate its visual understanding capabilities for text-to-image evaluation, which we observe performs well in most cases, as below:

Prompt: “ {image} This image is generated by a prompt: {prompt}. Does this image accurately represent the prompt? Please answer yes or no without explanation.”

Instruction: “ {image} This image is generated by a prompt: {prompt}. Does this image accurately represent the prompt? Please answer yes or no without explanation.”

The ‘{image}’ and ‘{prompt}’ denote and generated image from Show-o [29] and the input textual prompt. This model assesses the quality of candidate images, providing binary responses, ‘yes’ (good quality) or ‘no’ (low quality). The candidate image with the highest probability of ‘yes’ is then selected as the final output.

Response: “Yes” or “No”

• ORM Ranking Data Curation: To enhance the accuracy of outcome rewards, we curate a dataset of 288K text-to-image ranking examples for fine-tuning ORM. First, we prompt GPT-4 [70] to generate a list of 200 countable daily object names with specific colors. Using

• Fine-tuned ORM: Using the curated ranking dataset, we fine-tune LLaVA-OneVision to enhance its capability for assessing image quality and cross-modal alignment. The training data format is consistent with the prompt template used in the zero-shot ORM, incorporated

with our constructed 288K text prompts and associated images. The model is fine-tuned for one epoch, using a batch size of 8 and a learning rate of 1e−5. This fine-tuning process enables the ORM to capture more intricate aspects of object composition and nuanced visual-text relationships, resulting in more reliable scoring.

###### PRM.

Different from ORM that evaluates only the final output, we utilize PRM to provide a reward score to each candidate with different steps throughout the generation process. Similar to our previous investigation, we start from a zero-shot PRM, LLaVA-OneVision, and then curate 10K step-wise text-to-image ranking data to obtain a fine-tuned PRM.

- • Zero-shot PRM: We also utilize the pretrained LLaVA-OneVision (7B) as our zero-shot PRM, applying a similar prompt template used in ORM as:

Prompt: “ {image} This is an intermediate image in the generation process by a prompt: {prompt}. Does this intermediate image accurately represent the prompt? Please answer yes or no without explanation.”

At each intermediate step in the generation process, the zero-shot PRM assesses each candidate image with a binary response, ‘yes’ or ‘no’. We then adopt a step-level best-of-N strategy, selecting the most confident candidate and following this path for subsequent decoding. By iteratively employing the PRM at each step, the generation process is guided step by step towards the final.

- • PRM Ranking Data Curation: We observe that the images generated at intermediate steps tend to appear very blurry, as only partial visual tokens in specific regions are decoded while others remain unresolved. Since LLaVA-OneVision is pre-trained only on natural images (similar to those generated at the final step), the zero-shot PRM has limited capability for precise step-wise evaluation. To address this issue, we curate a 300K step-wise text-to-image ranking dataset to fine-tune an improved PRM. We adopt the same prompt in the instruction as the zero-shot PRM, formulated as:

Instruction: “ {image} This is an intermediate image in the generation process by a prompt: {prompt}. Does this intermediate image accurately represent the prompt? Please answer yes or no without

explanation.” Response: “Yes” or “No”

First, we utilize the 13K unique text prompts from our ORM ranking dataset, generating 18 intermediate-step images per prompt using Show-o. Inspired by Math-Shepherd [34], we employ an automated annotation approach to obtain accurate step-wise labels, eliminating the need for costly human labor or GPT assistance. For instance, to label the image at step i (1 ≤ i ≤ 18), we condition Show-o on that image and then produce four different paths for the remaining 18 - i steps. By evaluating the final images from each of these paths, if any path receives a ‘yes’ score, it indicates that step i has a high potential to lead to a correct final output, and thus it is labeled as ‘yes’; otherwise, it is labeled as ‘no’. This automated approach allows us to efficiently obtain step-wise annotations for assessing the generation.

• Fine-tuned PRM: With the step-wise ranking data, the LLaVA-OneVision is fine-tuned to boost the visual comprehension of intermediatestep images. The data format and training configurations are the same as those used for fine-tuning the ORM. After training, the PRM becomes more capable of interpreting blurry images within the decoding process for more accurate step-by-step selection.

###### Experiments and Insights.

As showcased in the middle of Table 1, we compare the test-time verification results between ORM and PRM with a best-of-20 strategy. The observations are summarized below:

- • Test-time verification can significantly boost generation performance. Compared to the baseline scores of 53% on GenEval, the fine-tuned ORM as a test-time verifier achieves the highest gains of +10%. This finding suggests that current autoregressive image generation models, similar to LLMs and LMMs, face challenges with inconsistent and unstable decoding paths. Consequently, a test-time verification strategy is essential to identify and follow the most reliable reasoning path.
- • ORM exhibits stronger enhancement capabilities than PRM. In contrast to ORM providing a clear benefit, PRM yields only marginal improvements, e.g., +2% on GenEval after fine-tuning. This discrepancy arises from the unique characteristics of the autoregressive image generation task in two key ways:

1) Images at early steps are too blurry for

The early-stage images are too blurry The later-stage images are too similar

Text Prompt:

“A refrigerator.”

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

Text Prompt:

“A microwave oven.”

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

Text Prompt:

“A book with a beautiful cover.”

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

Text Prompt:

“A cup.”

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

Text Prompt:

“A carrot in front of the TV.”

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

- Fig. 5: Visualization of Early-stage and Later-stage Images. We visualize the generated images in the intermediate steps of Show-o [29], where the early-stage images are too blurry to interpret, while the later-stage images are too similar to discriminate, posing great challenges for PRMs to evaluate.

PRM to effectively interpret their visual features and image-text alignment. 2) Images at later steps tend to exhibit minimal differences, making it challenging for PRM to discriminate. Whereas, ORM evaluates images at the final step, which provides sufficient visual and semantic information for accurate judgment.

refining reward accuracy and benefiting scalability across a broader range of candidates.

#### 3.3 Test-time Verifiers vs Preference Alignment

Post-training has been widely utilized in existing LLMs and LMMs to align model outputs with human preferences. Common techniques include reinforcement learning with reward models, e.g., Proximal Policy Optimization (PPO) [61], and its streamlined counterpart with classification objectives, e.g., Direct Preference Optimization (DPO) [62]. Given that most autoregressive image generation models inherently function within a classification framework, we leverage the simplicity of DPO alignment to enhance the quality of generated images.

• Fine-tuning by ranking data enhances verification results and demonstrates improved scaling performance. As illustrated in Figure 3, both fine-tuned ORM and PRM outperform their zero-shot counterparts, achieving higher scores with larger N values in the best-of-N selection. Additionally, as N increases, fine-tuned reward models show greater improvements, indicated by steeper curves, reflecting a better scaling response to test-time computation. This highlights the effectiveness of our curated ranking dataset in

Table 1: Test-time Verifiers (ORM vs PRM) vs Preference Alignment. We evaluate text-toimage generation on the GenEval [38] benchmark and adopt Show-o [29] as the autoregressive baseline model. ‘ORM/PRM’ and ‘DPO’ denote Outcome/Process Reward Model and Direct Preference Optimization [62], respectively. We adopt the best-of-N selection for test-time verifiers, setting N = 20, and highlight the better-performed variant of each reasoning strategy in green.

Attribute binding

Reasoning Strategy

Single object

Two object

Counting Colors Position

Method Setting

###### Overall

Baseline - - 0.95 0.52 0.49 0.82 0.11 0.28 0.53

Zero-Shot 0.99 0.63 0.63 0.84 0.19 0.39 0.61

ORM

Test-time Verifier

Fine-tuned 0.99 0.72 0.65 0.84 0.25 0.33 0.63 PRM

Zero-Shot 0.98 0.51 0.54 0.82 0.11 0.23 0.53

Fine-tuned 0.98 0.55 0.54 0.83 0.13 0.29 0.55 Preference

- 0.96 0.70 0.50 0.82 0.30 0.43 0.62 Alignment Iterative 0.98 0.72 0.53 0.84 0.40 0.46 0.65

DPO

###### DPO Ranking Data Curation.

To bypass reinforcement learning, DPO leverages an implicit rewarding mechanism by training on a ranking dataset of paired preferred and dispreferred responses, corresponding to well-generated and poor-quality images in our case. Fortunately, we have already constructed a substantial amount of ranking data for training ORM, annotated with ‘yes’ and ‘no’ labels to indicate the generation quality. Building on this, we utilize the 13K unique text prompts from the ORM training dataset and, for each prompt, randomly pair two generated images, one labeled ‘yes’ and the other ‘no’, yielding 10K paired data for DPO alignment.

DPO for Autoregressive Image Generation. Since autoregressive image generation models are also trained using a cross-entropy loss, we can directly apply the maximum likelihood objective in DPO to our setting. In detail, the parameterized policy is initialized from Show-o and optimized during training, while the reference policy is also initialized from Show-o but kept frozen. The objective encourages the model to assign a higher likelihood to preferred images over dispreferred images, aligning with the curated preference structure. The training is conducted over one epoch with a batch size of 10 and a learning rate of 1e−5.

###### DPO with Iterative Training.

Following the initial stage of DPO alignment, the model has learned to generate images that better align with the preferred responses. Inspired by iterative DPO [85], we further refine this alignment by applying the newly aligned model to generate updated ranking data based on the text prompts in D. We annotate these new images with ‘yes’ or ‘no’ labels using the same method in Sec. 3.2. For each prompt, we collect paired images labeled yyes and yno, and exclude samples

where all images receive the same label, resulting in a refined DPO ranking dataset of 7K samples. By conducting another round of DPO, the model can be further improved by learning from more informative preference relations. We iterate the DPO training process once with the same training configurations.

###### Experiments and Insights.

In the bottom of Table 1, we present the evaluation results of DPO alignment and compare it with the performance with test-time verification. The observations are summarized below:

- • DPO alignment can effectively reinforce the generation performance. On GenEval, initial DPO alignment improves the baseline model’s performance by +9%. With iterative training, these gains are further extended as

+11%, highlighting the effectiveness of a refined preference dataset in strengthening model alignment with desired outputs. This demonstrates that DPO alignment can serve as a powerful method for enhancing autoregressive generation models, especially in scenarios where explicit preference data is available to guide training.

- • Initial DPO matches test-time verification, while iterative DPO surpasses. After the initial alignment, the model achieves performance comparable to that of the fine-tuned ORM, the top-performing variant for verification. However, with iterative alignment on refined ranking data, the model outperforms all test-time verifiers, i.e., +2% over the finetuned ORM, indicating the potential of iterative DPO to progressively reinforce image generation capabilities through updated ranking data.

Table 2: Test-time Verifiers plus Preference Alignment. We evaluate text-to-image generation on the GenEval [38] benchmark and adopt Show-o [29] as the autoregressive baseline model. ‘Ft. ORM’ and ‘It. DPO’ denote the fine-tuned ORM and iterative DPO [62]. We explore three combination approaches (‘1st, 2nd, and 3rd Integration’) of reward models and preference alignment, comparing ‘individual’ results. We adopt the best-of-N selection for test-time verifiers, setting N = 20, and highlight the bestperforming integration in green.

Attribute binding

Reasoning Strategy

Test-time Verifier

Preference Alignment

Reward Guidance

Single object

Two object

Counting Colors Position

Overall Baseline - - - 0.95 0.52 0.49 0.82 0.11 0.28 0.53 Individual

Ft. ORM - - 0.99 0.72 0.65 0.84 0.25 0.33 0.63 - It. DPO - 0.98 0.72 0.53 0.84 0.40 0.46 0.65

1st Integration - It. DPO Ft. ORM 0.98 0.78 0.44 0.81 0.50 0.48 0.67 2nd Integration Ft. ORM It. DPO - 0.98 0.80 0.62 0.83 0.59 0.54 0.72 3rd Integration Ft. ORM It. DPO Ft. ORM 0.98 0.84 0.64 0.85 0.66 0.52 0.75

#### 3.4 DPO Alignment plus Test-time Verifiers

The investigations above have demonstrated the individual effectiveness of test-time verification and DPO alignment. Next, we explore three approaches to integrate these two techniques to assess their complementary potential in image generation, leveraging both the adaptability of verifiers and the reinforcement of DPO.

###### DPO with Reward Model Guidance.

As discussed in previous works [64], DPO can struggle with out-of-distribution responses due to distribution shifts from the ranking dataset. A potential solution [68, 86] is to incorporate prompt-only datasets during post-training and leverage a reward model to provide online preference guidance. Following this approach, we adopt our fine-tuned ORM as the explicit reward model to offer more generalized preference feedback, and add the online objectives with the original DPO loss We maintain the same training data and configurations as the initial DPO alignment stage.

###### Verification after DPO Alignment.

We observe that verification and DPO techniques may naturally complement each other in two key ways: 1) They operate independently at different stages of implementation, i.e., post-training and test-time; and 2) DPO refines the internal knowledge distribution within the model to enhance reasoning, while verification focuses on selecting the optimal reasoning path within this refined distribution. Therefore, we apply the fine-tuned ORM for best-of-N selection directly on the model after DPO alignment.

Verification after DPO with Reward Model Guidance.

In this approach, we combine the strengths of both DPO with reward model guidance and testtime verification. Our goal is to achieve optimal alignment, enhancing the model’s generalization capabilities during training, while also ensuring reliable image generation paths at inference time.

###### Experiments and Insights.

Table 2 reports the text-to-image generation scores with different integration methods. The observations are summarized below:

- • Verification and alignment perform expected complementary characteristics. Across all three approaches, verification and alignment complement each other effectively. For instance, the third integration method outperforms DPO alignment alone by +10%, and surpasses the verification alone by +12%. These results highlight the potential of combining verification and alignment techniques in future autoregressive image generation tasks, enabling the production of high-quality outputs that are both preference-aligned and test-time reliable.
- • Applying verifiers to both training and test time yields maximum enhancement. We observe the third combination approach delivers the most significant gains, outperforming the first approach by +8%, and the second by +3%. This suggests that, even with a model already aligned to preferences, the fine-tuned ORM can play complementary roles in the testtime decoding. These dual functions reinforce each other, highlighting the versatility of reward models in autoregressive image generation.

### 4 Potential Assessment Reward Model

From our comprehensive investigation, reward models prove valuable by enabling both decoding path selection and preference reward guidance. However, we still observe considerable room for enhancing reward models.

###### Limitation of ORM and PRM.

1) ORM showcases strong performance by selecting optimal final outputs, yet it lacks the capacity to provide fine-grained, step-wise evaluation at each generation step. 2) While PRM has demonstrated effectiveness in understanding tasks such as mathematics, it is less suitable for autoregressive image generation. As analyzed in Sec. 3.2, PRM struggles with early-stage images that are too blurry for reliable evaluation, given that only a few regions are decoded. In later stages, images derived from similar previous steps lack sufficient distinction, challenging for PRM to discriminate.

###### PARM.

Motivated by these observations, we propose the Potential Assessment Reward Model (PARM), a specialized reward model tailored for autoregressive image generation, as illustrated in Figure 4. PARM combines the best of both worlds: 1) it operates adaptively in a step-wise manner, using a potential assessment mechanism to overcome PRM’s evaluation challenges; and 2) it performs a best-of-N′ selection across N′ (N′ ≤ N) high-potential reasoning paths, thus inheriting ORM’s advantage. Specifically, the methodology of PARM contains three progressive tasks:

- 1. Clarity Judgment. In the best-of-N setting, we first sample N different reasoning paths for image generation. Then, at each intermediate step, PARM evaluates whether the partially generated image contains enough visual clarity to be meaningfully assessed, assigning a binary label If labeled ‘no’, the model skips to the next step. If labeled ‘yes’, the model proceeds to the next task for potential assessment. This pre-judgment prevents scoring on early, blurry images that lack informative content (as seen in PRM), ensuring only sufficiently clear steps are considered for rewarding.
- 2. Potential Assessment. For each clear step that passes the clarity judgment, PARM assesses the potential of the current step to determine whether it can lead to a highquality final image, again using a binary label. If labeled ‘no’, the generation path is truncated immediately. If labeled ‘yes’, the path is preserved to produce the final image. This approach is based on the observation that,

once an image at a given step is clear enough to evaluate, its overall layout and structure are unlikely to change significantly in subsequent steps, making it a reliable candidate for potential assessment. This task helps identify promising intermediate steps, effectively pruning low-potential candidates during inference.

3. Best-of-N′ Selection. After completing the above two tasks, suppose there are N′ highpotential paths remaining to produce the final images (N′ ≤ N). PARM then performs a bestof-N′ selection to identify the most promising image candidates as the output. If N′ = 0, the model defaults to selecting the reasoning path with the lowest probability of a ‘no’ label as the output. This final task leverages ORM’s global selection capabilities to ensure a high-quality generated image.

###### PARM Ranking Data Curation.

In Figure 5, we illustrate why PRM is less suitable for autoregressive image generation. As shown, the early-stage images are too blurry for reliable evaluation, given that only a few regions are decoded, while the later-stage images derived from similar previous steps lack sufficient distinction, challenging for discrimination. To integrate the advantage of both ORM and PRM, we propose Potential Assessment Reward Model (PARM) and curate a new ranking dataset with 400K instances by re-annotating the 13K text prompts from ORM ranking data. The dataset is structured into three subsets corresponding to the three tasks:

• Clarity Judgment Data (120K): Through comprehensive analysis, we observe that the baseline model (Show-o) typically produces its first clear image between steps 8 and 12 within the 18-step generation, qualifying it for potential assessment. Based on this, we simplify the annotation by labeling steps after 11 as ‘yes’ and those before 10 as ‘no’. Although this approach is static, the trained PARM acquires generalization skills to adaptively identify the first ‘yes’ label within steps 8∼12 during inference. The data format is shown below:

Instruction: “ {image} This image is a certain step in the text-to-image generation process with a prompt: {prompt}. It is not the final generated one, and will keep iterating better. Do you think this image can be used to judge whether it has the potential to iterate to the image satisfied the prompt? (The image, which needn’t to be confused but can be clear and basically judged the object, can be used to judge the potential) Answer yes or no

- Table 3: Performance Comparison on the GenEval [38] Benchmark. Compared to existing diffusion and autoregressive models, we investigate the potential of Chain-of-Thought (CoT) reasoning strategies in text-to-image generation. ‘Zs.’, ‘Ft.’, and ‘It. DPO’ denote the zero-shot, fine-tuned verifiers, and iterative DPO [62], repsectively. PARM refers to our proposed Potential Assessment Reward Model specialized for autoregressive image generation. We adopt the best-of-20 selection for test-time verifiers, highlighting our best result in green and the previous best model in red.

Attribute binding

Test-time Verifier

Preference Alignment

Reward Guidance

Single object

Two object

Counting Colors Position

Model

###### Overall

PixArt-α [74] - - - 0.98 0.50 0.44 0.80 0.08 0.07 0.48 SD v2.1 [87] - - - 0.98 0.51 0.44 0.85 0.07 0.17 0.50 DALL-E 2 [75] - - - 0.94 0.66 0.49 0.77 0.10 0.19 0.52 SDXL [76] - - - 0.98 0.74 0.39 0.85 0.15 0.23 0.55 SD 3 (d=24) [41] - - - 0.98 0.74 0.63 0.67 0.34 0.36 0.62

LlamaGen [30] - - - 0.71 0.34 0.21 0.58 0.07 0.04 0.32 Chameleon [73] - - - - - - - - - 0.39 LWM [88] - - - 0.93 0.41 0.46 0.79 0.09 0.15 0.47 SEED-X [89] - - - 0.97 0.58 0.26 0.80 0.19 0.14 0.49

- - - 0.95 0.52 0.49 0.82 0.11 0.28 0.53

Zs. ORM - - 0.99 0.63 0.63 0.84 0.19 0.39 0.61 Ft. ORM - - 0.99 0.72 0.65 0.84 0.25 0.33 0.63 Zs. PRM - - 0.98 0.51 0.54 0.82 0.11 0.23 0.53 Ft. PRM - - 0.98 0.55 0.54 0.83 0.13 0.29 0.55

PARM - - 0.99 0.77 0.68 0.86 0.29 0.45 0.67

- - DPO - 0.96 0.70 0.50 0.82 0.30 0.43 0.62

- - It. DPO - 0.98 0.72 0.53 0.84 0.40 0.46 0.65

Show-o [29]

Zs. ORM It. DPO - 0.99 0.79 0.63 0.85 0.44 0.50 0.70 Ft. ORM It. DPO - 0.98 0.80 0.62 0.83 0.59 0.54 0.72

PARM It. DPO - 0.98 0.83 0.64 0.84 0.59 0.62 0.74

- - It. DPO Ft. ORM 0.98 0.80 0.62 0.83 0.59 0.54 0.72

- - It. DPO PARM 0.97 0.75 0.60 0.83 0.54 0.53 0.69

Ft. ORM It. DPO Ft. ORM 0.98 0.84 0.64 0.85 0.66 0.52 0.75 PARM It. DPO PARM 0.99 0.86 0.67 0.84 0.66 0.64 0.77

without explanation.” Response: “Yes” or “No”

Instruction: “ {image} This image is generated by a prompt: {prompt}. Does this image accurately represent the prompt? Please answer yes or no without explanation.”

- • Potential Assessment Data (80K): We assign intermediate images from steps after 11 with a ‘yes’ or ‘no’ label, which is based on the final output label of that path in the ORM data annotation. In practice, if the previous clarity judgment task yields ‘yes’, the data of this task is organized as a follow-up question-answering within a multi-turn conversation. The data sample of this task is formulated as:

Instruction: “ {image} Do you think whether the image has the potential to iterate to the image satisfied the prompt? Please answer yes or no without explanation.”

Response: “Yes” or “No”

- • Best-of-N′ Selection Data (200K): We directly utilize the labels in the ORM ranking dataset, with the format as

Response: “Yes” or “No”

###### Experiments and Insights.

With the new reward model, we revisit our previous investigation by applying PARM to different approaches enhancing autoregressive image generation. The observations are summarized below:

• PARM demonstrates the best-performing reward model across different strategies. Table 3 and Figure 3 present the effectiveness of PARM as test-time verifiers, significantly outperforming other reward models, e.g., +6% to the fine-tuned ORM. Additionally, PARM scales effectively with increasing N, highlighting its potential for further improvement with larger test-time computation. PARM also outperforms iterative DPO, the enhanced preference alignment strategy with refined data.

- Table 4: Generalization Results of DPO, Finetuned ORM, and PARM on Sequential Autoregressive Generative Models, LlamaGen [72] and Janus-Pro [90]. ‘Zs.’ and ‘Ft.’ denote the zero-shot, fine-tuned verifiers. PARM refers to our proposed Potential Assessment Reward Model specialized for autoregressive image generation. We adopt the best-of-20 selection for test-time verifiers. Our methods consistently improve performance across both models, highlighting their model-agnostic and plug-and-play nature.

Attribute binding

Single object

Two object

Counting Colors Position

Model Method

###### Overall

- 0.71 0.34 0.21 0.58 0.07 0.04 0.32

Zs. ORM 0.75 0.37 0.24 0.61 0.08 0.06 0.35 Ft. ORM 0.80 0.41 0.29 0.65 0.10 0.08 0.39

LlamaGen [72]

DPO 0.78 0.43 0.30 0.63 0.19 0.14 0.41

- PARM 0.84 0.45 0.31 0.67 0.20 0.21 0.46

Janus-Pro [90]

- 0.99 0.89 0.59 0.90 0.79 0.66 0.80

Zs. ORM 0.99 0.91 0.77 0.90 0.79 0.67 0.84 Ft. ORM 0.99 0.93 0.82 0.92 0.91 0.76 0.89

DPO 0.99 0.89 0.65 0.92 0.82 0.72 0.83

- PARM 1.00 0.95 0.80 0.93 0.91 0.85 0.91

[Figure 209]

Baseline with DPO

TheOverallScoreonGenEval

N in Best-of-N Selection

- Fig. 6: Comparison of Reward Models as Test-time Verifiers with DPO Alignment. We adopt Show-o [29] with DPO alignment as the ‘Baseline with DPO’ and evaluate Best-of-N selection on the GenEval [38] benchmark.

[Figure 210]

Baseline with It. DPO

TheOverallScoreonGenEval

N in Best-of-N Selection

Fig. 7: Comparison of Reward Models as Test-time Verifiers with Iterative DPO Alignment. We adopt Show-o [29] with iterative DPO alignment as the ‘Baseline with It. DPO’ and evaluate Best-of-N selection on GenEval [38].

Furthermore, PARM better harnesses the complementary strengths with post-training, consistently attaining higher integration scores than the fine-tuned ORM. These results underscore PARM’s capability as a versatile, robust reward model for autoregressive image generation.

• With PARM, our baseline model (Showo) is enhanced to achieve leading generation performance. Compared to other image generation models in Table 3, our best-performing configuration, i.e., integrating PARM with iterative DPO in both post-training and test-time, achieves a score of 77%, improving the baseline by +24% and surpassing the advanced Stable Diffusion 3 [41] by +15%. In particular, substantial gains are observed in ‘Two Obj.’, ‘Colors’, ‘Position’, and ‘Attribute binding’ emphasizing the robustness in handling challenging compositional generation. In

Figures 6 and 7, we present the performance of test-time verification integrated with DPO [62] and iterative DPO, respectively, instead of the test-time verification only in Figure 2. As shown, our propose PARM both achieves the best results as the N increases for best-of-N selection. Additionally, in Figures 10, 11, 12, 13, and 14, we showcase extensive qualitative examples. We observe that the baseline model often generates inaccurate spatial relationships, strange appearances, and object attributes. In contrast, our approach consistently mitigates such issues, ensuring that the spatial relations, object features, and overall fidelity to the text prompt are preserved.

• Generalization to Sequential Autoregressive Generative Models. While our method is validated on Show-o, which uses adaptive generation ordering with a discrete diffusion

Potential Assessment Reward Model ++ (PARM++)

approach, we further extend it to sequential autoregressive models that generate image tokens via standard next-token prediction. Specifically, we apply DPO, Finetuned ORM, and PARM to LlamaGen-3B [72] and JanusPro-7B [90], both of which follow token-bytoken generation in a fixed order. As shown in Table 4, our method consistently improves performance across both models without requiring retraining or architectural changes, demonstrating its generality and plug-and-play nature.

Text Prompt

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

Clarity Judgment

[Figure 216]

Potential Assessment

Best-of-N’ Selection

[Figure 217]

### 5 Potential Assessment Reward Model ++

Reflection Evaluation

[Figure 218]

Low Quality High Quality

Besides sequential step-by-step reasoning, humans often engage in a reflection process to verify the correctness of their previous thoughts. To explore its potential in image generation, we introduce PARM++, as shown in Figure 4, which enhances PARM with a reflection mechanism to refine the text-to-image quality.

This image can

This image cannot

[Figure 219]

[Figure 220]

accurately

accurately align with the prompt.

align with the prompt.

The reason is that …

Self-correction Process

[Figure 221]

[Figure 222]

[Figure 223]

Text Prompt Reason

###### Reflection in Image Generation.

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

The reflection strategy has also been recently applied in LLMs [91, 92], resulting in improved performance through self-correction. Unlike LLMs, which can produce and interpret freeform language to review and refine their outputs, image generation models typically rely on a text prompt (often descriptive rather than instructive) as input and output only the image modality. Consequently, the reflection capability must be primarily handled by an external reward model, tasked with identifying misalignments and providing explanations. Additionally, the image generation model itself also needs to be fine-tuned to effectively understand and respond to these reflection texts for self-correction.

Fig. 8: The Reflection Mechanism in Potential Assessment Reward Model ++ (PARM++). As an upgraded version of PARM, PARM++ incorporates a reflection evaluation task, enabling the generative model to self-correct its low-quality images.

and the image-text correspondence. We set the maximum number of reflection iterations to 3.

###### PARM++ Ranking Data Curation.

Building on the 400K dataset used for PARM, we curate an additional 120K instances for the reflection evaluation task, resulting in a total of 520K data points for training PARM++. For the negative data, we select samples labeled as ‘no’ from the ORM ranking dataset, representing lowquality images to be refined, and leverage GPT4o [69] to provide concise annotations detailing the image-text discrepancies. For the positive data, we directly utilize samples labeled as ‘yes’ from the ORM ranking dataset, representing high-quality images passing the reflection evaluation. The ratio of negative to positive instances is approximately 80% : 20% in the dataset.

###### PARM++.

After selecting the optimal output from N′ image candidates, PARM++ incorporates a reflection evaluation task that examines the alignment between the final image and the input text prompt. If the image satisfies the alignment criteria, PARM++ outputs a ‘yes’ and considers it the final result. Otherwise, it provides a detailed analysis of the discrepancies, including specific reasons for the misalignment between the image and the prompt. Then, we feed three inputs into the image generation model to self-correct its image output, including the original text prompt, the previously generated suboptimal image, and the identified misalignment reasons. This iterative refinement process continues until PARM++ produces a ‘yes’ in the reflection evaluation, thereby progressively improving both the visual fidelity

###### Self-correction Fine-tuning.

As the baseline model, e.g., Show-o, is not pretrained to refine low-quality images based on textual instructions, we specifically fine-tune Show-o to endow it with the capability of self-correcting

Before Self-correction:

Reflection Evaluation by PARM++:

After Self-correction:

Text Prompt:

The structure of the chair is unrealistic, as the shape

[Figure 229]

[Figure 230]

does not seem practical, especially with the tilted seat

###### and the way the armrests are connected. The perspective and proportions also appear unnatural,

“A photo of a bench.”

particularly in the relationship between the armrests

and the backrest.

[Figure 231]

[Figure 232]

###### The number of sport balls in the image does not

match the prompt. While the lighting effect is decent,

“A photo of three sports balls.”

###### some areas have unnatural reflections and shadowing, which affects the overall realism.

[Figure 233]

[Figure 234]

The image does not align with the prompt. Instead of

“A photo of a green bus and a purple microwave.”

generating a separate green bus and a purple

microwave, the elements seem merged, with parts of

the bus incorporating microwave-like features.

[Figure 235]

[Figure 236]

The image does not accurately match the prompt.

###### Instead of placing a zebra below a broccoli, it places

“A photo of a zebra

the zebra on top, and the composition blends the elements in an unnatural way, making the broccoli

below a broccoli.”

appear as terrain.

- Fig. 9: Qualitative Results with Reflection in PARM++. The proposed PARM++ incorporates a reflection evaluation stage to detect text-image misalignments and provides detailed explanations to guide the self-correction process in autoregressive image generation models.

- Table 5: PARM++ with Reflection as Test-time Verifiers. We evaluate text-to-image generation on the GenEval [38] benchmark and adopt Show-o [29] as the autoregressive baseline model. ‘Ft. ORM’ and ‘PARM’ denote the fine-tuned ORM and our Potential Assessment Reward Model, respectively. We adopt the best-of-N selection, setting N = 20, and highlight the best-performing variant in green.

Reasoning Strategy

Test-time Verifier

Single object

Two object

Attribute binding

Reflection

Counting Colors Position

###### Overall

Baseline - - 0.95 0.52 0.49 0.82 0.11 0.28 0.53

- - 0.92 0.50 0.47 0.79 0.10 0.27 0.51 Ft. ORM - 0.97 0.57 0.54 0.86 0.20 0.35 0.58

Self-correction Fine-tuning

PARM - 0.98 0.65 0.61 0.91 0.26 0.40 0.64

PARM++ - 0.97 0.60 0.58 0.89 0.24 0.38 0.61 PARM++ ✓ 0.99 0.71 0.69 0.95 0.36 0.49 0.70

generated images. Fortunately, Show-o supports simultaneous inputs of both text and images, enabling image refinement guided by textual feedback. To curate the training data, we extract 10K instance groups from the PARM++ ranking dataset. Each group consists of a text prompt, a low-quality (negative) image, a high-quality (positive) image, and the annotated reflection reasons. This dataset is used to fine-tune Show-o for iterative image refinement, progressively improving the quality and alignment of the generated images.

###### Experiments and Insights.

In Table 5, we showcase the results of reflection and self-correction using PARM++. Note that we do not incorporate any DPO alignment upon Show-o, in case of the training conflict with self-correction fine-tuning. The observations are summarized below:

• The reflection mechanism in PARM++ significantly enhances image quality. As an ablation, when no reflection is performed (‘-’ in the table), PARM++ performs slightly worse

than PARM due to the integration of reflectionspecific training data. However, when the reflection mechanism is enabled (‘✓’ in the table), the overall score on GenEval improves substantially by +10%, underscoring the effectiveness of the reflection strategy in PARM++. In Figure 9, we illustrate a comparison of images before and after the self-correction process. Guided by the reflection evaluation of PARM++, the selfcorrected images address issues such as unrealistic elements, incorrect numbers, wrong colors, and improper object layouts.

• The Self-correction Fine-tuning slightly impacts the baseline model’s performance. While PARM++ ultimately improves performance through its reflection mechanism, the self-correction fine-tuning introduces a minor trade-off, resulting in a -2% drop in accuracy on GenEval. This decline occurs because training for new capabilities inevitably influences the model’s original knowledge. We anticipate that this limitation can be mitigated in the future with the advancement of more general large models, which are expected to inherently possess the robustness to handle multi-modal information seamlessly.

### 6 Conclusion

In this work, we investigate the adaption and potential of CoT reasoning strategies in autoregressive image generation. Through a systematic investigation, we demonstrate that different reasoning strategies can effectively improve image generation, e.g., test-time verification, preference alignment, and their integration. Given our observation, we further introduce two tailored reward models for autoregressive image generation, termed Potential Assessment Reward Model (PARM) and PARM++, which evaluate the step-wise generation for adaptive reward scoring, and incorporate a reflection mechanism for self-corrected image generation. Our experiments underscore the promise of CoT reasoning in autoregressive image generation, advancing this field in new directions.

“A couple is relaxing in a hammock under the shade of a tree.”

Text Prompt:

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

Baseline Model:

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

With Reasoning:

“A person is looking at a waterfall and feeling awestruck.”

Text Prompt:

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

Baseline Model:

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

With Reasoning:

Text Prompt: “A leather jacket and a glass vase.”

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

Baseline Model:

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

With Reasoning:

###### Fig. 10: Qualitative Results using Our Reasoning Strategies. Show-o [29] is adopted as the baseline model, and compared to our best-performing reasoning strategy: integrating PARM with iterative DPO for both reward model guidance and test-time verification.

Text Prompt: “The fluffy towel and metallic hook hang on the wooden hook.”

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

Baseline Model:

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

With Reasoning:

Text Prompt: “The black chair is on top of the blue rug.”

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

Baseline Model:

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

With Reasoning:

Text Prompt: “The black sofa was on the left of the white coffee table.”

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

Baseline Model:

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

With Reasoning:

###### Fig. 11: Qualitative Results using Our Reasoning Strategies. Show-o [29] is adopted as the baseline model, and compared to our best-performing reasoning strategy: integrating PARM with iterative DPO for both reward model guidance and test-time verification.

Text Prompt: “The fluffy white cat snuggled up next to the warm brown blanket.”

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

Baseline Model:

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

With Reasoning:

Text Prompt: “The metallic pen and notebook jot down ideas on the wooden desk.”

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

Baseline Model:

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

With Reasoning:

Text Prompt: “The leather chair and metallic lamp provide comfort and light for the wooden desk on the rug.”

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

Baseline Model:

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

With Reasoning:

###### Fig. 12: Qualitative Results using Our Reasoning Strategies. Show-o [29] is adopted as the baseline model, and compared to our best-performing reasoning strategy: integrating PARM with iterative DPO for both reward model guidance and test-time verification.

Text Prompt:

“The white shirt was on the black hanger.”

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

Baseline Model:

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

With Reasoning:

Text Prompt:

“The bright blue bird perched on the rough brown branch.”

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

Baseline Model:

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

With Reasoning:

Text Prompt:

“The smooth metal surface reflected the bright sky and the dark clouds.”

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

Baseline Model:

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

With Reasoning:

###### Fig. 13: Qualitative Results using Our Reasoning Strategies. Show-o [29] is adopted as the baseline model, and compared to our best-performing reasoning strategy: integrating PARM with iterative DPO for both reward model guidance and test-time verification.

Text Prompt:

“The sleek bike zoomed down the smooth road and the bumpy trail.”

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

Baseline Model:

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

With Reasoning:

“The red apple was next to the yellow pear.”

Text Prompt:

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

Baseline Model:

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

With Reasoning:

Text Prompt: “The leather wallet and keychain hang on the metallic hook by the wooden door.”

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

Baseline Model:

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

With Reasoning:

###### Fig. 14: Qualitative Results using Our Reasoning Strategies. Show-o [29] is adopted as the baseline model, and compared to our best-performing reasoning strategy: integrating PARM with iterative DPO for both reward model guidance and test-time verification.

### References

- [1] Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., Cobbe, K.: Let’s verify step by step. arXiv preprint arXiv:2305.20050 (2023)
- [2] Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V., Zhou, D., et al.: Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems 35, 24824– 24837 (2022)
- [3] Zhang, R., Jiang, D., Zhang, Y., Lin, H., Guo, Z., Qiu, P., Zhou, A., Lu, P., Chang, K.-W., Gao, P., et al.: Mathverse: Does your multimodal llm truly see the diagrams in visual math problems? ECCV 2024 (2024)
- [4] Zhang, R., Wei, X., Jiang, D., Zhang, Y., Guo, Z., Tong, C., Liu, J., Zhou, A., Wei, B., Zhang, S., et al.: Mavis: Mathematical visual instruction tuning. arXiv preprint arXiv:2407.08739 (2024)
- [5] Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozi`ere, B., Goyal, N., Hambro, E., Azhar, F., et al.: Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023)
- [6] Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al.: Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288

(2023)

- [7] Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are few-shot learners. In: Advances in Neural Information Processing Systems, pp. 1877–1901 (2020)
- [8] OpenAI: GPT-4V(ision) System Card

(2023). https://openai.com/research/ gpt-4v-system-card

- [9] Zhang, R., Han, J., Zhou, A., Hu, X., Yan, S., Lu, P., Li, H., Gao, P., Qiao, Y.: LLaMA-adapter: Efficient fine-tuning of large language models with zero-initialized attention. In: The Twelfth International Conference on Learning Representations (2024). https://openreview.net/forum?id=d4UiXAHN2W

- [10] Gao, P., Han, J., Zhang, R., Lin, Z., Geng, S., Zhou, A., Zhang, W., Lu, P., He, C., Yue, X., Li, H., Qiao, Y.: Llama-adapter v2: Parameter-efficient visual instruction model. arXiv preprint arXiv:2304.15010 (2023)
- [11] Contributors, O.: OpenCompass: A Universal Evaluation Platform for Foundation Models. https://github.com/open-compass/ opencompass (2023)
- [12] OpenAI: ChatGPT. https://chat.openai.com

(2023)

- [13] Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., Wu, Y., Ji, R.: Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint

- arXiv:2306.13394 (2023)

[14] Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al.: Mmbench: Is your multi-modal model an all-around player? arXiv preprint

- arXiv:2307.06281 (2023)

- [15] Fu, C., Dai, Y., Luo, Y., Li, L., Ren, S., Zhang, R., Wang, Z., Zhou, C., Shen, Y., Zhang, M., et al.: Video-mme: The firstever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075 (2024)
- [16] Han, J., Zhang, R., Shao, W., Gao, P., Xu, P., Xiao, H., Zhang, K., Liu, C., Wen, S., Guo, Z., et al.: Imagebind-llm: Multimodality instruction tuning. arXiv preprint arXiv:2309.03905 (2023)
- [17] Guo, Z., Zhang, R., Zhu, X., Tang, Y., Ma, X., Han, J., Chen, K., Gao, P., Li, X., Li, H., et al.: Point-bind & point-llm: Aligning point cloud with multi-modality for 3d understanding, generation, and instruction following. arXiv preprint arXiv:2309.00615 (2023)
- [18] Guo, Z., Zhang, R., Zhu, X., Tong, C., Gao, P., Li, C., Heng, P.-A.: Sam2point: Segment any 3d as videos in zero-shot and promptable manners. arXiv preprint arXiv:2408.16768

(2024)

- [19] Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., Zhou, D.: Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171 (2022)
- [20] Zhang, Z., Zhang, A., Li, M., Zhao, H.,

- Karypis, G., Smola, A.: Multimodal chain-ofthought reasoning in language models. arXiv preprint arXiv:2302.00923 (2023)
- [21] Jiang, D., Zhang, R., Guo, Z., Wu, Y., Lei, J., Qiu, P., Lu, P., Chen, Z., Song, G., Gao, P., et al.: Mmsearch: Benchmarking the potential of large models as multi-modal search engines. arXiv preprint arXiv:2409.12959 (2024)
- [22] OpenAI: OpenAI o1. [Online]. https://openai.com/index/ learning-to-reason-with-llms/ (2024)
- [23] Lu, P., Bansal, H., Xia, T., Liu, J., Li, C.y., Hajishirzi, H., Cheng, H., Chang, K.-W., Galley, M., Gao, J.: Mathvista: Evaluating math reasoning in visual contexts with gpt4v, bard, and other large multimodal models. ArXiv abs/2310.02255 (2023)
- [24] Guo, Z., Zhang, R., Chen, H., Gao, J., Gao, P., Li, H., Heng, P.-A.: Sciverse. https://sciverse-cuhk.github.io (2024)
- [25] Saikh, T., Ghosal, T., Mittal, A., Ekbal, A., Bhattacharyya, P.: Scienceqa: A novel resource for question answering on scholarly articles. International Journal on Digital Libraries 23(3), 289–301 (2022)
- [26] Zhu, Q., Guo, D., Shao, Z., Yang, D., Wang, P., Xu, R., Wu, Y., Li, Y., Gao, H., Ma, S., et al.: Deepseek-coder-v2: Breaking the barrier of closed-source models in code intelligence. arXiv preprint arXiv:2406.11931 (2024)
- [27] Guo, D., Zhu, Q., Yang, D., Xie, Z., Dong, K., Zhang, W., Chen, G., Bi, X., Wu, Y., Li, Y., et al.: Deepseek-coder: When the large language model meets programming– the rise of code intelligence. arXiv preprint arXiv:2401.14196 (2024)
- [28] Chang, H., Zhang, H., Jiang, L., Liu, C., Freeman, W.T.: Maskgit: Masked generative image transformer. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11315–11325

(2022)

- [29] Xie, J., Mao, W., Bai, Z., Zhang, D.J., Wang, W., Lin, K.Q., Gu, Y., Chen, Z., Yang, Z., Shou, M.Z.: Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528 (2024)
- [30] Sun, L., Liang, H., Zhang, W.: Beats: Optimizing llm mathematical capabilities

- with backverify and adaptive disambiguate based efficient tree search. arXiv preprint arXiv:2409.17972 (2024)
- [31] Wang, X., Zhang, X., Luo, Z., Sun, Q., Cui, Y., Wang, J., Zhang, F., Wang, Y., Li, Z., Yu, Q., et al.: Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869

(2024)

- [32] Snell, C., Lee, J., Xu, K., Kumar, A.: Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters (2024). https://arxiv.org/abs/ 2408.03314
- [33] Ma, Q., Zhou, H., Liu, T., Yuan, J., Liu, P., You, Y., Yang, H.: Let’s reward step by step: Step-level reward model as the navigators for reasoning. arXiv preprint arXiv:2310.10080

(2023)

- [34] Wang, P., Li, L., Shao, Z., Xu, R., Dai, D., Li, Y., Chen, D., Wu, Y., Sui, Z.: Math-shepherd: Verify and reinforce llms step-by-step without human annotations. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 9426–9439 (2024)
- [35] Lu, Z., Zhou, A., Wang, K., Ren, H., Shi, W., Pan, J., Zhan, M.: Step-controlled dpo: Leveraging stepwise error for enhanced mathematical reasoning. arXiv preprint arXiv:2407.00782 (2024)
- [36] Havrilla, A., Du, Y., Raparthy, S.C., Nalmpantis, C., Dwivedi-Yu, J., Zhuravinskyi, M., Hambro, E., Sukhbaatar, S., Raileanu, R.: Teaching large language models to reason with reinforcement learning. arXiv preprint arXiv:2403.04642 (2024)
- [37] Lai, X., Tian, Z., Chen, Y., Yang, S., Peng, X., Jia, J.: Step-dpo: Step-wise preference optimization for long-chain reasoning of llms. arXiv preprint arXiv:2406.18629 (2024)
- [38] Ghosh, D., Hajishirzi, H., Schmidt, L.: Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems 36 (2024)
- [39] Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Li, Y., Liu, Z., Li, C.: Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326 (2024)
- [40] Pang, R.Y., Yuan, W., Cho, K., He, H.,

- Sukhbaatar, S., Weston, J.: Iterative reasoning preference optimization. arXiv preprint arXiv:2404.19733 (2024)
- [41] Esser, P., Kulal, S., Blattmann, A., Entezari, R., Mu¨ller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., Podell, D., Dockhorn, T., English, Z., Lacey, K., Goodwin, A., Marek, Y., Rombach, R.: Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (2024). https://arxiv.org/ abs/2403.03206
- [42] Zhang, D., Zhoubian, S., Hu, Z., Yue, Y., Dong, Y., Tang, J.: Rest-mcts*: Llm self-training via process reward guided tree search. arXiv preprint arXiv:2406.03816

(2024)

- [43] Xin, H., Ren, Z., Song, J., Shao, Z., Zhao, W., Wang, H., Liu, B., Zhang, L., Lu, X., Du, Q., et al.: Deepseek-prover-v1. 5: Harnessing proof assistant feedback for reinforcement learning and monte-carlo tree search. arXiv preprint arXiv:2408.08152 (2024)
- [44] DeLorenzo, M., Chowdhury, A.B., Gohil, V., Thakur, S., Karri, R., Garg, S., Rajendran, J.: Make every move count: Llm-based highquality rtl code generation using mcts. arXiv preprint arXiv:2402.03289 (2024)
- [45] Ni, A., Iyer, S., Radev, D., Stoyanov, V., Yih, W.-t., Wang, S., Lin, X.V.: Lever: Learning to verify language-to-code generation with execution. In: International Conference on Machine Learning, pp. 26106–26128 (2023). PMLR
- [46] Zhang, J., Xiang, J., Yu, Z., Teng, F., Chen, X., Chen, J., Zhuge, M., Cheng, X., Hong, S., Wang, J., et al.: Aflow: Automating agentic workflow generation. arXiv preprint arXiv:2410.10762 (2024)
- [47] Gal, R., Haviv, A., Alaluf, Y., Bermano, A.H., Cohen-Or, D., Chechik, G.: Comfygen: Prompt-adaptive workflows for text-to-image generation. arXiv preprint arXiv:2410.01731

(2024)

- [48] Xue, X., Lu, Z., Huang, D., Ouyang, W., Bai, L.: Genagent: Build collaborative ai systems with automated workflow generation– case studies on comfyui. arXiv preprint arXiv:2409.01392 (2024)
- [49] Kojima, T., Gu, S.S., Reid, M., Matsuo, Y., Iwasawa, Y.: Large language models are

- zero-shot reasoners. Advances in neural information processing systems 35, 22199–22213 (2022)
- [50] Wang, X., Zhou, D.: Chain-of-thought reasoning without prompting. arXiv preprint arXiv:2402.10200 (2024)
- [51] Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al.: Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168

(2021)

- [52] OpenAI: Learning to Reason with LLMs. https://openai.com/index/ learning-to-reason-with-llms/. Accessed: 2024-11-06 (2024)
- [53] Christiano, P.F., Leike, J., Brown, T., Martic, M., Legg, S., Amodei, D.: Deep reinforcement learning from human preferences. Advances in neural information processing systems 30

(2017)

- [54] Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., Chen, A., Goldie, A., Mirhoseini, A., McKinnon, C., et al.: Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073 (2022)
- [55] Kupcsik, A., Hsu, D., Lee, W.S.: Learning dynamic robot-to-human object handover from human feedback. Robotics Research: Volume 1, 161–176 (2018)
- [56] Pacchiano, A., Saha, A., Lee, J.: Dueling rl: reinforcement learning with trajectory preferences. arXiv preprint arXiv:2111.04850

(2021)

- [57] Jain, A., Wojcik, B., Joachims, T., Saxena, A.: Learning trajectory preferences for manipulators via iterative improvement. Advances in neural information processing systems 26 (2013)
- [58] Busa-Fekete, R., Sz¨r´enyi, B., Weng, P., Cheng, W., Hu¨llermeier, E.: Preference-based reinforcement learning: evolutionary direct policy search using a preference-based racing algorithm. Machine learning 97, 327–351

(2014)

- [59] Zhao, Y., Joshi, R., Liu, T., Khalman, M., Saleh, M., Liu, P.J.: Slic-hf: Sequence likelihood calibration with human feedback. arXiv preprint arXiv:2305.10425 (2023)
- [60] Bradley, R.A., Terry, M.E.: Rank analysis

- of incomplete block designs: I. the method of paired comparisons. Biometrika 39(3/4), 324–345 (1952)
- [61] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O.: Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 (2017)
- [62] Rafailov, R., Sharma, A., Mitchell, E., Manning, C.D., Ermon, S., Finn, C.: Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems 36 (2024)
- [63] Luo, H., Sun, Q., Xu, C., Zhao, P., Lou, J., Tao, C., Geng, X., Lin, Q., Chen, S., Zhang, D.: Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. arXiv preprint arXiv:2308.09583 (2023)
- [64] Xu, S., Fu, W., Gao, J., Ye, W., Liu, W., Mei, Z., Wang, G., Yu, C., Wu, Y.: Is dpo superior to ppo for llm alignment? a comprehensive study. arXiv preprint arXiv:2404.10719

(2024)

- [65] Miao, Y., Gao, B., Quan, S., Lin, J., Zan, D., Liu, J., Yang, J., Liu, T., Deng, Z.: Aligning codellms with direct preference optimization. arXiv preprint arXiv:2410.18585 (2024)
- [66] Gee, L., Gritta, M., Lampouras, G., Iacobacci, I.: Code-optimise: Self-generated preference data for correctness and efficiency. arXiv preprint arXiv:2406.12502 (2024)
- [67] Zhang, R., Zhang, B., Li, Y., Zhang, H., Sun, Z., Gan, Z., Yang, Y., Pang, R., Yang, Y.: Improve vision language model chain-of-thought reasoning. arXiv preprint arXiv:2410.16198 (2024)
- [68] AI@Meta: Llama 3 Model Card (2024). https://github.com/meta-llama/llama3/ blob/main/MODEL CARD.md

- [69] OpenAI: Hello GPT-4o. https://openai.com/ index/hello-gpt-4o/ (2024)
- [70] OpenAI: Gpt-4 technical report. ArXiv abs/2303.08774 (2023)
- [71] Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., Sutskever,

I.: Zero-shot text-to-image generation. In: International Conference on Machine Learning, pp. 8821–8831 (2021). Pmlr

- [72] Sun, P., Jiang, Y., Chen, S., Zhang, S., Peng,

- B., Luo, P., Yuan, Z.: Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525 (2024)
- [73] Team, C.: Chameleon: Mixed-Modal EarlyFusion Foundation Models (2024). https:// arxiv.org/abs/2405.09818
- [74] Chen, J., YU, J., GE, C., Yao, L., Xie, E., Wang, Z., Kwok, J., Luo, P., Lu, H., Li, Z.: Pixart-$\alpha$: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In: The Twelfth International Conference on Learning Representations (2024). https://openreview.net/forum?id=eAKmQPe3m1
- [75] Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M.: Hierarchical Text-Conditional Image Generation with CLIP Latents (2022). https://arxiv.org/abs/2204.06125
- [76] Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Mu¨ller, J., Penna, J., Rombach, R.: SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis

(2023). https://arxiv.org/abs/2307.01952

- [77] Zhou, C., Yu, L., Babu, A., Tirumala, K., Yasunaga, M., Shamis, L., Kahn, J., Ma, X., Zettlemoyer, L., Levy, O.: Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039 (2024)
- [78] Jiang, D., Song, G., Wu, X., Zhang, R., Shen, D., Zong, Z., Liu, Y., Li, H.: Comat: Aligning text-to-image diffusion model with imageto-text concept matching. arXiv preprint arXiv:2404.03653 (2024)
- [79] Van Den Oord, A., Vinyals, O., et al.: Neural discrete representation learning. Advances in neural information processing systems 30

(2017)

- [80] Razavi, A., Oord, A., Vinyals, O.: Generating diverse high-fidelity images with vq-vae-2 (supplementary material)
- [81] Gu, S., Chen, D., Bao, J., Wen, F., Zhang, B., Chen, D., Yuan, L., Guo, B.: Vector quantized diffusion model for text-to-image synthesis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10696–10706 (2022)
- [82] Esser, P., Rombach, R., Ommer, B.: Taming transformers for high-resolution image

- synthesis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12873–12883 (2021)
- [83] Li, T., Tian, Y., Li, H., Deng, M., He, K.: Autoregressive image generation without vector quantization. arXiv preprint arXiv:2406.11838 (2024)
- [84] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10684–10695 (2022)
- [85] Pang, R.Y., Yuan, W., Cho, K., He, H., Sukhbaatar, S., Weston, J.: Iterative reasoning preference optimization, 2024. URL https://arxiv. org/abs/2404.19733
- [86] Yang, A., Yang, B., Hui, B., Zheng, B., Yu, B., Zhou, C., Li, C., Li, C., Liu, D., Huang, F., Dong, G., Wei, H., Lin, H., Tang, J., Wang, J., Yang, J., Tu, J., Zhang, J., Ma, J., Xu, J., Zhou, J., Bai, J., He, J., Lin, J., Dang, K., Lu, K., Chen, K., Yang, K., Li, M., Xue, M., Ni, N., Zhang, P., Wang, P., Peng,

- R., Men, R., Gao, R., Lin, R., Wang, S., Bai,
- S., Tan, S., Zhu, T., Li, T., Liu, T., Ge, W., Deng, X., Zhou, X., Ren, X., Zhang, X., Wei, X., Ren, X., Fan, Y., Yao, Y., Zhang, Y., Wan, Y., Chu, Y., Liu, Y., Cui, Z., Zhang, Z., Fan, Z.: Qwen2 technical report. arXiv preprint arXiv:2407.10671 (2024)

- [87] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-Resolution Image Synthesis with Latent Diffusion Models

(2022). https://arxiv.org/abs/2112.10752

- [88] Liu, H., Yan, W., Zaharia, M., Abbeel, P.: World Model on Million-Length Video And Language With Blockwise RingAttention

(2024). https://arxiv.org/abs/2402.08268

- [89] Ge, Y., Zhao, S., Zhu, J., Ge, Y., Yi, K., Song, L., Li, C., Ding, X., Shan, Y.: SEEDX: Multimodal Models with Unified Multigranularity Comprehension and Generation

(2024). https://arxiv.org/abs/2404.14396

- [90] Chen, X., Wu, Z., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., Ruan, C.: Janus-Pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811 (2025)
- [91] Kumar, A., Zhuang, V., Agarwal, R., Su, Y., Co-Reyes, J.D., Singh, A., Baumli, K.,

Iqbal, S., Bishop, C., Roelofs, R., et al.: Training language models to self-correct via reinforcement learning. arXiv preprint arXiv:2409.12917 (2024)

[92] Huang, J., Chen, X., Mishra, S., Zheng, H.S., Yu, A.W., Song, X., Zhou, D.: Large language models cannot self-correct reasoning yet. arXiv preprint arXiv:2310.01798 (2023)

