## Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization

Weiyun Wang2,1, Zhe Chen3,1, Wenhai Wang4,1, Yue Cao3,1, Yangzhou Liu3,1, Zhangwei Gao1, Jinguo Zhu1, Xizhou Zhu5,1, Lewei Lu6, Yu Qiao1, Jifeng Dai5,1

1OpenGVLab, Shanghai AI Laboratory, 2Fudan University, 3Nanjing University, 4The Chinese University of Hong Kong, 5Tsinghua University, 6SenseTime Research Project Page

# arXiv:2411.10442v2[cs.CL]7Apr2025

|54.0<br><br>56.0<br><br>58.0<br><br>60.0<br><br>62.0<br><br>64.0<br><br>66.0<br><br>68.0<br><br>54.0 56.0 58.0 60.0 62.0 64.0 66.0 68.0<br><br>AccuracywithCoTresponses<br><br>Accuracy with direct-answer responses<br><br>Our model InternVL2 MiniCPM-V-2.6<br><br>LLaVA-OneVision<br><br>Qwen2VL<br><br>InternVL2-76B (+1.4)<br><br>InternVL2-40B (-2.0)<br><br>InternVL2-26B (-1.6)<br><br>Qwen2VL-7B (-3.4)<br><br>InternVL2-8B-MPO (+1.7)<br><br>MiniCPM-V-2.6-8B (-3.6)<br><br>LLaVA-OneVision-7B (-2.4)<br><br>InternVL2-8B (-1.5)<br><br>8B 26B 40B 76B<br><br>Diameter|
|---|

#### Abstract

Existing open-source multimodal large language models (MLLMs) generally follow a training process involving pretraining and supervised fine-tuning. However, these models suffer from distribution shifts, which limit their multimodal reasoning, particularly in the Chain-of-Thought (CoT) performance. To address this, we introduce a preference optimization (PO) process to enhance the multimodal reasoning capabilities of MLLMs. Specifically, (1) on the data side, we design an automated preference data construction pipeline to create MMPR, a high-quality, large-scale multimodal reasoning preference dataset; and (2) on the model side, we explore integrating PO with MLLMs, developing a simple yet effective method, termed Mixed Preference Optimization (MPO), which boosts multimodal CoT performance. Our approach enhances the multimodal reasoning abilities of both InternVL2-8B and InternVL2-76B. Notably, our model, InternVL2-8B-MPO, achieves an accuracy of 67.0 on MathVista, outperforming InternVL2-8B by 8.7 points and achieving performance comparable to the 10× larger InternVL2-76B. We hope this study could inspire further advancements in MLLMs. Code, data, and model are released in this page.

Figure 1. Open-source model performance on MathVista. The X- and Y-axes represent the accuracy evaluated with direct-answer responses and CoT responses, respectively. The bubble size is positively correlated with the number of model parameters. The values in parentheses indicate the performance gap between CoT and direct-answer responses. Notably, most open-source models perform worse when answering with CoT.

pre-training corpora [44, 49, 83, 93, 102, 121] and highquality SFT data [20, 24, 54, 56, 101], a series of opensource MLLMs [6, 20, 45, 47, 53, 99, 101, 109] exhibit strong performance across various domain and tasks, some even achieving results comparable to commercial models such as GPT-4o [72] and Gemini [81, 91].

#### 1. Introduction

However, open-source MLLMs still exhibit limited reasoning capabilities. As shown in Figure 1, InternVL2-8B [20] achieves a score of 58.3 on MathVista [62], a benchmark for multimodal reasoning, when using direct answers but drops to 56.8 with Chain-of-Thought (CoT) reasoning, indicating that CoT reasoning actually reduces its performance. This decline is commonly observed across opensource MLLMs [20, 45, 99, 109]. We attribute this phenomenon primarily to a distribution shift introduced by the

With the remarkable success of large language models (LLMs) [1, 5, 10, 11, 27, 92, 95, 96] in the field of natural language processing, the training paradigm comprising pre-training and supervised fine-tuning (SFT) have also swept the multimodal field, becoming the primary choice for the research and development of multimodal large language models (MLLMs). Benefiting from the large-scale

Corresponding Author: daijifeng@tsinghua.edu.cn

SFT loss. Specifically, SFT relies on teacher forcing, where the model is trained to predict the next token based on previ-

- ous ground-truth tokens. However, during inference, models must predict each token based on their own prior outputs, leading to a distribution shift between training and inference. Since the direct-answer approach requires only brief responses, while CoT reasoning involves generating a long rationale, the distribution shift problem becomes more severe during CoT. This results in models performing worse with CoT reasoning compared to direct-answer responses.

To address the limitations of CoT reasoning in MLLMs, we draw inspiration from recent NLP approaches [43, 76, 106] that use Preference Optimization (PO) techniques to align model outputs with desired reasoning patterns. Specifically, methods like Direct Preference Optimization (DPO) [79] allow models to learn from preference signals to generate responses that better align with user requirements, offering the foundation for Reinforcement Learning from Human Feedback (RLHF). While RLHF has been explored for MLLMs primarily to reduce hallucinations [18, 88, 110], its application for enhancing multimodal reasoning remains under-explored. Building on these insights, we conduct a systematic study on using PO to strengthen the multimodal reasoning capabilities of MLLMs.

Enhancing the multimodal reasoning abilities of MLLMs through PO presents several challenges: (1) Limited multimodal reasoning preference data and high annotation cost. Existing multimodal preference datasets [48, 88, 110, 111, 117] primarily address hallucination issues and focus on natural images and perception data, lacking scientific images and reasoning data. Annotating these types of data requires human annotators to carefully compare the given reasoning processes, making it both time-consuming and costly. (2) Lack of open-source methods for improving multimodal reasoning via PO. Although previous works have explored fine-tuning MLLMs using feedback from various sources, these models typically exhibit performance gains on hallucination benchmarks, with little enhancement in general reasoning abilities. Thus, leveraging PO to improve multimodal reasoning capabilities remains largely under-explored.

This work addresses these challenges from both the data and model sides. (1) On the data side, we design an automated preference data construction pipeline to create MMPR, a high-quality, large-scale multimodal reasoning preference dataset. (2) On the model side, we explore various PO methods with MLLMs, introducing a simple yet effective method, termed Mixed Preference Optimization (MPO), which boosts multimodal CoT performance with-

- out the requirement for a reward model. Specifically, we propose a continuation-based pipeline

called Dropout Next Token Prediction (DropoutNTP) for samples lacking clear ground truth and a correctness-based

pipeline for samples with clear ground truth. In DropoutNTP, the responses generated by InternVL2 series are considered as positive samples. For a given chosen response, we truncate it by half and then prompt InternVL2 series to complete the remaining portion of the truncated answer without access to the image input. This generated completion serves as the rejected answer for the paired sample. Experimental results in Section 5.2 demonstrate that this straightforward method achieves comparable performance in reducing hallucinations compared to the divideand-conquer method proposed in RLAIF-V [111]. In the correctness-based pipeline, multiple solutions to each question are sampled from InternVL2 series. Solutions matching the ground truth answer are used as chosen responses, while those that do not are used as rejected responses.

Additionally, we propose the MPO method. The key insight behind this algorithm is that an effective PO process should enable the model to learn the relative preference between pairs of responses, the absolute quality of individual responses, and the process for generating preferred responses. Compared to previous multimodal PO methods [48, 77, 88, 110, 111, 117], our approach excels in the following aspects: (1) Efficient automated data construction pipeline: Our pipeline enables high-quality preference pair generation at a controlled cost. (2) Effectiveness across diverse domains: Models fine-tuned with our data and approach show superior performance across reasoning, question-answering, and hallucination benchmarks. (3) Improvements over SoTA settings: Our results demonstrate that our method greatly enhances the reasoning abilities of both InternVL2-8B and InternVL2-76B, further highlighting the potential and scalability of our method.

In summary, our main contributions are as follows:

- (1) We propose an efficient preference data construc-

tion pipeline. Based on this pipeline, we create MMPR, a high-quality, large-scale multimodal reasoning preference dataset containing approximately 3 million samples.

- (2) We introduce MPO, an effective PO algorithm de-

signed to improve the reasoning abilities of MLLMs. The resulting models, InternVL2-8B-MPO and InternVL2-76BMPO, exhibit enhanced multimodal reasoning ability compared to their baseline models before MPO.

- (3) We conduct extensive experiments to explore prac-

tical approaches for improving multimodal reasoning via PO. Results show that PO significantly improves reasoning abilities over SFT. Notably, the proposed InternVL28B-MPO achieves an accuracy of 67.0 on MathVista [62], outperforming InternVL2-8B by 8.7 points and achieving performance comparable to the 10× larger InternVL2-76B.

#### 2. Related Work

Multimodal Large Language Models. With advancements in LLMs, significant progress has also been made in

MLLMs. To leverage the abilities of pre-trained LLMs [5, 11, 27] and Vision Foundation Models (VFMs) [19, 80], a series of works [20, 46, 47, 54, 57, 99, 102, 103] employ a connector to align their latent space, achieving promising performance at a controllable cost. Besides, another series of works [2, 27, 94, 100] extend pre-trained LLMs with additional fusion layers for vision features, reducing the number of visual tokens required by LLMs while introducing extra training costs. Recently, there have been explorations into vision encoder-free architectures [7, 51, 63, 90, 104], which consist of a single transformer model that jointly processes visual and textual information without a separate encoder. In addition to exploring model architectures, recent works [28, 49, 56, 101, 108, 115] also try to construct highquality training data to improve multimodal reasoning abilities. Despite these advancements, MLLMs typically rely on a training paradigm comprising pre-training and supervised fine-tuning, which suffers from the curve of distribution shift and exhibits limited multimodal reasoning abilities. In this work, we conduct a systematic study on using preference optimization to enhance the multimodal reasoning ability of MLLMs.

Preference Optimization. Preference optimization (PO) is a crucial technique for advancing LLMs and MLLMs. Specifically, Reinforcement Learning from Human Feedback (RLHF) uses human preferences as a reward signal to fine-tune models, aligning them with human preferences. InstructGPT [74] employs a reward model as a proxy for human preferences and maximizes this reward via the PPO algorithm [84], improving the model’s ability to follow user intent and become more helpful, honest, and harmless (3H). PPO-Max [97, 118] carefully explores the implementation details of PPO, proposing a more stable version of the algorithm. Additionally, DPO [79] proposes an efficient PO algorithm based on the Bradley-Terry model [9], removing the need for an explicit reward model. Subsequent works [4, 21, 26, 29, 33, 43, 55] have further analyzed and refined this method from various perspectives. In natural language processing, a series of works [43, 76] have explored how to leverage PO to enhance reasoning ability. In the multimodal field, however, most methods [48, 88, 110, 111, 117] primarily focus on reducing hallucination, leaving the potential for PO to improve multimodal reasoning ability under-explored. This work demonstrates that PO not only mitigates hallucinations but also strengthens multimodal reasoning abilities, highlighting its broader applicability in MLLM development.

#### 3. Scalable Multimodal Preference Dataset Generation

To address the scarcity of multimodal preference data, we introduce a scalable data construction pipeline. Based on this pipeline, we construct a million-level MultiModal

PReference dataset (MMPR).

##### 3.1. Data Engine

Definition. Each data sample in our MMPR consists of an image I ∈ I, an instruction x ∈ X, a chosen response yc ∈ Yp, and a rejected response yr ∈ Yn, where yc is preferable to yr. The image sets I and instruction sets X are collected from existing datasets. Yp and Yn represent the positive and negative response set, respectively. Given a certain image I and instruction x, we sample the candidate response y from an initial instruction model M0 as follows:

y ∼ M0(y | x,I), (1)

where M0(y | x,I) represents the response distribution of M0 conditioned on image I and instruction x.

For instructions with clear ground truths, the model is prompted to first provide the reasoning process and then give the final answer in the format like “Final Answer: ***”. Responses matching the ground truth answer constitute the positive set Yp, while those that do not match make up the negative set Yn. Additionally, responses that fail to provide a clear final answer are also merged into Yn. Given these responses labeled as positive or negative, we build the preference pairs by selecting a chosen response yc from Yp and a negative response yr from Yn.

For instructions without clear ground truths, we propose a simple yet effective method: Dropout Next-Token Prediction (Dropout NTP). Specifically, we directly consider all responses generated from Equation 1 as positive set Yp. To generate the negative set Yn, we sample a response y from Yp and drop the last half of this response. The model is required to complete the remained response as follows:

y˜≥j ∼ M0(˜y≥j | x,y<j), (2)

where y<j and y≥j is the remained part and truncated part of y, respectively. y˜≥j is the completion of y<j without the image input. The original response y = [y<j,y≥j] serves as the chosen response yc and the completed response y˜ = [y<j,y˜≥j] serves as the rejected response yr. It is worth noting that while the responses generated by M0 may not be perfect, the completions generated without the image input will introduce more hallucinations than those generated with the image input. Therefore, the partial order relationship between y and y˜ holds true.

Compared with previous methods, our data engine is as effective as the more complex divide-and-conquer method proposed in RLAIF-V [111] (see the experimental results in Section 5.2.2), while more efficient. Taking data generation for M3CoT as an example, our pipeline incurs a token cost of 571.2 per preference pair, compared to 992.7 tokens for the divide-and-conquer approach used in RLAIFV. Thus, the cost of our pipeline is only 57.5% of that of RLAIF-V. Additionally, a comparison with other recent data pipelines [25, 71, 120] is also presented in Section 5.2.2.

Task Dataset General VQA VQAv2 [30], GQA [35], OKVQA [64], IconQA [60] Science AI2D [40], ScienceQA [61], M3CoT [16] Chart ChartQA [65], DVQA [38], MapQA [13] Mathematics

GeoQA+ [12], CLEVR-Math [52], Geometry3K [59], GEOS [85], GeomVerse [39], Geo170K [28]

OCRVQA [69], InfoVQA [67], TextVQA [86], STVQA [8], SROIE [34]

OCR

Document DocVQA [66]

Table 1. Datasets used to build our preference dataset.

##### 3.2. Multimodal Preference Dataset

Dataset Statistics. Using this pipeline, we build a largescale multimodal preference dataset, MMPR. Data examples are presented in Figure 2. See more examples in the Appendix. This dataset comprises approximately 750K samples without clear ground truths and 2.5M samples with clear ground truths. For samples without clear ground truths, each instruction averages 25.0 tokens, while the chosen and rejected responses average 211.4 and 171.2 tokens, respectively. The longest chosen and rejected responses consist of 1,342 and 1,642 tokens, respectively, whereas the shortest chosen and rejected responses contain 20 and 17 tokens, respectively. For samples with clear ground truths, the average instruction length is 79.5 tokens, with the chosen and rejected responses averaging 300.0 and 350.5 tokens, respectively. The longest chosen and rejected responses are composed of 2,018 and 4,097 tokens, while the shortest responses contain 32 and 33 tokens, respectively.

Data Source. As shown in Table 1, to ensure the diversity of instructions and images, we collect samples from diverse domains, including general visual question answering (VQA) [30, 35, 60, 64], science [16, 40, 61], chart [13, 38, 65], mathematics [12, 28, 39, 52, 59, 85], OCR [8, 34, 67, 69, 86], and document [22]. Notably, when constructing open-ended samples, we collect instructions from all the data sources mentioned above and prompt the model to answer the original question without additional requirements. On the other side, when building samples through the correctness-based pipeline, we exclude questions from general VQA and document sources, as verifying the correctness of the generated answers using heuristic rules is challenging for datasets in these domains. For example, the ground truths in VQAv2 [30] consist of a single word or phrase, which may lead to false-negative responses when the model outputs a complete sentence or a synonym as the final answer. Such false-negative responses can negatively impact training effectiveness.

#### 4. Improved Multimodal Large Language Model with Preference Optimization

To enhance the multimodal reasoning capabilities of MLLMs, we propose mixed preference optimization

(MPO), a method that blends supervised fine-tuning (SFT) loss with various preference optimization losses to enhance training effectiveness. Additionally, we investigate different Chain-of-Thought (CoT) approaches with multimodal input to improve reasoning performance.

##### 4.1. Mixed Preference Optimization

We observed that when MLLMs are trained on large-scale preference datasets using direct preference optimization (DPO), they might fail to generate reasonable rationales and produce repetitive responses, as shown in Section 10. This phenomenon aligns with the analysis presented in Smaug [75]. To address this issue, we introduce the MPO in this work, aiming to learn the relative preference between pairs of responses, the absolute quality of individual responses, and the process for generating preferred responses. Training Objective. MPO is defined as a combination of preference loss Lp, quality loss Lq, and generation loss Lg, which can be formulated as follows:

L = wpLp + wqLq + wgLg, (3)

where w∗ represents the weight assigned to each loss component. In this work, we empirically compare different variants of preference loss [4, 14, 21, 33, 37, 55, 68, 70, 79, 105]. Based on the experimental results, we use DPO [79] as our preference loss and BCO [37] as our quality loss.

Preference Loss. The DPO [79] serves as the preference loss to enable the model to learn the relative preference between chosen and rejected responses. DPO eliminates the requirement of training an explicit reward model based on the assumption of the Bradley-Terry model [9] and optimizes the following loss function:

πθ (yr | x) π0 (yr | x)

πθ (yc | x) π0 (yc | x) − β log

, (4)

Lp = − log σ β log

where β is the KL penalty coefficient, and x, yc, and yr are user query, chosen response, and rejected response, respectively. The policy model πθ is initialized from model π0.

Quality Loss. The BCO loss [37] is employed as the quality loss, which helps the model to understand the absolute quality of individual responses. This algorithm trains a binary classifier, where the logit serves as a reward and effectively maps the chosen response to 1 and the rejected response to 0. The loss function is defined as:

Lq = L+q + L−q , (5)

where L+q and L−q represent the loss for chosen and rejected responses, respectively. They are calculated independently, requiring the model to differentiate the absolute quality of individual responses. The loss terms are given by:

πθ (yc | x) π0 (yc | x) − δ , (6)

L+q = −log σ β log

###### Instruction with clear ground truths

Question:

Your task is to answer the question below. Give step by step reasoning before you answer, and when you’re ready to answer, please use the format “Final answer: …” Question: What could be the reason for a person to have a brown teddy bear tied with a ribbon around its neck, a pack of colored pencils and a colorful patterned blanket?

[Figure 1]

- A. The person likes to collect teddy bears, color pencils, and blankets as they are a fan of art
- B. The person received the brown teddy bear as a gift from someone and owns the colored pencils because they are a fan of art
- C. The person is a child who loves to play with teddy bears and draw with color pencils
- D. The person is a school teacher who uses these items to decorate their classroom for their students

###### Chosen Response:

###### Rejected Response:

To determine the answer, let’s analyze each option:

To determine the reason for a person to have a brown teddy bear tied with a ribbon around its neck, a pack of colored pencils, and a colorful patterned blanket, we need to consider the context and meaning behind these items.

- A. The person likes to collect teddy bears, color pencils, and blankets as they are a fan of art. This option assumes that the person collects multiple items as part of a fan interest in art. The items are not just limited to teddy bears, colored pencils, and blankets.
- B. The person received the brown teddy bear as a gift from someone and owns the colored pencils because they are a fan of art. This option suggests that the teddy bear was received as a gift and that the person uses colored pencils as they are a fan of art. This explanation is plausible.
- C. The person is a child who loves to play with teddy bears and draw with color pencils. This option simply states that the person is a child, but it does not provide a clear connection between the other items like the vibrant blanket.
- D. The person is a school teacher who uses these items to decorate their classroom for their students. This option seems less likely because the items (teddy bear, colored pencils, and blanket) might not be common classroom decoration items. Final answer: B

- Option A: “The person likes to collect teddy bears, color pencils, and blankets as they are a fan of art.” This option suggests that the person has a specific interest in collecting items, which may not necessarily imply a connection to art.
- Option B: “The person received the brown teddy bear as a gift from someone and owns the colored pencils because they are a fan of art.” This option suggests a personal connection to a gift while considering an individual interest in art.
- Option C: “The person is a child who loves to play with teddy bears and draw with color pencils.” This option assumes that the person is a child, which could be a plausible explanation for the items.
- Option D: “The person is a school teacher who uses these items to decorate their classroom for their students.” This option suggests a professional context rather than personal interest or collecting items. Given the context of the items in the image and the most plausible explanation based on the presence of children’s items and a child-friendly material like a teddy bear with a ribbon, the most likely explanation is: Final answer: C

Instruction without clear ground truths

###### Question:

Where is the woman in the picture located?

[Figure 2]

###### Chosen Response:

###### Rejected Response:

The woman in the picture is located in a public park. Here are the observations that support this conclusion:

The woman in the picture is located in a public park. Here are the observations that support this conclusion:

- 1.Surroundings: The image shows a well-maintained area with grass, trees, and benches, which is typical of public parks. The presence of other people walking in the background also suggests a public space.
- 2.Lighting and Ambience: The bright daylight and open space are characteristic of outdoor environments like parks, rather than indoor locations like libraries.
- 3.Bench Design: The style of the bench, which appears to be made of wood or metal with a decorative back, is common in public parks. Such benches are often placed along walking paths for public use.
- 4.Vegetation: The variety and arrangement of trees and shrubs look like planned landscaping, which is typical of parks rather than the natural, untamed growth found in a virgin forest. These elements collectively indicate that the setting is a public park.

- 1.Surroundings: The image shows a well-maintained area with grass, trees, and benches, which is typical of public parks. The presence of other people walking in the background also suggests a public space.
- 2.Lighting and Ambience: The bright daylight and open space are characteristic of outdoor environments like parks, rather than indoor locations like libraries.
- 3.Bench: If she were in a private garden or library, it would likely be more private and less likely to have communal spaces such as a bench.
- 4.Background Activity: The presence of other people in the background suggests a public space where people gather, such as a park. Given these observations, the picture most likely shows a woman in a public park.

- Figure 2. Data examples in MMPR. For instructions with clear ground truths, we propose a correctness-based pipeline, which samples multiple solutions and considers those with correct answers as chosen responses and those with incorrect answers as rejected responses. For instructions without clear ground truths, we propose DropoutNTP to generate rejected responses. Differences between the chosen and rejected responses are emphasized in italicized text. Red highlights incorrect responses.

πθ (yr | x) π0 (yr | x) − δ , (7)

is used for samples from chart, OCR, and document domains. (3) Grounded CoT: The model generates a text response while simultaneously linking all referenced objects in the response to corresponding regions in the image. This approach is applied to general VQA domain samples.

L−q = −log σ − β log

where δ represents the reward shift, calculated as the moving average of previous rewards to stabilize training.

Generation Loss. The SFT loss is used as the generation loss to help the model learn the generation process of preferred responses. The loss function is defined as:

Responses generated by these above CoT methods are mixed with those sampled using the prompt shown in the bottom case of Figure 2. These approaches not only effectively integrate multimodal information into the reasoning process but also enhance data diversity. Furthermore, including the background knowledge and visual contents at the start of responses also improves the quality of the negative responses generated by DropoutNTP, preventing a significant quality gap between positive and negative samples that could reduce training effectiveness.

log πθ (yc | x) |yc|

. (8)

Lg = −

##### 4.2. Chain-of-Thought with Multimodal Input

During the data sampling process, we require the model to provide a detailed CoT reasoning process instead of directly answering the final answer. For most samples, we sample the responses using the prompt shown in the bottom case of Figure 2, which requires the model to perform a step-bystep analysis. Considering that MLLMs involve non-textual inputs, we further introduce the following CoT methods: (1) Background Knowledge-based CoT: The model first introduces relevant background knowledge related to the problem or image, followed by reasoning steps and the final answer. This approach is applied to samples from the science domain. (2) Visual Content-based CoT: The model begins by analyzing the visual contents in the image, then proceeds with reasoning and the final answer. This method

#### 5. Experiments

##### 5.1. Main Results

In this section, we compare our models with leading MLLMs on multimodal reasoning [16, 62, 78, 98, 107, 113, 114], complex Visual Question Answering (VQA) [112], and hallucination evaluation [88] tasks.

Benchmarks. We evaluate the reasoning abilities of MLLMs across seven benchmarks, including M3CoT [16], MMMU [113], MathVista [62], MathVision [98], Math-

Model Name M3CoT MMMU MathVista MathVision MathVerse-VO WeMath LogicVista MMVet MMHal Proprietary Models

Gemini-1.5-Pro [81] - - 63.9 19.2 - - - - GPT-4o [73] 64.3 70.7 63.8 30.4 40.6 45.8 52.8 69.1 4.0 GPT-4o-Mini [73] 61.9 - 52.4 27.3 - - - 66.9 3.6

Open Source Models

LLaVA-1.5-13B [53] 39.5 35.7 27.6 11.1 11.4 1.4 7.0 36.3 2.4 Qwen2-VL-7B [99] 57.8 53.7 58.2 21.1 19.2 11.0 22.3 60.6 3.4 MiniCPM-V-2-6-8B [109] 56.0 49.8 60.6 23.4 18.9 16.4 27.5 57.4 3.6 LLaVA-OneVision-7B [45] 52.3 47.9 63.2 18.4 18.3 9.0 20.9 51.4 3.1

InternVL Models

InternVL2-8B [20] 59.3 51.2 58.3 20.4 20.4 20.2 33.6 54.2 3.3 InternVL2-8B-MPO (ours) 79.2 54.0 67.0 25.7 25.0 28.4 39.8 56.2 3.5 InternVL2-26B [20] 58.2 50.7 59.4 23.4 19.5 18.8 34.0 62.1 3.7 InternVL2-40B [20] 63.6 55.2 63.7 21.4 23.5 22.2 39.8 65.5 3.9 InternVL2-76B [20] 65.4 58.3 67.5 23.7 23.7 32.1 45.6 65.7 3.8 InternVL2-76B-MPO (ours) 82.0 64.4 70.9 30.6 37.7 39.4 49.0 69.5 4.2

Table 2. Results on multimodal benchmarks. M3CoT [16] and MMMU [113] are multidisciplinary reasoning benchmarks. MathVista [62], MathVision [98], MathVerse [114], and WeMath [78] are mathematics benchmarks. For MathVerse, we report the performance on Vision-Only (VO) split. LogicVista [107] is a logical reasoning benchmark. Additionally, MMVet [112] and MMHal [88] are designed for general VQA and hallucination evaluation, respectively. Our InternVL2-8B-MPO demonstrates superior performance compared to InternVL2-8B across multimodal reasoning, VQA, and hallucination evaluation benchmarks. Notably, both InternVL2-8B-MPO and InternVL2-76B-MPO exhibit significant performance improvements over their counterparts before MPO.

Verse [114], WeMath [78], and LogicVista [107]. The evaluation samples include subject-based, mathematical, and logical reasoning problems. We report the overall accuracy for these benchmarks. For MathVerse, we report the performance on the Vision-Only split. Additionally, we evaluate the general VQA abilities and hallucinations of MLLMs using MMVet [112] and MMHalBench [88], respectively.

Results. As shown in Table 2, InternVL2-8B-MPO and InternVL2-76B-MPO achieve superior performance across all benchmarks compared to their counterparts before MPO, particularly excelling in multimodal reasoning tasks. On the MathVista benchmark, InternVL2-8B-MPO achieves an accuracy of 67.0%, outperforming InternVL2-8B by 8.7 points and achieving performance comparable to the 10× larger InternVL2-76B. Similarly, on the MathVision benchmark, InternVL2-76B-MPO achieves an accuracy of 30.6%, establishing a new state-of-the-art performance. These results highlight the effectiveness of our preference optimization approach in enhancing multimodal reasoning capabilities. Furthermore, our models also show superior performance over their pre-MPO counterparts on VQA [112] and hallucination [88] benchmarks, indicating that their general abilities are also improved, benefiting from enhanced reasoning abilities and mitigated hallucinations. Notably, InternVL2-76B-MPO exhibits a more significant performance improvement over InternVL2-76B than the improvement observed in the 8B model, demonstrating the scalability of our method.

Model Name Setting M3CoT MathVista MMVet POPE InternVL2-8B

Direct 59.3 58.3 54.2 86.9

CoT 57.0 56.8 54.7 82.9 InternVL2-8B-SFT

Direct 63.9 62.7 54.7 86.5

CoT 67.8 64.2 53.8 84.0 InternVL2-8B-MPO

Direct 77.2 64.5 55.1 87.0 CoT 79.2 67.0 56.2 88.1

Table 3. Results of models trained with SFT and MPO. The SFT training data consists of the chosen responses from the preference pairs used in MPO training. In the Direct setting, the model is prompted to provide the answer directly, while in the CoT setting, the model is instructed to answer with detailed rationales.

##### 5.2. Ablation Study

In this section, we present ablation studies to analyze the effects of preference optimization and SFT on multimodal reasoning abilities. Additionally, we compare our proposed DropoutNTP method with recent related data pipelines [25, 71, 120] and the divide-and-conquer approach from RLAIF-V [111], demonstrating the effectiveness of our approach. Furthermore, we conduct extensive experiments to analyze the effects of different preference optimization algorithms. We also present analysis of the effects on text-only performance.

###### 5.2.1. Comparison between MPO and SFT

To compare the impact of MPO and SFT on improving multimodal reasoning ability, we use the chosen responses

in MMPR as SFT data to fine-tune InternVL2-8B. As shown in Table 3, the results indicate that the model trained with MPO consistently outperforms that trained with SFT across all benchmarks. For example, the MPO-trained model achieves a score of 79.2 on the multimodal reasoning benchmark M3CoT, surpassing its SFT counterpart by 11.4 points. Furthermore, the MPO-trained model also performs better on the general benchmark [112] and the hallucination benchmark [50]. Notably, the SFT-trained model performs worse with CoT responses than with direct-answer responses on MMVet and POPE, demonstrating that SFT alone is insufficient to enhance multimodal CoT abilities. These results demonstrate that while SFT provides moderate improvement, preference optimization is more effective in improving the overall performance of the model.

###### 5.2.2. Comparison with Different Data Pipelines

Here, we compare our proposed Dropout NTP method with SeVa [71], POVID [120], STIC [25], and the divideand-conquer approach from RLAIF-V [111]. To ensure a fair comparison, we use the same prompts and chosen responses as in RLAIF-V and replace the rejected responses with those generated by different data pipelines. Following RLAIF-V, we report the hallucination rates in responselevel (Resp.) and mention-level (Ment.) for Object HalBench [82] and overall score and hallucination rates (Hall.) for MMHal-Bench [88].

As shown in Table 4, the model trained with our data achieves performance comparable to that of the model trained with RLAIF-V, demonstrating the effectiveness of our method. Specifically, the response-level hallucination rate of the model trained with our data on Object HalBench is 7.6, compared to 7.3 for its counterpart. Besides, this model achieves a score of 3.6 on the MMHal-Bench, compared to 3.5 for its counterpart. Note that our method requires the model to generate only a single continuation for each sample, while RLAIF-V requires the model to decompose the response into atomic claims and then verify each one individually. Therefore, our method is more efficient. A quantitative analysis is provided in Section 3.1.

Furthermore, DropoutNTP outperforms SeVa, POVID, and STIC. We manually review the generated samples and find that responses generated by DropoutNTP are stronger negative samples, which leads to better training effectiveness. We attribute this to the fact that most open-source MLLMs are trained on text-only samples and those paired with clean images. Therefore, corrupted images are out-ofdomain inputs for these models, and responses generated conditioned on corrupted images are much worse than those generated with text-only input.

###### 5.2.3. Effects of Optimization Algorithms

Here, we empirically compare the effectiveness of different optimization algorithms, including (1) DPO [79],

Object HalBench MM HalBench Resp. (↓) Ment. (↓) Score Hall. (↓)

Method

InternVL2-8B 18.4 8.7 3.3 40.6 POVID [120] 10.5 5.5 3.4 38.5 STIC [25] 9.4 5.3 3.5 37.5 SeVa [71] 9.1 4.6 3.4 39.6 RLAIF-V [111] 7.3 3.9 3.5 32.3 DropoutNTP (ours) 7.6 4.1 3.6 31.3

Table 4. Comparison of DropoutNTP and other data construction pipelines. We replace the negative samples in RLAIF-V with responses generated using different pipelines.

which directly fine-tunes the model on an offline preference dataset without explicitly constructing a reward function. (2) RSO [55], which applies a hinge loss on the normalized likelihood instead of the sigmoid loss used in DPO. (3) IPO [4], which introduces a modified loss function to address overfitting in DPO by averaging log-likelihoods and controlling the gap between chosen and rejected completions via a beta parameter. (4) cDPO [70], which is a modification of the DPO loss that accounts for potential label noise in preference data. (5) RobustDPO [21], which provides an unbiased estimate of the DPO loss designed to handle preference noise in data. Similar to cDPO, it assumes that labels are noisy with a certain probability. (6) BCO [37], which introduces a binary classifier trained to output logits used as reward values. (7) SPPO [105], which iteratively pushes chosen rewards toward 1/2 and rejected rewards toward -1/2 to approximate a Nash equilibrium, aiming to reduce data sparsity issues. (8) AOT [68], which applies Distributional Preference Alignment via Optimal Transport. (9) TR-DPO [29], which adds synchronization between the model and a reference model every few steps to mitigate overfitting during DPO training. (10) ORPO [33], a reference model-free preference optimization algorithm that uses a log odds ratio penalty appended to the NLL loss, allowing for preference-aligned fine-tuning without an additional preference alignment phase. For all algorithms, we set the learning rate to 5e-6 and use the hyper-parameters suggested in their corresponding paper. Additionally, we extend these algorithms with SFT loss to analyze its impact. The SFT model trained with the chosen responses of the reasoning preference data is also included as a baseline.

Notably, most current benchmarks lack corresponding in-distribution training samples, and the data distribution of our MMPR may differ from that of these benchmarks. This discrepancy can introduce additional variability when analyzing the impact of different optimization algorithms on training results. Therefore, we use the training and validation sets of M3CoT [16] for ablation studies.

The visualization results are illustrated in Figure 3, and the numerical results are presented in Table 6 and 8. We

###### Direct CoT

79.1

78.9

78.4

78.0 78.4 78.0

78.1

77.7

77.4

76.8

76.7 75.9

76.5

76.4

76.3

76.2

76.0

75.8

75.1

74.3

74.2

74.2

74.2

73.9

73.1

72.8

72.7

71.6

68.5

67.4

66.8

66.6

66.2

65.7

59.3

57.0

InternVL2-8B SFT DPO RSO IPO cDPO RobustDPO BCO SPPO AOT TR-DPO ORPO DPO+ cDPO+ RobustDPO+ BCO+ AOT+ MPO

###### Baseline DPO Variants w/o SFT Loss DPO Variants w. SFT Loss

- Figure 3. Results of models trained with different preference optimization algorithms on M3CoT. The algorithm X extended with the SFT loss is called X+ for brevity. For instance, DPO+ denotes the combination of DPO loss and SFT loss.

Setting MMLU Gaokao TriviaQA NQ C3 Race-h BBH GSM8K Math TheoremQA IFEval HumanEval MBPP Average

Baseline 73.2 75.0 62.0 28.1 94.2 90.8 72.7 75.6 39.5 15.6 52.3 69.5 58.8 62.1 SFT 71.8 74.4 63.7 28.2 94.3 90.6 72.1 75.5 40.1 15.8 53.6 68.3 58.0 62.0 MPO 71.0 74.8 64.2 29.3 94.2 90.6 71.8 75.0 40.4 20.8 56.4 68.9 61.5 63.0

Table 5. Results on text-only benchmarks. The model fine-tuned through MPO exhibits superior overall performance on text-only tasks compared to the baseline model and its SFT counterpart, particularly on TheoremQA and IFEval.

cant improvements are observed on TheoremQA and IFEval. Specifically, our model trained with MPO achieves an accuracy of 20.8 on TheoremQA, a benchmark consisting of complex science problems, outperforming the baseline model by 5.2 points and the SFT counterpart by 5.0 points. Additionally, since our dataset considers responses that fail to follow instructions as negative samples when constructing data using our correctness-based pipeline, our model also exhibits enhanced instruction-following abilities on IFEval, outperforming the baseline model by 4.1 points and the SFT counterpart by 2.8 points.

can observe that almost all preference optimization methods outperform their SFT counterpart in both the Direct and CoT settings. However, DPO and its variants struggle to enhance the CoT reasoning abilities of the model as the resulting models exhibit trivial or no improvement when answering with CoT reasoning responses compared to directanswer responses. On the other hand, when combining SFT Loss with these DPO variants, all algorithms are able to improve the model’s CoT reasoning abilities, demonstrating that the SFT loss is a key component for enhancing CoT reasoning abilities. Additionally, models trained with TR-DPO, a DPO variant that updates the reference model every few steps, perform much worse when using CoT reasoning compared to direct-answer responses. Similarly, the model trained with ODPO, a reference-model-free method, achieves worse overall performance compared to other methods extended with SFT Loss.

#### 6. Conclusion

In this work, we introduce a preference optimization (PO) process to enhance the multimodal reasoning capabilities of MLLMs. On the data side, we design an automated pipeline for preference data construction, which is applicable to instructions both with and without clear ground truths. Using this pipeline, we create MMPR, a high-quality, largescale multimodal reasoning preference dataset. On the model side, we propose a simple yet effective method called Mixed Preference Optimization (MPO). This algorithm aims to learn the relative preference between pairs of responses, the absolute quality of individual responses, and the process for generating preferred responses. The resulting models, InternVL2-8B-MPO and InternVL2-76BMPO, exhibit enhanced multimodal reasoning ability and fewer hallucinations compared to their counterparts before MPO. Notably, InternVL2-8B-MPO even achieves performance comparable to the 10× larger InternVL2-76B on MathVista. We hope this study could inspire further advancements in MLLMs.

These results indicate that the reference model constraint on policy updates is crucial for enhancing overall reasoning abilities, and the reference model should remain frozen during training. Notably, models trained with DPO+ and BCO+ exhibit the best CoT performance among existing algorithms. Therefore, we use DPO and BCO as the preference loss and quality loss. The resulting algorithm (i.e., MPO) further improves the overall performance.

##### 5.3. Effects on Text-only Performance

We evaluate the text-only performance of our models on a series of benchmarks [3, 15, 17, 23, 31, 32, 36, 41, 42, 87, 89, 116, 119] and report the average performance across them. As shown in Table 5, although our MMPR dataset does not include any text-only data, the MPO-trained model achieves superior average performance on these benchmarks compared to the baseline model. The most signifi-

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 1
- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. NIPS, 35:23716–23736, 2022. 3
- [3] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021. 8
- [4] Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. A general theoretical paradigm to understand learning from human preferences. In International Conference on Artificial Intelligence and Statistics, pages 4447–4455. PMLR, 2024. 3, 4, 7, 1
- [5] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 1, 3
- [6] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966,

2023. 1

- [7] Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, and Sa˘gnak Ta¸sırlar. Introducing our multimodal models, 2023. 3
- [8] Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez, Mar¸cal Rusinol, Ernest Valveny, CV Jawahar, and Dimosthenis Karatzas. Scene text visual question answering. In ICCV, pages 4291–4301, 2019. 4
- [9] Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952. 3, 4
- [10] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NIPS, 2020. 1
- [11] Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, et al. Internlm2 technical report. arXiv preprint arXiv:2403.17297, 2024. 1, 3
- [12] Jie Cao and Jing Xiao. An augmented benchmark dataset for geometric question answering through dual parallel text encoding. In COLING, pages 1511–1520, 2022. 4
- [13] Shuaichen Chang, David Palzer, Jialin Li, Eric FoslerLussier, and Ningchuan Xiao. Mapqa: A dataset for question answering on choropleth maps. arXiv preprint

arXiv:2211.08545, 2022. 4

- [14] Huayu Chen, Guande He, Hang Su, and Jun Zhu. Noise contrastive alignment of language models with explicit rewards. arXiv preprint arXiv:2402.05369, 2024. 4
- [15] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021. 8
- [16] Qiguang Chen, Libo Qin, Jin Zhang, Zhi Chen, Xiao Xu, and Wanxiang Che. M3cot: A novel benchmark for multidomain multi-step multi-modal chain-of-thought. arXiv preprint arXiv:2405.16473, 2024. 4, 5, 6, 7, 2
- [17] Wenhu Chen, Ming Yin, Max Ku, Pan Lu, Yixin Wan, Xueguang Ma, Jianyu Xu, Xinyi Wang, and Tony Xia. Theoremqa: A theorem-driven question answering dataset. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7889–7901, 2023. 8
- [18] Yangyi Chen, Karan Sikka, Michael Cogswell, Heng Ji, and Ajay Divakaran. Dress: Instructing large vision-language models to align and interact with humans via natural language feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14239–14250, 2024. 2
- [19] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Zhong Muyan, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023. 3
- [20] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with opensource suites. arXiv preprint arXiv:2404.16821, 2024. 1, 3, 6
- [21] Sayak Ray Chowdhury, Anush Kini, and Nagarajan Natarajan. Provably robust dpo: Aligning language models with noisy feedback. arXiv preprint arXiv:2403.00409, 2024. 3, 4, 7, 1
- [22] Christopher Clark and Matt Gardner. Simple and effective multi-paragraph reading comprehension. In ACL, pages 845–855, 2018. 4
- [23] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. 8
- [24] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning. NIPS, 36, 2024. 1
- [25] Yihe Deng, Pan Lu, Fan Yin, Ziniu Hu, Sheng Shen, Quanquan Gu, James Zou, Kai-Wei Chang, and Wei Wang. Enhancing large vision language models with self-training on image comprehension. arXiv preprint arXiv:2405.19716,

2024. 3, 6, 7

- [26] Hanze Dong, Wei Xiong, Bo Pang, Haoxiang Wang, Han Zhao, Yingbo Zhou, Nan Jiang, Doyen Sahoo, Caiming Xiong, and Tong Zhang. Rlhf workflow: From reward modeling to online rlhf. arXiv preprint arXiv:2405.07863, 2024. 3
- [27] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 1, 3

- [28] Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, et al. G-llava: Solving geometric problem with multi-modal large language model. arXiv preprint arXiv:2312.11370, 2023. 3, 4
- [29] Alexey Gorbatovski, Boris Shaposhnikov, Alexey Malakhov, Nikita Surnachev, Yaroslav Aksenov, Ian Maksimov, Nikita Balagansky, and Daniil Gavrilov. Learn your reference model for real good alignment. arXiv preprint arXiv:2404.09656, 2024. 3, 7, 1
- [30] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In CVPR, pages 6904–6913, 2017. 4
- [31] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020. 8
- [32] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021. 8
- [33] Jiwoo Hong, Noah Lee, and James Thorne. Orpo: Monolithic preference optimization without reference model. arXiv preprint arXiv:2403.07691, 2(4):5, 2024. 3, 4, 7, 1
- [34] Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and CV Jawahar. Icdar2019 competition on scanned receipt ocr and information extraction. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1516–1520. IEEE, 2019. 4
- [35] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, pages 6700–6709, 2019. 4
- [36] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. arXiv preprint arXiv:1705.03551, 2017. 8
- [37] Seungjae Jung, Gunsoo Han, Daniel Wontae Nam, and Kyoung-Woon On. Binary classifier optimization for large language model alignment. arXiv preprint arXiv:2404.04656, 2024. 4, 7, 1
- [38] Kushal Kafle, Brian Price, Scott Cohen, and Christopher Kanan. Dvqa: Understanding data visualizations via question answering. In CVPR, pages 5648–5656, 2018. 4
- [39] Mehran Kazemi, Hamidreza Alvari, Ankit Anand, Jialin Wu, Xi Chen, and Radu Soricut. Geomverse: A systematic

- evaluation of large models for geometric reasoning. arXiv preprint arXiv:2312.12241, 2023. 4
- [40] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In ECCV, pages 235–251, 2016. 4
- [41] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453–466, 2019. 8
- [42] Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. Race: Large-scale reading comprehension dataset from examinations. arXiv preprint arXiv:1704.04683, 2017. 8
- [43] Xin Lai, Zhuotao Tian, Yukang Chen, Senqiao Yang, Xiangru Peng, and Jiaya Jia. Step-dpo: Step-wise preference optimization for long-chain reasoning of llms. arXiv preprint arXiv:2406.18629, 2024. 2, 3
- [44] Hugo Laurenc¸on, Lucile Saulnier, L´eo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander Rush, Douwe Kiela, et al. Obelics: An open web-scale filtered dataset of interleaved image-text documents. NIPS, 36, 2024. 1
- [45] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 1, 6
- [46] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML, pages 12888–12900, 2022. 3
- [47] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, pages 19730–19742. PMLR, 2023. 1, 3
- [48] Lei Li, Zhihui Xie, Mukai Li, Shunian Chen, Peiyi Wang, Liang Chen, Yazheng Yang, Benyou Wang, and Lingpeng Kong. Silkie: Preference distillation for large visual language models. arXiv preprint arXiv:2312.10665, 2023. 2, 3
- [49] Qingyun Li, Zhe Chen, Weiyun Wang, Wenhai Wang, Shenglong Ye, Zhenjiang Jin, Guanzhou Chen, Yinan He, Zhangwei Gao, Erfei Cui, et al. Omnicorpus: An unified multimodal corpus of 10 billion-level images interleaved with text. arXiv preprint arXiv:2406.08418, 2024. 1, 3
- [50] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In EMNLP, pages 292–305,

2023. 7

- [51] Xi Victoria Lin, Akshat Shrivastava, Liang Luo, Srinivasan Iyer, Mike Lewis, Gargi Gosh, Luke Zettlemoyer, and Armen Aghajanyan. Moma: Efficient early-fusion pretraining with mixture of modality-aware experts. arXiv preprint arXiv:2407.21770, 2024. 3
- [52] Adam Dahlgren Lindstr¨om and Savitha Sam Abraham. Clevr-math: A dataset for compositional lan-

- guage, visual and mathematical reasoning. arXiv preprint arXiv:2208.05358, 2022. 4
- [53] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023. 1, 6
- [54] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. NIPS, 36, 2023. 1, 3
- [55] Tianqi Liu, Yao Zhao, Rishabh Joshi, Misha Khalman, Mohammad Saleh, Peter J Liu, and Jialu Liu. Statistical rejection sampling improves preference optimization. arXiv preprint arXiv:2309.06657, 2023. 3, 4, 7, 1
- [56] Yangzhou Liu, Yue Cao, Zhangwei Gao, Weiyun Wang, Zhe Chen, Wenhai Wang, Hao Tian, Lewei Lu, Xizhou Zhu, Tong Lu, et al. Mminstruct: A high-quality multi-modal instruction tuning dataset with extensive diversity. arXiv preprint arXiv:2407.15838, 2024. 1, 3
- [57] Zhaoyang Liu, Yinan He, Wenhai Wang, Weiyun Wang, Yi Wang, Shoufa Chen, Qinglong Zhang, Zeqiang Lai, Yang Yang, Qingyun Li, Jiashuo Yu, et al. Interngpt: Solving vision-centric tasks by interacting with chatgpt beyond language. arXiv preprint arXiv:2305.05662, 2023. 3
- [58] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 1
- [59] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. arXiv preprint arXiv:2105.04165,

2021. 4

- [60] Pan Lu, Liang Qiu, Jiaqi Chen, Tony Xia, Yizhou Zhao, Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu. Iconqa: A new benchmark for abstract diagram understanding and visual language reasoning. arXiv preprint arXiv:2110.13214, 2021. 4
- [61] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. NIPS, 35:2507–2521, 2022. 4
- [62] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023. 1, 2, 5, 6
- [63] Gen Luo, Xue Yang, Wenhan Dou, Zhaokai Wang, Jifeng Dai, Yu Qiao, and Xizhou Zhu. Mono-internvl: Pushing the boundaries of monolithic multimodal large language models with endogenous visual pre-training. arXiv preprint arXiv:2410.08202, 2024. 3
- [64] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In CVPR, pages 3195–3204, 2019. 4
- [65] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In ACL, pages 2263–2279, 2022. 4

- [66] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In WACV, pages 2200–2209, 2021. 4
- [67] Minesh Mathew, Viraj Bagal, Rub`en Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In WACV, pages 1697–1706, 2022. 4
- [68] Igor Melnyk, Youssef Mroueh, Brian Belgodere, Mattia Rigotti, Apoorva Nitsure, Mikhail Yurochkin, Kristjan Greenewald, Jiri Navratil, and Jerret Ross. Distributional preference alignment of llms via optimal transport. arXiv preprint arXiv:2406.05882, 2024. 4, 7, 1
- [69] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In ICDAR, pages 947–952, 2019. 4
- [70] Eric Mitchell. A note on dpo with noisy preferences & relationship to ipo, 2023. 4, 7, 1
- [71] Kushin Mukherjee, Holly Huey, Xuanchen Lu, Yael Vinker, Rio Aguina-Kang, Ariel Shamir, and Judith Fan. Seva: Leveraging sketches to evaluate alignment between human and machine visual abstraction. Advances in Neural Information Processing Systems, 36:67138–67155, 2023. 3, 6, 7
- [72] OpenAI. Gpt-4v(ision) system card. https://cdn. openai.com/papers/GPTV_System_Card.pdf,

2023. 1

- [73] OpenAI. Gpt-4o system card. https://openai.com/ index/gpt-4o-system-card/, 2024. 6
- [74] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35: 27730–27744, 2022. 3
- [75] Arka Pal, Deep Karkhanis, Samuel Dooley, Manley Roberts, Siddartha Naidu, and Colin White. Smaug: Fixing failure modes of preference optimisation with dpo-positive. arXiv preprint arXiv:2402.13228, 2024. 4
- [76] Richard Yuanzhe Pang, Weizhe Yuan, Kyunghyun Cho, He He, Sainbayar Sukhbaatar, and Jason Weston. Iterative reasoning preference optimization. arXiv preprint arXiv:2404.19733, 2024. 2, 3
- [77] Renjie Pi, Tianyang Han, Wei Xiong, Jipeng Zhang, Runtao Liu, Rui Pan, and Tong Zhang. Strengthening multimodal large language model with bootstrapped preference optimization. arXiv preprint arXiv:2403.08730, 2024. 2
- [78] Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Zhuoma GongQue, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? arXiv preprint arXiv:2407.01284, 2024. 5, 6
- [79] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024. 2, 3, 4, 7, 1

- [80] Sylvestre-Alvise Rebuffi, Hakan Bilen, and Andrea Vedaldi. Learning multiple visual domains with residual adapters. NIPS, 30, 2017. 3
- [81] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 1, 6
- [82] Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. Object hallucination in image captioning. arXiv preprint arXiv:1809.02156, 2018. 7
- [83] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. NIPS, 35:25278– 25294, 2022. 1
- [84] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 3
- [85] Minjoon Seo, Hannaneh Hajishirzi, Ali Farhadi, Oren Etzioni, and Clint Malcolm. Solving geometry problems: Combining text and diagram interpretation. In Proceedings of the 2015 conference on empirical methods in natural language processing, pages 1466–1476, 2015. 4
- [86] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In CVPR, pages 8317–8326, 2019. 4
- [87] Kai Sun, Dian Yu, Dong Yu, and Claire Cardie. Investigating prior knowledge for challenging chinese machine reading comprehension. Transactions of the Association for Computational Linguistics, 8:141–155, 2020. 8
- [88] Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liang-Yan Gui, Yu-Xiong Wang, Yiming Yang, et al. Aligning large multimodal models with factually augmented rlhf. arXiv preprint arXiv:2309.14525, 2023. 2, 3, 5, 6, 7
- [89] Mirac Suzgun, Nathan Scales, Nathanael Sch¨arli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, , and Jason Wei. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022. 8
- [90] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818,

2024. 3

- [91] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 1
- [92] InternLM Team. Internlm: A multilingual language model with progressively enhanced capabilities. https:// github.com/InternLM/InternLM, 2023. 1
- [93] Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth,

- and Li-Jia Li. Yfcc100m: The new data in multimedia research. Communications of the ACM, 59(2):64–73, 2016. 1
- [94] Changyao Tian, Xizhou Zhu, Yuwen Xiong, Weiyun Wang, Zhe Chen, Wenhai Wang, Yuntao Chen, Lewei Lu, Tong Lu, Jie Zhou, et al. Mm-interleaved: Interleaved image-text generative modeling via multi-modal feature synchronizer. arXiv preprint arXiv:2401.10208, 2024. 3
- [95] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1
- [96] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 1
- [97] Binghai Wang, Rui Zheng, Lu Chen, Yan Liu, Shihan Dou, Caishuang Huang, Wei Shen, Senjie Jin, Enyu Zhou, Chenyu Shi, et al. Secrets of rlhf in large language models part ii: Reward modeling. arXiv preprint

- arXiv:2401.06080, 2024. 3

[98] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. arXiv preprint

- arXiv:2402.14804, 2024. 5, 6

- [99] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1, 3, 6
- [100] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079, 2023. 3
- [101] Weiyun Wang, Yiming Ren, Haowen Luo, Tiantong Li, Chenxiang Yan, Zhe Chen, Wenhai Wang, Qingyun Li, Lewei Lu, Xizhou Zhu, et al. The all-seeing project v2: Towards general relation comprehension of the open world. arXiv preprint arXiv:2402.19474, 2024. 1, 3
- [102] Weiyun Wang, Min Shi, Qingyun Li, Wenhai Wang, Zhenhang Huang, Linjie Xing, Zhe Chen, Hao Li, Xizhou Zhu, Zhiguo Cao, et al. The all-seeing project: Towards panoptic visual recognition and understanding of the open world. In ICLR, 2024. 1, 3
- [103] Weiyun Wang, Shuibo Zhang, Yiming Ren, Yuchen Duan, Tiantong Li, Shuo Liu, Mengkang Hu, Zhe Chen, Kaipeng Zhang, Lewei Lu, et al. Needle in a multimodal haystack. arXiv preprint arXiv:2406.07230, 2024. 3
- [104] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 3
- [105] Yue Wu, Zhiqing Sun, Huizhuo Yuan, Kaixuan Ji, Yiming Yang, and Quanquan Gu. Self-play preference optimization for language model alignment. arXiv preprint arXiv:2405.00675, 2024. 4, 7, 1

- [106] Zhiheng Xi, Wenxiang Chen, Boyang Hong, Senjie Jin, Rui Zheng, Wei He, Yiwen Ding, Shichun Liu, Xin Guo, Junzhe Wang, et al. Training large language models for reasoning through reverse curriculum reinforcement learning. arXiv preprint arXiv:2402.05808, 2024. 2
- [107] Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. arXiv preprint arXiv:2407.04973, 2024. 5, 6
- [108] Zhen Yang, Jinhao Chen, Zhengxiao Du, Wenmeng Yu, Weihan Wang, Wenyi Hong, Zhihuan Jiang, Bin Xu, Yuxiao Dong, and Jie Tang. Mathglm-vision: Solving mathematical problems with multi-modal large language model. arXiv preprint arXiv:2409.13729, 2024. 3
- [109] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024. 1, 6
- [110] Tianyu Yu, Yuan Yao, Haoye Zhang, Taiwen He, Yifeng Han, Ganqu Cui, Jinyi Hu, Zhiyuan Liu, Hai-Tao Zheng, Maosong Sun, et al. Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13807– 13816, 2024. 2, 3
- [111] Tianyu Yu, Haoye Zhang, Yuan Yao, Yunkai Dang, Da Chen, Xiaoman Lu, Ganqu Cui, Taiwen He, Zhiyuan Liu, Tat-Seng Chua, et al. Rlaif-v: Aligning mllms through open-source ai feedback for super gpt-4v trustworthiness. arXiv preprint arXiv:2405.17220, 2024. 2, 3, 6, 7
- [112] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 5, 6, 7
- [113] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In CVPR, 2024. 5, 6
- [114] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? arXiv preprint arXiv:2403.14624, 2024. 5, 6
- [115] Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Yichi Zhang, Ziyu Guo, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Bin Wei, Shanghang Zhang, et al. Mavis: Mathematical visual instruction tuning. arXiv preprint arXiv:2407.08739, 2024. 3
- [116] Xiaotian Zhang, Chunyang Li, Yi Zong, Zhengyu Ying, Liang He, and Xipeng Qiu. Evaluating the performance of large language models on gaokao benchmark. arXiv preprint arXiv:2305.12474, 2023. 8

- [117] Yongting Zhang, Lu Chen, Guodong Zheng, Yifeng Gao, Rui Zheng, Jinlan Fu, Zhenfei Yin, Senjie Jin, Yu Qiao, Xuanjing Huang, et al. Spa-vl: A comprehensive safety preference alignment dataset for vision language model. arXiv preprint arXiv:2406.12030, 2024. 2, 3
- [118] Rui Zheng, Shihan Dou, Songyang Gao, Yuan Hua, Wei Shen, Binghai Wang, Yan Liu, Senjie Jin, Qin Liu, Yuhao Zhou, et al. Secrets of rlhf in large language models part i: Ppo. arXiv preprint arXiv:2307.04964, 2023. 3
- [119] Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023. 8
- [120] Yiyang Zhou, Chenhang Cui, Rafael Rafailov, Chelsea Finn, and Huaxiu Yao. Aligning modalities in vision large language models via preference fine-tuning. arXiv preprint arXiv:2402.11411, 2024. 3, 6, 7
- [121] Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, and Yejin Choi. Multimodal c4: An open, billion-scale corpus of images interleaved with text. NIPS, 36, 2024. 1

## Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization

### Supplementary Material

#### 7. Implementation Details

During the construction of samples with clear ground truths, we sample at most 32 reasoning processes and construct at most 15 preference pairs for each query. When constructing data using DropoutNTP, we truncate the original response by half and ask InternVL2-8B to complete the response without the image input. Our ablation studies in Section 8.2 show that truncating the original response by 25% or 75% has negative effects on the final performance. We set the temperature to 1.0 during sampling to ensure response diversity. Besides, the maximum tiles for dynamic resolution are set to 6 for the general VQA domain and 12 for OCR-, document-, and chart-related domains.

During the MPO process, the global batch size is set to 256 during training. We employ the AdamW optimizer [58] with the β1 of 0.9, the β2 of 0.999, and the weight decay of 0.05. The learning rate is initialized as 5e-6. The training phases include a linear warmup that lasts until the first 5% of training steps. The warmup is followed by a cosine decay strategy with a minimum learning rate of 0. The KL penalty coefficient β is set to 0.1. For the Equation 3, we set wp to 0.8, wq to 0.2, and wg to 1. The model is initialized from InternVL2-8B [20], and all parameters are trainable during training. We train the model for 1 epoch.

#### 8. More Ablation Studies 8.1. Ablation Studies about PO Variants

In this section, we present the numerical experimental results of ablation studies on the effects of different preference optimization algorithms in Table 6. We define ∆ as the performance gap between CoT reasoning responses and direct-answer responses to quantitatively assess the effects of different preference optimization algorithms on CoT reasoning abilities. Our results indicate that introducing an additional SFT loss can significantly improve the CoT performance compared to each algorithm’s vanilla counterpart.

Note that, to reduce computational costs, we only extend the DPO variants, which exhibit superior performance

- in Table 6 compared to DPO, with SFT Loss. Furthermore, the comparison of different loss combinations is presented
- in Table 7. We can observe that our MPO outperforms other methods. Notably, the model trained with DPO+cDPO performs worse than the one trained with DPO alone. After adding generation loss, the performance of DPO+cDPO improves greatly, but it still lags behind MPO.

In addition to the ablation studies based on M3CoT, we

###### Method Direct CoT ∆

InternVL2-8B 59.3 57.0 -2.3 SFT 65.7 68.5 +2.8

DPO [79] 75.8 72.7 -3.1 RSO [55] 74.2 74.3 +0.1 IPO [4] 72.8 73.1 +0.3 cDPO [70] 76.2 76.8 +0.6 RobustDPO [21] 75.1 74.2 -0.9 BCO [37] 78.1 78.4 +0.3 SPPO [105] 66.2 67.4 +1.2 AOT [68] 76.7 76.0 -0.7 TR-DPO [29] 75.9 66.8 -9.1

ORPO [33] 66.6 73.9 +7.3 DPO+ 76.4 78.9 +2.5 cDPO+ 71.6 74.2 +2.7 RobustDPO+ 76.5 78.0 +1.5 BCO+ 77.4 78.4 +1.0 AOT+ 76.3 78.0 +1.7

MPO (ours) 77.7 79.1 +1.4

- Table 6. Results of models trained with different preference optimization algorithms on M3CoT. ∆ represents the performance gap between CoT responses and direct-answer responses. The algorithm X extended with the SFT loss is referred to as X+ for brevity. For example, DPO+ is the combination of DPO and the SFT loss. Note that ORPO is also equipped with the SFT loss.

Setting DPO

BCO w. DPO

BCO w. cDPO

DPO w. cDPO

DPO w. cDPO+

MPO (ours)

Direct 75.8 77.5 76.1 76.6 77.2 77.7 CoT 72.7 78.5 77.8 67.1 78.4 79.1

- Table 7. Comparison of different loss combinations on M3CoT.

also present the performance of models trained with DPO+ and BCO+ using our MMPR, as shown in Table 8. The experimental results show that the model trained with MPO exhibits superior overall performance compared to those trained with DPO+ and BCO+.

- 8.2. Ablation Studies on DropoutNTP

Here, we present the ablation results for the Dropout Ratio (DR) in our proposed DropoutNTP. By default, we set DR to 0.5, which means that we truncate the positive response by half. Notably, setting DR to 0.25 means using the first quarter of the positive responses for continuation.

InternVL2-8B 59.3 58.3 20.4 54.2 73.2 86.9 75.0 3.3 InternVL2-8B-DPO+ 80.4 66.4 23.4 58.3 74.1 87.6 75.5 3.4 InternVL2-8B-BCO+ 79.6 66.1 18.8 55.5 78.6 88.5 75.5 3.5

InternVL2-8B-MPO (ours) 79.2 67.0 25.7 56.2 76.7 88.1 75.4 3.5

- Table 8. Results of models trained with DPO+, BCO+ and MPO using our MMPR. The model trained with MPO exhibits superior overall performance compared to those trained with DPO+ and BCO+.

Method

Object HalBench MM HalBench Resp. (↓) Ment. (↓) Score Hall. (↓)

DR=0.25 9.3 4.8 3.3 40.6 DR=0.50 7.6 4.1 3.6 31.3 DR=0.75 11.6 6.2 3.3 36.5

- Table 9. Results of DropoutNTP with different Dropout Ratios (DR). The model trained on samples generated with a DR value of 0.50 achieves the best performance.

80.0

80.0

Direct

75.0

78.0

CoT

70.0

76.0

65.0

74.0

Accuracy

60.0

Accuracy

55.0

72.0

50.0

70.0

45.0

Direct

68.0

40.0

CoT

66.0

35.0

30.0

64.0

5e-7 2e-6 5e-6 2e-5 5e-5

10K 40K 70K 100K

Learning Rate

Data Scale

(a)

(b)

80.0

79.8

Direct

Direct

79.0

79.3

CoT

CoT

78.0

78.8

Accuracy

Accuracy

77.0

78.3

76.0

77.8

Following the experimental settings in Section 5.2.2, we replace the negative samples in RLAIF-V with the completions based on different dropout ratios.

75.0

77.3

74.0

76.8

73.0

76.3

0.01 0.5 1.0 2.0 5.0

1:1 3:1 4:1 5:1 9:1

SFT coefficient

PO coefficient

- As shown in Table 9, the model trained with data gener-

(c)

(d)

ated using a DR of 0.75 performs the worst. We attribute this to the fact that, with the first three-quarters of the prefix being identical, the difference in quality between the chosen and rejected responses becomes less apparent, reducing training effectiveness. Additionally, the model trained with a DR of 0.25 performs worse than that trained with a dropout ratio of 0.5. We believe this is because the majority of the content in the rejected responses is generated without image input, resulting in noticeably lower quality compared to the chosen responses, which similarly hampers the training effectiveness. Therefore, we set the DR to 0.5.

Figure 4. Results of models trained with different data scales or hyer-parameters on M3CoT. The X-axis represents the corresponding data scale or hyper-parameter for this point, while the Y-axis indicates the accuracy on M3CoT.

achieving higher performance across all scales. These results highlight the importance of scaling up reasoning preference data to improve model performance.

##### 8.4. Effects of Hyper-parameters.

##### 8.3. Effects of Data Scale.

To evaluate the effects of the data scale, we train the model with different amounts of preference reasoning data sampled from M3CoT [16]. The M3CoT training set contains 7,861 samples annotated with corresponding rationales. To control the data volume, we adjust the maximum number of preference pairs generated for each sample, resulting in datasets of different sizes: 10K, 40K, 70K, and 100K.

As illustrated in Figure 4a, model accuracy consistently improves with the increasing data volume. As the data volume rises to 100K, the model achieves its highest accuracy of 76.4 when directly answering the final answer and 78.9 when answering with CoT. Furthermore, both the Direct and CoT performance exhibit a positive correlation between data scale and accuracy, with the CoT performance

We conduct ablation studies on M3CoT to study the impact of the hyper-parameters, including learning rate, PO coefficient wp,wq, and SFT coefficient wg. For the PO coefficient, we control the sum of wp and wq to equal 1.0 and adjust different proportions. Unless specifically mentioned, we set the learning rate to 5e-6, wp to 0.8, wq to 0.2, and wg to 1. As shown in Figure 4b, the learning rate significantly affects the model’s performance. With a relatively low learning rate of 5e-7, the model shows moderate improvement. As the learning rate increases to 5e-6, the model’s performance improves further, reaching optimal results across the tested learning rates and surpassing the baseline by 19.6 points. However, further increasing the learning rate to 5e-5 causes a drastic performance drop, suggesting that a higher learning rate may lead to overfitting or instability in training. Additionally, the PO coefficient

w/o multimodal CoT 79.5 66.7 26.2 54.6 71.4 88.0 75.5 3.6 w. multimodal CoT 79.2 67.0 25.7 56.2 76.7 88.1 75.4 3.5

- Table 10. Comparison of models trained on the dataset with or without multimodal CoT samples. The model trained on the dataset augmented with multimodal CoT samples achieve superior performance on complex Visual Question Answering (VQA) benchmarks.

w0,w1 and SFT coefficient w2 are crucial. As shown in Figure 4c and 4d, the model achieves optimal performance with wp set to 0.8, wq set to 0.2, and wg set to 1. Notably, when wg is set to 0.01, the performance of the CoT approach is inferior to that of directly answering the final answer, indicating the importance of the SFT Loss during the direct preference optimization.

##### 8.5. Effects of Multimodal Chain-of-Thought.

In this section, we analyze the effects of multimodal CoT. Specifically, we exclude the samples generated by the multimodal CoT methods proposed in Section 4.2 from MMPR. As shown in Table 10, the model trained on the dataset including multimodal CoT samples (i.e., MMPR) achieves superior performance on complex Visual Question Answering (VQA) benchmarks. Specifically, this model outperforms its counterpart by 1.6 points on MMVet and 5.3 points on LLaVA-Bench. We attribute these improvements to the fact that the multimodal CoT methods not only effectively integrate multimodal information into the reasoning process but also enhance data diversity. Furthermore, including background knowledge and visual content at the start of responses improves the quality of the negative responses generated by DropoutNTP, preventing a significant quality gap between positive and negative samples that could reduce training effectiveness.

#### 9. More Data Examples in MMPR

In this section, we provide data examples in MMPR for each task described in Table 1. Specifically, Figure 7a to 7f are examples from data constructed using DropoutNTP, while Figure 7g to 7j are examples from data constructed using correctness-based pipeline. Additionally, the examples for multimodal CoT, which is introduced in Section 4.2, are shown in Figure 7k to 7m.

#### 10. Qualitative Results

In previous sections, we evaluated our model across various benchmarks and observed its strong performance. In this section, we conduct a qualitative comparison between our model and two baselines: the model before MPO (i.e., InternVL2-8B) and the model fine-tuned with DPO. As shown in Figure 5, the model fine-tuned with DPO is more

prone to generating repetitive responses. To further quantify this phenomenon, we counted the number of responses generated by InternVL2-8B-DPO that failed to produce a parsable answer. Our analysis revealed that, despite its reasonable CoT performance, 16.4% of its responses consisted of gibberish or repetitive outputs that could not be parsed into a valid answer. In contrast, for DPO+ and MPO, only 0.4% and 0.3% of the responses were invalid, respectively. Additionally, we present a qualitative comparison of the model before and after MPO in Figure 6. These examples demonstrate that our InternVL2-8B-MPO achieves superior performance in recognizing information from images and reasoning based on this information.

#### 11. Discussion

In this work, we introduce supervised fine-tuning (SFT) loss as the generation loss in Mix Preference Optimization (MPO), which may seem to contradict our original premise that SFT loss induces distribution shifts. However, we emphasize that incorporating generation loss does not violate this premise. While generation loss and SFT loss share a similar form, their objectives are fundamentally different.

SFT training data contains only positive samples, and its goal is to teach the model to mimic high-quality response generation. As analyzed in Section 1, this approach can lead to a distribution shift. In contrast, Preference Optimization (PO) training data includes both positive and negative samples, with the objective of ensuring that the generation probability of positive samples is higher than that of negative samples, thereby correcting the model’s response distribution. However, in practice, we observe that while the relative probability of generating positive samples is indeed higher than that of negative samples, the absolute probability of generating both positive and negative samples decreases simultaneously.

To address this, we introduce generation loss to ensure that while the generation probability of negative samples decreases, the probability of generating positive samples increases, thereby preventing model collapse. Although our generation loss and SFT loss share a similar form, the generation loss primarily serves to correct the distribution, thereby mitigating the distribution shift.

###### M3CoT - Language Science

###### M3CoT - Language Science

###### Question:

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: ..“ Question: What is one reason people can't go to Mars? A. Option A in the image B. Option B in the image C. None of the options are correct.

###### Question:

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: ..“ Question: What is one reason people can't go to Mars? A. Option A in the image B. Option B in the image C. None of the options are correct.

###### InternVL2-8B:

###### InternVL2-8B-MPO:

It takes too long to get there, and it's not an easy place to live. So, scientists sent a robot to look around Mars for them. The robot is named Curiosity. One of its jobs is to find out if anything can live on Mars. Curiosity has many cameras. It takes pictures as it moves around Mars. Curiosity sends the pictures back to the scientists on Earth. Scientists hope that the pictures will make it easier for us to visit Mars someday. People can't make it to Mars yet. It takes too long to get there, and it's not an easy place to live. So, scientists sent a robot to look around Mars for them. The robot is named Curiosity. One of its jobs is to find out if anything can live on Mars. Curiosity has many cameras. It takes pictures as it moves around Mars. Curiosity sends the pictures back to the scientists on Earth. Scientists hope that the pictures will make it easier for us to visit Mars someday. People can't make it to Mars yet. It takes too long to get there, and it's not an easy place to live. So, scientists sent a robot to look around Mars for them. The robot is named Curiosity. ...(Repeat until truncated.)

To determine the correct answer, let's analyze the passage step by step:

###### InternVL2-8B:

###### InternVL2-8B-MPO:

- 1. The passage states that "People can't make it to Mars yet." This indicates that there is a reason why people cannot currently travel to Mars.
- 2. The passage further explains that "It takes too long to get there, and it's not an easy place to live." This suggests that the primary reasons are the long travel time and the challenging living conditions on Mars.
- 3. The passage does not mention anything about the temperature on Mars being too hot. Given this information, the correct answer is: Final answer: A

It takes too long to get there, and it's not an easy place to live. So, scientists sent a robot to look around Mars for them. The robot is named Curiosity. One of its jobs is to find out if anything can live on Mars. Curiosity has many cameras. It takes pictures as it moves around Mars. Curiosity sends the pictures back to the scientists on Earth. Scientists hope that the pictures will make it easier for us to visit Mars someday. People can't make it to Mars yet. It takes too long to get there, and it's not an easy place to live. So, scientists sent a robot to look around Mars for them. The robot is named Curiosity. One of its jobs is to find out if anything can live on Mars. Curiosity has many cameras. It takes pictures as it moves around Mars. Curiosity sends the pictures back to the scientists on Earth. Scientists hope that the pictures will make it easier for us to visit Mars someday. People can't make it to Mars yet. It takes too long to get there, and it's not an easy place to live. So, scientists sent a robot to look around Mars for them. The robot is named Curiosity. ...(Repeat until truncated.)

To determine the correct answer, let's analyze the passage step by step:

- 1. The passage states that "People can't make it to Mars yet." This indicates that there is a reason why people cannot currently travel to Mars.
- 2. The passage further explains that "It takes too long to get there, and it's not an easy place to live." This suggests that the primary reasons are the long travel time and the challenging living conditions on Mars.
- 3. The passage does not mention anything about the temperature on Mars being too hot. Given this information, the correct answer is: Final answer: A

###### M3CoT - Mathematics Algebra

###### M3CoT - Mathematics Algebra

[Figure 5]

###### Question:

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: .." Question: What statement is true about Rebecca Soni's performance in the 200 meter meet at the 2008 summer olympics and the 2009 duel in the pool?

[Figure 6]

###### Question:

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: ..“ Question: What statement is true about Rebecca Soni's performance in the 200 meter meet at the 2008 summer olympics and the 2009 duel in the pool?

- A. She finished at the same time in both meets
- B. She finished earlier in the 2008 summer Olympics
- C. She finished later in the 2008 summer Olympics
- D. There is not enough information to answer the question

- A. She finished at the same time in both meets
- B. She finished earlier in the 2008 summer Olympics
- C. She finished later in the 2008 summer Olympics
- D. There is not enough information to answer the question

###### InternVL2-8B-DPO:

###### InternVL2-8B-MPO:

- Step 1: To determine which statement is true about Rebecca Soni's performance in the 200-meter meet at the 2008 Summer Olympics and the 2009 Duel in the Pool, we need to compare the times she achieved in both events.
- Step 2: From the given table, we can see that in the 2008 Summer Olympics, Rebecca Soni finished the 200-meter breaststroke in 2:20.22, while in the 2009 Duel in the Pool, she finished the same event in 2:14.57.
- Step 3: Comparing the times, we see that Rebecca Soni finished earlier in the 2009 Duel in the Pool (2:14.57) than in the 2008 Summer Olympics (2:20.22).
- Step 4: Therefore, the statement that is true about Rebecca Soni's performance in both events is that she finished later in the 2008 Summer Olympics.
- Step 5: Based on the information provided, we can conclude that the correct answer is C.
- Step 6: Therefore, the final answer is C.
- Step 7: [Final solution: C]\nStep 8: \nStep 9: \nStep 10: \nStep 11: \nStep 12: ...(Repeat until truncated.)

To determine the correct answer, we need to compare the times for the 200-meter breaststroke event in the 2008 Summer Olympics and the 2009 Duel in the Pool for Rebecca Soni.

###### InternVL2-8B:

###### InternVL2-8B-MPO:

- Step 1: To determine which statement is true about Rebecca Soni's performance in the 200-meter meet at the 2008 Summer Olympics and the 2009 Duel in the Pool, we need to compare the times she achieved in both events.
- Step 2: From the given table, we can see that in the 2008 Summer Olympics, Rebecca Soni finished the 200-meter breaststroke in 2:20.22, while in the 2009 Duel in the Pool, she finished the same event in 2:14.57.
- Step 3: Comparing the times, we see that Rebecca Soni finished earlier in the 2009 Duel in the Pool (2:14.57) than in the 2008 Summer Olympics (2:20.22).
- Step 4: Therefore, the statement that is true about Rebecca Soni's performance in both events is that she finished later in the 2008 Summer Olympics.
- Step 5: Based on the information provided, we can conclude that the correct answer is C.
- Step 6: Therefore, the final answer is C.
- Step 7: [Final solution: C]\nStep 8: \nStep 9: \nStep 10: \nStep 11: \nStep 12: ...(Repeat until truncated.)

To determine the correct answer, we need to compare the times for the 200-meter breaststroke event in the 2008 Summer Olympics and the 2009 Duel in the Pool for Rebecca Soni.

- 1. 2008 Summer Olympics:

- - Event: 200 m breaststroke
- - Time: 2:20.22

- 2. 2009 Duel in the Pool:

- 1. 2008 Summer Olympics:

- - Event: 200 m breaststroke
- - Time: 2:20.22

- 2. 2009 Duel in the Pool:

- - Event: 200 m breaststroke (SC)
- - Time: 2:14.57

Comparing the times:

- - 2:20.22 (2008 Summer Olympics) is later than 2:14.57 (2009 Duel in the Pool). Therefore, Rebecca Soni finished later in the 2008 Summer Olympics compared to the 2009 Duel in the Pool. Final answer: C

- - Event: 200 m breaststroke (SC)
- - Time: 2:14.57

Comparing the times:

- - 2:20.22 (2008 Summer Olympics) is later than 2:14.57 (2009 Duel in the Pool). Therefore, Rebecca Soni finished later in the 2008 Summer Olympics compared to the 2009 Duel in the Pool. Final answer: C

(a)

###### M3CoT - Natural Science

###### M3CoT - Natural Science

###### Question:

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: ..“ Question: Compare the average kinetic energies of the particles in each sample. Which sample has the higher temperature? A. sample B B. sample A C. neither; the samples have the same temperature

###### Question:

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: .." Question: Compare the average kinetic energies of the particles in each sample. Which sample has the higher temperature?

|[Figure 7]<br><br>InternVL2-8B:<br><br>Temperature is a measure of the average kinetic energy of the particles in a substance. Kinetic energy is proportional to the mass of the particle and its speed. The mass of each particle in both samples A and B is 44 u, and the average particle speed in sample B is higher (760 m/s) compared to sample A (770 m/s). Step-by-step reasoning:<br><br>1. Temperature is a measure of the average kinetic energy of the particles in a substance.<br>2. Kinetic energy is proportional to the mass of the particle and its speed.<br>3. The mass of each particle in both samples A and B is 44 u.<br>4. Kinetic energy is proportional to the mass of the particle and its speed.<br>5. The mass of each particle in both samples A and B is 44 u.<br>6. Kinetic energy is proportional to the mass of the particle and its speed.<br>7. The mass of each particle in both samples A and B is 44 u.<br>8. Kinetic energy is proportional to the mass of the particle and its speed.<br>9. The mass of each particle in both samples A and B is 44 u.<br><br><br>...(Repeat until truncated.)<br><br>[Figure 8]<br><br>A. sample B B. sample A C. neither; the samples have the same temperature<br><br>InternVL2-8B-DPO:<br><br>Temperature is a measure of the average kinetic energy of the particles in a substance. Kinetic energy is proportional to the mass of the particle and its speed. The mass of each particle in both samples A and B is 44 u, and the average particle speed in sample B is higher (760 m/s) compared to sample A (770 m/s). Step-by-step reasoning:<br><br>1. Temperature is a measure of the average kinetic energy of the particles in a substance.<br>2. Kinetic energy is proportional to the mass of the particle and its speed.<br>3. The mass of each particle in both samples A and B is 44 u.<br>4. Kinetic energy is proportional to the mass of the particle and its speed.<br>5. The mass of each particle in both samples A and B is 44 u.<br>6. Kinetic energy is proportional to the mass of the particle and its speed.<br>7. The mass of each particle in both samples A and B is 44 u.<br>8. Kinetic energy is proportional to the mass of the particle and its speed.<br>9. The mass of each particle in both samples A and B is 44 u.<br><br><br>...(Repeat until truncated.)|InternVL2-8B-MPO:| |
|---|---|---|
| |To compare the average kinetic energies of the particles in each sample, we need to use the formula for kinetic energy:<br><br>𝐾𝐸 =<br><br>1<br>2<br><br><br>𝑚𝑣2<br><br>where 𝑚 is the mass of the particle and 𝑣 is the speed of the particle. Given:<br><br>- Mass of each particle in both samples: 44 u<br>- Average particle speed in Sample A: 770 m/s<br>- Average particle speed in Sample B: 760 m/s Let's calculate the kinetic energy for each sample:<br><br><br>Sample A: 𝐾𝐸𝐴 = 12 × 44 u × (770 m/s)2 Sample B: 𝐾𝐸B = 12 × 44 u × (760 m/s)2 Since the mass of the particles is the same in both samples, the kinetic energy is directly proportional to the square of the speed. Therefore, we can compare the speeds:<br><br>𝐾𝐸𝐴 ∝ (770 m/s)2 𝐾𝐸B ∝ (760 m/s)2 Since 770 m/s is greater than 760 m/s, the kinetic energy of the particles in Sample A will be higher than that in Sample B. Temperature is a measure of the average kinetic energy of the particles in a substance. Therefore, the sample with higher kinetic energy will have a higher temperature. Final answer: B<br><br>InternVL2-8B-MPO:<br><br>To compare the average kinetic energies of the particles in each sample, we need to use the formula for kinetic energy:<br><br>𝐾𝐸 =<br><br>1<br>2<br><br><br>𝑚𝑣2<br><br>where 𝑚 is the mass of the particle and 𝑣 is the speed of the particle. Given:<br><br>- Mass of each particle in both samples: 44 u<br>- Average particle speed in Sample A: 770 m/s<br>- Average particle speed in Sample B: 760 m/s Let's calculate the kinetic energy for each sample:<br><br><br>Sample A: 𝐾𝐸𝐴 = 12 × 44 u × (770 m/s)2 Sample B: 𝐾𝐸B = 12 × 44 u × (760 m/s)2 Since the mass of the particles is the same in both samples, the kinetic energy is directly proportional to the square of the speed. Therefore, we can compare the speeds:<br><br>𝐾𝐸𝐴 ∝ (770 m/s)2 𝐾𝐸B ∝ (760 m/s)2 Since 770 m/s is greater than 760 m/s, the kinetic energy of the particles in Sample A will be higher than that in Sample B. Temperature is a measure of the average kinetic energy of the particles in a substance. Therefore, the sample with higher kinetic energy will have a higher temperature.| |
| | | |
| |Final answer: B| |
| | | |

(b)

###### M3CoT - Language Science

###### Question:

[Figure 9]

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: .." Question: What is one reason people can't go to Mars? A. Option A in the image B. Option B in the image C. None of the options are correct.

###### InternVL2-8B-DPO:

###### InternVL2-8B-MPO:

It takes too long to get there, and it's not an easy place to live. So, scientists sent a robot to look around Mars for them. The robot is named Curiosity. One of its jobs is to find out if anything can live on Mars. Curiosity has many cameras. It takes pictures as it moves around Mars. Curiosity sends the pictures back to the scientists on Earth. Scientists hope that the pictures will make it easier for us to visit Mars someday. People can't make it to Mars yet. It takes too long to get there, and it's not an easy place to live. So, scientists sent a robot to look around Mars for them. The robot is named Curiosity. One of its jobs is to find out if anything can live on Mars. Curiosity has many cameras. It takes pictures as it moves around Mars. Curiosity sends the pictures back to the scientists on Earth. Scientists hope that the pictures will make it easier for us to visit Mars someday. People can't make it to Mars yet. It takes too long to get there, and it's not an easy place to live. So, scientists sent a robot to look around Mars for them. The robot is named Curiosity. ...(Repeat until truncated.)

To determine the correct answer, let's analyze the passage step by step:

- 1. The passage states that "People can't make it to Mars yet." This indicates that there is a reason why people cannot currently travel to Mars.
- 2. The passage further explains that "It takes too long to get there, and it's not an easy place to live." This suggests that the primary reasons are the long travel time and the challenging living conditions on Mars.
- 3. The passage does not mention anything about the temperature on Mars being too hot. Given this information, the correct answer is: Final answer: A

(c)

- Figure 5. Examples of the model fine-tuned with DPO and MPO. Green highlights the standout answers, while red highlights incorrect responses. We observe that the model fine-tuned with DPO is more prone to generating repetitive responses.

4

###### M3CoT - Mathematics Algebra

###### Question:

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: ..“ Question: What statement is true about Rebecca Soni's performance in the 200 meter meet at the 2008 summer olympics and the 2009 duel in the pool?

- A. She finished at the same time in both meets
- B. She finished earlier in the 2008 summer Olympics
- C. She finished later in the 2008 summer Olympics
- D. There is not enough information to answer the question

###### InternVL2-8B:

###### InternVL2-8B-MPO:

- Step 1: To determine which statement is true about Rebecca Soni's performance in the 200-meter meet at the 2008 Summer Olympics and the 2009 Duel in the Pool, we need to compare the times she achieved in both events.
- Step 2: From the given table, we can see that in the 2008 Summer Olympics, Rebecca Soni finished the 200-meter breaststroke in 2:20.22, while in the 2009 Duel in the Pool, she finished the same event in 2:14.57.
- Step 3: Comparing the times, we see that Rebecca Soni finished earlier in the 2009 Duel in the Pool (2:14.57) than in the 2008 Summer Olympics (2:20.22).
- Step 4: Therefore, the statement that is true about Rebecca Soni's performance in both events is that she finished later in the 2008 Summer Olympics.
- Step 5: Based on the information provided, we can conclude that the correct answer is C.
- Step 6: Therefore, the final answer is C.
- Step 7: [Final solution: C]\nStep 8: \nStep 9: \nStep 10: \nStep 11: \nStep 12: ...(Repeat until truncated.)

To determine the correct answer, we need to compare the times for the 200-meter breaststroke event in the 2008 Summer Olympics and the 2009 Duel in the Pool for Rebecca Soni.

- 1. 2008 Summer Olympics:

- - Event: 200 m breaststroke
- - Time: 2:20.22

- 2. 2009 Duel in the Pool:

- - Event: 200 m breaststroke (SC)
- - Time: 2:14.57

Comparing the times:

- - 2:20.22 (2008 Summer Olympics) is later than 2:14.57 (2009 Duel in the Pool). Therefore, Rebecca Soni finished later in the 2008 Summer Olympics compared to the 2009 Duel in the Pool. Final answer: C

###### M3CoT - Natural Science

###### Question:

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: ..“ Question: Compare the average kinetic energies of the particles in each sample. Which sample has the higher temperature? A. sample B B. sample A C. neither; the samples have the same temperature

|<br><br>InternVL2-8B:<br><br>Temperature is a measure of the average kinetic energy of the particles in a substance. Kinetic energy is proportional to the mass of the particle and its speed. The mass of each particle in both samples A and B is 44 u, and the average particle speed in sample B is higher (760 m/s) compared to sample A (770 m/s). Step-by-step reasoning:<br><br>1. Temperature is a measure of the average kinetic energy of the particles in a substance.<br>2. Kinetic energy is proportional to the mass of the particle and its speed.<br>3. The mass of each particle in both samples A and B is 44 u.<br>4. Kinetic energy is proportional to the mass of the particle and its speed.<br>5. The mass of each particle in both samples A and B is 44 u.<br>6. Kinetic energy is proportional to the mass of the particle and its speed.<br>7. The mass of each particle in both samples A and B is 44 u.<br>8. Kinetic energy is proportional to the mass of the particle and its speed.<br>9. The mass of each particle in both samples A and B is 44 u.<br><br><br>...(Repeat until truncated.)|InternVL2-8B-MPO:<br><br>To compare the average kinetic energies of the particles in each sample, we need to use the formula for kinetic energy:<br><br>𝐾𝐸 =<br><br>1<br>2<br><br><br>𝑚𝑣2<br><br>where 𝑚 is the mass of the particle and 𝑣 is the speed of the particle. Given:<br><br>- Mass of each particle in both samples: 44 u<br>- Average particle speed in Sample A: 770 m/s<br>- Average particle speed in Sample B: 760 m/s Let's calculate the kinetic energy for each sample:<br><br><br>Sample A: 𝐾𝐸𝐴 = 12 × 44 u × (770 m/s)2 Sample B: 𝐾𝐸B = 12 × 44 u × (760 m/s)2 Since the mass of the particles is the same in both samples, the kinetic energy is directly proportional to the square of the speed. Therefore, we can compare the speeds:<br><br>𝐾𝐸𝐴 ∝ (770 m/s)2 𝐾𝐸B ∝ (760 m/s)2 Since 770 m/s is greater than 760 m/s, the kinetic energy of the particles in Sample A will be higher than that in Sample B. Temperature is a measure of the average kinetic energy of the particles in a substance. Therefore, the sample with higher kinetic energy will have a higher temperature. Final answer: B| |
|---|---|---|
| | | |

###### MathVista - Geometry Reasoning

###### Question:

Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end. Question: Find 𝑧. Choices: (A) 7 (B) 9 (C) 12 (D) 15

###### InternVL2-8B:

InternVL2-8B-MPO:

To find the value of 𝑧, we can use the fact that the sum of the interior angles of a triangle is 180∘. The triangle ΔRST has angles ∠S, ∠R, and ∠T. Given:

To find the value of 𝑧, we need to use the properties of the triangle ΔRST. The triangle has two sides marked as equal, indicating that ΔRST is an isosceles triangle with 𝑅𝑆 = 𝑅𝑇. Given:

∠S = 90° ∠R = 2z − 15 ∠T = 90° We can set up the equation for the sum of the interior angles of the triangle:

∠R = 2z − 15 RT = 9 Since 𝑅𝑆 = 𝑅𝑇, we can set up the equation:

∠S + ∠R + ∠T = 180° Substitute the given values:

2𝑧 − 15 = 9 Now, solve for 𝑧:

90° + 2z − 15 + 90° = 180° Combine like terms:

- 1. Add 15 to both sides of the equation: 2𝑧 − 15 + 15 = 9 + 15

2𝑧 = 24

- 2. Divide both sides by 2: 2𝑧

180° + 2z − 15 = 180° Simplify the equation:

2z − 15 = 0 Solve for 𝑧:

24 2

=

15 2

2

2𝑧 = 15 𝑧 =

𝑧 = 7.5

𝑧 = 12

Since none of the given options match 7.5, it seems there might be an error in the problem or the given choices. However, based on the given choices, the closest value to 7.5 is 9.

Therefore, the correct option is (C) 12. Final answer: C

Final answer: (B) 9

###### MathVista - Arithmetic Reasoning

###### Question:

Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end. Question: Subtract all rubber balls. Subtract all yellow shiny things. How many objects are left?

[Figure 13]

###### InternVL2-8B:

###### InternVL2-8B-MPO:

To answer the question, let's first identify all the objects in the image and then follow the instructions step by step.

To solve the problem, we need to follow these steps:

- 1. Identify all the rubber balls in the image.
- 2. Identify all the yellow shiny things in the image.
- 3. Subtract the number of rubber balls from the total number of objects.
- 4. Subtract the number of yellow shiny things from the result of step 3. Let's go through each step:

- 1. Identify all objects in the image:

- - Blue cube - Yellow rubber sphere - Blue rubber cylinder - Red cylinder
- - Red cylinder (different from the first red cylinder) - Purple shiny sphere
- - Teal rubber sphere - Teal rubber cube - Yellow cylinder

- 2. Subtract all rubber balls:

- Yellow rubber sphere - Teal rubber sphere

- 3. Subtract all yellow shiny things:

- No yellow shiny objects in the image

- 4. Count the remaining objects:

- 1. Identify all the rubber balls:

- There are two rubber balls: one yellow and one turquoise.

- 2. Identify all the yellow shiny things:

- There is one yellow shiny cylinder.

- 3. Subtract the number of rubber balls from the total number of objects:

- Total objects: 9 - Rubber balls: 2 - Result: 9 - 2 = 7

- 4. Subtract the number of yellow shiny things from the result of step 3:

- Blue cube - Blue rubber cylinder - Red cylinder - Red cylinder - Purple shiny sphere Teal rubber cube - Yellow cylinder There are 7 objects left after removing the rubber balls and the yellow shiny objects. Final answer: 7

- Result after step 3: 7 - Yellow shiny things: 1 - Final result: 7 - 1 = 6 Final answer: 6

###### MathVision - Statistics Reasoning

###### Question:

[Figure 14]

Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end. Question: The pie charts below indicate the percent of students who prefer golf, bowling, or tennis at East Junior High School and West Middle School. The total number of students at East is 2000 and at West, 2500. In the two schools combined, the percent of students who prefer tennis is

Choices: (A) 30% (B) 31% (C) 32% (D) 33% (E) 34%

###### MathVista - Geometry Reasoning

###### Question:

###### InternVL2-8B:

###### InternVL2-8B-MPO:

[Figure 15]

Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end. Question: Find 𝑧. Choices: (A) 7 (B) 9 (C) 12 (D) 15

First, we need to find the number of students who prefer tennis at each school. At East Junior High School:

To determine the combined percentage of students who prefer tennis at East Junior High School and West Middle School, we need to follow these steps:

- - Total students: 2000
- - Percentage of students who prefer tennis: 48%
- - Number of students who prefer tennis: 2000 × 0.48 = 960 At West Middle School:
- - Total students: 2500
- - Percentage of students who prefer tennis: 40%
- - Number of students who prefer tennis: 2500 × 0.40 = 1000 Next, we add the number of students who prefer tennis from both schools:
- - Total number of students who prefer tennis: 960 + 1000 = 1960 Now, we need to find the total number of students in both schools:
- - Total number of students: 2000 + 2500 = 4500 Finally, we calculate the percentage of students who prefer tennis in both schools:
- - Percentage of students who prefer tennis: 1960 ÷ 4500 × 100 ≈ 43.56% Since the closest answer choice to 43.56% is 34%, the correct answer is (E). Final answer: (E)

- 1. Identify the percentages of students who prefer tennis in each school:

- - East JHS: 48% prefer bowling, 22% prefer tennis, and 30% prefer golf.
- - West MS: 40% prefer tennis, 36% prefer bowling, and 24% prefer golf.

- 2. Calculate the total number of students who prefer tennis in each school:

- - East JHS: 22% of 2000 students = 0.22 * 2000 = 440 students.
- - West MS: 40% of 2500 students = 0.40 * 2500 = 1000 students.

- 3. Add the number of students who prefer tennis in both schools:

- Total number of students who prefer tennis = 440 (East JHS) + 1000 (West MS) = 1440 students.

- 4. Calculate the combined percentage of students who prefer tennis:

###### InternVL2-8B:

###### InternVL2-8B-MPO:

To find the value of 𝑧, we can use the fact that the sum of the interior angles of a triangle is 180∘. The triangle ΔRST has angles ∠S, ∠R, and ∠T. Given:

To find the value of 𝑧, we need to use the properties of the triangle ΔRST. The triangle has two sides marked as equal, indicating that ΔRST is an isosceles triangle with 𝑅𝑆 = 𝑅𝑇. Given:

∠S = 90° ∠R = 2z − 15 ∠T = 90° We can set up the equation for the sum of the interior angles of the triangle:

∠R = 2z − 15 RT = 9 Since 𝑅𝑆 = 𝑅𝑇, we can set up the equation:

∠S + ∠R + ∠T = 180° Substitute the given values:

2𝑧 − 15 = 9 Now, solve for 𝑧:

90° + 2z − 15 + 90° = 180° Combine like terms:

- - Total number of students in both schools = 2000 (East JHS) + 2500 (West MS) = 4500 students.
- - Combined percentage of students who prefer tennis = (1440 / 4500) * 100 = 32%. Final answer:(C)

- 1. Add 15 to both sides of the equation: 2𝑧 − 15 + 15 = 9 + 15

2𝑧 = 24

- 2. Divide both sides by 2: 2𝑧

180° + 2z − 15 = 180° Simplify the equation:

2z − 15 = 0 Solve for 𝑧:

24 2

=

15 2

2

2𝑧 = 15 𝑧 =

𝑧 = 7.5

𝑧 = 12

Since none of the given options match 7.5, it seems there might be an error in the problem or the given choices. However, based on the given choices, the closest value to 7.5 is 9.

Therefore, the correct option is (C) 12. Final answer: C

Final answer: (B) 9

(a)

###### MathVision - Metric Geometry

###### MathVista - Arithmetic Reasoning

###### Question:

###### Question:

[Figure 16]

Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end. Question: Three circles with centres A, B, C touch each other in pairs from the outside (see diagram). Their radii are 3,2 and 1. How big is the area of the triangle ABC?

Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end. Question: Subtract all rubber balls. Subtract all yellow shiny things. How many objects are left?

[Figure 17]

###### InternVL2-8B:

###### InternVL2-8B-MPO:

###### InternVL2-8B:

###### InternVL2-8B-MPO:

To find the area of triangle ABC, we can use the formula for the area of a triangle, which is: Area =

To find the area of triangle ABC, we need to follow these steps:

To answer the question, let's first identify all the objects in the image and then follow the instructions step by step.

To solve the problem, we need to follow these steps:

- 1
- 2

- 1. Identify the radii of the circles:

- Circle A has a radius of 3. - Circle B has a radius of 2. - Circle C has a radius of 1.

- 2. Determine the side lengths of triangle ABC:

- - The distance between the centers of circles A and B is the sum of their radii: 3 + 2 = 5.
- - The distance between the centers of circles B and C is the sum of their radii: 2 + 1 = 3.
- - The distance between the centers of circles A and C is the sum of their radii: 3 + 1 = 4.

- 3. Use Heron's formula to find the area of triangle ABC:

- Heron's formula states that the area of a triangle with sides 𝑎, 𝑏, and 𝑐 is given by:

Area = 𝑠(𝑠 − 𝑎)(𝑠 − 𝑏)(𝑠 − 𝑐) where \(s\) is the semi-perimeter of the triangle: 𝑠 = 𝑎 +2𝑏 +𝑐

- 4. Calculate the semi-perimeter 𝑠:

- - The sides of the triangle are 5, 3, and 4.
- - Therefore, the semi-perimeter 𝑠 is: 𝑠 = 5+32 +4 = 122 = 6

- 5. Apply Heron's formula:

- 1. Identify all the rubber balls in the image.
- 2. Identify all the yellow shiny things in the image.
- 3. Subtract the number of rubber balls from the total number of objects.
- 4. Subtract the number of yellow shiny things from the result of step 3. Let's go through each step:

× base × height

- 1. Identify all objects in the image:

- - Blue cube - Yellow rubber sphere - Blue rubber cylinder - Red cylinder
- - Red cylinder (different from the first red cylinder) - Purple shiny sphere
- - Teal rubber sphere - Teal rubber cube - Yellow cylinder

- 2. Subtract all rubber balls:

- Yellow rubber sphere - Teal rubber sphere

- 3. Subtract all yellow shiny things:

- No yellow shiny objects in the image

- 4. Count the remaining objects:

In this case, the base of the triangle is the sum of the radii of the two smaller circles, which is 2 + 1 = 3. The height of the triangle is the radius of the largest circle, which is 3. So, the area of triangle ABC is:

- 1. Identify all the rubber balls:

- There are two rubber balls: one yellow and one turquoise.

- 2. Identify all the yellow shiny things:

- There is one yellow shiny cylinder.

- 3. Subtract the number of rubber balls from the total number of objects:

- Total objects: 9 - Rubber balls: 2 - Result: 9 - 2 = 7

- 4. Subtract the number of yellow shiny things from the result of step 3:

- 1
- 2

× 3 × 3 Area =

Area =

9 2

= 4.5

Therefore, the area of triangle ABC is 4.5 square units. Final answer: 4.5

- Blue cube - Blue rubber cylinder - Red cylinder - Red cylinder - Purple shiny sphere Teal rubber cube - Yellow cylinder There are 7 objects left after removing the rubber balls and the yellow shiny objects. Final answer: 7

- Result after step 3: 7 - Yellow shiny things: 1 - Final result: 7 - 1 = 6 Final answer: 6

- The area of triangle ABC is:

Area = 6(6 − 5)(6 − 3)(6 − 4) = 6 ∙ 1 ∙ 3 ∙ 2 = 36 = 6 Thus, the area of triangle ABC is 6 square units. Final answer: 6

(b)

###### MathVision - Statistics Reasoning

###### Question:

[Figure 18]

- Figure 6. Examples of the model before and after MPO. Green highlights the standout answers, while red highlights incorrect responses. The model fine-tuned with MPO exhibits superior recognition and reasoning abilities.

Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end. Question: The pie charts below indicate the percent of students who prefer golf, bowling, or tennis at East Junior High School and West Middle School. The total number of students at East is 2000 and at West, 2500. In the two schools combined, the percent of students who prefer tennis is

Choices: (A) 30% (B) 31% (C) 32% (D) 33% (E) 34%

###### InternVL2-8B:

###### InternVL2-8B-MPO:

First, we need to find the number of students who prefer tennis at each school. At East Junior High School:

To determine the combined percentage of students who prefer tennis at East Junior High School and West Middle School, we need to follow these steps:

- - Total students: 2000
- - Percentage of students who prefer tennis: 48%
- - Number of students who prefer tennis: 2000 × 0.48 = 960 At West Middle School:
- - Total students: 2500
- - Percentage of students who prefer tennis: 40%
- - Number of students who prefer tennis: 2500 × 0.40 = 1000 Next, we add the number of students who prefer tennis from both schools:
- - Total number of students who prefer tennis: 960 + 1000 = 1960 Now, we need to find the total number of students in both schools:
- - Total number of students: 2000 + 2500 = 4500 Finally, we calculate the percentage of students who prefer tennis in both schools:
- - Percentage of students who prefer tennis: 1960 ÷ 4500 × 100 ≈ 43.56% Since the closest answer choice to 43.56% is 34%, the correct answer is (E). Final answer: (E)

- 1. Identify the percentages of students who prefer tennis in each school:

- - East JHS: 48% prefer bowling, 22% prefer tennis, and 30% prefer golf.
- - West MS: 40% prefer tennis, 36% prefer bowling, and 24% prefer golf.

- 2. Calculate the total number of students who prefer tennis in each school:

- - East JHS: 22% of 2000 students = 0.22 * 2000 = 440 students.
- - West MS: 40% of 2500 students = 0.40 * 2500 = 1000 students.

- 3. Add the number of students who prefer tennis in both schools:

- Total number of students who prefer tennis = 440 (East JHS) + 1000 (West MS) = 1440 students.

- 4. Calculate the combined percentage of students who prefer tennis:

- - Total number of students in both schools = 2000 (East JHS) + 2500 (West MS) = 4500 students.
- - Combined percentage of students who prefer tennis = (1440 / 4500) * 100 = 32%. Final answer:(C)

5

###### MathVision - Metric Geometry

###### Question:

[Figure 19]

Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end. Question: When does the expenditure per student in Jamaica have the greatest increase?

###### InternVL2-8B:

###### InternVL2-8B-MPO:

To find the area of triangle ABC, we can use the formula for the area of a triangle, which is: Area =

To find the area of triangle ABC, we need to follow these steps:

- 1
- 2

- 1. Identify the radii of the circles:

- Circle A has a radius of 3. - Circle B has a radius of 2. - Circle C has a radius of 1.

- 2. Determine the side lengths of triangle ABC:

- - The distance between the centers of circles A and B is the sum of their radii: 3 + 2 = 5.
- - The distance between the centers of circles B and C is the sum of their radii: 2 + 1 = 3.
- - The distance between the centers of circles A and C is the sum of their radii: 3 + 1 = 4.

- 3. Use Heron's formula to find the area of triangle ABC:

- Heron's formula states that the area of a triangle with sides 𝑎, 𝑏, and 𝑐 is given by:

Area = 𝑠(𝑠 − 𝑎)(𝑠 − 𝑏)(𝑠 − 𝑐) where \(s\) is the semi-perimeter of the triangle: 𝑠 = 𝑎 +2𝑏 +𝑐

- 4. Calculate the semi-perimeter 𝑠:

- - The sides of the triangle are 5, 3, and 4.
- - Therefore, the semi-perimeter 𝑠 is: 𝑠 = 5+32 +4 = 122 = 6

- 5. Apply Heron's formula:

× base × height

In this case, the base of the triangle is the sum of the radii of the two smaller circles, which is 2 + 1 = 3. The height of the triangle is the radius of the largest circle, which is 3. So, the area of triangle ABC is:

- 1
- 2

× 3 × 3 Area =

Area =

9 2

= 4.5

Therefore, the area of triangle ABC is 4.5 square units. Final answer: 4.5

- The area of triangle ABC is:

Area = 6(6 − 5)(6 − 3)(6 − 4) = 6 ∙ 1 ∙ 3 ∙ 2 = 36 = 6 Thus, the area of triangle ABC is 6 square units. Final answer: 6

###### General VQA

Question: What are the toppings on the pizza?

Question: What are the toppings on the

Question: What are the toppings on the

Question: What are the toppings on the pizza?

Question: What are the toppings on the pizza?

[Figure 20]

[Figure 21]

[Figure 22]

###### Chosen Response:

Rejected Response:

[Figure 23]

###### Chosen Response:

Rejected Response:

###### Chosen Response:

Rejected Response:

###### Chosen Response:

Rejected Response:

[Figure 24]

The pizza in the image has the following toppings:

The pizza in the image has the following toppings:

The pizza in the image has the following toppings:

The pizza in the image has the following toppings:

The pizza in the image has the following toppings:

The pizza in the image has the following toppings:

###### Chosen Response:

###### Rejected Response:

The pizza in the image has the following toppings:

The pizza in the image has the following toppings:

- 1. Tomatoes: There are halved cherry tomatoes scattered across the pizza, adding a fresh and juicy element.
- 2. Artichokes: Pieces of artichoke hearts are visible, providing a tender and slightly tangy flavor.
- 3. Cheese: A layer of melted cheese covers the pizza, likely mozzarella or a mozzarella blend, providing richness and an anchor for the other toppings.
- 4. Peppers: Slices of green bell pepper contribute both color and crunch.
- 5. Olives: Kalamata olives are scattered across the pizza, adding a briny and slightly salty taste.
- 6. Onion: Thinly sliced onions contribute a sweet and pungent flavor. These toppings create a delicious balance of flavors and textures, making the pizza a crowdpleaser!

- 1. Tomatoes: There are halved cherry tomatoes scattered across the pizza, adding a fresh and juicy element.
- 2. Artichokes: Pieces of artichoke hearts are visible, providing a tender and slightly tangy flavor.
- 3. Cheese: A layer of melted cheese covers the pizza, likely mozzarella or a similar type, providing a creamy and rich texture.
- 4. Pesto or Herb Sauce: There appears to be a green sauce, possibly pesto, spread beneath the toppings, adding a herby and savory flavor. The pizza has a rustic appearance with a slightly charred crust, suggesting it was baked at a high temperature.

- 1. Tomatoes: There are halved cherry tomatoes scattered across the pizza, adding a fresh and juicy element.
- 2. Artichokes: Pieces of artichoke hearts are visible, providing a tender and slightly tangy flavor.
- 3. Cheese: A layer of melted cheese covers the pizza, likely mozzarella or a mozzarella providing and an for the other toppings.
- 4. : of green contribute both color and .
- 5. : are scattered across the pizza, adding a and slightly .
- 6. : contribute a and flavor. These toppings a of and the pizza a -

- 1. Tomatoes: There are halved cherry tomatoes scattered across the pizza, adding a fresh and juicy element.
- 2. Artichokes: Pieces of artichoke hearts are visible, providing a tender and slightly tangy flavor.
- 3. Cheese: A layer of melted cheese covers the pizza, likely mozzarella or a similar providing a and .
- 4. or : There to a green the adding a and flavor. The pizza has a with a slightly at a high .

- 1. Tomatoes: There are halved cherry tomatoes scattered across the pizza, adding a fresh and juicy element.
- 2. Artichokes: Pieces of artichoke hearts are visible, providing a tender and slightly tangy flavor.
- 3. Cheese: A layer of melted cheese covers the pizza, likely mozzarella or a mozzarella providing and an for the other toppings.
- 4. : of green contribute both color and .
- 5. : are scattered across the pizza, adding a and slightly .
- 6. : contribute a and flavor. These toppings a of and the pizza a -

- 1. Tomatoes: There are halved cherry tomatoes scattered across the pizza, adding a fresh and juicy element.
- 2. Artichokes: Pieces of artichoke hearts are visible, providing a tender and slightly tangy flavor.
- 3. Cheese: A layer of melted cheese covers the pizza, likely mozzarella or a similar providing a and .
- 4. or : There to a green the adding a and flavor. The pizza has a with a slightly at a high .

The pizza in the image has the following toppings:

The pizza in the image has the following toppings:

- 1. Tomatoes: There are halved cherry tomatoes scattered across the pizza, adding a fresh and juicy element.
- 2. Artichokes: Pieces of artichoke hearts are visible, providing a tender and slightly tangy flavor.
- 3. Cheese: A layer of melted cheese covers the pizza, likely mozzarella or a mozzarella blend, providing richness and an anchor for the other toppings.
- 4. Peppers: Slices of green bell pepper contribute both color and crunch.
- 5. Olives: Kalamata olives are scattered across the pizza, adding a briny and slightly salty taste.
- 6. Onion: Thinly sliced onions contribute a sweet and pungent flavor. These toppings create a delicious balance of flavors and textures, making the pizza a crowdpleaser!

- 1. Tomatoes: There are halved cherry tomatoes scattered across the pizza, adding a fresh and juicy element.
- 2. Artichokes: Pieces of artichoke hearts are visible, providing a tender and slightly tangy flavor.
- 3. Cheese: A layer of melted cheese covers the pizza, likely mozzarella or a similar type, providing a creamy and rich texture.
- 4. Pesto or Herb Sauce: There appears to be a green sauce, possibly pesto, spread beneath the toppings, adding a herby and savory flavor. The pizza has a rustic appearance with a slightly charred crust, suggesting it was baked at a high temperature.

- 1. Tomatoes: There are halved cherry tomatoes scattered across the pizza, adding a fresh and juicy element.
- 2. Artichokes: Pieces of artichoke hearts are visible, providing a tender and slightly tangy flavor.
- 3. Cheese: A layer of melted cheese covers the pizza, likely mozzarella or a mozzarella blend, providing richness and an anchor for the other toppings.
- 4. Peppers: Slices of green bell pepper contribute both color and crunch.
- 5. Olives: Kalamata olives are scattered across the pizza, adding a briny and slightly salty taste.
- 6. Onion: Thinly sliced onions contribute a sweet and pungent flavor. These toppings create a delicious balance of flavors and textures, making the pizza a crowdpleaser!

- 1. Tomatoes: There are halved cherry tomatoes scattered across the pizza, adding a fresh and juicy element.
- 2. Artichokes: Pieces of artichoke hearts are visible, providing a tender and slightly tangy flavor.
- 3. Cheese: A layer of melted cheese covers the pizza, likely mozzarella or a similar type, providing a creamy and rich texture.
- 4. Pesto or Herb Sauce: There appears to be a green sauce, possibly pesto, spread beneath the toppings, adding a herby and savory flavor. The pizza has a rustic appearance with a slightly charred crust, suggesting it was baked at a high temperature.

(a)

###### Science

###### Question:

###### Question:

###### Question:

###### Question:

Two magnets are placed as shown. Hint: Magnets that attract pull together. Magnets that repel push apart. Will these magnets attract or repel each other? A. repel B. attract

magnets are as . : Magnets that attract . Magnets that repel . magnets attract or repel each A. repel B. attract

magnets are as . : Magnets that attract . Magnets that repel . magnets attract or repel each A. repel B. attract

###### Question:

Two magnets are placed as shown. Hint: Magnets that attract pull together. Magnets that repel push apart. Will these magnets attract or repel each other? A. attract B. repel

Two magnets are placed as shown. Hint: Magnets that attract pull together. Magnets that repel push apart. Will these magnets attract or repel each other? A. repel B. attract

###### Chosen Response:

###### Rejected Response:

###### Chosen Response:

###### Rejected Response:

###### Chosen Response:

###### Rejected Response:

###### Chosen Response:

###### Rejected Response:

In the image, there are two magnets. Each magnet has two poles labeled as "N" for North and "S" for South. Magnetic Interaction:

In the image, there are two magnets. Each magnet has two poles labeled as "N" for North and "S" for South. Magnetic Interaction:

In the image, there are two magnets. Each magnet has two poles labeled as "N" for North and "S" for South. Magnetic Interaction:

In the image, there are two magnets. Each magnet has two poles labeled as "N" for North and "S" for South. Magnetic Interaction:

In the image, there are two magnets. Each magnet has two poles labeled as "N" for North and "S" for South. Magnetic Interaction:

In the image, there are two magnets. Each magnet has two poles labeled as "N" for North and "S" for South. Magnetic Interaction:

###### Chosen Response:

###### Rejected Response:

In the image, there are two magnets. Each magnet has two poles labeled as "N" for North and "S" for South. Magnetic Interaction:

In the image, there are two magnets. Each magnet has two poles labeled as "N" for North and "S" for South. Magnetic Interaction:

In the image, there are two magnets. Each magnet has two poles labeled as "N" for North and "S" for South. Magnetic Interaction:

In the image, there are two magnets. Each magnet has two poles labeled as "N" for North and "S" for South. Magnetic Interaction:

- - Like Poles Repel: If similar poles (North-North or South-South) face each other, they repel.
- - Unlike Poles Attract: If opposite poles (North-South) face each other, they attract. Analysis of the Image:

- - Like Poles Repel: If similar poles (North-North or South-South) face each other, they repel.
- - Unlike Poles Attract: If opposite poles (North-South) face each other, they attract. Analysis of the Image:

[Figure 25]

- - Like Poles Repel: If similar poles (North-North or South-South) face each other, they repel.
- - Unlike Poles Attract: If opposite poles (North-South) face each other, they attract. Analysis of the Image:

- - Like Poles Repel: If similar poles (North-North or South-South) face each other, they repel.
- - Unlike Poles Attract: If opposite poles (North-South) face each other, they attract. Analysis of the Image:

[Figure 26]

- - Like Poles Repel: If similar poles (North-North or South-South) face each other, they repel.
- - Unlike Poles Attract: If opposite poles (North-South) face each other, they attract. Analysis of the Image:

- - Like Poles Repel: If similar poles (North-North or South-South) face each other, they repel.
- - Unlike Poles Attract: If opposite poles (North-South) face each other, they attract. Analysis of the Image:

[Figure 27]

- - Like Poles Repel: If similar poles (North-North or South-South) face each other, they repel.
- - Unlike Poles Attract: If opposite poles (North-South) face each other, they attract. Analysis of the Image:

- - Like Poles Repel: If similar poles (North-North or South-South) face each other, they repel.
- - Unlike Poles Attract: If opposite poles (North-South) face each other, they attract. Analysis of the Image:

[Figure 28]

- - Like Poles Repel: If similar poles (North-North or South-South) face each other, they repel.
- - Unlike Poles Attract: If opposite poles (North-South) face each other, they attract. Analysis of the Image:

- - Like Poles Repel: If similar poles (North-North or South-South) face each other, they repel.
- - Unlike Poles Attract: If opposite poles (North-South) face each other, they attract. Analysis of the Image:

[Figure 29]

- 1. Left Magnet: The North pole is on the left, and the South pole is on the right.
- 2. Right Magnet: The North pole is on the right, and the South pole is on the left. These arrangements show that the North poles are on opposite ends of the magnets, and so are the South poles. Conclusion: Since a North pole on one magnet faces a North pole on the other, and a South pole on one magnet faces a South pole on the other: The magnets will repel each other because the North poles repel and the South poles repel. Answer: B. repel

- 1. Left Magnet: The North pole is on the left, and the South pole is on the right.
- 2. Right Magnet: The North pole is on the left, and the South pole is on the right. Orientation:

- 1. Left Magnet: The North pole is on the left, and the South pole is on the right.
- 2. Right Magnet: The North pole is on the and the South pole is on the left. These that the North poles are on opposite of the and are the South poles. Conclusion: Since a North pole on one magnet faces a North pole on the other, and a South pole on one magnet faces a South pole on the other: The magnets will repel each other

- 1. Left Magnet: The North pole is on the left, and the South pole is on the right.
- 2. Right Magnet: The North pole is on the left, and the South pole is on the right. :

- 1. Left Magnet: The North pole is on the left, and the South pole is on the right.
- 2. Right Magnet: The North pole is on the and the South pole is on the left. These that the North poles are on opposite of the and are the South poles. Conclusion: Since a North pole on one magnet faces a North pole on the other, and a South pole on one magnet faces a South pole on the other: The magnets will repel each other

- 1. Left Magnet: The North pole is on the left, and the South pole is on the right.
- 2. Right Magnet: The North pole is on the left, and the South pole is on the right. :

- 1. Left Magnet: The North pole is on the left, and the South pole is on the right.
- 2. Right Magnet: The North pole is on the right, and the South pole is on the left. These arrangements show that the North poles are on opposite ends of the magnets, and so are the South poles. Conclusion: Since a North pole on one magnet faces a North pole on the other, and a South pole on one magnet faces a South pole on the other: The magnets will repel each other because the North poles repel and the South poles repel. Answer: B. repel

- 1. Left Magnet: The North pole is on the left, and the South pole is on the right.
- 2. Right Magnet: The North pole is on the left, and the South pole is on the right. Orientation:

- 1. Left Magnet: The North pole is on the left, and the South pole is on the right.
- 2. Right Magnet: The North pole is on the right, and the South pole is on the left. These arrangements show that the North poles are on opposite ends of the magnets, and so are the South poles. Conclusion: Since a North pole on one magnet faces a North pole on the other, and a South pole on one magnet faces a South pole on the other: The magnets will repel each other because the North poles repel and the South poles repel. Answer: B. repel

- 1. Left Magnet: The North pole is on the left, and the South pole is on the right.
- 2. Right Magnet: The North pole is on the left, and the South pole is on the right. Orientation:

- - The North pole of the left magnet is facing the North pole of the right magnet.
- - The South pole of the left magnet is facing the South pole of the right magnet. Conclusion: Since both pairs of like poles (North-North and South-South) are facing each other, the magnets will repel each other. Therefore, the correct answer is: B. repel

- - The North pole of the left magnet is facing the North pole of the right magnet.
- - The South pole of the left magnet is facing the South pole of the right magnet. Conclusion: Since both of poles (North-North and South-South) are facing each other, the magnets will repel each other. Therefore, the is: B. repel

- - The North pole of the left magnet is facing the North pole of the right magnet.
- - The South pole of the left magnet is facing the South pole of the right magnet. Conclusion: Since both of poles (North-North and South-South) are facing each other, the magnets will repel each other. Therefore, the is: B. repel

- - The North pole of the left magnet is facing the North pole of the right magnet.
- - The South pole of the left magnet is facing the South pole of the right magnet. Conclusion: Since both pairs of like poles (North-North and South-South) are facing each other, the magnets will repel each other. Therefore, the correct answer is: B. repel

- - The North pole of the left magnet is facing the North pole of the right magnet.
- - The South pole of the left magnet is facing the South pole of the right magnet. Conclusion: Since both pairs of like poles (North-North and South-South) are facing each other, the magnets will repel each other. Therefore, the correct answer is: B. repel

the North poles repel and the South poles repel. : B. repel

the North poles repel and the South poles repel. : B. repel

(b)

###### Chart

Question: How many color does the graph shows?

Question: How many color the graph

Question: How many color the graph

Question: How many color does the graph shows?

Question: How many color does the graph shows?

[Figure 30]

[Figure 31]

[Figure 32]

###### Chosen Response:

###### Rejected Response:

###### Chosen Response:

###### Rejected Response:

###### Chosen Response:

###### Rejected Response:

[Figure 33]

###### Chosen Response:

###### Rejected Response:

[Figure 34]

###### The graph uses five distinct colors, each representing a different country. Here's a detailed breakdown:

###### The graph uses five distinct colors, each representing a different country. Here's a detailed breakdown:

###### The graph uses five distinct colors, each representing a different country. Here's a detailed breakdown:

###### The graph uses five distinct colors, each representing a different country. Here's a detailed breakdown:

###### The graph uses five distinct colors, each representing a different country. Here's a detailed breakdown:

###### The graph uses five distinct colors, each representing a different country. Here's a detailed breakdown:

###### Chosen Response:

###### Rejected Response:

###### The graph uses five distinct colors, each representing a different country. Here's a detailed breakdown:

###### The graph uses five distinct colors, each representing a different country. Here's a detailed breakdown:

###### The graph uses five distinct colors, each representing a different country. Here's a detailed breakdown:

###### The graph uses five distinct colors, each representing a different country. Here's a detailed breakdown:

- 1. Blue - Represents Brazil, with an internet access percentage of 49.2%.
- 2. Purple - Represents Latvia, with an internet access percentage of 76%.
- 3. Green - Represents Austria, with an internet access percentage of 82.4%.
- 4. Red - Represents France, with an internet access percentage of 82.6%.
- 5. Orange - Represents Norway, with an internet access percentage of 96.6%. Each color is used to differentiate the countries and their corresponding internet access percentages for 2015.

- 1. Blue - Represents Brazil, with an internet access percentage of 49.2%.
- 2. Purple - Represents Latvia, with an internet access percentage of 76%.
- 3. Green - Represents Austria, with an internet access percentage of 59.9%.
- 4. Orange - Represents Mexico, with an internet access percentage of 41.5%.
- 5. Red - Represents New Zealand, with an internet access percentage of 85%. Each country is distinctly colored and plotted at different points on the graph, which allows us to differentiate between the various countries and their corresponding internet access percentages.

- 1. Blue - Represents Brazil, with an internet access percentage of 49.2%.
- 2. Purple - Represents Latvia, with an internet access percentage of 76%.
- 3. Green - Represents Austria, with an internet access percentage of 82.4%.
- 4. Red - Represents with an internet access percentage of 82.6%.
- 5. Orange - Represents with an internet access percentage of .6%. Each color is to differentiate the countries and their corresponding internet access percentages for .

- 1. Blue - Represents Brazil, with an internet access percentage of 49.2%.
- 2. Purple - Represents Latvia, with an internet access percentage of 76%.
- 3. Green - Represents Austria, with an internet access percentage of . %.
- 4. Orange - Represents with an internet access percentage of .5%.
- 5. Red - Represents with an internet access percentage of %. Each country is and at different on the which us to differentiate the countries and their corresponding internet access percentages.

- 1. Blue - Represents Brazil, with an internet access percentage of 49.2%.
- 2. Purple - Represents Latvia, with an internet access percentage of 76%.
- 3. Green - Represents Austria, with an internet access percentage of 82.4%.
- 4. Red - Represents with an internet access percentage of 82.6%.
- 5. Orange - Represents with an internet access percentage of .6%. Each color is to differentiate the countries and their corresponding internet access percentages for .

- 1. Blue - Represents Brazil, with an internet access percentage of 49.2%.
- 2. Purple - Represents Latvia, with an internet access percentage of 76%.
- 3. Green - Represents Austria, with an internet access percentage of . %.
- 4. Orange - Represents with an internet access percentage of .5%.
- 5. Red - Represents with an internet access percentage of %. Each country is and at different on the which us to differentiate the countries and their corresponding internet access percentages.

- 1. Blue - Represents Brazil, with an internet access percentage of 49.2%.
- 2. Purple - Represents Latvia, with an internet access percentage of 76%.
- 3. Green - Represents Austria, with an internet access percentage of 82.4%.
- 4. Red - Represents France, with an internet access percentage of 82.6%.
- 5. Orange - Represents Norway, with an internet access percentage of 96.6%. Each color is used to differentiate the countries and their corresponding internet access percentages for 2015.

- 1. Blue - Represents Brazil, with an internet access percentage of 49.2%.
- 2. Purple - Represents Latvia, with an internet access percentage of 76%.
- 3. Green - Represents Austria, with an internet access percentage of 59.9%.
- 4. Orange - Represents Mexico, with an internet access percentage of 41.5%.
- 5. Red - Represents New Zealand, with an internet access percentage of 85%. Each country is distinctly colored and plotted at different points on the graph, which allows us to differentiate between the various countries and their corresponding internet access percentages.

- 1. Blue - Represents Brazil, with an internet access percentage of 49.2%.
- 2. Purple - Represents Latvia, with an internet access percentage of 76%.
- 3. Green - Represents Austria, with an internet access percentage of 82.4%.
- 4. Red - Represents France, with an internet access percentage of 82.6%.
- 5. Orange - Represents Norway, with an internet access percentage of 96.6%. Each color is used to differentiate the countries and their corresponding internet access percentages for 2015.

- 1. Blue - Represents Brazil, with an internet access percentage of 49.2%.
- 2. Purple - Represents Latvia, with an internet access percentage of 76%.
- 3. Green - Represents Austria, with an internet access percentage of 59.9%.
- 4. Orange - Represents Mexico, with an internet access percentage of 41.5%.
- 5. Red - Represents New Zealand, with an internet access percentage of 85%. Each country is distinctly colored and plotted at different points on the graph, which allows us to differentiate between the various countries and their corresponding internet access percentages.

(c)

###### Mathematics

Question: Subtract all balls. How many objects are left?

Question: Subtract all balls. How many objects are

Question: Subtract all balls. How many objects are

Question: Subtract all balls. How many objects are left?

Question: Subtract all balls. How many objects are left?

###### Chosen Response:

###### Rejected Response:

###### Chosen Response:

###### Rejected Response:

###### Chosen Response:

###### Rejected Response:

###### Chosen Response:

###### Rejected Response:

[Figure 35]

To determine how many objects are left after subtracting all the balls, let's examine the image and identify each object:

To determine how many objects are left after subtracting all the balls, let's examine the image and identify each object:

[Figure 36]

[Figure 37]

To determine how many objects are left after subtracting all the balls, let's examine the image and identify each object:

To determine how many objects are left after subtracting all the balls, let's examine the image and identify each object:

To determine how many objects are left after subtracting all the balls, let's examine the image and identify each object:

To determine how many objects are left after subtracting all the balls, let's examine the image and identify each object:

###### Chosen Response:

###### Rejected Response:

[Figure 38]

To determine how many objects are left after subtracting all the balls, let's examine the image and identify each object:

To determine how many objects are left after subtracting all the balls, let's examine the image and identify each object:

[Figure 39]

To determine how many objects are left after subtracting all the balls, let's examine the image and identify each object:

To determine how many objects are left after subtracting all the balls, let's examine the image and identify each object:

- 1. Balls: - There are three spherical objects (balls): - A small grey ball on the far left.

- A shiny metallic ball next to the small grey ball. - A medium-sized blue ball near the center.

- 2. Other Objects: - A purple cube, which is a square-shaped object. - A large blue cylinder with a red top and base, which is a cylindrical object. - A green rectangular prism, which is a rectangular-shaped object. Step-by-step Calculation:

- 1. Balls: - There are three spherical objects (balls): - A small grey ball on the far left.

- A shiny metallic ball next to the small grey ball. - A medium-sized blue ball near the center.

- 2. Other Objects: - A purple cube, which is a square-shaped object. - A large blue cylinder in the foreground. - A small orange cylinder behind the purple cube. - A small purple cylinder next to the large blue cylinder. - A medium-sized blue cylinder in the background. Total Objects: - Total objects initially present = 8 (3 balls + 5 non-ball objects). Subtracting the Balls: - Removing the 3 balls leaves us with the non-ball objects. Remaining Objects: - 1 purple cube - 3 cylinders (1 large blue, 1 small orange, 1 small purple, and 1 medium blue). Therefore, after subtracting all the balls, there are 5 objects left.

- 1. Balls: - There are three spherical objects (balls): - A small grey ball on the far left.

- A shiny metallic ball next to the small grey ball. - A medium-sized blue ball near the center.

- 2. Other Objects: - A purple cube, which is a square-shaped object. - A large blue cylinder with a and which is a object. - A green rectangular which is a rectangular-shaped object.

- 1. Balls: - There are three spherical objects (balls): - A small grey ball on the far left.

- A shiny metallic ball next to the small grey ball. - A medium-sized blue ball near the center.

- 2. Other Objects: - A purple cube, which is a square-shaped object. - A large blue cylinder in the . - A small cylinder the purple cube. - A small purple cylinder next to the large blue cylinder. - A medium-sized blue cylinder in the . Total Objects: - Total objects = 8 (3 balls 5 non-ball .

- 1. Balls: - There are three spherical objects (balls): - A small grey ball on the far left.

- A shiny metallic ball next to the small grey ball. - A medium-sized blue ball near the center.

- 2. Other Objects: - A purple cube, which is a square-shaped object. - A large blue cylinder with a and which is a object. - A green rectangular which is a rectangular-shaped object.

- 1. Balls: - There are three spherical objects (balls): - A small grey ball on the far left.

- A shiny metallic ball next to the small grey ball. - A medium-sized blue ball near the center.

- 2. Other Objects: - A purple cube, which is a square-shaped object. - A large blue cylinder in the . - A small cylinder the purple cube. - A small purple cylinder next to the large blue cylinder. - A medium-sized blue cylinder in the . Total Objects: - Total objects = 8 (3 balls 5 non-ball .

- 1. Balls: - There are three spherical objects (balls): - A small grey ball on the far left.

- A shiny metallic ball next to the small grey ball. - A medium-sized blue ball near the center.

- 2. Other Objects: - A purple cube, which is a square-shaped object. - A large blue cylinder with a red top and base, which is a cylindrical object. - A green rectangular prism, which is a rectangular-shaped object. Step-by-step Calculation:

- 1. Balls: - There are three spherical objects (balls): - A small grey ball on the far left.

- A shiny metallic ball next to the small grey ball. - A medium-sized blue ball near the center.

- 2. Other Objects: - A purple cube, which is a square-shaped object. - A large blue cylinder in the foreground. - A small orange cylinder behind the purple cube. - A small purple cylinder next to the large blue cylinder. - A medium-sized blue cylinder in the background. Total Objects: - Total objects initially present = 8 (3 balls + 5 non-ball objects). Subtracting the Balls: - Removing the 3 balls leaves us with the non-ball objects. Remaining Objects: - 1 purple cube - 3 cylinders (1 large blue, 1 small orange, 1 small purple, and 1 medium blue). Therefore, after subtracting all the balls, there are 5 objects left.

- 1. Balls: - There are three spherical objects (balls): - A small grey ball on the far left.

- A shiny metallic ball next to the small grey ball. - A medium-sized blue ball near the center.

- 2. Other Objects: - A purple cube, which is a square-shaped object. - A large blue cylinder with a red top and base, which is a cylindrical object. - A green rectangular prism, which is a rectangular-shaped object. Step-by-step Calculation:

- 1. Balls: - There are three spherical objects (balls): - A small grey ball on the far left.

- A shiny metallic ball next to the small grey ball. - A medium-sized blue ball near the center.

- 2. Other Objects: - A purple cube, which is a square-shaped object. - A large blue cylinder in the foreground. - A small orange cylinder behind the purple cube. - A small purple cylinder next to the large blue cylinder. - A medium-sized blue cylinder in the background. Total Objects: - Total objects initially present = 8 (3 balls + 5 non-ball objects). Subtracting the Balls: - Removing the 3 balls leaves us with the non-ball objects. Remaining Objects: - 1 purple cube - 3 cylinders (1 large blue, 1 small orange, 1 small purple, and 1 medium blue). Therefore, after subtracting all the balls, there are 5 objects left.

-by- :

-by- :

- 1. Count the total number of objects: There are five distinct objects in total.
- 2. Subtract the number of balls: There are three balls in total.
- 3. Calculate the remaining objects: - Total objects: 5 - Subtract the number of balls: 3 Remaining objects: 5 - 3 = 2 Thus, after subtracting all the balls, there are two objects left.

- 1. the total number of objects: There are five distinct objects in total.
- 2. Subtract the number of balls: There are three balls in total.
- 3. the objects: - Total objects: 5 - Subtract the number of balls: 3 Remaining objects: 5 - 3 = 2

the Balls: - the 3 balls us with the non-ball objects. Remaining Objects: - 1 purple cube - 3 (1 large 1 small 1 small

- 1. the total number of objects: There are five distinct objects in total.
- 2. Subtract the number of balls: There are three balls in total.
- 3. the objects: - Total objects: 5 - Subtract the number of balls: 3 Remaining objects: 5 - 3 = 2

the Balls: - the 3 balls us with the non-ball objects. Remaining Objects: - 1 purple cube - 3 (1 large 1 small 1 small

- 1. Count the total number of objects: There are five distinct objects in total.
- 2. Subtract the number of balls: There are three balls in total.
- 3. Calculate the remaining objects: - Total objects: 5 - Subtract the number of balls: 3 Remaining objects: 5 - 3 = 2 Thus, after subtracting all the balls, there are two objects left.

- 1. Count the total number of objects: There are five distinct objects in total.
- 2. Subtract the number of balls: There are three balls in total.
- 3. Calculate the remaining objects: - Total objects: 5 - Subtract the number of balls: 3 Remaining objects: 5 - 3 = 2 Thus, after subtracting all the balls, there are two objects left.

and 1 medium .

and 1 medium .

Therefore, after subtracting all the balls, there are 5 objects left.

Therefore, after subtracting all the balls, there are 5 objects left.

after subtracting all the balls, there are two objects left.

after subtracting all the balls, there are two objects left.

(d)

###### OCR

[Figure 40]

Question: What is the name of the event?

[Figure 41]

Question: What is the of the

[Figure 42]

Question: What is the of the

[Figure 43]

Question: What is the name of the event?

[Figure 44]

Question: What is the name of the event?

###### Chosen Response:

###### Rejected Response:

###### Chosen Response:

###### Rejected Response:

###### Chosen Response:

###### Rejected Response:

###### Chosen Response:

###### Rejected Response:

The event is part of the "RBC Summer Kids Stuff." It includes two programs:

The event is part of the "RBC Summer Kids Stuff." It includes two programs:

The event is part of the "RBC Summer Kids Stuff." It includes two programs:

The event is part of the "RBC Summer Kids Stuff." It includes two programs:

The event is part of the "RBC Summer Kids Stuff." It includes two programs:

The event is part of the "RBC Summer Kids Stuff." It includes two programs:

###### Chosen Response:

###### Rejected Response:

The event is part of the "RBC Summer Kids Stuff." It includes two programs:

The event is part of the "RBC Summer Kids Stuff." It includes two programs:

- 1. Camp Kilimanjaro: This is a Vacation Bible School (VBS) program for children in Kindergarten through 5th grade. It is themed as "An Epic Expedition Through Proverbs.“
- 2. Cross Training: This is a middle school and high school student summer program with multiple activities, including sports, recreation, community service, and Bible lessons.

- 1. Camp Kilimanjaro: This is a Vacation Bible School (VBS) program for children in Kindergarten through 5th grade. It is themed as "An Epic Expedition Through Proverbs.“
- 2. Cross Training: This is for junior high students in 6th through 8th grade, organized by Rancho Baptist Church. The event is scheduled from June 22 to June 26, with activities running from 8:45 AM to 12:00 PM. Registration is open, and the cost is free.

- 1. Camp Kilimanjaro: This is a Vacation Bible School (VBS) program for children in Kindergarten through 5th grade. It is themed as "An Epic Expedition Through Proverbs.“
- 2. Cross Training: This is a school and high school student program with recreation, community and Bible lessons.

- 1. Camp Kilimanjaro: This is a Vacation Bible School (VBS) program for children in Kindergarten through 5th grade. It is themed as "An Epic Expedition Through Proverbs.“
- 2. Cross Training: This is for high students in 6th through 8th grade, by

- 1. Camp Kilimanjaro: This is a Vacation Bible School (VBS) program for children in Kindergarten through 5th grade. It is themed as "An Epic Expedition Through Proverbs.“
- 2. Cross Training: This is a school and high school student program with recreation, community and Bible lessons.

- 1. Camp Kilimanjaro: This is a Vacation Bible School (VBS) program for children in Kindergarten through 5th grade. It is themed as "An Epic Expedition Through Proverbs.“
- 2. Cross Training: This is for high students in 6th through 8th grade, by

The event is part of the "RBC Summer Kids Stuff." It includes two programs:

The event is part of the "RBC Summer Kids Stuff." It includes two programs:

- 1. Camp Kilimanjaro: This is a Vacation Bible School (VBS) program for children in Kindergarten through 5th grade. It is themed as "An Epic Expedition Through Proverbs.“
- 2. Cross Training: This is a middle school and high school student summer program with multiple activities, including sports, recreation, community service, and Bible lessons.

- 1. Camp Kilimanjaro: This is a Vacation Bible School (VBS) program for children in Kindergarten through 5th grade. It is themed as "An Epic Expedition Through Proverbs.“
- 2. Cross Training: This is for junior high students in 6th through 8th grade, organized by Rancho Baptist Church. The event is scheduled from June 22 to June 26, with activities running from 8:45 AM to 12:00 PM. Registration is open, and the cost is free.

- 1. Camp Kilimanjaro: This is a Vacation Bible School (VBS) program for children in Kindergarten through 5th grade. It is themed as "An Epic Expedition Through Proverbs.“
- 2. Cross Training: This is a middle school and high school student summer program with multiple activities, including sports, recreation, community service, and Bible lessons.

- 1. Camp Kilimanjaro: This is a Vacation Bible School (VBS) program for children in Kindergarten through 5th grade. It is themed as "An Epic Expedition Through Proverbs.“
- 2. Cross Training: This is for junior high students in 6th through 8th grade, organized by Rancho Baptist Church. The event is scheduled from June 22 to June 26, with activities running from 8:45 AM to 12:00 PM. Registration is open, and the cost is free.

. The event is from June to June with from 8: to : . is and the is .

. The event is from June to June with from 8: to : . is and the is .

(e)

###### Science

###### Question:

|[Figure 45]|
|---|

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: ..“ Question: Read the text. A population's growth is affected by factors in the environment, such as space, available food, predators, and disease. When a population inhabits an environment with abundant resources and few limiting factors, it can experience exponential growth. Under these conditions, a population grows increasingly rapidly. Plotted on a graph with time on the x-axis and population size on the y-axis, exponential growth resembles a J-shaped curve. Logistic growth, in contrast, occurs when resources are scarce or a population faces considerable limiting factors, such as predators. Logistic growth resembles an S-shaped curve: it rises steeply at first but then levels off. When growth levels off, the population has reached the environment's carrying capacity, or the population limit it can support. Which term matches the picture?

- A. logistic growth
- B. exponential growth

###### Chosen Response:

###### Rejected Response:

###### Document

To determine which term matches the picture, let's analyze the image and the descriptions provided:

To determine which term matches the picture, let's analyze the characteristics of each type of growth:

- 1. Exponential Growth: - Description: When a population inhabits an environment with abundant resources and few limiting factors, it can experience exponential growth. - Graph Shape: Exponential growth resembles a J-shaped curve.
- 2. Logistic Growth: - Description: Logistic growth occurs when resources are scarce or a population faces considerable limiting factors, such as predators. - Graph Shape: Logistic growth resembles an S-shaped curve: it rises steeply at first but then levels off. Image Analysis: - The graph shows a J-shaped curve where the population grows exponentially with time, without any leveling off. Based on the description and the shape of the graph, the term that matches the picture is: Final answer: B

- 1. Exponential Growth: This type of growth occurs when a population inhabits an environment with abundant resources and few limiting factors. In this case, the population grows increasingly rapidly, leading to a J-shaped curve. The growth is rapid and unbounded over time.
- 2. Logistic Growth: This type of growth occurs when resources are scarce or a population faces considerable limiting factors, such as predators. In this scenario, the population rises steeply at first but then levels off as it reaches the carrying capacity, which is the environment's maximum supportable population level. The growth curve resembles an S-shaped curve: it increases rapidly, reaches a limit, and then stabilizes. The graph in the picture shows a population that grows rapidly at first and then levels off, resembling an Sshaped curve. This is characteristic of logistic growth. Final answer: A

Question: What is the name of entity mentioned in the given form ?

[Figure 46]

###### Chosen Response:

###### Rejected Response:

The form is a "Principal Investigator’s Statement of Economic Interests" related to research funding. It details information about a research project led by Paul Saltman from the Biology department at the University of California, San Diego. The specific section of interest is Section A, which requires details about the non-governmental entity providing funding for the project. The information in this section pertains to the funding organization, their role, and the type of funds provided. The organization mentioned is the "Cystic Fibrosis Foundation ® (CFF ®)". So, the name of the entity mentioned in the form is the Cystic Fibrosis Foundation® (CFF ®). The CFF ® provides grants to support innovative and promising research projects in the field of cystic fibrosis, a genetic disorder affecting the lungs and digestive system. The funding primarily supports research aimed at improving patient outcomes, disease understanding, and treatments. In this case, Dr. Saltman's project aims to explore the molecular mechanisms underlying bone development during the fetus stages, potentially connecting them to cystic fibrosis and other conditions.

The form is a “Principal Investigator’s Statement of Economic Interests" related to research funding. It details information about a research project led by Paul Saltman from the Biology department at the University of California, San Diego. The specific section of interest is Section A, which requires details about the nongovernmental entity providing funding for the research. According to the form:

- - Name of Entity: Proctor & Gamble
- - Address of Entity: Cincinnati, Ohio
- - Principal Business of Entity: Household products and foods
- - Amount of Funding: $25,000.00 (Actual) This section is crucial as it discloses the financial backing from a non-governmental source, which is essential for transparency and compliance in research funding.

###### Chart

[Figure 47]

###### Question:

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: ..“ Question: What is the value of total in fence?

(f)

###### Chosen Response:

###### Rejected Response:

To determine the value of the "total" category in the "fence" row, we need to look at the colorcoded legend and the corresponding values for each category in the "fence" row.

To determine the value of the total in fence, we need to analyze the data given for each category in the "fence" row:

###### Science

1. legs: The value is 109 (one billion).

1. The "fence" row has the following categories: - legs: blue - index: green - engine: red -

| |[Figure 48]<br><br>[Figure 49]|Question:<br><br>Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: ..“ Question:<br><br>the . A growth is by factors in the such as and . When a population inhabits an environment with abundant resources and few limiting factors, it can experience exponential growth. these a population grows increasingly rapidly. on a graph with time on the x- and population on the y- exponential growth resembles a J-shaped curve. Logistic in occurs when resources are scarce or a population faces considerable limiting factors, such as predators. Logistic growth resembles an S-shaped curve: it rises steeply at first but then levels off. When growth levels off, the population has the environment's carrying<br><br>Question:<br><br>Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: ..“ Question: Read the text. A population's growth is affected by factors in the environment, such as space, available food, predators, and disease. When a population inhabits an environment with abundant resources and few limiting factors, it can experience exponential growth. Under these conditions, a population grows increasingly rapidly. Plotted on a graph with time on the x-axis and population size on the y-axis, exponential growth resembles a J-shaped curve. Logistic growth, in contrast, occurs when resources are scarce or a population faces considerable limiting factors, such as predators. Logistic growth resembles an S-shaped curve: it rises steeply at first but then levels off. When growth levels off, the population has reached the environment's carrying<br><br>2. index: The value is 103 (one thousand).<br>3. engine: The value is 105(one hundred thousand).<br>4. total: The value is not directly given, but we can infer it by summing the values of legs, index, and engine. Let's calculate the total: total=legs+index+engine, 𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕 = 𝟏𝟏𝟏𝟏𝟗𝟗 + 𝟏𝟏𝟏𝟏𝟑𝟑 + 𝟏𝟏𝟏𝟏𝟓𝟓, 𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕 =<br><br><br>𝟏𝟏𝟏𝟏𝟗𝟗 + 𝟏𝟏𝟏𝟏𝟑𝟑 + 𝟏𝟏𝟏𝟏𝟓𝟓, 𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕 = 𝟏𝟏𝟏𝟏𝟗𝟗 + 𝟏𝟏𝟏𝟏𝟑𝟑 + 𝟏𝟏𝟏𝟏𝟓𝟓, ...(Repeat until truncated.)<br><br>total: purple<br><br>2. According to the legend, the value for "total" is represented by the purple color.<br>3. By examining the "fence" row, we can see that the purple square is in the "total" column.<br>4. The purple square is aligned with the value on the right side of the image, which corresponds to 𝟏𝟏𝟏𝟏𝟑𝟑. Therefore, the value of the "total" in the "fence" row is 103. Final answer: 1000<br>|
|---|---|---|
| |Mathematics|capacity, or the population it can .<br><br>term matches the capacity, or the population limit it can support. Which term matches the picture?<br><br>A. logistic growth<br>B. exponential growth<br><br><br>A. logistic growth<br>B. exponential growth<br>|

###### Chosen Response:

###### Rejected Response:

Question:

###### Chosen Response:

###### Rejected Response:

[Figure 50]

To determine which term matches the picture, let's analyze the image and the descriptions provided:

To determine which term matches the picture, let's analyze the characteristics of each type of growth:

To determine which term matches the picture, let's analyze the and the provided:

To determine which term matches the picture, let's analyze the of each type of growth:

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: ..“ Question: Let quadrilateral ABCD have intersecting diagonals AC and BD at point O, and satisfy the conditions AC+BD=16 and CD=6. What expression represents the perimeter of triangle ABO? Use variables to represent the sides and diagonals to derive the expression. Choices: A: 8 B: 10 C: 12 D: 14

- 1. Exponential Growth: - Description: When a population inhabits an environment with abundant resources and few limiting factors, it can experience exponential growth. - Graph Shape: Exponential growth resembles a J-shaped curve.
- 2. Logistic Growth: - Description: Logistic growth occurs when resources are scarce or a population faces considerable limiting factors, such as predators. - Graph Shape: Logistic growth resembles an S-shaped curve: it rises steeply at first but then levels off. Image Analysis: - The graph shows a J-shaped curve where the population grows exponentially with time, without any leveling off. Based on the description and the shape of the graph, the term that matches the picture is: Final answer: B

- 1. Exponential Growth: This type of growth occurs when a population inhabits an environment with abundant resources and few limiting factors. In this case, the population grows increasingly rapidly, leading to a J-shaped curve. The growth is rapid and unbounded over time.
- 2. Logistic Growth: This type of growth occurs when resources are scarce or a population faces considerable limiting factors, such as predators. In this scenario, the population rises steeply at first but then levels off as it reaches the carrying capacity, which is the environment's maximum supportable population level. The growth curve resembles an S-shaped curve: it increases rapidly, reaches a limit, and then stabilizes. The graph in the picture shows a population that grows rapidly at first and then levels off, resembling an Sshaped curve. This is characteristic of logistic growth. Final answer: A

- 1. Exponential Growth: - Description: When a population inhabits an environment with abundant resources and few limiting factors, it can experience exponential growth. - Graph Shape: Exponential growth resembles a J-shaped curve.
- 2. Logistic Growth: - Description: Logistic growth occurs when resources are scarce or a population faces considerable limiting factors, such as predators. - Graph Shape: Logistic growth resembles an S-shaped curve: it rises steeply at first but then levels off.

- 1. Exponential Growth: This type of growth occurs when a population inhabits an environment with abundant resources and few limiting factors. In this the population grows increasingly rapidly, to a J-shaped curve. The growth is and time.
- 2. Logistic Growth: This type of growth occurs when resources are scarce or a population faces considerable limiting factors, such as predators. In this the population rises steeply at first but then levels off as it reaches the carrying capacity, which is the environment's population . The growth curve resembles an S-shaped curve: it rapidly, reaches a and then . The graph in the picture shows a population that grows rapidly at first and then levels off, an Sshaped curve. This is of logistic growth. Final answer: A

###### : - The graph shows a J-shaped curve the population grows with without any off.

###### Chosen Response:

###### Rejected Response:

on the and the of the the term that matches the picture is: Final answer: B

Step-by-Step Solution:

Let's break down the problem step by step:

- 1. Identify the Given Information:

- - Quadrilateral ABCD has diagonals AC and BD intersecting at point O.
- - The sum of the lengths of the diagonals is AC + BD = 16.
- - The length of side CD is CD = 6.

- 2. Properties of the Diagonals:

- Since O is the intersection point of the diagonals, we can denote the segments of the diagonals as follows: AO = CO, BO = DO.

- 3. Express the Diagonals in Terms of Segments: Let AO = CO = x and BO = DO = y. Therefore, AC = 2x and BD = 2y .
- 4. Use the Given Sum of Diagonals:

- - From the given information, AC + BD = 16.
- - Substituting the expressions for AC and BD , we get: 2x + 2y = 16
- - Simplifying, we find: x + y = 8

- 5. Perimeter of Triangle ABO : The perimeter of triangle ABO is the sum of the lengths of its sides: Perimeter of ∠ABO = AO + BO + AB. We know AO = x and BO = y, so: Perimeter of ∠ABO = 𝑥𝑥 + y + AB
- 6. Determine the Length of AB : Since AB = CD (opposite sides of a parallelogram are equal), and CD = 6 , we have: AB = 6
- 7. Calculate the Perimeter: Substitute x + y = 8 and AB = 6 into the perimeter formula: Perimeter of ∠ABO = 8 + 6 = 14 Final Answer: D

- 1. Identify the given sides of quadrilateral ABCD: AC (one diagonal) + BD (the other diagonal) = 16 units. CD (one side of quadrilateral ABCD) = 6 units.
- 2. Understand the problem context: The diagonals AC and BD intersect at point O. We need to find the perimeter of triangle ABO.
- 3. Identify missing lengths: The length of AB, AD, and BO are required to find the perimeter of triangle ABO. There is no direct information about these lengths, so we need to find relationships or additional lengths that can help us.
- 4. Use the given information to set up an equation: Since the diagonals of a cyclic quadrilateral intersect at a point that divides each diagonal into segments that are proportional, we can use the concept of similar triangles or segments' properties.
- 5. Apply the intersecting chords theorem (if quadrilateral ABCD were cyclic and AC and BD its diagonals): In a cyclic quadrilateral, the product of the segments of one chord is equal to the product of the segments of the other chord. Let AB = x, AD = y, BO = z. As the diagonals intersect at O, triangle ADO and ABO will be similar by AA similarity.
- 6. Calculate the perimeter of triangle ABO: Perimeter of triangle ABO = AB + AO + BO = x + y/2 + z.
- 7. Since the full lengths of AB and BO are unknown, we need to use the lengths provided: From the given CD = 6 units, we can use triangles or properties to find lengths. Since no further lengths are provided, assume they are proportional using segments or similarity properties.
- 8. Conclusion: Given AC + BD = 16 units and knowing that we don't need the specific lengths of segments due to the given perimeter problem context: The triangles can potentially be found using segment lengths from given proportionalities. Given all these relationships, knowing the lengths provided, and without further calculation: The correct option is 14, Final answer: D

(g)

###### Chart

[Figure 51]

###### Question:

[Figure 52]

###### Question:

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: ..“ Question: What is the value of total in fence?

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: ..“ Question: What is the value of total in

###### Chosen Response:

###### Rejected Response:

###### Chosen Response:

###### Rejected Response:

To determine the value of the "total" category in the "fence" row, we need to look at the colorcoded legend and the corresponding values for each category in the "fence" row.

To determine the value of the total in fence, we need to analyze the data given for each category in the "fence" row:

To determine the value of the "total" category in the "fence" row, we need to at the colorand the values for each category in the "fence" row.

To determine the value of the total in we need to analyze the given for each category in the "fence" row:

- 1. legs: The value is 109 (one billion).
- 2. index: The value is 103 (one thousand).
- 3. engine: The value is 105(one hundred thousand).
- 4. total: The value is not directly given, but we can infer it by summing the values of legs, index, and engine. Let's calculate the total: total=legs+index+engine, 𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕 = 𝟏𝟏𝟏𝟏𝟗𝟗 + 𝟏𝟏𝟏𝟏𝟑𝟑 + 𝟏𝟏𝟏𝟏𝟓𝟓, 𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕 =

- 1. The "fence" row has the following categories: - legs: blue - index: green - engine: red total: purple
- 2. According to the legend, the value for "total" is represented by the purple color.
- 3. By examining the "fence" row, we can see that the purple square is in the "total" column.
- 4. The purple square is aligned with the value on the right side of the image, which corresponds to 𝟏𝟏𝟏𝟏𝟑𝟑. Therefore, the value of the "total" in the "fence" row is 103. Final answer: 1000

- 1. legs: The value is 109 (one .
- 2. index: The value is 103 (one thousand).
- 3. engine: The value is 105(one thousand).
- 4. total: The value is but we can it by the values of and engine.

- 1. The "fence" row has the : - legs: - index: - engine: total: purple
- 2. to the the value for "total" is by the purple color.
- 3. the "fence" row, we can that the purple square is in the "total" .
- 4. The purple square is with the value on the side of the which to 𝟏𝟏𝟏𝟏𝟑𝟑.

Let's the total: , 𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕 = 𝟏𝟏𝟏𝟏𝟗𝟗 + 𝟏𝟏𝟏𝟏𝟑𝟑 + 𝟏𝟏𝟏𝟏𝟓𝟓, 𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕 = 𝟏𝟏𝟏𝟏𝟗𝟗 + 𝟏𝟏𝟏𝟏𝟑𝟑 + 𝟏𝟏𝟏𝟏𝟓𝟓, 𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕 = 𝟏𝟏𝟏𝟏𝟗𝟗 + 𝟏𝟏𝟏𝟏𝟑𝟑 + 𝟏𝟏𝟏𝟏𝟓𝟓, .)

𝟏𝟏𝟏𝟏𝟗𝟗 + 𝟏𝟏𝟏𝟏𝟑𝟑 + 𝟏𝟏𝟏𝟏𝟓𝟓, 𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕 = 𝟏𝟏𝟏𝟏𝟗𝟗 + 𝟏𝟏𝟏𝟏𝟑𝟑 + 𝟏𝟏𝟏𝟏𝟓𝟓, ...(Repeat until truncated.)

Therefore, the value of the "total" in the "fence" row is 103. Final answer:

(h)

###### OCR

###### Mathematics

###### Question:

Question:

Question:

[Figure 53]

[Figure 54]

[Figure 55]

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: ..“ Question: What is the total amount of this receipt?

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: ..“ Question: Let quadrilateral ABCD have intersecting diagonals AC and BD at point O, and satisfy the conditions AC+BD=16 and CD=6. What expression represents the perimeter of triangle ABO? Use variables to represent the sides and diagonals to derive the expression. Choices: A: 8 B: 10 C: 12 D: 14

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: ..“ Question: Let quadrilateral ABCD have intersecting diagonals AC and BD at point O, and the 16 and 6. What expression the perimeter of triangle Use

###### Chosen Response:

###### Rejected Response:

to the sides and diagonals to the expression. : A: 8 B: 10 : 12 D: 14

To find the total amount on the receipt, let's follow the steps provided:

To find the total amount on the receipt, let's break down the expenses listed on the receipt.

- 1. Identify the items listed:

- - Chicken Katsu Bento (BEN02): 1, RM 12.90
- - Omu Curry Fried Rice Set (Set10): 1, RM 6.90
- - Katsu Don (Fish) Sate (TS02 Kat): 1, RM 9.90

- 2. Calculate the subtotal:

- Add the prices of each item together: 𝟏𝟏𝟏𝟏.𝟗𝟗𝟏𝟏 + 𝟔𝟔.𝟗𝟗𝟏𝟏 + 𝟗𝟗.𝟗𝟗𝟏𝟏 = 𝟏𝟏𝟗𝟗.𝟕𝟕𝟏𝟏

- 3. Check for any discounts, service charges, or rounding adjustments:

- - There is a "Sub Total Before Discount" of RM 29.70.
- - There is no mention of any service charge or service discount.
- - The "Rounding Adjustment" is RM 0.00.

- 4. Calculate the total amount:

- 1. Chicken Katsu Bentod (Beno2):

- Quantity (QTY): 1 - Unit Price (RM): 12.90 - Discount (%): 0% - Amount (RM): 12.90

- 2. Omu Curry Fried Rice Set (Set1):

- Quantity (QTY): 1 - Unit Price (RM): 6.90 - Discount (%): 0% - Amount (RM): 6.90

- 3. Katsu Don (Fish) Sat (Ts02 Kats):

###### Chosen Response:

###### Rejected Response:

###### Chosen Response:

###### Rejected Response:

Step-by-Step Solution:

Let's break down the problem step by step:

Step-by-Step :

Let's break down the problem step by step:

- - Quantity (QTY): 1 - Unit Price (RM): 9.90 - Discount (%): 0% - Amount (RM): 9.90 Now, let's add up these amounts:
- - Chicken Katsu Bentod: RM 12.90
- - Omu Curry Fried Rice Set: RM 6.90
- - Katsu Don Sat: RM 9.90 Total before any discounts or service charges: 𝟏𝟏𝟏𝟏.𝟗𝟗𝟏𝟏 + 𝟔𝟔.𝟗𝟗𝟏𝟏 + 𝟗𝟗.𝟗𝟗𝟏𝟏 = 𝟏𝟏𝟗𝟗.𝟕𝟕𝟏𝟏 The receipt also mentions a "Sub Total Before Discount" of RM 29.70, which matches our calculation. Looking further, there are other details indicating adjustments: - Service Charge: RM 0.00 - Total Sales: RM 29.70 - Rounding Adjustment: RM 0.00 Finally, the receipt shows cash payment was RM 30.00 and the change was RM 0.30. Final answer: 30.00

- 1. Identify the Given Information:

- - Quadrilateral ABCD has diagonals AC and BD intersecting at point O.
- - The sum of the lengths of the diagonals is AC + BD = 16.
- - The length of side CD is CD = 6.

- 2. Properties of the Diagonals:

- Since O is the intersection point of the diagonals, we can denote the segments of the diagonals as follows: AO = CO, BO = DO.

- 3. Express the Diagonals in Terms of Segments: Let AO = CO = x and BO = DO = y. Therefore, AC = 2x and BD = 2y .
- 4. Use the Given Sum of Diagonals:

- - From the given information, AC + BD = 16.
- - Substituting the expressions for AC and BD , we get: 2x + 2y = 16
- - Simplifying, we find: x + y = 8

- 5. Perimeter of Triangle ABO : The perimeter of triangle ABO is the sum of the lengths of its sides: Perimeter of ∠ABO = AO + BO + AB. We know AO = x and BO = y, so: Perimeter of ∠ABO = 𝑥𝑥 + y + AB
- 6. Determine the Length of AB : Since AB = CD (opposite sides of a parallelogram are equal), and CD = 6 , we have: AB = 6
- 7. Calculate the Perimeter: Substitute x + y = 8 and AB = 6 into the perimeter formula: Perimeter of ∠ABO = 8 + 6 = 14 Final Answer: D

- 1. Identify the given sides of quadrilateral ABCD: AC (one diagonal) + BD (the other diagonal) = 16 units. CD (one side of quadrilateral ABCD) = 6 units.
- 2. Understand the problem context: The diagonals AC and BD intersect at point O. We need to find the perimeter of triangle ABO.
- 3. Identify missing lengths: The length of AB, AD, and BO are required to find the perimeter of triangle ABO. There is no direct information about these lengths, so we need to find relationships or additional lengths that can help us.
- 4. Use the given information to set up an equation: Since the diagonals of a cyclic quadrilateral intersect at a point that divides each diagonal into segments that are proportional, we can use the concept of similar triangles or segments' properties.
- 5. Apply the intersecting chords theorem (if quadrilateral ABCD were cyclic and AC and BD its diagonals): In a cyclic quadrilateral, the product of the segments of one chord is equal to the product of the segments of the other chord. Let AB = x, AD = y, BO = z. As the diagonals intersect at O, triangle ADO and ABO will be similar by AA similarity.
- 6. Calculate the perimeter of triangle ABO: Perimeter of triangle ABO = AB + AO + BO = x + y/2 + z.
- 7. Since the full lengths of AB and BO are unknown, we need to use the lengths provided: From the given CD = 6 units, we can use triangles or properties to find lengths. Since no further lengths are provided, assume they are proportional using segments or similarity properties.
- 8. Conclusion: Given AC + BD = 16 units and knowing that we don't need the specific lengths of segments due to the given perimeter problem context: The triangles can potentially be found using segment lengths from given proportionalities. Given all these relationships, knowing the lengths provided, and without further calculation: The correct option is 14, Final answer: D

- 1. Identify the Given :

- - ABCD has diagonals AC and BD intersecting at point O.
- - The sum of the lengths of the diagonals is AC + BD = 16.
- - The length of side CD is CD = 6.

- 2. of the Diagonals:

- Since O is the point of the we can the segments of the diagonals as : AO = BO = DO.

- 3. the Diagonals in of : Let AO = = x and BO = DO = y. Therefore, AC = 2x and BD = 2y .
- 4. Use the Given of Diagonals:

- - From the given AC + BD = 16.
- - the for AC and BD , we : 2x + 2y = 16
- - we find: x + y = 8

- 5. Perimeter of ABO : The perimeter of triangle ABO is the sum of the lengths of its sides: Perimeter of ∠ABO = AO + BO + AB. We AO = x and BO = y, so: Perimeter of ∠ABO = + y + AB
- 6. the of AB : Since AB = CD sides of a are and CD = 6 , we have: AB = 6
- 7. Calculate the Perimeter: x + y = 8 and AB = 6 into the perimeter : Perimeter of ∠ABO = 8 + 6 = 14 Final : D

- 1. Identify the given sides of quadrilateral ABCD: AC (one diagonal) + BD other diagonal) = 16 units. CD (one side of quadrilateral = 6 units.
- 2. the problem context: The diagonals AC and BD intersect at point O. We need to find the perimeter of triangle ABO.
- 3. Identify lengths: The length of and BO are to find the perimeter of triangle ABO. There is no information these so we need to find or lengths that can .
- 4. Use the given information to up an : Since the diagonals of a cyclic quadrilateral intersect at a point that each into segments that are we can use the of similar triangles or

properties.

- 5. the intersecting quadrilateral ABCD cyclic and AC and BD its : In a cyclic the product of the segments of chord is to the product of the segments of the other chord. Let

AB = = y, BO = z. the diagonals intersect at O, triangle and ABO be similar by similarity.

- 6. Calculate the perimeter of triangle ABO: Perimeter of triangle ABO = AB + AO + BO = x + 2 + z.
- 7. Since the lengths of AB and BO are we need to use the lengths provided: From the given CD = 6 we can use triangles or properties to find lengths. Since no further lengths are provided, are using segments or similarity properties.
- 8. : Given AC + BD = 16 units and knowing that we need the lengths of segments to the given perimeter problem context: The triangles can be using lengths given

- The total sales amount is already provided, which is RM 29.70. Thus, the total amount of this receipt is RM 29.70. Final answer: RM 29.70

(i)

. Given these knowing the lengths provided, and without further calculation: The is 14,

Final answer: D

7

###### OCR

###### Question:

###### Question:

[Figure 56]

[Figure 57]

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: ..“ Question: What is the total amount of this receipt?

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: ..“ Question: What is the total amount of this

###### Chosen Response:

###### Rejected Response:

###### Chosen Response:

###### Rejected Response:

To find the total amount on the receipt, let's break down the expenses listed on the receipt.

To find the total amount on the receipt, let's follow the steps provided:

To find the total amount on the receipt, let's break down the listed on the receipt.

To find the total amount on the receipt, let's the provided:

- 1. Chicken Katsu Bentod (Beno2):

- Quantity (QTY): 1 - Unit Price (RM): 12.90 - Discount (%): 0% - Amount (RM): 12.90

- 2. Omu Curry Fried Rice Set (Set1):

- Quantity (QTY): 1 - Unit Price (RM): 6.90 - Discount (%): 0% - Amount (RM): 6.90

- 3. Katsu Don (Fish) Sat (Ts02 Kats):

- 1. Identify the items listed:

- - Chicken Katsu Bento (BEN02): 1, RM 12.90
- - Omu Curry Fried Rice Set (Set10): 1, RM 6.90
- - Katsu Don (Fish) Sate (TS02 Kat): 1, RM 9.90

- 2. Calculate the subtotal:

- Add the prices of each item together: 𝟏𝟏𝟏𝟏.𝟗𝟗𝟏𝟏 + 𝟔𝟔.𝟗𝟗𝟏𝟏 + 𝟗𝟗.𝟗𝟗𝟏𝟏 = 𝟏𝟏𝟗𝟗.𝟕𝟕𝟏𝟏

- 3. Check for any discounts, service charges, or rounding adjustments:

- - There is a "Sub Total Before Discount" of RM 29.70.
- - There is no mention of any service charge or service discount.
- - The "Rounding Adjustment" is RM 0.00.

- 4. Calculate the total amount:

- 1. Chicken Katsu Bentod 2):

- Quantity (QTY): 1 - Unit Price (RM): 12.90 - Discount (%): 0% - Amount (RM): 12.90

- 2. Omu Curry Fried Rice Set (Set1):

- Quantity (QTY): 1 - Unit Price (RM): 6.90 - Discount (%): 0% - Amount (RM): 6.90

- 3. Katsu Don (Fish) Sat 02 :

- 1. Identify the listed:

- - Chicken Katsu 02): 1, RM 12.90
- - Omu Curry Fried Rice Set (Set10): 1, RM 6.90
- - Katsu Don (Fish) 02 : 1, RM 9.90

- 2. Calculate the :

- the of each : 𝟏𝟏𝟏𝟏.𝟗𝟗𝟏𝟏 + 𝟔𝟔.𝟗𝟗𝟏𝟏 + 𝟗𝟗.𝟗𝟗𝟏𝟏 = 𝟏𝟏𝟗𝟗.𝟕𝟕𝟏𝟏

- 3. for any service or adjustments:

- - There is a "Sub Total Before Discount" of RM 29.70.
- - There is no of any service or service .
- - The is RM 0.00.

- 4. Calculate the total amount:

- - Quantity (QTY): 1 - Unit Price (RM): 9.90 - Discount (%): 0% - Amount (RM): 9.90 Now, let's add up these amounts:
- - Chicken Katsu Bentod: RM 12.90
- - Omu Curry Fried Rice Set: RM 6.90
- - Katsu Don Sat: RM 9.90 Total before any discounts or service charges: 𝟏𝟏𝟏𝟏.𝟗𝟗𝟏𝟏 + 𝟔𝟔.𝟗𝟗𝟏𝟏 + 𝟗𝟗.𝟗𝟗𝟏𝟏 = 𝟏𝟏𝟗𝟗.𝟕𝟕𝟏𝟏 The receipt also mentions a "Sub Total Before Discount" of RM 29.70, which matches our calculation. Looking further, there are other details indicating adjustments: - Service Charge: RM 0.00 - Total Sales: RM 29.70 - Rounding Adjustment: RM 0.00 Finally, the receipt shows cash payment was RM 30.00 and the change was RM 0.30. Final answer: 30.00

- - Quantity (QTY): 1 - Unit Price (RM): 9.90 - Discount (%): 0% - Amount (RM): 9.90 let's up these :
- - Chicken Katsu Bentod: RM 12.90
- - Omu Curry Fried Rice Set: RM 6.90
- - Katsu Don Sat: RM 9.90 Total before any or service : 𝟏𝟏𝟏𝟏.𝟗𝟗𝟏𝟏 + 𝟔𝟔.𝟗𝟗𝟏𝟏 + 𝟗𝟗.𝟗𝟗𝟏𝟏 = 𝟏𝟏𝟗𝟗.𝟕𝟕𝟏𝟏 The receipt a "Sub Total Before Discount" of RM 29.70, which matches calculation. are other adjustments: - : RM 0.00 - Total : RM 29.70 - : RM 0.00

- The total sales amount is already provided, which is RM 29.70. Thus, the total amount of this receipt is RM 29.70. Final answer: RM 29.70

- The total amount is provided, which is RM 29.70.

the total amount of this receipt is RM 29.70. Final answer: RM 29.70

the receipt shows was RM 30.00 and the was RM 0.30. Final answer: 30.00

###### Science

###### Question:

||
|---|

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: ..“ Question: Read the text. A population's growth is affected by factors in the environment, such as space, available food, predators, and disease. When a population inhabits an environment with abundant resources and few limiting factors, it can experience exponential growth. Under these conditions, a population grows increasingly rapidly. Plotted on a graph with time on the x-axis and population size on the y-axis, exponential growth resembles a J-shaped curve. Logistic growth, in contrast, occurs when resources are scarce or a population faces considerable limiting factors, such as predators. Logistic growth resembles an S-shaped curve: it rises steeply at first but then levels off. When growth levels off, the population has reached the environment's carrying capacity, or the population limit it can support. Which term matches the picture?

- A. logistic growth
- B. exponential growth

###### Chosen Response:

###### Rejected Response:

To determine which term matches the picture, let's analyze the image and the descriptions provided:

To determine which term matches the picture, let's analyze the characteristics of each type of growth:

- 1. Exponential Growth: - Description: When a population inhabits an environment with abundant resources and few limiting factors, it can experience exponential growth. - Graph Shape: Exponential growth resembles a J-shaped curve.
- 2. Logistic Growth: - Description: Logistic growth occurs when resources are scarce or a population faces considerable limiting factors, such as predators. - Graph Shape: Logistic growth resembles an S-shaped curve: it rises steeply at first but then levels off. Image Analysis: - The graph shows a J-shaped curve where the population grows exponentially with time, without any leveling off. Based on the description and the shape of the graph, the term that matches the picture is: Final answer: B

- 1. Exponential Growth: This type of growth occurs when a population inhabits an environment with abundant resources and few limiting factors. In this case, the population grows increasingly rapidly, leading to a J-shaped curve. The growth is rapid and unbounded over time.
- 2. Logistic Growth: This type of growth occurs when resources are scarce or a population faces considerable limiting factors, such as predators. In this scenario, the population rises steeply at first but then levels off as it reaches the carrying capacity, which is the environment's maximum supportable population level. The growth curve resembles an S-shaped curve: it increases rapidly, reaches a limit, and then stabilizes. The graph in the picture shows a population that grows rapidly at first and then levels off, resembling an Sshaped curve. This is characteristic of logistic growth. Final answer: A

###### Chart

[Figure 59]

###### Question:

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: ..“ Question: What is the value of total in fence?

###### Chosen Response:

###### Rejected Response:

To determine the value of the "total" category in the "fence" row, we need to look at the colorcoded legend and the corresponding values for each category in the "fence" row.

To determine the value of the total in fence, we need to analyze the data given for each category in the "fence" row:

- 1. legs: The value is 109 (one billion).
- 2. index: The value is 103 (one thousand).
- 3. engine: The value is 105(one hundred thousand).
- 4. total: The value is not directly given, but we can infer it by summing the values of legs, index, and engine. Let's calculate the total: total=legs+index+engine, 𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕 = 𝟏𝟏𝟏𝟏𝟗𝟗 + 𝟏𝟏𝟏𝟏𝟑𝟑 + 𝟏𝟏𝟏𝟏𝟓𝟓, 𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕 =

- 1. The "fence" row has the following categories: - legs: blue - index: green - engine: red total: purple
- 2. According to the legend, the value for "total" is represented by the purple color.
- 3. By examining the "fence" row, we can see that the purple square is in the "total" column.
- 4. The purple square is aligned with the value on the right side of the image, which corresponds to 𝟏𝟏𝟏𝟏𝟑𝟑. Therefore, the value of the "total" in the "fence" row is 103. Final answer: 1000

𝟏𝟏𝟏𝟏𝟗𝟗 + 𝟏𝟏𝟏𝟏𝟑𝟑 + 𝟏𝟏𝟏𝟏𝟓𝟓, 𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕𝒕 = 𝟏𝟏𝟏𝟏𝟗𝟗 + 𝟏𝟏𝟏𝟏𝟑𝟑 + 𝟏𝟏𝟏𝟏𝟓𝟓, ...(Repeat until truncated.)

###### Mathematics

###### Question:

[Figure 60]

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: ..“ Question: Let quadrilateral ABCD have intersecting diagonals AC and BD at point O, and satisfy the conditions AC+BD=16 and CD=6. What expression represents the perimeter of triangle ABO? Use variables to represent the sides and diagonals to derive the expression. Choices: A: 8 B: 10 C: 12 D: 14

###### Chosen Response:

###### Rejected Response:

Step-by-Step Solution:

Let's break down the problem step by step:

- 1. Identify the Given Information:

- - Quadrilateral ABCD has diagonals AC and BD intersecting at point O.
- - The sum of the lengths of the diagonals is AC + BD = 16.
- - The length of side CD is CD = 6.

- 2. Properties of the Diagonals:

- Since O is the intersection point of the diagonals, we can denote the segments of the diagonals as follows: AO = CO, BO = DO.

- 3. Express the Diagonals in Terms of Segments: Let AO = CO = x and BO = DO = y. Therefore, AC = 2x and BD = 2y .
- 4. Use the Given Sum of Diagonals:

- - From the given information, AC + BD = 16.
- - Substituting the expressions for AC and BD , we get: 2x + 2y = 16
- - Simplifying, we find: x + y = 8

- 5. Perimeter of Triangle ABO : The perimeter of triangle ABO is the sum of the lengths of its sides: Perimeter of ∠ABO = AO + BO + AB. We know AO = x and BO = y, so: Perimeter of ∠ABO = 𝑥𝑥 + y + AB
- 6. Determine the Length of AB : Since AB = CD (opposite sides of a parallelogram are equal), and CD = 6 , we have: AB = 6
- 7. Calculate the Perimeter: Substitute x + y = 8 and AB = 6 into the perimeter formula: Perimeter of ∠ABO = 8 + 6 = 14 Final Answer: D

- 1. Identify the given sides of quadrilateral ABCD: AC (one diagonal) + BD (the other diagonal) = 16 units. CD (one side of quadrilateral ABCD) = 6 units.
- 2. Understand the problem context: The diagonals AC and BD intersect at point O. We need to find the perimeter of triangle ABO.
- 3. Identify missing lengths: The length of AB, AD, and BO are required to find the perimeter of triangle ABO. There is no direct information about these lengths, so we need to find relationships or additional lengths that can help us.
- 4. Use the given information to set up an equation: Since the diagonals of a cyclic quadrilateral intersect at a point that divides each diagonal into segments that are proportional, we can use the concept of similar triangles or segments' properties.
- 5. Apply the intersecting chords theorem (if quadrilateral ABCD were cyclic and AC and BD its diagonals): In a cyclic quadrilateral, the product of the segments of one chord is equal to the product of the segments of the other chord. Let AB = x, AD = y, BO = z. As the diagonals intersect at O, triangle ADO and ABO will be similar by AA similarity.
- 6. Calculate the perimeter of triangle ABO: Perimeter of triangle ABO = AB + AO + BO = x + y/2 + z.
- 7. Since the full lengths of AB and BO are unknown, we need to use the lengths provided: From the given CD = 6 units, we can use triangles or properties to find lengths. Since no further lengths are provided, assume they are proportional using segments or similarity properties.
- 8. Conclusion: Given AC + BD = 16 units and knowing that we don't need the specific lengths of segments due to the given perimeter problem context: The triangles can potentially be found using segment lengths from given proportionalities. Given all these relationships, knowing the lengths provided, and without further calculation: The correct option is 14, Final answer: D

(j)

###### OCR

###### Background Knowledge-based CoT

###### Question:

[Figure 61]

###### Question:

[Figure 62]

Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format "Final answer: ..“ Question: What is the total amount of this receipt?

Which state is highlighted?

- A. Idaho
- B. Washington
- C. Oregon
- D. Hawaii Please analyze the relevant background knowledge before answering the question.

###### Chosen Response:

###### Rejected Response:

To find the total amount on the receipt, let's follow the steps provided:

To find the total amount on the receipt, let's break down the expenses listed on the receipt.

- 1. Chicken Katsu Bentod (Beno2):

- Quantity (QTY): 1 - Unit Price (RM): 12.90 - Discount (%): 0% - Amount (RM): 12.90

- 2. Omu Curry Fried Rice Set (Set1):

- Quantity (QTY): 1 - Unit Price (RM): 6.90 - Discount (%): 0% - Amount (RM): 6.90

- 3. Katsu Don (Fish) Sat (Ts02 Kats):

- 1. Identify the items listed:

- - Chicken Katsu Bento (BEN02): 1, RM 12.90
- - Omu Curry Fried Rice Set (Set10): 1, RM 6.90
- - Katsu Don (Fish) Sate (TS02 Kat): 1, RM 9.90

- 2. Calculate the subtotal:

- Add the prices of each item together: 𝟏𝟏𝟏𝟏.𝟗𝟗𝟏𝟏 + 𝟔𝟔.𝟗𝟗𝟏𝟏 + 𝟗𝟗.𝟗𝟗𝟏𝟏 = 𝟏𝟏𝟗𝟗.𝟕𝟕𝟏𝟏

- 3. Check for any discounts, service charges, or rounding adjustments:

- - There is a "Sub Total Before Discount" of RM 29.70.
- - There is no mention of any service charge or service discount.
- - The "Rounding Adjustment" is RM 0.00.

- 4. Calculate the total amount:

###### Chosen Response:

###### Rejected Response:

The map of the United States shows all states in green, but one state is highlighted in a darker shade. The highlighted state is located in the Pacific Ocean, separate from the mainland United States. This state is an archipelago and is situated southwest of the continental U.S. It is known for its

The map of the United States shows all states in green, but one state is highlighted in a darker shade. The highlighted state is located in the Pacific Ocean, separate from the mainland United States. This state is an archipelago and is situated southwest of Oregon and northwest of Hawaii. It consists of over 1,400 islands and atolls spread over 1,500 mi (2,400 km)2. The state covers an area of about 70,000 sq miles (0.7% of the total United States land area) and is the only U.S. state whose mainland part is fully in one archipelago. It is the 20th and second smallest state by area, and by population the 45th. The only state that fits this description is Hawaii, so the answer is D.

tropical climate and is a popular tourist destination. Given these characteristics, the highlighted state is: D. Hawaii

- - Quantity (QTY): 1 - Unit Price (RM): 9.90 - Discount (%): 0% - Amount (RM): 9.90 Now, let's add up these amounts:
- - Chicken Katsu Bentod: RM 12.90
- - Omu Curry Fried Rice Set: RM 6.90
- - Katsu Don Sat: RM 9.90 Total before any discounts or service charges: 𝟏𝟏𝟏𝟏.𝟗𝟗𝟏𝟏 + 𝟔𝟔.𝟗𝟗𝟏𝟏 + 𝟗𝟗.𝟗𝟗𝟏𝟏 = 𝟏𝟏𝟗𝟗.𝟕𝟕𝟏𝟏 The receipt also mentions a "Sub Total Before Discount" of RM 29.70, which matches our calculation. Looking further, there are other details indicating adjustments: - Service Charge: RM 0.00 - Total Sales: RM 29.70 - Rounding Adjustment: RM 0.00 Finally, the receipt shows cash payment was RM 30.00 and the change was RM 0.30. Final answer: 30.00

Hawaii is the only U.S. state made entirely of islands and is located in the central Pacific Ocean, distinct from the mainland states.

(k)

- The total sales amount is already provided, which is RM 29.70. Thus, the total amount of this receipt is RM 29.70. Final answer: RM 29.70

###### Visual Content-based CoT

###### Question:

[Figure 63]

What was the real GDP of New York in 2020? Please analyze the given image and then provide step-by-step reasoning.

###### Chosen Response:

###### Rejected Response:

The line chart shows the Real GDP in billion U.S. dollars from 2000 to 2020. The GDP started at 1084.84 in 2000 and steadily increased to 1323.4 in 2011. It then experienced a slight dip in 2012 and 2013, but continued to increase, reaching a peak of 1490.68 in 2019. In 2020, the GDP decreased to 1402.41. Here’s a detailed breakdown:

The line chart shows the Real GDP in billion U.S. dollars from 2000 to 2020. The GDP started at 1084.84 in 2000 and steadily increased to 1323.4 in 2011. It then experienced a slight dip in 2012 and 2013, but continued to increase, reaching a peak of 1490.68 in 2019. In 2020, the GDP decreased to 1402.41. Here’s a detailed breakdown:

- 1.Trend Overview: From 2000 to 2020, New York’s real GDP generally increased, with some fluctuations along the way. The steady growth reflects the state’s expanding economy over the two decades.
- 2.Early Years (2000-2009): Between 2000 and 2009, the GDP grew from 1,084.84 billion to 1,177.92 billion U.S. dollars, experiencing occasional dips, such as during the 2008 financial crisis.
- 3.Recovery and Growth (2010-2019): Post-2009, the GDP experienced a stronger upward trajectory, reaching a peak of 1,490.68 billion U.S. dollars in 2019, indicating robust economic recovery and growth.
- 4.Decline in 2020: In 2020, there was a notable decline to 1,402.41 billion U.S. dollars, likely due to the economic impact of the COVID-19 pandemic, which affected various sectors and led to a decrease in economic activity. This data is sourced from Statista, a reliable provider of market and consumer data. The graph reflects the economic dynamics of New York over two decades, highlighting both growth periods and downturns.

- 1.Trend Overview: From 2000 to 2020, New York’s real GDP generally increased, with some fluctuations along the way. The steady growth reflects the state’s expanding economy over the two decades.
- 2.Early Years (2000-2009): Between 2000 and 2009, the GDP grew from 1,084.84 billion to 1,177.92 billion U.S. dollars, experiencing occasional dips, such as during the 2008 financial crisis.
- 3.Post-2009 Growth (2010-2019): After the recession, the GDP continued to grow, reaching 1,490.68 billion U.S. dollars in 2019. This growth can be attributed to various factors, including technological advancements, increased investment, and a strong job market.
- 4. 2020 Decline: In 2020, the GDP decreased to 1,402.41 billion U.S. dollars. This decline can be attributed to the COVID-19 pandemic, which severely impacted the global economy and New York City, causing a temporary halt to many businesses and events.
- 5. Future Outlook: As the pandemic gradually subsides, it is expected that New York’s real GDP will gradually recover and continue its upward trend, albeit at a slower pace than before the pandemic.

(l)

###### Grounded CoT

[Figure 64]

###### Question:

What sport is being played? Answer the question with grounding.

###### Chosen Response:

###### Rejected Response:

The <ref>players</ref><box>[[621, 294, 785, 748]]</box> are playing soccer, as indicated by their uniforms and the <ref>soccer ball</ref><box>[[583, 659, 618, 721]]</box> on the <ref>field</ref><box>[[0, 559, 998, 1000]]</box>.

The <ref>players</ref><box>[[621, 294, 785, 748]]</box> are playing soccer, as indicated by their uniforms and the fact that they're dressed in the appropriate cleats and stadium attire.

(m)

Figure 7. More data examples from MMPR. Figure 7a to 7f are examples from data constructed using DropoutNTP, while Figure 7g to 7j are examples from data constructed using correctness-based pipeline. Additionally, the examples for multimodal CoT, which is introduced in Section 4.2, are shown in Figure 7k to 7m.

