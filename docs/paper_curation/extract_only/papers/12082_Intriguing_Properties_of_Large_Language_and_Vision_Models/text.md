# arXiv:2410.04751v1[cs.CV]7Oct2024

## INTRIGUING PROPERTIES OF LARGE LANGUAGE AND VISION MODELS

Young-Jun Lee 1 Byungsoo Ko 1 Han-Gyu Kim 2 Yechan Hwang 1 Ho-Jin Choi 1 1 School of Computing, KAIST 2 NAVER Cloud Multimodal AI {yj2961, yemintmint, hojinc}@kaist.ac.kr Kobiso62@gmail.com hangyu.kim@navercorp.com

Code: https://github.com/passing2961/IP-LLVM

ABSTRACT

Recently, large language and vision models (LLVMs) have received significant attention and development efforts due to their remarkable generalization performance across a wide range of tasks requiring perception and cognitive abilities. A key factor behind their success is their simple architecture, which consists of a vision encoder, a projector, and a large language model (LLM). Despite their achievements in advanced reasoning tasks, their performance on fundamental perception-related tasks (e.g., MMVP) remains surprisingly low. This discrepancy raises the question of how LLVMs truly perceive images and exploit the advantages of the vision encoder. To address this, we systematically investigate this question regarding several aspects: permutation invariance, robustness, math reasoning, alignment preserving and importance, by evaluating the most common LLVM’s families (i.e., LLaVA) across 10 evaluation benchmarks. Our extensive experiments reveal several intriguing properties of current LLVMs: (1) they internally process the image in a global manner, even when the order of visual patch sequences is randomly permuted; (2) they are sometimes able to solve math problems without fully perceiving detailed numerical information; (3) the cross-modal alignment is overfitted to complex reasoning tasks, thereby, causing them to lose some of the original perceptual capabilities of their vision encoder; (4) the representation space in the lower layers (< 25%) plays a crucial role in determining performance and enhancing visual understanding. Lastly, based on the above observations, we suggest potential future directions for building better LLVMs and constructing more challenging evaluation benchmarks.

1 INTRODUCTION

Large Language and Vision Models (LLVMs)(Liu et al., 2024c;b; Lee et al., 2024c;b) have demonstrated remarkable generalization capabilities across a wide variety of tasks, including coding and mathematics, showcasing their potential for practical applications. These impressive advancements have been achieved through a straightforward yet effective architecture based on the concept of model-stitching(Lenc & Vedaldi, 2015; Bansal et al., 2021). This approach integrates a pre-trained vision encoder (Radford et al., 2021) with a pre-trained large language model (LLM) (Touvron et al., 2023; Zheng et al., 2023b) via a simple cross-modal alignment module. This method significantly benefits from the power of well-established pre-trained representations. Consequently, this structure has become the de facto standard in the field, extending into other modality domains such as video, audio, and unified modalities (Xie et al., 2024; Erfei Cui, 2024).

Despite their significant generalization performance, recent studies have revealed several interesting phenomena regarding LLVMs. For instance, they struggle with tasks that are easy for humans to perceive (e.g., MMVP (Tong et al., 2024), BLINK (Fu et al., 2024)) and have limited understanding of domain-specific images (Zhai et al., 2024; Verma et al., 2024). In contrast to the computer vision domain, where demystifying the properties of vision encoders has been more thoroughly explored (Naseer et al., 2021; Kim et al., 2023; Vishniakov et al., 2023), the underlying properties of LLVMs are still largely under-explored. Therefore, in this work, we scrutinize the current de facto

structure of LLVMs under various partial conditions, such as permutation, occlusion, and synthetic images.

In this paper, we systematically conduct a comprehensive evaluation of widely used LLVM families, specifically the LLaVA-series, across 10 diverse benchmarks, which encompass tasks such as math, chart, and basic perception. Our extensive experiments reveal several intriguing properties of current LLVMs, which we summarize as follows:

- • In LLVMs, the visual patch tokens processed through the projector exhibit varying magnitudes of localized visual information. Remarkably, even when the order of these patch sequences is randomly shuffled before being fed into the LLM, the performance does not significantly degrade. For instance, in the case of LLaVA 1.5 (Li et al., 2024b), the average performance drop across 10 benchmarks is 0.19 (< 1%), indicating that LLVMs exhibit permutation-invariant properties.
- • LLVMs effectively handle tasks when given synthetic versions of the MathVista (Lu et al.,

2023) dataset, with only a small performance decline (1.8% for LLaVA 1.5). Furthermore, we discovered that, in certain scenarios, LLVMs can solve problems even without access to the full image, including detailed numerical and chart elements.

- • Following alignment and visual instruction tuning, LLVMs fail to preserve their initial perceptual capacities, with up to a 20% drop in image classification tasks (e.g., CIFAR100 (Krizhevsky et al., 2009)), a phenomenon known as catastrophic forgetting (Zhai et al., 2024). Furthermore, they struggle to understand shared-world concepts within the representation space, according to the platonic representation hypothesis (Huh et al., 2024).
- • Our analysis of model behavior reveals that LLVMs tend to concentrate more on the central region of the image. Furthermore, the lower layers in LLVM architectures are crucial for better generalization. In these layers (i.e., the bottom 20% of the LLM layers), the model primarily processes visual information, while the higher layers focus on interpreting the text.

In addition to our findings, we present and discuss several points regarding LLMs and evaluation benchmarks. Specifically, we highlight the need to develop more interactive and complex evaluation benchmarks to mitigate selection bias Zheng et al. (2023a) and improve applicability to real-world scenarios. Furthermore, when developing new LLMs, it is crucial to preserve cross-modal alignment. We hope that our findings will assist other ML researchers and engineers in building a new paradigm for LLMs.

- 2 RELATED WORKS

Model-Stitching. The model-stitching (Lenc & Vedaldi, 2015; Bansal et al., 2021) is a technique first introduced to study the internal representations of neural networks by measuring the representational similarity between two given models. Consider two models defined as f = fm ◦ ··· ◦ f1 and g = gn◦···◦g1. Specifically, the stitched model is formalized as F = gn◦···◦gk+1◦s◦fk◦···◦f1, where s is a simple stitching layer (e.g., a linear layer or a 1×1 convolution). Therefore, even if the two models f and g differ in training methodology (e.g., supervised vs. self-supervised) or modalities (e.g., text vs. image), if F exhibits good performance, then f and g have strongly correlated and compatible internal representations at layer k, apart from the stitching layer s. Merullo et al. (2022) have the similar concept of model-stitching to verify their strong hypothesis that the conceptual representations from a frozen LLM and a visual encoder are sufficiently similar such that a simple linear mapping layer can align them.

Large Language and Vision Models. Recent advancements in LLVMs have predominantly adopted simplistic yet highly effective architectures, notably through the model-stitching concept. Numerous prior studies have introduced various design modifications to bridge the performance gap with closed-source LLVMs (OpenAI, 2023; Anthropic, 2024). These efforts include focusing intently on high-resolution processing (Li et al., 2024d; Liu et al., 2024b; Shi et al., 2024), implementing locality-enhanced projectors (Cha et al., 2024), and incorporating knowledge embeddings (Lee et al., 2024c), layer traversal technique (Lee et al., 2024b) and leveraging a diverse array of vision

encoders (Lu et al., 2024; Tong et al., 2024) have also been explored. Additionally, integrating external, task-specific computer vision modules (Lee et al., 2024d;e; Jiao et al., 2024; Lai et al., 2024) and incorporating different modalities — including video and audio (Wang et al., 2024; Li et al., 2024a; Erfei Cui, 2024; Xie et al., 2024) — have expanded the models’ capabilities. Moreover, enabling the handling of interleaved input formats (Li et al., 2024b; Xue et al., 2024) has further broadened the versatility of these models. While these models have been developed based on a simplistic structure, model-stitching, the effectiveness of this architecture remains under-explored.

Investigating Intriguing Properties of LLVMs. Alongside these advancements, recent studies have investigated and uncovered several crucial properties of current LLVMs. For instance, some studies have rigorously evaluated LLVMs on basic perception tasks that are trivially easy for humans by introducing “blind” pairs of image datasets (Tong et al., 2024; Fu et al., 2024; Rahmanzadehgervi et al., 2024). Other studies have explored cross-modal alignment by focusing on domain-specific visual capabilities (Verma et al., 2024) and examining the alignment of representation spaces across modalities between independently pre-trained LLMs and vision encoders (Li et al., 2024c; Huh et al., 2024). Zhai et al. (2024) examine the phenomenon of catastrophic forgetting in LLVMs within the context of image classification tasks. Additional studies (Zhou et al., 2023b; Chen et al., 2024b) analyze the persistent issue of object hallucination in LLVMs. Moreover, research has explored spatial reasoning capabilities (Kamath et al., 2023). While vision encoders (e.g., ViT (Dosovitskiy, 2020), DeiT (Touvron et al., 2021)) in the computer vision field have been rigorously examined across a wide range of image settings, the study of these intriguing properties in LLVMs remains relatively under-explored. In this paper, we aim to address this by conducting an in-depth investigation into LLVMs, examining their permutation invariance, robustness, alignment preservation, and importance in scenarios involving occluded and synthesized images.

- 3 DEMYSTIFYING INTRIGUING PROPERTIES OF LVLMS

In this section, we explore the intriguing properties of current LLVMs that have de facto structure of modal-stitching in terms of various aspects: permutation invariance, robustness to occlusion, synthetic data, alignment preserving, and importance.

- 3.1 BACKGROUND

Overview of LVLM. Current LVLMs M have widely adopted the model-stitching architecture, which consists of three main components: a pre-trained vision encoder fV, a projector fP, and a pre-trained LLM fL. The overall model is represented as M = fL ◦ fP ◦ fV. The vision encoder fV converts the input image I ∈ R3×H×W into visual features Fv ∈ RN×d

v, where N = HW/P2 is the number of visual features, P is the patch size, and dv is the dimension of the vision encoder’s output. The projector fP transforms these visual features Fv into visual patch tokens XV ∈ RN×d

l

in the representation space of the LLM, where dl is the embedding dimension of the LLM. This mapping allows the LLM to perceive and conceptually understand the given image. The LLM fL produces an appropriate response Y = {yi}L

Y

i=1 in an autoregressive manner, given both the visual patch tokens XV and the text tokens XT ∈ RL

T×dl, where LT denotes the length of the input text sequence, and LY is the length of the output sequence. The probability of generating the response is given by:

p(Y | XV,XT) =

LY

i=1

p(yi | XV,XT,y<i) (1)

- 3.2 EVALUATION SETUP

Evaluation Benchmarks. To ensure a comprehensive and rigorous evaluation, we employ 10 standard and widely adopted benchmarks: MMVP (Tong et al., 2024), Q-Bench (Wu et al., 2023), MME (Fu et al., 2023), MMStar (Chen et al., 2024a), MM-Vet (Yu et al., 2023), LLaVA-W (Liu et al., 2024c), MathVista (Lu et al., 2023), SQA-IMG (Lu et al., 2022a), ChartQA (Masry et al., 2022), and AI2D (Kembhavi et al., 2016). Detailed descriptions of each dataset are provided in Appendix A.

LLVMs MMVP Q-Bench MME MMStar MM-Vet LLaVAW MathVista SQAI ChartQA AI2D Avg. ∆ LLaVA-1.5 34.67 59.73 1850.07 34.20 31.50 67.50 24.70 65.59 16.92 53.34

36.00

59.60

1874.60

33.33

30.40

66.20

21.20

65.44

14.08

52.69

▼ 0.59 LLaVA-NeXT 36.67 63.55 1874.42 37.80 43.50 75.50 32.00 62.12 66.06 64.02

+ Perm.

(▲ 1.33)

(▼ 0.13)

(▲ 24.53)

(▼ 0.87)

(▼ 1.10)

(▼ 1.30)

(▼ 3.50)

(▼ 0.15)

(▼ 2.84)

(▼ 0.65)

37.33

62.54

1890.19

36.87

43.40

75.80

21.70

62.12

34.55

64.02

▼ 2.71 LLaVA-OneVision 60.67 77.26 1982.5 59.87 57.80 87.40 61.80 94.00 93.52 81.25

+ Perm.

(▲ 0.67)

(▼ 1.00)

(▲ 15.78)

(▼ 0.93)

(▼ 0.10)

(▲ 0.30)

(▼ 10.30)

(▼ 0.00)

(▼ 31.51)

(▼ 0.00)

59.33

76.99

1964.3

54.93

47.60

82.30

53.50

89.24

58.26

75.58

▼ 9.40

+ Perm.

(▼ 1.33)

(▼ 0.27)

(▼ 18.2)

(▼ 4.93)

(▼ 10.20)

(▼ 5.10)

(▼ 8.30)

(▼ 4.76)

(▼ 35.26)

(▼ 5.67)

Table 1: Results of drop ratio (∆) when random permutation is applied. We run five experiments.

Evaluation Models. Recently, a large number of LVLM models have been actively introduced, owing to their remarkable flexibility and versatility across multiple domains. Consequently, it is challenging and inefficient to conduct holistic evaluations on all LVLMs. Therefore, we select most standard LLVMs: LLaVA-1.5-7B (Li et al., 2024b), LLaVA-NeXT-7B (Liu et al., 2024b), and LLaVA-OneVision-8B (Li et al., 2024a). For our customized experiments, before evaluating LVLMs under diverse settings (e.g., occlusion), we first attempt to reproduce the baseline performance of LVLMs on 10 evaluation benchmarks. To do this, we implement our customized evaluation toolkits by referring to the code of UniBench 1 (Al-Tahan et al., 2024). Detailed descriptions of each model are provided in Appendix B.

- 3.3 DO LLVMS PERCEIVE IMAGES GLOBALLY?

Current LVLMs commonly adopt ViT (Dosovitskiy, 2020)-based vision encoders, such as CLIP ViT (Radford et al., 2021) and SigLIP (Zhai et al., 2023), making their image perception dependent on these encoders. Specifically, ViT is designed to learn interactions across all image patches, providing properties (Naseer et al., 2021; Vishniakov et al., 2023) such as permutation invariance and robustness to occlusion. This raises the question of whether these ViT properties might transfer to current LVLM models.

[Figure 1]

Each visual patch token encapsulates localized visual information. We first investigate whether each visual patch token XV from the projector fP captures a localized understanding of the patch area corresponding to its position in the image. Specifically, given an image I, the projector outputs N visual patch tokens (e.g., N = 576 for LLaVA-1.5-7B). We then select a single token (removing all others) and feed it into the LLM fL. To quantify this, we define the patch information loss (PIL) as the ratio of the performance drop to the original performance. However, performing computations on each individual visual token is computationally intensive, especially for models such as LLaVA 1.5-7B that process 576 visual tokens arranged in a 24×24 grid of patches. To accelerate computation and reduce complexity, we aggregate the original

Figure 1: We demonstrate the extent to which group-wise visual tokens capture region-specific information (PIL) for LLaVA-1.5-7B on the MMStar (Chen et al., 2024a) and MME (Fu et al., 2023). Darker regions indicate areas where the model retains more localized information for those specific groups.

N visual tokens into M tokens, where M < N, by grouping neighboring tokens. As shown in Figure 1, the group-wise visual tokens in the LLaVA-1.5-7B model demonstrate varying levels of performance on the MMStar (Chen et al., 2024a) and MME (Fu et al., 2023), suggesting that each token captures localized visual information rather than global concept understanding. Additionally, the central visual tokens contain more informative content compared to those at the edges.

LLVMs are permutation invariant in terms of visual patch tokens. From our above results, we empirically verify that each visual patch token from the projector contains localized visual information. Here, we aim to understand how LLVMs systematically process and perceive images based on these visual patch tokens. Given that LLVMs generate answers in an autoregressive manner, we

1https://github.com/facebookresearch/unibench

Figure 3: We present examples of shuffled images with different grid sizes (2, 4, 8, 14) derived from a MathVista dataset image. As the grid size increases, the chart image becomes more artistically styled.

Original Image 2 X 2 4 X 4 8 X 8 14 X 14

|[Figure 2]|
|---|

|[Figure 3]|
|---|

|[Figure 4]|
|---|

|[Figure 5]|
|---|

|[Figure 6]|
|---|

investigate whether they exhibit order bias regarding visual patch tokens. To study this, we strongly hypothesize that if LLVMs have permutation variance, the performance drop (∆) will be significant when a random permutation is applied to the visual patch tokens XV.

As shown in Table 1, the overall performance across most benchmarks declines when the visual patch tokens are randomly shuffled. However, the performance gap between the original and the shuffled (Perm.) versions is not substantial, remaining within a 0–2% range, for LLaVA-1.5 and LLaVA-NeXT. Considering that LLaVA-1.5 uses 576 visual tokens, this is an intriguing observation. It suggests that current LLVMs interpret images in a global manner, despite each visual patch token containing localized information (see Figure 1), and even though they process both images and text autoregressively. In the case of LLaVA-OneVision which has many visual tokens (729), the avg. performance drop (∆) is non-trivial. We hypothesize that this global interpretation may result from recent LLVMs being trained via backpropagation, with the loss signal primarily derived from the text output of the Assistant: side. Based on these experiments, we argue that while LLVMs are trained with an autoregressive objective, they internally handle images globally. This observation offers a possible explanation for the success of pixel shuffling (Chen et al., 2024c) in achieving both strong performance and efficiency.

##### MMVP

MM-Vet

60

60

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

40

LLVMs are sensitive to spatial structures. Instead of treating visual patch tokens as permutation invariants, we explore how LLVMs behave when the sequence of image patches is permuted. To examine the sensitivity to spatial structure, we randomly shuffle image patches at varying grid sizes (2, 4, 8, 14), as shown in Figure 2. In our experiments, we observe that LLaVA-OneVision is sensitive to spatial structures on the MathVista (Lu et al., 2023) and AI2D (Kembhavi et al., 2016) tasks, despite the ViT learning all interactions between image patches. This result contrasts with previous study (Naseer

40

20

20

0 2 4 8 14

0 2 4 8 14

Grid Size

Grid Size

MathVista

##### AI2D

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

80

60

40

60

20

0 2 4 8 14

0 2 4 8 14

Grid Size

Grid Size

LLaVA-1.5-7B LLaVA-NeXT-7B LLaVA-OneVision-7B

Figure 2: We present the performance across different grid sizes (2, 4, 8, 14) on the MMVP, MM-Vet, MathVista, and AI2D datasets, using three models: LLaVA-1.5, LLaVA-NeXT, and LLaVA-OneVision.

- et al., 2021) suggesting that ViT-based vision encoders exhibit high permutation invariance to patch positions than CNN counterparts. We posit that on the MMVP Tong et al. (2024) dataset, which involves perception task, LLaVA-OneVision would also show permutation invariance to randomly shuffled patches, similar to existing work (Naseer et al., 2021) analyzing the ImageNet (Deng et al.,

2009) val. dataset. However, unlike ImageNet, the MathVista and AI2D datasets contain more structurally complex images (e.g., charts, code screenshots) that are highly sensitive to spatial structure, as the original numerical understanding is significantly disrupted. Interestingly, both LLaVA-1.5 and LLaVA-NeXT exhibit insensitivity to spatial structure, particularly in the MathVista dataset, where performance drops were minimal. These results suggest the need for further investigation, which we address in the following sections.

- 3.4 DO LLVMS PERCEIVE NUMERICAL INFORMATION WELL?

Here, we study whether LLVMs truly perceive text-rich images (e.g., charts, geometric shapes) that contain highly detailed numerical and shape information. To do this, we construct synthetic datasets for MMVP (Tong et al., 2024) and MathVista (Lu et al., 2023). Specifically, we first generate an image description of a given image using LLaVA-OneVision (Li et al., 2024a) with the prompt: “Please generate a caption of this image.”. Next, we generate an image corresponding to the image description leveraging the SDXL-Lightning (Lin et al., 2024) model, ensuring both quality and

Figure 5: We present examples of images (left) synthesized by SDXL-Lightning and (right) occluded using three methods: Random, Salient, and Non-Salient. The original images are from the MathVista and MME datasets. Occluded areas are marked in black to indicate zero pixel values.

Synthesize Image

Salient PatchDrop

Non-Salient PatchDrop

Random PatchDrop

Original Image

Original Image

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

efficiency. As a result, we get synthesized version (Syn.) to the original version (Org.), illustrated in Figure 5.

MMVP MathVista LLVMs Text-rich? Scale Orig. Syn. Orig. Syn. Freq. CLIP-I - - - 0.84 - 0.61 LLaVA-1.5 ✗ 158K 34.7

In some cases, LLVMs can solve problems without seeing the image. Table 2 presents the performance comparison on both original and synthesized datasets. For comparison, we evaluate the knowledge-embedded-specific LLVM, Meteor 7B (Lee et al., 2024c). Overall, compared to the basic perception task (i.e., MMVP), the performance drop in MathVista is not significantly larger across four LLVMs. Given that the generated images show distorted chart and function shapes, with detailed numerical and formula information missing, as shown in Figure 5 (CLIP-I scores lower than in MMVP), it is surprising that LLVMs are still able to solve math problems requiring advanced cognitive reasoning, without key information. This observation leads us to more in-depth analysis on MathVista dataset. We analyze how LLVMs solve math problems using synthesized images. In instances where they answer correctly, LLVMs frequently choose “No” for MCQs and tend to generate “1” for free-form responses. A deeper analysis reveals that many of these questions ask “What is the smallest value?”, causing the models to select “1” using commonsense reasoning, without needing to interpret the image itself. Table 2 shows how often the models produce “1,” with a noticeable drop in frequency for LLaVA-OneVision and Meteor models. This suggests that these models, likely due to extensive training with million-scale datasets, struggle with “smallest value” questions when images are unclear, demonstrating their ability to interpret images effectively.

22.9

20.7

81.0 LLaVA-NeXT ✓ 760K 36.7

24.7

(▼ 14.0)

(▼ 1.8)

16.7

27.7

50.0 Meteor ✓ 1.1M 51.3

32.0

(▼ 20.0)

(▼ 4.3)

35.3

31.4

9.5 LLaVA-OneVision ✓ 3.1M 60.7

52.1

(▼ 16.0)

(▼ 20.7)

37.3

37.0

61.8

12.0

(▼ 23.3)

(▼ 24.8)

Table 2: We present the performance on the synthesized versions of the MMVP (Tong et al., 2024) and MathVista (Lu et al., 2023) datasets across the models LLaVA-1.5, LLaVA-NeXT, Meteor, and LLaVA-OneVision. Additionally, we provide the scale of the visual instruction training datasets used by each model and specify whether chart, math, and diagram datasets were included. CLIP-I indicates the image similarity using CLIP-ViT-L/14. Freq. denotes the frequency with which the model generates the answer “1” in free-form question types.

| | | | | |
|---|---|---|---|---|
| | | | | |
| |Mete|or| | |
| | | |LLaVA-OneVision| |
|LLaVA-Next| | | |158K 760K<br><br>|
|LLaVA| | | |1.1M 3.2M<br><br>|

Scaling up visual instruction tuning datasets improves text-only mathematical reasoning. Here, we explore whether enhancing math reasoning in a visual context can improve standard text-only math reasoning. We evaluate four LLVMs on the GSM8K (Cobbe et al., 2021) dataset in an 8-shot setting using Chain-of-Thought (CoT) prompting (Wei

70

Perf.onGSM8K(8shotCoT)

60

50

40

30

20

- et al., 2022). As shown in Figure 4, we observe that models performing well in visual math contexts also achieve strong performance on GSM8K. Moreover, as the scale of the dataset used for training increases, so does model performance. This suggests that using high-quality, large-scale datasets (e.g., rationalestyle datasets, as used in Meteor) is beneficial, and that there is compatibility between visual math and text-only math reasoning, aligning with the datacentric AI perspective (Xu et al., 2023; Zhou et al., 2024).

30 40 50 60

Perf. on MathVista

Figure 4: We present performance on the GSM8K dataset using 8-shot Chain-of-Thought prompting. Additionally, we demonstrate that scaling up the instruction-tuning dataset enables LLVMs to solve text-only math reasoning problems more effectively.

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

(a) (b)

Figure 6: We present robustness performance under occlusion conditions. (a) ViT variant vision encoders demonstrate greater robustness to occlusion compared to ResNet-50. (b) LLVMs also show robustness to occlusion, benefiting from the use of ViT encoders.

- 3.5 ARE LLVMS ROBUST TO OCCLUSIONS?

Existing studies (Naseer et al., 2021; Vishniakov et al., 2023) have demonstrated that ViT models exhibit a remarkable degree of robustness to occlusions, such as patch dropping, than CNN counterparts. Since most LVLMs utilize CLIP ViT-L as their vision encoder, we aim to explore whether this robustness transfers to LVLMs in scenarios involving occluded images. Following the simple masking method presented in prior work (Naseer et al., 2021), we manipulate the input image I = {xi}Ni=1, where N represents the number of patches. Specifically, we mask out N′ patches (where N′ < N) by setting the pixel values of these patches to zero, creating an occluded image, denoted as I˜. We then apply three distinct occlusion methods to the image I: (1) Random PatchDrop: A subset of N′ patches is randomly selected and dropped from the image, effectively simulating random occlusion; (2) Salient PatchDrop: We strategically select and drop salient patches by leveraging the self-supervised ViT model dino-small (Caron et al., 2021); (3) Non-Salient PatchDrop: In this case, we drop non-salient, background patches, retaining only the salient information. This method also utilizes dino-small, following a similar approach to the Salient PatchDrop but focusing on removing the background regions. Figure 5 presents example images with different occlusion methods applied.

LLVMs are robust against occlusion. Before evaluating LLVMs on occluded images, we first verify whether ViT-based encoders are more robust than their CNN counterparts in this scenario. To do this, we assess several ViT variants and ResNet-50 (He et al., 2016) on occluded ImageNet (Krizhevsky et al., 2009) images, applying the same masking process as mentioned above. As shown in Figure 6 (left), compared to ResNet-50, ViT variants demonstrate greater robustness in occlusion scenarios, consistent with findings from the prior study (Naseer et al., 2021). Due to this robustness, LLVMs also exhibit relatively strong performance under occlusion. This result is surprising given that in the AI2D dataset, which contains text-rich diagram images, 50%–70% of the patches are missing, yet LLVMs can still provide correct answers to some extent. This may be because AI2D involves selecting one answer from multiple options, suggesting the possibility of selection bias (Zheng et al., 2023a), a significant issue that we leave for future work.

- 3.6 DO LLVMS PRESERVE CROSS-MODAL ALIGNMENT?

In the de facto structure of LLVMs, a projector fP enables LLMs to perceive and understand images by transforming visual representations into the LLM’s representation space. While a recent work (McKinzie et al., 2024) suggests that the type of projector has minimal impact on performance, other studies (Zhai et al., 2024; Verma et al., 2024) have argued that the projector have limitations in preserving cross-modal understanding and issues such as catastrophic forgetting. In this work, we investigate (1) how effectively a trained projector maintains its visual recognition capability relative to the LLVM’s original vision encoders (e.g., CLIP-ViT-14 for LLaVA-NeXT), and (2) how well a trained projector preserves cross-modal alignment, based on the platonic representation hypothesis (Huh et al., 2024), compared to representation expressivity without alignment learning.

Models Caltech100 CIFAR-100 Food101 Pets Country211 EuroSAT AirCraft Avg. CLIP ViT 336 85.2 76.9 92.9 93.0 31.7 59.0 33.2 67.4 LLaVA-1.5 44.4 (▼ 40.8) 50.8 (▼ 26.1) 57.9 (▼ 35.0) 73.0 (▼ 20.0) 12.2 (▼ 19.5) 11.8 (▼ 47.2) 17.6 (▼ 15.6) 38.2 (▼ 29.2) CLIP ViT 14 85.2 76.3 92.4 93.2 27.5 57.7 32.9 66.5 LLaVA-NeXT 57.2 (▼ 28.0) 56.3 (▼ 20.0) 64.5 (▼ 27.9) 76.1 (▼ 17.1) 14.6 (▼ 12.9) 23.8 (▼ 33.9) 18.3 (▼ 14.6) 44.4 (▼ 22.1)

Table 3: We report the Top-1 accuracy (%) of LLVMs and their corresponding vision encoder models on 1K subsampled datasets from Caltech100, CIFAR-100, Food101, Pets, Country211, EuroSAT, and AirCraft.

LLVMs struggle to preserve the original visual understanding capability. Ideally, after alignment and visual instruction tuning, LLVMs should retain the visual perception abilities of their original vision encoders, allowing them to understand and classify images effectively. To assess this, we evaluate LLVMs on zero-shot image classification tasks using widely adopted datasets such as Caltech100 (Higgins et al., 2017), Food101 (Bossard et al., 2014), CIFAR-100 (Krizhevsky et al., 2009), Pets (Parkhi et al., 2012), Country211 2, EuroSAT (Helber et al., 2019), and AirCraft (Maji et al., 2013). Following the method in previous work (Zhai et al., 2024), we use ChatGPT (gpt-3.5-turbo) (OpenAI, 2023) to extract a single label with the use of prompt: Is this prediction correct?. As shown in Table 3, the performance of LLVMs significantly degrades across all datasets compared to their vision encoders, suggesting that LLVMs do not fully retain the perception capabilities of their original vision encoders. This may be due to: (1) LLVMs being trained to solve complex tasks (e.g., chart or math reasoning) with the use of instruction, which may cause them to lose basic perception abilities (e.g., recognizing simple objects), a phenomenon known as catastrophic forgetting (Zhai et al., 2024) 3, and (2) the vision encoder’s relatively small parameter size (307M for CLIP ViT-L/336px) compared to the LLM (7B for Vicuna-1.5), which could result in a loss of visual perception capability during projection, as the more powerful LLM dominates.

LLVMs lose the ability to understand and interpret shared world concepts. Beyond visual recognition capabilities, we analyze cross-modal alignment based on the platonic representation hypothesis (Huh et al., 2024), which argues that neural networks, despite being trained on different objectives, data, and modalities, should converge to a shared statistical model of reality in their representation spaces. To measure representation similarity between two modalities, the original authors of this hypothesis use mutual nearest-neighbor alignment metrics, a type of kernel-alignment metric. In our work, we assess how much alignment is lost after visual instruction tuning by applying this metric within the context of the platonic representation hypothesis. We evaluate 10 LLMs and measure alignment between these LLMs and vision encoders (LLVMs) using the DOCCI (Onoe et al., 2024) dataset which contains long image descriptions requiring localized scene understanding. As shown in Figure 7, after visual instruction tuning, both LLaVA-1.5 and LLaVA-NeXT show degraded alignment performance

0.3

CLIP-L/336px

LLaVA-1.5

0.2

bloom0.55bbloom1.1bbloom1.7bbloom3bbloom7bolmo1bolmo7bgemma2bgemma7bvicuna1.57b

0.30

CLIP-L/14

0.25

LLaVA-NeXT

0.20

bloom0.55bbloom1.1bbloom1.7bbloom3bbloom7bolmo1bolmo7bgemma2bgemma7bvicuna1.57b

Figure 7: We present how alignment preservation changes (CLIP → LLaVA) in the representation space across various LLM families, BLOOM (Le Scao et al., 2023), OLMo (Groeneveld et al., 2024), Gemma (Team et al., 2024), Vicuna (Chiang et al., 2023), with different parameter sizes on the DOCCI dataset.

with respect to representations compared to their original vision encoder. This suggests doubts about the actual role of the projector in causing the degradation in alignment preservation. From this observation, we speculate that current LLVMs are trained on a variety of datasets to achieve generalization (i.e., multi-task learning). However, during visual instruction tuning, the models might overemphasize capabilities requiring complex cognition while potentially reducing representations related to other tasks, such as localized scene understanding (i.e., DOCCI). This results in a lower alignment

2https://github.com/openai/CLIP/blob/main/data/country211.md 3On the CLEVR/Count (Johnson et al., 2017) dataset, we observed a 16.6% performance improvement in

the LLaVA-NeXT model compared to the previous vision encoder (i.e., CLIP-ViT-L/14.)

score and catastrophic forgetting, as shown in Table 3. For future work, one potential direction is to develop a localized enhanced alignment module similar to HoneyBee (Cha et al., 2024).

- 3.7 MODEL BEHAVIOR: WHICH MODALITY AND LAYERS ARE MORE IMPORTANT?

Here, we conduct an in-depth analysis of model behavior to assess the importance of either a layer or a visual token when performing downstream tasks. We hypothesize that if adding arbitrary noise to a specific component—either a layer block or a visual token—results in a significant drop in model performance, then that component is crucial and actively involved in the model’s reasoning process. To quantify this, we define an importance score (I) inspired by the concept of “sharpness of minima” (Keskar et al., 2016; Lee et al., 2024f). This concept aims to identify flat minima, which promote stable training and better generalization, by measuring the sensitivity of the training function around a local minimum. In our work, we adapt this concept for the inference stage.

Definition 3.1 (Importance Score). Let xt ∈ Rd be the d-dimensional input embedding for a target subject t. For xt, we define the constraint candidate set Ct for t as:

Ct = {zt ∈ Rd : −ϵ + |xt| ≤ zt ≤ ϵ + |xt|}, ϵ ∼ Uniform(−1,1), (2) where zt is a noise vector. The importance score I for target t is then defined as:

f(xt) − maxz

t∈Ct f(xt + zt) f(xt) × 100. (3)

It :=

Note that while the concept of “sharpness of minima” was originally used to find flat minima during training by defining a square-bound constraint set, this is feasible because the model is trained via stochastic gradient descent (SGD), which indirectly allows the evaluation of all noise values z in the constraint set C. However, since our experiment focuses on downstream task performance during inference, we adapt this concept by sampling K noise vectors, {zt1,zt2,...,ztK} ∼ Ct, with different random seeds. For computational efficiency, we set K = 10.

1.0

[Figure 19]

0.912 1.000 0.872 0.944 0.704 0.936

LLVMs strongly focus on the center of the image. To assess the extent to which LLVM utilizes visual token information, we add a noise vector zt to each visual token information based on the importance score (I). However, performing computations on each individual visual token is computationally intensive, especially for models such as LLaVA-1.5-7B that process 576 visual tokens arranged in a 24 × 24 grid of patches. To accelerate computation and reduce complexity, we adopt the same process as 3.3. As shown in Figure 8, LLaVA-1.5-7B places strong emphasis on the central part of the images in the MM-Vet dataset, while the edge regions have minimal influence on the final performance compared to the central visual tokens. This result suggests that not all visual tokens are necessary, aligning with recent works (Cha et al., 2024; Xue et al., 2024) that reduce redundant visual tokens in the projector to enhance efficiency.

0.8

0.664 0.648 0.368 0.272 0.504 0.368

0.6

0.320 0.304 0.008 0.056 0.248 0.240

0.224 0.392 0.072 0.072 0.096 0.240

0.4

0.376 0.000 0.048 0.120 0.032 0.256

0.2

0.352 0.544 0.640 0.464 0.560 0.720

0.0

Figure 8: We report the degree of utility of group-wise visual tokens for LLaVA 1.5 7B on the MM-Vet dataset. Darker regions indicate that the LLVM relies heavily on information from those specific group parts.

Layer-wise

Modality

1.0

1.0

| | | | |
|---|---|---|---|
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

Lower block layers in LLVMs are more important. Figure 9 (left) shows that the lower layers (< 6) play a crucial role in handling the integrated capabilities required for tasks in the MMStar dataset. This suggests that these layers contain more beneficial representations for perception and cognition. This finding aligns with recent work on LLVMs, specifically TroL (Lee et al., 2024b), which introduces the concept of “layer traversal.” This technique revisits layer representations, resulting in a highly generalizable model despite its small size (1.8B parameters). In their paper (Figure 6), the traversal pattern is more pronounced in the lower layers, which is consistent with our findings. Therefore, we believe our results may provide insights into why traversing layers leads to improved generalization.

Importance

0.5

0.5

0.0

0.0

1 10 20 30

1 10 20 30

Figure 9: We present the results of (left) layerwise importance and (right) modality importance within the layers.

Textual modality is more important than visual modality. In addition to layer-wise importance, we measure modality importance using the score II

IT , which calculates the relative importance of textual and image modalities. Specifically, to obtain the image modality importance score II, we feed noise vectors to the positions corresponding to image tokens (e.g., 576 tokens in LLaVA-1.5), and

vice versa for text modality IT. As shown in Figure 9 (right), until the lower layers (< 8), the image modality is more important than the textual modality. However, as layers progress, we observe that the textual modality becomes increasingly important, likely because generating responses requires a stronger focus on text with the perspective of autoregressive modeling. This suggests that LLVMs initially focus on global image perception (section 3.3), and by the middle layers, they have processed the image and shifted toward solving complex user queries to generate coherent answers. Similarly, in TroL, layer traversal occurs more actively in the lower layers, which we interpret as the model attempting to better comprehend the image when it fails to do so in a single pass, enabling it to solve complex reasoning tasks more effectively. These findings highlight the value of strong visual perception, which may explain the success of models utilizing large visual tokens (Wang et al., 2024; Li et al., 2024a) or high-resolution image processing (Li et al., 2024a).

- 4 DISCUSSIONS

Building more interactive evaluation benchmarks. As mentioned in section 3.4, LLVM can effectively solve problems even without seeing the input image. However, current evaluation benchmarks are designed for single-turn interactions and lack applicability to real-world, interactive scenarios. For example, in standard OCR tasks, we typically assess whether the LLVM correctly transcribes text from an image. But consider a practical situation: you’re traveling in a foreign country and visiting a local restaurant. Translating the menu is challenging, and while an application with strong OCR capabilities would be helpful, this is only the first step. When ordering, the LLVM should not only recognize the menu items but also understand the user’s preferences — what they like or dislike — by incorporating knowledge of their persona (Lee et al., 2022). Therefore, future benchmarks should be more interactive and socially grounded (Zhou et al., 2023a), extending beyond multiple-choice, binary, or non-interactive free-form tasks. These benchmarks should involve multi-turn interactions and be based on the user’s preferences (Lee et al., 2024g) or persona in long-term social interactions (Jang et al., 2023; Lee et al., 2024h).

A new paradigm enhancing cross-modal alignment. Current LLVMs have widely adopted the model-stitching structure, which demonstrates impressive capabilities on tasks requiring higher-level cognitive reasoning. However, they exhibit significantly degraded performance in zero-shot image classification tasks (Table 3). Additionally, they cannot effectively preserve alignment (Figure 7) in terms of relatively simple perception when compared to text-rich images (e.g., charts, mathematical expressions). Recently, although recent studies (Li et al., 2024a; Lee et al., 2024c) has been extensively scaling up model sizes with larger datasets to achieve higher performance on increasingly difficult tasks — which we believe is the correct direction - we think it is necessary to deeply consider innovative model architectures (e.g., layer of traversal (Lee et al., 2024b), hidden dimension expansion (Lee et al., 2024a)) to enhance cross-modal alignment at least once. For example, in recent unified architectures (Xie et al., 2024; Erfei Cui, 2024), enabling LLMs to generate images is akin to how drawing can be substantially more challenging for humans than simply viewing a picture. This is because drawing requires a comprehensive and simultaneous understanding of complex world concepts such as relationships between objects, lighting, perspective, and more. Therefore, by projecting visual imagination abilities (Lu et al., 2022b; Lee et al., 2023) onto LLVMs to enable them to generate images, it might significantly help in better preserving cross-modal alignment.

- 5 CONCLUSION

In this paper, we systemically reveals intriguing properties of LLVMs with respect to permutation invariance, robustness, alignment preserving, and importance under various image settings such as occlusion and synthesized images. We hope these findings will assist academic researchers and ML developers in advancing the next frontier of LLMs by providing a foundational basis for future model design choices.

REFERENCES

Haider Al-Tahan, Quentin Garrido, Randall Balestriero, Diane Bouchacourt, Caner Hazirbas, and Mark Ibrahim. Unibench: Visual reasoning requires rethinking vision-language beyond scaling. arXiv preprint arXiv:2408.04810, 2024.

Anthropic. The claude 3 model family: Opus, sonnet, haiku. https: //www.anthropic.com, 2024. URL https://www-cdn.anthropic.com/ de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf.

Yamini Bansal, Preetum Nakkiran, and Boaz Barak. Revisiting model stitching to compare neural representations. Advances in neural information processing systems, 34:225–236, 2021.

Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101–mining discriminative components with random forests. In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part VI 13, pp. 446–461. Springer, 2014.

Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, et al. Internlm2 technical report. arXiv preprint arXiv:2403.17297, 2024.

Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 9650–9660, 2021.

Junbum Cha, Wooyoung Kang, Jonghwan Mun, and Byungseok Roh. Honeybee: Locality-enhanced projector for multimodal llm. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13817–13827, 2024.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024a.

Xuweiyi Chen, Ziqiao Ma, Xuejun Zhang, Sihan Xu, Shengyi Qian, Jianing Yang, David F Fouhey, and Joyce Chai. Multi-object hallucination in vision-language models. arXiv preprint arXiv:2407.06192, 2024b.

Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024c.

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023. URL https: //lmsys.org/blog/2023-03-30-vicuna/.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

et al. Erfei Cui. Sharegpt-4o: Comprehensive multimodal annotations with gpt-4o, 2024. https:

### //sharegpt4o.github.io/.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.

Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. arXiv preprint arXiv:2404.12390, 2024.

Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, et al. Olmo: Accelerating the science of language models. arXiv preprint arXiv:2402.00838, 2024.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7):2217–2226, 2019.

Irina Higgins, Loic Matthey, Arka Pal, Christopher P Burgess, Xavier Glorot, Matthew M Botvinick, Shakir Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a constrained variational framework. ICLR (Poster), 3, 2017.

Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. The platonic representation hypothesis. arXiv preprint arXiv:2405.07987, 2024.

Jihyoung Jang, Minseong Boo, and Hyounghun Kim. Conversation chronicles: Towards diverse temporal and relational dynamics in multi-session conversations. arXiv preprint arXiv:2310.13420, 2023.

Qirui Jiao, Daoyuan Chen, Yilun Huang, Yaliang Li, and Ying Shen. Enhancing multimodal large language models with vision detection models: An empirical study. arXiv preprint arXiv:2401.17981, 2024.

Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2901–2910, 2017.

Amita Kamath, Jack Hessel, and Kai-Wei Chang. What’s” up” with vision-language models? investigating their struggle with spatial reasoning. arXiv preprint arXiv:2310.19785, 2023.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pp. 235–251. Springer, 2016.

Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang. On large-batch training for deep learning: Generalization gap and sharp minima. arXiv preprint arXiv:1609.04836, 2016.

Junho Kim, Byung-Kwan Lee, and Yong Man Ro. Demystifying causal features on adversarial examples and causal inoculation for robust network by adversarial instrumental variable regression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12302–12312, 2023.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9579–9589, 2024.

Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ili´c, Daniel Hesslow, Roman Castagn´e, Alexandra Sasha Luccioni, Fran¸cois Yvon, Matthias Gall´e, et al. Bloom: A 176bparameter open-access multilingual language model. 2023.

Byung-Kwan Lee, Sangyun Chung, Chae Won Kim, Beomchan Park, and Yong Man Ro. Phantom of latent for large language and vision models. arXiv preprint arXiv:2409.14713, 2024a.

Byung-Kwan Lee, Sangyun Chung, Chae Won Kim, Beomchan Park, and Yong Man Ro. Trol: Traversal of layers for large language and vision models. arXiv preprint arXiv:2406.12246,

- 2024b.

Byung-Kwan Lee, Chae Won Kim, Beomchan Park, and Yong Man Ro. Meteor: Mamba-based traversal of rationale for large language and vision models. arXiv preprint arXiv:2405.15574,

- 2024c.

Byung-Kwan Lee, Beomchan Park, Chae Won Kim, and Yong Man Ro. Collavo: Crayon large language and vision model. arXiv preprint arXiv:2402.11248, 2024d.

Byung-Kwan Lee, Beomchan Park, Chae Won Kim, and Yong Man Ro. Moai: Mixture of all intelligence for large language and vision models. arXiv preprint arXiv:2403.07508, 2024e.

Joonhyung Lee, Jeongin Bae, Byeongwook Kim, Se Jung Kwon, and Dongsoo Lee. To fp8 and back again: Quantifying the effects of reducing precision on llm training stability. arXiv preprint arXiv:2405.18710, 2024f.

Seongyun Lee, Sue Hyun Park, Seungone Kim, and Minjoon Seo. Aligning to thousands of preferences via system message generalization. arXiv preprint arXiv:2405.17977, 2024g.

Young-Jun Lee, Chae-Gyun Lim, Yunsu Choi, Ji-Hui Lm, and Ho-Jin Choi. Personachatgen: Generating personalized dialogues using gpt-3. In Proceedings of the 1st Workshop on Customized Chat Grounding Persona and Knowledge, pp. 29–48, 2022.

Young-Jun Lee, Jonghwan Hyeon, and Ho-Jin Choi. Large language models can share images, too! arXiv preprint arXiv:2310.14804, 2023.

Young-Jun Lee, Dokyong Lee, Junyoung Youn, Kyeongjin Oh, Byungsoo Ko, Jonghwan Hyeon, and Ho-Jin Choi. Stark: Social long-term multi-modal conversation with persona commonsense knowledge. arXiv preprint arXiv:2407.03958, 2024h.

Karel Lenc and Andrea Vedaldi. Understanding image representations by measuring their equivariance and equivalence. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 991–999, 2015.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024a.

Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023.

Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024b.

Jiaang Li, Yova Kementchedjhieva, Constanza Fierro, and Anders Søgaard. Do vision and language models share concepts? a vector space alignment study, 2024c. URL https://arxiv.org/ abs/2302.06555.

Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814, 2024d.

Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxl-lightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929, 2024.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26296–26306, 2024a.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024b.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024c.

Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Yaofeng Sun, et al. Deepseek-vl: towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525, 2024.

Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In The 36th Conference on Neural Information Processing Systems (NeurIPS), 2022a.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, KaiWei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.

Yujie Lu, Wanrong Zhu, Xin Eric Wang, Miguel Eckstein, and William Yang Wang. Imaginationaugmented natural language understanding. arXiv preprint arXiv:2204.08535, 2022b.

Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi. Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151, 2013.

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022.

Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Floris Weers, et al. Mm1: Methods, analysis & insights from multimodal llm pre-training. arXiv preprint arXiv:2403.09611, 2024.

Jack Merullo, Louis Castricato, Carsten Eickhoff, and Ellie Pavlick. Linearly mapping from image to text space. arXiv preprint arXiv:2209.15162, 2022.

Muhammad Muzammal Naseer, Kanchana Ranasinghe, Salman H Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Intriguing properties of vision transformers. Advances in Neural Information Processing Systems, 34:23296–23308, 2021.

Yasumasa Onoe, Sunayana Rane, Zachary Berger, Yonatan Bitton, Jaemin Cho, Roopal Garg, Alexander Ku, Zarana Parekh, Jordi Pont-Tuset, Garrett Tanzer, et al. Docci: Descriptions of connected and contrasting images. arXiv preprint arXiv:2404.19753, 2024.

OpenAI. ChatGPT. https://openai.com/blog/chatgpt/, 2023. OpenAI. Gpt-4v(ision) technical work and authors, 2023. https://openai.com/

contributions/gpt-4v, Last accessed on 2024-02-13.

Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. Cats and dogs. In 2012 IEEE conference on computer vision and pattern recognition, pp. 3498–3505. IEEE, 2012.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Pooyan Rahmanzadehgervi, Logan Bolton, Mohammad Reza Taesiri, and Anh Totti Nguyen. Vision language models are blind. arXiv preprint arXiv:2407.06581, 2024.

Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree Radhakrishnan, De-An Huang, Hongxu Yin, Karan Sapra, Yaser Yacoob, Humphrey Shi, et al. Eagle: Exploring the design space for multimodal llms with mixture of encoders. arXiv preprint arXiv:2408.15998, 2024.

Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivi`ere, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024.

Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9568–9578, 2024.

Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herv´e J´egou. Training data-efficient image transformers & distillation through attention. In International conference on machine learning, pp. 10347–10357. PMLR, 2021.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Gaurav Verma, Minje Choi, Kartik Sharma, Jamelle Watson-Daniels, Sejoon Oh, and Srijan Kumar. Cross-modal projection in multimodal llms doesn’t really project visual attributes to textual space. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 657–664, 2024.

Kirill Vishniakov, Zhiqiang Shen, and Zhuang Liu. Convnet vs transformer, supervised vs clip: Beyond imagenet accuracy. arXiv preprint arXiv:2311.09215, 2023.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Haoning Wu, Zicheng Zhang, Erli Zhang, Chaofeng Chen, Liang Liao, Annan Wang, Chunyi Li, Wenxiu Sun, Qiong Yan, Guangtao Zhai, et al. Q-bench: A benchmark for general-purpose foundation models on low-level vision. arXiv preprint arXiv:2309.14181, 2023.

Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.

Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Demystifying clip data. arXiv preprint arXiv:2309.16671, 2023.

Le Xue, Manli Shu, Anas Awadalla, Jun Wang, An Yan, Senthil Purushwalkam, Honglu Zhou, Viraj Prabhu, Yutong Dai, Michael S Ryoo, et al. xgen-mm (blip-3): A family of open large multimodal models. arXiv preprint arXiv:2408.08872, 2024.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 11975–11986, 2023.

Yuexiang Zhai, Shengbang Tong, Xiao Li, Mu Cai, Qing Qu, Yong Jae Lee, and Yi Ma. Investigating the catastrophic forgetting in multimodal large language model fine-tuning. In Conference on Parsimony and Learning, pp. 202–227. PMLR, 2024.

Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. Large language models are not robust multiple choice selectors. In The Twelfth International Conference on Learning Representations, 2023a.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023b.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36, 2024.

Xuhui Zhou, Hao Zhu, Leena Mathur, Ruohong Zhang, Haofei Yu, Zhengyang Qi, Louis-Philippe Morency, Yonatan Bisk, Daniel Fried, Graham Neubig, et al. Sotopia: Interactive evaluation for social intelligence in language agents. arXiv preprint arXiv:2310.11667, 2023a.

Yiyang Zhou, Chenhang Cui, Jaehong Yoon, Linjun Zhang, Zhun Deng, Chelsea Finn, Mohit Bansal, and Huaxiu Yao. Analyzing and mitigating object hallucination in large vision-language models. arXiv preprint arXiv:2310.00754, 2023b.

- A DESCRIPTION OF EVALUATION BENCHMARKS

- • MM-Vet (Yu et al., 2023) dataset is a benchmark designed to evaluate large vision-language models (LVLMs) across six core vision-language (VL) capabilities: recognition, knowledge, optical character recognition (OCR), spatial awareness, language generation, and mathematical reasoning. The dataset includes open-ended, real-world questions based on image-text pairs, requiring models to integrate multiple capabilities to solve complex tasks. MM-Vet benchmark consists of 200 images paired with 218 open-ended questions.
- • Q-Bench (Wu et al., 2023) evaluates the capabilities of large vision-language models in three main areas related to low-level vision tasks. These tasks focus on evaluating how well LVLMs can perform basic low-level perception tasks that are traditionally associated with human visual perception. In the Q-Bench dataset, the questions are of three types: Yes-or-No, What, and How.

- – Low-Level Visual Perception: Assesses how accurately LVLMs can answer questions about low-level image attributes (e.g., clarity, color, distortion). LLVisionQA dataset includes 2,990 images, each with a corresponding question about low-level features.
- – Low-Level Visual Description: Evaluates the ability of LVLMs to describe images. LLDescribe dataset has 499 images with expert-labeled descriptions averaging 58 words each. LVLMs are compared against these to assess completeness, preciseness, and relevance.
- – Visual Quality Assessment: Evaluates LVLMs’ ability to predict quantifiable quality scores for images by assessing how well they align with human-rated mean opinion scores (MOS) on low-level visual appearances, using 81,284 samples.

- • SQA-IMG (Lu et al., 2022a) is a portion of the Science Question Answering (SQA) dataset that contains questions from a wide range of scientific domains, each paired with corresponding image contexts. The dataset includes 10,332 examples of multimodal multiplechoice questions, along with lectures and explanations that detail the reasoning behind the correct answers.

- • ChartQA (Masry et al., 2022) dataset is a benchmark designed to test AI models on their ability to perform question-answering tasks over various types of charts. It focuses specifically on questions requiring complex reasoning, such as visual and logical interpretation, going beyond simpler template-based datasets. ChartQA includes 9,608 human-authored open-ended questions as well as 23,111 questions that are automatically generated from chart summaries.
- • SEED-IMG (Li et al., 2023), a subset of SEED-Bench, focuses on evaluating spatial comprehension of images by testing models on various dimensions like scene understanding, object identification, and spatial relationships. In terms of scale, the dataset includes 19,000 multiple-choice questions that evaluate both image and video comprehension, covering 12 evaluation dimensions such as scene understanding, instance identity, spatial relations, and action recognition.
- • MME (Fu et al., 2023) evaluates both perception and cognition abilities of LVLMs. It features 14 subtasks, including recognition tasks (such as object existence, count, position, color) and reasoning tasks (such as commonsense reasoning, numerical calculation, text translation, and code reasoning). MME uses manually created instruction-answer pairs, ensuring no overlap with public datasets. MME uses ”yes/no” responses for quantitative evaluations.
- • MathVista (Lu et al., 2023) is a benchmark designed to evaluate the mathematical reasoning capabilities of foundation models in visual contexts. It integrates challenges from diverse mathematical and visual tasks, with a focus on fine-grained, deep visual understanding and compositional reasoning. MathVista consists of 6,141 examples including 3,392 multiple-choice questions and 2,749 free-form questions derived from 28 existing multimodal datasets and 3 newly created datasets: IQTest, FunctionQA, and PaperQA.
- • LLaVA-W (Liu et al., 2024c) is a challenging evaluation benchmark created to assess the generalization and instruction-following capabilities LVLMs in complex, real-world situations. It consists of 24 images and 60 questions, including diverse scenes like indoor environments, outdoor settings, memes, paintings, and sketches. Each image is associated with a highly detailed and manually curated description, and the questions focus on extracting intricate details and reasoning about the visual content. LLaVA-W involves a variety of tasks, including detailed descriptions, conversational answers, and complex reasoning.
- • MMStar (Chen et al., 2024a) is a vision-dependent multimodal benchmark designed to evaluate the multimodal capabilities of LVLMs. It addresses two main issues identified in previous benchmarks: the reliance on textual information without visual input and data leakage during training. MMStar is composed of 1,500 samples carefully selected to ensure that visual content is necessary for solving each problem. MMStar evaluates six core capabilities across 18 detailed axes, which include tasks like image perception and logical reasoning. MMStar uses multiple-choice as the primary answer type.
- • MMVP (Tong et al., 2024) evaluates the visual grounding capabilities of large visionlanguage models by identifying scenarios where they fail to distinguish simple visual patterns in images. These patterns include aspects like orientation, counting, viewpoint, and relational context. The benchmark is constructed using 150 pairs of images, resulting in 300 multiple-choice questions.

- B DESCRIPTION OF EVALUATION LVLMS

- • LLaVA-1.5 (Liu et al., 2024a) incorporates academic task-oriented datasets to enhance performance in VQA tasks and features an MLP vision-language connector, which improves upon the original linear layer utilized in LLaVA (Liu et al., 2024c). It uses CLIP ViT-L/14 (Radford et al., 2021) with a 336px resolution as its vision encoder, resulting in a total of (336/14)2 = 576 visual tokens. LLaVA-1.5 is built on Vicuna with either 7B or 13B parameters. The training dataset includes 558K samples for pre-training and 665K for fine-tuning, totaling 1.2M image-text pairs from publicly available datasets
- • LLaVA-NeXT (Liu et al., 2024b) (also known as LLaVA-1.6) enhances visual reasoning, OCR, and world knowledge, offering four times higher image resolution (up to 1344x336)

- and improved performance in visual conversations. Its architecture includes a CLIP ViTL/14 as a vision encoder, paired with Vicuna models ranging from 7B to 34B as a backbone language model. It utilizes 1.3M visual instruction tuning data samples for training, maintaining efficiency with approximately one day of training on 32 A100 GPUs. The architecture’s high resolution and dynamic grid scheme improve detailed image processing capabilities.
- • LLaVA-OneVision (Li et al., 2024b) is a LVLM designed for task transfer across singleimage, multi-image, and video scenarios, with strong capabilities in video understanding through image-to-video task transfer. Its architecture consists of a Qwen2 language model (Yang et al., 2024) with 8B to 72B parameters, and the SigLIP vision encoder (Zhai et al., 2023), which processes images at a base resolution of 384x384, producing 729 visual tokens. The model employs a 2-layer MLP projector. The training utilized 3.2M singleimage data samples and 1.6M multi-modal data samples, focusing on high-quality visual instruction tuning data to enhance its multimodal capabilities.
- • Meteor (Lee et al., 2024c) is a large vision-language model that uniquely embeds multifaceted rationales using a Mamba-based architecture (Gu & Dao, 2023), enabling efficient processing of lengthy rationales to enhance its vision-language understanding. This approach allows Meteor to achieve superior performance without scaling up model size or employing additional vision encoders. Its architecture includes a CLIP-L/14 vision encoder with an image resolution of 490x490, comprising 428M parameters, and InternLM27B (Cai et al., 2024) as a foundational LLM. Meteor was trained on 2.1M question-answer pairs, with 1.1M curated triples.
- • TroL (Lee et al., 2024b) uses a unique characteristic called layer traversing, which reuses layers in a token-wise manner, allowing it to simulate retracing the answering process without physically adding more layers, making it efficient despite smaller model sizes. TroL uses CLIP-L and InternViT as vision encoders, containing 428M and 300M parameters, respectively, and supports 24 layers. The image resolution is adjusted using MLPs in the vision projector. For its foundational LLM, TroL utilizes Phi-3-mini with 3.8B parameters and InternLM2 with 1.8B and 7B parameters. The training dataset comprises 2.3M visual instruction tuning samples.

