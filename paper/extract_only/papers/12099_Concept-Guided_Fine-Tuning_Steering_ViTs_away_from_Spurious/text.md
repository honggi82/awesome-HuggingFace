## Concept-Guided Fine-Tuning: Steering ViTs away from Spurious Correlations to Improve Robustness

Yehonatan Elisha Tel Aviv University

Oren Barkan The Open University

Noam Koenigstein Tel Aviv University

# arXiv:2603.08309v2[cs.CV]15Mar2026

### Abstract

Vision Transformers (ViTs) often degrade under distribution shifts because they rely on spurious correlations, such as background cues, rather than semantically meaningful features. Existing regularization methods, typically relying on simple foreground-background masks, which fail to capture the fine-grained semantic concepts that define an object (e.g., “long beak” and “wings” for a “bird”). As a result, these methods provide limited robustness to distribution shifts. To address this limitation, we introduce a novel finetuning framework that steers model reasoning toward concept-level semantics. Our approach optimizes the model’s internal relevance maps to align with spatially grounded concept masks. These masks are generated automatically, without manual annotation: class-relevant concepts are first proposed using an LLM-based, label-free method, and then segmented using a VLM. The finetuning objective aligns relevance with these concept regions while simultaneously suppressing focus on spurious background areas. Notably, this process requires only a minimal set of images and uses half of the dataset classes. Extensive experiments on five out-of-distribution benchmarks demonstrate that our method improves robustness across multiple ViT-based models. Furthermore, we show that the resulting relevance maps exhibit stronger alignment with semantic object parts, offering a scalable path toward more robust and interpretable vision models. Finally, we confirm that concept-guided masks provide more effective supervision for model robustness than conventional segmentation maps, supporting our central hypothesis. Our code is provided at: https://github.com/yonisGit/cft

### 1. Introduction

Modern ViTs [20, 29] achieve remarkable performance on standard benchmarks like ImageNet [18], yet their robustness under distribution shifts remains limited. A growing body of evidence shows that these models often rely on spu-

[Figure 1]

Figure 1. Motivation for CFT: Relevance maps produced by ViTs often concentrate on spurious background cues rather than semantically meaningful concepts. The figure illustrates this issue using ViT-B on ImageNet-A and ImageNet-R, showing relevance maps before and after applying CFT. By encouraging the model to focus on class-relevant, discriminative concepts, CFT substantially improves the semantic alignment of relevance maps. Notably, after CFT, the model highlights meaningful object parts, such as the beak and wings of the bird (top row) or the fins and mouth of the fish (bottom row), despite never being fine-tuned on these datasets.

rious correlations, such as background textures or contextual cues, rather than the semantic content of the target object [24, 30]. This reliance manifests as catastrophic failures on out-of-distribution (OOD) data, including natural adversarial examples [31], images with altered viewpoints [4], or artistic renditions [56]. While such behavior may be sufficient for in-distribution accuracy, it undermines trustworthiness in real-world deployment, where environmental conditions are rarely controlled.

A promising avenue to improve robustness is to align the model’s internal reasoning with semantically meaningful image regions. Prior work has shown that models relying on object foregrounds exhibit better generalization under dis-

tribution shifts, introducing methods that leverage groundtruth object segmentation masks, for example, by guiding data augmentation strategies [51] or by informing the design of architectural components [60]. However, existing approaches either require extensive retraining or annotated ground-truth segmentation masks. These limitations hinder scalability and practical adoption, especially for large pretrained models where fine-tuning must be both efficient and effective. Moreover, binary foreground–background separation can often be too coarse to support robust recognition, as it treats the foreground as a uniform region and overlooks its internal semantic structure. Consider recognizing a “bird”: robust models should attend to discriminative parts like “wings” and “long beak” (top row of Fig. 1) rather than the entire silhouette. Similarly, relevant features may extend beyond the primary object - a ”branch” can provide contextual evidence for “lorikeet”, while “water” can support “duck” recognition. In this work, we introduce Concept-Guided Fine-Tuning (CFT), a post-hoc framework that steers ViTs toward semantically meaningful reasoning without requiring groundtruth masks or full retraining. CFT operates in three stages. First, an LLM-based, label-free method [40] proposes a set of context-aware semantic concepts per class. Second, a vision-language grounding model (GroundedSAM [44]) spatially localizes these concepts in each training image, producing an adaptive guidance mask. Third, the model is optimized by aligning its relevance map, computed via the transformer-faithful AttnLRP method [2], with this concept-based mask, encouraging high relevance within concept regions while suppressing spurious background cues. A concurrent classification-consistency objective ensures classification accuracy is preserved throughout fine-tuning. Following the protocol of [58], we train on half of ImageNet-1K classes, amounting to only 1,500 images (three per class for half the ImageNet-1K classes) with no manual annotation. Despite this minimal supervision, CFT consistently improves robustness across five OOD benchmarks and three ViT-based models while largely maintaining, and in some cases improving, in-distribution accuracy. The resulting relevance maps exhibit significantly stronger alignment with ground-truth object masks, and robustness gains generalize to held-out classes unseen during fine-tuning, confirming that CFT refines the model’s underlying reasoning rather than memorizing class-specific cues. Taken together, CFT represents a step toward vision models that are both more robust and more interpretable.

### 2. Related Work

Robustness and Shortcut Learning. A primary challenge for modern vision models is their tendency to learn shortcuts, spurious correlations in the training data, such as background textures, that do not generalize [24]. This reliance

limits model robustness on out-of-distribution (OOD) data. Consequently, a suite of challenging benchmarks has been developed to measure this vulnerability, including datasets with natural adversarial examples (ImageNet-A [31]), novel viewpoints and contexts (ObjectNet [4]), artistic renditions (ImageNet-R [30]), sketches (ImageNet-Sketch [56]), and synthetic transformations (SI-Score [19]). Model performance is typically contrasted with in-distribution accuracy on standard benchmarks like ImageNet [18, 46] and its variants (ImageNet-v2 [43]). Our work evaluates extensively on these OOD datasets to demonstrate meaningful improvements in robustness.

Saliency-Guided Model Regularization. One prominent approach to combatting shortcut learning is to explicitly guide the model’s reasoning. This is often achieved by regularizing the model’s explanations to focus on predefined foreground regions. For example, Right for the Right Reasons (RRR) [45] constrains model explanations to match annotated foreground regions via an input-gradient regularizer. GradMask [50] uses saliency-based gradient masking during backpropagation to reduce overfitting, and RRDA [47] employs data augmentation strategies guided by explanation methods to preserve foreground relevance. However, these methods are fundamentally limited by their reliance on this foreground-background dichotomy, which is often insufficient for robust reasoning. Robust recognition often depends on a structured hierarchy of semantic cues, rather than a single undifferentiated foreground region. Furthermore, this approach can be overly restrictive, penalizing focus on relevant context or failing to distinguish between visually similar but semantically different concepts. Beyond this primary conceptual flaw, some of these methods present further gaps: (i) they are typically formulated as regularizers during full training or retraining [45, 47, 55], rendering them less computationally feasible for large-scale, pretrained models, and (ii) many rely on input gradients as a proxy for explanation [45, 50], which is particularly problematic for ViTs, where such explanations can be unstable or unfaithful [35, 39]. In contrast, our method integrates concept-based cues and classifier confidence, rather than relying solely on foreground or background features. In addition, it is applied post hoc as a lightweight finetuning procedure, making it practical even for large-scale models. Finally, our method is fully automatic and does not require any ground-truth segmentation masks.

Vision Models Explainability. Explainable AI has advanced rapidly in recent years, with significant developments across multiple modalities [5, 8, 11, 12, 14, 21– 23, 27, 28]. Explainability methods aim to reveal the reasoning behind model predictions. In vision models, relevance maps can highlight the regions that influence a classifier’s decision, and may expose cases where the model over-

looks salient features (see Fig. 1, middle column). A dominant family of interpretation methods relies on gradients [6– 8, 10, 13, 17, 21, 22, 37, 49], which have been refined by incorporating additional input signals [25, 48, 52, 53]. Other prominent approaches include permutation-based techniques grounded in Shapley values [36, 48] and theorydriven attribution propagation methods, such as Layer-wise Relevance Propagation (LRP) [3, 38], which propagates the output prediction backward through the network. When applied to transformer architectures, initial work demonstrated that combining gradients and attention values can yield viable interpretations [9, 15]. Yet, the technical limitations of purely gradient-based explanations for ViTs have motivated the development of more faithful, propagationbased alternatives. AttnLRP [2] specifically adapts the LRP principle for transformers by properly attributing relevance through the integration of information from both the attention and MLP blocks. This approach yields stable and faithful relevance maps that are better suited for model refinement than raw gradient-based signals. Consequently, the demonstrated stability and faithfulness [2] of AttnLRP make it the clear choice for the explanation backbone of our fine-tuning framework. This choice is further supported by empirical comparisons with alternative saliency methods, provided in the Appendix.

Semantic Guidance from Vision-Language Models. The bottleneck of requiring human-annotated masks is being rapidly obviated by the rise of powerful vision-language models (VLMs) [42]. Models like Grounding DINO [34] and Segment Anything (SAM) [33], combined in tools like GroundedSAM [44], can segment arbitrary semantic concepts in a zero-shot manner from text prompts. This technology unlocks the ability to generate dynamic, conceptlevel guidance maps, moving decisively beyond the insufficient static foreground-background dichotomy. While prior work has used VLMs for tasks like pseudo-labeling [59] or data augmentation [32], their use as a supervisory signal for spatially grounding a model’s internal explanations with specific concepts remains unexplored.

### 3. Method

We propose Concept-guided Fine-Tuning (CFT), a dataefficient framework to improve the robustness of ViTs. CFT aligns the model’s internal relevance with semantically meaningful image regions (concept regions), steering the model away from spurious correlations [24]. CFT performs fine-tuning on a small set of examples to guide the model toward more conceptually grounded reasoning. While our primary focus is on ViTs, we also provide an alternative implementation for CNNs in Sec. 3.2, with additional evaluation results reported in Sec. 4.

#### 3.1. Problem Setup

Let fθ : X → Y be a pretrained ViT, where X is the input image space and Y is the label space. The model is defined by its parameters θ. For an input image I ∈ X, the model produces a prediction fθ(I). We can also compute the model’s relevance map Φ(I;θ), which indicates which parts of the image I were most important for the prediction. Given a small finetuning dataset D = {(Ij,yj)}Nj=1, consisting of N image-label pairs, our goal is to find optimal parameters θ∗. These new parameters should align the model’s relevance map Φ(I;θ∗) with a concept-based semantic mask S(I), without harming classification accuracy. This objective is formulated as finding the parameters θ∗ that minimize a total loss L:

θ∗ = arg min

E(I,y)∼D L(θ,I,y) . (1)

θ

The total loss L combines a relevance loss and a classification loss, which are detailed in Section 3.3.

#### 3.2. Relevance and Semantic Guidance

Relevance Extraction. We compute a patch-level relevance map Φ(I;θ) ∈ [0,1]H×W, where H and W denote the height and width of the ViT patch grid. Relevance is derived using Attention-aware Layer-wise Relevance Propagation (AttnLRP) [2], which backpropagates the class out-

put score through the model. The relevance Φ(iℓ−1) of token

- i at layer ℓ−1 is computed from the relevance Φ(jℓ) of tokens
- j at the next layer ℓ as:

Φ(iℓ−1) =

j

A(ijℓ)Φ(jℓ) k A(kjℓ) + ϵ

, (2)

where A(ijℓ) denotes the attention weight from token i to token j, and ϵ ensures numerical stability. For CNNs,

we adapt AttnLRP by replacing attention maps with intermediate feature representations, following the approach of Barkan et al. [8], combining activation magnitudes with standard LRP relevance scores. We favor LRP-based methods as they satisfy the conservation property, which guarantees that the total relevance propagated through the network sums to the model’s output score. This ensures that relevance maps constitute a faithful redistribution of the prediction signal rather than an arbitrary approximation, making them well-suited as an optimization target in our fine-tuning objective.

Concept Set Creation and Validation. For a dataset D′ containing classes C with P examples per class, we extract class-discriminative textual attributes ξc for each c ∈ C using the procedure of [40]. This produces linguistically interpretable, class-specific candidate concepts. To ensure reliability, we apply an automated validation

step based on visual grounding. Given an image I with label l, we provide its corresponding attribute set ξl to GroundedSAM [44], a zero-shot grounding model combining GroundingDINO [34] with SAM [33]. For each concept k ∈ ξl, GroundedSAM returns corresponding segmentation masks when the concept is visually present, and no mask otherwise. Thus, for each image I and concept k, we obtain a segment set Segk(I), which is empty when k is absent in the image. Concepts are validated according to two criteria: (1) Occurrence Rate — the fraction of images in class c where k is detected, i.e., |{I ∈ D′c : Segk(x) ̸= ∅}|/|D′c|, where D′c is a subset of the D′ containing images of class c, and (2) Spatial Coverage — the mean IoU between

k Segk and the corresponding class-level segmentation mask, measuring how well concepts visually cover their target class regions. Concepts that fail to meet the occurrence criterion are discarded, yielding a validated set of spatially grounded, frequently occurring concepts per class. Although this validation phase produces higher-quality concept sets, one may alternatively use the initial concept set ξc for each class c without validation. As demonstrated in the Appendix, the validation step yields superior results but is not mandatory. This process is performed once prior to the fine-tuning stage.

Semantic Mask Generation. For each image I, we generate a binary semantic guidance mask S(I) ∈ {0,1}H×W. Using the validated concept sets from the previous step, we again employ GroundedSAM [44] to obtain binary segmentation masks Mk(I) for all concepts k. If a concept is not present in I, Mk(I) is set to zero. The final semantic guidance mask S(I) is formed by applying the maximum operator across all individual concept masks.

#### 3.3. Training Objective

The total loss L consists of two weighted components: an alignment loss Lalign and a classification loss Lcls. Alignment Loss. To align the relevance map Φ(I) with the semantic mask S(I), we define two complementary terms.

The first term, Lconcept, promotes high attribution within the concept regions by minimizing the following objective over all concept pixels:

1 |S| p∈S

log Φp(I), (3)

Lconcept = −

where Φp(I) denotes the relevance value at pixel p, and S indexes the set of concept pixels where S(I) = 1. This term drives the attribution values inside the concept mask toward their maximum.

The second term, Lnon-concept, suppresses spurious attribution in background regions:

###### 1 |S¯|

Lnon-concept = −

log 1 − Φp(I) , (4)

p ∈ S¯

where S¯ indexes all pixels where S(I) = 0. This term penalizes any residual relevance assigned to non-concept areas. The total alignment loss combines these two terms:

Lalign = λconcept Lconcept + λnon-concept Lnon-concept. (5)

Classification Loss. In the absence of an explicit regularization objective, fine-tuning drives the model to produce explanations that closely resemble the ground-truth segmentation but at the expense of a severe drop in accuracy. To prevent this collapse, it is essential to introduce an auxiliary loss that constrains the fine-tuned model’s output distribution to remain consistent with that of the original model. To achieve this balance, we incorporate a classificationconsistency loss, defined as follows:

Lcls = CrossEntropy fθ(I), arg max fθ(I) , (6)

where arg max fθ(I) denotes the class predicted by the model fθ for the input image I. The loss computes the cross-entropy between the model’s output distribution and a one-hot target vector that assigns a probability of 1 to the predicted class. In essence, this objective reinforces the model’s confidence in its own predictions by amplifying the probability associated with the predicted class. In Section 4.4, we compare the performance of CFT when using our classification-consistency loss with that achieved using a standard ground-truth cross-entropy loss.

Final Loss. The final loss L is the weighted sum of these two objectives:

L = λalign Lalign + λcls Lcls. (7)

### 4. Experiments

In what follows, we present a comprehensive experimental evaluation of CFT. Our experiments are designed to answer three research questions:

- (i) Does CFT improve robustness on real-world and synthetic out-of-distribution benchmarks?
- (ii) Does CFT produce relevance maps that better align with object foregrounds?
- (iii) Does the benefit of CFT generalize beyond the finetuned classes? We compare CFT against four state-of-the-art baselines that similarly regularize saliency maps during training or fine-tuning: GradMask [50], Right for the Right Reasons (RRR) [45], and RRDA [47]. All experiments are conducted on four modern vision models: DINOv2 [41], ViT-B [20], DeiT-III (DeiT) [54], and ConvNeXt-V2 (CNv2) [57]. All models were sourced from the timm library and utilize their corresponding dataset pretrained weights. For DINOv2, we employ the fine-tuned variant designed for image classification.

#### 4.1. Experimental Setup

Datasets. We evaluate robustness on five standard out-ofdistribution benchmarks:

- 1. ImageNet-A (IN-A) [31]: a collection of natural adversarial examples where standard ImageNet models fail.
- 2. ObjectNet [4]: images with controlled object pose, background, and viewpoint variations.
- 3. ImageNet-R (IN-R) [30]: renditions of ImageNet classes in the form of art, cartoons, and sculptures.
- 4. ImageNet-Sketch (IN-Sketch) [56]: sketch-based depictions of ImageNet categories.
- 5. SI-Score [19]: a synthetic benchmark that systematically varies object location, scale, and rotation.

We use the standard ImageNet validation set (denoted INV) [46] and ImageNet-v2 (denoted IN-V2) [43] as the indistribution reference. For segmentation evaluation, we employ the ImageNet-Segmentation dataset [26], which provides pixel-level masks for a subset of ImageNet classes.

Baselines. We compare CFT against baselines that are most closely aligned with our goal: improve model robustness by modifying its saliency behavior. To ensure fairness, all methods are adapted to our fine-tuning setting:

- 1. GradMask [50]: constrains model explanations to foreground regions via an input-gradient regularizer with human-annotated masks.
- 2. RRR [45]: applies saliency-based gradient masking during backpropagation to reduce overfitting.
- 3. RRDA [47]: employs explanation-guided data augmentation to preserve foreground relevance.

Although these methods were originally designed to operate during full training, this is impractical for modern largescale vision models due to the substantial computational cost. To ensure a fair evaluation, we integrate each baseline objective into a comparable fine-tuning procedure that mirrors our own setup.

- 4.2. Training and Implementation Details

Training Procedure. In most experiments, we follow the protocol of [58], which examined transfer learning on half of the classes. Specifically, we construct a small finetuning dataset by sampling three images per class for half of the classes in ImageNet-1K [18], totaling 1,500 images. This sparse sampling is computationally motivated and tests the method’s data efficiency. We select classes randomly to ensure diverse semantic coverage. All models are initialized from publicly available, ImageNet-1K pre-trained checkpoints. Fine-tuning for all models is performed for 50 epochs using the AdamW optimizer with a batch size of 8. Learning rates are selected via grid search in [5×10−7,5×10−6] for every model. CFT uses fixed loss weights λnon-concept = 1.2, λconcept = 0.5, λalign = 0.8, and λcls=0.2 across all models and datasets.

[Figure 2]

Figure 2. Qualitative examples of CFT correcting prediction failures on OOD datasets using the ViT-B model: the baseline model (Original) misclassifies the images, with relevance maps often highlighting misleading context. Our CFT-finetuned model successfully corrects the prediction (e.g., “scorpion” → “common newt”) by focusing its relevance on the object’s core semantic concepts, demonstrating improved reasoning.

Concept Set Creation. For concept set creation, we used P = 30 samples for every class, guided by the feedback from the occurrence rate and spatial coverage measurements. This process was conducted using occurrence rate ≥15% and spatial coverage ≥20%, and resulted with a total of 1852 concepts over 500 classes (half of the classes in ImageNet-1K [18]). In contrast to Oikarinen et al. [40], who use GPT-3 for concept set creation, we employ the GPT-4omini model [1], while keeping the remainder of the setup consistent with the original work.

All experiments are conducted on NVIDIA A100 GPUs using PyTorch. Code and reproducibility details are available in the Github repository. Further implementation details are provided in the Appendix.

Best results are in bold, second-best are underlined.

#### 4.3. Results

Figure 2 presents qualitative examples from the IN-A and ObjectNet datasets using the ViT-B model, illustrating cases where CFT successfully corrected the model’s predictions. These examples vividly demonstrate the baseline model’s failure mode: a strong reliance on spurious contextual cues. For instance, in the top row (IN-A), the baseline model misclassifies a “common newt” as a “scorpion”, with its relevance map (Original) incorrectly diffusing across the textured background. After fine-tuning with CFT, the model not only corrects the prediction but also shifts its relevance to be tightly concentrated on the object’s body. This provides qualitative evidence that CFT is successfully steering the model’s reasoning from misleading cues toward the core object.

Robustness under distribution shift. Table 1 presents the Top-1 (R@1) and Top-5 (R@5) accuracies across different models and datasets. As shown, CFT consistently achieves

Table 1. Out-of-distribution (OOD) robustness and in-distribution accuracy: metrics are Top-1 and Top-5 accuracy (%).

Dataset Model Metric Original GradMask RRR RRDA CFT

R@1 80.41 80.65 80.89 81.12 81.35

ViT-B

- R@5 94.88 95.03 95.17 95.39 95.51

DINOv2

R@1 81.07 80.95 80.48 81.61 81.44

- R@5 95.20 95.08 94.83 95.53 95.65

IN-V

R@1 82.20 82.04 81.85 82.78 82.61

DeiT

- R@5 95.04 95.23 95.37 95.64 95.77

CNv2

R@1 86.49 86.68 86.74 87.18 87.27

- R@5 95.05 95.24 95.09 95.42 95.71

R@1 68.32 68.51 68.68 69.04 69.19 R@5 83.74 83.95 84.18 84.60 84.77

ViT-B

R@1 71.32 71.55 71.45 72.08 71.91 R@5 87.42 87.61 87.53 88.02 88.15

DINOv2

IN-V2

R@1 72.43 72.68 72.75 72.85 73.11

DeiT

- R@5 87.86 88.05 88.11 88.35 88.58

CNv2

R@1 74.49 74.70 74.83 75.09 75.25

- R@5 88.77 88.98 88.93 89.34 89.50

R@1 13.26 15.37 18.45 25.12 27.76 R@5 32.54 38.29 45.43 59.87 62.75

ViT-B

R@1 14.92 16.73 19.25 25.74 27.71

DINOv2

- R@5 34.22 39.14 46.91 60.01 62.36

DeiT

R@1 15.37 17.65 19.94 25.61 27.72

- R@5 35.43 41.08 47.92 60.09 62.20

CNv2

R@1 16.25 18.46 20.93 25.89 27.93

- R@5 36.07 42.34 49.15 59.88 62.40

IN-A

R@1 33.26 36.44 40.32 51.12 54.28

ViT-B

- R@5 50.54 56.72 62.18 73.02 75.46

DINOv2

R@1 34.93 38.27 42.51 52.42 53.89

- R@5 51.81 58.02 63.17 74.12 75.58

DeiT

R@1 35.42 38.86 43.26 52.81 54.24

- R@5 52.26 58.49 63.83 74.03 75.46

CNv2

R@1 36.18 39.43 43.82 52.54 54.19

- R@5 53.01 59.04 64.21 74.03 75.62

ObjectNet

R@1 30.26 33.19 37.21 47.12 48.47

ViT-B

- R@5 45.54 50.37 56.41 69.02 70.50

DINOv2

R@1 31.43 34.25 38.02 47.93 48.53

- R@5 46.12 51.05 57.12 70.12 70.73

IN-R

R@1 32.14 35.06 38.75 47.68 48.33

DeiT

- R@5 46.77 51.63 57.41 70.04 70.65

CNv2

R@1 32.86 35.78 39.16 47.72 48.37

- R@5 47.28 52.08 57.64 70.03 70.68

R@1 35.49 36.00 36.56 36.85 37.06 R@5 54.89 56.12 57.75 61.94 62.59

ViT-B

R@1 41.32 42.14 42.92 44.50 44.74

DINOv2

- R@5 63.67 64.67 65.65 68.61 68.90

DeiT

R@1 42.43 43.09 43.61 44.50 44.83

- R@5 64.84 65.71 66.52 69.21 69.55

CNv2

R@1 42.49 43.16 43.80 45.87 46.14

- R@5 65.17 66.07 66.93 70.52 70.81

IN-Sketch

substantial performance gains on real-world datasets, including adversarial variants (IN-A) and those featuring randomized or controlled backgrounds, rotations, and viewpoints (ObjectNet). In contrast, the improvement is less pronounced for datasets depicting artistic or abstract representations (IN-R, IN-Sketch), which often lack complex or varied backgrounds. This behavior is expected, as such datasets inherently minimize background biases. Furthermore, while baseline methods largely maintain their accu-

Table 2. Robustness to geometric transformations on the SI-Score benchmark: metrics are Top-1 and Top-5 accuracy (%).

Dataset Model Metric Original GradMask RRR RRDA CFT

R@1 34.26 35.18 36.42 38.54 39.15 R@5 50.54 52.33 54.21 60.95 62.48

ViT-B

R@1 35.11 36.05 37.22 38.95 39.07 R@5 52.85 54.67 56.40 61.33 62.15

DINOv2

SI-location

R@1 34.74 35.41 36.93 38.79 39.08

DeiT

- R@5 51.24 53.06 55.38 60.65 62.04

CNv2

R@1 35.05 36.20 37.14 38.93 39.00

- R@5 52.10 54.45 56.89 61.27 62.11

R@1 40.26 42.15 44.83 50.73 52.15 R@5 56.54 59.07 62.88 69.02 71.48

ViT-B

R@1 41.12 43.26 45.74 50.35 51.92 R@5 58.17 60.32 63.09 69.83 70.94

DINOv2

SI-rotation

R@1 42.03 43.95 46.40 50.27 51.64

DeiT

- R@5 57.96 60.41 63.88 69.10 70.66

CNv2

R@1 41.66 43.75 46.00 50.11 51.57

- R@5 58.05 60.72 63.16 69.04 70.45

R@1 55.26 57.45 59.14 62.72 63.15

ViT-B

- R@5 74.54 77.02 80.19 86.21 87.48

DINOv2

R@1 56.01 58.33 59.88 62.85 63.02

- R@5 75.11 78.46 81.17 86.03 87.12

SI-size

R@1 55.88 57.93 59.54 62.66 63.07 R@5 75.20 77.38 80.46 86.11 87.24

DeiT

R@1 56.14 58.12 59.67 62.91 63.10 R@5 75.48 78.02 81.08 86.25 87.36

CNv2

racy on datasets drawn from the original ImageNet distribution (IN-V and IN-V2), they exhibit clear degradation on real-world out-of-distribution datasets such as IN-A and ObjectNet. This observation suggests that existing methods are less effective at mitigating overfitting to the ImageNet domain. Finally, we can observe that sometimes CFT introduces a minor reduction in accuracy on in-distribution data (IN-V and IN-V2), which can be reasonably interpreted as a result of improved regularization that alleviates overfitting to the training distribution. On the synthetic SI-Score benchmark (Table 2), CFT demonstrates even more pronounced gains. This suggests that concept-focused reasoning inherently improves invariance to geometric transformations, as the model learns to rely on object structure and relevant features rather than absolute position or orientation cues.

Relevance map alignment. To verify that CFT indeed shifts model focus toward the object’s relevant features and foreground information, we evaluate segmentation metrics on relevance maps (Table 3). Using the ImageNetSegmentation dataset [26], we compute pixel accuracy (PA), mean Intersection-over-Union (mIoU), and mean Average Precision (mAP) between relevance maps and ground-truth masks. CFT improves all metrics across all architectures, confirming that our fine-tuning successfully aligns model explanations with object regions.

##### Generalization across classes. To verify that the robust-

- Table 3. Alignment of relevance maps with ground-truth object masks: pixel-level agreement between model relevance maps and human-annotated masks. Additional details are provided in Sec. 4.

Metric ViT-B DINOv2 DeiT CNv2

mIoU 62.91 60.35 60.45 62.64 mAP 78.67 80.14 82.37 84.46 PA 72.23 74.85 77.16 79.32

Original

mIoU 68.23 70.84 72.91 74.21 mAP 84.26 86.45 88.16 89.32 PA 80.34 81.92 83.74 84.58

CFT

ness improvements induced by CFT fine-tuning extend beyond the classes used during training, we evaluate performance separately on training and non-training classes. Table 4 reports the average improvement across models. The results indicate that both subsets achieve comparable gains on robustness benchmarks. As expected, classes included in the training set exhibit slightly higher accuracy on datasets derived from the original ImageNet distribution, reflecting their direct exposure during fine-tuning.

In summary, CFT consistently enhances model robustness across architectures and distribution shifts by explicitly guiding relevance toward concepts and object foregrounds. Its gains generalize to unseen classes and are most pronounced in scenarios where background cues are misleading, a common failure mode in real-world deployment [24].

#### 4.4. Ablation Studies

In what follows, we present three sets of experiments: (1) a comparison between concept-based and object-based segmentation, (2) an ablation study on the loss terms of the training objective, and (3) an evaluation of different saliency methods for generating relevance maps. These results demonstrate the clear advantage of using AttnLRP as the explanation method for CFT compared with alternative approaches.

- Table 5 compares Top-1 accuracy (%) results of conceptbased guidance during fine-tuning (CFT) with objectsegmentation–based guidance (Segmentation) for ViT-B and DINOv2 across all datasets. For this evaluation, we use the same loss function as in CFT, but replace the concept segmentation map S(I) with the ground-truth object segmentation mask. Notably, we also experimented with GroundedSAM [44] by using the class label as a prompt and using the response mask instead of the ground-truth object mask. This approach produced results nearly identical to the Similarity baseline, maintaining the same performance trends. All experiments follow the same training setup as described previously. The goal of this experiment is to assess whether fine-grained semantic concepts provide a superior guidance signal for robustness than uniform ob-

ject segments. As shown in the table, CFT consistently outperforms Segmentation guidance across both in-distribution and out-of-distribution datasets. This highlights the advantage of leveraging concept-based cues to enhance model robustness and, in some cases, improve in-distribution accuracy. While object-segmentation maps provide a reasonable level of robustness, using concept-guided masks further improves performance on standard in-distribution data.

- Table 6 presents the ablation results for the λnon-concept, λconcept, and λcls loss terms using the ViT-B model across all datasets, evaluated by Top-1 accuracy (%). Moreover, we conduct an ablation to assess the role of our classificationconsistency classification loss (Eq. 6) by substituting it with the standard cross-entropy loss computed using the ground-truth label. Results show that performance on INV and IN-V2 is relatively insensitive to the removal of

λnon-concept. In contrast, this term plays a crucial role in outof-distribution datasets, as its absence leads to a significant accuracy drop. Furthermore, λcls proves essential for maintaining robustness, as its removal results in substantial performance degradation. Finally, our ablation study highlights the advantage of employing the classification-consistency loss over the standard ground-truth cross-entropy. While the ground-truth variant maintains slightly higher original accuracy, the classification-consistency loss consistently yields greater improvements in model robustness.

- Table 7 reports CFT Top-1 accuracy (%) performance using alternative relevance methods: Gradient-Rollout [8], IIA [8], and GradCAM [15]. We evaluated ViT-B, on INA and IN-R. Across all evaluations, AttnLRP yields superior performance, highlighting its effectiveness for relevance propagation in the CFT approach.

### 5. Conclusion

We introduced Concept-Guided Fine-Tuning (CFT), a fully automated framework designed to address a key limitation of modern vision models: their reliance on spurious correlations for classification. By steering the model’s internal reasoning away from such cues and toward semantically meaningful concepts, CFT substantially improves OOD robustness. Extensive experiments across five OOD benchmarks demonstrate that CFT consistently outperforms prior saliency-regularization approaches. Importantly, the robustness improvements generalize to classes that are not observed during fine-tuning, indicating that CFT promotes a more robust reasoning process rather than merely replacing one set of cues with another. Our ablation studies further support a central hypothesis: fine-grained semantic concepts provide a significantly stronger supervision signal for robustness than conventional foreground–background segmentation masks. Overall, CFT offers a scalable and interpretable pathway toward more reliable vision models. Lim-

- Table 4. Generalization to unseen classes: robustness evaluation was performed separately on classes included in the fine-tuning set and those excluded from it. Last row reports average change for both training and non-training classes across all models and datasets.

IN-V IN-A IN-R IN-Sketch IN-V2 ObjectNet

Model Train Classes Method

R@1 R@5 R@1 R@5 R@1 R@5 R@1 R@5 R@1 R@5 R@1 R@5

Original 82.02 96.26 14.86 35.26 32.96 47.45 36.85 58.42 71.39 89.86 36.03 55.82

✓

###### CFT 82.39 96.42 22.95 46.10 35.53 50.59 38.16 59.73 71.46 89.92 43.51 64.67 ×

ViT-B

Original 81.04 95.95 17.20 38.74 34.73 49.62 34.02 56.41 71.33 89.85 34.33 53.88 CFT 81.33 96.12 25.43 48.74 37.04 52.17 35.73 57.94 71.21 89.63 41.01 62.33

Original 82.89 96.04 24.10 50.39 38.65 55.75 43.52 65.29 73.22 91.09 38.58 63.68

✓

###### CFT 82.80 96.48 31.45 57.87 43.43 59.68 46.00 67.54 73.67 91.39 44.01 69.76 ×

DINOv2

Original 83.80 96.15 21.42 45.52 41.11 57.52 40.33 63.74 72.11 91.24 41.38 61.54 CFT 84.09 96.53 29.08 53.68 43.59 61.01 40.79 64.96 72.30 91.68 47.24 67.98

Original 82.82 95.89 24.57 50.85 38.70 55.47 43.66 65.44 73.30 90.87 38.38 64.06

✓

###### CFT 83.58 96.22 30.55 57.52 43.63 59.69 46.22 67.20 73.52 91.28 43.55 69.38 ×

DeiT

Original 84.21 95.79 20.96 45.85 41.60 57.07 40.36 63.42 72.13 91.62 40.88 61.55 CFT 84.34 96.17 28.63 53.57 43.69 61.38 40.37 65.21 72.01 91.55 47.73 67.61

Original 83.82 97.29 25.32 51.64 39.84 56.97 44.75 66.52 74.45 92.35 39.81 64.91

✓

###### CFT 84.28 97.56 33.12 59.87 45.36 61.66 47.93 69.48 75.63 92.90 45.96 71.65 ×

CNv2

Original 85.03 97.35 22.68 46.78 42.35 58.74 41.59 64.94 73.34 92.49 42.61 62.77

CFT 85.70 97.48 31.04 55.63 45.55 62.95 42.72 66.88 73.69 92.83 49.17 69.96 Avg. Change (✓) +0.38 +0.30 +7.31 +8.31 +4.45 +4.00 +2.38 +2.07 +0.48 +0.33 +6.06 +6.75 Avg. Change (×) +0.35 +0.27 +7.98 +8.68 +2.52 +3.64 +0.83 +1.62 +0.08 +0.12 +6.49 +7.04

Table 5. Ablation study: Concept-level guidance vs. object-level guidance.

Model Method IN-V IN-A IN-R IN-Sketch IN-V2 ObjectNet SI-location SI-rotation SI-size

Original 81.53 16.09 33.81 35.47 71.13 35.12 33.36 39.18 55.65 CFT 82.57 26.05 36.69 36.28 71.65 42.29 38.60 46.82 61.94 Segmentation 80.25 24.12 35.95 36.01 70.84 41.70 38.13 46.31 61.40

ViT-B

Original 83.15 17.52 35.20 36.83 72.81 36.78 34.63 40.25 57.09 CFT 83.62 27.88 38.02 37.61 73.04 43.91 39.82 48.19 63.11 Segmentation 82.04 25.91 37.14 37.12 72.25 43.15 39.07 47.73 62.85

DINOv2

- Table 6. Ablation study on loss components: we evaluate the impact of removing each of our three main loss terms using the ViT-B model.

Method IN-V IN-A IN-R IN-Sketch IN-V2 ObjectNet SI-location SI-rotation SI-size

Original 81.53 16.09 33.81 35.47 71.15 35.12 33.36 39.18 55.65 CFT 82.57 24.11 36.34 36.28 72.03 42.29 38.60 46.25 62.18 w/o Lcls (Eq. 6) 79.82 17.96 34.29 34.84 69.41 39.65 37.27 43.09 58.46 w/o Lnon-concept (Eq. 4) 82.50 19.14 35.13 36.26 72.01 41.70 34.93 42.31 58.40 w/o Lconcept (Eq. 3) 81.26 24.10 34.37 35.22 71.69 42.14 38.55 45.61 61.79 w/ ground-truth 82.61 21.35 34.56 35.72 72.08 41.55 37.02 44.68 61.36

- Table 7. Ablation study on relevance methods: Top-1 accuracy using different relevance methods.

itations and future work are discussed in the Appendix.

### References

Method GradCAM Gradient-Rollout IIA AttnLRP

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 5

IN-A 25.88 26.43 26.94 27.82 IN-R 43.32 46.92 47.58 48.54

- [2] Reduan Achtibat, Maximilian Dreyer, Ilia Shumailov, Yugeng Liu, Sayak Paul, Jan Philipp Kretzer, and Ullrich K¨othe. Attnlrp: Attention-aware layer-wise relevance propagation for transformers, 2024. 2, 3
- [3] Sebastian Bach, Alexander Binder, Gr´egoire Montavon, Frederick Klauschen, Klaus-Robert M¨uller, and Wojciech Samek. On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. In PloS one, page e0130140. Public Library of Science San Francisco, CA USA, 2015. 3
- [4] Andrei Barbu, David Mayo, Julian Alverio, William Luo, Christopher Wang, Dan Gutfreund, Josh Tenenbaum, and Boris Katz. Objectnet: A large-scale bias-controlled dataset for pushing the limits of object recognition models. In Advances in Neural Information Processing Systems, 2019. 1, 2, 5
- [5] Oren Barkan, Yonatan Fuchs, Avi Caciularu, and Noam Koenigstein. Explainable recommendations via attentive multi-persona collaborative filtering. In Proceedings of the 14th ACM Conference on Recommender Systems, pages 468–473, 2020. 2
- [6] Oren Barkan, Omri Armstrong, Amir Hertz, Avi Caciularu, Ori Katz, Itzik Malkiel, and Noam Koenigstein. Gam: Explainable visual similarity and classification via gradient activation maps. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management, pages 68–77, 2021. 3
- [7] Oren Barkan, Edan Hauon, Avi Caciularu, Ori Katz, Itzik Malkiel, Omri Armstrong, and Noam Koenigstein. Gradsam: Explaining transformers via gradient self-attention maps. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management, pages 2882–2887, 2021.
- [8] Oren Barkan, Yehonatan Elisha, Yuval Asher, Amit Eshel, and Noam Koenigstein. Visual explanations via iterated integrated attributions. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 2073–2084. IEEE,

2023. 2, 3, 7

- [9] Oren Barkan, Yehonatan Elisha, Jonathan Weill, Yuval Asher, Amit Eshel, and Noam Koenigstein. Deep integrated explanations. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, pages 57–67, 2023. 3
- [10] Oren Barkan, Yehonatan Elisha, Jonathan Weill, Yuval Asher, Amit Eshel, and Noam Koenigstein. Stochastic integrated explanations for vision models. In 2023 IEEE International Conference on Data Mining (ICDM), pages 938–943. IEEE, 2023. 3
- [11] Oren Barkan, Veronika Bogina, Liya Gurevitch, Yuval Asher, and Noam Koenigstein. A counterfactual framework for learning and evaluating explanations for recommender systems. In Proceedings of the ACM Web Conference 2024, pages 3723–3733, 2024. 2
- [12] Oren Barkan, Yonatan Toib, Yehonatan Elisha, Jonathan Weill, and Noam Koenigstein. Llm explainability via attributive masking learning. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 9522–

9537, 2024. 2

- [13] Oren Barkan, Yehonatan Elisha, Jonathan Weill, and Noam Koenigstein. Bee: Metric-adapted explanations via baseline exploration-exploitation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1835–1843, 2025. 3
- [14] Oren Barkan, Yahlly Schein, Yehonatan Elisha, Veronika Bogina, Mikhail Baklanov, and Noam Koenigstein. Fidelityaware recommendation explanations via stochastic path integration. arXiv preprint arXiv:2511.18047, 2025. 2
- [15] Hila Chefer, Shir Gur, and Lior Wolf. Transformer interpretability beyond attention visualization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 782–791, 2021. 3, 7
- [16] Nurit Cohen-Inger, Yehonatan Elisha, Bracha Shapira, Lior Rokach, and Seffi Cohen. Forget what you know about llms evaluations–llms are like a chameleon. arXiv preprint arXiv:2502.07445, 2025. 3
- [17] Piotr Dabkowski and Yarin Gal. Real time image saliency for black box classifiers. Advances in neural information processing systems, 30, 2017. 3
- [18] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, pages 248–255, 2009. 1, 2, 5
- [19] Josip Djolonga, Jessica Yung, Michael Tschannen, Rob Romijnders, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Matthias Minderer, Alexander D’Amour, Dan I Moldovan, et al. On robustness and transferability of convolutional neural networks. In CVPR, pages 16453–16463,

2021. 2, 5

- [20] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 1, 4
- [21] Yehonatan Elisha, Oren Barkan, and Noam Koenigstein. Probabilistic path integration with mixture of baseline distributions. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, pages 570–580, 2024. 2, 3
- [22] Yehonatan Elisha, Seffi Cohen, Oren Barkan, and Noam Koenigstein. Rethinking saliency maps: A cognitive human aligned taxonomy and evaluation framework for explanations. arXiv preprint arXiv:2511.13081, 2025. 3
- [23] Ruth Fong, Mandela Patrick, and Andrea Vedaldi. Understanding deep networks via extremal perturbations and smooth masks. In Proceedings of the IEEE International Conference on Computer Vision, pages 2950–2958, 2019. 2
- [24] Robert Geirhos, J¨orn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020. 1, 2, 3, 7
- [25] Jiuxiang Gu, Zhenhua Wang, Jason Kuen, Lianyang Ma, Amir Shahroudy, Bing Shuai, Ting Liu, Xingxing Wang, Gang Wang, Jianfei Cai, et al. Recent advances in convolutional neural networks. Pattern recognition, 77:354–377,

2018. 3

- [26] Matthieu Guillaumin, Daniel K¨uttel, and Vittorio Ferrari. Imagenet auto-annotation with segmentation propagation. International Journal of Computer Vision, 110:328–348,

2014. 5, 6

- [27] Liya Gurevitch, Veronika Bogina, Oren Barkan, Yahlly Schein, Yehonatan Elisha, and Noam Koenigstein. Lxr: Learning to explain recommendations. ACM Transactions on Recommender Systems, 4(2):1–39, 2025. 2
- [28] Ziv Weiss Haddad, Oren Barkan, Yehonatan Elisha, and Noam Koenigstein. Soft local completeness: Rethinking completeness in xai. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 19794–19804, 2025. 2
- [29] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16000– 16009, 2022. 1
- [30] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8340–8349, 2021. 1, 2, 5
- [31] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In CVPR, pages 15262–15271, 2021. 1, 2, 5
- [32] Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. Visual prompt tuning. In ECCV, pages 709–727, 2022. 3
- [33] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 3, 4
- [34] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision, pages 38–55. Springer,

2024. 3, 4

- [35] Yibing Liu, Haoliang Li, Yangyang Guo, Chenqi Kong, Jing Li, and Shiqi Wang. Rethinking attention-model explainability through faithfulness violation test. In International conference on machine learning, pages 13807–13824. PMLR,

2022. 2

- [36] Scott M Lundberg and Su-In Lee. A unified approach to interpreting model predictions. Advances in neural information processing systems, 30, 2017. 3
- [37] Aravindh Mahendran and Andrea Vedaldi. Visualizing deep convolutional neural networks using natural pre-images. International Journal of Computer Vision, 120(3):233–255,

2016. 3

- [38] Gr´egoire Montavon, Sebastian Lapuschkin, Alexander Binder, Wojciech Samek, and Klaus-Robert M¨uller. Explaining nonlinear classification decisions with deep taylor decomposition. Pattern Recognition, 65:211–222, 2017. 3

- [39] Muzammal Naseer, Kanchana Ranasinghe, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Intriguing properties of vision transformers. In Advances in neural information processing systems, pages 23296–23308, 2021. 2
- [40] Tuomas Oikarinen, Subhro Das, Lam M Nguyen, and TsuiWei Weng. Label-free concept bottleneck models. arXiv preprint arXiv:2304.06129, 2023. 2, 3, 5, 1
- [41] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision,

2023. 4

- [42] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021. 3
- [43] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do imagenet classifiers generalize to imagenet? In International conference on machine learning, pages 5389–5400. PMLR, 2019. 2, 5
- [44] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, et al. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159,

2024. 2, 3, 4, 7

- [45] Andrew Slavin Ross, Michael C Hughes, and Finale DoshiVelez. Right for the right reasons: Training differentiable models by constraining their explanations. In IJCAI, pages 2664–2670, 2017. 2, 4, 5
- [46] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115(3):211–252, 2015. 2, 5
- [47] Fl´avio Arthur Oliveira Santos and Cleber Zanchettin. Exploring image classification robustness and interpretability with right for the right reasons data augmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, pages 4147–4156,

2023. 2, 4, 5

- [48] Avanti Shrikumar, Peyton Greenside, and Anshul Kundaje. Learning important features through propagating activation differences. In International conference on machine learning, pages 3145–3153. PMlR, 2017. 3
- [49] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. In arXiv preprint arXiv:1312.6034, 2013. 3
- [50] Becks Simpson, Francis Dutil, Yoshua Bengio, and Joseph Paul Cohen. Gradmask: Reduce overfitting by regularizing saliency. arXiv preprint arXiv:1904.07478, 2019. 2, 4, 5
- [51] Krishna Kumar Singh, Dhruv Mahajan, Kristen Grauman, Yong Jae Lee, Matt Feiszli, and Deepti Ghadiyaram. Don’t judge an object by its context: Learning to overcome contextual bias. In CVPR, pages 11070–11078, 2020. 2

- [52] Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Vi´egas, and Martin Wattenberg. Smoothgrad: removing noise by adding noise. arXiv preprint arXiv:1706.03825, 2017. 3
- [53] Suraj Srinivas and Franc¸ois Fleuret. Full-gradient representation for neural network visualization. Advances in neural information processing systems, 32, 2019. 3
- [54] Hugo Touvron, Alexandre Sablayrolles, Armand Joulin, Matthijs Douze, and Herv´e J´egou. Deit iii: Revenge of the vit. In ECCV, pages 25–41, 2022. 4
- [55] Joseph D Viviano, Becks Simpson, Francis Dutil, Yoshua Bengio, and Joseph Paul Cohen. Saliency is a possible red herring when diagnosing poor generalization. arXiv preprint arXiv:1910.00199, 2019. 2
- [56] Haohan Wang, Songwei Ge, Zachary Lipton, and Eric P Xing. Learning robust global representations by penalizing local predictive power. Advances in neural information processing systems, 32, 2019. 1, 2, 5
- [57] Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, and Saining Xie. Convnext v2: Co-designing and scaling convnets with masked autoencoders. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16133– 16142, 2023. 4
- [58] Jason Yosinski, Jeff Clune, Yoshua Bengio, and Hod Lipson. How transferable are features in deep neural networks? In NeurIPS, 2014. 2, 5
- [59] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for vision-language models. In CVPR, pages 16816–16825, 2022. 3
- [60] Xizhou Zhu, Han Hu, Stephen Lin, and Jifeng Dai. Deformable convnets v2: More deformable, better results. In CVPR, pages 9308–9316, 2019. 2

## Concept-Guided Fine-Tuning: Steering ViTs away from Spurious Correlations to Improve Robustness Supplementary Material

### 6. Implementation Details

Hyperparameters. In Table 8, we summarize the hyperparameter configurations used across all experiments. Our method is highly stable and relies on a consistent set of hyperparameters, with the exception of the learning rate. In contrast, RRR requires model-specific hyperparameter tuning and is notably sensitive to these choices. GradMask also demands careful learning rate tuning for each model. Its performance varies substantially with small adjustments to this parameter, and achieving convergence of the background loss proved challenging in both cases. RRDA shares no common hyperparameters with the other methods aside from the learning rate. CFT uses fixed loss weights λnon-concept = 1.2, λconcept = 0.5, λalign = 0.8, and λcls=0.2 across all models and datasets. We heavily weight the Lnon-concept loss, as reliance on spurious cues (areas without concepts in this case) is the primary issue to be corrected.

Ablation on Concept Validation Thresholds. Following the initial label-free concept discovery procedure of [40], we further refined the resulting pool to obtain a high-quality concept set. To this end, we enforced minimum thresholds of an occurrence rate of at least 15% and spatial coverage of at least 20%. Applying these criteria to IN produced 1,852 validated concepts. Across the dataset, concepts appeared in 29% of images on average, and those satisfying the filtering criteria covered roughly 35% of the relevant region. We examined the effect of varying the occurrencerate and spatial-coverage thresholds on Top-1 accuracy for ViT-B evaluated on IN-A and IN-R. The best results were obtained using our default thresholds of 15% and 20%. Increasing the thresholds to 40%/40% reduced the number of concepts to 694 and led to a noticeable drop in performance (IN-A: 24.59, IN-R: 44.23), presumably because many informative concepts were discarded. Relaxing the thresholds to 5%/10% increased the concept count to 2,435 but introduced substantial noise, which similarly harmed performance (IN-A: 25.13, IN-R: 44.92).

Concept Set Creation. For concept set construction, we used P = 30 samples per class, guided by the occurrence rate and spatial coverage feedback. Using thresholds of occurrence rate ≥ 15% and spatial coverage ≥ 20%, this process yielded a total of 1852 concepts across 500 classes (half of ImageNet-1K [18]). The filtering (occur-

rence rate and spatial coverage feedback) proceeded in two stages: we first applied the occurrence-rate threshold, and then evaluated the remaining candidates using the spatialcoverage criterion. In our experiments, all concepts that passed the occurrence-rate filter also satisfied the 20% coverage threshold. While not required for our study, this procedure could be extended with an iterative refinement step — potentially assisted by an LLM to identify additional concepts that jointly satisfy both constraints.

Clarification on the P parameter. The parameter P is used exclusively during the validation of the initial concept sets. We first generate the initial concept sets using the procedure of Oikarinen et al. [40]. Then, for each class, we examine P = 30 images to compute the occurrence rate and spatial coverage. These measurements are subsequently used to filter and refine the initial concept sets.

### 7. Concept Validation Effect

Effect of the optional concept validation step. We further compared CFT performance with and without the concept validation stage, evaluating Top-1 accuracy on both INA and IN-R. Without validation, CFT achieves 26.01 on INA (vs. 27.92 with validation) and 47.19 on IN-R (vs. 48.51 with validation). Although the validation step provides a consistent performance boost, the non-validated variant remains competitive and continues to outperform several robustness-oriented baselines (Tab. 1). Yet, using the validation step provides state-of-the-art performance, outperforming all other approaches.

Ablation on Concept Validation Thresholds. Following the initial label-free concept discovery procedure of [40], we further refined the resulting pool to obtain a high-quality concept set. To this end, we enforced minimum thresholds of an occurrence rate of at least 15% and spatial coverage of at least 20%. Applying these criteria to IN produced 1,852 validated concepts. Across the dataset, concepts appeared in 29% of images on average, and those satisfying the filtering criteria covered roughly 35% of the relevant region. We examined the effect of varying the occurrencerate and spatial-coverage thresholds on Top-1 accuracy for ViT-B evaluated on IN-A and IN-R. The best results were obtained using our default thresholds of 15% and 20%. Increasing the thresholds to 40%/40% reduced the number of concepts to 694 and led to a noticeable drop in performance

Table 8. Hyperparameter selection for all methods.

Model λalign λcls λnon-concept λconcept Learning rate

ViT-B 0.8 0.2 1.2 0.5 5e−7 DINOv2 0.8 0.2 1.2 0.5 6e−7 DeiT 0.8 0.2 1.2 0.5 8e−7 CNv2 0.8 0.2 1.2 0.5 3e−6

CFT

ViT-B - 2e−6 1e−10 - 2e−6 DINOv2 - 2e−8 1e−10 - 1e−5 DeiT - 2e−6 1e−10 - 5e−6 CNv2 - 2e−6 1e−8 - 3e−6

RRR

ViT-B - 0.1 50 - 0.001 DINOv2 - 0.1 50 - 0.005 DeiT - 0.1 50 - 0.001 CNv2 - 0.1 50 - 0.05

GradMask

ViT-B - - - - 2e-6 DINOv2 - - - - 1e-5 DeiT - - - - 5e-6 CNv2 - - - - 3e-6

RRDA

(IN-A: 24.59, IN-R: 44.23), presumably because many informative concepts were discarded. Relaxing the thresholds to 5%/10% increased the concept count to 2,435 but introduced substantial noise, which similarly harmed performance (IN-A: 25.13, IN-R: 44.92).

### 8. Main evaluation - full results

The results in Table 1 are averaged over five random seeds, where the subset of ImageNet classes used for fine-tuning is varied while keeping all other parameters fixed. Table 9 reports the corresponding standard deviations for this experiment.

### 9. Limitations and Future Work 9.1. Failure Cases

Despite strong overall performance, CFT exhibits identifiable failure modes:

Abstract or non-visual concepts. GPT-4o-mini rarely generates concepts that are semantically appropriate but not visually grounded (e.g., “aggressive behaviour” for a lion). GroundedSAM cannot localize such concepts, resulting in empty high-confidence masks.

Very small object parts. For parts occupying < 2% of image area (e.g., the beak of a distant bird), GroundedSAM’s hit rate decreases. The impact on final accuracy is limited, as the remaining concepts provide sufficient coverage, but fine-grained part-level reasoning may be impaired.

Domain mismatch between LLM and target domain. In specialized domains (medical imaging, satellite imagery), GPT-4o-mini’s concept vocabularies may be imprecise or incomplete. In such settings, domain-specific LLMs or expert-curated concept lists are recommended.

#### 9.2. Limitations

While our proposed CFT framework demonstrates improvements in model robustness across multiple benchmarks, several limitations warrant discussion.

Dependency on Vision-Language Models. Our approach relies on the quality and capabilities of GroundedSAM for concept localization. While this eliminates the need for manual annotations, it introduces a dependency on the grounding model’s performance. In cases where GroundedSAM fails to accurately segment concepts, particularly for abstract or fine-grained semantic attributes, the quality of guidance masks may degrade, potentially limiting CFT’s effectiveness.

Computational Overhead. While CFT is designed as a lightweight fine-tuning procedure requiring only 1,500 images, the initial concept creation and validation stage involves processing 30 samples per class through GroundedSAM, which introduces non-negligible computational costs. For datasets with thousands of classes, this preprocessing step could become a practical bottleneck. Moreover, computing relevance maps via AttnLRP during training adds overhead compared to standard gradient-based methods, though this cost is amortized across the finetuning procedure.

Table 9. Evaluation over 5 different seeds.

Model Metric IN-V IN-V2 IN-A ObjectNet IN-R IN-Sketch ViT-B

R@1 81.35 ±0.28 69.19 ±0.51 27.76 ±0.14 54.28 ±0.82 48.47 ±0.39 37.06 ±0.66 R@5 95.51 ±0.47 84.77 ±0.19 62.75 ±0.73 75.46 ±0.32 70.50 ±0.56 62.59 ±0.21

R@1 81.44 ±0.61 71.91 ±0.34 27.71 ±0.80 53.89 ±0.17 48.53 ±0.54 44.74 ±0.42 R@5 95.65 ±0.42 88.15 ±0.77 62.36 ±0.54 75.58 ±0.59 70.73 ±0.18 68.90 ±0.69

DINOv2

R@1 82.61 ±0.44 73.11 ±0.34 27.72 ±0.57 54.24 ±0.71 48.33 ±0.23 44.83 ±0.36 R@5 95.77 ±0.68 88.58 ±0.49 62.20 ±0.31 75.46 ±0.15 70.65 ±0.75 69.55 ±0.53

DeiT

R@1 87.27 ±0.43 75.25 ±0.63 27.93 ±0.41 54.19 ±0.50 48.37 ±0.78 46.14 ±0.27 R@5 95.71 ±0.52 89.50 ±0.38 62.40 ±0.65 75.62 ±0.22 70.68 ±0.46 70.81 ±0.60

CNv2

Architecture Specificity. Although we demonstrate CFT’s applicability to both ViTs and CNNs (ConvNeXtV2), the primary design and optimization were conducted with transformer architectures in mind. The adaptation to CNNs, while successful, required modifications to the relevance computation procedure. Extending CFT to other emerging architectures may require additional architectural considerations.

LLM-Based Concept Generation. While LLMs enable the automated, label-free discovery of semantic concepts [40], they introduce inherent risks concerning the reliability of the generated concept sets. Despite their sophistication, LLMs are susceptible to “hallucinations”—proposing attributes that are semantically plausible but lack visual grounding or are factually absent from the specific image. Furthermore, there is a risk that an LLM might inadvertently reinforce “shortcuts” by suggesting concepts based on frequent co-occurrences in its own training data rather than true class-discriminative features [16]. Finally, in specialized domains such as medical or satellite imagery, a domain mismatch can result in imprecise or incomplete vocabularies, ultimately degrading the quality of the spatial guidance masks.

#### 9.3. Future Work

Several promising directions emerge from this work that could further advance concept-guided robustness in vision models.

could be achieved through simple masking response-based approaches or by using concept activation vectors (CAVs) to produce concept-class importance weights.

Hierarchical and Compositional Concepts. Our framework currently treats concepts as independent entities. However, real-world objects exhibit hierarchical structure and compositional semantics. Incorporating compositional reasoning, where complex concepts are built from simpler primitives, could enhance both interpretability and robustness.

Application to Other Domains. Although this work focuses on image classification, the underlying principle of aligning model reasoning with semantically meaningful concepts extends naturally to other computer vision tasks (e.g., object detection, semantic segmentation, video understanding) and potentially to non-vision domains where structured, interpretable representations are valuable. Exploring these extensions could validate the generality of concept-guided learning as a robustness paradigm.

Adaptive Concept Weighting. Our current approach treats all validated concepts equally during fine-tuning. However, different concepts may contribute unequally to robustness for specific distribution shifts. Developing methods to dynamically weight concepts based on their discriminative power or relevance to particular OOD scenarios could yield more targeted robustness improvements. This

