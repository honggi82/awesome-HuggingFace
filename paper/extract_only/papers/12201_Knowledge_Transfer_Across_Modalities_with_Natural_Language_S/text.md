## If you can describe it, they can see it: Cross-Modal Learning of Visual Concepts from Textual Descriptions

# arXiv:2411.15611v2[cs.CV]17Dec2025

Carlo Alberto Barbano University of Turin

Luca Molinaro University of Turin

Vito Paolo Pastore University of Genoa

Massimiliano Ciranni University of Genoa

Marco Grangetto University of Turin

Emanuele Aiello Politecnico di Torino

### Abstract

Humans can visualize new and unknown concepts from their natural language description, based on their experience and previous knowledge. Insipired by this, we present a way to extend this ability to Vision-Language Models (VLMs), teaching them novel concepts by only using a textual description. We refer to this approach as Knowledge Transfer (KT). Our hypothesis is that the knowledge of a pre-trained VLM can be re-used to represent previously unknown concepts. Provided with a textual description of the novel concept, KT works by aligning relevant features of the visual encoder, obtained through model inversion, to its text representation. Differently from approaches relying on visual examples or external generative models, KT transfers knowledge within the same VLM by injecting visual knowledge directly from the text. Through an extensive evaluation on several VLM tasks, including classification, segmentation, image-text retrieval, and captioning, we show that: 1) KT can efficiently introduce new visual concepts from a single textual description; 2) the same principle can be used to refine the representation of existing concepts; and 3) KT significantly improves the performance of zero-shot VLMs.

### 1. Introduction

Can a blind person who gained sight recognize the objects they previously knew only by touch? This is a philosophical riddle posed by William Molyneux in 1668 to John Locke [45], which has been relevant in vision neuroscience for decades. Recent research has shown that, while this does not happen immediately after sight restoration, crossmodal mappings develop rapidly in human subjects, within days [20]. While recent research in multimodal neural networks has focused on this cross-modal interaction [62], in

this paper we aim to answer a slightly revisited version of Molyneux’s riddle, in which our model already has some previous knowledge of the world. We hypothesize that prior knowledge of a pre-trained VLM can be used to produce a reasonable visual representation of an unknown concept, if an explicative textual description is provided. This prior knowledge can be obtained with multimodal pre-training, for example by employing image-text alignment as done in CLIP and other similar works [15, 33, 56, 74]. Learning novel concepts starting from a textual description has been explored in different works such as SynthCLIP [19] and others [9, 27, 60, 76, 81]. These approaches, however, rely on the availability of generative models, which is not trivial in low-data contexts such as medical imaging. Furthermore, instead of just retrieving concepts learned during pre-training as in zero-shot VLMs [56], we explicitly incorporate new visual concepts by injecting novel textual descriptions into the VLM, without relying on external visual data or generative model (Fig. 1a). This enables performance improvements across a wide range of downstream tasks without using any real images, as shown in Fig. 1b. To the best of our knowledge, this is the first work to exploit existing knowledge of a VLM to learn novel concepts across modalities.

Leveraging natural language supervision to learn novel visual concepts is a process we call Knowledge Transfer (KT), inspired by previous research on zero-shot modality generalization [13]. An illustrative example of the goal of KT is shown in Fig. 2, where a CLIP-based zero-shot classifier is presented with unknown concepts. In this work, we propose a novel framework for KT that does not require parameters to be shared across visual and textual encoders, thus being general with respect to the VLM architecture. Specifically, starting from the textual description of the novel concepts, we synthesize matching imaging via model inversion [31], later employed to fine-tune the model with a visual-text matching loss.

175

| |MedCLIP +159% CLIP ViT<br><br>| |
|---|
<br><br>| |
|---|
<br><br>MedCLIP-SAMv2 ViLT<br><br>| |
|---|
<br><br>| |
|---|
<br><br>CoCa|
|---|---|
| | |
| | |

|Model<br><br>Synthetic Training<br><br>[Figure 1]<br><br>[Figure 2]<br><br>2. Finetune on generated images<br><br>Generative<br><br>Model<br><br>1. Prompt: "Generate an image of a gyroscope"<br><br>[Figure 3]<br><br>(From ChatGPT)|
|---|

150

RelativeImprovement(%)

20

+16%

Generative

Model

Model

+11%

10

+4% +5%

+5%

+2%

+1% +1%

0

|Model<br><br>"A gyroscope is a series of gleaming silver rings, each nested perfectly within the next, surrounds a central disk that spins smoothly...."<br><br>Knowledge Transfer (KT) 2. Model now visually<br><br>1. Transfer knowledge from recognizes novel concept text representation<br><br>[Figure 4]<br><br>[Figure 5]<br><br>|
|---|

Zero-shot Classif.(JSRT, Lung Cancer)Zero-shot Classif.(CheXpert-5x200c)Zero-shot Segm.(UnitoChest)Zero-shot Segm.(SIIM Pneumothorax)Zero-shot Segm.(UDIAT)Zero-shot Segm.(BraTS23)Text Retrieval(Flickr30k)Image Retrieval(Flickr30k)Captioning(MSCOCO)

(a) Our research question: Can we leverage existing knowledge in a pretrained VLM to teach it novel concepts? Differently from training with synthetic data or generative models [9, 19], with Knowledge Transfer we aim at leveraging existing knowledge within a model to teach it novel concepts, or improve existing ones.

(b) Knowledge Transfer improves performance across a variety of tasks and architectures. Differently from existing approaches, such as K-LITE [63], LLaMP [79], and SynthClip [19], this is achieved without using any real image nor generative model.

- Figure 1. Overview of Knowledge Transfer (KT): (a) shows our research question, while (b) summarizes performance improvements.

[Figure 6]

(a) CLIP (B) Top-3 zero-shot predictions: Triumphal Arch, Stone Wall, Steel Arch Bridge.

CLIP (B) + KT Top-3 zero-shot predictions: Moongate, Triumphal Arch, Stone Wall

[Figure 7]

(b) CLIP (L) Top-3 zero-shot predictions: (Cocktail Shaker, Odometer, Dragonfly.

CLIP (L) + KT Top-3 zero-shot predictions: Tonometer, Cocktail Shaker, Espresso Maker

- Figure 2. Knowledge Transfer can introduce novel concepts in a multimodal model, by leveraging prior visual knowledge of the visual encoder and a textual description of the target concept. In the example, a CLIP model [56] learns the concepts Moongate and Tonometer, without using any real image, while retaining a good accuracy on general zero-shot classification (58.10% vs 56.43% and 70.79% vs 70.61% on ImageNet-1k).

and textual features in a shared embedding space, empowering zero-shot and few-shot learning in various visual tasks. Efforts have been made to understand how VLMs internally process cross-modal information (e.g. multimodal neurons) [16, 53, 62]. Here, we provide an overview of works related to KT in different fields.

Cross-Modal Transfer Cross-modal knowledge distillation [18, 24, 66] is a strategy for transferring knowledge between modalities, to enrich representations. Methods like VidLanKD [66] and C2KD [24] employ modality-bridging techniques to improve generalization in zero and few-shot scenarios. These approaches typically require substantial multimodal data and complex training procedures. In contrast, our method uses textual descriptions to introduce new visual concepts with minimal data by efficient reuse of prior knowledge. Lin et al. [42] recently showed that integrating cues from multiple modalities can enhance concept learning, mirroring human learning. Their approach leverages few-shot examples of paired multimodal data to enhance unimodal downstream tasks. Differently from them, we use single-modal text data to introduce new visual knowledge.

Our findings show that KT:

- 1. Successfully introduces novel concepts in pre-trained VLMs with only textual descriptions;
- 2. Improves the visual accuracy on already existing concepts;
- 3. Improves zero-shot downstream tasks such as classification, segmentation, and image-text retrieval and shows potential for out-of-domain generalization.

Text-based zero-shot methods Methods such as KLITE [63] leverage structured text rather than simple captions as in CLIP [56], to improve pre-training. With a similar goal, Zheng et al. [79], propose to augment zeroshot prompts with an LLM, or to jointly fine-tune it in their LLaMP framework, to obtain richer classification prompts. Differently from these approaches, we use text data to describe novel concepts without using any real image. Additionally, text-only methods have been proposed to achieve visual understanding. For example, CapDec [51] and CLOSE [17], leverage the alignment between the vi-

### 2. Related Works

Multimodal representation learning aims to bridge the gap between different modalities (e.g. visual and textual), enabling models to process them jointly. VLMs like CLIP [56], CoCa [74], Flamingo [3] and ImageBind [15], align visual

sual and text encoder in CLIP by training a downstream model (e.g. for captioning or VQA) on text embedding and then using it on images. Our approach, on the other hand, aims at achieving knowledge transfer in the VLM itself via fine-tuning.

Synthetic Training Other works [9, 27, 81] involve synthetic data generation to train discriminative models. For example, SynthCLIP [19] leverages Stable Diffusion [58] to train a CLIP model entirely on synthetic data. While effective, this approach depends on the quality and diversity of generated data. Our approach differs by integrating novel concepts into existing models without relying on external knowledge of a text-to-image generative model and a computationally expensive data generation pipeline.

Incremental Learning Due to the similar objective, which is introducing novel concepts, we could also frame KT as a form of incremental learning. Recent works such as CLIP [43], TPPT [46], and ENGINE [80] exploit textual prompts or external knowledge to stabilize updates across tasks. Other approaches incorporate language cues or generative models to enrich class representations [6, 10, 78]. Despite these advances, there is still a notable gap: none (to our knowledge) explicitly considers incremental introduction of new visual classes solely via textual descriptions, with no visual exemplars, and then deploy the model to recognise images of the new class (i.e., a “text-only → visual class incremental” paradigm). Our method fills precisely this gap, by using textual descriptions of novel concepts (without real images) to integrate them into a pre-trained VLM, thereby enabling zero- or few-shot recognition of the new class while avoiding large-scale retraining or storage of exemplar images.

### 3. Knowledge Transfer

In this section, we present our proposed method for Knowledge Transfer, which we will simply refer to as KT.

##### 3.1. Goal of KT

Let fT : RL → Rn be a text encoder (where L is the sequence length) and fV : Rw×h → Rn be a visual encoder (with w and h being the size of the image), our goal is to introduce new concepts into fV through fT using only the text modality. Let XT be a set of unpaired captions pertaining to a novel concept that we want to learn, and XV∗ a set of ideal ground truth images corresponding to that concept. What we would like to achieve is:

−sim(fv(x∗v),ft(xk))

sim(fv(x∗v),ft(xt)) st

> 0

sk

∀x∗v ∈ XV∗ , xt ∈ XT, xk ∈ XK

(1)

where XK is the set of all other captions pertaining to other concepts (XK ∩ XT = ∅). The condition in Eq. 1 means that all ideal visual samples should be mapped closer to the true corresponding captions then all other captions. In practice, if XV∗ is available, we can satisfy Eq. 1 by optimizing its approximation [5]:

1 |X∗ V | x

exp(st) exp(st) + x

1 |XT| x

−

min

log

exp(sk)

fV

∗v

k

t

(2) which corresponds to the InfoNCE loss [8, 32, 56]. In our setting, however, XV∗ is not available, thus we propose to estimate it (e.g. with model inversion [31]) and then use the estimated values to jointly train fV and fT with the contrastive approach using InfoNCE [15, 56].

As a practical example, a caption xt for the concept Moongate could be “A perfectly circular archway built from uniformly cut stones or bricks, set into a larger wall. It forms a smooth circle, framing views of gardens or landscapes beyond, creating a picturesque portal.”. More examples can be found in the supplementary material. As shown by this example, this method requires the visual encoder to have some prior knowledge about the visual concepts contained in the caption (e.g., stones, walls, circles and gardens). We argue that this is reasonable, as humans also struggle to visualize unfamiliar concepts, especially in specialized domains beyond their prior experience. Nonetheless, in the results section we provide an experiment on out-of-domain KT, showing how the proposed approach can still reach zero-shot out-of-domain generalization.

###### 3.1.1. Estimating Xv∗ by inversion

The most straightforward way to estimate Xv∗ is to compute it by inverting the visual encoder fV starting from the textual embeddings of XT, in order to obtain an approximation XˆV∗ ≈ XV∗ . To do so, we solve the following optimization problem, starting from random noise:

XˆV∗ = fV−1(fT(XT)) ≈ max

sim(A(fV (XˆV∗ )),fT(XT)) + αR(XˆV∗ ) (3)

XˆV∗

where, as in [31], f−1 is the inversion operator, A is a random augmentation operation applied at each step (e.g. random affine) and R is a regularization term based on Total Variation (TV) [50], weighted by α. Augmentation and regularization help in producing more naturally-looking images.

Inversion as an effective alternative to generative models Learning new visual concepts from natural language description can be solved, in principle, by synthesizing training images based on textual descriptions, employing generative models (e.g. DALL-E [57] or Stable Diffusion [54]). As such, a question that may automatically arise is why

ImageNet 0-shot (Baseline)

Target Acc. (Baseline)

ImageNet 0-shot (KT)

Target Acc. (KT)

CLIP ViT-B/32 and ViT-L/14: Baseline vs KT (Target & ImageNet)

| | |Rare Concepts (Rigid)<br><br>+10% +21% +10%| | | | | | |Deformable Categories (Fabric)| | | | | | |Geometric (Parquet)| | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | |+30% +47%| | | | | | |+20%<br><br>+20%<br><br>+10%| | | | | | | | | | |
| | |+60%| | | | | | | | | | | | | || |
|---|
<br><br>+50%| | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | |+20%<br><br>+20%| | | | | | | | | | |
| | | | | | | | | |+10%| | | | | | |+10%| | | |
| | | | | | | | | | | | | | | | | | | | |

100

Accuracy(%)

80

60

40

20

0

CLIPViT-B/32MoongateCLIPViT-B/32TonometerCLIPViT-B/32GyroscopeCLIPViT-L/14MoongateCLIPViT-L/14TonometerCLIPViT-L/14GyroscopeCLIPViT-B/32

CLIPViT-B/32 Floral CLIPViT-B/32MadrasCLIPViT-L/14 BengalStripe CLIPViT-L/14 Floral CLIPViT-L/14MadrasCLIPViT-B/32 ParquetChantilly

CLIPViT-L/14 ParquetChantilly

BengalStripe

- Figure 3. Knowledge Transfer (KT) on novel and rare concepts (high-level and fine-grained concepts). KT achieves improvements (even notable) in the target accuracy on the novel concept in all instances. We also make sure that catastrophic forgetting does not occur by monitoring zero-shot accuracy on ImageNet, which remains comparable with the baseline.

not employing such generative models for KT. First, using external generative models for augmenting the training set [9, 19, 27, 60, 76, 81] is fundamentally different from the posed research question, as shown in Fig. 1a, which is transferring the knowledge from textual to visual modality within the same model, thus being independent from the availability of any external models. Besides, there are at least two other concerncs on the direct usage of generative models for the task at hand: i.) we do not know if they already include the target concepts in their training set; ii.) even if the target concepts are available in the original training set, in settings such as low-data domains (e.g. medical imaging), the availability of a generative model is not trivial, especially for clinically-relevant conditions. On top of this, employing external generative models would require more efforts in terms of computational time and resources.

###### 3.1.2. Finetuning on the new concepts

After images XˆV∗ have been synthesized via model inversion, we can use them to train fv and fT with an image-text alignment objective such as InfoNCE (Eq. 2). In order to successfully match visual features to the desired concepts, we augment XT by prepending the corresponding concept name to each caption. In the example presented earlier, the fine-tuning caption would be represented by “A moongate is a perfectly circular archway built from uniformly cut stones [...]”. This step is required in order to finally map the already learned low-level visual concepts to the high-level one itself.

### 4. Experimental Setup

We use several datasets to benchmark KT: an original dataset RareConcepts (for proof-of-concept experiments), ImageNet-1k [11], CheXpert-2x500c [23], JSRT [65], UnitoChest [7], UDIAT [72], SIIM Pneumothorax [75], BraTS23 Glioma [1], Flickr30k [73], and MSCOCO [41]. We target many different tasks, such as classification, seg-

mentation, captioning and text-image retrieval, across both natural and medical images. A more detailed presentation of the datasets can be found in the supplementary material, along with a detailed description of the training details. In summary, we produce the target concepts’ captions using a mix of LLM-based text and handcrafted captions. Inversion is run for 5k steps, and a quick fine-tuning of the visual encoder is done on the inverted images, with small learning rates, between 10−6 and 10−4, for one single epoch.

### 5. Controlled Experiments

In this section, we perform preliminary controlled experiments to assess the potential of Knowledge Transfer (KT). We aim at learning novel concepts in natural and medical images, and out-of-domain concepts.

##### 5.1. Learning novel concepts

As a proof of concept, we evaluate KT on the RareConcepts dataset, which includes three uncommon classes on natural images (Moongate, Tonometer, and Gyroscope) and four classes of fine-grained and deformable categories and geometric patterns (i.e. fabric and parquet patterns). These were selected by probing CLIP models on web-sourced concepts. We test two CLIP variants (ViT-B/32 and ViT-L/14), using the official checkpoints released by OpenAI [56]. The inversion-finetuning setup follows Sec. 4.

Results are shown in Fig. 3. We report zero-shot accuracy on target concepts before and after KT. Baseline models show poor recognition of unseen concepts-e.g., CLIP ViTB/32 fails on Moongate (0%) and Tonometer. After KT, all models improve on the target classes while largely preserving ImageNet accuracy, indicating minimal forgetting [35] with a proper choice of learning rate. With proper tuning, CLIP base and large even reach 100% target accuracy with negligible loss on ImageNet. Overall, these experiments

Method Needs real images

Needs Generative Model

Text type Transfer of knowledge*

SynthCLIP [19] NO YES Gen. prompt NO K-LITE [63] YES NO Structured NO LLaMP [79] YES YES Gen. prompt NO

LLM-aug [48, 79] NO NO Description NO

KT (ours) NO NO Description YES

- Table 1. Comparison with existing approaches. *refers to transfer within the same model.

Moongate Tonometer Gyroscope

CLIP L/14 (base) 78.95% 31.58% 90% CLIP L/14 + LLM [79] 100% 57.89% 60%↓ CLIP L/14 + KT (ours) 100% 78.95% 100%

- Table 2. Knowledge Transfer (KT) vs LLM-augmented CLIP (accuracy). Augmenting classification prompts with an LLM can help by introducing contextual cues (e.g. background for moongate), but no transfer of knowledge happens between textual and visual representations. KT provides more reliable improvements.

show that KT succesfully introduces unknown concepts and improves existing ones. Detailed results across different learning rate values are reported in the supplementary material, together with an analysis of ViLT.

Comparison with other approaches To our knowledge, no prior work directly addresses our research question, though some methods use textual information to enhance visual understanding. We qualitatively compare KT with three related directions: (i) improving textual supervision during pre-training (e.g., K-LITE [63]), (ii) tuning textual prompts for zero-shot classification (e.g., LLaMP [79], [48]), and (iii) generating synthetic images via text-to-image models (e.g., SynthCLIP [19]). As shown in Tab. 1, KT uniquely operates without real or synthetic images, requires no generative model, and performs knowledge transfer within the same model. Due to these differences, the only reasonable quantitative comparison can be done with the LLM-augmented approaches proposed in [48, 79]. We compare KT to LLMaugmented prompting methods, using identical captions (see Supplementary Material), in Tab. 2. While both methods succeed on Moongate (likely due to contextual cues such as “garden”), KT performs markedly better on Tonometer and Gyroscope, where LLM prompting even reduces accuracy. This reflects a key difference: KT achieves genuine cross-modal transfer, whereas LLM prompting merely alters text inputs. For completeness, in our ablation studies (Sec. 6.4), we also perform a comparison with a generative setup using Stable Diffusion XL, showing that inverted images outperform synthetic ones, likely because they are explicitly optimized for the target model.

Concept Baseline KT Benign Nodule Target Acc. 54.55% 54.55%

CheXpert 0-shot 62.10% 62.30% Lung Cancer Target Acc. 83.93% 92.86% CheXpert 0-shot 62.10% 61.50%

- Table 3. KT on MedCLIP on the JSRT dataset (accuracy). The model successfully learns the novel concept of malignant nodules (lung cancer) on CXR images. Benign nodules, on the other hand, are harder to visually differentiate from other findings in CXRs.

Model Atelect. Cardiom. Cons. Edema P. Effusion Top-1

MedCLIP Reference 49% 69.50% 32.50% 75.50% 84% 62.10% CLIP (B) Baseline 0% 2.5% 0% 0% 94.50% 19.40%

KT 0% 21.5% 0% 0% 85% 21.30% CLIP (L) Baseline 59.50% 16.50% 0% 0% 35.50% 22.40%

KT 4% 32.5% 0% 0% 92.5% 25.90%

- Table 4. Learning out-of-domain concepts (natural images → medical) shows potential. Accuracy on CheXpert-5x200c.
- 5.2. Knowledge Transfer on Medical Imaging

Next, we target KT on medical imaging. Medical images are a perfect task for KT, as we can leverage existing medical knowledge in the form of text (e.g. from medical textbooks and encyclopedias) to accurately describe concepts and visual appearance of different pathologies on images such as Chest X-rays (CXR), Computed Tomography (CT) scans, Magnetic Resonance Images (MRI), and Ultrasound images. Our experiments use MedCLIP [69], a CLIP-based model with BioClinicalBERT [4] as text encoder and Swin Transformer [44] as visual encoder. MedCLIP is pre-trained on MIMIC-CXR [28] and CheXpert [26], containing CXR images and radiological reports covering concepts such as Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion, and others. We introduce two new concepts, benign and malignant nodules, into MedCLIP following the same KT protocol as for CLIP, and evaluate zero-shot accuracy on the external JSRT dataset [65].

Results are reported in Tab. 3, with captions listed in the Supplementary Material. As before, we monitor the zero-shot accuracy on previous knowledge using CheXpert5x200c, to spot instances of catastrophic forgetting. KT improves malignant nodule detection from 83.93% to 92.86% while retaining comparable results on the source dataset CheXpert-5x200c. For benign nodules, accuracy remains unchanged, likely due to less distinctive visual cues. Interestingly, we observe a slight overall gain on the source dataset, suggesting more robust feature representations.

##### 5.3. Out of domain Knowledge Transfer

Lastly, we assess the potential of KT to introduce novel concepts outside of the training domain. Specifically, we aim to

introduce medical concepts into a model trained on natural images. For this purpose, we fine-tune a CLIP model on all five CheXpert classes (atelectasis, cardiomegaly, consolidation, edema, and pleural effusion). The results are reported in Tab. 4. We report the performance of MedCLIP as a reference for a model trained on CheXpert. Looking at the top-1 accuracy, we achieve improved results with both versions of CLIP, with the large variant scoring a higher increase from 22.40% to 25.90%. However, breaking down the accuracy per class reveals that i.) classes with a starting accuracy of 0% did not improve, and ii.) performance in some classes got worse (i.e. pleural effusion and atelectasis). This may be due to the domain gap between the prior knowledge of the model (natural images) and the features specific to the medical domain. Nevertheless, considering this limitation, KT shows potential in zero-shot out-of-domain generalization.

### 6. Real-world Experiments

With an extensive evaluation on different datasets and domains, we aim to thoroughly evaluate the potential of KT. Here, we focus on improving zero-shot downstream tasks on real-world scenarios, on both novel and known concepts. Namely, we target segmentation, image-text retrieval, and captioning. The detailed description about the experimental setup can be found in the supplementary material.

##### 6.1. Captioning

We perform experiments on captioning on the MSCOCO dataset. For this task, we employ the CoCa architecture [74], which is a state-of-the-art captioner. Specifically, we employ the open-source version released by LAION [25] as the original one is proprietary. CoCa is built by adding an autoregressive text decoder to a CLIP model, thus when fine-tuning we apply the InfoNCE loss jointly with a cap-

tioning loss [74] which aims at predicting the next token yt given the previous tokens y<t and the image x. As captions,

MSCOCO (5K) Model BLEU@4 METEOR CIDEr SPICE

CLIP-ViL [64] 40.2 29.7 134.2 23.8 BLIP [39] 40.4 - 136.7 VinVL [77] 41.0 31.1 140.9 25.4 SimVLM [70] 40.6 33.7 143.3 25.4 LEMON [22] 41.5 30.8 139.1 24.1 CoCa [74] (proprietary) 40.9 33.9 143.6 24.7

CoCa 6.9 12.8 31.1 9.1

CoCa + KT 17.9 19.4 60.8 13.7 CoCa FT 34.9 29.7 123.1 23.5 CoCa FT + KT 35.2 29.8 124.0 23.3

Table 5. Image captioning on MSCOCO. CoCa refers to the baseline model pre-trained on LAION-2B [61], while CoCa FT refers to the model fine-tuned for captioning on MSCOCO. We highlight in bold the best results and the improvements by Knowledge Transfer.

Sample

[Figure 8]

[Figure 9]

[Figure 10]

Actual Caption

A shot of a clock in the train station.

A baseball player hitting a ball in a professional game.

A baseball player holding a bat during a baseball game.

###### Baseline (CoCa)

grand - central - station - new - york.jpg (METEOR 0.0)

Aaron judge 2016 new york Yankees (METEOR 5.0)

20080419 mariners 0001 | by Mike. Smith (METEOR 4.2)

CoCa+KT A black and white photo of a clock at Grand Central terminal. (METEOR 92.9)

A baseball player swings his bat at a batter. (METEOR 83.6)

A baseball player takes a swing at a pitch. (METEOR 86.9)

Table 6. Visual examples of captioning on MSCOCO. Illustrative cases where KT improves captioning, with METEOR scores in parentheses.

we utilize a simple set of templates such as “A photo of a X”, containing different alterations. They are listed in the supplementary material.

Results are presented in Tab. 5. We report different evaluation metrics (BLEU, METEOR, CIDEr, SPICE) computed using the standard pycocoevalcap package [59]. We perform experiments with two variants of CoCa: one pre-trained on LAION-2B [61], and one further fine-tuned (FT) for captioning on MSCOCO. We also report reference results from proprietary CoCa [74] for comparison, along with other methods. With KT, we improve on CoCa FT across almost all metrics, reaching a BLEU@4 of 35.2. A notable result is achieved with the pre-trained only CoCa, where we improve all metrics by a large margin, sometimes even doubling them (e.g. BLEU@4 from 6.9 to 17.9). We want to point out again that this model is not originally trained for captioning on MSCOCO, and the improvement is introduced by KT alone, without using any real image at all. We report some visual results in Tab. 6, where the improvement is notable; more examples can be found in the supplementary material.

##### 6.2. Segmentation

For segmentation, we employ the zero-shot method MedCLIP-SAMv2 [36, 37]. It works by computing activation maps from a pre-trained CLIP model, and using them as query for the Segment Anything Model (SAM) [34]. Activation maps are computed using Multi-Modal Information Bottleneck Attribution (M2IB) [68], using a target image and a query prompt. Here, we aim at improving the quality of the

Lung Nodules† Lung Pneumothorax† Breast Ultrasound Brain MRI Model DSC NSD IoU DSC NSD IoU DSC NSD IoU DSC NSD IoU MedCLIP-SAMv2 14.83% 17.30% 8.64% 6.30% 7.61% 3.75% 56.25% 59.44% 47.81% 17.20% 20.97% 12.05%

- KT (1e-5) 13.95% 17.45% 8.75% 6.28% 7.59% 3.77% 58.23% 61.56% 49.52% 15.90% 19.36% 11.10%

- KT (2e-5) 14.10% 17.65% 8.83% 6.41% 7.76% 3.83% 54.36% 57.30% 46.30% 18.13% 22.26% 12.62% KT (1e-4) 14.35% 18.03% 9.04% 6.02% 7.29% 3.59% - - - - - -

- Table 7. Improvements in zero-shot segmentation. † denotes novel concepts that are not included in the original MedCLIP-SAMv2 training data [36]. Prompts used for segmentation are reported here: P1 A medical chest CT scan showing circular spots of varying size within the lungs, suggesting either benign or malignant nodules; P2 A medical chest x-ray showing an abnormal collection of air within the pleural cavity, suggesting a pneumothorax; P3 A medical breast mammogram showing an irregularly shaped, spiculated mass suggestive of a malignant breast tumor; P4 A brain MRI showing a bright or dark mass with irregular edges suggestive of a brain tumor or glioma.

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Image Ground Truth MedCLIPSAMv2

[Figure 20]

[Figure 21]

[Figure 22]

KT

Figure 4. Qualitative evaluation of KT on breast tumor segmentation (UDIAT dataset). We report illustrative examples where knowledge transfer improved segmentation, in terms of DSC.

Flickr30k (1K) Text Retrieval Image Retrieval

Model R@1 R@5 R@10 R@1 R@5 R@10 ViLBERT [47] - - - 31.9% 61.1% 72.8% Unicoder-VL [38] 64.3% 85.8% 92.3% 48.4% 76.0% 85.2% ImageBERT [55] 70.7% 90.2% 94.0% 54.3% 79.6% 87.5% ViLT-B/32 (original) [33] 73.2% 93.6% 96.5% 55.0% 82.5% 89.8% ViLT-B/32 (huggingface) 73.8% 93.5% 96.5% 57.3% 83.9% 90.4% ViLT-B/32 + KT (lr 9e-7) 74.6% 93.8% 96.4% 57.8% 84.0% 90.5% ViLT-B/32 + KT (lr 2e-6) 74.6% 93.7% 96.5% 57.8% 84.0% 90.5%

- Table 8. Text and image retrieval on Flickr30k. Recall scores are shown at top 1, 5 and 10 levels. Our results are based on huggingface’s ViLT. Original results and other comparisons from [33].

in the supplementary material. To compute the M2IB activation maps on the fine-tuned models, we employ descriptive prompts as suggested in [37]. The prompts are reported in Tab. 7 as P1 to P4 for each task. We also report reference results of MedCLIP-SAMv2 on each task. Compared to the original setting of MedCLIP-SAMv2, lung nodules and lung pneumothorax are completely novel concepts. There is also a slight difference in the brain glioma class compared to the original brain tumor task, explained in the supplementary file. We employ three metrics to assess the segmentation quality, namely the Dice-Sørensen Coefficient (DSC), Normalized Surface Distance (NSD), and Intersection over Union (IoU). We report results with different values of fine-tuning learning rate. We can observe an increase in segmentation metrics across all tasks, notably in breast ultrasound (NSD 59.44% to 61.56%) and brain MRIs (NSD 20.97% to 22.26%). For lung nodules and pneumothorax, the improvement is less pronounced, probably because the novelty of the task makes improving more difficult in the MedCLIP-SAM setting. We report some visual examples on breast tumor segmentation in Fig. 4, showcasing the improvements of KT.

##### 6.3. Text and image retrieval

We perform experiments on text and image retrieval on the Flickr30k dataset. For these experiments, we employ the huggingface version of ViLT [33]. To fine-tune ViLT with KT, we employ captions of common concepts that may help improve the model’s general knowledge. For this purpose, we use the 80 object categories from MSCOCO as target concepts, using the method presented in Sec. 4, using ChatGPT4. All captions are reported in the supplementary material. For each caption, we generate 10 inverted images, for a total of 800 inverted images. Fine-tuning is performed as in Sec. 5.1, by maximizing the ITM score for positive pairs and minimizing it for negative ones.

activation maps on different concepts by leveraging KT. This, in turn, should result in a higher accuracy of the final segmentation. We target four different segmentation tasks: lung nodules segmentation on CT images (UnitoChest), pneumothorax segmentation on CXR images (SIIM Pneumothorax), breast nodule segmentation on ultrasound images (UDIAT), and glioma segmentation in MRIs (BraTS23).

The results of zero-shot text and image retrieval are reported in Tab. 8. For comparison, we also report the original results of ViLT from [33], alongside other relevant baselines. Results are shown in terms of recall (marked as R) computed at different levels (top-1, top-5, and top-10 recall). As we

The overall results across all segmentation tasks are presented in Tab. 7. The captions used for inversion are reported

ViT-L/14 - Gyroscope

100

Accuracy(%)

ViT-L/14 - Moongate ViT-L/14 - Tonometer

75

50

25

0 1000 2000 3000 4000 5000

Num. of inversion steps

- Figure 5. KT results at different inversion steps. Better inversion quality (more steps) generally leads to improved results.

observe from the results, KT consistently improves the results across all metrics for both image and text retrieval tasks. Notably, we score an improvement of almost 1% from 73.8% to 74.6% on the text retrieval task. Additional configurations are available in the supplementary material.

- 6.4. Ablation studies

In this section, we analyze the main factors influencing KT’s performance using the RareConcepts dataset, namely inversion hyperparameters and quality of the textual description. We also report computational details and scaling of inversion, compared to other approaches such as generative models. Additional ablations on fine-tuning strategy and caption construction are provided in the Supplementary Material.

Inversion hyperparameters Figure 5 and Table 9 summarize how inversion quality affects downstream accuracy. Increasing the number of inversion steps generally improves reconstruction fidelity, resulting in higher KT accuracy (Fig. 5). Similarly, applying image augmentation and total variation regularization during inversion leads to more stable and accurate representations (Tab. 9). The full configuration combining both techniques achieves the best accuracy.

Description quality Tab. 10 evaluates KT’s robustness to caption length and quality. KT consistently performs well across short (P1), medium (P2), and long (P3) captions, showing only minor variation. Human-written captions with concise, relevant descriptions slightly outperform LLMgenerated or verbose ones, which may also contain irrelevant text. Overall, KT remains stable across prompt variations. The captions used are listed in the supplementary material.

Model inversion vs Stable Diffusion In this ablation, we employ a generative approach based on SDXL using the same prompts of previous experiments (P3) on the RareConcepts dataset, to obtain images used for fine-tuning. As summarized in the last column of Tab. 10, KT outperforms the generative SDXL baseline (average 88.3% vs. 86.7%), while requiring much less compute and memory.

Compute efficiency and scaling As shown in Fig. 6, our inversion-based KT scales efficiently with dataset size. Compared to SDXL, KT requires substantially less compute and memory (e.g., 3.5k s vs. 27k s for 1k images), remaining practical even for large-scale applications.

Name Augmentation TV Strength (α) Moongate Tonometer Gyroscope Avg. Baseline - - 0% 50% 90% 46.66%

- A1 No 0 20% 80% 100% 66.67%
- A2 Yes 0.001 30% 80% 100% 70% Full Yes 0.005 60% 80% 100% 80%

- Table 9. Inversion hyperparameters on CLIP (B). Augmentation and regularization improve quality, achieving better overall results.

Model Concept Baseline P1 P2 P3 SDXL

CLIP ViT-B/32

Moongate 0% 50% 70% 60% 70% Tonometer 50% 80% 80% 80% 70% Gyroscope 90% 100% 100% 100% 90%

CLIP ViT-L/14

Moongate 78.95% 100% 100% 100% 90% Tonometer 31.58% 100% 90% 90% 100% Gyroscope 90% 100% 100% 100% 100%

Avg 56.76% 88.33% 90% 88.33% 86.67%

- Table 10. Robustness of KT to prompt variations and comparison with SDXL (P1: short human caption < 8 words; P2: mid-size human caption < 16 words; P3: longer LLM caption > 32 words).

OOM KT (B)

55 GiB

60000

Memory(MB)

###### SDXL

SDXL

Runtime(s)

| | |
|---|---|
| | |
| | |

27420

40000

20000

24 GiB 13 GiB

20000

KT (ViT-B)

3460

0

2 GiB 4 GiB

101 102 103

0

10 100 1000

Tot. Num. of images

Tot. Num. of images

Figure 6. Compute and scaling of inversion (left) total runtime (right) memory required to generate 10, 100, and 1k images.

### 7. Conclusions and Future Works

We present a way to learn novel visual concepts by only using their textual descriptions, with a method we call Knowledge Transfer (KT). Through extensive evaluation, we show that KT can successfully introduce novel concepts in pre-trained VLMs, without hurting performance on previous tasks. We also show that KT can improve the results of downstream zero-shot tasks, such as segmentation, text-image retrieval, and captioning, and also shows potential for out-of-domain generalization, for example on medical images. The proposed method is based on model inversion to synthesize ideal images for a target concept, that are later used to fine-tune the VLM in an image-text matching fashion, such as CLIP [56]. Our method leverages prior knowledge in pre-trained VLMs, with the aim of aligning known visual concepts to novel high-level ones.

One key design choice in our KT framework is not requiring parameters to be shared between visual and textual encoders, thus making it generally applicable regardless of the specific architecture of the VLM. We believe that by relaxing this constraint, it should possible to achieve KT with an alternative framework, for example by leveraging multimodal neurons [62] and employing only textual captions (e.g., fine-tuning the VLM with masked language modeling). We provide a brief presentation of such alternative frame-

work in the supplementary material, leaving an in-depth investigation to future work.

To the best of our knowledge, this is the first work attempting to leverage prior knowledge inside a neural network to teach it novel concepts. We believe this work can pave the way for future research on this topic, especially in datalimited domains where imaging data is scarce but we have access to textual human knowledge, such as medical imaging.

### References

- [1] Maruf Adewole, Jeffrey D Rudie, Anu Gbdamosi, Oluyemisi Toyobo, Confidence Raymond, Dong Zhang, Olubukola Omidiji, Rachel Akinola, Mohammad Abba Suwaid, Adaobi Emegoakor, et al. The brain tumor segmentation (brats) challenge 2023: glioma segmentation in sub-saharan africa patient population (brats-africa). ArXiv, 2023. 4, 15
- [2] AI@Meta. Llama 3 model card. 2024. 14
- [3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736,

2022. 2

- [4] Emily Alsentzer. emilyalsentzer/Bio_ClinicalBERT · Hugging Face — huggingface.co. https://huggingface. co/emilyalsentzer/Bio_ClinicalBERT. 5
- [5] Carlo Alberto Barbano, Benoit Dufumier, Enzo Tartaglione, Marco Grangetto, and Pietro Gori. Unbiased supervised contrastive learning. In The Eleventh International Conference on Learning Representations, 2023. 3
- [6] Xusheng Cao, Haori Lu, Linlan Huang, Xialei Liu, and MingMing Cheng. Generative multi-modal models are good class incremental learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28706–28717, 2024. 3
- [7] Hafiza Ayesha Hoor Chaudhry, Riccardo Renzulli, Daniele Perlo, Francesca Santinelli, Stefano Tibaldi, Carmen Cristiano, Marco Grosso, Giorgio Limerutti, Attilio Fiandrotti, Marco Grangetto, et al. Unitochest: A lung image dataset for segmentation of cancerous nodules on ct scans. In International Conference on Image Analysis and Processing, pages 185–196. Springer, 2022. 4, 15
- [8] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pages 1597–1607. PMLR, 2020. 3
- [9] Yunhao Chen, Zihui Yan, and Yunjie Zhu. A unified framework for generative data augmentation: A comprehensive survey. arXiv preprint arXiv:2310.00277, 2023. 1, 2, 3, 4
- [10] Marco D’Alessandro, Alberto Alonso, Enrique Calabrés, and Mikel Galar. Multimodal parameter-efficient few-shot class incremental learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3393–3403,

2023. 3

- [11] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database.

- In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 4, 15
- [12] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171– 4186, Minneapolis, Minnesota, 2019. Association for Computational Linguistics. 18
- [13] Mohamed Elhoseiny, Babak Saleh, and Ahmed Elgammal. Write a classifier: Zero-shot learning using purely textual descriptions. In Proceedings of the IEEE international conference on computer vision, pages 2584–2591, 2013. 1
- [14] Leon A Gatys, Alexander S Ecker, and Matthias Bethge. Image style transfer using convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2414–2423, 2016. 13
- [15] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15180–15190, 2023. 1, 2, 3
- [16] Gabriel Goh, Nick Cammarata, Chelsea Voss, Shan Carter, Michael Petrov, Ludwig Schubert, Alec Radford, and Chris Olah. Multimodal neurons in artificial neural networks. Distill, 6(3):e30, 2021. 2, 13
- [17] Sophia Gu, Christopher Clark, and Aniruddha Kembhavi. I can’t believe there’s no images! learning visual tasks using only language supervision. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2672– 2683, 2023. 2
- [18] Saurabh Gupta, Judy Hoffman, and Jitendra Malik. Cross modal distillation for supervision transfer. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2827–2836, 2016. 2
- [19] Hasan Abed Al Kader Hammoud, Hani Itani, Fabio Pizzati, Philip Torr, Adel Bibi, and Bernard Ghanem. Synthclip: Are we ready for a fully synthetic clip training? arXiv preprint arXiv:2402.01832, 2024. 1, 2, 3, 4, 5
- [20] Richard Held, Yuri Ostrovsky, Beatrice de Gelder, Tapan Gandhi, Suma Ganesh, Umang Mathur, and Pawan Sinha. The newly sighted fail to match seen with felt. Nature neuroscience, 14(5):551–553, 2011. 1
- [21] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 13, 14
- [22] Xiaowei Hu, Zhe Gan, Jianfeng Wang, Zhengyuan Yang, Zicheng Liu, Yumao Lu, and Lijuan Wang. Scaling up visionlanguage pretraining for image captioning. 2022 ieee. In CVF Conference on computer vision and pattern recognition (CVPR), pages 17959–17968, 2021. 6, 21
- [23] Shih-Cheng Huang, Liyue Shen, Matthew P Lungren, and Serena Yeung. Gloria: A multimodal global-local representation learning framework for label-efficient medical image

- recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3942–3951, 2021. 4, 15
- [24] Fushuo Huo, Wenchao Xu, Jingcai Guo, Haozhao Wang, and Song Guo. C2kd: Bridging the modality gap for cross-modal knowledge distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16006–16015, 2024. 2
- [25] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, 2021. 6
- [26] Jeremy Irvin, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, et al. Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In Proceedings of the AAAI conference on artificial intelligence, pages 590–597, 2019. 5, 15
- [27] Ali Jahanian, Xavier Puig, Yonglong Tian, and Phillip Isola. Generative models as a data source for multiview representation learning. In International Conference on Learning Representations, 2022. 1, 3, 4
- [28] Alistair EW Johnson, Tom J Pollard, Seth J Berkowitz, Nathaniel R Greenbaum, Matthew P Lungren, Chih-ying Deng, Roger G Mark, and Steven Horng. Mimic-cxr, a deidentified publicly available database of chest radiographs with free-text reports. Scientific data, 6(1):317, 2019. 5
- [29] Kaggle. ImageNet100. https://www.kaggle.com/ datasets/ambityga/imagenet100. [Accessed Nov. 2024]. 17
- [30] Andrej Karpathy and Li Fei-Fei. Deep visual-semantic alignments for generating image descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3128–3137, 2015. 15
- [31] Hamid Kazemi, Atoosa Chegini, Jonas Geiping, Soheil Feizi, and Tom Goldstein. What do we learn from inverting clip models? arXiv preprint arXiv:2403.02580, 2024. 1, 3, 15
- [32] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning. Advances in neural information processing systems, 33:18661–18673,

2020. 3

- [33] Wonjae Kim, Bokyung Son, and Ildoo Kim. ViLT: Visionand-language transformer without convolution or region supervision. In International conference on machine learning, pages 5583–5594. PMLR, 2021. 1, 7, 13, 15, 17, 19
- [34] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023. 6
- [35] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13): 3521–3526, 2017. 4

- [36] Taha Koleilat, Hojat Asgariandehkordi, Hassan Rivaz, and Yiming Xiao. Medclip-sam: Bridging text and image towards universal medical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 643–653. Springer, 2024. 6, 7
- [37] Taha Koleilat, Hojat Asgariandehkordi, Hassan Rivaz, and Yiming Xiao. Medclip-samv2: Towards universal text-driven medical image segmentation. arXiv preprint arXiv:2409.19483, 2024. 6, 7, 17
- [38] Gen Li, Nan Duan, Yuejian Fang, Ming Gong, and Daxin Jiang. Unicoder-vl: A universal encoder for vision and language by cross-modal pre-training. Proceedings of the AAAI Conference on Artificial Intelligence, 34(07):11336–11344,

2020. 7

- [39] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified visionlanguage understanding and generation. In International conference on machine learning, pages 12888–12900. PMLR,

2022. 6, 21

- [40] Liunian Harold Li, Haoxuan You, Zhecan Wang, Alireza Zareian, Shih-Fu Chang, and Kai-Wei Chang. Unsupervised vision-and-language pre-training without parallel images and captions. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5339–5350, Online, 2021. Association for Computational Linguistics. 13
- [41] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 4, 15
- [42] Zhiqiu Lin, Samuel Yu, Zhiyi Kuang, Deepak Pathak, and Deva Ramanan. Multimodality helps unimodality: Crossmodal few-shot learning with multimodal models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19325–19337, 2023. 2
- [43] Wenzhuo Liu, Fei Zhu, Longhui Wei, and Qi Tian. C-CLIP: Multimodal continual learning for vision-language model. In The Thirteenth International Conference on Learning Representations, 2025. 3
- [44] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021. 5
- [45] John Locke. An essay concerning human understanding,

1690. 1948. 1

- [46] Haodong Lu, Xinyu Zhang, Kristen Moore, Jason Xue, Lina Yao, Anton van den Hengel, and Dong Gong. Continual learning on clip via incremental prompt tuning with intrinsic textual anchors. arXiv preprint arXiv:2505.20680, 2025. 3
- [47] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2019. 7
- [48] Sachit Menon and Carl Vondrick. Visual classification via description from large language models. In The Eleventh

- International Conference on Learning Representations, 2023. 5
- [49] Shentong Mo and Pedro Morgado. Unveiling the power of audio-visual early fusion transformers with dense interactions through masked modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27186–27196, 2024. 13
- [50] A. Mordvintsev, Christopher Olah, and Mike Tyka. Inceptionism: Going deeper into neural networks. 2015. 3
- [51] David Nukrai, Ron Mokady, and Amir Globerson. Text-only training for image captioning using noise-injected clip. arXiv preprint arXiv:2211.00575The 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2022. 2
- [52] OpenAI. Chatgpt, 2024. Nov 2024 Version. 15
- [53] Haowen Pan, Yixin Cao, Xiaozhi Wang, and Xun Yang. Finding and editing multi-modal neurons in pre-trained transformer. arXiv preprint arXiv:2311.07470, 2023. 2
- [54] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 3
- [55] Di Qi, Lin Su, Jia Song, Edward Cui, Taroon Bharti, and Arun Sacheti. Imagebert: Cross-modal pre-training with large-scale weak-supervised image-text data, 2020. 7
- [56] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 1, 2, 3, 4, 8, 15, 17
- [57] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on machine learning, pages 8821–8831. Pmlr, 2021. 3
- [58] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. 2022 ieee. In CVF Conference on Computer Vision and Pattern Recognition (CVPR),

2021. 3

- [59] salaniz. GitHub - salaniz/pycocoevalcap: Python 3 support for the MS COCO caption evaluation tools — github.com. https://github.com/salaniz/pycocoevalcap. 6
- [60] Veit Sandfort, Ke Yan, Perry J Pickhardt, and Ronald M Summers. Data augmentation using generative adversarial networks (cyclegan) to improve generalizability in ct segmentation tasks. Scientific reports, 9(1):16884, 2019. 1, 4
- [61] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa R Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5b: An open large-scale dataset for training next generation image-text models. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022. 6, 21

- [62] Sarah Schwettmann, Neil Chowdhury, Samuel Klein, David Bau, and Antonio Torralba. Multimodal neurons in pretrained text-only transformers. In 2023 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), pages 2854–2859, 2023. 1, 2, 8, 13, 18
- [63] Sheng Shen, Chunyuan Li, Xiaowei Hu, Yujia Xie, Jianwei Yang, Pengchuan Zhang, Zhe Gan, Lijuan Wang, Lu Yuan, Ce Liu, et al. K-lite: Learning transferable visual models with external knowledge. Advances in Neural Information Processing Systems, 35:15558–15573, 2022. 2, 5
- [64] Sheng Shen, Liunian Harold Li, Hao Tan, Mohit Bansal, Anna Rohrbach, Kai-Wei Chang, Zhewei Yao, and Kurt Keutzer. How much can CLIP benefit vision-and-language tasks? In International Conference on Learning Representations, 2022. 6, 21
- [65] Junji Shiraishi, Shigehiko Katsuragawa, Junpei Ikezoe, Tsuneo Matsumoto, Takeshi Kobayashi, Ken-ichi Komatsu, Mitate Matsui, Hiroshi Fujita, Yoshie Kodera, and Kunio Doi. Development of a digital image database for chest radiographs with and without a lung nodule: receiver operating characteristic analysis of radiologists’ detection of pulmonary nodules. American journal of roentgenology, 174(1):71–74, 2000. 4, 5, 15
- [66] Zineng Tang, Jaemin Cho, Hao Tan, and Mohit Bansal. Vidlankd: Improving language understanding via video-distilled knowledge transfer. Advances in Neural Information Processing Systems, 34:24468–24481, 2021. 2
- [67] Radiopaedia Team. Radiopaedia. https : / / radiopaedia.org/. [Accessed Nov. 2024]. 15
- [68] Ying Wang, Tim G. J. Rudner, and Andrew Gordon Wilson. Visual explanations of image-text representations via multimodal information bottleneck attribution. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 6
- [69] Zifeng Wang, Zhenbang Wu, Dinesh Agarwal, and Jimeng Sun. MedCLIP: Contrastive learning from unpaired medical images and text. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 3876–3887, Abu Dhabi, United Arab Emirates, 2022. Association for Computational Linguistics. 5
- [70] Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, and Yuan Cao. SimVLM: Simple visual language model pretraining with weak supervision. In International Conference on Learning Representations, 2022. 6, 13, 21
- [71] Han Xu, Yao Ma, Hao-Chen Liu, Debayan Deb, Hui Liu, JiLiang Tang, and Anil K Jain. Adversarial attacks and defenses in images, graphs and text: A review. International journal of automation and computing, 17:151–178, 2020. 14
- [72] Moi Hoon Yap, Gerard Pons, Joan Marti, Sergi Ganau, Melcior Sentis, Reyer Zwiggelaar, Adrian K Davison, and Robert Marti. Automated breast ultrasound lesions detection using convolutional neural networks. IEEE journal of biomedical and health informatics, 22(4):1218–1226, 2017. 4, 15
- [73] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Association for Computational Linguistics, 2:67–78, 2014. 4, 15

- [74] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. Transactions on Machine Learning Research, 2022. 1, 2, 6, 16, 21
- [75] Anna Zawacki, Carol Wu, Shih George, Julia Elliott, Mikhail Fomitchev, Hussain Mohannad, Paras Lakhani, Phil Culliton, and Shunxing Bao. Siim-acr pneumothorax segmentation challenge. Kaggle, 2019. 4, 15
- [76] Huijuan Zhang, Zongrun Huang, and Zhongwei Lv. Medical image synthetic data augmentation using gan. In Proceedings of the 4th International Conference on Computer Science and Application Engineering, pages 1–6, 2020. 1, 4
- [77] Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. Vinvl: Revisiting visual representations in vision-language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5579–5588, 2021. 6, 21
- [78] Wentao Zhang, Tong Yu, Ruixuan Wang, Jianhui Xie, Emanuele Trucco, Wei-Shi Zheng, and Xiaobo Yang. Visual class incremental learning with textual priors guidance based on an adapted vision-language model. IEEE Transactions on Multimedia, 2025. 3
- [79] Zhaoheng Zheng, Jingmin Wei, Xuefeng Hu, Haidong Zhu, and Ram Nevatia. Large language models are good prompt learners for low-shot image classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28453–28462, 2024. 2, 5
- [80] Da-Wei Zhou, Kai-Wen Li, Jingyi Ning, Han-Jia Ye, Lijun Zhang, and De-Chuan Zhan. External knowledge injection for clip-based class-incremental learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 3314–3325, 2025. 3
- [81] Yongchao Zhou, Hshmat Sahak, and Jimmy Ba. Training on thin air: Improve image classification with generated data. arXiv preprint arXiv:2305.15316, 2023. 1, 3, 4

### A. Impact and Limitations

We believe that Knowledge Transfer has the potential to be an impactful technique for introducing novel concepts in pre-trained models. Overall, Knowledge Transfer is quite cheap in terms of computational requirements, as it works by only fine-tuning on just a handful of synthesized samples. Thus, it is very quick and does not need a large amount of memory. In this sense, it may be comparable to parameterefficient fine-tuning (PEFT) techniques, such as low-rank adaptation (LoRA) [21], which minimize the amount of memory required for fine-tuning. However, compared to PEFT, Knowledge Transfer does not require any real data besides a single textual description for each novel concept.

From this point onward, we will refer to the KT algorithm described in the main paper as Explicit Knowledge Transfer, namely the variant that relies on an inversion step to synthesize a visual example before fine-tuning. In contrast, we use the term Implicit Knowledge Transfer to denote approaches that avoid this inversion step and instead rely on shared parameters between modalities (e.g., multimodal neurons), enabling transfer through purely textual objectives such as MLM.

The main limitation of Explicit Knowledge Transfer lies in the inversion step, which takes the most time compared to fine-tuning. If this step could be avoided, we could achieve near real-time learning of novel concepts with minimal computational requirements. This could enable the development of rapidly improving intelligent agents in many realworld applications. We hypothesize this is possible with Implicit Knowledge Transfer, for example by using MaskedLanguage Modeling (MLM) as a proxy for knowledge transfer. However, in this work, we do not focus on this topic, as preliminary experiments (shown in Sec. D.3) did not achieve satisfactory results compared to Explicit Knowledge Transfer.

Another limitation lies in the limited comparison with state-of-the-art approaches; however, to the best of our knowledge, we are not aware of other works sharing the same goal as ours.

### B. Knowledge Transfer

A general overview of explicit Knowledge Transfer can be found in Fig. 7.

- B.1. Examples of inverted images Examples of inverted images can be found in Fig. 8.
- B.2. Possible improvements of Explicit Transfer We start from the inversion equation:

XˆV∗ = fV−1(fT(XT)) ≈ max

sim(A(fV (XˆV∗ )),fT(XT)) + αR(XˆV∗ ) . (4)

XˆV∗

###### B.2.1. Relaxation of Eq. 4

Computing XˆV∗ as in Eq. 4 might produce images that are widely different from the training distribution of natural images, as shown in Fig. 8. So, instead of inverting the whole visual encoder fV , we can invert just a subset of layers ΨV ⊂ fV , starting from the top of the model:

ZˆV∗ = Ψ−V1(fT(XT)) ≈ max

sim(ΨV (ZˆV∗ ),fT(XT))

ZˆV∗

+R(ZˆV∗ )

(5)

where R could be a regularization similar to style transfer [14] to encourage ZˆV∗ to be similar to the intermediate representations of natural images.

##### B.3. Implicit Knowledge Transfer

Although in this work we focus on explicit knowledge transfer, we briefly present the idea behind Implicit Knowledge Transfer for the sake of completeness. It has been shown how multi-modal neurons can be found in multi-modal models [16, 62]. These neurons exhibit high activation on the same concepts in either modality, meaning that they are able to capture cross-modal representations. We hypothesize that in a shared-parameter architecture (e.g. early-fusion transformers [33, 49]) it should be possible to exploit these neurons for knowledge transfer, for example with simple masked language modeling on the novel concept description, effectively eliminating the need for model inversion. For this purpose, early-fusion architectures that can process single modalities independently would be required. However, to the best of our knowledge, we are not aware of many large pre-trained models satisfying these requirements at the time being, hence we leave an in-depth exploration of this path for future research. Hints that training on different modalities independently can help can be found in the literature, for example during pre-training of U-VisualBERT [40]. Even more relevant to our research, authors in [70] report some capabilities of cross-modal transfer on SimVLM, however, the model is proprietary and we are unable to reproduce their claims. Thus, here we focus on ViLT and we report some preliminary analysis in Sec. D.3.

##### B.4. Open questions

Q1 Domain Gap. Inverted images, as shown in Fig. 8, appear widely different from natural images. However, as shown by the results in the paper, fine-tuning the models

|Trainable<br><br>[Figure 23]|
|---|

###### INVERSION STAGE FINE-TUNING STAGE

Inversion InfoNCE

Vision Encoder

Text Encoder

Vision Encoder

Text Encoder

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

"A series of gleaming silver rings, each nested perfectly within the next, surrounds a central disk that spins smoothly...."

"A gyroscope is made by ... a series of gleaming silver rings, each nested perfectly within the next...."

[Figure 29]

[Figure 30]

Figure 7. Graphical overview of Knowledge Transfer. Starting from a textual description of the target concept, we synthesize images via model inversion (left) then, using an image-text matching loss, we fine-tune the visual encoder to match the concept (right). In this way, we leverage prior knowledge contained in the model (from pre-training) to learn novel concepts.

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

(a) Moongate. Caption: A perfectly circular archway built from uniformly cut stones or bricks, set into a larger wall. It forms a smooth circle, framing views of gardens or landscapes beyond, creating a picturesque portal.

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

(b) Tonometer. Caption: A slender, pen-like probe attached to a small base equipped with precise dials and gauges. This tool is often part of a larger medical apparatus, featuring a metallic finish and a refined, professional appearance.

Figure 8. Example of inverted images (top) and real images (bottom) from rare concepts that CLIP struggles to classify correctly.

on them leads to improved results. Is a domain gap present between inverted and real images? Or is it indicative of a fundamental difference in which deep models process visual information? This phenomenon may be linked with adversarial attacks [71].

Q2 Generalizability of inversion An interesting point to analyze, which could provide some insights for Q1, is the generalizability of the inverted images. For example, can images inverted with a certain model (e.g. CLIP) be used for training some other model from scratch? Or are they “fitted” to only work with the specific model used for inversion?

Q2 Catastrophic Forgetting To what extent can we prevent catastrophic forgetting when applying Knowledge Transfer? In this work, we show that lower learning rates generally achieve a good trade-off between learning novel concepts and preserving previous information. However, there is still room for improvement. For example, LoRA [21] has been shown to help in avoiding catastrophic forgetting during

fine-tuning, hence applying it during Knowledge Transfer could further improve the results. Also, Implicit Transfer (on shared-parameter models) might avoid catastrophic forgetting better than Explicit Transfer, for example by focusing on multi-modal neurons.

### C. Experimental Setup

##### C.1. Training Details

Captioning To produce descriptive captions for the new concepts, we employ a LLM-based approach. Specifically, for natural images, we use Llama-3 Instruct (with 8B parameters) [2] with the following prompt: “Generate a small description of the ImageNet class <class name> without using the word itself. The description must contain visual cues useful for recognizing the subject with low-level and accurate details. Please don’t insert anything else in the response except the description.”, where we insert the appropriate class name for each new concept. Note that we employ an LLM only for the sake of convenience (e.g. captioning

all 1000 ImageNet classes), but this is not a requirement. For medical data, we actually employ a mix of hand-crafted captions based on Radiopaedia [67] augmented with some elements from ChatGPT-4 [52]. All captions can be found in the supplementary material.

Inversion We run inversion for 5k steps, using a cosine learning rate annealing schedule. For the regularization term, we use the default value α = 0.005 [31]. The augmentation we employ is composed of random affine transformations (rotation comprised between -30 and +30 degrees, a translation of 10%, and a scaling comprised between 70% and 100% of the image size), with a probability of 0.5. An example of inverted images can be found in Fig. 8. For each concept, we generate ten inverted samples.

Fine-tuning Fine-tuning is performed using the InfoNCE loss to achieve alignment between the inverted images and the textual descriptions. We only fine-tune the visual encoder, while keeping the text encoder frozen. The motivation is that we wish to align features extracted from the visual encoder to those extracted from the text encoder. For most experiments, we perform a quick fine-tuning consisting of only one single epoch, with small learning rates between 10−6 and 10−4. For CLIP-based models, we generally employ a weight decay of 0.2 as in [56]. More details are provided in the description of each experiment.

##### C.2. Datasets

We employ a variety of datasets in different domains and for different downstream tasks. Here we provide a complete list, divided by task. Note that we do not use any training data from these datasets, as we only use them for testing. All improvements come from the textual description.

###### Natural images classification

- 1.) RareConcepts is a collection of images of rare concepts gathered from the web. We release the dataset as part of this work. In our experiments, we focus on concepts that are relatively unknown to different large multi-modal architectures: Moongate, Gyroscope and Tonometer, together with four additional fine-grained or deformable categories and geometric patterns (fabric patterns: Bengal Strip, Madras, Floral; parquet pattern: Chantilly). For each concept, we collect 10 images.
- 2.) ImageNet-1k [11] is a large-scale benchmark for visual recognition, with 1000 classes and 3.2M natural images.

Medical images classification

- 3.) CheXpert-2x500c [23] is a dataset of Chest X-Rays obtained from the large-scale CheXpert dataset [26] by considering 200 examples for the classes Atelectasis, Cardiomegaly,

- Edema, Consolidation, and Pleural Effusion.
- 4.) JSRT [65] is a Chest X-Ray dataset containing 154 conventional chest radiographs with a lung nodule of different types (malignant and benign nodules).

Medical images segmentation

- 5.) UnitoChest [7] is a collection of 306,440 chest CT slices coupled with nodules segmentation masks. We consider slices where nodules are present, for a total of 4179 images.
- 6.) UDIAT [72] is a dataset of breast masses in ultrasound images, containing 110 benign and 54 malignant cases.
- 7.) SIIM Pneumothorax [75] is a Chest X-ray dataset for pneumothorax segmentation, released as a challenge in 2019. We consider a total of 500 images.
- 8.) BraTS23 Glioma [1] is a brain MRI dataset of adult patients with brain gliomas. We consider all slices where a tumor is present for a total of 14,746 images.

Image-Text retrieval and captioning

- 9.) Flickr30k [73] is a dataset of 31,783 images from Flickr, each one associated with 5 captions provided by human annotators. For our experiments, we used Karpathy’s test split [30], which contains 1000 images and 5000 captions.
- 10.) MSCOCO [41] is a large-scale dataset of more than 330k images with textual captions. We use Karpathy’s test split [30], containing 5000 images.

##### C.3. Controlled Experiments

###### C.3.1. Rare Concepts (CLIP and ViLT)

We train using the Adam optimizer, with a batch size of 4, a weight decay of 0.2, and learning rates between 1e-5 and 5e-5 as reported in the table in the main text. We train using 10 inverted images for each concept. The captions used for inversion can be found in Tab. 20.

Details about image inversion for ViLT For ViLT we use a slightly different approach, in order to accommodate the different architecture. To run inversion, we start from a pair of input < xt,xˆ∗v ∼ N(0;1) > composed by the textual caption and random noise. We then optimize xˆ∗v by optimizing the image-text matching (ITM) score computed on the ITM head of ViLT [33]. This head outputs two values: one indicating no match and the other indicating a match. To optimize this, we use the cross-entropy loss during inversion, aiming to maximize the output corresponding to a match while minimizing the output for no match. The rest of the setup is the same as CLIP. Furthermore, we disabled the random affine augmentation, as it produced noisy inverted images. Additionally, we use a weight decay value of 0.01, which is consistent with the one used by the authors of ViLT. The captions used for inversion with ViLT can be found in Tab. 21.

###### C.3.2. KT on Medical Images (MedCLIP)

For MedCLIP, we use the same setup as CLIP on rare concepts, see Sec. C.3.1. Namely, we employ Adam with a batch size of 4 and a weight decay of 0.2, using 10 inverted images for each concept. The descriptions used for inversion with MedCLIP can be found in Tab. 22.

###### C.3.3. CLIP on medical images (out of domain KT)

For ViT-B/32, we use a learning rate of 5e-5 with a batch size of 8, and we train for 5 epochs; for ViT-L/14 we use a learning rate of 1e-5, a batch size of 4, and we train for 2 epochs. The captions used for inversion are reported in Tab. 23.

##### C.4. Real-world experiments C.4.1. Captioning (CoCa)

In these experiments, we deal with two types of captions: the first is the concept caption, that we use for inversion and fine-tuning with InfoNCE as in all other experiments (listed in Sec. G), the second is the target caption that we use to fine-tune the autoregressive captioning decoder of CoCa with Lcap.

Captioning Loss When fine-tuning on inverted images, we apply an autoregressive captioning loss, as defined in [74]:

T

log Pθ(yt|y<t,x) (6)

Lcap = −

t=1

which aims at predicting the next token yt given the previous tokens y<t and the image x. The final objective function that we optimize is the combination of the InfoNCE loss and the captioning loss:

L = λ1LCLIP + λ2Lcap (7)

where λ1,λ2 ≥ 0. In our fine-tuning, we use λ1 = 1 and λ2 = 0.1.

Target captions template We use a set of 26 different templates as target captions during fine-tuning. At each optimization step, we select a random template for each sample in the following manner:

- 1 TEMPLATES = (

- 2 lambda c: f’a bad photo of a {c}.’,

- 3 lambda c: f’a low resolution photo of the {c}.’,

- 4 lambda c: f’a rendering of a {c}.’,

- 5 lambda c: f’a bad photo of the {c}.’,

- 6 lambda c: f’a cropped photo of the {c}.’,

- 7 lambda c: f’a photo of a hard to see {c}.’,

- 8 lambda c: f’a bright photo of a {c}.’,

- 9 lambda c: f’a photo of a clean {c}.’,

- 10 lambda c: f’a photo of a dirty {c}.’,

- 11 lambda c: f’a dark photo of the {c}.’,

- 12 lambda c: f’a photo of my {c}.’,

- 13 lambda c: f’a bright photo of the {c}.’,

- 14 lambda c: f’a cropped photo of a {c}.’,

- 15 lambda c: f’a photo of the {c}.’,

- 16 lambda c: f’a good photo of the {c}.’,

- 17 lambda c: f’a rendering of the {c}.’,

- 18 lambda c: f’a photo of one {c}.’,

- 19 lambda c: f’a close-up photo of the {c}.’,

- 20 lambda c: f’a photo of a {c}.’,

- 21 lambda c: f’a low resolution photo of a {c}.’,

- 22 lambda c: f’a photo of a large {c}.’,

- 23 lambda c: f’itap of the {c}.’,

- 24 lambda c: f’a jpeg corrupted photo of the {c}.’,

- 25 lambda c: f’a good photo of a {c}.’,

- 26 lambda c: f’itap of a {c}.’,

- 27 lambda c: f’a photo of the large {c}.’,

- 28 )

- 29

- 30 template_idx = torch.randint(

- 31 0, len(TEMPLATES), (1,)

- 32 ).item()

- 33 template = TEMPLATES[template_idx]

- 34 return tokenize(template(class_name))

These templates are inspired by OpenAI prompt ensembling for zero-shot classifiers1. We use these captions although they are not in the exact style of MSCOCO, as we do not want to leverage information from MSCOCO besides the concept classes. Target captions crafted specifically for MSCOCO might further improve the results.

### D. Additional Results

##### D.1. Controlled Experiments

- D.1.1. Rare Concepts (CLIP and ViLT)

Here we report results across differen learning rate values, in Fig. 9. Results in numerical forms can be found in Tab.11.

- D.1.2.KTonFine-grainedanddeformablerare-concepts

Tab. 12, we perform additional experiments on fine-grained categories, including both deformable (fabric) and nondeformable (parquet) patterns.

- D.1.3. KT on medical images (MedCLIP)

The full results on JSRT with MedCLIP can be found in Tab. 13.

##### D.2. Real-world Experiments D.2.1. Segmentation

Results of knowledge transfer on MedCLIP-SAMv2 with different values of learning rate are shown in Tab. 14. We report illustrative examples of the improvements achieved by knowledge transfer in Fig. 10 and Fig. 11. The captions used for inversion for segmentation can be found in Tab. 24.

1https://github.com/mlfoundations/open_clip/ blob/main/src/open_clip/zero_shot_metadata.py

CLIP ViT-B/32 CLIP ViT-L/14 ViLT

Moongate

Tonometer

Gyroscope

- 2.0 5

- 3.0 5

- 4.0 5

- 4.0 5

- 5.0 5

70

70

70

ImageNet0-shotAccuracy(%)

ImageNet0-shotAccuracy(%)

ImageNet0-shotAccuracy(%)

Baseline 1.0 5 2.0 5 3.0 5

5.0 5

Baseline 1.0 5

Baseline 1.0 5 3.0 5 2.0 5

65

65

60

- 2.0 5

- 3.0 5

- 4.0 5

- 5.0 5

60

Baseline 1.0 5

- 4.0 5

- 5.0 5

60

Baseline 1.0 5 2.0 5

55

50

1.0 5 2.0 5

3.0 5

Baseline

- 3.0 5

- 4.0 5

- 5.0 5

ViLT

ViLT

ViLT

55

- 4.0 5

- 5.0 5

50

24.25

24.25

40

- 1.0 5

- 2.0 5

Baseline

4.0 5 5.0 5

- 4.0 5

- 5.0 5

23.75

3.0 5

45

50

- 1.0 5

- 2.0 5

3.0 5 2.0 5

- 3.0 5

- 4.0 5

24.00

24.00

23.50

1.0 5

40

30

45

23.25

Baseline

Baseline

5.0 5

23.75

23.75

35

1 0 1

20 40

40 60

40

0 20 40 60 80 100

30 40 50 60 70 80 90 100

88 90 92 94 96 98 100 102

Target Accuracy (%)

Target Accuracy (%)

Target Accuracy (%)

- Figure 9. Knowledge Transfer (KT) on novel and rare concepts (CLIP and ViLT) across different learning rates. In most instances, we achieve improvement (even notable) in the target accuracy on the novel concept, preserving original knowledge (measured as accuracy on ImageNet). We also observe that on ViLT the accuracy on ImageNet generally improves when performing KT.

Learning Rate Model Concept Baseline 1e-5 2e-5 3e-5 4e-5 5e-5 CLIP ViT-B/32 [56] Moongate Target Acc. 0% 10% 60% 90% 100% 100%

ImageNet 0-shot 58.10% 57.78% 56.43% 53.95% 50.37% 42.30% Tonometer Target Acc. 50% 80% 80% 100% 100% 100%

ImageNet 0-shot 58.10% 57.52% 55.62% 51.98% 42.80% 23.73% Gyroscope Target Acc. 90% 100% 100% 100% 100% 100%

ImageNet 0-shot 58.10% 57.86% 56.84% 53.96% 48.28% 34.48% CLIP ViT-L/14 [56] Moongate Target Acc. 78.95% 78.95% 100% 100% 100% 100%

ImageNet 0-shot 70.79% 70.74% 70.51% 69.96% 68.57% 62.35% Tonometer Target Acc. 31.58% 52.63% 78.95% 100% 100% 100%

ImageNet 0-shot 70.79% 70.74% 70.61% 70.08% 69.06% 66.92% Gyroscope Target Acc. 90% 90% 100% 100% 100% 100%

ImageNet 0-shot 70.79% 70.65% 70.42% 69.84% 69.39% 68.35% ViLT [33] Moongate Target Acc. 0% 0% 0% 0% 0% 0%

ImageNet* 0-shot 23.74% 23.90% 24.02% 24.16% 24.18% 24.16% Tonometer Target Acc. 10% 30% 30% 30% 40% 40%

ImageNet* 0-shot 23.74% 23.88% 24.02% 24.04% 24.22% 23.94% Gyroscope Target Acc. 50% 60% 50% 50% 40% 30%

ImageNet* 0-shot 23.74% 23.80% 23.88% 23.72% 23.38% 23.12%

- Table 11. Knowledge Transfer on novel and rare concepts (CLIP and ViLT) in terms of accuracy. * for VilT, we employ ImageNet-100 [29] due to the computational requirements of evaluating every possible image-caption pair for zero-shot classification.

Differences in downstream tasks As said in the main text, lung nodules and pneumothorax segmentations are novel tasks on which MedCLIP-SAMv2 was not pre-trained. Regarding brain tumors, we employ the BraTS 2023 glioma dataset, which contains brain gliomas in adult patients. With respect to the original performance reported in [37] on brain tumors, we notice a significant gap. However, the preprocessing of the images is quite different, as data from BraTS 2023 is more heavily preprocessed (e.g. skull stripping) than in [37]. We were not able to compare MedCLIP-SAMv2 on

the original data, as, at the time of writing, details about the data split are missing.

###### D.2.2. Text-image retrieval

In this section, we show the full results for text and image retrieval tasks on Flickr30k with ViLT. Tab. 15 is the extended version of the results in which we report the huggingface’s pre-trained baseline, along with the results of the experiments we performed while tuning the learning rate and the batch size. We report the best batch size for each learning

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

[Figure 79]

[Figure 80]

(a) Image (b) Ground Truth (c) M2IB Map (Baseline)

(d) Final Segmentation (Baseline)

(e) M2IB Map (Knowledge Transfer)

(f) Final Segmentation (Knowledge Transfer)

- Figure 10. Qualitative evaluation of knowledge transfer on breast tumor segmentation (UDIAT dataset). We report the top ten most illustrative examples in which knowledge transfer improved segmentation, in terms of DSC.

Fabric Parquet Model Bengal Stripe Floral Madras Chantilly

CLIP ViT-B/32 0% 10% 0% 20% CLIP ViT-B/32 + KT (ours) 20% 30% 10% 70% ImageNet (KT) 57.08% 57.63% 57.33% 56.12%

CLIP ViT-L/14 70% 60% 80% 0% CLIP ViT-L/14 + KT (ours) 90% 80% 90% 10% ImageNet (KT) 70.82% 70.73% 70.79% 69.90%

- Table 12. KT applied to fine-grained and deformable categories (fabric and parquet patterns), expanding the Rare Concepts dataset.

rate. As can be seen our method works best with smaller learning rates in this setting. The captions used for inversion (mscoco) can be found in Tab. 25.

###### D.2.3. Captioning

We report additional results of captioning, with and without the use of captioning loss in Tab. 16. Even without captioning loss, we are achieve improvements over both version CoCa (pre-trained on LAION-2B) and CoCa FT (fine-tuned on MSCOCO). By applying Lcap we achieve a further increase in the reported metrics. We showcase some

improvements in captioning on the CoCa model pre-trained on LAION-2B in Fig. 17. The concept captions used for inversion can be found in Tab. 25.

##### D.3. Preliminary results with Implicit Knowledge Transfer

In this section, we show preliminary results about Implicit Knowledge Transfer, presented in Sec. B.3. In Implicit Knowledge Transfer, the objective is to teach the model a novel concept by only training it on text, without using inverted images. To do so with ViLT, we no longer use the image-text matching objective as it requires images, instead, we employ masked language modeling (MLM) [12], using as input a pair composed of the textual description of the concept and random noise instead of the image. The assumption is that, in a model with parameters shared between modalities, fine-tuning on one modality (text), will also benefit the other modality. Here, our hypothesis is that during finetuning, multi-modal neurons [62] can help in transferring knowledge across modalities.

Learning Rate (multiplier) Concept Baseline ×1 ×2 ×3 ×4 ×5 Benign Nodule Target Acc. (base lr 1e-5) 54.55% 54.55% 54.55% 54.55% 54.55% 54.55%

- CheXpert-5x200c 0-shot 62.10% 61.80% 62.30% 62.10% 62% 62.20%

Lung Cancer Target Acc. (base lr 1e-4) 83.93% 87.50% 92.86% 94.64% 92.86% 92.86%

- CheXpert-5x200c 0-shot 62.10% 62.20% 61.50% 53.70% 48.20% 44.50%

###### Table 13. Knowledge Transfer on MedCLIP on the JSRT dataset (accuracy). Full results across learning rates.

Lung Nodules† Lung Pneumothorax† Breast Ultrasound Brain MRI Model DSC NSD IoU DSC NSD IoU DSC NSD IoU DSC NSD IoU MedCLIP-SAMv2 14.83% 17.30% 8.64% 6.30% 7.61% 3.75% 56.25% 59.44% 47.81% 17.20% 20.97% 12.05%

- Transf. (1e-5) 13.95% 17.45% 8.75% 6.28% 7.59% 3.77% 58.23% 61.56% 49.52% 15.90% 19.36% 11.10%
- Transf. (2e-5) 14.10% 17.65% 8.83% 6.41% 7.76% 3.83% 54.36% 57.30% 46.30% 18.13% 22.26% 12.62%
- Transf. (3e-5) 14.10% 17.65% 8.85% 6.25% 7.55% 3.73% 55.70% 59.00% 47.49% 15.47% 18.85% 10.78%
- Transf. (4e-5) 14.25% 17.85% 8.94% 6.24% 7.57% 3.71% 53.86% 56.82% 45.61% 15.26% 18.63% 10.62%
- Transf. (5e-5) 14.20% 17.78% 8.92% 6.20% 7.51% 3.70% 54.90% 57.97% 46.09% 16.22% 19.81% 11.34%

- Transf. (1e-4) 14.35% 18.03% 9.04% 6.02% 7.29% 3.59% - - - - - -
- Transf. (2e-4) 10.74% 13.64% 6.66% 4.71% 5.54% 2.86% - - - - - -

Table 14. Full results on zero-shot segmentation with MedCLIP-SAMv2.

###### D.3.1. Implicit Knowledge Transfer with MLM

For Implicit Knowledge Transfer we used the same masked language modeling setup as in ViLT [33], which means that we use whole-word masking and a masking probability of 15%. We use 10 examples for fine-tuning, each of which is composed by a random noise image and a masked caption. The masked captions are generated starting from the same caption by masking differently each time. For the caption we use the template “A X is Y ”, where X is the name of the concept and Y is the concept’s description (from Tab. 21). We use a batch size of 4 with different learning rates, for a total of 3 train steps. Weight decay is set, as in the other experiments, to 0.01.

Explicit Knowledge Transfer baseline with MLM For comparison, we also evaluate the results of explicit knowledge transfer with the masked language modeling objective instead of the image-text matching objective. We use the same setup as the implicit one, with the only exception that instead of random noise images, we use inverted images. In particular, we use the same inverted images we used for the explicit knowledge transfer with the image-text matching objective.

###### D.3.2. Results discussion

Tab. 18 reports the results for both implicit and explicit knowledge transfer with masked language modeling. In both cases, no improvements are observed for the moongate concept, whose accuracy stays at 0%. For tonometer, explicit knowledge transfer seems to work better since with the implicit one, there is a loss of performance, while for

gyroscope the opposite is true. In all cases, we observe an increase in the accuracy over the ImageNet-100 classes, as observed when using image-text matching objective. The only improvement is registered for the gyroscope concept in the implicit transfer setting, from 50% to 60%. Overall we can say that implicit knowledge transfer with masked language modeling does not work for the ViLT model, this is probably due to the fact that ViLT was pre-trained on image-text pairs, which means that it expects both modalities in input. Regarding explicit knowledge transfer with MLM, more experiments are needed to determine the correct algorithm and set of hyperparameters to make it work, for example, we may have to use more examples generated from different textual descriptions.

### E. Ablation studies

##### E.1. Fine-tuning strategy

We perform an ablation study on our fine-tuning strategy. In our experiments, during fine-tuning, we freeze the text encoder and only train the visual encoder. Here we evaluate fine-tuning with different configurations. The results are illustrated in Fig. 12. When fine-tuning both encoders, we observe a rapid collapse of target accuracy and ImageNet accuracy for all concepts. We also observe a similar trend when fine-tuning the text encoder only, while leaving the visual encoder frozen. This is, however, expected as our assumption is that the knowledge contained in the text encoder is already good enough to represent the target concept, and we just wish to align visual features to it. Moreover, if we alter the text encoder weights, correspondence between cap-

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

(a) Image (b) Ground Truth (c) M2IB Map (Baseline)

(d) Final Segmentation (Baseline)

(e) M2IB Map (Knowledge Transfer)

(f) Final Segmentation (Knowledge Transfer)

- Figure 11. Qualitative evaluation of knowledge transfer on brain tumor segmentation (BraTS 2023 glioma dataset). We report the top ten most illustrative examples in which knowledge transfer improved segmentation, in terms of DSC.

Flickr30k (1K) Text Retrieval Image Retrieval

Model LR Batch Size R@1 R@5 R@10 R@1 R@5 R@10 ViLT-B/32 (huggingface) - - 73.8% 93.5% 96.5% 57.3% 83.9% 90.4%

- ViLT-B/32 8e-7 32 74.5% 93.8% 96.4% 57.7% 84.0% 90.4%
- ViLT-B/32 9e-7 32 74.6% 93.8% 96.4% 57.8% 84.0% 90.5%

- ViLT-B/32 1e-6 16 74.4% 93.8% 96.5% 57.7% 84.1% 90.5%
- ViLT-B/32 2e-6 128 74.6% 93.7% 96.5% 57.8% 84.0% 90.5%
- ViLT-B/32 3e-6 256 74.5% 93.9% 96.5% 57.7% 83.9% 90.5%
- ViLT-B/32 4e-6 32 73.8% 93.6% 96.5% 57.4% 84.0% 90.5%
- ViLT-B/32 5e-6 256 74.5% 93.9% 96.5% 57.6% 84.0% 90.5% ViLT-B/32 8e-6 32 73.2% 93.7% 96.1% 57.4% 83.7% 90.4%

- ViLT-B/32 1e-5 128 74.4% 93.8% 96.8% 56.8% 83.7% 90.6%
- ViLT-B/32 2e-5 32 71.8% 93.2% 96.4% 56.7% 83.6% 90.4%
- ViLT-B/32 3e-5 32 70.8% 92.1% 95.7% 56.0% 82.9% 90.2%

- Table 15. Full results for text and image retrieval on Flickr30k with ViLT. The first section reports baseline results, while the second shows the outcome of each tested learning rate and its optimal batch size (chosen among 16, 32, 64, 128, and 256). Recall scores at top 1, 5, and 10 are reported.

MSCOCO (5K) Model BLEU@4 METEOR CIDEr SPICE

CLIP-ViL [64] 40.2 29.7 134.2 23.8 BLIP [39] 40.4 - 136.7 VinVL [77] 41.0 31.1 140.9 25.4 SimVLM [70] 40.6 33.7 143.3 25.4 LEMON [22] 41.5 30.8 139.1 24.1 CoCa [74] (proprietary) 40.9 33.9 143.6 24.7

CoCa 6.9 12.8 31.1 9.1 CoCa (transf. 6e-5) 13.6 18.5 47.3 13.6 CoCa† (transf. 9e-5) 17.9 19.4 60.8 13.7

CoCa FT 34.9 29.7 123.1 23.5 CoCa FT (transf. 2e-5) 35.2 29.8 123.1 23.2 CoCa FT† (transf. 5e-6) 35.2 29.8 124.0 23.3

- Table 16. Image captioning on MSCOCO. † means the decoder is also fine-tuned. CoCa refers to the baseline model pre-trained on LAION-2B [61], while CoCa FT refers to the model fine-tuned for captioning on MSCOCO. We highlight in bold the best results overall and the improvements achieved by Knowledge Transfer.

tions and inverted images may be lost, leading to degenerate cases.

##### E.2. Captions construction

We focus on the construction of the captions for fine-tuning. As explained in the main text, during fine-tuning we prepend each caption with the name of the concept, for example “A moongate is [...]”. Here we motivate why this is necessary by comparing captions prepended with the name and captions without the name. The results are shown in Fig. 13. As we can observe, using the name of the concept during fine-tuning is necessary in order to map visual features to its textual description.

##### E.3. Captions Quality

In Tab. 19 we reports the prompts used in the ablation study on prompt quality and length, with P1 indicating a short human-written prompt, P2 mid-sized human caption with some clear visual hints, and P3 the original LLM-generated description that we used in our experiments (which can be found in Sec. G).

#### F. Code Code will be publicly released upon paper acceptance.

###### MSCOCO Sample

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

Actual A shot of a clock in

A pianist in a suit and glasses playing a keyboard.

A baseball player hitting a ball in a professional game.

A baseball holding a baseball bat during a baseball game.

Two people that are sitting on a table.

the train station.

Baseline grand - central - station - new - york.jpg (METEOR 0.0)

Paul Kagasoff, president and chief executive officer of Intel corp., speaks during the 2012 Computex trade show in Taipei, Taiwan (METEOR 8.6)

Aaron judge 2016 new york Yankees (METEOR 5.0)

20080419 _ mariners _ 0001 | by Mike. Smith (METEOR 4.2)

Dinner in our tiny studio apartment in Amsterdam. (METEOR 0.0)

Knowledge Transfer A black and white photo of a clock at Grand Central terminal. (METEOR 92.9)

A photo of a man in a suit sitting at a keyboard. (METEOR 86.1)

A baseball player swings his bat at a batter. (METEOR 83.6)

A baseball player takes a swing at a pitch. (METEOR 86.9)

A photo of a man and a woman sitting at a kitchen table. (METEOR 80.8)

###### Sample

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Actual A baseball batter up at the plate that just hit a ball

A pair of red scissors sitting on a newspaper.

A young man sitting at a table with a pizza.

A man in a black suit kicking around a soccer ball.

The man in a black tie sits in a chair with his shirt sleeves rolled up.

Baseline david - ortiz _ display _ image _ display _ image _ display _ image _ display _ image _ display _ image _ display _ image (METEOR 3.9)

Your scissors are now digitized (METEOR 9.5)

Pizza and beer in Chicago, Illinois. Photo via Flickr: David J. Abbott M.D. (METEOR 8.3)

blatter - foot ball.jpeg (METEOR 5.6)

Avatar for Marc Anthony (METEOR 5.3)

Knowledge Transfer A baseball player swings his bat at a pitch. (METEOR 80.3)

A photo of a pair of red scissors on a piece of paper. (METEOR 84.9)

A photo taken of a man sitting at a table with a plate of pizza. (METEOR 83.5)

A man in a suit kicking a soccer ball on a field. (METEOR 80.7)

A man in a white shirt and black tie sits on a chair. (METEOR 78.4)

- Table 17. Visual example of captioning on MSCOCO. We report the top ten most illustrative examples in which knowledge transfer improved captioning, in terms of METEOR score.

Learning Rate Type Concept Baseline 1e-5 2e-5 3e-5 4e-5 5e-5 Implicit Moongate Target Acc. 0% 0% 0% 0% 0% 0%

ImageNet* 0-shot 23.74% 23.82% 23.90% 23.98% 23.94% 23.86% Tonometer Target Acc. 10% 10% 10% 10% 10% 0%

ImageNet* 0-shot 23.74% 23.84% 23.86% 23.70% 23.64% 23.60% Gyroscope Target Acc. 50% 50% 60% 60% 60% 50%

ImageNet* 0-shot 23.74% 23.74% 23.62% 23.42% 23.44% 23.46% Explicit Moongate Target Acc. 0% 0% 0% 0% 0% 0%

ImageNet* 0-shot 23.74% 23.80% 24.08% 24.02% 24.10% 24.20% Tonometer Target Acc. 10% 10% 10% 10% 10% 10%

ImageNet* 0-shot 23.74% 23.80% 23.74% 23.72% 23.70% 23.56% Gyroscope Target Acc. 50% 50% 50% 50% 40% 30%

ImageNet* 0-shot 23.74% 23.74% 23.84% 23.84% 23.84% 23.82%

- Table 18. Knowledge Transfer on novel and rare concepts using masked language modeling with ViLT. In the Implicit Knowledge Transfer, we pass noise images along with a corresponding masked caption to ViLT; in the explicit one, we replace noise images with inverted images.

Moongate

Tonometer

Gyroscope

visual (target)

100

100

100

visual (imagenet)

0-shotAccuracy

0-shotAccuracy

0-shotAccuracy

80

80

80

text (target)

text (imagenet)

60

60

60

both (target)

both (imagenet)

40

40

40

20

20

20

0

0

0

Baseline 1e-5 2e-5 3e-5 4e-5 5e-5

Baseline 1e-5 2e-5 3e-5 4e-5 5e-5

Baseline 1e-5 2e-5 3e-5 4e-5 5e-5

Learning Rate

Learning Rate

Learning Rate

- Figure 12. Comparison of fine-tuning strategies. Fine-tuning both the text and the visual encoders, or just the text encoder leads to a collapse in accuracy. Fine-tuning only the visual encoder correctly aligns prior visual features to the novel concept. A good choice of learning rate leads to higher accuracy on the novel concept (target) while limiting catastrophic forgetting on previous tasks (imagenet).

Moongate

Tonometer

Gyroscope

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

100

100

100

w/ name (target)

w/ name (imagenet)

90

90

80

w/o name (target)

80

w/o name (imagenet)

0-shotAccuracy

0-shotAccuracy

0-shotAccuracy

80

70

60

70

60

40

60

50

50

40

20

40

30

0

20

Baseline 1e-5 2e-5 3e-5 4e-5 5e-5

Baseline 1e-5 2e-5 3e-5 4e-5 5e-5

Baseline 1e-5 2e-5 3e-5 4e-5 5e-5

Learning Rate

Learning Rate

Learning Rate

Figure 13. Ablation study on caption construction for finetuning.

Moongate P1 A circular stone archway P2 A circular archway built from uniformly cut stones or bricks P3 original LLM caption Tonometer P1 An instrument to measure pressure P2 A pen-like instrument with dials and pressure gauges P3 original LLM caption Gyroscope P1 A device to measure orientation and angular velocity P2 A measuring device composed of a spinning wheel or disc to measure orientation and

angular velocity P3 original LLM caption

Table 19. Prompt ablations (P1 - short human caption; P2 - mid human caption; P3 - longer LLM caption).

### G. List of captions

Table 20. Descriptions for rare concepts (generated with Llama-3-8B-Instruct).

Moongate A perfectly circular archway built from uniformly cut stones or bricks, set into a larger wall. It forms a smooth circle, framing views of gardens or landscapes beyond, creating a picturesque portal.

Tonometer A slender, pen-like probe attached to a small base equipped with precise dials and gauges. This tool is often part of a larger medical apparatus, featuring a metallic finish and a refined, professional appearance.

Gyroscope A series of gleaming silver rings, each nested perfectly within the next, surrounds a central disk that spins smoothly. The rings are connected by intersecting axes, allowing the disk to tilt and rotate freely while maintaining a sophisticated, mechanical look.

Table 21. Manually shortened descriptions for rare concepts (to fit into ViLT’s 40 token input)

Moongate A perfectly circular archway built from uniformly cut stones or bricks, set into a larger wall. It forms a smooth circle, framing views of gardens, creating a picturesque portal. Tonometer A slender, pen-like probe attached to a small base equipped with precise dials and gauges. This tool

is often part of a larger medical apparatus. Gyroscope A series of rings each nested within the next, surrounds a central disk that spins. The rings are connected by intersecting axes allowing the disk to rotate freely.

Table 22. Descriptions for medical classes for JSRT (Mix with Radiopaedia and ChatGPT-4).

Benign Nodule A small, round spots appearing in Chest X-Ray, typically well-defined with smooth, regular borders. These spots are often uniformly dense and do not cause distortion of surrounding structures. Lung Cancer A dense and irregular mass on Chest X-Ray images often with spiked or uneven edges. It may appear in the lung’s periphery or near the airways.

Table 23. Descriptions for medical classes for CheXpert-5x200c (obtained with a mix of Radiopaedia and ChatGPT-4).

Atelectasis A small areas of collapsed lung. It is usually seen on Chest X-Rays as small volume linear shadows, usually peripherally or at lung bases, appearing more opaque and shrunken. Cardiomegaly Enlargement of the heart usually seen in Chest X-Rays. The central shadow of the chest appears enlarged, extending beyond half the width of the entire chest cavity. Pleural Effusion A collection of fluid between the lungs and the chest, which makes the area appear white and smooth in Chest X-Ray images. The area does not present visible lung markings. Consolidation An area inside the lungs that appears as branching low attenuating (lucent) bronchi surrounded by high attenuating (dense) consolidated/opacified alveoli on Chest X-Ray images.

Edema An abnormal accumulation of fluid in the extravascular compartments of the lung, which makes the area whiter in Chest X-Ray images. It is usually present on both lungs.

Table 24. Descriptions for medical classes for segmentation (Mix with Radiopaedia and ChatGPT-4).

Lung Nodules Circular spots appearing within the lung fields, with clear and defined edges in CT images. They are denser than the surrounding tissue, often appearing in shades of gray or white, with varying size.

Breast Tumor A dark, irregularly shaped area is visible against the lighter surrounding tissue. The borders may appear uneven or spiculated, and the area is typically less uniform in texture. Shadowing can often be seen beneath the mass.

Pneumothorax An abnormal collection of air in the pleural space, which allows the parietal and visceral pleura to separate and the lung to collapse. The pleura edge is thin and no lung markings are visible. Brain Tumor An irregular bright mass in brain MRI, often with thick and irregular margins, surrounded by vasogenic-type edema or fluid accumulation. It may also have a hemorrhagic component.

Table 25. Descriptions for MSCOCO classes used for text and image retrieval experiments (With ChatGPT-4).

person A human figure, typically with visible head, torso, arms, and legs, in various postures. bicycle A two-wheeled vehicle with a frame, handlebars, and pedals, usually ridden by a person. car A four-wheeled enclosed vehicle with windows and doors, commonly seen on roads. motorcycle A two-wheeled motorized vehicle with a seat and handlebars, typically ridden by one or two people. airplane A large flying vehicle with wings and a tail, often seen with windows along the sides for passengers. bus A large, rectangular vehicle with many windows and seating rows, designed to carry multiple

passengers. train A long, linked series of vehicles running on tracks, often with a locomotive at the front. truck A large vehicle with a separate cab and an open or enclosed cargo area for transporting goods. boat A small to medium-sized watercraft with a hull and often visible sails or an engine. traffic light A vertical or horizontal post with red, yellow, and green lights, used to control vehicle flow at

intersections. fire hydrant A small, red, metal cylinder with nozzles on the side, often found on sidewalks for fire emergencies. stop sign A red, octagonal sign with the word "STOP" in white, used to indicate where vehicles must halt. parking meter A tall, narrow post with a small display and slot, used to pay for parking time. bench A long seat, often with a backrest, typically found in parks or public areas. bird A small animal with feathers, wings, and a beak, often shown perched or flying. cat A small, furry animal with pointed ears, whiskers, and a long tail, often seen sitting or grooming. dog A furry, four-legged animal with a tail, usually seen with a collar or leash. horse A large, four-legged animal with a mane and tail, often depicted standing or galloping. sheep A woolly animal with a round body, small head, and short legs, often seen in groups in fields. cow A large animal with a boxy body, horns, and a long face, often shown grazing or with an udder. elephant A massive, gray animal with a long trunk, large ears, and tusks. bear A large, sturdy animal with thick fur, rounded ears, and a short tail, often shown standing or walking

on all fours. zebra A horse-like animal with black and white stripes across its body. giraffe A very tall animal with a long neck and legs, spotted coat, and small horns on its head. backpack A bag with shoulder straps, typically worn on the back and used for carrying personal items. umbrella A foldable, rounded canopy on a stick, used for protection from rain or sun. handbag A small to medium-sized bag with handles, often carried by hand and used to hold personal items. tie A long, narrow piece of fabric worn around the neck, often knotted at the collar of a shirt. suitcase A rectangular, boxy container with a handle, used for carrying clothes and personal items when

traveling. frisbee A flat, round disc often made of plastic, used for throwing and catching. skis Long, narrow pieces of equipment attached to boots, used for gliding on snow. snowboard A flat, wide board attached to boots, used for sliding on snow. sports ball A round object of varying sizes, such as a soccer ball or basketball, used in sports. kite A lightweight object with a string, often shaped like a diamond or triangle, designed to fly in the

wind.

baseball bat A smooth, cylindrical wooden or metal stick used to hit a baseball. baseball glove A padded, leather glove worn on one hand, used to catch baseballs.

skateboard A narrow board with wheels, used for rolling and performing tricks. surfboard A long, flat board used for riding waves in the ocean. tennis racket An oval-shaped frame with strings and a handle, used to hit a tennis ball. bottle A narrow-necked container with a cap, often used to hold liquids like water or soda. wine glass A stemmed glass with a wide bowl at the top, used for drinking wine. cup A small, handleless vessel used for drinking, usually made of ceramic or plastic. fork A utensil with multiple prongs, used to pick up food. knife A utensil with a long, sharp blade, used for cutting food. spoon A utensil with a shallow bowl at the end of a handle, used for eating or serving food. bowl A round, deep dish, often used to hold soup or other foods. banana A long, yellow fruit with a curved shape and soft interior. apple A round fruit, typically red or green, with a stem at the top. sandwich Two slices of bread with filling in between, such as meat, cheese, or vegetables. orange A round, orange-colored fruit with a thick, textured peel. broccoli A green vegetable with a tree-like shape, featuring a thick stalk and small florets. carrot A long, orange vegetable with a pointed end, often with green leaves at the top. hot dog A sausage in a bun, often with condiments like ketchup or mustard. pizza A round, flatbread topped with cheese, sauce, and various toppings, often cut into slices. donut A round, fried pastry with a hole in the middle, often glazed or topped with sprinkles. cake A sweet, layered dessert, often decorated with frosting or fruit. chair A piece of furniture with a backrest and four legs, designed for sitting. couch A large, cushioned seat with a backrest and arms, designed for multiple people. potted plant A plant growing in a container, often with green leaves or flowers. bed A large, rectangular piece of furniture for sleeping, with a mattress and pillows. dining table A flat, often rectangular surface with legs, designed for eating meals. toilet A porcelain fixture with a seat and flushing mechanism, used in bathrooms. tv A rectangular screen on a stand or wall, used for viewing shows and movies. laptop A portable computer with a hinged screen and keyboard. mouse A small, handheld device used to control a cursor on a computer screen. remote A small, rectangular device with buttons, used to control electronics like TVs. keyboard A flat, rectangular panel with keys, used for typing on computers. cell phone A handheld electronic device with a screen and buttons or touchscreen, used for communication. microwave A box-like appliance with a door, used for heating food quickly. oven A large appliance with a door and interior racks, used for baking or roasting. toaster A small appliance with slots, used to toast bread. sink A basin with a faucet, used for washing hands, dishes, or food. refrigerator A large, box-like appliance with doors, used to store perishable food at low temperatures. book A collection of pages bound together with a cover, containing text or images. clock A circular or rectangular device with hands or digital display, showing the current time. vase A decorative container, often made of glass or ceramic, used to hold flowers. scissors A handheld tool with two blades, used for cutting paper or fabric. teddy bear A soft, stuffed toy shaped like a bear, often used by children. hair drier A handheld device that blows warm air, used to dry hair. toothbrush A small brush with a handle, used for cleaning teeth.

