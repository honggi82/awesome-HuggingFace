## Learning Vision from Models Rivals Learning Vision from Data

# arXiv:2312.17742v1[cs.CV]28Dec2023

Yonglong Tian1,† Lijie Fan2,†,* Kaifeng Chen1 Dina Katabi2 Dilip Krishnan1 Phillip Isola2 1Google Research, 2MIT CSAIL, †equal contribution

Github Repo: https://github.com/google-research/syn-rep-learn

#### Abstract

Image dataset

Text dataset

|Learner<br><br>|e.g. CLIP|
|---|
|!<br><br>|
|---|---|
| | |

We introduce SynCLR, a novel approach for learning visual representations exclusively from synthetic images and synthetic captions, without any real data. We synthesize a large dataset of image captions using LLMs, then use an offthe-shelf text-to-image model to generate multiple images corresponding to each synthetic caption. We perform visual representation learning on these synthetic images via contrastive learning, treating images sharing the same caption as positive pairs. The resulting representations transfer well to many downstream tasks, competing favorably with other general-purpose visual representation learners such as CLIP and DINO v2 in image classification tasks. Furthermore, in dense prediction tasks such as semantic segmentation, SynCLR outperforms previous self-supervised methods by a significant margin, e.g., improving over MAE and iBOT by 6.2 and 4.3 mIoU on ADE20k for ViT-B/16.

[Figure 1]

[Figure 2]

Learning from data

### !

| |Linear probe acc. on ImageNet|
|---|---|
|CLIP<br><br>|80.2%|
|StableRep<br><br>|76.7%|
|Ours|80.3%|

f : X ! Y

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

80.2%

IN lin. acc

Text dataset

Image Generator

Learner

! !

f : X ! Y

Hybrid

e.g. StableRep

###### 76.7%

IN lin. acc

Image Generator

LLM

Learner

Learning from models

! !

f : X ! Y

- 80.2% (CLIP [XX])

76.7% (StableRep [XX])

- 80.3% (Ours)

Ours

80.7%

IN lin. acc

Figure 1. Three paradigms for visual representation learning. Top row: Traditional methods, such as CLIP [71], learn only from real data; Middle row: Recent methods, such as StableRep [91], learn from real text and generated images; Bottom row: Our method, SynCLR, learns from synthetic text and synthetic images, and rival the linear transfer performance of CLIP on ImageNet despite not directly observing any real data.

#### 1. Introduction

Representation learning extracts and organizes information from raw, often unlabeled data. The quality, quantity, and diversity of the data determines how good a representation the model can learn. The model becomes a reflection of the collective intelligence that exists in the data. We get what we feed in.

recent work has indeed shown that this can lead to strong performance gains at scale [68], but this path is costly to pursue.

Unsurprisingly, the current best-performing visual representation learning methods [68, 71] rely on large scale real datasets. However, the collection of real data has its own dilemmas. Collecting large scale uncurated data [80] is relatively cheap and thus quite achievable. However, for selfsupervised representation learning, this approach exhibits poor scaling behavior –i.e., adding more uncurated data has little effect at large data scales [38, 90]. Collecting small scale curated data [24] also is achievable, but models trained in this way are limited to relatively narrow tasks. The ideal would be large scale curated datasets of real images, and

To alleviate the cost, in this paper we ask if synthetic data, sampled from off-the-shelf generative models, is a viable path toward large scale curated datasets that can train state-of-the-art visual representations.

We call such a paradigm learning from models, in contrast to directly learning from data. Models have several advantages as a data source for building large scale training sets: via their latent variables, conditioning variables, and hyperparameters, they provide new controls for curating data; we will make use of these controls in the method we propose. Models also can be easier to share and store (because models are more compressed than data), and can

*Work done while interning at Google.

produce an unlimited number of data samples (albeit with finite diversity). A growing literature has studied these properties and other advantages (and disadvantages) of using generative models as a data source for training downstream models [3, 30, 45, 48, 78, 91]. Some of these methods use a hybrid mode – either mixing real and synthetic datasets [3] or needing a real dataset to generate another synthetic dataset [91]. Other methods try to learn representations from purely synthetic data [78] but lag far behind the best performing models. Instead, we show that learning from models, without training on any real data, can yield representations that match the top-performing representations learnt from real data. For instance, as illustrated in Figure 1, representations learnt by our method are able to transfer as well as OpenAI’s CLIP [71] on ImageNet (both methods using ViT-B [28]).

|class 1|
|---|

|class 2|
|---|

|class 3|
|---|

|class 4|
|---|

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

SimCLR

| |class 1| |
|---|---|---|
| | | |

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Supervised CE

| |class 1| |
|---|---|---|
| | | |

| |class 2| |
|---|---|---|
| | | |

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

SynCLR

Figure 2. Different learning objectives treat classification granularity differently. These images are generated by two prompts “a golden retriever, wearing sunglasses and a beach hat, rides a bike" and “a cute golden retriever sits in a house made of sushi". SimCLR treats each image as a class, while supervised cross-entropy treats them all as the same “golden retrieval” class. The former does not consider shared semantics between images, and the latter is coarse-grained and ignores actions or relationships between subjects/background. Our approach, SynCLR, defines visual classes by sentences.

Our approach leverages generative models to re-define the granularity of visual classes. As shown in Figure 2, consider we have four images generated using two prompts: “a golden retriever, wearing sunglasses and a beach hat, rides a bike" and “a cute golden retriever sits in a house made of sushi". Traditional self-supervised method such as SimCLR [13] will treat each of these images as a different class; embeddings for different images are pushed apart with no explicit consideration of the shared semantics between images. On the other extreme, supervised learning methods (i.e. SupCE) will regard all these images as a single class (e.g., “golden retriever”). This ignores nuances in the semantics of the images, such as the fact that the dogs are riding a bike in one pair of images and sitting inside a sushi house in the other pair of images. Instead, our method, SynCLR, treats captions as classes, i.e., each caption describes a visual class (this level of granularity was also explored in StableRep [91]). This allows us to group images by the concepts of “riding a bike” and “sitting in a sushi house”, in addition to grouping by a coarser class label like “golden retrieval”. This level of granularity is difficult to mine in real data, since collecting multiple images described by a given caption is non-trivial, especially when scaling up the number of captions. However, text-to-image diffusion models are fundamentally built with this ability; simply by conditioning on the same caption and using different noise inputs, a textto-image diffusion model will produce different images that all match the same caption. In our experiments, we find the caption-level granularity outperforms both SimCLR and supervised training. Another advantage is that this definition of visual classes has good scalability. Unlike ImageNet-1k/21k where a given number of classes is fixed, we can augment existing classes (or data) in an online fashion, and theoretically scale up to as many classes as needed.

pability of large language models (LLMs), where we present examples of word-to-caption translations. Next, a text-toimage diffusion model is adopted to synthesize multiple images for each synthetic caption. This yields a synthetic dataset of 600M images. Then we train visual representation models by a combination of multi-positive contrastive learning [50] and masked image modeling [110].

Our learned representations transfer well. With SynCLR pre-training, our ViT-B and ViT-L models achieve 80.7% and 83.0% top-1 linear probing accuracy on ImageNet-1K, respectively, which is on par with OpenAI’s CLIP [71]. On fine-grained classification tasks, SynCLR outperforms CLIP by 3.3% for ViT-B and 1.5% for ViT-L, and performs similarly to DINO v2 [68] models, which are distilled from a pre-trained ViT-g model. For semantic segmentation on ADE20k, SynCLR outperforms MAE pre-trained on ImageNet by 6.2 and 4.1 in mIoU for ViT-B and ViT-L under the same setup, showing strong transfer ability for dense prediction tasks similar to DINO v2, which additionally involves a training period on 518x518 resolution images that SynCLR does not have.

#### 2. Related Works

Self-supervised representation learning approaches in vision develop domain-specific pre-text tasks, such as colorization [106], rotation prediction [36], and solving jigsaw

Our system consists of three steps. The first step is to synthesize a large corpus of image captions. We design a scalable approach by leveraging the in-context learning ca-

puzzles [65]. Domain-agnostic approaches have been popular, such as contrastive learning [6, 13, 40, 43, 66, 88, 97] and masked image modeling [2, 4, 5, 33, 44, 96, 100, 110]. Contrastive learning promotes invariance [89] for two views of the same image and pushes apart representations for different images [95] (or only invariance [11, 39]); the resulting representations yield strong performance for linear or zeroshot transfer. Masked image modeling reconstructs the pixels [44, 100] or local features [4], often producing excellent fine-tuning transfer performance, especially in dense prediction tasks [44]. The state of the art DINO v2 [68] leverages both approaches, and our approach shares a similar spirit.

Supervised learning [41, 52, 84] used to be the dominant approach for learning transferable visual representations for various tasks [26, 37, 81]. Recent studies [42, 57] has shown that, the transferability of representations learned in this way is limited, e.g., pre-training has no improvement over random initialization for dense prediction tasks (e.g., object detection) when the fine-tuning is long enough. Such limitation continues when the model has been scaled up to 22B [23]. An alternative paradigm learns visual representations from text supervision [49, 71], e.g., CLIP [71]. This approach is more flexible (i.e., not requiring classes) and provides richer supervision, often learning generalizable representations.

Generative models as representation learners. A number of papers have explored the representations that are learned by generative models for various recognition tasks [25, 56]. As might be expected intuitively, such models indeed learn especially good representations for dense tasks, such as optical flow estimation [79], semantic segmentation [8, 101], and depth estimation [107]. Another line of work [19, 55] adapt pre-trained diffusion models for zero-shot image recognition via analysis-by-synthesis. These approaches may need to be adapted when the architectures of the generative models change or a new family of generative model emerge. Our approach treats images as universal interfaces with the hope of better generality.

Learning from synthetic data from generative models. Synthetic data has been explored to train machine learning models in various domains [31, 53, 62, 63, 74, 75, 83, 87, 102]. In computer vision, the utilization of synthetic data for training models is common, ranging from optical flow [61] and autonomous driving [1] to semantic segmentation [15] and human pose estimation [94]. Others [48, 58] have explored synthetic data for representation learning, with the predominant approach of altering the latent variables of deep generative models. Our approach aligns with this research paradigm, but it diverges in its use of text-toimage models, which have also been investigated by other researchers [45, 78, 111]. But they use synthetic data for supervised learning [30, 78]. The closet work is StableRep [91], which also conducts representation learning but still needs a real text dataset.

#### 3. Approach

In this paper, we study the problem of learning a visual encoder f in the absence of real images or textual data. Our approach hinges on the utilization of three key resources: a language generation model (g1), a text-to-image generative model (g2), and a curated list of visual concepts (C). Our exploration include three steps: (1) we employ g1 to synthesize a comprehensive set of image descriptions T, which encompass the range of visual concepts in C; (2) for each caption in T, we generate multiple images using g2, culminating in an extensive synthetic image dataset X; (3) we train on X to obtain a visual representation encoder f.

We use Llama-2 7B [93] and Stable Diffusion 1.5 [73] as g1 and g2, respectively, because of their fast inference speed. We anticipate that better g1 and g2 in the future will further enhance the effectiveness of this approach.

##### 3.1. Synthesizing captions

To harness the capability of powerful text-to-image models for generating a substantial dataset of training images, we initially require a collection of captions that not only precisely depict an image but also exhibit diversity to encompass a broad spectrum of visual concepts.

We have developed a scalable approach to create such a large collection of captions, leveraging the in-context learning capability of LLMs [9]. Our method involves crafting specific prompt engineering templates that guide the LLM to produce the required captions. We start by gathering the concept list C from some existing datasets, such as ImageNet21k [24] and Places-365 [108]. For each concept c ∈ C, we consider three straightforward templates to generate captions effectively.

- • c –> caption. As the most direct and simple approach, we have the Llama-2 model sample a sentence for the concept c.
- • c, bg –> caption. We combine the visual concept c with a background or setting bg. A naïve approach would randomly select both c and bg, where bg may correspond to a class name from a places dataset like [108]. However, this method often leads to unlikely combinations in the real world, such as a blue whale in a football field. Our ablation experiments demonstrate that this strategy results in suboptimal performance, likely because the generated cap-

tions fall far outside the training distribution of g2. Instead, we employ GPT-4 [67] to generate a list of suitable backgrounds for the chosen concepts. This approach increases the likelihood of generating more plausible combinations, such as a tiger in a forest or a cat in a kitchen, enhancing the overall quality of the results.

- • c, rel –> caption. Given a visual concept c, we consider pairing it with a positional relationship word, rel. Take for instance, if c signifies cat and rel translates to in front

Templates In context examples c –> caption revolver –> Multiple antique revolvers lie on a wooden table, gleaming under soft, ambient light.

closet –> The compact closet, brimming with clothes and shoes, exudes a feeling of organization. zebra –> A zebra is gallantly trotting across the vast, sunlit plains of the African savannah, creating a captivating black and white spectacle. bus station –> The bustling bus station thrums with restless energy, as travelers navigate through the crowded space, awaiting their journeys amid the echoes of departing buses.

c,bg –> caption tiger, forest –> Two tigers are running together in the forest. lighter, motorhome –> In the cozy, cluttered environment of a well-traveled motorhome, a sleek silver lighter holds dominion on the rustic wooden table. sunset, lake –> Golden sunset hues reflect on a calm lake, silhouetting a lone canoeist against a backdrop of fiery clouds.

c,rel –> caption kit fox, in front of –> A group of small, fluffy, golden kit foxes is playfully gathered in front of a lush, green, towering forest backdrop. cabbage, besides –> A vibrant image portrays a lush, green cabbage, glistening with dewdrops, nestled besides a rustic, wooden crate full of freshly harvested vegetables.

- Table 1. We show examples for the three synthesis templates. Such examples are used as demonstrations for Llama-2 to perform the in-context learning task. We have 176 such examples in total. Most of them are generated by prompting GPT-4 [67], while a handful of others are human generated (in a 10M scale pilot study of synthetic captions, we do not notice significant differences between including or excluding human generated examples.)

[Figure 19]

Figure 3. In-context caption generation using Llama-2 [93]. We randomly sample three in-context examples for each inference run.

of, our objective is to prompt the LLM to create captions such as a cute yellow cat is enjoying the fish in front of the sofa. To add variety, we have a selection of 10 different positional relationship words that we randomly choose from.

For each of the three templates, we have prepared multiple demonstration examples that serve as instructions for the LLM to complete the caption synthesis task. Table 1 shows a couple of examples for each template. In total, we have 106 examples for c–>prompt, 50 examples for c,bg–>prompt, and 20 examples for c,rel–>prompt. Such examples are mostly collected by prompting GPT-4, with a handful from human. In a pilot study, we do not observe difference between including or excluding human generated examples.

In the stage of generating captions in-context, we select a concept and one of the three templates. Next, we randomly pick three examples from the chosen template and frame the caption generation as a text completion task. This process is illustrated in Figure 3.

##### 3.2. Synthesizing Images

For each text caption, we generate a variety of images by initiating the reverse diffusion process with different random noise. The Classifier-Free Guidance (CFG) scale is a crucial factor in this process. A higher CFG scale enhances the quality of the samples and the alignment between text and image, whereas a lower scale results in more diverse samples and better adherence to the original conditional distribution of images based on the given text. Following the approach used in StableRep [91], we opt for a lower CFG scale, specifically

- 2.5, and produce 4 images for each caption. Examples of these images can be seen in Figure 4.
- 3.3. Representation Learning

Our representation learning method is built upon StableRep [91]. The key component of our approach is the multi-positive contrastive learning loss [50] which works by aligning (in the embedding space) images generated from the same caption. We additionally combine multiple techniques from other self-supervised learning methods, including a patch-level masked image modeling objective. We briefly review StableRep and elaborate on the added modules.

StableRep [91] minimizes the cross-entropy loss between a ground-truth assignment distribution and a contrastive assignment distribution. Consider an encoded anchor sample a and a set of encoded candidates {b1,b2,...,bK}. The contrastive assignment distribution q describes how likely the model predicts a and each b to be generated from the same caption, and the ground-truth distribution is the actual match

[Figure 20]

[Figure 21]

A plate of paella, a mixed rice dish with chicken, beans, and seafood A vintage electric locomotive rolls along a railway line through a quaint paddy field in a tranquil rural landscape.

[Figure 22]

[Figure 23]

An industrial power plant with its smokestacks belching black smoke. On a desk, a glass water bed is surrounded by a chaotic, messy workspace.

[Figure 24]

[Figure 25]

A fluffy, black and white junco bird perches on a snow-covered fence, overlooking a dark forest.

A combine harvester pulling a trailer full of hay, driving along a narrow road with a lake in the distance.

- Figure 4. Random examples of synthetic captions and images generated in our SynCLR pipeline. Each caption comes with 4 images.

between a and b (a is allowed to match multiple b):

exp(a · bi/τ) K j=1 exp(a · bj/τ)

qi =

pi = K match(a,bi)

j=1 match(a,bj)

(1)

(2)

where τ ∈ R+ is the scalar temperature, a and all b have been ℓ2 normalized, and the indicator function match(·,·) indicates whether two samples are from the same caption. The contrastive loss for a is given as

K

pi log qi (3)

L(a) = H(p,q) = −

i=1

iBOT [110] is a masked image modeling objective, wherein a localized patch is masked, and the model is tasked with predicting the tokenized representation of said masked patch. It adapts the DINO [11] objective from the image level into the patch level. We follow [76] to replace the softmaxcentering method with the iterative Sinkhorn-Knopp (SK) algorithm [22]. We run SK for 3 iterations to build the prediction target.

Exponential Moving Average (EMA) is firstly introduced into self-supervised learning by MoCo [43]. We use EMA to encode crops as b and to produce the targets for iBOT loss. We update the EMA model as θema ← λθema + (1 − λ)θ, following a cosine schedule for λ from 0.994 to 1 during training [39, 68]. We find the EMA module not only increases the final performance, but also improves the training stability for long training schedules.

Multi-crop strategy is introduced by [10] as a smart way to improve computation efficiency, and is adopted in this paper.

For these local crops, we only employ the contrastive loss, omitting the iBOT loss. Local crops are encoded only by the student network, and matched to global crops from the same caption encoded by the EMA model. Such reuse of global crops saves computation. For each image x, where we generate a single global crop xg alongside n local crops xl, the final loss can be expressed as follows:

1 n

L(xg) +

n

L(xli) + LiBOT(xg) (4)

i=1

##### 3.4. Implementation

Concept list. We concatenate class names from various datasets, including IN-1k [24], IN-21k (we keep the most frequent 13k classes), Aircraft [60], Cars [51], DTD [18], Flowers [64], Pets [69], Sun397 [98], Caltech-101 [34], Food101 [7], and Places-365 [108]. If the concept is a place (i.e. SUN397 and Places) or a texture (i.e. DTD), we only apply the c –> caption template. For fine-grained classes such as pets or flowers, we employ GPT-4 to generate a consolidated list of probable backgrounds, rather than producing distinct lists for each specific class. We favor more frequent sampling from IN-1k, Food101, Cars, Aircraft, and Flowers. Batches. For each training batch, we sample 2048 captions (except when noted), and use all of the 4 images generated by each caption. We generate 1 global and 4 local crops for each image. As a result, each batch contains 8192 global crops, which is similar with prior work [13, 14, 39, 91].

Masking. For the iBOT loss, we randomly choose 50% images inside a batch to mask, and randomly mask 50% of the tokens in each chosen image. We use 65536 prototypes. While the target from the EMA model is ascertained using

StableRep SynCLR IN avg. IN avg.

captions

cc12m 73.0 81.6 77.1 85.3 IN+h+Places 75.4 80.0 78.7 83.0 IN+Places+LLM 73.7 76.9 77.6 81.8 IN+OurBG+LLM 75.3 78.5 78.2 81.9

###### our final config. 75.8 85.7 78.8 88.1

- Table 2. Comparison of different caption synthesis strategies. We report top-1 ImageNet linear evaluation accuracy and the average accuracy over 9 fine-grained datasets. Every item here includes 10M captions and 4 images per caption.

CFG 2 3 4 IN top-1 72.8 72.6 72.6

- Table 3. Classifier-free guidance scale (CFG). Contrastive loss prefers small CFG scale but is not very sensitive to it.

the SK algorithm, we apply softmax normalization to the output of the student model.

Projection heads. We follow the design in MoCo v3 [14] and DINO [11] for the contrastive and iBOT loss heads, respectively, ensuring consistency with established methods. Other hyper-parameters. We set the temperature in the contrastive loss to 0.08. For the temperature used in the iBOT loss, we linearly increase it from 0.04 to 0.07 over 4000 iterations, and keep it as 0.07 afterwards, as in DINO [11]. Additionally, the weight decay parameter is incrementally adjusted from 0.04 to 0.2, adhering to a cosine schedule.

- 4. Experiment

We first perform an ablation study to evaluate the efficacy of various designs and modules within our pipeline. Then we proceed to scale up the volume of synthetic data.

##### 4.1. Study different components

We analyze each component of SynCLR, and ablate their effectiveness in two measurements: (1) linear probing performance on IN-1k; (2) average accuracy of linear transfer on fine-grained datasets Aircraft [60], Cars [51], DTD [18], Flowers [64], Pets [69], Sun397 [98], Caltech-101 [34], Food-101 [7], and Pascal VOC [29]. For analysis conducted in this subsection, we train ViT-B/16 [28] models for 85000 iterations, and use the cls token as image representation.

Synthesize captions. Following [91], we use cc12m [12] real captions as our baseline, which has 10M sentences. To synthesize captions, we design the following variants: (a) IN+h+Places randomly combines one IN class plus its hypernyms in WordNet graph, with one place class; (b) IN+Places+LLM uses the c,bg –> caption in-context synthesis template with c from IN and bg from places; (c) IN+ourBG+LLM uses the background classes output by GPT-4, instead of Places; (d) ours means our full configu-

method EMA iBOT MC IN avg. ADE20k StableRep 75.8 85.7 -

✓ 76.7 86.7 48.0

- ✓ ✓ 77.6 87.1 50.5
- ✓ ✓ 78.6 87.8 49.5

SynCLR ✓ ✓ ✓ 78.8 88.1 50.8

- Table 4. Important components for our model. ViT-B/16 models are trained for 85000 iterations. We study the modules that affect the ImageNet linear evaluation, the fine-grained classification (avg.), and ADE20k segmentation.

method IN avg. Supervised CE 71.9 75.0 SimCLR 63.6 67.9 SynCLR 75.3 78.5

- Table 5. Comparison of different learning objectives. These objectives assume different level of classification granularity, as shown in Figure 2. Our modeling, i.e., defining classes as captions, outperforms the other two. To accomondate Supervised CE training, all items here used IN+OurBG+LLM entry in Table 2.

ration specified in Section 3.1. For each of the config, we generate 10M captions. If not enough, we do duplication.

Results are summarized in Table 2, where we train both StableRep and SynCLR to avoid biases favored by a single method. Compared to a real caption dataset cc12m, simply concatenating IN and Places class names improves the ImageNet linear accuracy but reduces the fine-grained classification performance. Interestingly, naively asking Llama to combine IN and Places classes into captions yields the worst performance. Replacing random background from places with GPT generated background improves the accuracy. This shows the importance of synthesizing captions that follow the distribution of real captions, which were used to train the text-to-image model. Finally, our full configuration achieves the best accuracy on both ImageNet and fine-grained classification. Another advantage of our synthesis method is its scalability – scale up to hundreds of millions of captions with little duplication. In contrast, if we concatenate IN classes with Places classes, there are at most 365k unique captions. Synthesize images. There are two major parameters in this process: number of images per caption and classifier free guidance scale. For the former, we find generating 4 images is almost able to reproduce StableRep [91]’s performance (10 images) when using cc12m captions (ours 73.0% v.s. StableRep 73.5% on ImageNet). Thus we stick to 4. For guidance scale, we briefly find the contrastive loss is not very sensitive to CFG in a pilot study, as shown in Table 3. Thus we stick to 2.5, similar as StableRep [91].

Model components. We present the improvement of accuracy brought by different modules in Table 4. Compared

|text img # imgs|ImageNet<br><br>|Aircraft<br><br>Cars<br><br>DTD<br><br>Flowers<br><br>Pets<br><br>SUN397<br><br>Caltech-101<br><br>Food-101<br><br>VOC2007<br><br>Average|
|---|---|---|
|StableRep real syn 100M ViT-B/16 CLIP real real 400M<br><br>ViT-B/16 ViT-L/14<br><br>OpenCLIP real real<br><br>400M ViT-B/16 400M ViT-L/14<br><br>2B ViT-L/14 DINO v2* - real 142M<br><br>ViT-B/14 ViT-L/14<br><br>|75.7 80.2 83.9 78.9 82.3 83.4<br><br>83.9† 85.7†|59.2 83.5 80.1 97.3 88.3 74.3 94.7 85.1 87.9 83.4 59.5 86.7 79.2 98.1 93.1 78.4 94.7 92.8 89.2 85.7 69.4 90.9 82.1 99.2 95.1 81.8 96.5 95.2 89.6 88.9 61.1 92.3 81.9 98.2 91.5 77.9 95.2 90.9 88.0 86.3 67.1 94.0 83.6 98.8 92.5 81.0 96.4 93.4 88.8 88.4 71.7 95.3 85.3 99.0 94.2 82.2 97.5 94.1 88.9 89.8 79.4 88.2 83.3 99.6 96.2 77.3 96.1 92.8 88.2 89.0 81.5 90.1 84.0 99.7 96.6 78.7 97.5 94.3 88.3 90.1<br><br>|
|SynCLR syn syn 600M<br><br>ViT-B/16 ViT-L/14<br><br>|80.7 83.0|81.7 93.8 79.9 99.1 93.6 76.2 95.3 91.6 89.4 89.0 85.6 94.2 82.1 99.2 94.1 78.4 96.1 93.4 90.3 90.4<br><br>|

Caltech-101

###### ImageNet

VOC2007

Food-101

SUN397

Average

Flowers

Aircraft

DTD

Cars

Pets

- Table 6. Comparison on ImageNet linear evaluation and fine-grained classificaton. SynCLR achieves comparable results with OpenAI’s CLIP and DINO v2 models, despite only using synthetic data. *DINO v2 modes are distilled from a ViT-g model, thus advantageous in this comparison. † we rerun only using cls token instead of concatenating multiple layers presented in the original DINO v2 paper [68].

to the baseline StableRep, adding a teacher EMA model improves the IN linear accuracy by 0.9%. Further adding iBOT local objective or the multi-crop strategy increases the accuracy by 0.9% and 1.9%, respectively. Combining all of them results in our full SynCLR model, which achieves 78.8% top-1 IN linear accuracy. The fine-grained classification performance follows a similar trend, and reaches 88.1%. Besides, we test the transfer ability to semantic segmentation on ADE20k. The iBOT objective brings 1.0 more mIoU than multi-crop strategy, demonstrating the effectiveness of masked image modeling for dense prediction tasks.

Compare to SimCLR and supervised training. We compare the three different representation learning objectives shown in Figure 2, which classify images at different levels of granularity. Since supervised cross-entropy training requires a fixed set of balanced classes (indeed both fixed set and balance are limitations of such method), we use the IN+ourBG+LLM configuration where we have 1000 balanced classes (i.e., each class has 40k images). The supervised training recipe follows [86]. For a fair comparison with SimCLR, we remove all unmatched modules (i.e., EMA, iBOT, and MC) to make sure that the only difference between SimCLR and our SynCLR is the classification granularity defined by the contrastive loss. For all of them, we do pre-training and then linear probing on the target dataset.

Table 5 presents the comparison. Our multi-positive objective, which defines images as the same class if they are generated by the same caption, achieves the best performance. It outperforms supervised cross-entropy training and SimCLR by 3.4% and 11.7% for top-1 accuracy on ImageNet linear evaluation, and by 3.5% and 10.6% on finegrained classification tasks. Besides, our objective does not require balance between samples from a fixed set of classes, making it easier to scale up.

##### 4.2. Scaling up

After we have ablated different components, we scale up our experiments. Specifically, we synthesize a dataset of 150M captions, called SynCaps-150M, from which we generate 600M images. We train both ViT-B/16 and ViT-L/14 (no SwiGLU [82] or LayerScale [92]), and extend the training schedules to 500k steps with a batch size of 8192 captions. We use 224x224 resolution for all pre-training tasks.

We compare SynCLR with OpenAI’s CLIP [71], OpenCLIP [17], and DINO v2 [68], which represent learning from data. We note that ViT-B/14 and ViT-L/14 from DINO v2 are distilled from a ViT-g [104] model, which makes DINO v2 advantageous in our comparison. We also includes StableRep [91], which uses the hybrid paradigm.

ImageNet linear evaluation. For fair comparison, cls token from the last block is used as representation across all models (whereas in DINO v2, results are from concatenating multiple layers). As shown in Table 6, SynCLR achieves 80.7% with ViT-B and 83.0% with ViT-L. This is similar as CLIP, but still lags behind DINO v2 by 3.2% and 2.7%, respectively, partially because of the extra distillation in DINO v2. We note SynCLR has already outperformed other self-supervised methods pre-trained directly on ImageNet1k (e.g., DINO achieves 78.2% with ViT-B/16 and iBOT reaches 81.0% with ViT-L/16).

Fine-grained classification. On the nine fine-grained datasets we have evaluated in Table 6, SynCLR achieves very similar average accuracy as DINO v2, e.g., 89.0% v.s. 89.0% for ViT-B, and 90.1% vs 90.4% for ViT-L. Both SynCLR and DINO v2 have curated the pre-training data to include the distribution for these datasets (but in different ways and portions), and end up with similar performance. Interestingly, SynCLR outperforms others on Aircraft and Cars, possibly because we favor more frequent sampling

method pre-train data distill ViT-B ViT-L StableRep hybrid, 100M 49.4 MoCo v3 real, IN1K-1M 47.3 49.1 BEiT real, IN1K-1M+DALLE 47.1 53.3 MAE real, IN1K-1M 48.1 53.6 iBOT real, IN1K-1M 50.0 CLIP real, WIT-400M 52.6 BEiT v2 real, WIT-400M, IN1K ✓ 53.1 56.7 DINO v2 real, LVD-142M ✓ 54.4 † 57.5† SynCLR synthetic, 600M 54.3 57.7 †

- Table 7. ADE20K semantic segmentation (mIoU) using UperNet, with single scale at 512x512 resolution. † use patch size of 14x14, thus adapt to 518x518 resolution.

method pre-train data ViT-B ViT-L MoCo v3 real, IN1K-1M 83.2 84.1 SimMIM real, IN1K-1M 83.8 MAE real, IN1K-1M 83.6 85.9 PeCo real, IN1K-1M 83.6 85.9 data2vec real, IN1K-1M 84.2 86.6 iBOT real, IN21K-14M 84.4 86.6 BEiT v2 real, WIT-400M+IN1k-1M 85.5 87.3 CLIP real, WIT-400M 85.2 87.5† OpenCLIP

real, LAION-400M 85.0 86.6† real, LAION-2B - 87.1†

SynCLR synthetic, 600M 85.8 87.9†

- Table 8. Top-1 accuracy on ImageNet with fine-tuning evaluation.. Models are fine-tuned at 224x224 resolution. † use patch size of 14x14.

towards them. This can be an advantage for synthetic data when we know what downstream tasks to solve. Besides, SynCLR outperforms CLIP and StableRep by 3.3% and by

- 5.6% for ViT-B, respectively. Semantic segmentation. To evaluate the pixel-level understanding ability of SynCLR, we fine-tune the pre-trained models on ADE20k [109], following the setup in [5, 44]. UperNet [99] is used as the task layer, and we evaluate with a single-scale, i.e. 512x512. Besides CLIP and DINO v2, we also compare to self-supervised methods pre-trained on ImageNet, as well as BEiT v2 [70], which distills from CLIP. Table 7 shows that our SynCLR outperforms self-supervised methods trained on IN-1k by a clear marge, e.g., 4.3 higher mIoU than iBOT. Despite not involving a high resolution pretraining period like DINO v2 (e.g., 518x518), SynCLR performs similarly with DINO v2 (0.1 lower for ViT-B possibly because DINO v2 uses a smaller patch size of 14x14, but 0.2 higher for ViT-L). This suggests SynCLR pre-training is suitable for dense prediction tasks. ImageNet fine-tuning. We evaluate the fine-tuning transfer ability of SynCLR on ImageNet. We compare with other

| |EuroSAT<br><br>GTSRB<br><br>Country211<br><br>MNIST<br><br>RESISC45<br><br>KITTI|Average<br><br>|
|---|---|---|
|CLIP ViT-B/16 ViT-L/14<br><br>DINO v2 ViT-B/14 ViT-L/14<br><br>SynCLR ViT-B/16 ViT-L/14|97.1 86.6 33.3 99.0 92.7 64.7<br><br>98.2 92.5 42.9 99.2 94.1 69.2<br><br><br>96.0 72.8 21.6 98.6 92.5 75.3 96.7 74.1 24.1 98.2 93.8 76.9<br><br>96.6 78.6 21.0 98.4 93.7 77.3<br><br>96.7 79.2 24.3 98.5 93.8 78.0<br><br><br>|78.9 82.7 76.1 77.3 77.6 78.4|

Country211

RESISC45

EuroSAT

Average

GTSRB

MNIST

KITTI

- Table 9. Generalization to concepts not seen by DINO v2 and SynCLR. SynCLR outperforms DINO v2. CLIP achieves the best accuracy, possibly because its training data includes similar concepts as these datasets.

SynCLR CLIP IN avg. IN avg.

SynCaps-150M 80.7 89.0 78.3 87.7 Laion-400M 78.9 86.5 76.6 84.9

- Table 10. Compare SynCLR with CLIP on the same synthetic data. We observe that: (1) SynCLR outperforms CLIP; (2) in our setup, i.e., generating 4 images per caption, SynCaps-150M yields better representations for both SynCLR and CLIP.

state of the art self-supervised methods [4, 5, 14, 27, 44, 100, 110] in Table 8. Our SynCLR outperforms models trained on ImageNet images or large scale image datasets. Specifically, SynCLR outperforms OpenCLIP ViT-L trained on Laion-2B, which is the dataset Stable Diffusion (the text2image model we used) is trained on. This contrasts with [30, 78], which shows that directly training a classifier on synthetic images yields bad classification accuracy. Our finding suggests synthetic images are good for training representations, which later can be easily adapted to a downstream task with limited amount of real data.

##### 4.3. Further analysis

SynCLR requires a list of concepts C to start off. But how will SynCLR transfer to concepts outside our list?

Generalize to unseen concepts. We consider additional datasets whose classes are outside the synthesis list, including EuroSAT [46], GTSRB [85], Country211 [71], MNIST [54], RESISC45 [16], and KITTI distances [35]. These datasets, except for KITTI, are also outside the curation list of DINO v2. Therefore, it is also a generalization test for DINO v2. Table 9 shows the linear probing results. SynCLR outperforms DINO v2 by 1.5% for ViT-B and 1.1% for ViT-L, respectively. This suggests the representations of SynCLR generalize. CLIP outperforms SynCLR and DINO v2, with most gains coming from Country211. An explanation is CLIP’s training data contains similar country flags which are not in the training sets of SynCLR and DINO v2.

Dino v2 SynCLR (ours) Dino v2 SynCLR (ours) Dino v2 SynCLR (ours)

[Figure 26]

[Figure 27]

[Figure 28]

(a)

(b) (c)

- Figure 5. PCA visualization. Follow DINO v2 [68], we compute a PCA between the patches of the images from the same set and colorize by their first 3 components. Compared to DINO v2, SynCLR produces more accurate maps for cars (e.g., zoom-in to see the two bars on the roof of the first car, and the three side windows of the third car) and airplanes (e.g., the boundaries), while being slightly worse for dogs (e.g., heads). We use ViT-L/14 for both methods. Images are resized to 336x448 resolution before being fed into the networks, yielding 24x32 visualization grids.

ViT-B/16 ViT-L/14

82

ImageNetLinearAcc.(%)

80

78

76

74

1 3 10 40 150 Number of Synthetic Captions (M)

- Figure 6. ImageNet linear accuracy w/ different training scales.

| |ViT-B/16 ViT-L/14<br><br>|
|---|---|
| | |
| | |
| | |
| | |
| | |

1 3 10 40 150 Number of Synthetic Captions (M)

- 85
- 86
- 87
- 88
- 89
- 90

Fine-grainedCls.Acc.(%)

- Figure 7. Fine-grained classification w/ different training scales.

Given that both captions and images are synthesized, a natural question arises: how would CLIP training perform on such data?

Compare to CLIP training. We use the same data to train a ViT-B CLIP model. For each caption, we randomly choose 1

out of the 4 synthesized images in each iteration. Following common practice [71], we train for 32 epochs with a batch size of 32768. This model achieves 44.4% zero-shot accuracy on IN-1k. The SynCaps-150M row in Table 10 presents the linear probing results. Synthetic CLIP learns reasonably good features, reaching 78.3% on IN-1k and 87.7% on fine-grained datasets. However, SynCLR is still better.

We have also repeated our experiments with Laion-400M captions, i.e., generate 4 images for each caption and train SynCLR and CLIP. The comparison between rows SynCaps150M and Laion-400M in Table 10 suggests synthetic captions are also favorable on a large scale.

PCA visualization. Following the method used in DINO v2 [68], we present visualizations derived from the Principal Component Analysis (PCA) conducted on patch features extracted using our model SynCLR. As depicted in Figure 5, a comparative analysis is conducted between SynCLR and DINO v2, both utilizing the ViT-L/14 architecture. The results demonstrate that SynCLR effectively accentuates the features of cars and planes, while efficiently minimizing background clutter.

Scaling behavior. We train ViT-BViT-L models using random subsets of varying sizes: 1M, 3M, 10M, 40M, and the comprehensive 150M (measured in the number of captions). These models are trained over a reduced schedule of 300,000 steps and utilizes a smaller batch size of 2048. The outcomes of linear probing are illustrated in Figures 6 and 7. These results indicate that the ViT-B model delivers robust performance at the 10M scale, with diminishing returns observed beyond this point. In contrast, the ViT-L model exhibits a greater demand for data (i.e., it underperforms ViT-B at the 3M scale) and scales better with data.

#### 5. Discussions and Conclusion

Why learn from generative models? One compelling reason is that a generative model can act like hundreds of datasets simultaneously. Traditionally, researchers have to spend separate effort collecting datasets for different image categories, e.g., cars, flowers, cats, dogs, and so on. DINO v2 [68] achieves robust representations by curating and amalgamating numerous such datasets. Such a process introduces complexities such as clustering and search challenges. In contrast, advanced text-to-image generative models like Stable Diffusion [72] or Imagen [77] have the capability to generate many diverse datasets. These models provide the flexibility to produce an infinite number of samples (albeit finite diversity) and control the generation process through textual input. Thus, generative models offer a convenient and effective method for curating training data. In our study, we harness this advantage to synthesize images encompassing a broad spectrum of visual concepts.

What can be further improved? Enhanced caption sets can be achieved through various methods, such as enriching the set of in-context examples, optimizing the sampling ratios among different concepts, and utilizing more advanced LLMs. In terms of the learning process, one approach is to distill knowledge from a larger model, and incorporate an additional high-resolution training phase (as discussed in [68]) or an intermediate IN-21k fine-tuning stage (as per [5, 70]). Regarding architectural improvements, the integration of SwiGLU and LayerScale, coupled with superior model initialization strategies (referenced in [32]), can be beneficial. However, due to limited resources and the scope of this paper not being focused on achieving the highest possible metrics, we propose these areas for further exploration in future research endeavors.

In summary, this paper studies a new paradigm for visual representation learning – learning from generative models. Without using any real data, SynCLR learns visual representations that are comparable with those achieved by state of the art general-purpose visual representation learners.

#### References

- [1] Hassan Abu Alhaija, Siva Karthik Mustikovela, Lars Mescheder, Andreas Geiger, and Carsten Rother. Augmented reality meets computer vision: Efficient data generation for urban driving scenes. IJCV, 2018. 3
- [2] Mahmoud Assran, Mathilde Caron, Ishan Misra, Piotr Bojanowski, Florian Bordes, Pascal Vincent, Armand Joulin, Mike Rabbat, and Nicolas Ballas. Masked siamese networks for label-efficient learning. In ECCV, 2022. 3
- [3] Shekoofeh Azizi, Simon Kornblith, Chitwan Saharia, Mohammad Norouzi, and David J Fleet. Synthetic data from diffusion models improves imagenet classification. arXiv preprint arXiv:2304.08466, 2023. 2
- [4] Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Ji-

- atao Gu, and Michael Auli. Data2vec: A general framework for self-supervised learning in speech, vision and language. In ICML, 2022. 3, 8
- [5] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. arXiv preprint arXiv:2106.08254, 2021. 3, 8, 10, 14
- [6] Suzanna Becker and Geoffrey E Hinton. Self-organizing neural network that discovers surfaces in random-dot stereograms. Nature, 1992. 3
- [7] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101–mining discriminative components with random forests. In ECCV, 2014. 5, 6
- [8] Emmanuel Asiedu Brempong, Simon Kornblith, Ting Chen, Niki Parmar, Matthias Minderer, and Mohammad Norouzi. Denoising pretraining for semantic segmentation. In CVPR,

2022. 3

- [9] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NeurIPS, 2020. 3
- [10] Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In NeurIPS, 2020. 5
- [11] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021. 3, 5, 6, 14
- [12] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In CVPR,

2021. 6

- [13] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020. 2, 3, 5, 15
- [14] Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In ICCV, 2021. 5, 6, 8, 14
- [15] Yuhua Chen, Wen Li, Xiaoran Chen, and Luc Van Gool. Learning semantic segmentation from synthetic data: A geometrically guided input-output adaptation approach. In CVPR, 2019. 3
- [16] Gong Cheng, Junwei Han, and Xiaoqiang Lu. Remote sensing image scene classification: Benchmark and state of the art. Proceedings of the IEEE, 2017. 8
- [17] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In CVPR, 2023. 7
- [18] Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In CVPR, 2014. 5, 6
- [19] Kevin Clark and Priyank Jaini. Text-to-image diffusion models are zero-shot classifiers. arXiv preprint arXiv:2303.15233, 2023. 3

- [20] Kevin Clark, Minh-Thang Luong, Quoc V Le, and Christopher D Manning. Electra: Pre-training text encoders as discriminators rather than generators. arXiv preprint arXiv:2003.10555, 2020. 14
- [21] Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le. Randaugment: Practical automated data augmentation with a reduced search space. In CVPR workshops, 2020. 14
- [22] Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. In NeurIPS, 2013. 5
- [23] Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al. Scaling vision transformers to 22 billion parameters. In ICML, 2023. 3
- [24] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 1, 3, 5
- [25] Jeff Donahue and Karen Simonyan. Large scale adversarial representation learning. NeurIPS, 2019. 3
- [26] Jeff Donahue, Yangqing Jia, Oriol Vinyals, Judy Hoffman, Ning Zhang, Eric Tzeng, and Trevor Darrell. Decaf: A deep convolutional activation feature for generic visual recognition. In ICML, 2014. 3
- [27] Xiaoyi Dong, Jianmin Bao, Ting Zhang, Dongdong Chen, Weiming Zhang, Lu Yuan, Dong Chen, Fang Wen, Nenghai Yu, and Baining Guo. Peco: Perceptual codebook for bert pre-training of vision transformers. In AAAI, 2023. 8
- [28] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 2, 6
- [29] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. IJCV, 2010. 6
- [30] Lijie Fan, Kaifeng Chen, Dilip Krishnan, Dina Katabi, Phillip Isola, and Yonglong Tian. Scaling laws of synthetic images for model training ... for now. arXiv:2312.04567,

2023. 2, 3, 8

- [31] Lijie Fan, Dilip Krishnan, Phillip Isola, Dina Katabi, and Yonglong Tian. Improving clip training with language rewrites. In NeurIPS, 2023. 3
- [32] Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva-02: A visual representation for neon genesis. arXiv preprint arXiv:2303.11331, 2023. 10
- [33] Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva: Exploring the limits of masked visual representation learning at scale. In CVPR, 2023. 3
- [34] Li Fei-Fei, Rob Fergus, and Pietro Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In CVPR, 2004. 5, 6
- [35] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In CVPR, 2012. 8

- [36] Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rotations. In ICLR, 2018. 2
- [37] Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In CVPR, 2014. 3
- [38] Priya Goyal, Dhruv Mahajan, Abhinav Gupta, and Ishan Misra. Scaling and benchmarking self-supervised visual representation learning. In ICCV, 2019. 1
- [39] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. In NeurIPS, 2020. 3, 5, 14, 15
- [40] Raia Hadsell, Sumit Chopra, and Yann LeCun. Dimensionality reduction by learning an invariant mapping. In CVPR,

2006. 3

- [41] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR,

2016. 3

- [42] Kaiming He, Ross Girshick, and Piotr Dollár. Rethinking imagenet pre-training. In ICCV, 2019. 3
- [43] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020. 3, 5, 14
- [44] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022. 3, 8, 14
- [45] Ruifei He, Shuyang Sun, Xin Yu, Chuhui Xue, Wenqing Zhang, Philip Torr, Song Bai, and Xiaojuan Qi. Is synthetic data from generative models ready for image recognition? arXiv preprint arXiv:2210.07574, 2022. 2, 3
- [46] Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019. 8
- [47] Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger. Deep networks with stochastic depth. In ECCV,

2016. 14

- [48] Ali Jahanian, Xavier Puig, Yonglong Tian, and Phillip Isola. Generative models as a data source for multiview representation learning. arXiv preprint arXiv:2106.05258, 2021. 2, 3
- [49] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In ICML, 2021. 3
- [50] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning. In NeurIPS,

2020. 2, 4

- [51] Jonathan Krause, Jia Deng, Michael Stark, and Li Fei-Fei. Collecting a large-scale dataset of fine-grained cars. tech report, 2013. 5, 6

- [52] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In NeurIPS, 2012. 3
- [53] Varun Kumar, Ashutosh Choudhary, and Eunah Cho. Data augmentation using pre-trained transformer models. arXiv preprint arXiv:2003.02245, 2020. 3
- [54] Yann LeCun. The mnist database of handwritten digits. http://yann. lecun. com/exdb/mnist/, 1998. 8
- [55] Alexander C Li, Mihir Prabhudesai, Shivam Duggal, Ellis Brown, and Deepak Pathak. Your diffusion model is secretly a zero-shot classifier. arXiv preprint arXiv:2303.16203,

2023. 3

- [56] Tianhong Li, Huiwen Chang, Shlok Mishra, Han Zhang, Dina Katabi, and Dilip Krishnan. Mage: Masked generative encoder to unify representation learning and image synthesis. In CVPR, 2023. 3
- [57] Yanghao Li, Saining Xie, Xinlei Chen, Piotr Dollar, Kaiming He, and Ross Girshick. Benchmarking detection transfer learning with vision transformers. arXiv preprint arXiv:2111.11429, 2021. 3
- [58] Hao Liu, Tom Zahavy, Volodymyr Mnih, and Satinder Singh. Palm up: Playing in the latent manifold for unsupervised pretraining. arXiv preprint arXiv:2210.10913, 2022. 3
- [59] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 14, 15
- [60] Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi. Fine-grained visual classification of aircraft. arXiv:1306.5151, 2013. 5, 6
- [61] Nikolaus Mayer, Eddy Ilg, Philip Hausser, Philipp Fischer, Daniel Cremers, Alexey Dosovitskiy, and Thomas Brox. A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation. In CVPR, 2016. 3
- [62] Yu Meng, Jiaxin Huang, Yu Zhang, and Jiawei Han. Generating training data with language models: Towards zero-shot language understanding. arXiv preprint arXiv:2202.04538,

2022. 3

- [63] Masato Mimura, Sei Ueno, Hirofumi Inaguma, Shinsuke Sakai, and Tatsuya Kawahara. Leveraging sequence-tosequence speech synthesis for enhancing acoustic-to-word speech recognition. In SLT, 2018. 3
- [64] Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In Indian Conference on Computer Vision, Graphics & Image Processing, 2008. 5, 6
- [65] Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In ECCV,

2016. 3

- [66] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. 3
- [67] OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 3, 4
- [68] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision.

- arXiv preprint arXiv:2304.07193, 2023. 1, 2, 3, 5, 7, 9, 10, 15
- [69] Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. Cats and dogs. In CVPR, 2012. 5, 6
- [70] Zhiliang Peng, Li Dong, Hangbo Bao, Qixiang Ye, and Furu Wei. Beit v2: Masked image modeling with vector-quantized visual tokenizers. arXiv preprint arXiv:2208.06366, 2022. 8, 10
- [71] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 1, 2, 3, 7, 8, 9
- [72] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 10
- [73] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 3
- [74] Andrew Rosenberg, Yu Zhang, Bhuvana Ramabhadran, Ye Jia, Pedro Moreno, Yonghui Wu, and Zelin Wu. Speech recognition with augmented synthesized speech. In ASRU,

2019. 3

- [75] Nick Rossenbach, Albert Zeyer, Ralf Schlüter, and Hermann Ney. Generating synthetic audio data for attention-based speech recognition systems. In ICASSP, 2020. 3
- [76] Yangjun Ruan, Saurabh Singh, Warren Morningstar, Alexander A Alemi, Sergey Ioffe, Ian Fischer, and Joshua V Dillon. Weighted ensemble self-supervised learning. arXiv preprint arXiv:2211.09981, 2022. 5
- [77] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022. 10
- [78] Mert Bulent Sariyildiz, Karteek Alahari, Diane Larlus, and Yannis Kalantidis. Fake it till you make it: Learning transferable representations from synthetic imagenet clones. In CVPR, 2023. 2, 3, 8
- [79] Saurabh Saxena, Charles Herrmann, Junhwa Hur, Abhishek Kar, Mohammad Norouzi, Deqing Sun, and David J Fleet. The surprising effectiveness of diffusion models for optical flow and monocular depth estimation. arXiv preprint arXiv:2306.01923, 2023. 3
- [80] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. In NeurIPS, 2022. 1
- [81] Ali Sharif Razavian, Hossein Azizpour, Josephine Sullivan, and Stefan Carlsson. Cnn features off-the-shelf: an astounding baseline for recognition. In CVPR workshops, 2014. 3
- [82] Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020. 7
- [83] David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lu-

- cas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. Nature, 2017. 3
- [84] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014. 3
- [85] Johannes Stallkamp, Marc Schlipsing, Jan Salmen, and Christian Igel. The german traffic sign recognition benchmark: a multi-class classification competition. In IJCNN,

2011. 8

- [86] Andreas Steiner, Alexander Kolesnikov, Xiaohua Zhai, Ross Wightman, Jakob Uszkoreit, and Lucas Beyer. How to train your vit? data, augmentation, and regularization in vision transformers. arXiv preprint arXiv:2106.10270, 2021. 7
- [87] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Alpaca: A strong, replicable instructionfollowing model. Stanford Center for Research on Foundation Models., 2023. 3
- [88] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. arXiv:1906.05849, 2019. 3, 14
- [89] Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, and Phillip Isola. What makes for good views for contrastive learning? In NeurIPS, 2020. 3
- [90] Yonglong Tian, Olivier J Henaff, and Aäron van den Oord. Divide and contrast: Self-supervised learning from uncurated data. In ICCV, 2021. 1
- [91] Yonglong Tian, Lijie Fan, Phillip Isola, Huiwen Chang, and Dilip Krishnan. Stablerep: Synthetic images from text-toimage models make strong visual representation learners. In NeurIPS, 2023. 1, 2, 3, 4, 5, 6, 7, 14
- [92] Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, and Hervé Jégou. Going deeper with image transformers. In ICCV, 2021. 7
- [93] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 3, 4
- [94] Gul Varol, Javier Romero, Xavier Martin, Naureen Mahmood, Michael J Black, Ivan Laptev, and Cordelia Schmid. Learning from synthetic humans. In CVPR, 2017. 3
- [95] Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In ICML, 2020. 3
- [96] Chen Wei, Haoqi Fan, Saining Xie, Chao-Yuan Wu, Alan Yuille, and Christoph Feichtenhofer. Masked feature prediction for self-supervised visual pre-training. In CVPR, 2022. 3
- [97] Zhirong Wu, Yuanjun Xiong, Stella X Yu, and Dahua Lin. Unsupervised feature learning via non-parametric instance discrimination. In CVPR, 2018. 3
- [98] Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva, and Antonio Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In CVPR, 2010. 5, 6
- [99] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding. In ECCV, 2018. 8, 14

- [100] Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. Simmim: A simple framework for masked image modeling. In CVPR, 2022. 3, 8
- [101] Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello. Open-vocabulary panoptic segmentation with text-to-image diffusion models. In CVPR, 2023. 3
- [102] Yiben Yang, Chaitanya Malaviya, Jared Fernandez, Swabha Swayamdipta, Ronan Le Bras, Ji-Ping Wang, Chandra Bhagavatula, Yejin Choi, and Doug Downey. Generative data augmentation for commonsense reasoning. arXiv preprint arXiv:2004.11546, 2020. 3
- [103] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In ICCV, 2019. 14
- [104] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers. In CVPR, 2022. 7
- [105] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412, 2017. 14
- [106] Richard Zhang, Phillip Isola, and Alexei A Efros. Colorful image colorization. In ECCV, 2016. 2
- [107] Wenliang Zhao, Yongming Rao, Zuyan Liu, Benlin Liu, Jie Zhou, and Jiwen Lu. Unleashing text-to-image diffusion models for visual perception. arXiv preprint arXiv:2303.02153, 2023. 3
- [108] Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, and Antonio Torralba. Learning deep features for discriminative localization. In CVPR, 2016. 3, 5
- [109] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset. IJCV,

2019. 8, 14

- [110] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. ibot: Image bert pre-training with online tokenizer. arXiv preprint arXiv:2111.07832,

2021. 2, 3, 5, 8

- [111] Yongchao Zhou, Hshmat Sahak, and Jimmy Ba. Training on thin air: Improve image classification with generated data. arXiv preprint arXiv:2305.15316, 2023. 3

#### A. Concept Sampling

The concepts used to synthesize captions are randomly sampled from the names of various datasets. The rough ratios are presented in Table 11. It is likely that different combinations of these ratios lead to different results, but we do not optimize over this dimension. For example, we simply concatenate IN-21k concepts with the classes of other datasets (e.g., Caltech-101, Pets), and do uniform sampling from the concatenated list. This may lead to under-sampling for other datasets, as the list is dominated by IN-21 classes.

|source<br><br>|prob.|
|---|---|
|IN-1k Aircraft Cars Food Flowers Places-365, SUN397 IN-21k and others|0.47 0.05 0.05 0.05 0.03 0.09 0.26<br><br>|

Table 11. Rough concept sampling probabilities.

#### B. Implementation Details

- B.1. Pre-training The setting for our final long schedule training in Section

- 4.2 is summarized in Table 12, where models are trained for 500k steps with a batch size of 8192 captions. For ablation study present in Section 4.1, we only train for 85k steps with a batch size of 2048 captions; for the scaling plots in Section
- 4.3, we train all models for 300k steps with a batch size of 2048.

|config<br><br>|value|
|---|---|
|batch size optimizer peak learning rate weight decay optimizer momentum learning rate schedule steps warmup steps stoch. depth [47] augmentation<br><br>|8192 AdamW [59] 2e-3 (B), 1.5e-3 (L) 0.04 –> 0.2, cosine β1, β2=0.9, 0.999 cosine decay 500k 80k 0.1 (B), 0.4 (L) Downsample [91] + BYOL Aug. [39]|

Table 12. SynCLR pre-training settings.

- B.2. ImageNet linear probing

We use the cls token from the final transformer block as the image representation. This is different from DINO v2,

which tries to concatenate cls token with average pooled patch tokens and sweep over whether to use multiple layers.

We follow prior work [11, 14] to train the linear classifier. It has been generally observed that regularization such as weight decay hurts the performance [43, 88]. Therefore, we set weight decay as 0, and we sweep the base_lr over {0.1,0.2,0.5,1,2,5,10,20,50} × 10−2.

|config|value<br><br>|
|---|---|
|batch size optimizer base learning rate peak learning rate weight decay optimizer momentum learning rate schedule epochs augmentation<br><br>|1024 SGD sweep blr × bsz/256 0 0.9 cosine decay 90 RandomResizedCrop, Flip|

Table 13. ImageNet linear probing settings.

##### B.3. End-to-End ImageNet fine-tuning

Following common practice [5, 44], we append a linear classifier on top of the CLS token of the last transformer block, and fine-tune the whole network. We use layer-wise lr decay [20]. Table 14 shows the settings.

|config<br><br>|value|
|---|---|
|optimizer base learning rate peak learning rate optimizer momentum layer-wise lr decay batch size learning rate schedule warmup epochs epochs RandAugment [21] label smoothing erasing prob. mixup [105] cutmix [103] stoch. depth [47] test crop ratio ema|AdamW [59] 5e-5 blr × bsz/256 β1, β2=0.9, 0.999 0.65 (B), 0.8 (L) 1024 cosine decay 20 (B), 5 (L) 100 (B), 50 (L) 9/0.5<br><br>0.1 (B), 0.2 (L)<br><br>0.25 0.8 1.0<br><br>0.1 (B), 0.3 (L)<br><br><br>0.95 (B), 1.0 (L) 0.9999<br><br>|

Table 14. ImageNet end-to-end fine-tuning settings.

##### B.4. Semantic segmentation on ADE20k

We conduct the experiments on ADE20k [109]. Following [5, 44], we use UperNet [99] as the task adaptation layer. We use the common single-scale [5] setup, with a resolution

of 512×512 for models with a patch size of 16×16 and a resolution of 518×518 for models with a patch size of 14×14. The hyper-parameters are summarized in Table 15.

|config<br><br>|value|
|---|---|
|batch size optimizer peak learning rate optimizer momentum weight decay layer-wise lr decay steps warmup steps stoch. depth|32 (B), 16 (L) AdamW [59] 8e-5 β1, β2=0.9, 0.999 0.05 0.6 (B), 0.8 (L) 60k (B), 160k (L) 1500 0.1 (B), 0.2 (L)<br><br>|

Table 15. ADE20k semantic segmentation settings.

##### B.5. Fine-grained linear classification

Following prior works [13, 39], we train a regularized multinomial logistic regression model upon the output CLS token. In training and testing, we do not perform any data augmentation; images are resized to 224 pixels along the shorter side, followed by a center crop of 224×224. We minimize the cross-entropy objective using L-BFGS with ℓ2-regularization. We select this ℓ2-regularization constant on the validation set over 45 logarithmically spaced values between 10−6 and 105. The maximum number of L-BFGS iterations is set to 1000, similar as that in DINO v2 [68].

#### C. In-context Learning Examples

All of the three types of in-context examples are summarized in Table 16, Table 17, and Table 18, respectively.

Table 16. Detailed in-context learning examples for Template 1: c –> Caption. Here c is the concept.

- 1 coucal –> A vibrant coucal is perched on the branch of a lush green tree, surrounded by wildflowers.

- 2 bee eater –> A lively bee eater is elegantly perched on a branch, peering intently.

- 3 three-toed sloth –> A three-toed sloth is lazily hanging from a sturdy, tropical rainforest tree.

- 4 hay –> In the serene countryside, hundreds of neatly stacked hay bales lay scattered under the

softly glowing golden sunset sky.

- 5 station wagon –> A shiny, red station wagon is parked under the dappled shade of a large oak tree,

highlighting its spacious and family-friendly design.

- 6 zebra –> A zebra is gallantly trotting across the vast, sunlit plains of the African savannah, creating

a captivating black and white spectacle.

- 7 vase –> In the well-lit living room, a beautifully designed, delicate vase stands out as the center-

piece, exuding an aura of elegance.

- 8 barber chair –> A shiny black barber chair sits invitingly in a bustling, well-lit barbershop.

- 9 carbonara –> A heaping plate of creamy carbonara pasta topped with fresh parsley sprigs.

- 10 mink –> In the midst of a dense forest with shimmering green leaves, a sleek mink gracefully

navigates the underbrush, showcasing its rich, brown fur.

- 11 small white butterfly –> A small white butterfly gracefully flutters amongst vibrant, blooming summer flowers.

- 12 christmas stocking –> A vibrant red Christmas stocking is hanging delicately from a festively decorated man-

telpiece.

- 13 horse-drawn vehicle –> An antique horse-drawn vehicle is stationed amidst a peaceful country landscape, its

rustic wooden structure gleaming under the warm afternoon sun.

- 14 ruler measuring stick –> A manual craftsman is precisely measuring a wooden log with a ruler stick.

- 15 picket fence –> A tranquil suburban scene featuring multiple white picket fences surrounding well-

maintained green lawns, punctuated by diverse, colorful flowerbeds.

- 16 suspension bridge –> Depicting a long suspension bridge, its steel cables elegantly stretching towards the sky,

connecting two ends over a scenic river.

- 17 brain coral –> A vibrant brain coral stands out amidst the serene backdrop of underwater marine life.

- 18 revolver –> Multiple antique revolvers lie on a wooden table, gleaming under soft, ambient light.

- 19 slip-on shoe –> A pair of slip-on shoes, with their sleek, black leather exterior and comfortable, cushioned

interior, are neatly placed on a wooden floor.

- 20 hand-held computer –> A hand-held computer, compact and portable, rests on a well-lit desk, surrounded by

various technological paraphernalia and a steaming cup of coffee.

- 21 mattress –> A teddy bear lying face down on a bedspread covered mattress in front of a window.

- 22 refrigerator –> A nicely decorated kitchen with metallic refrigerator and blue counter.

- 23 ball –> Silver balls are lined up in the sand as people mill about in the background.

- 24 wheel –> The motorcycle’s gleaming steering wheel, vivid red door reflected in the side mirror,

and a youth passing by, creating a dynamic urban tableau.

- 25 plane –> A group of trick planes turned upside down leaving smoke trails.

- 26 vehicle –> Army vehicles, including a U.S. Army jeep and aircraft in a hangar or on display

- 27 boy –> a little boy wearing sunglasses laying on a shelf in a basement.

- 28 fence –> a man standing near a fence as reflected in a side-view mirror of a red car.

- 29 wood table –> A footed glass with water in front of a glass with ice tea, and green serpentine bottle

with pink flowers, all on a wood table in front of chair, with a window to city view.

- 30 toilet –> A black and white toilet sitting in a bathroom next to a plant filled with waste.

- 31 table lamp –> A textured brass table lamp, casting a warm, golden glow, accents a cozy reading nook

beside a leather armchair and a stack of books.

- 32 hair dryer –> A modern sleek and white hair dryer, with a textured grip, stands next to a set of

hairbrushes.

- 33 street sign –> The street signs indicate which way a car can and cannot turn while the signal light

controls traffic.

- 34 instrument –> Man dressed in Native American clothes protecting musical instruments from the rain

with an umbrella.

- 35 train –> A man and a cow’s faces are near each other as a train passes by on a bridge.

- 36 giraffe –> A couple of large giraffe standing next to each other.

- 37 red admiral butterfly –> a red admiral butterfly, alights upon a dew-kissed sunflower, wings glistening under the

soft morning light.

- 38 stupa –> Surrounded by verdant foliage, a white stupa rises, adorned with golden accents and

intricate patterns, while devotees circle its base offering prayers.

- 39 elephant –> A group of elephants being led into the water.

- 40 bottle –> Motorcycles parked on a street with a bottle sitting on the seat of the nearest the camera.

- 41 trombone –> On a polished wooden stage, a gleaming brass trombone rests, its slide extended, next to

scattered sheet music and a muted trumpet.

- 42 keyboard –> Sleek black keyboard with illuminated backlit keys, a soft wrist rest, and a nearby

wireless mouse on a textured matte desk surface.

- 43 bear –> The brown bear sits watching another bear climb the rocks

- 44 snowboard –> A man standing next to his snowboard posing for the camera.

- 45 railway –> a woman and her son walking along the tracks of a disused railway.

- 46 sand –> the waves and the sand on the beach close up

- 47 pixel –> very colorful series of squares or pixels in all the colors of the spectrum , from light to

dark

- 48 cigar –> a burning cigar in a glass ashtray with a blurred background.

- 49 music –> happy girl listening music on headphones and using tablet in the outdoor cafe.

- 50 earring –> this gorgeous pair of earrings were featured in april issue.

- 51 cliff –> Steep cliff, jagged edges against azure sky, with seabirds soaring and waves crashing

below.

- 52 corn cob –> Fresh corn cob, golden kernels glistening with dew, nestled amid green husks in a sunlit

field.

- 53 archaeological exca-

vation

–> In this intriguing scene, archaeologists meticulously uncover ancient relics at an archaeological excavation site filled with historical secrets and enigmas.

- 54 formal garden –> This is an immaculately kept formal garden, with perfectly trimmed hedges, colorful,

well-arranged flower beds, and classic statuary, giving a vibe of tranquil sophistication.

- 55 veterinarians office –> The busy veterinarian’s office is a hive of activity with pets awaiting treatment and care.

- 56 elevator –> A modern, well-lit elevator interior with shiny metal walls and sleek buttons.

- 57 heliport –> Situated in a lively area, the heliport stands out with numerous helicopters taking off and

landing against the city’s skyline.

- 58 airport terminal –> In the spacious airport terminal, travelers hurriedly navigate through check-ins and

security, making it a hive of constant activity.

- 59 car interior –> Inside the car, the leather seats exude luxury, contrasted by the high-tech dashboard,

creating an atmosphere of sleek comfort and convenience.

- 60 train interior –> The inside of the train offers a spacious setting with numerous comfortable seats.

- 61 candy store –> The sweet aroma of sugared treats fills the air in a vibrant candy store, adorned with

colourful candies and cheerful customers.

- 62 bus station –> The bustling bus station thrums with restless energy, as travelers navigate through the

crowded space, awaiting their journeys amid the echoes of departing buses.

- 63 castle –> Nestled amidst towering mountains, the majestic castle spews ancient grandeur, with its

stone walls and towering turrets exuding tranquility and timeless mystique.

- 64 palace –> The grand palace exudes regality, radiant under the sun, showcasing ornate decorations,

intricate sculptures, and exquisite architectural sophistication.

- 65 kitchen –> The heart of the home unfolds in the kitchen, characterized by stainless steel appliances,

navy blue cabinets, and a patterned tile backsplash.

- 66 raceway –> The high-speed adrenaline-filled atmosphere of the raceway is pulsing with the roars of

powerful engines and excited cheering fans.

- 67 bakery –> The warm, inviting bakery is filled with the intoxicating aroma of fresh bread, assorted

pastries, and brewing coffee.

- 68 medina –> This ancient, labyrinth-like medina exudes an air of mystique with its vibrantly decorated

shops lining narrow, stone-cobbled pathways.

- 69 skyscraper –> The city skyline is dominated by towering skyscrapers, creating a captivating blend of

technology and architectural innovation.

- 70 supermarket –> The supermarket scene is lively, filled with individuals scanning shelves, children reach-

ing for treats, and clerks restocking fresh produce.

- 71 closet –> The compact closet, brimming with clothes and shoes, exudes a feeling of organization.

- 72 assembly line –> In the heart of a busy factory, an orderly assembly line hums with continuous activity,

filled with workers focused on their precision tasks.

- 73 palace room –> A man in military dress uniform stands in an ornate palace room with antique furniture

and Christmas decorations.

- 74 barn doorway –> A farmer holding an animal back while another farmer stands in a barn doorway.

- 75 food court –> A bustling food court with a variety of culinary stalls, featuring vibrant signage, aromatic

dishes, and communal seating, creates a diverse dining experience.

- 76 mountain –> Majestic mountains, their peaks dusted with snow, overlook a serene alpine lake where

hikers and photographers gather to enjoy the breathtaking scenery.

- 77 squash court –> Against a clear glass wall, a squash court with gleaming wooden floors, white boundary

lines, and two rackets awaits players.

- 78 subway station –> Dimly lit subway station with graffiti-covered walls, commuters waiting

- 79 restaurant –> Cozy restaurant with wooden tables, ambient lighting, patrons chatting, and plates filled

with colorful dishes, framed by exposed brick walls and hanging green plants.

- 80 field –> there is a large heard of cows and a man standing on a field.

- 81 aquarium –> Amidst vivid coral formations, an aquarium teems with colorful fish, shimmering under

soft blue lights.

- 82 market –> A large group of bananas on a table outside in the market.

- 83 park –> a young boy is skating on ramps at a park

- 84 beach –> old fishing boats beached on a coastal beach in countryside.

- 85 grass –> little boy sitting on the grass with drone and remote controller.

- 86 woven –> The woven basket’s intricate pattern creates a visually captivating and tactile surface.

- 87 knitted –> The knitted blanket envelops with cozy warmth

- 88 flecked –> The stone surface was flecked, giving it a uniquely speckled and rough appearance.

- 89 bubbly –> The liquid gleamed, showcasing its bubbly, effervescent texture vividly.

- 90 cobwebbed –> The dusty corner was cobwebbed, displaying years of untouched, eerie beauty.

- 91 stained –> A weather-worn wall manifests an intriguing pattern of stained texture.

- 92 scaly –> The image showcases a close-up of a lizard’s scaly, rough texture.

- 93 meshed –> A patterned image depicting the intricate, tightly-knit texture of meshed fabric.

- 94 waffled –> A fresh, golden-brown waffle displays its distinct crisply waffled texture invitingly.

- 95 pitted –> The image portrays an intriguing terrain, characterized by a pitted, moon-like surface.

- 96 studded –> A studded leather jacket gleams, highlighting its rough, tactile texture.

- 97 crystalline –> The picture showcases an exquisite, crystalline texture with stunning brilliance and

clarity.

- 98 gauzy –> A delicate veil of gauzy texture enhances the ethereal, dreamy atmosphere.

- 99 zigzagged –> The photo captures the zigzagged texture, emphasizing the rhythmic, sharp-edged pat-

terns.

- 100 pleated –> A flowing skirt delicately showcasing the intricate detail of pleated texture.

- 101 veined –> A detailed image showcasing the intricate, veined texture of a leaf.

- 102 spiralled –> The spiralled texture of the seashell creates a captivating, tactile pattern.

- 103 lacelike –> The delicate veil features an intricate, lacelike texture, exuding elegant sophistication.

- 104 smeared –> A wall coated with thick, smeared paint exudes a rough texture.

- 105 crosshatched –> A worn, vintage book cover, richly crosshatched, exuding old-world charm.

- 106 particle –> abstract background of a heart made up of particles.

- Table 17. Detailed in-context learning examples for Template 2: c,bg –> caption. Here c is the concept, and bg is the background.
- 107 stick insect, under-

growth

–> A stick insect, masterfully camouflaged, clings to a fern amidst the sprawling, dense undergrowth of a lush, tropical forest.

- 108 black swan, public

garden

–> In the peaceful ambiance of a lush public garden, a majestic black swan gracefully glides across a shimmering emerald-green pond.

- 109 st. bernard, family-

photo

–> In the heartwarming family photo, a gregarious St. Bernard dog is seen joyfully nestled among his adoring human companions.

- 110 measuring cup, food

prep area

–> In the food prep area, multiple transparent measuring cups are neatly organized on the marble countertop.

- 111 can opener, hotel

room

–> A sleek, stainless steel can opener is sitting on the glossy dark-wood kitchenette counter of a modern, well-appointed hotel room.

- 112 small white butterfly,

pond side

–> A delicate, small white butterfly flutters gracefully above the tranquil pond side, creating a serene image amidst lush greenery.

- 113 hair dryer, theatre –> A sleek, professional hair dryer is positioned center stage amidst the dramatic velvet

curtains and ornate details of a bustling theatre.

- 114 water bottle, airport –> A reusable water bottle sits on the glossy surface of a bustling airport terminal counter,

amidst a backdrop of hurried travelers and departure screens.

- 115 leonberger, horse

ranch

–> Several Leonbergers are joyfully romping around a bustling horse ranch.

- 116 lighter, motorhome –> In the cozy, cluttered environment of a well-traveled motorhome, a sleek silver lighter

holds dominion on the rustic wooden table.

- 117 slug, foliage –> A solitary, glistening slug meanders slowly amidst lush, dense green foliage, leaving a

slimy trail on dewy leaves in its path.

- 118 ring binder, educa-

tion department

–> The ring binder, filled with important documents, sits prominently on a well-organized desk in the bustling education department.

- 119 weimaraner, pet store –> A sleek, silver-gray Weimaraner is spotted curiously sniffing around various pet supplies

in a well-stocked and vibrant pet store.

- 120 norfolk terrier, coun-

tryside

–> A lively Norfolk terrier joyfully bounds across a lush, green countryside, its red fur contrasting vividly with the vast open surroundings.

- 121 dalmatian, apple or-

chard

–> A lively Dalmatian is playfully darting amongst the lush rows of a bountiful apple orchard, its spots contrasting against the ruby fruits.

- 122 television, mountain

lodge

–> A sleek, modern television sits prominently against the rustic, wooden walls of an inviting mountain lodge, surrounded by pine-furnished decor.

- 123 guillotine, horror

story

–> In the shadowy landscape of a suspenseful horror story, a grim, menacing guillotine looms ominously, exuding a petrifying sense of imminent dread.

- 124 hot tub, condo-

minium

–> A luxurious hot tub is nestled in the private balcony of a high-rise condominium, boasting spectacular cityscape views.

- 125 leaf beetle, plant nurs-

eries

–> A vibrant leaf beetle is diligently navigating through a lush plant nursery, its metallic sheen contrasting against the abundant green foliage.

- 126 carolina anole, hiking

trails

–> A small Carolina Anole lizard basks in the warm sunlight, gracefully draped over a gnarled tree root next to a bustling hiking trail.

- 127 girl, laboratory –> teenage girl and boy working in a laboratory on an experiment.

- 128 tiger, forest –> Two tigers are running together in the forest.

- 129 sunset, lake –> Golden sunset hues reflect on a calm lake, silhouetting a lone canoeist against a backdrop

of fiery clouds.

- 130 building, mountain –> town of skyline over roofs of historic buildings with the mountains in the background.

- 131 block plane, weath-

ered wood

–> A block plane, its sharp blade gleaming, rests on weathered wood

- 132 olive tree, soil –> single olive tree planted in the center of a dry and cracked soil

- 133 hamster, pet store –> A curious hamster peers out, with pet store shelves stacked with supplies behind.

- 134 bag, factory –> plastic bags production line in a factory.

- 135 restaurant, ocean –> young pretty couple dining in a romantic atmosphere at restaurant on the boat with ocean

on the background

- 136 helicopter, burning

forest

–> a helicopter flies over a portion of burning forest.

- 137 pipe organ, commem-

oration event

–> striking pipe organ dominates with its notes resonating, while a somber commemoration event unfolds in the backdrop

- 138 rotisserie, wedding

reception

–> Rotisserie turning golden meats, with a bustling wedding reception, twinkling lights, and guests mingling.

- 139 duck, taiga –> A group of ducks paddle on a tranquil pond, dense taiga and towering conifers looming

in the background.

- 140 tiger beetle, rice

fields

–> Amidst verdant rice fields, a shimmering tiger beetle perches prominently on a dewkissed blade of grass.

- 141 girl, barn –> slow motion clip of a girl walking with her horse through a barn

- 142 headmaster, gradua-

tion ceremony

–> the headmaster addresses the graduating seniors during graduation ceremonies.

- 143 businessperson, mu-

sic festival

–> businessperson and guest attend music festival.

- 144 fountain, park –> Water cascades from an ornate fountain, surrounded by autumn-hued trees in a serene

park.

- 145 speedboat, water –> A sleek speedboat glides on shimmering waters, powered by twin high-horsepower

outboard motors.

- 146 pipe, beach –> a rusty water pipe on the beach.

- 147 pretzel, home kitchen –> Golden pretzel rests on a wooden board, with a cozy home kitchen, pots and tiled

backsplash, behind.

- 148 forklift, paper mill –> A forklift transports hefty paper rolls amidst the industrial bustling paper mill.

- 149 lotion, therapy center –> Blue lotion bottles lined up at a thalasso therapy center by the ocean.

- 150 guinea pig, sand

dunes

–> Guinea pig exploring vast golden sand dunes, with tiny footprints trailing behind.

- 151 groom, wedding cere-

mony

–> father of groom congratulating him after the wedding ceremony.

- 152 fishing boat, village –> fishing boats moored at fishing village a suburb of capital of the state,

- 153 red fox, yard –> wild red fox sitting on a partially snow covered front yard of a house in the suburbs of a

small city

- 154 grey wolf, woodland

areas

–> A grey wolf prowls silently, eyes alert, through dense, misty woodland areas with moss-covered trees.

- 155 cheetah, edges of

swamplands

–> A cheetah crouches, poised and watchful, at the lush edges of murky swamplands.

- 156 wine bottle, living

room

–> in the living room, a person si opening a wine bottle with corkscrew with wooden barrel

Table 18. Detailed in-context learning examples for Template 3: c,rel –> caption. Here c is the concept, and rel is the relation.

- 157 product packet / pack-

aging, next to

–> A vibrant product packet, adorned with colorful labels and intricate designs, is neatly placed next to an elegant crystal glass.

- 158 croquet ball, behind –> A vivid, red croquet ball rests serenely, hiding behind a worn, rustic wooden fence in a

sun-kissed, lush green lawn.

- 159 bassoon, in front of –> A beautifully crafted bassoon stands elegantly in front of a backdrop of velvet curtains,

ready to perform at a concert.

- 160 grand piano, above –> A gorgeous, antique chandelier is suspended above the glossy black grand piano, illumi-

nating it with warm, opulent light.

- 161 bolo tie, behind –> A beautifully crafted bolo tie is casually hung, indicating its previous use, behind a rustic,

well-polished wooden shelf.

- 162 waffle iron, next to –> A large, black waffle iron is placed next to a sparkling glass jar filled with golden maple

syrup on a wooden countertop.

- 163 komodo dragon, be-

low

–> A young child grins excitedly, peering down from a secure bridge, as a colossal Komodo dragon sprawls lazily below in the wildlife park.

- 164 vaulted or arched ceil-

ing, besides

–> Besides the grand marble statue, glimpses of an intricate vaulted or arched ceiling add to the room’s majestic charm.

- 165 gossamer-winged

butterfly, next to

–> A lovely, vibrant gossamer-winged butterfly is gently perched next to a dew-kissed red rose in an early morning garden.

- 166 kit fox, in front of –> A group of small, fluffy, golden kit foxes is playfully gathered in front of a lush, green,

towering forest backdrop.

- 167 koala, in –> A cute, fuzzy koala is visibly relaxed, nestled contentedly in the crook of a towering,

lush green eucalyptus tree.

- 168 centipede, above –> A vibrant green centipede is effortlessly crawling on a tree branch, positioned distinctly

above a patch of untouched fern leaves.

- 169 mountain bike, above –> A mountain bike is displayed prominently above the rustic mantlepiece, showcasing its

sleek design and intricate details.

- 170 wallaby, above –> A fluffy, brown wallaby is leaping high, appearing as if it is effortlessly floating above a

lush, green Australian field.

- 171 giant panda, on –> A playful giant panda is perched on a sturdy tree branch, munching on fresh green

bamboo amidst the tranquil forest ambiance.

- 172 beagle, on –> A pack of adorable beagles are spotted lounging on an expansive, sunbathed meadow

with colorful wildflowers sprouting around them.

- 173 beach, on –> A vivid sunset is on display over a sprawling beach, casting warm hues on the waves

gently lapping at the sandy shore.

- 174 grey whale, on –> A voluminous grey whale is majestically breaching, its massive body on display against

the azure backdrop of the expansive ocean.

- 175 tractor, in front of –> A bright red tractor is parked in front of a rustic, weathered barn, casting long shadows

under the golden afternoon sun.

- 176 cabbage, besides –> A vibrant image portrays a lush, green cabbage, glistening with dewdrops, nestled

besides a rustic, wooden crate full of freshly harvested vegetables.

