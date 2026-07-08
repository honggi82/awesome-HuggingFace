### Aligning Text, Images and 3D Structure Token-by-Token

# arXiv:2506.08002v2[cs.CV]6Jan2026

Aadarsh Sahoo∗ Vansh Tibrewal∗ Georgia Gkioxari California Institute of Technology

[Figure 1]

Figure 1. Kyvo: a decoder-only transformer aligns a structured 3D modality with language and vision. This 3D modality represents scenes as lists of objects, each defined by its 3D shape, type, 3D position, pose and size parameters. Kyvo unifies the token space of images, text, and 3D to enable a variety of complex visual 3D tasks.

#### Abstract

Creating machines capable of understanding the world in 3D is essential in assisting designers that build and edit 3D environments and robots navigating and interacting within a three-dimensional space. Inspired by advances in language and image modeling, we investigate the potential of autoregressive models for a new modality: structured 3D scenes. To this end, we propose a unified LLM framework that aligns language, images, and 3D scenes and provide a detailed

“cookbook” outlining critical design choices for achieving optimal training and performance addressing key questions related to data representation, modality-specific objectives, and more. We show how to tokenize complex 3D objects to incorporate into our structured 3D scene modality. We evaluate performance across four core 3D tasks – rendering, recognition, instruction-following, and question-answering – and four 3D datasets, synthetic and real-world. We show our model’s effectiveness on reconstructing complete 3D scenes consisting of complex objects from a single image and on real-world 3D object recognition tasks. Project webpage: https://glab-caltech.github.io/kyvo.

∗Equal contribution

#### 1. Introduction

Large language models (LLMs) that fuse text and images have unlocked unprecedented visual capabilities, such as visual captioning and text-guided image generation. Inspired by this success, we explore a third modality – structured 3D scenes – and show how aligning it with text and images allows LLMs to tackle a new suite of visual tasks in 3D, such as 3D reconstruction from a single image and 3D-conditioned image generation. We start from a language-pretrained transformer and extend it with a structured 3D modality that is designed to try to leverage its linguistic reasoning and generalization capabilities.

Our structured 3D modality encodes a scene as a list of its objects, where every object is specified by its 3D shape (e.g., 3D mesh, 3DGS), type, 3D position, 3D size and 3D pose, shown in Fig. 1. This modality captures aspects of the physical world that are not directly conveyed through language or images alone. Additionally, it slots naturally into the unified token space shared by language and vision through an object-by-object tokenization scheme that integrates seamlessly with image and text tokens, so any modality can serve as input or output. As a result, it supports a broad range of tasks in 3D, such as image generation conditioned on the 3D scene structure (3D → image), predicting 3D objects, their shapes and locations from a single image (image → 3D),

and language-guided object-centric 3D editing (3D + image + text → 3D + image) – all within the same model design – shown in Fig. 2. These capabilities can reshape workflows. Robots can parse an image into a 3D scene composed of 3D objects, while designers can create complex scenes of objects through language in one forward pass of an LLM instead of wrestling with hard-to-use software like Blender.

But, how does one design and train an LLM that aligns this structured 3D modality with image and text? While there is extensive work on language-only [1, 14, 38] or vision-and-language models [6, 21, 39], there is limited work on training with an additional structured 3D modality. We explore the vast design space and evaluate the impact of design choices in architecture, training strategies, data representation, modality-specific objectives, and more. Our testbed consists of four core 3D tasks – image generation, recognition, instruction-following, and question-answering – and four challenging datasets, both synthetic and real-world. Through an extensive “cookbook” of rigorously validated empirical findings we hope to provide guidelines to design 3D-aligned multi-modal LLMs, including guidelines on tokenizing complex 3D shapes. We demonstrate the effectiveness of our unified LLM framework, Kyvo (Greek for “3D cube”), on single-view 3D reconstruction of complex object geometries. Finally, we apply our model to real-world domains of indoor and outdoor scenes for the task of imagebased 3D object recognition, where we show that our model can tackle tasks that previously required task-specific vision specialists.

#### 2. Related Work

LLMs. The success of LLMs is attributed to the scalability of transformers [42] taking the form of encoderdecoder [11, 36] and most popular decoder-only [1, 14, 38] models. LLMs showcase the effectiveness of autoregressive formulations in achieving task generalization, a finding we extend to 3D structured representations.

Vision-Language Models (VLMs). Early VLMs [3, 18, 32] learned from image–text pairs for tasks like zero-shot classification, retrieval, and QA. Modern VLMs [9, 23, 24, 28, 43, 45] extend these abilities using internet-scale data and stronger alignment. Recent studies [6, 21, 39] analyze vision-centric design choices but treat images only as input, while others [37, 40, 48] explore early fusion and diffusion for image generation. Nonetheless, current VLMs remain weak in 3D reasoning and geometry prediction, tasks that we tackle in this work.

LLMs & 3D. Prior work integrates LLMs with diverse 3D formats for tasks like VQA and grounding [27]. Some [10, 22, 26] extend VLMs to 3D via depth-aware queries, while others [16, 20, 29, 30, 33, 35, 47] embed 3D data (e.g., point clouds, NeRFs) with CLIP-style features for open-ended

QA. 3D-LLM [16] uses holistic 3D point clouds for captioning and QA, whereas our model’s structured 3D modality decomposes scenes into objects, enabling direct alignment across language and vision. This supports tasks like 3D shape and pose prediction from a single image and image generation from object-centric 3D inputs. Similarly, while SceneScript [4] autoregressively predicts 3D boxes from videos and full scene clouds, our approach aligns images with structured 3D object and scene representations.

3D Tokenization. Prior work on 3D tokenization targets limited tasks, mainly single-asset generation. SAR3D [8] uses triplanes, while AToken [25] builds on Trellis [46] to train a joint tokenizer for images, 3D, and video. In contrast, we design a 3D tokenizer for structured scene encoding, optimized for compactness to represent multiple objects per scene. Whereas AToken and SAR3D use over 20k and 2040 tokens per asset respectively, our tokenizer achieves comparable or better reconstruction with far fewer tokens (see Sec. 3.2).

#### 3. Building our model token-by-token

We present our approach, Kyvo, that aligns the 3D modality with text and images. The result is a unified multimodal LLM framework that can perform a range of visual 3D tasks – rendering, reconstruction, recognition and instruction-following – as shown in Fig. 2.

Sec. 3.1 outlines our experimental setup, including tasks and datasets. Sec. 3.2 presents a bottom-up analysis of key design choices – covering data representation, 3D tokenization, sequence design, and scaling – and addresses challenges in modality-specific tokenization and optimization. We summarize empirical insights and adopt optimal configurations for each subsequent experiment. Finally, we explore generalization to complex scene layouts (Sec. 3.3), 3D shape representations (Sec. 3.4), and real-world recognition (Sec. 3.5).

##### 3.1. Setup: tasks and datasets

We design and train an autoregressive model aligning 3D with images and language, capable of performing 3D tasks. Tasks. We focus on four core 3D tasks: image generation from 3D scene specifications (rendering), 3D object recognition & reconstruction from a single image, instructionfollowing 3D editing and question-answering.

Rendering: 3D → Image. Given a 3D scene described by our structured modality (object types, shapes, locations, poses), the model generates a corresponding image. This task tests whether rendering – typically requiring tools (e.g., Blender) and complex processes (e.g., ray tracing, rasterization) – can be reframed as feed-forward next-token prediction.

Reconstruction: Image → 3D. This is a dual of rendering. Given an image, the model predicts the underlying 3D scene structure, including object shapes, types, positions and poses.

[Figure 2]

Figure 2. 3D task examples with Kyvo’s unified autoregressive framework using a structured 3D modality. (1) 3D shape and scene reconstruction: From a single input image, Kyvo reconstructs individual objects with accurate geometry and spatial relationships. (2) 3D object recognition: Given an input image, Kyvo identifies objects and predicts their 3D positions in real-world scenes. (3) Shape and scene rendering: Kyvo generates semantically consistent images from structured 3D scene inputs. (4) Instruction-Following: Given an image, 3D scene and text instruction, Kyvo produces coherent modifications to both image and the 3D representation.

Instruction-Following: (Image,3D,TextI) → (Image,3D). We define a set of tasks to manipulate and modify 3D scenes given a text instruction. These include: (1) modifying the appearance of objects, (2) adding new objects, (3) removing objects, and (4) moving an object to a desired location. For each subtask, we craft templated natural language instructions to evaluate the model’s ability to follow instructions in 3D, e.g. “Remove the red mug behind the yellow bottle”.

Question-Answering: (Image,3D,TextQ) → TextA. We generate QA pairs using templated questions. Given the 3D scene, an image, and a query (e.g. “Is there a green object the same size as the glass vase?”), the model predicts a natural language answer. Details are in the Appendix.

All four tasks require spatial reasoning and grounding. Two of them explicitly involve text as input or output and all four implicitly require linguistic reasoning in the use of the structured 3D modality as input or output.

Datasets. We consider four datasets for our experiments. Two synthetic ones: CLEVR [19] features scenes of simple shapes in varying layouts; ObjaWorld features complex objects of any geometry sourced from Objaverse [12]. Two realworld ones: Objectron [2] and ARKitScenes [5] comprising real-world indoor and outdoor scenes of various object types.

Evaluation. Per our task definitions, Kyvo’s output can be of either modality: text, image, or 3D.

Recognition. We evaluate predicted 3D scenes using the Jaccard Index, J = tp+fptp+fn, which measures object-level agreement between prediction and ground truth based on matching attributes (type, size, color, material) and spatial proximity within threshold τ. True positives are matched objects within τ, false positives are extra predictions, and false negatives are misses. We report the mean Jaccard Index over τ ∈ {0.05,0.10,0.15,0.20,0.25}, reflecting both

recognition and spatial accuracy.

Question-Answering. We report answer accuracy based on exact match between predicted and ground-truth text (typically 1–2 tokens). A random baseline yields 0.359 accuracy, while a frequency-based baseline reaches 0.430. See Appendix for more details.

Rendering. In the rendering task, models generate images from 3D scene inputs. Standard image metrics (L2, SSIM [44], FID [15], PSNR) fail to capture object placement or attribute accuracy. We therefore use human evaluations: annotators rank anonymized outputs against ground truth, and we report the mean rank. While L2 and SSIM roughly follow these trends, human judgment better reflects scene correctness (details in Appendix).

Instruction-Following. Here, the model predicts both images and 3D scenes. We report the Jaccard Index for 3D outputs, as this is the primary focus of our work.

##### 3.2. Cookbook

Our model, Kyvo, starts from language as the principal modality. Our backbone is the decoder-only Llama-3.21B-Instruct [14] initialized from language-only pretrained weights. We extend it with modality-specific tokenizers for images and the structured 3D modality, along with modified input embeddings and output projections (Fig. 1). We do this in order to leverage the generalization and reasoning capabilities of LLMs, a hypothesis we later confirm in Tab. 5b. We evaluate key architectural choices – each with significant performance impact – which we hope offer insights to guide future multimodal LLM development in the 3D domain.

We begin by exploring how to represent and tokenize our structured 3D modality. We explore the tokenization of individual 3D objects of diverse geometry and appearance

###### Aux Rec Loss Mean Rank(↓)

✗ 2.828 Fixed Single View 1.672 Multiple Views 1.500

Table 1. Effect of auxiliary reconstruction loss.

3D Tokenizer Mean Rank(↓)

SAR3D 1.605 Kyvo 3D VQ-VAE 1.395

Table 2. 3D tokenization comparison using human evaluation.

from Objaverse as well as the tokenization of scenes containing multiple objects of known geometry in CLEVR. This setup helps establish best practices for integrating 3D into a unified LLM framework. We then extend these findings to complex scenes in ObjaWorld (Sec. 3.3 & Sec. 3.4) and validate generalization to real-world scenes in Objectron and ARKitScenes (Sec. 3.5).

###### 3.2.1. What is the optimal 3D representation?

Our model handles three modalities – images, text, and structured 3D scenes – by converting each into token sequences for autoregressive modeling.

Text. We employ an off-the-shelf text tokenizer from Llama-3.2 [14] with a 128,000 size vocabulary. Text instructions, questions, and answers are tokenized and enclosed within special, learnable tokens [TEXT-START] and [TEXT-END].

Images. Our framework addresses tasks involving images as both inputs and outputs. To enable image generation within an autoregressive paradigm, we adopt discrete image representations using VQGAN [13]. This approach maps continuous image features to discrete tokens through a learned codebook. Specifically, we fine-tune a pre-trained VQGAN model to optimize the codebook for our visual domain. Image tokens are enclosed within special, learnable tokens, [IMAGE-START] and [IMAGE-END].

3D scene tokenization. We represent 3D scenes as structured token sequences, where each list element encodes one object. Object attributes – shape, size, location, color, material – are expressed through special learnable tokens (e.g., [SHAPE]), whose values may be text (“car”, “yellow”), numbers (for size or position), or learned 3D embeddings. Each object is enclosed by [OBJECT-START] and [OBJECT-END], and the full scene by [SCENE-START] and [SCENE-END]. An example scene with two objects is shown below:

[SCENE-START][OBJECT-START][SHAPE]<v11, v21,...,v5121 >[LOCATION]-0.15 1.05 0.00 [POSE]0.00 0.00 3.00 [OBJECT-END][OBJECT-START][SHAPE]<v12,v22,..., v5122 >[LOCATION]0.25 2.10 0.00[POSE]0.00 0.00

-2.05[OBJECT-END][SCENE-END]

We discuss two key design decisions:

- (1) How to tokenize 3D shapes (geometry & texture)? We aim for our structured 3D modality to encode complex objects via compact, autoregressively decodable 3D shape representations. To this end, we adopt Trellis [46], which

Multi View Reconstructions

Multi View Renders

|[Figure 3]|
|---|

|[Figure 4]|
|---|

Dense Tokenized Representation

Reconstructed SLAT

GT SLAT

Ground Truth Asset

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

|[Figure 9]|
|---|

|[Figure 10]|
|---|

3D VQ-VAE

TRELLIS

3D VQ-VAE

TRELLIS

Decoder

Decoder

Encoder

Encoder

Latent Space Reconstruction Loss

|[Figure 11]|
|---|

|[Figure 12]|
|---|

Decoded Pixel Space Reconstruction Loss

Figure 3. 3D VQ-VAE training involves the standard VQ-VAE losses (including reconstruction loss) applied in latent space as well as an auxiliary reconstruction loss applied in decoded pixel space.

encodes geometry and texture as sparse voxel features (slats) z = {(zi,pi)}Li=1, where each zi ∈ R8 represents local features and pi indexes active voxels in an N3 grid. Although slats are sparse (L ≪ N3, typically L ≈ 20k for N = 64), their length makes autoregressive modeling intractable.

We therefore train a 3D VQ-VAE [41] to compress slats from 643×8 to a dense 83×128 latent, vector-quantized with an 8192-token codebook. Each object is thus represented by 512 discrete tokens—a ∼40× reduction—enabling efficient autoregressive decoding.

How do we train such a 3D VQ-VAE in latent slat space while preserving essential geometric and appearance information? We find that training the VQ-VAE in the latent space of slats is insufficient to learn an effective representation for reconstruction. To overcome this, we apply an auxiliary reconstruction loss to the decoded reconstructed slats in pixel-space, as shown in Fig. 3. We use the same pixel-space reconstruction loss (L1, D-SSIM and LPIPS) as in [46]. This yields significantly better reconstructions, despite achieving a similar latent space reconstruction loss, as shown in Fig. 4a. Additionally, we find that imposing a multi-view reconstruction loss of the asset (randomly sampled from 150 views) leads to better reconstructions than a single fixed view. We quantify the improvement using human evaluations in Tab. 1.

Our Trellis-based 3D VQ-VAE matches or surpasses SAR3D [8] while using 4× fewer tokens (512 vs. 2040). Qualitative and quantitive comparisons are shown in Fig. 4b and Tab. 2. Importantly, our compact 3D tokens integrate seamlessly into our unified vocabulary with image and language tokens, enabling efficient multi-object scene encoding and autoregressive decoding, while SAR3D tackles singleasset generation only.

While our representation enables effective 3D asset tokenization and reconstruction, we further test its suitability for autoregressive decoding. Using the same Llama-3.2-Instruct backbone, we evaluate reconstruction and rendering on unseen assets, in Fig. 4c. The learned 3D encodings generalize well, confirming their effectiveness for both reconstruction and decoding. These 3D tokens are then incorporated into our unified vocabulary alongside image and language tokens for training and inference.

Reconstruction

Rendering

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

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

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

GT (Blender)

Single-View Rec Loss

Multi-View Rec Loss

GT (Blender) Kyvo 3D Tokenizer (Ours)

SAR3D

Kyvo Image Rendering

Input Image

No Rec Loss

Kyvo 3D Reconstructions

GT (Blender)

(a)

(b)

(c)

Figure 4. 3D Tokenization findings and comparisons. (a) Effect of auxiliary reconstruction loss. An auxiliary pixel-space reconstruction loss on decoded renders from multiple views of the 3D object leads to much better reconstructions. (b) 3D tokenizer comparison. Reconstructions from our Trellis-based VQ-VAE exceed the quality of SAR3D reconstructions with fewer tokens; improved textures stem from the Trellis slat representation rather than triplanes. (c) Learned 3D shape encodings are effective during decoding. The 3D tokens used in Kyvo are sufficient for both reconstruction and rendering using Llama 3.2 as decoder.

[Figure 39]

Granularity Rendering(↓) Recognition(↑) Instruction(↑) QA(↑)

0.005 1.380 0.5707 0.8643 0.5185 0.05 1.200 0.9212 0.8666 0.4980 0.5 2.020 0.2352 0.2427 0.4730

Table 3. Effect of granularity. A value of 0.05 yields the best overall performance on CLEVR.

- (2) How to tokenize 3D location and orientation? In addition to 3D shape, the 3D location and orientation of an object are key attributes within our 3D modality and critical for understanding 3D spatial relationships and performing grounded 3D actions based on instructions. Thus, accurately encoding coordinates is essential. However, LLMs are struggle with numbers [31, 34]. To mitigate this, we encode each object’s x,y,z coordinates separately as individual tokens, allowing the model to learn distinct embeddings for each coordinate. We discretize the coordinate values using equally spaced bins based on a chosen granularity. We find this granularity to be decisive for performance – if it is too coarse, spatial inaccuracies arise; if it is too fine, it leads to an exponential increase in tokens, with fewer training samples per bin and thus difficulty in learning. Tab. 3 shows the effect of granularity across the four tasks on CLEVR. We choose CLEVR, which features simple, known shapes, to isolate and study the effect of number encodings in nexttoken prediction frameworks. We find that a granularity of 0.05 outperforms the coarser 0.5 and the finer 0.005. Fig. 5 compares image generation for the rendering task on the test set across varying levels of granularity.

- Figure 5. Effect of Granularity. A 0.05 granularity more accurately captures object locations and shapes.

[Figure 40]

- Figure 6. Hybrid number encodings are robust across data scales.

translate to decreased memory requirements and faster training and inference times. The final vocabulary comprises 137,607 tokens covering all modalities.

###### 3.2.2. What matters in input sequence design?

We combine three modalities to construct the input sequence to the model. How should we order the modalities? And how should we encode numbers? We discuss our findings for input sequence design.

Furthermore, our tokenization approach substantially improves efficiency by reducing sequence length compared to standard text tokenizers, which typically fragment floatingpoint values (e.g., "0.000" becomes "0", ".", "000"). We achieve a mean sequence length reduction from 271.4 tokens using the standard Llama-3.2 tokenizer to 93.2 tokens – a 2.91× compression ratio. These efficiency gains directly

Sine-cosine encoding of numbers. While we independently tokenized coordinate values, naively learning embeddings for these tokens may fail to capture the inherent ordering of the numbers (e.g., 2 is between 1 and 3). We investi-

###### CT reorder Weighted loss Rendering(↓)

- ✗ ✗ 2.66

✓ ✗ 3.56

- ✗ ✓ 2.78

###### ✓ ✓ 1.00

Table 4. Effect of center-token reordering and weighted loss.

gate whether augmenting the learned embeddings with sinecosine encodings can enforce numeric ordering relationships. Specifically, we evaluate three encoding strategies: (1) fixed sine-cosine encodings, (2) learned embeddings initialized from scratch, and (3) a hybrid approach where embeddings are learned but augmented with sine-cosine encodings. Fig. 6 plots the effect of these encoding strategies on the recognition task with varying training data sizes. While all methods perform on par in the high data regime, standalone fixed sine-cosine and learned embeddings collapse with low data. Consequently, we adopt the hybrid approach as it demonstrates robustness across data regimes. We show the performance of these encoding strategies across all four tasks in the Appendix.

Should the image or 3D come first? We investigate the impact of the ordering between the image and 3D modality for instruction-following and QA tasks. Our experiments reveal that placing the image before the 3D sequence leads to better performance compared to the reverse order. Specifically, we achieve an accuracy of 0.8666 with the sequence (I,3D,TI) → (I,3D) compared to 0.8350 with (3D,I,TI) → (3D,I). Moreover, we obtain an accuracy of 0.4980 with (I,3D,TQ) → TA compared to 0.4720 with (3D,I,TQ) → TA. This improvement could be attributed to the 3D tokens attending to the entirety of the preceding image tokens, enabling better conditioning and performance.

###### 3.2.3. What matters in output sequence design?

We outline the important design decisions concerning the output sequence and objectives.

Initial token prediction matters. The next-token prediction scheme caused challenges in generating reliable image outputs during inference. Despite achieving low training loss, the model’s decoded (next-token) predictions during inference deviated from expected outputs. At a high level, the issue stems from predicting rich outputs (images) from less informative conditions (3D specifications).

Further investigation revealed the issue: the first token. During inference, the model decodes sequentially, with the first token guiding the output. For CLEVR images, this token, representing the top-left corner, was biased toward a few codes – see Fig. 7a (blue dash plot) – due to CLEVR’s uniform gray background. This caused overfitting and during inference a wrongly predicted first token caused the decoding to diverge. This finding is not just applicable to CLEVR but to any image set with a more uniform background, often the case in graphic design. Moreover, this trend is even evident with real-world images which we show in the Appendix.

To mitigate this issue, we incorporate a center-token reordering scheme to balance the token distribution at the sequence’s starting position, by starting from the center token of the image and alternating hops after that as shown in Fig. 7a. This reordering means the first token now captures a representative part of the scene, instead of an uninformative background patch. The token distribution and method and results are shown in Fig. 7a and Fig. 7b respectively.

Token-specific loss weighting. We apply a weighted loss during training by assigning a higher weight (10.0) to the loss for the first five tokens of the output image sequence. This enforces a stronger constraint to correctly predict the initial tokens, which proved critical for autoregressive decoding.

Tab. 4 shows the impact of center-token reordering and weighted loss for the rendering task. Best performance is achieved when both are combined, also shown in Fig. 7b.

##### 3.3. Generalization to complex scene layouts

Above, we validated our structured 3D modality and developed a cookbook for training unified autoregressive models with tokenized 3D shapes and attributes. We verified our findings for 3D shape encodings on diverse Objaverse objects and for 3D layout attributes (e.g., locations) on CLEVR scenes with known shapes but diverse layouts. We now show these findings generalize to ObjaWorld scenes with complex shapes (from Objaverse assets [12]) and diverse layouts.

We focus on two categories: park scenes (person, bench, lamppost, bird) and living room scenes (person, sofa, coffee table), featuring realistic textured objects with varied geometry at diverse locations and heights. We generate 50,000 scenes per category (total 100,000) to train rendering and recognition models, and 2,000 additional scenes with unseen object layouts for testing. The rendering and reconstruction tasks are defined the same as with CLEVR.

For recognition, Kyvo achieves a Jaccard Index of 0.6415, well below its CLEVR score (0.9212, Tab. 3), reflecting ObjaWorld’s increased complexity. Llama3.2-V, prompted with in-context examples for the same attributes, performs near zero, failing to accurately predict 3D locations.

For rendering, the model renders capture accurate object types, counts, and positions, though some errors occur in fine pose alignment (Fig. 8a).

Generalizing to novel scene configurations. We test on out-of-distribution inputs (e.g., park-only objects placed in living room and vice-versa). Fig. 8b shows our model can generalize to such novel scenarios for the rendering task.

Chaining tasks. Chaining our recognition and rendering models enables image reconstruction to test model robustness. Fig. 8c shows an example of input image, chainedmodel reconstruction, and Blender rendering of the predicted scene. Blender shows accurate object types, poses, and positions, and the reconstructed images also closely match the

[Figure 41]

[Figure 42]

[Figure 43]

(a) (b)

- Figure 7. (a) Top: Schematic representation of center token reordering. Bottom: We convert CLEVR images into 256-token sequences and analyze token frequency. Over 25% of images share the same first token, causing biased predictions. Center token reordering significantly reduces this issue. (b) Inaccurate first-token predictions can cause catastrophic object- and scene-level diversions in autoregressive generations.

[Figure 44]

(a) (b)

[Figure 45]

[Figure 46]

(c)

- Figure 8. (a) Rendering examples. The model predicts images autoregressively from 3D inputs. Errors are largely pose mispredictions, e.g., the bird. (b) Novel scene configurations. (c) Chaining tasks. Our recognition model predicts the 3D scene representation for an input image, which is then visualized through both our rendering model and Blender. input, with only minor pose errors (e.g., the birds).

spatial layouts via our structured, object-centric 3D modality, while Trellis [46], a diffusion-based image-to-3D model trained on Objaverse that treats the scene holistically, often yields distorted shapes (e.g., a deformed bottle) and misaligned layouts (e.g., linearly arranged objects).

For rendering, the model takes the structured 3D modality with encoded shape tokens and outputs the corresponding image, mapping shape to appearance and placing objects at the correct positions and poses. Fig. 9b compares our outputs with Blender renderings, showing that our model reliably captures shapes and spatial relationships, with only minor distortions in challenging cases (e.g., the occluded cheeseburger behind the basketball in the third image).

Qualitative scaling behavior. Fig. 10 shows that with limited data the rendering model produces amorphous color blobs that lack semantic coherence and captures only coarse layouts, while increasing training data progressively improves object geometry, appearance, and spatial relationships, yielding more accurate and consistent renderings.

##### 3.5. Generalization to real-world recognition

We now evaluate our model’s effectiveness for 3D object recognition in real-world scenes. We conduct experiments on two challenging real-world datasets: Objectron [2], which features indoor and outdoor scenes of diverse object categories (e.g., bicycle, camera, car, cereal box); and ARKitScenes [5], featuring complex indoor environments of many object categories (e.g., bathtub, bed, cabinet, chair). ARKitScenes presents additional complexity due to its scene density and ground truth annotation noise. Following the traintest splits by [7], we train a recognition model to detect objects by type and predict their 3D center coordinates and size dimensions in metric space. During training, we augment the data with random horizontal flipping. Table 5a reports the Jaccard Index of our Kyvo and a state-of-the-art 3D object detector Cube R-CNN [7] – we apply a 0.05 confidence threshold to their predictions, as recommended by

##### 3.4. Unified 3D shape and scene understanding

Sec. 3.3 focused on the effectiveness of our structured 3D modality for diverse scene layouts with known complex shapes. Therefore, we used type-specific object descriptions (e.g., “bird” or “person”). Here, we bring together all our findings from Sec. 3.2, including the learned 3D shape tokenization. We show that Kyvo can reconstruct and render complex 3D objects and scenes, using a variant of ObjaWorld with complex objects (e.g., barrel, chicken) from Objaverse [12]. We train recognition and rendering models on 100k image-scene pairs.

The recognition model must (1) reconstruct full 3D geometry for each object, and (2) infer each object’s 3D position and pose from a single image. Fig. 9a shows results on unseen scenes: our model recovers object geometries and

[Figure 47]

[Figure 48]

(a) (b)

- Figure 9. (a) Unified shape and scene reconstruction example. Given a single input image, Kyvo predicts shape sequences and reconstructs individual objects (candle, barrel, bottle) along with their 3D locations and poses via our structured 3D modality, effectively reconstructing the 3D scene with consistent spatial relations between the objects, visualized using Blender. (b) Shape and scene rendering examples. Given the structured 3D modality as input, Kyvo renders images with consistent object appearance and spatial relationships.

Recipe Rendering(↓) Recognition(↑) Instruction(↑) QA(↑) Scratch 1.36 0.6265 0.7744 0.4645 LoRA 1.82 0.8684 0.8680 0.3950 FFT 1.26 0.9212 0.8666 0.4980

Model Objectron ARKitScenes

Backbone Rendering(↓) Recognition(↑) Instruction(↑) QA(↑) Llama-3.2-1B 1.38 0.8948 0.8674 0.4490 Llama-3.2-1B-Instruct 1.28 0.9212 0.8666 0.4980 Llama-3.2-3B-Instruct 1.18 0.8626 0.8763 0.2345

Cube R-CNN (ResNet-34) 0.3276 0.2043 Cube R-CNN (DLA-34) 0.4012 0.2208 Kyvo (Ours) 0.4784 0.2118

(c) Effect of backbone size and instruction-tuning. Table 5. Comparing real-world recognition performance; Quantifying the effect of training recipes and different backbones for Kyvo.

(b) Training recipe: Scratch vs LoRA vs FFT.

(a) Jaccard Acc. vs Cube R-CNN.

[Figure 49]

Figure 10. Qualitative scaling behavior of rendering model.

the authors. Kyvo significantly outperforms Cube R-CNN with two backbone variants on Objectron and performs on par on the more challenging ARKitScenes dataset. This result demonstrates the potential of a general autoregressive framework that aligns images with the 3D modality.

##### 3.6. Analysis and observations

We explore the effects of training strategies and model backbones for Kyvo.

Training recipes. Table 5b compares three approaches for model adaptation: training from scratch, LoRA [17], and full fine-tuning (FFT). FFT from pre-trained language-only weights yields superior performance even when adapting to image and 3D modalities unseen during pre-training – suggesting effective cross-modal transfer with limited domainspecific data. Notably, LoRA performs worse despite its

established efficacy for text-only adaptation, indicating limitations when incorporating entirely new modalities.

Instruction-tuned backbones and model sizes. Instructiontuned backbones match or outperform non-instruction-tuned ones across all tasks, as shown in Table 5c. Increasing model size from 1B to 3B provides no significant gains (even performing notably worse for question-answering), indicating that the 1B model sufficiently captures our dataset’s complexity while avoiding overfitting.

#### 4. Conclusion & Limitations

We introduce Kyvo, a autoregressive model that aligns structured 3D with language and vision to support a broad range of 3D tasks. Our empirical “cookbook”, based on training 307 models, outlines effective design choices, including for tokenizing 3D scene attributes and complex 3D shapes. We will release code and data.

We cover limitations through scaling behavior, qualitative and quantitative results. A key challenge is the limited availability of 3D data. We show that strong performance and within-domain generalization can be achieved with relatively modest training data, but achieving cross-domain generalization demands larger datasets not readily available. A promising direction is extending Kyvo to handle mixed training data, enabling generalization to new domains even when 3D data is not always available as a paired modality.

#### Acknowledgments

We thank Damiano Marsili, Raphi Kang, Ilona Demler, and Ziqi Ma for their valuable feedback. Aadarsh is supported by the Kortschak Scholarship. Georgia is supported by the Powell Foundation, Meta through the LLM evaluation research grant, Google, and Amazon.

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 2
- [2] Adel Ahmadyan, Liangkai Zhang, Artsiom Ablavatski, Jianing Wei, and Matthias Grundmann. Objectron: A large scale dataset of object-centric videos in the wild with pose annotations. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7822–7831,

- 2021. 3, 7

[3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736,

- 2022. 2

- [4] Armen Avetisyan, Christopher Xie, Henry Howard-Jenkins, Tsun-Yi Yang, Samir Aroudj, Suvam Patra, Fuyang Zhang, Duncan Frost, Luke Holland, Campbell Orme, et al. Scenescript: Reconstructing scenes with an autoregressive structured language model. In ECCV, 2024. 2
- [5] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, and Elad Shulman. ARKitscenes - a diverse real-world dataset for 3d indoor scene understanding using mobile RGB-d data. In NeurIPS Datasets and Benchmarks Track (Round 1), 2021. 3, 7
- [6] Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024. 2
- [7] Garrick Brazil, Abhinav Kumar, Julian Straub, Nikhila Ravi, Justin Johnson, and Georgia Gkioxari. Omni3d: A large benchmark and model for 3d object detection in the wild. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13154–13164, 2023. 7, 12
- [8] Yongwei Chen, Yushi Lan, Shangchen Zhou, Tengfei Wang, and Xingang Pan. Sar3d: Autoregressive 3d object generation and understanding via multi-scale 3d vqvae. In CVPR, 2025. 2, 4
- [9] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024. 2

- [10] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision language model. arXiv preprint arXiv:2406.01584, 2024. 2
- [11] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instructionfinetuned language models. Journal of Machine Learning Research, 25(70):1–53, 2024. 2
- [12] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13142–13153, 2023. 3, 6, 7, 12
- [13] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 4, 24
- [14] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, and (additional authors not shown). The llama 3 herd of models,

2024. 2, 3, 4, 13

- [15] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 3
- [16] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. Advances in Neural Information Processing Systems, 2023. 2
- [17] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 8
- [18] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pages 4904–4916. PMLR, 2021. 2
- [19] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2901–2910,

2017. 3, 11

- [20] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. Lerf: Language embedded radiance fields. In ICCV, 2023. 2
- [21] Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models? arXiv preprint arXiv:2405.02246, 2024. 2
- [22] Yuan-Hong Liao, Rafid Mahmood, Sanja Fidler, and David Acuna. Reasoning paths with reference objects elicit quantita-

- tive spatial reasoning in large vision-language models. arXiv preprint arXiv:2409.09788, 2024. 2
- [23] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024. 2
- [24] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, et al. Deepseek-vl: towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525, 2024. 2
- [25] Jiasen Lu, Liangchen Song, Mingze Xu, Byeongjoo Ahn, Yanjun Wang, Chen Chen, Afshin Dehghan, and Yinfei Yang. Atoken: A unified tokenizer for vision, 2025. 2
- [26] Chenyang Ma, Kai Lu, Ta-Ying Cheng, Niki Trigoni, and Andrew Markham. Spatialpin: Enhancing spatial reasoning capabilities of vision-language models through prompting and interacting 3d priors. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 2
- [27] Xianzheng Ma, Yash Bhalgat, Brandon Smart, Shuai Chen, Xinghui Li, Jian Ding, Jindong Gu, Dave Zhenyu Chen, Songyou Peng, Jia-Wang Bian, et al. When llms step into the 3d world: A survey and meta-analysis of 3d tasks via multi-modal large language models. arXiv preprint arXiv:2405.10255, 2024. 2
- [28] Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Liang Zhao, et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. arXiv preprint arXiv:2411.07975, 2024. 2
- [29] Ziqi Ma, Yisong Yue, and Georgia Gkioxari. Find any part in 3d. arXiv preprint arXiv:2411.13550, 2024. 2
- [30] Damiano Marsili, Rohun Agrawal, Yisong Yue, and Georgia Gkioxari. Visual agentic ai for spatial reasoning with a dynamic api. In CVPR, 2025. 2
- [31] Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy Bengio, and Mehrdad Farajtabar. Gsm-symbolic: Understanding the limitations of mathematical reasoning in large language models. arXiv preprint arXiv:2410.05229,

2024. 5

- [32] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 2
- [33] William Shen, Ge Yang, Alan Yu, Jansen Wong, Leslie Pack Kaelbling, and Phillip Isola. Distilled feature fields enable few-shot language-guided manipulation. arXiv preprint arXiv:2308.07931, 2023. 2
- [34] Aaditya K Singh and DJ Strouse. Tokenization counts: the impact of tokenization on arithmetic in frontier llms. arXiv preprint arXiv:2402.14903, 2024. 5
- [35] Yuan Tang, Xu Han, Xianzhi Li, Qiao Yu, Yixue Hao, Long Hu, and Min Chen. Minigpt-3d: Efficiently aligning 3d point clouds with large language models using 2d priors. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 6617–6626, 2024. 2
- [36] Yi Tay, Mostafa Dehghani, Vinh Q Tran, Xavier Garcia, Jason Wei, Xuezhi Wang, Hyung Won Chung, Siamak Shakeri, Dara

- Bahri, Tal Schuster, et al. Ul2: Unifying language learning paradigms. arXiv preprint arXiv:2205.05131, 2022. 2
- [37] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 2
- [38] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 2
- [39] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024. 2
- [40] Shengbang Tong, David Fan, Jiachen Zhu, Yunyang Xiong, Xinlei Chen, Koustuv Sinha, Michael Rabbat, Yann LeCun, Saining Xie, and Zhuang Liu. Metamorph: Multimodal understanding and generation via instruction tuning. arXiv preprint arXiv:2412.14164, 2024. 2
- [41] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 4
- [42] A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems, 2017. 2
- [43] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 2
- [44] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 3
- [45] Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, et al. Deepseek-vl2: Mixture-of-experts vision-language models for advanced multimodal understanding. arXiv preprint arXiv:2412.10302, 2024. 2
- [46] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. In CVPR, 2025. 2, 4, 7, 18, 24
- [47] Runsen Xu, Xiaolong Wang, Tai Wang, Yilun Chen, Jiangmiao Pang, and Dahua Lin. Pointllm: Empowering large language models to understand point clouds. In ECCV, 2024. 2
- [48] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024. 2

### Appendix Contents

- A Dataset details. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11

- A.1 Data generation. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- A.2 3D scene serialization. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- A.3 Tokenization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- A.4 Task sequence formation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

- B Additional qualitative examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- C Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- C.1 3D scenes. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- C.2 Images . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- C.3 3D assets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- C.4 Text . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- D Additional implementation details. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23

- D.1 Image VQGAN architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- D.2 3D VQ-VAE training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- D.3 Encoding of numbers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- D.4 Compute resources and time . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25

- E Additional experiments and observations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- F Failure cases . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

#### A. Dataset details

In this section, we present a detailed overview of the four datasets used in our experiments. We discuss dataset creation, statistics, serialization, tokenization, and task formulation.

##### A.1. Data generation

CLEVR. We generate CLEVR scenes using the dataset creation code from [19] and render the corresponding images with Blender. Each scene outputs a JSON file describing the scene along with its rendered image.

Rendering. We generate 120,000 unique CLEVR scenes as training data for rendering. The JSON files serve as input for the 3D scene representation, while the corresponding images are used to generate output sequences. Further details are provided in the following sections. For evaluation, we use a test set of 2,000 image-JSON pairs.

Recognition. We use the same training data as in rendering but with reversed input-output roles. Here, images serve as input sequences, while JSON files generate the 3D scene output. For evaluation, we use a test set of 2,000 JSONimage pairs.

Instruction-following. We consider four different types of

instructions for 3D scene modification: (1) modifying the appearance of objects, (2) adding new objects, (3) removing objects, and (4) moving an object to a desired location. Using 16 to 28 text instruction templates, we generate 20,000 input-output pairs per instruction type that we build on top of the initial 20,000 CLEVR scenes. Specifically, we sample a CLEVR scene, apply an instruction template, and generate the corresponding modified scene. Additionally, for instructions involving object appearance modification, we generate an extra 20,000 pairs that do not reference other objects, ensuring modifications apply solely to uniquely identifiable objects within the scene. This results in a total of 100,000 input-output pairs forming the training set. For evaluation, we sample 500 input-output pairs per instruction type, creating a test set of 2,500 pairs. This approach enhances dataset diversity, improving the model’s ability to generalize across different instruction types. Example text instructions for each type are listed in Table 6.

Question-answering. For question-answer pair generation, we use the question generation engine by [19] that uses functional programs and generate 20,000 question-answer pairs for the training data. For evaluation, we use a test set of 2,000 question-answer pairs.

ObjaWorld. We extend the CLEVR framework to support

###### Instruction type # pairs Example text instruction

Modifying the appearance of objects (no reference to other objects) 20,000

“Change the gray object to have purple color” “Transform the small yellow rubber sphere to have metal material”

“Change the size of the small purple metal cylinder to the behind of large green rubber sphere to large” “Set the material of the gray metal cube object to the left of small purple rubber cylinder to rubber”

Modifying the appearance of objects 20,000

“Put a small gray rubber cylinder to the left of small yellow rubber sphere” “Insert a red rubber cylinder object to the front of large cyan rubber cube”

Adding new objects 20,000

“Remove the small red rubber cylinder to the right of large yellow rubber cube” “Take out the small rubber sphere object to the right of small gray rubber cylinder”

Removing objects 20,000

“Move the small cyan rubber cylinder object to left and behind” “Change the position of the large rubber sphere object towards right and behind”

Moving an object to a desired location 20,000

Table 6. Instruction templates. Each template type, its pair count, and two representative text instructions.

3D Blender assets from Objaverse [12], for inclusion of objects beyond basic geometric shapes such as cubes, cylinders, and spheres. Specifically, we adapt the CLEVR code to include objects like person, bird, bench, etc..

First, for experiments showing generalization to complex scene layouts (Section 3.3), we consider two scene setups: park and living room. Park scenes are composed of the assets person, bird, bench, and lamppost, while living room scenes use person, sofa, and table to construct the scenes. For training, we generate 50,000 scenes for each setup, resulting in a total of 100,000 scenes. Our test set comprises 4,000 scenes, evenly split between 2,000 park and 2,000 living room scenes.

For experiments showing unified 3D shape and scene understanding (Section 3.4), we select 20 complex Objaverse objects like barrel, chicken, cheeseburger, etc., and generate 100,000 training scenes containing 2-3 randomly sampled objects. The test set contains 1,000 unseen scenes.

Objectron. We adhere to the official dataset splits by [7] to construct our training and test sets. For each object in a scene, we extract the category name, 3D center camera coordinates, and object dimensions from the annotation files, generating image–3D scene pairs.

ARKitScenes. Similar to Objectron, we follow [7] and use the provided dataset splits and extract object category labels, 3D center coordinates, and dimensions from annotations to generate image-3D scene pairs.

Objaverse. For training the 3D VQ-VAE, we use ∼ 168k Objaverse assets from the Sketchfab subset. However, the computational cost of extracting slats from assets is high using the original TRELLIS pipeline, as it requires extracting 150 renders of the assets which are then passed through the DINOv2 encoder. Instead, we extract a single render of each asset only, and pass it as image-conditioning to the pretrained TRELLIS slat generator. We use the resultant slats as the inputs to the 3D VQ-VAE encoder during training. We found that the slats generated via this synthetic pipeline are relatively faithful to the original asset. Further, the synthetic slats are effective substitutes in the training data as during

evaluation, we use slats extracted from the original TRELLIS pipeline and find that performance transfers well, indicating the distributions of slats are similar with the two methods.

##### A.2. 3D scene serialization

As described above, each scene consists of an image paired with a structured 3D scene representation in JSON format. Before tokenization, we preprocess these JSON files by parsing and converting them into a single string representation. Specifically, we extract relevant attributes from the JSON and structure them using special markers like [SHAPE], [LOCATION], etc., which vary depending on the dataset. These special markers are registered as special tokens in the tokenizer, which we discuss in the next section. For instance, in ObjaWorld, when we encode the explicit geometry, 512 tokens fetched from the 3D VQ-VAE codebook follow the [SHAPE] marker. We provide examples of serialized outputs for each dataset below.

CLEVR:

[SCENE-START][OBJECT-START][SIZE]large [COLOR]cyan[MATERIAL]metal[SHAPE]cube [LOCATION]-0.55 0.05 0.70 [OBJECT-END][OBJECT-START][SIZE]small [COLOR]yellow[MATERIAL]metal[SHAPE]cylinder

- [LOCATION]1.25 2.50 0.35 [OBJECT-END][SCENE-END]

ObjaWorld:

[SCENE-START][OBJECT-START][SHAPE]table [LOCATION]-2.70 -2.20 0.20[POSE]0.00 0.00

- -0.10 [OBJECT-END][OBJECT-START][SHAPE]person [LOCATION]-0.20 -0.70 0.85[POSE]0.00 0.00

- 0.55[OBJECT-END][OBJECT-START][SHAPE]person [LOCATION]-0.75 -2.80 0.85[POSE]0.00 0.00

-2.55[OBJECT-END][OBJECT-START][SHAPE]table [LOCATION]2.75 1.90 0.20[POSE]0.00 0.00

- 1.95[OBJECT-END][OBJECT-START][SHAPE]sofa [LOCATION]0.40 2.75 0.30[POSE]0.00 0.00

-0.95[OBJECT-END][SCENE-END]

ObjaWorld (with explicit shape representations):

[SCENE-START][OBJECT-START][SHAPE]<v11, v21,...,v5121 >[LOCATION]-2.45 0.60 1.20 [POSE]0.00 0.00 2.30 [OBJECT-END][OBJECT-START][SHAPE]<v12,v22,..., v5122 >[LOCATION]2.50 -1.35 0.00[POSE]0.00 0.00 1.55[OBJECT-END][OBJECT-START][SHAPE]< v13,v23,...,v5123 >[LOCATION]-1.80 -0.90 0.00 [POSE]0.00 0.00 1.45[OBJECT-END][SCENE-END]

Objectron:

[SCENE-START][OBJECT-START][CATEGORY]bicycle [CENTER_CAM]0.00 -0.10 2.45[DIMENSIONS]0.60

1.10 1.00[OBJECT-END][SCENE-END]

ARKitScenes:

[SCENE-START][OBJECT-START][CATEGORY]sofa [CENTER_CAM]-0.14 0.04 1.50[DIMENSIONS]1.70

0.80 0.90 [OBJECT-END][OBJECT-START][CATEGORY]table [CENTER_CAM]0.02 0.08 1.60[DIMENSIONS]0.50 0.40 0.50 [OBJECT-END][OBJECT-START][CATEGORY]table [CENTER_CAM]-0.30 0.22 0.00[DIMENSIONS]1.30

0.80 0.40

[OBJECT-END][OBJECT-START][CATEGORY]cabinet [CENTER_CAM]0.46 -0.02 0.00[DIMENSIONS]0.70

0.80 0.30 [OBJECT-END][OBJECT-START][CATEGORY]cabinet [CENTER_CAM]0.40 0.24 0.00[DIMENSIONS]1.90 0.90 0.70[OBJECT-END][SCENE-END]

##### A.3. Tokenization

Text. We employ an off-the-shelf text tokenizer from Llama3.2 [14] with a vocabulary size of 128,000 for tokenization. Images. For image tokenization, we train a domain-specific VQGAN on the training set of each dataset. This model encodes images into discrete representations by mapping them to codebook indices, which are then used for downstream processing.

3D Scenes. To enable tokenization of 3D scene representations, we augment the vocabulary of the Llama-3.2 tokenizer with two additional token types: (1) special tokens that serve as markers for scene attributes and (2) numerical tokens, ensuring that each location coordinate is encoded as a distinct token.

3D Shapes. For encoding the explicit shape geometries for the assets, we train a 3D VQ-VAE for tokenization as discussed in Section 3.2.1 of the main paper. We provide more details on training the 3D VQ-VAE in Section D.2 below.

##### A.4. Task sequence formation

After tokenizing all three modalities, we construct sequences tailored to the specific task, which are then used to train

Kyvo. In this section, we present examples of complete sequences for each of the tasks across the four datasets. CLEVR:

Rendering: 3D → Image

[BOS][SCENE-START][OBJECT-START][SIZE]small [COLOR]green[MATERIAL]metal[SHAPE]sphere [LOCATION]0.85 1.85 0.35 [OBJECT-END][OBJECT-START][SIZE]small [COLOR]green[MATERIAL]metal[SHAPE]sphere [LOCATION]0.80 -2.00 0.35 [OBJECT-END][OBJECT-START][SIZE]large [COLOR]brown[MATERIAL]metal[SHAPE]cylinder

- [LOCATION]-1.35 2.65 0.70 [OBJECT-END][OBJECT-START][SIZE]small [COLOR]purple[MATERIAL]rubber[SHAPE]sphere [LOCATION]-0.90 -2.20 0.35 [OBJECT-END][OBJECT-START][SIZE]large [COLOR]red[MATERIAL]rubber[SHAPE]cylinder

- [LOCATION]-2.70 -2.90 0.70 [OBJECT-END][OBJECT-START][SIZE]large [COLOR]red[MATERIAL]metal[SHAPE]cylinder [LOCATION]2.25 -1.15 0.70 [OBJECT-END][OBJECT-START][SIZE]small [COLOR]red[MATERIAL]metal[SHAPE]cube [LOCATION]2.15 -2.65 0.35 [OBJECT-END][SCENE-END][OUTPUT-SEP] [IMAGE-START]<image-tokens>[IMAGE-END][EOS]

Recognition: Image → 3D

[BOS][IMAGE-START]<image-tokens> [IMAGE-END][OUTPUT-SEP][SCENE-START] [OBJECT-START][SIZE]small[COLOR]brown [MATERIAL]rubber[SHAPE]sphere [LOCATION]-2.50 0.30 0.35 [OBJECT-END][OBJECT-START][SIZE]large [COLOR]blue[MATERIAL]metal[SHAPE]cylinder [LOCATION]2.95 1.60 0.70 [OBJECT-END][OBJECT-START][SIZE]large [COLOR]red[MATERIAL]metal[SHAPE]cylinder [LOCATION]-0.85 0.60 0.70 [OBJECT-END][OBJECT-START][SIZE]small [COLOR]green[MATERIAL]metal[SHAPE]sphere [LOCATION]0.85 1.85 0.35 [OBJECT-END][SCENE-END][EOS]

Instruction-Following: (Image,3D,TextI) → (Image,3D)

[BOS][IMAGE-START]<image-tokens> [IMAGE-END][SCENE-START][OBJECT-START][SIZE] small[COLOR]brown[MATERIAL]rubber [SHAPE]sphere[LOCATION]-2.50 0.30 0.35 [OBJECT-END][OBJECT-START][SIZE]large [COLOR]blue[MATERIAL]metal[SHAPE]cylinder [LOCATION]2.95 1.60 0.70 [OBJECT-END][OBJECT-START][SIZE]large [COLOR]red[MATERIAL]metal[SHAPE]cylinder [LOCATION]-0.85 0.60 0.70

[OBJECT-END][OBJECT-START][SIZE]small [COLOR]green[MATERIAL]metal[SHAPE]sphere [LOCATION]0.85 1.85 0.35 [OBJECT-END][SCENE-END][TEXT-START]Change the brown object to have purple color [TEXT-END] [OUTPUT-SEP][IMAGE-START]<image-tokens> [IMAGE-END][SCENE-START][OBJECT-START][SIZE] small[COLOR]purple[MATERIAL]rubber [SHAPE]sphere[LOCATION]-2.50 0.30 0.35 [OBJECT-END][OBJECT-START][SIZE]large [COLOR]blue[MATERIAL]metal[SHAPE]cylinder [LOCATION]2.95 1.60 0.70 [OBJECT-END][OBJECT-START][SIZE]large [COLOR]red[MATERIAL]metal[SHAPE]cylinder [LOCATION]-0.85 0.60 0.70 [OBJECT-END][OBJECT-START][SIZE]small [COLOR]green[MATERIAL]metal[SHAPE]sphere [LOCATION]0.85 1.85 0.35 [OBJECT-END][SCENE-END][EOS]

Question-Answering: (Image,3D,TextQ) → TextA

[BOS][IMAGE-START]<image-tokens> [IMAGE-END][SCENE-START][OBJECT-START][SIZE] small[COLOR]brown[MATERIAL]rubber [SHAPE]sphere[LOCATION]-2.50 0.30 0.35 [OBJECT-END][OBJECT-START][SIZE]large [COLOR]blue[MATERIAL]metal[SHAPE]cylinder [LOCATION]2.95 1.60 0.70 [OBJECT-END][OBJECT-START][SIZE]large [COLOR]red[MATERIAL]metal[SHAPE]cylinder [LOCATION]-0.85 0.60 0.70 [OBJECT-END][OBJECT-START][SIZE]small [COLOR]green[MATERIAL]metal[SHAPE]sphere [LOCATION]0.85 1.85 0.35 [OBJECT-END][SCENE-END][TEXT-START]What size is the rubber sphere? [TEXT-END][OUTPUT-SEP][TEXT-START]small [TEXT-END][EOS]

Similar structure is followed for ObjaWorld (Rendering and Recognition), Objectron (Recognition), and ARKitScenes (Recognition). For ObjaWorld, when we encode the explicit shape representations, the complete sequences look like the following.

ObjaWorld: (with explicit shape representations)

Rendering: 3D → Image

[BOS][SCENE-START][OBJECT-START][SHAPE]<v11, v21,...,v5121 >[LOCATION]2.10 0.15 -0.75 [POSE]0.00 0.00 -2.10 [OBJECT-END][OBJECT-START][SHAPE]<v12,v22,..., v5122 >[LOCATION]-1.25 2.85 0.05[POSE]0.00 0.00 1.85[OBJECT-END][OBJECT-START][SHAPE]< v13,v23,...,v5123 >[LOCATION]0.45 -2.35 -1.50 [POSE]0.00 0.00 0.60 [OBJECT-END][SCENE-END][OUTPUT-SEP] [IMAGE-START]<image-tokens>[IMAGE-END][EOS]

Recognition: Image → 3D

[BOS][IMAGE-START]<image-tokens> [IMAGE-END][OUTPUT-SEP][SCENE-START] [OBJECT-START][SHAPE]<v11,v21,...,v5121 > [LOCATION]1.15 -2.85 0.40[POSE]0.00 0.00

-1.75[OBJECT-END][OBJECT-START][SHAPE]<v12, v22,...,v5122 >[LOCATION]-0.35 2.10 -0.50 [POSE]0.00 0.00 2.45 [OBJECT-END][OBJECT-START][SHAPE]<v13,v23,..., v5123 >[LOCATION]2.75 0.15 -1.60[POSE]0.00 0.00 -0.25[OBJECT-END][SCENE-END][EOS]

#### B. Additional qualitative examples

In this section, we provide additional qualitative examples.

3D Tokenization. We present additional qualitative results for our 3D tokenization scheme. Fig. 11 provides additional examples of how the 3D tokens are effective for reconstructions, as shown in Figure 4 (b) in the paper. Fig. 12 provides additional examples of how the 3D tokens are effective for decoding, as shown in Figure 4 (c) in the paper.

Rendering. We present qualitative results for the rendering task on CLEVR and ObjaWorld with complex shapes in Figures 13 and 14, respectively. Results on ObjaWorld with explicit shape representations are shown in Figure 15. The model takes only the structured 3D scene representation as input and predicts the corresponding image tokens. These tokens are then decoded using the VQGAN decoder, which reconstructs the image by mapping them from tokenspace to pixel-space. For each example, we also provide the ground truth image rendered using Blender, allowing direct comparison with the model-generated output. As observed, the model effectively captures the 3D scene structure based solely on the JSON input and accurately synthesizes the corresponding image. Notably, Figure 15 demonstrates the model’s ability to integrate multiple information sources: it successfully maps spatial arrangements from JSON scene descriptions and visual properties from asset sequences to generate coherent, realistic pixel-space representations.

However, certain failure cases highlight the model’s limitations. For instance, in the first example of Figure 13, the model fails to predict a small red cube positioned at the front. Similarly, in the last example of Figure 14, the model mispredicts the bird’s pose, causing it to face the wrong direction. In the third column of comparisons in Figure 15, some distortions in the shapes and poses can be seen as well. Despite these occasional errors, the overall rendering quality demonstrates strong spatial understanding and scene reconstruction capabilities.

Recognition. In Figure 16 we show example recognition results on the ObjaWorld dataset with complex shapes. The model takes the image (tokenized) as input and outputs the sequential 3D representation. We then parse the predicted

## GT Kyvo 3D Tokenizer GT Kyvo 3D Tokenizer

[Figure 50]

- Figure 11. Reconstruction examples for Trellis-based 3D VQ-VAE. VQ-VAE reconstructions, shown from multiple views, compared to ground truth for unseen Objaverse assets.

Kyvo

Kyvo

GT GT

GT Render GT Render

3D Reconstructions

3D Reconstructions

[Figure 51]

[Figure 52]

(a)

(b)

- Figure 12. Decoding examples for Trellis-based 3D VQ-VAE on unseen Objaverse assets. (a) Reconstruction of 3D shape encoding using Llama 3.2 as decoder. (b) Rendering from 3D shape encoding using Llama 3.2 as decoder.

3D scene into JSON format and display it alongside the ground truth JSON in the figure. To evaluate recognition performance, we match the predicted objects with ground truth objects to compute the Jaccard Index (see Algorithm 1). This matching is based on attribute similarity and spatial location

criteria. Additional details on the Jaccard Index calculation are provided in Section C. The matching results, visualized using colored numbers in Figure 16, illustrate the model’s strong ability to accurately predict object attributes and their 3D spatial locations almost accurately. While minor spatial

[Figure 53]

- Figure 13. Rendering examples for CLEVR. Example image generations for the rendering task on CLEVR. The model takes a 3D scene as input and produces a corresponding image. Additionally, we show the ground-truth image rendered using Blender.

[Figure 54]

- Figure 14. Rendering examples for ObjaWorld. Example image generations for the rendering task on ObjaWorld with complex shapes. The model takes a 3D scene as input and produces a corresponding image. Additionally, we show the ground-truth image rendered using Blender.

deviations may occur, the model effectively reconstructs the structured 3D scene from image input.

In Figure 17 we show examples of recognition task on ObjaWorld with explicit shape representations on unseen

[Figure 55]

- Figure 15. Rendering examples for ObjaWorld with explicit shape encodings. Example image generations for the rendering task on ObjaWorld. The model takes a 3D scene with embedded shape encodings as input and produces a corresponding image. Additionally, we show the ground-truth image rendered using Blender.

scenes. Specifically, the model takes a single images as input and reconstructs the full 3D geometry per object and infers each object’s 3D position and pose to accurately reconstruct the 3D scene. As can be seen from Figure 17, Kyvo accurately recovers object geometries and spatial layouts via our structured, object-centric 3D modality. In contrast, Trellis [46] often merges two objects into one (e.g. the second example) or hallucinates shapes (e.g., blue can in the first example) and misaligned layouts.

Question-answering. In Figure 18, we present qualitative examples from the question-answering task on the CLEVR dataset. The figure showcases model predictions across a diverse set of question types, including binary (True/False) responses, categorical answers such as object sizes (“small” or “large”), and numerical values. These examples highlight the model’s ability to understand and reason about structured 3D scenes, demonstrating accurate comprehension of spatial relationships, object attributes, and numerical reasoning.

#### C. Evaluation

In this section we provide more details on the evaluation strategies that we adopt. We discuss the evaluation method for the three primary modalities as well as for the 3D shape encodings.

##### C.1. 3D scenes

We evaluate predicted 3D scenes using the Jaccard Index, computed via Algorithm 1. Objects are matched between predicted and ground-truth scenes based on attribute similarity and spatial proximity.

Matching criteria by dataset:

- • CLEVR: Match shape, size, color, and material
- • ObjaWorld: Match shape with pose constraint (predicted pose within ±0.15 radians)
- • Objectron: Match category with dimension constraint

- (mean absolute error ≤ 0.05)

• ARKitScenes: Match category with dimension constraint

- (mean absolute error ≤ 1.00) Spatial proximity thresholds (τ): We average Jaccard

Index across multiple threshold values:

- • CLEVR, ObjaWorld, Objectron: τ ∈ {0.05,0.10,0.15,0.20,0.25}
- • ARKitScenes: τ ∈ {1.25,1.50,1.75,2.00,2.25} Lower τ values impose stricter spatial constraints, requir-

ing predicted objects to be closer to ground-truth positions. Figure 19 illustrates how τ affects Jaccard Index on CLEVR across different training data sizes.

Comparison with Trellis. We compare our unified shape and scene reconstruction against Trellis (Figure 9, main pa-

[Figure 56]

- Figure 16. Recognition examples for ObjaWorld. Two example predictions from the recognition task on ObjaWorld. The colored numbers indicate object matching between the predicted and ground-truth scenes, based on the criteria for Jaccard Index as defined in Algorithm 1. Note that the fourth number in the list is the azimuth pose value, this format of prediction saves sequence length.

per; Figure 17). Trellis reconstructs scenes by inputting an image to a rectified flow transformer (DiT) that generates a scene-level SLAT representation, which is subsequently decoded into a single holistic 3DGS. In contrast, Kyvo employs a different approach through two key distinctions: (1) Scene decomposition: Kyvo decomposes scenes into constituent objects, each parameterized by shape, 3D location, and orientation within our structured 3D modality. (2) Quantized representation: While both methods utilize SLAT representations for object shapes, Kyvo vectorquantizes these representations via our 3D VQ-VAE, to slot naturally into Kyvo’s autoregressive generation framework. This decomposition enables Kyvo to achieve precise reconstruction of individual objects while simultaneously inferring their 3D spatial locations and relationships. Moreover, we obtained an average Jaccard Index of 0.666 averaged over τ ∈ {0.50,0.75,1.00,1.25,1.50} for this recognition model. We observed that the model seems to have a relatively harder time predicting the location coordinates when explicit shape sequences are involved as compared to when the shapes are identified using a word token like in CLEVR.

##### C.2. Images

Human Evaluation. As discussed in the main paper, we rely on human evaluations to assess image generations by the model, as object-level nuances often go beyond the scope of quantitative metrics. To facilitate this evaluation, we designed a user interface, a snapshot of which is shown in Figure 20. For each comparison involving N models,

the interface presents users with N generated images alongside the ground truth image, all displayed in a shuffled and anonymized order. This ensures that users remain unaware of which model generated each image, mitigating potential biases in evaluation. In each comparison, users are asked to assign both a score and a rank to every generated image based on its visual fidelity and alignment with the ground truth. Figure 20 shows a snapshot of the evaluation of images from the experiment where we studied the effect of center-token reordering and weighted loss on model generation involving four models (results reported in Table 4 of the main paper).

Score: The score takes a binary value of “0” or “1” and signifies the complete correctness of an image generation. The user is asked to provide a score of “1” only if the user believes that all the objects in the scene were accurately generated and accurately placed spatially. If the generated image has any differences with the groundtruth, e.g. a cube was not generated correctly, the user provides a score of “0". The score value is independent of any other model involved in the comparison and solely depends on the model under consideration and the groundtruth.

Rank: The rank takes a value from {“1”, “2”, ..., “N”}. The user is expected to rank the N images from “1" to “N" by comparing the generation quality among them. If the user is unable to decide between two images, then we allow equal rank assignment to more than one image, e.g. if two models equally perform for a given scene, they get the same rank value.

[Figure 57]

[Figure 58]

- Figure 17. Unified shape and scene reconstruction examples. Given a single input image, Kyvo predicts shape sequences and reconstructs individual objects (bottle, cheeseburger, etc.) along with their 3D locations and poses via our structured 3D modality, effectively reconstructing the 3D scene with consistent spatial relations between the objects, visualized using Blender.

We consider a test set of 50 scenes and use the user interface for scoring and ranking the generations. Specifically, let’s say we want to have an evaluation on the effect of granularity, then we consider the image generated by the models for the 50 scenes and score and rank them as discussed above. In the main paper, we report the mean rank (the lower, the better) over the 50 images in the tables. In addition to the mean rank, we obtain the mean score (the higher, the better)

for every model. We also compute the winning rate (the higher, the better) that is defined as the fraction of times a given model was assigned a rank of 1. We report the mean score and winning rate for all the models in Table 7.

SSIM and L2-loss. Evaluation of the generated images requires careful assessment of the objects and their attributes in the scene. For example, consider the example predictions in Figure 21. For both cases, the predicted image is incorrect

[Figure 59]

- Figure 18. Question-answering examples for CLEVR. Example cases from the question-answering task on CLEVR. The model takes an image, a 3D scene, and a question as input to generate the corresponding answer.

[Figure 60]

- Figure 19. Effect of τ. The plot shows the impact of τ on the Jaccard Index for models trained with increasing amounts of training data on CLEVR for the recognition task. The drop in Jaccard Index with decreasing τ is more pronounced for models trained on smaller datasets. Higher-performing models demonstrate greater robustness to changes in τ.

and minutely differs from the groundtruth. For the first case, the small cyan sphere gets an incorrect gray color, while in the second case, the small cube gets mispredicted as a cylinder. Quantitative metrics like SSIM and L2-loss fail to capture these subtle differences between the predicted and the groundtruth images that occupy a very small pixel region, leading us to qualitative human evaluations. However, for

experimental completeness, we computed the SSIM and pixel-wise L2-loss for all the models and reported them in Table 7. While the values show similar trends to human evals, we report human evals in the main paper as they directly assess image correctness.

Algorithm 1 Compute Jaccard Index

- 1: Input: GTS (list of ground-truth scenes), PS (list of predicted scenes), distance threshold τ
- 2: Output: Average JaccardIndex
- 3: Initialize JaccardIndex = 0
- 4: for (G, P) in zip(GTS, PS) do
- 5: GObjs ← G.objects
- 6: PObjs ← P.objects
- 7: matchedFlags ← Boolean array of length |GObjs|, initialized to False
- 8: Initialize TP = 0, FP = 0, FN = 0
- 9: for p in PObjs do
- 10: foundMatch ← False
- 11: for j = 1 to |GObjs| do
- 12: if ¬matchedFlags[j] and attributesMatch(p,GObjs[j]) and dist(p.coords,GObjs[j].coords) < τ then
- 13: matchedFlags[j] ← True
- 14: foundMatch ← True
- 15: TP ← TP + 1
- 16: break
- 17: end if
- 18: end for
- 19: if foundMatch = False then
- 20: FP ← FP + 1
- 21: end if
- 22: end for
- 23: FN ← FN + count(matchedFlags = False)
- 24: JaccardIndex+ = TP+TPFP+FN

- 25: end for
- 26: JaccardIndexAvg ← JaccardIndex/|PS|
- 27: return JaccardIndexAvg

[Figure 61]

- Figure 20. Human Evaluation User Interface. A snapshot of the user interface used for human evaluation of generated images. This example is taken from an experiment analyzing the effect of center-token reordering and weighted loss, comparing four models. The results of this experiment are presented in Table 2 of the main paper.

##### C.3. 3D assets

For the same reasons as above, we use human evaluations to assess the quality of 3D assets when it comes to the 3D

tokenizer. In Table 1 in the main paper, we use human evaluations to quantify the effect of auxiliary reconstruction losses on the reconstruction quality of the 3D VQ-VAE, and report mean rank. In Table 2 in the main paper, we use human evaluations to compare the reconstruction quality of our Trellis-based 3D VQ-VAE with that of SAR3D, and report mean rank. In both cases, the human evaluation is facilitated using an anonymized, shuffled interface as described above, and only relative rank is assessed and reported.

##### C.4. Text

Among the four tasks we consider, only question-answering produces text output. The question templates used in CLEVR cover a diverse range of answer types. Some questions require binary (True/False) responses, others expect numerical values ranging from 0 to 10, while some answers involve text words describing attributes like “small”,“green”, “metal”, etc.. Table 8 provides a breakdown of the different question types in the training set.

Using these distributions, we establish two baseline accuracies on the test set: random and frequency. For the random baseline, we predict answers uniformly at random for each

[Figure 62]

- Figure 21. Object-level nuances are challenging for image metrics, like SSIM and L2-loss, to capture, prompting the need for human evaluation. The predicted image is incorrect in both cases but differs only subtly from the groundtruth.

|Comparison|Metrics<br><br>| |
|---|---|---|
|Granularity|Mean Rank (↓) Winning Rate (↑) Mean Score (↑)|SSIM (↑) L2-loss (↓)|
|0.005 0.05 0.5|1.38 0.70 0.74<br><br>1.20 0.82 0.80<br><br>2.02 0.28 0.52<br>|0.9527 0.0021 0.9505 0.0010 0.9030 0.0010<br><br>|
|Number Encoding|Mean Rank (↓) Winning Rate (↑) Mean Score (↑)<br><br>|SSIM (↑) L2-loss (↓)|
|Fixed Sine-Cosine Learned Fixed Sine-Cosine + Learned|1.44 0.64 0.64 1.58 0.60 0.60 1.28 0.78 0.78<br><br>|0.9525 0.0010 0.9487 0.0011 0.9505 0.0010<br><br>|
|CT Reordering, Weighted Loss|Mean Rank (↓) Winning Rate (↑) Mean Score (↑)|SSIM (↑) L2-loss (↓)<br><br>|
|✗, ✗<br><br>✓, ✗<br><br>✗, ✓<br><br><br>✓, ✓|2.66 0.00 0.00<br><br>3.56 0.00 0.00<br><br><br>2.78 0.00 0.00 1.00 1.00 0.84<br><br>|0.8575 0.0049 0.8410 0.0061 0.8668 0.0044 0.9505 0.0010<br><br>|
|Recipe<br><br>|Mean Rank (↓) Winning Rate (↑) Mean Score (↑)|SSIM (↑) L2-loss (↓)|
|Scratch LoRA FFT|1.36 0.68 0.72 1.82 0.48 0.56 1.26 0.8 0.82<br><br>|0.9515 0.0010 0.9507 0.0010 0.9505 0.0010|
|Backbone|Mean Rank (↓) Winning Rate (↑) Mean Score (↑)|SSIM (↑) L2-loss (↓)<br><br>|
|Llama-3.2-1B Llama-3.2-1B-Instruct Llama-3.2-3B-Instruct|1.38 0.76 0.78 1.28 0.72 0.80 1.18 0.84 0.86<br><br>|0.9521 0.0010 0.9505 0.0010 0.9539 0.0010<br><br>|

Table 7. Image Metrics. Comparison of mean rank, winning rate, and mean score from human evaluation across all models for the CLEVR rendering task. Additionally, we provide SSIM and pixel-wise L2 loss values for each model.

question type, yielding a mean accuracy of 0.359 with a standard deviation of 0.009 over 100 runs. For the frequency baseline, we predict the most common answer for each question type based on its distribution in the training data (as shown in Table 8), achieving a test accuracy of 0.430.

#### D. Additional implementation details

##### D.1. Image VQGAN architecture

In this section, we provide additional implementation details for the VQGAN architecture employed in our experiments to represent images. Figure 22 illustrates the complete network

Question type # Questions Majority answer

True/False 8,080 False Number (0–10) 4,401 0 Shape 1,840 cylinder Color 1,919 cyan Material 1,840 metal Size 1,920 small

###### Baseline accuracy (test set)

Random (100 runs) 0.359 ± 0.009 Frequency 0.430

Table 8. Question-Answering Data. Statistics of various question types in the training dataset and baseline accuracies on the test set.

architecture, with output shapes shown for each layer given an input of shape (1, 3, 256, 256). Our VQGAN config-

uration uses a 1024-entry codebook with 256-dimensional embeddings, trained on 256×256 resolution images. The encoder produces quantized embeddings of shape (1, 256, 16, 16), yielding 16 × 16 = 256 discrete tokens per image that correspond to learned codebook entries. For each dataset, we initialize the VQGAN model with ImageNet pretrained weights from [13], then fine-tune on the respective training set for 100 epochs. This fine-tuned model serves as both the encoder for converting images to token sequences and the decoder for reconstructing images from predicted tokens during evaluation.

##### D.2. 3D VQ-VAE training

In Section 3.2.1 of the main paper we describe how we train a 3D VQ-VAE to compress the 643 × 8 slats to a dense 83 × 128 latent. Here, we provide some additional details on the training. A slat contains ∼ 20k sparse voxels (in a 643 grid, with vectors of dimension 8 at active positions). We train the 3D VQ-VAE with these slats in 3 steps: (i) it densifies the sparse tensor onto a 643× 8 grid, (ii) encodes that grid with a 3-D convolutional U-Net down to an 83×128 latent volume, and (iii) vector-quantizes every latent with an 8192-entry codebook.

Architecture. The original Trellis SLAT encoder and decoder remain frozen and provide supervision. Sparse latents (zi,pi) are first rasterised into a zero-initialised 643×8 grid; using a learnable padding vector instead of zeros showed no measurable gain and is therefore omitted. The dense grid is processed by a 3D U-Net that downsamples as 643 → 323 → 163 → 83 with channel widths (32,128,512,1024). Each 83 cell outputs a 128-dim vector, quantized to the nearest of 8192 code vectors updated by exponential moving average (τ=0.99). Gradients flow to the encoder through the straight-through estimator.

Optimization. We train for 200k steps on ∼ 168k Sketchfab assets from Objaverse (further details provided in Sec. A.1) with a batch size of 8. We use an AdamW optimizer (β1=0.9,β2=0.999) with a constant learning-rate of 3×10−4, no weight decay, mixed precision, and adaptive gradient clipping.

The training objective combines four terms:

###### L = ∥x − xˆ∥22

###### +β ∥z − sg[e]∥22

dense-SLAT recon

commit

+ λKL DKL + γ Lrender

(1)

3×8 is the rasterised SLAT voxel, xˆ its VQ–VAE reconstruction, sg[·] denotes the stop-gradient operator. In the commitment term, z ∈ R8

where x ∈ R64

3×128 denotes the

pre-quantization output of the encoder, while e is the corresponding vector-quantized output selected from the codebook. The Lrender follows TRELLIS:

Lrender = L1(I,Iˆ)+0.2 1 − SSIM(I,Iˆ) +0.2LPIPS(I,Iˆ),

computed between images rendered from ground-truth images I and rendering from Gaussian reconstructions Iˆ. This is the auxiliary pixel-space reconstruction loss discussed in the main paper. We set β = 0.25, λKL = 10−6, γ = 0.1.

The resulting codebook is used to represent objects using 83 = 512 discrete tokens, enabling unified processing with structured 3D representations for autoregressive modeling. We employ bi-directional attention within shape token sequences for full intra-sequence connectivity when training the LLM.

Codebook usage. Figure 23 shows codebook utilization across 5000 training assets (log-scale, decreasing order). The heavy-tailed distribution indicates effective learning: frequent codes capture common 3D primitives while rare codes encode geometric variations. Active utilization of the whole codebook indicates it is not over-parameterised. Additionally, increasing codebook size did not help empirically indicating it is not under-parameterised.

The full training script and code for 3D VQ-VAE are included in the code.zip file. The code was built on top of Trellis [46] code.

##### D.3. Encoding of numbers

In Section 3.2.2 (Figure 6), we showed the robustness of a hybrid approach that we used for encoding numbers, where the embeddings are learned but are augmented with sinecosine encodings. Here, we provide more details on how we implement the encodings. Specifically, if we have N numbers to encode, then the nth number is encoded using an embedding of dimension d. First, we obtain the sine-cosine encoding as follows:

NE(pos,2i) = sin

NE(pos,2i+1) = cos

pos 100002di

pos 100002di

, (1)

(2)

where i indexes the embedding dimensions. On top of this, we incorporate a learned embedding layer, implemented as nn.Embedding in PyTorch, of the same dimensionality d. The final representation for each number is obtained by summing its learned embedding with the corresponding static sine-cosine encoding. This hybrid approach allows the model to leverage both the flexibility of learned embeddings and the structured inductive bias introduced by the sinecosine encoding for numbers.

[Figure 63]

- Figure 22. VQGAN Architecture. We show the detailed architecture of the VQGAN model that we use to train a domain-specific codebook. We show the output shapes for an input shape (1, 3, 256, 256).

[Figure 64]

- Figure 23. Codebook usage. Histogram of usage counts (log scale) for the 8192 latent codes on 5000 assets used for training. The heavytailed distribution reveals a compact core of frequently reused codes that capture ubiquitous 3D primitives, while the long tail represents rarer geometric variations. In total, all of the vocabulary is exercised at least once, indicating that the codebook is not over-parameterised. Additionally, increasing codebook size did not help empirically indicating it is not under-parameterised.

##### D.4. Compute resources and time

Experiments were executed on a single Ubuntu 22.04.5 server equipped with 8×NVIDIA A100–SXM4 GPUs (80

GB each, driver 570.124, CUDA 12.8), dual-socket AMD EPYC 7763 CPUs (256 threads) and 2 TB RAM. The software stack comprised Python 3.11 and PyTorch 2.4.1 built

against CUDA 12.1 and cuDNN 9.1. Each model was trained on a single GPU, with an average training throughput of ∼ 8,800 tokens sec−1 per GPU for all language model training. The LLM experiments were implemented using the torchtune PyTorch library, providing a unified framework for fine-tuning language models. We employed PagedAdamW8bit optimizer with learning rate 1 × 10−4, batch size 32, and trained for 10 epochs using bfloat16 precision and used the cross-entropy loss function.

The code and configs are included in the code.zip file. The code was built on top of the torchtune code.

#### E. Additional experiments and observations

Failure of modern-day LLMs. To demonstrate the complexity of our 3D tasks, we evaluated several state-of-the-art large language models on simple CLEVR scenes. While CLEVR scenes appear visually simple, the tasks they address are complex. For instance, Google’s Gemini-Pro and OpenAI’s latest GPT4o, when prompted to generate an image with CLEVR’s 3D scene specifications, produce wrong images – failing to adhere to relative object positions and often hallucinating new objects (see Figure 24). For the recognition task, Meta’s state-of-the-art VLM, Llama3.2Vision (from the same family of backbones as ours), achieves near-zero performance. Figure 25 visualizes Llama3.2Vision’s predictions rendered in Blender alongside ground truth scenes and the predictions from Kyvo recognition model. As can be observed, it fails to accurately predict xyz coordinates, despite the simplicity of CLEVR scenes and its objects. These failures demonstrate that 3D spatial reasoning remains a significant challenge for internet-scale trained models, even within controlled synthetic environments, underscoring the difficulty of the tasks we address.

Number Encoding Rendering(↓) Recognition(↑) Instruction(↑) QA(↑) Fixed Sine–Cosine 1.44 0.9229 0.8678 0.4845 Learned 1.58 0.9185 0.8572 0.4680 Fixed Sine–Cosine + Learned 1.28 0.9212 0.8666 0.4980

Table 9. Encoding strategies for numbers. Performance of each strategy across the four tasks.

Number encodings. As detailed in Section 3.2.2 of the main paper, we employ a hybrid approach for number encoding that combines learned representations with sine-cosine positional encodings. We demonstrated the robustness of this hybrid encoding strategy across varying data regimes in Figure 6 of the main paper, with implementation details provided in Section D.3. While Figure 6 assessed the performance of these strategies across data regimes, Table 9 summarizes the performance of these encoding strategies across all four tasks for the largest training set size.

Embedding dimension projector. We use a projector to

embed the VQGAN codes and the 3D VQ-VAE codes into the same dimensional space as the Llama embeddings, creating a unified sequence of dimensions. A simple linear layer without biases proves sufficient for this task. We tried more complex alternatives, such as a two-layer MLP with ReLU activation, but did not show any notable performance improvements.

Token visualization plot. As discussed in Section 3.2.3 of the main paper, our analysis of 256-token image sequences revealed that over 25% of CLEVR images shared identical first tokens, creating a substantial bias attributed to the uniform gray backgrounds prevalent in synthetic scenes. Interestingly, in Figure 26 we demonstrate that similar positional biases emerge in the Objectron and ARKitScenes datasets, which contain images of real-world 3D scenes, though with notably reduced magnitude. These plots illustrate the percentage of images sharing the most frequent token value at each sequence position.

3D scene conditioning in QA. To assess the impact of 3D scene data on question-answering, we train a model excluding 3D scene inputs, i.e., (I,TQ) → TA. This model achieves a Jaccard accuracy of 0.4465, compared to 0.4980 for the (I,3D,TQ) → TA model, demonstrating the critical role of 3D semantic information for accurate answers.

#### F. Failure cases

In this section, we discuss a failure case of Kyvo and potential areas for improvement. Among the four tasks used in our experiments for CLEVR, Instruction-Following presents the highest complexity. This task requires the model to process three modalities – 3D scenes, images, and text instructions – as input and generate both a modified image and a modified 3D scene. This requires precise comprehension of the text instruction and accurate application of the specified modifications across both the image and 3D scene sequences.

Our experiments indicate that while Kyvo effectively predicts the modified 3D scene sequence, it struggles with image sequence modifications. In the main paper, we report Jaccard Index values for the instruction-following task, demonstrating the model’s effectiveness in handling 3D scenes. Additionally, Figure 27 presents qualitative examples of the model’s image outputs. Although the predicted images are somewhat close to the groundtruth, the model often fails to accurately modify the scene within the image sequence. For instance, in the first example, the red sphere was incorrectly assigned a purple color and was moved behind but not to the left as instructed.

While this highlights areas for improvement, an interesting direction for future work is exploring whether decomposing complex tasks like instruction-following into sequences of simpler tasks can enhance performance. For example, instead of predicting both images and 3D scenes simulta-

[Figure 65]

###### Figure 24. Failure of modern-day LLMs on rendering task. Each row depicts one test scene described by our 3D structured modality; columns show the ground-truth Blender render (GT) and images produced by GPT4o, Gemini 2.0 Flash, and Kyvo (ours). GPT4o and Gemini frequently hallucinate extra objects, omit specified ones, or displace shapes, violating fundamental xyz and relational constraints.

[Figure 66]

###### Figure 25. Failure of Llama3.2-V on recognition task. For six randomly selected scenes, we render (from left to right) the ground-truth scene, the recognition prediction scene by Llama 3.2-Vision, and the predicted scene by Kyvo (ours). Llama 3.2-Vision frequently collapses objects toward the centre or misplaces them entirely, failing to recover the true spatial layout even in this simple synthetic setting. Moreover, it also misidentifies some objects in the scene.

[Figure 67]

###### (a) Objectron

[Figure 68]

###### (b) ARKitScenes

neously, the task could be divided into two stages: first, predicting the modified 3D scene, and then using a rendering model to generate the corresponding image.

- Figure 26. For 256-token image sequences, the plots show the percentage of images that have the most common token at each position in Objectron (left, blue) and ARKitScenes (right, red). Early positions repeat more often, so there is still some bias, but it is weaker than in synthetic CLEVR scenes.

[Figure 69]

- Figure 27. Instruction-following failure cases for the image modality on CLEVR.. As observed, the images generated by the model do not accurately reflect the intended modifications based on the input image and text instruction. On the other hand, the output 3D scenes are correct, meaning that our Kyvo accurately modified them based on the instructions. This suggests that a better avenue for predicting instruction-modified images is by task decomposition: first predict the modified 3D scene and then render the final image.

