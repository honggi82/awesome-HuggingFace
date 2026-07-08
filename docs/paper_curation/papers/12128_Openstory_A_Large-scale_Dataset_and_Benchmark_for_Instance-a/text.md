# arXiv:2408.03695v1[cs.CV]7Aug2024

## Openstory++ : A Large-scale Dataset and Benchmark for Instance-aware Open-domain Visual Storytelling

∗Zilyu Ye1,2, ∗Jinxiu Liu1,2, ∗Ruotian Peng1,2, JinJin Cao2, Zhiyang Chen2,4, Yiyang Zhang1, Ziwei Xuan3, Mingyuan Zhou3, Xiaoqian Shen5, Mohamed Elhoseiny5, †Qi Liu1 and †Guo-Jun Qi2

1South China University of Technology 2Westlake University 3OPPO US Research Center 4Foundation Model Research Center, CASIA 5King Abdullah University of Science and Technology

[Figure 1]

Figure 1: The visualization of our dataset. On the left is a data case with visual annotation that corresponds to each entity word in the sentence, where different color stands for different instance visual annotations, and on the right is the general pipeline of our dataset annotation process.

### Abstract

Recent image generation models excel at creating high-quality images from brief captions. However, they fail to maintain consistency of multiple instances across images when encountering lengthy contexts. This inconsistency is largely due

- *Equal contribution. †Corresponding authors.
- *{zilyuye,jinxiuliu0628}@foxmail.com, prtprt666@gmail.com, drliuqi@scut.edu.cn, guojunq@gmail.com

Preprint. Under review.

to in existing training datasets the absence of granular instance feature labeling in existing training datasets. To tackle these issues, we introduce Openstory++, a large scale dataset combining additional instance-level annotations with both images and text. This dataset can be utilized to train multi-modal generated models, allowing for the training of instance-focused story visualization models. Furthermore, we develop a tailored training methodology that emphasizes entity-centric image-text generation, ensuring that the models learn to effectively interweave visual and textual information. Specifically, Openstory++ streamlines the process of keyframe extraction from open-domain videos, employing vision-language models to generate captions that are then polished by a large language model for narrative continuity. It surpasses previous datasets by offering a more expansive open-domain resource, which incorporates automated captioning, high-resolution imagery tailored for instance count, and extensive frame sequences for temporal consistency. Additionally, we present Cohere-Bench, a pioneering benchmark framework for evaluating the image generation tasks when long multimodal context is provided, including the ability to keep the background, style, instances in the given context coherent. Compared to existing benchmarks, our work fills critical gaps in multi-modal generation, propelling the development of models that can adeptly generate and interpret complex narratives in open-domain environments. Experiments conducted within Cohere-Bench confirm the superiority of Openstory++ in nurturing high-quality visual storytelling models, enhancing their ability to address sophisticated and open-domain generation tasks. More details can be found at https://openstorypp.github.io/

### 1 Introduction

The domain of artificial intelligence has witnessed a surge of interest due to the emergence of highly capable generative models. Modern Multi-modal Large Language Models (MLLMs) have achieved remarkable fluency in synthesizing text [47, 43, 41], while state-of-the-art text-to-image models have shown an impressive ability to create realistic images [35, 4]. Despite these advancements, crafting high-quality visual stories that span an indefinite number of frames continues to be a significant challenge. Unlike the generation of a single image, creating multi-frame visual narratives necessitates the maintenance of subject continuity across frames, presenting a substantial hurdle.

While contemporary efforts have explored the image-text interleaved generation [26, 51, 38, 40, 32, 14, 13, 16, 26], they have predominantly concentrated on sustaining general semantic and contextual relationships between text and images, falling short of achieving instance-level semantic coherence. Moreover, some studies [25, 22, 29, 24] have only managed to maintain instance-level cross-frame visual consistency within confined domain datasets. This limitation is primarily due to the lack of open-domain datasets that encapsulate the narrative and temporal dynamics essential for training models to produce coherent stories. The existing datasets often fail to provide the required instance focus for coherent storytelling.

In response to these limitations and the pressing demand for automated and scalable methods to generate data for visual storytelling, we introduce Openstory++, a comprehensive dataset that underscores narrative continuity around pivotal instances with instance-level visual segmentation annotations, as depicted in Figure 1. Our dataset processes video content to extract keyframes, evaluates them aesthetically, and employs BLIP2 [19] to produce descriptive captions. These captions are further refined by a Large Language Model (LLM) to ensure narrative coherence. Additionally, our sub-pipeline identifies valid instances within the images and utilizes the Segment Anything Model (SAM) [17] to create masks for these instances. Openstory++ offers narratively coherent visual sequences that are tailored to specific instances, making it an ideal resource for training models that can generate visual stories with coherent instances across frames.

Additionally, to better reveal the model’s ability in interleaved image-text generation, especially the consistency of visual appearance representing the same semantic instance across frames, we designed Cohere-Bench. It evaluates visual storytelling models across progressive dimensions, including single and multi-entity preservation in text-to-image generation and multi-turn generation capabilities, providing a comprehensive assessment of models’ performance in visual storytelling tasks.

- • We created the Openstory++, a large-scale visual storytelling dataset enhancing instancefocused story visualization models by providing contextually coherent frames featuring recurring instances.
- • We have developed a training methodology specifically tailored for entity-focused imagetext interleaved generation, addressing the issue of inconsistency in existing models when dealing with lengthy contexts featuring multiple instances.
- • We present Cohere-Bench, a pioneering benchmark framework for evaluating image-text generation. It overcomes the limitations of existing benchmarks by assessing long-context entity consistency and multi-turn generation capabilities.
- • Experiments in Cohere-Bench demonstrate Openstory++ ’s advantage in developing highquality instance-aware open-domain visual storytelling models.

### 2 Related Work

#### 2.1 Datasets for Story Visualization

Story visualization datasets are crucial for training models to generate images that correspond to narrative descriptions. However, existing datasets have several limitations. The VIST dataset[42], one of the earliest, did not include instance extraction within frames, which is essential for effective instancefocused story visualization tasks. More recent datasets like PororoSV[22], FlintstonesSV[29], and StorySalon[25] offer improved continuity in frame sequences and caption coherence but are limited to close-domain scenarios such as cartoons, restricting the model’s capability for open-domain scene generation. Additionally, artificial annotations in captions of PororoSV, FlintstonesSV, and VIST limit the size of the datasets and increase production costs. On the other hand, datasets like DideMoSV[30] that rely on video subtitles for caption generation may compromise the accuracy of captions in describing frame scenes. To overcome these limitations, we have developed a pipeline capable of extracting instance-focused keyframes and sequences from open-domain videos. This pipeline has been used to create the OpenStory++ dataset, a large-scale dataset that includes caption-frame pairs with long-term sequences. We believe that the OpenStory++ dataset will significantly enhance the proficiency of models in instance-focused story visualization tasks and provide a robust benchmark for multi-modal long-context understanding and generation.

#### 2.2 Benchmarks for Generative Multi-modal Model

The advancement of multimodal large language models (MLLMs) has necessitated the development of benchmarks to evaluate their capabilities. Several studies have proposed benchmarks to assess various aspects of MLLMs. Benchmarks such as OwlEval[45], LLaVA-Bench[26], LAMM[12], Touchstone[3], MME[11], and MMBench[27] evaluate the contextual understanding of text-image information and basic visual localization abilities of MLLMs, but they do not focus on generation capabilities. Other benchmarks, like MMMU[46], emphasize the model’s reasoning and cognitive abilities across multiple domains. Seed-bench-2[18] specifically evaluates the image generation capabilities of multimodal large models. However, none of these benchmarks assess the thematic consistency of image generation in long-context scenarios for MLLMs. To address this gap, we introduce Cohere-Bench, a benchmark designed to evaluate the instance consistency of image generation in MLLMs over extended contexts.

### 3 Openstory++

Openstory++ is a dataset that enhances instance-focused story visualization models by providing contextually coherent frames featuring recurring instances, which promote the creation of coherent and contextually relevant visual narratives. Unlike previous approaches that generated a single keyframe per textual cue, our method extracts key frames emphasizing significant segments of the video, capturing the essential narrative aspects and offering a detailed analysis of instance activities. The result can be seen in Figure 3.

Avg. Length

Masks

per frames Seriality PororoSV[22] close Manual 73K 5 - ✓

Dataset Domain Caption Frames

FlintstonesSV[29] close Manual 123K 5 - ✓ DideMoSV[30] close Manual 53K 3 - ✓

VIST[42] open Manual 145K 5 - ✓ StorySalon[25] close ASR 159K 14 1 ✓

SDD[28] open Generated 76M 1 3 ✗ Openstory++ open Generated 100M+1M 28 2.5 ✓

Table 1: Openstory++ statistics and comparison with other story-visualization datasets. Avg.Length represents the average length of storytelling data for each scene. Our dataset consists of about 100 million high-quality, fully annotated unique samples, as well as an additional 1 million fully annotated sequence samples.

|Task: Enhancing Continuity and Storytelling in Captions Objective: Given a series of English captions and a list of English nouns, your task is to replace words in the captions with related nouns from the noun list. The goal is to maintain the narrative flow and coherence of the captions while integrating relevant nouns to enrich the storytelling.<br><br>Sample Input: Caption List: ["A man with a beard and a hat is standing", "A man holding a small piece of wood in front of a tree", "A man with a beard and hat is holding a knife"] Noun List: ["knife", "man", "hat"] Output: ["A man clad in a worn hat and sporting an unkempt beard stood motionless", "The man is clutching a small knife, standing before a towering tree and memory flooding back", "the bearded man gripped the knife in his calloused hands ready for the next chapter"]<br><br>Instructions:<br><br>1. Analyze each caption in the caption list to identify words that can be replaced with related nouns from the provided noun list.<br>2. Subsequently, substitute those words with corresponding nouns, ensuring that the overall meaning and coherence of the captions remain intact<br>3. Format the output as a list enclosed in square brackets, with each caption separated by commas and enclosed in quotes without any other explanation<br>4. Emphasize the improvement of narrative continuity and storytelling coherence between captions.<br><br><br>Caption List: {captions} Noun List: {nouns}|
|---|

- Figure 2: This figure presents a prompt designed to enhance narrative flow and coherence across scenes, which contains refined captioning guidelines aimed at enriching imagery with descriptive details while preserving the core content. Additionally, the prompt emphasizes maintaining consistent instances throughout the storytelling process.

#### 3.1 Data Sources

Our sequence dataset incorporates a broad range of video datasets sourced from platforms such as Pandas-70M[6] and InternVid[44]. Besides, we have carefully planned to add some high-quality data sources with inherent storytelling properties to enhance the data quality of the dataset. In contrast to the closed-domain datasets like StorySalon [25] and Flintstones-SV [29], our dataset encompasses images from a wide array of scenarios, offering enhanced resources for training long-context models in open-domain environments. Using these data sources, we successfully labeled large-scale and fully annotated data. We compared other related datasets in Table 1.

[Figure 2]

- Figure 3: This figure shows the video frame sequence generated by our pipeline, with the subject’s mask and the subject’s bounding box.

#### 3.2 Pipeline Overview

Our data processing pipeline is designed to construct a coherent series of keyframes featuring the same instance across different scenes, along with storytelling captions. This approach ensures narrative consistency and leverages a Large Language Model (LLM) [10] to enhance captions, maintaining coherence throughout the storytelling process. The pipeline culminates in creating precise instance masks from the refined keyframes and captions, essential for instance-focused visual storytelling tasks. The workflow is illustrated in Figure 4.

Keyframe Extraction and Deduplication The pipeline initiates with extracting I-frames from the video content. Then, they are processed with DINOv2 [31] to identify and eliminate redundant frames with high visual similarity. This ensures that each frame has distinct visual expression and captures the essence of the narrative.

Single-Image Captioning and Instance-Masking Workflow For each keyframe, we employ BLIP2 [20] to autonomously generate basic captions. We subsequently utilize NLTK [5] to extract entity labels from these captions. Combined with YOLO-World [7], we define the bounding box for each entity instance. The EfficientViT-SAM model [48] then creates pixel-level instance masks, providing fine-grained visual annotations for each labeled entity in the captions. In addition, using this single-image annotation workflow, we can easily obtain fully annotated images from large-scale mature datasets[37, 39, 17].

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Video-LLaVA

(Video Captioning)

NLTK (Instance Highlight)

###### BLIP2 (Image Captioning)

|The video features two canines, one black and white and the other<br><br>brown and white, sitting on the floor and looking at a beverage. The black and white canine sniffs the bottle and then licks it, while the brown and white canine also sniffs the bottle and licks it. The video ends with the two canines sitting on the floor.|
|---|

|a man with glasses and a dog in front of a barn|
|---|

|a man with<br><br>glasses is taking a<br><br>picture of a dog|
|---|

|a man in glasses is petting a doberman|
|---|

|a man in glasses is standing next to a dog|
|---|

|a man with glasses and a dog standing next to him|
|---|

Large Language Model (Object Alignment)

Prompt: “Enhance continuity and storytelling in captions…”

|canine are lounging next to a<br><br>beverage|canine are frolicking with a beverage|
|---|---|

|canine are positioned beside the bottled beverage|
|---|

|a human is clasping a bottled<br><br>beverage a canine take a seat on it|
|---|

|a human is gripping a bottled beverage while canine lingers in the distance|
|---|

|YOLO-World (Object Detection)|
|---|

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

DINO (Image-Text Entity Alignment) EfficientViT-SAM (Instance Segmentation)

Aligned caption list Target frame sequence

|canine0 and<br>canine1 are positioned beside the<br><br><br>bottled<br><br>beverage0 and<br>beverage1 and<br>beverage2<br>|a human is gripping a<br><br>bottled<br><br>beverage0 while canine0 lingers in the distance|a human is clasping a bottled beverage0<br><br>a canine0<br><br>take a seat on it|canine0 and<br>canine1 are lounging next to a<br><br><br>beverage0|canine0 and<br>canine1 are frolicking<br><br><br>with a<br><br>beverage0|
|---|---|---|---|---|

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

- Figure 4: This figure showcases the workflow of our pipeline. After obtaining a sequence of frames devoid of redundancy, we first utilized BLIP2 to generate basic image captions. Subsequently, Video-LLaVA was employed to produce a sequence of captions that encapsulate the narrative flow. Guided by the sequence caption, a LLM was prompted to align the entities in the image captions, thus enhancing the narrative coherence across consecutive frames. Next, YOLO-World was applied to detect bounding boxes for the entities. To ensure that labels for the same entities across frames are unique and consistent, we blended the bounding box labels with the assistance of Dino and a facial feature module. Finally, we employed EfficientVIT-SAM to obtain the masks for the entities, thereby providing a comprehensive understanding of the spatial extent and characteristics of each entity within the frames.

Frame-Caption Alignment for Narrative Coherence Then we align the refined captions with the keyframes to ensure consistent labeling of the same subject across images. VideoLLaVA [23] processes multiple keyframes to generate sequence-level captions, enabling the standardization of entity labels across continuous scenes. ChatGLM3-Turbo [47], based on the Instance given by VideoLLaVA, use the prompt shown in Figure 2 to refine the subjects represented by different concepts in captions annotated by BLIP2, ensuring that the subjects in multiple captions remain consistent, further refines individual image captions, focusing on narrative cohesion and enhancing the storytelling object alignment by accurately distinguishing between human instances.

Instance Masking The concluding step involves creating instance masks based on the labeled captions from the image sequence. With YOLO-World and DINOv2, we delineate the bounding box for each entity instance. The EfficientViT-SAM model fabricates pixel-level instance masks, concentrating our dataset on the most relevant visual storytelling instances. Additionally, an integrated facial detection mechanism enhances the precision of differentiating between human instances, overcoming the limitations of DINOv2.

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

2

4

6

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

“Please help me to Visual Detokenizer Visual Detokenizer Visual Detokenizer

[Figure 38]

[Figure 39]

write a story about

[Figure 40]

[Figure 41]

MAPLE

Large Language Model

a golden retriever based on the reference image.”

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

[Figure 52]

1

7 “OK! Once upon

3

5

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Once upon a time, there was a gentle golden retriever named Sunny, who brought happiness to his family with his often licking

Spring brought a new baby to the family, and their dog Sunny adapted with ease and curiosity.

The golden retriever always protects the baby and plays with the baby.

Then, ...

[Figure 57]

a time...

tongue.

[Figure 58]

[Figure 59]

[Figure 60]

Segment & Retrival Segment & Retrival Segment & Retrival

- Figure 5: Overview of the interleaved image-text generation: Both the image and text are produced by the MLLM. During image generation, we take the diffusion model as the visual detokenizer.

### 4 Training Data Challenge

#### 4.1 Model Settings

In our experiment, we aim to generate M images, denoted as Simg = {I1,...,IM}, from M lines of text prompts, Stxt = {L1,...,LM}. Unlike traditional text-to-image models that generate images independently, our approach considers the contextual relationships and sequence of images. We use a multi-modal guided generation model that extends the autoregressive capabilities of MiniGPT-5. It is built on a Large Language Model (LLM) with pre-trained weights from Vicuna-7B [50], adept at producing consistent outputs from interleaved inputs of images and text. For visual encoding, we use a pretrained Vision Transformer (ViT) [9] to convert visual content into embeddings and inject into LLM space by linear projection. The visual encoder extracts features from both original images and segmented objects, with segmented images encoded by a visual tokenizer. During image generation, the visual encoder and LLM process input images and text together. A visual de-tokenizer, which is Stable Diffuiosn 2.1 [36], uses the LLM’s output features to condition the final image generation. The architecture of model is shown in Figure 5.

#### 4.2 Dataset Comparsion

Instance

Style

Training Data

Semantic

Consistency↑ Perplexity↓ Aesthetic↑ VIST [42] 0.183 0.742 0.598 35.626 4.748

Consistency↑

Alignment↑

StorySalon [25] 0.245 0.754 0.727 31.256 4.941 FlintstoneSV [29] 0.252 0.772 0.720 30.254 4.864 PororoSV [22] 0.247 0.662 0.637 52.703 4.755 Openstory++ 0.262 0.783 0.765 29.045 4.843

Table 2: Comparison of various metrics across different datasets.

We evaluated our model across various datasets for its ability to generate coherent and stylistically consistent text, as detailed in Section 4.1. Key metrics included semantic alignment, style consistency, and instance consistency. Perplexity, using phi3-mini-4k [1], assessed word prediction accuracy. Aesthetic judgment evaluated visual appeal and distribution similarity to real images, respectively. From the Table 2, it is evident that our model trained on Openstory++ outperforms other datasets in terms of semantic alignment, style consistency, and instance consistency, as indicated by the higher values in these metrics. The lower perplexity score for our dataset suggests that our model is more effective at predicting the next word in a sequence. The aesthetic scores further validate the quality of our model’s generated content, with a higher aesthetic score indicating better visual appeal except compared to close domain cartoon datasets. This demonstrates the crucial role of our dataset in generating high quality image-text interleaved models with cross-frame instance consistency.

#### 4.3 Cohere-Bench

Our Cohere-Bench is designed to evaluate the quality of generated images and text against a ground truth dataset. We have sampled a subset of our dataset with 1600 items to serve as this evaluation dataset. The benchmark assesses the similarity of generated images to the ground truth and the effectiveness of the generated text. To evaluate text effectiveness, we employ a pipeline that involves extracting keyframes from videos, tagging them with Large Language Models (LLMs) and Video LLaVA [23], and applying YOLO-World [7] and BLIP2 [20] to the generated images to add captions and identify the main instance’s position, which is then aligned using LLM. This process results in a sequence of generated images with captions and instance-level segmentation. For consistency of the generated instance with the given instance, we measure semantic similarity between the segmented instance and the ground truth. We also calculate the similarity between the instance and previously appearing instances, as well as semantic alignment with the current instruction, style consistency, and instance consistency for both single-instance (s) and multiple-instance (m) settings. The specific calculation methods are as follows:

- • Semantic Alignment: We use CLIP [34] to measure the semantic similarity between visual and text features, ensuring that the generated image aligns with the instruction prompt.
- • Background Consistency: We detect entities using YOLO-World [7] and calculate the similarity of the masked and inpainted images to ensure background consistency.
- • Style Consistency: We measure the similarity across previous frames and compute the average similarity score to gauge style consistency.
- • Instance Consistency: For both single and multiple instances, we calculate the similarity between the generated image and the reference instance using YOLO-World [7] to segment the instance.
- • Instance Integrity: We assess the completeness of the segmented instance in the current image compared to the ground truth, expressed as a percentage.
- • BLEU4: Due to the challenges in evaluating the accuracy of text continuation, we apply our dataset’s annotation pipeline directly to the generated image sequence. By doing so, we annotate the generated images and use BLEU4 [33] scores to measure the similarity between the generated captions and the ground truth text. This approach allows us to assess the accuracy of content generation based on context, which measures the similarity of the recaptioned text to the ground truth.

Evaluation analysis We evaluate the semantic alignment of generated images with their current captions and their visual consistency within a multi-modal context. The results are presented in Table 3. For text alignment, our model, our model, outperforms all others except for GPT-4V and MiniGemini, which also demonstrate strong performance due to their robust captioning capabilities. Additionally, our model excels in maintaining cross-plot background consistency and instance consistency in both single and multiple instance settings. Unlike other models that rely on global features to encode images into the MLLM, resulting in less detailed visual features, our model enriches the inference phase with detailed, instance-level visual annotations. This approach integrates instance-level segmentation into the autoregressive space of the LLM, enhancing the model’s ability to understand and generate detailed visual features.

#### 4.4 Human Evaluation

Our human evaluation framework integrates assessments of visual story quality across various dimensions, including image-text alignment, image style, story consistency, and instance consistency. We employ two complementary evaluation methods:

Comparative Evaluation: This method quantitatively measures the visual story quality through numerical ratings. Participants rate random samples from our model and baseline models on a scale of 1 to 5, with 1 being the lowest and 5 the highest quality. This approach provides detailed insights into each model’s performance across different aspects of visual storytelling.

Semantic

Background Consistency↑

Style

Instance

Instance

Instance Integrity↑ BLEU4↑

Models

Alignment↑

Consistency↑

Consistency (s)↑

Consistency (m)↑

DreamLLM[8] 0.270 0.615 0.615 0.271 0.292 0.144 0.055 MiniGPT-5[49] 0.209 0.634 0.214 0.214 0.219 0.115 0.011

SEED-X[15] 0.272 0.775 0.762 0.744 0.774 0.421 0.057

Emu2[40] 0.258 0.788 0.762 0.818 0.787 0.351 0.058 GPT4-V[2] 0.286 0.762 0.781 0.753 0.761 0.424 0.062

MiniGemini[21] 0.271 0.710 0.577 0.602 0.610 0.203 0.052

Our model

(w/o visual anno) 0.254 0.748 0.766 0.693 0.696 0.383 0.054

Our model

(w/ visual anno) 0.279 0.791 0.784 0.821 0.782 0.429 0.064

- Table 3: Quantitative comparison of various models on semantic alignment, style consistency, and instance consistency in single-instance (s) and multiple-instance (m) settings. The table also presents BLEU4 scores, which are metrics for evaluating the quality of generated text.

Preference Evaluation: This method captures the instanceive preferences of participants regarding the overall appeal of visual stories. For a given storyline, participants choose their most preferred image sequence from those generated by our model and other methods. This reveals which model’s storytelling is most engaging to the audience.

The evaluation criteria, including semantic alignment, image style, story consistency, character consistency, plot continuity, and image quality, are chosen to reflect the key elements of visual storytelling. As shown in Table 4, our model trained on Openstory++ (w/ visual anno), shows a strong performance across most metrics, indicating its effectiveness in generating coherent and high-quality visual narratives. However, it slightly underperforms in the text alignment metric compared to GPT4V and MiniGemini. This discrepancy could be attributed to the advanced language understanding and visual generation capabilities of GPT4-V and the sophisticated image synthesis techniques provided by MiniGemini’s training with DALL-E 3’s API [4]. Despite this, our model trained on Openstory++ (w/ visual anno) excels in plot continuity and image quality, which are crucial for maintaining the narrative flow and visual appeal of the stories. This suggests that training on datasets with visual annotations significantly enhances the model’s ability to generate images that are not only consistent with the story context but also contribute to a coherent and engaging visual storytelling experience.

|Model / Training Data Align Style Consist Char<br><br>Plot Continuity<br><br>Image Quality<br><br>|Pref|
|---|---|
|SEED-LLAMA [13] 2.87 2.58 2.63 2.95 3.55 3.57 SEED-X [15] 3.76 3.78 3.67 3.67 3.66 3.93 Emu2 [40] 3.20 2.92 3.85 2.75 3.70 4.11 MiniGemini [21] 4.03 3.81 2.85 2.70 2.65 3.62 GPT4-V [2] 4.30 2.87 3.85 3.73 4.25 4.75 VIST [42] 1.98 2.29 3.25 2.15 2.31 2.72<br><br>StorySalon [25] 3.95 3.30 4.07 3.22 3.29 3.46 Openstory++ (w/o visual anno) 3.81 4.01 3.53 3.62 4.09 4.12<br><br>Openstory++ (w/ visual anno) 4.02 4.14 4.25 4.31 4.40 4.24|3.47%<br><br>13.53%<br><br>14.21%<br><br><br>8.34%<br><br>20.66%<br><br>1.56% 5.28% 11.67%<br><br>21.28%<br>|

- Table 4: Comparison results of human evaluation. The metrics are text-image alignment, style consistency, content consistency, character consistency, plot continuity, image quality, and preference, respectively. The scores for each metric have been adjusted to maintain a consistent relationship with the preference percentage.
- 5 Conclusion

In summary, our work introduces significant advancements in the field of multi-modal generation. The Openstory++ dataset, coupled with our tailored training methodology, addresses the limitations of current image generation models by providing a rich, instance-focused resource that promotes

entity consistency and narrative continuity. Furthermore, the Cohere-Bench benchmark framework sets a new standard for evaluation, focusing on long-context coherence and multi-turn capabilities. These contributions not only enhance the capabilities of existing models but also pave the way for future innovations in generating and interpreting complex narratives within open-domain settings.

### 6 Limitations

Although this work shows some progress, there are still some limitations. While broad, the Openstory++ dataset may not cover all possible scenarios encountered in visual storytelling. In addition, there are some errors in the algorithm of the data set, and the data set is not completely error-free, which may bring some biases to the training results.

### References

- [1] Marah Abdin et al. “Phi-3 technical report: A highly capable language model locally on your phone”. In: arXiv preprint arXiv:2404.14219 (2024).
- [2] Josh Achiam et al. “Gpt-4 technical report”. In: arXiv preprint arXiv:2303.08774 (2023).
- [3] Shuai Bai et al. “Touchstone: Evaluating vision-language models by language models”. In: arXiv preprint arXiv:2308.16890 (2023).
- [4] James Betker et al. Improving Image Generation with Better Captions. 2023.
- [5] Steven Bird. “NLTK: the natural language toolkit”. In: Proceedings of the COLING/ACL 2006 Interactive Presentation Sessions. 2006, pp. 69–72.
- [6] Tsai-Shien Chen et al. “Panda-70M: Captioning 70M Videos with Multiple Cross-Modality Teachers”. In: arXiv preprint arXiv:2402.19479 (2024).
- [7] Tianheng Cheng et al. “YOLO-World: Real-Time Open-Vocabulary Object Detection”. In: Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR). 2024.
- [8] Runpei Dong et al. “Dreamllm: Synergistic multimodal comprehension and creation”. In: arXiv preprint arXiv:2309.11499 (2023).
- [9] Alexey Dosovitskiy et al. “An image is worth 16x16 words: Transformers for image recognition at scale”. In: arXiv preprint arXiv:2010.11929 (2020).
- [10] Zhengxiao Du et al. “GLM: General Language Model Pretraining with Autoregressive Blank Infilling”. In: Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2022, pp. 320–335.
- [11] Chaoyou Fu et al. “MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models”. In: arXiv preprint arXiv:2306.13394 (2023).
- [12] Jingsheng Gao et al. “LAMM: Label Alignment for Multi-Modal Prompt Learning”. In: Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. 3. 2024, pp. 1815– 1823.
- [13] Yuying Ge et al. “Making llama see and draw with seed tokenizer”. In: arXiv preprint arXiv:2310.01218 (2023).
- [14] Yuying Ge et al. “Seed-x: Multimodal models with unified multi-granularity comprehension and generation”. In: arXiv preprint arXiv:2404.14396 (2024).
- [15] Yuying Ge et al. “Seed-x: Multimodal models with unified multi-granularity comprehension and generation”. In: arXiv preprint arXiv:2404.14396 (2024).
- [16] Hexiang Hu et al. “Instruct-Imagen: Image generation with multi-modal instruction”. In: arXiv preprint arXiv:2401.01952 (2024).
- [17] Alexander Kirillov et al. “Segment Anything”. In: arXiv:2304.02643 (2023).
- [18] Bohao Li et al. “Seed-bench-2: Benchmarking multimodal large language models”. In: arXiv preprint arXiv:2311.17092 (2023).
- [19] Junnan Li et al. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. arXiv:2301.12597 [cs]. June 2023. URL: http:// arxiv.org/abs/2301.12597 (visited on 01/31/2024).
- [20] Junnan Li et al. “Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models”. In: International conference on machine learning. PMLR. 2023, pp. 19730–19742.
- [21] Yanwei Li et al. “Mini-gemini: Mining the potential of multi-modality vision language models”. In: arXiv preprint arXiv:2403.18814 (2024).
- [22] Yitong Li et al. StoryGAN: A Sequential Conditional GAN for Story Visualization. 2019. arXiv: 1812.02784 [cs.CV].
- [23] Bin Lin et al. “Video-LLaVA: Learning United Visual Representation by Alignment Before Projection”. In: arXiv preprint arXiv:2311.10122 (2023).
- [24] Han Lin et al. VideoDirectorGPT: Consistent Multi-scene Video Generation via LLM-Guided Planning. 2023. arXiv: 2309.15091 [cs.CV].
- [25] Chang Liu et al. Intelligent Grimm – Open-ended Visual Storytelling via Latent Diffusion Models. arXiv:2306.00973 [cs]. Mar. 2024. URL: http://arxiv.org/abs/2306.00973 (visited on 03/06/2024).
- [26] Haotian Liu et al. “Visual instruction tuning”. In: Advances in neural information processing systems 36 (2024).

- [27] Yuan Liu et al. “Mmbench: Is your multi-modal model an all-around player?” In: arXiv preprint arXiv:2307.06281 (2023).
- [28] Jian Ma et al. Subject-Diffusion:Open Domain Personalized Text-to-Image Generation without Test-time Fine-tuning. arXiv:2307.11410 [cs]. July 2023. URL: http://arxiv.org/abs/ 2307.11410 (visited on 02/28/2024).
- [29] Adyasha Maharana and Mohit Bansal. Integrating Visuospatial, Linguistic and Commonsense Structure into Story Visualization. 2021. arXiv: 2110.10834 [cs.CL].
- [30] Adyasha Maharana, Darryl Hannan, and Mohit Bansal. StoryDALL-E: Adapting Pretrained Text-to-Image Transformers for Story Continuation. 2022. arXiv: 2209.06192 [cs.CV].
- [31] Maxime Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. 2023.
- [32] Xichen Pan et al. “Kosmos-g: Generating images in context with multimodal large language models”. In: arXiv preprint arXiv:2310.02992 (2023).
- [33] Kishore Papineni et al. “Bleu: a method for automatic evaluation of machine translation”. In: Proceedings of the 40th annual meeting of the Association for Computational Linguistics. 2002, pp. 311–318.
- [34] Alec Radford et al. “Learning transferable visual models from natural language supervision”. In: International conference on machine learning. PMLR. 2021, pp. 8748–8763.
- [35] Robin Rombach et al. High-Resolution Image Synthesis with Latent Diffusion Models. 2022. arXiv: 2112.10752 [cs.CV].
- [36] Robin Rombach et al. “High-resolution image synthesis with latent diffusion models”. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022, pp. 10684–10695.
- [37] Christoph Schuhmann et al. “Laion-400m: Open dataset of clip-filtered 400 million image-text pairs”. In: arXiv preprint arXiv:2111.02114 (2021).
- [38] Xiaoqian Shen and Mohamed Elhoseiny. “Large Language Models as Consistent Story Visualizers”. In: arXiv preprint arXiv:2312.02252 (2023).
- [39] Keqiang Sun et al. “Journeydb: A benchmark for generative image understanding”. In: Advances in Neural Information Processing Systems 36 (2024).
- [40] Quan Sun et al. “Generative multimodal models are in-context learners”. In: arXiv preprint arXiv:2312.13286 (2023).
- [41] Gemini Team. Gemini: A Family of Highly Capable Multimodal Models. 2023. arXiv: 2312. 11805 [cs.CL].
- [42] Ting-Hao et al. Visual Storytelling. arXiv:1604.03968 [cs]. Apr. 2016. URL: http://arxiv. org/abs/1604.03968 (visited on 03/14/2024).
- [43] Hugo Touvron, Louis Martin, Kevin Stone, et al. Llama 2: Open Foundation and Fine-Tuned Chat Models. 2023. arXiv: 2307.09288 [cs.CL].
- [44] Yi Wang et al. “Internvid: A large-scale video-text dataset for multimodal understanding and generation”. In: arXiv preprint arXiv:2307.06942 (2023).
- [45] Qinghao Ye et al. “mplug-owl: Modularization empowers large language models with multimodality”. In: arXiv preprint arXiv:2304.14178 (2023).
- [46] Xiang Yue et al. “Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi”. In: arXiv preprint arXiv:2311.16502 (2023).
- [47] Aohan Zeng et al. “Glm-130b: An open bilingual pre-trained model”. In: arXiv preprint arXiv:2210.02414 (2022).
- [48] Zhuoyang Zhang, Han Cai, and Song Han. “EfficientViT-SAM: Accelerated Segment Anything Model Without Performance Loss”. In: arXiv preprint arXiv:2402.05008 (2024).
- [49] Kaizhi Zheng, Xuehai He, and Xin Eric Wang. “Minigpt-5: Interleaved vision-and-language generation via generative vokens”. In: arXiv preprint arXiv:2310.02239 (2023).
- [50] Lianmin Zheng et al. “Judging llm-as-a-judge with mt-bench and chatbot arena”. In: Advances in Neural Information Processing Systems 36 (2024).
- [51] Deyao Zhu et al. “Minigpt-4: Enhancing vision-language understanding with advanced large language models”. In: arXiv preprint arXiv:2304.10592 (2023).

### A Dataset Details

Our dataset mainly consists of 100M single-image data with instance-level annotation and 1M image sequence data with narrative content. In this section, we introduce some details of the dataset composition and the pipeline for building the dataset.

#### A.1 Data Source and Data selection

- A.1.1 Unique Data

- 1. Laion-400m This is one of the currently popular large-scale datasets, featuring open-domain largescale images and traceable URLs. However, the original image captions vary significantly in quality. We utilized BLIP2 to re-caption these images and processed them through our single-image pipeline, resulting in a dataset format with instance-level annotations.
- 2. SA-1B This is a large-scale image dataset that includes high-quality street view images. Due to privacy protection measures, facial data in this dataset has been blurred, which is not ideal for pretraining our instance-level model. Therefore, we only incorporated a portion of the high-quality images into our dataset.
- 3. JourneyDB This is a large-scale dataset containing 4,429,295 high-quality Midjourney images, annotated with high-quality image captions. Since these images are generated by Midjourney, they exhibit a uniform style. We included this dataset as a separate subset and performed instance-level annotations. Due to the significant stylistic differences from real-world images, it serves as a highquality subset for training instance-focused models.

Once we have these image datasets, we can filter and annotate them through our single-image instance-level annotation pipeline.

- A.1.2 Continous Sequence Data

- 1. Pandas-70M This is a large-scale dataset with 70 million high-quality video-caption pairs. This dataset is generated using a multi-step process that includes video collection, captioning, and filtering. The dataset is designed to provide a comprehensive and diverse set of video-text pairs for various multimodal tasks
- 2. InternVid This is a large-scale video-centric multimodal dataset designed to facilitate the development of powerful and transferable video-text representations for multimodal understanding and generation. This dataset contains over 7 million videos, yielding 234 million video clips accompanied by detailed descriptions totaling 4.1 billion words. The dataset is built using a scalable approach that leverages large language models (LLMs) to generate high-quality video-text pairs

We utilized a subset of data from Pandas-70M and Intenvid, primarily sourced from naturally story-driven YouTube categories such as film and how-to, as well as subject-driven YouTube categories like pet and people. Additionally, we extracted videos corresponding to captions containing subjects from their inherent video captions. For instance, if a caption reads "A person is holding a long-haired dachshund in their arms," it indicates a video containing the subject we’re interested in. From these, we curated high-quality source videos to build our dataset. Furthermore, we gathered numerous high-quality videos from narrative-focused channels with a focus on subjects, such as reality TV shows and outdoor adventures from explorer channels.

#### A.2 Dataset Construction Pipeline Details

We set up a multi-faceted filtering strategy to improve the dataset’s quality. For example, aesthetic score, number of Instances, etc.

#### A.2.1 Filtering strategy

Instance filtering We filtered out images without instances and images with more than eight subjects. The number of instances was determined by the bounding boxes annotated by YOLO-World. This is because images without instances add noise to the dataset, while images with too many instances become too complex for the model to learn effectively.

Aesthetic filtering To improve the quality of the dataset, we employed an aesthetic evaluation model to filter out images with low aesthetic scores. For unique images, we selected those with an aesthetic score greater than five, resulting in a high-aesthetic-score dataset that constitutes 15% of the total unique image dataset. Additionally, in the image sequence dataset, we retained only origin frames with scores higher than 4.5 in our

[Figure 61]

- Figure 6: Some statistics about bounding box confidence, caption word count, instance count, and aesthetic score, as well as a word cloud of instance label of our sequence dataset

pipeline. Through human evaluation, this approach effectively filtered out transitional and blurry shots, thereby strengthening the continuity between frames

#### A.2.2 Pipeline Performance and Models Details

The performance of the entire pipeline described above is primarily constrained by the speed bottleneck of the LLM in refining the captions and the instance-level data annotation after obtaining the refined captions. We use ChatGLM3-Turbo’s API as our LLM provider, and it returns our refined caption request approximately once every 5 seconds. We employed the latest YOLO-Worldv2-X and YOLO-Worldv2-L models for mixed annotation of bounding boxes in our dataset. Additionally, we utilized EfficientViT-SAM for generating instance masks within the bounding boxes. Compared to other related approaches, these models demonstrate rapid and high-performance annotation at the instance level. Their combined performance on a single A100 GPU can handle approximately 30 images per second.

#### A.3 Dataset Statistics

As illustrated in Figure 6, we have compiled statistics on the aesthetic scores, bounding box confidences, caption word counts, and the number of entities per image in our sequence data. Additionally, we present a word cloud to display the information on labels within our bounding boxes, which intuitively shows the diversity of our data.

Besides, It is worth noting that when we mentioned filtering out images with aesthetic scores lower than 4.5, this filtering was done at the source image stage. Since our final target images were resized to match the typical square size used by encoders, the aesthetic scores exhibited some variations upon recalculation.

### B Experiments

In this section, we start by describing our experimental settings and then compare them with other models. Additionally, we present results for ablation experiments to prove the effectiveness of our dataset construction pipeline modules.

#### B.1 Experimental Settings

Validation Dataset Set We selected two subsets from the OpenStory++ dataset for our evaluation: the single dataset 7 and the multi dataset 8. These subsets represent scenes with single subjects and multiple subjects, respectively. Our selection process involved manual evaluation. Specifically, we categorized the collected dataset by video ID into different stories. We then observed each story to ensure that all frames within a story contained the same subject performing different actions. Only those stories that met this criterion were included in our evaluation data. The single dataset typically consists of stories with a main instance in each frame, while the multi dataset includes stories with multiple main instances. This careful curation ensures that the evaluation data accurately represents the complexity and variety of real-world scenarios, allowing us to robustly assess the generative capabilities of the models.

[Figure 62]

##### Figure 7: Some single instance sample from our dataset

[Figure 63]

##### Figure 8: Some multi instance sample from our dataset

Cohere-Bench Task Design Once we have a validation set, we designed two tasks within the Cohere-Bench framework to evaluate the generative capabilities of current large multi-modal models: story generation and story continuation. For the story generation task, we provided the multi-modal models with a text prompt to generate an image for the first scene. Subsequently, the models used the generated image and a new text prompt for the second scene to generate the corresponding image, continuing this pattern for subsequent scenes. In the story continuation task, we initially provided both the image of the first scene and the text prompt for the second scene, aiming to generate the image for the next scene. For both tasks, each story comprised 2-5 scenes.

Test Details We tested these tasks on many models: GPT4-V, Seed-X, Emu2, DreamLLm, MiniGemini, MiniGPT-5 and our model. For each model, we used the official open-source inference settings as default parameters. For models that did not support simultaneous text and image input to generate images, we manually adjusted the input and generation methods without making any other modifications. All inferences were performed on a single A100 GPU.For each model, the text prompt followed the format "Generate an image + caption for the next scene", with MiniGPT-5 using its default prompt description as input. The image prompt for each scene was the generated image of the previous scene. In addition, when we evaluate other models, we will delete the dino alignment index in caption, such as 0 in "woman0". One results of a model comparison can be referred to Figure 9. We assessed the generative quality of each model using the metrics introduced in the Section 4.3.

Cohera-Benchmark Evaluation After we have the experimental results obtained in section B, based on which, we can introduce in detail how we evaluate each model, which is mainly divided into six parts: Semantic Alignment, Background Consistency, Style Consistency, Instance Consistency, Instance Integerity, and BLEU4.

- • Semantic Alignment Since we use the dataset’s captions as part of the prompts, we leverage CLIPViT-B-32 to encode the given captions and the model-generated images from the validation set. We then compute the similarity between their feature vectors. This similarity serves as our Semantic Alignment score.
- • Background Consistency We employ an open-domain detector, Yolo-World, to detect all instances in the images, focusing on generic classes like characters and animals. After masking these instances and inpainting the masked regions, we obtain background images. By encoding the generated image sequences with CLIP-ViT-B-32, we calculate the similarity between the background of the reference image (the first image of each scene in the validation set) and the backgrounds of subsequent images. The average similarity score gives us the Background Consistency score.
- • Style Consistency We use dino-vits16 to encode all images in the generated sequence. We then compute the feature similarity between consecutive frames and take their average to obtain the Style Consistency score.
- • Instance Integrity Similar to the Background Consistency calculation, we use Yolo-World to obtain bounding boxes for all instances. Using the bounding box of characters from the reference image, we encode these regions with dino-vits16 to get the base features. For the subsequently generated images, we encode the bounding boxes of characters to obtain their visual features. We then construct a similarity matrix between these features and the base features. Specifically,

Instance Integrity Score = k

Similarity(ficurrentk , fjbasek ) len(Fbase)

(1)

where (ik, jk) are the optimal matching index pairs found using the Hungarian algorithm on the cost matrix C.

- • BLEU4 The BLEU4 score is calculated by comparing the caption provided in the prompt with the caption generated by BLIP2 for the model-generated image.

The similarity between all feature vectors is calculated using the following formula:

1 N

Similarity =

N

fifeature1 · (fifeature2)T (2)

i=1

where fifeature1 and fifeature2 are the feature vectors from feature set 1 and feature set 2, respectively, and N is the total number of feature vectors. The dot product fifeature1 · (fifeature2)T calculates the similarity between each corresponding pair of feature vectors, and the mean of these similarities provides the overall similarity score.

Human Evaluation To better assess human preferences for the generated stories, we conducted a human evaluation using several criteria: Align, Style, Consistency, Character, Plot Continuity, Image Quality, and Preference Evaluation.

[Figure 64]

Figure 9: Here are the results of evaluating different models. The first image is the reference image. We used the captions of the last three images from our validation set as part of the prompt input to each model. These three captions are: "a female in sunglasses is driving a vehicle while sipping from a cup," "a female operator of a vehicle with her hand0 and hand1 placed firmly on the steering wheel," and "woman0 is capturing a self-portrait in front of a structure."

- • Align Participants were given each scene’s image and the corresponding text prompt. They rated the alignment between the image and the text on a scale of 1 to 5, with 1 being the lowest and 5 the highest. The participants were not informed about which model generated the images.
- • Style Participants were shown the image sequences for each story and rated the stylistic similarity between the images in the sequence on a scale of 1 to 5. The rating scale was the same, and the participants were not aware of the model origins.
- • Consistency Different story text and image sequences were presented to participants, who rated the consistency between the story text and the generated images on a scale of 1 to 5. The identity of the generating model was kept anonymous.
- • Character For each story, participants evaluated the consistency of the same subject across different images. The primary focus was on whether the subject remained consistent with previous appearances. Ratings were given on a scale of 1 to 5, without knowledge of the image’s source model.
- • Plot Continuity To ensure that the generated scenes were logically connected, participants rated the logical coherence between different scenes within each story based on both text and images. The rating scale was 1 to 5, and participants were blinded to the model generating the images.
- • Image Quality The quality of the images is crucial for story generation. Participants evaluated the aesthetic quality of the generated images on a scale of 1 to 5, again without knowing the source model.
- • Preference Evaluation Participants were asked to provide an overall assessment of the images generated for each story. They viewed image sequences of the same story generated by different models, without knowing which model produced each sequence, and selected their preferred image sequence based on personal preference.

#### B.2 Ablation Studies of Pipeline

Semantic

Alignment ↑ Perplexity ↓ BLIP2 Captions 0.228 38.024

Refined Captions 0.262 29.045 Table 5: Performance comparison of BLIP2 and Refined Captions in terms of Semantic Alignment and Perplexity.

To validate the effectiveness of LLM-refined captions, we measured the semantic similarity between the captions and images both before and after refinement. Additionally, we assessed the perplexity of the captions prior to refinement. The results are shown in Table 5. It can be seen that using LLM to refine the caption can make the caption more narrative and more accurate in describing the image.

