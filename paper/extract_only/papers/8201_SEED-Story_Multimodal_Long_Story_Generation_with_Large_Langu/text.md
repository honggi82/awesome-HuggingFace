### SEED-STORY: MULTIMODAL LONG STORY GENERATION WITH LARGE LANGUAGE MODEL

##### Shuai Yang 1∗ Yuying Ge 2† Yang Li 1 Yukang Chen 4 Yixiao Ge 2,3 Ying Shan 2,3 Yingcong Chen 1,5† 1HKUST(GZ) 2ARC Lab, Tencent PCG 3Tencent AI Lab 4CUHK 5HKUST

#### 1 2 3

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

## arXiv:2407.08683v2[cs.CV]11Oct2024

[Figure 5]

He looked around with a curious expression, wondering what adventures awaited him …

To his surprise, the noise was George‘s friend, a small brown dog …

USER: Here is the beginning of a story about George, Please continue it …

Suddenly, George heard a noise. …

#### 4 5 6 7

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

George and the dog then played a game of hide and seek. George hid behind a couch …

The next day, George and the dog decided to explore the city …

George stopped on the city sidewalk, looking up at the sky …

George then noticed a building with a reflective glass …

8 9 10 11

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

George and the dog stood in front of the building, looking up at the lit windows …

They were in a room with a door, waiting for their friend to join them …

He seemed deep in thought, unaware of George and the dog watching him from below ….

[Figure 15]

Suddenly, the door opened, and a man in a yellow suit walked in …

[Figure 16]

Figure 1: The introduced SEED-Story, powered by MLLM, is capable of generating multimodal long stories based on user-provided images and texts as the story’s beginning. The generated story features rich and coherent narrative texts, accompanied by images that maintain consistency in characters and style. The story can span up to 25 multimodal sequences (see Appendix), even though we only use a maximum of 10 sequences for training.

ABSTRACT

With the remarkable advancements in image generation and open-form text generation, the creation of interleaved image-text content has become an increasingly

∗Worked done at ARC Lab, Tencent PCG †Corresponding Authors

intriguing field. Multimodal story generation, characterized by producing narrative texts and vivid images in an interleaved manner, has emerged as a valuable and practical task with broad applications. However, this task poses significant challenges, as it necessitates the comprehension of the complex interplay between texts and images, and the ability to generate long sequences of coherent, contextually relevant texts and visuals. In this work, we propose SEED-Story, a novel method that leverages a Multimodal Large Language Model (MLLM) to generate extended multimodal stories. Our model, built upon the powerful comprehension capability of MLLM, predicts text tokens as well as visual tokens, which are subsequently processed with an adapted visual de-tokenizer to produce images with consistent characters and styles. We further propose multimodal attention sink mechanism to enable the generation of stories with up to 25 sequences (only 10 for training) in a highly efficient autoregressive manner. Additionally, we present a large-scale and high-resolution dataset named StoryStream for training our model and quantitatively evaluating the task of multimodal story generation in various aspects. All models, training and inference codes are released in https://github.com/TencentARC/SEED-Story.

- 1 INTRODUCTION

Interleaved image-text data is ubiquitous on the internet, characterized by multiple images interspersed with text. In recent years, there has been a surge of interest in generating interleaved image-text content Tian et al. (2024); Ge et al. (2024); Aiello et al. (2023); Dong et al. (2023); Team (2024), driven by the remarkable advancements in image generation Rombach et al. (2022); Lin et al. (2023); Chen et al. (2023); Patashnik et al. (2021); Wang et al. (2023) and open-form text generation Touvron

- et al. (2023); Taori et al. (2023); Zheng et al. (2023). This has given rise to Multimodal Story Generation, an intriguing and valuable task that involves the generation of a sequence of consecutive storylines along with their corresponding vivid images in an interleaved manner, similar to that of a serialized comic. Different from Personalized Story Visualization Ruiz et al. (2023); Avrahami et al. (2024); Tewel
- et al. (2024), which aims to generate consistent images based on the provided captions following the pattern of text-to-image generation, multimodal story generation poses a more significant challenge due to the complexity of the inputs and the high demands of the outputs. Firstly, this task necessitates a thorough comprehension of interleaved data, where text is not only abstract and narrative in nature, but also deeply intertwined with complex images. The model must be adept at deciphering the intricate relationships between images and texts to maintain a coherent narrative flow. Secondly, this task requires the generation of not only a plausible text plot, but also visually captivating images that are consistent in characters and styles. The model should be capable of achieving coherence in the generation of both text and visuals, ensuring an engaging storytelling output. Recently, Multimodal Large Language Models (MLLMs) Li et al. (2023); Zhu et al. (2023a); Peng

- et al. (2023a); Bai et al. (2023a); Liu et al. (2023c); Zhang et al. (2023); Lin et al. (2023); Laurençon
- et al. (2024) have showcased powerful comprehension abilities in understanding multimodal data, which makes them ideally suited for interleaved image-text content in multimodal stories. Consequently, we introduce SEED-Story, as shown in Figure 1, a novel approach that builds upon the MLLM to harness its comprehension strength, while further equipping it with the capability to generate coherent images align with the narrative texts.

Specifically, following previous work Sun et al. (2023a); Ge et al. (2024), we utilize a pre-trained image tokenizer and de-tokenizer, which can decode realistic images with SD-XL Podell et al. (2023) by taking the features of a pre-trained ViT as input. During training, given the interleaved visual and textual data, we adopt the next-word prediction and image feature regression training objectives to regularize multimodal generation. A fixed number of learnable queries are fed into the MLLM, where the output hidden states are trained to reconstruct the ViT features of the target images. To further enhance the consistency of characters and styles in generated images, we propose de-tokenizer adaptation, where the regressed image features from the MLLM are fed into the de-tokenizer for tuning SD-XL. This adaptation allows for better maintenance of coherence in low-level image details from the de-tokenizer, ensuring a more visually consistent storytelling output.

Furthermore, to enable the efficient generation of coherent long stories, we propose a multimodal attention sink mechanism based on window attention Beltagy et al. (2020), which maintains a fixedsize sliding window on the Key-Value (KV) states of the most recent tokens, as well as the beginning of text tokens, images tokens, and the end of image tokens. We empirically find that retaining these tokens will largely mitigate the model’s failure with window attention when the token length surpasses the cache size, allowing our model to generalize to longer sequences than the training sequence length in an efficient manner. Our model with the proposed multimodal attention sink mechanism can generate long stories with up to 30 multimodal sequences, featuring rich text plots and diverse visual scenarios.

Additionally, we introduce a dataset named StoryStream for training and evaluating multimodal story generation. We design an automatic pipeline that leverages MLLMs to obtain a large-scale and high-resolution dataset featuring a sequence of narrative-rich texts and intriguing images, derived from animated videos. StoryStream is four times larger in terms of data volume compared to the existing largest story dataset Liu et al. (2023a), and it boasts higher image resolution, longer sequence lengths, and more detailed story narratives. We further meticulously design evaluation metrics to assess multimodal story generation, taking into account image style consistency, story engagement, and image-text coherence. The evaluation results demonstrate that our model, SEED-Story, achieves superior performance in these aspects.

In summary, Our contributions are three-fold. (1) We propose SEED-Story, a novel method that leverages an MLLM to generate multimodal stories with rich narrative text and contextually relevant images. (2) We propose multimodal attention sink to enable the efficient generation of long stories with sequence lengths larger than those used during training. (3) We introduce StoryStream, a large-scale dataset specifically designed for training and benchmarking multimodal story generation.

- 2 RELATED WORK

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

StoryGeneration Personalized

Multimodal

Feeling tired from their adventure, George climbed onto the bed, snuggling under the blanket. The room was peaceful, and George was quickly drifting off to sleep.

George followed the man into a colorful hallway. The man was gesturing towards a closed door, and George was curious about what was behind it.

George decided to take the cat for a walk. He held the cat by the collar, looking quite pleased with himself.

George found a cream-colored cat with a collar. He gently stroked the cat's head, glad to have found a new friend.

Here is the beginning of a story about George. Please continue it: The man led George into a nearby building.

[Figure 23]

StoryVisualization

“hiking in the mountains with walking stick”

“drinking hot tea on a rainy night”

[Figure 24]

“meeting a friend in the park” “graduating from the college”

“standing in the street”

[Figure 25]

“a photorealistic illustration of a bunny, full body”

- Figure 2: Comparison between personalized story visualization and multimodal story generation. Contents in grey boxes refer to user’s input. In the former, a sequence of captions are given (referred to as a “story”) for consistent text-to-image generation. By contrast, multimodal story generation involves creating a sequence of consecutive storylines along with their corresponding images.

Personalized Story Visualization v.s. Multimodal Story Generation Personalized Story Visualization, such as Li et al. (2019); Maharana et al. (2021); Maharana & Bansal (2021); Maharana

- et al. (2022); Pan et al. (2024); Rahman et al. (2023); Gong et al. (2023); Liu et al. (2023a); Ruiz
- et al. (2023); Avrahami et al. (2024); Tewel et al. (2024), aims to generate images depicting specified characters engaged in various actions or within different scenes, based on the provided captions (so-called "story") as shown in Fig. 2, which follows the pattern of text-to-image generation. For example, StoryDALL-E Maharana et al. (2022) utilizes pre-trained models augmented with crossattention layers to support story progression from an initial image. Innovations like AR-LDM Pan
- et al. (2024) and Story-LDM Rahman et al. (2023) have introduced auto-regressive diffusion models to create coherent sequences of images, while TaleCrafter Gong et al. (2023) has employed LoRA and optimization techniques to ensure consistent characters throughout complex visual narratives.

- Table 1: Comparison of multimodal story generation datasets. The table provides details on the number of images, their resolution, the total length of visual stories, and the average text length per sentence, which indicates the narrative detail of the text. Note that StorySalon has various size of images and we choose one of the typical sizes presented here.

##### Datasets Number of Resolution Story Avg Text

Images Length Length Flintstones Gupta et al. (2018) 122,560 128 × 128 5 86 Pororo Li et al. (2019) 73,665 128 × 128 5 74 StorySalon Liu et al. (2023a) 159,778 432 × 803 14 106 StoryStream 257,850 480 × 854 30 146

Multimodal Story Generation (Shen & Elhoseiny (2023)) aims to generate a sequence of consecutive storylines along with their corresponding images, similar to that of a serialized comic. To achieve this, a model must be capable of predicting reasonable story developments and generating corresponding illustrations, by incorporating the previous results as context in an auto-regressive manner. As shown in Figure 2, the generated story images exhibit rapidly changing backgrounds and characters, different from personalized story visualization where the main character consistently appears in the images.

The task of multimodal story generation presents a more substantial challenge, and we follow previous research Shen & Elhoseiny (2023) to adopt a closed-set setting. We believe the ultimate goal of multimodal story generation should be to generate highly diverse scenarios while also generalizing to unseen characters, which will be explored in our future work.

MLLM for Multimodal Story Generation In the rapidly evolving domain of large language models (LLMs) Touvron et al. (2023); Brown et al. (2020); Chowdhery et al. (2022) and multimodal large language models (MLLMs) Li et al. (2023); Zhu et al. (2023a); Liu et al. (2023d); Peng et al. (2023b); Bai et al. (2023b); Liu et al. (2023b); Zhang et al. (2023); Lin et al. (2023); Sun et al. (2023c); Yu et al. (2023); Ge et al. (2023c;b); Wu et al. (2023); Dong et al. (2023); Zhu et al. (2023b); Sun et al. (2023b); Li et al. (2024), recent work StoryGPTV Shen & Elhoseiny (2023) explores using MLLMs for story generation by converting visual features into token embeddings for image generation, but requires aditional character and object masks for training. MM-interleaved Tian et al. (2024) designs a multi-scale and multi-image feature synchronizer module (MMFS) to process interleaved text-image data and generates next image conditioned on the previous context features from LLM, which makes it difficult to generate long multimodal stories due to the complex multi-scale attention mechanism.

Visual Story Dataset In the landscape of datasets for visual storytelling, various collections have been developed as shown in Table 1. The VIST Huang et al. (2016) dataset is noteworthy for its use of realistic images, though it struggles with maintaining character consistency across stories. Pororo Li et al. (2019) and Flintstones Gupta et al. (2018) datasets, while popular for animation-based story datasets, are hindered by their low resolution and the simplicity of their accompanying texts. Another significant dataset is StorySalon Liu et al. (2023a), which offers high-resolution images and is large in scale, but it lacks global consistency across images. To address these gaps, we introduce StoryStream, a globally consistent, large-scale, high-resolution animated style dataset with engaging, narrative-rich text for complex storytelling, overcoming the limitations of existing datasets.

- 3 METHOD

- 3.1 STORY GENERATION WITH MULTIMODAL LARGE LANGUAGE MODEL

Visual Tokenization and De-tokenization The overview of our method is presented in Figure 3. To effectively extend visual stories, our model must comprehend and generate both images and text. Drawing inspiration from recent advancements in generative MLLMs that unify image comprehension and generation Podell et al. (2023), we develop a multimodal story generation model. Our model employs a pre-trained Vision Transformer Dosovitskiy et al. (2020) (ViT) as the visual tokenizer and a pre-trained diffusion model as the visual de-tokenizer to decode images by using ViT’s features as inputs. Specifically, visual embeddings from the ViT tokenizer are fed into a learnable module, which

Training sample

Target

[Figure 26]

[Figure 27]

[Figure 28]

George, the small monkey, was on an icy adventure…

…The man comforted George…

They stood together against the snowy backdrop…

Training pipeline

- Stage1: Visual Tokeniza on & De-tokeniza on

SEED-Story: Mul modal Large Language Model (LoRA )

|298|
|---|

|16|
|---|

|89|
|---|

|…|
|---|

|…|
|---|

|BOI|
|---|

|EOI|
|---|

|16|
|---|

|218|
|---|

|82|
|---|

|…|
|---|

|…|
|---|

|BOI|
|---|

|EOI|
|---|

|BOS|
|---|

|344|
|---|

|218|
|---|

|82|
|---|

|898|
|---|

|…|
|---|

|BOI|
|---|

|EOI|
|---|

|344|
|---|

|218|
|---|

|82|
|---|

|…|
|---|

|…|
|---|

|BOI|
|---|

|EOI|
|---|

- Stage2: Instruc on Tuning

Regression Loss

|344|
|---|

|218|
|---|

|82|
|---|

|898|
|---|

|…|
|---|

|BOI|
|---|

Next-token prediction

Label Target Image features

Learnable queries

MLLM outputs

The man comfort They stood together

- Stage3: De-tokenizer Adapta on

Image features

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

SDXL Image De-tokenizer

ViT Tokenizer

[Figure 33]

❄ 🔥

[Figure 34]

[Figure 35]

[Figure 36]

George the small

[Figure 37]

🔥

Diffusion MLLM outputs SDXL Image Loss

|[Figure 38]|
|---|

|[Figure 39]|
|---|

[Figure 40]

🔥

De-tokenizer

- Figure 3: Overview of the SEED-Story Training Pipeline: In Stage 1, we pre-trains an SD-XL-based de-tokenizer to reconstruct images by taking the features of a pre-trained ViT as inputs. In Stage 2, we sample an interleaved image-text sequence of a random length and train the MLLM by performing next-word prediction and image feature regression between the output hidden states of the learnable queries and ViT features of the target image. In Stage 3, the regressed image features from the MLLM are fed into the de-tokenizer for tuning SD-XL, enhancing the consistency of the characters and styles in the generated images.

then serves as inputs for the U-Net of the pre-trained SD-XL Podell et al. (2023). This process replaces the original text features with visual embeddings. During this stage, the parameters are optimized using open-world text-image pair data as well as story data to enhance the model’s encoding-decoding capability. After this training phase, we expect the visual tokenizer and de-tokenizer modules to preserve as much image information as possible in the feature space.

Story Instruction Tuning In our instruction tuning process for story generation, we sample a random-length subset of a story data point for each iteration. The model is tasked with predicting the next image and the next sentence of the story text. Within MLLM, all images are converted into image features using a pre-trained ViT tokenizer. For the target text tokens, we perform next-token prediction and use Cross Entropy loss to train for this discrete target. For target image features, the model uses a series of learnable queries as inputs and continuously outputs a series of latents. We then compute the cosine similarity loss between the MLLM’s output and the target image features. During this stage, we fine-tune the SEED-Story model using a LoRA Hu et al. (2021) module.

[Figure 41]

| |
|---|

| |
|---|

| |
|---|

Normal tokens

Sink tokens Evicted tokens

Current token

Attention

- (a) Dense a en on

(d) Mul modal a en on sink

(c) Attention sink

| | |
|---|---|

…

| | | |
|---|---|---|

| | |
|---|---|

- (b) Window a en on

| | |
|---|---|

| | | |
|---|---|---|

…

| | | |
|---|---|---|

| | |
|---|---|

| | |
|---|---|

…

…

…

| | |
|---|---|

…

“0” token Punctua on Beginning of Image (BoI)

End of Image (EoI)

| | |
|---|---|

| | | |BOI|IMG|EOI|
|---|---|---|---|---|---|

| | |
|---|---|

…

…

- Figure 4: Left: Visualization of the attention map when predicting the next token for multimodal story generation. We observe that important attentions are aggregated into the first token of the whole sequence (“0” token), punctuation tokens, tokens adjacent to BoI, and tokens adjacent to EoI. Right: The diagram of (a) Dense attention, which preserves all tokens in KV cache. (b) Window attention, which evicts preceding tokens by a sliding window. (c) Attention sink, which preserves the beginning tokens based on window attention. (d) Multimodal attention sink, which preserves the beginning of text tokens, images tokens, and the end of image token based on window attention. It can efficiently enable our model to generalize to generating longer sequences than the training sequence length.

De-tokenizer Adaptation After instruction tuning, the SEED-Story MLLM effectively produces story images with correct semantics but lacks style consistency and details. We attribute this issue to the misalignment between the latent space of the MLLM output and the image features. To address this, we perform de-tokenizer adaptation for style and texture alignment. In this stage, only the SD-XL image de-tokenizer is trained. Conditioned on the MLLM output embeddings, SD-XL is expected to generate images that are pixel-level aligned with the ground truth. The separate training of the de-tokenizer offers two key advantages. First, it avoids optimization conflicts between the LLM and the de-tokenizer. Second, it conserves memory, making the process executable on GPUs with limited memory. Please see more analysis in Section B of our appendix.

- 3.2 LONG STORY GENERATION WITH MULTIMODAL ATTENTION SINK

Generating long visual stories has substantial potential in various applications, including education and entertainment. However, creating these stories with MLLMs presents significant challenges. Datasets for extended, interleaved stories are not only rare but also impede the training process due to their complexity. To address this, we have to employ a train-short-test-long approach, training models on shorter narratives and extending to longer generations during inference.

Moreover, during inference, generating significantly longer stories than the training data often leads to model degradation, producing lower-quality images, as illustrated in the first row of Figure 8. This process also requires extensive token usage to ensure continuity and coherence, which in turn increases memory and computational demands.

A simplistic solution for this is to use a sliding window technique, depicted in Figure 4 right (b). However, this method disrupts the token relationships in the Key-Value (KV) cache, resulting in subpar generative outcomes, as demonstrated by StreamingLLM Xiao et al. (2023). To overcome this, StreamingLLM introduces an attention sink mechanism that preserves the initial tokens, thus allowing for efficient processing of lengthy generations without quality compromise. While effective in language models, its efficacy diminishes in multimodal contexts, as shown in Figure 4 right (c).

To enhance long multimodal generation, we revisit the attention maps of MLLMs. After conducting numerous experiments (see Section D of the appendix for more details) across various models and cases, we analyze the attention maps across different layers and heads. Our analysis reveals that most

queries predominantly focus on four types of tokens: (1) starting tokens, (2) punctuation tokens, (3) beginning-of-image (BoI) tokens, and (4) end-of-image (EoI) tokens. Unlike language-only models, MLLMs place considerable attention on specific image tokens, particularly those near the BoI and EoI, as illustrated in Figure 4 left.

Building on these insights, we propose a new mechanism for extended generation in MLLMs, termed the multimodal attention sink. During generation, we consistently retain the starting tokens and the image tokens adjacent to the BoI and EoI. Although punctuation tokens receive high attention values, their latent value norms are minimal, contributing insignificantly to the final output, so we do not keep them, as noted by Ge et al. (2023a). Our proposed mechanism enables our model to generate high-quality images while maintaining a low computational footprint.

128

Flintstone

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Length=5Length=5Length=30

128128

Wilma is standing in a room, talking.

Fred is holding his things in the room.

Wilma is walking across the room.

Fred is standing in a room talking while holding a pile of clothes and a hat over one arm.

Wilma is speaking in the room

128

Pororo

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Eddy smiles in front of a jail where his friends are.

A pirate pig runs to the open door. Pororo is hiding.

Eddy runs away and a pirate pig gets up.

A pirate pig gets back to sleep and wakes up again being surprised.

A sleeping pirate pig wakes up.

854

StoryStream

[Figure 52]

[Figure 53]

[Figure 54]

480

They stood together against the snowy backdrop, trying to figure out their next move. George and his friend found themselves standing on an icy platform surrounded by water, with large icebergs in the background.

The snowflakes continued to fall around them, creating a beautiful,

George, the small monkey, was on an icy adventure with his friend, the man in the yellow snowsuit.

albeit cold, scene. Despite the cold, the man comforted George, who was showing signs of concern.

[Figure 55]

[Figure 56]

[Figure 57]

…

Inside an icy cave, the man held George close. They peered curiously into the snowy landscape, cautious of what might be out there.

Suddenly, the man had to run through a snowy canyon. It was a treacherous path with ice floes floating on the water and snowflakes falling from the sky.

George found comfort in the arms of his friend. They stood together, ready to face whatever came next. The man took a moment to rest, seated against the icy backdrop, still wearing his hat and carrying a backpack, ready for their next move.

# …

- Figure 5: Data sample of our StoryStream dataset and existing multimodal story generation datasets. Our multimodal story sequences consist of high-resolution images that are visually engaging, and detailed narrative texts as underlined, closely resembling the real-world storybooks. Additionally, our stories are more extended in length.

(a)

(b) Image Style Consistency (c) Story Engagement (d) Image-Text Coherence

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

SEED wins

- Figure 6: Quantitative evaluation of multimodal story generation between MM-interleaved versus SEED-Story. (a): Histograms displaying FID scores. (b, c, d): GPT-4V is employed to choose a preferred result generated by MM-interleaved and SEED-Story respectively. Pie charts show the win rate, where “Ties” indicates GPT-4V assesses their outcomes with equal scores.

- 4 STORYSTREAM DATASET

- 4.1 DATASET CONSTRUCTION

An ideal source for creating a multimodal story generation dataset is cartoon series, which inherently contain rich plots and consistent character portrayals. We selected three cartoon series to construct our dataset and we present the Curious George in the main body of our paper. The process begins with collecting various series, from which we extract keyframes and their associated subtitles Kilian et al. (2023). Each keyframe is then processed by GPT-4V OpenAI (2023) or Qwen-VL Bai et al. (2023b) to generate a detailed image description. These elements—keyframe, subtitle, and description—are compiled into a single group. We aggregate 30 such groups and input them into GPT-4, supplemented with background information about the cartoon series. Following our instructions, GPT-4 generates high-quality narrative texts suitable for training story generation models.

During dataset construction, we discovered that employing the above chain of thought approach not only produces more accurate narrative text but also speeds up the construction process. Unlike directly feeding all images directly to GPT-4, which is limited to 10 images due to API constrains, our approach produces longer stories. We also significantly improve the model’s understanding of each image by incorporating detailed descriptions. This enhancement in image comprehension enriches the narrative details, providing a richer story generation reference.

- 4.2 KEY FEATURES

Large-scale. Our StoryStream dataset comprises three subsets totaling 257,850 released images. This represents a significant improvement over existing datasets in terms of scale, specific numbers are presented in column 2 of Table 1. To the best of my knowledge, it is the largest visual story generation dataset featuring consistent main characters.

High Resolution. Unlike existing story generation datasets which offer images at a resolution of 128x128, our story stream dataset provides high-resolution images of 480x768.

Narrative Text. Our dataset diverges from existing ones that utilize simple and descriptive language. We offer abstract, narrative, detailed, and story-toned texts that are more akin to real-world applications, such as visualizing narratives from a storybook, examples are shown in Figure 5. Story text of existing datasets obey the form of “name” + “action”, like “Poby is playing the violin.”. Contrarily, our story text involves more intrinsic elements. This effectively enhances the engagement of audiences. An analysis of the average text length per sentence is shown in column 5 of Table 1.

Long Sequence. Moreover, our dataset enhances long story comprehension by offering up to 30 images per story point. Within these 30 images, our corresponding texts present a cohesive narrative, effectively conveying the progression and intricacies of extended stories.

[Figure 62]

1 2 3 4

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

USER: Please continue the story: “On sunny day, George and his friend, the man in the yellow hat, decided to explore their farm.”

- 5 6 7 8 9

George was so excited that he jumped high in the air, his arms stretched out in front of him.

As George landed, he noticed a bee hovering nearby and reached out to touch it, curious about the small creature with stripes

George was right beside a bee, both of them in a grassy field with a building in the background where a small flower grew between the bricks

George was having a great time, smiling and extending his hand towards a bee that was flying near him, with the building and greenery behind him under a clear sky

He signaled to George, who was standing next to a house, that they had to go, with a lush green landscape and a wooden fence suggesting a peaceful, rural setting

George, wearing his yellow hat, looked puzzled but ready for adventure, standing against a simple grassy background with a stone wall partially visible

They hopped into a yellow car, George in the back seat, looking curious, with a building on the right and greenery around as they drove off into the distance

They drove through a picturesque rural landscape, with a large red barn, a small cottle, lush greenery, and a winding dirt road, all under a clear sky

Meanwhile, the man in the yellow hat was in a bit of a pickle, looking worried as he stood in front of a stone wall and a wooden structure, with a small window above him

[Figure 68]

USER: Please continue the story: “After a day brimming with fun, they went back to their cheerful countryside residence, preparing for new adventures.”

1 2 3 4

- 5 6 7 9

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

Next, George captured his friend holding a book, with a vibrant landscape behind them, making the moment last forever.

With a magnifying glass and a camera, George and his friend began their adventure, excited to explore the world around them.

George was thrilled to capture the beauty of the hills, the trees, and the wooden fence under the calm sky.

He took a picture of a quaint house surrounded by lush greenery, noticing small details through his magnifying lens.

8

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

George then took a picture of a child holding a chicken, with a chicken coop in the background, capturing a fun moment on earth.

He continued capturing the vibrant farm with his camera, capting the beauty of the farm, the farmhouse, and the lush greenery.

With a smile, George held his camera, ready to capture another moment, excited to capture more of the world's beauty.

George then took a picture of his friend smiling, standing beside a bushy tree under the clear sky.

Next, George took a picture of four chickens in front of a small house, noticing a hand holding a blue object in the foreground.

- Figure 7: Examples of multimodal story generation from SEED-Story with the same initial image but different texts as the beginning. The top branch starts with the text referencing “the man in the yellow hat”, which leads to the generation of images that include the character. The bottom branch starts without mentioning the man, resulting in a different storyline.

- 5 EXPERIMENT

- 5.1 QUANTITATIVE RESULTS

For quantitative evaluation of multimodal story generation, since there are relatively few established methods for generating multimodal stories, we first fine-tune the recently developed MM-interleaved model on our training dataset for a fair comparison as the baseline model. The quantitative results are listed in Figure 6. The FID is employed to assess the visual quality of the generated images. Additionally, we ask GPT-4V (“gpt-4-turbo-2024-04-09”) to compare and choose a preferred option between each of the generation results of MM-interleaved and SEED-Story across several dimensions: (a) Style Consistency, which evaluates the stylistic uniformity across different images; (b) Story Engagement, which measures the ability of narratives to captivate and maintain audience interest; (c) Image-Text Coherence, which assesses the alignment and relevance between images and their accompanying texts. The details of evaluation are introduced in Section G of Appendix. We also conduct a user study, which compares SEED-Story and MM-Interleaved in Section A of Appendix. The quantitative evaluations demonstrate that SEED-Story outperforms the baseline model in terms of story engagement and image-text coherence, and achieves slightly higher preference in image style consistency. We also provide quantitative comparisons of story visualization in Section E of Appendix.

≤ Length InferenceLength >TrainingLength

Inference Length

Training

≤ Length InferenceLength >TrainingLength

Inference Length

Training

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

attention Window

…

[Figure 94]

[Figure 95]

…

… …

Dense

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

attention Attention

…

[Figure 106]

[Figure 107]

… …

…

… … …

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

…

…

sink Multimodal

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

[Figure 130]

[Figure 131]

Attn.sink

… …

…

- Figure 8: The visualization of generating long stories with different attention mechanisms. Without multimodal attention sink, MLLM cannot generate meaningful long image sequences. As highlighted in the red boxes, other methods produce meaningless images in the later frames.

- Table 2: Quantitative evaluation of long story generation with various attention mechanisms. FID and CLIP scores are calculated by comparing ground truth images with generated images. Inference time and memory usage are calculated by generating 50 sequences multiple times for average.

Metrics FID ↓ CLIP Score ↑ Inference Time (s) ↓ Memory (GB) ↓ Dense Attn 119.72 0.705 569.67 37.99 Window Attn 334.90 0.598 450.64 30.81 Attn Sink 221.53 0.676 451.94 30.81 Multimodal Attn Sink 79.67 0.728 473.98 31.82

- 5.2 QUALITATIVE RESULTS

For qualitative evaluations, we first demonstrate that our SEED-story can effectively generate stories with different plots and corresponding illustrations based on the different beginnings provided by the user, as shown in Fig. 7. We provide more qualitative results of multimodal story generation in Section F of Appendix. As shown in Figure F, Figure G and Figure H, SEED-story can generate long sequences with engaging plots and vivid images. We further provide qualitative comparisons of story visualization in Section E of Appendix.

- 5.3 MULTIMODAL ATTENTION SINK

To verify the effectiveness of multimodal attention sink in long story generation, we conduct an experiment visualizing a long story using the SEED-Story model, but with varying attention mechanisms. We chunk our data into stories of length of 10 considering the training efficiency. We set the window size as the same as the training length. Qualitative results presented in Figure 8 demonstrate that window attention quickly collapses when the inference length exceeds the training length. Both dense attention and attention sink approaches fare better, yet still fail to produce meaningful images as the inference sequence lengthens. In contrast, the multimodal attention sink consistently produces highquality images. In terms of efficiency, the multimodal attention sink exhibits significant improvement over dense attention, with only a modest increase in time and memory costs compared to window attention and vanilla attention sink. These additional costs stem from retaining extra image tokens in the KV cache. Quantitative results presented in Table 2 substantiate the above conclusion.

- 6 CONCLUSION

This work introduces SEED-Story, a pioneering approach that leverages a Multimodal Large Language Model to generate multimodal long stories with rich narrative text and contextually relevant images. We propose a multimodal attention sink mechanism to enable our model to generalize to generating long sequences in an efficient manner. We further present a high-quality dataset named StoryStream for training and benchmarking the task of multimodal story generation effectively.

REFERENCES

Emanuele Aiello, Lili Yu, Yixin Nie, Armen Aghajanyan, and Barlas Oguz. Jointly training large autoregressive multimodal models. arXiv preprint arXiv:2309.15564, 2023.

Chenxin An, Shansan Gong, Ming Zhong, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu. L-eval: Instituting standardized evaluation for long context language models, 2023.

Animaj. Animaj official website, 2024a. URL https://www.animaj.com/#pocoyo/. Accessed: 2024-05-22.

Animaj. Rabbids invasion official youtube channel, 2024b. URL https://www.youtube.com/ @RabbidsInvasion. Accessed: 2024-05-22.

Omri Avrahami, Amir Hertz, Yael Vinker, Moab Arar, Shlomi Fruchter, Ohad Fried, Daniel Cohen-Or, and Dani Lischinski. The chosen one: Consistent characters in text-to-image diffusion models. In ACM SIGGRAPH 2024 Conference Papers, pp. 1–12, 2024.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. 2023a.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond, 2023b.

Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pp. 1877–1901, 2020.

Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis, 2023.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.

Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, et al. Dreamllm: Synergistic multimodal comprehension and creation. arXiv preprint arXiv:2309.11499, 2023.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. Model tells you

what to discard: Adaptive kv cache compression for llms. arXiv preprint arXiv:2310.01801, 2023a. Yuying Ge, Yixiao Ge, Ziyun Zeng, Xintao Wang, and Ying Shan. Planting a seed of vision in large

language model. arXiv preprint arXiv:2307.08041, 2023b. Yuying Ge, Sijie Zhao, Ziyun Zeng, Yixiao Ge, Chen Li, Xintao Wang, and Ying Shan. Making llama see and draw with seed tokenizer, 2023c.

Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation, 2024.

Yuan Gong, Youxin Pang, Xiaodong Cun, Menghan Xia, Haoxin Chen, Longyue Wang, Yong Zhang, Xintao Wang, Ying Shan, and Yujiu Yang. Talecrafter: Interactive story visualization with multiple characters. arXiv preprint arXiv:2305.18247, 2023.

Tanmay Gupta, Dustin Schwenk, Ali Farhadi, Derek Hoiem, and Aniruddha Kembhavi. Imagine this! scripts to compositions to videos. In Proceedings of the European conference on computer vision (ECCV), pp. 598–613, 2018.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Ting-Hao Huang, Francis Ferraro, Nasrin Mostafazadeh, Ishan Misra, Aishwarya Agrawal, Jacob Devlin, Ross Girshick, Xiaodong He, Pushmeet Kohli, Dhruv Batra, et al. Visual storytelling. In Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: Human language technologies, pp. 1233–1239, 2016.

Maciej Kilian, Romain Beaumont, Daniel Mendelevitch, Sumith Kulal, and Andreas Blattmann. video2dataset: Easily turn large sets of video urls to a video dataset. https://github.com/ iejMac/video2dataset, 2023.

Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander Rush, Douwe Kiela, et al. Obelics: An open web-scale filtered dataset of interleaved image-text documents. Advances in Neural Information Processing Systems, 36, 2024.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In Proceedings of the International Conference on Machine Learning (ICML), 2023.

Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814, 2024.

Yitong Li, Zhe Gan, Yelong Shen, Jingjing Liu, Yu Cheng, Yuexin Wu, Lawrence Carin, David Carlson, and Jianfeng Gao. Storygan: A sequential conditional gan for story visualization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6329–6338, 2019.

Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, et al. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575, 2023.

Chang Liu, Haoning Wu, Yujie Zhong, Xiaoyun Zhang, and Weidi Xie. Intelligent grimm–open-ended visual storytelling via latent diffusion models. arXiv preprint arXiv:2306.00973, 2023a.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction

- tuning. arXiv preprint arXiv:2310.03744, 2023b.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction

- tuning. arXiv preprint arXiv:2310.03744, 2023c.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023d.

Adyasha Maharana and Mohit Bansal. Integrating visuospatial, linguistic and commonsense structure into story visualization. arXiv preprint arXiv:2110.10834, 2021.

Adyasha Maharana, Darryl Hannan, and Mohit Bansal. Improving generation and evaluation of visual stories via semantic consistency. arXiv preprint arXiv:2105.10026, 2021.

Adyasha Maharana, Darryl Hannan, and Mohit Bansal. Storydall-e: Adapting pretrained text-toimage transformers for story continuation. In European Conference on Computer Vision, pp. 70–87. Springer, 2022.

OpenAI. Gpt-4v: Optimizing language models for dialogue. https://www.openai.com/ chatgpt, 2023.

Xichen Pan, Pengda Qin, Yuhong Li, Hui Xue, and Wenhu Chen. Synthesizing coherent story with auto-regressive latent diffusion models. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 2920–2930, 2024.

Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, and Dani Lischinski. Styleclip: Textdriven manipulation of stylegan imagery. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 2085–2094, October 2021.

PBS Kids. Curious george official website, 2024a. URL https://pbskids.org/ curiousgeorge/. Accessed: 2024-05-22.

PBS Kids. Curious george official youtube channel, 2024b. URL https://www.youtube. com/@CuriousGeorge. Accessed: 2024-05-22.

Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint

- arXiv:2306.14824, 2023a.

Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint

- arXiv:2306.14824, 2023b.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Tanzila Rahman, Hsin-Ying Lee, Jian Ren, Sergey Tulyakov, Shweta Mahajan, and Leonid Sigal. Make-a-story: Visual memory conditioned consistent story generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2493–2502, 2023.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 22500–22510, 2023.

Xiaoqian Shen and Mohamed Elhoseiny. Storygpt-v: Large language models as consistent story visualizers, 2023.

Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. 2023a.

Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative pretraining in multimodality. 2023b.

Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative pretraining in multimodality. arXiv preprint arXiv:2307.05222, 2023c.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford_alpaca, 2023.

Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv e-prints, pp. arXiv–2405, 2024.

Yoad Tewel, Omri Kaduri, Rinon Gal, Yoni Kasten, Lior Wolf, Gal Chechik, and Yuval Atzmon. Training-free consistent text-to-image generation. ACM Transactions on Graphics (TOG), 43(4): 1–18, 2024.

TheLandBeforeTime. The land before time official youtube channel, 2024a. URL https://www. youtube.com/@TheLandBeforeTime. Accessed: 2024-05-22.

TheLandBeforeTime. The land before time official website, 2024b. URL https:// thelandbeforetime.org. Accessed: 2024-05-22.

Changyao Tian, Xizhou Zhu, Yuwen Xiong, Weiyun Wang, Zhe Chen, Wenhai Wang, Yuntao Chen, Lewei Lu, Tong Lu, Jie Zhou, et al. Mm-interleaved: Interleaved image-text generative modeling via multi-modal feature synchronizer. arXiv preprint arXiv:2401.10208, 2024.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Luozhou Wang, Shuai Yang, Shu Liu, and Ying cong Chen. Not all steps are created equal: Selective diffusion distillation for image manipulation, 2023.

Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal llm. arXiv preprint arXiv:2309.05519, 2023.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023.

Lili Yu, Bowen Shi, Ramakanth Pasunuru, Benjamin Muller, Olga Golovneva, Tianlu Wang, Arun Babu, Binh Tang, Brian Karrer, Shelly Sheynin, et al. Scaling autoregressive multi-modal models: Pretraining and instruction tuning. arXiv preprint arXiv:2309.02591, 2023.

Pan Zhang, Xiaoyi Dong, Bin Wang, Yuhang Cao, Chao Xu, Linke Ouyang, Zhiyuan Zhao, Shuangrui Ding, Songyang Zhang, Haodong Duan, Hang Yan, et al. Internlm-xcomposer: A visionlanguage large model for advanced text-image comprehension and composition. arXiv preprint arXiv:2309.15112, 2023.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023.

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023a.

Jinguo Zhu, Xiaohan Ding, Yixiao Ge, Yuying Ge, Sijie Zhao, Hengshuang Zhao, Xiaohua Wang, and Ying Shan. Vl-gpt: A generative pre-trained transformer for vision and language understanding and generation. arXiv preprint arXiv:2312.09251, 2023b.

APPENDIX

- A USER STUDY

|Criteria|MM-Interleaved<br><br>|SEED-Story|Ties|
|---|---|---|---|
|Image Quality|21.3%<br><br>|66.2%<br><br>|12.5%|
|Image Style Consistency|7.5%|78.8%<br><br>|13.7%|
|Image Diversity|21.3%<br><br>|75.0%<br><br>|3.7%|
|Story Engagement<br><br>|37.5%<br><br>|55.0%|7.5%|
|Image-Story Coherence<br><br>|5.0%|86.3%|8.7%|

Table 3: The results of user study between MM-Interleaved, SEED-Story, and Ties

Participants were asked to choose their preference based on image quality, image style consistency, image diversity, story engagement, and image-story coherence. The results were obtained from 125 samples, as shown in Table 3.

The results indicate that SEED-Story clearly outperforms the baseline in image generation ability and text-to-image coherence. Additionally, SEED-Story shows a slightly higher preference in text quality for story generation.

- B ABLATION STUDY ON DE-TOKENIZER ADAPTATION

We find that the generated images before the de-tokenizer adaptation stage exhibit semantic relevance with consistent backgrounds and characters, thanks to MLLM’s context preservation, as shown in Figure A. However, they suffer from texture distortion and inconsistency in style. After de-tokenizer adaptation, the images show improved consistency in style and character appearance. The calculated FID scores in table 4 confirm that de-tokenizer adaptation enhances image quality.

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

[Figure 143]

w/ de-tokenizer adaptation

w/o de-tokenizer adaptation

- Figure A: Generated story images with and without the 3rd stage: de-tokenizer adaptation. The images generated before the third stage preserve semantic information, with mostly correct backgrounds and characters. However, they display low-quality textures and inconsistency in cartoon style. Our third stage effectively enhances these aspects.

|Model<br><br>|FID|
|---|---|
|Before 3rd stage<br><br>|153.93|
|After 3rd stage<br><br>|99.79|

Table 4: FID scores before and after the 3rd stage.

- C IMPLEMENTATION DETAILS

- C.1 VISUAL TOKENIZATION AND DE-TOKENIZATION

For visual tokenization, we use Qwen-VL pre-trained ViT-G. We first resize the image to 448x448 images and then use ViT to produce its feature of length256 with 4096 dimension. (shape: [256,

4096]). Inside the MLLM, we use a Q-Former architecture to process the image embedding. It takse the ViT image feature as key and Value, and conduct attention with its learnable queries. The length of learnable queries are 64. For de-tokenization, we also use a Q-Former architecture to transform the MLLM output to the shape of SD-XL condition embedding.

- C.2 INSTRUCTION TUNING

Instruction tuning data is formatted as follows: for each story, we sample a random length and compute losses on the last sequence (highlighted in red text). The sequence format is structured as:

<bos>[start of the story.][User prompt: ][following sequence 1][following sequence 2][following sequence 3][following sequence 4] ... [target sequence]<eos>

For our language model (LLM), we utilize the LLAMA2-7B pre-trained model and finetune it using LoRA, supported by the peft library. The hyperparameter r is set to 6, and lora_alpha is set to 32. The modules optimized include the q_projection_layer, v_projection_layer, k_projection_layer, o_projection_layer, gate_projection_layer, down_projection_layer, and up_projection_layer. We employ a learning rate of 1 × 10−4 to finetune this model on our dataset across approximately 6 epochs, utilizing 8 NVIDIA-A800 GPUs.

- C.3 DE-TOKENIZER ADAPTATION

In this stage we fully finetune the SD-XL model. The data format is as the same as instruction tuning, but we fix all MLLM params and optimize only the SD-XL. It takes the MLLM output and is asked to produce image correspond to the ground truth. The SD-XL model was trained using 4 NVIDIA-A800 GPUs. A learning rate of 1 × 10−4 was chosen to facilitate gradual weight updates, ensuring stable convergence, while a weight decay of 0.03 was applied for regularization to prevent overfitting. Training was performed using mixed precision (bf16), which significantly reduced memory usage and accelerated the training process without compromising the model’s accuracy. The model underwent three training epochs, balancing the learning of complex patterns against computational resource use, optimized for large-scale datasets and sophisticated model architectures.

- D ANALYSIS OF MULTIMODAL ATTENTION SINK

- D.1 ATTENTION MAP VISUALIZATION

In this section, we present additional visualizations of attention maps. These maps are derived from various model runs, including varying data lengths, attention heads, and layers. The visualizations consistently reveal a pattern of attention focused on “0” tokens, punctuation, tokens adjacent to Begin-of-Image (BoI), and tokens adjacent to End-of-Image (EoI).

- D.2 STATISTICS ON TOKEN’S ATTENTION

To demonstrate that more attention is paid to image tokens adjacent to <BOI> and <EOI>, we analyzed over 5600 attention maps from various models, layers, and input sequences, to identify “key tokens” with the highest mean attention values. For each attention map, we computed the mean attention value across the query dimension for every key and selected the top 10 keys. We then aggregated these results to determine how often each token appeared as a key token. Table 5 shows the tokens with the highest occurrences, supporting our multimodal attention sink mechanism, with most queries focusing on four key token categories.

- • Starting tokens (BOS)
- • Punctuation (“,” “.” and “;” ...)
- • Image tokens near BOI (BOI, IMG04)
- • Image tokens near EOI (from IMG57 to IMG63, added EOI)

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

- Figure B: Visualization of attention maps from various model runs, showcasing attention patterns across different data lengths, attention heads, and model layers. Notably, the maps highlight consistent focus on ’0’ tokens, punctuation, tokens adjacent to Begin-of-Image (BoI), and tokens adjacent to End-of-Image (EoI).

| |BOS|IMG57<br><br>|EOI|IMG04<br><br>|,|IMG60|
|---|---|---|---|---|---|---|
|Key Token Occurrence|4320<br><br>|4320<br><br>|4320|4140<br><br>|4140|4120|
| |IMG61|IMG59|IMG62|IMG63<br><br>|BOI|IMG56|
|Key Token Occurrence|3730|3132|1651<br><br>|607<br><br>|603|361|

Table 5: Key Token Occurrence for Various Tokens

- E STORY VISUALIZATION COMPARISON

Previous story generation approaches primarily utilize diffusion models, focusing on visualizing story images. These models take the previous image and text as input, and then generate only the next image based on the current text prompt. For a fair comparison, we adapt our model to a visualization-only format. For StoryGen Liu et al. (2023a), we also train it to produce images with previous images and texts. For LDM Rombach et al. (2022), we only give it text-image pairs. The visual results are shown in Figure C. SEED-Story model shows better style and character consistency and higher quality compared to baselines. We also showcase the visualization result of our model on Rabbids Invasion and The Land Before Time. Please see Figure D and E. We further conduct a quantitative evaluation in Table 6 to demonstrate our effectiveness.

Table 6: Quantitative evaluation for story visualization.

##### Model FID ↓ CLIP Score↑

LDM 67.29 0.7585 StoryGen 73.74 0.7573 SEED-Story 67.01 0.7793

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

a) And indeed, it had! A small, brown monkey named George was sitting on a nearby tree branch, hugging his legs with

SEED-StoryLDMStoryGenSEED-StoryLDMStoryGen

- a curious expression.
- b) To his surprise, two raccoons, with mouths open and startled expressions, were looking at George. He was only partially visible, with his red hat peeking out from the bush.
- c) Suddenly, a character wearing a red and white shirt, blue pants, and a yellow cap appeared. He stood a red bike with a basket, in the lush green park with trees and a wooden fence.
- d) The character stood next to his red and yellow bicycle in the green park. He raised a hand to wave at George.

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

- a) Suddenly, the man in the yellow hat stopped to look at something oﬀ-screen. George, the curious monkey, also turned to look in the same direc on.
- b) The bellhop con nued his conversa on on the phone while another ﬁgure held wooden planks. The room was ﬁlled with a classic armchair, console table with a ﬂower vase, and a framed pain ng.
- c) Suddenly, there was a spill! An eggs had fallen, and cracked eggs, yolk, and eggshells were sca ered.
- d) George sat on the ﬂoor with the man in the yellow suit and the bellhop in the red coat, amidst sca ered pieces of paper and a broken object.

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

- a) The man looked to the side with a surprised expression. The background appeared to be the interior of a rounded structure with a window.
- b) A small monkey looked up at a person wearing an ou it. In the background, another person stood next to a large dinosaur skeleton inside what appeared to be a museum.
- c) A dinosaur skeleton stood inside a museum, with a woman in a coat gesturing towards it. In the background, a small ﬁgure waved enthusias cally from behind a glass window.
- d) An woman with short hair and a lab coat stood with hands on hips, smirking. The background showed a wall with two light switches.

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

SEED-StoryLDMStoryGen

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

Figure C: Story visualization comparison of SEED-Story and other story visualization methods.

- F MULTIMODAL STORY GENERATION RESULTS

In this section, we present more multimodal story generation results of our SEED-Story. It keeps produce story image and text with high quality. Figure F, Figure G, and Figure H prove our multimodal

[Figure 189]

1 2 3 4

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

USER: Here is the beginning of the story. Please help me visualize the following…

[Figure 194]

1 2 3 4

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

USER: Here is the beginning of the story. Please help me visualize the following…

[Figure 200]

Figure D: Story visualization result on Rabbids Invasion data.

1 2 3 4

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

USER: Here is the beginning of the story. Please help me visualize the following…

[Figure 206]

1 2 3 4

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

USER: Here is the beginning of the story. Please help me visualize the following…

[Figure 212]

Figure E: Story visualization result on The Land Before Time data.

long story generation capabilities. SEED-story can generate long sequences with engaging plots and vivid images.

- G DETAILS ABOUT GPT-4V EVALUATION

- G.1 COMPARATIVE EVALUATION

To evaluate the effectiveness of MM-interleaved and SEED-Story in multimodal story generation, we initiate an experiment where each model produces a story of five segments, based on a common starting image and text. The segment limit is set to five to accommodate the constraints of GPT-4V, which can handle a maximum of ten images per input session. In total, we generate 180 stories for assessment. For evaluation, we employ GPT-4 or GPT-4V to determine which model produces the better story in each case, based on the framework established in L-Eval An et al. (2023). We calculate the win rate for each model to determine its performance relative to its counterpart. The prompt we used is shown below.

“Please act as an impartial judge and evaluate the quality of the generation story contents provided by two AI assistants. Your job is to evaluate which assistant’s generation is better. Your evaluation should consider {the style consistency of the

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

George stood next to a duck in front of a barn, surrounded by a grassy field and trees. The sky was blue with a few clouds.

A duck stood in the foreground, with a barn and trees in the background. The sky was blue with a small cloud.

George, wearing a puzzled expression, stood in front of a rural landscape with a large barn and a small house.

The background showed green fields, trees, and a blue sky with a single cloud.

[Figure 218]

USER: Here is the beginning of a story about George, Please continue it …

5 6 7 8 9

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

A man in overalls and a hat was holding a bucket, standing next to a pile of dirt. George and a duck were there also, with a small house in the background.

George, with a curious expression, stood in front of a house with a barn in the background. The sky was blue with a small cloud.

A man in overalls was kneeling beside a pile of dirt, holding a shovel. George and a duck were there also, looking at the house in the background.

A duck stood in a grassy area, with a barn and trees in the background. The sky was blue with a small cloud.

George stood in a farm setting, looking at a small puddle. Behind him, a barn and a house were visible, along with greenery and a fence

10 11 12 13 14

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

A man in overalls was digging a hole with a shovel in a grassy area. George and a duck were there also, looking on with curiosity. A small house sat in the background.

A figure in overalls was holding a stick, standing next to a small house in a grassy area. There was a large bush on the side.

George stood in a farm setting, looking towards a barn and a house. A puddle of water was on the ground in front of him, with greenery and a fence behind.

George stood in a grassy area, looking at a small puddle of water. Behind him, a barn and a house were visible, along with a wooden fence and trees.

A man in a hat and overalls was holding a stick, standing next to George. Both were looking at a muddy puddle in a grassy area, near a house and a tree.

16 17 18 19

15

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

A character dressed in a suit and hat was standing in front of a rural landscape with a barn and a tree.

A man in a hat and shirt stood outside, looking surprised or confused. A tree and a building were visible in the background.

A man in a shirt and hat stood outside, looking surprised. George, the monkey, held onto the man's leg, looking up at him. A barn, a house, and a tree were in the background.

A man in a hat and overalls was kneeling in the grass, reaching out to George. A house and a tree were in the background.

George, the cartoon monkey, looked surprised or worried. The background showed a grassy field with a small puddle of water and a tree trunk.

20 21 22 23 24

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

A man in a hat and shirt looked surprised, standing in front of a barn and a tree. His mouth was open, and one hand was raised to his head.

George, the cartoon monkey, stood in a grassy area, looking curious. Behind him, a small puddle of water and a tree were visible. George appeared surprised or intrigued, with wide eyes and raised eyebrows.

George, the monkey, stood on a grassy field, looking curiously at a muddy puddle. Near the puddle, a pair of boots and a hat were visible. Behind, a fence and lush trees suggested a rural setting.

A man in a hat and shirt looked surprised, standing in front of a barn and a tree.

A man in a hat and shirt stood in front of a barn, looking surprised with one hand on the chest. A tree and grassy hill were visible in the background.

###### Figure F: Multimodal long story generation results of SEED-Story.

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

George stood near the parked car, looking up at the man in the yellow hat. They were surrounded by trees and a blue sky.

Soon, George and the man in the yellow hat were back in their car, driving through a green tunnel. The car's interior was a comforting sight.

They drove along a curved road surrounded by grass and trees. The sky was clear, and George's heart was full of excitement.

Suddenly, George noticed a cat lying on the road. He crouched near the car's front wheel, watching the cat.

[Figure 244]

USER: Here is the beginning of a story about George, Please continue it …

5 6 7 8 9

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

George, with his big eyes and mischievous smile, stood ready for the next adventure.

The man in the yellow hat smiled as he looked at George, who was standing beside the car. The grassy hills and trees in the background added to the beauty of the scene.

George and the man in the yellow hat stood beside the parked car in the green landscape. They were ready for their next adventure.

Their journey led them to a vibrant landscape with a large slide in the foreground. A slide extended into a pool, inviting them to play.

They reached a colorful park with a slide, a ladder, and a playground. The green bushes and trees added to the playful atmosphere.

10 11 12 13 14

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

The amusement park was colorful and bustling with visitors. A large Ferris wheel stood in the foreground, with buildings and greenery in the distance.

They reached a vibrant roller coaster track set against a blue sky with fluffy white clouds. The ride featured red and yellow accents, adding to its charm.

Their journey took them to a colorful amusement park with a towering roller coaster. The park was surrounded by greenery and a cityscape.

The man in the yellow hat and George stood in front of a roller coaster track, looking concerned. They were unsure of what to do next.

George, with his big, curious eyes, looked up at the man in the yellow hat. He was eager to have some fun.

16 17 18 19

15

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

At the end of the day, they returned to their vibrant yellow house. The walls and roof glowed in the twilight, and the trees and hills in the background added to the peacefulness of their home.

On their way home, they took a boat ride. The man held onto the steering wheel, while George and a child peeked out from the boat's window, smiling. They were having a great time.

The next day, George and the man in the yellow hat found themselves on a roller coaster. George appeared excited while the man looked concerned. Their adventure had just begun.

Finally, they returned to their colorful house. The walls and the roof under the clear sky made the house look inviting and peaceful. It was the perfect end to a perfect day.

The man in the yellow hat and George waved goodbye from the doorway of their house. They were ready for another day of adventures.

20 21 22 23 24

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

Finally, George and the man in the yellow hat sat on a bench inside a colorful cage with a bright wheel. The man looked concerned while George appeared curious. They were ready to go home, but their adventurous spirit was ready for the next day.

The man in the yellow hat and George waved goodbye from the doorway of their house. They were ready for another day of adventures in the park, near the river.

Finally, George and the man in the yellow hat sat on a bench inside a colorful cockpit with a steering wheel and control levers. They were ready to go home, but their adventurous spirit was ready for the next day.

Their last ride before heading home was a boat. The man held the steering wheel, while George and a child peeked out from the window. They were all smiling, looking forward to their next adventure.

Their last ride before heading home was a boat. The man held the steering wheel, while George and a child peeked out from a window. They were all smiling, looking forward to their next adventure.

###### Figure G: Multimodal long story generation results of SEED-Story.

1 2 3 4

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

Feeling tired from their adventure, George climbed onto the bed, snuggling under the blanket. The room was peaceful, and George was quickly drifting off to sleep.

The man led George into a nearby building, where they found a cozy room with a wooden bed adorned with flowers. A lamp beside the bed cast a warm glow, making George feel right at home.

George lay back down, resting his head on the pillow. His eyes were wide with curiosity, wondering what new adventures the next day would bring.

Suddenly, George woke up in the middle of the night, looking around curiously. The room was quiet, and George felt a sense of mystery.

[Figure 270]

USER: Here is the beginning of a story about George, Please continue it …

5 6 7 8 9

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

George followed the man into a colorful hallway. The man was gesturing towards a closed door, and George was curious about what was behind it.

George sat on the bed, watching as the man in the yellow suit walked towards a door. He wondered where the man was going, and if their adventure would lead them there.

Behind the door, George found a sleeping cat. He gently pulled the door closed, leaving the cat to sleep in peace.

The next morning, George woke up feeling cheerful. He sat up in bed, looking around the room with a smile, ready for another day of adventures.

George stood in front of a door, ready to push it open and see what was behind it. He was excited about the new discoveries he might find.

10 11 12 13 14

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

George held the cat in his arms, looking at the curious cat. He was enjoying this new experience of being a cat walkkeeper.

George decided to take the cat for a walk. He held the cat by the collar, looking quite pleased with himself.

George decided to take the cat for another walk. He held the cat in his arms, looking at the wall, ready for another adventure.

George found a cream-colored cat with a collar. He gently stroked the cat's head, glad to have found a new friend.

George decided to take the cat for a walk around the building. He held the cat in his arms, looking at the curious cat. They passed by a man in a coat, who smiled at the sight of them.

1 2 3 4

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

Meanwhile, a gray pigeon had found a new perch on a building's edge. It overlooked a small balcony, where a man was enjoying a morning cup of coffee, oblivious to the pigeon's presence.

In a nearby park, a pigeon had found a new friend, George, the curious monkey. They were standing on a wooden ledge, peering over it. Nearby, a taller bird observed them with interest.

George, the small brown monkey, was standing next to a large bird on a ledge. The bird looked content, while George seemed curious. The background showed a building with windows, a perfect playground for a curious monkey.

Elsewhere, a blue-gray pigeon was pecking at seeds on the ground. A person nearby watched intently, their eyes following the bird's every move.

[Figure 286]

USER: Here is the beginning of a story about George, Please continue it …

5 6 7 8 9

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

Meanwhile, a cartoon bird was pecking at some food on the ground. A human hand reached towards the bird, offering a treat.

In a playful mood, George mimicked a pigeon, pecking at a crumb-covered ledge. The pigeon watched him, amused. In the background, a building stood tall, perfect for their little playground.

George was having a great time, leaning on the bird's perch with a smile. The background showed a building, perfect for a little monkey adventure.

George was back on the ledge, this time looking puzzled. The background showed a building, perfect for his little puzzle.

Suddenly, George noticed a small object on the ground. He stood on his two legs, reaching out to pick it up. Nearby, a pigeon rested on a ledge.

10 11 12 13 14

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

Suddenly, George noticed a small object on the floor. He stood on the ledge, looking at it with curiosity. The background showed a building, perfect for his little investigations.

Feeling adventurous, George stood on a city sidewalk, waving with a happy expression. The background showed colorful buildings and a clear sky, perfect for his little adventure.

George stood on a city sidewalk, looking puzzled. The background showed colorful buildings and a street with a parked car, perfect for his little thought bubble.

George sat on a ledge, looking puzzled at a crumby area. The background showed a building, perfect for his little thought bubble.

George stood on a city sidewalk, waving with one hand. The background showed colorful buildings and a clear sky, perfect for his little adventure.

Figure H: Multimodal story generation results of SEED-Story.

story images / the engagement of the story / the coherence of the generated text and images}. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of

the assistants.Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: “[[A]]” if assistant A is better, “[[B]]” if assistant B is better, and “[[C]]” for a tie.”

- G.2 SCORE EVALUATION

We also provide a prompt for directly estimating the performance of the generated results without comparing to others. The prompt we used is shown below. We present the direct estimation score is shown in Table 7

“Please act as an impartial judge and evaluate the quality of the generation story contents provided by an AI assistant. Your job is to give a score out of 10. Your evaluation should consider {the style consistency of the story images / the engagement of the story / the coherence of the generated text and images}. Do not allow the length of the responses to influence your evaluation. Be as objective as possible. After providing your explanation, output your final score by strictly following this format: “[[score]]”, such as “[[7]]”.”

Table 7: GPT4 score evaluation results in 3 different aspects-style consistency, story engaging level, and text-image coherence.

Style ↑ Engaging↑ Coherence↑ SEED-Story 8.61 6.27 8.24

- H STORY VIDEO

To showcase the capabilities of our multimodal generation model, we employ a video generation technique to animate the images. We then synchronize these moving images with audio to create a narrative video, which is available in our supplementary materials.

- I DATA USAGE AND LICENSE

- I.1 CURIOUS GEORGE

Curious George is an animated series featuring George, a curious monkey whose adventures teach preschoolers about math, science, and engineering. Guided by The Man with the Yellow Hat, George explores the world through problem-solving and experimentation, making it a delightful and educational experience for young viewers.

Curious George is released on PBS KIDS PBS Kids (2024a;b), a not-for-profit institution. It is a production of Imagine, WGBH and Universal. Curious George and related characters, created by Margret and H.A. Rey, are copyrighted and trademarked by Houghton Mifflin Harcourt and used under license. Licensed by Universal Studios Licensing LLC. Television Series: ©2024 Universal Studios. The terms of use of them are provided in https://www.pbs.org/about/ about-pbs/terms-of-use/.

Our usage fully comply with the terms of use. 1) Personal Uses Permitted: My project is noncommercial and educational, which aligns with personal uses as outlined by PBS. we are not using the information for commercial purposes or exploiting it in a manner inconsistent with PBS rules. The use is strictly for educational and research purposes within an academic setting. 2) User’s Obligation to Abide By Applicable Law: We will ensure all research activities comply with local laws, particularly those relating to copyright and intellectual property rights. Our use will not involve unauthorized reproduction, distribution, or exhibition that violates Intellectual Property Laws. All data are for research only. 3) Content of Information: We will responsibly use the "Curious George" materials, ensuring that all content used in our research is accurately cited and acknowledged. Any PBS content incorporated into your project will be clearly attributed to PBS.

- I.2 RABBIDS INVASION

“Rabbids Invasion” is a French-American computer-animated TV series that breathes life into the zany antics of Ubisoft’s popular Rabbids video game characters. Created by Jean-Louis Momus and featuring the voice of Damien Laquet, the show is a dynamic blend of humor and adventure tailored for a family audience. Since its debut on August 3, 2013, on France 3, the series has enjoyed multiple seasons and a global reach. The Rabbids are mischievous rabbit-like creatures whose escapades lead them into all sorts of unpredictable and hilarious situations, making “Rabbids Invasion” a delight for both kids and adults alike. Thanks to their release, we derive some subsets from the cartoon series Rabbids Invasion Animaj (2024b;a).

- I.3 THE LAND BEFORE TIME

The Land Before Time, an iconic animated film series created by Judy Freudberg and Tony Geiss and distributed by Universal Pictures, debuted in 1988 with significant contributions from Don Bluth, George Lucas, and Steven Spielberg. This franchise, consisting of an initial film followed by 13 sequels, a TV series, video games, and extensive merchandising, explores the adventures of five young dinosaurs who learn key life lessons about friendship and teamwork through their prehistoric trials. Despite the absence of the original creators in the sequels, the series has continued to captivate audiences, emphasizing themes of community and perseverance across its extensive narrative arc. Thanks to their release, we derive some subsets from their websites TheLandBeforeTime (2024a;b).

- I.4 APPRECIATION

Leveraging the data derived from "Curious George," "Rabbids Invasion," and "The Land Before Time," we have significantly advanced the capabilities of our story generation models. This progress has direct and impactful implications for children’s education by enhancing their imaginative faculties and fostering a keen interest in learning. By integrating elements from these animated series into our models, we not only engage young minds but also deepen their affection for animated storytelling. Consequently, this not only meets but also amplifies educational objectives, such as improving literacy and cognitive skills through enjoyable and interactive content. The successful application of data from these beloved animations in our research exemplifies how academic pursuits can harmoniously blend with educational entertainment, ultimately delivering multifaceted benefits that extend well beyond conventional learning environments.

Lastly, we extend our profound appreciation to the creators and maintainers of "Curious George," "Rabbids Invasion," and "The Land Before Time," each a rich and vibrant resource that has significantly contributed to the scope and success of our research. The engaging narratives and characters from these series, especially the ever-curious George, the mischievous Rabbids, and the adventurous dinosaurs from The Land Before Time, have provided invaluable data that enhanced our narrative generation models. This project benefited immensely from the educational and entertaining content crafted with meticulous attention to detail, fostering imagination and learning in young audiences. We acknowledge the pivotal role that these animated series have played in advancing academic research aimed at educational technology. The commitment of the teams behind these beloved series to fostering curiosity and learning is both inspiring and exemplary. We are immensely grateful for the opportunity to incorporate such cherished resources into our scholarly work.

- J BROADER IMPACTS

This project may potentially produce copyrighted content, particularly when used inappropriately or without adherence to existing intellectual property laws. To mitigate this risk, we will implement a rigorous compliance framework that respects the copyrights of third parties. This involves setting strict usage licenses that align with the legal standards dictated by our data sources. Our aim is to protect intellectual property rights while fostering innovation and ethical use of our technology. We also commit to educating users on the importance of respecting intellectual property rights when using our technology. This will be achieved through detailed user guidelines, training sessions, and readily available support to help users understand and navigate the complexities of copyright laws. By taking these measures, we aim not only to comply with legal standards but also to promote a

culture of respect for intellectual property within our user community, thereby contributing positively to the broader digital ecosystem.

- K LIMITATIONS

Lack of Realistic Data Experimentation: This limitation points to a potential gap in the validation of the SEED-Story model under practical, real-world conditions. Without experiments using realistic data, it’s difficult to ascertain how the model would perform in scenarios that are not perfectly controlled or that deviate from the training conditions. This can be crucial, especially in applications like storytelling where the context and variability of real-world data play significant roles. A possible solution would be to incorporate a broader range of test conditions, including noisy data or data from "in-the-wild" storytelling scenarios, to evaluate the robustness and adaptability of the model.

Training on a Non-Diverse Dataset: The second limitation is the restriction of the model’s training to animation datasets which does not cover a large scale or diverse styles. This can severely limit the model’s ability to generalize and produce outputs in styles that are not represented in the training data. This is particularly limiting in creative tasks such as storytelling where the ability to adapt to various artistic and narrative styles is crucial. To mitigate this, expanding the dataset to include a wider array of styles, genres, and visual aesthetics could be beneficial.

