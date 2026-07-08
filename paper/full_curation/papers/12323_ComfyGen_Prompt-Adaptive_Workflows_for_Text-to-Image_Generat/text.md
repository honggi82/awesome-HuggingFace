# arXiv:2410.01731v1[cs.CV]2Oct2024

## COMFYGEN: PROMPT-ADAPTIVE WORKFLOWS FOR TEXT-TO-IMAGE GENERATION

Rinon Gal NVIDIA, Tel Aviv University

Adi Haviv Tel Aviv University

Yuval Alaluf Tel Aviv University

Amit H. Bermano Tel Aviv University

Daniel Cohen-Or Tel Aviv University

Gal Chechik NVIDIA

Input Prompt Standard Approach

[Figure 1]

[Figure 2]

“Envision a breathtaking waterfall cascading into a crystal-clear pool. The pool is home to elegant swans…”

T2I Model

ComfyGen

[Figure 3]

[Figure 4]

[Figure 5]

| |[Figure 6]|
|---|---|
| | |

[Figure 7]

Large Language Model

[Figure 8]

[Figure 9]

Workflow JSON

ComfyUI

Figure 1: The standard text-to-image generation flow (top) employs a single monolithic model to transform a prompt into an image. However, the user community often relies on complex multimodel workflows, hand-crafted by expert users for different scenarios. We leverage an LLM to automatically synthesize such workflows, conditioned on the user’s prompt (bottom). By choosing components that better match the prompt, the LLM improves the quality of the generated image.

ABSTRACT

The practical use of text-to-image generation has evolved from simple, monolithic models to complex workflows that combine multiple specialized components. While workflow-based approaches can lead to improved image quality, crafting effective workflows requires significant expertise, owing to the large number of available components, their complex inter-dependence, and their dependence on the generation prompt. Here, we introduce the novel task of prompt-adaptive workflow generation, where the goal is to automatically tailor a workflow to each user prompt. We propose two LLM-based approaches to tackle this task: a tuningbased method that learns from user-preference data, and a training-free method that uses the LLM to select existing flows. Both approaches lead to improved image quality when compared to monolithic models or generic, prompt-independent workflows. Our work shows that prompt-dependent flow prediction offers a new pathway to improving text-to-image generation quality, complementing existing research directions in the field.

1 INTRODUCTION

As the field of text-to-image generation (Rombach et al., 2022; Ramesh et al., 2021) matures, researchers and practitioners shift from simple, monolithic workflows to more complex ones. Instead

of relying on a single model to produce an image, those advanced workflows combine a variety of components, or blocks, designed to enhance the quality of the generated image (AUTOMATIC1111,

- 2022; Zhang, 2023; comfyanonymous, 2023). These components may include fine-tuned versions of the generative model, large language models (LLMs) for refining the input prompt, LoRAs (Luo et al., 2023; Ryu, 2023) trained to correct poorly generated hands or to introduce specific artistic styles, improved latent decoders for creating finer details, super resolution blocks, and more.

Importantly, effective workflows are prompt-dependent. The choice of blocks often depending on the text prompt and the content of the image being created. For example, a workflow aimed at emulating nature photographs may elect to use a model fine-tuned for photorealism, while workflows focused on generating human images often contain the term “bad anatomy” as a negative prompt or leverage specific super-resolution models that also correct distorted facial features, such as the eyes. Due to the richness of available blocks and complexity of workflows, building a well-designed workflow often requires considerable expertise.

In this work, we propose to learn how to build text-to-image generation workflows, conditioned on a user prompt. Specifically, we propose to leverage an LLM to take as input a prompt describing an image, and output a workflow that is specifically tailored to that prompt. Below, we outline two approaches to achieving this goal. The prompt-specific workflow can then be used to synthesize images for that prompt, resulting in improved quality compared to using fixed base models or popular human-crafted workflows. Importantly, using an LLM enables the model to leverage its extensive prior knowledge to parse the prompt and match its content to the most appropriate blocks.

To represent flows, we build on ComfyUI (comfyanonymous, 2023), a widely used tool that stores workflows as JSON files, which can be easily parsed by recent LLMs. The popularity of ComfyUI also provides access to multiple human-created workflows, which we then augment to create a more diverse training set. To teach the LLM the link between flow components and image quality, we collect 500 diverse prompts from human users.1 These prompts are used for generating images using each workflow in our training set, and the results are scored by an ensemble of aesthetic predictors and human preference estimators (Kirstain et al., 2023; Xu et al., 2024; Wu et al., 2023b). This process effectively creates a training set composed of triplets of (prompt, flow, score).

We then consider two approaches for matching flows to novel prompts. In the first, we leverage a closed-source LLM, and provide it with a table of flows and their scores across a closed-set of categories automatically derived from our training prompts. This table serves as a context for a followup request, where we ask the LLM to select the flow that is most suitable for a novel prompt. In the second approach, we fine-tune an open LLM (Dubey et al., 2024) so that, given a prompt and an ensemble score, it predicts the flow that achieved that score. During inference, we provide the LLM with an unseen prompt and a target score and ask it to provide us with an appropriate workflow. We name these approaches ComfyGen-IC and ComfyGen-FT respectively. The design choices behind each approach and their motivations are discussed below.

We compare our prompt-adaptive approach against several baselines, including: (1) single-model approaches (the baseline SDXL model (Podell et al., 2024), popular fine-tunes, and a DPOoptimized version (Rafailov et al., 2024; Wallace et al., 2024)), and (2) prompt-independent, popular workflows. ComfyGen outperforms all baselines on both human-preference and prompt-alignment benchmarks, highlighting the benefit of prompt-dependent flows.

Finally, we analyze the workflows selected by our method, demonstrate their relation to the domains represented in the input prompts, and investigate the scaling behaviors of our model.

- 2 RELATED WORK

Improving Text-to-image generation quality. With the growing popularity of text-to-image diffusion models (Rombach et al., 2022; Nichol et al., 2021; Ramesh et al., 2022), a range of works sought to improve the visual quality of their outputs, and their alignment to human preferences.

One approach is to fine-tune pretrained models using curated, high quality datasets or improved captioning techniques (Dai et al., 2023; Betker et al., 2023; Segalis et al., 2023). Instead of collecting data, a range of works use reward models (Kirstain et al., 2023; Wu et al., 2023b; Xu et al.,

1Sampled from https://civitai.com/ after filtering out NSFW content.

2024; Lee et al., 2023) to guide text-to-image generation. This can be done using reinforcementlearning (Black et al., 2024; Deng et al., 2024; Fan et al., 2024; Zhang et al., 2024). However, these methods can be computationally expensive and struggle to generalize effectively. As an alternative, the model can be fine-tuned using differentiable rewards (Clark et al., 2024; Prabhudesai et al.,

- 2023; Wallace et al., 2024). Instead of tuning the model directly, one can also use reward models to explore the diffusion input-noise space (Eyring et al., 2024; Qi et al., 2024), finding seeds for which the output is of higher quality. Finally, some approaches leverage self-guidance (Hong et al.,

2023) or frequency-based feature manipulations (Si et al., 2024; Luo et al., 2024) to drive the model towards more detailed and sharper outputs.

Our work proposes a new, orthogonal path to improving image quality. Instead of modifying the diffusion model or intervening in its sampling process, we use reward models to better match workflow components to a given prompt, aligning the entire pipeline towards human preferences.

LLM-based tool selection and Agents Recent advancements in large language models have demonstrated significant improvements in reasoning abilities and their capacity to adapt to novel content and tasks. This adaptability can be achieved through efficient fine-tuning methods, but more commonly simply through zero-shot prompting or in-context learning.

Building on these capabilities, a range of works proposed to leverage LLMs for tasks beyond text generation. A common line of work aims to equip the LLM with external tools (Schick et al., 2024), either through appropriate API tags within the generated text (Schick et al., 2024), by providing in-context API documentations (Wang et al., 2024; Sur´ıs et al., 2023), model descriptions (Shen et al., 2024) and code samples (Gupta & Kembhavi, 2023), or by retrieving models from a predefined collection. (Wu et al., 2023a). Such tools are often referred to as LLM agents, and their latest variants are often equipped with components such as memory mechanisms, retrieval modules or self-reflection and reasoning steps, all aimed at improving their overall performance.

Our work can similarly be viewed as an agent, as it employs an LLM to directly select and connect external tools. Here, we focus on the novel task of prompt-adaptive pipeline creation, and on tapping this under-explored path to improving the quality of downstream generations.

Worfklow generation An emerging trend in machine learning research is the use of compound systems, where multiple models are used in collaboration to achieve state-of-the-art results. These systems have been successfully used across various domains, ranging from coding competitions (AlphaCode Team, 2024) to olympiad-level problem solving (Trinh et al., 2024), medical reasoning (Nori et al., 2023) and video generation (Yuan et al., 2024). Crafting such compound systems can be a daunting task, as the components must be carefully selected and their parameters tuned to perform well in tandem, rather than optimized on each individual step of the task. To address this, recent approaches have proposed optimization-based frameworks that tune pipeline parameters for improved end-to-end performance (Khattab et al., 2023), or even optimize the connections within a graph representing the components of a complex system (Zhuge et al., 2024).

Our work similarly tackles the task of pipeline generation. Here, we focus on text-to-image models, and demonstrate that their performance can be enhanced by designing compound pipelines that depend on the user’s prompt.

- 3 METHOD

Given an input prompt describing an image, our goal is to match it with an appropriate text-to-image workflow, leading to improved visual quality and prompt alignment. We hypothesize that effective workflows will depend on the specific topics present in the prompt. Therefore, we propose to tackle this task by leveraging an LLM that can concurrently reason over the prompt and identify these topics, while also serving as a means to directly select or synthesize the new flow.

In the following section, we provide a detailed description of ComfyUI and our method, focusing on our training data, as well as our retrieval-based and score-based tuning approaches.

“a boat in the middle of the cherry blossoms, lush scenery, fairycore, cute and dreamy, pink and green...”

Histogram of Flow Scores

[Figure 10]

[Figure 11]

[Figure 12]

30k

25k

20k

Frequency

Score: 0.14 Score: 0.42

15k

[Figure 13]

[Figure 14]

10k

5k

0

0.1 0.2 0.3 0.4 0.5 0.6 0.7

Flow Score

Score: 0.56 Score: 0.66

(a) (b) (c)

- Figure 2: (a) A simple ComfyUI pipeline using a base model and a face restoration block, as well as both a positive and a negative prompt. (b) Distribution of scores for the prompt, flow pairs in our training set. (c) Example images produced for the same prompt by flows with different scores. A higher score typically correlates with more detailed and vibrant results, and fewer artifacts.

- 3.1 COMFYUI

Our work focuses on ComfyUI, an open-source software for designing and executing generative pipelines. In ComfyUI, users create pipelines by connecting a series of blocks that represent specific models or their parameter choices. In fig. 2a, we show a simple example ComfyUI pipeline. This pipeline includes blocks for loading a model, specifying prompts and latent dimensions, a sampler, a VAE decoder, and a face restoration upscaling model. More complex pipelines may involve additional components like LoRAs (Ryu, 2023) or embeddings (Gal et al., 2022), ControlNets (Zhang

- et al., 2023), IP-Adapters (Ye et al., 2023), blocks that re-write and enhance the input prompt, and more. In many cases, complex blocks are introduced into ComfyUI through user-created extensions, which are then shared across the community.

Importantly, each ComfyUI pipeline can be exported to a JSON file which outlines both the graph nodes and their connectivity. ComfyUI’s standard JSON format also contains UI information, such as the position and color of the blocks. We use the simpler API version which excludes this UIspecific information. Moreover, the API format of the flow can be used to trigger novel generations without using the UI, allowing us to automate much of our process.

We note that the concurrent work of Xue et al. (2024) also leverages ComfyUI pipelines. However, their work focuses on using ComfyUI as a test bed for exploring the stability of collaborative workflow generation approaches. Hence, their evaluation focuses on examining the rate at which ComfyUI-compliant workflows are created. In contrast, we focus on learning to tailor specific workflows to a user’s prompt, with the aim of improving downstream generation quality.

3.2 TRAINING DATA

As a starting point, we collect a set of approximately 500 human-generated ComfyUI workflows from popular generative-resource-sharing websites such as Civitai.com. We limit ourselves to textto-image workflows, flitering out video generation flows, and flows that take a control image as an input. We further discard highly complex flows, whose JSON representations often span tens of thousands of lines. Finally, we discard flows that use community-written blocks appearing in fewer than three flows. This leaves us with a small set of 33 flows, which we augment by randomly switching diffusion models (see supplementary for list), LoRAs and samplers, or changing parameters like the guidance scale and number of steps. In total, this process resulted in 310 distinct workflows.

Recall that our goal is to predict effective flows for a given prompt, which will enhance the quality of the generated output. To achieve this, we need a way to assess workflow performance. To do so, we begin by collecting a set of 500 popular prompts from Civitai.com and using them to synthesize images with each flow in our dataset. These images are then scored using an ensemble of quality prediction models (LAION Aesthetic Score (Schuhmann et al., 2022), ImageReward (Xu

- et al., 2024), HPS v2.1 (Wu et al., 2023b), and Pickscore (Kirstain et al., 2023)). We standardize

the outputs of these models so that they are of approximately the same scale, and sum them up, assigning higher weights to models that better align with human preferences. This process yields a single scalar score for each prompt and flow pair, where higher scores typically correlate with better image quality. Figure 2b,c shows the distribution of scores in our data set, along with visual examples of images created with the same prompt, across a range of scores.

Our final dataset consists of triplets of prompt, workflow, and score. We use these to implement both the in-context and the fine-tuning based approaches detailed below.

- 3.3 COMFYGEN-IC

As a first approach to providing prompt-dependent flows, we look to in-context based solutions that leverage a powerful, closed-source LLM. To do so, we first need to provide the LLM with some knowledge about the quality of results produced by each flow. We thus start by asking the LLM to come up with a list of 20 labels which will best fit our 500 training prompts. These include object-categories (“People”, “Wildlife”), scene categories (“Urban”, “Nature”) and styles (“Anime”, “Photo-realistic”). A complete list of the labels is provided in the supplementary. With these labels in hand, we can now calculate the average quality score of images produced by each flow across all prompts belonging to a specific label. Repeating this for all flows and all labels gives us a table of flows and a measure of their performance across all 20 categories.

Ideally, we would have liked to provide the LLM with the full JSON description of the flows, allowing it to learn the relationships between flow components and downstream performance on specific categories. Unfortunately, the flows are too lengthy to fit more than a handful into the context window of most LLMs. Hence, our table contains only flow identities, and we simply ask the LLM to choose the flow that it believes will perform best on a given, unseen prompt.

All in all, this approach provides us with a classifier capable of parsing new prompts, breaking them down into relevant categories, and selecting the flow that best matches these categories. We name this variation ComfyGen-IC.

- 3.4 COMFYGEN-FT

As an alternative approach, we can fine-tune an LLM to predict high-quality workflows for given prompts. One way to approach this problem could be to simply fine-tune the LLM so that, given an input prompt provided in-context, it would need to predict the flow that achieved the highest score for that prompt. However, this approach has several drawbacks: it significantly reduces the number of training tokens, using only one flow per prompt instead of all 310; it’s more sensitive to randomness in our data creation process, such as the seed chosen for each generated image; and it doesn’t allow learning from negative examples, which could help the model identify ineffective flow components or combinations.

Instead, we propose an alternative fine-tuning formulation where the LLM takes as its context both the prompt and a score. The model is then tasked with predicting the specific flow that achieved the given score for the corresponding prompt. This approach tackles the drawbacks of the best-scoringflow method. First, it greatly increases the number of tokens available for training by utilizing all available data points, not just the highest-scoring ones. Second, it reduces the impact of random fluctuations by considering a wider range of scores and their associated flows. Third, it allows the model to learn from negative examples, i.e., flows that achieved low scores for a given prompt. At inference time, we can simply provide the LLM with a prompt and a high score and have it predict an effective flow for our prompt. We name this variation ComfyGen-FT.

- 3.5 IMPLEMENTATION DETAILS

We implement ComfyGen-IC using Claude Sonnet 3.5, and ComfyGen-FT on top of pre-trained Meta Llama3.1 8B and 70B checkpoints (Dubey et al., 2024). Unless otherwise noted, all ComfyGen-FT results in the paper use the 70B model with a target score of 0.725. In all cases, we finetune for a single epoch using a LoRA of rank 16 and a learning rate of 2e − 4. Additional details are in the supplementary.

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

- Figure 3: Our method can generate higher quality images across diverse domains and styles. Prompts are available in the supplementary.
- 4 EXPERIMENTS

We begin by showcasing images generated with our approach across a range of prompts, including subject-focused, photo-realistic imagery, as well as artistic or abstract creations. These are shown in fig. 3, with additional large-scale figures in the supplementary.

Next, we compare the images produced by our approach with those generated by a series of baselines. We consider two types of alternative approaches: (1) Fixed, monolithic models, where we simply use the prompts to directly condition a pre-trained diffusion model. (2) Generic workflows, where we use the same workflow to generate all images, regardless of the specific prompt.

For (1), we consider the original SDXL model (Podell et al., 2024) and its two most popular (most downloaded on CivitAI) fine-tuned variations: JuggernautXL and DreamshaperXL. We further consider a version of SDXL fine-tuned with DPO (DPO-SDXL, (Wallace et al., 2024)). For (2), we selected the two most popular flows (based on download counts) from our training corpus. These flows use SSD-1B (Gupta et al., 2024) and Pixart-Σ (Chen et al., 2024) respectively.

We evaluate our models and the baselines on two fronts. First, we use the GenEval benchmark (Ghosh et al., 2024), which uses object detection to assess generative models across promptalignment tasks like single-object generation, counting, and attribute binding. Quantitative results

Single Two Attribute Model object object Counting Colors Position binding Overall

Single Model - SDXL 0.98 0.74 0.39 0.85 0.15 0.23 0.55 Single Model - JuggernautXL 1.00 0.73 0.48 0.89 0.11 0.19 0.57 Single Model - DreamShaperXL 0.99 0.78 0.45 0.81 0.17 0.24 0.57 Single Model - DPO-SDXL 1.00 0.81 0.44 0.90 0.15 0.23 0.59

Fixed Flow - Most Popular 0.95 0.38 0.26 0.77 0.06 0.12 0.42 Fixed Flow - 2nd Most Popular 1.00 0.65 0.56 0.86 0.13 0.34 0.59

ComfyGen-IC (ours) 0.99 0.78 0.38 0.84 0.13 0.25 0.56 ComfyGen-FT (ours) 0.99 0.82 0.50 0.90 0.13 0.29 0.61

- Table 1: GenEval comparisons. ComfyGen-FT outperforms all baseline approaches, despite being tuned with human preference scores, and not strictly for prompt alignment.

SDXL Juggernaut DreamShaper Flow 1 Flow 2 ComfyGen-IC ComfyGen-FT

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

“A photo of a cake and a stop sign”

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

“A photo of a bird left of a couch”

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

“A photo of a blue cell phone and a green apple”

- Figure 4: Qualitative results on GenEval prompts. ComfyGen shows better performance on multisubject prompts, colorization and attribute binding, but may struggle with positioning.

are reported in table 1 with qualitative samples shown in fig. 4. Our tuning-based model outperforms all baselines, despite only using human preference scores during training. The in-context approach underperforms. We hypothesize that it suffers due to GenEval’s short, simplistic prompts, which typically only describe a few objects in one or two words each. This challenges the LLM’s ability to match the prompts with labels, and performance degrades.

To better evaluate the visual quality of our images, we turn back to CivitAI, and sample 500 prompts from the 10,000 highest-ranked images on the website, after filtering out prompts used for training our model and prompts that contain nudity or excessive violence. We assess the results both automatically — using HPS V2.0, a model not included in the weighted score used during training and through a human preference study. For HPS, we follow Wallace et al. (2024); Qi et al. (2024) and perform a pair-wise comparison, each time pitting our method against a baseline using the same prompt. We report the fraction of prompts for which our approach received a higher score.

For the user study, we pit each version of our model against each baseline in a two-alternative forcedchoice setup and report the fraction of times our model was preferred over each baseline. We sample roughly 20 prompts for each baseline and ComfyGen version pair, for a total of 231 questions. We collected a total of 682 responses from 35 users. More details are provided in the supplementary.

The results on the CivitAI prompts are shown in fig. 5, with visual samples for our approach and the

- 4 best baselines provided in fig. 6. Both of our approaches outperform all baselines, with notable improvement over simply using the baseline SDXL model. Curiously, we observe that some finetuned versions of SDXL are competitive with fixed flows, further emphasizing the importance of tailoring flows to specific use cases.

##### SDXL

JuggernautXL

- 93% 7%

72% 28%

- 94% 6%

87% 13% 60% 40%

HPS v2.0 User Study HPS v2.0 User Study

HPS v2.0 User Study HPS v2.0 User Study

86% 14% 58% 42%

60% 40%

0% 50% 100% Win Rate

0% 50% 100% Win Rate

DreamShaperXL

##### DPO-SDXL

91% 9% 73% 27%

86% 14% 82% 18%

HPS v2.0 User Study HPS v2.0 User Study

HPS v2.0 User Study HPS v2.0 User Study

91% 9% 82% 18%

88% 12% 68% 32%

0% 50% 100% Win Rate

0% 50% 100% Win Rate

Fixed Flow (1 )

Fixed Flow (2 )

78% 22% 73% 27%

- 73% 27%

60% 40%

- 74% 26%

HPS v2.0 User Study HPS v2.0 User Study

HPS v2.0 User Study HPS v2.0 User Study

78% 22% 87% 13%

59% 41%

0% 50% 100% Win Rate

0% 50% 100% Win Rate

ComfyGen-FT (Ours) ComfyGen-IC (Ours) Monolithic Models Fixed Flows

- Figure 5: HPS V2.0 and User Study win rates. We compare each baseline against both ComfyGenFT (green) and ComfyGen-IC (teal). ComfyGen variants are favored over all baselines.

SDXL Juggernaut DreamShaper Flow 1 Flow 2 ComfyGen-IC ComfyGen-FT

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

“8k digital nature photography, an idyllic landscape with mountains in the distance, shiny ruby see-through, Sprawling labyrinth, bioluminescent”

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

“masterpiece, intricate detail, 8K, HDR, Cross section of a carrot growing in the ground. A scientific perspective on the cross section of a carrot growing”

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

“close-up photography of grey tabby cat, cooking fish, c4ttitude, in glasstech kitchen, hyper realistic, intricate detail, foggy, pov from below”

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

“oil painting, silhouette of a woman in the steppe wilderness, dramatic light, 34K uhd, masterpiece, high detail, 8k, intricate, detailed, high resolution, high res, high quality, highly detailed, Extremely high-resolution details, fine texture”

- Figure 6: Qualitative results on CivitAI prompts. Please zoom in to better appreciate the differences.

- 5 ANALYSIS

Having shown that our approach outperforms existing baselines, we next turn to analyzing its behavior. We examine three aspects of ComfyGen’s performance: (1) the originality and diversity of

the generated flows, (2) whether they show human-interpretable patterns, and (3) the effect of using the target score in the ComfyGen-FT prompts. The findings for these aspects are reported below.

- 5.1 ORIGINALITY AND DIVERSITY

We begin by assessing ComfyGen-FT’s ability to generate novel flows by comparing its predictions for the 500 CivitAI prompts to the nearest neighbors in our training corpus. While ComfyGenIC is retrieval-based and has an expected similarity score of 1.0, ComfyGen-FT achieves 0.9995 similarity, indicating that at the scale of our model, there is little to no generation of unseen flows. This indicates that our fine-tuning approach has also learned to classify flows. However, we note that in contrast to ComfyGen-IC, it has learned so directly from the data with limited ad-hoc choices in the process, and indeed it outperforms ComfyGen-IC in most of our evaluations. Looking ahead, we hope that future methods will also be able to synthesize unseen flows with novel graph structures.

In terms of diversity, we observe that over 500 prompts, ComfyGen-IC makes use of 41 unique flows, while ComfyGen-FT uses 79, indicating a higher diversity. Recall also that our base set contained only 33 human created templates, which were then augmented through random parameter changes. Hence, both variations identified useful flows which differ from the initial human-created set. This suggests that more data or a more involved search over the input parameter space could yield more diverse outputs and possibly improved performance.

- 5.2 ANALYZING THE CHOSEN FLOWS

Next, we want to see whether we can identify any patterns in the chosen flows that would provide useful information about the strengths of existing models. Towards this goal, we want to see which models are popular across different categories, and which ones are especially prevalent for prompts within a particular category.

To identify models most strongly associated with specific labels, we parse all flows selected for our 500-prompt test set. We scan each flow for base models, LoRAs, and upscaling models, appending their names to a to a document corresponding to each label associated with the prompt that generated the flow. Then, we use TF-IDF (Sparck Jones, 1972) to rank the models across these label-documents. In table 2, we report the top-scoring model for each of four distinct labels, as well as the most common models across the entire flow set (“General”).

We observe that, in many cases, the choices make intuitive sense. For example, the GFPGAN face restoration model is closely tied to the “People” category. Similarly, “Anime” prompts make more frequent use of models that better preserve human anatomy, or a LoRA tuned for an anime aesthetic. However, while such patterns exist in the data, the choices are not always intuitively clear. In the future, it may be beneficial to have the LLM explain the reasoning behind its component selections.

Category “People” “Nature” “Anime” “Abstract” General Top Base Model Proteus v3 Stable Cascade JibMixXL v9 “Better Bodies” SDVN7-NijiStyleXL crystalClearXL Top LoRA SDXL FaeTastic v24 Add-Detail XL AnimeTarot LogoRedmond MidJourney52 v1.2 Top Upscaler GFPGAN v1.4 Real-ESRGAN UltraSharp x4 None UltraSharp x4

Table 2: Top workflow components by TF-IDF scores for selected categories. In many cases, selections align with human intuition (e.g., a face upscaling model is favored for the “People” category).

- 5.3 THE EFFECT OF TARGET SCORES

Recall that ComfyGen-FT was fine-tuned to predict a flow based on a given prompt and a target score. Here, we examine the performance of the model according to the target score provided at inference time. To do so, we repeat the CivitAI prompt experiments of section 4, while adjusting the target score used in our prompts. We evaluate both a model tuned from the Llama3.1 8B version and one from Llama3.1 70B. The quality of the generated images is assessed using HPS v2.0, and we report the average outcomes. The results are presented in fig. 7. For reference, we provide the scores of the baseline SDXL model, as well as ComfyGEN-IC. We additionally examine a scenario where instead of tuning the model to predict a flow given a prompt and a score, we simply tune it to predict the highest scoring flow (”Predict best”).

“...red-haired female adventurer in medieval attire standing against a backdrop of a futuristic, geometric, neon-lit landscape, surrealism style, vibrant colors...”

Llama3.1-70b-ComfyGen-FT

0.30

LLama3.1-8b-ComfyGen-FT

ComfyGen-IC

[Figure 73]

[Figure 74]

0.29

Predict best

0.28

HPSV2.0Score()

SDXL

0.27

Target score: 0.296 Target score: 0.467

0.26

[Figure 75]

[Figure 76]

0.25

0.24

0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

Inference Target Score

Target score: 0.596 Target score: 0.725

- Figure 7: (left) Average HPS V2 score on CivitAI prompts as a function of the inference target score. The 8B and 70B variations of our model perform equally well, and significantly outperform the variation trained to predict the highest scoring flow (green). (right) Model outputs for the same prompt at different target scores.

We observe that the ComfyGen-FT model has indeed learned to associate the target scores with flows of varying quality. With an appropriate choice of score (near the top of the training score distribution), ComfyGen-FT achieves comparable performance to ComfyGen-IC. Notably, attempting to predict the best model instead of the score-based tuning leads to greatly diminished performance, highlighting the importance of our approach. We further note that both model sizes achieve comparable performance, hinting that we are far from saturating the capabilities of the models.

Finally, it is important to note that our target scores were calculated using an ensemble of humanpreference predictors, excluding HPS v2.0. Therefore, there should be no expectation for alignment between the values on the y-axis and the x-axis.

- 6 LIMITATIONS

While ComfyGen’s prompt-dependent workflow approach demonstrates improvements over monolithic models and constant flows, it is not free of limitations. Our current model is limited to text-toimage workflows, and cannot address more complex editing or control-based tasks. However, this could potentially be resolved in the future through the use of vision-language models (VLMs).

Similarly, both of our approaches require us to generate images using a large number of flows. With typical generations taking an order of 15 seconds, even a modest set of 500 prompts and 300 flows requires a month of GPU time to create. Therefore, scaling up the approach would likely require significant computational resources or more efficient ways (e.g., Reinforcement Learning) to explore the flow parameter space.

Finally, each of our two methods has its own unique drawbacks. The fine-tuning approach cannot easily generalize to new blocks as they become available, requiring retraining with new flows that include these blocks. On the other hand, the in-context approach can be easily expanded by including the new flows in the score table provided to the LLM. However, this increases the number of input tokens used, making it more expensive to run and eventually saturating the maximum context length. We hope that these limitations can be addressed through more advanced retrieval-based approaches or through the use of collaborative agents.

- 7 CONCLUSIONS

We introduced the task of prompt-adaptive workflow generation, and presented ComfyGen - a set of two approaches that tackle this task. Our experiments demonstrate that such prompt-dependent flows can outperform monolithic models or fixed, user created flows, in a sense providing us with a new path to improving downstream image quality.

In the future, we hope to further explore prompt-dependent workflow creation methods, increasing their originality and expanding their scope to image-to-image or even video-related tasks. Perhaps in the future we could collaborate with the language model on the creation of such flows, providing it feedback through additional instructions or examples of outputs, thereby enabling non-expert users to further push the boundary of content creation.

- 8 ACKNOWLEDGEMENTS

We would like to thank Yotam Nitzan, Ron Mokady, Yael Vinker and Linoy Tsaban for providing feedback on an early version of this manuscript or its figures. We thank Shahar Sarfaty for his assistance in collecting user prompts, Assaf Shocher, Yoad Tewel and Tomer Wolfson for useful discussions, and Yaki Tebeka for his assistance with compute infrastructure.

A very special thanks to Or Patashnik for always lending a willing ear, and for providing moral support throughout the project.

REFERENCES

Google DeepMind AlphaCode Team. Alphacode 2 technical report. https://storage. googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report. pdf, 2024.

AUTOMATIC1111. Stable diffusion web-ui. https://github.com/AUTOMATIC1111/ stable-diffusion-webui, 2022.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In The Twelfth International Conference on Learning Representations, 2024.

Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-\sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. arXiv preprint arXiv:2403.04692, 2024.

Kevin Clark, Paul Vicol, Kevin Swersky, and David J Fleet. Directly fine-tuning diffusion models on differentiable rewards. In The Twelfth International Conference on Learning Representations, 2024.

comfyanonymous. Comfyui. https://github.com/comfyanonymous/ComfyUI, 2023. Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon

Vandenhende, Xiaofang Wang, Abhimanyu Dubey, et al. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807, 2023.

Fei Deng, Qifei Wang, Wei Wei, Tingbo Hou, and Matthias Grundmann. Prdp: Proximal reward difference prediction for large-scale reward finetuning of diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7423–7433, 2024.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Luca Eyring, Shyamgopal Karthik, Karsten Roth, Alexey Dosovitskiy, and Zeynep Akata. Reno: Enhancing one-step text-to-image models through reward-based noise optimization. arXiv preprint arxiv:2406.04312, 2024.

Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Reinforcement learning for finetuning text-to-image diffusion models. Advances in Neural Information Processing Systems, 36, 2024.

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36, 2024.

Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14953–14962, 2023.

Yatharth Gupta, Vishnu V Jaddipal, Harish Prabhala, Sayak Paul, and Patrick Von Platen. Progressive knowledge distillation of stable diffusion xl using layer level loss. arXiv preprint arXiv:2401.02677, 2024.

Susung Hong, Gyuseong Lee, Wooseok Jang, and Seungryong Kim. Improving sample quality of diffusion models using self-attention guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7462–7471, 2023.

Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T Joshi, Hanna Moazam, et al. Dspy: Compiling declarative language model calls into self-improving pipelines. arXiv preprint arXiv:2310.03714, 2023.

Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview. net/forum?id=G5RwHpBUv0.

Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu. Aligning text-to-image models using human feedback. arXiv preprint arXiv:2302.12192, 2023.

Simian Luo, Yiqin Tan, Suraj Patil, Daniel Gu, Patrick von Platen, Apolin´ario Passos, Longbo Huang, Jian Li, and Hang Zhao. Lcm-lora: A universal stable-diffusion acceleration module, 2023.

Yang Luo, Yiheng Zhang, Zhaofan Qiu, Ting Yao, Zhineng Chen, Yu-Gang Jiang, and Tao Mei. Freeenhance: Tuning-free image enhancement via content-consistent noising-and-denoising process. arXiv preprint arXiv:2409.07451, 2024.

Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.

Harsha Nori, Yin Tat Lee, Sheng Zhang, Dean Carignan, Richard Edgar, Nicolo Fusi, Nicholas King, Jonathan Larson, Yuanzhi Li, Weishung Liu, et al. Can generalist foundation models outcompete special-purpose tuning? case study in medicine. arXiv preprint arXiv:2311.16452, 2023.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=di52zR8xgf.

Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, and Katerina Fragkiadaki. Aligning text-toimage diffusion models with reward backpropagation. arXiv preprint arXiv:2310.03739, 2023.

Zipeng Qi, Lichen Bai, Haoyi Xiong, et al. Not all noises are created equally: Diffusion noise selection and optimization. arXiv preprint arXiv:2407.14041, 2024.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.

Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pp. 8821–8831. PMLR, 2021.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pp. 10674–10685. IEEE, 2022. doi: 10.1109/CVPR52688.2022.01042. URL https://doi. org/10.1109/CVPR52688.2022.01042.

Simo Ryu. Low-rank adaptation for fast text-to-image diffusion fine-tuning. https://github. com/cloneofsimo/lora, 2023.

Timo Schick, Jane Dwivedi-Yu, Roberto Dess`ı, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36, 2024.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022.

Eyal Segalis, Dani Valevski, Danny Lumen, Yossi Matias, and Yaniv Leviathan. A picture is worth a thousand words: Principled recaptioning improves image generation. arXiv preprint arXiv:2310.16656, 2023.

Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. Advances in Neural Information Processing Systems, 2024.

Chenyang Si, Ziqi Huang, Yuming Jiang, and Ziwei Liu. Freeu: Free lunch in diffusion u-net. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4733–4743, 2024.

Karen Sparck Jones. A statistical interpretation of term specificity and its application in retrieval. Journal of documentation, 28(1):11–21, 1972.

D´ıdac Sur´ıs, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 11888–11898, 2023.

Trieu H Trinh, Yuhuai Wu, Quoc V Le, He He, and Thang Luong. Solving olympiad geometry without human demonstrations. Nature, 2024.

Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8228–8238, 2024.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https: //openreview.net/forum?id=ehfRiF0R3a.

Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. Visual chatgpt: Talking, drawing and editing with visual foundation models. arXiv preprint arXiv:2303.04671, 2023a.

Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-toimage synthesis. arXiv preprint arXiv:2306.09341, 2023b.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36, 2024.

Xiangyuan Xue, Zeyu Lu, Di Huang, Wanli Ouyang, and Lei Bai. Genagent: Build collaborative ai systems with automated workflow generation–case studies on comfyui. arXiv preprint arXiv:2409.01392, 2024.

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. 2023.

Zhengqing Yuan, Ruoxi Chen, Zhaoxu Li, Haolong Jia, Lifang He, Chi Wang, and Lichao Sun. Mora: Enabling generalist video generation via a multi-agent framework. arXiv preprint arXiv:2403.13248, 2024.

Lvmin Zhang. Fooocus. https://github.com/lllyasviel/Fooocus, 2023. Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image

diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3836–3847, 2023.

Yinan Zhang, Eric Tzeng, Yilun Du, and Dmitry Kislyuk. Large-scale reinforcement learning for diffusion models. arXiv preprint arXiv:2401.12244, 2024.

Mingchen Zhuge, Wenyi Wang, Louis Kirsch, Francesco Faccio, Dmitrii Khizbullin, and J¨urgen Schmidhuber. Gptswarm: Language agents as optimizable graphs. In Forty-first International Conference on Machine Learning, 2024.

# arXiv:2410.01731v1[cs.CV]2Oct2024

## COMFYGEN: PROMPT-ADAPTIVE WORKFLOWS FOR TEXT-TO-IMAGE GENERATION - SUPPLEMENTARY MATERIALS

- 1 ADDITIONAL RESULTS

In figs. 1 to 4 we show additional images generated using our method. figs. 1 and 2 show images generated using ComfyGen-IC for workflow selection, while figs. 3 and 4 show images created with flows from ComfyGen-FT.

These results showcase the generalizability of our approach to a wide range of prompts, from portraying unique artistic styles to portraying photo-realistic and imagined scenes.

- 2 ADDITIONAL IMPLEMENTATION DETAILS 2.1 COMFYGEN-IC

Here, we provide additional details on our ComfyGen-IC approach, including the prompts used to create the class labels, and those used for workflow selection.

To generate the set of category labels, we prompted Claude Sonnet 3.5 with the following:

|“Given the following list of image prompts, generate a list of 20 labels that describe the key elements, styles, or themes of the images these prompts might produce. Focus on general descriptors like ’people’, ’photo-realistic’, ’photo-artistic’, ’nature’, ’abstract’, etc.<br><br>Here are the prompts: [prompts]”|
|---|

This produced the following list of category labels: ’People’, ’Photo-realistic’, ’Photo-artistic’, ’Fantasy’, ’Sci-fi’, ’Horror’, ’Anime’, ’Abstract’, ’Surreal’, ’Cyberpunk’, ’Steampunk’, ’Gothic’, ’Digital art’, ’Portrait’, ’Nature’, ’Landscape’, ’Wildlife’, ’Urban’, ’Cosmic’, ’Underwater’.

To assign labels to each text-to-image training prompt, we prompted the LLM with the following:

|”Given the following image prompt and list of labels, select the most relevant labels that describe the key elements, styles, or themes of the image this prompt might produce. Provide only the selected labels, separated by commas.<br><br>Image prompt: [prompt] Available labels: [labels] Selected labels:”|
|---|

The LLM assigned an average of 4.96 ± 1.33 labels per prompt, with a maximum of 10. For 2 prompts (0.4% of the training set), the LLM failed to assign any label and so they were discarded.

The full table of flows and scores amounted to roughly 80,000 tokens. To further reduce this, we filter out all flows which achieved a score below the median for every single category. This filters out a total of 125 (or roughly 40% of the flows).

The filtered table serves at the context which we provide to the LLM with the following flow selection prompt:

|“[context] Please classify the following prompt into one of the flows mentioned above: [prompt] Provide the flow ID and a brief explanation for your classification.”|
|---|

As part of our investigation, we also ablated the use of 30 labels and different LLM labeling prompts, but were unable to improve results over this baseline.

2.2 COMFYGEN-FT

Our ComfyGen-FT models were finetuned using 4-bit quantization, with a batch size of 8 and 4 gradient accumulation steps. We used an AdamW optimizer with weight decay of 0.01. We further use 5 warmup steps, and a linear LR scheduler.

All experiments were conducted on a single NVIDIA H100 GPU. During training and inference, we use the following instruction when prompting the LLM:

|“Below is a prompt that describes an image a user wants to generate, and a numerical score describing the quality of an image. Please output a ComfyUI workflow in json format that will create an image with this score when given the prompt.<br><br>>>> Prompt: [prompt] >>> Score: [score] >>> Flow:”|
|---|

- 3 IMAGE PROMPTS

Below, we provide the prompts used to synthesize all the large-figure images in the core paper and in section 1. In each table, we point to the image location using the figure number and a grid position. The grids are 0 indexed, and list the image location from top-to-bottom and left-to-right.

- 4 LIST OF MODELS APPEARING IN OUR WORKFLOWS

Recall that we augment human created flows by randomly swapping the models that they use. Below is a list of all models that appear in our dataset. All base models are on the scale of SDXL and below.

- 5 ADDITIONAL USER STUDY DETAILS

In fig. 5 we show an example question from our user study form. To create the forms, we randomly sampled 120 prompts for ComfyGen-FT and 120 prompts for ComfyGen-IC. We divided each group of 120 questions into 6 blocks of 20, and matched each block to a baseline. From this matching, we created 240 questions that pit one of our methods against one of the baselines. We discarded all questions for which at least one of the images contained nudity or overly sexualized content, leaving us with a total of 231 questions. Each question was initially answered by at least 3 users, but we discarded all responses from users that picked at least 2 answers where the chosen image had no relation to the input prompt (e.g., a scenario where the prompt describes a cigar, one image shows a cigar, and the user picked an image that shows an empty street).

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

#### Figure 1: Curated images generated using ComfyGen-ICL. The list of prompts is available in the supplementary.

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

#### Figure 2: Curated images generated using ComfyGen-ICL. The list of prompts is available in the supplementary.

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

#### Figure 3: Curated images generated using ComfyGen-FT. The list of prompts is available in the supplementary.

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

#### Figure 4: Curated images generated using ComfyGen-FT. The list of prompts is available in the supplementary.

|Figure<br><br>|Prompt|
|---|---|
|Fig. 3 (0, 0)<br><br>|“The entire observable universe in a single bottle, Dreamlike, Surreal landscapes, Mystical creatures, Twisted reality, Surreal still life, Extremely Detailed Oil Painting, glow effects, god rays, Hand drawn, render, 8k, cartoon, octane render, cinema 4d, blender, dark, atmospheric 4k ultra detailed, cinematic sensual, Sharp focus, humorous illustration, big depth of field, Masterpiece, colors, 3d octane render, 4k, concept art, trending on artstation, hyper realistic, Vivid colors, extremely detailed CG unity 8k wallpaper, trending on ArtStation, trending on CGSociety, Intricate, High Detail, dramatic, masterpiece, best quality, ultra-detailed, unreal engine, octane render, HDR”|
|Fig. 3 (0, 1)<br><br>|“cottoncandy, In a whimsical candy kingdom, there stands a unicorn crafted from marshmallows, exuding childlike wonder and an air of mystery. Its body is as white and fluffy as a cloud, embodying the spirit of a playful sprite emerging from sweet dreams, Its head features a radiant rainbow candy gem that forms the iconic horn, shining with seven-colored brilliance, seemingly illuminating every innocent aspiration. The eyes are crafted from translucent hard candies, filled with tender yet inquisitive expressions, The mane consists of multi-hued strands of marshmallow fluff, each one dazzling like a rainbow, swaying gently in the breeze, whispering enchanting tales. Its tail resembles a billowy marshmallow cloud, soft and dreamy; when it gallops joyfully, it leaves behind a trail of vibrant marshmallow magic, Its hooves resemble those cast from creamy white chocolate, solid yet sweet, firmly planted on the ground made of colorful candies. This marshmallow unicorn stands tall, serving as both a tangible representation of children’s naive fantasies and a mystical presence brimming with magical allure within the fairy tale world, UHD, Extreme detail, natural light, volume light, fantasism, professional color, professional composition”|
|Fig. 3 (0, 2)|“Masterpiece double exposure of a girl silhouette blending with monochrome apocalypse aftermath and a colorful natural landscape in the underlying backdrop, sharp contrast, detailed crisp lines, in focus”<br><br>|
|Fig. 3 (1, 0)<br><br>|“A photo of a large man with beard riding a vespa wearing loriseg armor and helmet, red tunic”|
|Fig. 3 (1, 1)<br><br>|“A hyper-realistic, ultra-detailed painting featuring a fantastical landscape with a village at the end of time. The scene has a perfect composition and chiaroscuro inspired by Rembrandt, with concept art elements by Mariusz Lewandowski, Louis Aston Knight, Karine Eibatova, John Howe, Jessica Rossier, JarosAaw˚ JaAnikowski,˚ Inessa Garmash, HenriAtte˜ Ronner-Knip, and Harumi Hironaka. The painting is vibrant, beautiful, painterly, detailed, and textural. Watercolor painting, masterpiece, best quality, hyper detailed, ultra realistic, 32k, RAW photo, landscape, fantasy, village at the end of time, perfect composition, chiaroscuro by Rembrandt, concept art, vibrant, beautiful, painterly, detailed, textural, artistic, Rembrandt, Mariusz Lewandowski, Louis Aston Knight, Karine Eibatova, John Howe, Jessica Rossier, JarosAaw˚ JaAnikowski,˚ Inessa Garmash, HenriAtte˜ Ronner-Knip, Harumi Hironaka., extremely detailed, swirling ink”|
|Fig. 3 (1, 2)<br><br>|“true masterpiece, masterpiece cinematic lighting, cinematic shot from below, extremely detailed, high detail, hires textures, incredibly detailed, intricate details, photorealism, intricately designed, sea storm up inside a large transparent glass ball, drops outside the ball rainfall on floor background, HD32k, focus”|
|Fig. 3 (2, 0)|“no humans, scenery, closeup, branch, tree, leaf, nature, rain, outdoors, depth of field, droplets”<br><br>|
|Fig. 3 (2, 1)<br><br>|“fine art, oil painting, best quality, dark tales, illustration, each color adds depth, and the entire piece comes together to create a breathtaking spectacle of motion and tranquility., while the ball is adorned with an array of stripes in various hues. the figurine, while her right hand delicately holds a small, epic splash cover art in the van gogh style, starry sky, dan mumford, andy kehoe, 2d, flat, delightful, vintage, art on a cracked paper, patchwork, stained glass, fairytale, storybook detailed illustration, cinematic, ultra highly detailed, tiny details, beautiful details, mystical, luminism, vibrant colors, complex background”|
|Fig. 3 (2, 2)|“a ral-sun flying flower, dusty background, epic, heroic, bokeh, sharp detailed, hyperrealistic, amazing, macro photo, god rays, volumetric”|

#### Table 1: List of prompts used for Fig. 3 in the core paper.

|Figure|Prompt<br><br>|
|---|---|
|Supp. Fig. 1 (0, 0)<br><br>|“2D anime style, elegant girl floating in a vortex of city lights and digital elements, deep and expressive eyes, vibrant blue and purple colors, intricate details, magical and futuristic ambiance, highly detailed background with swirling buildings and lights, complex and dynamic scene”|
|Supp. Fig. 1 (0, 1)<br><br>|“A close-up portrait of an alien being, with luminescent skin and eyes that hold centuries of wisdom, set against the backdrop of their advanced, technology-filled habitat., 90th photos from photo album, film, vintage style, flash from camera, hyper realistic, fine textures, high quality textures of materials, volumetric textures, natural textures, natural colors, correct white balance, color correction, dehaze, clarity”|
|Supp. Fig. 1 (0, 2)<br><br>|“masterpiece, best quality, high quality, intricate, absurdres, very aesthetic, no humans, landscape, outdoors, mountain tops, wind, windy, wind lines, clouds, above clouds, cliff, wind magic, aurora, ultra wide angle shot, cinematic style, highly detailed, extremely detailed, sharp detail, majestic, shallow depth of field, movie still, soft light, circular polarizer, colorful, wallpaper, professional illustration, anime”|
|Supp. Fig. 1 (1, 0)|“adventurous cute funny lizard, donned a tiny explorer’s hat and carried a mini backpack, walk in wild colorfully jungle, detailed scales, by Jean-Baptiste Monge, Gilles Beloeil, Tyler Edlin, Marek Okon, Pixar, 8k, album art, comic style, golden ratio, perfect composition, a masterpiece, trending on artstation, extreme close up, shot from below . High dynamic range, vivid, rich details, clear shadows and highlights, realistic, intense, enhanced contrast, highly detailed”<br><br>|
|Supp. Fig. 1 (1, 1)|“Amazing detailed photography of a cute adorable samurai kitten holding Katana with 2 paws, Cherry Blossom Tree petals floating in air, high resolution, piercing eyes, lifelike fur, Anti-Aliasing, FXAA, De-Noise, Post-Production, SFX, insanely detailed and intricate, hypermaximalist, elegant, ornate, hyper realistic, super detailed, noir coloration, serene, 16k resolution, full body”<br><br>|
|Supp. Fig. 1 (1, 2)|“bg imgs, portrait, wallpaper, colorful, highres, absurdres, huge filesize, fantasy, foreshortening, black dress, Extremely gorgeous metal style, Metal crown with ornate stripes, Various metals background, Sputtered molten iron, floating hair, Hair like melted metal, Clothes made of silver, Clothes with gold lace, flowing gold and silver, everything flowing and melt, flowing iron, flowing silver, lace flowing and melt, best quality, masterpiece, illustration, an extremely delicate and beautiful, extremely detailed, CG, unity, 8k wallpaper, Amazing, finely detail, masterpiece, best quality, official art, extremely detailed CG unity 8k wallpaper, absurdres, incredibly absurdres, huge filesize, ultra-detailed, highres, extremely detailed, beautiful detailed girl, extremely detailed eyes and face, beautiful detailed eyes, light on face”<br><br>|
|Supp. Fig. 1 (2, 0)<br><br>|“4n1v3rs3, 4n1v3rs3, 2, Create an art deco inspired illustration of A scenic lighthouse perched on a rocky coastline, overlooking the turquoise waters of Cala Ratjada. Include vibrant pastel colors, sleek lines, and a retro summer atmosphere. The style should be reminiscent of vintage travel posters with a modern twist”|
|Supp. Fig. 1 (2, 1)|“lighting Style, dim light, low light, dramatic light, partially covered in shadow, award winning photography, RAW photo, Hyperrealistic, beautiful African woman, wearing traditional outfit, vibrant colors, intricate patterns, detailed textures, soft natural lighting, ethereal glow, participating in a traditional ceremony, serene and dignified expression, traditional headwrap, detailed jewelry, lush green background, realistic shadows, fine details in fabric and skin, ultra-quality, cultural richness, ceremonial setting, glowing skin, traditional makeup, colorful beads, elegant pose, cultural heritage, ceremonial attire Detailed natural skin”<br><br>|
|Supp. Fig. 1 (2, 2)|“perfect flower in the nature”|

#### Table 2: List of prompts used for Fig. 1 in the supplementary.

|Figure|Prompt<br><br>|
|---|---|
|Supp. Fig. 2 (0, 0)|“amazing quality, masterpiece, best quality, hyper detailed, ultra detailed, UHD, perfect anatomy, in castle, girl knight, holding a sword, dazzling, transparent, polishing, 1 arm up, waving sword, gold armor, glowing armor, glowing eyes, full armor, shine armor, dazzling armor, extremely detailed”<br><br>|
|Supp. Fig. 2 (0, 1)<br><br>|“closeup award winning photo of wolf, perfect environment, extremely detailed, dark shot”|
|Supp. Fig. 2 (0, 2)|“detailed, aesthetic, 8k unreal engine photorealism, ethereal lighting, purple, nighttime, darkness, surreal art, fantasy, glowing, night, dark environment, AyameNewYears, horns, long hair, side, red kimono, floral print, hair flower, sash, haori, scenery, ink, mountains, water, trees, full body, reflection, light, arm behind back, sandals, from side”<br><br>|
|Supp. Fig. 2 (1, 0)<br><br>|“Japanese Ink Drawing, Ink Dripping Drawing, horror-themed space-themed Surrealism, cartoon style, concept art surreal, anatomy, anatomical drawings, human body, fantastical organs, space demon, sharp teeth, wide grimace, masterpiece, best quality, highly detailed, sharp focus, dynamic lighting, vivid colors, texture detail, particle effects, storytelling elements, narrative flair, 16k, UE5, HDR, subject-background isolation . digital artwork, illustrative, painterly, matte painting, highly detailed, expressive, dramatic, organic lines and forms, dreamlike and mysterious, Surrealism . cosmic, celestial, stars, galaxies, nebulas, planets, science fiction, highly detailed . eerie, unsettling, dark, spooky, suspenseful, grim, highly detailed, ink drawing, dripping ink, ink drawing, inkwash, Japanese cartoon style, japanese torii”|
|Supp. Fig. 2 (1, 1)<br><br>|“a mystical, and high-resolution image of an anthropomorphic Elysium Knight in Radiant Darkness, The character should be depicted in a manga cover style with wealthy portraiture and poster art elements. Volumetric lighting, The image should feature rich colors and high contrast, focusing on the best quality, official art, and a beautiful and aesthetic appearance. wearing Sci-fi clothes and enhanced Astronaut suit, and standing in a cowboy shot pose. The character should have tattoos, muscular abs, shoulder armor, mystic Ethereal moon forest lake in the background. Emphasize a samurai theme with a detailed background and floral elements.”|
|Supp. Fig. 2 (1, 2)|“ova, background 2D anime, anime screencap, a pastel-colored character with long, flowing pink hair, large horns, and elf-like ears, wearing a cozy cable-knit sweater. The character is sitting in a beautiful, flower-filled meadow, surrounded by butterflies and small woodland creatures. She is gently petting a small bunny on her lap, smiling softly. The background is detailed with various types of flowers, trees, and a bright blue sky. The lighting is natural and bright, enhancing the cheerful and peaceful atmosphere. The camera shot is a medium shot, capturing the character and the detailed meadow.”<br><br>|
|Supp. Fig. 2 (2, 0)<br><br>|“watercolor art painting, watercolor, 1 forest queen, solo, sitting in the water, vintage anime aesthetic, back view, illustration, turn back her face, mysital, detailed eyes”|
|Supp. Fig. 2 (2, 1)|“the image portrays a tranquil scene of a boat floating gently on the water, surrounded by an expansive landscape. the moon, full and glowing with a warm, reddish orange hue, casts a mystical ambiance over the entire scene. its reflection shimmers off the surface of the water, adding to the serene atmosphere. in the distance, mountains loom under the moon’s soft glow, their peaks partially obscured by the low hanging clouds. they appear majestic yet gentle, as if watching over the peaceful night below. trees line the shore in the foreground, their silhouettes faintly visible against the darkening sky. this picturesque setting evokes a sense of calm and tranquility, inviting viewers to take a moment and appreciate the beauty of nature. it is a symphony of colors and shapes, each element working harmoniously together to create a visually captivating and emotionally soothing composition.”<br><br>|
|Supp. Fig. 2 (2, 2)|“spectacular digital rendering of a rear view of a transparent hyper car revealing internal mechanical components such as engine, car chassis, suspension, and internal wiring, detailed textures, accurate lighting and shadows, 8k quality, intricate patterns, high-definition, glossy finish, vivid reflections, perfect lighting, showroom”|

#### Table 3: List of prompts used for Fig. 2 in the supplementary.

|Figure|Prompt<br><br>|
|---|---|
|Supp. Fig. 3 (0, 0)<br><br>|“A photo of a abyssal destroyed robot covered in moss, post apocalyptic city, lush overgrowth, by Luis Royo, by Greg Rutkowski, dark, gritty, intricate, volumetric lighting, volumetric atmosphere, concept art, cover illustration, octane render, trending on artstation, 8k, dynamic pose”|
|Supp. Fig. 3 (0, 1)<br><br>|“by Mattias Adolfsson and Satoshi Kon, hyper realistic medium full shot photo of a otherworldly landscape, oppulent, glamourous, masterful, poster art, bold lines, hyper detailed, expressive, award winning, dark limited color palette, high contrast, depth of field, intricate details, masterpiece, best quality, rim lighting, looking at viewer, dynamic pose, wide angle panoramic view”|
|Supp. Fig. 3 (0, 2)|“close up photo of a rabbit, forest, haze, halation, bloom, dramatic atmosphere, centred, rule of thirds, 200mm 1.4f macro shot”<br><br>|
|Supp. Fig. 3 (1, 0)|“closeup of a circuit board city with very small futuristic cars, small flying vehicles, small robots and small humans, ultra hd, realistic, vivid colors, highly detailed, UHD drawing, pen and ink, perfect composition, beautiful detailed intricate insanely detailed octane render trending on artstation, 8k artistic photography, photorealistic concept art, soft natural volumetric cinematic perfect light”<br><br>|
|Supp. Fig. 3 (1, 1)|“Cross-Processing by Artur Amijewski and Rodelio Astudillo, award winning, kookaburra, aesthetic of symbolism with bubbling atmosphere, skybox / skydome, well-defined edges, creative tour de force with meticulous details, morganite pink and mauve colors”<br><br>|
|Supp. Fig. 3 (1, 2)<br><br>|“detailed ink, pen and ink, mail art, best quality, detailed epic ice transparent ethereal otherworldly ghost castle in the blue sky, clouds, smoke, fog, detailed landscape, ghost figures, lake, boat, green forest, detailed flying dragon at the sky, detailed scales, warm lights, glittering, Craola, Dan Mumford, Andy Kehoe, 2d, flat, art on a cracked paper, patchwork, stained glass, cute, adorable, fairytale, storybook detailed illustration, cinematic, ultra highly detailed, tiny details, beautiful details, mystical, luminism, vibrant colors, complex background”|
|Supp. Fig. 3 (2, 0)|“ethereal fantasy concept art of masterpiece, best quality, RAW macro photo of just some garbage that someone put in a box/frame . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy”<br><br>|
|Supp. Fig. 3 (2, 1)|“detailed realistic close up of a strawberry shaped like a muscular man, sitting, natural light”<br><br>|
|Supp. Fig. 3 (2, 2)|“minimalist, cinematic, movie poster, cold colors, purple theme, standing, weapon, outdoors, sky, cloud, gun, moon, helmet, robot, ground vehicle, motor vehicle, multiple planets, planet, spacecraft”|

#### Table 4: List of prompts used for Fig. 3 in the supplementary.

|Figure<br><br>|Prompt|
|---|---|
|Supp. Fig. 4 (0, 0)<br><br>|“Once upon a time, organized theory, systematic scientific papers, elegant, figuration, sharp lines, black and white colors, beautiful, trending on artstation, volumetric lighting, by Guy Denning, no text, colorized drawings, black and white, artistic”|
|Supp. Fig. 4 (0, 1)|“celestial promenade, the horror of stars, twilight sunshine, heaven help us, cityscape masterpiece, realistic, best quality, cosmic horror, bright horror”<br><br>|
|Supp. Fig. 4 (0, 2)<br><br>|“highly detailed and hyper realistic photo, by Alena Aenami, by Archibald Thorburn, by Daniele Afferni, a breathtaking otherworldly landscape with psychedelic colors, in the style of monochromatic silhouette reflection, limited dark palette, unusual dark colors, faded colors, atmospheric haze, highly dramatic cinematic lighting, motion blur, film grain, professional, excellent composition, finest details, maximized details, ultimate detail level, masterpiece, best quality”|
|Supp. Fig. 4 (1, 0)|“refreshing, vibrant glowing coconut juice drink, dew drops, refreshing, in the style of a product hero shot in motion, dynamic magazine ad image, photorealism, sleep and mystical elements around the background”<br><br>|
|Supp. Fig. 4 (1, 1)<br><br>|“The art of Origami, Paper folding, Swan on a lake, Amazing colours, Intricate details, Painstaking Attention to Details, UHD”|
|Supp. Fig. 4 (1, 2)<br><br>|“The devil with glowing eyes ingenious opus magnum by Michael Vincent Manalo and Elvira Vigna, pastel oil painting, high key with glistening light, cosmic anddireful and dramatic atmosphere, zestful quinacridone nickel azo gold and electricblue color swatch, raw details, crisp details, bokeh lights, analog photo, full body, tall height, against the background volcano mouth in the background, blue sky, epic clouds, oil on canvas, art by Jeremy Mann, god rays, dramatic light, art by Eduard Wilhelm Pose and Master of the Holy Blood and Ivan Bilibin, oxygen-rich air and sheltered atmosphere, snowy, aesthetic of hard - edge painting, symmetry and balance, pioneering unparalleled masterwork with superior details, skin texture, masterpiece, top quality, best quality, official art, highest detailed, atmospheric lighting, cinematic composition, complex multiple subjects, 4k HDRvaporwave style, cyberpunk, vibrant, neon colors, highly detailed, Leica Q2 with Summilux 35mm f/1.2 ASPH, clear face, Ultra High Resolution, wallpaper, 8K, Rich texture details, hyper detailed, detailed eyes, detailed background, dramatic angle”|
|Supp. Fig. 4 (2, 0)<br><br>|“In a lush, tropical rainforest, a vibrant, exotic bird like a parrot perches amidst a kaleidoscope of flowers, with dappled sunlight filtering through the dense canopy. This scene is captured in the style of Marco Lumiere, known for his vivid and lively color palettes and a slightly impressionistic touch, highlighting the vitality of nature and the majestic beauty of the bird in its natural habitat.”|
|Supp. Fig. 4 (2, 1)<br><br>|“wooden arch bridge, river, crowd, mount fuji, boats, cherry tree, scenery, outdoors, best quality, highly detailed”|
|Supp. Fig. 4 (2, 2)|“zentangle Flickering grass beneath a lone smart light bulb that casts an eerie glow in the otherwise serene sunlit meadow, as if waiting for something to emerge from the shadows. . intricate, abstract, monochrome, patterns, meditative, highly detailed”|

#### Table 5: List of prompts used for Fig. 4 in the supplementary.

### Base models and refiners:

- • AetherverseLightning v10
- • AlbedobaseXL v13
- • AnimagineXL v30
- • AnythingXL
- • crystalClearXL ccXL
- • DreamshaperXL turboDpmppSDEKarras
- • EnvyhyperdriveXL v10
- • GleipnirV0.3
- • JibMixXL v9 “BetterBodies”
- • JuggernautXL v9 Rdphoto2Lightning
- • LeosamsHelloworldXL v70
- • Proteus v03
- • RealismEngineSDXL v10
- • RealvisXL v40 BakedVAE
- • RealvisXL v40 LightningBakedVAE
- • SDXL Base 1.0 0.9VAE
- • SDXL Base 1.0
- • SDXL Refiner 1.0 0.9VAE
- • SDXL Refiner 1.0
- • SDVN7 - NijiStyleXL v1
- • SSD-1B
- • TurbovisionXL SuperFastXL V431BakedVAE
- • Stable Cascade
- • Pixart-Σ

### LoRAs and embeddings:

- • easynegative
- • bad-hands-5
- • nfixer
- • Add-Detail XL
- • EpicF4nta5yXL
- • AnimeTarot
- • JuggerCineXL2
- • LCM LoRA SSD-1B
- • LCM LoRA SDXL
- • LogoRedmond
- • MJ52 v2.0
- • MJ52
- • PerfectEyesXL
- • Pixel-Art-XL v1.1
- • Ral-Dissolve-SDXL
- • SDXL Glass
- • SDXLFaetastic v24
- • Sinfully Stylish SDXL

- • Werewolf SDXL
- • WowifierXL v2
- • XL more art-full-beta1

### Super resolution and face restoration models:

- • 4x NMKD Superscale - SP 178000 G
- • 4x UltraSharp
- • RealESRGAN x2 plus
- • codeformer
- • GFPGAN v1.4

### VAEs:

- • SharpSpectrum VAEXL
- • SDXL VAE fp16 fix
- • SDXL VAE

|[Figure 113]|
|---|

Figure 5: An example question from our user study.

