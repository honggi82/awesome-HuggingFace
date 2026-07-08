[Figure 1]

# DiffusionAgent : Navigating Expert Models for Agentic Image Generation

Jie Qin1∗, Jie Wu2∗, Weifeng Chen2∗, Yueming Lyu3 1Meituan 2ByteDance 3Nanjing University Project page: https://DiffusionAgent.github.io

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

! : ❌ " : ❌

Prompt

[Figure 8]

[Figure 9]

[Figure 10]

i.e., SD1.5, SDXL

[Figure 11]

! : ✅ " : ❌

[Figure 12]

Text-to-Image Genera.on System

I want to explore a mystical forest ﬁlled with ancient trees

I want to be a superman and soar into the sky

## arXiv:2401.10061v2[cs.CV]20Jan2026

a wooden toy horse with a mane made of rope

[Figure 13]

a stork playing a violin Dogs playing poker

[Figure 14]

Model

i.e., DALLE3

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

! : ✅

[Figure 21]

[Figure 22]

! : ❌ " : ✅ " : ✅

[Figure 23]

[Figure 24]

i.e., DiﬀusionGPT (Ours)

Agent

[Figure 25]

[Figure 26]

[Figure 27]

Imagine a world where cities ﬂoat in the

Capture the excitement and energy of a music festival with crowds dancing in sync.

Generate an image of a skilled carpenter crafting a beautiful piece of furniture with intricate detail

If I had a magic wand, I would bring toys to life and create a magical toyland.

If I could talk to trees, they would share their ancient wisdom with me

i.e., Civitai, LibLibAI

[Figure 28]

! : Input-ParseDiversePrompts

sky, connected by intricate bridges.

[Figure 29]

" :Output-PickExpertModel

Red: Prompt-based; Green: Inspiration-based; Purple: Instruction-based; Orange: Hypothesis-based

Fig. 1: We propose a unified generation system DiffusionAgent, which leverages Large Language Models (LLM) to seamlessly accommodate various types of prompts input and integrate domain-expert models for output. Our system is capable of parsing diverse forms of inputs, including Prompt-based, Instruction-based, Inspiration-based, and Hypothesis-based input types. It exhibits the ability to generate outputs of superior quality.

tasks. The first open-source text-to-image diffusion model, known as Stable Diffusion (SD) [2], which has rapidly gained popularity and widespread usage. Various techniques tailored for SD, such as Controlnet [5], Lora [6], further paved the way for the development of SD and foster its integration into various applications. SDXL [3] is tailored to deliver exceptional photorealistic outputs with intricate details and artistic composition. Moreover, the impact of SD extends beyond technical aspects. Community platforms such as Civitai, WebUI, and LibLibAI have emerged as vibrant hubs for discussions and collaborations among designers and creators.

Abstract—In the accelerating era of human-instructed visual content creation, diffusion models have demonstrated remarkable generative potential. Yet their deployment is constrained by a dual bottleneck: semantic ambiguity in diverse prompts and the narrow specialization of individual models. A single diffusion architecture struggles to maintain optimal performance across heterogeneous prompts, while conventional “parse-thencall” pipelines artificially separate semantic understanding from generative execution. To bridge this gap, we introduce DiffusionAgent, a unified, language-model-driven agent that casts the entire “prompt comprehension–expert routing–image synthesis” loop into a agentic framework. Our contributions are three-fold: (1) a tree-of-thought-powered expert navigator that performs fine-grained semantic parsing and zero-shot matching to the most suitable diffusion model via an extensible prior-knowledge tree; (2) an advantage database updated with human-in-the-loop feedback, continually aligning model-selection policy with human aesthetic and semantic preferences; and (3) a fully decoupled agent architecture that activates the optimal generative path for open-domain prompts without retraining or fine-tuning any expert. Extensive experiments show that DiffusionAgent retains high generation quality while significantly broadening prompt coverage, establishing a new performance and generality benchmark for multi-domain image synthesis. The code is available at https://github.com/DiffusionAgent/DiffusionAgent.

Despite making significant strides, current stable diffusion models face two key challenges when applied to realistic scenarios as shown in Fig. 1:

- • Model Limitation: Stable diffusion models like SD1.5 [2] are versatile but underperform in specific domains, whereas domain-specific models like SD1.5+Lora excel in targeted sub-fields at the cost of general adaptability.
- • Prompt Constraint: During stable diffusion training, text data only includes descriptive captions, yet the models face challenges in achieving optimal performance with diverse prompt, such as instructions and inspirations.

Index Terms—Agentic image generation, Large language Model, Tree-of-thought, Expert model

The mismatch between stable diffusion models and realworld applications often results in limited performance, poor generalization, and implementation difficulties. Research has sought to address these issues. While SDXL has improved specific-domain performance, but optimal results are still

I. INTRODUCTION

Recent years have witnessed the prevalence of diffusion models [1]–[4] in human-interactive image generation tasks, revolutionizing image editing, stylization, and other related

[Figure 30]

##### DiffusionAgent

Fig. 2: Overview of DiffusionAgent. The workflow of DiffusionAgent consists of four steps: Prompt Parse, Tree-of-thought of Models of Building and Searching, Model Selection, and Execution Generation. The four steps are shown from left to right and interact with LLM continuously. The upper side shows the detailed process of each step. The lower side shows the example of the whole workflow.

unattained. Techniques like prompt engineering and fixed templates have been used to enhance input quality and output, yet they fall short of offering a complete solution. This leads us to pose a key question: Can we create a unified framework to unleash prompt constraint and activate corresponding domain expert model ?

In order to address the aforementioned question, we propose DiffusionAgent, which leverages Large Language Model (LLM) [7]–[9] to offer a one-for-all generation system that seamlessly integrates superior generative models and effectively parses diverse prompts. DiffusionAgent constructs a Tree-of-Thought (ToT) structure, which encompasses various generative models based on prior knowledge and human feedback. When presented with an input prompt, the LLM first parses the prompt and then guides the ToT to identify the most suitable model for generating the desired output. Furthermore, we introduce Advantage Databases, where the Tree-of-Thought is enriched with valuable human feedback, aligning the LLM’s model selection with human preferences.

The contributions of this work can be summarized as:

- • New Insight: DiffusionAgent utilizes a Large Language Model (LLM) as the cognitive engine for its text-to-image generation system, processing various inputs and enabling expert selection for outputs.

- • All-in-one System: DiffusionAgent provides a versatile solution compatible with various diffusion models, unlike existing approaches that are restricted to descriptive prompts, thereby broadening its applicability to different prompt types.

- • Efficiency and Pioneering: DiffusionAgent is notable for its training-free, plug-and-play integration. By incorporating the Tree-of-Thought (ToT) and human feedback, our system enhances accuracy and facilitates a flexible process for aggregating multiple experts.

- • High Effectiveness: DiffusionAgent outperforms traditional stable diffusion models, showcasing significant advancements and offering an all-in-one system that

provides a more efficient and effective pathway for community development in image generation.

II. METHODOLOGY

DiffusionAgent is an all-in-one system specifically designed to generate high-quality images for diverse input prompts. Its primary objective is to parse the input prompt and identify the generative model that produces the most optimal results, which is high-generalization, high-utility, and convenient.

DiffusionAgent composes of a large language model (LLM) and various domain-expert generative models from the opensource communities (e.g. Hugging Face, Civitai). The LLM acts as the main controller and maintains the workflow of the system, which consists of four steps: Prompt Parse, Tree-ofthought of Models of Building and Searching, Model Selection with Human Feedback, and Execution of Generation. The overall pipeline of DiffusionAgent is shown in Fig. 2.

A. Prompt Parse

The Prompt Parse Agent plays a pivotal role in our methodology as it utilizes the large language model (LLM) to analyze and extract the salient textual information from the input prompt. Accurate parsing of the prompt is crucial for effectively generating the desired content, given the inherent complexity of user input. This agent is applicable to various types of prompts, including prompt-based, instruction-based, inspiration-based, hypothesis-based, etc.

Prompt-based: The entire input is used as the prompt for generation. For example, if the input is “a dog” the prompt used for generation would be “a dog”.

Instruction-based: The core part of the instruction is extracted as the prompt for generation. For instance, if the input is “generate an image of a dog”, the recognized prompt would be “an image of a dog”.

Inspiration-based: The target subject of the desire is extracted and used as the prompt for generation (e.g., Input: “I want to see a beach”; Recognized: “a beach”).

Hypothesis-based: It involves extracting the hypothesis condition (“If xxx, I will xxx”) and the object of the forthcoming action as the prompt for generation. For instance, if the input is “If you give me a toy, I will laugh very happily”, the recognized prompt would be “a toy and a laugh face”.

[Figure 31]

User Input

[Figure 32]

[Figure 33]

Given the user input text. Please judge the paradigm of the input text, and then recognize the main string

Prompt of text prompts according to the corresponding form. Parse

Prompt

The Prompt Parse Agent helps DiffusionAgent identify core content from prompts, reducing the impact of noisy text. This is essential for selecting suitable generative models and achieving high-quality results.

- • You are an information analyst who can analyze and summarize a set of words to abstract some representation categories.
- • You are an information analyst who can create a Knowledge Tree according to the input categories.
- • You are an information analyst who can add some input models to an input knowledge tree according to the similarity of the model tags and the categories of the knowledge tree.

[Figure 34]

[Figure 35]

Prompt

Tree-of-thought of Models Build

[Figure 36]

The built tree-of-thought of models.

- B. Tree-of-thought of Models

Following the prompt parsing stage, the subsequent step involves selecting appropriate generative models from an extensive model library to generate the desired images. However, considering the large number of models available, it is impractical to input all models simultaneously into the large language model (LLM) for selection. To address this issue and pinpoint the optimal model, we draw inspiration from the Chain-of-Thought (CoT) paradigm [10], [11] and introduce a Tree-of-Thought (ToT) model tree. By leveraging the search capabilities of the model tree, we can narrow down the candidate set of models and enhance the accuracy of the model selection process.

Constructing the Model Tree using TOT. The Tree-ofThought (TOT) of Model Building Agent is employed to automatically build the Model Tree based on the tag attributes of all models. The agent processes the tag attributes of all models to analyze and categorize them into Subject and Style Domains. Style categories become subcategories within the Subject category, forming a two-layer hierarchical tree. Models are assigned to appropriate leaf nodes based on their attributes, completing the model tree. This automatically constructed tree allows for easy extensibility, as new models are seamlessly integrated based on their attributes.

Searching the Model Tree using TOT. The Model Tree search process, guided by the Tree-of-Thought (TOT) of Models Searching Agent, aims to find models closely aligned with a given prompt. It uses a breadth-first search, systematically evaluating the best subcategory at each leaf node. Categories are compared against the prompt to find the closest match. This iterative process continues until the final node, yielding a candidate set of models for next selection.

- C. Model Selection

[Figure 37]

Identify and behave as five different experts that are appropriate to select one element from the input category list that best matches the input prompt. All experts will write down the selection result, then share it with the group. You then analyze all 5 analyses and output the consensus selected element or your best guess matched element.

[Figure 38]

#### LLM

ChatGPT

Tree-of-thought of Models Search

Prompt

[Figure 39]

The selected model list “model_list_T”.

[Figure 40]

[Figure 41]

The obtained model list “model_list_H” by the HF.

[Figure 42]

[Figure 43]

- • Please judge whether each name in this list {model_list_T} has highly similar name in the list {model_list_H}, if yes, output the similar model name “intersection_model”.
- • Please select one model name from the following model list {intersection_model} that has the highest frequency and top ranking according to the list {model_list_H}.

[Figure 44]

Prompt

Model Selection

[Figure 45]

The selected model.

[Figure 46]

[Figure 47]

Please follow the sentence pattern of the example to expand the description of the input paragraph. The output MUST preserve the contents of the input paragraph. Example: {example_prompt}.

Prompt Extension

Prompt

Fig. 3: Details of prompts during interactions with the LLM. Before being inputted into the LLM, the slots “{}” in figure are uniformly replaced with the corresponding text values.

scores. For a given input prompt, we calculate its semantic similarity to these prompts and identify the top 5 matches. The Model Selection Agent then retrieves precomputed model performances for these prompts from an offline database, selecting the top 5 models for each. This results in a candidate set of 25 models. The agent intersects the model set with the candidate set from the TOT stage, prioritizing models with higher occurrence probabilities and rankings. These models are chosen for the final model generation.

D. Execution of Generation

Once the most suitable model has been selected, the chosen generative model is utilized to generate the desired images using the obtained core prompts.

Prompt Extension. To enhance the quality of prompts during the generation process, a Prompt Extension Agent is employed to augment the prompt. This agent leverages prompt examples from the selected model to automatically enrich the input prompt. The example prompts and the input prompts are both sent to LLM in the in-context learning paradigm. Specially, this agent incorporates rich descriptions and detailed vocabulary to the input prompts following the sentence pattern of example prompts. The Prompt Extension Agent enhances it to a more detailed and expressive form, which significantly improves the quality of the outputs.

The model selection stage aims to identify the most suitable model for generating the desired image from the candidate set obtained in the previous stage. The candidate set, a subset of the model library, includes models closely matching the input prompt. However, limited attribute data from opensource sources complicates precise model selection and detailed information provision to the LLM. To tackle this, we propose a Model Selection Agent that uses human feedback and advanced database technology to align model selection with human preferences.

III. EXPERIMENTS A. Settings

In our experimental setup, the primary large language model (LLM) controller employed was ChatGPT [7], specifically uti-

The advantage database uses a reward model to score model-generated results from 10,000 prompts, storing these

Prompt Instruction Inspiration Hypothesis

[Figure 48]

Transport us to a snowy wonderland where children build snowmen and have snowball fights

If I had a magic wand, I would bring toys to life and create a magical toyland

Create an illustration of a chef expertly preparing a mouthwatering gourmet dish

The man who whistles tunes pianos, watercolor

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

SD 15

AlignmentAesthetics

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Ours

If I could visit outer space, I would witness breathtaking views of distant galaxies

I dream of floating in the clouds with colorful hot air balloons

Create an illustration of a romantic couple sharing a tender moment under a starry sky

a girl examining an ammonite fossil

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

SD 15

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Ours

- Fig. 4: When comparing SD1.5-based DiffusionAgent with SD1.5 [2], it is observed that DiffusionAgent excels in generating more realistic results at a fine-grained level for categories such as humans and scenes. The generated images demonstrate improved visual fidelity, capturing finer details and exhibiting a higher degree of realism compared to SD1.5.

- TABLE I: The selected public trained models for SD1.5 and SDXL based system.

|SD1.5 Based<br><br>|SDXL Based|
|---|---|
|FilmVelvia2 majicmixRealistic lou cartoonish animemix impasto-painting CineStyle5 dreamlabsoil ghibli-style dreamshaper C4D|SDXL weird-future-fashion fenrisxl-801Photorealistic dynavisionXLAllInOneStylized sdxlNijiSpecial copaxTimelessxlSDXL1 starcitizen lwmirXL kandinsky dreamshaperXL<br><br>|

- TABLE II: Quantitative Results: We evaluate the aesthetic score using image-reward and aesthetic score compare to “SD1.5” and “Random” select expert models to output.

|Method<br><br>|Image-reward Aes score|
|---|---|
|SD1.5 Random DiffusionAgent wo HF DiffusionAgent<br><br>|0.28 5.26 0.45 5.50 0.56 5.62 0.63 5.70|

lizing the text-davinci-003 version, which is accessible through the OpenAI API. To facilitate the guidance of LLM responses, we adopted the LangChain framework, which effectively controlled and directed the generated outputs. For the generation models utilized in our experiments, we selected a diverse range of models sourced from the Civitai and Hugging Face communities. The selection process involved choosing the most popular models across different types or styles available on these platforms. The selected models for different versions of DiffusionAgent are illustrated in Table I. The details of prompts during interactions with LLM are shown in Fig. 3.

B. Qualitative Results

1) Visualization of SD1.5 Version: To assess the efficacy of our system, we performed a comprehensive evaluation by comparing its generation performance against the baseline method, SD1.5 [2]. The comparative results are presented in Fig. 4. We conducted a detail analysis of four distinct prompt types and compared them along two key dimensions: semantic alignment and image aesthetics.

Upon examining the results, two issues with the base model were identified: i) Semantic Lack: The base model struggles to capture complete semantic information, particularly with prompts involving “man, chef, children, and toyland”, focusing too narrowly on specific classes. ii) Poor Performance on Human-Related Targets: The base model has difficulty generating accurate facial and body details, resulting in lower quality images, especially for “girl and couple” prompts.

In contrast, DiffusionAgent effectively addresses these limitations by producing images that capture complete semantic information from input prompts. For example, it successfully represents broader contexts like “man who whistles tunes pianos” and “a snowy wonderland where children build snowmen”. Additionally, it excels in generating detailed images of human-related objects, as seen with prompts like “a romantic couple sharing a tender moment under a starry sky”.

2) Visualization of SDXL Version.: With advancements in universal generation models, SDXL [3] has shown promising results. We improved our system by integrating various opensource models based on SDXL. All comparative results are shown in Fig. 5, all output images were 1024x1024. The result reveals that SDXL sometimes loses semantic information, as seen in prompts like “3D tiger” or “flying cars”. In contrast,

Prompt Instruction Inspiration Hypothesis

[Figure 68]

Create an image of a futuristic cityscape with flying cars and towering skyscrapers, illuminated by neon lights

On a real playground, a cartoon shark is pulling a 3D tiger while running

Transport us to a snowy wonderland where children build snowmen and have snowball fights

If I could talk to trees, they would share their ancient wisdom with me.

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

SD XL

### AlignmentAesthetics

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Ours

Create an image of a lone traveler walking an endless desert under a starry sky

It would be amazing to engage in a virtual reality game with an animated alien creature in a lush jungle setting.

If I could communicate with animals, I will stand on the building and discuss with the eagle

a white towel with a cartoon of a cat on it

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

SD XL

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Ours

- Fig. 5: Comparison of SDXL version of DiffusionAgent with baseline SDXL [3]. All generated iamges are 1024×1024 pixels.

[Figure 90]

[Figure 91]

[Figure 92]

A black dragon perched on top of a tall Egyptian obelisk and breathing flames at a knight on the gro

[Figure 93]

[Figure 94]

[Figure 95]

a cartoon of an angry shark.

[Figure 96]

[Figure 97]

a panda

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

a woman looking at a house

Random Select

Ours - TOT

Ours - TOT+HF

[Figure 102]

- Fig. 6: Ablation study of DiffusionAgent. The random selection is the baseline method for generating images. The TOT or TOT+HF represent the performance of different agents.

our system produces more visually appealing images, such as “a white towel with a cartoon cat” and a “starry sky”. C. Quantitative Results

The alignment between user preferences and the quantitative findings presented in Table II serves as strong evidence of the robustness and effectiveness of DiffusionAgent. To further evaluate the different generated results, we employed the aesthetic predictor and human feedback related reward model. By comparing the effects of our basic version with the baseline model SD1.5, the results in Table II demonstrate that our overall framework outperforms SD1.5 in terms of imagereward and aesthetic score, achieving improvements of 0.35% and 0.44% respectively.

D. Ablation Study

- 1) Tree-of-Thought and Human Feedback: To validate the

effectiveness of our designed components, we performed a visual analysis of various modules, as shown in Fig. 6. The “Random” variant represents the random sampling model, which produces many images that lack semantic coherence with the input prompts. In contrast, incorporating the treeof-thought (TOT) and human feedback (HF) modules significantly enhances image quality, resulting in improved realism, semantic alignment, and aesthetic appeal. This analysis highlights the advantages of our system in selecting superior models through the integration of TOT and HF components.

- 2) Prompt extension: To evaluate the effectiveness of the

prompt extension agent, we compared generation results using

[Figure 103]

###### Original Extended

The city intersection is eerily quiet, with no cars in sight. The absence of traffic creates a surreal atmosphere, illuminated by dramatic lighting that casts long shadows across the empty street.

a city intersection without cars

[Figure 104]

The image depicts an old man in a small boat, surrounded by a peaceful lake. He is wearing a tattered hat and a weathered coat, and his hands are resting on the oars. The sun is setting in the background, casting a warm, orange glow over the scene. The boat is weathered and worn, but still sturdy, and the old man looks content and at ease.

The old man and the boat

[Figure 105]

[Figure 106]

- Fig. 7: Ablation of Prompt Extension. It aims to provide the riched prompts that produces higher quality images.

[Figure 107]

- Fig. 8: User Study: Comparing DiffusionAgent with SD1.5. Users show a strong preference for the expert models selected by DiffusionAgent across all 10 prompt categories.

original and extended prompts, as shown in Fig. 7. The extended prompts provided more comprehensive descriptions, enriching the context for generating visually appealing outputs. Our analysis revealed that extended prompts significantly improved the aesthetics and detail of the generated images, guiding the model to produce outputs with greater fidelity to the desired artistic style. Additionally, the detailed descriptions in the extended prompts enhanced the model’s ability to capture intricate nuances, resulting in images that exhibited higher intricacy and refinement compared to those generated with the original prompts.

E. User Study

To assess human preferences for generated images, we conducted a user study comparing our model to a baseline model using 100 randomly selected prompts from PartiPrompts [12], generating four images per prompt. Feedback was collected from 20 users who rated the images, resulting in approximately 400 votes for each baseline model (SD1.5 and SD XL). As shown in Fig. 8 and 9, the results consistently favored our model, with users perceiving its outputs as of superior quality versus the baseline.

[Figure 108]

Fig. 9: Comparison of DiffusionAgent-Xl with base model. IV. RELATED WORK

- A. Text-based Image Generation

Initially, Generative Adversarial Networks (GANs) [13], [14] were widely used as the primary approach for Textbased image generation. However, the landscape of image generation has evolved, and diffusion models [1] have emerged as a dominant framework, especially when integrated with text encoders such as CLIP [15] and T5 [16], enabling precise text-conditioned image generation [17]–[19]. For instance, DAELL-2 [17] leverages CLIP’s image embeddings, derived from CLIP’s text embeddings through a prior model, to generate high-quality images. Similarly, Stable Diffusion [2] directly generates images from CLIP’s text embeddings. Imagen [18], on the other hand, utilizes a powerful language model like T5 [16] to encode text prompts, resulting in accurate image generation. To align text-to-image diffusion models with human preferences, recent methods [20]–[22] propose training diffusion models with reward signals. This ensures that the generated images not only meet quality benchmarks but also closely align with human intent and preferences.

- B. Large Language Models (LLMs) for Vision-Language Tasks

The field of natural language processing (NLP) has witnessed a significant transformation with the emergence of large language models (LLMs) [7], [8], [23], which have demonstrated remarkable proficiency in human interaction through conversational interfaces. To further enhance the capabilities of LLMs, the Chain-of-Thought (CoT) framework [10], [11], [24], [25] has been introduced. This framework guides LLMs to generate answers step-by-step, aiming for superior final answers. Recent research has explored innovative approaches by integrating external tools or models with LLMs [26]–[29]. For example, Toolformer [26] empowers LLMs with the ability to access external tools through API tags. Visual ChatGPT [28] and HuggingGPT [29] extend the capabilities of LLMs by enabling them to leverage other models to handle complex tasks that go beyond language boundaries. Drawing inspiration from these endeavors, we embrace the concept of LLMs as versatile tools and leverage this paradigm to guide T2I models to generate high-quality images.

V. CONCLUSION

We propose DiffusionAgent, a one-for-all framework that seamlessly integrates superior generative models and efficiently parses diverse prompts. By leveraging Large Language Models (LLMs), DiffusionAgent gains insights into the intent of input prompts and selects the most suitable model from a

Tree-of-Thought (ToT) structure. This framework offers versatility and exceptional performance across different prompts and domains. To sum up, DiffusionAgent is training-free and can be easily integrated as a plug-and-play solution, offers an efficient and effective pathway for community development.

REFERENCES

- [1] Jonathan Ho, Ajay Jain, and Pieter Abbeel, “Denoising diffusion probabilistic models,” in Advances in Neural Information Processing Systems, 2020, pp. 6840–6851.
- [2] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer, “High-resolution image synthesis with latent diffusion models,” in IEEE Conference on Computer Vision and Pattern Recognition, 2022, pp. 10684–10695.
- [3] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, and Robin Rombach, “Sdxl: Improving latent diffusion models for high-resolution image synthesis,” in International Conference on Learning Representations, 2024, pp. 1–13.
- [4] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman, “Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation,” in IEEE Conference on Computer Vision and Pattern Recognition, 2023, pp. 22500–22510.
- [5] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala, “Adding conditional control to text-to-image diffusion models,” in IEEE International Conference on Computer Vision, 2023, pp. 3836–3847.
- [6] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen, “Lora: Low-rank adaptation of large language models,” arXiv preprint arXiv:2106.09685, 2021.
- [7] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al., “Training language models to follow instructions with human feedback,” in Advances in Neural Information Processing Systems, 2022, pp. 27730–27744.
- [8] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, MarieAnne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al., “Llama: Open and efficient foundation language models,” arXiv preprint arXiv:2302.13971, 2023.
- [9] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al., “Qwen technical report,” arXiv preprint arXiv:2309.16609, 2023.
- [10] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al., “Chain-of-thought prompting elicits reasoning in large language models,” in Advances in Neural Information Processing Systems, 2022, pp. 24824–24837.
- [11] Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola, “Automatic chain of thought prompting in large language models,” arXiv preprint arXiv:2210.03493, 2022.
- [12] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, et al., “Scaling autoregressive models for content-rich text-to-image generation,” arXiv preprint arXiv:2206.10789, 2022.
- [13] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio, “Generative adversarial nets,” in Advances in Neural Information Processing Systems, 2014, pp. 0–9.
- [14] Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, and Dimitris Metaxas, “Stackgan: Text to photo-realistic image synthesis with stacked generative adversarial networks,” in IEEE International Conference on Computer Vision, 2017, pp. 5907–5915.
- [15] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al., “Learning transferable visual models from natural language supervision,” in International Conference on Machine Learning, 2021, pp. 8748–8763.
- [16] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu, “Exploring the limits of transfer learning with a unified text-to-text transformer,” Journal of Machine Learning Research, vol. 21, no. 140, pp. 1–67, 2020.
- [17] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen, “Hierarchical text-conditional image generation with clip latents,” arXiv preprint arXiv:2204.06125, pp. 1–27, 2022.

- [18] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al., “Photorealistic text-to-image diffusion models with deep language understanding,” in Advances in Neural Information Processing Systems, 2022, pp. 36479–36494.
- [19] Yueming Lyu, Tianwei Lin, Fu Li, Dongliang He, Jing Dong, and Tieniu Tan, “Deltaedit: Exploring text-free training for text-driven image manipulation,” arXiv preprint arXiv:2303.06285, 2023.
- [20] Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu, “Aligning text-to-image models using human feedback,” arXiv preprint arXiv:2302.12192, 2023.
- [21] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine, “Training diffusion models with reinforcement learning,” arXiv preprint arXiv:2305.13301, 2023.
- [22] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong, “Imagereward: Learning and evaluating human preferences for text-to-image generation,” in Advances in Neural Information Processing Systems, 2024, pp. 1–33.
- [23] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al., “Palm: Scaling language modeling with pathways,” Journal of Machine Learning Research, vol. 24, no. 240, pp. 1–113, 2023.
- [24] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa, “Large language models are zero-shot reasoners,” in Advances in Neural Information Processing Systems, 2022, pp. 22199– 22213.
- [25] Denny Zhou, Nathanael Sch¨arli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V Le, et al., “Least-to-most prompting enables complex reasoning in large language models,” in International Conference on Learning Representations, 2023, pp. 1–61.
- [26] Timo Schick, Jane Dwivedi-Yu, Roberto Dess`ı, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom, “Toolformer: Language models can teach themselves to use tools,” in Advances in Neural Information Processing Systems, 2024, pp. 1–13.
- [27] D´ıdac Sur´ıs, Sachit Menon, and Carl Vondrick, “Vipergpt: Visual inference via python execution for reasoning,” arXiv preprint arXiv:2303.08128, 2023.
- [28] Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan, “Visual chatgpt: Talking, drawing and editing with visual foundation models,” arXiv preprint arXiv:2303.04671, 2023.
- [29] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang, “Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face,” in Advances in Neural Information Processing Systems, 2024, pp. 1–27.

