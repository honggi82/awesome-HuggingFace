## InternLM-XComposer-2.5: A Versatile Large Vision Language Model Supporting Long-Contextual Input and Output

# arXiv:2407.03320v1[cs.CV]3Jul2024

Pan Zhang∗1, Xiaoyi Dong∗1,2, Yuhang Zang∗1, Yuhang Cao1, Rui Qian1,2, Lin Chen1, Qipeng Guo1, Haodong Duan1, Bin Wang1, Linke Ouyang1, Songyang Zhang1, Wenwei Zhang1, Yining Li1, Yang Gao1, Peng Sun1, Xinyue Zhang1, Wei Li1, Jingwen Li1, Wenhai Wang1,2, Hang Yan1,

Conghui He3, Xingcheng Zhang3, Kai Chen1, Jifeng Dai4,1, Yu Qiao1, Dahua Lin1,2, Jiaqi Wang1,

1Shanghai Artificial Intelligence Laboratory, 2The Chinese University of Hong Kong, 3SenseTime Group, 4Tsinghua University

internlm@pjlab.org.cn

#### Abstract

We present InternLM-XComposer-2.5 (IXC-2.5), a versatile large-vision language model that supports longcontextual input and output. IXC-2.5 excels in various text-image comprehension and composition applications, achieving GPT-4V level capabilities with merely 7B LLM backend. Trained with 24K interleaved image-text contexts, it can seamlessly extend to 96K long contexts via RoPE extrapolation. This long-context capability allows IXC-2.5 to excel in tasks requiring extensive input and output contexts. Compared to its previous 2.0 version, InternLMXComposer-2.5 features three major upgrades in visionlanguage comprehension: (1) Ultra-High Resolution Understanding, (2) Fine-Grained Video Understanding, and (3) Multi-Turn Multi-Image Dialogue. In addition to comprehension, IXC-2.5 extends to two compelling applications using extra LoRA parameters for text-image composition: (1) Crafting Webpages and (2) Composing High-Quality Text-Image Articles. IXC-2.5 has been evaluated on 28 benchmarks, outperforming existing open-source state-ofthe-art models on 16 benchmarks. It also surpasses or competes closely with GPT-4V and Gemini Pro on 16 key tasks. The InternLM-XComposer-2.5 is publicly available at https://github.com/InternLM/InternLMXComposer.

[Figure 1]

Figure 1. Overview of InternLM-XComposer-2.5 (IXC-2.5) performance on benchmarks in different domains, including Video Benchmarks, Structural High-resolution Benchmarks, General Visual QA Benchmarks, Multi-True Multi-Image Benchmark, and Webpage Crafting Benchmark. IXC-2.5 based on InternLM2-7B [143] matches or even surpasses GPT-4V [112] and Gemini Pro [142] in 15 benchmarks. Please refer to Table 3, 4, 5 for details.

#### 1. Introduction

Recent advancements in Large Language Models (LLMs) [29, 55, 111, 121, 146, 147] have sparked interest in the development of Large Vision Language Models (LVLMs) [31, 41, 84, 112, 173, 183]. Leading

* equal contribution. corresponding author.

paradigms like GPT-4 [112], Gemini Pro 1.5 [41], and Claude 3 [3] have achieved considerable success and significantly expanded the range of applications for LLMs. Open-source LVLMs are also being rapidly developed and can compete with proprietary APIs in several benchmarks. However, these open-source models still lag behind closedsource leading paradigms in versatility. They lack the

ability to perform diverse vision-language comprehension and composition tasks, largely due to limited diversity in training corpus and challenges in managing long-context input and output.

To further bridge the gap between proprietary APIs [41, 112] and open-sourced Large Vision Language Models, we are introducing InternLM-XComposer-2.5 (IXC-2.5), a versatile LVLM supporting long-contextual input and output with diverse comprehension and composition capacities. IXC-2.5 excels in existing open-sourced LVLMs with two advantages. (1) Versatility: IXC-2.5 supports a wide range of tasks related to comprehension and composition, such as free-form text-image conversation, OCR, video understanding, article composition with illustrations, and webpage crafting. (2) Long-context capabilities in both input and output: It is natively trained with 24K interleaved image-text data, whose context window can be extended to 96K through positional encoding extrapolation [94], empowering the long-term human-AI interaction and content creation.

Benefiting from the long contextual capability, compared to its previous 2.0 version [33], IXC-2.5 has upgraded three comprehension abilities: (1) Ultra-High Resolution Understanding: IXC-2.5 enhances the dynamic resolution solution proposed in IXC2-4KHD [34] with a native 560 × 560 ViT vision encoder, supporting high-resolution images with any aspect ratio. (2) Fine-Grained Video Understanding: IXC-2.5 treats videos as a ultra-high-resolution composite picture consisting of tens to hundreds of frames, allowing it to capture fine details through dense sampling and higher resolution for each frame. (3) Multi-Turn Multi-Image Dialogue: IXC-2.5 supports free-form multiturn multi-image dialogue, allowing it to naturally interact with humans in multi-round conversations.

Besides comprehension, IXC-2.5 also supports two notable applications by incorporating extra LoRA parameters for text-image composition: (1) Crafting Webpages: IXC-2.5 can be readily applied to create webpages by composing source code (HTML, CSS, and JavaScript) following text-image instructions. (2) Composing High-Quality Text-Image Articles: Compared to IXC-2, IXC-2.5 leverages specially designed Chain-of-Thought (CoT) [153] and Direct Preference Optimization (DPO) [124] techniques to significantly enhance the quality of its written content.

We evaluated the versatility of InternLM-XComposer-

- 2.5 (IXC-2.5) across a range of twenty-eight benchmarks, including five video benchmarks [38, 42, 71, 88, 181], nine structural high-resolution benchmarks [20, 89, 106– 108, 117, 133, 139, 140], twelve general VQA benchmarks [18, 40, 44, 61, 66, 87, 100, 155, 164, 166], one multi-true multi-image benchmark [92], and one webpage crafting benchmark [131]. Compared to previous opensource LVLMs, IXC-2.5 achieved state-of-the-art results in

16 out of 28 benchmarks based on InternLM2-7B [143] backend. As shown in Figure 1, the performance of IXC2.5 matches or even surpasses proprietary APIs, e.g., GPT4V [112] and Gemini Pro [41], in 16 benchmarks.

IXC-2.5 web demo now supports audio input and output using open-source tools [123, 179]. You may try it at https://huggingface.co/spaces/ Willow123/InternLM-XComposer.

#### 2. Related Works

LVLMs for Text-Image Conversation. Large Language Models (LLMs) [8, 12, 13, 29, 55, 60, 111, 115, 121, 143, 146, 147, 168] have received considerable attention because of their impressive performance in language comprehension and generation. Large vision-language models (LVLMs) [5, 9, 22–24, 31, 34, 35, 41, 68, 78, 112, 118, 159, 173, 183] have been developed by integrating LLMs with vision encoders [6, 14, 17, 25, 26, 33, 79, 91, 93, 113, 122, 138, 150, 167, 169, 170, 176] to extend the ability to understand vision content, enabling the application of text-image conversation. Most existing LVLMs are trained for single-image multi-round conversations, while some works [2, 6, 56, 78, 136, 178] have the ability to understand multi-image inputs. However, IXC-2.5 focuses on providing a free-form long-contextual multi-turn multiimage interaction experience [86, 92, 103], which has not been addressed yet.

LVLMs for High Resolution Images Analysis. Understanding high-resolution images has significant potential applications such as OCR and document/chart analysis, which is attracting increased attention in the LVLMs area. In recent works, there are two main strategies to enable high-resolution understanding: (1) High-resolution (HR) visual encoders [47, 74, 102, 141, 151, 177] directly support higher resolution images. (2) Patchification: A highresolution image is cropped into patches [34, 49, 67, 76, 79, 79, 85, 90, 157, 158, 158]. Each patch is processed with a low resolution vision encoder, e.g., CLIP [122] and visual embeddings of patches are further concatenated as inputs for LLM backends. IXC-4KHD [34] scales the supported resolution of open-source LVLMs into 4K and beyond for the first time. IXC-2.5 combines both solutions with a vision encoder trained with a resolution of 560x560 and a dynamic resolution solution proposed in IXC2-4KHD [34], resulting in further improvements.

LVLMs for Video Understanding. In addition to image understanding, the LVLMs area has also witnessed emerging efforts in video analysis [37, 39, 70, 88, 110, 134, 135]. To handle complex video inputs, existing works use sparse sampling or temporal pooling [52, 77, 101, 104, 162], compressed video tokens [57, 69, 73, 120, 154, 172], memory banks [45, 134, 135], and language as a bridge [54, 59, 171] for video understanding. Apart from these video-specific

|Chat on 4K-resolution Image| |
|---|---|
| | |

[Figure 2]

[Figure 3]

Analyze this image in detail

[Figure 4]

[Figure 5]

The infographic is a visual representation of various facts about Dubai. It begins with a statement about Palm Jumeirah, highlighting it as the largest artificial island visible from space. It then provides a historical context, noting that in 1968, there were only a few cars in Dubai, contrasting this with the current figure of more than 1.5

| |
|---|

million vehicles. The infographic also points out that

Dubai has the world's largest Gold Chain, with 7 of the top 10 tallest hotels located there.

[Figure 6]

Additionally, it mentions that the crime rate is near 0%, and the income tax rate is also 0%, with 20% of the world's total cranes operating in Dubai.

| |
|---|

Furthermore, it states that 17% of the population is Emirati, and 83% are immigrants. The Dubai Mall is highlighted as the largest shopping mall in the world, with 1200 stores. The infographic also notes that Dubai has no standard address system, with no zip codes, area codes, or postal services. It mentions that the Burj Khalifa is so tall

| |
|---|

[Figure 7]

that its residents on top floors need to wait longer to break

fast during Ramadan. The infographic also includes information about Dubai's climate-controlled City, with the Royal Suite at Burj Al Arab costing $24,000 per night. Lastly, it notes that the net worth of the four listed billionaires is roughly equal to the GDP of Honduras.

The HD image (1200 x 4679)

###### Multi-Turn Multi-Image Dialogue

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

| |
|---|

I want to buy a car from the three given cars, analyze their advantages and weaknesses one by one

[Figure 12]

When considering the purchase of a car from the three options provided, it's essential to evaluate each vehicle based on its advantages and weaknesses. The first image shows a Mercedes-Benz SUV, which is known for its luxury, comfort, and reliability. Its strengths include spacious interiors, advanced safety features, and a reputation for durability. However, it may not offer the same performance as the other two vehicles, especially in terms of speed and handling.

The second image depicts a Bugatti Veyron, an iconic supercar renowned for its power, speed, and engineering prowess. Its advantages lie in its exceptional performance capabilities, with a top speed exceeding 250 mph and a powerful engine that can deliver over 1,100 horsepower. However, this comes at a significant price, both financially and in terms of maintenance costs. Additionally, while the

Veyron offers unparalleled driving experience, it might be less practical for everyday use due to its limited seating capacity and high

fuel consumption. The third image shows a BMW i8, a hybrid sports car that combines electric propulsion with a combustion engine. This vehicle's benefits include lower emissions, efficient energy usage, and impressive acceleration times. It also boasts a stylish design and advanced technology features. However, one potential weakness could be the relatively short range compared to fully electric cars, requiring frequent charging stops if used extensively for long-distance travel.

[Figure 13]

[Figure 14]

How about this car

[Figure 15]

The car in the image is a red Ferrari, which is known for its high performance and luxury. This particular model appears to be a modern iteration of the Ferrari 812 Superfast or a similar high-performance vehicle from the brand. The advantages of this car include its powerful engine, likely delivering over 800 horsepower, making it one of the fastest cars on the road. Its aerodynamic design suggests excellent handling and stability at high speeds. Additionally, as a Ferrari, it comes with a reputation for superior craftsmanship and quality. However, there are also some potential weaknesses to consider. One might argue that the high price tag associated with owning such a vehicle could be a deterrent for many buyers. Maintenance costs can also be quite significant due to the complex nature of Ferrari engines and the specialized parts required. Furthermore, while the car's performance is unmatched by most standards, it may not offer the same level of comfort and practicality as more conventional vehicles, especially when considering long-distance travel or daily commuting.

Figure 2. The capabilities of IXC-2.5: Chat on 4K-resolution Image and Multi-Turn Multi-Image Dialogue.

###### Chat on Video

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

| |
|---|

frames

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

What is the most likely use of this video?

[Figure 25]

The most likely use of this video is as an advertisement for Starbucks. The sequence of events depicted in the video, such as the preparation of a coffee drink, the interaction between customers and baristas, and the closing shot of the Starbucks logo, are characteristic of a promotional piece designed to showcase the brand's products and services. The emphasis on the coffee-making process, the use of the Starbucks app, and the friendly interactions between customers and staff all serve to create a positive image of the brand. Additionally, the final shot of the Starbucks logo with the text "IT STARTS WITH YOU" suggests a call to action for viewers to engage with the brand, further supporting the idea that this video is an advertisement aimed at attracting new customers and encouraging existing ones to continue patronizing Starbucks.

|Resume-to-Homepage| |
|---|---|
| | |

|[Figure 26]<br><br>[Figure 27]|
|---|

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

| |
|---|

[Figure 32]

[Figure 33]

| |
|---|

[Figure 34]

| |
|---|

[Figure 35]

[Figure 36]

### Generate Code and Render

| |
|---|

[Figure 37]

[Figure 38]

Resume Homepage Some Crop Regions

| | | |
|---|---|---|
| |Instruction-aware Webpage Generation| |

[Figure 39]

[Figure 40]

[Figure 41]

Generate a website for Research institutions.

The name is Shanghai AI lab. Top

Navigation Bar is blue. Below left, an image shows the logo of the lab. In the right, there is a passage of text below that describes the mission of the laboratory. There are several images to show the research projects of Shanghai AI lab.

Generate Code and Render

Instruction

Figure 3. The capabilities of IXC-2.5: Chat on Video, Resume-to-Homepage and Instruction-aware Webpage Generation.

| | | |
|---|---|---|
| |Screenshot-to-Code| |

|[Figure 42]|
|---|

|[Figure 43]|
|---|

[Figure 44]

Generate Code and Render

| | |
|---|---|
|Article Composing| |

Generate an article about the Impact and Measures Regarding Global Climate Change. The outline is as follows:

- 1. Describe the impact of climate change on the natural environment, including rising sea levels and an increase in extreme weather events.
- 2. Analyze the effects of climate change on industries such as agriculture and tourism.
- 3. Explore international efforts to address climate change, such as the Paris Climate Agreement.
- 4. Discuss actions that individuals and communities can take at a personal level.

Impact and Measures Regarding Global Climate Change

The most likely use of this video is as an advertisement for Starbucks. The sequence of events depicted in the video, such as the preparation of a coffee drink, the interaction between customers and baristas, and the closing shot of the Starbucks logo, are characteristic of a promotional piece designed to showcase the brand's products and services. The emphasis on the coffee-making process, the use of the Starbucks app, and the friendly interactions between customers and staff all serve to create a positive image of the brand. Additionally, the final shot of the Starbucks logo with the text "IT STARTS WITH

YOU" suggests a call to action for viewers to engage with the brand, further supporting the idea that this video is an advertisement aimed at attracting new

customers and encouraging existing ones to continue patronizing Starbucks.

[Figure 45]

Impact of Climate Change on the Environment

One of the most visible impacts of climate change is the rise in sea levels. As temperatures increase,

glaciers and ice caps melt, contributing to a rise in sea level that threatens coastal communities around the world. This can lead to saltwater intrusion into freshwater supplies, affecting agriculture and drinking water quality. Additionally, increased precipitation can cause flooding, while droughts can result in crop failure and food shortages. Another significant impact of climate change is the alteration of ecosystems. As temperatures rise, plant and animal species are forced to adapt to new conditions or face extinction. This can lead to a loss of biodiversity and the collapse of entire ecosystems. The warming of oceans also poses a threat to marine life, including coral reefs, which serve as vital habitats for many ocean species.

[Figure 46]

Impact of Climate Change on Industry

Climate change has a profound impact on various industries, particularly those dependent on natural resources. Agriculture, for example, is highly vulnerable to changes in temperature and precipitation patterns. Droughts can reduce crop yields, while floods can destroy crops and livestock. This can lead to food shortages and higher food prices, affecting both producers and consumers. The tourism industry is another sector significantly impacted by climate change. Extreme weather events, such as hurricanes and typhoons, can disrupt travel plans and damage infrastructure, leading to a loss of revenue for businesses and jobs. Furthermore, the warming of oceans and the melting of glaciers can alter the aesthetic appeal of beach destinations, reducing their attractiveness to tourists.

International Efforts to Address Climate Change

In response to the global challenge posed by climate change, countries have come together to form international agreements aimed at mitigating its impact. The Paris Climate Agreement, signed by 195 countries in 2015, is a landmark effort to address climate change. The agreement sets a goal of limiting global warming to well below 2 degrees Celsius above pre-industrial levels and pursuing efforts to limit the temperature increase to 1.5 degrees Celsius.

The Paris Agreement recognizes the importance of reducing greenhouse gas emissions and increasing resilience to the impacts of climate change. Countries are required to submit nationally determined contributions (NDCs), outlining their efforts to reduce emissions and adapt to the impacts of climate change. While the implementation of these measures is voluntary, countries are encouraged to increase their ambition over time to meet the goals of the agreement.

[Figure 47]

[Figure 48]

Personal Level Actions While international efforts are crucial in addressing climate change, individuals and communities can also take actions at a personal level. Some measures include:

- - Reducing energy consumption: This can be achieved by using energy-efficient appliances, turning off lights when not in use, and reducing overall energy usage.
- - Using alternative transportation: Walking, cycling, carpooling, and using public transportation can help reduce greenhouse gas emissions from vehicles.
- - Reducing waste: Properly recycling and composting waste can help reduce methane emissions from landfills.
- - Supporting renewable energy: Investing in solar panels, wind turbines, and other forms of renewable energy can help reduce reliance on fossil fuels.
- - Planting trees: Trees absorb carbon dioxide and produce oxygen, making them a valuable tool in the fight against climate change.

In conclusion, climate change poses a significant threat to the natural environment, industry, and human populations. However, through international efforts and personal actions, it is possible to mitigate its impact and create a more sustainable future.

Figure 4. The capabilities of IXC-2.5: Screenshot-to-Code and Article Composing.

[Figure 49]

designs, video analysis can also be formulated to understand a high-resolution composite picture consisting of sampled video frames [63, 156, 174]. Benefiting from the ability to comprehend ultra-high-resolution images and long context, IXC-2.5 exhibits strong performance on various video benchmarks for LVLMs.

Encode & Merge

[Figure 50]

[Figure 51]

<IMAGE1>

[Figure 52]

<IMAGE 2>

[Figure 53]

| |
|---|
| |
|\n|
| |
| |
|\n|
|sp|

<IMAGE 3>

[Figure 54]

|\n|
|---|
|\n|

| | |
|---|---|
| | |

[Figure 55]

Flatten

<IMAGE N>

[Figure 56]

Global View

Resize

Resize

[Figure 57]

Webpage Generation. Pix2Code [10] presents an endto-end solution for UI-to-code transformation leveraging CNNs and RNNs. This approach contends with the challenges posed by intricate visual encoding and extensive text decoding when applied to real-world UIs. In the sphere of recent advancements, works such as Sightseer [64], DCGen [148], and Design2Code [131] have employed large vision-language models trained on synthetic screenshotHTML paired datasets like WebSight v0.1 or v0.2 [64] to facilitate HTML code generation. Nevertheless, the synthesized web page datasets have been critiqued for their simplicity and lack of diversity. These studies generally concentrate on the screenshot/sketch-to-code task. In contrast, our IXC-2.5 model extends these capabilities to include screenshot-to-code, instruction-aware webpage generation, and resume-to-homepage tasks. IXC-2.5 is trained using a combination of high-quality synthesized and realworld web data. Furthermore, IXC-2.5 is proficient in generating JavaScript code, thereby enabling the development of interactive front-end webpages.

<IMAGE1>

[Figure 58]

| | |
|---|---|
| | |
|\n| |
| | |
| | |
|\n| |
| | |
| | |
|\n| |
| | |
| | |
| | |
|\n| |
| | |
| | |
|\n| |
| | |
| | |
|\n| |
| | |
| | |
|\n| |
| | |
| | |
|\n| |

<IMAGE 2>

[Figure 59]

<IMAGE 3>

[Figure 60]

Local View

[Figure 61]

| | |
|---|---|
| | |

|\n|
|---|
|\n|

<IMAGE N>

[Figure 62]

PLoRA

|\n|
|---|
|\n|

| | |
|---|---|
| | |

Video Dynamic Partition

Image

Dynamic Partition

Flatten

|\n|
|---|
|\n|

| | |
|---|---|
| | |

|[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br><IMAGE 3><br><br><IMAGE 2><br><br><IMAGE1><br><br>[Figure 66]<br><br>[Figure 67]<br><br><IMAGE 3><br><br><br><IMAGE 2><br><br><IMAGE1><br><br>[Figure 68]<br><br>[Figure 69]<br><br><IMAGE 3><br><br><br><IMAGE 2><br><br><IMAGE1><br><br>[Figure 70]<br><br>[Figure 71]<br><br><IMAGE 2><br><br><IMAGE1>|
|---|
|[Figure 72]<br><br>[Figure 73]<br><br><IMAGE N><br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br><IMAGE N><br><br>[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br><IMAGE N><br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br><IMAGE N><br><br><IMAGE 3>|

[Figure 83]

[Figure 84]

| | |
|---|---|
| | |

|\n|
|---|
|\n|

[Figure 85]

[Figure 86]

| |
|---|

| |
|---|

[Figure 87]

| |
|---|

LLM

Tokenize

Preference Alignment. Reinforcement Learning from Human Feedback (RLHF) [115] and Reinforcement Learning from AI Feedback (RLAIF) [7] have shown great promise in aligning LLMs across various domains, including improving logical reasoning and generating helpful and harmless outputs. The typical approach involves training a reward model using human or AI preference data and finetuning the LLM to maximize the expected reward function with optimization algorithms like Proximal Policy Optimization (PPO) [126]. Alternatively, Direct Preference Optimization (DPO) [124] and the following works [36, 116] have emerged as leading methods that implicitly represent the reward score and eliminate the need for a separate reward model. Building on the success of RLHF and RLAIF in LLMs, recent studies have successfully extended RLHF/RLAIF algorithms for multimodal LVLMs [72, 119, 163, 180, 182] to reduce hallucination. In this work, we investigate the application of preference alignment techniques to the text-image article composition task, with a focus on generating high-quality and stable response results.

Encode & Merge

Pease describe theimage /video …………

Figure 5. Framework of IXC-2.5 that supports the multi-modal inputs, including text, single/multiple images, and videos.

XComposer2-4KHD [34] (IXC2 and IXC2-4KHD for simplicity), including a light-weight Vision Encoder OpenAI ViT-L/14 [122], Large Language Model InternLM27B [13], and Partial LoRA [33] for efficient alignment. We recommend the readers to the IXC2 and IXC2-4KHD papers for more details.

##### 3.2. Multi-modal Input

Our IXC-2.5 supports diverse input modalities, including text, single/multiple images, and videos. As showin in Figure 5, a Unified Dynamic Image Partition strategy is adopted for both videos and multiple images with any resolutions and aspect ratios.

#### 3. Method

Image Processing. We mainly follow the Dynamic Image Partition and Global-Local Format design used in IXC24KHD [34] with a few modifications. For the vision encoder, we reuse the ViT of 490 × 490 resolution used in IXC2 and further increase its resolution to 560 × 560, so that each sub-image has 400 tokens.

- 3.1. Model Architecture The model architecture of InternLM-XComposer-2.5 (IXC-

- 2.5 in the following for simplicity) mainly follows the design of InternLM-XComposer2 [33] and InternLM-

For the high-resolution strategy, we unify the different strategies used in the IXC-4KHD into a scaled identity strategy. Given a maximum partition number H, the image x with size [h,w] is resized and padded to the new image xˆ with size [ph × 560,pw × 560]. This process is subject to the following constraints:

- pw1 × ⌈pw1 × h/w⌉ ≤ H; (1)
- pw2 = ⌈w ∗ s/560⌉ (2) pw = min(pw1,pw2) ;ph = ⌈pw × h/w⌉ (3)

where s is the scale factor, pw and ph represent the number of patches in each row and column, respectively.

For multi-image input, we assign an index to each image like <IMAGE i>, i ∈ {1,2,3,...} and format the image and text in an interleaved format.

Video Processing. We sample frames from the given video and concatenate them along the short side of the frame, leading to a high-resolution image. The frame index is also written in the image to provide the temporal relation.

Audio Processing. IXC-2.5 web demo supports audio input and output using open-source tools. For audio input, we employ Whisper [123] to transcribe audio into text. For audio output, we utilize MeloTTS [179] to convert the text back into audio.

##### 3.3. Pre-training

During the pre-training phase, the LLM (InternLM27B [143]) is frozen while both the vision encoder and Partial LoRA [33] are fine-tuned to align the visual tokens with the LLM. The data used for pre-training is shown in Table 1.

In practice, we employ the CLIP ViT-L-14-490 [122] from IXC2 as the vision encoder and further increase its resolution to 560 × 560. For the Unified Dynamic Image Partition strategy [34], we set the maximum number H = 12 for the pertaining. For the Partial LoRA [33], we set a rank of 256 for all the linear layers in the LLM decoder block. Our training process involves a batch size of 4096 and spans across 2 epochs. The learning rate linearly increases to 2 × 10−4 within the first 1% of the training steps. Following this, it decreases to 0 according to a cosine decay strategy. To preserve the original knowledge of the vision encoder, we apply a layer-wise learning rate (LLDR) decay strategy [33], and the decay factor is set to 0.90.

##### 3.4. Supervised Fine-tuning

We fine-tune the model with data listed in Table 2. The maximum number H of the Unified Dynamic Image Partition strategy is 24 to handle extremely large images and videos. For video datasets, the IXC-2.5 is trained with large images concatenated by at most 64 frames. The largest training context is set to a 24,000 context window size, where the MMDU [92] dataset can achieve this limitation.

In practice, we jointly train all the components with a batch size of 2048 over 4000 steps. Data from multiple sources are sampled in a weighted manner, with the weights based on the number of data from each source. The maximum learning rate is set to 5 × 10−5, and each component has its own unique learning strategy. For the vision encoder, we set the LLDR to 0.9, which aligns with the pretraining strategy. For the LLM, we employ a fixed learning rate scale factor of 0.2. This slows down the update of the LLM, achieving a balance between preserving its original capabilities and aligning it with vision knowledge.

##### 3.5. Webpage Generation

We enhance the capabilities of the IXC-2.5 to include automated webpage generation. Specifically, the IXC-2.5 is now equipped to autonomously construct web pages, utilizing HTML, CSS, and JavaScript, based on input in the form of a visual screenshot, a set of free-form instructions, or a resume document. Current open-source general-purpose large language models frequently demonstrate suboptimal performance in generating HTML and CSS relative to their proficiency in natural language generation. To address this limitation, we propose training the screenshot-to-code task using extensive datasets from WebSight v0.1/v0.2 [64], and Stack v2 [95]. Subsequently, we fine-tune the model with a smaller, meticulously crafted dataset consisting of instruction-aware webpage generation and personal page generation examples.

Screenshot-to-code. In addition to the WebSight [64] datasets, we preprocess the HTML and CSS code from the Stack v2 [95] dataset to facilitate screenshot-to-code training. Initially, we combine the CSS and HTML code into a single file. Subsequently, we remove all comments, JavaScript code, and external links. Furthermore, we eliminate any CSS styles that are not referenced by the HTML code. We convert all files into screenshots, subsequently discarding those that did not render successfully. The remaining screenshots are then processed using the IXC24KHD [34] model to assess the quality of the web pages. Following the exclusion of low-quality web pages, we retained a final set of three remaining about 250,000 highquality web pages.

We conduct training on the LoRA model utilizing the three aforementioned datasets. The LoRA rank is set to 512. The training protocol employs a batch size of 512 and is executed over a single epoch. Initially, the learning rate is incremented linearly to 1 × 10−4 within the first 1% of the training iterations. Subsequently, the learning rate decreases to 0 following a cosine decay schedule.

Instruction-aware Webpage Generation. A pivotal attribute of large language models lies in their capability to adhere to human instructions. To facilitate web page generation based on freeform instructions, we propose con-

Task Dataset General Semantic Alignment ShareGPT4V-PT [17], COCO [21], Nocaps [1], TextCaps [132], LAION [125], SBU [114], CC 3M [129] ALLaVA [15] World Knowledge Alignment Concept Data [173] Vision Capability Enhancement WanJuan [46], Flicker[160], MMC-Inst[82], RCTW-17[130], CTW[165], LSVT[137], ReCTs[175], ArT[28]

Table 1. Datasets used for Pre-Training. The data are collected from diverse sources for the three objectives.

Task Dataset Caption ShareGPT4V [17], COCO [21], Nocaps [1] General QA VQAv2 [4], GQA [53], OK-VQA [105]

VD [32], RD [16], VSR [81], ALLaVA-QA [15] Multi-Turn QA MMDU [92] Science QA AI2D [61], SQA [98], TQA [62], IconQA [97] Chart QA DVQA [58], ChartQA [106], ChartQA-AUG [106] Math QA MathQA [161], Geometry3K [96], TabMWP [99],

CLEVR-MATH [80], Super [75] World Knowledge QA A-OKVQA [127], KVQA [128], ViQuAE [65] OCR QA TextVQA [133], OCR-VQA [109], ST-VQA [11] HD-OCR QA InfoVQA[108], DocVQA [107], TabFact [20],

WTQ [117], DeepForm [139], Visual MRC [140] Video ShareGPT4Video [19], ActivityNet [37] Conversation LLaVA-150k [84], LVIS-Instruct4V [149]

ShareGPT-en&zh [27], InternLM-Chat [143]

- Table 2. Datasets used for Supervised Fine-Tuning. We collect data from diverse sources to empower the model with different capabilities.

structing data through querying closed-source large language models. Specifically, we utilize GPT-4 to generate diverse instructions and concepts for web page creation, encompassing elements such as type, style, and layout. Subsequently, these instructions are harnessed to query Claude-3sonnet [3] for the actual web page generation process. This approach results in 18,000 high-quality, instruction-aware samples. Additionally, we employ Tailwind CSS instead of traditional CSS, given its succinct nature.

Resume-to-homepage. In addition to instruction-aware webpage generation, we introduce a more practical task. Specifically, given a resume, the model is designed to generate a personal homepage. This homepage not only encapsulates the information present in the resume but also presents it with a well-structured and visually appealing format, improving both content organization and aesthetic layout. To generate corresponding datasets, we propose an idea-resume-homepage data generation pipeline. Initially, we leverage GPT-4 to produce resume ideas tailored for diverse personas, such as researchers, students, and engineers. GPT-4 is tasked with generating these resumes in markdown format based on the provided ideas. Upon obtaining the generated resumes, we then prompt Claude-3-sonnet [3] to create corresponding homepages from these resumes. To enhance the interactivity of these webpages, Claude-3-

sonnet is also utilized to generate JavaScript events based on the HTML code. In total, we have constructed a dataset comprising 2,000 samples.

Upon constructing the dataset for instruction-aware webpage generation and resume-to-homepage, we subsequently fine-tuned the LoRA model for 10 epochs. All other experimental settings were maintained consistent with those employed during the screenshot-to-code training phase.

##### 3.6. Article Composing

Generating high-quality text-image articles (e.g., poetry, novels, short stories, and essays) is a crucial capability for AI assistants, with various applications in daily life, including education and entertainment. Building upon the IXC2.5 SFT model π in Section 3.4, we enhance creative writing capabilities for generating high-quality text-image articles. However, collecting high-quality text-image articles is a rare and expensive endeavor. Direct fine-tuning on scarce instruction data can lead to unstable responses from LVLMs in most cases. To overcome these challenges, we propose a scalable pipeline that integrates supervised fine-tuning, reward modeling, preference data collection, and DPO alignment for high-quality and stable article generation.

Supervised Fine-tuning. We begin with the SFT model π (Section 3.4) and a collection of 5,000 instruction tuning data samples D from IXC2 [33], focused on article writing. Due to the limited scale of the instruction data, we use the SFT model to rewrite the original prompts using the Chainof-Thought (CoT) technique [152], generating step-by-step prompts to supplement the instruction tuning data as augmented data D∗. We observe that the SFT model is more effective in generating long-form responses when using these augmented prompts. We then train the initial model π on the augmented instruction tuning data via LoRA [51] with the rank of 256 and get the model πref to establish a starting point of our alignment pipeline.

Preference Data Collection. We use the fine-tuned model πref to generate diverse responses for each prompt in the augmented instruction tuning data D∗, using different random seeds. This yields a collection of 80,000 promptresponse pairs. Next, we employ the GPT-4o model to label 2,000 responses with chosen or rejected decisions and give the reasons, which serve as our reward modeling data. We then train a reward model πrm, sharing the same architecture of πref, on the reward modeling data. The reward model is used to make the chosen or rejected prediction on the remaining prompt-response pairs. These selected responses

MME MMB∗1 Temp∗2 Doc Chart Info Text OCR

Deep Visual Tab Video Video Compass VQA QA VQA VQA Bench Form MRC Fact

MVBench MLVU

WTQ

Open-Source VideoChat InternVL LIVA InternVL Qwen-VL InternVL InternVL InternVL InternVL GLM-4v DocOwl DocOwl DocOwl DocOwl Previous SOTA 2-7B[71] 1.5-26B[26]34B[78]1.5-26B[26] 7B[6] 1.5-26B[26]1.5-26B[26]1.5-26B[26]1.5-26B[26] 9B[43] 1.5-8B[50]1.5-8B[50]1.5-8B[50]1.5-8B[50] Performance 60.4 50.4 59.0 42.0 58.4 90.9 83.8 72.5 80.6 77.6 40.6 68.8 246.4 80.2

Closed-source API

GPT-4V [112] 43.5 49.2 59.9 56.0 — 88.4 78.5 75.1 78.0 51.6 — — — Gemini-Pro [142] — — 75.0 49.3 70.6 88.1 74.1 75.2 74.6 68.0 — — — —

IXC-2.5-7B 69.1 58.8 55.8 46.9 67.1 90.9 82.2 69.9 78.2 69.0 53.6 71.2 307.5 85.2

- Table 3. Comparison with closed-source APIs and previous open-source SOTAs on Video Benchmarks and Structural High-resolution Benchmarks. The best results are bold and the second-best results are underlined. ∗1 We scale the score from 0 ∼ 3 to 0 ∼ 100 for easier understanding. ∗2 We report the determinism part (MCQA, Y/N, Caption Match) of TempCompass as the evaluation using GPT-3.5 is not stable.

MMDU MMStar RealWQAMathVista AI2D MMMU MME MMB MMBCN MMB1.1 SEEDI MM-Vet HallB

Open-Source LLaVa1.6 InternVL WeMM WeMM InternVL 360VL InternVL InternVL1.5InternVL1.5InternVL1.5 WeMM GLM-4v WeMM Previous SOTA 8B[83] 1.5-26B[26] 8B[145] 8B[145] 1.5-26B[26]70B[144]1.5-26B[26]1.5-26B[26] 1.5-26B[26] 1.5-26B[26]8B[145] 14B[43] 8B[145] Performance 42.8 57.1 68.1 54.9 80.6 53.4 2,189.6 82.3 80.7 79.7 75.9 58.0 47.5

Closed-source API

GPT-4V [112] 66.3 57.1 68.0 47.8 75.5 56.8 1,926.5 81.3 80.2 79.8 69.1 56.8 46.5 Gemini-Pro [142] — 42.6 64.1 45.8 70.2 47.9 1,933.3 73.9 74.3 73.9 70.7 59.2 45.2

IXC-2.5-7B 56.6 59.9 67.8 63.8 81.5 42.9 2,229.0 82.2 80.8 79.4 75.4 51.7 42.4

- Table 4. Comparison with closed-source APIs and previous open-source SOTAs on Multi-Turn Multi-Image Dialog and General Visual QA Benchmarks. The best results are bold and the second-best results are underlined.

are then used to construct the pair data Dp = {x,yw,yl}, while x, yw and yl refer to the prompt, chosen response and rejected response, respectively. Ultimately, we obtain a total of 30,000 preference data Dp for DPO [124] alignment. DPO Alignment. We use the DPO algorithm to update the SFT model πref on target policy from the preference data Dp:

LDPO(πθ,πref) = Ex,y

w,yl∼Dp

(4)

πθ(yw|x) πref(yw|x)

πθ(yl|x) πref(yl|x)

[−log σ(β log(

) − β log(

))].

In practice, we use LoRA with a rank of 256 to get the DPO model πθ. We observe that our model tends to prioritize minimizing the likelihood of dis-preferred responses yl over maximizing the likelihood of preferred responses yw to avoid generating inappropriate or low-quality content.

In summary, our scalable pipeline consists of three primary components. First, we address the challenge of limited instruction tuning data by re-writing original prompts into augmented prompts. Next, we generate diverse responses using different random seeds, enabling the exploration of various creative possibilities. Finally, we apply the DPO algorithm to the chosen and rejected responses to refine our model’s performance. Through our pipeline, our model is capable of generating high-quality articles.

#### 4. Experiments

In this section, we validate the benchmark performance of our InternLM-XComposer-2.5 (IXC-2.5) after supervised fine-tuning.

##### 4.1. LVLM Benchmark results.

In Table 3 and Table 4, we compare our IXC-2.5 on a list of benchmarks with both closed-source APIs and SOTA open-source LVLMs (with comparable model size). Here we report video understanding results on MVBench [71], MLVU [181], MME-Video [42], MMBench-Video [38], TempCompass [88]. For Structural High-resolution understanding, we report results on DocVQA [107], ChartQA [106], InfographicVQA [108], TextVQA [133], OCRBench [89], DeepForm [139], WikiTableQuestion (WTQ) [117], Visual MRC [140], and TabFact [20]. For general visual question answering, we report results on MMStar [18], RealWorldQA[155], MathVista [100], MMMU [166], AI2D [61], MME [40], MMBench (MMB) [87], MMBench-Chinese (MMBCN) [87], MMBench-v1.1 (MMBv1.1) [87], SEED-Bench Image Part (SEEDI)[66], MM-Vet [164], HallusionBench (HallB) [44]. For Multi-True Multi-Image dialogue, we evaluate IXC-2.5 on MMDU [92] benchmark. For webpage crafting, we report a subtask screenshot-to-code [131] since benchmarks for others are not available in the community.

The evaluation is mainly conducted on the OpenCompass VLMEvalKit [30] for the unified reproduction of the results.

Comparison on Video Understanding Benchmarks. As demonstrated in Table 3, IXC-2.5 exhibits competitive performance on fine-grained video understanding tasks, outperforming open-source models on 4 of the 5 benchmarks and being on par with Closed-Source APIs. For example, IXC-2.5 reaches 69.1 on the MVBench, +8.7% higher

Block-Match Text Position Color CLIP Average Closed-source API

GPT-4V [112] 85.8 97.4 80.5 73.3 86.9 84.8 Gemini-Pro [142] 80.2 94.6 72.3 66.2 83.9 79.4

Open-source WebSight VLM-8B [64] 55.9 86.6 77.3 79.4 86.5 77.1 CogAgent-Chat-18B [48] 7.1 18.1 13.3 13.0 75.5 25.4 Design2Code-18B [131] 78.5 96.4 74.3 67.0 85.8 80.4

IXC-2.5-7B 81.9 95.6 80.9 80.8 86.5 85.1

- Table 5. Screenshot-to-code. Comparison with closed-source APIs and open-source models on Design2Code benchmark. The best results are bold and the second-best results are underlined.

than the previous SOTA method VideoChat2-7B and outperforms GPT-4V with +25.6%. For the recent challenging MMBench-Video, IXC-2.5 reaches the SOTA performance on open-source models and performs close to Gemini-Pro.

Comparison on Structural High-resolution Benchmarks. Benefiting from the unified image partition strategy, IXC-2.5 could handle diverse kinds of images. Table 3 reports its performance on several structural highresolution benchmarks. IXC-2.5 with only 7B parameters performs on par with the current large open-source LVLMs and close-source APIs. For example, IXC-2.5 gets 90.9% on the DocVQA test set, the same as InternVL-1.5 which has nearly 4× parameters. For the highly structured form and table understanding tasks, IXC-2.5 outperforms DocOwl 1.5-8B with +13.0%, +2.4%, +5.0% on WikiTableQuestion, DeepForm and TableFace respectively.

Comparison on Multi-Image Multi-Turn Benchmarks. IXC-2.5 is capable of taking multiple images as input and conducting multi-round free-form dialogue based on them. We evaluate it quantitatively on the newly proposed MMDU benchmark [92]. As shown in Table 4, the IXC-2.5 model demonstrates superior performance, outperforming the previous SOTA open-source model by a significant margin of 13.8%. This notable improvement highlights the effectiveness of our approach in advancing the capabilities of multiimage and multi-turn understanding.

Comparison on General Visual QA Benchmarks. IXC2.5 is designed as a general LVLM to handle diverse multimodal tasks. Here we report its performance on general visual QA benchmarks. As shown in Table 4, the IXC-2.5 shows superb performance on these benchmarks and on par with current large open-source LVLMs and closed-source APIs. For example, IXC-2.5 gets 59.9% on the challenging MMStar and outperforms GPT-4V and Gemini-Pro. On the RealWorldQA, IXC-2.5 also performs better than GeminiPro and close to GPT-4V.

Comparison on Screenshot-to-code Benchmark. Table 5 presents the comparison results on the Design2Code [131] benchmark that assesses the ability to translate visual design into code implementation. Our IXC-2.5 even surpasses the GPT-4v on average performance, which highlights the

potential of IXC-2.5 to excel in bridging the gap between visual design and code implementation.

#### 5. Conclusion

We have introduced InternLM-XComposer-2.5 (IXC-2.5), a cutting-edge Large Vision-Language Model (LVLM) boasting long-contextual input and output capabilities that enable advanced features such as ultra-high resolution image understanding, fine-grained video understanding, multiturn multi-image dialogue, webpage generation, and article composing. Our comprehensive experiments demonstrate that IXC-2.5 achieves competitive performance, remarkably, with a relatively modest 7B Large Language Model (LLM) backend.

Our model sets out a promising research direction that can extend to a more contextual multi-modal environment, including long-context video understanding (e.g., long movies) and long-context interaction history, to better assist humans in real-world applications.

Acknowledgements: We deeply express our gratitude to Prof. Chao Zhang from Tsinghua University for suggestions about audio models and tools.

#### References

- [1] Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. Nocaps: Novel object captioning at scale. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019. 8
- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning,

2022. 2

- [3] Anthropic. Claude 3 haiku: our fastest model yet,

2024. Available at: https://www.anthropic.com/ news/claude-3-haiku. 1, 8

- [4] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi Parikh. VQA: Visual question answering. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2015. 8
- [5] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, and Ludwig Schmidt. Openflamingo: An opensource framework for training large autoregressive visionlanguage models. arXiv.org, 2023. 2
- [6] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-VL: A frontier large vision-language model with versatile abilities. arXiv.org, 2023. 2, 9
- [7] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional AI: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073, 2022. 6
- [8] Baichuan. Baichuan 2: Open large-scale language models. arXiv.org, 2023. 2
- [9] Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, and Sa˘gnak Ta¸sırlar. Introducing our multimodal models, 2023. 2
- [10] Tony Beltramelli. pix2code: Generating code from a graphical user interface screenshot. In Proceedings of the ACM SIGCHI symposium on engineering interactive computing systems, 2018. 6
- [11] Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez, Marc¸al Rusinol, Ernest Valveny, CV Jawahar, and Dimosthenis Karatzas. Scene text visual question answering. In ICCV, 2019. 8
- [12] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al.

- Language models are few-shot learners. Advances in Neural Information Processing Systems (NeurIPS), 33:1877– 1901, 2020. 2
- [13] Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, Xiaoyi Dong, Haodong Duan, Qi Fan, Zhaoye Fei, Yang Gao, Jiaye Ge, Chenya Gu, Yuzhe Gu, Tao Gui, Aijia Guo, Qipeng Guo, Conghui He, Yingfan Hu, Ting Huang, Tao Jiang, Penglong Jiao, Zhenjiang Jin, Zhikai Lei, Jiaxing Li, Jingwen Li, Linyang Li, Shuaibin Li, Wei Li, Yining Li, Hongwei Liu, Jiangning Liu, Jiawei Hong, Kaiwen Liu, Kuikun Liu, Xiaoran Liu, Chengqi Lv, Haijun Lv, Kai Lv, Li Ma, Runyuan Ma, Zerun Ma, Wenchang Ning, Linke Ouyang, Jiantao Qiu, Yuan Qu, Fukai Shang, Yunfan Shao, Demin Song, Zifan Song, Zhihao Sui, Peng Sun, Yu Sun, Huanze Tang, Bin Wang, Guoteng Wang, Jiaqi Wang, Jiayu Wang, Rui Wang, Yudong Wang, Ziyi Wang, Xingjian Wei, Qizhen Weng, Fan Wu, Yingtong Xiong, Chao Xu, Ruiliang Xu, Hang Yan, Yirong Yan, Xiaogui Yang, Haochen Ye, Huaiyuan Ying, Jia Yu, Jing Yu, Yuhang Zang, Chuyu Zhang, Li Zhang, Pan Zhang, Peng Zhang, Ruijie Zhang, Shuo Zhang, Songyang Zhang, Wenjian Zhang, Wenwei Zhang, Xingcheng Zhang, Xinyue Zhang, Hui Zhao, Qian Zhao, Xiaomeng Zhao, Fengzhe Zhou, Zaida Zhou, Jingming Zhuo, Yicheng Zou, Xipeng Qiu, Yu Qiao, and Dahua Lin. Internlm2 technical report. arXiv preprint arXiv:2403.17297, 2024. 2, 6
- [14] Yuhang Cao, Pan Zhang, Xiaoyi Dong, Dahua Lin, and Jiaqi Wang. DualFocus: Integrating macro and micro perspectives in multi-modal large language models. arXiv preprint arXiv:2402.14767, 2024. 2
- [15] Guiming Hardy Chen, Shunian Chen, Ruifei Zhang, Junying Chen, Xiangbo Wu, Zhiyi Zhang, Zhihong Chen, Jianquan Li, Xiang Wan, and Benyou Wang. ALLaVA harnessing gpt4v-synthesized data for a lite vision-language model. arXiv preprint arXiv:2402.11684, 2024. 8
- [16] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv.org, 2023. 8
- [17] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023. 2, 8
- [18] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024. 2, 9
- [19] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. ShareGPT4Video: Improving video understanding and generation with better captions. arXiv preprint arXiv:2406.04325, 2024. 8
- [20] Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, Shiyang Li, Xiyou Zhou, and William Yang Wang. TabFact: A large-scale dataset for table-based fact verification. In Proceedings of the Inter-

national Conference on Learning Representations (ICLR),

2020. 2, 8, 9

- [21] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollar, and C. Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server, 2015. 8
- [22] Xi Chen, Josip Djolonga, Piotr Padlewski, Basil Mustafa, Soravit Changpinyo, Jialin Wu, Carlos Riquelme Ruiz, Sebastian Goodman, Xiao Wang, Yi Tay, Siamak Shakeri, Mostafa Dehghani, Daniel Salz, Mario Lucic, Michael Tschannen, Arsha Nagrani, Hexiang Hu, Mandar Joshi, Bo Pang, Ceslee Montgomery, Paulina Pietrzyk, Marvin Ritter, AJ Piergiovanni, Matthias Minderer, Filip Pavetic, Austin Waters, Gang Li, Ibrahim Alabdulmohsin, Lucas Beyer, Julien Amelot, Kenton Lee, Andreas Peter Steiner, Yang Li, Daniel Keysers, Anurag Arnab, Yuanzhong Xu, Keran Rong, Alexander Kolesnikov, Mojtaba Seyedhosseini, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, and Radu Soricut. Pali-x: On scaling up a multilingual vision and language model, 2023. 2
- [23] Xi Chen, Xiao Wang, Lucas Beyer, Alexander Kolesnikov, Jialin Wu, Paul Voigtlaender, Basil Mustafa, Sebastian Goodman, Ibrahim Alabdulmohsin, Piotr Padlewski, Daniel Salz, Xi Xiong, Daniel Vlasic, Filip Pavetic, Keran Rong, Tianli Yu, Daniel Keysers, Xiaohua Zhai, and Radu Soricut. Pali-3 vision language models: Smaller, faster, stronger, 2023.
- [24] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Nan Ding, Keran Rong, Hassan Akbari, Gaurav Mishra, Linting Xue, Ashish Thapliyal, James Bradbury, Weicheng Kuo, Mojtaba Seyedhosseini, Chao Jia, Burcu Karagol Ayan, Carlos Riquelme, Andreas Steiner, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, and Radu Soricut. Pali: A jointly-scaled multilingual languageimage model, 2023. 2
- [25] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023. 2
- [26] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, Ji Ma, Jiaqi Wang, Xiaoyi Dong, Hang Yan, Hewei Guo, Conghui He, Botian Shi, Zhenjiang Jin, Chao Xu, Bin Wang, Xingjian Wei, Wei Li, Wenjian Zhang, Bo Zhang, Pinlong Cai, Licheng Wen, Xiangchao Yan, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. How far are we to gpt4v? closing the gap to commercial multimodal models with open-source suites, 2024. 2, 9
- [27] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, 2023. 8

- [28] Chee Kheng Chng, Yuliang Liu, Yipeng Sun, Chun Chet Ng, Canjie Luo, Zihan Ni, ChuanMing Fang, Shuaitao Zhang, Junyu Han, Errui Ding, et al. Icdar2019 robust reading challenge on arbitrary-shaped text-rrc-art. In International Conference on Document Analysis and Recognition (ICDAR), 2019. 8
- [29] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv.org, 2022. 1, 2
- [30] OpenCompass Contributors. Opencompass: A universal evaluation platform for foundation models. https: //github.com/open-compass/opencompass,

2023. 9

- [31] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning,

2023. 1, 2

- [32] Abhishek Das, Satwik Kottur, Khushi Gupta, Avi Singh, Deshraj Yadav, Jos´e M.F. Moura, Devi Parikh, and Dhruv Batra. Visual Dialog. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),

2017. 8

- [33] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer2: Mastering free-form text-image composition and comprehension in vision-language large model. arXiv preprint arXiv:2401.16420, 2024. 2, 6, 7, 8
- [34] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Zhe Chen, Xinyue Zhang, Wei Li, Jingwen Li, Wenhai Wang, Kai Chen, Conghui He, Xingcheng Zhang, Jifeng Dai, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer24khd: A pioneering large vision-language model handling resolutions from 336 pixels to 4k hd. arXiv preprint arXiv:2404.06512, 2024. 2, 6, 7
- [35] Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Pete Florence. Palm-e: An embodied multimodal language model. In arXiv preprint arXiv:2303.03378, 2023. 2
- [36] Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. KTO: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306, 2024. 6
- [37] Bernard Ghanem Fabian Caba Heilbron, Victor Escorcia and Juan Carlos Niebles. ActivityNet: A large-scale video

- benchmark for human activity understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2015. 2, 8
- [38] Xinyu Fang, Kangrui Mao, Haodong Duan, Xiangyu Zhao, Yining Li, Dahua Lin, and Kai Chen. MMBench-Video: A long-form multi-shot benchmark for holistic video understanding. arXiv preprint arXiv:2406.14515, 2024. 2, 9
- [39] Xinyu Fang, Kangrui Mao, Haodong Duan, Xiangyu Zhao, Yining Li, Dahua Lin, and Kai Chen. Mmbench-video: A long-form multi-shot benchmark for holistic video understanding, 2024. 2
- [40] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 2, 9
- [41] Chaoyou Fu, Renrui Zhang, Zihan Wang, Yubo Huang, Zhengye Zhang, Longtian Qiu, Gaoxiang Ye, Yunhang Shen, Mengdan Zhang, Peixian Chen, Sirui Zhao, Shaohui Lin, Deqiang Jiang, Di Yin, Peng Gao, Ke Li, Hongsheng Li, and Xing Sun. A challenger to gpt-4v? early explorations of gemini in visual expertise. arXiv preprint arXiv:2312.12436, 2023. 1, 2
- [42] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-MME: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 2, 9
- [43] Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, et al. ChatGLM: A family of large language models from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793, 2024. 9
- [44] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination & visual illusion in large vision-language models, 2023. 2, 9
- [45] Bo He, Hengduo Li, Young Kyun Jang, Menglin Jia, Xuefei Cao, Ashish Shah, Abhinav Shrivastava, and Ser-Nam Lim. Ma-lmm: Memory-augmented large multimodal model for long-term video understanding. arXiv preprint arXiv:2404.05726, 2024. 2
- [46] Conghui He, Zhenjiang Jin, Chaoxi Xu, Jiantao Qiu, Bin Wang, Wei Li, Hang Yan, Jiaqi Wang, and Da Lin. Wanjuan: A comprehensive multimodal dataset for advancing english and chinese large models. ArXiv, abs/2308.10755,

2023. 8

- [47] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. arXiv preprint arXiv:2312.08914,

2023. 2

- [48] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao

- Dong, Ming Ding, et al. CogAgent: A visual language model for gui agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 10
- [49] Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, et al. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding. arXiv preprint arXiv:2403.12895, 2024. 2
- [50] Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, et al. mPLUG-DocOwl 1.5: Unified structure learning for ocr-free document understanding. arXiv preprint arXiv:2403.12895, 2024. 9
- [51] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In Proceedings of the International Conference on Learning Representations (ICLR), 2022. 8
- [52] Suyuan Huang, Haoxin Zhang, Yan Gao, Yao Hu, and Zengchang Qin. From image to video, what do we need in multimodal llms? arXiv preprint arXiv:2404.11865, 2024. 2
- [53] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 8
- [54] Md Mohaiminul Islam, Ngan Ho, Xitong Yang, Tushar Nagarajan, Lorenzo Torresani, and Gedas Bertasius. Video recap: Recursive captioning of hour-long videos. arXiv preprint arXiv:2402.13250, 2024. 2
- [55] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, L´elio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timoth´ee Lacroix, and William El Sayed. Mistral 7b, 2023. 1, 2
- [56] Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max Ku, Qian Liu, and Wenhu Chen. Mantis: Interleaved multiimage instruction tuning, 2024. 2
- [57] Peng Jin, Ryuichi Takanobu, Caiwan Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. arXiv preprint arXiv:2311.08046, 2023. 2
- [58] Kushal Kafle, Brian Price, Scott Cohen, and Christopher Kanan. DVQA: Understanding data visualizations via question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

2018. 8

- [59] Kumara Kahatapitiya, Kanchana Ranasinghe, Jongwoo Park, and Michael S Ryoo. Language repository for long video understanding. arXiv preprint arXiv:2403.14622,

2024. 2

- [60] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for

neural language models. arXiv preprint arXiv:2001.08361,

2020. 2

- [61] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Proceedings of the European Conference on Computer Vision (ECCV), 2016. 2, 8, 9
- [62] Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, and Hannaneh Hajishirzi. Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 8
- [63] Wonkyun Kim, Changin Choi, Wonseok Lee, and Wonjong Rhee. An image grid can be worth a video: Zero-shot video question answering using a vlm, 2024. 6
- [64] Hugo Laurenc¸on, L´eo Tronchon, and Victor Sanh. Unlocking the conversion of web screenshots into html code with the websight dataset. arXiv preprint arXiv:2403.09029,

2024. 6, 7, 10

- [65] Paul Lerner, Olivier Ferret, Camille Guinaudeau, Herv´e Le Borgne, Romaric Besanc¸on, Jos´e G Moreno, and Jes´us Lov´on Melgarejo. Viquae, a dataset for knowledge-based visual question answering about named entities. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 3108–3120, 2022. 8
- [66] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension, 2023. 2, 9
- [67] Bo Li, Peiyuan Zhang, Jingkang Yang, Yuanhan Zhang, Fanyi Pu, and Ziwei Liu. Otterhd: A high-resolution multimodality model, 2023. 2
- [68] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. arXiv.org, 2023. 2
- [69] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 2
- [70] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. arXiv preprint arXiv:2311.17005,

2023. 2

- [71] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2, 9
- [72] Lei Li, Zhihui Xie, Mukai Li, Shunian Chen, Peiyi Wang, Liang Chen, Yazheng Yang, Benyou Wang, and Lingpeng Kong. Silkie: Preference distillation for large visual language models. arXiv preprint arXiv:2312.10665, 2023. 6
- [73] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. arXiv preprint arXiv:2311.17043, 2023. 2

- [74] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-Gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814,

2024. 2

- [75] Zhuowan Li, Xingrui Wang, Elias Stengel-Eskin, Adam Kortylewski, Wufei Ma, Benjamin Van Durme, and Alan L Yuille. Super-clevr: A virtual benchmark to diagnose domain robustness in visual reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 8
- [76] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. arXiv preprint arXiv:2311.06607, 2023. 2
- [77] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023. 2
- [78] Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models, 2024. 2, 9
- [79] Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, et al. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575, 2023. 2
- [80] Adam Dahlgren Lindstr¨om and Savitha Sam Abraham. Clevr-math: A dataset for compositional language, visual and mathematical reasoning. arXiv preprint arXiv:2208.05358, 2022. 8
- [81] Fangyu Liu, Guy Edward Toh Emerson, and Nigel Collier. Visual spatial reasoning. Transactions of the Association for Computational Linguistics (TACL), 2023. 8
- [82] Fuxiao Liu, Xiaoyang Wang, Wenlin Yao, Jianshu Chen, Kaiqiang Song, Sangwoo Cho, Yaser Yacoob, and Dong Yu. Mmc: Advancing multimodal chart understanding with large-scale instruction tuning. arXiv preprint arXiv:2311.10774, 2023. 8
- [83] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023. 9
- [84] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv.org, 2023. 1, 8
- [85] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 2
- [86] Shuo Liu, Kaining Ying, Hao Zhang, Yue Yang, Yuqi Lin, Tianle Zhang, Chuanhao Li, Yu Qiao, Ping Luo, Wenqi Shao, et al. Convbench: A multi-turn conversation evaluation benchmark with hierarchical capability for large vision-language models. arXiv preprint arXiv:2403.20194,

2024. 2

- [87] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhnag, Wangbo Zhao, Yike Yuan, Jiaqi Wang,

- Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player? arXiv:2307.06281, 2023. 2, 9
- [88] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. TempCompass: Do video llms really understand videos? arXiv preprint arXiv:2403.00476, 2024. 2, 9
- [89] Yuliang Liu, Zhang Li, Biao Yang, Chunyuan Li, Xucheng Yin, Cheng lin Liu, Lianwen Jin, and Xiang Bai. On the hidden mystery of ocr in large multimodal models, 2024. 2, 9
- [90] Yuliang Liu, Biao Yang, Qiang Liu, Zhang Li, Zhiyin Ma, Shuo Zhang, and Xiang Bai. Textmonkey: An ocr-free large multimodal model for understanding document. arXiv preprint arXiv:2403.04473, 2024. 2
- [91] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s, 2022. 2
- [92] Ziyu Liu, Tao Chu, Yuhang Zang, Xilin Wei, Xiaoyi Dong, Pan Zhang, Zijian Liang, Yuanjun Xiong, Yu Qiao, Dahua Lin, et al. MMDU: A multi-turn multi-image dialog understanding benchmark and instruction-tuning dataset for lvlms. arXiv preprint arXiv:2406.11833, 2024. 2, 7, 8, 9, 10
- [93] Ziyu Liu, Zeyi Sun, Yuhang Zang, Wei Li, Pan Zhang, Xiaoyi Dong, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. RAR: Retrieving and ranking augmented mllms for visual recognition. arXiv preprint arXiv:2403.13805, 2024. 2
- [94] LocalLLaMA. Dynamically scaled rope further increases performance of long context llama with zero fine-tuning,

2023. 2

- [95] Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, et al. Starcoder 2 and the stack v2: The next generation. arXiv preprint arXiv:2402.19173, 2024. 7
- [96] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In The 59th Annual Meeting of the Association for Computational Linguistics (ACL), 2021. 8
- [97] Pan Lu, Liang Qiu, Jiaqi Chen, Tony Xia, Yizhou Zhao, Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu. Iconqa: A new benchmark for abstract diagram understanding and visual language reasoning. arXiv preprint arXiv:2110.13214, 2021. 8
- [98] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, KaiWei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Advances in Neural Information Processing Systems (NeurIPS), 2022. 8
- [99] Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, SongChun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. arXiv preprint arXiv:2209.14610, 2022. 8

- [100] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR), 2024. 2, 9
- [101] Ruipu Luo, Ziwang Zhao, Min Yang, Junwei Dong, Minghui Qiu, Pengcheng Lu, Tao Wang, and Zhongyu Wei. Valley: Video assistant with large language model enhanced ability. arXiv preprint arXiv:2306.07207, 2023. 2
- [102] Tengchao Lv, Yupan Huang, Jingye Chen, Lei Cui, Shuming Ma, Yaoyao Chang, Shaohan Huang, Wenhui Wang, Li Dong, Weiyao Luo, Shaoxiang Wu, Guoxin Wang, Cha Zhang, and Furu Wei. Kosmos-2.5: A multimodal literate model, 2023. 2
- [103] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, Pan Zhang, Jiang Yu-Gang Pan, Liangming, Jiaqi Wang, Yixin Cao, and Aixin Sun. MMLongBench-Doc: Benchmarking long-context document understanding with visualizations. arXiv preprint arXiv:2407.01523, 2024. 2
- [104] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023. 2
- [105] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3195–3204, 2019. 8
- [106] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022. 2, 8, 9
- [107] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proc. of the IEEE Winter Conference on Applications of Computer Vision (WACV), 2021. 8, 9
- [108] Minesh Mathew, Viraj Bagal, Rub`en Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In Proc. of the IEEE Winter Conference on Applications of Computer Vision (WACV), 2022. 2, 8, 9
- [109] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. OCR-VQA: Visual question answering by reading text in images. In International Conference on Document Analysis and Recognition (ICDAR), 2019. 8
- [110] Munan Ning, Bin Zhu, Yujia Xie, Bin Lin, Jiaxi Cui, Lu Yuan, Dongdong Chen, and Li Yuan. Video-bench: A comprehensive benchmark and toolkit for evaluating video-based large language models. arXiv preprint arXiv:2311.16103, 2023. 2
- [111] OpenAI. Chatgpt. https://openai.com/blog/ chatgpt, 2022. 1, 2
- [112] OpenAI. Gpt-4 technical report, 2023. 1, 2, 9, 10
- [113] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,

- Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herv´e Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2024. 2
- [114] Vicente Ordonez, Girish Kulkarni, and Tamara L. Berg. Im2text: Describing images using 1 million captioned photographs. In Advances in Neural Information Processing Systems (NeurIPS), 2011. 8
- [115] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems (NeurIPS), 2022. 2, 6
- [116] Arka Pal, Deep Karkhanis, Samuel Dooley, Manley Roberts, Siddartha Naidu, and Colin White. Smaug: Fixing failure modes of preference optimisation with dpo-positive. arXiv preprint arXiv:2402.13228, 2024. 6
- [117] Panupong Pasupat and Percy Liang. Compositional semantic parsing on semi-structured tables. In The Annual Meeting of the Association for Computational Linguistics (ACL),

2015. 2, 8, 9

- [118] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv.org,

2023. 2

- [119] Renjie Pi, Tianyang Han, Wei Xiong, Jipeng Zhang, Runtao Liu, Rui Pan, and Tong Zhang. Strengthening multimodal large language model with bootstrapped preference optimization. arXiv preprint arXiv:2403.08730, 2024. 6
- [120] Rui Qian, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Shuangrui Ding, Dahua Lin, and Jiaqi Wang. Streaming long video understanding with large language models, 2024. 2
- [121] Qwen. Introducing Qwen-7B: Open foundation and human-aligned models (of the state-of-the-arts), 2023. 1, 2
- [122] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In Proceedings of the International Conference on Machine learning (ICML), 2021. 2, 6, 7
- [123] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. arxiv 2022. arXiv preprint arXiv:2212.04356, 10, 2022. 2, 7
- [124] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 2, 6, 9
- [125] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-

- 400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 8
- [126] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 6
- [127] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In Proceedings of the European Conference on Computer Vision (ECCV), 2022. 8
- [128] Sanket Shah, Anand Mishra, Naganand Yadati, and Partha Pratim Talukdar. Kvqa: Knowledge-aware visual question answering. In Proceedings of the Conference on Artificial Intelligence (AAAI), 2019. 8
- [129] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In The Annual Meeting of the Association for Computational Linguistics (ACL), 2018. 8
- [130] Baoguang Shi, Cong Yao, Minghui Liao, Mingkun Yang, Pei Xu, Linyan Cui, Serge Belongie, Shijian Lu, and Xiang Bai. Icdar2017 competition on reading chinese text in the wild (rctw-17). In International Conference on Document Analysis and Recognition (ICDAR), 2017. 8
- [131] Chenglei Si, Yanzhe Zhang, Zhengyuan Yang, Ruibo Liu, and Diyi Yang. Design2code: How far are we from automating front-end engineering? arXiv preprint arXiv:2403.03163, 2024. 2, 6, 9, 10
- [132] Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. Textcaps: a dataset for image captioning with reading comprehension. In Proceedings of the European Conference on Computer Vision (ECCV), 2020. 8
- [133] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 2, 8, 9
- [134] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Xun Guo, Tian Ye, Yan Lu, Jenq-Neng Hwang, et al. Moviechat: From dense token to sparse memory for long video understanding. arXiv preprint arXiv:2307.16449, 2023. 2
- [135] Enxin Song, Wenhao Chai, Tian Ye, Jenq-Neng Hwang, Xi Li, and Gaoang Wang. Moviechat+: Question-aware sparse memory for long video question answering. arXiv preprint arXiv:2404.17176, 2024. 2
- [136] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners, 2024. 2
- [137] Yipeng Sun, Zihan Ni, Chee-Kheng Chng, Yuliang Liu, Canjie Luo, Chun Chet Ng, Junyu Han, Errui Ding, Jingtuo Liu, Dimosthenis Karatzas, et al. Icdar 2019 competition on large-scale street view text with partial labeling-rrclsvt. In International Conference on Document Analysis and Recognition (ICDAR), 2019. 8
- [138] Zeyi Sun, Ye Fang, Tong Wu, Pan Zhang, Yuhang Zang, Shu Kong, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang.

- Alpha-CLIP: A clip model focusing on wherever you want. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2
- [139] S. Svetlichnaya. DeepForm: Understand structured documents at scale., 2020. 2, 8, 9
- [140] Ryota Tanaka, Kyosuke Nishida, and Sen Yoshida. VisualMRC: Machine reading comprehension on document images. In Proceedings of the Conference on Artificial Intelligence (AAAI), 2021. 2, 8, 9
- [141] Jingqun Tang, Chunhui Lin, Zhen Zhao, Shu Wei, Binghong Wu, Qi Liu, Hao Feng, Yang Li, Siqi Wang, Lei Liao, Wei Shi, Yuliang Liu, Hao Liu, Yuan Xie, Xiang Bai, and Can Huang. Textsquare: Scaling up text-centric visual instruction tuning, 2024. 2
- [142] Gemini Team. Gemini: A family of highly capable multimodal models, 2023. 1, 9, 10
- [143] InternLM Team. Internlm: A multilingual language model with progressively enhanced capabilities. https:// github.com/InternLM/InternLM, 2023. 1, 2, 7, 8
- [144] 360VL Team. 360vl, 2024. 9
- [145] WeMM Team. Wemm, 2024. 9
- [146] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv.org, 2023. 1, 2
- [147] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models,

- 2023. 1, 2

[148] Yuxuan Wan, Chaozheng Wang, Yi Dong, Wenxuan Wang, Shuqing Li, Yintong Huo, and Michael R Lyu. Automatically generating ui code from screenshot: A divide-andconquer-based approach. arXiv preprint arXiv:2406.16386,

- 2024. 6

- [149] Junke Wang, Lingchen Meng, Zejia Weng, Bo He, Zuxuan Wu, and Yu-Gang Jiang. To see is to believe: Prompting gpt-4v for better visual instruction tuning. arXiv preprint arXiv:2311.07574, 2023. 8
- [150] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. Cogvlm: Visual expert for pretrained language models, 2023. 2
- [151] Haoran Wei, Lingyu Kong, Jinyue Chen, Liang Zhao, Zheng Ge, Jinrong Yang, Jianjian Sun, Chunrui Han, and Xiangyu Zhang. Vary: Scaling up the vision vocabulary for large vision-language models. arXiv preprint arXiv:2312.06109, 2023. 2
- [152] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems (NeurIPS), 2022. 8

- [153] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023. 2
- [154] Yuetian Weng, Mingfei Han, Haoyu He, Xiaojun Chang, and Bohan Zhuang. Longvlm: Efficient long video understanding via large language models. arXiv preprint arXiv:2404.03384, 2024. 2
- [155] XAI. Grok-1.5 vision preview. 2024. 2, 9
- [156] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava : Parameter-free llava extension from images to videos for video dense captioning, 2024. 6
- [157] Ruyi Xu, Yuan Yao, Zonghao Guo, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, Maosong Sun, and Gao Huang. Llava-uhd: an lmm perceiving any aspect ratio and high-resolution images. arXiv preprint arXiv:2403.11703, 2024. 2
- [158] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, et al. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model. arXiv preprint arXiv:2310.05126, 2023. 2
- [159] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv.org, 2023. 2
- [160] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Association for Computational Linguistics (TACL), 2014. 8
- [161] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284, 2023. 8
- [162] Shoubin Yu, Jaemin Cho, Prateek Yadav, and Mohit Bansal. Self-chained image-language model for video localization and question answering. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 2
- [163] Tianyu Yu, Yuan Yao, Haoye Zhang, Taiwen He, Yifeng Han, Ganqu Cui, Jinyi Hu, Zhiyuan Liu, Hai-Tao Zheng, Maosong Sun, et al. RLHF-V: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

2024. 6

- [164] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 2, 9
- [165] Tai-Ling Yuan, Zhe Zhu, Kun Xu, Cheng-Jun Li, Tai-Jiang Mu, and Shi-Min Hu. A large chinese text dataset in the wild. Journal of Computer Science and Technology, 34(3): 509–521, 2019. 8

- [166] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502, 2023. 2, 9
- [167] Yuhang Zang, Wei Li, Jun Han, Kaiyang Zhou, and Chen Change Loy. Contextual object detection with multimodal large language models. arXiv preprint arXiv:2305.18279, 2023. 2
- [168] Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, Weng Lam Tam, Zixuan Ma, Yufei Xue, Jidong Zhai, Wenguang Chen, Zhiyuan Liu, Peng Zhang, Yuxiao Dong, and Jie Tang. GLM-130b: An open bilingual pretrained model. In Proceedings of the International Conference on Learning Representations (ICLR), 2023. 2
- [169] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training,

2023. 2

- [170] Beichen Zhang, Pan Zhang, Xiaoyi Dong, Yuhang Zang, and Jiaqi Wang. Long-CLIP: Unlocking the long-text capability of clip. arXiv preprint arXiv:2403.15378, 2024. 2
- [171] Ce Zhang, Taixi Lu, Md Mohaiminul Islam, Ziyang Wang, Shoubin Yu, Mohit Bansal, and Gedas Bertasius. A simple llm framework for long-range video question-answering. arXiv preprint arXiv:2312.17235, 2023. 2
- [172] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 2
- [173] Pan Zhang, Xiaoyi Dong Bin Wang, Yuhang Cao, Chao Xu, Linke Ouyang, Zhiyuan Zhao, Shuangrui Ding, Songyang Zhang, Haodong Duan, Hang Yan, et al. Internlmxcomposer: A vision-language large model for advanced text-image comprehension and composition. arXiv preprint arXiv:2309.15112, 2023. 1, 2, 8
- [174] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024. 6
- [175] Rui Zhang, Yongsheng Zhou, Qianyi Jiang, Qi Song, Nan Li, Kai Zhou, Lei Wang, Dong Wang, Minghui Liao, Mingkun Yang, et al. Icdar 2019 robust reading challenge on reading chinese text on signboard. In International Conference on Document Analysis and Recognition (ICDAR),

2019. 8

- [176] Tao Zhang, Xiangtai Li, Hao Fei, Haobo Yuan, Shengqiong Wu, Shunping Ji, Chen Change Loy, and Shuicheng Yan. Omg-llava: Bridging image-level, object-level, pixel-level reasoning and understanding, 2024. 2
- [177] Yi-Fan Zhang, Qingsong Wen, Chaoyou Fu, Xue Wang, Zhang Zhang, Liang Wang, and Rong Jin. Beyond llavahd: Diving into high-resolution large multimodal models,

2024. 2

- [178] Haozhe Zhao, Zefan Cai, Shuzheng Si, Xiaojian Ma, Kaikai An, Liang Chen, Zixuan Liu, Sheng Wang, Wenjuan Han, and Baobao Chang. Mmicl: Empowering vision-language model with multi-modal in-context learning. arXiv.org,

2023. 2

- [179] Wenliang Zhao, Xumin Yu, and Zengyi Qin. Melotts: Highquality multi-lingual multi-accent text-to-speech, 2023. 2, 7
- [180] Zhiyuan Zhao, Bin Wang, Linke Ouyang, Xiaoyi Dong, Jiaqi Wang, and Conghui He. Beyond hallucinations: Enhancing lvlms through hallucination-aware direct preference optimization. arXiv preprint arXiv:2311.16839, 2023. 6
- [181] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. MLVU: A comprehensive benchmark for multi-task long video understanding. arXiv preprint arXiv:2406.04264, 2024. 2, 9
- [182] Yiyang Zhou, Chenhang Cui, Rafael Rafailov, Chelsea Finn, and Huaxiu Yao. Aligning modalities in vision large language models via preference fine-tuning. arXiv preprint arXiv:2402.11411, 2024. 6
- [183] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing visionlanguage understanding with advanced large language models. arXiv.org, 2023. 1, 2

