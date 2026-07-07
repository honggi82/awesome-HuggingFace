# arXiv:2407.08739v2[cs.CV]1Nov2024

MAVIS: MATHEMATICAL VISUAL INSTRUCTION TUNING WITH AN AUTOMATIC DATA ENGINE

Renrui Zhang∗1, Xinyu Wei∗2, Dongzhi Jiang1, Ziyu Guo1, Shicheng Li2 Yichi Zhang2, Chengzhuo Tong3, Jiaming Liu2, Aojun Zhou1, Bin Wei5 Shanghang Zhang2, Peng Gao3, Chunyuan Li4, Hongsheng Li1

1CUHK 2Peking University 3Shanghai AI Laboratory 4ByteDance 5Oracle {renruizhang, dzjiang, ziyuguo}@link.cuhk.edu.hk

ABSTRACT

Multi-modal Large Language Models (MLLMs) have recently showcased superior proficiency in general visual scenarios. However, we identify their mathematical capabilities remain under-explored with three areas to be improved: visual encoding of math diagrams, diagram-language alignment, and chain-of-thought (CoT) reasoning. This draws forth an urgent demand for an effective training paradigm and a large-scale, comprehensive dataset with detailed CoT rationales, which is challenging to collect and costly to annotate manually. To tackle this issue, we propose MAVIS, a MAthematical VISual instruction tuning pipeline for MLLMs, featuring an automatic data engine to efficiently create mathematical visual datasets. We design the data generation process to be entirely independent of human intervention or GPT API usage, while ensuring the diagram-caption correspondence, question-answer correctness, and CoT reasoning quality. With this approach, we curate two datasets, MAVIS-Caption (558K diagram-caption pairs) and MAVIS-Instruct (834K visual math problems with CoT rationales), and propose four progressive stages for training MLLMs from scratch. First, we utilize MAVIS-Caption to fine-tune a math-specific vision encoder (CLIP-Math) through contrastive learning, tailored for improved diagram visual encoding. Second, we also leverage MAVIS-Caption to align the CLIP-Math with a large language model (LLM) by a projection layer, enhancing vision-language alignment in mathematical domains. Third, we adopt MAVIS-Instruct to perform the instruction tuning for robust problem-solving skills, and term the resulting model as MAVIS-7B. Fourth, we apply Direct Preference Optimization (DPO) to enhance the CoT capabilities of our model, further refining its step-wise reasoning performance. On various mathematical benchmarks, our MAVIS-7B achieves leading results among open-source MLLMs, e.g., surpassing other 7B models by +9.3% and the second-best LLaVANeXT (110B) by +6.9%, demonstrating the effectiveness of our method. Code and data will be released https://github.com/ZrrSkywalker/MAVIS.

1 INTRODUCTION

The pursuit of artificial general intelligence necessitates models to seamlessly interpret and generate multi-modal data. In recent years, the advent of Large-language Models (LLMs) (Brown et al., 2020; Touvron et al., 2023a;b; Chiang et al., 2023) and their Multi-modal extension (MLLMs) (Zhang et al., 2024a; Gao et al., 2023b; Su et al., 2023; Ye et al., 2023a) have significantly facilitated this process across various fields, such as healthcare (Singhal et al., 2023; Shu et al., 2023), autonomous driving (Yang et al., 2023; Jin et al., 2024), and robotics (Li et al., 2023b; Liu et al., 2024b). Although MLLMs exhibit remarkable performance in diverse tasks and benchmarks, one arena where they have yet to fully demonstrate their potential is mathematical problem-solving in visual contexts.

∗Equal Contribution

Math Diagram CLIP ViT-L Our CLIP-Math

|[Figure 1]<br><br>The image shows a geometric figure with points labeled A, B, C, D, and E. Point A is at the top of a vertical line segment AB, which measures 10 units. From B, a 30° angle is formed with the line segment extending to D. There is a quarter circle centered at B with a radius of 10 units, reaching point C. The figure includes an inscribed circle centered within the quadrilateral formed by points C, D, and E. The lines AB and DE are parallel ...<br><br>A sector ABC is drawn, with angle CBA measuring 30 degrees. A square CBDE is linked to edge CB, with angle BCE measuring 90 degrees. The distance between points A and B is 10.0 units, and the distance between points C and<br>B is also 10.0 units. The square CBDE has an inscribed circle.<br><br><br>[Figure 2]<br><br>[Figure 3]<br><br>GPT-4V:<br><br>MAVIS-7B:|
|---|

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

(a) Attention Comparison between CLIP and our CLIP-Math

(b) Captioning Comparison between GPT-4V and our MAVIS-7B

##### Question：

#### GPT-4V:

#### Gemini-Pro:

#### MAVIS-7B:

Find the domain and range of the function f using interval notation, which goes through (-2,-4), (-1,-2), (0,0), (1,-4).

The domain of the function is the

From the graph, The function starts at x=−4 and goes up to x=1.

✓

×

From the graph, we know that the function starts from x = -3 and ends at x=1.

- set of all x-values for which the function is defined.

The function is defined for all xvalues between -3 and 1, including 3 but excluding 1.

Thus, We can write this in interval

- notation as: [-3, 1)

The range of the function is the set of all y-values that the function can take on.

The function takes on all y-values between -4 and 0, including -4 and 0.

We can write this in interval

- notation as: [-4, 0]

✓ ✓ ✓ ✓ ✓ ✓

There is a hollow point at x=−3, which means the function is not defined there.

✓

Besides, there is a hallow point at x = -3.

× ×

Answer： domain(-3,1] range[-4,0]

×

Thus, the domain is [-4, -3)∪(-3,1]

Thus, We can determine that the domain is (-3, 1]

[Figure 10]

The range is the set of all possible output values (y-values).

✓ ✓

From the graph, we know that the function reaches maximum 0 at x = 0, and reaches minimus -4 at x = -2 and x = 1.

✓

From the graph: The lowest y-value is −4 (at x=−2 and x=1).

The highest y-value is 0, but since the point at (−3,0) is hollow, the function does not reach 0.

✓

The function attains all values from -4 to 0.

× ×

Thus, the range is [-4, 0]

✓

Thus, the range is [-4, 0)

(c) CoT Reasoning Comparison between Different MLLMs

- Figure 1: (a) We compare the attention map of class tokens from CLIP ViT-L (Radford et al., 2021) and our CLIP-Math. Our vision encoder can better capture significant mathematical information within diagrams. (b) We compare the diagram captioning capabilities between GPT-4V (OpenAI, 2023c) and our MAVIS-7B, where GPT-4V fall short of accurately recognizing mathematical elements. (c) We compare the chain-of-thought (CoT) reasoning between different models, showcasing that GPT-4V and Gemini-Pro (Gemini Team, 2023) suffer from low-quality reasoning process.

Existing efforts (OpenAI, 2023b;a; Zhou et al., 2023) for text-only mathematics have attained considerable progress, largely attributed to the availability of sufficient and easily accessible training data. In contrast, solving visual mathematical problems remains a significant challenge for MLLMs, primarily due to the absence of a fully validated, effective training pipeline and the acute shortage of large-scale, high-quality datasets. Visual mathematical data is not only more costly to collect from publicly available sources compared to text-only data, but also requires expensive manual annotation to produce accurate step-by-step chain-of-thought (CoT) rationales integrating diagram information. In light of these challenges, we identify three critical issues that impede the visual mathematical capabilities of MLLMs.

- i. Unsatisfactory math diagram embeddings by vision encoders. Most MLLMs adopt a frozen CLIP (Radford et al., 2021) as the vision encoder, which is pre-trained by natural images capturing real-world scenes with rich colors and textures. In contrast, math diagrams are composed of abstract curves, shapes, and symbols with a monochromatic color scheme, exhibiting large semantic gaps to general scenarios. As visualized in Figure 1 (a), the attention map of CLIP struggles to capture important information within math diagrams, which cannot provide satisfactory visual embeddings for LLMs to understand.
- ii. Diagram-language misalignment between vision encoders and LLMs. Likewise, the vision-language pre-training stage of MLLMs also adopts natural image-caption pairs for cross-modal alignment. Due to the domain gap, while they can generate accurate captions for

- real-world images, but fall short of recognizing basic mathematical elements and narrating their relations. As compared in Figure 1 (b), even GPT-4V (OpenAI, 2023c) produces low-quality descriptions for simple geometric figures and functions, indicating LLMs are not well aligned with the visual embedding space of math diagrams.
- iii. Inaccurate CoT reasoning capabilities with visual elements by MLLMs. Referring to the CoT evaluation in MathVerse (Zhang et al., 2024b), incorporating the diagram input would adversely affect the reasoning quality of MLLMs compared to using only the textonly question. As visualized in Figure 1 (c), we observe the problem-solving process of GPT-4V and Gemini-Pro (Gemini Team, 2023) both suffer from low-quality CoT reasoning accuracy. This demonstrates the incapability of MLLMs to leverage visual cues for precise step-by-step mathematical problem-solving.

Therefore, to mitigate these issues, it is essential to develop an extensive dataset and effective training approach tailored to visual mathematics. In this paper, we propose MAVIS, a MAthematical VISual instruction tuning paradigm and an automatic data generation engine for MLLMs, which aims to fully unleash their potential for diagram visual encoding and reasoning capabilities. We introduce two meticulously curated datasets, a progressive four-stage training pipeline, and a visual mathematical specialist, MAVIS-7B. We summarize the contributions of our work as follows.

- • Automatic Mathematical Visual Data Engine. To eliminate the need for labor-intensive annotation and expensive GPT API (OpenAI, 2023c;b) usage, we designed our data engine to be entirely rule-based and fully automated. This engine handles every aspect of mathematical data creation, including diagram drawing, caption generation, question-answer synthesis, and CoT rationale production. With this approach, we curate two large-scale, high-quality mathematical visual datasets, MAVIS-Caption and MAVIS-Instruct, widely covering plane geometry, analytic geometry, and function. MAVIS-Caption consists of 558K diagram-caption pairs automatically created by our data engine with accurate visionlanguage correspondence. MAVIS-Instruct includes 834K visual math problems, which includes 582K data constructed by our data engine and additional 252K data augmented by GPT-4V from manual collection and existing datasets (Chen et al., 2021b; Lu et al., 2021). Each problem is annotated with a CoT rationale, and modified to contain minimized textual redundancy that enforces MLLMs to pay more attention on visual diagrams.
- • Four-stage Training Pipeline. Our training framework involves four progressive stages designed to sequentially address the aforementioned identified deficiencies in MLLMs. Firstly, we utilize MAVIS-Caption to fine-tune a math-specific vision encoder by contrastive learning, termed CLIP-Math, to enable better visual representations of math diagrams. Subsequently, we align this encoder with the LLM to ensure effective diagram-language integration also by MAVIS-Caption. After that, our MAVIS-Instruct is adopted to instructiontune the MLLM, which provides sufficient step-wise problem-solving supervision. Finally, we employ Direct Preference Optimization (DPO) (Rafailov et al., 2024) with annotated CoT rationales in MAVIS-Instruct to further enhance the reasoning capabilities of our model.
- • Mathematical Visual Specialist. After the four-stage training, we develop MAVIS-7B, an MLLM specifically optimized for visual mathematical problem-solving. On various evaluation benchmarks, our model achieves leading performance compared to existing open-source MLLMs, e.g., surpassing other 7B models by +9.3% and the second-best LLaVA-NeXT (110B) (Li et al., 2024a) by +6.9% on MathVerse (Zhang et al., 2024b). The quantitative results and qualitative analysis both validate the significance of our approach.

2 AUTOMATIC DATA ENGINE

To cope with the substantial data requirements of MLLMs, it is essential to have access to extensive training instances. However, for visual mathematics, the paucity of publicly available datasets poses a challenge, and creating such data manually is also not feasible due to the high cost involved. Therefore, we develop an automatic data engine to efficiently generate high-quality math diagrams (Section 2.1), detailed captions (Section 2.2), and question-answer pairs with CoT rationales (Section 2.3).

## Table 1: Statistics of MAVIS-Caption.

## Table 2: Subject Distribution of MAVIS-Instruct.

Statistic Number Total Captions

- - Total number 588K
- - Average length (words) 62.85
- - Average length (characters) 339.68
- - Vocabulary size 418 Plane Geometry

- - Total number 299K (50.9%)
- - Average length (words) 69.77
- - Average length (characters) 385.85
- - Vocabulary size 195 Analytic Geometry

- - Total number 77K (13.1%)
- - Average length (words) 39.64
- - Average length (characters) 210.10
- - Vocabulary size 158 Function

- - Total number 212K (36.0%)
- - Average length (words) 61.48
- - Average length (characters) 321.46
- - Vocabulary size 149

Statistic Number Total questions 834K

- - Multiple-choice questions 615K (62.4%)
- - Free-form questions 218K (37.6%) Data Engine Generated Problems 582K

- - Geometry questions 466K (80.0%)
- - Function questions 116K (20.0%) Data Engine Captions Annotated by GPT-4 51K

- - Geometry questions 30K (58.8%)
- - Function questions 21K (41.2%) Manual Collection Augmented by GPT-4 83K

- - Geometry questions 72K (86.5%)
- - Function questions 11K (13.5%) Existing Datasets Augmented by GPT-4 118K

- - Geometry questions 118K (100.0%)
- - Function questions 0 (0%)

Number of unique images 611K (73.3%) Number of unique questions 804K (96.5%) Number of unique answers 675K (81.0%)

Average question length 44.60 Average answer length 62.82

- 2.1 DIAGRAM GENERATION

Covering most mathematical scenarios, we adopt three diagram types: plane geometry, analytic geometry, and function. Note that all the logic of the data engine is implemented in Python, and we employ Matplotlib for the graphical rendering of the diagrams.

Plane Geometry Diagram. As such diagrams typically consist of spatial combinations of various basic shapes, we utilize principles from multi-hop data curation to develop customized generation rules. These rules allow for the iterative integration of new shapes into existing configurations. Initially, we establish a core set of shapes, including squares, rectangles, triangles, sectors, etc, for diagram generation. Starting with a randomly selected shape, we extend another shape from the set along one of its straight sides. By iterating this process, we can construct diverse plane geometry diagrams featuring different combinations of shapes. Additionally, we randomly label the vertices with letters (e.g., A, B, C) and annotate numerical values relevant to geometric properties (e.g., side lengths and angles), simulating realistic plane geometry problems.

Analytic Geometry Diagram. Likewise, our approach begins by defining a basic figure set that differs slightly from that used in plane geometry; for example, we include additional elements such as points and line segments. We then construct a Cartesian coordinate system, complete with grid lines and scaled axes. The range of the coordinate system is randomly determined within a predefined scope. Subsequently, we select a number from 1 to 3 to indicate the number of figures to be drawn on the graph, and randomly choose coordinates for the top-left vertices to plot these figures at varied sizes (using these points as centers for circles). Unlike plane geometry, we ensure that the figures do not overlap, except for points and segments, and maintain the figure areas within a suitable scale.

Function Diagram. We focus on seven fundamental function types: polynomial, sine, cosine, tangent, logarithmic, absolute value, and piece-wise polynomial functions. For each function type, we parameterize the equations with random variables, such as coefficients and constants within a predefined range (e.g., a and b in y = ax + b), which facilitates the generation of diverse function graphs. We also adopt the same Cartesian coordinate system employed for analytic geometry. Additionally, for specific caption or question-answering samples, we also plot key features like extreme points and zero points of the functions, providing additional visual information that aids in the understanding and reasoning of these mathematical functions.

[Figure 11]

- Figure 2: MAVIS-Caption Dataset. We showcase three diagram-caption pairs of plane geometry, function, and analytic geometry in MAVIS-Caption, generated by our developed data engine.

- 2.2 MAVIS-CAPTION

With our mathematical visual data engine, we first curate a diagram-caption dataset, MAVIS-Caption, as shown in Figure 2, aiming to benefit the diagram visual representations and cross-modal alignment.

Data Overview. As presented in Table 1, the MAVIS-Caption dataset comprises 588K diagramcaption pairs. This includes 299K for plane geometry, 77K for analytic geometry, and 212K for function. The average word length of the captions is 61.48 words, reflecting their detailed descriptive nature. The overall vocabulary size is 149, indicating the diversity in language expression. We adopt different strategies to generate captions for three types of diagrams. It is important to note that GPT-4 (OpenAI, 2023b) is only utilized during the template creation stage; it is not used at any point during the automatic caption generation process.

Plane Geometry Caption. We follow the iterative geometric generation process to develop regulations for an accurate and detailed caption. We first prompt GPT-4 to create three sets of language templates: the descriptive content for fundamental shapes (e.g., “A Triangle {} with two congruent sides {} and {}”), the phrases to denote specific attributes (e.g., “Angle {} measures {} degrees”), and the conjunction to link two adjacent shapes (e.g., “Attached to edge {} of shape {}, there is a {}”). Then, based on various generation scenarios, we fill and merge these templates to acquire a coherent description of the geometric figure.

Function Caption. As function diagrams typically showcase a single curve, we directly utilize GPT4 to generate templates describing various properties of functions, including expressions, domains, ranges, extreme points, and zero points. Each template is then filled based on specific cases, such as

“The expression of the function is y = −3x3 − 2x2 − 2x − 2. Within the range of x values [−3.0,4.0], zero points occur at −0.83 ...”.

Analytic Geometry Caption. We also employ GPT-4 to obtain two sets of language templates: the description of coordinates and attribute information for basic figures (e.g., “The square with its base left corner at {} features sides of {} in length”) and the spatial relation for nearby figures (e.g., “On the bottom right of {}, there is a {}”). The captions are then formulated by filling in the coordinates and selecting appropriate spatial relationship templates through coordinate comparison.

- 2.3 MAVIS-INSTRUCT

Besides the diagram-caption data, we curate MAVIS-Instruct of extensive problem-solving data, which endows MLLMs with visual mathematical reasoning capabilities and serve as the basis for Direct Preference Optimization (DPO) (Rafailov et al., 2024), as shown in Figure 3.

[Figure 12]

- Figure 3: MAVIS-Instruct Dataset. We showcase the generated visual math problems from four sources within MAVIS-Instruct, which contain detailed rationales and minimized textual redundancy.

Data Overview. As illustrated in Table 2, the MAVIS-Instruct dataset consists of a total of 834K visual math problems. Given that the proportion of analytic geometry problems is relatively small, we classify them with function problems for simplicity. Each problem in MAVIS-Instruct includes a CoT rationale providing step-by-step solutions, with an average answer length of 150 words. We have minimized textual redundancy in the questions, eliminating unnecessary contextual information, distracting conditions, and attributes readily observable from the diagrams. This reduction in text forces MLLMs to enhance their capability to extract essential content from visual inputs. MAVISInstruct is assembled from four distinct sources to ensure broad coverage.

Data Engine Generated Problems. Within our data engine, we manually craft rigorous regulations to produce visual math problems with accurate CoT annotations. Similar to caption generation, GPT API is not involved in the automatic synthesis process of questions, answers, and CoT rationales.

- • Plane Geometry Problems. We initially prompt GPT-4 to compile a comprehensive set of mathematical formulas applicable to each basic shape (e.g., Pythagorean theorem for right triangles and area formula for circles). Then, for a geometric diagram, we randomly select a known condition within a shape as the final solution target, and systematically deduce backward to another condition, either within the same shape or an adjacent one, using a randomly selected mathematical formula. This deduced condition is then set as unknown, and we continue iterative backward deductions as necessary. The final condition, along with any conditions in the last step, are presented as initial attributes in the question. The rationales can be simply obtained by reversing this backward deduction process.
- • Function Problems. As the properties of functions are predetermined, we utilize GPT-4 to generate diverse reasoning templates. These templates facilitate the solving of one function property based on other provided properties, thereby ensuring the generation of high-quality function rationales. The related function properties include analytical expression, function values, zeros, extremum points, monotonicity, derivatives, and integrals. To accurately reason these properties, the CoT annotation incorporates understanding of function types, solving the analytical expressions of equations, and interpreting function graphs.

Data Engine Captions Annotated by GPT-4. Given the detailed captions and diagrams generated by our data engine, we can prompt GPT-4V with these sufficient conditions to synthesis questionanswering data and ensure its correctness. We first generate a new set of 17K diagram-caption pairs that do not overlap with the previous MAVIS-Caption, which avoids answer leakage within the

Stage-1 Stage-2 Stage-3 Stage-4

Aligning Diagram-Language

Preference Alignment with DPO

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Instruction Tuning

Training CLIP-MATH

- Figure 4: Four-stage Training Pipeline of MAVIS. With our curated MAVIS-Caption and MAVISInstruct, we adopt four progressive stages for training a mathematical visual specialist from scratch.

detailed caption. Then, we prompt GPT-4V to generate 3 new problems with rationales, obtaining 51K data in total from the diagram-caption pairs.

Manual Collection Augmented by GPT-4. To incorporate high-quality problems found in realworld contexts, we manually collect 4K math problems with diagrams from publicly available resources. Recognizing that these sources often lack detailed rationales and may contain redundant text, we initially utilize GPT-4V to annotate a detailed solving process and streamline the question text to reduce redundancy. Subsequently, for each collected instance, we input the question, rationale, and diagram into GPT-4 and employ customized few-shot prompts to generate 20 new problems per original, comprising 15 multiple-choice questions and 5 free-form questions. This process contributes a total of 83K problems to the dataset.

Existing Datasets Augmented by GPT-4. Given existing well-organized geometric datasets, we can also leverage them to expand MAVIS-Instruct. Referring to previous prompt designs, we augment the 8K training set from two dataset, Geometry-3K (Lu et al., 2021) and GeoQA+ (Chen et al., 2021a), into 80K visual problems with accompanying rationales, mapping each original problem to 10 new ones. Due to the scarcity of publicly available function data, we do not include function problems from this source.

- 3 MATHEMATICAL VISUAL TRAINING

With the curated datasets, we devise a four-stage training pipeline for endowing MLLMs with mathematical visual capabilities as shown in Figure 4. They respectively aim to mitigate the three deficiencies within existing MLLMs, i.e., diagram visual encoding, diagram-language alignment, and mathematical reasoning skills in visual contexts.

- 3.1 STAGE 1: TRAINING CLIP-MATH

To enhance CLIP’s (Radford et al., 2021) inadequate visual encoding of math diagrams, we utilize MAVIS-Caption to train a specialized CLIP-Math encoder. Specifically, we fine-tune a pre-trained CLIP-Base model following the conservative learning scheme. The math diagrams are fed into the learnable vision encoder, while the corresponding captions are processed by the text encoder, which remains frozen to provide reliable supervision. Via contrastive training, the model learns to adapt from its original natural image domain to mathematical contexts, increasing its focus on essential visual elements within diagrams, as demonstrated in Figure 1 (a). The optimized CLIP-Math encoder now delivers more precise and robust representations of math diagrams, establishing a solid foundation for the subsequent visual interpretation of LLMs.

- 3.2 STAGE 2: ALIGNING DIAGRAM-LANGUAGE

After acquiring the CLIP-Math encoder, we further integrate it with LLMs using MAVIS-Caption to boost cross-modal alignment between math diagrams and language embedding space. Using a simple two-layer MLP as the projection layer, we transform the visual encodings from CLIP-Math, and prepend them as a prefix to the LLM input. This process, guided by the diagram captioning task, enables the LLM to accurately recognize mathematical components and spatial arrangements. With the diagram-language alignment, LLMs are equipped with the interpretation capability in math diagrams, serving as an initial step toward deeper mathematical reasoning. In this stage, we freeze the CLIP-Math, and train the projection layer along with the LoRA-based (Hu et al., 2021) LLM.

- Table 3: Evaluation on MathVerse’s testmini Set with Six Problem Versions. ‘CoT-E’ and ‘Acc’ denote the scores of CoT evaluation strategy and the scores of direct ‘true or false’ accuracy, respectively. ‘∗’ denotes previous mathematical visual specialists. The highest scores for closed-source and open-source MLLMs are marked in red and blue respectively.

Text Dominant

Text Lite

Vision Intensive

Vision Dominant

Vision Only

All

Model LLMSize

CoT-E Acc CoT-E Acc CoT-E Acc CoT-E Acc CoT-E Acc CoT-E Acc Baselines

Random Chance - - 12.4 - 12.4 - 12.4 - 12.4 - 12.4 - 12.4 Human - - 64.9 - 71.2 - 70.9 - 61.4 - 68.3 - 66.7

LLMs

ChatGPT - - - 51.3 33.3 38.5 18.9 - - - - - GPT-4 - - - 63.4 46.5 40.7 20.7 - - - - - -

Closed-source MLLMs

Qwen-VL-Plus - 21.3 11.8 26.0 15.7 21.2 11.1 18.5 9.0 19.1 13.0 21.8 10.0 Gemini-Pro - 35.3 23.5 39.8 26.3 34.7 23.5 32.0 23.0 36.8 22.3 33.3 22.2 Qwen-VL-Max - 37.2 25.3 42.8 30.7 37.7 26.1 33.6 24.1 35.9 24.1 35.9 21.4 GPT-4V - 54.4 39.4 63.1 54.7 56.6 41.4 51.4 34.9 50.8 34.4 50.3 31.6

Open-source MLLMs LLaMA-Adapter-V2 7B 5.8 5.7 7.8 6.2 6.3 5.9 6.2 6.1 4.5 4.2 4.4 6.1 ImageBind-LLM 7B 10.0 9.2 13.2 11.4 11.6 11.3 9.8 8.9 11.8 11.2 3.5 3.4 mPLUG-Owl2 7B 10.3 5.9 11.6 6.6 11.4 6.3 11.1 6.3 9.4 5.6 8.0 4.9 LLaVA-1.5 7B 12.7 7.6 17.1 8.8 12.0 7.6 12.6 7.4 12.7 7.4 9.0 6.9 SPHINX-Plus 13B 14.0 12.2 16.3 13.9 12.8 11.6 12.9 11.6 14.7 13.5 13.2 10.4 G-LLaVA∗ 7B 15.7 16.6 22.2 20.9 20.4 20.7 16.5 17.2 12.7 14.6 6.6 9.4 LLaVA-NeXT 8B 17.2 15.6 21.6 19.4 19.7 15.2 17.6 16.8 14.9 15.2 12.1 11.3 ShareGPT4V 13B 17.4 13.1 21.8 16.2 20.6 16.2 18.6 15.5 16.2 13.8 9.7 3.7 SPHINX-MoE 8×7B 22.8 15.0 33.3 22.2 21.9 16.4 21.1 14.8 19.6 12.6 18.3 9.1 Math-LLaVA∗ 13B 24.1 19.0 34.2 21.2 22.7 19.8 21.1 20.2 20.3 17.6 22.2 16.4 InternLM-XC2. 7B 25.9 16.5 36.9 22.3 28.3 17.0 20.1 15.7 24.4 16.4 19.8 11.0 LLaVA-NeXT 110B 28.3 24.5 37.1 31.7 29.1 24.1 22.6 21.0 21.8 22.1 30.9 20.7 MAVIS-7B w/o DPO 7B 33.7 27.5 42.5 41.4 36.3 29.1 33.3 27.4 29.3 24.9 27.1 14.6 MAVIS-7B 7B 35.2 28.4 43.2 41.6 37.2 29.5 34.1 27.9 29.7 24.7 31.8 18.3

- 3.3 STAGE 3: INSTRUCTION TUNING

On top of that, we leverage MAVIS-Instruct to endow MLLMs with CoT reasoning and problemsolving capabilities in visual mathematics. The detailed rationales within each problem’s solution provide high-quality reasoning guidance for MLLMs, significantly enhancing their step-by-step CoT process. Furthermore, as we have minimized the redundancy within question texts during the construction process, such text-lite problem formats, referring to MathVerse (Zhang et al., 2024b), facilitate MLLMs to capture more essential information from the visual embeddings for problemsolving, rather than relying on shortcuts to only process the textual content. In this stage, we unfreeze both the projection layer and apply LoRA (Hu et al., 2021) for the LLM for a thorough tuning.

- 3.4 STAGE 4: PREFERENCE ALIGNMENT WITH DPO

After the instruction tuning phase, the resulting model gains the capability for CoT reasoning on visual math problems. However, it may still produce inaccurate intermediate steps due to insufficient supervision for generating the best reasoning path. To address this, we further apply CoT preference alignment using the DPO (Rafailov et al., 2024) algorithm to further enhance the model’s reasoning performance. Specifically, we adopt the instruction-tuned model to first infer CoT reasoning process on the 582K problems generated by data engine within MAVIS-Instruct. Then, we filter out the incorrect outputs (88K data) based on the final answer as the negative reasoning samples in DPO, and directly utilize the annotated CoT process as the positive samples. We only unfreeze the LoRA parameters for DPO training, and finally obtain our mathematical specialist, MAVIS-7B.

- 4 EXPERIMENT

We first detail our experimental settings in Section 4.1, and then discuss the quantitative on different benchmarks and qualitative examples in Sections 4.2 and 4.3, respectively. Please refer to the Appendix for more data details and ablation studies.

- Table 4: Evaluation on Six Mathematical Benchmarks. ‘MMMU-Math’ denotes the math problems within the test set of MMMU. ‘GPS’, ‘ALG’, and ‘GEO’ denote geometry problem solving, algebraic, and geometry in MathVista’s testmini set. ‘S1’, ‘S2’, and ‘S3’ denote different problem steps in We-Math’s testmini set. ‘∗’ denotes previous mathematical visual specialists. The highest scores for

closed-source and open-source MLLMs are marked in red and blue respectively.

LLM Size

MathVista We-Math

Model

GeoQA FunctionQA MMMU-Math MathVision

GPS ALG GEO S1 S2 S3

Baselines

Random Chance - 17.1 - 21.6 7.2 24.1 25.8 22.7 - - Human - 92.3 - 84.2 68.8 48.4 50.9 51.4 - - -

LLMs

ChatGPT - - - - 9.7 31.7 32.4 33.0 - - GPT-4 - - - 30.6 13.1 31.7 33.5 32.2 - - -

Closed-source MLLMs Qwen-VL-Plus - - - - 10.7 38.5 39.1 39.3 - - Qwen-VL-Max - - - 36.3 15.6 - - - 40.8 30.3 20.6 GPT-4V - - - 48.4 22.8 50.5 53.0 51.0 65.5 49.2 38.2 Open-source MLLMs

LLaMA-Adapter V2 7B - 30.6 23.0 - 25.5 26.3 24.3 - - mPLUG-Owl2 7B - - 18.8 - - - - - - UniMath - 50.0 - - - - - - - - LLaVA-1.5 13B 20.3 21.0 24.0 11.1 - - - - - ShareGPT4V 13B - - - 11.9 - - - - - SPHINX-MoE 8×7B - 33.9 - 14.2 31.2 31.7 30.5 - - G-LLaVA∗ 13B 67.0 - - - 56.7 - - 32.4 30.1 32.7 Math-LLaVA∗ 13B - - - - 57.7 53.0 56.5 - - InternLM-XC2. 7B - - 30.1 14.5 63.0 56.6 62.3 47.0 33.1 33.0 LLaVA-NeXT 110B - - - - - - - 53.7 36.9 31.5

MAVIS-7B w/o DPO 7B 66.7 40.3 39.2 18.6 63.2 58.3 63.0 56.9 37.1 33.2 MAVIS-7B 7B 68.3 50.0 42.4 19.2 64.1 59.2 63.2 57.2 37.9 34.6

- 4.1 EXPERIMENTAL SETTINGS

Implementation Details. We adopt a CLIP ViT-L (Radford et al., 2021) as the pre-trained model to fine-tune our CLIP-Math, and utilize Mammoth2-7B (Yue et al., 2024) as the base LLM to construct MAVIS-7B. In the first stage, we fine-tune the CLIP for 10 epochs with a batch size 16 and an initial learning rate 2e−6. In the second stage, we train the diagram-language alignment for 1 epoch with a batch size 32 and an initial learning rate 2e−6, and adopt LoRA (Hu et al., 2021) with a rank 128. In the third and fourth stages, we adopt the same training settings as the second one.

Evaluation Schemes. We evaluate our model MAVIS-7B on several popular mathematical benchmarks, MathVerse (Zhang et al., 2024b), GeoQA (Chen et al., 2021b), FunctionQA (function problems in MathVista (Lu et al., 2023)), MMMU-Math (the math problems in MMMU (Yue et al., 2023a)), MathVision (Wang et al., 2024a), three mathematical categories in MathVista, and We-Math (Qiao et al., 2024). We compare a variety of existing MLLMs, including two mathematical visual specialist (Gao et al., 2023a; Shi et al., 2024), two LLMs (OpenAI, 2023a;b), and other general MLLMs (Bai et al., 2023b; Gao et al., 2023b; Ye et al., 2023b; Liu et al., 2023a; Chen et al., 2023b; Gao et al., 2024; Dong et al., 2024; Liu et al., 2024a; Chen et al., 2023a; Gao et al., 2024).

- 4.2 QUANTITATIVE PERFORMANCE

As shown in Table 3 for the MathVerse benchmark, MAVIS-7B achieves the best overall scores in both CoT evaluation and accuracy among open-source MLLMs with only a 7B model size, and consistently surpasses the second-best method on different problem versions. Specifically, our model surpasses the powerful InternLM-XComposer2 (7B) (Dong et al., 2024) by +9.3% and ShareGPT4V (13B) (Chen et al., 2023b) by +17.8% CoT evaluation scores. Compared to other mathematical visual specialist, i.e., G-LLaVA (7B) (Gao et al., 2023a) and the concurrent Math-LLaVA (13B) (Shi et al., 2024), MAVIS-7B exhibits superior problem-solving capabilities with higher CoT evaluation scores of +19.5% and +11.1%, respectively. In addition, our model is also advantageous to the most powerful open-source MLLM series, LLaVA-NeXT (Li et al., 2024a), from 8B to 110B model sizes, demonstrating the math-specific proficiency of MAVIS-7B. Note that, the improvement brought by

[Figure 17]

## Figure 5: Problem-solving Comparison of MAVIS-7B and GPT-4V.

DPO (our fourth-stage training) is more apparent in CoT evaluation compared to the accuracy scores, indicating that the preference alignment learning can effectively boost the CoT reasoning capabilities.

Table 4 showcases the performance comparison on six other mathematical benchmarks, where our model still attains remarkable performance among other MLLMs. In detail, MAVIS-7B outperforms the closed-source Qwen-VL-Max (Bai et al., 2023a) by +6.1% in MMMU-Math, +3.6% in MathVision, and around +10% in three subsets of We-Math. Our model even exceeds GPT-4V (OpenAI, 2023b) in the three mathematical categories of MathVista, indicating our problem-solving and reasoning proficiency. We also observe that, the enhancement from DPO increases from ‘S1’ to ‘S3’ of We-Math, which well demonstrates its benefit on math problems with more intricate reasoning steps.

- 4.3 QUALITATIVE ANALYSIS In Figure 5, we compare the mathematical problem-solving examples between MAVIS-7B and GPT-
- 4V (OpenAI, 2023c). As presented, our model not only showcases better accuracy in understanding the geometric elements, function curves, and coordinate axes in mathematical diagrams, but also performs higher-quality step-by-step reasoning process for formula substitution and numerical calculation. This demonstrates the effectiveness of our four-stage training pipeline and automatic data engine for enhanced diagram understanding and CoT reasoning.
- 5 CONCLUSION

In this paper, we propose MAVIS, the first mathematical visual instruction tuning paradigm for MLLMs. We first introduce two high-quality datasets by a delicate data engine, MAVIS-Caption and MAVIS-Instruct, containing large-scale diagram-language and problem-solving data. Then, we customize a three-stage training framework to progressively train the math-specific vision encoder, the diagram-language alignment, and the mathematical reasoning capabilities of MLLMs. The obtained specialist model, MAVIS-7B, achieves superior performance across different mathematical visual benchmarks, demonstrating the potential to serve as a new standard for future research.

REFERENCES

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716– 23736, 2022.

Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. Openflamingo: An open-source framework for training large autoregressive vision-language models. arXiv preprint arXiv:2308.01390, 2023.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. ArXiv, abs/2308.12966, 2023a.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023b.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In Advances in neural information processing systems, pp. 1877–1901, 2020.

Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric P. Xing, and Liang Lin. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. ArXiv, abs/2105.14517, 2021a. URL https://api.semanticscholar.org/CorpusID: 235253782.

Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric P Xing, and Liang Lin. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. arXiv preprint arXiv:2105.14517, 2021b.

Jiaqi Chen, Tong Li, Jinghui Qin, Pan Lu, Liang Lin, Chongyu Chen, and Xiaodan Liang. Unigeo: Unifying geometry logical reasoning via reformulating mathematical expression. ArXiv, abs/2212.02746, 2022.

Jun Chen, Deyao Zhu1 Xiaoqian Shen1 Xiang Li, Zechun Liu2 Pengchuan Zhang, Raghuraman Krishnamoorthi2 Vikas Chandra2 Yunyang Xiong, and Mohamed Elhoseiny. Minigpt-v2: Large language model as a unified interface for vision-language multi-task learning. arXiv preprint arXiv:2310.09478, 2023a.

Lin Chen, Jinsong Li, Xiao wen Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. ArXiv, abs/2311.12793, 2023b. URL https://api.semanticscholar.org/CorpusID:265308687.

Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024.

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. https://lmsys.org/ blog/2023-03-30-vicuna/, March 2023.

Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, et al. Internlm-xcomposer2: Mastering free-form text-image composition and comprehension in vision-language large model. arXiv preprint arXiv:2401.16420, 2024.

Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024.

Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, et al. G-llava: Solving geometric problem with multi-modal large language model. arXiv preprint arXiv:2312.11370, 2023a.

Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, Hongsheng Li, and Yu Qiao. Llama-adapter v2: Parameter-efficient visual instruction model. arXiv preprint arXiv:2304.15010, 2023b.

Peng Gao, Renrui Zhang, Chris Liu, Longtian Qiu, Siyuan Huang, Weifeng Lin, Shitian Zhao, Shijie Geng, Ziyi Lin, Peng Jin, et al. Sphinx-x: Scaling data and parameters for a family of multi-modal large language models. arXiv preprint arXiv:2402.05935, 2024.

Google Gemini Team. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yujiu Yang, Minlie Huang, Nan Duan, Weizhu Chen, et al. Tora: A tool-integrated reasoning agent for mathematical problem solving. arXiv preprint arXiv:2309.17452, 2023.

Ziyu Guo, Renrui Zhang, Xiangyang Zhu, Yiwen Tang, Xianzheng Ma, Jiaming Han, Kexin Chen, Peng Gao, Xianzhi Li, Hongsheng Li, et al. Point-bind & point-llm: Aligning point cloud with multi-modality for 3d understanding, generation, and instruction following. arXiv preprint arXiv:2309.00615, 2023.

Jiaming Han, Renrui Zhang, Wenqi Shao, Peng Gao, Peng Xu, Han Xiao, Kaipeng Zhang, Chris Liu, Song Wen, Ziyu Guo, et al. Imagebind-llm: Multi-modality instruction tuning. arXiv preprint arXiv:2309.03905, 2023.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, MarieAnne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mixtral of experts. Arxiv 2401.04088, 2024.

Bu Jin, Yupeng Zheng, Pengfei Li, Weize Li, Yuhang Zheng, Sujie Hu, Xinyu Liu, Jinwei Zhu, Zhijie Yan, Haiyang Sun, et al. Tod3cap: Towards 3d dense captioning in outdoor scenes. arXiv preprint arXiv:2403.19589, 2024.

Mehran Kazemi, Hamidreza Alvari, Ankit Anand, Jialin Wu, Xi Chen, and Radu Soricut. Geomverse: A systematic evaluation of large models for geometric reasoning. arXiv preprint arXiv:2312.12241, 2023.

Bo Li, Kaichen Zhang, Hao Zhang, Dong Guo, Renrui Zhang, Feng Li, Yuanhan Zhang, Ziwei Liu, and Chunyuan Li. Llava-next: Stronger llms supercharge multimodal capabilities in the wild, May 2024a. URL https://llava-vl.github.io/blog/

- 2024-05-10-llava-next-stronger-llms/.

Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next: Tackling multi-image, video, and 3d in large multimodal models, June 2024b. URL https://llava-vl.github.io/blog/

- 2024-06-16-llava-next-interleave/.

Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models, 2024c. URL https://arxiv.org/abs/2407.07895.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In International Conference on Machine Learning, pp. 12888–12900. PMLR, 2022.

KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023a.

Xiaoqi Li, Mingxu Zhang, Yiran Geng, Haoran Geng, Yuxing Long, Yan Shen, Renrui Zhang, Jiaming Liu, and Hao Dong. Manipllm: Embodied multimodal large language model for object-centric robotic manipulation. arXiv preprint arXiv:2312.16217, 2023b.

Zhenwen Liang, Tianyu Yang, Jipeng Zhang, and Xiangliang Zhang. Unimath: A foundational and multimodal mathematical reasoner. In EMNLP, 2023.

Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, et al. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575, 2023.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023a.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023b.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024a. URL https: //llava-vl.github.io/blog/2024-01-30-llava-next/.

Jiaming Liu, Chenxuan Li, Guanqun Wang, Lily Lee, Kaichen Zhou, Sixiang Chen, Chuyan Xiong, Jiaxin Ge, Renrui Zhang, and Shanghang Zhang. Self-corrected multimodal large language model for end-to-end robot manipulation. arXiv preprint arXiv:2405.17418, 2024b.

Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. arXiv preprint arXiv:2105.04165, 2021.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chun yue Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating math reasoning in visual contexts with gpt-4v, bard, and other large multimodal models. ArXiv, abs/2310.02255, 2023.

OpenAI. Chatgpt. https://chat.openai.com, 2023a. OpenAI. Gpt-4 technical report. ArXiv, abs/2303.08774, 2023b. OpenAI. GPT-4V(ision) system card, 2023c. URL https://openai.com/research/

gpt-4v-system-card.

Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Zhuoma GongQue, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? arXiv preprint arXiv:2407.01284, 2024.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, 2021. URL https://api.semanticscholar.org/CorpusID:

231591445.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model, 2024. URL https://arxiv.org/abs/2305.18290.

Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See-Kiong Ng, Lidong Bing, and Roy Ka-Wei Lee. Math-llava: Bootstrapping mathematical reasoning for multimodal large language models. arXiv preprint arXiv:2406.17294, 2024.

Chang Shu, Baian Chen, Fangyu Liu, Zihao Fu, Ehsan Shareghi, and Nigel Collier. Visual med-alpaca: A parameter-efficient biomedical llm with visual capabilities, 2023.

Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Le Hou, Kevin Clark, Stephen Pfohl, Heather Cole-Lewis, Darlene Neal, et al. Towards expert-level medical question answering with large language models. arXiv preprint arXiv:2305.09617, 2023.

Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai. Pandagpt: One model to instruction-follow them all. arXiv preprint arXiv:2305.16355, 2023.

InternLM Team. Internlm: A multilingual language model with progressively enhanced capabilities, 2023.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023b.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. arXiv preprint arXiv:2402.14804,

- 2024a.

Ke Wang, Houxing Ren, Aojun Zhou, Zimu Lu, Sichun Luo, Weikang Shi, Renrui Zhang, Linqi Song, Mingjie Zhan, and Hongsheng Li. Mathcoder: Seamless code integration in LLMs for enhanced mathematical reasoning. In The Twelfth International Conference on Learning Representations,

- 2024b. URL https://openreview.net/forum?id=z8TW0ttBPp.

Runsen Xu, Xiaolong Wang, Tai Wang, Yilun Chen, Jiangmiao Pang, and Dahua Lin. Pointllm: Empowering large language models to understand point clouds. arXiv preprint arXiv:2308.16911, 2023.

Senqiao Yang, Jiaming Liu, Ray Zhang, Mingjie Pan, Zoey Guo, Xiaoqi Li, Zehui Chen, Peng Gao, Yandong Guo, and Shanghang Zhang. Lidar-llm: Exploring the potential of large language models for 3d lidar understanding. arXiv preprint arXiv:2312.14074, 2023.

Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, Chaoya Jiang, Chenliang Li, Yuanhong Xu, Hehong Chen, Junfeng Tian, Qi Qian, Ji Zhang, and Fei Huang. mplug-owl: Modularization empowers large language models with multimodality, 2023a.

Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration, 2023b.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284, 2023.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502, 2023a.

Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning. arXiv preprint arXiv:2309.05653, 2023b.

Xiang Yue, Tuney Zheng, Ge Zhang, and Wenhu Chen. Mammoth2: Scaling instructions from the web. arXiv preprint arXiv:2405.03548, 2024.

Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao. LLaMA-adapter: Efficient fine-tuning of large language models with zero-initialized attention. In The Twelfth International Conference on Learning Representations, 2024a. URL https://openreview.net/forum?id=d4UiXAHN2W.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? arXiv preprint arXiv:2403.14624, 2024b.

Aojun Zhou, Ke Wang, Zimu Lu, Weikang Shi, Sichun Luo, Zipeng Qin, Shaoqing Lu, Anya Jia, Linqi Song, Mingjie Zhan, et al. Solving challenging math word problems using gpt-4 code interpreter with code-based self-verification. arXiv preprint arXiv:2308.07921, 2023.

A APPENDIX

- A.1 RELATED WORK

Visual Instruction Tuning. The advancement of large language models (LLMs) (Brown et al., 2020; Jiang et al., 2024; Touvron et al., 2023b; Chiang et al., 2023) with instruction tuning has significantly enhanced zero-shot capabilities across a range of tasks. Drawing inspiration from this, LLaMAAdapter series (Zhang et al., 2024a; Gao et al., 2023b; Han et al., 2023) propose a zero-initialized attention mechanism to align frozen vision encoders (Radford et al., 2021) with LLaMA (Touvron et al., 2023a) for multi-modal learning. LLaVA series (Liu et al., 2023b;a) employ a linear projector for vision-language alignment, establishing visual instruction tuning as a standard training approach in the multi-modal field. Flamingo (Alayrac et al., 2022) and OpenFlamingo (Awadalla et al., 2023) have honed visual representation by integrating a cross-attention resampler with vision encoders. SPHINX series (Gao et al., 2024; Lin et al., 2023) utilize a blend of visual encoders to make the LLM cognizant of various image aspects. InternVL series (Chen et al., 2024; Dong et al., 2024; Team, 2023) employ a large vision encoder and QFormer (Li et al., 2022) to incorporate high-quality visual information through a multi-stage training methodology. LLaVA-NexT (Liu et al., 2024a; Li et al., 2024a;b) further introduces the ‘AnyRes’ technique to manage images at any given resolution, and LLaVA-NexT-Interleave (Li et al., 2024c) extends the scope widely to interleave multi-image settings. There are also recent efforts to apply visual instruction tuning to 3D (Guo et al., 2023; Xu et al., 2023) and video (Li et al., 2023a; Fu et al., 2024) scenarios. Despite the impressive strides made in both model capability and training efficiency by multi-modal large language models (MLLMs) through visual instruction tuning, there is currently no MLLM specifically designed for mathematical problem-solving, nor a substantial dataset available for such purposes in the open-source community. In this paper, we mitigate the issue by proposing MAVIS with high-quality mathematical visual datasets and training paradigms.

Mathematics in Large Models. Recent research has predominantly concentrated on text-only mathematical problem-solving using LLMs. MAmmoTH (Yue et al., 2023b; 2024) have compiled extensive collections of mathematical problems, training LLMs using the reasoning processes described in solutions. MetaMATH (Yu et al., 2023) has expanded upon this by rewriting existing problems to create a larger dataset. MathCoder (Wang et al., 2024b) and ToRA (Gou et al., 2023) introduced a tools agent approach, employing Python code and symbolic resolvers during the training phase, significantly outperforming traditional models that rely on text-only mathematical reasoning. However, in the multi-modal field, despite the introduction of several datasets such as Geometry3K (Lu et al., 2021), GeoQA (Chen et al., 2021a), UniGeo (Chen et al., 2022), UniMath (Liang et al., 2023), and GeomVerse (Kazemi et al., 2023), aiming at enhancing the performance of MLLMs in solving graphical mathematical problems, these datasets are quite limited in scale and domain. Based on these datasets, G-LLaVA (Gao et al., 2023a) has developed superior capabilities for understanding graphical geometries but struggles with mathematical problems in other domains. The comprehensive benchmark MathVerse (Zhang et al., 2024b) has also highlighted the existing MLLMs’ unsatisfactory capacity for encoding visual diagrams in diverse mathematical domains. Therefore, there is a pressing need for the development of more robust encoders for mathematical images and the tuning of MLLMs with mathematical visual instructions, for which we propose MAVIS to address the challenges.

- A.2 HUMAN EVALUATION OF MAVIS-INSTRUCT

To assess the dataset’s coverage, validity, and quality, human verification is employed. The creation process of our MAVIS-Instruct dataset can be broadly categorized into two approaches:

- • GPT-generated: This method leverages GPT-4 to generate new problems (including questions, rationales, and answers) based on existing problems with diagrams. While this approach produces fluent, human-like sentences, it may be influenced by the inherent capabilities and occasional instability of GPT-4V.
- • Data Engine: As the main source of our mathematical visual data, this method utilizes the custom automatic data engine to generate new problems (including diagrams, questions, rationales, and answers), without relying on GPT models. It guarantees 100% correctness due to the use of rigorous templates, though it may occasionally exhibit rigid expressions.

###### GPT-generated Data Engine

###### GPT-generated Rule-based

Score 3 Score 2 Score 1

Score 3 Score 2 Score 1

Diagram Question Rationale Answer

Diagram Question Rationale Answer

3.00

3.00

3.00 3.00

3.00

3.00

3.00 3.00

3.00

3.00

3.00 3.00

3.00

3.00

3.00 3.00

2.97

100

2.97

3.00

2.96

3.00

2.96

2.92

2.92

2.91

2.91

cases

2.85

2.85

2.83

2.83

2.00

2.00

Diagram Question Rationale Answer

Diagram Question Rationale Answer

GPT Engine GPT Engine GPT Engine GPT Engine GPT Engine GPT Engine GPT Engine

GPT- Rule- GPT- Rule- GPT- Rule- GPT- Rule- GPT- Rule- GPT- Rule- GPT- Rule-

100

3

3

4 3

3

4 3

3

8

9

8

9

cases

9

9

11

11

96 100 97 86 100 100 100 92 88 100 100 91 100 100

96 100 97 86 100 100 100 92 88 100 100 91 100 100

Figure 6: Human Evaluation Results on 200 randomly sampled problems in MAVISInstruct, 100 GPT-generated and 100 Data Engine. We set three levels (1, 2, and 3) for each metric, and report average scores.

Figure 7: Human Evaluation Statistics on 200 randomly sampled problems in MAVISInstruct, 100 GPT-generated and 100 Data Engine. We count the numbers of three score levels (1, 2, and 3) for each metric.

Specifically, we evaluate four aspects(Diagram, Question, Rationale and Answer) of each problem using seven metrics. Each metric is scored on a scale of 1 to 3, where 1 denotes poor, 2 denotes moderate, and 3 denotes good. The human evaluation results are shown in Figure 6 and score statistics are shown in Figure 7. In addition, we also showcase some specific examples in Figure 8 and Figure 9. We analyze each aspect as follows:

- • Diagram: The diagrams in GPT-generated problems are directly collected from existing sources with rigorous human filtering, ensuring high quality, resulting in scores close to 3. In contrast, for rule-based problems, the diagrams are drawn accurately using Python code driven by our data engine, which guarantees correctness. However, these diagrams may lack alignment with human aesthetic preferences, as indicated by 3% of them receiving an appearance score of 1.
- • Question: Regarding the questions, both GPT-generated and rule-based problems display a high degree of accuracy in aligning with the diagram elements. This is attributed to the well-crafted prompts used with GPT-4 and the meticulous template design of the data engine. Nevertheless, rule-based questions may occasionally exhibit minor fluency issues, as they lack human refinement.
- • Rationale: In terms of the rationales, most instances feature a precise and detailed chain-ofthought (CoT) reasoning process. However, in a few cases (3% receiving an accuracy score of 1), some GPT-generated rationales contain minor reasoning or calculation errors, which are inherent to GPT-4’s limitations in problem-solving. These errors usually affect only one or two steps and do not compromise the overall logic. Conversely, the rule-based rationales are highly accurate due to the carefully designed data engine, although there is still room for improvement in language fluency.
- • Answer: The answers in both methods achieve high correctness scores. For GPT-generated problems, we prompt GPT-4 to identify a known condition from the original problems as the answer. Similarly, for rule-based problems, we randomly select a known attribute from the generated diagrams to serve as the answer.

Overall, the randomly sampled instances show that our dataset exhibits good question quality and answer accuracy.

GPT-generated

[Figure 18]

Appearance Score: 3

Date Engine

[Figure 19]

Appearance Score: 3

Date Engine

[Figure 20]

Appearance Score: 3

Date Engine

[Figure 21]

Appearance Score: 1

- Figure 8: Diagram Examples in MAVIS-Instruct. The first three diagrams showcase superior correctness and appearance, while a small portion of Data Engine generated diagrams (3%) are not aligned with human preference, e.g., the fourth diagram.

[Figure 22]

- Figure 9: Accurate Rationale Examples in MAVIS-Instruct. Most GPT-generated and Data Engine-generated rationales ensure correctness.

