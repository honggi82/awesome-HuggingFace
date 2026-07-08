# arXiv:2508.17188v2[cs.AI]12Apr2026

## PosterGen: Aesthetic-Aware Multi-Modal Paper-to-Poster Generation via Multi-Agent LLMs

Zhilin Zhang1,2* Xiang Zhang3˚ Jiaqi Wei4 Yiwei Xu5 Chenyu You1†

1Stony Brook University 2New York University 3University of British Columbia 4Zhejiang University 5University of California, Los Angeles

#### Abstract

Multi-agent systems built on large language models (LLMs) and multimodal large language models (MLLMs) have recently shown strong capabilities on complex compositional tasks. We apply this paradigm to paper-to-poster generation, a practical but time-consuming step in research communication. Most of the existing methods ignore core design and aesthetic principles, and often produce posters that still require substantial manual refinement. To address these limitations, we propose PosterGen, a multi-agent framework that embeds fundamental design principles into a specialized agent workflow that mirrors professional designers. Our system incorporates collaborative agents to distill a paper’s narrative (CuratorAgent), map content to a balanced spatial layout (LayoutAgent), apply a cohesive visual system of color and typography (StylistAgents), and render the final poster in PPTX format (Renderer). This design-centric approach produces posters that are both semantically grounded and visually compelling. To systematically evaluate visual quality, we introduce a comprehensive VLM-based rubric measuring information fidelity and aesthetic design, which we validate against human preferences in a user study. Experimental results show that PosterGen achieves both content fidelity and design aesthetics comparable to human-designed posters while significantly and consistently outperforming state-of-the-art methods. Code is publicly available at https://github.com/YResearch-SBU/PosterGen.

#### 1. Introduction

Academic posters serve as an indispensable venue for fast, visual, and interactive research communication for conferences, as they provide a medium for direct, one-on-one dialogue between the target audience and the authors [10]. However, many researchers suffer from spending signifi-

*Equal contribution. †Corresponding author.

Table 1. Comparison of existing paper-to-poster methods and their supported features. Here ✓ denotes fully supported, △ denotes partially supported, and ✗ denotes not supported.

Output Format Outline Layout Styling / Editable Slides Hierarchy Alignment Cohesion

Method

PosterBot [55] LaTeX / △ △ △ ✗ P2P [44] HTML / ✗ ✓ ✓ ✗ PosterAgent [32] PPTX / ✓ ✓ ✗ ✗

PosterGen (ours) PPTX / ✓ ✓ ✓ ✓

cant efforts and time on the deliberate design of an academic poster. Therefore, there is urgent need for an automatic method for generating high-quality academic posters that can significantly relieve manual burdens.

Although the automatic generation of commercial posters has received significant research attention [4, 5, 12], research on academic poster generation is far less explored. Early works on academic posters often used neural models [35, 55]. Other works focused on specific sub-tasks like layout generation [47] or text summarization [25, 37]. These methods often produce posters with quality issues, such as content overflow [35], and require further manual adjustment. Recently, LLM-powered multi-agent systems have shown strong performance on solving complex tasks. P2P [44] and PosterAgent [32] were the first to apply this approach to academic poster generation. However, their works do not sufficiently consider aesthetics and design principles, e.g., well-organized layout design that ensures natural reading flow, and styling choices for color and typography to present visual hierarchy.

To move beyond the aesthetic limitations of current approaches and minimize the need for manual refinement, we propose the first aesthetic-driven multi-agent framework, PosterGen, designed to generate academic posters that are close to human-level aesthetic perfection. As shown in Table 1, existing methods [32, 44, 55] offer partial solutions, but PosterGen is the first to fully integrate both principled layout and stylings while supporting a more accessible PPTX format than LaTeX or HTML. We argue that embedding these aesthetic principles is indispensable for full

###### Content Layout Styling Renderer

[Figure 1]

Paper.pdf

[Figure 2]

[Figure 3]

|[Figure 4]|
|---|

|[Figure 5]|
|---|

|[Figure 6]|
|---|

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Curator Agent

Layout Agent

Color Agent

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

•••

Python-pptx

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Poster

###### ...

Structured Sections

Initial Storyboard

ABT Narrative

Initial Canvas

|[Figure 25]|
|---|

[Figure 26]

[Figure 27]

PDF Converter

[Figure 28]

[Figure 29]

|Parameter Calculation<br><br>[Figure 30]<br><br>Text box Height<br><br>[Figure 31]<br><br>Image Size<br><br>[Figure 32]<br><br>[Figure 33]|
|---|

[Figure 34]

[Figure 35]

Select & Filter

[Figure 36]

Assets

[Figure 37]

Font Agent

|Narrative-based Storyboard<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>...|
|---|

[Figure 41]

[Figure 42]

[Figure 43]

|[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>Balancing<br><br>Spatial Layout<br><br>[Figure 47]<br><br>[Figure 48]|
|---|

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Structured Sections

Figures & Tables

ABT Narrative

Initial Storyboard

Bold Size Highlight

Italic

- Figure 1. Overview of PosterGen. The process consists of three main stages: (1) ParserAgent processes the input paper, extracting all text and visual assets and organizing them into a structured format focusing on an ABT narrative. (2) A series of aesthetic-aware agents then transform this content into a styled layout: CuratorAgent creates a narrative-based storyboard, LayoutAgent calculates the precise spatial arrangement and balances the columns, and StylistAgents apply a harmonious color palette and a hierarchical typographic system.

(3) Finally, Renderer module takes the styled metadata and produces the output poster.

automation. PosterGen implements a collaborative multiagent workflow that mirrors a professional design process, and consists of four critical stages: (i) distilling the essential narrative and visual assets; (ii) constructing a narrativedriven storyboard based on the ABT structure; (iii) mapping the storyboard to a balanced and aesthetic-aware layout; and (iv) applying a cohesive system of colors and fonts.

We further evaluate PosterGen against state-of-the-art methods via VLM-as-Judge and human evaluation, which leads to several key findings: ❶ Text-to-image generation methods (GPT-4o-Image) frequently suffer from content hallucination and gibberish text. ❷ PosterGen achieves both content and aesthetic quality comparable to humandesigned posters while consistently outperforming state-ofthe-art multi-agent methods across almost all metrics. ❸ Quantitative and qualitative results confirm that our designcentric approach is highly effective, producing visually compelling and presentation-ready posters.

Overall, our main contributions are as follows:

- • We propose PosterGen, a novel aesthetic-aware multimodal multi-agent framework for academic poster generation that, to our knowledge, is the first to incorporate core design principles directly into its agent workflow.
- • We introduce a comprehensive VLM-based evaluation rubric to assess the functional and aesthetic quality of generated posters, covering information, layout, color, and typography, and validated against human preferences.
- • We present both quantitative and qualitative ablation studies that validate our fine-grained agent decomposition and the contribution of each specialized agent.

#### 2. Related Work

Poster Generation. Recent research has explored the automatic generation of artistic and product posters in a broad sense. For example, some works utilize modular [4] or unified [5] frameworks to achieve a high aesthetic quality in

generated posters. Other methods focus on precise generation control, such as layout structure control using language models [38], text accuracy [12], or handling multiple userprovided conditions [60], all of which excel at creating visually appealing posters for art or marketing purposes. While these methods excel at visual appeal, an academic poster differs in its primary goal to convey complex research with precision and clarity in limited space. Early works used neural models to generate posters from papers [35, 55]. Recently, [37, 47] proposed benchmarks for this task; however, these methods suffer from several limitations, such as content overflow [35], and restrictions to layout generation [47] or text summary [37] only.

Recent studies show that LLM-powered multi-agent frameworks can outperform single models on complex multimodal tasks [2, 14, 18, 21, 22, 26, 30, 42, 43, 50, 52, 54, 57–59, 61, 62, 66] by letting agents take on specialized roles and coordinate through mechanisms like selfreflection [1, 51]. P2P [44] and PosterAgent [32] were the first to apply this multi-agent solution to scientific poster generation. However, these methods lack a thorough consideration of design principles and aesthetics, and require extensive manual adjustments before they are ready for use in a conference poster session.

Aesthetic Design with MLLMs. Recent works have explored multimodal large language models (MLLMs) for aesthetics-aware visual design. For instance, DesignProbe [23] investigates the aesthetic reasoning capabilities of MLLMs through benchmarks that assess models in terms of color harmony, typography, and composition. Other works utilize MLLMs to generate aestheticsconstrained layouts, such as hierarchical layouts [6], generalized content-aware layouts [16] and aesthetic-aligned layout training schemes [34]. Beyond layout prediction, POSTA [4] integrates MLLMs with diffusion-based rendering to enable customizable artistic poster creation; Poster-

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Chaotic Reading Flow No Visual Anchor

Natural Reading Flow Anchor at Eye Level

Aesthetic-driven

[Figure 58]

[Figure 59]

##### Poster Layout

Introduction Method Main Results

Introduction & Motivation Methods

Visual and Design principles to follow for Poster Layout:

Motivation

Motivation

- 1. Multi-Column Layout
- 2. Left-aligned Titles
- 3. Eye-level Visual Anchor
- 4. Separate by White Space
- 5. Avoid distracting elements

Related Work Experiments

Results

Related Work

Takeaways

- Figure 2. A comparison of different layout structures. The vertically unaligned layout (left) results in a chaotic reading flow. In contrast, the vertically aligned grid (right), adopted by our LayoutAgent, establishes a natural reading flow and places the visual anchor at eye-level for emphasis.

Maker [12] introduces a high-quality rendering pipeline that improves the visual and linguistic fidelity for posters.

- 3. Design Principles

lish two hierarchy types: (1) a visual hierarchy using different font sizes (title, headings, body) to organize content structure, and (2) a semantic hierarchy using formatting like bold, italics, or contrast color for emphasis.

Academic posters are visual media that require deliberate design to initiate conversations. Instead of relying on arbitrary heuristics, our framework is built upon a foundation of established research into high-quality poster design [10]. We distill these best practices into four core principles that are embedded directly into our agent designs.

Narrative. A coherent narrative is the foundation of a design-aware poster. Following a schema widely adopted in scientific writing, we adopt the “And, But, Therefore” (ABT) structure [29] to distill the paper’s core message, which establishes context (And), identifies problems (But), and presents solutions (Therefore). This narrative then guides the creation of specific, content-driven section titles. Layout Structure. Since a poster is a two-dimensional space with width and height, a three-column grid is a common and effective method to ensure a natural reading flow, as shown in Figure 2 (right). This structure strategically places a key visual anchor at the eye-level hot zone (top of the center column) and utilizes white space to separate elements and reduce visual clutter.

Color Design. Color is used to create hierarchy and ensure accessibility. One optimal approach is to employ a restrained, theme-based, and monochromatic palette to maintain visual harmony (Figure 4). This framework establishes a three-tier system: theme color for primary emphasis, monochromatic variants for section backgrounds, and a high-contrast accent color for highlights, with all text following the WCAG 4.5:1 contrast ratio to ensure readability. Typography Design. Typography complements color to reinforce clarity and readability from a standard viewing distance. We prioritize legible sans-serif typefaces and estab-

#### 4. Method

In this work, we propose a novel multi-agent framework for generating functionally effective and aesthetic-aware scientific posters, as shown in Figure 1. Our framework implements the core design criteria from Section 3 by incorporating them as core logic within each specialized agent. This architecture establishes a cascade of structured design constraints throughout the entire generation process.

The PosterGen workflow consists of five specialist agents and one post-processing module: ParserAgent (Section 4.1), CuratorAgent (Section 4.2), LayoutAgent (Section 4.3), ColorAgent and FontAgent (Section 4.4), and Renderer (Section 4.5).

###### 4.1. ParserAgent

Given a research paper in PDF format, ParserAgent initiates the workflow and is responsible for extracting the raw text and all available visual assets (e.g., figures and tables). To accomplish this, we utilize an external PDF converter tool, Marker [33], which converts the paper’s content into Markdown format and saves all visual assets as PNG images.

To minimize token usage for downstream agents (particularly for lengthy papers), ParserAgent concurrently performs several processing functions. (a) It distills the paper’s core narrative into the ABT structure (see Section 3), to establish a guiding framework for all subsequent content organization; (b) it restructures the raw text into logical sections that focus on main content and essential details, under rigid limitation of maximum 1000 words per section; and (c) it classifies the extracted visual assets into distinct categories

Margin_s Padding_s

Section Content

Margin_v

Margin_t

Padding_v

Padding_t

Figure / Table

Text

[Figure 60]

[Figure 61]

●

◦

●

- Figure 3. An illustration of the CSS-like box model used to control the spacing between poster elements for XML files.

based on their narrative role: a single “key visual” representing the core research; visuals for “problem illustration” and “method workflow”; figures depicting “main results” and “comparative results”; and all other visual elements as “supporting” material. Prompts of ParserAgent can be found in Figures 9 to 12 in the supplementary material.

###### 4.2. CuratorAgent

CuratorAgent functions as a spatial narrative designer. Its primary design consideration is to orchestrate all parsed content elements tightly around the ABT narrative. This narrative-centric approach ensures that the poster’s structure is fluid and engaging. By establishing a strong narrative foundation early, it also minimizes the need for unnecessary content and visual refinements in later stages.

Operating on the ABT structure and structured sections provided by ParserAgent, CuratorAgent performs the initial strategic placement of content, and maps the narrative onto a preliminary three-column storyboard. To follow the narrative and visual strategy, CuratorAgent enforces a strict limit of five to eight sections for the entire poster. This constraint guarantees that the three-column layout is fully utilized while preventing content overflow, which imitates a typical human design pattern that progresses logically from introduction and methods to results and discussion. Representative prompts of CuratorAgent can be found in Figures 13 to 15 in the supplementary material.

###### 4.3. LayoutAgent

LayoutAgent is a hybrid procedure that transforms the storyboard from CuratorAgent into a precise spatial layout. It operates in three phases: Phases 1 and 3 are procedural, computing the exact coordinates and dimensions tx,y,w,hu for each element within a three-column grid

- (Figure 2); Phase 2 invokes an LLM-based Balancer to optimize column utilization based on the spatial analysis from Phase 1. We use two key tools support this process: an element height estimation algorithm (Algorithm 1) for accurate vertical space allocation, and a CSS-like box model
- (Figure 3) for precise whitespace control in XML. While calculating the height for visual assets is straight-

forward due to their fixed aspect ratios, determining the

Algorithm 1: Optimal TextFrame Height Estimate Input: Text T, width w, font attr. f, precision ε Data: Initial bounds for binary search Output: h‹ (estimated height)

- 1 hmin,hmax Ð initial bounds;
- 2 while hmax ´ hmin ą ε do

- 3 htest Ð phmin ` hmaxq{2;
- 4 B Ð SimulateTextboxpT,w,htest,fq;
- 5 if IsOverflowingpBq then

- 6 hmin Ð htest;
- 7 else

- 8 hmax Ð htest;
- 9 end if
- 10 DeletepBq;
- 11 end while
- 12 h‹ Ð hmax ` NewlineOffsetpT,f.sizeq;
- 13 return h‹;

height for textFrames is much more complex for PPTX. This challenge derives from a discrepancy between the python-pptx library, which acts as an XML editor, and the final rendering engine (e.g., Microsoft PowerPoint) that determines the actual appearance. To bridge this gap, we propose the estimation algorithm detailed in Algorithm 1. The algorithm first employs a binary search to identify the minimum text box height that avoids any font size reduction by the rendering engine. It then applies a corrective offset, calculated from the number of newline characters, to compensate for subtle deviations in the engine’s behavior.

To control whitespace precisely, we implement a CSSlike box model (Figure 3) that encapsulates every element with distinct ‘margin’ and ‘padding’ settings for finegrained spacing control. This approach significantly narrows the layout capability gap that typically exists between automated HTML-based and PPTX-based layout generation methods. The prompts of LayoutAgent can be found in Figures 16 and 17 in the supplementary material.

###### 4.4. StylistAgents

Once the spatial layout is determined, StylistAgents apply the visual and typographic details to generate styled layouts. This stage consists of two specialized components: ColorAgent and FontAgent. Rather than the simple assignment of colors and fonts, we highlight the importance of a design thinking process rooted in the principles of poster aesthetics. This perspective is based on a core understanding that in academic posters, color and typography are not merely decorative; instead, they serve as essential media for both visual and semantic hierarchy.

ColorAgent. ColorAgent focuses on creating a suitable and harmonious color palette, as shown in Figure 4. It

Theme Monochrome

###### Assets

[Figure 62]

Contrast

###### Theme

###### Contrast

###### Monochrome

Aﬃliation Logo / Key Figure

Accent Colors for Key Words

Section Background

- Figure 4. Overview of the color palette generation. A primary color is extracted from a source image to get monochromatic and contrast colors.

first searches for the author’s affiliation logo. If it exists, a VLM is adopted to analyze the image and extract a dominant theme color. This method leverages the institution’s official branding to ensure an official appearance. For a fallback plan, ColorAgent can also analyze the key figure from the paper to identify a suitable theme color. After selecting the primary theme color, the next step for ColorAgent is to generate a complete color scheme strictly following color theory principles. For instance, given the theme color, ColorAgent will create the following color scheme:

- • monochromatic shades for backgrounds and accents, e.g., monochromatic light and dark;
- • a high-contrast color that is used specifically for highlighting important keywords.

In this way, the ColorAgent generates a limited color palette that ensures aesthetic cohesion and high readability. Complete prompts of ColorAgent can be found in Figure 18 in the supplementary material.

FontAgent. FontAgent manages typography and works to establish a clear visual hierarchy and emphasize key information within the text. It operates in a two-stage process: it first employs one LLM call to analyze the summarized text of the paper, which extracts a list of important keywords for each section. Next, FontAgent applies styling by using a set of predefined interfaces to assign different font families and sizes. FontAgent also highlights the keywords identified in the previous stage via the contrast color from ColorAgent. To avoid a tedious appearance, we adopt diverse highlighting styles, i.e., bolding and italics, to make the poster more visually engaging. Complete prompts of FontAgent can be found in Figure 19 in the supplementary material.

###### 4.5. Renderer

The renderer post-processes the fully styled layout metadata from the previous agents and renders a standard PPTX file using the python-pptx library. Additionally, it attaches

- Table 2. VLM-as-Judge evaluation criteria for poster content and design, all on a 1-5 scale.

Domain Focus Area Dimension

Content

Information

Layering, Coverage, Depth, Completeness

Narrative ABT, Flow, Conciseness

Aesthetics

Layout

Grid, Whitespace, Anchor, Balance Color

Palette, Hierarchy, Contrast, Source Typography

Font, Emphasis, Legibility

Strictness Alignment, Robustness

- Table 3. Quantitative results on content metrics across different poster generation methods, with scores averaged over 30 posters rated on a 1-5 scale. The best scores for each content metric are bolded. The second best scores are underlined.

Information Narrative

VLM Method

Avg. Layer.Ò Cover.Ò DepthÒ Compl.Ò ABTÒ FlowÒ Conc.Ò

Human-designed 3.47 3.83 3.97 4.50 4.53 4.47 4.47 4.18 GPT-4o-Image 2.23 2.10 2.20 2.20 2.73 2.87 2.57 2.41 P2P [44] 2.63 3.33 3.17 3.83 4.67 4.57 4.53 3.82 PosterAgent [32] 3.00 3.33 3.33 4.80 4.07 4.17 4.07 3.82

GPT-4.1

PosterGen (ours) 3.27 3.93 3.83 4.70 4.87 4.90 4.83 4.33

Human-designed 3.73 3.33 4.13 4.47 4.40 4.27 3.73 4.01 GPT-4o-Image 2.23 2.03 2.87 2.10 2.83 2.60 2.13 2.40 P2P [44] 2.90 3.13 3.50 4.73 4.27 4.27 3.77 3.80 PosterAgent [32] 3.03 3.07 3.50 4.80 4.23 4.30 3.90 3.83

Claude Sonnet 4

PosterGen (ours) 3.97 3.43 4.27 4.80 4.60 4.87 4.53 4.35

affiliation and conference logos to the top-right corner of the poster. In the final step, the renderer uses LibreOffice (headless mode) to convert this PPTX file into a highquality PNG image for visual inspection and evaluation.

#### 5. Experiments

###### 5.1. Metrics

We evaluate the generated posters using the comprehensive criteria detailed in Table 2. To simulate how humans perceive and evaluate academic posters, we utilize VisionLanguage Models (VLMs) as judges, and use both GPT-4.1 and Claude Sonnet 4 to mitigate assessment biases.

The evaluation is divided into two fundamental domains: Content and Aesthetics, which are detailed in Table 2. The Content domain assesses the poster’s fidelity to the source paper, focusing on two key areas: Information (evaluating the layering, coverage, depth, and completeness) and Narrative (evaluating the ABT structure, logical flow, and conciseness). The Aesthetic domain evaluates the poster’s visual execution based on the principles from Section 3. This is broken down into four focus areas: Layout (grid, whites-

Table 4. Quantitative results on aesthetic metrics across different poster generation methods, with scores averaged over 30 posters rated on a 1-5 scale. The best scores for each aesthetic metric are bolded. The second best scores are underlined.

Layout Color Typography Strictness

VLM Method

Avg. GridÒ Wh.Sp.Ò Anch.Ò Bal.Ò P.l.t.Ò Hier.Ò Ctrst.Ò Src.Ò FontÒ Empha.Ò Legi.Ò Align.Ò Robust.Ò

Human-designed 3.77 3.80 4.47 3.83 3.83 3.93 4.97 4.10 3.93 4.30 4.70 4.93 5.00 4.27 GPT-4o-Image 2.80 3.07 4.00 2.80 2.90 2.73 5.00 2.43 3.37 3.17 4.13 3.33 1.13 3.14 P2P [44] 3.80 3.90 4.20 3.90 3.53 3.63 4.97 3.33 3.13 3.17 4.37 5.00 5.00 3.99 PosterAgent [32] 3.97 4.00 4.13 3.97 3.20 3.27 5.00 3.17 3.17 3.00 4.73 5.00 5.00 3.97

GPT-4.1

###### PosterGen (ours) 4.20 4.27 4.97 4.27 3.50 3.53 5.00 4.47 4.17 4.23 5.00 5.00 5.00 4.43

Human-designed 4.27 3.97 4.47 4.27 4.17 4.13 5.00 4.20 3.30 4.77 3.87 4.47 5.00 4.30 GPT-4o-Image 2.90 2.83 3.73 2.60 2.10 1.97 4.93 2.17 3.00 3.10 2.87 2.77 2.00 2.84 P2P [44] 4.30 3.90 4.37 4.10 3.37 3.60 5.00 3.43 2.97 3.70 3.50 4.87 5.00 4.01 PosterAgent [32] 4.50 3.93 4.03 4.27 2.90 3.10 5.00 2.83 3.10 3.70 3.63 4.80 5.00 3.91

Claude Sonnet 4

PosterGen (ours) 4.37 4.10 4.87 4.57 3.07 3.20 5.00 4.10 3.27 4.83 4.00 4.70 5.00 4.24

pace, anchor, balance), Color (palette, hierarchy, contrast, source), and Typography (font, emphasis, legibility). Finally, the Strictness focus area assesses the technical quality, such as element Alignment and Robustness against critical flaws like content overlap and overflow.

###### 5.2. Baselines

We choose two types of baselines: an end-to-end textto-image generation method, i.e., GPT-4o-Image, and the state-of-the-art multi-agent poster generation methods.

GPT-4o-Image Generation is directly based on the ChatGPT web interface. We provide the GPT-4o model with the source PDF file, along with a text prompt that instructs it to generate an academic poster of a given size. This method produces the final poster as a single image in an end-to-end way, without explicit intermediate generation stages.

P2P [44] is the first LLM-based multi-agent framework for academic poster generation. It uses three agents for visual element extraction, textual content generation, and final poster assembly, respectively. However, P2P renders its output exclusively in HTML and CSS, which is neither directly editable nor easily portable for practical use.

PosterAgent [32] proposes a top-down, multi-agent pipeline that consists of (1) Parser to distill the source paper into a structured asset library; and (2) Planner agent that arranges assets into a binary-tree layout, which is subsequently refined by a (3) Painter-Commenter loop that leverages VLM feedback to address layout issues. Although this baseline provides a solid technical solution for poster generation in the PPTX format, it does not sufficiently incorporate aesthetic and design principles into its agent workflow, which marks a key difference from our approach.

###### 5.3. Quantitative Results and Comparisons

We compare PosterGen with baseline approaches on 30 topconference papers, which are detailed in Sec. E in the supplementary material. The results are presented in Table 3

[Figure 63]

Figure 5. Human evaluation on aesthetic scores across 10 randomly picked sets of posters. Specific dimension scores are averaged to represent design aesthetics.

and Table 4. Overall, PosterGen achieves best or secondbest performance across almost every metric, thus significantly outperforming all baselines and achieving a level of quality that is comparable to human-designed posters.

Content Performance. As shown in Table 3, PosterGen achieves the highest average content score across both VLM judges. With GPT-4.1, PosterGen attains an average score of 4.33, surpassing both P2P [44] and PosterAgent [32] by 13.4%. The improvement is driven mainly by the superior narrative structure, as PosterGen achieves near-perfect scores in narrative metrics. This contrasts sharply with PosterAgent, which scores well on section completeness, but much lower across narrative metrics (-15.7%). This highlights the efficacy of our CuratorAgent, which leverages the ABT narrative structure to create a coherent storyboard rather than just aggregating content. Meanwhile, the Claude Sonnet 4 judge gives highly similar results, which further reinforces PosterGen’s informative capabilities.

Aesthetic Performance. As shown in Table 4, PosterGen again achieves the highest average aesthetic score of 4.43 (per GPT-4.1), which is comparable to the Human-designed score (4.27), outperforming P2P by 11.0% and PosterAgent

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Automation Human

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

(a) GPT-4o-Image (b) PosterAgent

(c) PosterGen (Ours) (d) Designed by humans

- Figure 6. Qualitative comparison on two representative papers [65, 68]. (a) Posters generated by GPT-4o-Image; (b) Posters generated by PosterAgent [32]; (c) Posters generated by PosterGen (ours); (d) Posters designed by humans (Ground truth).

by 11.6%. With regard to layout design, PosterGen leads across all four related metrics, including a near-perfect 4.97 on visual anchor (“Anch.”). Notably, although P2P benefits from HTML and CSS, which natively handle element alignment and responsive spacing, PosterGen surpasses it across all layout metrics under both VLM judges. This demonstrates the effectiveness of LayoutAgent in implementing core design principles, such as placing a visual anchor at eye-level, and using its CSS-like box model to effectively manage white space and visual balance.

While subjective metrics like color scheme (“P.l.t.”) are competitive with human designers, our StylistAgents perform much better in systematic and aesthetic design. Per GPT-4.1, PosterGen achieves perfect legibility and the highest score for color source, validating its principled approach to typography and color. This confirms that while agentbased methods are feasible, PosterGen’s design-centric agents are what truly deliver satisfactory aesthetic quality.

###### 5.4. Human Evaluation

Motivated by previous works [5, 56], we conduct a user study to establish correlation with human preferences. We randomly chose 10 sets of posters generated by PosterAgent, PosterGen, and humans to create 10 questions, and invited 30 researchers to evaluate these posters, rating each using a 1-5 Likert scale across four aesthetic dimensions, i.e., layout, color, typography, and technical quality. As shown in Figure 5, the results indicate that PosterGen significantly outperforms PosterAgent in all aesthetic scores. Moreover, PosterGen’s scores (4.15 for layout, 4.20 for color, 4.17 for typography) tend to be very close to those of the human-made posters (4.32, 4.32 and

- 4.21, respectively). This further verifies the effectiveness of our aesthetic-centric framework. Additionally, it also shows that human evaluators are stricter in comparison to VLM judges, as PosterAgent’s aesthetic scores in the hu-

man evaluation are around 15% lower than its VLM-asJudge scores, suggesting that it potentially requires more manual refinement. More details of the user study can be found in Section D in the supplementary material.

###### 5.5. Qualitative Results

To provide visually convincing demonstration, we conduct a qualitative comparison on two representative papers [65, 68], as shown in Figure 6. A visual inspection shows that PosterGen significantly outperforms the baselines.

A major flaw in the end-to-end GPT-4o-Image method (Figure 6 (a)) is the presence of critical content failures, despite a clear layout at first glance. Its outputs frequently suffer from sections of gibberish text, duplicated or broken content blocks, and the hallucination of visual assets not present in the source paper. In comparison, PosterAgent (Figure 6 (b)) represents a significant improvement in content fidelity using multi-agent workflow. However, it remains limited due to lack of aesthetic consideration. The layout often suffers from misaligned sections or huge wasted space, and fails to establish a logical reading flow. Furthermore, its stylings merely rely on identically sized, black bullet points and monotonous plain text that cannot create a visual hierarchy or emphasize any key information.

In contrast, the posters generated by PosterGen (Figure 6 (c)) exhibit a superior level of design quality that approaches the human-designed ones (Figure 6 (d)). The region around the title bar is designed with elegance, applying varied fonts to the title, authors, and affiliation, along with their respective logos, which impose an instant hierarchy. Instead of applying lined borders, PosterGen adopts an easy-reading approach that utilizes colored section blocks and deliberate use of whitespace. The textual content is also enriched, as key phrases are highlighted using contrasting colors and varied formats to direct the viewer’s attention. This is well demonstrated by the main anchor section,

Table 5. Quantitative results of ablation experiments over 10 posters. The best scores for each metric are bolded.

Agent Stage Content Metrics Aesthetic Metrics Curator Layout Styling InformationÒ NarrativeÒ Avg.Ò LayoutÒ ColorÒ TypographyÒ StrictnessÒ Avg.Ò

VLM Model

✓ ✗ ✗ 3.70 4.90 4.30 3.68 3.10 3.17 4.35 3.57 ✓ ✓ ✗ 3.80 4.83 4.32 4.40 3.25 3.47 5.00 4.03 ✓ ✓ ✓ 3.93 4.90 4.41 4.28 4.33 4.50 5.00 4.53

GPT-4.1

[Figure 74]

[Figure 75]

[Figure 76]

(a) Curator Agent (b) Curator Agent + Layout Agent (c) Curator Agent + Layout Agent + Stylist Agents

- Figure 7. Qualitative results of ablation experiments on [3]. (a) Output of CuratorAgent. Chaotic layouts are highlighted in the red dashed boxes. (b) Output of LayoutAgent. LayoutAgent applies spatial adjustments and balances space usage of columns. (c) Output of the entire multi-agent pipeline. StylistAgents apply visually appealing color and font elements to the poster.

which applies a light monochromatic background to provide emphasis without imposing visual strains. More detailed qualitative results can be found in Section E in the supplementary material.

###### 5.6. Ablation Study

We conduct an ablation study on 10 randomly selected posters to validate the contribution of each agent, with results reported in Table 5. The full pipeline consistently achieves the highest performance across both content and aesthetic metrics.

CuratorAgent effectively establishes the poster content, as “Information” and “Narrative” metrics remain high with negligible variation across all stages, suggesting that the core content is successfully curated at this early stage. LayoutAgent then significantly improves the “Layout” score and technical quality via spatial arrangement. StylistAgents further enhance color and typography, underscoring their contribution to the final visual quality.

We also visualize the progressive output on a representative paper [3] in Figure 7. CuratorAgent (a) produces the storyboard but with layout flaws such as imbalanced columns and content overflow, which are highlighted in the red dashed boxes. LayoutAgent (b) resolves these spatial issues through its box model and balancing loop, yielding a properly aligned three-column grid. StylistAgents (c) then apply aesthetic refinement by introducing themed colors to section titles and using a light background to emphasize the key method section, while highlighting keywords in contrasting colors and varied typography.

###### 5.7. Efficiency Analysis

We evaluate the runtime efficiency of PosterGen on a MacBook Air M3 without GPU acceleration or parallelism, over

Table 6. Runtime and cost analysis averaged over 10 posters.

Runtime Composition Parser Curator Layout Other

Runtime API Calls Cost (GPT-4.1)

- 5.32 min 8 $0.20/poster 62.73% 13.11% 17.89% 6.27%

10 randomly picked posters, as reported in Table 6. PosterGen generates a complete poster in 5.32 minutes on average, and requires 8 API calls at a cost of $0.20 per poster with GPT-4.1. ParserAgent accounts for the majority of runtime (62.73%), as it relies on Marker for PDF parsing and visual asset extraction. The remaining agents collectively complete in under two minutes, demonstrating the practical efficiency of our multi-agent workflow.

- 6. Conclusion

In this work, we present PosterGen, the first aestheticaware multi-agent framework capable of producing visually compelling posters approaching human-level quality. Our work is thoughtfully guided by fundamental design and aesthetic principles, incorporated into specialized agent collaboration that mirrors the process of professional designers. To systematically evaluate the visual quality, we introduce VLM-based criteria that measure information fidelity and aesthetic design, and conduct a user study to establish the correlation with human preferences. Experimental results demonstrate significant improvements in design aesthetics over state-of-the-art methods. Our method substantially alleviates the manual burden of preparing academic posters for researchers, and offers a principled and reproducible framework that narrows the aesthetic gap to humandesigned posters, which lays a foundation for future designaware multi-agent systems in scientific communication.

#### References

- [1] Xiaohe Bo, Zeyu Zhang, Quanyu Dai, Xueyang Feng, Lei Wang, Rui Li, Xu Chen, and Ji-Rong Wen. Reflective multi-agent collaboration based on large language models. Advances in Neural Information Processing Systems, 37: 138595–138631, 2024. 2
- [2] Juntai Cao, Xiang Zhang, Raymond Li, Jiaqi Wei, Chuyuan Li, Shafiq Joty, and Giuseppe Carenini. Multi2: Multi-agent test-time scalable framework for multi-document processing. In Proceedings of The 5th New Frontiers in Summarization Workshop, pages 135–156, 2025. 2
- [3] Tri Cao, Chengyu Huang, Yuexin Li, Wang Huilin, Amy He, Nay Oo, and Bryan Hooi. Phishagent: a robust multimodal agent for phishing webpage detection. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 27869–27877, 2025. 8
- [4] Haoyu Chen, Xiaojie Xu, Wenbo Li, Jingjing Ren, Tian Ye, Songhua Liu, Ying-Cong Chen, Lei Zhu, and Xinchao Wang. Posta: A go-to framework for customized artistic poster generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28694–28704, 2025. 1, 2
- [5] SiXiang Chen, Jianyu Lai, Jialin Gao, Tian Ye, Haoyu Chen, Hengyu Shi, Shitong Shao, Yunlong Lin, Song Fei, Zhaohu Xing, et al. Postercraft: Rethinking high-quality aesthetic poster generation in a unified framework. arXiv preprint arXiv:2506.10741, 2025. 1, 2, 7
- [6] Yutao Cheng, Zhao Zhang, Maoke Yang, Hui Nie, Chunyuan Li, Xinglong Wu, and Jie Shao. Graphic design with large multimodal model. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2473–2481, 2025. 2
- [7] Zhiyuan Cheng, James Chenhao Liang, Guanhong Tao, Dongfang Liu, and Xiangyu Zhang. Adversarial training of self-supervised monocular depth estimation against physical-world attacks. In The Eleventh International Conference on Learning Representations, 2023. 17
- [8] Chongyang Du, Tao Wang, Kaihao Zhang, Wenhan Luo, Lin Ma, Wei Liu, Xiaochun Cao, et al. Punctuation-level attack: Single-shot and single punctuation can fool text models. Advances in Neural Information Processing Systems, 36: 49312–49324, 2023. 18
- [9] Jiawei Du, Xin Zhang, Juncheng Hu, Wenxing Huang, and Joey T Zhou. Diversity-driven synthesis: Enhancing dataset distillation through directed weight adjustment. Advances in neural information processing systems, 37:119443–119465,

2024. 23

- [10] Zen Faulkes. Better posters: plan, design and present an academic poster. Pelagic Publishing Ltd, 2021. 1, 3
- [11] Tsu-Jui Fu, William Yang Wang, Daniel McDuff, and Yale Song. Doc2ppt: Automatic presentation slides generation from scientific documents. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 634–642, 2022. 1
- [12] Yifan Gao, Zihang Lin, Chuanbin Liu, Min Zhou, Tiezheng Ge, Bo Zheng, and Hongtao Xie. Postermaker: Towards high-quality product poster generation with accurate text rendering. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8083–8093, 2025. 1, 2, 3

- [13] Lan-Zhe Guo, Zhi Zhou, Yu-Feng Li, and Zhi-Hua Zhou. Identifying useful learnwares for heterogeneous label spaces. In International Conference on Machine Learning, pages 12122–12131. PMLR, 2023. 20
- [14] Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V Chawla, Olaf Wiest, and Xiangliang Zhang. Large language model based multi-agents: A survey of progress and challenges. arXiv preprint arXiv:2402.01680, 2024. 2
- [15] Chunsan Hong, ByungHee Cha, and Tae-Hyun Oh. Cas: A probability-based approach for universal condition alignment score. In The Twelfth International Conference on Learning Representations, 2024. 17
- [16] HsiaoYuan Hsu and Yuxin Peng. Postero: Structuring layout trees to enable language models in generalized contentaware layout generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8117–8127,

2025. 2

- [17] Yue Hu and Xiaojun Wan. Ppsgen: Learning-based presentation slides generation for academic papers. IEEE transactions on knowledge and data engineering, 27(4):1085–1097,

2014. 1

- [18] Zhi Jin, Sheng Xu, Xiang Zhang, Tianze Ling, Nanqing Dong, Wanli Ouyang, Zhiqiang Gao, Cheng Chang, and Siqi Sun. Contranovo: A contrastive learning approach to enhance de novo peptide sequencing. In Proceedings of the AAAI conference on artificial intelligence, pages 144–152,

2024. 2

- [19] Kyudan Jung, Hojun Cho, Jooyeol Yun, Soyoung Yang, Jaehyeok Jang, and Jaegul Choo. Talk to your slides: Languagedriven agents for efficient slide editing. arXiv preprint arXiv:2505.11604, 2025. 1
- [20] Keshav Kumar and Ravindranath Chowdary. Slidespawn: An automatic slides generation system for research publications. arXiv preprint arXiv:2411.17719, 2024. 1
- [21] Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for” mind” exploration of large language model society. Advances in neural information processing systems, 36:51991– 52008, 2023. 2
- [22] Xin Liang, Xiang Zhang, Yiwei Xu, Siqi Sun, and Chenyu You. Slidegen: Collaborative multimodal agents for scientific slide generation. arXiv preprint arXiv:2512.04529,

2025. 2

- [23] Jieru Lin, Danqing Huang, Tiejun Zhao, Dechen Zhan, and Chin-Yew Lin. Designprobe: A graphic design benchmark for multimodal large language models. arXiv preprint arXiv:2404.14801, 2024. 2
- [24] Jing Lin, Yao Feng, Weiyang Liu, and Michael J Black. Chathuman: Chatting about 3d humans with tools. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8150–8161, 2025. 21
- [25] Puyuan Liu, Xiang Zhang, and Lili Mou. A characterlevel length-control algorithm for non-autoregressive sentence summarization. Advances in Neural Information Processing Systems, 35:29101–29112, 2022. 1
- [26] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan

- Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang. Agentbench: Evaluating LLMs as agents. In The Twelfth International Conference on Learning Representations, 2024. 2
- [27] Ishani Mondal, S Shwetha, Anandhavelu Natarajan, Aparna Garimella, Sambaran Bandyopadhyay, and Jordan BoydGraber. Presentations by the humans and for the humans: Harnessing llms for generating persona-aware slides from documents. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2664–2684, 2024. 1
- [28] Seungeun Oh, Jihong Park, Sihun Baek, Hyelin Nam, Praneeth Vepakomma, Ramesh Raskar, Mehdi Bennis, and Seong-Lyun Kim. Differentially private cutmix for split learning with vision transformer. In First Workshop on Interpolation Regularizers and Beyond at NeurIPS 2022, 2022. 16
- [29] Randy Olson. Narrative is everything: The ABT framework and narrative evolution. Prairie Starfish Productions, 2019. 3
- [30] Jiazhen Pan, Bailiang Jian, Paul Hager, Yundi Zhang, Che Liu, Friedrike Jungmann, Hongwei Bran Li, Chenyu You, Junde Wu, Jiayuan Zhu, et al. Beyond benchmarks: Dynamic, automatic and systematic red-teaming agents for trustworthy medical language models. arXiv preprint arXiv:2508.00923, 2025. 2
- [31] Zizheng Pan, Jing Liu, Haoyu He, Jianfei Cai, and Bohan Zhuang. Stitched vits are flexible vision backbones. In European Conference on Computer Vision, pages 258–274. Springer, 2024. 22
- [32] Wei Pang, Kevin Qinghong Lin, Xiangru Jian, Xi He, and Philip Torr. Paper2poster: Towards multimodal poster automation from scientific papers. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025. 1, 2, 5, 6, 7, 3
- [33] Vik Paruchuri. Marker: Convert pdf to markdown and json quickly with high accuracy, 2025. 3
- [34] Sohan Patnaik, Rishabh Jain, Balaji Krishnamurthy, and Mausoom Sarkar. Aesthetiq: Enhancing graphic layout design via aesthetic-aware preference alignment of multimodal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 23701–23711, 2025. 2
- [35] Yu-Ting Qiang, Yan-Wei Fu, Xiao Yu, Yan-Wen Guo, ZhiHua Zhou, and Leonid Sigal. Learning to generate posters of scientific papers by probabilistic graphical models. Journal of Computer Science and Technology, 34(1):155–169, 2019. 1, 2
- [36] Stephane Rivaud, Louis Fournier, Thomas Pumir, Eugene Belilovsky, Michael Eickenberg, and Edouard Oyallon. PETRA: Parallel end-to-end training with reversible architectures. In The Thirteenth International Conference on Learning Representations, 2025. 23
- [37] Rohit Saxena, Pasquale Minervini, and Frank Keller. Postersum: A multimodal benchmark for scientific poster summa-

- rization. In Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, pages 1828–1844, 2025. 1, 2
- [38] Jaejung Seol, Seojun Kim, and Jaejun Yoo. Posterllama: Bridging design ability of language model to content-aware layout generation. In European Conference on Computer Vision, pages 451–468. Springer, 2024. 2
- [39] Jingwei Shi, Zeyu Zhang, Biao Wu, Yanjie Liang, Meng Fang, Ling Chen, and Yang Zhao. Presentagent: Multimodal agent for presentation video generation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 760–773,

2025. 1

- [40] Weixi Song, Zuchao Li, Lefei Zhang, hai zhao, and Bo Du. Sparse is enough in fine-tuning pre-trained large language models. In Forty-first International Conference on Machine Learning, 2024. 20
- [41] M Sravanthi, C Ravindranath Chowdary, and P Sreenivasa Kumar. Slidesgen: Automatic generation of presentation slides for a technical paper using summarization. In FLAIRS,

2009. 1

- [42] Li Sun, Liu He, Shuyue Jia, Yangfan He, and Chenyu You. Docagent: An agentic framework for multi-modal longcontext document understanding. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 17712–17727, 2025. 2
- [43] Shanlin Sun, Jiaqi Xu, Gabriel de Araujo, Shenghan Zhou, Hanwen Zhang, Ziheng Huang, Chenyu You, and Xiaohui Xie. Coma: Compositional human motion generation with multi-modal agents. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 9206–9214, 2026. 2
- [44] Tao Sun, Enhao Pan, Zhengkai Yang, Kaixin Sui, Jiajun Shi, Xianfu Cheng, Tongliang Li, Ge Zhang, Wenhao Huang, Jian Yang, and Zhoujun Li. P2p: Automated paper-to-poster generation and fine-grained benchmark. In The Fourteenth International Conference on Learning Representations, 2026. 1, 2, 5, 6
- [45] Yuchen Sun, Shanhui Zhao, Tao Yu, Hao Wen, Samith Va, Mengwei Xu, Yuanchun Li, and Chongyang Zhang. Guixplore: Empowering generalizable gui agents with one exploration. In Proceedings of the computer vision and pattern recognition conference, pages 19477–19486, 2025. 21
- [46] Siddharth Tourani, Ahmed Alwheibi, Arif Mahmood, and Muhammad Haris Khan. Pose-guided self-training with twostage clustering for unsupervised landmark discovery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23041–23051, 2024. 19
- [47] Hao Wang, Shohei Tanaka, and Yoshitaka Ushiku. Scipostlayout: A dataset for layout analysis and layout generation of scientific posters. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8136–8141, 2024. 1, 2
- [48] Linhan Wang, Kai Cheng, Shuo Lei, Shengkun Wang, Wei Yin, Chenyang Lei, Xiaoxiao Long, and Chang-Tien Lu. Dcgaussian: Improving 3d gaussian splatting for reflective dash cam videos. Advances in Neural Information Processing Systems, 37:99898–99920, 2024. 19

- [49] Naigang Wang, Chi-Chun Charlie Liu, Swagath Venkataramani, Sanchari Sen, Chia-Yu Chen, Kaoutar El Maghraoui, Vijayalakshmi Viji Srinivasan, and Leland Chang. Deep compression of pre-trained transformer models. Advances in Neural Information Processing Systems, 35:14140–14154,

2022. 18

- [50] Jiaqi Wei, Yuejin Yang, Xiang Zhang, Yuhan Chen, Xiang Zhuang, Zhangyang Gao, Dongzhan Zhou, Guangshuai Wang, Zhiqiang Gao, Juntai Cao, et al. From ai for science to agentic science: A survey on autonomous scientific discovery. arXiv preprint arXiv:2508.14111, 2025. 2
- [51] Jiaqi Wei, Hao Zhou, Xiang Zhang, Di Zhang, Zijie Qiu, Noah Wei, Jinzhe Li, Wanli Ouyang, and Siqi Sun. Retrieval is not enough: Enhancing RAG through test-time critique and optimization. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 2
- [52] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. Autogen: Enabling next-gen llm applications via multi-agent conversations. In First conference on language modeling, 2024. 2
- [53] You Wu, Xucheng Wang, Xiangyang Yang, Mengyuan Liu, Dan Zeng, Hengzhou Ye, and Shuiwang Li. Learning occlusion-robust vision transformers for real-time uav tracking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17103–17113,

2025. 22

- [54] Fei Xiong, Xiang Zhang, Aosong Feng, Siqi Sun, and Chenyu You. Quantagent: Price-driven multi-agent llms for high-frequency trading. arXiv preprint arXiv:2509.09995,

2025. 2

- [55] Sheng Xu and Xiaojun Wan. Posterbot: A system for generating posters of scientific papers with neural models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 13233–13235, 2022. 1, 2
- [56] Xiangyuan Xue, Zeyu Lu, Di Huang, Zidong Wang, Wanli Ouyang, and Lei Bai. Comfybench: Benchmarking llmbased agents in comfyui for autonomously designing collaborative ai systems. In Proceedings of the computer vision and pattern recognition conference, pages 24614–24624, 2025. 7
- [57] Yuwei Yin, Jean Kaddour, Xiang Zhang, Yixin Nie, Zhenguang Liu, Lingpeng Kong, and Qi Liu. Ttida: controllable generative data augmentation via text-to-text and textto-image models. arXiv preprint arXiv:2304.08821, 2023. 2
- [58] Chenyu You, Yifei Mint, Weicheng Dai, Jasjeet S Sekhon, Lawrence Staib, and James S Duncan. Calibrating multimodal representations: A pursuit of group robustness without annotations. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26140–26150, 2024.
- [59] Chenyu You, Haocheng Dai, Yifei Min, Jasjeet S Sekhon, Sarang Joshi, and James S Duncan. Uncovering memorization effect in the presence of spurious correlations. Nature Communications, 16(1):5424, 2025. 2
- [60] Hui Zhang, Dexiang Hong, Maoke Yang, Yutao Cheng, Zhao Zhang, Jie Shao, Xinglong Wu, Zuxuan Wu, and YuGang Jiang. Creatidesign: A unified multi-conditional diffu-

- sion transformer for creative graphic design. arXiv preprint arXiv:2505.19114, 2025. 2
- [61] Xiang Zhang, Muhammad Abdul-Mageed, and Laks VS Lakshmanan. Autoregressive+ chain of thought= recurrent: Recurrence’s role in language models’ computability and a revisit of recurrent transformer. arXiv preprint arXiv:2409.09239, 2024. 2
- [62] Xiang Zhang, Senyu Li, Ning Shi, Bradley Hauer, Zijun Wu, Grzegorz Kondrak, Muhammad Abdul-Mageed, and Laks VS Lakshmanan. Cross-modal consistency in multimodal large language models. arXiv preprint arXiv:2411.09273, 2024. 2
- [63] Xiang Zhang, Juntai Cao, Jiaqi Wei, Yiwei Xu, and Chenyu You. Tokenization constraints in llms: A study of symbolic and arithmetic reasoning limits. arXiv preprint arXiv:2505.14178, 2025. 1
- [64] Xiang Zhang, Juntai Cao, Chenyu You, and Dujian Ding. Why prompt design matters and works: A complexity analysis of prompt search space in llms. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 32525–32555,

2025. 2

- [65] Yizi Zhang, Yanchen Wang, Mehdi Azabou, Alexandre Andre, Zixuan Wang, Hanrui Lyu, International Brain Laboratory, Eva L Dyer, Liam Paninski, and Cole Lincoln Hurwitz. Neural encoding and decoding at scale. In Forty-second International Conference on Machine Learning, 2025. 7
- [66] Haokun Zhao, Xiang Zhang, Jiaqi Wei, Yiwei Xu, Yuting He, Siqi Sun, and Chenyu You. Timeseriesscientist: A general-purpose ai agent for time series analysis. arXiv preprint arXiv:2510.01538, 2025. 2
- [67] Hao Zheng, Xinyan Guan, Hao Kong, Wenkai Zhang, Jia Zheng, Weixiang Zhou, Hongyu Lin, Yaojie Lu, Xianpei Han, and Le Sun. Pptagent: Generating and evaluating presentations beyond text-to-slides. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 14413–14429, 2025. 1
- [68] Hengyu Zhou, Hui Zhang, and Bin Wang. Segmentationguided layer-wise image vectorization with gradient fills. In European Conference on Computer Vision, pages 165–180. Springer, 2024. 7

## PosterGen: Aesthetic-Aware Multi-Modal Paper-to-Poster Generation via Multi-Agent LLMs

### Supplementary Material

#### Table of Contents

- A. Extended Related Work 1
- B. Implementation Details 1

- B.1. Agent Specifications . . . . . . . . . . . . . 1
- B.2. Experimental Environment . . . . . . . . . . 1
- B.3. Configuration Parameters . . . . . . . . . . 2

- C. Prompts 2

- C.1. Baseline Prompt . . . . . . . . . . . . . . . 2
- C.2. PosterGen Prompts . . . . . . . . . . . . . . 2
- C.3. VLM-as-Judge Evaluation Prompt Template 2

- D. Human Evaluation Details 2
- E. Additional Qualitative Results 2
- F. Limitations 4
- G. Broader Impacts 4

#### A. Extended Related Work

Slide Generation. A similar task to poster generation is the automatic generation of presentation slides from documents [11, 17, 19, 20, 27, 39, 41, 63, 67]. Some works develop agents for general purposes, such as efficient slide editing [19] or narrated presentation videos [39]. Other methods [67] focus on holistically improving the content, design, and coherence of the slides. Among these, several works specifically aim to generate slides for academic presentations. Early approaches like PPSGen [17], SlidesGen [41], and SlideSpawn [20] utilized summarization and information extraction techniques to generate draft slides. More recent approaches utilize end-to-end systems [11] or persona-aware models [27] to generate more tailored slides. However, academic poster design is far more challenging than slide generation, as slides can distribute content across multiple pages, and work together with a presenter’s oral explanation to convey the full message of the paper, while an academic poster needs to contain all necessary information from a paper onto a single page, and ought to be visually appealing to attract the attention and help initiate a dialogue [10] with the authors.

Table 7. Agent specifications in PosterGen.

Agent/Module Type Model Role Parser Hybrid GPT 4.1 Structure section content ëMarker Tool N/A Extract content from PDF Curator LLM GPT 4.1 Create spatial content plan and storyboard Color LLM GPT 4.1 Generate color palette Layout Hybrid N/A Control precise element coords (CSS-style) ëBalancer LLM GPT 4.1 Optimize column utilization Font LLM GPT 4.1 Apply typography and keyword highlighting Renderer Proced. N/A Generate final PPTX and PNG files

Table 8. Key hyperparameters for PosterGen Implementations. The default LLM/VLM model is gpt-4.1-2025-04-14. The “Content Constraints” are controlled via instructions within the agent prompts. The symbol ‘#’ denotes “number of.”

Category Parameter Value

Model GPT-4.1 ë Alternatives GPT-4o

LLM/VLM Configuration

GPT-4.1 mini Claude-Sonnet-4 Temperature 0.7

# Sections r5,8s # Visual Assets r4,6s Max Words per Section 1000

Content Constraints

Height Precision (ϵ) 0.001 inches Newline Offset Ratio 1.0

Text Height Estimation

#### B. Implementation Details

###### B.1. Agent Specifications

Table 7 details the type and role of each agent in our pipeline. Most agents (CuratorAgent, ColorAgent, FontAgent) operate via a single LLM call, while ParserAgent and LayoutAgent adopt a hybrid design that runs mostly procedural logic and only calls the LLM for specific subtasks. Renderer is entirely procedural and requires no LLM call.

###### B.2. Experimental Environment

The framework is compatible with Windows, Linux and macOS operating systems, and is implemented in Python

###### 3.11. While a GPU is not strictly required, it is strongly recommended to accelerate document parsing and optical character recognition (OCR) tasks handled by the marker-pdf tool. The framework relies on several key libraries; the most critical dependencies for reproducing our work include:

- • python-pptx “ 1.0.2 for PPT generation.
- • langchain “ 0.3.25 and langgraph “ 0.4.8 for building the multi-agent workflow.

- • marker-pdf “ 1.8.2 for PDF-to-Markdown conversion and parsing. Notably, we slightly modify the official marker-pdf source code to extract tables as images rather than Markdown text.

- • Pillow “ 10.4.0 for image manipulation.

- B.3. Configuration Parameters

To ensure the reproducibility of our implementation and experiments, we list the most critical parameters that directly influence the behavior of the agents and the quality of generated posters in Table 8. Aesthetic parameters related to layout, color and typography (font sizes, margins, color values) are defined in the configuration files within our source code but omitted here for clarity.

- C. Prompts

- C.1. Baseline Prompt

We present the prompt of GPT-4o Image Generation via ChatGPT web interface (as illustrated in Figure 8), which is alongside the input paper file.

- C.2. PosterGen Prompts

We present the detailed prompt design [64] used in our PosterGen multi-agent workflow as follows.

ParserAgent. This includes: (1) Title and Authors Extraction (Figure 9); (2) Narrative (ABT) Extraction (Figure 10); (3) Visual Asset Classification (Figure 11); and (4) Structured Section Extraction (Figure 12).

CuratorAgent. CuratorAgent generates an effective storyboard through strategic content planning and by applying visual height constraints. Due to space limitations, we split the prompt into three parts: (1) Input and Design Patterns (Figure 13), (2) visual asset selection and content organization (Figure 14), and (3) output format (Figure 15). We also omit less important parts and replace them with ellipsis mark (...). The full prompt is available in the source code.

Layout Balancer. This is a sub-agent of LayoutAgent designed to improve column utilization and prevent overflows. Its prompt is detailed in Figure 16 and Figure 17.

ColorAgent. We present only the prompt for extracting the theme color from an affiliation logo using a VLM (as shown in Figure 18); the fallback method, which uses a key visual asset, is omitted for clarity.

FontAgent. FontAgent calls LLM once to extract and classify different keywords. The detailed prompt is illustrated in Figure 19.

- C.3. VLM-as-Judge Evaluation Prompt Template

We present the prompt template used for our VLM-as-Judge evaluation, as shown in Figure 20. This standardized template is applied to every evaluation focus area and uses a 5point scale. To counteract the tendency of Vision-Language

Models (VLMs) to provide overly generous scores, our prompt design incorporates a targeted few-shot example strategy. For high scores (4 and 5), we provide positive examples of desired qualities, while for low-to-mid scores (1, 2, and 3), we provide negative examples of common flaws. This approach is designed to calibrate the VLM’s judgment and yield more accurate, evidence-based scoring.

#### D. Human Evaluation Details

We invite 30 participants to take part in our user study using the Qualtrics platform. To gain objective evaluation results, we survey each user for information regarding their field of study and experience in making academic posters, as shown in Figure 22. This verifies that our group of human evaluators possesses adequate research background to evaluate academic posters properly.

To alleviate evaluator fatigue and cognitive load, we did not ask each participant to assess all of our benchmark. Instead, each participant only needs to evaluate aesthetic scores for a random subset of 10 posters. Each set consists of three variations of posters: generated by our approach PosterGen, the baseline PosterAgent [32], and designed by authors of the paper. A rigorous randomization procedure is adopted to reduce possible bias: (i) each of the 10 sets is randomly ordered for evaluation, and (ii) for each set, the order of the three posters is also randomized. This makes sure that the evaluation is completely blind to all of our approaches for its validity to be maximized. An example set of questions is shown in Figure 21.

In addition, we did not include the GPT-4o-Image approach in our human study. As discussed in the qualitative analysis in Section 5.5 of the main paper, diffusion-based generation via GPT-4o-Image constantly suffers from gibberish text, distorted text blocks, and the hallucination of visual assets. It is thus pointless to consider it for comparison with the multi-agent approach and human designs.

#### E. Additional Qualitative Results

In this section, we provide 15 additional representative qualitative results drawn from our 30-paper benchmark, as shown in Figures 24 to 38. All the 30 papers were selected from top-tier AI conferences, e.g., NeurIPS, ICML, ICLR, CVPR and ECCV in the last four years, with the distribution shown in Figure 23. Our selection criteria require that both the full paper and a corresponding high-quality humanmade poster are publicly available.

As illustrated in Figures 24 to 38, these supplementary qualitative results further validate the observations presented within the main paper, regarding the limitations of baseline approaches for aesthetic design. The end-toend GPT-4o-Image method is prone to basic layout constraints, manifesting serious boundary problems like too

GPT-4o Image Generation You are an expert specializing in designing automated academic poster. Primary Task: Analyze the provided research paper and autonomously design and generate a complete, professional

academic poster. Key Guidelines:

- • Fixed Dimensions: The final poster layout must be exactly width pixels wide and height pixels high.
- • Content Fidelity: All content (text, figures, tables) must be extracted or summarized exclusively from the source paper. Do not invent or infer information.
- • Visual Design: The poster must be visually appealing. Apply a clean and professional theme consistently across all elements.
- • Layout Design: The layout must be well-balanced, effectively utilizing the available space. Actively avoid element overflow and large, underutilized blank areas.

###### Required Structure & Content:

- • Header: Must include the full paper title and a complete list of all authors.
- • Body: The main content area must be organized into distinct sections. These sections should feature a wellbalanced and carefully arranged composition of:

- – Concise text summaries.
- – high-quality figures from paper pdf.
- – Key data tables from paper pdf.

Output Format: The final output must be a single PNG image file with dimensions of width ˆ height pixels.

Figure 8. Prompt for GPT-4o Image Generation.

ParserAgent (1): Title and Authors Extraction

You are an expert academic paper parser. Your task is to extract the title and authors from the provided academic paper text. Please extract:

- 1. Title: The main title of the paper
- 2. Authors: All author names using initials (no affiliations, emails, or other metadata) Strict Formatting Requirements:

- • Title: Use proper title case where each word has only the first letter capitalized, EXCEPT for established acronyms, technical terms, or proper nouns that are conventionally written in all uppercase letters (such as abbreviations for organizations, technologies, or methodologies). Keep such terms in their original case. Example: “A Study of Machine Learning Methods” but preserve acronyms like “Using LLM for Data Analysis” or “CNN Architecture Design”.
- • Authors: Use initials for authors’ names. Convert full names to initials format, preserving middle initials when present. Examples: “Kevin W. Jones” Ñ ”K.W. Jones”, “Yann LeCun” Ñ ”Y. LeCun”, ”Mary Smith Johnson” Ñ ”M.S. Johnson”. Separate multiple authors with “, ”. Remove all affiliations, emails, institutions, departments, addresses, and other metadata.

###### Input text: {{ markdown document }} Required JSON structure:

- 1 {
- 2 "title": "Title With Proper Case Formatting",
- 3 "authors": "F. Author, S. Author, T. Author"
- 4 }

Figure 9. Prompt for ParserAgent to extract the title and authors from the source paper.

much whitespace and cut-off vertical content, thus verifying that it is unable to have robust canvas control. Though PosterAgent [32] enhances content fidelity over diffusion-

based methods, it consistently shows aesthetic inadequacies like improper alignment of elements, poor whitespace use, and a monotonous visual aesthetic that lacks typographi-

ParserAgent (2): Narrative (ABT) Extraction Extract ABT narrative structure optimized for poster presentation: Input: {{ Academic paper markdown text }} Output: JSON with poster-ready ABT structure Guidelines:

- • Each section (and/but/therefore) should be 1-2 concise sentences
- • Focus on visual impact and poster audience understanding
- • Emphasize key contributions and results
- • Avoid technical jargon where possible Required JSON structure:

- 1 {
- 2 "and": "Current knowledge and established facts (background context)",
- 3 "but": "Specific problem, gap, or challenge identified",
- 4 "therefore": "Your solution, contribution, and key findings",
- 5 "poster_hook": "One compelling sentence that grabs attention",
- 6 "key_impact": "Why this research matters (practical implications)"
- 7 } Paper content: markdowndocument

Figure 10. Prompt for ParserAgent to extract ABT-structured narratives.

cal hierarchy to create visual flow. In contrast, PosterGen consistently generates high-quality artistic posters showcasing pleasant colors, effective visual hierarchy generation, and natural reading paths. Therefore, it demonstrates that PosterGen is capable of including effective aesthetic concepts within its agent workflow.

#### F. Limitations

Although PosterGen demonstrates significant improvement over existing methods for generating aesthetic-aware posters, we identify several limitations that present directions for future work:

- • Although the marker-pdf tool is very effective for most use cases for ParserAgent, occasionally it yields errors for extractions. Such artifacts may include lack of text information (for example, title or author names) or possibly for visual components at low resolutions. Areas for improvement may include development of a specialized document parser module designed for academic-style pdf files containing intricate formats.
- • CuratorAgent is now limited to using only the visual elements it finds in the source paper. It is not capable of creating generative visual elements based on paper content only. One promising direction to take this agent further is to empower its functionality to generate additional diagrams, such as creating a flowchart to intuitively summarize a text-heavy Introduction.
- • PosterGen is dependent on a three-column grid layout as its current paradigm. This acts as a solid and safe choice for landscape-oriented posters, but does come with its limitations in design flexibility. Going further, the ex-

ploration of LayoutAgent should include diverse ratios for various portrait-oriented positions, as well as dynamic and diverse layout designs.

#### G. Broader Impacts

In this paper, we are the first to explore the integration of design and aesthetic principles into a multi-agent framework for academic poster generation, and propose a novel workflow PosterGen. By mirroring the specialized workflow of professional designers, PosterGen achieves content fidelity that rivals the state-of-the-art while significantly enhancing the final poster’s visual design and aesthetic quality. Though it may not fully eliminate the need for human finetuning, its core contribution lies in systematically embedding design principles into multi-agent design, a step often overlooked even in manual creation.

We hope our work will advance this emerging yet meaningful field of automated scientific communication, and firmly believe that PosterGen can substantially relieve researchers of the time and effort required for poster creation, allowing them to focus more on the one-on-one scholarly dialogue that poster sessions are meant to facilitate.

- ParserAgent (3): Classify Visual Assets

Classify visual assets by column-aware poster placement and research role: Available visuals with captions: visuals list Classify each visual into exactly one category based on column-specific poster design: Column-Aware Categories:

- 1. key visual: Most important method visual representing core research innovation (max 1, middle column)
- 2. problem illustration: Visuals showing research problem, challenges, or motivation (left column introduction)
- 3. method workflow: Method architecture, system diagrams, algorithmic workflows (middle column method)
- 4. main results: Primary experimental results, performance tables, key findings (right column)
- 5. comparative results: Baseline comparisons, ablation studies, validation charts (right column)
- 6. supporting: Background concepts, supplementary analysis, minor details (flexible placement) Classification Guidelines:

- • Problem Context: Figures showing “what’s wrong” or “why this matters” Ñ problem illustration
- • Method Core: Most important technical diagram Ñ key visual
- • Method Details: Architecture/workflow diagrams Ñ method workflow
- • Primary Evidence: Main performance results Ñ main results
- • Validation Evidence: Comparisons with baselines Ñ comparative results
- • Background/Supplementary: Minor or supporting content Ñ supporting Consider:
- • Visual content and research narrative role
- • Optimal column placement for logical flow
- • Visual impact and audience comprehension Required JSON output:

- 1 {
- 2 "key_visual": "visual_id or null",
- 3 "problem_illustration": ["visual_id1", ...],
- 4 "method_workflow": ["visual_id1", ...],
- 5 "main_results": ["visual_id1", ...],
- 6 "comparative_results": ["visual_id1", ...],
- 7 "supporting": ["visual_id1", ...]
- 8 } Ensure every visual id appears exactly once across all categories.

Figure 11. Prompt for ParserAgent to classify extracted visual assets.

- ParserAgent (4): Structured Section Extraction

Extract structured sections from academic paper text for poster creation. Paper Text: {{ raw text }} Task: Extract all major sections from the paper and organize them with their content. Focus on sections that would be relevant for an academic poster. Section Extraction Guidelines:

###### 1. Identify Major Sections:

- • Introduction/Background
- • Related Work (if substantial)
- • Methodology/Approach
- • Experiments/Results
- • Discussion/Analysis

###### 2. Content Processing:

- • Extract the main content for each section
- • Keep section content under 1000 words
- • Preserve key technical details, formulas, and findings
- • Maintain important bullet points and lists
- • Remove excessive citations and references

###### 3. Section Classification:

- • foundation: Introduction, background, motivation, problem statement
- • method: Methodology, approach, algorithm, system design
- • evaluation: Experiments, results, analysis, validation

###### Required JSON structure:

- 1 {
- 2 "paper_sections": [
- 3 {
- 4 "section_name": "Introduction",
- 5 "section_type": "foundation",
- 6 "content": "Main content of the section (max 1000 words)",
- 7 "key_points": [ ... ],
- 8 "importance": "high|medium|low",
- 9 "contains_figures": ["figure_1", "figure_2"],
- 10 "contains_tables": ["table_1"]
- 11 }
- 12 ],
- 13 "paper_structure": {
- 14 "total_sections": 5,
- 15 "foundation_sections": 2,
- 16 "method_sections": 2,
- 17 "evaluation_sections": 1
- 18 }
- 19 } Critical Requirements:

- • Extract ALL major sections (don’t skip any)
- • Keep each section under 1000 words
- • Preserve technical accuracy
- • Identify which figures/tables belong to each section
- • Classify section importance for poster layout Generate structured sections that provide comprehensive paper coverage for poster creation.

Figure 12. Prompt for ParserAgent to extract structured sections.

CuratorAgent: Spatial Story Board Generation (Part 1 of 3)

You are an Expert Academic Poster Designer specializing in visual-dense poster layouts with strategic spatial organization. Mission: Transform research papers into spatially-organized poster sections that maximize visual asset utilization while following human design patterns. Prioritize visual impact over text density. Input:

- • Paper Structured Sections: {{ structured sections }}
- • Enhanced ABT Narrative: {{ narrative content }}
- • Classified Visuals: {{ classified visuals }}
- • Available Images: {{ available images }}
- • Available Tables: {{ available tables }}
- • Visual Heights Information: {{ visual heights info }}
- • Available Height Per Column: {{ available height per column }} Human Poster Design Patterns: Based on analysis of successful academic posters:

###### 1. Left Column Strategy - Foundation & Context:

- • Introduction/Background/Motivation (priority placement)
- • Problem definition and challenges
- • Related work and background context
- • Method overview or workflow diagrams

###### 2. Middle Column Strategy - Core Technical Content:

- • Primary methodology (highest priority content)
- • Technical details and algorithms
- • Theoretical analysis and key innovations
- • System architecture diagrams

###### 3. Right Column Strategy - Experiments & Results:

- • Experimental results (tables and performance charts)
- • Key findings and validation data
- • Performance comparisons and analysis

Figure 13. Part 1 of CuratorAgent prompt, focusing on the inputs, high-level instructions and human design patterns.

CuratorAgent: Spatial Story Board Generation (Part 2 of 3) Oversized Visual Exclusion:

- • Exclusion Rule: Any visual with height percentage ą 50% in visual heights info MUST BE EXCLUDED from poster
- • Reasoning: Even with 80% shrinking, these visuals would still exceed 40% column height
- • Smart Substitution: ...
- • FALLBACK RULE: If only ONE oversized visual (ą 50%) is selected, allow it to proceed. For multiple oversized visuals, only select the one with SMALLEST height percentage.

Visual Asset Strategic Selection Process

- 1. Key Visual Mandatory Placement:

- • Identify the “key visual” from classified visuals. This is the MOST important visual
- • Place key visual in middle column, top priority section
- • This anchors the entire poster layout around the core research contribution

- 2. Column-Based Visual Distribution:

- • Column 1 (Left) - Foundation & Context:

- – MINIMUM: 1 visual asset required
- – Purpose: Express core research problem or contradiction visually
- – Selection Priority: Choose visuals that illustrate problem context, background concepts, or prior work limitations
- – Maximum: 2 visual assets

- • Column 2 (Middle) - Methodology:

- – MANDATORY: Contains key visual from classified visuals
- – Additional: May include 1 supporting method diagram
- – Maximum: 2 visual assets

- • Column 3 (Right) - Results & Impact:

- – STRICT MAXIMUM: 2 visual assets ONLY
- – Selection Criteria: Choose the 2 most critical visuals that directly validate main claims
- – Priority Order: ...

- 3. Visual Distribution Enforcement: ...
- 4. Column Space Optimization Strategy: ... Core Task: Create 5-8 poster sections with BOTH content organization AND strategic spatial placement to achieve perfect space utilization across all three columns. DO NOT create any conclusion, takeaway, future work, or impact sections. Focus ONLY on problem, method, and results/experiments. Content Organization Guidelines:

- 1. Section Requirements:

- • Section titles: Maximum 4 words (e.g., “Our Method”, “Key Results”)
- • Text content: 2-3 concise entries using different rich hierarchical formatting (see examples below) based on section contents
- • Visual integration: Each visual assigned to exactly ONE section
- • Complete content: No ellipsis (...), write full bullet points

- 2. Rich Text Formatting Options:

- • A) Nested Bullet Structure:

- 1 "* Primary concept or finding",
- 2 " - Supporting detail or sub-point",
- 3 " - Additional supporting evidence"

- • Other formats like Bold Headers and Ordered Lists are also available.

Figure 14. Part 2 of CuratorAgent prompt, specifying the detailed rules for visual asset selection, content organization, and other planning requirements.

CuratorAgent: Spatial Story Board Generation (Part 3 of 3) Output Format:

- 1 {
- 2 "spatial_content_plan": {
- 3 "poster_strategy": {
- 4 "narrative_flow": "How the story progresses across columns",
- 5 "space_utilization_approach": "Strategy for filling all three columns",
- 6 "column_balance_rationale": "Why content is distributed this way"
- 7 },
- 8 "sections": [
- 9 {
- 10 "section_id": "unique_identifier",
- 11 "section_title": "Max 4 Words",
- 12 "column_assignment": "left|middle|right",
- 13 "vertical_priority": "top|middle|bottom",
- 14 "importance_level": 1,
- 15 "content_type": "foundation|method|results",
- 16 "expected_content_density": "high|medium|low",
- 17 "text_content": [
- 18 "* **Key Innovation:** Core contribution with bold emphasis",
- 19 " - Supporting technical detail",
- 20 "* **Impact:** Quantifiable result or benefit"
- 21 ],
- 22 "visual_assets": [
- 23 {
- 24 "visual_id": "figure_1",
- 25 "visual_purpose": "How this supports the section",
- 26 "placement_rationale": "Why this visual belongs in this spatial location"
- 27 }
- 28 ],
- 29 "spatial_rationale": "Why this section belongs in this column/position"
- 30 }
- 31 ]
- 32 },
- 33 "column_distribution": {
- 34 "left_column": {
- 35 "focus": "Foundation and context",
- 36 "assigned_sections": ["section_id_1", "section_id_2"],
- 37 "content_strategy": "Build problem understanding and motivation"
- 38 },
- 39 "middle_column": {
- 40 "focus": "Core methodology",
- 41 "assigned_sections": ["section_id_3", "section_id_4"],
- 42 "content_strategy": "Present technical innovation and approach"
- 43 },
- 44 "right_column": {
- 45 "focus": "Results and impact",
- 46 "assigned_sections": ["section_id_5", "section_id_6"],
- 47 "content_strategy": "Demonstrate effectiveness and validations"
- 48 }
- 49 }
- 50 }

Figure 15. Part 3 of CuratorAgent prompt, defining the exact JSON output format and data structure required from the agent.

Balancer Agent: Balance Column Space Utilization (Part 1 of 2)

You are an expert academic poster layout optimization specialist. Your goal is to achieve optimal three-column space utilization through conservative within-column content adjustments only. Current Column Status:

- • Column 1 (Left): {left utilization} utilization - {left status}
- • Column 2 (Middle): {middle utilization} utilization - {middle status}
- • Column 3 (Right): {right utilization} utilization - {right status}
- • Available Height per Column: {available height} inches Target Utilization: 85-95% for each column Core Optimization Principle: Prioritize content reduction over content expansion. Better to have 80% utilization than risk overflow beyond available space. Column Content Rules:

- 1. Left Column: Foundation & Context

- • Purpose: Introduction, background, prior work, problem setup, supporting context
- • Content Types: Motivation, challenges, related work, problem definitions, supporting materials
- • Reading Role: Sets up the research problem and provides necessary background

- 2. Middle Column: Core Methodology

- • Purpose: Method details, algorithms, implementation, technical innovation
- • Content Types: Core methods, algorithms, technical approach, key innovations
- • Reading Role: Presents the technical contribution and methodology
- • CRITICAL: Contains key visual (importance level=1). NEVER remove method sections

- 3. Right Column: Results & Impact

- • Purpose: Experiments, evaluation, findings, conclusions, future work
- • Content Types: Experimental results, performance analysis, conclusions, future directions
- • Reading Role: Demonstrates validation and impact of the proposed method

- Figure 16. Part 1 of the Balancer sub-agent prompt, outlining its role, the current column status, and the fundamental content rules for each column.

Balancer Agent: Balance Column Space Utilization (Part 2 of 2) Within-Column Optimization Strategies:

- 1. Strategy A: Conservative Text Content Adjustment (for 80-100% utilization)

- • When to use: Column utilization is close to optimal range (80-100%)
- • Actions allowed:

- – MINIMAL text expansion: Add only 1-2 short phrases to underutilized columns (75-85%)
- – Aggressive text reduction: Significantly shorten content in overflow columns (ą95%)
- – CONSERVATIVE APPROACH: Prefer slight underutilization over any risk of overflow

- • Text Length Limits:

- – Maximum per bullet: 25 words (count carefully)
- – Maximum sub-bullets: 2 per main bullet
- – Expansion limit: Add maximum 10-15 words total per section
- – Reduction target: Remove 30-50% of content from overflow sections

- 2. Strategy B: Section Management (for ă80% or ą100% utilization)

- • When to use: Column has severe underutilization (ă80%) or overflow (ą100%)
- • Actions allowed:

- – Add sections from structured sections: Use additional content from paper sections that fit the column’s purpose
- – Remove less important sections: Remove sections with importance level “ 3 or lower importance

- • Section Removal Priority:

- – NEVER remove: Method sections with key visual (importance level “ 1)
- – NEVER remove: Core experimental results or main findings
- – Remove first: Supporting context, minor experiments, supplementary details (importance level “ 3)
- – Remove second: Secondary analysis, additional background (importance level “ 2)

###### Strict Constraints:

- 1. NO CROSS-COLUMN MOVES: Never change column assignment for any existing section
- 2. PRESERVE READING FLOW: Maintain left→middle→right logical progression
- 3. SECTION ID PRESERVATION: Never change section id, section title, visual assets, or other identifying fields
- 4. IMPORTANCE RESPECT: Never remove critical sections (importance level=1 or core results)
- 5. TARGET UTILIZATION: Achieve 85-95% utilization for each column Input: {{structured sections}, {current story board}, {column analysis}} Output Format: Output the complete optimized story board JSON. Each section’s ‘text content’ must be an array of complete strings only:

- 1 "text_content": [
- 2 "* **Point Title:** Complete description text here",
- 3 " - Supporting detail in complete sentences",
- 4 "* **Another Point:** Full explanation without truncation"
- 5 ] Preserve all original structure and field names. Only modify content within string values.

- Figure 17. Part 2 of the Balancer sub-agent prompt, detailing the specific optimization strategies, strict constraints, and the required input/output format.

ColorAgent: Theme Color Extraction Extract a sophisticated theme color from an affiliation logo that will work well as a poster accent color. Core Task: Analyze the provided affiliation logo and identify the most prominent, meaningful color that can serve as a poster theme color. This color should be:

- • Representative of the organization’s visual identity
- • Suitable for poster design applications (text highlights, accents)
- • Professional and readable when used on white backgrounds
- • Harmonious for academic poster contexts Color Extraction Guidelines:

###### 1. Primary Color Identification:

- • Look for the main brand color of the organization
- • Ignore pure white, black, and very light grays (background/outline colors)
- • Focus on colored elements that define the logo’s visual identity
- • Consider text colors, graphic elements, symbols, and emblematic elements

###### 2. Color Suitability Assessment:

- • Too Bright: If the main color is very bright/saturated (e.g., neon yellow #FFFF00), generate a more subdued version
- • Appropriate Saturation: Aim for colors that are vibrant but professional
- • Readability: Ensure the color provides sufficient contrast on white backgrounds for text

###### 3. Color Adjustment Rules:

- • If original color is too bright (lightness ą 85% or saturation ą 90%), reduce brightness by 15-25%
- • If original color is too dark (lightness ą 25%), lighten slightly for better visibility
- • Maintain the color’s hue character while optimizing for poster applications

Output Requirements: Return ONLY a JSON object with the following structure:

- 1 {
- 2 "extracted_color": "#1E3A8A",
- 3 "color_name": "Professional Navy Blue",
- 4 "adjustment_made": "reduced_brightness | lightened | none",
- 5 "original_color": "#0000FF",
- 6 "suitability_score": 8.5,
- 7 "reasoning": "Extracted the primary blue from the university emblem. Reduced brightness from bright blue to professional navy to ensure readability and sophisticated appearance on white backgrounds.",
- 8 "usage_notes": "Excellent for text highlights, section headers, and accent elements. Provides strong contrast while maintaining professional appearance."
- 9 } Scoring Criteria (1-10 scale):

- • Contrast/Readability: How well it works on white background
- • Professional Appearance: Appropriate for academic/research contexts
- • Brand Representation: How well it represents the organization
- • Poster Suitability: Effectiveness for highlights and accents

Figure 18. Prompt for ColorAgent to extract theme color from affiliation logo.

FontAgent: Keyword Extraction Analyze poster content and identify keywords for strategic visual highlighting using three distinct formatting styles. Input Data:

- • Enhanced Narrative: enhanced narrative
- • Curated Content: curated content Core Task: For each section, identify keywords and assign them to specific highlighting styles based on their semantic importance and role in the research narrative. Highlighting Style Categories:

###### 1. BOLD + CONTRAST COLOR:

- • Purpose: Core method/methodology names that represent the paper’s unique contribution
- • Criteria: Novel algorithms, architectures, or techniques introduced by this work; the main methodological innovation that defines the paper; must be unique to this research (not generic terms)
- • Limit: Maximum 2 per section, prefer 1 if it captures the main contribution

###### 2. BOLD:

- • Purpose: Important quantitative results and core technical terms within each section
- • Criteria: Performance metrics and numerical results (e.g., “95% accuracy”, “5.2ˆ speedup”); key technical concepts central to understanding the section; architecture names, dataset names, established method names; word-level emphasis, not entire phrases
- • Limit: Maximum 3 per section

###### 3. ITALIC:

- • Purpose: Defining terms, single-word emphasis, and foreign terminology
- • Criteria: Technical terms being defined or introduced for the first time; single-word emphasis (e.g., “This was the only experiment”); foreign words, Latin terms, or specialized vocabulary; word-level application only, never entire sentences
- • Limit: Maximum 2 per section

###### Output Format:

- 1 {
- 2 "section_keywords": {
- 3 "motivation": {
- 4 "bold_contrast": ["DP-CutMixSL"],
- 5 "bold": ["Vision Transformers", "privacy leakage"],
- 6 "italic": ["federated"]
- 7 },
- 8 "method": {
- 9 "bold_contrast": ["CutMix", "differential privacy"],
- 10 "bold": ["95% accuracy", "ResNet-50"],
- 11 "italic": ["only"]
- 12 },
- 13 "results": {
- 14 "bold_contrast": ["TransformerNet"],
- 15 "bold": ["top-1 accuracy", "5.2x speedup", "CIFAR-10"],
- 16 "italic": ["a priori"]
- 17 }
- 18 },
- 19 "formatting_summary": {
- 20 "total_bold_contrast": 4,
- 21 "total_bold": 7,
- 22 "total_italic": 3,
- 23 ...
- 24 }
- 25 } Return a JSON object with the exact schema above that maximizes research impact through strategic visual emphasis.

Figure 19. Prompt for FontAgent to extract different types of keywords.

VLM-as-Judge Evaluation Template

You are an expert academic poster design critic. Your evaluation must be strict, detailed, and evidence-based. For each dimension, assess the poster against the specific examples provided in the 5-point scale. High scores require adherence to professional design principles (positive examples). Low scores should be assigned when common design failures (negative examples) are present. A poster that is merely functional but exhibits poor design choices must be scored significantly lower on design metrics than one that is both functional and visually excellent. Please carefully examine the provided poster image and evaluate it across the following metrics. Provide a detailed explanation and score on a 5-point scale. FOCUS AREA: ă Focus Area ą Metric : ă Metric Description ą

- • Score 5 (Excellent):

- – Descriptor: ...
- – Positive Examples: ...

- • Score 4 (Good):

- – Descriptor: ...
- – Positive Examples: ...

- • Score 3 (Acceptable):

- – Descriptor: ...
- – Negative Examples: ...

- • Score 2 (Poor):

- – Descriptor: ...
- – Negative Examples: ...

- • Score 1 (Failed):

- – Descriptor: ...
- – Negative Examples: ...

Output Format: Please provide your evaluation as a JSON array with exactly 1 object for the metric:

- 1 [
- 2 {
- 3 "metric": "Core Graphic Principles",
- 4 "explanation": "Detailed analysis of the application of repetition, alignment, contrast, and proximity principles...",
- 5 "score": 4
- 6 }
- 7 ] Evaluation Instructions: ... Please evaluate the poster step by step according to these criteria.

Figure 20. Prompt Template for VLM-as-Judge Evaluation.

[Figure 77]

Figure 21. A sample set of questions from Qualtrics survey for human evaluation.

[Figure 78]

Figure 22. Participant demographics regarding field of study and prior poster creation experience.

[Figure 79]

Figure 23. Data distribution of our evaluation benchmark from top-tier AI conferences.

[Figure 80]

- Figure 24. Qualitative results of Oh et al. [28].

[Figure 81]

###### Figure 25. Qualitative results of Hong et al. [15].

[Figure 82]

###### Figure 26. Qualitative results of Cheng et al. [7].

[Figure 83]

###### Figure 27. Qualitative results of Wang et al. [49].

[Figure 84]

###### Figure 28. Qualitative results of Du et al. [8].

[Figure 85]

###### Figure 29. Qualitative results of Wang et al. [48].

[Figure 86]

###### Figure 30. Qualitative results of Tourani et al. [46].

[Figure 87]

###### Figure 31. Qualitative results of Song et al. [40].

[Figure 88]

###### Figure 32. Qualitative results of Guo et al. [13].

[Figure 89]

###### Figure 33. Qualitative results of Lin et al. [24].

[Figure 90]

###### Figure 34. Qualitative results of Sun et al. [45].

[Figure 91]

###### Figure 35. Qualitative results of Wu et al. [53].

[Figure 92]

###### Figure 36. Qualitative results of Pan et al. [31].

[Figure 93]

###### Figure 37. Qualitative results of Du et al. [9].

[Figure 94]

###### Figure 38. Qualitative results of Rivaud et al. [36].

