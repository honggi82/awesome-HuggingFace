### SketchAgent: Language-Driven Sequential Sketch Generation

Yael Vinker1 Tamar Rott Shaham1 Kristine Zheng2 Alex Zhao1 Judith E Fan2 Antonio Torralba1

2Stanford University

1MIT

{jefan,kxzheng}@stanford.edu https://sketch-agent.csail.mit.edu/

{yaelvink,tamarott,alexzhao,torralba}@mit.edu

# arXiv:2411.17673v1[cs.CV]26Nov2024

“Butteﬂy” “Taj Mahal”

“Circuit Diagram” “Neural Network”

Figure 1. SketchAgent leverages an off-the-shelf multimodal LLM to facilitate language-driven, sequential sketch generation through an intuitive sketching language. It can sketch diverse concepts, engage in interactive sketching with humans, and edit content via chat.

#### Abstract

Sketching serves as a versatile tool for externalizing ideas, enabling rapid exploration and visual communication that spans various disciplines. While artificial systems have driven substantial advances in content creation and human-computer interaction, capturing the dynamic and abstract nature of human sketching remains challenging. In this work, we introduce SketchAgent, a language-driven, sequential sketch generation method that enables users to create, modify, and refine sketches through dynamic, conversational interactions. Our approach requires no training or fine-tuning. Instead, we leverage the sequential nature and rich prior knowledge of off-the-shelf multimodal large language models (LLMs). We present an intuitive sketching language, introduced to the model through in-context examples, enabling it to “draw” using string-based actions. These are processed into vector graphics and then rendered to create a sketch on a pixel canvas, which can be accessed again for further tasks. By drawing stroke by stroke, our agent captures the evolving, dynamic qualities intrinsic to sketching. We demonstrate that SketchAgent can generate sketches from diverse prompts, engage in dialogue-driven drawing, and collaborate meaningfully with human users.

#### 1. Introduction

Sketching is a powerful tool for distilling ideas into their simplest form. Its fluid and spontaneous nature makes sketching a uniquely versatile tool for visualization, rapid ideation, and communication across cultures, generations, and disciplines [31, 113]. For example, designers use sketches to explore new ideas [44, 114], scientists employ them to formulate problems [55, 82], and children engage in sketching to learn and express themselves [32, 33] (see Fig. 2). Artificial systems, in principle, have the potential to support and enhance human creativity, problem-solving, and visual expression through sketching, adapting flexibly to their exploratory nature [26, 109, 133].

Traditionally, sketch generation methods rely on humandrawn datasets to train generative models [6, 8, 20, 41, 48, 67]. However, fully capturing the diversity of sketches within datasets remains challenging [31], limiting these methods in both scale and diversity. Recent advancements in vision-language models, such as CLIP [88] and text-to-image diffusion [92], have enabled sketch generation methods that reduce reliance on human-drawn datasets [34, 53, 116]. These methods leverage pretrained model guidance and differentiable rendering [66] to optimize parametric curves, creating sketches that go beyond predefined styles and categories.

While representing a significant step toward a generalpurpose sketching system, these methods lack a crucial aspect of human drawing: the process itself. Current methods, though versatile, optimize all strokes simultaneously, making the intermediate sketching steps meaningless. As a result, the sketch cannot be decomposed into a coherent sequence of strokes that reflects the drawing process. In contrast, humans draw iteratively, stroke by stroke, incorporating visual feedback and continuously adapting—a dynamic, evolving process that fosters creativity, ideation, and communication [60, 98, 112].

- (A) Frank Gehry. Elevation Sketches. Guggenheim Museum
- (B) Alexander Bell’s drawing of the telephone

- (C) Children drawing
- (D) Basketball strategy (pick and roll)

(1) "Chair” by Vincent Van Gogh (2) Citibank logo by Paula Scher (3) Leonardo Da Vinci’s sketchbooks

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

(4) Alexander Bell’s first drawing of the telephone

(4) The "Phaeno Science Center" in Germany

[Figure 7]

[Figure 8]

In this work, we introduce SketchAgent, a sketch generation agent that leverages the prior knowledge and sequential nature of multimodal large language models (LLMs) to enable versatile, progressive, language-driven sketching. Our agent can generate sketches across a wide range of textual concepts—from animals to engineering principles (Fig. 1, left). Its sequential nature facilitates interactive humanagent sketching and supports iterative refinement and editing through a chat-based dialogue (Fig. 1, right).

Figure 2. Examples of sketches used across disciplines and goals. (A) Ideation and design: Process Elevation Sketches by the architect Frank Gehry, Guggenheim Museum. (B) Engineering: Alexander Bell’s telephone drawing. (C) Expressing emotions: Children’s sketches. (D) Visual communication: Planning and communicating game strategy in basketball.

Unlike vision-language models that directly generate images from text [85, 90, 92], multimodal LLMs [1, 3, 19, 64, 73, 84, 108] accept text and images as input but only output text. To produce visuals, they either utilize external “tools” (such as calling a text-to-image model) or are prompted to generate executable code (e.g., Python [49], SVG [12]) to create charts, diagrams, or graphics. However, prompting for such representations to directly produce sketches often results in a mechanical appearance with uniform, precise shapes that lack the subtle irregularities and spontaneous qualities characteristic of human sketches (see Fig. 3B). Additionally, despite their robustness in textual tasks, these models often struggle with fine-grained spatial reasoning [47, 129] as they are primarily optimized for text, making sketch editing more challenging.

sible to both the user and the agent throughout the session. The agent generates strokes sequentially and pauses according to an adjustable stopping token, allowing the user to add their own strokes directly to the canvas. These strokes are then integrated into the agent’s sequence, enabling it to continue drawing, with real-time canvas updates.

We demonstrate SketchAgent’s capabil to generate sketches of diverse concepts while capturing the inherently sequential and dynamic nature of sketching. We showcase our agent’s ability to collaborate effectively with humans in real time to create novel and meaningful sketches. Our method is the first to leverage pretrained multimodal LLMs for sequential sketching without additional training, paving the way for a general-purpose artificial sketching system that supports iterative, evolving interactivity.

To address these limitations, we introduce an intuitive sketching language that enables an off-the-shelf multimodal LLM agent to “draw” sketches on a canvas by providing string-based actions, without additional training or finetuning. We define the canvas as a numbered grid, allowing the agent to reference specific coordinates (e.g., x2y8) to enhance its spatial reasoning capabilities. We represent a sketch as a sequence of semantically meaningful strokes, each defined by a series of such coordinates. We leverage In-Context Learning (ICL) [9] to introduce the agent to the new representation, and Chain of Thought (CoT) [119] to enhance its planning capabilities. Given a sketching task, the agent produces a textual response following our representation, which we process by fitting a smooth B´ezier curve to each coordinate sequence. The curves are then rendered onto the canvas to form the final sketch. We find this approach useful in emulating a more natural sketch appearance. For collaborative sketching, the canvas remains acces-

#### 2. Related Work

Sketch Generation Early methods approached sketch generation by designing image filters to simulate sketchlike effects [13, 120]. With the advent of deep learning, data-driven approaches emerged to address a range of sketch-related tasks [128], including category-conditioned sketching [48, 86, 103], object sketching [67, 70], scenesketching [16, 65, 68, 125], sketch completion [7, 71, 72, 105], portrait drawing [4, 131, 132], part-based generation [7, 42, 48, 140], and more. While sketch data collection has been broadly explored [25, 39, 46, 81, 95, 124], the wide variation in sketch styles and their adaptation to specific tasks [28] makes collecting datasets that encompass this diversity challenging. For example, QuickDraw [54], the largest available sketch dataset with 50 mil-

lion sketches, covers only 345 object categories and primarily focuses on simple, iconic representations. This limits data-driven methods to the style, abstraction level, and concepts seen during training. Recently, large pretrained vision-language models [85, 88, 90, 92, 94] have shown remarkable text-to-image generation capabilities by leveraging extensive visual knowledge from billions of training images [99]. While these models can be prompted to generate sketch-like images (see Fig. 3A), they do so in a single step and in pixel space, lacking the sequential, strokebased process of human sketching. Subsequent approaches [18, 34, 36, 53, 116, 117, 126, 127, 135] leverage the priors of these models to guide an iterative optimization of parametric curves, with a differentiable rasterizer [66] linking pixel and vector representations. This approach reduces reliance on human-drawn datasets, enabling robust sketch generation beyond pre-defined categories. However, optimizing all strokes simultaneously results in sketches that lack temporal and semantic structure, and the process is time-consuming, taking 5 minutes to over an hour per sketch, making it suboptimal for collaborative sketching.

Sequential and Collaborative Sketching Collaborative human-machine sketching holds promise in enhancing creativity, ideation, communication, and learning, as explored in various fields, including human-computer interaction (HCI) [21, 52, 56, 57, 61, 62], computer graphics [63, 107], robotics [96, 97], cognitive science [29, 30, 40, 77], learning sciences [2, 43, 115], and more. Central to collaborative sketching is its sequential, adaptive, and dynamic process, with each action carrying intent. Existing methods employ diverse training strategies to account for the discrete nature of sequential sketches, including reinforcement and adversarial learning [37, 78, 140], multi-agent referential games [79, 87], transformers [6, 7, 14, 38, 69, 91, 123], and more. SketchRNN [48] is a pioneering work in this area, introducing the QuickDraw dataset [54], a crowd-sourced collection of real-time sketch sequences made by users. They utilize this dataset to train a recurrent neural network for sequential sketch generation, which was later shown [29, 83] to have potential for human-machine collaboration. However, this approach remains constrained by the predefined categories encountered during training.

Multimodel LLMs for Content Creation LLMs [10, 22, 89, 111] and multimodal LLMs [1, 3, 19, 64, 73, 84, 108] receive text as input (or text and images for multimodal) and output text. To enable visual content generation, these models are often paired with external “tools” that extend their functionality [51, 100, 122, 130]. For example, ChatGPT [84] generates images by internally calling a separate model, DALLE-3 [5]. Another approach involves prompting models to produce code in languages like Python [49], Processing [102], SVG [12], or TikZ [11] that can be ren-

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

- (A) DALLE3 [5]

(B) LLMs [3] (SVG)

(C) SketchAgent (D) Human [54]

Figure 3. Sketch appearance. (A) Text-to-image diffusion models operate in pixel space, lacking thesequential nature of sketches.

- (B) Prompting LLMs to produce visuals with SVG results in a uniform, mechanical appearance. (C) Sketches produced by our agent appear less mechanical, more closely resembling the nature of (D) Human sketches, which are often spontaneous and irregular.

dered into visuals such as graphs, charts, and vector graphics. However, such code-generated content often looks rigid, with uniform and overly precise shapes that lack the subtle irregularities and spontaneous qualities characteristic of freehand sketches (see Fig. 3B). In contrast, we propose a sketching language grounded in spatial information that encourages the model to produce a more natural sketch appearance, which we then process into vector graphics. Common strategies for enhancing LLMs capabilities include Chain-of-Thought prompting [17, 80, 93, 101, 138], which breaks down tasks into smaller, logical steps to mimic human reasoning, and In-Context Learning (ICL) [23, 106, 134, 136], where examples of input-output pairs are provided to help the model infer task patterns.

- P2
- P3

- P0
- P1

#### 3. Preliminaries

t=0.25

Vector Graphics and B´ezier Curves Vector graphics allow us to create visual images directly from geometric shapes such as points, lines, curves, and polygons. Unlike raster images (represented with pixels), vector graphics are resolution-free, more compact, and editable. SVG [121] is an XML-based format for storing vector graphics, popular for its scalability and compatibility with modern web browsers. The process of transferring vector graphics into pixel images is called rasterization or rendering. Cubic B´ezier curves are commonly used to represent sketches in vector graphics. A cubic B´ezier curve (Fig. 4) is a smooth parametric curve defined by four points: a start point P0, an end point P3, and two control points P1 and P2 that shape the curvature . The set P = {P0,P1,P2,P3} is often referred to as the curve’s control points. The curve is described by the following polynomial equation:

Control points

Sample at t

Figure 4. Cubic B´ezier curve.

B(t) = (1−t)3P0+3(1−t)2tP1+3(1−t)t2P2+t3P3, (1)

where t ∈ [0,1] is a parameter that moves the point along the curve from P0 at t = 0 to P3 at t = 1.

Sj

- P2
- P3

- P0
- P1

50

(xj,yj), t

“<thinking>To draw a shark.. <strokes>

System Prompt

|3 2 1<br><br>1 2 3 . . . . . . . . 50|
|---|

Human Sketcher

- <s1> <points>‘x5y8’ ‘x12y20’.. <t_values>’0’ ‘0.3’.. <id>dorsal fin</id>

</s1>

- <s2> <points>... <t_values>...

.......

Processing Render

User Prompt

50

(x,y) coordinates Control points

SketchAgent

.......

...

Canvas

<strokes>”

3 2 1

1 2 3 . . . . . . . . 50

Figure 5. Method Overview. SketchAgent (blue) receives drawing instructions and generates a string representing the intended sketch. Inputs include: (1) a system prompt (orange) introducing the sketching language and canvas, (2) a user prompt (pink) specifying the task (e.g., “draw a shark”), and (3) a numbered canvas. The agent’s response outlines a sketching strategy (in thinking tags) and a sequence of strokes defined by coordinates, which are processed into B´ezier curves and rendered onto the canvas.

#### 4. Method

Our goal is to enable an off-the-shelf pretrained multimodal LLM to draw sketches based on natural language instructions. An overview of our pipeline is illustrated in Fig. 5. We utilize a frozen multimodal LLM (“SketchAgent” shown in blue), which receives three inputs: (1) a system prompt containing guidelines for using our new sketching language, (2) a user prompt with additional task-specific instructions (e.g., “Draw a shark”), and (3) a blank canvas on which the agent can draw. Based on the given task, the agent generates a textual response, representing the sequence of strokes to be drawn, which we then process into vector graphics and render onto the canvas. The canvas can then be reused in two ways: it can be fed back into the model with an updated user prompt for additional tasks and editing, or it can be accessed by a human user who can draw directly on it to facilitate collaborative sketching. Next, we describe each component of the pipeline.

The Canvas Although multimodal LLMs demonstrate remarkable reasoning abilities, they often struggle with spatial reasoning tasks [35, 75, 110]. We present a simple example (see Fig. 6) to illustrate how this limitation affects the naive use of these models for sketch generation and interactive sketching. We provide GPT-4o [84] with an image depicting a simple line drawing of a partial house featuring five numbered points (from 1 to 5), and ask it to identify which points should be connected to complete the house. While the model correctly identifies the pair of points, it fails to select the correct pixel coordinates when given a basic draw line tool that connects two points, even after multiple attempts. To enhance the model’s spatial reasoning ability, we utilize a numbered canvas that forms a grid. This grid features numbers (1 to 50) along the x-axis and the y-axis (Fig. 5, left). Each cell is uniquely identified by a combination of the corresponding x-axis and y-axis numbers (e.g., the bottom-left cell is x1y1). The agent interacts with the canvas by specifying desired (x,y) coordinates.

[Figure 13]

[Figure 14]

User:

Agent:

“Draw a line between points 1 and 5.”

“Complete the drawing to form a house”

Figure 6. Although excelling in visual reasoning, multimodal LLMs often struggle to translate these abilities into spatial actions. In this example, GPT-4o [84] intends to draw a line between points 1 and 5 but fails to execute this with a draw line function that accepts pixel coordinates.

Sketch Representation We define a sketch as a sequence of n ordered strokes S = {S1,S2,...Sn}. Each stroke Si is defined by a sequence of m cell coordinates on the grid: Si = {(xj,yj)}mj=1, represented in string format as: <points>x1y1, x15y20, ...</points>.

A naive approach to processing the textual sequence of coordinates would be to use a polyline, connecting consecutive points with line segments. However, our gridbased representation sparsifies the canvas, resulting in a non-smooth and unnatural appearance when using polylines (see Fig. 7, left). To achieve a smoother appearance, an alternative approach is to treat the coordinates as a sequence of control points defining smooth curves. However, as illustrated in Fig. 4, the control points often do not lie directly on the curve. Consequently, if the agent aims for a stroke that passes through specific coordinates, it must derive the control points that define this stroke, which is challenging.

We propose an alternative approach: we treat the specified (x,y) coordinates as a set of desired points sampled along the curve, and fit a smooth B´ezier curve to them (Fig. 7, right). To accommodate curves with complex curvature, we also task the model with determining when each point on the curve should be passed through, corresponding to the t value described in Eq. (1). Thus, for each stroke Si, the agent provides a set of m sampled points Si = {(xj,yj)}mj=1, along with a corresponding set of t values: Ti = {tj}mj=1. Based on these, we fit a cubic B´ezier

[Figure 15]

[Figure 16]

[Figure 17]

(A) Polyline (B) Control Points (C) Ours

Figure 7. Methods for processing the agent’s coordinate sequence (in red): (A) Polyline results in an unnatural appearance. (B) Directly using coordinates as B´ezier control points is challenging as they do not lie on the curve. (C) Fitting a B´ezier curve to sampled coordinates provides smoother results.

curve to the sampled points by solving a system of linear equations using least squares, where the unknowns are the control points P = {P0,P1,P2,P3}:

P = argminP||AP − B||, (2)

where A ∈ Rm×4 contains the cubic B´ezier basis functions evaluated at specific tj values (as described in Eq. (1)), and B ∈ Rm×2 contains the m sampled points

{(xj,yj)}mj=1. The least squares solution minimizes the error between the fitted B´ezier curve and the sampled points. For long sequences resulting in a large fitting error, we recursively split the curve. Additionally, we account for B´ezier curves of lower degrees, including quadratic curves, linear lines, and points. Upon completing this process, we render the parametric curves onto the canvas.

Drawing Instructions We provide the model with a system prompt and a user prompt (marked in orange and pink in Fig. 5). In the system prompt, we supply the agent with context about its expertise (“You are an expert artist specializing in drawing sketches”) and introduce it to the grid canvas along with examples of how to use our sketching language for drawing single-stroke primitives (full prompts are provided in the Appendix). The system prompt is fixed and can be applied to a variety of sketching tasks. The user prompt includes a description of the desired task and an example of a simple sketch of a house drawn with our sketching language. We find this to be crucial in assisting the agent with preserving the correct format that could be parsed directly [9]. The agent is tasked with responding in the format shown in the gray text box in Fig. 5. In the <thinking> tags, the agent is tasked to outline the overall sketching strategy [119]. This typically includes describing the different components of the sketch, the intended sketching order, and the overall placement of each part. The agent is also tasked with providing an ID tag following each stroke, which is useful for further analysis and for producing annotated sketches in scale.

##### 4.1. In-Chat Editing and Collaborative Sketching

The above process can be repeated iteratively to support multiple sketching tasks and interactions. Text-based sketch

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Golden Gate Bridge

Mount Fuji

Eiffel Tower

DNA Double Helix

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Tic-tac toe

Pendulum Motion

Double-Slit Experiment

Flowchart

Figure 8. Sketches produced by SketchAgent for concepts beyond pre-defined categories. The textual input describing the desired concept shown below each image.

editing in a chat dialogue is enabled by feeding the rendered canvas back to the agent (see dashed arrow in Fig. 5) and updating the user prompt with the desired edits. To support collaborative human-agent sketching, the canvas remains accessible to both the human user and the agent throughout the entire sketching session. We define an adjustable stopping token, </s{j}>, which instructs the agent to pause generating the sequence at stroke number j. We then process and render the generated strokes onto the canvas up to that point, then the user can add strokes directly to the canvas to continue the sketch. The user-drawn strokes are processed and converted into the agent’s format by reversing our fitting process, i.e., sampling each stroke at multiple t values (as shown in Eq. (1)), and selecting the points closest to each cell’s center on the grid. The converted user strokes are then chained with the agent’s sequence, after which the agent resumes sketching until the next stopping token.

#### 5. Results

We evaluate the performance of our method qualitatively and quantitatively across a selected set of sketching tasks. Additional tasks, evaluations, and examples are provided in the Appendix. All results presented in the paper were generated using Claude3.5-Sonnet [3] as our backbone model, unless stated otherwise.

##### 5.1. Text-Conditioned Sketch Generation

Figures 1 and 8 demonstrate SketchAgent’s capability to generate sketches of various concepts that extend beyond standard categories, which includes scientific concepts (e.g., “the double-slit experiment”, “pendulum motion”), diagrams (e.g.,“circuit diagram”, “a flowchart”), and notable landmarks (e.g., “Taj Mahal”, “Eiffel Tower”). More examples are provided in the Appendix. To quantitatively

Claude3.5 -Sonnet (SVG)

GPT-4o -mini

Claude3 Opus

Claude3.5 -Sonnet*

Human (QD [54])

GPT-4o

|Top1<br><br>|0.15 0.04 0.13 0.23 ±0.04 ±0.03 ±0.04 ±0.05|0.23 0.27 ±0.04 ±0.07<br><br>|
|---|---|---|
|Top5|0.30 0.10 0.27 0.44 ±0.06 ±0.04 ±0.05 ±0.03<br><br>|0.43 0.49 ±0.06 ±0.06<br><br>|
|Vis.<br><br>|[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]|[Figure 30]<br><br>[Figure 31]|

Table 1. Sketch recognition evaluation. Average Top-1 and Top-5 sketch recognition accuracy computed with CLIP zero-shot classifier on 500 sketches from 50 categories. The last row visualizes one sample from each experiment. *Indicates our default settings, which receives the highest accuracy among all models.

evaluate text-conditioned generation we utilize the QuickDraw dataset [54]. We randomly sample 50 categories (out of 345), and apply our method to generate 10 sketch instances per category, resulting in 500 sketches in total. Following common practice [116, 117, 126, 127], we utilize a CLIP zero-shot classifier [88] to evaluate how well the generated sketches depict the intended category. We compare the performance of different multimodal LLMs by repeating the same process with GPT-4o-mini [84], GPT-4o [84], and Claude3-Opus [3] as our backbone model (in addition to Claude3.5-Sonnet [3], our default backbone). As a baseline, we include human-drawn sketches sampled from the QuickDraw dataset [54]. The average Top-1 and Top-5 sketch classification accuracy are presented in Table 1. As can be seen, human sketches achieve the highest recognition accuracy, with Claude3.5-Sonnet performing best among all models, approaching human-level rates under the CLIPscore metric. More evaluation of confusion patterns and visualization of the data are provided in the Appendix.

We additionally compare to prompting Claude3.5Sonnet to directly generate SVGs using the following prompt: “Write SVG string that draws a sketch of a <concept>. Use only black and white colors”. The corresponding scores are shown in the fifth column of Tab. 1. While this approach achieves recognition scores comparable to those of SketchAgent, the outputs are often characterized by uniform and precise shapes, failing to replicate the fluidity and natural irregularity of free-hand human sketches (e.g., Fig. 3). To evaluate how “humanlike” our agent’s sketches appear, we conduct a two alternative forced choice (2AFC) user study with 150 participants. Each participant was presented with pairs of sketches depicting the same object class produced by different methods, and asked to choose the sketch they believed was human-drawn. 150 sketches across 50 object classes were tested, comparing three methods: direct prompting, SketchAgent, and human sketches from Quick-

[Figure 32]

[Figure 33]

Figure 9. SketchAgent gradually draws stroke-by-stroke, each stroke is annotated by the agent with a semantic meaning.

Draw (see Appendix for details). Results indicate SketchAgent’s drawings appeared more human-like, being chosen as human-drawn in 74.90±3.35% of cases when compared with direct prompting. When compared to human drawings, users slightly preferred human sketches (54.68 ± 4.61%) over SketchAgent’s, while direct prompting was chosen only 38.9 ± 5.55% of the time.

##### 5.2. Sequential Sketching

Figure 9 shows stroke-by-stroke sketch generation by SketchAgent, with the labels on the right indicating the sketching order and the meaning our agent associates with each generated stroke (see Appendix for more examples). Stroke annotation during generation is enabled by utilizing the prior of the backbone LLM, providing a valuable feature for analysis and data collection [42, 74, 118, 137, 139]. In Fig. 10, we illustrate why accounting for the sequential nature of sketching more closely emulates the process of human drawing. We present the sketch creation process of SketchAgent alongside SVGDreamer [127], SketchRNN [48], and a human sketch sampled from QuickDraw [54]. SVGDreamer (first row), is an optimization-based method, where a set of randomly initialized parametric curves (leftmost column) are iteratively refined to form a sketch, guided by a pretrained text-to-image diffusion model [92]. This process is time-consuming, taking 2000 iterations (1.6 hours), which makes it unsuitable for interactive sketching. While the final sketch (rightmost column) appears detailed and artistic due to the powerful vision backbone, the intermediate sketching and individual strokes lack clear semantic meaning. In contrast, SketchRNN (second row) is a sequential generative model trained on human-drawn dataset, producing sketches in real-time with strokes added progressively, emulating closer a human-like sketching process (as shown in the last row). Similarly, SketchAgent (third row) produces sketches gradually, with each stroke carrying a semantic meaning, by utilizing the sequential nature of its backbone model. While SketchRNN is restricted to generating sketches only within the 345 categories it was trained on, SketchAgent leverages the extensive prior knowledge of its backbone multimodal LLM, enabling it to create sketches of general visual concepts.

SVGDreamer [127]. ≈ 1.6 hours

SketchAgent

Human

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

SketchRNN [48]. ≈ 4 seconds

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Figure 11. Sequential sketching analysis of SketchAgent (blue) and Humans [54] (orange). Left: Histograms of stroke distribution per sketch, showing QuickDraw sketches are more abstract on average. Right: CLIPScore as a function of the accumulated number of strokes for sketches containing 4-7 strokes, showing a similar recognition pattern over time.

SketchAgent (Ours). ≈ 20 seconds

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

tically meaningful and recognizable sketches. We design a web-based collaborative sketching environment (Fig. 12A) where users and SketchAgent take turns drawing on a shared canvas to create a recognizable sketch from a given textual concept. Following the evaluation protocol in collabdraw [29], we select 8 simple concepts, based on the agent’s demonstrated ability to draw them independently, to focus evaluations on assessing the impact of collaboration. Participants sketched concepts in two modes: solo, where users drew independently, and collab, where users and SketchAgent collaborated, adding one stroke at a time until either was satisfied with the drawing. We collect sketches from 30 participants, resulting in 480 sketches in total. Average CLIP recognition rates are shown in Figure 12B. Collaboratively produced sketches (blue) achieve recognition levels close to those made solely by users and higher than those produced by the agent alone (dashed lines). To assess the contribution of each party in collaborative mode, we analyze partial sketches with only agent-made strokes (pink) or user-made strokes (green), resulting in a significant reduction in recognizability. This suggests that both user and agent contribute meaningfully to the recognizability of the complete sketch.

Human [54]. ≤ 20 seconds

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Figure 10. Sequential sketching process. SVGDreamer [127] requires 2000 iterations (1.6 hours) with intermediate steps lacking semantic meaning. SketchRNN [48] operates in real-time with coherent steps but is limited to QuickDraw categories. SketchAgent creates sketches gradually with meaningful strokes and no category restrictions. Human sketches also evolve through gradual, meaningful steps.

We use the set of 500 samples described in Sec. 5.1 to quantitatively analyze the sequential nature of our agent’s sketches compared to human drawings. On the left of Fig. 11, we present histograms comparing the number of strokes in QuickDraw sketches (orange) and our sketches (blue). Most QuickDraw sketches contain 1 to 6 strokes, while our sketches show a broader distribution, peaking between 5 to 10 strokes. This suggests that, on average, QuickDraw sketches appear more abstract. To ensure a balanced comparison of sketches with similar levels of abstraction, we select sketches from both groups with a similar number of strokes (the largest intersection is found in sketches with 4-7 strokes, comprising 204 of our sketches and 120 from QuickDraw) and measure the change in CLIPScore as a function of the accumulated number of strokes (Fig. 11, right). Both QuickDraw and our sketches exhibit a generally similar pattern, with CLIPScore increasing as more strokes are added, suggesting that sketches become progressively more recognizable as they evolve.

##### 5.4. Chat-Based Sketch Editing

We next demonstrate the effectiveness of our method in performing interactive text-based sketch editing within a chat dialogue, where the input to the agent combines both text and images. Inspired by [102], we explore edits that involve spatial reasoning and object relations. We focus on three object categories: outdoor, indoor, and animals, with three objects each, and design editing prompts to add objects to the input sketches. For outdoor and indoor objects, we specify relative locations of added concepts, e.g., “left to”, “on top of” (see Fig. 47 left). For the animals category, we tasked the agent with adding accessories to each animal

##### 5.3. Human-Agent Collaborative Sketching

We demonstrate the potential of our system for facilitating interactive human-agent collaboration, resulting in seman-

[Figure 62]

1.0

solo user full sketches

0.8

###### Let’s Sketch a Sailboat

recognitionrate

solo agent full sketches

[Figure 63]

0.6

[Figure 64]

0.4

User Turn! Draw a stroke

solo user partial strokes

0.2

0.0

collab full sketches

collab agent-only strokes

collab user-only strokes

(A) Sketching interface (B) Collaborative user study results

- Figure 12. Collaborative sketching evaluation measured using CLIP classification. Sketches created collaboratively (blue) approaching those made solely by users (dashed lines). In collaborative sketches, keeping agent-only strokes (pink) or user-only strokes (green) significantly reduces recognizability.

“Building” “Nightstand” “Cat”

[Figure 65]

[Figure 66]

[Figure 67]

“Tree to the left” “Sun on top right” “Building to the right”

“Coffee mug on top” “Lamp on top” “Plant to the left”

“Add glasses” “Add a hat” “Add a skirt”

- Figure 13. Chat-based sketch editing. We iteratively prompt SketchAgent to add objects to sketches through chat dialogues.

without guidance on their exact placement, testing its ability to infer placement based on semantics (e.g., placing a hat on a head (see Fig. 47 right). The full list of object and editing instructions is provided in the Appendix. We produced a total of 54 sketches. Evaluating the edited sketches reveals that SketchAgent correctly follows instructions 92% of the time, with 94% accuracy for specified relations and 88% accuracy for inferred semantic relations.

#### 6. Ablation

We evaluate the impact of each component of our method by systematically removing them and measuring sketch recognition rates as detailed in 5.1. We assess the effects of removing the system prompt, omitting the CoT process (i.e., excluding thinking tags and ’think step-by-step’ instructions), and modifying ICL (the complete sketch example provided in the user prompt). When modifying ICL, we use a correctly formatted single-stroke example instead of the complete sketch, as fully removing ICL results in outputs that do not follow the expected format making them unparsable. The results in Table 2 show that the full SketchAgent pipeline achieves the highest performance, highlighting the importance of each component. Interestingly, not providing a complete sketch example significantly reduces performance. Additional visualizations and analyses are provided in the Appendix.

w/o System Prompt

Modified ICL

SketchAgent (full)

w/o CoT

Top1 0.20 ± 0.04 0.14 ±0.02 0.07 ± 0.02 0.23 ± 0.04 Top5 0.42 ± 0.03 0.29 ± 0.04 0.16 ± 0.03 0.43 ± 0.06

Table 2. Ablation study. Average Top-1 and Top-5 CLIP recognition accuracy. We systematically remove each component in our pipeline, showcasing all components contribute to the agent’s full performance.

#### 7. Limitations and Future Work

SketchAgent has several limitations. First, it is constrained by the priors of the backbone model, primarily optimized for text rather than visual content. As a result, the agent often produces rich textual descriptions of object parts but struggles to convert these into effective sketching actions, resulting in overly abstract and unrecognizable outputs. For example, in Fig. 14A, the agent effectively describes key parts of a unicorn (e.g., the horn), but the sketch is unrecognizable. This constraint also impacts the depiction of human figures (Fig. 14B). While distinctive features (e.g., Frida Kahlo’s eyebrows or Michael Jordan’s dunk) may be captured well in language, the resulting sketches are overly simple, with an amateur style, lacking expressivity. We expect this issue to improve as future models advance in vision capabilities. Lastly, the agent may struggle with drawing letters and numbers. This could be improved in future work by providing relevant in-context examples.

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Figure 14. Limitations. Sketches of complex concepts (A) and human figures (B) appear too abstract and unrecognizable with non-professional style. (C) Fail to depict letters and numbers.

#### 8. Conclusions

We presented a method for language-driven, sequential sketch generation, that can produce versatile sketches in real-time and meaningfully engage in collaborative sketching sessions with humans. We show that the prior knowledge embedded in pretrained multimodal LLMs can be effectively leveraged for sketch generation through an intuitive sketching language and a grid canvas, without requiring additional training or fine-tuning. We hope our work represents a meaningful step toward developing general-purpose sketching systems with the potential to enhance human-computer communication and computeraided ideation.

#### 9. Acknowledgements

We thank Yuval Alaluf, Hila Chefer, Assaf Ben Kish, Joanna Materzynska, Rinon Gal, Elad Richardson, Arnav Verma, and Ellie Arar for providing feedback on early versions of our manuscript. We are especially grateful to Yarden Frenkel for his insights, early explorations, and engaging discussions. This work was partially supported by NSF CAREER #2047191, NSF DRL #2400471, Stanford Human Centered AI Institute Hoffman-Yee Grant, Hyundai Motor Company, ARL grant W911NF-18-2-021, the Zuckerman STEM Leadership Program, and the Viterbi Fellowship. The funders had no role in the experimental design or analysis, the decision to publish, or manuscript preparation. The authors have no competing interests to report.

#### References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millicah, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning. In Proceedings of the 36th International Conference on Neural Information Processing Systems, Red Hook, NY, USA, 2024. Curran Associates Inc. 2, 3
- [2] Vanessa De Andrade, Sofia Freire, M´onica Baptista, and Yael Shwartz. Drawing as a space for social-cognitive interaction. Education Sciences, 12(1), 2022. 3
- [3] Anthropic. Claude. https://www.anthropic.com/ claude, 2023. 2, 3, 5, 6, 1
- [4] Itamar Berger, Ariel Shamir, Moshe Mahler, Elizabeth Carter, and Jessica Hodgins. Style and abstraction in portrait sketching. ACM Trans. Graph., 32(4), 2013. 2
- [5] James Betker, Gabriel Goh, Li Jing, † TimBrooks, Jianfeng Wang, Linjie Li, † LongOuyang, † JuntangZhuang, † JoyceLee, † YufeiGuo, † WesamManassra, † PrafullaDhariwal, † CaseyChu, † YunxinJiao, and Aditya Ramesh. Improving image generation with better captions. 3
- [6] Ayan Kumar Bhunia, Ayan Das, Umar Riaz Muhammad, Yongxin Yang, Timothy M. Hospedales, Tao Xiang, Yulia Gryaditskaya, and Yi-Zhe Song. Pixelor: a competitive sketching ai agent. so you think you can sketch? ACM Trans. Graph., 39:166:1–166:15, 2020. 1, 3
- [7] Ankan Kumar Bhunia, Salman Khan, Hisham Cholakkal, Rao Muhammad Anwer, Fahad Shahbaz Khan, Jorma Laaksonen, and Michael Felsberg. Doodleformer: Creative sketch drawing with transformers. ECCV, 2022. 2, 3
- [8] Ankan Kumar Bhunia, Salman Khan, Hisham Cholakkal, Rao Muhammad Anwer, Fahad Shahbaz Khan, Jorma Laaksonen, and Michael Felsberg. Doodleformer: Creative sketch drawing with transformers. In European Conference on Computer Vision, pages 338–355. Springer, 2022. 1

- [9] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems, pages 1877–1901. Curran Associates,

- Inc., 2020. 2, 5

[10] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems, pages 1877–1901. Curran Associates,

- Inc., 2020. 3

[11] S´ebastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-

- 4. arXiv preprint arXiv:2303.12712, 2023. 3

- [12] Mu Cai, Zeyi Huang, Yuheng Li, Haohan Wang, and Yong Jae Lee. Delving into LLMs’ visual understanding ability using SVG to bridge image and text, 2024. 2, 3
- [13] John Canny. A computational approach to edge detection. IEEE Transactions on pattern analysis and machine intelligence, (6):679–698, 1986. 2
- [14] Alexandre Carlier, Martin Danelljan, Alexandre Alahi, and Radu Timofte. Deepsvg: A hierarchical generative network for vector graphics animation, 2020. 3
- [15] Micah Carroll, Rohin Shah, Mark K Ho, Tom Griffiths, Sanjit Seshia, Pieter Abbeel, and Anca Dragan. On the utility of learning about humans for human-ai coordination. In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2019. 13
- [16] Caroline Chan, Fr´edo Durand, and Phillip Isola. Learning to generate line drawings that convey geometry and semantics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7915–7925,

2022. 2

- [17] Zhenfang Chen, Qinhong Zhou, Yikang Shen, Yining Hong, Zhiqing Sun, Dan Gutfreund, and Chuang Gan. Visual chain-of-thought prompting for knowledge-based visual reasoning. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1254–1262, 2024. 3
- [18] Changwoon Choi, Jaeah Lee, Jaesik Park, and Young Min Kim. 3doodle: Compact abstraction of objects with 3d strokes. ACM Trans. Graph., 43(4), 2024. 3
- [19] Xiangxiang Chu, Jianlin Su, Bo Zhang, and Chunhua Shen.

Visionllama: A unified llama backbone for vision tasks,

2024. 2, 3

- [20] Ayan Das, Yongxin Yang, Timothy Hospedales, Tao Xiang, and Yi-Zhe Song. B´eziersketch: A generative model for scalable vector sketches. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXVI 16, pages 632–647. Springer,

2020. 1

- [21] Nicholas Davis, Chih-PIn Hsiao, Kunwar Yashraj Singh, Lisa Li, Sanat Moningi, and Brian Magerko. Drawing apprentice: An enactive co-creative agent for artistic collaboration. In Proceedings of the 2015 ACM SIGCHI Conference on Creativity and Cognition, page 185–186, New York, NY, USA, 2015. Association for Computing Machinery. 3
- [22] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota, 2019. Association for Computational Linguistics. 3
- [23] Sivan Doveh, Shaked Perek, M Jehanzeb Mirza, Wei Lin, Amit Alfassy, Assaf Arbelle, Shimon Ullman, and Leonid Karlinsky. Towards multimodal in-context learning for vision & language models. arXiv preprint arXiv:2403.12736,

2024. 3

- [24] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. 4
- [25] Mathias Eitz, James Hays, and Marc Alexa. How do humans sketch objects? ACM Trans. Graph., 31(4), 2012. 2
- [26] Ziv Epstein, Aaron Hertzmann, Laura Mariah Herman, Robert Mahari, Morgan R. Frank, Matthew Groh, Hope Schroeder, Amy Smith, Memo Akten, Jessica Fjeld, Hany Farid, Neil Leach, Alex Pentland, and Olga Russakovsky. Art and the science of generative ai. Science, 380:1110 – 1111, 2023. 1
- [27] Hugging Face. clip-vit-large-patch14. https:// huggingface.co/openai/clip- vit- largepatch14. 1
- [28] Judy Fan, Wilma A. Bainbridge, Rebecca Chamberlain, and Jeffrey D. Wammes. Drawing as a versatile cognitive tool. Nature Reviews Psychology, 2:556 – 568, 2023. 2
- [29] Judith E. Fan, Monica Dinculescu, and David Ha. collabdraw: An environment for collaborative sketching with an artificial agent. In Proceedings of the 2019 Conference on Creativity and Cognition, page 556–561, New York, NY, USA, 2019. Association for Computing Machinery. 3, 7
- [30] Judith E. Fan, Robert D. Hawkins, Mike Wu, and Noah D. Goodman. Pragmatic Inference and Visual Abstraction Enable Contextual Flexibility During Visual Communication. Computational Brain & Behavior, 3(1):86–101, 2020. 3
- [31] Judith E Fan, Wilma A Bainbridge, Rebecca Chamberlain, and Jeffrey D Wammes. Drawing as a versatile cognitive tool. Nature Reviews Psychology, 2(9):556–568, 2023. 1

- [32] Logan Fiorella and Shelbi Kuhlmann. Creating drawings enhances learning by teaching. Journal of Educational Psychology, 112(4):811, 2020. 1
- [33] Kenneth Forbus, Jeffrey Usher, Andrew Lovett, Kate Lockwood, and Jon Wetzel. Cogsketch: Sketch understanding for cognitive science research and for education. Topics in Cognitive Science, 3(4):648–666, 2011. 1
- [34] Kevin Frans, L. B. Soros, and Olaf Witkowski. Clipdraw: exploring text-to-drawing synthesis through languageimage encoders. In Proceedings of the 36th International Conference on Neural Information Processing Systems, Red Hook, NY, USA, 2024. Curran Associates Inc. 1, 3
- [35] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. arXiv preprint arXiv:2404.12390, 2024. 4
- [36] Rinon Gal, Yael Vinker, Yuval Alaluf, Amit Bermano, Daniel Cohen-Or, Ariel Shamir, and Gal Chechik. Breathing life into sketches using text-to-video priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4325–4336, 2024. 3
- [37] Yaroslav Ganin, Tejas D. Kulkarni, Igor Babuschkin, S. M. Ali Eslami, and Oriol Vinyals. Synthesizing programs for images using reinforced adversarial learning. ArXiv, abs/1804.01118, 2018. 3
- [38] Yaroslav Ganin, Sergey Bartunov, Yujia Li, Ethan Keller, and Stefano Saliceti. Computer-aided design as language. In Neural Information Processing Systems, 2021. 3
- [39] Chengying Gao, Qi Liu, Qi Xu, Limin Wang, Jianzhuang Liu, and Changqing Zou. Sketchycoco: Image generation from freehand scene sketches. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5174–5183, 2020. 2
- [40] Simon Garrod, Nicolas Fay, John Lee, Jon Oberlander, and Tracy MacLeod. Foundations of representation: Where might graphical symbol systems come from? Cognitive Science, 31(6):961–987, 2007. 3
- [41] Songwei Ge, Vedanuj Goswami, C Lawrence Zitnick, and Devi Parikh. Creative sketch generation. arXiv preprint arXiv:2011.10039, 2020. 1
- [42] Songwei Ge, Vedanuj Goswami, Larry Zitnick, and Devi Parikh. Creative sketch generation. In International Conference on Learning Representations, 2021. 2, 6
- [43] Hannie Gijlers, Armin Weinberger, Alieke Mattia van Dijk, Lars Bollen, and Wouter van Joolingen. Collaborative drawing on a shared digital canvas in elementary science education: The effects of script and task awareness support. International Journal of Computer-Supported Collaborative Learning, 8(4):427–453, 2013. 3
- [44] Gabriela Goldschmidt. Serial sketching: visual problem solving in designing. Cybernetics and System, 23(2):191– 219, 1992. 1
- [45] Barbara J. Grosz and Sarit Kraus. Collaborative plans for complex group action. Artificial Intelligence, 86(2):269– 357, 1996. 13
- [46] Yulia Gryaditskaya, Mark Sypesteyn, Jan Willem Hoftijzer, Sylvia Pont, Fr´edo Durand, and Adrien Bousseau.

- Opensketch: a richly-annotated dataset of product design sketches. ACM Trans. Graph., 38(6), 2019. 2
- [47] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14375–14385, 2024. 2
- [48] David Ha and Douglas Eck. A neural representation of sketch drawings. CoRR, abs/1704.03477, 2017. 1, 2, 3, 6, 7
- [49] Yucheng Han, China. Xiaoyan Zhang, Xin Chen, Xu Yang, Zhibin Wang, Gang Yu, Bin Fu, and Hanwang Zhang. Chartllama: A multimodal llm for chart understanding and generation. ArXiv, abs/2311.16483, 2023. 2, 3
- [50] Robert D. Hawkins, Megumi Sano, Noah D. Goodman, and Judith E. Fan. Visual resemblance and communicative context constrain the emergence of graphical conventions,

2021. 13

- [51] Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. arXiv preprint arXiv:2406.09403, 2024. 3
- [52] Francisco Javier Ibarrola, Tomas Lawton, and Kazjon Grace. A collaborative, interactive and context-aware drawing agent for co-creative design. IEEE Transactions on Visualization and Computer Graphics, 30:5525–5537, 2022. 3
- [53] Ajay Jain, Amber Xie, and Pieter Abbeel. Vectorfusion: Text-to-svg by abstracting pixel-based diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1911–1920, 2023. 1, 3
- [54] Jongejan Jonas, Rowley Henry, Kawashima Takashi, Kim Jongmin, and Fox-Gieg Nick. The Quick, Draw! - A.I. Experiment, 2016. 2, 3, 6, 7, 8
- [55] David Kaiser. Drawing theories apart: The dispersion of Feynman diagrams in postwar physics. University of Chicago Press, 2019. 1
- [56] Pegah Karimi, Jeba Rezwana, Safat Siddiqui, Mary Lou Maher, and Nasrin Dehbozorgi. Creative sketching partner: an analysis of human-ai co-creativity. In Proceedings of the 25th International Conference on Intelligent User Interfaces, page 221–230, New York, NY, USA, 2020. Association for Computing Machinery. 3
- [57] Pegah Karimi, Jeba Rezwana, Safat Siddiqui, Mary Lou Maher, and Nasrin Dehbozorgi. Creative sketching partner: an analysis of human-ai co-creativity. In Proceedings of the 25th International Conference on Intelligent User Interfaces, page 221–230, New York, NY, USA, 2020. Association for Computing Machinery. 3
- [58] G¨unther Knoblich, Stephen Butterfill, and Natalie Sebanz. Chapter three - psychological research on joint action: Theory and data. In Advances in Research and Theory, pages 59–101. Academic Press, 2011. 13

- [59] Kozea. Cairosvg. https://cairosvg.org/, 2023. 1
- [60] Paul Laseau. Graphic thinking for architects and designers. John Wiley & Sons, 2000. 2
- [61] Tomas Lawton, Kazjon Grace, and Francisco J Ibarrola. When is a tool a tool? user perceptions of system agency in human–ai co-creative drawing. In Proceedings of the 2023 ACM Designing Interactive Systems Conference, page 1978–1996, New York, NY, USA, 2023. Association for Computing Machinery. 3
- [62] Tomas Lawton, Francisco J Ibarrola, Dan Ventura, and Kazjon Grace. Drawing with reframer: Emergence and control in co-creative&nbsp;ai. In Proceedings of the 28th International Conference on Intelligent User Interfaces, page 264–277, New York, NY, USA, 2023. Association for Computing Machinery. 3
- [63] Yong Jae Lee, C. Lawrence Zitnick, and Michael F. Cohen. Shadowdraw: real-time user guidance for freehand drawing. In ACM SIGGRAPH 2011 Papers, New York, NY, USA, 2011. Association for Computing Machinery. 3
- [64] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In Proceedings of the 39th International Conference on Machine Learning, pages 12888–12900. PMLR, 2022. 2, 3
- [65] Mengtian Li, Zhe Lin, Radomir Mech, Ersin Yumer, and Deva Ramanan. Photo-sketching: Inferring contour drawings from images. In 2019 IEEE Winter Conference on Applications of Computer Vision (WACV), pages 1403–1412. IEEE, 2019. 2
- [66] Tzu-Mao Li, Michal Luk´aˇc, Gharbi Micha¨el, and Jonathan Ragan-Kelley. Differentiable vector graphics rasterization for editing and learning. ACM Trans. Graph. (Proc. SIGGRAPH Asia), 39(6):193:1–193:15, 2020. 1, 3
- [67] Yi Li, Yi-Zhe Song, Timothy M. Hospedales, and Shaogang Gong. Free-hand sketch synthesis with deformable stroke models. CoRR, abs/1510.02644, 2015. 1, 2
- [68] Yijun Li, Chen Fang, Aaron Hertzmann, Eli Shechtman, and Ming-Hsuan Yang. Im2pencil: Controllable pencil illustration from photographs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1525–1534, 2019. 2
- [69] Hangyu Lin, Yanwei Fu, Yu-Gang Jiang, and X. Xue. Sketch-bert: Learning sketch bidirectional encoder representation from transformers by self-supervised learning of sketch gestalt. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6757–6766,

2020. 3

- [70] Difan Liu, Matthew Fisher, Aaron Hertzmann, and Evangelos Kalogerakis. Neural strokes: Stylized line drawing of 3d shapes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14204– 14213, 2021. 2
- [71] Fang Liu, Xiaoming Deng, Yu-Kun Lai, Yong-Jin Liu, Cuixia Ma, and Hongan Wang. Sketchgan: Joint sketch completion and recognition with generative adversarial network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 2

- [72] Fang Liu, Xiaoming Deng, Yu-Kun Lai, Yong-Jin Liu, Cuixia Ma, and Hongan Wang. Sketchgan: Joint sketch completion and recognition with generative adversarial network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5830– 5839, 2019. 2
- [73] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems, pages 34892–34916. Curran Associates, Inc., 2023. 2, 3
- [74] Bria Long, Judith Fan, Holly Huey, Zixian Chai, and Michael Frank. Parallel developmental changes in children’s production and recognition of line drawings of visual concepts. Nature Communications, 15, 2024. 6
- [75] Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal, Paul Mcvay, Oleksandr Maksymets, Sergio Arnaud, Karmesh Yadav, Qiyang Li, Ben Newman, Mohit Sharma, Vincent Berges, Shiqi Zhang, Pulkit Agrawal, Yonatan Bisk, Dhruv Batra, Mrinal Kalakrishnan, Franziska Meier, Chris Paxton, Sasha Sax, and Aravind Rajeswaran. Openeqa: Embodied question answering in the era of foundation models. In Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 4
- [76] William P. McCarthy, Robert D. Hawkins, Haoliang Wang, Cameron Holdaway, and Judith E. Fan. Learning to communicate about shared procedural abstractions, 2021. 13
- [77] William P. McCarthy, Justin Matejka, Karl D.D. Willis, Judith E. Fan, and Yewen Pu. Communicating design intent using drawing and text. In Proceedings of the 16th Conference on Creativity & Cognition, page 512–519, New York, NY, USA, 2024. Association for Computing Machinery. 3
- [78] John FJ Mellor, Eunbyung Park, Yaroslav Ganin, Igor Babuschkin, Tejas Kulkarni, Dan Rosenbaum, Andy Ballard, Theophane Weber, Oriol Vinyals, and SM Eslami. Unsupervised doodling and painting with improved spiral. arXiv preprint arXiv:1910.01007, 2019. 3
- [79] Daniela Mihai and Jonathon Hare. Learning to draw: Emergent communication through sketching. Advances in Neural Information Processing Systems, 34:7153–7166, 2021. 3
- [80] Chancharik Mitra, Brandon Huang, Trevor Darrell, and Roei Herzig. Compositional chain-of-thought prompting for large multimodal models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14420–14431, 2024. 3
- [81] Kushin Mukherjee, Holly Huey, Xuanchen Lu, Yael Vinker, Rio Aguina-Kang, Ariel Shamir, and Judith Fan. Seva: Leveraging sketches to evaluate alignment between human and machine visual abstraction. In Advances in Neural Information Processing Systems, 2023. 2
- [82] Omar W Nasim. Observing by hand: sketching the nebulae in the nineteenth century. University of Chicago Press,

2019. 1

- [83] Changhoon Oh, Jungwoo Song, Jinhan Choi, Seonghyeon Kim, Sungwoo Lee, and Bongwon Suh. I lead, you help but only with enough details: Understanding user experience

- of co-creation with artificial intelligence. In Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems, page 1–13, New York, NY, USA, 2018. Association for Computing Machinery. 3
- [84] OpenAI. Gpt-4 technical report, 2024. 2, 3, 4, 6
- [85] Dustin Podell, Zion English, Kyle Lacey, A. Blattmann, Tim Dockhorn, Jonas Muller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for highresolution image synthesis. ArXiv, abs/2307.01952, 2023. 2, 3
- [86] Yonggang Qi, Guoyao Su, Pinaki Nath Chowdhury, Mingkang Li, and Yi-Zhe Song. Sketchlattice: Latticed representation for sketch manipulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 953–961, 2021. 2
- [87] Shuwen Qiu, Sirui Xie, Lifeng Fan, Tao Gao, SongChun Zhu, and Yixin Zhu. Emergent graphical conventions in a visual communication game. arXiv preprint arXiv:2111.14210, 2021. 3
- [88] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. CoRR, abs/2103.00020, 2021. 1, 3, 6
- [89] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21(1), 2020. 3
- [90] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022. 2, 3
- [91] Leo Sampaio Ferraz Ribeiro, Tu Bui, John P. Collomosse, and Moacir Antonelli Ponti. Sketchformer: Transformer-based representation for sketched structure. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14141–14150, 2020. 3
- [92] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models, 2022. 1, 2, 3, 6
- [93] Daniel Rose, Vaishnavi Himakunthala, Andy Ouyang, Ryan He, Alex Mei, Yujie Lu, Michael Saxon, Chinmay Sonar, Diba Mirza, and William Yang Wang. Visual chain of thought: bridging logical gaps with multimodal infillings. arXiv preprint arXiv:2305.02317, 2023. 3
- [94] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. Photorealistic text-toimage diffusion models with deep language understanding,

2022. 3

- [95] Patsorn Sangkloy, Nathan Burnell, Cusuh Ham, and James Hays. The sketchy database: Learning to retrieve badly drawn bunnies. ACM Trans. Graph., 35(4), 2016. 2

- [96] Peter Schaldenbrand, James McCann, and Jean Oh. Frida: A collaborative robot painter with a differentiable, real2sim2real planning environment. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 11712–11718, 2023. 3
- [97] Peter Schaldenbrand, Gaurav Parmar, Jun-Yan Zhu, James McCann, and Jean Oh. Cofrida: Self-supervised fine-tuning for human-robot co-painting. In 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE,

2024. 3

- [98] Donald A Schon and Vincent DeSanctis. The reflective practitioner: How professionals think in action, 1986. 2
- [99] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. Laion-5b: An open large-scale dataset for training next generation image-text models, 2022. 3
- [100] Tamar Rott Shaham, Sarah Schwettmann, Franklin Wang, Achyuta Rajaram, Evan Hernandez, Jacob Andreas, and Antonio Torralba. A multimodal automated interpretability agent. In Forty-first International Conference on Machine Learning, 2024. 3
- [101] Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. arXiv preprint arXiv:2403.16999, 2024. 3
- [102] Pratyusha Sharma, Tamar Rott Shaham, Manel Baradad, Stephanie Fu, Adrian Rodriguez-Munoz, Shivam Duggal, Phillip Isola, and Antonio Torralba. A vision check-up for language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14410–14419, 2024. 3, 7
- [103] Jifei Song, Kaiyue Pang, Yi-Zhe Song, Tao Xiang, and Timothy Hospedales. Learning to sketch with shortcut cycle consistency, 2018. 2
- [104] Peter Stone, Gal Kaminka, Sarit Kraus, and Jeffrey Rosenschein. Ad hoc autonomous agent teams: Collaboration without pre-coordination. Proceedings of the AAAI Conference on Artificial Intelligence, 24(1):1504–1509, 2010. 13
- [105] Guoyao Su, Yonggang Qi, Kaiyue Pang, Jie Yang, Yi-Zhe Song, and CVSSP SketchX. Sketchhealer: A graph-tosequence network for recreating partial human sketches. In BMVC, page 5, 2020. 2
- [106] Yanpeng Sun, Qiang Chen, Jian Wang, Jingdong Wang, and Zechao Li. Exploring effective factors for improving visual in-context learning. arXiv preprint arXiv:2304.04748,

2023. 3

- [107] Ivan E. Sutherland. Sketchpad—a man-machine graphical communication system, page 391–408. Association for Computing Machinery, New York, NY, USA, 1998. 3
- [108] Gemini Team. Gemini: A family of highly capable multimodal models, 2024. 2, 3
- [109] Jakob Tholander and Martin Jonsson. Design ideation with ai - sketching, thinking and talking with generative machine

- learning models. In Proceedings of the 2023 ACM Designing Interactive Systems Conference, page 1930–1940, New York, NY, USA, 2023. Association for Computing Machinery. 1
- [110] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms, 2024. 4
- [111] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. ArXiv, abs/2302.13971, 2023. 3
- [112] Barbara Tversky. What do sketches say about thinking? AAAI Spring Symp. Sketch Understanding Worksh., 2002. 2
- [113] Barbara Tversky. Visualizing thought. In Handbook of human centric visualization, pages 3–40. Springer, 2013. 1
- [114] Barbara Tversky, Masaki Suwa, Maneesh Agrawala, Julie Heiser, Chris Stolte, Pat Hanrahan, Doantam Phan, Jeff Klingner, Marie-Paule Daniel, Paul Lee, et al. Sketches for design and design of sketches. Human Behaviour in Design: Individuals, Teams, Tools, pages 79–86, 2003. 1
- [115] Russell Tytler, Vaughan Prain, George Aranda, Joseph Ferguson, and Radhika Gorur. Drawing to reason and learn in science. Journal of Research in Science Teaching, 57(2): 209–231, 2020. 3
- [116] Yael Vinker, Ehsan Pajouheshgar, Jessica Y. Bo, Roman Christian Bachmann, Amit Haim Bermano, Daniel Cohen-Or, Amir Zamir, and Ariel Shamir. Clipasso: Semantically-aware object sketching. ACM Trans. Graph., 41(4), 2022. 1, 3, 6
- [117] Yael Vinker, Yuval Alaluf, Daniel Cohen-Or, and Ariel Shamir. Clipascene: Scene sketching with different types and levels of abstraction. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4146–4156, 2023. 3, 6
- [118] Jiawei Wang and Changjian Li. Contextseg: Sketch semantic segmentation by querying the context with attention. arXiv preprint arXiv:2311.16682, 2023. 6
- [119] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. Red Hook, NY, USA, 2024. Curran Associates Inc. 2, 5
- [120] Holger Winnem¨oller, Jan Eric Kyprianidis, and Sven C. Olsen. Xdog: An extended difference-of-gaussians compendium including advanced image stylization. Comput. Graph., 36:740–753, 2012. 2
- [121] World Wide Web Consortium (W3C). Scalable Vector Graphics (SVG), 1999. 3
- [122] Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. Visual chatgpt: Talking, drawing and editing with visual foundation models. arXiv preprint arXiv:2303.04671, 2023. 3
- [123] Rong Wu, Wanchao Su, Kede Ma, and Jing Liao. Iconshop: Text-guided vector icon synthesis with autoregressive trans-

formers. ACM Transactions on Graphics (TOG), 42:1 – 14,

2023. 3

- [124] Chufeng Xiao, Wanchao Su, Jing Liao, Zhouhui Lian, YiZhe Song, and Hongbo Fu. Differsketching: How differently do people sketch 3d objects? ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH Asia 2022), 41

(4):1–16, 2022. 2

- [125] Saining Xie and Zhuowen Tu. Holistically-nested edge detection. In Proceedings of the IEEE international conference on computer vision, pages 1395–1403, 2015. 2
- [126] XiMing Xing, Chuang Wang, Haitao Zhou, Jing Zhang, Qian Yu, and Dong Xu. Diffsketcher: Text guided vector sketch synthesis through latent diffusion models. In Advances in Neural Information Processing Systems, pages 15869–15889. Curran Associates, Inc., 2023. 3, 6
- [127] Ximing Xing, Haitao Zhou, Chuang Wang, Jing Zhang, Dong Xu, and Qian Yu. Svgdreamer: Text guided svg generation with diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4546–4555, 2024. 3, 6, 7
- [128] Peng Xu, Timothy M. Hospedales, Qiyue Yin, Yi-Zhe Song, Tao Xiang, and Liang Wang. Deep learning for freehand sketch: A survey and a toolbox, 2020. 2
- [129] Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of lmms: Preliminary explorations with gpt-4v(ision). ArXiv, abs/2309.17421, 2023. 2
- [130] Zhengyuan Yang, Jianfeng Wang, Linjie Li, Kevin Lin, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. Idea2img: Iterative self-refinement with gpt-4v (ision) for automatic image design and generation. arXiv preprint arXiv:2310.08541, 2023. 3
- [131] Ran Yi, Yong-Jin Liu, Yu-Kun Lai, and Paul L Rosin. Apdrawinggan: Generating artistic portrait drawings from face photos with hierarchical gans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10743–10752, 2019. 2
- [132] Ran Yi, Yong-Jin Liu, Yu-Kun Lai, and Paul L Rosin. Unpaired portrait drawing generation via asymmetric cycle mapping. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8217– 8225, 2020. 2
- [133] C. Zhang, Weijie Wang, Paul Pangaro, Nikolas Martelaro, and Daragh Byrne. Generative image ai using design sketches as input: Opportunities and challenges. Proceedings of the 15th Conference on Creativity and Cognition,

2023. 1

- [134] Jiahao Zhang, Bowen Wang, Liangzhi Li, Yuta Nakashima, and Hajime Nagahara. Instruct me more! random prompting for visual in-context learning. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2597–2606, 2024. 3
- [135] Peiying Zhang, Nanxuan Zhao, and Jing Liao. Text-tovector generation with neural path representation. ACM Trans. Graph., 43(4), 2024. 3
- [136] Yuanhan Zhang, Kaiyang Zhou, and Ziwei Liu. What makes good examples for visual in-context learning? Ad-

- vances in Neural Information Processing Systems, 36: 17773–17794, 2023. 3
- [137] Zhengming Zhang, Xiaoming Deng, Jinyao Li, Yukun Lai, Cuixia Ma, Yongjin Liu, and Hongan Wang. Stroke-based semantic segmentation for scene-level free-hand sketches. Vis. Comput., 39(12):6309–6321, 2022. 6
- [138] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-ofthought reasoning in language models. arXiv preprint arXiv:2302.00923, 2023. 3
- [139] Yixiao Zheng, Kaiyue Pang, Ayan Das, Dongliang Chang, Yi-Zhe Song, and Zhanyu Ma. Creativeseg: Semantic segmentation of creative sketches. IEEE Transactions on Image Processing, 33:2266–2278, 2024. 6
- [140] Tao Zhou, Chen Fang, Zhaowen Wang, Jimei Yang, Byungmoon Kim, Zhili Chen, Jonathan Brandt, and Demetri Terzopoulos. Learning to sketch with deep q networks and demonstrated strokes. ArXiv, abs/1810.05977, 2018. 2, 3

### SketchAgent: Language-Driven Sequential Sketch Generation Supplementary Material

### Table of Contents

- A. Technical Details 1
- B. More Results and Analysis 1

- B.1. Quantitative Text-Conditioned Analysis 2
- B.2. Sequential sketching . . . . . . . . . 7
- B.3. Human-Agent Collaborative Sketching 12
- B.4. Chat-Based Editing . . . . . . . . . . 13

- C. Ablation Study 14
- D. Prompts and More Results 15

IRB Disclosure We received IRB approvals for all user studies, from all of the institutions involved. Accordingly, we took measures to ensure participant anonymity and refrained from showing them potentially offensive content.

#### A. Technical Details

We will publicly release the full source code, including our interactive platform. Our default backbone model is Claude3.5-Sonnet (version 20240620) [3]. We use the official API of Anthropic, with an average cost of $0.05 per sketch. We employ CairoSVG [59] for rendering the SVG onto the canvas. Our output sketches are also provided in SVG format to facilitate further editing if needed. SketchAgent generates a complete sketch in approximately 20 seconds, with individual strokes in collaborative mode taking about 8 seconds each. For the CLIP zero-shot classification, we use the clip-vit-large-patch14 model from Hugging Face [27]. Our canvas is defined as a 50 × 50 grid with numbers labeled on the bottom and left edges. Each cell corresponds to a patch of size 12 × 12 pixels, chosen to ensure a clear display of the grid numbers along the edges. This configuration results in a 612 × 612 pixels grid, with the drawing area confined to a 600 × 600 pixel range. All prompts used in our method are provided in Figs. 51, 52 and 55. The examples provided to the agent in the system and user prompts are visualized in Fig. 15 and Fig. 16 respectively.

We use Claude3.5-Sonnet in its default settings, which results in significant variability in results, given the highly diverse nature of LLMs. For example, in Fig. 17, we present 12 sketches produced by our method for the concept “rabbit”, demonstrating high diversity in pose, structure, and quality. To generate variations in the experiments described in Section 5.1 of the main paper (where we applied our method 10 times per category), we use the default settings of Claude3.5-Sonnet. However, during controlled experimental conditions, we reduce variability by setting the temperature to 0 and top k to 1, ensuring deterministic outputs. For general use, we recommend the stochastic version to encourage more varied and creative outputs.

[Figure 73]

- Figure 15. Visualization of single-stroke primitives used in the system prompt to introduce the grid and sketching language to the agent.

|[Figure 74]|
|---|

- Figure 16. Visualization of the simple sketch of a house provided as an in-context example, represented with our sketching language through the user prompt.

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

- Figure 17. Sketch variability. Example of twelve different sketches produced for the concept “rabbit” by SketchAgent, with the same settings.

#### B. More Results and Analysis

As described in Section 5.1 of the main paper, SketchAgent is capable of generating sketches for a wide range of concepts that extend beyond standard categories. Here we provide additional results to support this claim. We define three unique categories that require general knowledge: Scientific Concepts, Diagrams, and Notable Landmarks, and utilize ChatGPT-4o to produce 10 ran-

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Double-slit experiment

Circuit diagram

[Figure 97]

[Figure 98]

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

Pendulum motion

Flowchart

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

[Figure 120]

Photosynthesis

Organizational chart

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

[Figure 132]

ER diagram (EntityRelationship)

DNA replication

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

[Figure 144]

Newton s laws of motion

Venn diagram

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

Electromagnetic spectrum

Mind map

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

Plate tectonics

Gantt chart

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

Quantum entanglement

Network topology diagram

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

Cell division (mitosis)

Pie chart

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

Black hole formation

Decision tree

[Figure 205]

[Figure 206]

Figure 18. Randomly selected sketches of scientific concepts. Ten textual concepts were randomly selected using GPT-4o. Five sketches were generated per concept, showcasing the variability and diversity of the outputs.

Figure 19. Randomly selected sketches of diagrams across fields. Ten textual concepts were randomly selected using GPT-4o. Five sketches were generated per concept, showcasing the variability and diversity of the outputs.

dom textual concepts for each category, resulting in the following random concepts:

- • Scientific Concepts: Double-slit experiment, Pendulum motion, Photosynthesis, DNA replication, Newton’s laws of motion, Electromagnetic spectrum, Plate tectonics, Quantum entanglement, Cell division (mitosis), Black hole formation.
- • Diagrams: Circuit diagram, Flowchart, Organizational chart, ER diagram (Entity-Relationship), Venn diagram, Mind map, Gantt chart, Network topology diagram, Pie chart, Decision tree.
- • Notable Landmarks: Taj Mahal, Eiffel Tower, Great Wall of China, Pyramids of Giza, Statue of Liberty, Colosseum, Sydney Opera House, Big Ben, Mount Fuji, Machu Picchu.

We generate five sketches for each concept (producing 50 sketches per category) by applying our method five times using its default

settings. Figures 18 to 20 present the results for Scientific Concepts, Diagrams, and Notable Landmarks, respectively. The resulting sketches generally depict the concepts well, demonstrating diversity in the outputs. As can be seen, our method can generate a diverse set of different types and instances per concept (see double-slit experiment, pendulum motion, Electromagnetic spectrum in Fig. 18 and Flowchart, Network typology diagram in Fig. 19). Naturally, within each set, some concepts were depicted very successfully, while some outputs were less successful (e.g., Statue of Liberty, photosynthesis, pie chart).

##### B.1. Quantitative Text-Conditioned Analysis

In Section 5.1 of the main paper, we presented a quantitative analysis of text-conditioned sketch generation across 50 selected categories from the QuickDraw dataset [54]. Here, we provide addi-

ant binoculars

0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 0 0 0 0 0 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 10 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

Taj Mahal

[Figure 212]

bus

camel dog eye fish

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

Eiffel Tower

giraffe

[Figure 218]

headphones mailbox octopus

TrueLabel

onion

pear potato

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

Great Wall of China

[Figure 224]

school bus shark

smiley face snake spider

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

Pyramids of Giza

squiggle

[Figure 230]

giraffe

snake

dog

pear

ant

squiggle

potato

spider

bus

fish

binoculars

camel

shark

eye

octopus

onion

headphones

smileyface

schoolbus

mailbox

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

Predicted Label

Statue of Liberty

[Figure 236]

- Figure 21. Confusion matrix (showing top 10 confused classes) for the set of 500 sketches generated with SketchAgent default settings (Claude3.5-Sonnet) across 50 categories

|[Figure 237]<br><br>shark fish<br><br>0.0%|[Figure 238]<br><br>octopus spider<br><br>30.0%|[Figure 239]<br><br>snake squiggle<br><br>30.0%|[Figure 240]<br><br>pear onion<br><br>50.0%|[Figure 241]<br><br>potato smiley face<br><br>50.0%|[Figure 242]<br><br>school bus bus<br><br>50.0%|
|---|---|---|---|---|---|

- Figure 22. Visualization of sketches from the six most confused classes. The correct category is highlighted in green, while the misclassified category is highlighted in red.

|[Figure 243]<br><br>fish<br><br>100.0%|[Figure 244]<br><br>house<br><br>90.0%|[Figure 245]<br><br>umbrella<br><br>90.0%|[Figure 246]<br><br>table<br><br>90.0%|[Figure 247]<br><br>television<br><br>70.0%|[Figure 248]<br><br>camera<br><br>70.0%|
|---|---|---|---|---|---|
|[Figure 249]<br><br>eye<br><br>60.0%|[Figure 250]<br><br>bus<br><br>60.0%|[Figure 251]<br><br>lighthouse<br><br>60.0%|[Figure 252]<br><br>dog<br><br>50.0%|[Figure 253]<br><br>hand<br><br>50.0%|[Figure 254]<br><br>tennis racquet<br><br>50.0%|
|[Figure 255]<br><br>goatee<br><br>40.0%|[Figure 256]<br><br>car<br><br>40.0%|[Figure 257]<br><br>frog<br><br>30.0%|[Figure 258]<br><br>giraffe<br><br>30.0%|[Figure 259]<br><br>hot air balloon<br><br>30.0%|[Figure 260]<br><br>airplane<br><br>20.0%|

- Figure 23. Visualization of the top recognized classes for the set of 500 sketches generated with our default settings (Claude3.5Sonnet) across 50 categories.

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

Colosseum

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

Sydney Opera House

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

Big Ben

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

Mount Fuji

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

Machu Picchu

[Figure 290]

Figure 20. Randomly selected sketches of notable landmarks. Ten textual concepts were randomly selected using GPT-4o. Five sketches were generated per concept, showcasing the variability and diversity of the outputs.

tional details, visual examples, and further analysis of the experiment. We begin by providing further analysis of the CLIP classification rates of our default settings (Claude3.5-Sonnet) to explore recognition patterns. Figure 21 shows the confusion matrix (top 10 confused categories out of 50) for our set of 500 sketches. The most commonly confused classes are: “shark”, which was often misclassified as a “fish”, “octopus”, which was frequently identified as a “spider”; and “snake”, which was misclassified as a “squiggle”. These confused classes often fall within highly related classes (such as a fish and a shark, or a school bus and a bus), suggesting that our method struggles with emphasizing distinctive features, likely due to its inherently abstract style. In Fig. 22, we visualize sketches from the six most confused classes with the correct class shown in green and the misclassified class shown in red. Figure 23 visualize the 10 top recognized classes.

The class “fish” was correctly identified across all seeds, followed by “house”, “umbrella”, and “table”, which were correctly recognized in 90% of trials (9 out of 10). Recognition rates for other classes ranged from 70% to 20%. In Section 5.1 of the main paper, we compared the performance of different multimodal LLMs (GPT-4o-mini, GPT-4o, and Claude3-Opus) using our default prompts and settings. Figure 24 visualizes the eight most recognized classes across all backbone models. For this analysis, we select the top two recognized categories from each model and display the sketches with the highest classification probability for each. Note that some categories were at the top two of multiple models (such as house, fish, and eye), in that case, we

GPT-4o mini

Claude3 Opus

Claude3.5 Sonnet

GPT-4o

|[Figure 291]|[Figure 292]|[Figure 293]|[Figure 294]|
|---|---|---|---|
|[Figure 295]|[Figure 296]|[Figure 297]|[Figure 298]|
|[Figure 299]|[Figure 300]|[Figure 301]|[Figure 302]|
|[Figure 303]|[Figure 304]|[Figure 305]|[Figure 306]|
|[Figure 307]|[Figure 308]|[Figure 309]|[Figure 310]|
|[Figure 311]|[Figure 312]|[Figure 313]|[Figure 314]|
|[Figure 315]|[Figure 316]|[Figure 317]|[Figure 318]|
|[Figure 319]|[Figure 320]|[Figure 321]|[Figure 322]|

housefisheyetablegoateeumbrellaairplanebus

Figure 24. Visualization of sketches from the most recognized classes across all backbone models. The classes selected based on the two most recognizable classes in each model.

GPT-4o mini

Claude3 Opus

Claude3.5 Sonnet

GPT-4o

|[Figure 323]|[Figure 324]|[Figure 325]|[Figure 326]|
|---|---|---|---|
| |[Figure 327]|[Figure 328]|[Figure 329]|
|[Figure 330]|[Figure 331]|[Figure 332]|[Figure 333]|
|[Figure 334]|[Figure 335]|[Figure 336]|[Figure 337]|
|[Figure 338]|[Figure 339]|[Figure 340]|[Figure 341]|
|[Figure 342]|[Figure 343]|[Figure 344]|[Figure 345]|
| |[Figure 346]|[Figure 347]|[Figure 348]|
| |[Figure 349]|[Figure 350]|[Figure 351]|

snakesaxophonesharkraccoonoctopusdolphinwatermelonschoolbus

Figure 25. Visualization of sketches from the least recognized classes across all backbone models. The classes selected based on the two least recognizable classes in each model.

select the next top recognized category. The chosen top two categories for each model are: GPT-4o: house and eye, GPT-4o-mini: table and goatee, Claude3-Opus: fish and airplane, Claude3.5Sonnet: umbrella and bus. Similarly, Figure 25 highlights the least recognized categories, chosen using the same selection criteria. The chosen worst two categories for each model are: GPT4o: snake and school bus, GPT-4o-mini: saxophone and raccoon, Claude3-Opus: octopus and dolphin, Claude3.5-Sonnet: shark and watermelon. Note that snake, octopus, and shark, were all confused under at least three of the four backbones. The visualizations align well with the quantitative results presented in Table 1 of the main paper. Among the Anthropic models, Claude3.5Sonnet produces better sketches than Claude3-Opus, and among the GPT models, GPT-4o outperforms GPT-4o-mini. Overall, the two best-performing backbone models are Claude3.5-Sonnet and GPT-4o. Interestingly, the sketching style differs between GPT-4o and Claude3.5-Sonnet. Although Claude3.5-Sonnet (our default backbone model) seems to yield the best results, this may be due to the fact that our method was primarily developed using this model. Consequently, the prompts we use were optimized for Claude3.5-

Sonnet, and improved results for other models might be achievable with additional prompt engineering. We leave this exploration for future work.

SketchAgent using an open-source model While opensource models currently lag behind commercial closed-source models, they are rapidly advancing in size and capability, showing significant potential for facilitating sketch generation.

We begin by experimenting with Llama-3.2-11B-Vision [24], a multimodal large language model developed by Meta AI, as SketchAgent’s backbone model. When used with our default prompts and framework, the model fails to generate meaningful sketches, frequently replicating the in-context example of a house provided in the user prompt (examples are shown in Fig. 26).

We therefore turn into exploring a larger available open-source model, Llama-3.1-405B-Instruct [24]. This model resulted in better sketches that manage to generalize well beyond the in-context example. We generated 500 random sketches and computed their classification rates using CLIP, as described in Section 5.1 of the

house backpack eye airplane

|[Figure 352]|[Figure 353]|[Figure 354]|[Figure 355]|
|---|---|---|---|

- Figure 26. Sketches generated using Llama-3.2-11B-Vision as our backbone models. The model frequently replicates the in-context example of a house provided in the user prompt.

|[Figure 356]|[Figure 357]|[Figure 358]|[Figure 359]|
|---|---|---|---|

house goatee eye fish

|[Figure 360]|[Figure 361]|[Figure 362]|[Figure 363]|
|---|---|---|---|

hand potato lighthouse skyscraper

- Figure 27. Visualization of the eight top recognized classes for the set of 500 sketches generated with Llama-3.1-405B-Instruct as our backbone model.

main paper. The results yielded lower scores compared to commercial models, with an average Top-1 recognition accuracy of 0.052 ± 0.03 and a Top-5 recognition accuracy of 0.1 ± 0.03. Visualizations of the top eight correctly classified classes are shown in Fig. 27, and the top eight most confused classes are presented in Fig. 28. Despite the lower recognition rates, the generated sketches are reasonable and visually coherent, showing promise as open-source models continue to improve. This experiment demonstrates the potential for SketchAgent to be implemented using publicly available models. While its performance does not match that of our default backbone, SketchAgent can still function effectively with open-source models, albeit with a slight compromise in performance.

Direct Prompting Analysis In Section 5.1 of the main paper, we compared our method to directly prompting Claude3.5-Sonnet for generating SVGs with a sketch-like appearance. In Fig. 29, we extend this analysis by visualizing the results of direct prompting with the other backbone models used in the quantitative experiment. This demonstrates how different models respond to direct SVG generation prompts. We present examples for the concepts “giraffe” and “lighthouse”, using the following SVG generation prompt: “Write an SVG string of a <concept>.”. For sketch-like SVGs, we used the same prompt as in the main paper (“Write an SVG string that draws a sketch of a <concept>. Use only black and white colors”). As shown, the outputs across all methods often feature uniform and precise geometric shapes (e.g., ellipses, triangles), which diverge from the natural variability and expres-

dolphin peanut pond table

|[Figure 364]|[Figure 365]|[Figure 366]|[Figure 367]|
|---|---|---|---|

fish snake pear river

|[Figure 368]|[Figure 369]|[Figure 370]|[Figure 371]|
|---|---|---|---|

Figure 28. Visualization of sketches from the eight least recognized classes for the set of 500 sketches generated with Llama3.1-405B-Instruct as our backbone model.

siveness characteristic of hand-drawn sketches. Interestingly, the SVGs generated by GPT-4o and Claude3.5-Sonnet appear more expressive and visually appealing compared to those produced by GPT-4o-mini and Claude3-Opus, aligning well with the performance differences observed in sketch generation.

2AFC experiment In section 5.1 of the main paper, we also presented a 2AFC experiment to evaluate how “human-like” our agent’s sketches appear compared to sketch-like SVGs generated with direct prompting and human sketches from the QuickDraw dataset. We utilize 50 sketches from 50 classes per method. We recruited a total of 150 workers through Amazon Mechanical Turk, each participating in 50 test sessions, as presented in Fig. 30. Before starting the test, workers were presented with instructions (Fig. 31). We filtered participants with a Mturk approval rate of 99.9% or higher and with a record of more than 1,000 surveys. Workers were paid $0.5 for completing the full test.

GPT-4o -mini

Claude3 Opus

Claude3.5 -Sonnet

GPT-4o

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

SVG “giraffe”

[Figure 376]

[Figure 377]

[Figure 378]

Sketch-like SVG “giraffe”

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

SVG “lighthouse”

[Figure 384]

Sketch-like SVG “lighthouse”

[Figure 385]

[Figure 386]

[Figure 387]

Figure 29. Direct prompting for SVG generation across different backbone models. The SVGs generated by GPT-4o and Claude3.5-Sonnet appear more expressive and visually appealing compared to those produced by GPT-4o-mini and Claude3-Opus, aligning well with the performance differences observed in sketch generation.

[Figure 388]

Figure 31. 2AFC instructions to users.

[Figure 389]

Figure 30. An example of our 2AFC session.

4 strokes 7 strokes

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

|Human|
|---|

1 stroke

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

15 strokes

[Figure 415]

[Figure 416]

[Figure 417]

7 strokes

[Figure 418]

4 strokes

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

SketchAgent

15 strokes

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

Figure 32. Distribution of human sketches [54] (top) and SketchAgent’s sketches (bottom) based on the number of strokes per sketch. Representative examples are shown for sketches drawn with 1, 4, 7, and 15 strokes. Notably, in the QuickDraw dataset, single-stroke sketches often consist of a single long continuous line.

##### B.2. Sequential sketching

- In Section 5.2 of the main paper, we analyze the sequential nature of our generated sketches. In Figs. 37 to 39, we present additional visualizations of annotated sequential sketches of 48 randomly selected animals, with the presented sketches also chosen randomly. As illustrated, due to the extensive prior knowledge of the backbone model, SketchAgent provides meaningful textual annotations for each stroke and sketches in a logical order. Typically, more significant body parts, such as the head and body, are drawn first. We next provide more details and visualizations of the quantitative analysis shown in Figure 11 of the main paper. Figure 32 displays histograms of the number of strokes in QuickDraw sketches (top) and our generated sketches (bottom), as shown in the main paper. Alongside these histograms, we include visualizations of sketches drawn with 1, 4, 7, and 15 strokes. Notably, in the QuickDraw dataset, single-stroke sketches often consist of a single long continuous line, making them recognizable after the first stroke. In contrast, sketches with a larger number of strokes rarely feature long continuous lines. For such cases, the sequential process of adding strokes gradually makes the sketches recognizable

after several strokes. Figures 33 to 36 also demonstrates the sequential sketching process for both QuickDraw sketches and those generated by our method, providing a visual context for the trends observed in Figure 11.

[Figure 441]

HumanHumanHumanOursOursOurs

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

- Figure 33. Sequential four-stroke sketches of a pear, purse, and screwdriver, created by humans [54] and by SketchAgent.

HumanHumanHumanOursOursOurs

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

- Figure 34. Sequential five-stroke sketches of a pear, purse, and screwdriver, created by humans [54] and by SketchAgent.

[Figure 453]

HumanHumanHumanOursOursOurs

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

- Figure 35. Sequential six-stroke sketches of a television, bed, and peanut, created by humans [54] and by SketchAgent.

HumanHumanHumanOursOursOurs

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

- Figure 36. Sequential seven-stroke sketches of a backpack, fish, and house, created by humans [54] and by SketchAgent.

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

###### Figure9 37

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

###### Figure 38

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

###### Figure 39

[Figure 513]

##### B.3. Human-Agent Collaborative Sketching

## Let’s Sketch a Sailboat

- In Section 5.3 of the main paper, we demonstrate that humans and SketchAgent can effectively collaborate to produce meaningful sketches through genuine interaction. The sketching interface (Fig. 40) consists of a 400 × 400 plain canvas shared between the user and the agent. It highlights the current concept to be sketched and displays the active sketching mode, which can be either solo or collab. Additionally, the interface includes a submit button that allows users to finalize the sketch when they consider it complete. In solo mode, users independently sketch the given concept using green strokes. In collab mode, users and SketchAgent take turns adding strokes, with user strokes displayed in green and agent strokes in pink. At the beginning of each session, users are provided with general instructions about the experiment and the types of sketches they will be asked to draw (Fig. 41). Specifically, they are instructed to create recognizable sketches, stroke by stroke, while minimizing the number of strokes by planning ahead. Next, users begin by sketching two warm-up concepts in both “solo” and “collab” modes to familiarize themselves with the web environment. Each session includes all eight primary concepts in a randomized order, resulting in a total of 10 sketches per user (including the two warm-up sketches). The concepts are as follows:

[Figure 514]

[Figure 515]

User Turn! Draw a stroke

Figure 40. Screenshot of our web interface.

[Figure 516]

- • Warm up concepts: jellyfish, house
- • Text concepts: butterfly, fish, rabbit, duck, sailboat, coffee mug, eyeglasses, car

For each concept, participants sketched in both solo and collaboration modes, with the order of these modes randomized to mitigate potential biases. The 30 users are counterbalanced: 15 users produced the first stroke in collaboration with the agent (and all odd-numbered strokes thereafter), while the other 15 users produced the second stroke in collaboration with the agent (and all even-numbered strokes). In total we collected responses from 32 users, however, two users were excluded from the analysis due to incomplete sketching sessions, leaving a total of 30 users. In Fig. 46 we present examples of sketches from each mode, focusing on those with high recognition rates across categories. Solo sketches are shown in green, agent-only sketches in pink, and collaborative sketches are depicted with a combination of both colors. To analyze “collab” and “solo” sketches, we rendered all complete and partial sketches (agent-only and user-only strokes) from SVG to pixel images. We then utilized a CLIP zero-shot classifier, as described in the main paper, to evaluate how effectively each sketch represented the intended concept. Tab. 3 summarizes the results (as shown in the graph in Fig. 12B of the main paper). These results highlight that both users and the agent contributed meaningfully to the final “collab” sketches. Variants of collaborative sketches containing only the user’s strokes or only the agent’s strokes were found to contain substantially less semantic information about the intended concept compared to the complete collaborative sketches. Additionally, the average number of strokes per completed sketch was consistent across modes, indicating similar levels of complexity. Specifically, the average stroke counts were as follows: collaborative full sketches: 7.333; solo agent sketches: 7.321; solo user sketches: 7.708. This suggests that collaboration produces sketches with a level of detail comparable to those created independently.

Figure 41. User instructions in the sketching interface.

We analyze the classification confusion patterns for collaborative and solo sketches (240 sketches each) in Figs. 43 and 44, revealing similar trends. For instance, a “coffee cup” was often misclassified as a “teapot”, a “car” was frequently identified as a “turtle”, and a “duck” was misclassified as a “bird”. Additionally, “car” sketches were sometimes mistaken for a “pickup truck”. In most cases, the misclassifications occur within closely related categories (e.g., “car” to “truck” or “pickup truck”) or among categories sharing similar visual structures (e.g., the rounded dome and four base components of a “car” resembling a “turtle”). This highlights a challenge in emphasizing distinctive features within specific categories, likely stemming from the inherently abstract nature of our sketches.

Figure 42 presents the recognition rates with 95% confidence interval (CI) error bars for each concept across all three sketching conditions: “collab” (blue), “solo-user” (green), and “solo-agent” (pink). Overall, the recognition rates for collaborative sketches are comparable to those produced by users alone or the agent alone for each unique category. Notably, sketches of “car” exhibit the lowest recognition rate across all conditions. This is likely due to confusion with semantically similar categories, such as “truck”, “pickup truck”, “airplane”, and “speedboat”, as indicated by the

###### Variation Recognition Rate 95% CI

Collab full sketch 0.75 [0.61, 0.85] Collab agent-only strokes 0.10 [0.06, 0.19] Collab user-only strokes 0.13 [0.07, 0.23]

Table 3. Recognition rate and 95% CI across collaborative full and partial sketches. In collaborative sketches, keeping agent-only strokes or user-only strokes significantly reduces recognizability.

1.0

0.8

recognitionrate

0.6

0.4

0.2

0.0

butterfly car coffee mug duck eyeglasses fish rabbit sailboat

| |
|---|

| |
|---|

| |
|---|

collab full solo user full solo agent full

Figure 42. CLIP recognition rate by class for collaborative, solo user, and solo agent full sketches.

confusion matrices. Similarly, as discussed earlier, “coffee cup” and “duck” are frequently misclassified as related categories with overlapping visual features.

We observe that in some cases of collaborative sketching (14 out of 240 sketches), the agent-human pair faces challenges in interpreting each other’s intentions and the meanings of strokes. Achieving effective collaboration and communication between different parties [45] is a challenge that often requires prior planning, social reasoning, and repeated interactions to establish shared intentions and representations. These complex processes continue to be studied across various contexts, including in interactions between humans [50, 58, 76], between humans and agents [15], and between agents [104]. Fig. 45 highlights the few collaborative sketches where the CLIP classification is correct, but the agent and user appear to lack a shared understanding of different stroke groups, resulting in the conflicting creation of duplicate concept components (i.e. two heads).

##### B.4. Chat-Based Editing

In section 5.4 of the main paper we demonstrate chat-based editing using SketchAgent. Below, we provide more details about the implementation of the experiment we performed. To enable chat editing, we use the following prompt: “<editing instruction>. Describe the location of the added concepts first in <thinking> tags. Only provide the added strokes. Respond in the same format as before. Be concise.”, where <editing instruction> contains the desired edit such as “Add glasses to the given cat”. The chosen objects per category, as well as the editing prompts, are provided: • Animals: fish, bird, cat. Editing instruction: “Add glasses”,

“Add a hat”, “Add a skirt”.

0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 2 0 0

butterfly

- 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 2 0 0 0

- 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

- 0 0 0 0 0 0 0 0 0 2 1 0 0 0 0 0 0 0 0 0 0

0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 2 0 0

- 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

car coffee cup

duck eyeglasses fish rabbit

sailboat bee bird

TrueLabel

- 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
- 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

camel coffee mug

dolphin

finger mouse

piano sea turtle

speedboat squiggle teapot underwear

dolphin

squiggle

sailboat

piano

rabbit

speedboat

fish

butterfly

finger

underwear

teapot

eyeglasses

duck

seaturtle

car

mouse

bird

camel

bee

coffeecup

coffeemug

Predicted Label

- Figure 43. Confusion matrix from CLIP classification with categories from the QuickDraw dataset for 240 collaborative sketches across 8 categories.

butterfly

car

duck

eyeglasses

fish

rabbit

sailboat

bird

coffeemug

goatee

golfclub

mouse

ocean

penguin

pickuptruck

shark

square

squiggle

teapot

truck

Predicted Label

butterfly car duck eyeglasses fish rabbit sailboat

bird coffee mug

goatee golf club mouse ocean

penguin pickup truck

shark square squiggle teapot truck

TrueLabel

- 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0

- 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 0 0 0 0 3 0 0 0 0 0 0 0 2 0 0 0 0 0 2 0 0 0 0 0 0
- 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 2 0 0

- 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 2 0 0

- 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
- 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0

- 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
- 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

- Figure 44. Confusion matrix from CLIP classification with categories from the QuickDraw dataset for 240 solo user sketches across 8 categories.

- • Outdoor: bus, building, boat. Editing instruction: “Add a tree to the left of the <concept>”, “Add a sun on the top right, above the <concept>”, “Add another smaller <concept> to the right of this <concept>”.
- • Indoor: shelf, nightstand, table. Editing instruction: “Add a coffee mug on the top of the <concept>”, “Add a lamp on the top of the <concept>”, “Add an indoor plant to the left of the <concept>”.

The resulting edited sketches are shown in Figure 47.

|rabbit<br><br>|rabbit<br><br>|duck|duck|
|---|---|---|---|

- Figure 45. Examples of sketches created in “collab” mode that were correctly classified by CLIP but considered unsuccessful as collaborations due to conflicting agent-user interpretations of subcomponents.

|[Figure 517]| | |
|---|---|---|
|[Figure 518]| | |
|[Figure 519]| | |
|[Figure 520]| | |
|[Figure 521]| | |
|[Figure 522]| | |
|[Figure 523]| | |
|[Figure 524]| | |

eyeglasses

solo user solo agent collab

duck

sailboat

butterfly

rabbit

coffee mug

car

fish

- Figure 46. Examples of sketches from our collaborative human study that received high recognition rates. From left to right are sketches drawn in “solo” mode by users, “solo” mode by the agent, and collaboratively by both.

“Tree to the left” “Sun on top right” “Smaller <concept>” “to the right”

“Add glasses” “Add a hat” “Add a skirt”

“Coffee mug on top” “Lamp on top” “Plant to the left”

“Building” “Nightstand” “Cat”

[Figure 525]

[Figure 526]

[Figure 527]

“Boat” “Shelf” “Bird”

[Figure 528]

[Figure 529]

[Figure 530]

“Bus” “Table” “Fish”

[Figure 531]

[Figure 532]

[Figure 533]

Figure 47. Chat-based sketch editing. We iteratively prompt SketchAgent to add objects to sketches through chat dialogues.

#### C. Ablation Study

In Section 6 of the main paper, we presented an ablation study by systematically removing key components of our method and computing the resulting classification rates. Here, we provide further analyses and discussions on the ablation study.

Table 2 in the main paper shows the CLIP classification rates for 500 sketches (across 50 categories) per experiment. In Fig. 48 we include a visualization of six sketches from six different concepts, covering both structures and animals. As shown, incorporating chain-of-thought reasoning and our in-context example of a house significantly enhances the quality of the results.

We find that the examples used in in-context learning (ICL) can influence both the quality and appearance of the generated sketches, suggesting an interesting direction for future research. Here, we analyze the impact of different types of in-context examples. To investigate whether the theme of the in-context example affects the output (e.g., whether using a house example aids in sketching related concepts like a hospital or if using a cat example helps with sketching other animals), we constructed an alternative sketch of a cat. This sketch used the same number of strokes as the house example to isolate the effect of the theme from complexity. We then applied our method using this alternative example in ICL. In Fig. 49 we illustrate the influence of different ICL examples on related concepts. The example used in each experiment is shown

adding more details and then applied our method with the new, more complex example. The results are presented in Fig. 50. As shown, when a more detailed example is used, the generated sketches tend to overfit, closely replicating the original cat sketch. In contrast, using a simpler example leads to greater variation in the output.

tiger frog dog hospital lighthouse skyscraper

|[Figure 534]|[Figure 535]|[Figure 536]|[Figure 537]|[Figure 538]|[Figure 539]|
|---|---|---|---|---|---|
|[Figure 540]|[Figure 541]|[Figure 542]|[Figure 543]|[Figure 544]|[Figure 545]|
|[Figure 546]|[Figure 547]|[Figure 548]|[Figure 549]|[Figure 550]|[Figure 551]|
|[Figure 552]|[Figure 553]|[Figure 554]|[Figure 555]|[Figure 556]|[Figure 557]|

w/o System Prompt

w/o CoT

ICL example cat tiger frog dog bear

|[Figure 558]|[Figure 559]|[Figure 560]|[Figure 561]|[Figure 562]|[Figure 563]|
|---|---|---|---|---|---|

Modified ICL

SketchAgent (Full)

|[Figure 564]|[Figure 565]|[Figure 566]|[Figure 567]|[Figure 568]|[Figure 569]|
|---|---|---|---|---|---|

- Figure 48. Visualization of sketches produced in different cases of our ablation study.

on the left, with the top figure presenting the effect on animal concepts and the bottom figure depicting the effect on structures. The results indicate that animal sketches are generally more influenced by an animal-based in-context example. For instance, the eyes in the generated sketches tend to resemble the eyes of the cat example more closely, while they vary more when a house example is used. However, there is no definitive conclusion regarding the overall quality or recognizability of these results. Conversely, for structures (bottom), the use of the cat example seems to result in smoother and more rounded shapes, while sketches generated using the house example generally appear more refined and cohesive.

|[Figure 570]|[Figure 571]|[Figure 572]|[Figure 573]|[Figure 574]|[Figure 575]|
|---|---|---|---|---|---|
|[Figure 576]|[Figure 577]|[Figure 578]|[Figure 579]|[Figure 580]|[Figure 581]|

ICL example cat tiger frog dog bear

|[Figure 582]|[Figure 583]|[Figure 584]|[Figure 585]|[Figure 586]|[Figure 587]|
|---|---|---|---|---|---|
|[Figure 588]|[Figure 589]|[Figure 590]|[Figure 591]|[Figure 592]|[Figure 593]|

ICL example house church lighthouse hospital skyscraper

- Figure 49. ICL example ablation study. We examine the impact of changing the concept in the ICL example (e.g., from a house to a cat) on the generation of related concepts. The example used in each experiment is shown on the left, with the top figure illustrating the effect on animal concepts and the bottom figure showing the effect on structural concepts.

ICL example house church lighthouse hospital skyscraper

|[Figure 594]|[Figure 595]|[Figure 596]|[Figure 597]|[Figure 598]|[Figure 599]|
|---|---|---|---|---|---|

|[Figure 600]|[Figure 601]|[Figure 602]|[Figure 603]|[Figure 604]|[Figure 605]|
|---|---|---|---|---|---|

Figure 50. ICL example ablation study. We examine the impact of varying the complexity of the sketch presented in the ICL example while keeping the semantic concept (a cat) constant. The example used in each experiment is shown on the left, with the top figure illustrating the effect on animal concepts and the bottom figure showing the effect on structural concepts.

#### D. Prompts and More Results

We present the full prompts used in our system, as well as our randomly generated sketches used for the quantitative evaluation presented in Section 5.1 of the main paper, and the full set of sketches made by users and in collaborative mode from our human study.

We also examine the impact of example complexity, specifically how using a more detailed sketch with additional strokes affects the output. To test this, we enhanced the cat example by

You are an expert artist specializing in drawing sketches that are visually appealing, expressive, and professional. You will be provided with a blank grid. Your task is to specify where to place strokes on the grid to create a visually appealing sketch of the given textual concept. The grid uses numbers (1 to res) along the bottom (x axis) and numbers (1 to res) along the left edge (y axis) to reference specific locations within the grid. Each cell is uniquely identified by a combination of the corresponding x axis numbers and y axis number (e.g., the bottom-left cell is ’x1y1’, the cell to its right is ’x2y1’). You can draw on this grid by specifying where to draw strokes. You can draw multiple strokes to depict the whole object, where different strokes compose different parts of the object. To draw a stroke on the grid, you need to specify the following: Starting Point: Specify the starting point by giving the grid location (e.g., ’x1y1’ for column 1, row 1). Ending Point: Specify the ending point in the same way (e.g., ’xresyres’ for column res, row res). Intermediate Points: Specify at least two intermediate points that the stroke should pass through. List these in the order the stroke should follow, using the same grid location format (e.g., ’x6y5’, ’x13y10’ for points at column 6 row 5 and column 13 row 10). Parameter Values (t): For each point (including the start and end points), specify a t value between 0 and 1 that defines the position along the stroke’s path. t=0 for the starting point. t=1 for the ending point. Intermediate points should have t values between 0 and 1 (e.g., ”0.3 for x6y5, 0.7 for x13y10”). Examples: To draw a smooth curve that starts at x8y6, passes through x6y7 and x6y10, ending at x8y11: Points = [’x8y6’, ’x6y7’, ’x6y10’, ’x8y11’] t values = [0.00,0.30,0.80,1.00] To close this curve into an ellipse shape, you can add another curve: Points = [’x8y11’, ’x11y10’, ’x11y7’, ’x8y6’] t values = [0.00,0.30,0.70,1.00] To draw a large circle that starts at x25y44 and ends at x25y44, passing through the cells x32y41, x35y35, x31y29, x25y27, x19y29, x15y35, x18y41: Points = [’x25y44’, ’x32y41’, ’x35y35’, ’x31y29’, ’x25y27’, ’x19y29’, ’x15y35’, ’x18y41’, ’x25y44’] t values = [0.00, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.00] To draw non-smooth shapes (with corners) like triangles or rectangles, you need to specify the corner points twice with adjacent corresponding t values. For example, to draw an upside-down ”V” shape that starts at x13y27, ends at x24y27, with a pick (corner) at x18y37: Points = [’x13y27’, ’x18y37’,’x18y37’, ’x24y27’] t values = [0.00,0.55,0.5,1.00] To draw a triangle with corners at x10y29, x15y33, and x9y35, start with drawing a ”V” shape that starts at x10y29, ends at x9y35, with a pick (corner) at x15y33: Points = [’x10y29’, ’x15y33’, ’x15y33’, ’x9y35’] t values = [0.00,0.55,0.5,1.00] and then close it with a straight line from x13y27 to x24y27 to form a triangle: Points = [’x13y27’, ’x24y27’] t values = [0.00,1.00] Note that for a triangle, the start and end points should be different from each other. To draw a rectangle with four corners at x13y27, x24y27, x24y11, x13y11: Points = [’x13y27’, ’x24y27’, ’x24y27’, ’x24y11’, ’x24y11’, ’x13y11’, ’x13y11’, ’x13y27’] t values = [0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00] To draw a small square with four corners at x26y25, x29y25, x29y21, x26y21: Points = [’x26y25’, ’x29y25’, ’x29y25’, ’x29y21’, ’x29y21’, ’x26y21’, ’x26y21’, ’x26y25’] t values = [0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00] To draw a single dot at x15y31 use: Points = [’x15y31’] t values = [0.00] To draw a straight linear line that starts at x18y31 and ends at x35y14 use: Points = [’x18y31’, ’x35y14’] t values = [0.00, 1.00]. If you want to draw a big and long stroke, split it into multiple small curves that connect to each other. These instructions will define a smooth stroke that follows a Bezier curve from the starting point to the ending point, passing through the specified intermediate points. To draw a visually appealing sketch of the given object or concept, break down complex drawings into manageable steps. Begin with the most important part of the object, then observe your progress and add additional elements as needed. Continuously refine your sketch by starting with a basic structure and gradually adding complexity. Think step-by-step.

Figure 51. System prompt.

I provide you with a blank grid. Your goal is to produce a visually appealing sketch of a {concept}. Here are a few examples:

<examples> {gt-sketches} </examples>

You need to provide x-y coordinates that construct a recognizable sketch of a concept. You will receive feedback on your sketch and you will be able to adjust and fix it. Note that you will not have access to any additional resources. Do not copy previous sketches.

Think before you provide the x-y coordinates in <thinking> tags. First, think through what parts of the concept you want to sketch and the sketching order. Then, think about where the parts should be located on the grid. Finally, provide your response in <answer> tags, using your analysis.

Provide the sketch in the following format with the following fields: <formatting> <concept>The concept depicted in the sketch.</concept> <strokes>This element holds a collection of individual stroke elements that define the sketch. Each stroke is uniquely identified by its own tag (e.g., <s1>, <s2>, etc.). Within each stroke element, there are three key pieces of information: <points>A list of x-y coordinates defining the curve. These points define the path the stroke follows.</points> <t values>A series of numerical timing values that correspond to the points. These values define the progression of the stroke over time, ranging from 0 to 1, indicating the order or speed at which the stroke is drawn.</t values> <id>A short descriptive identifier for the stroke, explaining which part of the sketch it corresponds to.</id> </strokes> </formatting>

Figure 52. User prompt. This prompt contains the specific sketching task as well as details about the expected format.

<example> To draw a house, start by drawing the front of the house: <concept>House</concept> <strokes>

- <s1> <points>’x13y27’, ’x24y27’, ’x24y27’, ’x24y11’, ’x24y11’, ’x13y11’, ’x13y11’, ’x13y27’</points> <t values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t values> <id>house base front rectangle</id>

- </s1>

<s2> <points>’x13y27’, ’x18y37’,’x18y37’, ’x24y27’</points> <t values>0.00,0.55,0.5,1.00</t values> <id>roof front triangle</id>

- </s2>

</strokes> Next we add the house’s right section: <concept>House</concept> <strokes>

- <s1> <points>’x13y27’, ’x24y27’, ’x24y27’, ’x24y11’, ’x24y11’, ’x13y11’, ’x13y11’, ’x13y27’</points> <t values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t values> <id>house base front rectangle</id>

- </s1>

<s2> <points>’x13y27’, ’x18y37’,’x18y37’, ’x24y27’</points> <t values>0.00,0.55,0.5,1.00</t values> <id>roof front triangle</id>

- </s2>

<s3> <points>’x24y27’, ’x36y28’, ’x36y28’, ’x36y21’, ’x36y21’, ’x36y12’, ’x36y12’, ’x24y11’</points> <t values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t values> <id>house base right section</id>

- </s3>

<s4> <points>’x18y37’, ’x30y38’, ’x30y38’, ’x36y28’</points> <t values>0.00,0.55,0.5,1.00</t values> <id>roof right section</id>

- </s4>

</strokes> Now that we have the general structure of the house, we can add details to it, like windows and a door: <concept>House</concept> <strokes>

- <s1> <points>’x13y27’, ’x24y27’, ’x24y27’, ’x24y11’, ’x24y11’, ’x13y11’, ’x13y11’, ’x13y27’</points> <t values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t values> <id>house base front rectangle</id>

- </s1>

<s2> <points>’x13y27’, ’x18y37’,’x18y37’, ’x24y27’</points> <t values>0.00,0.55,0.5,1.00</t values> <id>roof front triangle</id>

- </s2>

Figure 53. ICL example. This is the example of a sketch of a house we provide to the model.

- <s3> <points>’x24y27’, ’x36y28’, ’x36y28’, ’x36y21’, ’x36y21’, ’x36y12’, ’x36y12’, ’x24y11’</points> <t values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t values> <id>house base right section</id>

- </s3>

<s4> <points>’x18y37’, ’x30y38’, ’x30y38’, ’x36y28’</points> <t values>0.00,0.55,0.5,1.00</t values> <id>roof right section</id>

- </s4>

<s5> <points>’x26y25’, ’x29y25’, ’x29y25’, ’x29y21’, ’x29y21’, ’x26y21’, ’x26y21’, ’x26y25’</points> <t values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t values> <id>left window square</id>

- </s5>

<s6> <points>’x31y25’, ’x34y25’, ’x34y25’, ’x34y21’, ’x34y21’, ’x31y21’, ’x31y21’,’x31y25’</points> <t values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t values> <id>right window square</id>

- </s6>

<s7> <points>’x17y11’, ’x17y18’, ’x17y18’, ’x21y18’, ’x21y18’, ’x21y11’, ’x21y11’, ’x17y11’</points> <t values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t values> <id>front door</id>

- </s7>

</strokes> and here is the complete example: <concept>House</concept> <strokes>

- <s1> <points>’x13y27’, ’x24y27’, ’x24y27’, ’x24y11’, ’x24y11’, ’x13y11’, ’x13y11’, ’x13y27’</points> <t values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t values> <id>house base front rectangle</id>

- </s1>

<s2> <points>’x24y27’, ’x36y28’, ’x36y28’, ’x36y21’, ’x36y21’, ’x36y12’, ’x36y12’, ’x24y11’</points> <t values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t values> <id>house base right section</id>

- </s2>

<s3> <points>’x13y27’, ’x18y37’,’x18y37’, ’x24y27’</points> <t values>0.00,0.55,0.5,1.00</t values> <id>roof front triangle</id>

- </s3>

<s4> <points>’x18y37’, ’x30y38’, ’x30y38’, ’x36y28’</points> <t values>0.00,0.55,0.5,1.00</t values> <id>roof right section</id>

- </s4>

Figure 54. ICL example. This is the example of a sketch of a house we provide to the model.

- <s5> <points>’x26y25’, ’x29y25’, ’x29y25’, ’x29y21’, ’x29y21’, ’x26y21’, ’x26y21’, ’x26y25’</points> <t values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t values> <id>left window square</id>

- </s5>

<s6> <points>’x31y25’, ’x34y25’, ’x34y25’, ’x34y21’, ’x34y21’, ’x31y21’, ’x31y21’,’x31y25’</points> <t values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t values> <id>right window square</id>

- </s6>

<s7> <points>’x17y11’, ’x17y18’, ’x17y18’, ’x21y18’, ’x21y18’, ’x21y11’, ’x21y11’, ’x17y11’</points> <t values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t values> <id>front door</id>

- </s7>

</strokes> </example>

Figure 55. ICL example. This is the example of a sketch of a house we provide to the model.

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

mosquitosharktigerfishraccoonoctopusdogfrogsnakedolphingiraffehandeyegoateepopsiclecakepearwatermelonpeanutpotatohotairballoonschoolbusairplanecarcanoebusrivergardenpondlighthousehospitalhouseskyscraperbaseballtennisracquetscrewdriverdrillscissorsguitarsaxophoneumbrellatablebedtelevisionbackpackmailboxcameratelephonemarkerpurse

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

[Figure 807]

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

[Figure 822]

[Figure 823]

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

[Figure 851]

[Figure 852]

[Figure 853]

[Figure 854]

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

[Figure 862]

[Figure 863]

- Figure 56. Randomly generated sketches used in the quantitative analysis (ten sketches per category).

[Figure 864]

[Figure 865]

[Figure 866]

[Figure 867]

[Figure 868]

[Figure 869]

[Figure 870]

[Figure 871]

[Figure 872]

[Figure 873]

[Figure 874]

[Figure 875]

[Figure 876]

[Figure 877]

[Figure 878]

[Figure 879]

[Figure 880]

[Figure 881]

[Figure 882]

21

[Figure 883]

[Figure 884]

[Figure 885]

[Figure 886]

[Figure 887]

[Figure 888]

[Figure 889]

[Figure 890]

[Figure 891]

[Figure 892]

[Figure 893]

[Figure 894]

[Figure 895]

[Figure 896]

[Figure 897]

[Figure 898]

[Figure 899]

[Figure 900]

[Figure 901]

[Figure 902]

mosquitosharktigerfishraccoonoctopusdogfrogsnakedolphingiraffehandeyegoateepopsiclecakepearwatermelonpeanutpotatohotairballoonschoolbusairplanecarcanoebusrivergardenpondlighthousehospitalhouseskyscraperbaseballtennisracquetscrewdriverdrillscissorsguitarsaxophoneumbrellatablebedtelevisionbackpackmailboxcameratelephonemarkerpurse

[Figure 1308]

[Figure 1309]

[Figure 1310]

[Figure 1311]

[Figure 1312]

[Figure 1313]

[Figure 1314]

[Figure 1315]

[Figure 1316]

[Figure 1317]

[Figure 1318]

[Figure 1319]

[Figure 1320]

[Figure 1321]

[Figure 1322]

[Figure 1323]

[Figure 1324]

[Figure 1325]

[Figure 1326]

[Figure 1327]

[Figure 1328]

[Figure 1329]

[Figure 1330]

[Figure 1331]

[Figure 1332]

[Figure 1333]

[Figure 1334]

[Figure 1335]

[Figure 1336]

[Figure 1337]

[Figure 1338]

[Figure 1339]

[Figure 1340]

[Figure 1341]

[Figure 1342]

[Figure 1343]

[Figure 1344]

[Figure 1345]

[Figure 1346]

[Figure 1347]

[Figure 1348]

[Figure 1349]

[Figure 1350]

[Figure 1351]

[Figure 1352]

[Figure 1353]

[Figure 1354]

[Figure 1355]

[Figure 1356]

[Figure 1357]

[Figure 1358]

[Figure 1359]

[Figure 1360]

[Figure 1361]

[Figure 1362]

[Figure 1363]

[Figure 1364]

[Figure 1365]

[Figure 1366]

[Figure 1367]

[Figure 1368]

[Figure 1369]

[Figure 1370]

[Figure 1371]

[Figure 1372]

[Figure 1373]

[Figure 1374]

[Figure 1375]

[Figure 1376]

[Figure 1377]

[Figure 1378]

[Figure 1379]

[Figure 1380]

[Figure 1381]

[Figure 1382]

[Figure 1383]

[Figure 1384]

[Figure 1385]

[Figure 1386]

[Figure 1387]

[Figure 1388]

[Figure 1389]

[Figure 1390]

[Figure 1391]

[Figure 1392]

[Figure 1393]

[Figure 1394]

[Figure 1395]

[Figure 1396]

[Figure 1397]

[Figure 1398]

[Figure 1399]

[Figure 1400]

[Figure 1401]

[Figure 1402]

[Figure 1403]

[Figure 1404]

[Figure 1405]

[Figure 1406]

[Figure 1407]

[Figure 1408]

[Figure 1409]

[Figure 1410]

[Figure 1411]

[Figure 1412]

[Figure 1413]

[Figure 1414]

[Figure 1415]

[Figure 1416]

[Figure 1417]

[Figure 1418]

[Figure 1419]

[Figure 1420]

[Figure 1421]

[Figure 1422]

[Figure 1423]

[Figure 1424]

[Figure 1425]

[Figure 1426]

[Figure 1427]

[Figure 1428]

[Figure 1429]

[Figure 1430]

[Figure 1431]

[Figure 1432]

[Figure 1433]

[Figure 1434]

[Figure 1435]

[Figure 1436]

[Figure 1437]

[Figure 1438]

[Figure 1439]

[Figure 1440]

[Figure 1441]

[Figure 1442]

[Figure 1443]

[Figure 1444]

[Figure 1445]

[Figure 1446]

[Figure 1447]

[Figure 1448]

[Figure 1449]

[Figure 1450]

[Figure 1451]

[Figure 1452]

[Figure 1453]

[Figure 1454]

[Figure 1455]

[Figure 1456]

[Figure 1457]

[Figure 1458]

[Figure 1459]

[Figure 1460]

[Figure 1461]

[Figure 1462]

[Figure 1463]

[Figure 1464]

[Figure 1465]

[Figure 1466]

[Figure 1467]

[Figure 1468]

[Figure 1469]

[Figure 1470]

[Figure 1471]

[Figure 1472]

[Figure 1473]

[Figure 1474]

[Figure 1475]

[Figure 1476]

[Figure 1477]

[Figure 1478]

[Figure 1479]

[Figure 1480]

[Figure 1481]

[Figure 1482]

[Figure 1483]

[Figure 1484]

[Figure 1485]

[Figure 1486]

[Figure 1487]

[Figure 1488]

[Figure 1489]

[Figure 1490]

[Figure 1491]

[Figure 1492]

[Figure 1493]

[Figure 1494]

[Figure 1495]

[Figure 1496]

[Figure 1497]

[Figure 1498]

[Figure 1499]

[Figure 1500]

[Figure 1501]

[Figure 1502]

[Figure 1503]

[Figure 1504]

[Figure 1505]

[Figure 1506]

[Figure 1507]

[Figure 1508]

[Figure 1509]

[Figure 1510]

[Figure 1511]

[Figure 1512]

[Figure 1513]

[Figure 1514]

[Figure 1515]

[Figure 1516]

[Figure 1517]

[Figure 1518]

[Figure 1519]

[Figure 1520]

[Figure 1521]

[Figure 1522]

[Figure 1523]

[Figure 1524]

[Figure 1525]

[Figure 1526]

[Figure 1527]

[Figure 1528]

[Figure 1529]

[Figure 1530]

[Figure 1531]

[Figure 1532]

[Figure 1533]

[Figure 1534]

[Figure 1535]

[Figure 1536]

[Figure 1537]

[Figure 1538]

[Figure 1539]

[Figure 1540]

[Figure 1541]

[Figure 1542]

[Figure 1543]

[Figure 1544]

[Figure 1545]

[Figure 1546]

[Figure 1547]

[Figure 1548]

[Figure 1549]

[Figure 1550]

[Figure 1551]

[Figure 1552]

[Figure 1553]

[Figure 1554]

[Figure 1555]

[Figure 1556]

[Figure 1557]

[Figure 1558]

[Figure 1559]

[Figure 1560]

[Figure 1561]

[Figure 1562]

[Figure 1563]

[Figure 1564]

[Figure 1565]

[Figure 1566]

[Figure 1567]

[Figure 1568]

[Figure 1569]

[Figure 1570]

[Figure 1571]

[Figure 1572]

[Figure 1573]

[Figure 1574]

[Figure 1575]

[Figure 1576]

[Figure 1577]

[Figure 1578]

[Figure 1579]

[Figure 1580]

[Figure 1581]

[Figure 1582]

[Figure 1583]

[Figure 1584]

[Figure 1585]

[Figure 1586]

[Figure 1587]

[Figure 1588]

[Figure 1589]

[Figure 1590]

[Figure 1591]

[Figure 1592]

[Figure 1593]

- Figure 57. Randomly generated sketches used in the quantitative analysis (ten sketches per category).

sailboat

[Figure 1594]

[Figure 1595]

[Figure 1596]

[Figure 1597]

[Figure 1598]

[Figure 1599]

[Figure 1600]

[Figure 1601]

[Figure 1602]

[Figure 1603]

[Figure 1604]

[Figure 1605]

[Figure 1606]

[Figure 1607]

[Figure 1608]

[Figure 1609]

[Figure 1610]

[Figure 1611]

[Figure 1612]

[Figure 1613]

[Figure 1614]

[Figure 1615]

[Figure 1616]

[Figure 1617]

[Figure 1618]

[Figure 1619]

[Figure 1620]

[Figure 1621]

[Figure 1622]

[Figure 1623]

coffee mug

[Figure 1624]

[Figure 1625]

[Figure 1626]

[Figure 1627]

[Figure 1628]

[Figure 1629]

[Figure 1630]

[Figure 1631]

[Figure 1632]

[Figure 1633]

[Figure 1634]

[Figure 1635]

[Figure 1636]

[Figure 1637]

[Figure 1638]

[Figure 1639]

[Figure 1640]

[Figure 1641]

[Figure 1642]

[Figure 1643]

[Figure 1644]

[Figure 1645]

[Figure 1646]

[Figure 1647]

[Figure 1648]

[Figure 1649]

[Figure 1650]

[Figure 1651]

[Figure 1652]

[Figure 1653]

glasses

[Figure 1654]

[Figure 1655]

[Figure 1656]

[Figure 1657]

[Figure 1658]

[Figure 1659]

[Figure 1660]

[Figure 1661]

[Figure 1662]

[Figure 1663]

[Figure 1664]

[Figure 1665]

[Figure 1666]

[Figure 1667]

[Figure 1668]

[Figure 1669]

[Figure 1670]

[Figure 1671]

[Figure 1672]

[Figure 1673]

[Figure 1674]

[Figure 1675]

[Figure 1676]

[Figure 1677]

[Figure 1678]

[Figure 1679]

[Figure 1680]

[Figure 1681]

[Figure 1682]

[Figure 1683]

car

[Figure 1684]

[Figure 1685]

[Figure 1686]

[Figure 1687]

[Figure 1688]

[Figure 1689]

[Figure 1690]

[Figure 1691]

[Figure 1692]

[Figure 1693]

[Figure 1694]

[Figure 1695]

[Figure 1696]

[Figure 1697]

[Figure 1698]

[Figure 1699]

[Figure 1700]

[Figure 1701]

[Figure 1702]

[Figure 1703]

[Figure 1704]

[Figure 1705]

[Figure 1706]

[Figure 1707]

[Figure 1708]

[Figure 1709]

[Figure 1710]

[Figure 1711]

[Figure 1712]

[Figure 1713]

- Figure 58. Sketches generated by SketchAgent for the eight categories of our human-agent collaborative study.

butterfly

[Figure 1714]

[Figure 1715]

[Figure 1716]

[Figure 1717]

[Figure 1718]

[Figure 1719]

[Figure 1720]

[Figure 1721]

[Figure 1722]

[Figure 1723]

[Figure 1724]

[Figure 1725]

[Figure 1726]

[Figure 1727]

[Figure 1728]

[Figure 1729]

[Figure 1730]

[Figure 1731]

[Figure 1732]

[Figure 1733]

[Figure 1734]

[Figure 1735]

[Figure 1736]

[Figure 1737]

[Figure 1738]

[Figure 1739]

[Figure 1740]

[Figure 1741]

[Figure 1742]

[Figure 1743]

fish

[Figure 1744]

[Figure 1745]

[Figure 1746]

[Figure 1747]

[Figure 1748]

[Figure 1749]

[Figure 1750]

[Figure 1751]

[Figure 1752]

[Figure 1753]

[Figure 1754]

[Figure 1755]

[Figure 1756]

[Figure 1757]

[Figure 1758]

[Figure 1759]

[Figure 1760]

[Figure 1761]

[Figure 1762]

[Figure 1763]

[Figure 1764]

[Figure 1765]

[Figure 1766]

[Figure 1767]

[Figure 1768]

[Figure 1769]

[Figure 1770]

[Figure 1771]

[Figure 1772]

[Figure 1773]

rabbit

[Figure 1774]

[Figure 1775]

[Figure 1776]

[Figure 1777]

[Figure 1778]

[Figure 1779]

[Figure 1780]

[Figure 1781]

[Figure 1782]

[Figure 1783]

[Figure 1784]

[Figure 1785]

[Figure 1786]

[Figure 1787]

[Figure 1788]

[Figure 1789]

[Figure 1790]

[Figure 1791]

[Figure 1792]

[Figure 1793]

[Figure 1794]

[Figure 1795]

[Figure 1796]

[Figure 1797]

[Figure 1798]

[Figure 1799]

[Figure 1800]

[Figure 1801]

[Figure 1802]

[Figure 1803]

duck

[Figure 1804]

[Figure 1805]

[Figure 1806]

[Figure 1807]

[Figure 1808]

[Figure 1809]

[Figure 1810]

[Figure 1811]

[Figure 1812]

[Figure 1813]

[Figure 1814]

[Figure 1815]

[Figure 1816]

[Figure 1817]

[Figure 1818]

[Figure 1819]

[Figure 1820]

[Figure 1821]

[Figure 1822]

[Figure 1823]

[Figure 1824]

[Figure 1825]

[Figure 1826]

[Figure 1827]

[Figure 1828]

[Figure 1829]

[Figure 1830]

[Figure 1831]

[Figure 1832]

[Figure 1833]

- Figure 59. Sketches generated by SketchAgent for the eight categories of our human-agent collaborative study.

[Figure 1834]

[Figure 1835]

[Figure 1836]

[Figure 1837]

[Figure 1838]

[Figure 1839]

[Figure 1840]

[Figure 1841]

[Figure 1842]

[Figure 1843]

[Figure 1844]

[Figure 1845]

[Figure 1846]

[Figure 1847]

[Figure 1848]

[Figure 1849]

[Figure 1850]

[Figure 1851]

[Figure 1852]

[Figure 1853]

[Figure 1854]

[Figure 1855]

[Figure 1856]

[Figure 1857]

[Figure 1858]

[Figure 1859]

[Figure 1860]

[Figure 1861]

[Figure 1862]

[Figure 1863]

[Figure 1864]

[Figure 1865]

[Figure 1866]

[Figure 1867]

[Figure 1868]

[Figure 1869]

[Figure 1870]

[Figure 1871]

[Figure 1872]

[Figure 1873]

[Figure 1874]

[Figure 1875]

[Figure 1876]

[Figure 1877]

[Figure 1878]

[Figure 1879]

[Figure 1880]

[Figure 1881]

[Figure 1882]

[Figure 1883]

[Figure 1884]

[Figure 1885]

[Figure 1886]

[Figure 1887]

[Figure 1888]

[Figure 1889]

[Figure 1890]

[Figure 1891]

[Figure 1892]

[Figure 1893]

[Figure 1894]

[Figure 1895]

[Figure 1896]

[Figure 1897]

[Figure 1898]

[Figure 1899]

[Figure 1900]

[Figure 1901]

[Figure 1902]

[Figure 1903]

[Figure 1904]

[Figure 1905]

[Figure 1906]

[Figure 1907]

[Figure 1908]

[Figure 1909]

[Figure 1910]

[Figure 1911]

[Figure 1912]

[Figure 1913]

[Figure 1914]

[Figure 1915]

[Figure 1916]

[Figure 1917]

[Figure 1918]

[Figure 1919]

[Figure 1920]

[Figure 1921]

[Figure 1922]

[Figure 1923]

[Figure 1924]

[Figure 1925]

[Figure 1926]

[Figure 1927]

[Figure 1928]

[Figure 1929]

[Figure 1930]

[Figure 1931]

[Figure 1932]

[Figure 1933]

[Figure 1934]

[Figure 1935]

[Figure 1936]

[Figure 1937]

[Figure 1938]

[Figure 1939]

[Figure 1940]

[Figure 1941]

[Figure 1942]

[Figure 1943]

[Figure 1944]

[Figure 1945]

[Figure 1946]

[Figure 1947]

[Figure 1948]

[Figure 1949]

[Figure 1950]

[Figure 1951]

[Figure 1952]

[Figure 1953]

[Figure 1954]

[Figure 1955]

[Figure 1956]

[Figure 1957]

[Figure 1958]

[Figure 1959]

[Figure 1960]

[Figure 1961]

[Figure 1962]

[Figure 1963]

[Figure 1964]

[Figure 1965]

[Figure 1966]

[Figure 1967]

[Figure 1968]

[Figure 1969]

[Figure 1970]

[Figure 1971]

[Figure 1972]

[Figure 1973]

[Figure 1974]

[Figure 1975]

[Figure 1976]

[Figure 1977]

[Figure 1978]

[Figure 1979]

[Figure 1980]

[Figure 1981]

[Figure 1982]

[Figure 1983]

[Figure 1984]

[Figure 1985]

[Figure 1986]

[Figure 1987]

[Figure 1988]

[Figure 1989]

[Figure 1990]

[Figure 1991]

[Figure 1992]

[Figure 1993]

[Figure 1994]

[Figure 1995]

[Figure 1996]

[Figure 1997]

[Figure 1998]

[Figure 1999]

[Figure 2000]

[Figure 2001]

[Figure 2002]

[Figure 2003]

[Figure 2004]

[Figure 2005]

[Figure 2006]

[Figure 2007]

[Figure 2008]

[Figure 2009]

[Figure 2010]

[Figure 2011]

[Figure 2012]

[Figure 2013]

[Figure 2014]

[Figure 2015]

[Figure 2016]

[Figure 2017]

[Figure 2018]

[Figure 2019]

[Figure 2020]

[Figure 2021]

[Figure 2022]

[Figure 2023]

[Figure 2024]

[Figure 2025]

[Figure 2026]

[Figure 2027]

[Figure 2028]

[Figure 2029]

[Figure 2030]

[Figure 2031]

[Figure 2032]

[Figure 2033]

[Figure 2034]

[Figure 2035]

[Figure 2036]

[Figure 2037]

[Figure 2038]

[Figure 2039]

[Figure 2040]

[Figure 2041]

[Figure 2042]

[Figure 2043]

[Figure 2044]

[Figure 2045]

[Figure 2046]

[Figure 2047]

[Figure 2048]

[Figure 2049]

[Figure 2050]

[Figure 2051]

[Figure 2052]

[Figure 2053]

[Figure 2054]

[Figure 2055]

[Figure 2056]

[Figure 2057]

[Figure 2058]

[Figure 2059]

[Figure 2060]

[Figure 2061]

[Figure 2062]

[Figure 2063]

[Figure 2064]

[Figure 2065]

[Figure 2066]

[Figure 2067]

[Figure 2068]

[Figure 2069]

[Figure 2070]

[Figure 2071]

[Figure 2072]

[Figure 2073]

[Figure 2074]

[Figure 2075]

[Figure 2076]

[Figure 2077]

[Figure 2078]

[Figure 2079]

[Figure 2080]

[Figure 2081]

[Figure 2082]

[Figure 2083]

[Figure 2084]

[Figure 2085]

[Figure 2086]

[Figure 2087]

[Figure 2088]

[Figure 2089]

[Figure 2090]

[Figure 2091]

[Figure 2092]

[Figure 2093]

[Figure 2094]

[Figure 2095]

[Figure 2096]

[Figure 2097]

[Figure 2098]

[Figure 2099]

[Figure 2100]

[Figure 2101]

[Figure 2102]

[Figure 2103]

[Figure 2104]

[Figure 2105]

[Figure 2106]

[Figure 2107]

[Figure 2108]

[Figure 2109]

[Figure 2110]

[Figure 2111]

[Figure 2112]

[Figure 2113]

[Figure 2114]

[Figure 2115]

[Figure 2116]

[Figure 2117]

[Figure 2118]

[Figure 2119]

[Figure 2120]

[Figure 2121]

[Figure 2122]

[Figure 2123]

[Figure 2124]

[Figure 2125]

[Figure 2126]

[Figure 2127]

[Figure 2128]

[Figure 2129]

[Figure 2130]

[Figure 2131]

[Figure 2132]

[Figure 2133]

[Figure 2134]

[Figure 2135]

[Figure 2136]

[Figure 2137]

[Figure 2138]

[Figure 2139]

[Figure 2140]

[Figure 2141]

[Figure 2142]

[Figure 2143]

[Figure 2144]

[Figure 2145]

[Figure 2146]

[Figure 2147]

[Figure 2148]

[Figure 2149]

[Figure 2150]

[Figure 2151]

[Figure 2152]

[Figure 2153]

[Figure 2154]

[Figure 2155]

[Figure 2156]

[Figure 2157]

[Figure 2158]

[Figure 2159]

[Figure 2160]

[Figure 2161]

[Figure 2162]

[Figure 2163]

[Figure 2164]

[Figure 2165]

[Figure 2166]

[Figure 2167]

[Figure 2168]

[Figure 2169]

[Figure 2170]

[Figure 2171]

[Figure 2172]

[Figure 2173]

[Figure 2174]

[Figure 2175]

[Figure 2176]

[Figure 2177]

[Figure 2178]

[Figure 2179]

[Figure 2180]

[Figure 2181]

[Figure 2182]

[Figure 2183]

[Figure 2184]

[Figure 2185]

[Figure 2186]

[Figure 2187]

[Figure 2188]

[Figure 2189]

[Figure 2190]

[Figure 2191]

[Figure 2192]

[Figure 2193]

[Figure 2194]

[Figure 2195]

[Figure 2196]

[Figure 2197]

[Figure 2198]

[Figure 2199]

[Figure 2200]

[Figure 2201]

[Figure 2202]

[Figure 2203]

[Figure 2204]

[Figure 2205]

[Figure 2206]

[Figure 2207]

[Figure 2208]

[Figure 2209]

[Figure 2210]

[Figure 2211]

[Figure 2212]

[Figure 2213]

[Figure 2214]

[Figure 2215]

[Figure 2216]

[Figure 2217]

[Figure 2218]

[Figure 2219]

[Figure 2220]

[Figure 2221]

[Figure 2222]

[Figure 2223]

[Figure 2224]

[Figure 2225]

[Figure 2226]

[Figure 2227]

[Figure 2228]

[Figure 2229]

[Figure 2230]

[Figure 2231]

[Figure 2232]

[Figure 2233]

[Figure 2234]

[Figure 2235]

[Figure 2236]

[Figure 2237]

[Figure 2238]

[Figure 2239]

[Figure 2240]

[Figure 2241]

[Figure 2242]

[Figure 2243]

[Figure 2244]

[Figure 2245]

[Figure 2246]

[Figure 2247]

[Figure 2248]

[Figure 2249]

[Figure 2250]

[Figure 2251]

[Figure 2252]

[Figure 2253]

[Figure 2254]

[Figure 2255]

[Figure 2256]

[Figure 2257]

[Figure 2258]

[Figure 2259]

[Figure 2260]

[Figure 2261]

[Figure 2262]

[Figure 2263]

[Figure 2264]

[Figure 2265]

[Figure 2266]

[Figure 2267]

[Figure 2268]

[Figure 2269]

[Figure 2270]

[Figure 2271]

[Figure 2272]

[Figure 2273]

[Figure 2274]

[Figure 2275]

[Figure 2276]

[Figure 2277]

[Figure 2278]

[Figure 2279]

[Figure 2280]

[Figure 2281]

[Figure 2282]

[Figure 2283]

[Figure 2284]

[Figure 2285]

[Figure 2286]

[Figure 2287]

[Figure 2288]

[Figure 2289]

[Figure 2290]

[Figure 2291]

[Figure 2292]

[Figure 2293]

[Figure 2294]

[Figure 2295]

[Figure 2296]

[Figure 2297]

[Figure 2298]

[Figure 2299]

[Figure 2300]

[Figure 2301]

[Figure 2302]

[Figure 2303]

[Figure 2304]

[Figure 2305]

[Figure 2306]

[Figure 2307]

[Figure 2308]

[Figure 2309]

[Figure 2310]

[Figure 2311]

[Figure 2312]

[Figure 2313]

