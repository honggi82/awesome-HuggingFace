### arXiv:2604.22875v2[cs.CV]28Apr2026

#### SketchVLM: Vision language models can annotate images to explain thoughts and guide users

Brandon Collins1∗

blc0063@auburn.edu

Logan Bolton1∗

logan.bolton@auburn.edu

Hung Huy Nguyen1∗

huyhung.dknec@gmail.com

Mohammad Reza Taesiri

mtaesiri@gmail.com

Trung Bui2

bui@adobe.com

Anh Totti Nguyen1

anh.ng8@gmail.com

1Auburn University 2Adobe Research

Abstract. When answering questions about images, humans naturally point, label, and draw to explain their reasoning. In contrast, modern vision–language models (VLMs) such as Gemini-3-Pro and GPT-5 only respond with text, which can be difficult for users to verify. We present SketchVLM, a training-free, model-agnostic framework that enables VLMs to produce non-destructive, editable SVG overlays on the input image to visually explain their answers. Across seven benchmarks spanning visual reasoning (maze navigation, ball-drop trajectory prediction, and object counting) and drawing (part labeling, connecting-thedots, and drawing shapes around objects), SketchVLM improves visual reasoning task accuracy by up to +28.5 percentage points and annotation quality by up to 1.48× relative to image-editing and fine-tuned sketching baselines, while also producing annotations that are more faithful to the model’s stated answer. We find that single-turn generation already achieves strong accuracy and annotation quality, and multi-turn generation opens up further opportunities for human-AI collaboration. An interactive demo and code are at https://sketchvlm.github.io/.

Keywords: Vision-language models · Explainable AI · Computer Vision

##### 1 Introduction

From browsers (Atlas, MS Edge [39]) to office products (MS Copilot, Gemini [43]), the market for LLM-powered chatbots for general end users is estimated to grow from $1.1B in 2023 to $83B in 2032 [6]. As these assistants increasingly answer image-based questions in high-frequency consumer and productivity workflows [6, 35, 39, 43], users need responses they can quickly understand and verify [29].

However, modern VLMs such as Gemini-3-Pro and GPT-5 typically respond with a block of text, which can be difficult to verify [55] (Fig. A2). To explain

* All three co-first authors made major contributions to dataset creation, experiments, and manuscript. See Sec. 7 for a detailed author contribution statement.

"annotations": [ {

"annotations": [ {

## multi-turn

# Final

"type": "rect",

"type": "text", "content": "1. OIL DIPSTICK",

- "x": 128,
- "y": 381, "width": 390, "height": 178, "color": "#00FF00", "strokeWidth": 3, "fill": "none"

- "x": 515,
- "y": 444, "color": "#FFFF00", "fontSize": 25

[Figure 1]

###### 2 Collins, Bolton, Nguyen et al.

}, {

To check your car's oil, follow these steps:

[Figure 2]

- 1. Locate the oil dipstick, which usually has a brightly colored handle (often yellow or orange). In this image, it is the yellow handle near the center.

- 2. Pull the dipstick out and wipe it clean with a rag or paper towel.

- 3. Reinsert the dipstick fully, then pull it out again.

- 4. Check the oil level on the end of the stick; it should be between the two marks. Also, check the color and consistency of the oil.

- 5. If you need to add oil, remove the oil ﬁller cap (labeled 'dexos' in this image) and pour in the recommended oil type.

[Figure 3]

"annotations": [ {

[Figure 4]

1

How do I check my car’s oil?

}, {

2. OIL FILLER CAP

"type": "text", "content": "1. OIL

"type": "arrow",

1. OIL DIPSTICK

DIPSTICK",

(a)

[Figure 5]

- "x": 515,
- "y": 444, "color": "#f1c232", "fontSize": 25

- "x1": 510,
- "y1": 444,

Sketch VLM

"type": "text", "content": "Safe Operating

}, {

how do I check my car's oil level?

"type": "circle", "cx": 496, "cy": 444, "r": 25,

To check your car's oil, follow these steps:

- "x2": 496,
- "y2": 444, "color": "#FFFF00", "strokeWidth": 4

[Figure 6]

ChatGPT 5.2

Range (Crosshatched)",

- 1. Locate the oil dipstick, which usually has a brightly colored handle (often yellow or orange). In this image, it is the yellow handle near the center...

- 2. Locate the oil filler cap

... }...

Thought for 52s

In your photo: Oil fill cap: the black cap that says “dexos” on the engine cover (upper left). Oil dipstick: the yellow handle near the front-center, tucked just under the front edge of the plastic cover. How to check it: Park on level ground and shut the engine off. Best: check first thing in the morning (cold). Pull the dipstick (yellow handle), wipe it clean with a paper towel or rag Reinsert fully until it seats, then pull it again...

System Prompt

- "x": 128,
- "y": 360, "color": "#00FF00", "fontSize": 20

“answer”: “To check your car's oil…”

Here’s a list of annotations you can use:

(b)

[Figure 7]

Text

"annotations": [ {

}, {

"type": "rect",

[Figure 8]

- "x": 128,
- "y": 381, "width": 390, "height": 178,

2

[Figure 9]

Is this enough oil?

Safe Operating Range

}, {

...

}, {

"type": "circle", "cx": 496, "cy": 444, "r": 25, "color": "#FFFF00", "strokeWidth": 3, "fill": "none"

[Figure 10]

Current Oil Level

Sketch VLM

"type": "text", "content": "Safe

Operating Range",

"type": "arrow",

- "x": 128,
- "y": 360,

Yes, the oil level appears to be sufficient. On this dipstick, the crosshatched area represents the safe operating range ...

... },…

Yes, the oil level appears to be suﬃcient. On this dipstick, the crosshatched area represents the safe operating range. The oil line is clearly visible covering the entire crosshatched section and extending slightly beyond it towards the tip. As long as the oil level is within or slightly above the crosshatched area (and not excessively overﬁlled), it is considered enough.

- "x1": 650,
- "y1": 350,

“answer”: “Yes, the oil level appears to be sufficient…”

- "x2": 518,
- "y2": 450, "color": "#FF0000", "strokeWidth": 3

Fig. 1: For complex questions, modern chatbots like ChatGPT often return long text responses (a) that are hard for users to understand, verify, and follow. In contrast, SketchVLM guides users (b) step-by-step by annotating the input image and grounding answers to relevant image regions—here, guiding a user how to check their car’s oil level (source: https://www.youtube.com/watch?v=tNNyu9S65E4).

}, {

}, {

"type": "text", "content": "2. OIL FILLER CAP",

"type": "text", "content": "Current Oil Level",

- "x": 355,
- "y": 265, "color": "#FF0000", "fontSize": 25

their thoughts on an image problem, humans often point, circle, underline, label, and annotate directly on the image. For example, when a user asks to check a car’s oil level, a visual annotation is easier to verify than a paragraph of text (Fig. 1) [26,31].

- "x": 660,
- "y": 340, "color": "#FF0000", "fontSize": 20

}, {

Some VLMs emit point coordinates for referencing objects (e.g., MoonDream [45], Molmo [2]), but they do not support free-form visual annotation. Other models are trained to generate visual reasoning traces but do not generalize well beyond their training domains [18,48]. Image-editing models can visualize intermediate reasoning steps for multimodal questions [57], but risk altering the source image in unintended ways, undermining user trust [41].

}, {

"type": "arrow",

- "x1": 350,
- "y1": 265,

"type": "circle", "cx": 236, "cy": 472, "r": 15, "color": "#FFFF00", "strokeWidth": 2, "fill": "none"

- "x2": 305,
- "y2": 265, "color": "#FF0000", "strokeWidth": 4

To address these issues, we propose SketchVLM, a state-of-the-art (SotA) system that draws SVG annotations in a separate layer overlaid on top of the input image to explain its reasoning. SketchVLM grounds its annotations directly on the original image without modifying it and without requiring taskspecific training. We test SketchVLM with multiple VLM backbones, including Gemini-3-Pro-Preview ( ) [36] and GPT-5 ( ) [30]. We collectively refer to these models harnessed with SketchVLM as SketchVLMs and individually as sketchVLM and sketchVLM. We compare against the leading alternative approaches for generating visual annotations such as the SotA image-editing model Nano Banana Pro ( ) [13] and specialized VLMs fine-tuned to produce image annotations, ViLaSR ( ) [48] and ThinkMorph ( ) [18].

}, {

}, {

"type": "circle", "cx": 305, "cy": 265, "r": 45, "color": "#FF0000", "strokeWidth": 3, "fill": "none"

[Figure 11]

"type": "circle", "cx": 332, "cy": 472, "r": 15, "color": "#FFFF00", "strokeWidth": 2, "fill": "none"

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

} ]

On a comprehensive evaluation of seven tasks: (a) three drawing tasks (connecting-the-dots, labeling parts of an object, and drawing shapes around

}, {

"type": "circle", "cx": 428, "cy": 468, "r": 15, "color": "#FFFF00", "strokeWidth": 2, "fill": "none"

} ]

objects) and (b) four visual reasoning tasks (two physics understanding tasks, one counting task and one maze navigation task), our main findings are:

[Figure 16]

[Figure 17]

Input sketchVLM ++ ViLaSR [48]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

BallDrop

###### (a)

Which bucket will the ball fall into?

Answer: 2 Answer: 2 Ball goes through platform

Answer: 3 Ball goes through platform

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Input sketchVLM ThinkMorph [18] ViLaSR [48]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Connect-the-Dots

###### (b)

Connect the dots in order.

Correct Improper Lines Improper Lines

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Input sketchVLM ++ ThinkMorph [18]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

MazeNavigation

###### (c)

Is the path “right, right, up, up” from green valid?

Answer: Yes Answer: No Wrong path

Answer: Yes Wrong path

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Fig. 2: Our sketchVLM (Gemini-3-Pro-Preview) draws more accurate predicted trajectories in Ball Drop (a); connects the dots more accurately (b); and sketches more plausible maze navigation paths (c). Nano Banana ( ++ ) often undesirably alters the image and draws implausible trajectories in ball drop and maze navigation. Specialist VLMs ( and ) fine-tuned to sketch often fail to generalize to new tasks.

[Figure 47]

[Figure 48]

[Figure 49]

- – SketchVLMs based on frontier models ( and ) generate annotations of superior generalizability, accuracy, and annotation quality compared to those by specialized fine-tuned sketching models ( , ) (Secs. 5.2, 5.3 and 5.6 to 5.10).

[Figure 50]

[Figure 51]

[Figure 52]

- – Nano Banana ( ) is unable to generate a separate overlay layer and frequently alters the original image when generating in-image annotations (Secs. 5.4 and 5.5).

[Figure 53]

- – SketchVLMs have similar accuracy in single-turn and multi-turn settings, but are significantly faster with single-turn generation (Sec. 5.11).
- – Adding an external grid of x–y coordinates to the input image improves sketchVLM’s drawing capability and question answering accuracy but is not necessary for sketchVLM (Sec. 5.1).

[Figure 54]

##### 2 Related Work

Native image-editing models (e.g., GPT-Image-1.5 and ) can directly modify images to add annotations, but their performance can be inconsistent for reasoning-heavy tasks [57]. Open-source native multimodal autoregressive models (e.g., Chameleon [42] and Bagel [15]) enable interleaved text–image generation, but they lack an editable annotation layer aligned to the input. In contrast, SketchVLMs generate a non-destructive SVG overlay on the input image (Tab. 1).

[Figure 55]

Tool-calling and code generation Agentic systems improve visual reasoning by writing code that invokes external tools or manipulates the input image. V* [49] uses LLM-guided visual search to localize target objects in highresolution images, then crops relevant regions for closer inspection. Visual Sketchpad [19] and OpenThinkIMG [40] equip VLMs with modular vision tools such as segmentation, object detection, and OCR to support multi-step reasoning. PyVision [54] generates Python code to draw structured overlays on input images across multiple turns. Other training-free methods leverage internal attention or gradient maps to automatically crop and zoom into salient regions [53]. These systems excel at fine-grained visual understanding, but they typically rely on external tools or code execution rather than prompting a single VLM to produce user-facing free-form annotations directly on the image.

Visual prompting e.g., drawing coordinate points or horizontal lines directly on an input image can improve the vision capabilities of VLMs [20,22,50]. Similarly, SketchAgent [46] appends a coordinate grid to the edge of the input image to allow the model to reference precise x–y positions in the image.

Visual sketching Whiteboard-of-Thought [27] prompts an LLM to produce Matplotlib code rendered on a blank canvas to give the model a space to draw before responding to text-based questions. D2R [33] interleaves textual chainof-thought with rendered visual drafts of its proposed actions overlaid on the input image at each reasoning step, enhancing dynamic spatial reasoning across multiple turns. VDLM [47] converts images into SVG and then into a more LLM-interpretable format to improve visual understanding. SketchAgent [46] and SketchFormer [38] focus on sketch generation as a standalone task on a blank canvas. In contrast, SketchVLMs generate non-destructive, editable SVG annotations directly on existing input images so that users can inspect the model’s reasoning without altering the source image.

Fine-tuned sketching models MVoT [23] fine-tunes Chameleon [42] to generate interleaved text and image reasoning traces, visualizing intermediate states to support multi-step spatial reasoning. LatentSketchpad [52] and DeepSketcher [51] both move visual reasoning into learned latent or embedding spaces.

LatentSketchpad is built with Gemma3 and Qwen2.5-VL-7B, while DeepSketcher uses Qwen2.5-VL-7B. ViLaSR [48] post-trains Qwen-2.5-VL-7B to sketch on the input image with an SVG overlay before responding. Similarly, ThinkMorph is fine-tuned from BAGEL-7B-MoT [15] to generate visual sketches that support its answers, though it directly edits the input image rather than overlaying SVG annotations. Unlike these approaches, we build a harness around SotA VLMs to enable them to annotate on input images by generating a layer of SVGs. Therefore, different from the literature, our approach is training-free and enjoys the generalizability of SotA VLMs to new domains.

Table 1: Comparison of sketching models and methods. Annotation type describes the visual artifact used during reasoning: Vector overlay denotes structured, nondestructive marks (e.g., strokes/boxes/text) aligned to an image or canvas, while Image edit denotes pixel-space image modification or synthesis that may change image content. Input image is marked only when a provided image is the primary visual context (vs. blank canvas or purely generative visual thoughts). Free-form drawing indicates support for arbitrary stroke-like annotations beyond a fixed mark set.

Model name Training-free Multi-turn Input image Free-form drawing Annotation type SketchVLM (Ours) ✓ ✓ ✓ ✓ Vector overlay VisualSketchPad [19] ✓ ✓ ✓ ✗ Vector overlay PyVision [54] ✓ ✓ ✓ ✗ Vector overlay SketchAgent [46] ✓ ✓ ✗ ✓ Vector overlay D2R [33] ✓ ✓ ✓ ✗ Image edit Whiteboard-of-Thought [27] ✓ ✗ ✗ ✗ Vector overlay ViLaSR [48] ✗ ✓ ✓ ✗ Vector overlay OpenThinkIMG [40] ✗ ✓ ✓ ✗ Image edit ThinkMorph [18] ✗ ✗ ✓ ✗ Image edit MVoT [23] ✗ ✓ ✓ ✗ Image edit LatentSketchpad [52] ✗ ✓ ✓ ✗ Image edit DeepSketcher [51] ✗ ✓ ✓ ✗ Image edit

##### 3 SketchVLM

SketchVLM combines three components: visual prompting to aid spatial reference, a system prompt that elicits structured stroke outputs, and XML-to-SVG conversion that renders those strokes as an overlay on the source image (Fig. 1). Visual prompting To make VLMs draw more reliably on tasks that require precision, such as Connect-the-Dots, we follow SketchAgent [46] and append a coordinate grid to the left and bottom of each image, scaled to the image resolution (Fig. D12).

Input prompt To enable VLMs to generate annotations, we introduce a system prompt (Sec. F.2) that instructs the model to produce stroke sequences in a specific format (e.g., XML-style <s1>, <s2>, ... <sN> tags each containing a list of points) corresponding to reasoning steps. We provide instructions for drawing primitives including rectangles, arrows, text labels, straight lines, and Bézier curves. The model is then given the task prompt (e.g., “Which bucket will

the ball fall into?” (Fig. 2)) and is instructed to annotate its reasoning on the image before responding with a final answer (Fig. 1).

SVG conversion We parse the model’s XML output to a standardized SVG output that can be overlaid on top of the original input image. If there are exactly two points in a stroke, we overlay a straight line. Otherwise, given a stroke described by m ordered samples Si = {(xj,yj)}mj=1 and corresponding normalized timestamps Ti = {tj}mj=1 with tj ∈ [0,1], we fit a smooth Bézier curve and render it as SVG. Following SketchAgent [46], a least squares solution is found for the control points in a cubic Bézier curve. See Fig. F5 for an example of the stroke output and the corresponding overlay. Following SketchAgent, we run all experiments using XML format. Given that VLMs produce JSON at comparable quality and JSON is more human-readable, we use JSON for our interactive demo.

##### 4 Evaluation

###### 4.1 7 Tasks

- 1. Connect-the-Dots contains 100 images spanning three subsets: 21 randomly generated dot patterns, 30 connect-the-dots puzzles derived from silhouette SVGs, and 49 worksheet-style images collected from online sources. Models must locate each dot and connect them in order (Secs. B.1 and D.3 and Fig. 2).

[Figure 56]

- 2. Counting Objects contains 746 images drawn from CountBench [5,34], and Pixmo-Count [14]. We include object counts from 0 to 10 and filter out unsuitable Pixmo-Count examples. Models must count the target objects and place numbered markers on each one (Secs. B.2 and D.4 and Fig. 5).

[Figure 57]

- 3. Drawing Shapes around Objects uses 1,000 images selected from the 5,000-image COCO validation set [25]. We choose images to balance object count and object size across classes. Models must localize objects by drawing rectangles or ovals (Secs. B.3 and D.5 and Fig. 6).

[Figure 58]

- 4. Part Labeling contains 985 images selected from PACO [37] and Pascal-Part [10], covering 52 object classes. We keep images with a single target object occupying at least 10% of the image area and with at least four annotated part labels, while maintaining class balance. Models must place the correct text labels at the corresponding part locations (Secs. B.4 and D.6 and Fig. 7).

[Figure 59]

- 5. Maze Navigation contains 200 generated 3 × 3 grid mazes. We vary the shortest path length from 3 to 8 steps and create invalid paths by perturbing one direction in the ground-truth path. Models must trace a proposed path and determine whether it reaches the goal without crossing walls (Secs. B.5 and D.2 and Fig. 2).
- 6. VPCT is the Visual Physics Comprehension Test [8], which contains 100 hand-crafted images. Models must predict which container a dropped ball will land in (Sec. B.6 and Figs. 2 and D1).

Single vs multiturn

- <s1>

<points>'x500y100','x500y270','x500y270','x 350y400'</points>

<t_values>0.00,0.50,0.50,1.00</t_values> <id>path_1</id>

- </s1>

<s2> <points>'x350y400','x325y470','x300y540'</p oints>

<t_values>0.00,0.50,1.00</t_values> <id>drop_1</id>

- </s2>

<s3> <points>'x300y540','x390y470','x500y670'</p oints>

<t_values>0.00,0.50,1.00</t_values> <id>path_2</id>

- </s3>

<s4> <points>'x500y670','x570y690','x640y710'</p oints>

<t_values>0.00,0.50,1.00</t_values> <id>path_bounce</id>

- </s4>

<s5> <points>'x640y710','x700y850','x730y950'</p oints>

<t_values>0.00,0.60,1.00</t_values> <id>drop_2</id>

- </s5> <final_answer>3</final_answer>

\caption{Single-turn and multi-turn generation on the same VPCT example.

<s3>\n <points>'x635y775','x680y850','x725 y965'</points>

<s2>\n <points>'x500y618','x568y697','x635 y775'</points>

SketchVLM 7

- (a) input image
- (b) system prompt
- (c) task prompt
- (d) rendered image
- (e) previous annotations

In \textit{single-turn}, \SketchVLM receives (a) the input image, (b) the system prompt, and (c) the task prompt, then outputs all annotation strokes and the final answer in a single model call.

System Prompt Input Image

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Here’s a list of annotations you can use:

Sketch VLM

Sketch VLM

- <s1>…</s1>

- <s2>…</s2>

- <s3>…</s3>

[Figure 67]

[Figure 68]

- <s4>…</s4>

[Figure 69]

- <s5>…</s5>

Text

…<s2><points>'x500 y618'…'x635y775'

<s1><points>'x500y 105','x500y618'… </points></s1>

In \textit{multi-turn}, after earlier turns have produced partial annotations, each new call receives

Turn 1

Turn 2 </points></s2>

[Figure 70]

Task Prompt

[Figure 71]

Draw the trajectory of the ball. What bucket does it land in?

- - Last edited image
- - XMLs of all past turns

- (d) the rendered image containing the annotations drawn so far and
- (e) the text representation of those previous annotations, together with the same (b) system prompt and (c) task prompt, and outputs one new annotation.

Final answer: Bucket 3 ✅

[Figure 72]

Sketch

Answer: Bucket 3

###### +

Answer: Bucket 3

VLM Sketch

…<s3><points>'x635y 775','x680y850'… </points></s3>

[Figure 73]

VLM

Turn N

- (d)

- (e)

Final answer: Bucket 3 ✅

The model produces its final text answer on the last turn.}

(b) Multi-turn

(a) Single-turn

(a) Single-turn (b) Multi-turn

New version

- Fig. 3: Single-turn and multi-turn generation on the same VPCT sample. In (a) singleturn, SketchVLM receives the system prompt, the task prompt, and the input image, then outputs all annotations and the final answer in a single model call. In (b) multiturn, Turn 1 uses the same inputs and outputs one annotation. For later turns, the model reuses the system prompt, the task prompt, and the previous annotations, which are provided in both the rendered image and text form. This process repeats until the model outputs its final text answer on the last turn.

<s3>…</s3>

<final_answer>3</final_answer>

<s2>…</s2>

\caption{Single-turn and multi-turn generation on the same VPCT sample. In \textit{single-turn}, SketchVLM receives (a) the system prompt, (b) the task prompt, and (c) the input image, then outputs all annotation strokes and the final answer in a single model call. In \textit{multi-turn}, Turn 1 uses the same inputs (a)--(c) and outputs one annotation stroke. For later turns, the model reuses (a) the system prompt, (b) the task prompt, and (d) the previous annotations, which are provided in both rendered-image and text form. This process repeats until the model outputs its final text answer on the last turn.}

Fig x: In (a) single-turn, SketchVLM receives (b) the input image, (c) the system prompt, and (d) the task prompt, then outputs all annotation strokes and the final answer in a single model call. In (e) multi-turn, the model produces one stroke at a time, and then the newly generated stroke is rendered onto the image and shown in text and image form back to the model (f). This process repeats itself until the model then produces its final text answer on the last turn.

[Figure 74]

- 7. Ball Drop contains 198 synthetically generated images with harderto-guess answers and ground-truth ball trajectories produced using PHYRE [4]. We generate equal numbers of images with 1, 2, and 3 randomly placed lines, and randomize the ball’s horizontal position. Models must predict the landing container out of four choices and trace the ball trajectory (Sec. B.6 and Figs. D2, D3 and 10).

###### 4.2 Setup

Single-turn and multi-turn We evaluate SketchVLMs in (1) single-turn, where the models output all annotations and their final answer in one response, and (2) multi-turn, producing one stroke per turn, to simulate iterative, realworld conversations. During each turn, the VLM receives the image with all previous annotations rendered, and the annotations’ text representations. The model then outputs its final text answer on the last turn (Fig. 3).

Visual prompting We evaluate the necessity of a coordinate grid for the model to reference specific points in an image. When omitting the grid, the model outputs coordinates on a normalized 1000×1000 scale (Fig. D14).

###### 4.3 Baselines

We compare SketchVLMs against three baselines that represent the SotA approaches for producing visual annotations on images (Fig. 4):

1. Image Editing Model + VLM: is an image-editing model that can annotate images, but produces no text answer. To obtain a text response, we feed its edited image to (denoted ++ ).

[Figure 75]

[Figure 76]

[Figure 77]

|[Figure 78]<br><br>Image|
|---|

|<br><br>Task Prompt|
|---|

Sketching Prompt

###### 8 Collins, Bolton, Nguyen et al.

[Figure 80]

|[Figure 81]<br><br>Image|
|---|

Text Answer

|[Figure 82]<br><br>VLM|
|---|

[Figure 83]

[Figure 84]

Annotated Image

[Figure 85]

|[Figure 86]<br><br>SketchVLM Structured Output|
|---|

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Edited Image

[Figure 92]

Text Answer

Edited Image

Text Answer

[Figure 93]

[Figure 94]

SVG Strokes

|[Figure 95]<br><br>SVG strokes|
|---|

|[Figure 96]<br><br>Final Answer|
|---|

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

[Figure 109]

Task Prompt

Image Image Task Prompt

Task Prompt

Sketching Prompt

Task Image Prompt

[Figure 110]

Image

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

(a) (b) sketchVLM (c) (d)

- Fig. 4: Four approaches for making VLMs answer visual questions and annotate images.

- (a) outputs text only. No drawings generated.
- (b) sketchVLM draws on the image while outputting text.
- (c) only edits the image.

[Figure 117]

- (d) ++ takes the edited image from and gives it to to respond.

[Figure 118]

[Figure 119]

- 2. Fine-tuned Sketching Models: is a model fine-tuned from Qwen-2.5VL-7B [3] to autoregressively generate SVG annotations on the input image over multiple turns, and is a model fine-tuned from BAGEL-7B-MoT [15] to directly edit the input image while also producing a text answer (Tab. F1 and Sec. G.1).

[Figure 120]

[Figure 121]

- 3. Default VLMs: We include text-only and as simple baselines.

[Figure 122]

- 4.4 Metrics

A correct text answer from the model is not enough if the annotation is uninformative, and a plausible annotation can be misleading if it contradicts the text response. We therefore measure three distinct aspects of model performance:

Accuracy serves as the primary measure of task performance and tests the effect of sketching on answer correctness.

Annotation–text alignment measures how faithful the visual traces are to the text answer. We ask a VLM judge [9] to infer the answer from the annotations alone and report how often the judge’s inferred answer matches the model’s final text answer.

Annotation quality allows us to distinguish models that produce informative annotations from those that output low-quality or incoherent drawings. We adopt a VLM-as-a-Judge approach using a rubric scored from 1 to 5 (Sec. E) that evaluates annotation plausibility and visual clarity for each task.

###### Table 2: SketchVLMs produce visual reasoning traces while maintaining competitive accuracy. ++ underperforms default , and fine-tuned sketching models ( , ) perform near random chance on visual reasoning tasks.

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

draw shapes connect dots

[Figure 128]

[Figure 129]

[Figure 130]

VPCT

[Figure 131]

video physics ball drop maze trace counting labeling RMSE Order Acc%

Model

|sketchVLM|96.0 ± 1.4 79.7 ± 2.8 98.0 ± 1.7 94.5 89.3 ± 2.2 83.8 ± 3.4 99.3 ± 0.8 93.0|60.3 58.8 55.4 5.92 99.0 64.1 63.1 59.8 † †<br><br>|
|---|---|---|
|[Figure 132]<br><br>sketchVLM<br><br>[Figure 133]|70.0 ± 2.9 68.5 ± 2.2 92.8 ± 2.5 75.4 63.5 ± 2.5 66.0 ± 4.7 92.3 ± 2.3 72.0|20.4 18.7 11.2 46.69 74.0 19.1 22.4 15.4 † †<br><br>|
|[Figure 134]<br><br>++<br><br>[Figure 135]<br><br>[Figure 136]|63.0 62.6 93.3 ± 3.0 91.7 37.0 35.9 50.8 ± 1.5 48.6 27.0 30.3 62.5 ± 2.1 68.1|† † † † 39.0<br><br>– – † 198.74 9.0 † † † † 0.0|

† Model is unable to output this format.

##### 5 Results

- 5.1 Grid prompting improves sketchVLM annotation precision, but hurts sketchVLM

[Figure 137]

To understand how each component of our framework affects the model, we run ablations to see how sketching and grid prompting affect sketchVLM and sketchVLM .

[Figure 138]

Experiment We evaluate four input configurations in single-turn mode: the base image alone, image with grid, image with sketching prompt, and image with both. We report accuracy on VPCT, Ball Drop, and Maze Navigation, and RMSE on Connect-the-Dots (Tab. 3).

Results performs best with both the grid and sketching prompt, with the grid providing a consistent boost to spatial precision (Fig. D12), consistent with prior findings [20]. performs better without the grid, and adding it notably degrades localization on Connect-the-Dots (RMSE increases from 5.92 to 99.34). We therefore report all SketchVLM results in Tab. 2 with the input grid for but without the grid for (Fig. D14).

[Figure 139]

[Figure 140]

[Figure 141]

###### 5.2 SketchVLMs can localize points and connect them in order

Connecting the dots in an image tests both spatial grounding and the ability to produce multiple coherent strokes in a row.

Experiment We evaluate the root mean squared error (RMSE) in pixels of the difference between the ground truth position of each of the points and the SketchVLMs’ closest generated point. To evaluate whether models connect points in the correct order, we compare each predicted segment i against all ground-truth segment pairs using MSE. If segment i has lower MSE to a different ground-truth pair than its expected pair (i, i+1), it is counted as an ordering error. Because and only produce an image response with no x-y coordinates, we manually evaluate their outputs.

[Figure 142]

[Figure 143]

Results sketchVLM can accurately output the correct location of up to 35 points with a very low RMSE of only 5.92 (Tab. 2). sketchVLM has a

[Figure 144]

- Table 3: Ablation across inputs in single-turn mode. “Sketch” adds strokes/system prompt; “Grid” additionally overlays the coordinate grid. RMSE is reported for Connect-the-Dots while accuracy is reported for the other tasks. sketchVLM works best with the grid while sketchVLM works best without the grid

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

Model Input VPCT

Image 63.5 66.0 92.3 N/A + Grid 67.0 67.2 98.0 N/A + Sketch 59.0 63.1 88.7 134.40

[Figure 149]

+ Sketch + Grid + Multi-turn 68.0 63.1 82.1 50.68 + Sketch + Grid (i.e., sketchVLM) 70.0 68.5 92.8 46.69

[Figure 150]

Image 89.3 83.8 99.3 N/A + Grid 90.0 81.3 99.5 N/A + Sketch (i.e., sketchVLM ) 96.0 79.7 98.0 5.92 + Sketch + Multi-turn 80.0 79.8 98.2 12.37 + Sketch + Grid 91.0 82.3 99.5 99.34

[Figure 151]

higher RMSE of 46.69, but still performs much more reliably than ’s 198.74 RMSE. Regarding ordering, , , and frequently produce ordering errors as the number of strokes increases, often connecting points out of order and inadvertently altering the input image (Sec. C.2). In contrast, sketchVLM and sketchVLM correctly order points 74% and 99% of the time, respectively, demonstrating that SketchVLMs can reliably scale to many strokes while maintaining spatial accuracy and logical coherence (Fig. D13).

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

###### 5.3 SketchVLM improves counting accuracy

(b) sketchVLM

Input (a) ++

(c)

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

Answer: 8

Answer: 3

Answer: 9

[Figure 162]

[Figure 163]

[Figure 164]

- Fig. 5: (a) ++ generates a different image and predicts an incorrect count. (c) directly outputs only a number without annotations and severely undercounts. In

[Figure 165]

contrast, our sketchVLM (b) outputs the correct answer and produces visual annotations to explain its answer.

Existing VLMs can output point coordinates to mark counted objects, but these points are unlabeled and can be tedious to verify. In contrast, SketchVLMs explicitly place numeric markers on each object, enabling direct visual verification of the predicted count.

Experiment We measure the accuracy of SketchVLMs and also how well they ground their markers on counting datasets (Sec. 4.1). We consider a marker

correct if it lies within the corresponding ground-truth bounding box, obtained via SAM-3 [7], allowing at most one marker per object. Results sketchVLM exhibits strong consistency between counting accuracy (94.5) and numeric-marker location accuracy (95.9), whereas sketchVLM achieves high counting accuracy (75.4) but substantially lower numeric-marker location accuracy (51.0), indicating that sketchVLM often places markers incorrectly despite producing the correct count (Tabs. 2 and C3). attains low counting accuracy (48.6) and numeric-marker location accuracy (59.9), indicating limited performance in both aspects (Tabs. 2 and C3). SketchVLM improves counting accuracy, yielding gains of (+1.5) points for and (+3.4) points for (Tab. 2). The explicit numeric markers further enable direct visual verification of model outputs (Figs. D9 and 5).

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

- 5.4 SketchVLMs localize objects more accurately with pre-defined shape primitives than with free-form annotations

[Figure 170]

- (a) sketchVLM (b) sketchVLM (c) (d) Free-form Ovals Free-form Rectangles Oval Primitive

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

Fig. 6: When prompted to outline the classes “person” and “sports-ball”, (c) replaces the original image with a newly generated one, whereas SketchVLM in (a) and

[Figure 176]

- (b) preserves the original image and draws shapes that accurately align with object boundaries and locations as compared to default (d) .

A design choice we face is whether to have SketchVLMs generate all of their drawings through free-form strokes, or whether to allow them to output shape information (such as the x–y position of the center of the circle and the length of the radius) and then delegate the actual rendering of the drawing to the SVG conversion. Free-form strokes are more flexible but require point-by-point drawing and demand higher spatial precision. In contrast, pre-defined primitives for rectangles, ovals, etc. can be specified by a few parameters and can be rendered automatically. We compare both approaches in object localization in order to understand how it affects the output of annotations.

Experiment We compare SketchVLMs’ stroke-based annotations against the baseline models that directly output parameters for shape locations and shape size (Sec. 4.1). To evaluate oval annotations, we convert the rendered shape into its tight enclosing bounding box before computing metrics. Performance is measured using Average Precision (AP) at an IoU threshold of 0.5.

Results sketchVLM with stroke-based rectangles is effective for medium and large objects, but remains limited for small-object detection. Sketch-based

rectangles slightly improve performance on medium (+0.6) and large objects (+1.4), but significantly degrade small-object detection (-10.1), reducing overall AP50 from 63.1 to 58.8 (Figs. D10 and 6 and Tab. C4).

We observe that stroke-based outputs in SketchVLMs underperform the original model on small objects (Tab. C4). We examine detection statistics and prompt ablations and find that SketchVLMs match the original model in precision but exhibit lower recall (Tab. C5) and do not significantly change with variations in the drawing prompt (Tab. C6). Therefore, in order to get the accuracy of pre-defined shapes as well as the expressiveness of free-form strokes, we allow the model to output both simultaneously, such as in Fig. 1.

###### 5.5 SketchVLM improves localization accuracy for but not for when labeling parts of an object

[Figure 177]

[Figure 178]

Input (a) (b) sketchVLM (c)

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

- Fig. 7: Qualitative comparison on the part labeling task. (b) SketchVLM places each part label directly on its corresponding region while preserving the original image, producing more interpretable part annotations than (a) or (c) .

[Figure 184]

A useful feature of SketchVLMs is pointing at parts of an image and explaining them, for example, labeling engine components in a car maintenance guide (Fig. 1) or annotating regions in a screenshot (Fig. A1). We test SketchVLMs’ ability to produce text that is both semantically correct and spatially well-placed, a capability that underpins step-by-step instructions, explainable visual descriptions, and annotation tools [17,28,56].

Experiment In SketchVLM, the framework automatically renders the predicted text at the predicted location (with adaptive size/color for visibility), whereas the original VLM model outputs only the label and coordinates. For both SketchVLMs and the original VLMs, we prompt the models with a predefined set of valid part labels and require all predictions to be selected strictly from this set. For the original VLMs, we use a Python post-processing pipeline to overlay the predicted label onto the image at the predicted coordinates, where

- Table 4: Labels placed by SketchVLMs land very close to the correct location. sketchVLM is more accurate than the baseline at every tolerance level, and sketchVLM matches it within a few pixels.

[Figure 185]

Dilation r (px) sketchVLM sketchVLM

[Figure 186]

[Figure 187]

| | | |
|---|---|---|
|0 3 5 7 10 15|64.1 60.3 (-3.8)<br><br>69.0 66.5 (-2.5)<br>70.8 69.3 (-1.5)<br>71.8 71.2 (-0.6) 73.1 73.5 (+0.4) 75.1 78.3 (+3.2)<br>|19.1 20.4 (+1.3) 21.9 23.3 (+1.4) 23.9 25.3 (+1.4) 25.6 27.2 (+1.6) 28.6 30.2 (+1.6) 33.7 35.9 (+2.2)|

[Figure 188]

Fig. 8: From r = 0 (red) to modest dilation (r = 7, yellow), small boundary offsets become visually negligible.

text size and color are manually chosen for consistent visibility across examples. We follow [11] and define boundary dilation with radius r as expanding the ground-truth boundary by r pixels in all directions to allow tolerance in spatial matching.

Results SketchVLM improves part labeling for (+1.3) but slightly underperforms at strict boundary matching (-3.8) (Tab. 4). For , SketchVLM becomes increasingly robust under boundary dilation [11], achieving higher accuracy as tolerance increases, with remaining errors dominated by wrong-position mistakes (>79%; Tab. C7). For , the gap narrows from (-3.8) at r = 0 to (-0.6) at r = 7, reaching near parity under modest tolerance (Tab. 4). These errors largely correspond to small boundary offsets that are visually negligible (Figs. D11, 7 and 8), where the original models produce more missing-label errors and SketchVLMs more position errors (Tab. C7). These findings indicate that SketchVLMs primarily introduce minor spatial shifts rather than semantic labeling failures, and remain competitive under reasonable boundary tolerance.

[Figure 189]

[Figure 190]

- 5.6 SketchVLMs outperform models fine-tuned directly on path-tracing tasks

Maze Navigation evaluates spatial reasoning by requiring models to follow a sequence of directions to reach a goal while avoiding obstacles. Given that and are trained on a similar maze task, we expect them to perform well. Experiment Given a set of directions, the model must sketch out the path while also determining if the path reaches the goal without crossing any walls.

[Figure 191]

[Figure 192]

[Figure 193]

Results Surprisingly, we find that has an accuracy of 50.8% and has an accuracy of 62.5%, both of which are near the random choice baseline of 50%. + performs much stronger with an accuracy of 93.3%; however, it sometimes alters the entire image and has other unusual outputs (Fig. D4). Both sketchVLM and sketchVLM perform well with accuracies of 98.0% and 92.8% respectively. The difference in performance between the default model baseline and the SketchVLM mode is minor (Tab. 2), but the annotations benefit user verification.

[Figure 194]

[Figure 195]

[Figure 196]

###### (a) Input (b) sketchVLM (c) Input (d) sketchVLM

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

Input Path: Up, Up, Right, Down, Down : “... the path is invalid.”

Input Path: Up, Right, Up, Right, Down : “... the path is valid.”

[Figure 201]

[Figure 202]

- Fig. 9: Models are presented with a blank maze such as (a) or (c) and are asked to verify whether a proposed path from the green square to the red square is feasible through annotations (b) and (d). SketchVLMs correctly verify both valid and invalid paths by drawing the trajectory and marking where an invalid move occurs.

5.7 Fine-tuned sketching models fail to generalize to unseen physics understanding tasks

[Figure 203]

Input sketchVLM ThinkMorph ViLaSR

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

- Fig. 10: SketchVLM generates the most accurate Ball Drop images compared to other baselines.

In addition to spatial reasoning, we evaluate whether SketchVLM can predict trajectories involving physical dynamics, such as a ball falling and rolling. Experiment Given an image with a ball and platforms, the model must sketch the ball’s trajectory and output the container number it lands in.

[Figure 212]

Results We find that and perform poorly on VPCT with accuracies of 37% and 27%, which are near the random choice baseline of 33.3%. On our Ball Drop dataset, they have accuracies of 35.9% and 30.3%, only slightly above the random choice baseline of 25.0%, showing how these models fail to generalize to tasks that they are not trained on. ++ often removes ledges in the image (Fig. D1) and performs significantly worse than the baseline accuracy (Tab. 2). In contrast, SketchVLMs output coherent annotations and have high accuracy, and sketchVLM reaches 96.0% accuracy on VPCT and 79.7% on Ball Drop

[Figure 213]

[Figure 214]

- (Fig. 10).

###### 5.8 SketchVLM has higher annotation–text alignment than image-editing and fine-tuned models

- Table 5: VLM-judged annotation–text alignment (higher is better) and annotation quality (1–5, higher is better). Models fine-tuned to generate annotations show lower alignment, while sketchVLM has the highest alignment and annotation quality.

Model Annotation–text Alignment Annotation Quality (1-5)

| |VPCT Ball Drop Maze Navigation Mean<br><br>|VPCT Ball Drop Maze Navigation Mean|
|---|---|---|
|[Figure 215]<br><br>sketchVLM sketchVLM|99.0 99.0 88.4 95.5<br>100.0 98.5 84.2 94.2<br>|1.83 1.74 3.20 2.26 3.12 4.28 3.69 3.70<br><br>|
|[Figure 216]<br><br>++<br><br>[Figure 217]<br><br>[Figure 218]|93.0 41.4 80.3 71.6 32.0 45.5 8.2 28.6 54.0 38.4 48.1 46.8|1.56 2.56 3.68 2.60 1.36 1.28 2.78 1.81 1.62 2.11 1.17 1.63|

Experiment To measure annotation–text alignment, we show each model’s annotations for VPCT, Ball Drop and Maze Navigation (without its text answer) to a VLM judge (Gemini-3-Flash-Preview [12]) and ask it to infer what the final answer should be from the annotations alone.

Results Despite being training-free, SketchVLMs achieve substantially higher annotation–text alignment than all baselines (Tab. 5). sketchVLM and sketchVLM reach mean alignment scores of 95.5% and 94.2%, respectively, compared to 71.6% for ++ , 46.8% for , and 28.6% for . The low alignment of the fine-tuned models means their annotations frequently contradict their own text answers, making them unreliable as visual explanations. In contrast, SketchVLMs’ high alignment means that users can reliably look at the annotations to verify whether the model’s reasoning makes sense, which is the framework’s primary goal.

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

###### 5.9 sketchVLM has significantly higher quality annotations than image-editing and fine-tuned models

Beyond faithfulness, an annotation must also be visually clear and logically coherent to be useful. A trajectory that clips through walls, a label that floats in empty space, or uninformative, overlapping annotations all undermine the user’s ability to interpret the model’s reasoning, even if the final text answer is correct

- (Fig. 11). Experiment As described in Sec. 4.4, we measure drawing quality using a VLM-as-a-Judge approach with Gemini-3-Flash-Preview [12]. Each annotation is scored on a 1–5 rubric (Sec. E) in which the judge is asked to evaluate annotation plausibility and visual clarity. Results Based on VLM-as-a-Judge ratings, sketchVLM achieves the highest mean annotation quality score of 3.70, followed by ++ at 2.60, sketchVLM

[Figure 223]

[Figure 224]

[Figure 225]

###### Input

ViLaSR

sketchVLM

[Figure 226]

[Figure 227]

[Figure 228]

Annotation Quality Score: 1 “The physics are completely unrealistic.”

Annotation Quality Score: 5 “The path is logical,

Prompt: “Draw the path of the ball.”

[Figure 229]

[Figure 230]

follows gravity, and does not clip through any walls.” Input

[Figure 231]

[Figure 232]

ThinkMorph

sketchVLM

[Figure 233]

[Figure 234]

[Figure 235]

Annotation Quality Score: 1 “The drawn path

Annotation Quality Score: 5

Prompt: “Draw the proposed path: Right, Right, Down, Down.”

[Figure 236]

“The sketch follows the proposed path...avoids all solid black walls...There are no logical errors.”

[Figure 237]

completely contradicts the given text path...clips through two solid black walls.”

[Figure 238]

[Figure 239]

- Fig. 11: Low-quality annotations from ThinkMorph and ViLaSR may still lead to the correct final answer, but contain logical errors that are harder for users to verify than the high-quality annotations from SketchVLMs.

[Figure 240]

at 2.26, at 1.81, and at 1.63 (Tab. 5). Similarly, based on human ratings, sketchVLM achieves the highest mean quality score of 4.14, followed by sketchVLM at 3.70, Nano Banana Pro at 3.08, at 1.74, and at 1.24 (Tab. E4). Notably, ++ produces annotations that appear visually polished but sometimes contain subtle logical errors (e.g., trajectories passing through solid platforms) that even the VLM judge can miss (Sec. E.1). The fine-tuned models and score lowest, and manual inspection confirms that their annotations are often incoherent and difficult to interpret (Sec. E.1). These low quality scores also correlate with their low task accuracy in Tab. 2, suggesting that models unable to produce clear annotations also struggle to reason about the underlying tasks.

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

###### 5.10 VLM judge ratings positively correlate with human judgments

We validate our annotation quality ratings from our VLM judge (Gemini-3Flash-Preview [12] (Tab. F1)) against three human annotators on 2,250 annotations across Ball Drop, VPCT, and Maze Navigation using a quadratic Kappa score and Pearson Correlation. Agreement between humans and the VLM judge is moderate (quadratic Kappa of 0.51 ± 0.02, Pearson correlation of 0.52 ± 0.01) compared to high inter-human agreement (quadratic Kappa of 0.84±0.04, Pearson correlation of 0.85±0.04). Further details are in Sec. E. While the VLM judge is not as consistent as the human evaluators, its ratings are positively correlated with human judgments across all tasks and models. We therefore treat the VLM judge scores as a useful proxy metric for quick and cost-effective evaluation that should be taken into consideration with both task accuracy and annotation–text alignment.

###### 5.11 Single-turn is as accurate as multi-turn but requires significantly fewer turns

[Figure 248]

First, you need to select your subject by clicking Select...

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

thebackground? 1

next? 2

How do I remove

What do I do

To make the background transparent, add a mask…

Final Output ✅

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

Click the layer mask button

Click ‘Select’, then ‘Subject’

[Figure 267]

- Fig. 12: Multi-turn example of SketchVLM guiding a user through how to remove an image’s background. At each turn, the model receives a screenshot then annotates the screenshot with labeled arrows and highlights UI elements to indicate the next step.

[Figure 268]

[Figure 269]

[Figure 270]

We want SketchVLMs to be able to visually guide users through tasks that require multiple turns, such as removing the background of a photo (Fig. 12) or setting up an EC2 instance on AWS (Fig. A1). We therefore evaluate how multiturn generation affects model accuracy compared to single-turn, and identify the most effective configuration for multi-turn interaction.

Experiment We compare single-turn and multi-turn generation on VPCT,

[Figure 271]

[Figure 272]

[Figure 273]

, , and . In multi-turn, the model receives all previously drawn strokes rendered onto the input image at each turn. We also test whether providing the text representation of rendered strokes across turns is necessary for maintaining annotation quality.

First, you need to select your subject by clicking Select...

[Figure 274]

[Figure 275]

[Figure 276]

thebackground? 1

next? 2

How do I remove

What do I do

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

Click ‘Select’, then ‘Subject’

Results Single-turn achieves comparable or higher accuracy than multi-turn across all tasks (Tab. 3), while requiring about 5.92x fewer turns (Tab. C9). This demonstrates that SketchVLMs can produce high-quality annotations and accurate answers in a single pass. When we remove the text representation of prior strokes to force the model to rely solely on the rendered image, we observe notable degradation in annotation quality (Fig. D15). We therefore report all

multi-turn results with both the rendered image and the text history provided to the model.

##### 6 Conclusion

We present SketchVLM, a training-free framework that prompts frontier VLMs to produce editable, non-destructive SVG annotations grounded on the input image. These visual explanations let users verify model reasoning at a glance, something that text-only responses and image-editing baselines fail to reliably provide. Our results show that this approach outperforms sketching models by +28.5 percentage points in accuracy (Tab. C1) and +48.3% in annotation quality (Tab. C2), while generalizing well to real-world tasks (Sec. A).

Limitations and Future Work SketchVLM works best with and and can transfer to strong open-source VLMs like Kimi K2.5 [44] (Tab. C8 and Sec. D), but does not perform well on small VLMs that struggle with instruction following like Qwen2.5-VL-7B [3]. Additionally, enabling models to undo and edit strokes could be explored as a way to improve multi-turn performance.

[Figure 282]

##### 7 Author contribution statement

Brandon Collins (BC), Logan Bolton (LB), and Hung Nguyen (HN) are major contributors who (a) created or curated benchmark datasets and (b) ran experiments.

- – HN curated the Counting Objects, Drawing Shapes around Objects, and Part Labeling benchmarks from existing datasets, led their evaluation, and led the human-versus-VLM agreement study.
- – BC created the Connect-the-Dots benchmark and led method development, inference and evaluation code, ablations, and evaluation on Connect-theDots and VPCT.
- – LB created the Ball Drop and Maze Navigation benchmarks, led their evaluation, and ran the open-source model experiments as well as the annotationquality and annotation-text alignment evaluations.

BC and LB led the writing of the manuscript while all authors contributed to editing and reviewing. BC, LB, and Anh Nguyen (AN) developed the demo. BC led development of the project website, with additional contributions from LB. BC and LB are technical team leads. AN supervised the project.

###### Acknowledgement

We thank Pooyan Rahmanzadehgervi at Auburn University for feedback and discussions of results. AN was supported by the NSF Grant No. 2145767, and donations from NaphCare Foundation & Adobe Research. LB was supported by the Auburn University URF program. HN was supported by the Auburn University AU PGRF program.

##### References

- 1. Acharya, M., Kafle, K., Kanan, C.: Tallyqa: Answering complex counting questions. In: Proceedings of the AAAI conference on artificial intelligence. vol. 33, pp. 8076– 8084 (2019)
- 2. Allen Institute for AI: Molmo: An open vision-language model from allen ai. https: //github.com/allenai/molmo (2024), open-source multimodal model family for vision-language tasks; accessed 2026-01-18
- 3. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J.: Qwen2.5-vl technical report (2025), https://arxiv.org/abs/2502.13923
- 4. Bakhtin, A., van der Maaten, L., Johnson, J., Gustafson, L., Girshick, R.: Phyre: A new benchmark for physical reasoning. arXiv:1908.05656 (2019)
- 5. Beyer, L., Steiner, A., Pinto, A.S., Kolesnikov, A., Wang, X., Salz, D., Neumann, M., Alabdulmohsin, I., Tschannen, M., Bugliarello, E., et al.: Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726 (2024)
- 6. Bloomberg Intelligence: Generative ai outlook. Tech. rep., Bloomberg, New York

(2025), https://assets.bbhub.io/professional/sites/41/Generative-AIOutlook.pdf, accessed: 2026-01-18

- 7. Carion, N., Gustafson, L., Hu, Y.T., Debnath, S., Hu, R., Suris, D., Ryali, C., Alwala, K.V., Khedr, H., Huang, A., et al.: Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719 (2025)
- 8. cbrower: Vpct ball drop benchmark. https://cbrower.dev/vpct (2025), accessed: 2025-11-09
- 9. Chen, D., Chen, R., Zhang, S., Wang, Y., Liu, Y., Zhou, H., Zhang, Q., Wan, Y., Zhou, P., Sun, L.: MLLM-as-a-judge: Assessing multimodal LLM-as-a-judge with vision-language benchmark. In: Salakhutdinov, R., Kolter, Z., Heller, K., Weller, A., Oliver, N., Scarlett, J., Berkenkamp, F. (eds.) Proceedings of the 41st International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 235, pp. 6562–6595. PMLR (21–27 Jul 2024), https://proceedings. mlr.press/v235/chen24h.html
- 10. Chen, X., Mottaghi, R., Liu, X., Fidler, S., Urtasun, R., Yuille, A.: Detect what you can: Detecting and representing objects using holistic models and body parts. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1971–1978 (2014)
- 11. Cheng, B., Girshick, R., Dollar, P., Berg, A.C., Kirillov, A.: Boundary iou: Improving object-centric image segmentation evaluation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 15334–15342 (June 2021)
- 12. DeepMind, G.: Gemini 3 flash: frontier intelligence built for speed. The Keyword (Google Blog) (Dec 2025), https://blog.google/products-and-platforms/ products/gemini/gemini-3-flash/
- 13. DeepMind, G.: Introducing nano banana pro (Nov 2025), https://blog.google/ innovation-and-ai/products/nano-banana-pro/, . Accessed: 2026-01-25
- 14. Deitke, M., Clark, C., Lee, S., Tripathi, R., Yang, Y., Park, J.S., Salehi, M., Muennighoff, N., Lo, K., Soldaini, L., et al.: Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 91–104 (2025)

- 15. Deng, C., Zhu, D., Li, K., Gou, C., Li, F., Wang, Z., Zhong, S., Yu, W., Nie, X., Song, Z., Shi, G., Fan, H.: Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683 (2025)
- 16. Douglas, D.H., Peucker, T.K.: Algorithms for the reduction of the number of points required to represent a digitized line or its caricature. The Canadian Cartographer 10(2), 112–122 (1973)
- 17. Evernote Corporation: Skitch: Snap. mark up. share. (2026), https://apps.apple. com/us/app/skitch-snap-mark-up-share/id425955336, accessed: 2026-01-28
- 18. Gu, J., Hao, Y., Wang, H.W., Li, L., Shieh, M.Q., Choi, Y., Krishna, R., Cheng, Y.: Thinkmorph: Emergent properties in multimodal interleaved chain-of-thought reasoning. arXiv preprint arXiv:2510.27492 (2025)
- 19. Hu, Y., Shi, W., Fu, X., Roth, D., Ostendorf, M., Zettlemoyer, L., Smith, N.A., Krishna, R.: Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. Advances in Neural Information Processing Systems 37, 139348–139379 (2024)
- 20. Izadi, A., Banayeeanzade, M.A., Askari, F., Rahimiakbar, A., Vahedi, M.M., Hasani, H., Soleymani Baghshah, M.: Visual structures helps visual reasoning: Addressing the binding problem in vlms. arXiv preprint arXiv:2506.22146 (2025). https://doi.org/10.48550/arXiv.2506.22146
- 21. Latif, E., Khan, Z., Zhai, X.: Sketchmind: A multi-agent cognitive framework for assessing student-drawn scientific sketches. arXiv preprint arXiv:2507.22904 (2025)
- 22. Lei, X., Yang, Z., Chen, X., Li, P., Liu, Y.: Scaffolding coordinates to promote vision-language coordination in large multi-modal models. In: Rambow, O., Wanner, L., Apidianaki, M., Al-Khalifa, H., Eugenio, B.D., Schockaert, S. (eds.) Proceedings of the 31st International Conference on Computational Linguistics. pp. 2886–2903. Association for Computational Linguistics, Abu Dhabi, UAE (Jan 2025), https://aclanthology.org/2025.coling-main.195/
- 23. Li, C., Wu, W., Zhang, H., Xia, Y., Mao, S., Dong, L., Vulić, I., Wei, F.: Imagine while reasoning in space: Multimodal visualization-of-thought (2025), https:// arxiv.org/abs/2501.07542
- 24. Li, H., Wu, J., Sun, Q., Li, G., Tian, J., Zhang, H., Lai, Y., An, R., Peng, H., Dai, Y., et al.: Gebench: Benchmarking image generation models as gui environments. arXiv preprint arXiv:2602.09007 (2026)
- 25. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: European conference on computer vision. pp. 740–755. Springer (2014)
- 26. Masters, K.: Why OpenAI’s ad announcement should worry retail media networks (Jan 2026), https://www.thedrum.com/opinion/why-openai-s-adannouncement-should-worry-retail-media-networks
- 27. Menon, S., Zemel, R., Vondrick, C.: Whiteboard-of-thought: Thinking step-by-step across modalities. arXiv (2024)
- 28. Microsoft Corporation: Draw on slides during a presentation (2026), https: / / support . microsoft . com / en - us / office / draw - on - slides - during - a presentation-80a78a11-cb5d-4dfc-a1ad-a26e877da770, accessed: 2026-01-28
- 29. Nguyen, T., Bolton, L., Taesiri, M.R., Bui, T., Nguyen, A.T.: Hot: Highlighted chain of thought for referencing supporting facts from inputs. arXiv preprint arXiv:2503.02003 (2025)
- 30. OpenAI: Openai gpt-5 system card (2025), https://arxiv.org/abs/2601.03267
- 31. OpenAI: Fix with chatgpt (Feb 2026), https://www.youtube.com/watch?v= PHKpsVIdAcc

- 32. Openclipart Contributors: Openclipart silhouette collection. https : //openclipart.org/search/?query=silhouette (2025), accessed: 2025-1110
- 33. Ou, S., Liu, H., Wang, P., Liao, Y., Xuan, C., Wang, Y., Wang, Y.: Bridging the dynamic perception gap: Training-free draft chain-of-thought for dynamic multimodal spatial reasoning (2025), https://arxiv.org/abs/2505.16579
- 34. Paiss, R., Ephrat, A., Tov, O., Zada, S., Mosseri, I., Irani, M., Dekel, T.: Teaching CLIP to Count to Ten. arXiv preprint arXiv:2302.12066 (2023)
- 35. Perez, S.: Chatgpt’s user growth has slowed, report finds | techcrunch (12 2025), https://techcrunch.com/2025/12/05/chatgpts-user-growth-has-slowedreport-finds/, [Online; accessed 2026-01-28]
- 36. Pichai, S., Hassabis, D., Kavukcuoglu, K.: A new era of intelligence with gemini 3. The Keyword (Google Blog) (Nov 2025), https://blog.google/products-andplatforms/products/gemini/gemini-3/
- 37. Ramanathan, V., Kalia, A., Petrovic, V., Wen, Y., Zheng, B., Guo, B., Wang, R., Marquez, A., Kovvuri, R., Kadian, A., et al.: Paco: Parts and attributes of common objects. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7141–7151 (2023)
- 38. Ribeiro, L.S.F., Bui, T., Collomosse, J., Ponti, M.: Sketchformer: Transformerbased representation for sketched structure (2020), https://arxiv.org/abs/2002. 10381
- 39. Shah, B.A.: Keep ai browsers out of your enterprise, warns gartner – computerworld, https://www.computerworld.com/article/4102569/keep-ai-browsersout-of-your-enterprise-warns-gartner.html?utm_source=chatgpt.com, [Online; accessed 2026-01-28]
- 40. Su, Z., Li, L., Song, M., Hao, Y., Yang, Z., Zhang, J., Chen, G., Gu, J., Li, J., Qu, X., et al.: Openthinkimg: Learning to think with images via visual tool reinforcement learning. arXiv preprint arXiv:2505.08617 (2025)
- 41. Taesiri, M.R., Collins, B., Bolton, L., Lai, V.D., Dernoncourt, F., Bui, T., Nguyen, A.T.: Understanding generative ai capabilities in everyday image editing tasks. arXiv preprint arXiv:2505.16181 (2025). https://doi.org/10.48550/arXiv. 2505.16181, https://arxiv.org/abs/2505.16181, version 2, submitted 26 May 2025
- 42. Team, C.: Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818 (2024)
- 43. Team, G., Anil, R., Borgeaud, S., Alayrac, J.B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A.M., Hauth, A., Millican, K., et al.: Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805 (2023)
- 44. Team, K.: Kimi k2.5: Visual agentic intelligence (2026), https://arxiv.org/abs/ 2602.02276
- 45. Vikhyat: Moondream: Tiny vision language model. https://github.com/vikhyat/ moondream (2023), open-source vision-language model with small-footprint multimodal capabilities
- 46. Vinker, Y., Shaham, T.R., Zheng, K., Zhao, A., E Fan, J., Torralba, A.: Sketchagent: Language-driven sequential sketch generation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 23355–23368 (2025)
- 47. Wang, Z., Hsu, J., Wang, X., Huang, K.H., Li, M., Wu, J., Ji, H.: Visually descriptive language model for vector graphics reasoning. Transactions on Machine Learning Research (2025), https://openreview.net/forum?id=WzS33L1iPC

- 48. Wu, J., Guan, J., Feng, K., Liu, Q., Wu, S., Wang, L., Wu, W., Tan, T.: Reinforcing spatial reasoning in vision-language models with interwoven thinking and visual drawing (2025), https://arxiv.org/abs/2506.09965
- 49. Wu, P., Xie, S.: V*: Guided visual search as a core mechanism in multimodal llms

(2023). https://doi.org/10.48550/arXiv.2312.14135, https://arxiv.org/ abs/2312.14135

- 50. Yu, T., et al.: Visual prompting in multimodal large language models: A survey. arXiv preprint arXiv:2409.15310 (2024). https://doi.org/10.48550/arXiv. 2409.15310, https://arxiv.org/abs/2409.15310
- 51. Zhang, C., Qiu, H., Zhang, Q., Zeng, Z., Ma, L., Zhang, J.: Deepsketcher: Internalizing visual manipulation for multimodal reasoning (2025), https://arxiv.org/ abs/2509.25866
- 52. Zhang, H., Wu, W., Li, C., Shang, N., Xia, Y., Huang, Y., Zhang, Y., Dong, L., Zhang, Z., Wang, L., Tan, T., Wei, F.: Latent sketchpad: Sketching visual thoughts to elicit multimodal reasoning in mllms. arXiv preprint arXiv:2510.24514 (2025)
- 53. Zhang, J., Khayatkhoei, M., Chhikara, P., Ilievski, F.: MLLMs know where to look: Training-free perception of small visual details with multimodal LLMs. In: The Thirteenth International Conference on Learning Representations (2025), https: //arxiv.org/abs/2502.17422
- 54. Zhao, S., Zhang, H., Lin, S., Li, M., Wu, Q., Zhang, K., Wei, C.: Pyvision: Agentic vision with dynamic tooling. (2025), https://agents-x.space/pyvision/
- 55. Zhou, R., Nguyen, G., Kharya, N., Nguyen, A.T., Agarwal, C.: Improving human verification of llm reasoning through interactive explanation interfaces. arXiv preprint arXiv:2510.22922 (2025)
- 56. Zoom Video Communications, Inc.: Using annotation tools for collaboration

(2026), https://support.zoom.com/hc/en/article?id=zm_kb&sysparm_ article=KB0067931, accessed: 2026-01-28

- 57. Zou, K., Huang, Z., Dong, Y., Tian, S., Zheng, D., Liu, H., He, J., Liu, B., Qiao, Y., Liu, Z.: Uni-MMMU: A massive multi-discipline multimodal unified benchmark. arXiv preprint arXiv:2510.13759 (2025)

#### Table of Contents

SketchVLM: Vision language models can annotate images to explain thoughts and guide users . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1

Brandon Collins, Logan Bolton, Hung Huy Nguyen, Mohammad Reza Taesiri, Trung Bui, Anh Totti Nguyen

- 1 Introduction. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
- 2 Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 3 SketchVLM . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 4 Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 4.1 7 Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 4.2 Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 4.3 Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 4.4 Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8

- 5 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 5.1 Grid prompting improves sketchVLM annotation precision, but hurts sketchVLM . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

[Figure 283]

- 5.2 SketchVLMs can localize points and connect them in order 9

[Figure 284]

- 5.3 SketchVLM improves counting accuracy . . . . . . . . . . . . . . . . . . 10

[Figure 285]

- 5.4 SketchVLMs localize objects more accurately with pre-defined shape primitives than with free-form annotations . . . . . . . . 11

[Figure 286]

- 5.5 SketchVLM improves localization accuracy for but not for when labeling parts of an object . . . . . . . . . . . . . . . . . . . . . . 12

[Figure 287]

[Figure 288]

- 5.6 SketchVLMs outperform models fine-tuned directly on path-tracing tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 5.7 Fine-tuned sketching models fail to generalize to unseen physics understanding tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

[Figure 289]

- 5.8 SketchVLM has higher annotation–text alignment than image-editing and fine-tuned models . . . . . . . . . . . . . . . . . . . . . . . . . 15
- 5.9 sketchVLM has significantly higher quality annotations than image-editing and fine-tuned models . . . . . . . . . . . . . . . . . . . . 15
- 5.10 VLM judge ratings positively correlate with human judgments . . 17
- 5.11 Single-turn is as accurate as multi-turn but requires significantly fewer turns . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

- 6 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- 7 Author contribution statement. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- A Real World Applications . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- B Dataset Creation. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

- B.1 Connect-the-Dots . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- B.2 Counting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- B.3 Drawing Shapes around Objects . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- B.4 Part Labeling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

- B.5 Maze Navigation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- B.6 Ball Drop . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

- C Additional Task Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29

- C.1 Combined Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- C.2 Connect-the-Dots . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- C.3 Counting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- C.4 Drawing Shapes around Objects . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- C.5 Part Labeling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- C.6 Additional Model Results. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32

- D Qualitative Samples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33

- D.1 Ball Drop . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
- D.2 Maze Navigation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
- D.3 Connect Dots . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
- D.4 Counting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
- D.5 Drawing Shape. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
- D.6 Part Labeling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
- D.7 Connect-the-Dots: Grid versus No Grid . . . . . . . . . . . . . . . . . . . . . . 44
- D.8 Connect-the-Dots: Bézier Curves versus Lines . . . . . . . . . . . . . . . . . 45
- D.9 Gemini-3-Pro-Preview Coordinate Systems . . . . . . . . . . . . . . . . . . . 46
- D.10 Multi-turn Ablation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47

- E VLM-Judge Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48

- E.1 Qualitative Examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50
- E.2 Rubric Prompts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 55

- F Model Settings and Prompts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 57

- F.1 API Settings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 57
- F.2 SketchVLM System Prompt . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 58
- F.3 SketchVLM Output Example . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 62

- G Other Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 63 G.1 Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 63

##### A Real World Applications

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

- Fig. A1: Beginning at the top left image, our SketchVLM framework visually guides the user how to set up a free EC2 instance through the notoriously non-user-friendly Amazon Web Services web interface.

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

- Fig. A2: SketchVLM responds via visual annotations while ChatGPT responds with text only.

[Figure 310]

[Figure 311]

[Figure 312]

How do I turn off my hot water?

Where is the ground nutmeg?

I’ve got two sticks of RAM. Where do they go?

Fig. A3: SketchVLM can be used in a variety of real-world use cases.

1

[Figure 313]

Go to searchbar and lookup an official PyTorch image.

How do I setup a PyTorch container?

2

[Figure 314]

Download the pytorch image that has 10 million downloads.

3

[Figure 315]

Which of these images do I pick?

Click the play button to start and then click exec to interact…

Which of these images do I pick?

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

Click ‘Select’, then ‘Subject’

[Figure 326]

[Figure 327]

Search for ‘PyTorch’

[Figure 328]

Interact with container

Start container

Click pytorch/pytorch

###### Fig. A4: Multi-turn session where the model explains how to set up PyTorch in a Docker container.

After committing this file, all future pushes will have the linter enabled.

To setup your linter, you should configure Pylint for your project

[Figure 329]

[Figure 330]

[Figure 331]

CI/CDlinter? 1

next? 2

it? 3

How do I setup a

What do I do

How do I enable

First, navigate to the GitHub Actions tab.

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

Click ‘Select’, then ‘Subject’

Click ‘Configure’

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

Click ‘Actions’

ClickClick‘Actions’ ‘Commit’

Click ‘Actions’

| |
|---|

###### Fig. A5: Multi-turn session where the model explains how to set up a CI/CD pipeline in GitHub.

- B Dataset Creation

- B.1 Connect-the-Dots

We collect 100 connect-the-dots images spanning three subsets with varying dot counts and background clutter.

- 1. Random dots: We use a Python script with the Pillow image library to generate 3 images each with 4, 5...10 dots randomly placed for a total of 21 images. Each dot has a number that corresponds to the order it should be connected.
- 2. Outlines: We convert 30 silhouette SVGs from Openclipart [32] into connectthe-dots puzzles. Each SVG was flattened to polylines, simplified with the Douglas–Peucker algorithm [16], and normalized to a unit square. The main contour was then resampled to place 30 evenly spaced dots along the boundary, with number labels slightly offset. We manually filtered out cases with distorted or self-intersecting shapes, which often occurred for concave outlines.
- 3. Worksheets: We gather 49 images of preexisting connect-the-dot worksheets from online sources. These images differ from the outlined images in how they often contain irrelevant information such as lines at the top of the image for students to mark their name. To obtain the ground truth strokes for these images, we manually annotate the coordinates of each dot.

- B.2 Counting

The counting dataset combines three sources: CountBench [5], TallyQA [1], and Pixmo-Count [14]. The CountBench and TallyQA subsets together contain 746 samples, covering object counts ranging from 0 to 10. Pixmo-Count contributes 443 samples after removing unsuitable cases from the original 526-image test split and similarly includes object counts from 1 to 10.

- B.3 Drawing Shapes around Objects

The testing dataset consists of 1,000 carefully selected images from the 5,000 COCO validation images, chosen to ensure a balanced distribution of object counts across classes and object sizes (small, medium, and large).

- B.4 Part Labeling

We carefully selected images from two datasets, PACO [37] and Pascal-Part [10]. The selected images satisfy the following criteria:

- 1. Each image contains only one object corresponding to the target class name.
- 2. The object’s size occupies at least 10% of the total image area.
- 3. Each selected object has at least four part labels annotated.
- 4. The dataset maintains a balanced distribution of objects across different classes.

After selection, the final dataset used for the part labeling task consists of 985 images covering 52 class names.

###### B.5 Maze Navigation

Given a start point, an end point, and a set of direction commands (e.g., Up, Down, Left, Right), the model must determine if the path reaches the goal without crossing any border walls. We create 200 unique 3x3 grids where the shortest path length between the starting green square and ending red square varies from 3 to 8 steps. For each maze, we take the ground truth path and randomly change one of the direction steps in order to make an invalid path (for example, Left, Right, Down could be changed to Right, Right, Down) (Fig. 9).

###### B.6 Ball Drop

We evaluate our framework using Visual Physics Comprehension Test (VPCT [8]), which consists of 100 hand-crafted images where the model must determine which of the buckets the ball will fall into after it is dropped. While VPCT does provide a simple way to evaluate physics understanding of VLMs, it does not contain any ground truth data of the trajectory of the ball. Therefore, in order to evaluate how well SketchVLM models can draw the true trajectory of the ball paths, we generate our own benchmark (Ball Drop). We simulate the trajectory of the ball using PHYRE [4] to obtain ground truth ball trajectory data. In contrast to VPCT, our Ball Drop benchmark is synthetically generated. We generate 198 unique images, with an equal number of images containing 1, 2, and 3 randomly placed lines. We randomize the ball’s X position and fix its Y position near the top. There are four containers at the bottom of the image compared to VPCT’s three containers, making it harder to guess answers correctly.

##### C Additional Task Results

###### C.1 Combined Results

Table C1: SketchVLM improves visual reasoning task accuracy by +28.5 points over alternative sketching approaches. Averages are computed over VPCT, Ball Drop, Maze, and Counting tasks.

[Figure 344]

[Figure 345]

[Figure 346]

Category Model VPCT Avg ∆

| | | |
|---|---|---|
|SketchVLM<br><br>sketchVLM sketchVLM<br><br>[Figure 347]|96.0 79.7 98.0 94.5 70.0 68.5 92.8 75.4<br><br>|84.4 +28.5|
|Other Sketching<br><br>[Figure 348]<br><br>++<br><br>[Figure 349]<br><br>[Figure 350]|63.0 62.6 93.3 91.7 37.0 35.9 50.8 48.6 27.0 30.3 62.5 68.1|55.9|

Table C2: SketchVLM produces higher quality annotations, scoring +48.3% above alternative sketching approaches on a 1–5 VLM-judged drawing quality scale.

[Figure 351]

[Figure 352]

Category Model VPCT Avg ∆

| | | |
|---|---|---|
|SketchVLM (Ours)<br><br>sketchVLM sketchVLM<br><br>[Figure 353]|3.12 4.28 3.69 1.83 1.74 3.20<br><br>|2.98 +48.3%|
|Other Sketching<br><br>[Figure 354]<br><br>++<br><br>[Figure 355]<br><br>[Figure 356]|1.56 2.56 3.68 1.36 1.28 2.78 1.62 2.11 1.17|2.01|

###### C.2 Connect-the-Dots

Random Dots (n=21) Worksheets (n=49) Outlines (n=30) Total (n=100) Model No Grid With Grid ∆ No Grid With Grid ∆ No Grid With Grid ∆ No Grid With Grid ∆ Gemini-2.5-Pro 1212.54 429.86 -64.55% 14769.86 1188.59 -91.95% 7848.27 1597.67 -79.65% 9846.22 1151.49 -88.31% Gemini-3-Pro 15.70 584.32 +3621.78% 48.60 1256.37 +2485.12% 25.00 30436.03 +121644.11% 34.61 9869.14 +28414.46% GPT-5 (low) 11988.22 402.50 -96.64% 16816.74 4627.17 -72.48% 24355.01 2885.04 -88.15% 18063.82 2179.59 -87.93% GPT-5 (med) 3098.15 427.02 -86.22% 15060.98 4627.17 -69.27% 22520.39 2885.04 -87.19% 14785.98 3222.40 -78.21% GPT-5 (high) 2415.63 419.35 -82.64% 14859.81 2219.47 -85.06% 20789.13 3362.99 -83.82% 14024.76 2183.90 -84.43% Qwen3-8B 12667.46 17728.22 +40.00% 80952.85 157661.81 +94.76% 484212.61 542690.99 +12.08% 187590.15 243784.56 +29.96% Qwen3-235B 2857.16 845.90 -70.38% 21472.98 26530.78 +23.56% 34420.65 227671.27 +561.40% 21447.25 81479.11 +280.00% ViLaSR 26658.05 N/A N/A 37153.54 N/A N/A 52312.84 N/A N/A 39497.28 N/A N/A Kimi K2.5 947.41 386.91 -59.16% 8874.52 3256.55 -63.30% 31596.88 6010.74 -80.98% 14026.53 3480.18 -75.19%

Fig. C2: Connect-the-Dots Mean MSE with categories as columns. Each entry shows the mean MSE for with grid and without grid, and the percent change ∆ (negative is better).

###### Connect-the-Dots - gemini-2.5-flash-image

Connect-the-Dots - gemini-3-pro-image-preview

100

100

91.0% (91)

80

80

61.0% (61)

59.0% (59)

60

60

###### Percentage(%)

###### Percentage(%)

37.0% (37)

40

40

20

20

13.0% (13)

9.0% (9)

4.0% (4)

0.0% (0)

0

0

Success No Change Hallucinated Change in Image

Improper Line Order

Success No Change Hallucinated Change in Image

Improper Line Order

Fig. C1: Gemini-2.5-Flash-Image frequently adds non-existent points to the image or does not correctly follow the proper order of the connected dots. Nano Banana Pro is significantly better than Gemini-2.5-Flash-Image, but still only completes the task without any errors 37% of the time.

###### C.3 Counting

Table C3: sketchVLM achieves high text-location accuracy, indicating that predicted counts are well aligned with target objects, whereas sketchVLM shows substantially weaker grounding.

[Figure 357]

|Model<br><br>|Accuracy (%)|
|---|---|
|sketchVLM<br><br>|95.9|
|[Figure 358]<br><br>sketchVLM|51.0<br><br>|
|[Figure 359]|59.9|

###### C.4 Drawing Shapes around Objects

Table C4: Sketch-based localization improves accuracy for medium and large objects, while reducing performance on small objects, compared to coordinate-based bounding boxes (AP50).

sketchVLM sketchVLM Output format Bounding box Rectangle Oval

AP50 (all) 63.1 58.8 (-4.3) 55.4 (-7.7) AP50 (small) 35.6 25.5 (-10.1) 26.4 (-9.2) AP50 (medium) 62.6 63.2 (+0.6) 55.4 (-7.2) AP50 (large) 77.7 79.1 (+1.4) 77.6 (-0.1)

###### Table C5: and SketchVLM achieve the same precision; however, SketchVLM has lower recall than the baseline because it produces fewer true positive (TP) detections.

|GT|P R AP|SketchVLM P R AP|TP sketchVLM|FP sketchVLM<br><br>|FN sketchVLM|
|---|---|---|---|---|---|
|1060|18.7 47.9 33.1|18.7 35.4 26.0|508 375|2204 1629|552 685|

###### Table C6: Different prompt variations do not improve AP performance for the drawing shape task.

| | |SketchVLM Draw rectangles|SketchVLM Draw all visible/part visible|SketchVLM Scan region before draw|SketchVLM Count object before draw|
|---|---|---|---|---|---|
|Output shape|Bounding box|Rectangles|Rectangles|Rectangles<br><br>|Rectangles|
|AP50 (all) AP50 (small) AP50 (medium) AP50 (large)|63.1 35.6 62.6 77.7|58.8 25.5 63.2 79.1|57.1 27.5 60.7 77.8|59.0 26.8 59.9 80.7|56.2 23.2 58.9 78.3|

- C.5 Part Labeling

###### Table C7: Error type breakdown for labeling and part labeling with and original models and SketchVLM.

[Figure 360]

|Wrong Type|(%) SketchVLM(%)|[Figure 361]<br><br>(%) SketchVLM(%)<br><br>[Figure 362]<br><br>|
|---|---|---|
|Missing Label Wrong Position|10.8 3.6 25.1 36.1|0.69 0.32 80.17 79.32<br><br>|
|Total|35.9 39.7|80.86 79.64|

###### C.6 Additional Model Results

Table C8: Ablation across inputs in single-turn mode for additional models. “Sketch” adds strokes/system prompt; “Grid” additionally overlays the coordinate grid. RMSE is reported for Connect-the-Dots while accuracy is reported for the other tasks. “Order Accuracy” is the percentage of connect-the-dots samples with correct point ordering (higher is better).

[Figure 363]

[Figure 364]

[Figure 365]

Model Input VPCT Order Accuracy GPT-5 (med)

Image 63.3 72.2 – 121.60 – + Sketch + Grid 77.0 71.7 – 56.76 –

Image 38.0 31.1 57.5 – – + Sketch + Grid 51.0 50.7 70.0 – –

Gemini-2.5-Flash

Image 50.0 43.5 72.2 99.23 – + Sketch + Grid 61.0 57.5 81.5 33.93 –

Gemini-2.5-Pro

Image 37.0 22.3 46.3 – – + Sketch + Grid 34.0 1.0 34.0 – –

Qwen-2.5VL-7B

Image 44.0 50.50 93.5 118.43 59%

+ Grid 55.0 58.08 97.0 – – + Sketch 56.0 48.48 88.00 – – + Sketch + Grid 56.0 50.50 89.75 58.99 67%

Kimi K2.5

Table C9: Average number of turns per task group in multi-turn evaluation.

Task Gemini-3-Pro GPT-5 (low) VPCT 5.10 6.18

[Figure 366]

5.00 2.58 16.61 19.00 3.42 4.29

[Figure 367]

[Figure 368]

###### Overall 5.91 5.92

##### D Qualitative Samples

###### D.1 Ball Drop

Gemini-3-Pro-Preview Multi-turn

Source Image Gemini-3-Pro-Preview

GPT-5 (low) GPT-5 (low) Multi-turn Kimi K2.5 NanoBanana Pro ThinkMorph ViLaSR

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

GT=3 Answer=3 Answer=3 Answer=3 Answer=2 Answer=2 Answer=2 Answer=2 Answer=1

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

GT=2 Answer=2 Answer=2 Answer=2 Answer=2 Answer=2 Answer=1 Answer=3 Answer=2

[Figure 387]

- GT=1 Answer=1 Answer=1 Answer=1 Answer=1 Answer=1 Answer=1 Answer=1 Answer=1

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

- GT=2 Answer=2 Answer=2 Answer=2 Answer=3 Answer=3 Answer=1 Answer=2 Answer=2

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

- GT=1 Answer=1 Answer=1 Answer=2 Answer=1 Answer=1 Answer=2 Answer=2 Answer=2

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

- GT=2 Answer=2 Answer=2 Answer=3 Answer=2 Answer=1 Answer=2 Answer=2 Answer=2

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

- GT=3 Answer=3 Answer=3 Answer=2 Answer=3 Answer=1 Answer=2 Answer=2 Answer=3

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

- GT=1 Answer=1 Answer=1 Answer=1 Answer=2 Answer=3 Answer=1 Answer=2 Answer=1

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

- GT=1 Answer=1 Answer=1 Answer=3 Answer=1 Answer=3 Answer=2 Answer=2 Answer=3

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

GT=1 Answer=1 Answer=1 Answer=2 Answer=2 Answer=1 Answer=1 Answer=2 Answer=2

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

- GT=2 Answer=2 Answer=2 Answer=2 Answer=2 Answer=2 Answer=2 Answer=N/A Answer=3

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

###### Fig. D1: Qualitative examples of models on VPCT. Gemini-3-Pro-Preview in singleturn produces the most accurate annotations, while NanoBanana Pro, ThinkMorph and ViLaSR often draw paths that cross walls and provide the wrong answer.

Gemini-3-Pro-Preview Multi-turn

Source Image Gemini-3-Pro-Preview

GPT-5 (low) GPT-5 (low) Multi-turn Kimi K2.5 NanoBanana Pro ThinkMorph ViLaSR

[Figure 468]

- GT=1 Answer=1 Answer=1 Answer=1 Answer=3 Answer=1 Answer=3 Answer=1 Answer=3

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

- GT=2 Answer=2 Answer=2 Answer=2 Answer=2 Answer=2 Answer=2 Answer=2 Answer=4

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

- GT=3 Answer=3 Answer=1 Answer=3 Answer=3 Answer=1 Answer=3 Answer=2 Answer=4

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

- GT=2 Answer=2 Answer=2 Answer=4 Answer=4 Answer=2 Answer=1 Answer=2 Answer=2

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

- GT=3 Answer=3 Answer=3 Answer=3 Answer=3 Answer=3 Answer=2 Answer=1 Answer=4

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

- GT=2 Answer=2 Answer=2 Answer=3 Answer=2 Answer=2 Answer=3 Answer=1 Answer=4

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

GT=1 Answer=1 Answer=1 Answer=1 Answer=1 Answer=2 Answer=1 Answer=2 Answer=2

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

- GT=3 Answer=2 Answer=3 Answer=3 Answer=3 Answer=3 Answer=3 Answer=4 Answer=3

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

###### Fig. D2: Qualitative examples of models on our Ball Drop dataset. Gemini-3-ProPreview in single-turn produces the most accurate annotations, while NanoBanana Pro, ThinkMorph and ViLaSR often draw paths that cross walls and provide the wrong answer.

[Figure 540]

###### Fig. D3: The SketchVLM framework boosts VLM accuracy on the Ball Drop task while also letting them draw ball trajectory paths that closely simulate the ground truth data.

###### D.2 Maze Navigation

Gemini-3-Pro-Preview Multi-turn

Source Image Gemini-3-Pro-Preview

GPT-5 (low) GPT-5 (low) Multi-turn Kimi K2.5 NanoBanana Pro ThinkMorph ViLaSR

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

GT=valid Pred=valid Pred=valid Pred=valid Pred=valid Pred=valid Pred=valid Pred=valid Pred=N/A

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

GT=valid Pred=valid Pred=valid Pred=invalid Pred=valid Pred=valid Pred=valid Pred=valid Pred=valid

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

GT=valid Pred=valid Pred=valid Pred=valid Pred=valid Pred=valid Pred=valid Pred=valid Pred=valid

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

GT=valid Pred=valid Pred=valid Pred=valid Pred=invalid Pred=valid Pred=valid Pred=valid Pred=valid

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

GT=valid Pred=valid Pred=valid Pred=invalid Pred=valid Pred=invalid Pred=valid Pred=invalid Pred=valid

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

GT=valid Pred=valid Pred=valid Pred=valid Pred=valid Pred=valid Pred=invalid Pred=valid Pred=valid

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

GT=valid Pred=valid Pred=valid Pred=valid Pred=valid Pred=valid Pred=valid Pred=valid Pred=valid

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

GT=valid Pred=valid Pred=valid Pred=invalid Pred=valid Pred=invalid Pred=valid Pred=invalid Pred=valid

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

GT=valid Pred=valid Pred=valid Pred=valid Pred=valid Pred=valid Pred=valid Pred=valid Pred=valid

###### Fig. D4: Qualitative examples of models on valid paths in Maze Navigation.

Gemini-3-Pro-Preview Multi-turn

Source Image Gemini-3-Pro-Preview

GPT-5 (low) GPT-5 (low) Multi-turn Kimi K2.5 NanoBanana Pro ThinkMorph ViLaSR

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

GT=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=valid

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

GT=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

GT=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=valid

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

GT=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=valid Pred=valid

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

GT=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=valid

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

GT=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=valid

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

GT=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=N/A Pred=invalid Pred=invalid Pred=valid

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

GT=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=valid

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

GT=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=invalid Pred=valid

###### Fig. D5: Qualitative examples of models on invalid paths in Maze Navigation.

###### D.3 Connect Dots

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

Source GT (MSE: 0) Kimi MSE: 1906

Kimi + Grid MSE: N/A

Qwen3-235B MSE: 1960

Qwen3-235B + Grid MSE: 658

Gemini-2.5Pro MSE: 3419

Gemini-2.5Pro + Grid MSE: 299

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

Source GT (MSE: 0) GPT-5 (low) MSE: 34784

GPT-5 (low) + Grid MSE: 387

GPT-5 (med) MSE: 5799

GPT-5 (med) + Grid MSE: 617

GPT-5 (high) MSE: 2545

GPT-5 (high) + Grid MSE: 390

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

Source GT (MSE: 0) ViLaSR MSE: 5951

ThinkMorph MSE: N/A

Gemini-3Pro MSE: 3

Gemini-3Pro + Grid MSE: 537

Gemini-3Pro (multi) MSE: 5

GPT-5 (low) (multi) MSE: 467

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

Source GT (MSE: 0) Kimi MSE: 1279

Kimi + Grid MSE: 479

Qwen3-235B MSE: 417

Qwen3-235B + Grid MSE: 791

Gemini-2.5Pro MSE: 1414

Gemini-2.5Pro + Grid MSE: 542

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

Source GT (MSE: 0) GPT-5 (low) MSE: 1647

GPT-5 (low) + Grid MSE: 428

GPT-5 (med) MSE: 866

GPT-5 (med) + Grid MSE: 364

GPT-5 (high) MSE: 757

GPT-5 (high) + Grid MSE: 428

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

Source GT (MSE: 0) ViLaSR MSE: 275029

ThinkMorph MSE: N/A

Gemini-3Pro MSE: 15

Gemini-3Pro + Grid MSE: 601

Gemini-3Pro (multi) MSE: 5

GPT-5 (low) (multi) MSE: 456

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

Source GT (MSE: 0) Kimi MSE: 573

Kimi + Grid MSE: 644

Qwen3-235B MSE: 3420

Qwen3-235B + Grid MSE: 563

Gemini-2.5Pro MSE: 957

Gemini-2.5Pro + Grid MSE: 624

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

Source GT (MSE: 0) GPT-5 (low) MSE: 3137

GPT-5 (low) + Grid MSE: 412

GPT-5 (med) MSE: 3369

GPT-5 (med) + Grid MSE: 393

GPT-5 (high) MSE: 4506

GPT-5 (high) + Grid MSE: 528

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

Source GT (MSE: 0) ViLaSR MSE: N/A

ThinkMorph MSE: N/A

Gemini-3Pro MSE: 11

Gemini-3Pro + Grid MSE: 698

Gemini-3Pro (multi) MSE: 4

GPT-5 (low) (multi) MSE: 501

Fig. D6: Connect-the-Dots qualitative comparisons on random dots. Each item spans three rows: (top) Kimi/Qwen3-235B/Gemini-2.5-Pro, (middle) GPT-5 (low/med/high), (bottom) multi-turn variants (Gemini-3-Pro and GPT-5 (low)), with ViLaSR and ThinkMorph added to the third row. Each cell shows the overlay and MSE.

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

Source GT (MSE: 0) Kimi MSE: 3446

Kimi + Grid MSE: 1294

Qwen3-235B MSE: 1243

Qwen3-235B + Grid MSE: 817

Gemini-2.5Pro MSE: 2626

Gemini-2.5Pro + Grid MSE: 377

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

Source GT (MSE: 0) GPT-5 (low) MSE: 11618

GPT-5 (low) + Grid MSE: 878

GPT-5 (med) MSE: 7432

GPT-5 (med) + Grid MSE: 1033

GPT-5 (high) MSE: 5420

GPT-5 (high) + Grid MSE: 651

[Figure 791]

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

Source GT (MSE: 0) ViLaSR MSE: 275029

ThinkMorph MSE: N/A

Gemini-3Pro MSE: 6

Gemini-3Pro + Grid MSE: 637

Gemini-3Pro (multi) MSE: 173

GPT-5 (low) (multi) MSE: 959

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

Source GT (MSE: 0) Kimi MSE: 1595

Kimi + Grid MSE: 5989

Qwen3-235B MSE: 3422

Qwen3-235B + Grid MSE: 36302

Gemini-2.5Pro MSE: 4162

Gemini-2.5Pro + Grid MSE: 823

[Figure 807]

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

Source GT (MSE: 0) GPT-5 (low) MSE: 32273

GPT-5 (low) + Grid MSE: 1119

GPT-5 (med) MSE: 26805

GPT-5 (med) + Grid MSE: 1786

GPT-5 (high) MSE: 34658

GPT-5 (high) + Grid MSE: 1521

[Figure 815]

[Figure 816]

[Figure 817]

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

[Figure 822]

Source GT (MSE: 0) ViLaSR MSE: 671536

ThinkMorph MSE: N/A

Gemini-3Pro MSE: 56

Gemini-3Pro + Grid MSE: 3787

Gemini-3Pro (multi) MSE: 104

GPT-5 (low) (multi) MSE: 1982

[Figure 823]

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

Source GT (MSE: 0) Kimi MSE: 1092

Kimi + Grid MSE: N/A

Qwen3-235B MSE: 304443

Qwen3-235B + Grid MSE: 26515

Gemini-2.5Pro MSE: 3728

Gemini-2.5Pro + Grid MSE: 1824

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

Source GT (MSE: 0) GPT-5 (low) MSE: 50688

GPT-5 (low) + Grid MSE: 3109

GPT-5 (med) MSE: 23417

GPT-5 (med) + Grid MSE: 3798

GPT-5 (high) MSE: 47798

GPT-5 (high) + Grid MSE: 6210

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

Source GT (MSE: 0) ViLaSR MSE: 1222760

ThinkMorph MSE: N/A

Gemini-3Pro MSE: 25

Gemini-3Pro + Grid MSE: 3712

Gemini-3Pro (multi) MSE: 32

GPT-5 (low) (multi) MSE: 5732

Fig. D7: Connect-the-Dots (worksheets) qualitative comparisons for connect-the-dot worksheets. Each item spans three rows: (top) Kimi/Qwen3-235B/Gemini-2.5-Pro, (middle) GPT-5 (low/med/high), (bottom) multi-turn variants (Gemini-3-Pro and GPT-5 (low)), with ViLaSR and ThinkMorph added to the third row.

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

[Figure 851]

[Figure 852]

[Figure 853]

| |
|---|

Kimi MSE: 11024

Kimi + Grid MSE: 1503

Qwen3235B MSE: N/A

Qwen3235B + Grid MSE: N/A

Gemini2.5-Pro MSE: 2685

Gemini2.5-Pro + Grid MSE: 1080

Source GT (MSE: 0)

[Figure 854]

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

GPT-5 (low) MSE: 22042

GPT-5 (low) + Grid MSE: 2362

GPT-5 (med) MSE: 27095

GPT-5 (med) + Grid MSE: 2473

GPT-5 (high) MSE: 30159

GPT-5 (high) + Grid MSE: 1134

Source GT (MSE: 0)

[Figure 862]

[Figure 863]

[Figure 864]

[Figure 865]

[Figure 866]

[Figure 867]

[Figure 868]

[Figure 869]

ViLaSR MSE: 2372280

ThinkMorph

Gemini-3Pro MSE: 23

Gemini-3Pro + Grid MSE: 1370

Gemini-3Pro (multi) MSE: 17

GPT-5 (low) (multi) MSE: 1737

Source GT (MSE: 0)

MSE: N/A

[Figure 870]

[Figure 871]

[Figure 872]

[Figure 873]

[Figure 874]

[Figure 875]

[Figure 876]

[Figure 877]

Kimi MSE: 1290

Kimi + Grid MSE: 1595

Qwen3235B MSE: 19328

Qwen3235B + Grid MSE: 34308

Gemini2.5-Pro MSE: 6867

Gemini2.5-Pro + Grid MSE: 1261

Source GT (MSE: 0)

[Figure 878]

[Figure 879]

[Figure 880]

[Figure 881]

[Figure 882]

[Figure 883]

[Figure 884]

[Figure 885]

GPT-5 (low) MSE: 24565

GPT-5 (low) + Grid MSE: 8363

GPT-5 (med) MSE: 30512

GPT-5 (med) + Grid MSE: 10865

GPT-5 (high) MSE: 32064

GPT-5 (high) + Grid MSE: 2765

Source GT (MSE: 0)

[Figure 886]

[Figure 887]

[Figure 888]

[Figure 889]

[Figure 890]

[Figure 891]

[Figure 892]

[Figure 893]

ViLaSR MSE: N/A

ThinkMorph

Gemini-3Pro MSE: 20

Gemini-3Pro + Grid MSE: 1670

Gemini-3Pro (multi) MSE: 23

GPT-5 (low) (multi) MSE: 3047

Source GT (MSE: 0)

MSE: N/A

[Figure 894]

[Figure 895]

[Figure 896]

[Figure 897]

[Figure 898]

[Figure 899]

[Figure 900]

[Figure 901]

Kimi MSE: 2984

Kimi + Grid MSE: 7336

Qwen3235B MSE: 686

Qwen3235B + Grid MSE: 534622

Gemini2.5-Pro MSE: 7418

Gemini2.5-Pro + Grid MSE: 664

Source GT (MSE: 0)

[Figure 902]

[Figure 903]

[Figure 904]

[Figure 905]

[Figure 906]

[Figure 907]

[Figure 908]

[Figure 909]

GPT-5 (low) MSE: 24851

GPT-5 (low) + Grid MSE: 1840

GPT-5 (med) MSE: 20791

GPT-5 (med) + Grid MSE: 1755

GPT-5 (high) MSE: 5917

GPT-5 (high) + Grid MSE: 2447

Source GT (MSE: 0)

[Figure 910]

[Figure 911]

[Figure 912]

[Figure 913]

[Figure 914]

[Figure 915]

[Figure 916]

[Figure 917]

ViLaSR MSE: 426532

ThinkMorph

Gemini-3Pro MSE: 15

Gemini-3Pro + Grid MSE: 2702

Gemini-3Pro (multi) MSE: 28

GPT-5 (low) (multi) MSE: 2327

Source GT (MSE: 0)

MSE: N/A

Fig. D8: Connect-the-Dots (outlines) qualitative comparisons for connect-the-dot outlines. Each item spans three rows: (top) Kimi/Qwen3-235B/Gemini-2.5-Pro pairs, (middle) GPT-5 (low/med/high), (bottom) multi-turn variants (Gemini-3-Pro and GPT-5 (low)), with ViLaSR and ThinkMorph added to the third row.

###### D.4 Counting

Original Nano Banana Gemini 3.0 Pro SketchVLM Gemini 3.0 Pro GPT-5 SketchVLM

How many people are there? GT: 5

[Figure 918]

[Figure 919]

[Figure 920]

- Pred: 5 Pred: 5 Pred: 4

[Figure 921]

[Figure 922]

How many sausages are there? GT: 5

[Figure 923]

[Figure 924]

[Figure 925]

- Pred: 6 Pred: 5 Pred: 6

[Figure 926]

[Figure 927]

###### How many guitars are there? GT: 5

[Figure 928]

[Figure 929]

[Figure 930]

[Figure 931]

[Figure 932]

Pred: 5 Pred: 5 Pred: 5

Original Nano Banana Gemini 3.0 Pro SketchVLM Gemini 3.0 Pro GPT-5 SketchVLM

How many airplanes are there? GT: 9

[Figure 933]

[Figure 934]

[Figure 935]

[Figure 936]

[Figure 937]

Pred: 9 Pred: 9 Pred: 10

###### How many helicopters are there? GT: 5

[Figure 938]

[Figure 939]

[Figure 940]

[Figure 941]

[Figure 942]

Pred: 5 Pred: 5 Pred: 5

###### How many cups are there? GT: 4

[Figure 943]

[Figure 944]

[Figure 945]

[Figure 946]

[Figure 947]

Pred: 4 Pred: 4 Pred: 4

Original Nano Banana Gemini 3.0 Pro SketchVLM Gemini 3.0 Pro GPT-5 SketchVLM

How many airplanes are there? GT: 9

[Figure 948]

[Figure 949]

[Figure 950]

[Figure 951]

[Figure 952]

Pred: 9 Pred: 9 Pred: 7

###### How many horses are there? GT: 6

[Figure 953]

[Figure 954]

[Figure 955]

- Pred: 6 Pred: 6 Pred: 6

[Figure 956]

[Figure 957]

How many people are there? GT: 7

[Figure 958]

[Figure 959]

[Figure 960]

- Pred: 7 Pred: 7 Pred: 7

[Figure 961]

[Figure 962]

###### Fig. D9: Qualitative comparison on counting task between sketchVLM , sketchVLM, and .

[Figure 963]

[Figure 964]

###### D.5 Drawing Shape

Original Nano Banana Gemini 3.0 Pro SketchVLM (Ovals) Gemini 3.0 Pro SketchVLM (Rects) Gemini 3.0 Pro (Ovals)

Locate every instance that belongs to the following categories: oven, refrigerator.

[Figure 965]

[Figure 966]

[Figure 967]

[Figure 968]

[Figure 969]

Original Nano Banana Gemini 3.0 Pro SketchVLM (Ovals) Gemini 3.0 Pro SketchVLM (Rects) Gemini 3.0 Pro (Ovals)

Locate every instance that belongs to the following categories: car, truck.

[Figure 970]

[Figure 971]

[Figure 972]

[Figure 973]

[Figure 974]

Original Nano Banana Gemini 3.0 Pro SketchVLM (Ovals) Gemini 3.0 Pro SketchVLM (Rects) Gemini 3.0 Pro (Ovals)

Locate every instance that belongs to the following categories: cup, person, wineglass.

[Figure 975]

[Figure 976]

[Figure 977]

[Figure 978]

[Figure 979]

Original Nano Banana Gemini 3.0 Pro SketchVLM (Ovals) Gemini 3.0 Pro SketchVLM (Rects) Gemini 3.0 Pro (Ovals)

Locate every instance that belongs to the following categories: motorbike.

[Figure 980]

[Figure 981]

[Figure 982]

[Figure 983]

[Figure 984]

Original Nano Banana Gemini 3.0 Pro SketchVLM (Ovals) Gemini 3.0 Pro SketchVLM (Rects) Gemini 3.0 Pro (Ovals)

Locate every instance that belongs to the following categories: cellphone, person.

[Figure 985]

[Figure 986]

[Figure 987]

[Figure 988]

[Figure 989]

Original Nano Banana Gemini 3.0 Pro SketchVLM (Ovals) Gemini 3.0 Pro SketchVLM (Rects) Gemini 3.0 Pro (Ovals)

Locate every instance that belongs to the following categories: firehydrant.

[Figure 990]

[Figure 991]

[Figure 992]

[Figure 993]

[Figure 994]

###### Fig. D10: Qualitative comparison on drawing shape tasks between , sketchVLM , and .

[Figure 995]

###### D.6 Part Labeling

Original Nano Banana Gemini 3.0 Pro SketchVLM Gemini 3.0 Pro

The object in the image is a bench. Label ONLY the following parts of the bench: arm, back, leg, seat, stretcher, table_top.

[Figure 996]

[Figure 997]

[Figure 998]

[Figure 999]

Original Nano Banana Gemini 3.0 Pro SketchVLM Gemini 3.0 Pro

The object in the image is a spoon. Label ONLY the following parts of the spoon: bowl, handle, neck, tip.

[Figure 1000]

[Figure 1001]

[Figure 1002]

[Figure 1003]

Original Nano Banana Gemini 3.0 Pro SketchVLM Gemini 3.0 Pro

The object in the image is a bottle. Label ONLY the following parts of the bottle: body, cap, neck, ring, shoulder.

[Figure 1004]

[Figure 1005]

[Figure 1006]

[Figure 1007]

Original Nano Banana Gemini 3.0 Pro SketchVLM Gemini 3.0 Pro

The object in the image is a bus. Label ONLY the following parts of the bus: front license plate, front side, headlight, right mirror, right side, wheel, window.

[Figure 1008]

[Figure 1009]

[Figure 1010]

[Figure 1011]

Original Nano Banana Gemini 3.0 Pro SketchVLM Gemini 3.0 Pro

The object in the image is a bird. Label ONLY the following parts of the bird: beak, head, left eye, left foot, left leg, left wing, neck, right foot, right leg, right wing, tail, torso.

[Figure 1012]

[Figure 1013]

[Figure 1014]

[Figure 1015]

Original Nano Banana Gemini 3.0 Pro SketchVLM Gemini 3.0 Pro

The object in the image is a person. Label ONLY the following parts of the person: hair, head, left eye, left foot, left hand, left lower arm, left lower leg, left upper arm, left upper leg, mouth, nose, right eye, right hand, right lower arm, right upper arm, right upper leg, torso.

[Figure 1016]

[Figure 1017]

[Figure 1018]

[Figure 1019]

###### Fig. D11: Qualitative comparison on part labeling tasks between , sketchVLM , and .

[Figure 1020]

###### D.7 Connect-the-Dots: Grid versus No Grid

Source Sketch without Grid Sketch with Grid

[Figure 1021]

[Figure 1022]

[Figure 1023]

[Figure 1024]

[Figure 1025]

- Fig. D12: Appending a reference coordinate grid to the edge of the input image allows to be more precise, but does not help .

[Figure 1026]

###### D.8 Connect-the-Dots: Bézier Curves versus Lines

[Figure 1027]

[Figure 1028]

Straight Lines: 12 strokes Bézier Curves: 2 strokes

[Figure 1029]

[Figure 1030]

Straight Lines: 9 strokes Bézier Curves: 1 stroke

[Figure 1031]

[Figure 1032]

Straight Lines: 17 strokes Bézier Curves: 9 strokes

[Figure 1033]

[Figure 1034]

Straight Lines: 9 strokes Bézier Curves: 3 strokes

[Figure 1035]

[Figure 1036]

Straight Lines: 19 strokes Bézier Curves: 1 stroke

- Fig. D13: Using Bézier curves instead of straight lines allows the model to connect the dots in fewer strokes, and it can draw less jagged shapes, leading to a more aesthetically pleasing result.

###### D.9 Gemini-3-Pro-Preview Coordinate Systems

[Figure 1037]

0 ¡ 1000 coordinate system 0 ¡ 2000 coordinate system

- Fig. D14: Comparing Gemini-3-Pro-Preview annotations under two coordinate systems. The 0–2000 system (vs. the 0–1000 system the model is typically used to) often preserves the overall shape but produces annotations that are compressed or shifted. In most cases the model adapts to the new coordinate system, but these failure cases lead to a noticeable decrease in performance. We hypothesize this change is what causes Gemini-3-Pro-Preview to also perform worse with the grid (which has a different coordinate system).

###### D.10 Multi-turn Ablation

Multiturn: Prompts + Image + Text Annotations sent back each turn

Multiturn: Prompts + Image sent back each turn

[Figure 1038]

[Figure 1039]

[Figure 1040]

[Figure 1041]

- Fig. D15: When testing Gemini-3-Pro-Preview in the multi-turn setting, not sending back the text representation of prior annotations degrades drawing quality: the model often attempts to redraw earlier strokes and fails to properly connect new strokes to the existing trajectory.

##### E VLM-Judge Details

We use a VLM judge (Gemini-3-Flash-Preview [12]) to evaluate annotation quality and annotation–text alignment. To verify that this automated metric is meaningful, we measure how well VLM judge ratings agree with human annotators.

Experiment Three independent human annotators each rate 50 annotations across the Ball Drop, VPCT, and Maze Navigation tasks from all 5 models ( , , , , and ), totaling 2,250 labeled images. Both human and VLM judges use the same 1–5 grading rubric (Sec. E.2). We report agreement using quadratically weighted Cohen’s κ, which penalizes larger disagreements more heavily on ordinal scales [21], and Pearson correlation, following recent VLMjudge benchmarks [24].

[Figure 1042]

[Figure 1043]

[Figure 1044]

[Figure 1045]

Results Human–human agreement is high, with κ = 0.84 ± 0.04 and Pearson r = 0.85 ± 0.04 (Tab. E1), indicating strong consistency between annotators. Detailed per-dataset agreement statistics are reported in Tab. E3.

Human–VLM agreement is moderate, with κ = 0.51 ± 0.02 and Pearson r = 0.52 ± 0.01, with task-level results shown in Tab. E2. The VLM judge’s ratings are positively correlated with human judgments across all tasks and models, making it a useful proxy for large-scale evaluation. However, the gap between human–human and human–VLM agreement indicates that the VLM judge can miss subtle errors, particularly logical violations such as trajectories clipping through walls (Sec. E.1).

We therefore use the VLM judge for cost-effective large-scale comparison across models, while acknowledging that human annotation remains more reliable for fine-grained evaluation.

- Table E1: Agreement is measured by quadratic κ and Pearson correlation (mean ± std). Human annotators are highly consistent, while the VLM judge shows moderate alignment with human judgment.

Comparison Kappa (Quadratic) Pearson

Human–Human 0.84 ± 0.04 0.85 ± 0.04 Human–VLM Judge 0.51 ± 0.02 0.52 ± 0.01

###### Table E2: Human–VLM agreement measured by quadratically weighted Cohen’s κ and Pearson correlation. The VLM judge achieves consistent moderate agreement with all annotators, with higher reliability on valid trajectories and lower agreement when annotations violate maze constraints.

Annotator 1 Annotator 2 Annotator 3 Dataset κquad Pearson κquad Pearson κquad Pearson

| | | |
|---|---|---|
|ball-path 0.53 0.53 maze-invalid 0.40 0.41 maze-valid 0.53 0.58 vpct 0.42 0.49|0.51 0.51 0.44 0.46 0.60 0.64 0.41 0.44|0.51 0.52 0.42 0.44 0.63 0.65 0.46 0.50|

###### Table E3: Human–human agreement across datasets. Annotators show consistently high agreement (κ ≈ 0.85–0.92), indicating reliable human evaluation, with slightly lower agreement on maze-invalid cases.

Annotator 1 vs Annotator 2 Annotator 1 vs Annotator 3 Annotator 2 vs Annotator 3 Dataset κquad Pearson κquad Pearson κquad Pearson

| | | |
|---|---|---|
|ball-path 0.90 0.90 maze-invalid 0.80 0.82 maze-valid 0.91 0.92 vpct 0.89 0.90|0.90 0.90 0.50 0.52<br>0.91 0.92 0.88 0.89<br>|0.85 0.85 0.53 0.53 0.92 0.93<br>0.86 0.87<br>|

###### Table E4: Human evaluation score (1–5, higher is better). We report mean ± standard deviation for each task and the overall mean across all tasks. Best mean in each column is bolded. SketchVLMs have higher mean annotation quality than other annotation models.

Model Human Score (1–5)

| |VPCT Ball Drop Maze Navigation (Invalid) Maze Navigation (Valid) Mean<br><br>|
|---|---|
|[Figure 1046]<br><br>sketchVLM sketchVLM|3.18 ± 1.27 3.24 ± 1.08 4.13 ± 1.24 4.45 ± 1.22 3.70 ± 1.32<br>4.56 ± 0.85 3.79 ± 1.30 3.92 ± 1.27 4.36 ± 1.07 4.14 ± 1.18<br><br><br>|
|[Figure 1047]<br><br>++<br><br>[Figure 1048]<br><br>[Figure 1049]|2.44 ± 1.30 2.94 ± 1.56 2.77 ± 1.41 4.29 ± 1.36 3.08 ± 1.57 1.26 ± 0.88 1.01 ± 0.12 3.02 ± 1.34 1.96 ± 1.05 1.74 ± 1.20 1.44 ± 1.01 1.10 ± 0.38 1.17 ± 0.45 1.25 ± 0.79 1.24 ± 0.72|

###### E.1 Qualitative Examples

###### Source Image Gemini-3-Pro-Preview GPT-5 (low) ThinkMorph NanoBanana Pro ViLaSR

[Figure 1050]

[Figure 1051]

[Figure 1052]

[Figure 1053]

[Figure 1054]

[Figure 1055]

Sketch: valid Judge: valid ✓

Sketch: valid Judge: valid ✓

Sketch: valid Judge: invalid ✗

Sketch: valid Judge: valid ✓

Sketch: valid Judge: invalid ✗

[Figure 1056]

[Figure 1057]

[Figure 1058]

[Figure 1059]

[Figure 1060]

[Figure 1061]

Sketch: valid Judge: valid ✓

Sketch: valid Judge: valid ✓

Sketch: valid Judge: invalid ✗

Sketch: valid Judge: valid ✓

Sketch: valid Judge: invalid ✗

[Figure 1062]

[Figure 1063]

[Figure 1064]

[Figure 1065]

[Figure 1066]

[Figure 1067]

Sketch: valid Judge: valid ✓

Sketch: valid Judge: valid ✓

Sketch: valid Judge: invalid ✗

Sketch: valid Judge: valid ✓

Sketch: valid Judge: invalid ✗

[Figure 1068]

[Figure 1069]

[Figure 1070]

[Figure 1071]

[Figure 1072]

[Figure 1073]

Sketch: valid Judge: invalid ✗

Sketch: valid Judge: valid ✓

Sketch: valid Judge: valid ✓

Sketch: valid Judge: valid ✓

Sketch: valid Judge: invalid ✗

[Figure 1074]

[Figure 1075]

[Figure 1076]

[Figure 1077]

[Figure 1078]

[Figure 1079]

Sketch: valid Judge: valid ✓

Sketch: valid Judge: valid ✓

Sketch: valid Judge: invalid ✗

Sketch: valid Judge: valid ✓

Sketch: valid Judge: invalid ✗

[Figure 1080]

[Figure 1081]

[Figure 1082]

[Figure 1083]

[Figure 1084]

[Figure 1085]

Sketch: valid Judge: valid ✓

Sketch: valid Judge: valid ✓

Sketch: valid Judge: invalid ✗

Sketch: valid Judge: invalid ✗

Sketch: valid Judge: valid ✓

[Figure 1086]

[Figure 1087]

[Figure 1088]

[Figure 1089]

[Figure 1090]

[Figure 1091]

Sketch: valid Judge: valid ✓

Sketch: valid Judge: valid ✓

Sketch: valid Judge: invalid ✗

Sketch: valid Judge: valid ✓

Sketch: valid Judge: invalid ✗

###### Fig. E1: Comparison between the original sketching model’s answer and the judge’s answer for valid paths.

###### Source Image Gemini-3-Pro-Preview GPT-5 (low) ThinkMorph NanoBanana Pro ViLaSR

[Figure 1092]

[Figure 1093]

[Figure 1094]

[Figure 1095]

[Figure 1096]

[Figure 1097]

Sketch: invalid Judge: invalid ✓

Sketch: invalid Judge: invalid ✓

Sketch: invalid Judge: invalid ✓

Sketch: invalid Judge: invalid ✓

Sketch: valid Judge: invalid ✗

[Figure 1098]

[Figure 1099]

[Figure 1100]

[Figure 1101]

[Figure 1102]

[Figure 1103]

Sketch: invalid Judge: invalid ✓

Sketch: invalid Judge: invalid ✓

Sketch: valid Judge: invalid ✗

Sketch: invalid Judge: invalid ✓

Sketch: valid Judge: invalid ✗

[Figure 1104]

[Figure 1105]

[Figure 1106]

[Figure 1107]

[Figure 1108]

[Figure 1109]

Sketch: invalid Judge: invalid ✓

Sketch: invalid Judge: invalid ✓

Sketch: invalid Judge: invalid ✓

Sketch: invalid Judge: invalid ✓

Sketch: valid Judge: invalid ✗

[Figure 1110]

[Figure 1111]

[Figure 1112]

[Figure 1113]

[Figure 1114]

[Figure 1115]

Sketch: invalid Judge: invalid ✓

Sketch: invalid Judge: invalid ✓

Sketch: valid Judge: invalid ✗

Sketch: invalid Judge: invalid ✓

Sketch: valid Judge: invalid ✗

[Figure 1116]

[Figure 1117]

[Figure 1118]

[Figure 1119]

[Figure 1120]

[Figure 1121]

Sketch: invalid Judge: invalid ✓

Sketch: invalid Judge: invalid ✓

Sketch: invalid Judge: invalid ✓

Sketch: invalid Judge: invalid ✓

Sketch: invalid Judge: invalid ✓

[Figure 1122]

[Figure 1123]

[Figure 1124]

[Figure 1125]

[Figure 1126]

[Figure 1127]

Sketch: invalid Judge: invalid ✓

Sketch: invalid Judge: invalid ✓

Sketch: invalid Judge: invalid ✓

Sketch: invalid Judge: invalid ✓

Sketch: valid Judge: invalid ✗

[Figure 1128]

[Figure 1129]

[Figure 1130]

[Figure 1131]

[Figure 1132]

[Figure 1133]

Sketch: invalid Judge: invalid ✓

Sketch: invalid Judge: invalid ✓

Sketch: invalid

Sketch: invalid Judge: invalid ✓

Sketch: valid Judge: invalid ✗

Judge: valid ✗

###### Fig. E2: Comparison between the original sketching model’s answer and the judge’s answer for invalid paths.

###### Source Image Gemini-3-Pro-Preview GPT-5 (low) ThinkMorph NanoBanana Pro ViLaSR

[Figure 1134]

[Figure 1135]

[Figure 1136]

[Figure 1137]

[Figure 1138]

[Figure 1139]

Score: 1 Score: 5 Score: 2 Score: 1 Score: 1

[Figure 1140]

[Figure 1141]

- Score: 4 Score: 4 Score: 1 Score: 5 Score: 2

[Figure 1142]

[Figure 1143]

[Figure 1144]

[Figure 1145]

[Figure 1146]

[Figure 1147]

- Score: 1 Score: 4 Score: 2 Score: 1 Score: 2

[Figure 1148]

[Figure 1149]

[Figure 1150]

[Figure 1151]

[Figure 1152]

[Figure 1153]

Score: 5 Score: 1 Score: 1 Score: 1 Score: 1

[Figure 1154]

[Figure 1155]

[Figure 1156]

[Figure 1157]

[Figure 1158]

[Figure 1159]

- Score: 1 Score: 1 Score: 2 Score: 1 Score: 1

[Figure 1160]

[Figure 1161]

[Figure 1162]

[Figure 1163]

[Figure 1164]

[Figure 1165]

- Score: 2 Score: 1 Score: 1 Score: 1 Score: 1

[Figure 1166]

[Figure 1167]

[Figure 1168]

[Figure 1169]

[Figure 1170]

[Figure 1171]

- Score: 2 Score: 1 Score: 2 Score: 5 Score: 1

[Figure 1172]

[Figure 1173]

[Figure 1174]

[Figure 1175]

[Figure 1176]

[Figure 1177]

[Figure 1178]

[Figure 1179]

[Figure 1180]

[Figure 1181]

Score: 4 Score: 1 Score: 1 Score: 1 Score: 1

###### Fig. E3: VLM-Judge quality scores for VPCT.

###### Source Image Gemini-3-Pro-Preview GPT-5 (low) ThinkMorph NanoBanana Pro ViLaSR

[Figure 1182]

[Figure 1183]

[Figure 1184]

[Figure 1185]

[Figure 1186]

[Figure 1187]

Score: 2 Score: 2 Score: 1 Score: 5 Score: 1

Path: right, down right, down, left, left

[Figure 1188]

[Figure 1189]

[Figure 1190]

[Figure 1191]

[Figure 1192]

[Figure 1193]

Score: 1 Score: 1 Score: 1 Score: 2 Score: 1

Path: up, up, left down, down, left

[Figure 1194]

[Figure 1195]

[Figure 1196]

[Figure 1197]

[Figure 1198]

[Figure 1199]

Score: 5 Score: 5 Score: 1 Score: 5 Score: 2

Path: right, right down, down

[Figure 1200]

[Figure 1201]

[Figure 1202]

[Figure 1203]

[Figure 1204]

[Figure 1205]

Path: left, down, left Score: 2 Score: 1 Score: 5 Score: 5 Score: 1

[Figure 1206]

[Figure 1207]

[Figure 1208]

[Figure 1209]

[Figure 1210]

[Figure 1211]

Score: 5 Score: 4 Score: 5 Score: 5 Score: 1

Path: left, down down

[Figure 1212]

[Figure 1213]

[Figure 1214]

[Figure 1215]

[Figure 1216]

[Figure 1217]

Score: 5 Score: 5 Score: 1 Score: 1 Score: 1

Path: right, right, up up

[Figure 1218]

[Figure 1219]

[Figure 1220]

[Figure 1221]

[Figure 1222]

[Figure 1223]

Score: 5 Score: 5 Score: 1 Score: 1 Score: 1

Path: right, down down

###### Fig. E4: VLM-Judge quality scores for valid paths in the Maze Navigation task.

###### Source Image Gemini-3-Pro-Preview GPT-5 (low) ThinkMorph NanoBanana Pro ViLaSR

[Figure 1224]

[Figure 1225]

[Figure 1226]

[Figure 1227]

[Figure 1228]

[Figure 1229]

Score: 5 Score: 2 Score: 1 Score: 4 Score: 1

Path: left, down, right down, left, left

[Figure 1230]

[Figure 1231]

[Figure 1232]

[Figure 1233]

[Figure 1234]

[Figure 1235]

Score: 5 Score: 1 Score: 1 Score: 5 Score: 5

Path: right, up, left down, down, left

[Figure 1236]

[Figure 1237]

[Figure 1238]

[Figure 1239]

[Figure 1240]

[Figure 1241]

Score: 5 Score: 2 Score: 1 Score: 1 Score: 5

Path: left, right, down down

[Figure 1242]

[Figure 1243]

[Figure 1244]

[Figure 1245]

[Figure 1246]

[Figure 1247]

Path: left, up, left Score: 1 Score: 2 Score: 1 Score: 1 Score: 1

[Figure 1248]

[Figure 1249]

[Figure 1250]

[Figure 1251]

[Figure 1252]

[Figure 1253]

Score: 5 Score: 5 Score: 1 Score: 5 Score: 1

Path: down, down down

[Figure 1254]

[Figure 1255]

[Figure 1256]

[Figure 1257]

[Figure 1258]

[Figure 1259]

Score: 5 Score: 1 Score: 1 Score: 5 Score: 5

Path: right, left, up up

[Figure 1260]

[Figure 1261]

[Figure 1262]

[Figure 1263]

[Figure 1264]

[Figure 1265]

Score: 5 Score: 5 Score: 2 Score: 5 Score: 1

Path: left, down down

###### Fig. E5: VLM-Judge quality scores for invalid paths in the Maze Navigation task.

###### E.2 Rubric Prompts

###### Rubric for VPCT and Ball Drop Quality

# Overall Instructions You will be shown two images: the original image and an AI-annotated version. The annotation is

supposed to show how the path the ball will take when it is dropped and where it will eventually end up. Your job is to grade the quality of the sketch using a rubric. You should

grade only based on the following attributes that are given.

A minor issue is a violation that has little effect on the final answer of what bucket the ball will fall into. For example , if the ball path slightly clips through the walls , but the trajectory of the path still makes sense , then this is only a minor issue and not a major issue. A major issue is a violation that has a significant effect on the final answer of what bucket the ball will fall into. For example , if the ball path clips through the walls and the trajectory of the path is not logical , then this is a major issue. If there is no sketch provided , then the score should be a 1. The score should be based on the scale of 1 -

5 # Quality Checks Below are more details: ## Things that are bad

- 1) The drawn path clips through any of the black walls.
- 2) The drawn path has very unrealistic physics. For example , the ball path momentarily disobeys gravity by moving upwards or moving in a direction that is not logical for gravity.
- 3) The drawn path substantially alters the original image (like adding or removing walls)
- 4) The drawn path contains multiple different paths instead of a single path. ## Things that are good

- 1) The drawn path is a single path that clearly shows where the final resting position of the ball will be.
- 2) The drawn path does not cut through any of the black walls. ## Things that are okay

- 1) If the path slightly clips through the walls , but the trajectory of the path still makes sense , then this is only a minor issue and not a major issue. We want to heavily penalize the ball going through walls and following a totally illogical path.
- 2) It’s okay if the drawn path does not start exactly at the origin of the ball. As long as it is close , then this is not a minor or major issue.

# Scoring breakdown

- 1) The sketch has several critical flaws
- 2) The sketch has a critical flaw
- 3) The sketch contains some errors , overall direction of the path is valid and makes sense.
- 4) The sketch contains one minor logical error.
- 5) The sketch contains zero errors.

# Output Format You should follow this output format EXACTLY with no other output:

{reasoning for logical consistency score} Quality Score: {integer from 1 - 5}

# Example Output

- <example_1 > The drawing contains multiple errors. The ball path barely clips through one of the platforms , but

it is a minor issue as the path of the ball still makes sense. Additionally , the ball path defies gravity and begins to momentarily float straight upwards for no logical reason. The minor error combined with the critical error results in a logical score of 2/5.

Quality Score: 2

- </example_1 >

<example_2 > Overall , the path the ball takes is logical. It properly reflects off the first wall and then rolls

to the right. Then , it falls down to the left and lands in bucket 2. The ball path slightly clips through one of the black edges , but this is a minor issue because the big picture trajectory of the ball is still logical. Given the overall path makes sense and there was just one minor issue , the quality score is 4/5.

Quality Score: 4

- </example_2 >

###### Fig. E6: Rubric for VPCT and Ball Drop Quality

###### Rubric for Path Navigation Quality

# Overall Instructions You will be shown two images: the original image and an AI-annotated version. Your job is to grade the quality of the sketch using a rubric. You should grade only based on the following attributes that are given.

# Sketch Instructions The original sketch was drawn according to the following prompt: "You are given an image of a maze where the green square marks the START cell and the red square

marks the END cell of the maze. The walls of the maze are solid black lines. Dashed gray lines mark cell boundaries that can be crossed. You are given a proposed sequence of moves to reach the end of the maze starting from the green square and ending at the red square. Each move will move exactly one cell length in that direction. For example , "right" means move one cell in the maze to the right. A valid path must NOT cross any solid black walls and must end up in the red square cell. A valid path can also move through any of the dashed

gray cell lines. The grid on the outside of the image is only there to help provide a

reference for you. Moving one right means go one big cell in the maze right." # Quality Checks The main things that you should be looking for are:

- 1) Clipping through walls when it is not required to.
- 2) The drawn path does not go to the CENTER of each cell that it goes through.
- 3) The drawn path contradicts the given text path. Below are more details: ## Things that are bad

- 1) The drawn path clips through any of the black walls when it is not required to. For example , even if the directions of the drawn path are correct , if the path touches or goes through a wall , then it is a bad sketch. That means that if the path goes through a wall even when it is not absolutely required to, then it is a bad sketch.
- 2) Each move in the drawn path should go to the **center** of the next cell in the path. If the drawn path is a curved path , then this does not apply. This is important! Look at each step in the path and make sure that the drawn path goes to the center of the next cell.
- 3) The sketch contains additional moves that are not in the path
- 4) The drawn sketch contradicts the given text path.
- 5) Even if the directions of the drawn path are correct , the end of the path does not end up touching the red square.
- 6) The drawn path does not start by touching the green square ## Things that are not an issue

- 1) If the proposed path is not valid , the drawn sketch shows exactly what the path should look like (even if it has to clip through walls or double back on itself).
- 2) If the proposed path is not valid , and the drawing ends as soon as there is an invalid move that is taken (such as requiring to go through a wall), then that is not an issue. It’s okay for the drawing to not show all the steps of the path here since the sketch is emphasizing that the path is invalid.

## Score breakdown

- 1) The sketch makes absolutely no logical sense.
- 2) The sketch has some critical flaws that breaks the logic of the sketch.
- 3) The sketch contains multiple logical errors.
- 4) The sketch contains a minor error.
- 5) The sketch contains zero errors.

# Output Format You should follow this output format EXACTLY with no other output:

{reasoning for quality score} Quality Score: {integer from 1 - 5}

# Example Output <example_1 > The drawing contains multiple errors. The drawn path goes up, up, left instead of up, up, right.

This contradicts the given text path. Additionally , the end of the drawn path slightly clips through the solid black wall. Additionally , the draw path does NOT go to the center of each cell that it goes through. Instead , it only moves about 60 percent of the way across each

cell and doesn’t therefore doesn’t follow the grid logic of moving from one center of each cell to the center of the next cell. The minor error combined with the multiple critical errors results in a score of 1/5.

Quality Score: 1 </example_1 >

The original proposed path that the model should have followed is: {INSERT_ORIGINAL_PROMPT_PATH}

###### Fig. E7: Rubric for Maze Navigation Quality

##### F Model Settings and Prompts F.1 API Settings

Table F1: Model inference settings across providers.

Model Details GPT-5 Model: gpt-5

Temperature: Default Reasoning Effort: Low, Medium, High Using OpenAI Responses API

Gemini 2.5 Flash Model: gemini-2.5-flash Temperature: Default Using Google AI Studio API

- Gemini 2.5 Pro Model: gemini-2.5-pro Temperature: Default Using Google AI Studio API

- Gemini 3 Pro Model: google/gemini-3-pro-preview Temperature: Default Reasoning Effort: Default Using OpenRouter API (Google AI Studio provider)

Gemini 3 Flash Model: google/gemini-3-flash-preview Temperature: Default Reasoning Effort: Default Using OpenRouter API (Google AI Studio provider)

Nano Banana Pro Model: google/gemini-3-pro-image-preview Temperature: Default Output Mode: Image Only Using OpenRouter API (Google AI Studio provider)

Qwen 2.5 VL 7B Model: qwen/qwen-2.5-vl-7b-instruct Temperature: Default Using OpenRouter API (Hyperbolic provider)

ThinkMorph Model: ThinkMorph/ThinkMorph-7B Temperature: Default Ran locally using inference code provided at https://github.com/ThinkMorph/ThinkMorph.

ViLaSR Model: inclusionAI/ViLaSR Temperature: Default Ran locally using inference code provided at https://github.com/AntResearchNLP/ViLaSR/.

###### F.2 SketchVLM System Prompt

###### System prompt (1/3)

# prompts.py SYSTEM_PROMPT_BASE = """You are an expert artist specializing in drawing sketches that are visually appealing, expressive, and

professional. You will be provided with a blank grid. Your task is to specify where to place strokes on the grid to create a visually appealing sketch to complete the request.

The grid uses numbers (0 to {res_x}) along the bottom (x axis) and numbers (0 to {res_y}) along the left edge (y axis) to reference specific locations within the grid. The {origin_corner} is the origin. Each cell is uniquely identified by a combination of the corresponding x axis numbers and y axis number (\eg, the bottom-left cell is ’{example_bottom_left}’,

the cell to its right is ’{example_right_of_bottom_left}’). You can draw on this grid by specifying where to draw strokes. You can draw multiple strokes to depict the whole object, where

different strokes compose different parts of the object. To draw a stroke on the grid, you need to specify the following: Starting Point: Specify the starting point by giving the grid location (\eg, ’x0y0’ for column 0, row 0). Ending Point: Specify the ending point in the same way (\eg, ’x{res_x}y{res_y}’ for column {res_x}, row {res_y}). Intermediate Points: Specify at least two intermediate points that the stroke should pass through. List these in the order the

stroke should follow, using the same grid location format (\eg, ’x6y5’, ’x13y10’ for points at column 6 row 5 and column 13 row 10). Parameter Values (t): For each point (including the start and end points), specify a t value between 0 and 1 that defines the

position along the stroke’s path. t=0 for the starting point. t=1 for the ending point. Intermediate points should have t values between 0 and 1 (\eg, "0.3 for x6y5, 0.7 for x13y10"). Examples: To draw a smooth curve that starts at x8y6, passes through x6y7 and x6y10, ending at x8y11: Points = [’x8y6’, ’x6y7’, ’x6y10’, ’x8y11’] t_values = [0.00,0.30,0.80,1.00] To close this curve into an ellipse shape, you can add another curve: Points = [’x8y11’, ’x11y10’, ’x11y7’, ’x8y6’] t_values = [0.00,0.30,0.70,1.00] To draw a large circle that starts at x25y44 and ends at x25y44, passing through the cells x32y41, x35y35, x31y29, x25y27,

x19y29, x15y35, x18y41:

- Points = [’x25y44’, ’x32y41’, ’x35y35’, ’x31y29’, ’x25y27’, ’x19y29’, ’x15y35’, ’x18y41’, ’x25y44’]

- t_values = [0.00, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.00] To draw non-smooth shapes (with corners) like triangles or rectangles, you need to specify the corner points twice with adjacent

corresponding t values. For example, to draw an upside-down "V" shape that starts at x13y27, ends at x24y27, with a pick (corner) at x18y37: Points = [’x13y27’, ’x18y37’,’x18y37’, ’x24y27’] t_values = [0.00,0.55,0.5,1.00] To draw a triangle with corners at x10y29, x15y33, and x9y35, start with drawing a "V" shape that starts at x10y29, ends at

x9y35, with a pick (corner) at x15y33: Points = [’x10y29’, ’x15y33’, ’x15y33’, ’x9y35’]

- t_values = [0.00,0.55,0.5,1.00] and then close it with a straight line from x13y27 to x24y27 to form a triangle: Points = [’x13y27’, ’x24y27’]
- t_values = [0.00,1.00] Note that for a triangle, the start and end points should be different from each other. To draw a rectangle with four corners at x13y27, x24y27, x24y11, x13y11: Points = [’x13y27’, ’x24y27’, ’x24y27’, ’x24y11’, ’x24y11’, ’x13y11’, ’x13y11’, ’x13y27’] t_values = [0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00] To draw a small square with four corners at x26y25, x29y25, x29y21, x26y21:

Points = [’x26y25’, ’x29y25’, ’x29y25’, ’x29y21’, ’x29y21’, ’x26y21’, ’x26y21’, ’x26y25’] t_values = [0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00] To draw a single dot at x15y31 use: Points = [’x15y31’] t_values = [0.00] To draw a straight linear line that starts at x18y31 and ends at x35y14 use: Points = [’x18y31’, ’x35y14’]

- t_values = [0.00, 1.00] If you want to draw a big and long stroke, split it into multiple small curves that connect to each other. """

###### Fig. F1: System Prompt (1/3). Used for all SketchVLM models in single-turn and multi-turn.

###### System prompt (2/3)

# Sketch Methods Below are the different sketching methods you can use for your task. ## FREEHAND SKETCH

- - Emit one or more stroke blocks with points on the grid, no <text>.
- - Use multiple strokes to compose shapes; curves/lines are both fine.
- - the <id> tag should describe the part being drawn.

- <s1> <points>’x12y20’,’x13y20’,’x14y21’,’x15y22’</points> <t_values>0.00,0.33,0.66,1.00</t_values>

- <id>part_1</id>

- </s1>

<s2> <points>’x20y18’,’x20y14’,’x24y14’,’x24y18’,’x20y18’</points> <t_values>0.00,0.25,0.50,0.75,1.00</t_values> <id>part_2</id>

- </s2>

## STRAIGHT LINE <sN>

<points>’x10y19’,’x40y19’</points> <t_values>0.00,1.00</t_values> <id>line_1</id>

</sN> ## Arrow (draw the shaft, and the arrowhead as two separate parts)

- <s1> <points>’x12y32’,’x6y32’</points> <t_values>0.00,1.00</t_values> <id>arrow_shaft</id>

- </s1>

<s2> <points>’x7y33’,’x6y32’</points> <t_values>0.00,1.00</t_values> <id>arrowhead_top</id>

- </s2>

<s3> <points>’x7y31’,’x6y32’</points> <t_values>0.00,1.00</t_values> <id>arrowhead_bottom</id>

- </s3>

## BOX / RECTANGLE (list the 4 corners in order) <sN>

<points>’x12y12’,’x20y12’,’x20y18’,’x12y18’,’x12y12’</points> <t_values>0.00,0.25,0.50,0.75,1.00</t_values> <id>box_1</id>

</sN> ## COUNTING (place numerals near each instance; one stroke per number; change text size based on object size and image

resolution, so can be text size="1.0" or "2.0" up to "32.0" etc)

<sN> <points>’x08y22’</points> <t_values>0.00</t_values> <text size="4.0" color="black">’1’</text> <id>count_1</id>

</sN> ## LABELING (anchor a text label to the cell closest to center of the object/part; change text size based on object/part size

and image resolution, so can be text size="1.0" or "2.0" up to "32.0" etc)

<sN> <points>’x26y17’</points> <t_values>0.00</t_values> <text size="3.2" color="black">’handlebar’</text> <id>label_handlebar</id>

</sN> # Rules

- - Output only <answer>...</answer> with a single <strokes>...</strokes> section.
- - For counting/labeling tasks, prefer <text> with short values (’1’,’2’,... or ’wheel’,’seat’,...).
- - Use <points> with exactly one anchor cell for each text label/number (one per item/part).
- - Do not mix patterns: if the user asks to label, do not draw boxes; if the user asks to count, do not label names.
- - Keep each stroke in its own <sN>...</sN> block; increment N in order without gaps.
- - If the question requires an answer (\eg, "How many?"), include it at the end of your response, after the </strokes> tag, in a new <final_answer> tag.

"""

###### Fig. F2: System Prompt (2/3), continued from Fig. F1.

###### System prompt (3/3)

def build_system_prompt(res_x: int, res_y: int, origin: str = "bottom_left") -> str:

if origin == "top_left": origin_corner = "top left" example_bottom_left = f"x0y{res_y}" example_right_of_bottom_left = f"x1y{res_y}"

else: origin_corner = "bottom left" example_bottom_left = "x0y0" example_right_of_bottom_left = "x1y0"

return SYSTEM_PROMPT_BASE.format(

- res_x=res_x,
- res_y=res_y, origin_corner=origin_corner, example_bottom_left=example_bottom_left, example_right_of_bottom_left=example_right_of_bottom_left,

)

COUNTING_PROMPT = """ Task:

- COUNT all the {object} by placing numbered SVG text strokes on them (no curves). Output example could be: <answer> <concept>Numbering each {object}</concept> <strokes> <s1>

<text size="1.6" color="#ff0066">’1’</text> <points>’xAyB’</points> <t_values>0.00</t_values> <id>marker_{object}1</id>

</s1> <!-- s2, s3, ... one per object --> </strokes>

Rules:

- - Use ONLY text strokes (no curves).
- - Exactly one point per stroke (’xAyB’) at the object’s center-ish cell.
- - You MAY style numbers: <text size="1.8" color="#0057ff"> or <style><font_size>...</font_size><color>...</color></style>.
- - size is cells (multiplier) unless you suffix ’px’
- - choose bigger text size for larger objects, smaller for tiny objects. use bigger size for higher resolution images, smaller for lower resolution.
- - choose readable colors that will contrast well with the object that you are numbering and the background.
- - If 0 objects, still return the full wrapper with an empty <strokes> block.

- - Do not write anything outside <answer>...</strokes>. """

GENERIC_LABEL_PROMPT = """ Task:

- - The object in the image is a {concept}.
- - Label ONLY the following parts of the {concept}: {labels_hint}.
- - Do not invent or add any new part names beyond this list.
- - Use SVG text strokes (no curves) to place each label.

Output EXACTLY this XML shape: <answer> <concept>Labeling: {concept}</concept> <strokes>

<!-- one <sN> per label --> <s1>

<text size="1.6" color="#ff0066">’head’</text> <!-- size: cells or ’px’ --> <points>’xAyB’</points> <t_values>0.00</t_values> <id>label_head</id>

</s1> </strokes> Rules:

- - Use ONLY text strokes (no curves).
- - Anchor each label at the center of the corresponding part (’xAyB’).
- - You MAY style the text labels: <text size="1.8" color="#0057ff"> or <style><font_size>...</font_size><color>...</color></style >
- - Use one <sN> per label name in the list above.
- - Choose bigger text size for larger objects, smaller for tiny objects. use bigger size for higher resolution images, smaller for lower resolution.
- - Choose readable colors that will contrast well with the object/part that you are labeling and the background.
- - Do not write anything outside <answer>...</strokes>. """

###### Fig. F3: System Prompt (3/3), continued from Fig. F2. Parameters are res_x, res_y, and origin. res_x controls how many x coordinates there are. res_y controls how many y coordinates there are. origin controls whether the origin is at the top left or bottom left.

One-stroke guard (multiturn)

Final-answer guard (multiturn)

|[Mode: stepwise] You are in stepwise mode. On this turn you<br><br>output EXACTLY ONE stroke block: <answer><br><br><strokes> <sN>...</sN><br><br></strokes> </answer> Do NOT output any other <sM> blocks, no<br><br><final_answer>, no explanations.<br><br>If the drawing is already complete and no further stroke is needed, output an empty <answer> with NO <strokes> block.<br><br>Stop immediately after </answer>.<br><br>|
|---|

|[Mode: stepwise] All strokes have already been provided. On<br><br>this turn output ONLY: <final_answer> ... </final_answer> Do not output the previous strokes again.<br><br>Stop immediately after </final_answer>.<br><br>|
|---|

- Fig. F4: During multi-turn, when the model must draw a stroke, we append the "Onestroke guard" to clarify to the model that it should only draw one stroke with no final answer. After the model has decided that no more strokes are needed (it does this by outputting no strokes on a turn), we run one more turn where we prompt the model with "Final-answer guard."

###### F.3 SketchVLM Output Example

[Figure 1266]

###### Stroke annotations (XML)

- <s1> <points>’x500y100’,’x500y270’,’x500y270’,’x350y400’</points> <t_values>0.00,0.50,0.50,1.00</t_values> <id>path_1</id>

- </s1>

<s2> <points>’x350y400’,’x325y470’,’x300y540’</points> <t_values>0.00,0.50,1.00</t_values> <id>drop_1</id>

- </s2>

<s3> <points>’x300y540’,’x390y470’,’x500y670’</points> <t_values>0.00,0.50,1.00</t_values> <id>path_2</id>

- </s3>

<s4> <points>’x500y670’,’x570y690’,’x640y710’</points> <t_values>0.00,0.50,1.00</t_values> <id>path_bounce</id>

- </s4>

<s5> <points>’x640y710’,’x700y850’,’x730y950’</points> <t_values>0.00,0.60,1.00</t_values> <id>drop_2</id>

- </s5> <final_answer>3</final_answer>

- Fig. F5: Example sketch output and annotations for prompting SketchVLM with Gemini-3-Pro-Preview on VPCT. Each stroke is colored differently for viewing purposes.

##### G Other Baselines

###### G.1 Baselines

Image-Editing Models. To benchmark the image-editing model Nano Banana Pro (commonly referred to as Nano Banana Pro), we first prompt Nano Banana Pro to generate a sketch for the input image. We then provide the resulting edited image to Gemini-3-Pro-Preview to produce the final task answer. For Connectthe-Dots, we additionally report a manual evaluation of Nano Banana Pro, since its outputs do not always map cleanly to our structured sketch representation.

Fine-tuned Sketching Models. For fine-tuned sketching models such as ViLaSR and ThinkMorph, we use the same task prompts as for SketchVLM. Because these models are trained to always output a sketch, they do not have a meaningful “no-sketch” baseline mode; therefore, we omit baseline VQA accuracy for these models in the main results and report only their sketch-conditioned performance.

