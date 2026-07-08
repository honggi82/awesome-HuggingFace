# arXiv:2604.25806v1[cs.CL]28Apr2026

## MAIC-UI: Making Interactive Courseware with Generative UI

Shangqing Tu1, Yanjia Li1, Keyu Chen2, Sichen Zhang3, Jifan Yu1, Daniel Zhang-Li1, Lei Hou1, Juanzi Li1, Yu Zhang1, Huiqin Liu1

1Tsinghua University, 2Guangzhou University, 3Zhejiang University Beijing, Guangzhou, Hangzhou, China https://github.com/THU-MAIC/MAIC-UI tsq25@mails.tsinghua.edu.cn

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Interactive simulations make the lens formula intuitive by allowing students to manipulate object distance and visualize real-time changes in image position and ray convergence.

Interactive simulations make the lens formula intuitive by allowing students to manipulate object distance and visualize real-time changes in image position and ray convergence.

[Figure 9]

[Figure 10]

A C

##### B

[Figure 11]

[Figure 12]

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

Figure 1: MAIC-UI enables zero-code creation and rapid editing of interactive courseware. In this example, a physics teacher creates an interactive simulation about lens focal length from a textbook chapter. (A) The Page Navigation panel displays the uploaded document pages, allowing teachers to browse content and select specific sections for generation. (B) The system generates an interactive simulation with a left panel showing the step-by-step procedure and a right panel with interactive controls for students to manipulate objects. (C) The Click-to-Locate Editing interface enables teachers to click on any element to select it, then describe desired changes in natural language, and apply Unified Diff-based updates in under 10 seconds.

### ABSTRACT

interactive courseware from textbooks, PPTs, and PDFs. MAICUI employs: (1) structured knowledge analysis with multi-modal understanding to ensure pedagogical rigor; (2) a two-stage generateverify-optimize pipeline separating content alignment from visual refinement; and (3) Click-to-Locate editing with Unified Diff-based incremental generation achieving sub-10-second iteration cycles. A controlled lab study with 40 participants shows MAIC-UI reduces editing iterations (4.9 vs. 7.0) and significantly improves learnability and controllability compared to direct Text-to-HTML generation. A three-month classroom deployment with 53 high school students demonstrates that MAIC-UI fosters learning agency and reduces outcome disparities—the pilot class achieved 9.21-point gains in STEM subjects compared to -2.32 points in control classes. Our code is available at https://github.com/THU-MAIC/MAIC-UI.

Creating interactive STEM courseware traditionally requires HTML/CSS/JavaScript expertise, leaving barriers for educators. While generative AI can produce HTML codes, existing tools generate static presentations rather than interactive simulations, struggle with long documents, and lack pedagogical accuracy mechanisms. Furthermore, full regeneration for modifications requires 200–600 seconds, disrupting creative flow. We present MAIC-UI, a zero-code authoring system that enables educators to create and rapidly edit

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

### CCS CONCEPTS

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY © 2018 Association for Computing Machinery. ACM ISBN 978-1-4503-XXXX-X/18/06...$15.00 https://doi.org/XXXXXXX.XXXXXXX

• Computing methodologies → Discourse, dialogue and pragmatics; Natural language generation.

### KEYWORDS

AI for Education, End-User Programming, Generative UI

ACM Reference Format:

Shangqing Tu1, Yanjia Li1, Keyu Chen2, Sichen Zhang3, Jifan Yu1, Daniel Zhang-Li1, Lei Hou1, Juanzi Li1, Yu Zhang1, Huiqin Liu1. 2018. MAICUI: Making Interactive Courseware with Generative UI. In Proceedings of Make sure to enter the correct conference title from your rights confirmation emai (Conference acronym ’XX). ACM, New York, NY, USA, 18 pages. https: //doi.org/XXXXXXX.XXXXXXX

### 1 INTRODUCTION

Interactive courseware has emerged as a powerful tool for fostering student engagement and deepening conceptual understanding in STEM education [1, 7, 12, 21]. By enabling learners to manipulate parameters, observe real-time feedback, and explore procedural knowledge through hands-on experimentation, interactive simulations transform passive reception into active construction of knowledge [8, 16, 39]. However, creating such materials traditionally requires substantial programming expertise—teachers must write HTML, CSS, and JavaScript code to implement interactive components, debug cross-browser compatibility issues, and ensure visual polish [18, 40]. This technical barrier leaves many educators unable to realize their pedagogical visions, forcing them to rely on static presentations or pre-made simulations that may not align with their specific instructional goals.

Recent advances in generative AI have transformed content creation across domains, offering new possibilities for educational material development. Large language models can now generate code from natural language descriptions, potentially democratizing the creation of interactive web content [44, 54]. However, existing tools face significant limitations when applied to educational contexts: they often produce static text and images rather than interactive simulations [45, 50], struggle to process long-form teaching materials such as textbooks and lecture slides [6, 33], and lack mechanisms to ensure pedagogical accuracy—a critical requirement for educational content [53, 57]. Furthermore, the iteration cycles in current systems are painfully slow: when teachers request modifications, systems typically regenerate entire files, requiring 200–600 seconds per edit and disrupting creative flow [15, 22].

To understand these challenges from educators’ perspectives, we conducted a formative study with six participants who had teaching experience. Our findings revealed four key challenges: (F1) Knowledge accuracy concerns—teachers worried that AI-generated content might misrepresent concepts or hallucinate wrong information; (F2) Editing limitations—participants expressed frustration with ambiguous natural language interfaces that made precise modifications difficult; (F3) Passive learning experiences—existing tools produced presentation-style content rather than interactive simulations that support learning-by-doing; and (F4) Theory-practice gaps—procedural knowledge was often presented as static descriptions rather than visualizable, explorable processes.

Based on these insights, we developed MAIC-UI (Making Interactive Courseware with Generative UI [29]), a web-based authoring system that enables educators to create interactive courseware from textbooks, PPTs, and PDFs without programming expertise. MAICUI addresses the four challenges through four key design goals:

(DG1) Pedagogical scientificity and visual professionalismensuring generated content is both factually accurate and aesthetically polished; (DG2) Zero-code precise editing—enabling teachers to make fine-grained adjustments without writing code; (DG3) Active learning and personalized exploration—supporting interactive components that foster student agency; and (DG4) Procedural knowledge visualization—making abstract processes concrete and explorable.

MAIC-UI employs three core technical innovations to achieve these goals. First, a structured knowledge analysis framework with multi-modal understanding extracts pedagogical content from documents (up to 50 pages), ensuring accurate content understanding before generation. Second, a two-stage generate-verify-optimize pipeline balances pedagogical accuracy with visual quality: Stage

- 1 creates content-aligned interactive simulations, while Stage 2 applies visual polish through layout verification, theme application, and animation smoothing. Third, Click-to-Locate natural language editing with Unified Diff-based incremental generation enables teachers to select UI elements by clicking and describe changes in natural language, with the system applying only the necessary code patches in under 10 seconds—a 90% reduction compared to full regeneration.

We evaluated MAIC-UI through a controlled user study with 40 participants and a three-month classroom deployment with 53 high school students. Results show that MAIC-UI reduces authoring time from days to minutes, achieves significantly faster edit response times compared to full regeneration baselines, and improves student engagement and learning outcomes. Teachers reported feeling empowered to create interactive content without programming, while students benefited from personalized, self-paced exploration experiences that traditional materials cannot provide.

In summary, this paper makes the following contributions:

- • A formative study identifying key challenges educators face when creating AI-generated interactive courseware, informing four design goals for pedagogical content creation tools.
- • The design and implementation of MAIC-UI, a zero-code authoring system featuring structured knowledge analysis, two-stage generation with validation, and Click-to-Locate editing with Unified Diff-based incremental updates.
- • Empirical evidence from a controlled study (N=40) and classroom deployment demonstrating that MAIC-UI significantly reduces authoring burden, enables rapid iteration, and improves learning outcomes.

- 2 RELATED WORK 2.1 LLMs in Education

Large language models are reshaping the production of educational content by simulating authentic dialogues and narrative logic to enhance learner immersion. For instance, Oak Story leverages LLMmediated interactive narratives to construct immersive ecological learning environments [11], while TutorCraftEase supports automated generation of pedagogical questions and reduces teachers’ burden [24]. Intelligent tutoring systems are also moving toward dynamic generation of personalized learning paths: LearnMate tailors learning plans for students with diverse backgrounds [49], and GuideAI further incorporates physiological feedback to adapt instructional pacing [42]. In project-based learning, LLM-assisted

systems are transitioning from static guidance to dynamic interaction. While AutoPBL supports autonomous exploration through intelligent checkpoints [60], another toolkit-based practice [30] improves students’ engagement through multimodal creation.

However, despite improved content-production efficiency, existing educational LLM tools still struggle to ensure scientific rigor and visual professionalism, and often exhibit knowledge hallucination or misalignment between generated content and learning objectives [37, 47]. In addition, many systems still produce primarily presentation-oriented courseware and lack effective parametric interaction mechanisms that can genuinely support personalized learning-by-doing [52, 58].

To address these issues, MAIC-UI establishes a two-stage workflow to improve pedagogical reliability and visual quality, and supports interactive component generation with real-time feedback to foster exploration through learner-controlled manipulation.

### 2.2 AI Co-creation Authoring Tools

In human-AI co-creation, a central design question is how to calibrate system agency so that AI can provide inspiration while preserving human decision authority. The COFI framework systematically characterizes this interaction design space [38], and Reframer demonstrates how systems can balance emergent inspiration with user control [28]. Domain-specific authoring tools are also improving creative quality by integrating contextual knowledge. For example, AIdeation supports concept designers with efficient workflows for reorganizing reference materials [48], while VRCopilot improves layout control in VR through intermediate representations such as wireframes [56]. In UI-oriented creation, recent work has focused on automated evaluation and collaborative generation of high-fidelity components: DynaVis synthesizes interactive widgets from natural language [46], and LLM-based evaluation can detect logical issues in UI mockups and provide optimization suggestions [14].

Although current co-creation tools are effective at inspiration support, educators still face a single-change-affects-all dilemma in fine-grained revisions due to limited local editability. Slow feedback loops also interrupt creative flow and increase iteration cost [32, 35].

MAIC-UI addresses these gaps with Unified Diff-based incremental code updates, reducing file-modification latency from minutes to seconds, and with click-to-locate zero-code precise editing, enabling educators to revise specific interaction details directly while avoiding the instability of full regeneration.

### 2.3 End-User Programming

LLM-driven end-user programming (EUP) is evolving toward a responsible collaboration model that emphasizes guidance rather than one-shot code generation. CodeAid adopts cognitively supportive outputs such as pseudocode and explanatory comments [26], while SketchGPT introduces multimodal communication channels (e.g., sketches and language) for novice users [23]. For understanding and modifying existing programs, interactive code-morphing approaches lower barriers via visual feedback: TweakIt allows nonexperts to iteratively transform code behavior through real-time interaction [27], and Ply introduces clear boundary management for trigger-action programming [31]. EUP is also expanding into emerging domains such as mixed reality and robotics: agentAR

supports rapid AR application construction through natural language [59], and Alchemist simplifies robot behavior authoring into collaborative goal specification [25]. Recent work further explores support for developer’s reflection in AI-assisted workflows [4].

Despite these advances, non-expert users still face barriers when modifying AI-generated artifacts. Existing EUP tools primarily support generating code from scratch but offer limited mechanisms for localizing and editing specific elements within generated programs [34, 51]. Users must either describe changes ambiguously in natural language or directly manipulate low-level code, creating cognitive barriers that violate the core EUP principle of letting users focus on what to achieve rather than how to implement it.

MAIC-UI addresses this gap through click-to-locate: a mechanism that bridges visual elements and their underlying code by allowing users to select UI components through direct manipulation and describe desired changes in natural language. This approach enables teachers to iteratively refine interactive courseware without understanding DOM structures or CSS syntax, embodying the EUP vision of democratizing programming capabilities for domain experts.

### 3 FORMATIVE STUDY

To inform the design of MAIC-UI, we conducted a formative study to understand how educators experience AI-generated interactive courseware, the challenges they face during creation, and their perceptions of classroom integration. While prior research has explored AI-powered educational tools, limited understanding exists regarding teachers’ experiences with generative UI systems that transform static teaching materials into interactive websites.

Our study was guided by the following research questions:

- (1) How do educators navigate the interactive courseware creation process with AI assistance?
- (2) What cognitive challenges do educators face when articulating instructional requirements to AI systems?
- (3) How do educators perceive the integration of AI-generated interactive materials into classroom education?

### 3.1 Process

We conducted semi-structured interviews with 6 participants recruited through local university networks. All participants were senior students from top Chinese universities with at least one teaching experience. Each participant completed an 1-hour hands-on session with MAIC-UI’s initial version followed by a semi-structured interview exploring their creation experience, perceived learning costs, creative amplification, classroom integration concerns, and views on procedural knowledge visualization. We performed qualitative thematic analysis on the interview transcripts to identify recurring patterns and insights that informed our design goals. See Appendix B for detailed recruitment criteria and participant quotes.

### 3.2 Findings

3.2.1 F1: Knowledge Accuracy Concerns in AI-Generated Content. Participants expressed concerns about knowledge accuracy and content fidelity in AI-generated courseware. P3 noted concerns about how the system “represents knowledge,” while P4 stated that “sometimes the knowledge it produces is simply incorrect.” Participants also identified mismatches between input content and

generated output (P6: “The generated website didn’t include the content I had specified”). These findings reveal that teachers need mechanisms to ensure knowledge accuracy and content alignment. See Appendix C for extended participant quotes.

- 3.2.2 F2: Limitations of Current Editing Mechanisms. Participants expressed the need for localized, granular editing without coding. P3 noted that “modification and editing aren’t that easy,” while P5 estimated needing “three to four” iterations to achieve desired results. P6 described frustration with failed edits when the system “didn’t follow my instructions.” These findings highlight a demand for “what you see is what you get” editing control that precisely targets specific elements without requiring code knowledge or multiple regeneration cycles.
- 3.2.3 F3: Passive Courseware Fails to Engage Active Exploration. Participants identified that traditional courseware promotes passive reception rather than active exploration. P4 described traditional PPT teaching as “fixed content” where “students may find it boring.” P2 emphasized that interactive tools “offer students a buffet” in an immediately actionable way, leading to higher participation. This reveals a need for hands-on, self-paced exploration opportunities.
- 3.2.4 F4: Static Materials Cannot Bridge Theory-Practice Gaps. Participants highlighted that students struggle to bridge the gap between conceptual knowledge and practical application. P4 explained that “what they learn from textbooks and real-world scenarios have a gap,” while P2 noted that “problems are written elaborately, but the underlying knowledge is simple—some students cannot cross this chasm.” These findings reveal a demand for dynamic visualizations that make abstract procedural steps concrete and visible.

### 3.3 Design Goals

Drawing on our formative study findings and learning theories, we identify four key design goals for an AI-powered interactive courseware authoring system:

- • DG1: Ensure pedagogical scientificity and visual professionalism. Address knowledge hallucinations and visual inconsistencies through a structured knowledge analysis framework and two-stage generate-verify-optimize pipeline, ensuring generated courseware maintains both scientific rigor and professional aesthetics.
- • DG2: Enable zero-code precise editing. Support finegrained localized adjustments through Click-to-Locate element selection and Unified Diff-based incremental generation, reducing edit response times from 200–600 seconds to under 10 seconds while preserving creative flow.
- • DG3: Foster active learning and personalized exploration. Enable hands-on interaction with customizable parameters and real-time feedback, embodying Constructionist “Learning by Doing” principles to transform passive content consumption into active knowledge construction.
- • DG4: Visualize procedural knowledge to bridge learning gaps. Generate dynamic visualizations at low cost to make abstract procedural logic concrete and visible [41], reducing cognitive load and bridging the gap between conceptual understanding and practical application.

### 4 THE MAIC-UI SYSTEM

Drawing on findings from our formative study (Section 3), we create a web-based authoring system, MAIC-UI, that enables educators to create interactive courseware from language instructions, PPTs, and PDFs without requiring programming expertise. Figure 1 illustrates the overall system architecture, which consists of three interconnected stages: Content Analysis, Two-Stage Generation, and Click-to-Locate Editing. Below, we present an illustrative scenario demonstrating how teachers use MAIC-UI in practice, followed by detailed descriptions of the core features and their implementation.

### 4.1 Example Scenario

Ms. Chen is a middle school physics teacher preparing a lesson on gravitational potential energy. She uploads a 45-page textbook chapter to MAIC-UI (Figure 1, left). The system analyzes the content and identifies the subject area (Physics), key concepts (gravitational potential energy, mass, height), learning objectives, and the core formula 𝐸𝑝 = 𝑚𝑔ℎ. Then the Two-Stage Generation Pipeline executes:

- Stage 1 creates an aligned simulation with a left panel showing the step-by-step process and a right panel with interactive controls;
- Stage 2 applies visual polish using a blue theme appropriate for physics content (Figure 1, center). The complete generation takes 2.5 minutes. Noticing that the title appears plain, Ms. Chen clicks directly on it in the preview and types “make this gradient red and bold.” The system identifies the selected element, processes her instruction, and applies the change in 8 seconds (Figure 1, right).

In the classroom, students interact with the generated simulation on their tablets. Each student can freely adjust the mass and height parameters, observing how the gravitational potential energy changes in real-time. One student comments: “I can try different values without worrying about making mistakes. When I get stuck, the simulation shows me what happens—it’s like having a tutor.” This personalized exploration embodies the active learning principle (DG3), where students construct knowledge through hands-on experimentation rather than passive reception. The simulation also includes an animated visualization showing energy bars that grow and shrink as parameters change, making the abstract formula concrete and visible (DG4).

### 4.2 Key Features

MAIC-UI integrates four key features that address the challenges identified in our formative study: structured knowledge analysis with multi-modal understanding (addressing F1, DG1), two-stage generation with validation (F1, DG1), and Click-to-Locate natural language editing with Unified Diff-based incremental generation (F2, DG2). These features collectively enable teachers to create pedagogically sound, visually professional interactive courseware while supporting student-centered active learning (F3, DG3) and procedural knowledge visualization (F4, DG4).

4.2.1 Structured Knowledge Analysis with Multi-Modal Understanding. MAIC-UI supports two input modes to accommodate different teacher workflows: (1) uploading PDFs for automatic concept extraction, or (2) directly entering structured pedagogical content. This dual-input design ensures that teachers can either leverage existing teaching materials or craft custom content from scratch.

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

A B

[Figure 30]

[Figure 31]

MAIC-UI

[Figure 32]

[Figure 33]

- Figure 2: PDF Document Analysis and HTML Courseware Generation Process. (A) Teachers upload PDF documents containing pedagogical content. The system extracts page images and processes them through a vision-language model with structured analysis prompts to extract key concepts. (B) Based on the extracted structured knowledge, MAIC-UI applies subject-specific visual themes and generates interactive HTML courseware with pedagogically accurate content and professional styling.

PDF-Based Input with Automatic Extraction. Addressing participants’ concerns about knowledge accuracy in AI-generated content (F1), MAIC-UI processes uploaded PDF documents through a multi-modal analysis pipeline (Figure 2) that ensures accurate content understanding before generation. The system handles large documents (up to 50 pages) by extracting page images and sending

- them to a vision-language model with a structured analysis prompt. This approach prevents knowledge hallucinations and distinguishes MAIC-UI from simple Text-to-HTML systems that often produce pedagogically unsound content.

Direct Concept Input. Alternatively, teachers can directly input structured pedagogical content without uploading documents (Figure 3). This mode is particularly useful when teachers want to create courseware from scratch or when source materials are not available in PDF format. Teachers provide the subject, concept name, overview, mastery points, and design ideas through a structured form interface. This direct input bypasses the extraction step while still leveraging the same structured knowledge framework and generation pipeline.

Both input modes populate the same structured knowledge representation that drives subsequent generation. The analysis extracts six key fields that drive generation decisions: Main Topics identify the broad subject areas covered; Key Concepts capture specific terminology and principles students must master; Learning Objectives define measurable outcomes; Prerequisite Knowledge identifies foundational concepts; Procedural Concepts represent step-by-step processes that can be transformed into interactive simulations; and Subject Area and Grade Level enable theme selection. The procedural concepts field is particularly important—it captures processes that can become interactive simulations, distinguishing MAIC-UI from systems that only extract static knowledge.

Based on the extracted subject area, the system selects a visual theme from a predefined palette (e.g., blue tones for Physics, green for Biology, orange for Chemistry). Each theme defines primary and accent colors that ensure visual consistency throughout the generated courseware. This structured knowledge analysis framework ensures that all subsequent generation decisions are grounded in pedagogically accurate content understanding (DG1).

4.2.2 Two-Stage Generation with Validation. MAIC-UI employs a two-stage generation pipeline (Figure 1, center panel) that balances pedagogical accuracy (F1, DG1) with visual professionalism (F1, DG1). This design addresses our formative finding that single-pass generation often produces either “correct but ugly” or “beautiful but wrong” outputs. As P4 emphasized, “Sometimes the knowledge it produces is simply incorrect,” highlighting the need for validation.

###### Stage 1: Content-Aligned Interactive Simulations. The first

stage generates interactive simulations tightly aligned with extracted content. The layout consists of a left-side process panel displaying step-by-step procedural knowledge and a right-side simulation panel providing interactive visualization where students manipulate parameters and observe real-time effects. A coupling mechanism ensures that changes in the process panel automatically update the simulation state. The AI generates HTML/JavaScript code implementing this layout, and the output is validated to ensure interactive elements function correctly and respond to user input in real-time. If validation fails, the system attempts refinements, feeding validation errors back to the AI for regeneration.

###### Stage 2: Layout Polish and Visual Refinement. The second

stage receives Stage 1 output and applies visual enhancements including layout verification, theme color application, typography improvements, and animation smoothing. An HTML validator checks for well-formed structure and proper syntax. Multiple refinement attempts are allowed to address any issues.

[Figure 34]

[Figure 35]

Original Concept Input

[Figure 36]

###### SUBJECT: Physics

[Figure 37]

A B

CONCEPT NAME: Newton's First Law (Inertia)

CONCEPT OVERVIEW: An object remains at rest or in uniform linear motion unless acted upon by an external force. The greater the mass, the greater the inertia.

[Figure 38]

MAIC-UI

###### MASTERY POINTS:

① Why do people lean forward when braking? ② Why does shaking the tablecloth keep plates on the table? ③ Inertia is not a force; it is an object's property.

DESIGN IDEA: To help students understand the relationship between inertia and mass, I designed an ice skating simulator: Students adjust the skater's mass, apply the same initial velocity, and observe differences in gliding distance and the time required to come to a stop. A friction slider can be added to demonstrate how friction affects the deceleration process.

- Figure 3: Concept-to-Interactive-Courseware Generation Pipeline. (A) Teachers input structured pedagogical content including subject, concept name, overview, mastery points, and design ideas. (B) MAIC-UI processes this input through the two-stage generation pipeline to produce interactive HTML courseware. The example shows a physics simulation for Newton’s First Law where students can adjust skater mass, initial velocity, and friction parameters to observe how inertia affects stopping distance. The generated output includes interactive controls, real-time visualization, scientific principles sidebar, and simulation results.

The pipeline implements graceful degradation: if Stage 1 fails, the system falls back to single-pass generation; if Stage 2 fails, basic styling is applied to Stage 1 output; if both fail, an emergency template is returned with a user-friendly message. This robust error handling ensures teachers always receive usable results.

- 4.2.3 Click-to-LocateNaturalLanguage Editing. The Click-to-Locate Editing Module (Figure 1, right panel) enables teachers to make finegrained adjustments to generated content without writing code. This module directly addresses the editing limitations identified in our formative study (F2), where participants expressed frustration with existing tools. As P3 noted, “Modification and editing aren’t that easy...sometimes AI pretends to understand but gets it wrong.” P5 estimated that achieving desired results required “three to four times” of iterations, while P6 described feeling stuck: “I kept trying to change the website...but finally, it didn’t follow my instructions.”

Element Citation System. The frontend implements a citationbased element selection system inspired by browser developer tools. When a teacher clicks on an element in the rendered preview, the system captures the element’s structure and displays it in a side panel with an index number. Visual highlighting shows a border overlay on the selected element. This approach eliminates the need for teachers to understand CSS selectors or DOM traversal—they simply point and click. As P6 highlighted, the ability to directly connect visual elements to their underlying code “reduces the uncertainty that comes from modifying through instructions alone,” enabling true “what you see is what you get” editing.

Natural Language to Code Translation. Once an element is selected, teachers describe desired changes in natural language. The system constructs a prompt including the selected element, the modification instruction, and the full HTML source. The AI model processes this prompt and returns changes in a structured format for efficient application.

Incremental Editing with Unified Diff. A key innovation in MAIC-UI is using Unified Diff format for incremental code generation. This addresses the “full regeneration bottleneck” identified in our formative study, where systems output entire modified files resulting in wait times of 200-600 seconds. Instead, MAIC-UI outputs only changed lines plus minimal context, reducing output size by approximately 90% and completing edits in under 10 seconds. The system implements fuzzy context matching to handle minor style drift, achieving high patch application rates even when the AI’s expected context differs slightly from actual content.

The performance improvement enables teachers to stay in a creative flow state. As one participant reflected: “Before, I would lose my train of thought waiting 5-10 minutes for each change. Now it’s instant—I can iterate like I’m sketching on paper.” This rapid iteration capability is essential for supporting teachers’ pedagogical creativity without technical friction (DG2).

### 4.3 Implementation

MAIC-UI is implemented as a full-stack web application designed for multi-user interaction. The frontend is built with React 18 and

TypeScript [10], providing interfaces for visual editing with splitpane preview, element citation display, and natural language chat commands. We chose React for its component-based architecture, which aligns well with our modular courseware generation approach. The backend uses Python FastAPI with SQLite [17] for data storage, implementing service layers for content generation and editing operations. The system integrates Zhipu AI’s GLM-4.7 [55] for text generation and GLM-4.6V [20] for multi-modal analysis, with fallback support for other endpoints [2, 43].

The Click-to-Locate editing module implements a DOM-aware element citation system. When a user clicks on a rendered element, the frontend captures the element’s XPath [9] and CSS selector,

- then displays the corresponding HTML snippet in a side panel. The natural language instructions are sent to the backend along with the full HTML context and selected element. The AI model processes this prompt and returns changes in Unified Diff format [36].

See Appendix E for complete API specifications, AI model configurations, prompt templates, and code-level implementation details.

### 5 EVALUATION

Ethics. This evaluation received approval from the Institutional Review Board (IRB) of the authors’ institution. All participants provided informed consent before joining the study, and demographic data (e.g., discipline background and prior teaching experience) were recorded anonymously.

### 5.1 Research Questions

To evaluate MAIC-UI in both controlled and authentic settings, we combine a lab user study on authoring performance with an in-class deployment on learning outcomes. The following research questions guide this evaluation.

- • RQ1 (Lab User Study): To what extent does MAIC-UI improve the pedagogical correctness and visual professionalism of generated instructional webpages compared with direct Text-to-HTML generation?
- • RQ2 (Lab User Study): To what extent does MAIC-UI support fine-grained, predictable, and efficient editing for users without programming experience?
- • RQ3 (In-Class Deployment): How can students’ learning agency be fostered in traditional classroom settings?
- • RQ4 (In-Class Deployment): How can gaps in learning outcomes among students with different learning abilities be mitigated?

### 5.2 Lab User Study

5.2.1 Conditions. We compare two conditions to isolate the contribution of MAIC-UI’s full pipeline, especially its intermediate structuring and targeted editing support.

- • Baseline A simplified version of MAIC-UI in which both initial prompts and subsequent revision instructions were sent directly to the AI model for HTML generation. Unlike the full MAIC-UI pipeline, this baseline did not include any intermediate processing, structured decomposition, or validation before output.
- • MAIC-UI The full version of the MAIC-UI including complete step of generation.

- 5.2.2 Participants. We recruited 40 graduate students with prior teaching practicum experience as proxy instructors for the controlled authoring evaluation. Their disciplinary backgrounds included Social Sciences and Management (𝑛 = 2), Computer Science (𝑛 = 7), Basic Sciences (𝑛 = 8), and Engineering (𝑛 = 23). Participants were evenly assigned to the two study conditions, with 20 in the experimental group and 20 in the control group.
- 5.2.3 Authoring Tasks. We designed the study task to reflect the complete MAIC-UI workflow, spanning both teacher preparation and student learning. Each participant engaged in both interactive courseware authoring and the review of webpages created by others, allowing them to evaluate the system from the perspectives of both content creators and learners.

To ensure realism and pedagogical validity, we used authentic STEM teaching materials collected from real educational settings ranging from primary school to graduate-level courses. These materials were collected with IRB approval and consent from both the instructors and the student authors. Each task package consisted of a 20–30 page slide deck and a corresponding teaching outline, covering subjects such as science, chemistry, biology, mathematics, and geography, and was assigned to participants based broadly on their disciplinary backgrounds.

- 5.2.4 Procedure. Each session lasted approximately 80 minutes and consisted of four stages: material familiarization, interactive courseware authoring, peer review, and a post-study interview.

- Stage 1: Material Familiarization. Participants first spent ap-

proximately 10 minutes familiarizing themselves with the assigned teaching materials. This stage helped them develop an initial understanding of the lesson content and how it might be organized into an interactive instructional webpage.

- Stage 2: Interactive Courseware Authoring. Participants then

spent approximately 45 minutes using the assigned system to create and modify interactive instructional webpages. An initial version had been pre-generated from the slide materials for efficiency, but participants were still required to upload PDF or text materials themselves so that they could experience the full document-based generation workflow before further revising the webpage.

- Stage 3: Peer-work Review. Participants then spent approx-

imately 10 minutes reviewing and interacting with webpages created by other participants from a learner’s perspective.

- Stage 4: Post-task Questionnaire and Interview. Participants

thencompletedapost-taskquestionnaire followed by a semi-structured interview, in which we explored their perceptions of the system’s usability, generation quality, and overall value for teaching. See Appendix D for the complete questionnaire and interview guide.

- 5.2.5 Results. In this subsection, we report quantitative and qualitative findings from the lab study. We first present editing-behavior outcomes, and then summarize questionnaire and interview results to provide a more complete view of usability and perceived value.

Editing Accuracy and Learnability. As shown in Table 1, editing behavior differed noticeably across the two conditions. Participants in the MAIC-UI condition generally completed refinement in fewer iterations, most often within about 3–7 rounds, whereas

Table 1: Comparison of editing behavior between MAIC-UI and the baseline condition (𝑛 = 20 per condition).

Method M SD Mdn IQR

MAIC-UI 4.90 2.88 4.50 2.75–7.00 Baseline 7.00 2.20 7.00 5.00–9.00

Mann–Whitney 𝑈 test: 𝑈 = 113.0, 𝑝 = 0.019, effect size 𝑟 = 0.37.

those in the baseline condition more commonly needed around 5–9 rounds. This suggests that MAIC-UI supported a more efficient and stable path from initial draft to target outcome (supporting RQ2). One explanation is that the Click-to-Locate interaction enabled localized, intention-aligned revisions, thereby reducing the need for repeated global regeneration (Figure 1, right).

Questionnaires. For each learnability and usability item, we encoded five-point Likert responses as ordinal scores from 1 (strongly disagree) to 5 (strongly agree), and compared MAIC-UI and the baseline using two-sided Mann–Whitney 𝑈 tests (𝑛 = 20 per condition). We used 𝑝 < 0.05 as the threshold for statistical significance. As illustrated in Figure 4, the results show that MAIC-UI was rated significantly higher than the baseline on two of the four items, while the remaining two items did not reach statistical significance. Specifically, Item 1 (𝑀 = 3.80 vs. 3.15, 𝑝 = 0.042) and Item 3 (𝑀 = 4.25 vs. 3.60, 𝑝 = 0.033) showed significant differences, whereas Item 2 (𝑀 = 3.90 vs. 3.55, 𝑝 = 0.199) and Item 4 (𝑀 = 4.20 vs. 3.90, 𝑝 = 0.305) did not. For the two non-significant items, ratings in both conditions were already relatively high, suggesting a possible ceiling effect that limited further between-condition separation. Taken together, these results indicate that MAIC-UI better supports users without programming experience in performing editing in a more predictable and efficient manner, especially in helping them become familiar with the system more quickly and obtain results that better aligned with their editing intentions (supporting RQ2).

To further examine webpage quality beyond editing experience, we analyzed how participants in the MAIC-UI condition perceived the generated webpages themselves. Experimental-group participants rated all six RQ1-specific items positively, including layout intuitiveness (𝑀 = 4.15), attention attraction (𝑀 = 4.20), accurate presentation of key concepts and procedural steps (𝑀 = 4.30), coverage of key teaching points (𝑀 = 4.45), clarity of instructional language (𝑀 = 4.40), and intuitive presentation of key concepts (𝑀 = 4.35) (Figure 5). All six items received mean ratings above 4.0, indicating that MAIC-UI achieved consistently positive evaluations in terms of visual quality, pedagogical soundness, and accessibility (supporting RQ1).

Interview Findings. To complement the behavioral and questionnaire results, we analyzed participants’ semi-structured interviews and identified two themes.

Theme 1. MAIC-UI helps translate ideas into workable instructional presentations. Participants described being able to offload initial structuring work to the system. P1 noted that while making slides manually requires finding images and typesetting formulas, MAIC-UI generates “the layout and even questions,” feeling “very efficient overall.” P17 highlighted a case where the system

“generated a presentation using a running track to illustrate the relationship between linear velocity and angular velocity”—described as “a very brilliant classroom introduction.” P8 estimated efficiency improvements of “around three times,” noting that “you only need to give it the goal you want, and it can directly make it for you.”

Theme 2. Effective editing requires explicit intentions but is learnable. P1 emphasized that “if you describe to the AI in more detail and with more precision, it can often give you the effect you want most.” Participants viewed this skill as developable rather than expert-only—P4 noted that “if you do it a few more times or practice more diligently, the results will get better,” and P1 suggested that with documentation and training, even novice users could find the system “quite easy to pick up.”

### 5.3 In-Class Deployment

- 5.3.1 Classroom Context. Beyond the controlled user study, we conducted a three-month in-the-wild deployment of MAIC-UI in a second-year class at a county-level public high school in China. Rather than aiming for a strictly controlled evaluation, this deployment sought to situate MAIC-UI within an authentic and sustained teaching context, allowing us to examine its practical role in everyday classroom use. Specifically, we explored whether MAIC-UI could foster students’ learning agency in traditional classrooms and whether its long-term use could help reduce disparities in learning outcomes among students with different levels of academic ability.

The grade comprised 11 classes, from which one class with 53 students was selected as the pilot class based on the school’s instructional arrangements and deployment feasibility. All students in this class were enrolled in the physics–chemistry–biology elective track, a science-oriented subject combination broadly aligned with STEM-related learning in the Chinese upper-secondary curriculum. Prior to the deployment, the classroom was equipped with the basic hardware infrastructure required for daily use of MAIC-UI, including tablets and charging cabinets.

- 5.3.2 Study Procedure. After the November monthly examination in 2025(pre-exam), Class C1 officially adopted MAIC-UI and continued using it until the final examination in February 2026(post-exam), resulting in a deployment period of approximately three months. Before each class, the teacher uploaded courseware created with MAIC-UI to the system. During class, in addition to listening to the teacher’s instruction, students could also use tablets to independently interact with the embedded interactive components to support their understanding of the course content.

As this deployment took place in an authentic and ongoing classroom setting, we did not structure it as a formal study under strictly controlled conditions. Instead, we addressed the two research questions by comparing changes in students’ performance across the two examinations and by conducting interviews with a subset of students and teachers after the deployment.

- 5.3.3 Results. In this section, we present quantitative outcomes from the classroom deployment. We report overall score gains, the distribution of gains within class, and performance changes among lower-performing students.

Overall Score Gains. We first compared score gains between Class C1 and the other classes in the same grade from the pre-exam

###### Learnability and Usability Comparison: MAIC-UI VS. Baseline

[Figure 39]

I can quickly get familiar with MAIC-UI even without professional training(p = 0.042)

MAIC-UI won't take up too much of my preparation time.(p = 0.198)

I believe that MAIC-UI can understand editing prompts and generate the results I expect.(p = 0.032)

[Figure 40]

I would prefer to use MAIC-UI in my daily teaching. (p = 0.305)

###### Figure 4: Questionnaire results comparing MAIC-UI and the baseline in the lab user study (𝑛 = 20 per condition). Stacked bars show response distributions for the four core items: learnability, time cost, editing controllability, and usage preference.

Feature-Specific Ratings of MAIC-UI

Visual Appeal and Simplicity

[Figure 41]

I think the layout of the generated webpage is intuitive.

MAIC-UI helps me better attract students’ attention.

Pedagogical Purpose and Rigor

Key concepts and procedural steps in the lesson are presented accurately.

The generated content covers the key teaching points of the lesson.

Ease of Understanding

The instructional language is clear and easy to understand, without excessive use of technical terms.

key concepts in the lesson is intuitive and easy to understand.

###### Figure 5: The six items summarize participants’ judgments along three dimensions: visual appeal and simplicity, pedagogical purpose and rigor, and ease of understanding.

- to the post-exam. Given that the classroom deployment of MAIC-UI primarily supported STEM subjects, including physics, chemistry, and biology, we separately calculated score gains for STEM subjects and humanities subjects.

of Class C1 in humanities was less pronounced. This difference is broadly consistent with the design focus of MAIC-UI, whose strengths are more likely to emerge in STEM learning content that benefits from dynamic presentation, interactive manipulation, and support for understanding processes.

The results showed that, in STEM subjects, Class C1 exhibited the most substantial improvement, with an average gain of 9.21 points, while most other classes showed only modest gains or declines. Exploratory student-level comparison indicated that C1 (𝑛 = 53, 𝑀 = 9.21) had higher STEM gains than the other classes (𝑛 = 493, 𝑀 = −2.32; Mann–Whitney 𝑈 = 16691.5, 𝑝 < 0.001, 𝑟 = 0.14).

Variance of Score Gains. Beyond overall score gains, we examined the variance of score changes across classes as an indicator of how evenly improvement was distributed among students. As shown in Figure 7, Class C1 exhibited the lowest score-gain variance in the grade (562), compared with 598–1054 for the other classes.

In humanities subjects, Class C1 also showed a positive gain, with an average increase of 6.43 points, although this was not the highest in the grade. Compared with STEM subjects, the relative advantage

This pattern suggests that learning gains in C1 were more evenly distributed across students, rather than being concentrated among a

[Figure 42]

###### Figure 6: Score gains in STEM and humanities across classes from the November 2025 monthly examination to the February 2026 final examination.

[Figure 43]

###### Figure 7: Variance of STEM score gains across classes over the same period. C1 denotes the MAIC-UI pilot class.

small subset of higher-performing learners. In relation to RQ4, this provides descriptive evidence that gaps in learning outcomes between students with different ability levels may have been mitigated. This pattern is also consistent with the mechanism proposed in our method: by aligning animated demonstrations with procedural knowledge in the generated interactive pages, the system externalizes content that would otherwise require students to mentally infer and internalize on their own, thereby making key problem-solving steps more directly learnable and potentially more accessible to lower-performing students.

Gains Among Lower-Performing Students. We further examined students in the bottom 25% of the November 2025 monthly examination. In STEM subjects, the bottom 25% students in Class C1 (𝑛 = 14) showed a larger average score gain than their counterparts in the other classes (𝑛 = 137, 15.46 vs. 12.42), a higher proportion of positive gains (78.6% vs. 63.5%), and a lower variance of score gains (251.02 vs. 671.34).

These results suggest that the benefits of MAIC-UI were not limited to students who were already more self-motivated or academically stronger. Instead, the observed pattern indicates that its support may extend to a broader range of learners, including those who initially performed at a lower level.

5.3.4 Post-Deployment Interviews. After the three-month classroom deployment, we interviewed students and teachers from Class C1. From these interviews, we identified two recurring themes.

- Theme 1. MAIC-UI expanded opportunities for active participation. Participants perceived that MAIC-UI created more entry points for students to explore. One student noted becoming “more willing to ask questions,” whereas previously they had been reluctant due to concerns about how others might see them. Teachers observed that previously quiet students had “completely opened themselves up in the smart online learning environment.”
- Theme 2. Visual interaction supported accessible understanding. Both students and teachers reported that interactive visualization helped students understand processes and relationships. One teacher noted that students could more intuitively observe parameter changes “through dragging and interaction” when explaining force analysis and projectile motion. One student remarked that while they had previously only memorized that “gravitational acceleration is 9.8,” after interacting with the system they could observe that “the higher the height, the longer the falling time,” so that they “understood it immediately.”

### 6 DISCUSSION

Our findings demonstrate that MAIC-UI successfully addresses key challenges in teacher-facing generative UI systems for educational content creation. We discuss key implications here; see Appendix A for the complete analysis.

Balancing Automation and Pedagogical Control. Teachers want AI assistance but cannot compromise on pedagogical accuracy. MAIC-UI’s two-stage pipeline separates content validation from visual polish, ensuring substance is verified before aesthetics are applied. This design reveals a broader principle: effective educational GenUI requires transparent communication of AI to help teachers leverage strengths while compensating for weaknesses.

The Importance of Rapid Iteration Cycles. Iteration speed fundamentally shapes creative engagement. Our sub-10-second iterationcycles—enabledbyUnified Diff-based incremental generationtransform authoring from a batch-oriented process into an interactive conversation. Delays exceeding approximately 10 seconds create cognitive discontinuities that disrupt creative flow.

Addressing Equity in STEM Learning. MAIC-UI’s benefits were not limited to high-performing students. Lower-performing students in the pilot class demonstrated larger score gains than their counterparts in control classes, and variance of score gains was substantially lower. This suggests interactive courseware may help address persistent achievement gaps by making abstract concepts accessible through multiple entry points.

Limitations. MAIC-UI currently focuses on single-page simulations, limiting applicability for extended narratives. The lab study used proxy instructors rather than practicing K-12 teachers, and the classroom deployment occurred in a single Chinese public high school, requiring further investigation for generalizability.

### 7 CONCLUSION

This paper presented MAIC-UI, a zero-code authoring system that enables educators to create interactive STEM courseware without

programming. The system introduces three technical contributions: structured knowledge analysis for pedagogical rigor, a generateverify-optimize pipeline, and Click-to-Locate editing with Unified Diff-based incremental generation achieving sub-10-second iteration cycles. Our three-month classroom deployment with 53 high school students demonstrated that MAIC-UI reduces editing iterations, improves learnability for non-programmers, and fosters learning agency—the pilot class achieved 9.21-point STEM gains compared to -2.32 points in control classes. Our findings suggest that well-designed interactive courseware can help address persistent achievement gaps in STEM education. Future work will explore template-based courseware support, additional disciplines, and long-term integration across diverse educational contexts.

### REFERENCES

- [1] Azza Abouhashem, Rana Magdy Abdou, Jolly Bhadra, Malavika Santhosh, Zubair Ahmad, and Noora Jabor Al-Thani. 2021. A Distinctive Method of Online Interactive Learning in STEM Education. Sustainability 13, 24 (2021). https://doi.org/10.3390/su132413909
- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774

(2023).

- [3] Eman A Alasadi and Carlos R Baiz. 2023. Generative AI in education and research: Opportunities, concerns, and solutions. Journal of Chemical Education 100, 8

(2023), 2965–2971.

- [4] Timothy J. Aveni, Hila Mor, Armando Fox, and Björn Hartmann. 2025. Generative Trigger-Action Programming with Ply. In Proceedings of the 38th Annual ACM Symposium on User Interface Software and Technology (UIST ’25). Association for Computing Machinery, New York, NY, USA, 1–17. https://doi.org/10.1145/ 3746059.3747638
- [5] Steve Olusegun Bada and Steve Olusegun. 2015. Constructivism learning theory: A paradigm for teaching and learning. Journal of Research & Method in Education 5, 6 (2015), 66–70.
- [6] Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei Hou, Yuxiao Dong, et al. 2025. Longbench v2: Towards deeper understanding and reasoning on realistic long-context multitasks. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 3639–3664.
- [7] Peter C Bell and Robert M O’keefe. 1987. Visual interactive simulation—history, recent developments, and major issues. Simulation 49, 3 (1987), 109–116.
- [8] Mhamed Ben Ouahi, Driss Lamri, Taoufik Hassouni, and El Mehdi Al Ibrahmi.

2022. Science Teachers’ Views on the Use and Effectiveness of Interactive Simulations in Science Teaching and Learning. International journal of instruction 15, 1 (2022), 277–292.

- [9] Michael Benedikt and Christoph Koch. 2009. XPath leashed. ACM Computing Surveys (CSUR) 41, 1 (2009), 1–54.
- [10] Gavin Bierman, Martín Abadi, and Mads Torgersen. 2014. Understanding typescript. In European Conference on Object-Oriented Programming. Springer, 257– 281.
- [11] Alan Y. Cheng, Carolyn Q. Zou, Anthony Xie, Matthew Hsu, Felicia Yan, Felicity Huang, David K. Zhang, Arjun Sharma, Rashon Poole, Daniel Wan Rosli, Andrea Cuadra, Roy Pea, and James A. Landay. 2025. Oak Story: Improving Learner Outcomes with LLM-Mediated Interactive Narratives. In Proceedings of the 38th Annual ACM Symposium on User Interface Software and Technology (UIST ’25). Association for Computing Machinery, New York, NY, USA, 1–17. https://doi. org/10.1145/3746059.3747698
- [12] Juhriyansyah Dalle et al. 2017. Interactive courseware for supporting learners competency in practical skills. Turkish Online Journal of Educational TechnologyTOJET 16, 3 (2017), 88–99.
- [13] Kevin Doherty and Gavin Doherty. 2018. Engagement in HCI: conception, theory and measurement. ACM computing surveys (CSUR) 51, 5 (2018), 1–39.
- [14] Peitong Duan, Jeremy Warner, Yang Li, and Bjoern Hartmann. 2024. Generating Automatic Feedback on UI Mockups with Large Language Models. In Proceedings of the CHI Conference on Human Factors in Computing Systems (CHI ’24). Association for Computing Machinery, New York, NY, USA, 1–20. https://doi.org/10.1145/3613904.3642782
- [15] Sarah Fakhoury, Aaditya Naik, Georgios Sakkas, Saikat Chakraborty, and Shuvendu K Lahiri. 2024. Llm-based test-driven interactive code generation: User study and empirical evaluation. IEEE Transactions on Software Engineering 50, 9

(2024), 2254–2268.

- [16] Xinxin Fan and DaviD Geelan. 2013. Enhancing students’ scientific literacy in science education using interactive simulations: A critical literature review. Journal of Computers in Mathematics and Science Teaching 32, 2 (2013), 125–171.
- [17] Kevin P Gaffney, Martin Prammer, Larry Brasfield, D Richard Hipp, Dan Kennedy, and Jignesh M Patel. 2022. SQLite: past, present, and future. Proceedings of the VLDB Endowment 15, 12 (2022).
- [18] Ahmed M Gharib, Gregory M Peterson, Ivan K Bindoff, and Mohammed S Salahudeen. 2023. Potential barriers to the implementation of computer-based simulation in pharmacy education: a systematic review. Pharmacy 11, 3 (2023), 86.
- [19] James Hollan, Edwin Hutchins, and David Kirsh. 2000. Distributed cognition: toward a new foundation for human-computer interaction research. ACM Transactions on Computer-Human Interaction (TOCHI) 7, 2 (2000), 174–196.
- [20] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. 2025. Glm-4.5 v and glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv preprint arXiv:2507.01006 (2025).
- [21] Teoh Sian Hoon, Toh Seong Chong, and Nor Azilah Binti Ngah. 2010. Effect of an Interactive Courseware in the Learning of Matrices. Journal of Educational Technology & Society 13, 1 (2010), 121–132.
- [22] Yingbing Huang, Lily Jiaxin Wan, Hanchen Ye, Manvi Jha, Jinghua Wang, Yuhong Li, Xiaofan Zhang, and Deming Chen. 2024. New solutions on LLM acceleration, optimization, and application. In Proceedings of the 61st ACM/IEEE Design Automation Conference. 1–4.
- [23] Zeyuan Huang, Cangjun Gao, Yaxian Shan, Haoxiang Hu, Qingkun Li, Xiaoming Deng, Cuixia Ma, Yu-Kun Lai, Yong-Jin Liu, Feng Tian, Guozhong Dai, and Hongan Wang. 2025. SketchGPT: A Sketch-based Multimodal Interface for Application-Agnostic LLM Interaction. In Proceedings of the 38th Annual ACM Symposium on User Interface Software and Technology. ACM, Busan Republic of Korea, 1–18. https://doi.org/10.1145/3746059.3747598
- [24] Wenhui Kang, Lin Zhang, Xiaolan Peng, Hao Zhang, Anchi Li, Mengyao Wang, Jin Huang, Feng Tian, and Guozhong Dai. 2025. TutorCraftEase: Enhancing Pedagogical Question Creation with Large Language Models. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems (CHI ’25). Association for Computing Machinery, New York, NY, USA, 1–22. https://doi. org/10.1145/3706598.3713731
- [25] Ulas Berk Karli, Juo-Tung Chen, Victor Nikhil Antony, and Chien-Ming Huang.

2024. Alchemist: LLM-Aided End-User Development of Robot Applications. In Proceedings of the 2024 ACM/IEEE International Conference on Human-Robot Interaction. ACM, Boulder CO USA, 361–370. https://doi.org/10.1145/3610977. 3634969

- [26] Majeed Kazemitabaar, Runlong Ye, Xiaoning Wang, Austin Zachary Henley, Paul Denny, Michelle Craig, and Tovi Grossman. 2024. CodeAid: Evaluating a Classroom Deployment of an LLM-based Programming Assistant That Balances Student and Educator Needs. In Proceedings of the CHI Conference on Human Factors in Computing Systems. 1–20. https://doi.org/10.1145/3613904.3642773 arXiv:2401.11314 [cs]
- [27] Sam Lau, Sruti Srinivasa Srinivasa Ragavan, Ken Milne, Titus Barik, and Advait Sarkar. 2021. TweakIt: Supporting End-User Programmers Who Transmogrify Code. In Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems (CHI ’21). Association for Computing Machinery, New York, NY, USA, 1–12. https://doi.org/10.1145/3411764.3445265
- [28] Tomas Lawton, Francisco J Ibarrola, Dan Ventura, and Kazjon Grace. 2023. Drawing with Reframer: Emergence and Control in Co-Creative AI. In Proceedings of the 28th International Conference on Intelligent User Interfaces. ACM, Sydney NSW Australia, 264–277. https://doi.org/10.1145/3581641.3584095
- [29] Yaniv Leviathan, Dani Valevski, et al. 2025. Generative UI: LLMs are effective UI generators. Technical Report. Technical report, Google Research, 2025. Available at generativeui. github. io.
- [30] Hanqi Li, Ruiwei Xiao, Hsuan Nieu, Ying-Jui Tseng, and Guanze Liao. 2025. “From Unseen Needs to Classroom Solutions”: Exploring AI Literacy Challenges & Opportunities with Project-Based Learning Toolkit in K-12 Education. Proceedings of the AAAI Conference on Artificial Intelligence 39, 28 (2025), 29145–29152. https: //doi.org/10.1609/aaai.v39i28.35187
- [31] Sarah Lim, Joshua Hibschman, Haoqi Zhang, and Eleanor O’Rourke. 2018. Ply: A Visual Web Inspector for Learning from Professional Webpages. In Proceedings of the 31st Annual ACM Symposium on User Interface Software and Technology (UIST ’18). Association for Computing Machinery, New York, NY, USA, 991–1002. https://doi.org/10.1145/3242587.3242660
- [32] Yuyu Lin, Jiahao Guo, Yang Chen, Cheng Yao, and Fangtian Ying. 2020. It Is Your Turn: Collaborative Ideation With a Co-Creative Robot through Sketch. In Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems (CHI ’20). Association for Computing Machinery, New York, NY, USA, 1–14. https://doi.org/10.1145/3313831.3376258
- [33] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Transactions of the association for computational linguistics 12

(2024), 157–173.

- [34] Boxuan Ma, Huiyong Li, Gen Li, Li Chen, Cheng Tang, Yinjie Xie, Chenghao Gu, Atsushi Shimada, and Shin’ichi Konomi. 2025. Scaffolding Metacognition in Programming Education: Understanding Student-AI Interactions and Design Implications. https://doi.org/10.48550/arXiv.2511.04144 arXiv:2511.04144 [cs]
- [35] Caterina Moruzzi and Solange Margarido. 2024. A User-centered Framework for Human-AI Co-creativity. In Extended Abstracts of the CHI Conference on Human Factors in Computing Systems (CHI EA ’24). Association for Computing Machinery, New York, NY, USA, 1–9. https://doi.org/10.1145/3613905.3650929
- [36] Yusuf Sulistyo Nugroho, Hideaki Hata, and Kenichi Matsumoto. 2020. How different are different diff algorithms in Git? Use–histogram for code changes. Empirical Software Engineering 25, 1 (2020), 790–823.
- [37] Minju Park, Sojung Kim, Seunghyun Lee, Soonwoo Kwon, and Kyuseok Kim.

2024. Empowering Personalized Learning through a Conversation-based Tutoring System with Student Modeling. In Extended Abstracts of the CHI Conference on Human Factors in Computing Systems (CHI EA ’24). Association for Computing Machinery, New York, NY, USA, 1–10. https://doi.org/10.1145/3613905.3651122

- [38] Jeba Rezwana and Mary Lou Maher. 2023. Designing Creative AI Partners with COFI: A Framework for Modeling Interaction in Human-AI Co-Creative Systems. ACM Transactions on Computer-Human Interaction 30, 5 (2023), 1–28. https://doi.org/10.1145/3519026
- [39] John Richards, William Barowy, and Dov Levin. 1992. Computer simulations in the science classroom. Journal of Science Education and Technology 1, 1 (1992), 67–79.
- [40] Georges L Savoldelli, Viren N Naik, Stanley J Hamstra, and Pamela J Morgan.

2005. Barriers to use of simulation-based education. Canadian Journal of Anesthesia/Journal canadien d’anesthésie 52, 9 (2005), 944–950.

- [41] Qifan Shu. 2025. Towards Conversational End-User Programming for Homes Prototyping and Evaluation of a Visually-Augmented Voice Interface. Thesis. University of Sussex.
- [42] Ananya Shukla, Chaitanya Modi, Satvik Bajpai, and Siddharth Siddharth. 2026. GuideAI: A Real-Time Personalized Learning Solution with Adaptive Interventions. https://doi.org/10.48550/arXiv.2601.20402 arXiv:2601.20402 [cs]
- [43] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530 (2024).
- [44] Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Y Charles, HS Che, Cheng Chen, Guanduo Chen, et al. 2026. Kimi K2. 5: Visual Agentic Intelligence. arXiv preprint arXiv:2602.02276 (2026).
- [45] Shangqing Tu, Zheyuan Zhang, Jifan Yu, Chunyang Li, Siyu Zhang, Zijun Yao, Lei Hou, and Juanzi Li. 2023. LittleMu: Deploying an online virtual teaching assistant via heterogeneous sources integration and chain of teach prompts. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management. 4843–4849.
- [46] Priyan Vaithilingam, Elena L. Glassman, Jeevana Priya Inala, and Chenglong Wang. 2024. DynaVis: Dynamically Synthesized UI Widgets for Visualization Editing. In Proceedings of the CHI Conference on Human Factors in Computing Systems (CHI ’24). Association for Computing Machinery, New York, NY, USA, 1–17. https://doi.org/10.1145/3613904.3642639
- [47] Shen Wang, Tianlong Xu, Hang Li, Chaoli Zhang, Joleen Liang, Jiliang Tang, Philip S. Yu, and Qingsong Wen. 2025. Large Language Models for Education: A Survey and Outlook. IEEE Signal Processing Magazine 42, 6 (2025), 51–63. https://doi.org/10.1109/MSP.2025.3594309
- [48] Wen-Fan Wang, Chien-Ting Lu, Nil Ponsa I Campanyà, Bing-Yu Chen, and Mike Y. Chen. 2025. AIdeation: Designing a Human-AI Collaborative Ideation System for Concept Designers. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems (CHI ’25). Association for Computing Machinery, New York, NY, USA, 1–28. https://doi.org/10.1145/3706598.3714148
- [49] Xinyu Jessica Wang, Christine P. Lee, and Bilge Mutlu. 2025. LearnMate: Enhancing Online Education with LLM-Powered Personalized Learning Plans and Support. In Proceedings of the Extended Abstracts of the CHI Conference on Human Factors in Computing Systems (CHI EA ’25). Association for Computing Machinery, New York, NY, USA, 1–10. https://doi.org/10.1145/3706599.3719857
- [50] Yucheng Wang, Jifan Yu, Daniel Zhang-Li, Joy Jia Yin Lim, Shangqing Tu, Haoxuan Li, Zhiyuan Liu, Huiqin Liu, Lei Hou, Juanzi Li, et al. 2025. EduCraft: A system for generating pedagogical lecture scripts from long-context multimodal presentations. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management. 6153–6160.
- [51] Litao Yan, Jeffrey Tao, Lydia B Chilton, and Andrew Head. 2025. Answering Developer Questions with Annotated Agent-Discovered Program Traces. In Proceedings of the 38th Annual ACM Symposium on User Interface Software and Technology. ACM, Busan Republic of Korea, 1–14. https://doi.org/10.1145/3746059.3747652
- [52] Soojeong Yoo, Sunkyung Kim, and Youngho Lee. 2020. Learning by doing: evaluation of an educational VR application for the care of schizophrenic patients. In Extended abstracts of the 2020 CHI conference on human factors in computing systems. 1–6.
- [53] Jifan Yu, Xiaozhi Wang, Shangqing Tu, Shulin Cao, Daniel Zhang-Li, Xin Lv, Hao Peng, Zijun Yao, Xiaohan Zhang, Hanming Li, Chunyang Li, Zheyuan Zhang,

- Yushi Bai, Yantao Liu, Amy Xin, Kaifeng Yun, Linlu GONG, Nianyi Lin, Jianhui Chen, Zhili Wu, Yunjia Qi, Weikai Li, Yong Guan, Kaisheng Zeng, Ji Qi, Hailong Jin, Jinxin Liu, Yu Gu, Yuan Yao, Ning Ding, Lei Hou, Zhiyuan Liu, Xu Bin, Jie Tang, and Juanzi Li. 2024. KoLA: Carefully Benchmarking World Knowledge of Large Language Models. In The Twelfth International Conference on Learning Representations. https://openreview.net/forum?id=AqN23oqraW
- [54] Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chengxing Xie, Cunxiang Wang, et al. 2026. GLM-5: from Vibe Coding to Agentic Engineering. arXiv preprint arXiv:2602.15763 (2026).
- [55] Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, et al. 2025. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. arXiv preprint arXiv:2508.06471

(2025).

- [56] Lei Zhang, Jin Pan, Jacob Gettig, Steve Oney, and Anhong Guo. 2024. VRCopilot: Authoring 3D Layouts with Generative AI Models in VR. In Proceedings of the 37th Annual ACM Symposium on User Interface Software and Technology. ACM, Pittsburgh PA USA, 1–13. https://doi.org/10.1145/3654777.3676451
- [57] Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, et al. 2025. Siren’s Song in the AI Ocean: A Survey on Hallucination in Large Language Models. Computational Linguistics 51, 4 (2025), 1373–1418.
- [58] Chengbo Zheng, Kangyu Yuan, Bingcan Guo, Reza Hadi Mogavi, Zhenhui Peng, Shuai Ma, and Xiaojuan Ma. 2024. Charting the Future of AI in Project-Based Learning: A Co-Design Exploration with Students. In Proceedings of the CHI Conference on Human Factors in Computing Systems. ACM, Honolulu HI USA, 1–19. https://doi.org/10.1145/3613904.3642807
- [59] Chenfei Zhu, Shao-Kang Hsia, Xiyun Hu, Ziyi Liu, Jingyu Shi, and Karthik Ramani. 2025. agentAR: Creating Augmented Reality Applications with ToolAugmented LLM-based Autonomous Agents. In Proceedings of the 38th Annual ACM Symposium on User Interface Software and Technology. ACM, Busan Republic of Korea, 1–23. https://doi.org/10.1145/3746059.3747676
- [60] Yihao Zhu, Zhoutong Ye, Yichen Yuan, Wenxuan Tang, Chun Yu, and Yuanchun Shi. 2025. AutoPBL: An LLM-powered Platform to Guide and Support Individual Learners Through Self Project-based Learning. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems (CHI ’25). Association for Computing Machinery, New York, NY, USA, 1–26. https://doi.org/10.1145/ 3706598.3714261

### A DETAILED DISCUSSION

This appendix presents the complete discussion of implications and limitations that were summarized in Section 6.

### A.1 Implications for Teacher-Facing GenUI Systems

- A.1.1 Balancing Automation and Pedagogical Control. Our formative study revealed a fundamental tension in educational GenUI systems: teachers want AI assistance to reduce workload, but they cannot compromise on pedagogical accuracy. This finding aligns with prior work highlighting educators’ concerns about AI-generated content misrepresenting scientific concepts [3]. MAIC-UI addresses this tension through its two-stage generation pipeline, which separates content-aligned simulation creation from visual polish. This design ensures that pedagogical substance is validated before aesthetic enhancements are applied.

However, our findings also suggest that effective human-AI collaboration in educational contexts requires more than workflow design—it demands clear communication of AI capabilities and limitations. Participants in our study who understood that the system performed structured content analysis were better able to leverage its strengths while compensating for its weaknesses. Future systems should consider transparent visualization of the extraction and analysis process, helping teachers understand what content has been identified and how it will drive generation.

- A.1.2 The Importance of Rapid Iteration Cycles. A key insight from our evaluation is that iteration speed fundamentally shapes creative

engagement. When modifications require 200–600 seconds, as in baseline systems, teachers report losing their “train of thought” and abandoning refinement attempts. MAIC-UI’s sub-10-second iteration cycles, enabled by Unified Diff-based incremental generation, transform the authoring experience from a batch-oriented process into an interactive conversation.

This finding has broader implications for GenUI design beyond education. Any domain requiring iterative refinement—such as data visualization, dashboard design, or document formatting—may benefit from incremental update mechanisms that preserve context while minimizing latency. The key principle is that creative flow requires feedback loops tighter than human working memory can comfortably sustain; delays exceeding approximately 10 seconds create noticeable discontinuities in cognitive engagement [13].

- A.1.3 Zero-Code Editing as End-User Programming. MAIC-UI’s Click-to-Locate interface represents a novel approach to end-user programming for web content. By allowing teachers to select elements visually and describe changes in natural language, the system bridges the gap between the rendered interface and its underlying code representation. This approach contrasts with traditional developer tools that require understanding of DOM structures and CSS selectors.

Our findings suggest that this design successfully embodies the EUP principle of enabling users to focus on what they want to achieve rather than how to implement it. Teachers could make precise modifications without writing code, yet the system preserved their ability to articulate specific intentions through natural language. Future GenUI systems for non-expert users should consider similar mechanisms for bridging visual and code representations, particularly in domains where users have strong domain expertise but limited programming knowledge.

- A.2 Implications for Procedural Knowledge Learning in STEM Education

- A.2.1 Externalizing Procedural Thinking. Traditional STEM instruction often relies on symbolic representations—equations, formulas, and static diagrams—that require students to mentally simulate dynamic processes. MAIC-UI’s interactive simulations externalize these mental simulations, making procedural knowledge visible and manipulable. Our classroom deployment findings suggest this externalization is particularly beneficial for lower-performing students, who showed larger score gains and more even distribution of learning outcomes.

This pattern aligns with theories of distributed cognition, which posit that cognitive processes can be offloaded onto environmental structures [19]. By making abstract relationships concrete and interactive, MAIC-UI reduces the working memory load required for understanding procedural concepts. Future educational technology design should consider how interactive representations can serve as “cognitive scaffolds” that make invisible thought processes visible and explorable.

- A.2.2 Fostering Learning Agency Through Interactive Exploration. Our in-class deployment revealed that interactive courseware can shift classroom dynamics from teacher-centered presentation to student-centered exploration. Students reported becoming “more

willing to ask questions” and engage actively with content, while teachers described “playing with knowledge points together with students” rather than “explaining concepts to the blackboard.”

This transformation represents a shift from passive reception to active construction of knowledge, consistent with constructivist learning theories [5]. However, our findings also highlight that such shifts depend on the quality of interactive design—poorly designed interactions may distract from learning rather than enhance it. The key appears to be alignment between the interactive affordances and the underlying procedural structure: when manipulations directly correspond to conceptual variables, exploration supports understanding; when they do not, interaction may become mere entertainment.

- A.2.3 Addressing Equity in STEM Learning. Perhaps most significantly, our classroom deployment showed that MAIC-UI’s benefits were not limited to already high-performing students. Lowerperforming students in the pilot class demonstrated larger score gains than their counterparts in control classes, and the variance of score gains was substantially lower, suggesting more even distribution of learning benefits.

This finding suggests that interactive courseware may help address persistent achievement gaps in STEM education. By externalizing procedural knowledge and providing multiple entry points for engagement, such systems may make abstract concepts accessible to students who struggle with traditional symbolic representations. Future research should explore how such tools can be designed to maximize equity benefits, potentially through adaptive scaffolding or personalized interaction pathways.

A.3 Limitations

While our evaluation demonstrates MAIC-UI’s effectiveness, several limitations should be acknowledged.

- A.3.1 System Limitations. MAIC-UI currently focuses on generating single-page interactive simulations rather than comprehensive multi-page courseware. While this scope aligns with our goal of supporting focused procedural knowledge visualization, it limits applicability for topics requiring extended narrative or sequential lesson structures. Future work should explore how the Click-toLocate editing paradigm can scale to multi-page experiences while maintaining rapid iteration cycles.

Additionally,thecurrentimplementation relies on vision-language

models for content extraction, which may occasionally misinterpret specialized notation or complex diagrams. While our structured analysis prompt reduces these errors, domain-specific fine-tuning or hybrid approaches combining OCR with visual understanding may be necessary for optimal performance across diverse STEM disciplines.

A.3.2 Evaluation Limitations. Our lab user study employed proxy instructors (graduate students with teaching experience) rather than practicing K-12 teachers. While this approach enabled controlled comparison, the participant population may differ from our target users in terms of technical comfort, pedagogical training, and authentic classroom pressures. The three-month classroom deployment with actual high school teachers and students helps address

this limitation, but longer-term studies examining sustained use and integration into regular teaching practice are needed.

Furthermore, our classroom deployment was conducted in a single school with a specific student population (physics-chemistrybiology track in a Chinese public high school). The generalizability of our findings to other educational contexts—including different countries, subject areas, or student demographics—requires further investigation. Cultural differences in teaching practices and student expectations may influence how interactive courseware is received and integrated.

- A.3.3 Technical Constraints. The Unified Diff-based incremental generation depends on stable code structure to apply patches correctly. While our fuzzy context matching handles minor drift, substantial structural changes from extensive editing sessions may occasionally require full regeneration. Additionally, the current system requires internet connectivity for AI model access, limiting use in settings with poor connectivity or strict data privacy requirements. Future iterations could explore local model deployment or hybrid approaches that balance capability with accessibility.

B FORMATIVE STUDY METHODOLOGY

- B.1 Participant Recruitment

We recruited 6 participants for the formative study through purposive sampling from local university networks. The recruitment criteria included: (1) senior undergraduate or graduate students from top-tier Chinese universities, (2) at least one formal teaching experience (tutoring, teaching assistantship, or formal instruction), and (3) no prior exposure to MAIC-UI or similar AI-powered courseware generation systems.

Recruitment channels: Participants were recruited through departmental mailing lists, teaching center announcements, and snowball sampling from existing contacts. Potential participants completed a screening survey assessing their teaching experience, disciplinary background, and familiarity with educational technology tools.

###### Participant demographics:

- • P1: Graduate student, Computer Science, 2 years tutoring experience
- • P2: Graduate student, Physics, 1 year teaching assistantship
- • P3: Senior undergraduate, Mathematics, 3 years tutoring experience
- • P4: Graduate student, Chemistry, 1.5 years teaching assistantship
- • P5: Graduate student, Biology, 2 years tutoring experience
- • P6: Senior undergraduate, Engineering, 2 years tutoring experience

All participants provided written informed consent and received a compensation of 200 RMB (approximately $28 USD) for their participation.

### B.2 Interview Protocol

Each participant completed a 1-hour hands-on session with MAICUI’s initial prototype, followed by a semi-structured interview lasting approximately 45 minutes. The interview protocol covered five main areas:

###### 1. Creation Experience (10 minutes)

- • Walk me through your process of creating the interactive courseware.
- • What aspects of the system felt intuitive or confusing?
- • How did you decide what changes to make during the editing process?

###### 2. Perceived Learning Costs (5 minutes)

- • How long do you think it would take to become proficient with this system?
- • What background knowledge would teachers need to use this effectively?
- • How does this compare to learning traditional courseware creation tools?

###### 3. Creative Amplification (10 minutes)

- • How did the system support or constrain your creative ideas?
- • Weretheredesignideasyouwanted to implement but couldn’t?
- • How did the generated output compare to your initial vision?

###### 4. Classroom Integration (10 minutes)

- • How do you envision using this tool in your actual teaching?
- • What concerns would you have about using AI-generated materials in class?
- • How might students respond to interactive versus traditional materials?

###### 5. Procedural Knowledge Visualization (10 minutes)

- • How well did the system handle step-by-step procedures or processes?
- • What types of content do you think would benefit most from interactive visualization?
- • Can you describe a specific concept that would be difficult to teach without interactive elements?

Interviews were audio-recorded with participant consent and transcribed verbatim for thematic analysis.

### B.3 Data Analysis

Two researchers independently coded the interview transcripts using deductive and inductive thematic analysis. We first developed a preliminary coding scheme based on our research questions and then refined it through iterative coding. Inter-rater reliability was calculated on 20% of the transcripts, achieving Cohen’s kappa of 0.82. Discrepancies were resolved through discussion until consensus was reached.

C EXTENDED PARTICIPANT QUOTES

This section presents extended verbatim quotes from the formative study that were abbreviated or omitted from the main text due to space constraints.

### C.1 Knowledge Accuracy Concerns P3 on pedagogical scientificity:

“The main issue lies in how it represents knowledge. When you’re teaching, you can’t just present facts; you need to show the logic, the derivation, the scientific rigor. The system generates content quickly, but it doesn’t always get the scientific concepts quite

right. For example, when I asked it to explain Newton’s laws, it gave a correct statement but missed the nuances that students typically struggle with. As a teacher, I need to verify every piece of generated content, which takes time.”

###### P4 on content verification needs:

“Sometimes the knowledge it produces is simply incorrect. Not often, but even once is too much when you’re teaching. Imagine standing in front of thirty students and presenting something wrong—your credibility is gone. So I would need to check everything carefully, maybe even rewrite portions. The speed is helpful, but only if the quality is there.”

###### P6 on content alignment:

“The generated website didn’t include the content I had specified. I gave it a PDF with specific examples I wanted to use, but the output focused on generic explanations instead. It felt like the system was making assumptions about what was important rather than actually using my materials. If I’m going to use this, it needs to respect my input—I’m the teacher, I know what my students need.”

### C.2 Editing Experience

- P3 on the difficulty of editing:

“Modification and editing aren’t that easy. You describe what you want changed, and sometimes AI pretends to understand but gets it wrong. Then you have to explain again, and again. It becomes this backand-forth where you’re not sure if the problem is your instructions or the system’s understanding. After three or four tries, I started wondering if it would be faster to just do it myself.”

- P5 on iteration requirements:

“In cases like this, you might need to edit three to four times. The first generation gives you something close but not quite right. The second edit gets closer. By the third or fourth iteration, it’s usually usable. But that’s a lot of waiting—each regeneration takes time, and you’re sitting there hoping this time it will be right. For simple text changes, it’s frustrating.”

- P6 on granular control limitations:

“I kept trying to change the website—specifically, I wanted to adjust the layout of a particular section to better match my teaching flow. But in the end, it didn’t follow my instructions. It changed something, but not what I asked for. When it comes to detailed issues, like breaking a procedure into several steps with specific visual hierarchy, it may not handle those modifications well. I felt like I had to accept whatever it gave me.”

C.3 Passive vs. Active Learning

- P4 on traditional teaching limitations:

“Traditional PPT and blackboard teaching—it’s fixed content. You write it, you present it, and students watch. They may find it boring and fail to concentrate, especially for abstract concepts. There’s no exploration, no discovery. With interactive tools, you can generate random parameters, let students experiment. They become more concentrated because each interaction is unique. They get randomized experiences instead of the same example every time.”

###### P2 on active learning benefits:

“Traditional teaching requires feeding knowledge to students’ mouths before they attempt it. You’re essentially saying, ‘Trust me, this is how it works.’ But interactive tools offer students a buffet—they can explore in an immediately actionable way. Everyone is lazy, but if you give them a better opportunity to try, they will be more active. When students manipulate variables themselves and see the results, they own that understanding. It’s not just received knowledge anymore.”

### C.4 Theory-Practice Gap

###### P4 on the knowledge-application disconnect:

“What they learn from textbooks and real-world scenarioshaveagap. Textbookspresent idealized situationsfrictionless surfaces, perfect gases, ideal circuits. But real-world scenarios are messy. If students don’t think actively about how to bridge this gap themselves, they won’t make the connection. The knowledge they learn is ‘dead’—it’s memorized but not truly understood. But when facing more flexible applications in exams or real life, they become confused because they’ve never seen the principles in action.”

P2 on the challenge of abstraction:

“Problems are written elaborately, with complex scenarios and multiple steps, but the underlying knowledgeissimple—somestudentscannot cross this chasm. They get lost in the problem description and can’t see the basic principle underneath. Visualization helps bridge that gap. If they can see the force vectors, see the motion, suddenly the abstract symbols make sense. Without that connection, they’re just memorizing problem-solving templates.”

###### P5 on intuitive understanding:

“Students lack intuitive feelings for concepts when first encountering them. They can recite definitions, but they don’t ‘feel’ what the concept means. For velocity, they can say 𝑣 = 𝑑/𝑡, but do they intuitively understand what changing velocity feels like? Interactive visualizations give them that feeling—they can see it, manipulate it, experience it. That builds the intuition that textbooks alone can’t provide.”

### D FULL QUESTIONNAIRE ITEMS

This section presents the complete questionnaire items used in the lab user study (Section 5.2).

### D.1 Post-Task Questionnaire (RQ2: Usability and Editing Experience)

Participants rated the following items on a 5-point Likert scale (1 = Strongly Disagree, 5 = Strongly Agree):

Learnability and Usability Items I1. Learnability: I could quickly learn to use the system effec-

tively. I2. Time Cost: The time required to create and refine the courseware was acceptable. I3. Editing Controllability: I could precisely control what changes were made to the generated courseware. I4. Usage Preference: I would prefer using this system over traditional courseware creation methods.

###### Perceived Quality Items (MAIC-UI condition only)

- Q1. Layout Intuitiveness: The layout of the generated webpage is intuitive and easy to follow.
- Q2. Attention Attraction: The visual design attracts and maintains learner attention.
- Q3. Concept Accuracy: The webpage accurately presents key concepts and procedural steps.
- Q4. Content Coverage: The webpage covers all key teaching points from the source material.
- Q5. Language Clarity: The instructional language used is clear and appropriate for learners.
- Q6. Concept Intuitiveness: The presentation of key concepts is intuitive and easy to understand.

### D.2 Demographic Questionnaire

- (1) What is your current academic status? (Undergraduate / Master’s student / Doctoral student / Other)
- (2) What is your disciplinary background? (Social Sciences / Computer Science / Basic Sciences / Engineering / Other)
- (3) How many hours of teaching experience do you have? (tutoring, TAship, or formal instruction)
- (4) How would you rate your programming experience? (None / Beginner / Intermediate / Advanced)
- (5) How would you rate your experience with AI tools (e.g., ChatGPT)? (None / Beginner / Intermediate / Advanced)
- (6) Have you created educational courseware before? (Yes / No)

### D.3 Post-Study Interview Guide (Lab Study) Usability and Experience:

- • What aspects of the system were most helpful during courseware creation?
- • What aspects were most frustrating or difficult?
- • How did you decide what changes to request during the editing phase?
- • How did the editing experience compare to your expectations?

###### Perceived Quality:

• How satisfied were you with the final courseware quality?

- • What additional features or capabilities would you want?
- • How would you use this tool in your actual teaching context?

### E DETAILED IMPLEMENTATION

This section provides technical implementation details of the MAICUI system.

### E.1 System Architecture

MAIC-UI follows a client-server architecture with the following components:

Frontend:

- • Framework: React 18 with TypeScript 5.0
- • State Management: Zustand for global state, React Query for server state
- • UI Components: Custom components with Tailwind CSS for styling
- • Code Editor: Monaco Editor for HTML/CSS viewing
- • Preview: Sandboxed iframe with bi-directional message passing

Backend:

- • Framework: Python 3.11 with FastAPI
- • Database: SQLite with SQLAlchemy ORM
- • Task Queue: Celery with Redis for async processing
- • File Storage: Local filesystem with CDN support

### E.2 AI Model Configuration Multi-modal Analysis:

- • Model: GLM-4.6V (Zhipu AI)
- • Parameters: temperature=0.2, max_tokens=4096
- • Fallback: GLM-4.5V

Text Generation:

- • Model: GLM-4.7 (Zhipu AI)
- • Parameters: temperature=0.3, max_tokens=8192
- • Fallback: GLM-4.6

### E.3 Structured Analysis Prompt

The multi-modal analysis uses the following structured prompt template:

ANALYSIS_PROMPT = """ Analyze the provided educational document and extract the

following information in JSON format:

- 1. Main Topics: List 3-5 broad subject areas covered

- 2. Key Concepts: Specific terminology and principles students must master

- 3. Learning Objectives: Measurable outcomes students should achieve

- 4. Prerequisite Knowledge: Foundational concepts required beforehand

- 5. Procedural Concepts: Step-by-step processes suitable for simulation

- - Name of the procedure

- - List of steps

- - Adjustable parameters

- 6. Subject Area: One of [Physics, Chemistry, Biology, Math, Geography, Other]

- 7. Grade Level: One of [Primary, Middle, High, Undergraduate , Graduate]

Focus on identifying content that would benefit from interactive visualization. Be precise and comprehensive.

Response format: Valid JSON only, no markdown formatting. """

### E.4 Two-Stage Generation Pipeline Stage 1 Prompt (Content-Aligned Simulation):

- STAGE1_PROMPT = """ Generate an interactive HTML/JavaScript simulation based on

the following educational content:

Subject: {subject_area} Key Concepts: {key_concepts} Procedural Concepts: {procedural_concepts}

Requirements:

- 1. Left panel: Step-by-step process display with current step highlighting

- 2. Right panel: Interactive controls for adjusting parameters

- 3. Real-time coupling between process and simulation panels

- 4. Scientific accuracy is paramount---verify all formulas and relationships

- 5. Include explanatory tooltips for technical terms

- 6. Use vanilla JavaScript (no external dependencies)

- 7. Responsive layout for tablet devices (min-width: 768px)

Generate complete, valid HTML with embedded CSS and

JavaScript. """

Stage 2 Prompt (Visual Polish):

- STAGE2_PROMPT = """ Apply visual polish to the following HTML simulation:

Current HTML: {stage1_html} Theme: {theme_config}

Enhancements to apply:

- 1. Apply theme colors consistently (primary: {primary}, accent: {accent})

- 2. Improve typography hierarchy

- 3. Add smooth animations for state transitions

- 4. Ensure consistent spacing and alignment

- 5. Validate all HTML structure

- 6. Maintain all interactive functionality

Return complete polished HTML. """

### E.5 Click-to-Locate Implementation

The Click-to-Locate feature is implemented through DOM-aware element tracking:

// Frontend element selection function handleElementClick(event: MouseEvent) { const element = event.target as HTMLElement; const selector = {

xpath: getXPath(element), cssSelector: getCSSSelector(element), elementHtml: element.outerHTML.substring(0, 500), boundingBox: element.getBoundingClientRect()

};

// Send to backend with instruction const editRequest = {

element_selector: selector, instruction: userInstruction, context_html: document.documentElement.outerHTML

};

streamEditRequest(editRequest); }

// Generate XPath for element def getXPath(element: HTMLElement): string {

if (element.id) return `//*[@id="${element.id}"]`; const path = []; while (element.parentElement) {

const siblings = Array.from(element.parentElement. children)

.filter(e => e.tagName === element.tagName); const index = siblings.indexOf(element) + 1; path.unshift(`${element.tagName}[${index}]`); element = element.parentElement;

} return `/${path.join('/')}`;

}

#### E.6 Unified Diff Processing The system uses Unified Diff format for incremental updates: import difflib

def generate_unified_diff(original: str, modified: str) -> str:

"""Generate unified diff between original and modified HTML."""

original_lines = original.splitlines(keepends=True) modified_lines = modified.splitlines(keepends=True)

diff = difflib.unified_diff( original_lines, modified_lines, fromfile='original.html', tofile='modified.html'

)

return ''.join(diff)

def apply_diff(original: str, diff: str) -> str:

"""Apply unified diff to original HTML.""" # Fuzzy matching for context drift lines = original.splitlines() diff_lines = diff.splitlines()

result = [] i = 0 for line in diff_lines:

if line.startswith('---') or line.startswith('+++'): continue

elif line.startswith('@@'): # Parse hunk header continue

elif line.startswith('-'): # Remove line (with fuzzy matching) continue

elif line.startswith('+'): # Add line result.append(line[1:])

else: # Context line result.append(line)

return '\n'.join(result)

### E.7 Performance Optimization Response Time Targets:

• Initial generation: 30-60 seconds

- • Edit iterations: <10 seconds (p50: 6.2s, p95: 8.8s)
- • Element selection: <100ms
- • Preview rendering: <500ms

###### Optimization Strategies:

- (1) Token Reduction: Unified Diff format reduces output tokens by ∼90% vs. full HTML regeneration
- (2) Streaming: Progressive rendering with Server-Sent Events (SSE)
- (3) Caching: Analysis results cached for 24 hours, generation templates cached indefinitely
- (4) Async Processing: Non-blocking I/O for concurrent requests
- (5) Connection Pooling: Keep-alive connections to AI API endpoints

### E.8 Error Handling and Fallbacks

The system implements a graceful degradation strategy:

- (1) Analysis Failure: Return basic metadata extraction; allow manual input
- (2) Stage 1 Failure: Fall back to single-pass generation with reduced validation
- (3) Stage 2 Failure: Apply basic CSS styling to Stage 1 output
- (4) Both Stages Fail: Return emergency template with userfriendly error message
- (5) Edit Failure: Retry with expanded context; fall back to full regeneration after 3 failures

