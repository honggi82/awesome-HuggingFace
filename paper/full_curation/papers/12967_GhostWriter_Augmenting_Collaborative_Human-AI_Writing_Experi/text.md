arXiv:2402.08855v3[cs.HC]4Jul2025

# GhostWriter: Augmenting Collaborative Human-AI Writing Experiences Through Personalization and Agency

Catherine Yeh

catherineyeh@g.harvard.edu Harvard University Cambridge, Massachusetts, USA

Gonzalo Ramos

Microsoft Research Redmond, Washington, USA goramos@microsoft.com

Rachel Ng

Microsoft Redmond, Washington, USA rng@microsoft.com

Andy Huntington

Microsoft London, UK anhuntin@microsoft.com

Richard Banks

Microsoft Research Cambridge, UK rbanks@microsoft.com

[Figure 1]

- Figure 1: Overview of GhostWriter, an AI-powered environment that personalizes the writing process through (a) editable style and context information. (b) Personalized content can be generated using features such as inline LLM prompts. (c) Users can also explicitly teach the system about their style preferences by highlighting likes & dislikes.

## Abstract

LLMs to implicitly learn the user’s intended writing style for seamless personalization, while exposing explicit teaching moments for style refinement and reflection. We study 18 participants who use GhostWriter on two distinct writing tasks, observing that it helps users craft personalized text generations and empowers them by providing multiple ways to control the system’s writing style. Based on this study, we present insights on how specific design choices can promote greater user agency in AI-assisted writing and discuss people’s evolving relationships with such systems. We conclude by offering design recommendations for future work.

Large language models (LLMs) have become ubiquitous in providing different forms of writing assistance to different writers. However, LLM-powered writing systems often fall short in capturing the nuanced personalization and control needed to effectively support users – particularly for those who lack experience with prompt engineering. To address these challenges, we introduce GhostWriter, an AI-enhanced design probe that enables users to exercise enhanced agency and personalization during writing. GhostWriter leverages

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

Conference acronym ’XX, Woodstock, NY © 2018 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-XXXX-X/2018/06 https://doi.org/XXXXXXX.XXXXXXX

## CCS Concepts

• Human-centered computing → Interactive systems and tools; Natural language interfaces; HCI design and evaluation methods; • Computing methodologies → Artificial intelligence.

## Keywords

AI-assistedwriting,large languagemodels, generative AI, co-creation, personalization, design probe

#### ACM Reference Format:

Catherine Yeh, Gonzalo Ramos, Rachel Ng, Andy Huntington, and Richard Banks. 2018. GhostWriter: Augmenting Collaborative Human-AI Writing Experiences Through Personalization and Agency. In Proceedings of Make sure to enter the correct conference title from your rights confirmation email (Conference acronym ’XX). ACM, New York, NY, USA, 17 pages. https://doi. org/XXXXXXX.XXXXXXX

## 1 Introduction

With the rise of Large Language Models (LLMs), there is growing interest in leveraging their text generation capabilities [8, 45] to enhance the process and product of writing. Many emerging systems use LLMs to assist with various writing tasks, from brainstorming and drafting content [36, 54, 83, 89] to summarizing [14, 91, 92] and refining existing text [69, 89].

These systems have immense potential to augment writing experiences, but also introduce persistent challenges. To illustrate, imagine Sarah, a software engineer looking to use AI to help write a short introduction email to her new team at work. When asking ChatGPT for assistance, she might receive the following as output:

Subject: Introduction and Excitement for Joining the Team! Dear Team,

I hope this email finds you all in good spirits. My name is Sarah, and I am thrilled to introduce myself as the newest member of the team. I am joining as a software engineer, and I couldn’t be more excited to embark on this journey with all of you...

This example illustrates two key challenges. First, the output can be too general, reflecting a lack of personalization in the generated text [3, 12, 23, 40, 54, 57, 86, 89]. The LLM likely did not write in Sarah’s voice, making the output seem “average” and as if it came from someone else [63]. Additionally, the LLM may be missing relevant context about Sarah or her situation that may be helpful to know when writing this email (e.g., her knowledge about a topic, what project she is working on with the new team, etc.).

Second, the complexity of articulating a prompt can elicit feelings of limited agency1 when working with LLMs [12, 13, 31, 51, 89, 90]. If Sarah, for example, wants to change the writing style of the generated email, it can be difficult for her to make those changes, as this requires one to understand what needs to change and why [89]. An LLM’s probabilistic nature can also make Sarah feel uncertain about how her edits will influence the model’s behavior [46, 50], as minor prompt alternations can lead to major changes in output [74].

These challenges of personalization and agency in the use of LLMs can limit their benefits when applied to writing and underscore opportunities for more meaningful co-creation with these systems. We argue that understanding and addressing these challenges requires not only technical advances, but also tools that provoke reflection on how users interact with AI during writing [55]. We see design and technology probes [7, 21, 25, 30] – exploratory tools intended to elicit insights into user values, practices, and interactions with technology – as key instruments to meet, understand, and ultimately address these challenges.

1Interpreted as the action of steering a system behavior toward a desired outcome with confidence – distinct from the more low-level notion of control.

In this work, we introduce GhostWriter, a design probe that explores how AI writing tools can support user agency and personalization. Specifically, GhostWriter aims to help us investigate new human-AI interaction ideas around these design goals:

- • DG1: Provide rich personalization options and feedback to align LLM-based writing outcomes with user intents.
- • DG2: Expose and champion user agency in AI-powered writing interfaces.

Embodied as an AI-powered editor, GhostWriter gives people an LLM-augmented writing experience by providing agency in style and context personalization through implicit and explicit means (Figure 1). Having the ability to steer the model toward an intended style can help users like Sarah generate text that is more aligned with their goals and intentions, rather than relying on the LLM’s default “generic” style. Communicating in a personalized style is central to many writing tasks [13, 89], serving as an expression of one’s voice, identity, and rhetorical goals. We believe one way to provide agency over style is through contextual guidance: by shaping the context given to the model, users can more effectively influence the style and content of its output.

Following methods from works such as Hohman et al. [27], we use GhostWriter as a probe to evaluate our design ideas to support a personalized experience that empowers users with the ability to shape LLM outputs and infuse their writing with a desired style while co-editing with AI. In a user study, we examined how 18 participants reacted to and used GhostWriter in two tasks that represent a good cross-section of writing as an activity: professional editing and creative writing.

Our results reveal that GhostWriter helped users exert control over the direction of LLM outputs and offered value in providing them with flexible ways to customize style and context. We observed that different methods of style control were useful at different points in a task, and across different kinds of writing. At the same time, our study surfaced challenges around interpreting the system’s behavior during style personalization and navigating questions about outcome ownership when writing with AI. These findings informed a set of design recommendations for creating AI-infused tools that amplify human intent while preserving ownership.

Our contributions include:

- • GhostWriter, an AI-powered writing experience that supports content personalization through style and context specifications.
- • User study results from using GhostWriter as a probe that indicate the perceived utility and effectiveness of our system in generating personalized text and championing user agency.
- • Design recommendations and directions to help shape the design of future LLM-infused writing systems.

## 2 Related Work 2.1 AI-Assisted Writing

Much work investigates how AI can enhance human writing [12, 14, 19, 23, 28, 36, 40, 79, 84, 88] across various users and tasks [45]. Clark et al. [13] explores how a machine-in-the-loop system can amplify creativity for short story and slogan writing tasks, and

Singh et al. [77] present a multimodal interface for creative writing powered by generative AI. Wordcraft [31, 89] is an LLM-augmented writing editor, offering features such as rephrasing or continuing a text passage, and just-in-time custom controls. Dramatron [54] also uses LLMs for creative text generation, exploring the use of these models to co-write screenplays. However, a common theme in these systems is the lack of context awareness and user agency when generating text, which we aim to address.

System extensions, such as Gmail Smart Compose [10], Grammarly,2 and Wordtune [93], integrate AI features into existing experiences to support writing suggestions and corrections. Other platforms like Notion have introduced document search, text generation, and style analysis features, which capture a vision similar to ours for AI-assisted writing. Our work with GhostWriter, developed in parallel, relates to these design patterns but takes a different path, focusing more on style capture, context definition, and output personalization.

## 2.2 Personalized AI Experiences

- 2.2.1 Explicit vs. Implicit Personalization. Personalization can be achieved through implicit or explicit methods. Many recommendation systems use implicit forms of personalization to model user preferences over time, by looking at user interaction histories [48], assessment performance [4], or mouse click patterns [41]. Some AI systems also incorporate explicit approaches such as having users bookmark [65, 67] or react to content [58] to receive personalized suggestions while retaining user agency.

We take inspiration from and aim to combine both kinds of personalization in having GhostWriter implicitly learn the user’s style as they write, while also providing opportunities for explicit style refinement. Extending the concept of “natural language user profiles” from recommendation systems [56, 68], we explore creating editable style and context profiles to guide the generation of personalized text. While Impressona [5] creates personas with different writing styles to generate stylized feedback, we focus on personalizing the written content itself – a core challenge and need when writing with LLMs [57, 63].

- 2.2.2 Personalization Without Retraining. Our work builds on the paradigm of personalizing downstream user experiences without retraining the underlying ML models [65, 67]. This approach is powerful because it allows for flexible, user-specific customization without compromising the performance or scalability of pretrained models. In writing contexts, recent research explores how LLMs can help generate personalized text without model retraining [37, 73, 95]. Some work proposes incorporating social factor modeling to personalize the writing experience with LLMs [43], while others draw on principles from writing education [47]. LMCanvas [39] transforms the traditional text editor into a canvas-based interface, where malleable “blocks” are used to create a personalized writing environment. We offer a complementary perspective by studying personalization through the lens of writing style and context.

We also draw on ideas from interactive machine teaching [70, 76], where a human teacher communicates information to a machine learner in an iterative process that has been shown to enhance

2https://www.grammarly.com/ai-writing-tools

both the user experience and the building of an efficient learning set [81, 97]. With GhostWriter, users iteratively and interactively construct their target style and context to guide the generation of (personalized) text.

## 2.3 Working with Style and Context

GhostWriter can infer a writing style from a written sample using an LLM – a human-interpretable alternative to neural methods such as style representation learning [66] and activation steering [42]. Another relevant idea from NLP is text style transfer (TST) [20, 29, 32], which aims to preserve the content of generated text while adjusting attributes like tone or voice. While TSTs are limited to preexisting text, our system can apply learned styles to produce new text, similar to Li et al. [47]. A related task is controllable text generation (CTG), where various aspects of generated text can be manipulated, such as context [78] or topic [18]. Our work explores how natural language can enable end-users to “perform” personalized TSTs and controllable text generation with LLMs. We also draw from context-faithful prompting [96] when incorporating the user’s writing style and context.

ChatGPT’s custom instructions feature enables users to specify information that the system should consider when generating output.3 Similarly, we allow users to tweak context and style information through natural language, but our system uses these details to generate personalized writing rather than chat output. While some of these features may overlap, the user experiences (i.e., separated chat vs. in-document interactions) remain distinct, and we provide tangible insights about their perceived value through our user studies.

3 System Overview

We address personalization and agency in the context of AIassisted writing through GhostWriter, an LLM-powered writing environment. We use GhostWriter as a design probe [7, 25, 30] to explore the possibility of using AI to craft personalized outputs through the manipulation of style and context in ways that emphasize user agency.

Design probes in HCI are adapted from cultural probes [21], designed objects that promote participant engagement in the design process [7, 27, 33, 65, 67]. These probes focus on identifying areas of value and improvement, rather than evaluating usability or comparing against existing solutions [65]. As such, GhostWriter helps to elicit feedback about how our design and implementation choices meet the challenges of personalization and agency.

## 3.1 Design Principles

The design, development, and study of GhostWriter was guided by our design goals (Section 1). These goals led us to ideate how we could enhance agency and personalization in an LLM-powered writing experience. We propose that these goals can be met by considering both implicit (i.e., learning as you write) and explicit (i.e., learning from direct user input) forms of feedback, which together can help yield more goal-aligned LLM outputs through the iterative definition of context and writing style. Building on this

3https://openai.com/blog/custom-instructions-for-chatgpt. Announced on July 20, 2023, post our original GhostWriter implementation.

[Figure 2]

### Figure 2: Personalization through refining text. (a) Upon invoking the context menu, the user can choose the rewrite or the “apply” prompt option. (b) The “apply” prompt will apply the inputted text as a prompt to the current text selection. (c) Sample output from using an “apply” prompt. Users can regenerate, delete, or insert the outputted text.

[Figure 3]

### Figure 3: Personalization through generating new text. (a) Upon invoking the slash menu, the user can choose the continue text or inline (“GhostWriter”) prompt option. (b) The inline prompt takes any general prompt as input. (c) Sample output from an inline prompt. Users can regenerate, delete, or insert the outputted text.

strategy and informed by existing literature on AI writing systems, we define the following design principles (DPs):

- 3.1.1 DP1: Leverage Machine Capabilities While Championing Agency. Our work explores opportunities to use LLMs’ text generation and analysis capabilities [8, 54, 89] to extract writing styles that can be used to produce bespoke outputs [DG1]. At the same time, we aim to champion user agency when co-writing with AI [DG2]. This position places technology in the service of human expression and can foster a writing experience that addresses previous concerns about personalization and control [12, 31, 89].

- 3.1.2 DP2: Use Familiar Editor Metaphors. To reduce the cognitive load [80] of navigating a new system, we use existing text editor metaphors in our design [60] (e.g., highlighting text, section blocks). By exploring how AI augmentations blend into familiar writing experiences, we can focus on distilling their effects and project our learnings into grounded, relatable scenarios [DG1].
- 3.1.3 DP3: Blend Into the Writer’s Existing Workflow. Users should not have to deviate from existing flows when using our system. We prioritize simplicity and a non-fragmented user experience, like Notion or Microsoft Loop, which offer straightforward, yet feature-rich writing interfaces [DG1]. In these systems, writing

and most interactions occur in one central editor, reducing interface complexity and unnecessary context-switching [52].

- 3.1.4 DP4: Provide Transparency to Support Reflection and Discovery. GhostWriter strives to offer transparency about its internal state. Users should know and be able to inspect what information the LLM has access to, which can be achieved in an easy-tounderstand way through natural language [56, 68]. In addition to fostering reflection and experimentation with alternative styles and contexts [51], this transparency can help users understand what they can do with the system and how to fix problems when they arise – a key aspect of championing agency [2] [DG2].

3.2 Interface Design

We outline the key components of GhostWriter’s interface, as shaped by our DPs. Our design and interaction strategies are broadly applicable to LLMs, and present an alternative to emerging experiences that rely on linear, turn-based chat interfaces.

- 3.2.1 Main Editor. GhostWriter’s central view is the main editor (Figure 1), which mirrors existing text editors [DP2] and provides a space for users to author documents. One way users can teach GhostWriter about their target writing style is simply by writing.

[Figure 4]

### Figure 4: Teaching style through manual edits. (a) Users can view and edit the full description of their current style by pressing the Style button. (b) The style description (excerpt shown) is editable like a normal text file.

After each n (default: 100) new characters, the system analyzes the document to extract its style [DP1]. Users can apply LLMpowered features to refine existing text or generate new text given the current writing style and context [DP1] (Table 1). All features are embedded inline in the editor to support a non-fragmented, familiar writing experience [DP2, DP3].

To refine written text, users can invoke the context menu by selecting a portion of the document. Then, they can rewrite or use a contextual “apply” prompt to refine the selected text (Figure 2a). Both operations apply the current style to generate personalized content (Figure 2b). To generate text, users can invoke the slash menu by typing a forward slash (“/”) anywhere in the document

- (Figure 3a). Then, they can continue the text from the current point or generate new text by invoking the inline “GhostWriter” prompt. These operations use both the current learned style and context to generate content (Figure 3b). For all context and slash menu features, users can regenerate, delete, or insert LLM output [DP1] (Figure 2c, Figure 3c).

The user can explicitly teach GhostWriter about their target writing style by indicating likes and dislikes [DP1]. They do so by highlighting a portion of text [DP2], which invokes the context menu (Figure 5a). Then, they can like or dislike the selection and optionally write why to help the system learn about their style (Figure 5b). Allowing users to explicitly nudge GhostWriter toward their preferences builds on work showing the value of fine-grained feedback for personalization [58, 67]. We also draw from active reading practices, where annotation and highlighting are key strategies in engaging with and reflecting on the text [75].

- 3.2.2 Left Panel. In the left panel, users can view their document list [DP2, DP3] and explore the system’s current style and context [DP4] (Figure 1). The style summary is automatically refreshed when the system’s writing style updates. Clicking the refresh icon will force a style update based on the current document [DP1]. Users can disable automatic style updates by pressing the lock icon.

GhostWriter avoids a cold start by providing a default (generic) writing style that evolves based on user input and interaction. When extracting style from a sample, the system is prompted to analyze five style characteristics: tone, voice, word choice, sentence structure,

### Table 1: GhostWriter’s LLM-powered features for personalized text generation.

Feature Description Rewrite (Context menu)

Rewrite text selection to match the system’s learned style

Apply LLM prompt to text selection to update its content using the system’s learned style

“Apply” prompt (Context menu)

Generate new text to continue the current document using the system’s learned style and context

Continue (Slash menu)

Generate new content based on an LLM prompt using the system’s learned style and context

Inline prompt (Slash menu)

[Figure 5]

Figure 5: Teaching style through likes and dislikes. (a) Users can like (thumbs up) or dislike (thumbs down) any text selection through the invoked context menu. (b) Once the corresponding icon is selected, the user can optionally provide feedback as to why they like or dislike the highlighted text.

and paragraph structure. We chose these dimensions to establish a relatable language for communicating style, based on informal pilot studies and work such as Reinhart et al. [71].

Users can edit the system’s style directly [DP1] by pressing “Style” in the left sidebar (Figure 4a). This opens the full style description in the main editor [DP3, DP4] (Figure 4b). In allowing users to inspect and modify the system’s style, we aim to support reflection and experimentation during writing [DP4]. To (optionally) provide additional context for grounding text generations, users

[Figure 6]

- Figure 6: (a) Users can view all their past writing styles on the Style History page. (b) The blue boxes on the left each display a writing style and (c) the gray boxes on the right display a comparison of each pair of adjacent writing styles. (d) Each comparison also includes the difference rating between adjacent styles. (e) Users can revert to prior writing styles as well.

can edit the “Context” page in the main editor [DP1, DP3, DP4] (Figure 1). There are no constraints on what can be included as “context,” and the user has full agency over this page [DP1].

- 3.2.3 Style Toolbar. The style toolbar, located above the main editor, exposes features to customize the user’s experience [DP1] and inspect the system’s style knowledge [DP4] (Figure 1). Users can toggle the “Track Style Of This Document” flag to turn on/off automatic style updates for the current document (vs. the global style lock in the left panel). They can also turn “Feedback Mode” on/off, which shows or hides all highlights in the current document.

Users also have the option to examine current and past writing styles through the “Style History” page (Figure 6a) [DP3], where more recent styles are displayed at the top. Styles are displayed in blue boxes on the left (Figure 6b), while comparisons between each pair of adjacent styles are shown in gray boxes on the right (Figure 6c). These LLM-generated comparisons [DP1] offer deeper insight into how the system-learned style changed over time by providing a “difference rating” (Figure 6d) that quantifies the difference between adjacent styles from 0 to 10 (0: identical, 10: entirely different) [DP4]. Users can revert to a previous style by clicking the revert icon in any style box (Figure 6e) [DP1].

Similarly, users can view their collection of highlighted “Likes & Dislikes” (Figure 7a) [DP3]. At the top of this page, there are LLMgenerated summaries of the user’s (qualified) liked and disliked text (Figure 7b) [DP1]. These summaries can encourage additional reflection [DP4] and help users assess whether GhostWriter correctly

understands their feedback. Users can add additional likes and dislikes (Figure 7c) or toggle/delete highlights in this view (Figure 7d) [DP1]. Only active highlights are used to compute the like/dislike summaries, which in turn guide text generations.

## 3.3 Implementation

GhostWriter is a React web application connected to a Python backend through RESTful endpoints. The main editor interface is built using Tiptap, a headless editor framework.

3.3.1 Backend AI Services. We orchestrated all backend LLM operations with LangChain, most of which use GPT-4. Style updates are computed using GPT-3.5-Turbo,4 which provided comparable, but faster results during experimentation.

We iteratively crafted and refined the prompts used by GhostWriter (included as Supplementary Material). Given our goal of producing a design probe, we did not optimize them to achieve perfect outcomes. Instead, we used GPT-4 to evaluate the generated output as a sanity check to supplement qualitative inspections [94]. The model generally assigned high confidence scores (e.g., ≥ 8 out of 10) to LLM-generated style descriptions and comparisons (where 10 = a perfectly accurate style description of the writing sample, or comparison between two styles). While adequate for a design probe, practitioners who deploy similar systems for production should rigorously validate LLM output quality [38].

While most of our prompts are straightforward, computing style updates requires additional logic (Figure 8). We first pass the current

4Both were among the leading models from OpenAI at the time of implementation.

[Figure 7]

### Figure 7: (a) Users can view all their likes and dislikes on the Likes & Dislikes page. (b) A system-generated summary of the user’s likes and dislikes is displayed at the top of the page. (c) The user can also manually add additional likes and dislikes. (d) Upon hovering on a like or dislike card, users have the option of toggling its active state or deleting it from their collection.

[Figure 8]

### Figure 8: How style updates are computed by GhostWriter. (a) The current document, likes & dislikes, and style description are passed as inputs from the frontend. (b) In the backend, we then ask the LLM to generate a new style description given this information. The LLM also generates a style comparison between the old and new styles, and computes a difference rating. (c) If the difference rating is greater than some threshold (e.g., 3 out of 10), the new style and comparison are passed as outputs back to the frontend. Otherwise, the user will be informed that there is no style update needed.

style description, user’s likes & dislikes summaries, and full text document to the backend. The LLM produces a new style description based on the provided text and likes & dislikes, formatted as HTML

- (Figure 4), along with a short style summary (Figure 1). Next, we ask the LLM to write a comparison of the new and old styles, including a difference rating to determine whether a style update is necessary (Figure 6). If the rating is greater than a threshold (currently set at 3 out of 10 based on pilot studies), we return the new style and comparison as outputs to the frontend. Otherwise, the system outputs a “no style update needed” message.

## 3.4 Design Limitations

Our current implementation only stores one global style and context, and simplifies style to five dimensions. This may not capture all aspects of the desired writing style. However, users are not limited to this (initial) setup, as they can directly edit the system’s style and add explicit likes & dislikes to better guide text generations.

## 4 User Study

We conducted a two-part study with 18 participants, using GhostWriter as a probe to examine how our designed experience shapes

user behaviors with and reactions to AI in editing and creative writing tasks. These two tasks were informed by background research and discussions with collaborators about common writing scenarios (1) where personalization is important and (2) that are relevant to a broad range of stakeholders [13, 54, 69, 77, 89]. We also chose these tasks to allow users to (potentially) explore different features in GhostWriter.

Our user study aims to address the following research questions:

- • RQ1: How does having the ability to affect style and context help users achieve their writing goals?
- • RQ2: How do users react to the different ways to craft personalized content in GhostWriter?
- • RQ3: What new challenges emerge for users from interacting with GhostWriter?
- • RQ4: How do users perceive the relationship between writers and AI?

## 4.1 Participants

We recruited 18 participants (16 women, 2 men) via mailing list at a large technology company. Our selection process was self-selective and targeted individuals whose professions involve writing in significant ways. This led to 7 content designers, 6 UX researchers, 4 communications managers, and 1 executive assistant (Table 2). All participants were based in the United States.

## 4.2 Procedure

Participants interacted with GhostWriter during two 1-hour sessions, each focused on a distinct writing task. Session 1 was designed to familiarize participants with the system, and we erred on the side of building expertise with a shorter editing task, before diving deeper into a longer creative writing task in Session 2. We understand the possibility of learning effects between sessions, but

- as a design probe looking into mostly qualitative insights, we think the benefits of a gradual introduction to GhostWriter outweigh any negative ordering effects.

Sessions took place online using Microsoft Teams where we asked participants to think aloud. We also recorded their screens and logged system events. Participants were compensated after study completion with a $100 Amazon gift card. This study was approved by our company’s Institutional Review Board.

- 4.2.1 Pre-study Survey. Prior to Session 1, participants completed a survey about their experience with generative AI (GenAI) systems. 15 of the 18 participants reported working with GenAI systems for less than one year (one had never interacted with them). Five participants reported interacting with such systems multiple times a week, another five reported multiple interactions per month, and the remainder reported less frequent interactions.

Six participantsindicatedthattheygenerally write multi-sentence prompts, with the rest writing simpler prompts. 10 out of 18 participants noted that they do not attach contextual documents when prompting GenAI systems.

- 4.2.2 Tutorial & Practice. Participants started Session 1 by watching a 5-minute tutorial. Afterwards, participants had 10-15 minutes to familiarize themselves with GhostWriter.

### Table 2: Overview of user study participants.

#### ID Role Gender

- P01 Senior content designer Female
- P02 Senior content designer Male
- P03 Content designer Female
- P04 UX researcher II Female
- P05 Content designer II Female
- P06 Executive assistant Female
- P07 Senior communications manager Female
- P08 Senior UX researcher Female
- P09 Senior UX researcher Female
- P10 UX researcher Female
- P11 Senior communications manager Female
- P12 Senior content designer Female
- P13 Senior communications manager Female
- P14 Senior UX researcher Male
- P15 Senior content designer Female
- P16 UX researcher II Female
- P17 Senior communications manager Female
- P18 Senior content designer Female

- 4.2.3 Task 1: Professional Editing. The main task in Session 1 was to edit and refine a document to fit a particular writing style based on these instructions:

Imagine you are a freelance content writer working with a new client. The client runs a travel blog, EpicWanderlust, and wants you to help write a new post about Seattle. You are given a draft with some basic ideas, but the client feels it does not fit the “style” of their blog. Here is where you come in: Your job is to polish the draft (with GhostWriter’s help!) so that it feels more cohesive and consistent with the other posts on EpicWanderlust.

One additional request: the client wants you to help them reach younger audiences by tweaking the post’s writing style.

We included an example blog post from EpicWanderlust to help participants get started and extract the desired writing style, along with some default context on Seattle. Participants were given ∼30 minutes to complete this task.

- 4.2.4 Task 2: Creative Writing. The main task in Session 2 was to generate a document following a writing style chosen by the participant. We asked participants to bring a short writing sample containing a style they wanted to emulate (written by them or someone else) and then write a story based on one of these prompts:

- Story 1: “Write a short story about an intern at a tech company having an adventure on an alien planet.”
- Story 2: “Write a short story about a group of friends who find themselves trapped in a haunted mansion.”
- Story 3: “Write a short story about a musician who finds a magical device that can control the weather.”

Participants chose which prompt to work on, and in each case, we provided some default context about a possible setting and characters to build upon. Participants were given ∼45 minutes to complete this more open-ended, creative task.

- 4.2.5 Post-task Survey. After each session, participants completed the same, ∼10 minute post-task survey. The survey started with Likert scale questions (where 1: strongly disagree - 5: strongly agree) about user satisfaction, ease of use, perceived agency, and output ownership, as well as system trust and understanding. Then, participants filled out open-ended questions about their overall experience with GhostWriter, likes, and dislikes. After completing both sessions, we asked participants how they perceived their relationship with the AI system. All survey questions are included as Supplementary Material.

## 4.3 Data Analysis

Our approach of surfacing insights through participant task interactions and post-task feedback is well-aligned with established design probe methodology, which prioritizes collecting reflections and reactions from users [7, 27].

We adopted a mixed-methods approach for data analysis. First, we quantitatively analyzed our event logs to examine how participants interacted with GhostWriter, focusing on event counts (e.g., likes, style updates) and common interaction sequences. We then performed a thematic analysis by coding recurring patterns in participants’ inline and “apply” prompts. Next, we computed metrics based on post-task Likert scale responses. Finally, we synthesized themes from the qualitative survey data and think-aloud interview transcripts to collect participant impressions about their experience with our AI-powered writing assistant.

Thematic analyses were performed inductively, with two authors independently coding a subset of the data to establish initial themes. A shared codebook was developed through iterative discussion and refinement, which was used to code the remaining data.

## 5 Results

All participants (P01-P18) successfully completed both writing tasks. The Editing task took an average of 28.7 minutes to complete, while the Creative Writing task took 46.7 minutes.

Overall Impressions. Participants gave positive responses about their experience with GhostWriter (Figure 9). We visualize aggregated responses from both tasks, since our goal was to collect impressions about GhostWriter as a general writing tool.5 The statements, “I trust the information generated by the system” (mean: 3.61), along with “I have a strong sense of ownership of the creative outcome” (mean: 3.44), yielded the most variation and lowest mean ratings. In each case, however, 61.1% and 52.7% of responses were still “agree” or “strongly agree,” respectively.

“I felt in control of the experience” (mean: 4.00) and “The system is learning from me” (mean: 4.17) received the highest mean scores for both tasks, reflecting positively on our goals of personalization and agency. 83.3% of responses were “agree” or “strongly agree” with the former statement, and 86.0% showed agreement with the latter. Ratings indicated that most participants were able to communicate their intended style (mean: 3.75), were satisfied with GhostWriter’s customized text generations (mean: 3.94), and understood the system’s behavior (mean: 3.92).

- 5We also did not observe significant differences in scores after each task.

## 5.1 RQ1: How does having the ability to affect style and context help users achieve their writing goals?

Participants iteratively crafted their desired style through implicit or explicit methods – showing how agency over style manifested in different ways throughout GhostWriter. There was an average of 6.71 style updates per task: 3.39 were automatic, 1.87 were direct edits, and 1.45 were manually requested. During each task, users also added an average of 3.77 likes and 2.81 dislikes. This suggests participants were not just passive recipients of stylistic suggestions, but engaged in intentional, often mixed-mode refinement to guide GhostWriter toward their individual goals. On average, participants viewed the style history page 0.97 times per task, and the likes & dislikes page 1.97 times. In contrast, they viewed the main style page an average of 4.99 times, showing a preference for in-the-moment control over retrospective style comparisons.

- 5.1.1 Editing vs. Creative Writing. Participants viewed the context page on average 0.86 times during Editing, compared to 4.41 times in Creative Writing. This shift – illustrating how user agency manifests through adaptive context-seeking behavior – is reasonable given the tasks’ differing natures; during Creative Writing, participants frequently referenced context information for composing text, while during Editing, they focused on refining the text.

On average, participants used the rewrite feature 3.52 times per task. The continue feature was used less in Editing (1.14 times) compared to Creative Writing (2.59 times). Similarly, for Editing, participants used an average of 1.14 inline prompts, while for Creative Writing, they used an average of 5.18. The latter’s focus on content creation likely explains the higher usage of features that support idea generation and progression. However, participants used an average of 4.57 “apply” prompts during Editing, compared to 3.47 in Creative Writing – reflecting a preference for more targeted forms of style control during the former task.

- 5.1.2 Intents By Prompt Type. To better understand usage differences between our two LLM prompt features and the intents behind them, we analyzed the 97 inline and 134 “apply” prompts composed by participants across both tasks. Overall, we found that participants used inline prompts for brainstorming and creative agency, and “apply” prompts for precise style refinement.

For inline prompts, the most common intent was adding more content (𝑛 = 42; 43%), e.g., “Elaborate on how John comes across PlayStation while exploring Starfield” or “Add a paragraph with additional places to visit.” Participants frequently wanted to generate full drafts of documents (𝑛 = 30; 31%), e.g., “Write a horror story in the style of Edgar Allan Poe. The story should have a strong plot with a surprise twist” or “Write a blog article with an introduction, 4 paragraphs with catchy titles and a summary to convince my friend to visit Seattle.” Ten inline prompts (10%) asked for a document introduction, and nine (9%) asked for a conclusion.

With “apply” prompts, users often wanted to expand the selected text (𝑛 = 53; 40%), e.g., “Add more details about his fellow researchers; reference a second character named Bob.” 43 prompts (32%) aimed to rewrite the selected text, e.g., by changing the perspective (“Change to first person”), tone (“Make this more positive

[Figure 9]

- Figure 9: Aggregated participant responses to survey questions regarding system usability and satisfaction after completing each user study task with GhostWriter. Each question was scored on a 5-point Likert scale (1: strongly disagree - 5: strongly agree). Each horizontal bar contains 36 responses (2 per participant), with average values displayed on the right.

and enthusiastic”), or audience (“Rewrite for younger readers”). Participants also used “apply” prompts to condense text (𝑛 = 6; 4%), e.g., “Shorten this poem by half”, or request specific formatting (𝑛 = 7;

- 5%), e.g., “Bullet form this paragraph.” In 3 cases (2%), users wanted to add transitions between paragraphs. 3 prompts (2%) asked for suggestions or critique, e.g., “Is this a well-written sentence?”

- 5.1.3 Interaction Patterns Over Time. Writing tasks are not monolithic, and people’s behaviors (like any story) can shift over the course of a task. As such, we looked at differences in interaction patterns between the first and second half of each task.

Participants requested manual style updates more frequently in the first half (1.19 times) vs. the second half of tasks (0.29 times; 𝑊 = 10.0,𝑝 < .004).6 Similarly, visits to the main style page dropped from an average of 3.32 times in the first half of each task to 1.68 in the second half (𝑊 = 72.5,𝑝 < .004). The context page followed a similar trend, with 2.10 visits in the first half compared to 0.71 in the second half (𝑊 = 24.0,𝑝 < .004). Conversely, visits to the likes & dislikes page increased from 0.58 to 1.39 between task halves (𝑊 = 65.0,𝑝 = .01). These observed behaviors reflect how during early-phase exploration and personalization, users may employ various techniques to shape their desired writing style and context. However, as writing progresses, they prefer more lightweight methods for finetuning the system’s style knowledge.

- 5.1.4 Responding to System Feedback. We plotted telemetry timelines to gain additional insights into participant interaction patterns (Figure 10). Across both writing tasks, users viewed the main style

- 6We report Wilcoxon rank-sum test statistics and use a Bonferroni correction of 𝛼 = 013.05 = .004, as we perform one test per 𝑛 = 13 log event types.

page 61 times (36% of the time) directly after a manual or automatic style update. After editing the description on the style page, participants returned to the home document 54 times (71%). Similarly, visiting the context page was often followed by navigating to the home (58 times = 55%) or style page (25 times = 24%). These patterns suggest users’ desire to examine or reflect on the system’s state when it changes, and eagerness to test GhostWriter’s knowledge after revising style and context. Users also frequently added consecutive likes and dislikes to steer style (160 times = 70%). However, after adding a highlight, participants only viewed the likes & dislikes page 72 times (17%), indicating a potentially reduced need or motivation for reflection in these cases.

5.1.5 Strategies for Personalization. Our analysis highlights differences in how participants experienced personalization and agency when interacting with GhostWriter. For example, P07 and P12 used the rewrite feature more extensively compared to P02 and P17 while writing (Figure 10). After rewriting text, participants also took different next actions, using a “apply” prompt 21 times (18%; e.g., P12, P17), adding a like 19 times (17%; e.g., P02, P07) or dislike 13 times (11%), and performing another rewrite 17 times (15%; e.g., P07, P12). Some participants (e.g., P07, P12) liked using two inline or “apply” prompts in a row to generate personalized content; the former occurred 28 times (25%) and the latter 31 times (23%).

Users added likes and dislikes at different points, highlighting the value of allowing flexible moments for user agency during writing. P12 mainly added them at the start of the task, while P02 and P07 did so at both the beginning and end, and P17 spread their likes and dislikes throughout the session.

[Figure 10]

### Figure 10: Example participant telemetry timelines from the (a) editing and (b) generation tasks. Key events are plotted in different horizontal lanes to avoid visual overlap, and time is plotted along the x-axis. Notable interaction patterns are highlighted with the gray dashed boxes.

## 5.2 RQ2: How do users react to the different ways to craft personalized content in GhostWriter?

Several participants described the potential of our system to augment human creativity and productivity through content personalization (𝑛 = 8): “[I] would definitely use this kind of tool to help me be more productive in a work setting” (P17). P02 added, “I think [GhostWriter] offers a good starting point for any kind of writing,” and P07 loved “seeing how creative it can be and how it boosted my own imagination.”

- 5.2.1 Ease of Use and Style Learning. Overall, participants viewed GhostWriter as easy to use and intuitive (𝑛 = 15). P04 described our system as “fun and robust,” and P14 reported, “This was an incredibly positive experience. It was easy to build and iterate on the story. The tool has an easy learning curve, and a simple interface.”

Participants were particularly excited about the ease of style learning (𝑛 = 18). P06 said, “I was impressed how the system understood and incorporated my style, both from the sample text and [my own] writing.” Similarly, P09 appreciated GhostWriter’s “responsiveness to the style guide” and P18 was “pleasantly surprised at how much content I was able to generate in a style I preferred.” P12 mentioned “[GhostWriter was] helpful for real time feedback on [writing] style” as well.

- 5.2.2 Multiple Personalization Paths. Consistent with the results from our timeline analysis (Section 5.1), participants enjoyed having different ways to interact with style and provide feedback (𝑛 = 9), valuing the agency to select between methods “that are optimal for different contexts and objectives” (P09). 7 participants particularly appreciated the likes & dislikes feature, which was perceived as “easy to use and effective” for tailoring LLM outputs (P08). Users also frequently filled out the optional feedback field, i.e., specifying why they highlighted a portion of text.

The context page (𝑛 = 7) and “apply” prompt (𝑛 = 4) were other common favorite features. P08 said the former “helped organize my thinking,” and P10 thought that “context adds value into the prompt.” Participants like P07 were “fascinated by how [GhostWriter] combines the sample, style, and context to find the right words.”

- 5.3 RQ3: What new challenges emerge for users from interacting with GhostWriter?

Our study with GhostWriter revealed challenges around designing for personalization and agency that may be relevant to other AIenhanced writing systems.

- 5.3.1 Confusion Over Style and Context Interactions. Many participants explored personalization through editing both style and context information. However, some did not fully grasp the role of style vs. context (𝑛 = 4), which can limit the effectiveness of these personalization controls: “[I’m] trying to understand how style impacted context when they were conflicting in tone” (P15), or “[I’m not] sure if I should put [information] in the context or style” (P04). Others like P16 hypothesized that “style has more influence than context” in steering text generations.

Participants also expressed confusion about how GhostWriter behaved after learning a new style. To preserve agency, our system

never rewrites documents without explicit user action. However, P14 thought that after a style update, GhostWriter would automatically ensure “[all] the text gets updated” as well. P03 added, “Refreshing the style [and] applying it was not very easy to wrap my mind around.” The impact of explicitly refreshing style was also unclear to some: “If I hit refresh, is it going to extract the style of this [document] or apply the style I pasted?” (P15).

- 5.3.2 Need for Contextual Awareness. Another issue was GhostWriter’s occasional lack of contextual awareness (𝑛 = 5) when generating content in the middle of a document or building on existing text (e.g., “Explain how Riley dealt with the negative impact on his internship” or “Finish the story with the characters running outside”). P02 noted how GhostWriter “was unable to continue writing content from the previous sentence. It just wrote more [unrelated] text.” Similarly, P08 hoped inline generations could start “at a midpoint [vs.] always having to restart.” This is a limitation of our current implementation, but it highlights a key consideration for designing effective personalized writing experiences.
- 5.3.3 Additional Style and Personalization Controls. Participants wanted GhostWriter to support a richer personalization infrastructure, e.g., by storing multiple styles and offering preset style templates (𝑛 = 8). Particularly for users working on different writing tasks and crafting diverse documents, it could be helpful to “apply context/style to individual documents” (P05) or “allow different styles for different parts of the document” (P08). P10 suggested having a style “library” to facilitate applying different styles, similar to document templates in Microsoft Word or Google Docs.

Six participants requested alternative forms of style expression. Some mentioned that GhostWriter’s open-ended way of specifying style through natural language could be a drawback: “it was difficult to articulate a style that I wanted to replicate” (P09). P04 wanted to “make style more structured,” while P08 and P10 proposed having “suggestions on styles (drop-downs of various options to spark individual creativity).” P15 suggested “quantifying” style using “variables that can be defined or fed [like] audience or venue.” Five participants wanted to validate style accuracy as well. As P15 expressed, it may be helpful to include “some kind of a score—how close are you with the style intended.”

Participants also wanted more fine-grained options for asserting agency over LLM outputs (𝑛 = 11). P06 said, “I wish I could’ve accepted only parts of the text it generated [and] provided feedback so it could learn what I liked and didn’t.”

- 5.4 RQ4: How do users perceive the relationship between writers and AI?

After both sessions, we asked participants how they view the role of AI in a system like GhostWriter. These perspectives appear to be closely related to how users engaged with our probe while writing.

Eight participants saw AI as a tool due to observing GhostWriter’s reliance on human prompts and the lack of collaboration in generating outputs: “It’s very powerful in generating well-written text. But it [needed] instructions to do that” (P11). P08 added, “it would become more of a collaborator” if they “spent more time adding my own text and getting to the point where we were working together.” P16 explained, “Collaborator feels too strong. [The system] feels more

like a sounding board. But I don’t think it actually understands my ideas,” emphasizing the human-AI communication gap and a more utilitarian relationship where the user maintains creative agency.

Conversely, 6 participants, who iterated with the system or explored multiple directions, framed it as a collaborator: “It helps you get started and then you can react to it” (P12). Similarly, P07 said systems like GhostWriter could help generate ideas and “boost creative thought if you have writer’s block.” P15 saw AI as a “second [pair] of eyes, like [how] you run presentations by your colleague” but would view it as more of a “potential collaborator” if there was more back and forth between writer and system – suggesting that dialogue is a key factor in perceived co-creation.

Four participants regarded AI as both a tool and collaborator, and 3 thought AI could take an advisor role as well, e.g., if it gave “feedback or ideas [to] improve” like pointing out “plot holes or inconsistencies’’ (P04). This aligns with participants’ use of GhostWriter to ask for writing critiques and suggestions (Section 5.1).

## 6 Design Recommendations & Discussion

We presentdesignrecommendations for creating similar AI-supported writing experiences, shaped by our findings and overarching themes of personalization (Section 6.1) and agency (Section 6.2). We also share observations about how these experiences can better support reflection (Section 6.3), and insights on how people view the role of AI and ownership in the context of writing (Section 6.4).

## 6.1 Designing for Personalization on People’s Terms

- 6.1.1 Providing multiple paths to personalization is important. One of GhostWriter’s perceived strengths was supporting different ways to personalize a desired target style (Section 5.2). As users differ in their writing workflows and preferences, it makes sense to offer agency in how they can interact with AI writing systems by including both implicit and explicit paths [45].

Our work surfaces how this flexibility is especially beneficial because one’s preferred mode of controlling style may change throughout the writing process. For example, manual style and context edits generally decreased throughout each task, suggesting these forms of style tweaking may be more useful earlier in the writing process, while users are exploring and defining their creative direction (Section 5.1). On the other hand, users highlighted more likes & dislikes as writing progressed, pointing to the importance of temporal personalization – adapting support based on writing phase – in future AI-infused systems.

- 6.1.2 Explicit teaching moments can be worth the effort. Users regularly engaged with the likes & dislikes feature (Figure 10), which was unexpected, as annotations require extra effort during writing. Several noted that this was their favorite feature, illustrating the value of offering explicit feedback opportunities in AI-augmented systems. Participants even requested additional opportunities to provide feedback, i.e., “directly on the text [GhostWriter] generated” (P08) and “at all input levels” (P06). This finding aligns with the guideline “Encourage granular feedback” from Amershi et al. [2] and principles for leveraging smaller “units of information” [67], suggesting that such mechanisms are key toward the goal of personalization. Moreover, our observations indicate that explicit feedback

- empowers users to assert agency in identifying and underscoring personal style dimensions that the system’s extraction process might miss (e.g., audience or formatting).
- 6.1.3 Substance and style are important. So is format. Although our system analyzes sentence and paragraph structure when updating style, it currently does not incorporate other aspects of formatting. However, many participants viewed document formatting as an integral part of style, wishing to use GhostWriter’s features to personalize and affect the format of generated text (Section 5.1).

Formatting was also used as a non-literal part of participants’ (teaching) language when explicitly editing the system’s style (Figure 4b), e.g., bolding words to underscore their importance. This type of weight specification is consistent with observations by Ng et al. [59] on teaching languages and presents a promising avenue for giving users richer expressive control over personalizing style. We see connections to work such as Textoshop [53], which use drawing software-inspired interactions to stylize writing.

- 6.1.4 Having one style is good. Having many is better. Several participants thought GhostWriter could be even more powerful if it allowed them to define and opportunistically select different styles and contexts, rather than having one global profile for all documents (Section 5.3). Going from having one style and context to many is a trend in the emerging capabilities of independent agents like OpenAI’s custom GPTs.7 Having the ability to invite these different styles and contexts into the same document is the natural next step to satisfy users, enabling AI-powered systems to support and personalize a wider range of writing tasks [45, 61].

## 6.2 Designing for Layered and Contextual Agency

Preserving user agency was top of mind when creating GhostWriter. However, we observed some trade-offs when designing for agency, which warrant further investigation.

- 6.2.1 Natural language as an interface can enhance agency. It can also add unwanted effort. We initially hypothesized that open-ended style expression would provide users with empowering flexibility. However, some participants found it challenging to articulate style through free-form natural language, and desired a strongly structured format such as dropdown selectors—even at the cost of reduced agency (Section 5.3). From our results, we see room to investigate different style specification languages, which may depend on a user’s role and background, and the particular writing scenario. One possibility is to support mixed-initiative style configuration – where natural language can be supplemented with visual scaffolds, example-based style selection, or other interactive controls – to reduce cognitive effort without compromising expressivity [13, 77].
- 6.2.2 Consider expectations regarding system behavior after learning a new style. Our decision to prevent automatic document rewrites after style changes contradicted some participants’ expectations (Section 5.3), underscoring an agency / transparency vs. responsiveness trade-off. On one hand, instantly refreshing documents

7https://openai.com/blog/introducing-gpts

after style updates could reduce users’ perceived sense of control and awareness when using GhostWriter, decreasing system trust [72]. On the other hand, people want immediacy in their actions and to avoid the repetitive steps of style change and application. We advocate for designs that balance agency and awareness during AI-assisted writing, e.g., by previewing changes to users and working to align system behavior to their mental models.

- 6.2.3 Agency is desired at different levels. Our work focuses on providing agency in style and context definition. After experiencing this agency, participants requested similar control over refining LLM generated outputs (Section 5.3), which is consistent with Gmeiner and Yildirim [24], Ippolito et al. [31], Yuan et al. [89]. Our observations and the intrinsic, iterative nature of writing underscore the importance of providing users with tools to confidently steer LLM outputs at multiple levels of text generation and style crafting: draft, intermediate, and final [17, 22, 24, 45]. Users may also require different kinds of agency across different tasks (e.g., fine-grained control for editing vs. high-level ideation support for creative writing – Section 5.1).

## 6.3 Supporting Reflection During AI-Mediated Writing

To create opportunities for reflection (see DP4, Section 3), we included features such as the main style page, likes & dislikes page, and style history page where users can explore learned styles. The style page was frequently viewed by participants (Figure 10), who enjoyed examining its content and seeing how (and how well) GhostWriter interpreted their style (Section 5.2). However, our other reflection features were less utilized, potentially due to the cost of information processing (Section 5.1). As P15 said, “These pages [aren’t] telling me much, or at least I didn’t have the patience to go through all of it.” P14 shared, “It would be cool if you could toggle to see the style, history, or likes in the right nav so you don’t have to go between the home page and other options,” suggesting ways to more seamlessly integrate lightweight reflection mechanisms into the natural flow of writing [82].

Overall, our work highlights how reflection can help people understand and act on AI responses [1, 49, 51, 87], pointing to important design opportunities. For example, given the literature on writing as a technology for thinking (e.g., Ong [62]), it seems worthwhile to provide interactions for writers to reflect on their style and how it changes across a document or over time. Additionally, when multiple styles are learned (Section 6.1), future work can explore how writers decide what constitutes a desirable or undesirable style in a particular context. Systems such as TextFocals [35] also reveal opportunities for using adaptive AI-generated views to encourage reflection and self-driven revision of writing without directly generating text with LLMs.

## 6.4 Considering the Role of AI and Ownership in AI-Infused Systems

Our study allowed us to investigate the evolving relationship between writers and AI, focusing on the connection between perceived ownership and process agency.

- 6.4.1 The role of AI in collaborative writing. For any collaborative human-AI task, it is crucial to study perceptions about AI and design with this information in mind. Given participants’ diverse perspectives on the relationship between writers and AI (Section 5.4), we encourage future work to look at shaping AI-mediated experiences to help AI fulfill different roles [45, 61, 83]. For instance, how can we transform LLM-powered writing systems to help AI serve more as collaborators rather than tools?

Works such as Biermann et al. [6], Chakrabarty et al. [9], Guo et al. [26] suggest that the role of AI can fluctuate throughout the writing process and depends on an author’s values regarding which parts of their writing they wish to maintain control over (e.g., craftsmanship and authenticity). We observed a similar phenomenon with GhostWriter, where many users switched preferred modes of style personalization over the course of writing, and between different tasks (Section 5.1).

Exploring this dynamic interplay between user values, agency, and perceptions of AI is critical to supplement and extend existing research on designing AI-enhanced productivity and creativity support tools (e.g., Chung et al. [11], Lawton et al. [44]). This fluctuation also suggests the need for systems to support role fluidity – allowing users to move between different types of AI assistance (e.g., exploratory vs. prescriptive) as their goals and writing evolve.

- 6.4.2 Agency over the writing process influences one’s sense of outcome ownership, but is not sufficient to define it. With GhostWriter, we explored whether preserving agency could increase users’ perceived ownership over the results of collaborative AI writing. In some cases, this was true: “I do [feel ownership] because there were so many different things I could inform to make [it] my own” (P18), or “I didn’t actually create” the document, but “I created the prompt. Yeah, I’m putting my name on it” (P01). Others were hesitant to claim ownership over the generated text: “I architected it, but I didn’t build it” (P09), or “There’s something about not feeling like the owner when using a product like this - more like a partner I should split the by-line with” (P13). P04 shared that they would only feel ownership “if I weren’t using the tool.”

The tension between ownership and agency raises questions about how ownership should be perceived and (possibly) redefined in the age of generative AI [85]. Draxler et al. [16] frames this tension as the “AI ghostwriter effect,” where users resist labeling AIassisted writing as AI-authored, but also do not consider themselves the owner of this work. However, the authors find that increasing user control over co-written text bolsters perceived ownership, mirroring the sentiments of participants like P18 when using GhostWriter’s various style controls. Similarly, Joshi and Vogel [34] shows that crafting longer, detailed prompts enhances user ownership, but this requires more metacognitive effort [90]. Counterquill [15] demonstrates that breaking down the human-AI writing process into three distinct phases — learning, brainstorming, and co-writing — can also increase writers’ perceived ownership.

These findings, along with those from Palani and Ramos [64], suggest that to preserve ownership, AI-powered systems should assign writers a key orchestrator role throughout the writing process. This role can empower writers to regulate and navigate the shifting boundaries of initiative and responsibility in humanAI co-creation. GhostWriter takes initial steps in this direction,

but our results highlight opportunities to build on this foundation, encouraging the development of systems that more flexibly support user agency, authorship, and creative control in AI-infused writing.

## 7 Limitations & Considerations

Our findings provide encouraging signals on how design can enhance alignment between people and AI writing assistance, while supporting user agency throughout the experience. By design, design probes are not intended to serve as comparisons to a baseline; instead, they function as instruments to explore ideas and formulate new questions during early stages of research. We consider our work a starting point for comparative studies and future research to build upon. More work is also needed to validate our results in other types of writing activities and over longer-term use.

The focus on UX-centric professionals from the United States may restrict the generalizability of our findings to more diverse populations. Since all participants were employees of a large technology company, they may have more exposure to generative AI than the average person. Our participant pool also exhibited a gender imbalance, possibly reflecting biases in certain industry roles. However, irrespective of role and self-reported background, we did not observe noticeable differences in participants’ abilities to access GhostWriter’s features.

Designing personalized systems of any kind requires careful consideration of data privacy trade-offs. While personalization offers numerous benefits, e.g., learning and applying custom writing styles, it also poses risks. After engaging with GhostWriter, P14 asked, “If people put sensitive data in here, is it safe?” Many users may not even be aware of the privacy risks associated with such data-driven systems. To mitigate these challenges, we advocate for a design approach rooted in transparency and user consent. By providing users with clear communication about how their information is used and stored, as well as global controls to turn on/off data collection [2, 33, 72], we can build human-centered, personalized experiences that protect user privacy and trust.

## 8 Conclusion

In this work, we explore how to design AI-assisted writing experiences that allow control over personalization and champion user agency. Following a set of design goals and principles, we created GhostWriter, an AI-infused editor, and used it as a design probe to study the potential of large language models in crafting personalized writing experiences through style and context.

A study in which participants used GhostWriter on two writing tasks revealed that they valued the tool (arguably compared to previous writing experiences and tools from their daily lives) and perceived it as allowing them to exert rich agency and align AI writing to their goals. Participant feedback illustrated the benefits of offering both implicit and explicit mechanisms to teach the system about one’s writing preferences – particularly as these preferences may evolve throughout the writing process. Guided by these findings, we present design lessons to help others build the new generation of collaborative AI writing systems.

We hope that our work inspires others looking to harness generative AI to enhance and complement human capabilities, providing a reference for exploring the challenges and opportunities that arise when designing and using these emerging technologies.

## References

- [1] Saleema Amershi, Maya Cakmak, William Bradley Knox, and Todd Kulesza. 2014. Power to the people: The role of humans in interactive machine learning. Ai Magazine 35, 4 (2014), 105–120.
- [2] Saleema Amershi, Dan Weld, Mihaela Vorvoreanu, Adam Fourney, Besmira Nushi, Penny Collisson, Jina Suh, Shamsi Iqbal, Paul N Bennett, Kori Inkpen, et al. 2019. Guidelines for human-AI interaction. In Proceedings of the 2019 chi conference on human factors in computing systems. ACM, Glasgow, Scotland, 1–13.
- [3] Barrett R Anderson, Jash Hemant Shah, and Max Kreminski. 2024. Homogenization effects of large language models on human creative ideation. In Proceedings of the 16th Conference on Creativity & Cognition. ACM, Chicago, IL, 413–425.
- [4] Soulef Benhamdi, Abdesselam Babouri, and Raja Chiky. 2017. Personalized recommender system for e-Learning environment. Education and Information Technologies 22 (2017), 1455–1477.
- [5] Karim Benharrak, Tim Zindulka, Florian Lehmann, Hendrik Heuer, and Daniel Buschek. 2023. Writer-Defined AI Personas for On-Demand Feedback Generation.
- [6] Oloff C Biermann, Ning F Ma, and Dongwook Yoon. 2022. From tool to companion: Storywriters want AI writers to respect their personal values and writing strategies. In Proceedings of the 2022 ACM Designing Interactive Systems Conference. ACM, Online, 1209–1227.
- [7] Kirsten Boehner, Janet Vertesi, Phoebe Sengers, and Paul Dourish. 2007. How HCI interprets the probes. In Proceedings of the SIGCHI conference on Human factors in computing systems. ACM, San Jose, CA, 1077–1086.
- [8] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems 33 (2020), 1877–1901.
- [9] Tuhin Chakrabarty, Vishakh Padmakumar, Faeze Brahman, and Smaranda Muresan. 2024. Creativity Support in the Age of Large Language Models: An Empirical Study Involving Professional Writers. In Proceedings of the 16th Conference on Creativity & Cognition. ACM, Chicago, IL, 132–155.
- [10] Mia Xu Chen, Benjamin N Lee, Gagan Bansal, Yuan Cao, Shuyuan Zhang, Justin Lu, Jackie Tsay, Yinan Wang, Andrew M Dai, Zhifeng Chen, et al. 2019. Gmail smart compose: Real-time assisted writing. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, Anchorage, AK, 2287–2295.
- [11] John Joon Young Chung, Shiqing He, and Eytan Adar. 2022. Artist support networks: Implications for future creativity support tools. In Designing Interactive Systems Conference. ACM, Online, 232–246.
- [12] John Joon Young Chung, Wooseok Kim, Kang Min Yoo, Hwaran Lee, Eytan Adar, and Minsuk Chang. 2022. TaleBrush: Sketching stories with generative pretrained language models. In Proceedings of the 2022 CHI Conference on Human Factors in Computing Systems. ACM, New Orleans, LA, 1–19.
- [13] Elizabeth Clark, Anne Spencer Ross, Chenhao Tan, Yangfeng Ji, and Noah A Smith. 2018. Creative writing with a machine in the loop: Case studies on slogans and stories. In 23rd International Conference on Intelligent User Interfaces. ACM, Tokyo, Japan, 329–340.
- [14] Hai Dang, Karim Benharrak, Florian Lehmann, and Daniel Buschek. 2022. Beyond text generation: Supporting writers with continuous automatic text summaries. In Proceedings of the 35th Annual ACM Symposium on User Interface Software and Technology. ACM, Bend, OR, 1–13.
- [15] Xiaohan Ding, Kaike Ping, Uma Sushmitha Gunturi, Buse Carik, Sophia Stil, Lance T Wilhelm, Taufiq Daryanto, James Hawdon, Sang Won Lee, and Eugenia H Rho. 2024. CounterQuill: Investigating the Potential of Human-AI Collaboration in Online Counterspeech Writing.
- [16] Fiona Draxler, Anna Werner, Florian Lehmann, Matthias Hoppe, Albrecht Schmidt, Daniel Buschek, and Robin Welsch. 2024. The AI ghostwriter effect: When users do not perceive ownership of AI-generated text but self-declare as authors. ACM Transactions on Computer-Human Interaction 31, 2 (2024), 1–40.
- [17] Wanyu Du, Zae Myung Kim, Vipul Raheja, Dhruv Kumar, and Dongyeop Kang.

2022. Read, revise, repeat: A system demonstration for human-in-the-loop iterative text revision.

- [18] Nouha Dziri, Ehsan Kamalloo, Kory W Mathewson, and Osmar Zaiane. 2018. Augmenting neural response generation with context-aware topical attention.
- [19] Xiaoxuan Fang, Davy Tsz Kit Ng, Jac Ka Lok Leung, and Samuel Kai Wah Chu.

2023. A systematic review of artificial intelligence technologies used for story writing. Education and Information Technologies 28 (2023), 1–37.

- [20] Zhenxin Fu, Xiaoye Tan, Nanyun Peng, Dongyan Zhao, and Rui Yan. 2018. Style transfer in text: Exploration and evaluation. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 32. AAAI, New Orleans, LA, 9 pages.
- [21] Bill Gaver, Tony Dunne, and Elena Pacenti. 1999. Design: cultural probes. interactions 6, 1 (1999), 21–29.
- [22] Katy Gero, Alex Calderwood, Charlotte Li, and Lydia Chilton. 2022. A design space for writing support tools using a cognitive process model of writing. In

- Proceedings of the first workshop on intelligent and interactive writing assistants (In2Writing 2022). ACL, Dublin, Ireland, 11–24.
- [23] Katy Ilonka Gero, Tao Long, and Lydia B Chilton. 2023. Social dynamics of AI support in creative writing. In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems. ACM, Hamberg, Germany, 1–15.
- [24] Frederic Gmeiner and Nur Yildirim. 2023. Dimensions for Designing LLM-based Writing Support. In In2Writing Workshop at CHI. ACM, Hamberg, Germany, 2 pages.
- [25] Connor Graham and Mark Rouncefield. 2008. Probes and participation. In Proceedings of the Tenth Anniversary Conference on Participatory Design 2008. ACM, Bloomington, IN, 194–197.
- [26] Alicia Guo, Shreya Sathyanarayanan, Leijie Wang, Jeffrey Heer, and Amy Zhang.

2024. From Pen to Prompt: How Creative Writers Integrate AI into their Writing Practice.

- [27] Fred Hohman, Andrew Head, Rich Caruana, Robert DeLine, and Steven M Drucker. 2019. Gamut: A design probe to understand how data scientists understand machine learning models. In Proceedings of the 2019 CHI conference on human factors in computing systems. ACM, Glasgow, Scotland, 1–13.
- [28] Md Naimul Hoque, Bhavya Ghai, Kari Kraus, and Niklas Elmqvist. 2023. Portrayal: Leveraging NLP and Visualization for Analyzing Fictional Characters. In Proceedings of the 2023 ACM Designing Interactive Systems Conference (Pittsburgh, PA, USA) (DIS ’23). Association for Computing Machinery, New York, NY, USA, 74–94. doi:10.1145/3563657.3596000
- [29] Zhiqiang Hu, Roy Ka-Wei Lee, Charu C Aggarwal, and Aston Zhang. 2022. Text style transfer: A review and experimental evaluation. ACM SIGKDD Explorations Newsletter 24, 1 (2022), 14–45.
- [30] Hilary Hutchinson, Wendy Mackay, Bo Westerlund, Benjamin B Bederson, Allison Druin, Catherine Plaisant, Michel Beaudouin-Lafon, Stéphane Conversy, Helen Evans, Heiko Hansen, et al. 2003. Technology probes: inspiring design for and with families. In Proceedings of the SIGCHI conference on Human factors in computing systems. ACM, Fort Lauderdale, FL, 17–24.
- [31] Daphne Ippolito, Ann Yuan, Andy Coenen, and Sehmon Burnam. 2022. Creative writing with an ai-powered writing assistant: Perspectives from professional writers.
- [32] Di Jin, Zhijing Jin, Zhiting Hu, Olga Vechtomova, and Rada Mihalcea. 2022. Deep learning for text style transfer: A survey. Computational Linguistics 48, 1 (2022), 155–205.
- [33] Matthew Jörke, Yasaman S Sefidgar, Talie Massachi, Jina Suh, and Gonzalo Ramos.

2023. Pearl: A Technology Probe for Machine-Assisted Reflection on Personal Data. In Proceedings of the 28th International Conference on Intelligent User Interfaces. ACM, Sydney, Australia, 902–918.

- [34] Nikhita Joshi and Daniel Vogel. 2024. Writing with AI Lowers Psychological Ownership, but Longer Prompts Can Help.
- [35] Jiho Kim, Ray C Flanagan, Noelle E Haviland, ZeAi Sun, Souad N Yakubu, Edom A Maru, and Kenneth C Arnold. 2024. Towards Full Authorship with AI: Supporting Revision with AI-Generated Views.
- [36] Jeongyeon Kim, Sangho Suh, Lydia B Chilton, and Haijun Xia. 2023. Metaphorian: Leveraging Large Language Models to Support Extended Metaphor Creation for Science Writing. In Proceedings of the 2023 ACM Designing Interactive Systems Conference. ACM, Pittsburgh, PA, 115–135.
- [37] Jaehyung Kim and Yiming Yang. 2024. Few-shot personalization of llms with mis-aligned responses.
- [38] Tae Soo Kim, Yoonjoo Lee, Jamin Shin, Young-Ho Kim, and Juho Kim. 2023. EvalLM: Interactive Evaluation of Large Language Model Prompts on UserDefined Criteria.
- [39] Tae Soo Kim, Arghya Sarkar, Yoonjoo Lee, Minsuk Chang, and Juho Kim. 2023. LMCanvas: Object-Oriented Interaction to Personalize Large Language ModelPowered Writing Environments.
- [40] Tae Wook Kim and Quan Tan. 2023. Repurposing Text-Generating AI into a Thought-Provoking Writing Tutor.
- [41] Yong Soo Kim and Bong-Jin Yum. 2011. Recommender system based on click stream data using association rule mining. Expert Systems with Applications 38

(2011), 13320–13327.

- [42] Kai Konen, Sophie Jentzsch, Diaoulé Diallo, Peer Schütt, Oliver Bensch, Roxanne El Baff, Dominik Opitz, and Tobias Hecking. 2024. Style vectors for steering generative large language model.
- [43] Vivek Kulkarni and Vipul Raheja. 2023. Writing Assistants Should Model Social Factors of Language.
- [44] Tomas Lawton, Kazjon Grace, and Francisco J Ibarrola. 2023. When is a Tool a Tool? User Perceptions of System Agency in Human–AI Co-Creative Drawing. In Proceedings of the 2023 ACM Designing Interactive Systems Conference. ACM, Pittsburgh, PA, 1978–1996.
- [45] Mina Lee, Katy Ilonka Gero, John Joon Young Chung, Simon Buckingham Shum, Vipul Raheja, Hua Shen, Subhashini Venugopalan, Thiemo Wambsganss, David Zhou, Emad A Alghamdi, et al. 2024. A Design Space for Intelligent and Interactive Writing Assistants. In Proceedings of the CHI Conference on Human Factors in Computing Systems. ACM, Honolulu, HI, 1–35.
- [46] Mina Lee, Percy Liang, and Qian Yang. 2022. Coauthor: Designing a humanai collaborative writing dataset for exploring language model capabilities. In

- Proceedings of the 2022 CHI conference on human factors in computing systems. ACM, New Orleans, LA, 1–19.
- [47] Cheng Li, Mingyang Zhang, Qiaozhu Mei, Yaqing Wang, Spurthi Amba Hombaiah, Yi Liang, and Michael Bendersky. 2023. Teach LLMs to Personalize–An Approach inspired by Writing Education.
- [48] Jinming Li, Wentao Zhang, Tian Wang, Guanglei Xiong, Alan Lu, and Gerard Medioni. 2023. GPT4Rec: A generative framework for personalized recommendation and user interests interpretation.
- [49] Chenchen Liu, Jierui Hou, Yun-Fang Tu, Youmei Wang, and Gwo-Jen Hwang.

2023. Incorporating a reflective thinking promoting mechanism into artificial intelligence-supported English writing environments. Interactive Learning Environments 31, 9 (2023), 5614–5632.

- [50] Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. 2021. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity.
- [51] Cerstin Mahlow. 2023. Writing Tools: Looking Back to Look Ahead.
- [52] Gloria Mark, Daniela Gudith, and Ulrich Klocke. 2008. The cost of interrupted work: more speed and stress. In Proceedings of the SIGCHI conference on Human Factors in Computing Systems. ACM, Florence, Italy, 107–110.
- [53] Damien Masson, Young-Ho Kim, and Fanny Chevalier. 2024. Textoshop: Interactions Inspired by Drawing Software to Facilitate Text Editing.
- [54] Piotr Mirowski, Kory W Mathewson, Jaylen Pittman, and Richard Evans. 2023. Co-Writing Screenplays and Theatre Scripts with Language Models: Evaluation by Industry Professionals. In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems. ACM, Hamberg, Germany, 1–34.
- [55] Meredith Ringel Morris, Carrie J Cai, Jess Holbrook, Chinmay Kulkarni, and Michael Terry. 2023. The design space of generative models.
- [56] Sheshera Mysore, Mahmood Jasim, Andrew McCallum, and Hamed Zamani. 2023. Editable User Profiles for Controllable Text Recommendation.
- [57] Sheshera Mysore, Zhuoran Lu, Mengting Wan, Longqi Yang, Steve Menezes, Tina Baghaee, Emmanuel Barajas Gonzalez, Jennifer Neville, and Tara Safavi. 2023. Pearl: Personalizing large language model writing assistants with generationcalibrated retrievers.
- [58] Arvind Narayanan. 2023. Understanding social media recommendation algorithms. https://academiccommons.columbia.edu/doi/10.7916/khdk-m460. Columbia University Academic Commons. DOI: 10.7916/khdk-m460.
- [59] Felicia Ng, Jina Suh, and Gonzalo Ramos. 2020. Understanding and Supporting Knowledge Decomposition for Machine Teaching. In Proceedings of the 2020 ACM Designing Interactive Systems Conference (Eindhoven, Netherlands) (DIS ’20). Association for Computing Machinery, New York, NY, USA, 1183–1194. doi:10.1145/3357236.3395454
- [60] Jakob Nielsen. 2005. Ten usability heuristics.
- [61] Ibukun Olatunji. 2023. Interactive writing systems and why small (er) could be more beautiful. In In2Writing Workshop at CHI. ACM, Hamberg, Germany, 4 pages.
- [62] Walter J Ong. 1992. Writing is a technology that restructures thought. In The linguistics of literacy. John Benjamins, Amsterdam, Netherlands, 293.
- [63] Vishakh Padmakumar and He He. 2023. Does Writing with Language Models Reduce Content Diversity?
- [64] Srishti Palani and Gonzalo Ramos. 2024. Evolving roles and workflows of creative practitioners in the age of generative AI. In Proceedings of the 16th Conference on Creativity & Cognition. ACM, Chicago, IL, 170–184.
- [65] Haekyu Park, Gonzalo Ramos, Jina Suh, Christopher Meek, Rachel Ng, and Mary Czerwinski. 2023. FoundWright: A System to Help People Re-find Pages from Their Web-history.
- [66] Ajay Patel, Delip Rao, and Chris Callison-Burch. 2023. Learning Interpretable Style Embeddings via Prompting LLMs.
- [67] Napol Rachatasumrit, Gonzalo Ramos, Jina Suh, Rachel Ng, and Christopher Meek.

2021. ForSense: Accelerating online research through sensemaking integration and machine research support. In 26th International Conference on Intelligent User Interfaces. ACM, Online, 608–618.

- [68] Filip Radlinski, Krisztian Balog, Fernando Diaz, Lucas Dixon, and Ben Wedin. 2022. On Natural Language User Profiles for Transparent and Scrutable Recommendation. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, Madrid, Spain, 2863–2874.
- [69] Vipul Raheja, Dhruv Kumar, Ryan Koo, and Dongyeop Kang. 2023. CoEdIT: Text Editing by Task-Specific Instruction Tuning.
- [70] Gonzalo Ramos, Christopher Meek, Patrice Simard, Jina Suh, and Soroush Ghorashi. 2020. Interactive machine teaching: a human-centered approach to building machine-learned models. Human–Computer Interaction 35, 5-6 (2020), 413–451.
- [71] Alex Reinhart, Ben Markey, Michael Laudenbach, Kachatad Pantusen, Ronald Yurko, Gordon Weinberg, and David West Brown. 2025. Do LLMs write like humans? Variation in grammatical and rhetorical styles. Proceedings of the National Academy of Sciences 122, 8 (2025), e2422455122.
- [72] Tylea Richard. 2024. User Agency in AI Deployments: Why Responsible AI Leads to Better Products. www.grammarly.com/blog/company/user-agency-inai-deployments.

- [73] Alireza Salemi, Sheshera Mysore, Michael Bendersky, and Hamed Zamani. 2023. LaMP: When Large Language Models Meet Personalization.
- [74] Abel Salinas and Fred Morstatter. 2024. The butterfly effect of altering prompts: How small changes and jailbreaks affect large language model performance.
- [75] Bill N Schilit, Gene Golovchinsky, and Morgan N Price. 1998. Beyond paper: supporting active reading with free form digital ink annotations. In Proceedings of the SIGCHI conference on Human factors in computing systems. ACM, Los Angeles, CA, 249–256.
- [76] Patrice Y Simard, Saleema Amershi, David M Chickering, Alicia Edelman Pelton, Soroush Ghorashi, Christopher Meek, Gonzalo Ramos, Jina Suh, Johan Verwey, Mo Wang, et al. 2017. Machine teaching: A new paradigm for building machine learning systems.
- [77] Nikhil Singh, Guillermo Bernal, Daria Savchenko, and Elena L. Glassman. 2023. Where to Hide a Stolen Elephant: Leaps in Creative Writing with Multimodal Machine Intelligence. ACM Trans. Comput.-Hum. Interact. 30, 5, Article 68 (Sept. 2023), 57 pages. doi:10.1145/3511599
- [78] Alessandro Sordoni, Michel Galley, Michael Auli, Chris Brockett, Yangfeng Ji, Margaret Mitchell, Jian-Yun Nie, Jianfeng Gao, and Bill Dolan. 2015. A neural network approach to context-sensitive generation of conversational responses.
- [79] Jessi Stark, Anthony Tang, Young-Ho Kim, Joonsuk Park, and Daniel Wigdor.

2023. Can AI Support Fiction Writers Without Writing For Them?. In Proceedings of the Second Workshop on Intelligent and Interactive Writing Assistants. ACM, Hamberg, Germany, 3 pages.

- [80] John Sweller. 2011. Cognitive load theory. InPsychology of learning and motivation. Vol. 55. Elsevier, Amsterdam, Netherlands, 37–76.
- [81] Karan Taneja, Harshvardhan Sikka, and Ashok Goel. 2022. A framework for interactive knowledge-aided machine teaching.
- [82] Lukas Teufelberger, Xintong Liu, Zhipeng Li, Max Moebus, and Christian Holz.

2024. LLM-for-X: Application-agnostic Integration of Large Language Models to Support Personal Writing Workflows.

- [83] Qian Wan, Siying Hu, Yu Zhang, Piaohong Wang, Bo Wen, and Zhicong Lu. 2024. " It Felt Like Having a Second Mind": Investigating Human-AI Co-creativity in Prewriting with Large Language Models. Proceedings of the ACM on HumanComputer Interaction 8, CSCW1 (2024), 1–26.
- [84] Zijie J Wang, Aishwarya Chakravarthy, David Munechika, and Duen Horng Chau. 2024. Wordflow: Social Prompt Engineering for Large Language Models.
- [85] Azmine Toushik Wasi, Mst Rafia Islam, and Raima Islam. 2024. Llms as writing assistants: Exploring perspectives on sense of ownership and reasoning. In Proceedings of the Third Workshop on Intelligent and Interactive Writing Assistants.

ACM, Honolulu, HI, 38–42.

- [86] Azmine Toushik Wasi, Raima Islam, and Mst Rafia Islam. 2024. Ink and individuality: Crafting a personalised narrative in the age of llms. In Proceedings of the Third Workshop on Intelligent and Interactive Writing Assistants. ACM, Honolulu, HI, 43–47.
- [87] Zekun Xi, Wenbiao Yin, Jizhan Fang, Jialong Wu, Runnan Fang, Ningyu Zhang, Jiang Yong, Pengjun Xie, Fei Huang, and Huajun Chen. 2025. OmniThink: Expanding Knowledge Boundaries in Machine Writing through Thinking.
- [88] Daijin Yang, Yanpeng Zhou, Zhiyuan Zhang, Toby Jia-Jun Li, and Ray LC. 2022. AI as an Active Writer: Interaction strategies with generated text in human-AI collaborative fiction writing. In Joint Proceedings of the ACM IUI Workshops, Vol. 10. CEUR-WS Team, ACM, Online, 10 pages.
- [89] Ann Yuan, Andy Coenen, Emily Reif, and Daphne Ippolito. 2022. Wordcraft: story writing with large language models. In 27th International Conference on Intelligent User Interfaces. ACM, Online, 841–852.
- [90] JD Zamfirescu-Pereira, Richmond Y Wong, Bjoern Hartmann, and Qian Yang.

2023. Why Johnny can’t prompt: how non-AI experts try (and fail) to design LLM prompts. In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems. ACM, Hamberg, Germany, 1–21.

- [91] Haopeng Zhang, Xiao Liu, and Jiawei Zhang. 2023. SummIt: Iterative Text Summarization via ChatGPT.
- [92] Tianyi Zhang, Faisal Ladhak, Esin Durmus, Percy Liang, Kathleen McKeown, and Tatsunori B Hashimoto. 2023. Benchmarking large language models for news summarization.
- [93] Xin Zhao. 2022. Leveraging artificial intelligence (AI) technology for English writing: Introducing Wordtune as a digital writing assistant for EFL writers. RELC Journal 54 (2022), 00336882221094089.
- [94] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems 36 (2023), 46595–46623.
- [95] Hanxun Zhong, Zhicheng Dou, Yutao Zhu, Hongjin Qian, and Ji-Rong Wen.

2022. Less is more: Learning to refine dialogue history for personalized dialogue generation.

- [96] Wenxuan Zhou, Sheng Zhang, Hoifung Poon, and Muhao Chen. 2023. Contextfaithful prompting for large language models.
- [97] Zhongyi Zhou. 2022. Exploiting and Guiding User Interaction in Interactive Machine Teaching. In Adjunct Proceedings of the 35th Annual ACM Symposium on User Interface Software and Technology. ACM, Bend, OR, 1–5.

Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009

