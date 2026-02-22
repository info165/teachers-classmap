'use strict';

const { onDocumentCreated } = require("firebase-functions/v2/firestore");
const { onRequest } = require("firebase-functions/v2/https");
const { VertexAI } = require('@google-cloud/vertexai');
const admin = require('firebase-admin');
const crypto = require('crypto');
const { Resolver } = require('dns').promises;

// ─────────────────────────────────────────────────────────────────────────────
// INITIALIZATION
// ─────────────────────────────────────────────────────────────────────────────

admin.initializeApp();
const vertex_ai = new VertexAI({ project: process.env.GCLOUD_PROJECT, location: 'us-central1' });
const db = admin.firestore();
const storage = admin.storage();
const dnsResolver = new Resolver();

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────

const MATH_SUBJECTS = [
    "maths", "mathematics", "applied mathematics",
    "applied maths", "pure mathematics", "additional mathematics"
];

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// ─────────────────────────────────────────────────────────────────────────────
// UTILITY FUNCTIONS
// ─────────────────────────────────────────────────────────────────────────────

function hashKey(plainKey) {
    return crypto.createHash('sha256').update(plainKey).digest('hex');
}

function isLiteratureSubject(subject) {
    if (!subject) return false;
    const literatureKeywords = [
        "english", "hindi", "sociology", "political science",
        "history", "civics", "social studies", "general studies",
        "mass media & communication"
    ];
    return literatureKeywords.includes(subject.trim().toLowerCase());
}

function extractJsonFromString(text) {
    if (!text || typeof text !== 'string') return null;
    let targetText = text.trim();

    if (targetText.startsWith('[') && !targetText.endsWith(']')) {
        const lastBrace = targetText.lastIndexOf('}');
        if (lastBrace !== -1) {
            targetText = targetText.substring(0, lastBrace + 1) + ']';
        } else {
            targetText += ']';
        }
    }

    const match = targetText.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
    const cleaned = match ? match[1] : targetText;

    try {
        return JSON.parse(cleaned);
    } catch (e) {
        const firstBrace = cleaned.indexOf('{');
        const firstBracket = cleaned.indexOf('[');
        let startIndex = (firstBrace === -1) ? firstBracket : (firstBracket === -1) ? firstBrace : Math.min(firstBrace, firstBracket);
        let endIndex = Math.max(cleaned.lastIndexOf(']'), cleaned.lastIndexOf('}'));
        if (startIndex === -1 || endIndex === -1) return null;
        try {
            return JSON.parse(cleaned.substring(startIndex, endIndex + 1));
        } catch (e2) {
            return null;
        }
    }
}

function cleanUndefined(obj) {
    if (Array.isArray(obj)) return obj.map(item => cleanUndefined(item));
    if (obj !== null && typeof obj === 'object') {
        return Object.fromEntries(
            Object.entries(obj)
                .filter(([_, v]) => v !== undefined)
                .map(([k, v]) => [k, cleanUndefined(v)])
        );
    }
    return obj;
}

function normalizeForComparison(str) {
    if (!str) return '';
    return String(str).toLowerCase()
        .replace(/\b(ans|answer|q|question|sol|solution|pt|part|or|alt|alternative)\b/gi, '')
        .replace(/[^a-z0-9]/g, '')
        .trim();
}

/**
 * FIX #5: Validates and normalizes ERP question objects to the internal schema.
 * Prevents silent failures in the librarian/grader when fields are missing.
 */
function validateAndNormalizeQuestions(questions) {
    if (!Array.isArray(questions) || questions.length === 0) {
        throw new Error("400: questions_json must be a non-empty array");
    }
    return questions.map((q, i) => ({
        // ── Core fields (were already mapped) ────────────────────────────
        questionNumber: String(q.questionNumber || q.question_number || i + 1),
        text:           q.text           || q.question_text            || "",
        answer:         q.answer         || q.model_answer             || "",
        marks:          Number(q.marks   || q.max_marks                || 1),
        type:           q.type           || "SA",
        topicAnchors:   Array.isArray(q.topicAnchors)   ? q.topicAnchors
                      : Array.isArray(q.topic_anchors)  ? q.topic_anchors : [],
        checkingInstructions: q.checkingInstructions || q.checking_instructions || "",
        rubric:         q.rubric || null,

        // ── ADDED: Fields that were silently dropped ──────────────────────

        // Preserve the original question ID so Firestore references still work.
        // If the ERP sends back a question that came from /v1/digitize, the id
        // field (e.g. "q_1770714346542_0") is preserved through the pipeline.
        id:             q.id || null,

        // bloom_level is stored in the question pattern and echoed in reports.
        // Dropping it meant the output report lost Bloom's taxonomy data.
        bloom_level:    q.bloom_level || q.bloomLevel || null,

        // imagePrompt is essential for diagram questions. The grader uses this
        // to know what to look for in a student's hand-drawn diagram.
        // Without it, diagram questions were graded as if they had no diagram.
        imagePrompt:    q.imagePrompt || q.image_prompt || null,

        // options is CRITICAL for MCQ. The grading system instruction says:
        //   "Extract Model_Answer_Letter from the 'modelAnswer' field"
        // but the actual options object (A/B/C/D text) must reach the grader
        // so the context is available. Without this, MCQ grading can hallucinate.
        options:        q.options || null,

        // topic is stored and echoed in the per-question report. Losing it
        // meant topic-level analytics in the ERP had no data.
        topic:          q.topic || "",

        // ragContext contains additional domain context for complex questions.
        // The grading system uses this as extra knowledge for ambiguous answers.
        ragContext:     q.ragContext || q.rag_context || "",

        // kept flags whether this question was included in the final paper.
        // The Teacher PWA uses this to hide draft questions. ERP should respect it.
        kept:           q.kept !== undefined ? q.kept : true,
    }));
}


// ─────────────────────────────────────────────────────────────────────────────
// GEMINI RETRY WRAPPER
// ─────────────────────────────────────────────────────────────────────────────

async function callGeminiWithRetry(model, request, retries = 3) {
    for (let i = 0; i <= retries; i++) {
        try {
            return await model.generateContent(request);
        } catch (error) {
            const errStr = error.message ? error.message.toLowerCase() : "";
            const isRateLimit = errStr.includes('429') || errStr.includes('too many requests');
            if (isRateLimit && i < retries) {
                const delay = Math.pow(2, i) * 500 + Math.random() * 1000;
                console.warn(`[VertexAI] Rate limit (429). Retrying in ${Math.round(delay)}ms...`);
                await sleep(delay);
                continue;
            }
            throw error;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SYSTEM PROMPTS (UNCHANGED — Brain preserved exactly)
// ─────────────────────────────────────────────────────────────────────────────

const LATEX_MATH_INSTRUCTIONS = `
**MATHEMATICAL EXPRESSIONS (NON-NEGOTIABLE):**
- **RULE 1: ALWAYS USE LATEX.** All mathematical content MUST be generated using valid LaTeX syntax.
- **RULE 2: USE CORRECT DELIMITERS.** For INLINE math, use \\( ... \\). For BLOCK math, use \\[ ... \\].
- **RULE 3: YOU MUST ESCAPE BACKSLASHES FOR JSON.** All backslashes (\\) must be escaped (\\\\). Correct: { "text": "The answer is \\\\\\\\(x = \\\\\\\\frac{1}{2}\\\\\\\\)" }
- **RULE 4: NEVER FORGET THE CLOSING DELIMITER.** Every opening delimiter MUST have a matching closing delimiter.
`;

const LITERATURE_GRADING_INSTRUCTIONS = `
**LITERATURE SUBJECT GRADING (OVERRIDE RULES):**
- **LANGUAGE OF FEEDBACK:** Your 'finalFeedback', 'strength', and 'improvementArea' MUST be in the SAME language as the student's answer.
- **DEEP ERROR ANALYSIS:** You must be extremely strict about: 1. Spelling Errors, 2. Grammatical Errors, 3. Conceptual Errors.
- **SPECIFIC FEEDBACK:** In the 'finalFeedback', you MUST explicitly mention the type of errors found (spelling, grammar, conceptual).
`;

const OCR_SYSTEM_INSTRUCTION = `
You are a high-precision Vision OCR sensor.

Your ONLY responsibility is to visually transcribe student handwriting EXACTLY as it appears on the page and report WHERE it appears.
You DO NOT understand the question paper.
You DO NOT know question numbers, ownership, or grading structure.
You NEVER infer meaning beyond visible ink.

You act strictly as a camera + transcription layer.

# INPUT
- Handwritten student answer sheets (scanned images).

# OUTPUT REQUIREMENT (STRICT)
Return ONLY a valid JSON object:
{
  "text": "Full verbatim transcription with inline [#P:p,y,x] coordinate tags"
}

Do NOT include explanations, summaries, or extra keys.

--------------------------------------------------
- Every physical line of student handwriting MUST end with the tag [#P:p,y,x].
- "p" = 1-based page number.
- "y" = vertical position (0-1000).
- "x" = horizontal position (0-1000).
- PLACEMENT: Place the tag at the far right of the page (x between 910-940), aligned perfectly with the vertical baseline of the handwriting on that current line (y). This ensures markers appear neatly in the right-hand margin next to the student's work.
--------------------------------------------------
# 2. DELETION & CANCELLATION LAW (STRICT VISION)

- Any BLUE/BLACK/PENCIL cancellation mark (scribble, horizontal line, large X, dense cross-hatching, or shading) makes the underlying content VOID.
- SURGICAL OMISSION (MANDATORY): Content covered by a cancellation mark MUST be omitted entirely from your transcription. 
- You are STRICTLY FORBIDDEN from transcribing text that has been struck through or scribbled over.
- Do NOT use [DELETED] tags. Simply skip that section as if the paper were blank in that area.
- If a question identifier (e.g., "Ans-5") is scribbled out, you MUST NOT generate a [Q:N] or [#P:...] tag for it.

- Content is only VOID if it is clearly and intentionally SCRIBBLED OUT or crossed with a HEAVY X.
- LAYOUT MARKERS ARE NOT DELETIONS: Do NOT treat dashes (-), arrows (->), bullet points, or underlines as cancellation marks. These must be transcribed or skipped as whitespace, but the text next to them is VALID.

# RED INK EXCEPTION (TEACHER MARKS)
- Red ink marks (ticks, crosses, circles, underlines) are grading marks.
- IGNORE red ink completely.
- Transcribe the student handwriting underneath exactly as written.

--------------------------------------------------
# 3. NOISE SUPPRESSION (VISION ONLY)

- DO NOT transcribe:
  • Printed questions
  • Printed instructions
  • Headers / footers
  • Page numbers
  • Grid lines or ruled lines

- If a student writes INSIDE a printed table or grid:
  • Transcribe ONLY the handwritten content.
  • Do NOT describe the printed structure.

- If NO handwriting is visible on a page, output:
  "[NO HANDWRITING DETECTED]"

--------------------------------------------------
# 4. VERBATIM TRANSCRIPTION RULES & PRESERVATION LAWS

- TRANSCRIBE EXACTLY what is written. Messy, faint, or ugly handwriting MUST be preserved.
- Verbatim transcription overrides readability. You are NOT allowed to clean, simplify, or normalize handwriting.
- NO AUTOCORRECT. 'Mitochundria' stays 'Mitochundria'.
- NO COMPLETION. 'Photosyn...' stays 'Photosyn...'.

# CHARACTER PRESERVATION LAW
- Every visible comma, dot, dash, bracket, underline, or stroke MUST be transcribed. 
- ALPHA-NUMERIC PRECISION: Pay extreme attention to sub-part alphabets (e.g., 6a, 6b). The distinction between 'a' and 'b' is critical. If a 'B' has a small top loop, do not mistake it for an 'A'. same for a & c and b & d (their attention to distinction is required)
- Examples: "6." must not become "6"; "1(a)" must not become "1a"; a trailing dot MUST be preserved.

# GREEDY VISUAL RECALL OVERRIDE (CRITICAL)
- You are REQUIRED to prefer RECALL over PRECISION. If handwriting is legible, transcribe it.
- ONLY omit text if it is obscured by a dense scribble that makes the ink unreadable. 
- When in doubt if a mark is a "bullet point" or a "deletion", assume it is a bullet point and transcribe the text.
- Prioritize the "DELETION & CANCELLATION LAW" above all other transcription rules. If ink appears scribbled, it is no longer valid data.

--------------------------------------------------

# 5. MATHEMATICS (MANDATORY LaTeX)

- ALL math MUST be in LaTeX.
- Inline math: \\( x^2 \\)
- Block math: \\[ \\int f(x) dx \\]

- If math is unfinished, transcribe exactly as written.

--------------------------------------------------
# 6. LANGUAGE RULES

- Hindi / Devanagari:
  • Transcribe EXACTLY.
  • Preserve matras and shirorekha.
  • NO transliteration.

--------------------------------------------------
# 7. DIAGRAMS (STRUCTURAL DESCRIPTION)

If a diagram is present, output a structural description inside:
[DIAGRAM] ... [/DIAGRAM]

- Circuits:
  • Nodes
  • Components
  • Values
  • Current directions

- Geometry:
  • Shapes
  • Angles
  • Side lengths

- Graphs:
  • Axes
  • Scale
  • Key points

DO NOT invent details.
Describe ONLY what is visible.

--------------------------------------------------
# 8. PAGE BOUNDARIES (MANDATORY)

- Each image MUST start with:
  [PAGE N]

- NEVER merge text across pages.

------------------------------------

# QUESTION IDENTIFIER LOCK (CRITICAL OVERRIDE)

- Any standalone number at the beginning of a line followed by a dot or bracket
  (e.g., "2.", "6.", "6B", "6(b)") MUST be treated as a Question Identifier.

- You are STRICTLY FORBIDDEN from normalizing these values.

- DO NOT convert:
    2 → 1
    6B → 6A
    b → a
    c → a
    d → b

- When a character is ambiguous:
    You MUST transcribe the character EXACTLY as visually written,
    even if it looks unusual.

- The shape of letters matters:
    'a' has one loop.
    'b' has a vertical stem + loop.
    'c' is open.
    'd' has loop + stem on right.

- If uncertain, preserve the more complex character.
    Example:
      If unsure between A and B → choose B.
      If unsure between a and c → choose a.


# CORE SAFETY RULE

If you cannot SEE ink, it does NOT exist.
Never assume, infer, repair, or interpret.
`;

const GRADING_SYSTEM_INSTRUCTION = `You are a strict, objective, and precise Senior Examiner for CBSE/ICSE Boards.
You are an AI Teacher's Assistant, an expert in educational evaluation with a specialization in analyzing both handwritten text and diagrams.

Your task is to grade a BATCH of questions from a student's answer sheet based on the provided OCR text, with precision, fairness, empathy, and the intelligence of an experienced teacher, focusing on logical understanding over perfect presentation and with absolute consistency.

CRITICAL OVERRIDE: If 'type' is 'MCQ', immediately skip all rules regarding 'Empathy', 'Logical Understanding', 'Semantic Matching', or 'Spelling'. Apply ONLY the 'MCQ TUNNEL VISION' rules.

${LATEX_MATH_INSTRUCTIONS}

CRITICAL: Before applying any grading logic, check the 'type' field in the "ALL Questions to Grade" list for each specific question.

1. EXCLUSIVE APPLICATION: The 'MCQ TUNNEL VISION LAW' applies ONLY to questions with "type": "MCQ". 
2. SUBJECTIVE PROTECTION: For questions with type "Subjective", "VSA", "SA", or "LA", you are STRICTLY FORBIDDEN from using the MCQ feedback format (e.g., "A - Good work"). 

3. MCQ TUNNEL VISION (Type: MCQ Only):
MCQs (Type: MCQ) are strictly EXEMPT from all logic, math verification, spelling, grammar, or reasoning checks.

1. THE ANCHOR-LETTER RULE: Identify the Question Number. The VERY NEXT single letter (A, B, C, or D) or parenthesized letter ( (a), (b) ) is the student's final choice. 

2. CASE INSENSITIVITY: Treat 'a' as 'A', 'b' as 'B', etc. Case must NEVER result in a point deduction.

4. JUNK DATA DESTRUCTION: Any math symbols (\\emptyset, \\infty, etc.), text, or scribbles following the chosen letter are "Ghost Ink." You are STRICTLY FORBIDDEN from interpreting them. Even if the student writes "A. (but actually I think its B)", the choice is "A".
5. BINARY MATCHING ONLY: 
   - Extract Student_Letter.
   - Extract Model_Answer_Letter 
      - MODEL_CHOICE: Look ONLY at the 'modelAnswer' field provided for that question. Extract ONLY the single letter (A, B, C, or D) from it. Ignore all other words.
   - COMPARISON: Compare them using Case-Insensitive logic (a == A).
   - If (Student_Letter == Model_Answer_Letter) -> marksAwarded = FULL MARKS.
   - If (Student_Letter != Model_Answer_Letter) -> marksAwarded = 0.

7. FEEDBACK:
   - FORMAT: "[Letter chosen by student] - [Feedback message]"
   - Correct: "A - Good work." (Replace 'A' with the actual letter found).
   - Incorrect: "C - Incorrect" (Replace 'C' with the actual letter found).
   - If no letter is found: "No option detected."
8. NO OVERRIDES: This rule overrides the 'Consistency Rule', 'Incorrect Answer Rule', and 'Concept Error Rule'. Do not look for mistakes once the letter is identified.
9. FORBIDDEN REASONING: You must not use "Incorrect option selected" as feedback if the letters match. 
10. STRICT BINARY LOCK: The 'Strictness Level' (Lenient, Moderate, etc.) provided below MUST be completely IGNORED for MCQs. An incorrect MCQ option is ALWAYS 0 marks, regardless of how 'Lenient' the grading setting is. You are strictly forbidden from awarding 'Partial Marks' or 'Effort Marks' for an MCQ.

---CORE DIRECTIVE: SUBJECTIVE & NON-MCQ LOGIC

(The following rules apply ONLY to Subjective, VSA, SA, and LA questions)

Consistency Rule (NON-MCQ ONLY):
For Subjective questions (VSA, SA, LA), if you identify a mistake in the 'finalFeedback', you MUST deduct marks. This rule DOES NOT apply to MCQs.
- MCQ LOGIC PARITY: If your 'finalFeedback' for an MCQ contains the string 'Incorrect option selected', the 'marksAwarded' field for that JSON object MUST be 0. Discrepancy between feedback and marks is a critical system error.

**DIAGRAM-EQUATION RECONCILIATION:** Treat any text inside [DIAGRAM] tags as the "Source of Truth" for the student's logic. If a student writes a KVL/KCL equation (e.g., '4(I-I1) + ...'), check if the variables (I, I1) and values (4) match the components described in the diagram transcription. A correct equation based on a correctly labeled diagram is worth full marks, even if the diagram itself is messy.

Concept Error Rule:
If the student applies the wrong formula or wrong concept, the score for that step must be 0
(EXCEPT for MCQs, where all working is ignored).

HALLUCINATION PREVENTION & EVIDENCE LAW

1. VERBATIM QUOTES ONLY: You are STRICTLY FORBIDDEN from inventing student quotes. Any text you put in "finalFeedback" as a quote from the student MUST exist word-for-word in the "Full OCR Text" provided below.
2. MISMATCH PROTECTION: If the "Full OCR Text" is about a completely different topic than the question (e.g., student answered a question about 'Plant Cells' but the question is about 'Calculus'), treat the question as UNATTEMPTED. 
3. UNATTEMPTED LOGIC (STRICT): A question is only considered "Not detected" if the provided text block is completely empty or contains only [NO HANDWRITING DETECTED]. 
   - If any student handwriting is provided, you MUST grade it against the question prompt. 
   - DO NOT award 0 marks just because the question number isn't repeated inside the snippet.

# CRITICAL OWNERSHIP LOCK:
You are STRICTLY FORBIDDEN from discovering, inferring, or guessing question boundaries.

You will receive one specific answer block per question.
PRE-MAPPED DATA: Every word inside the provided block has been pre-verified by a librarian to belong to this question.
TRUSTED MAPPING: You are STRICTLY FORBIDDEN from requiring a physical question label (e.g. "Q1", "Ans 3") to be present within the text block. If text exists, it is an attempt.
You MUST grade ONLY the text provided for that question.
If text is empty, treat as unattempted.

# CANCELLATION SENSITIVITY:
- DELETED CONTENT RULE (NON-NEGOTIABLE): Any text enclosed within [DELETED] ... [/DELETED] tags (or fragments like "[MESSY]", "[SCRIBBLE]") represents cancelled work.
- You are STRICTLY FORBIDDEN from grading, evaluating, or awarding marks for deleted content.
- Award 0 marks for any question where the only visible answer content appears to be a discarded or scribbled-out attempt.

STEP-WISE MARKING PROTOCOL (SUBJECTIVE)

For Short Answer (SA) and Long Answer (LA):

1. Breakdown:
   Break the model answer into:
   - Concept / Formula
   - Substitution
   - Calculation
   - Final Answer + Unit

2. Evaluation:
   Check each component in student text.

3. Deduction Logic:
- Wrong formula structure (missing variables or incorrect formula type) → 0 for that step
- Incorrect numerical substitution → deduct only substitution marks
- Arithmetic error → deduct only calculation marks
- Do NOT zero entire question unless the core method is absent
   - Correct formula, wrong substitution → marks ONLY for formula
  - Calculation error → deduct only for the specific erroneous step; do not zero out preceding correct logic or correct formulas.
   - Final answer correct but essential steps missing → deduct 50%
     (unless Strictness = Lenient)


     MATH PARTIAL CREDIT PROTECTION (CRITICAL):

If the student uses the correct formula structure
but substitutes a wrong numeric value,
this is NOT a conceptual error.

It is an Application or Calculation error.

Do NOT zero the entire question for numeric substitution mistakes.
Award marks for correct structure.



"RUBRIC DOMINANCE: If 'step_marking' is present, it is the SOLE authority for defining the 'stepWiseEvaluation' array. You must generate exactly one marker for every component described in the 'step_marking' string."

     1. GOAL: Precisely visualize WHERE marks were earned and WHERE they were lost.
2. EARNED MARKS: Create entries in 'stepWiseEvaluation' for correct formulas/steps. Use marks > 0.
3. DEDUCTIONS (MISTAKES): If marks are deducted for conceptual, calculation, or analytical errors:
   - Create an entry in 'stepWiseEvaluation' with 'marks': 0.
   - COORDINATE: Copy the [#P:p,y,x] coordinate from the specific line in the evidence list for that line.
   - NO COMMENTS: Do not add descriptions/comments to these steps to minimize token cost.

ERROR CLASSIFICATION (STRICT 4-CATEGORY LIMIT)

You are STRICTLY FORBIDDEN from using categories like "Final Answer Error", "Missing Steps", or "Incomplete". Every mistake MUST be classified into exactly one of these four:

1. Conceptual (Misunderstanding of principles/theory)
2. Calculation (Arithmetic or algebraic processing error)
3. Application (Error in applying a known formula/concept to the problem)
4. Analytical (Failure in logic, reasoning, or data interpretation)

STRICTNESS ENFORCEMENT MATRIX (CRITICAL HIERARCHY)

For the SAME identified answer:
Lenient marks ≥ Moderate marks ≥ Strict marks

Rules:
1. If Moderate deducts for an error, Strict MUST deduct at least the same amount.
2. Strict is STRICTLY FORBIDDEN from ignoring any error penalized in Moderate.
3. Lenient may ignore minor errors that are penalized in Moderate.
4. Strict must never award more total marks than Moderate for the same answer.
5. Any violation of this hierarchy makes the grading INVALID.

STRICTNESS DEFINITIONS (FINAL 3-TIER MODEL)

There are ONLY 3 valid levels: Lenient, Moderate, and Strict.

1. LENIENT (Encouragement Mode)
- Goal: Reward understanding and encourage students. Simple checking.
- Minor calculation mistakes → deduct minimal (max 0.5 per error).
- Final answer wrong but correct method → award up to 80% of marks.
- Missing units/labels → ignore; do not deduct.
- Reward every correct element found.
- Round UP to the nearest 0.5.


2. MODERATE (Standard Board Marking)
- Goal: Fair evaluation following the provided rubric or model answer.
- RUBRIC DOMINANCE: If a rubric/step marking is present (often labeled as 'rubric' or 'step_marking'), follow it strictly.
- Step-wise marking is mandatory.
- Correct formula → award marks.
- Calculation error → deduct marks ONLY for that specific step.
- Final answer wrong → cannot award full marks.
- Round normally to the nearest 0.5.

CASCADE PROTECTION RULE (for lenient and moderate checking)

If an earlier numeric error affects later calculations,
do NOT repeatedly deduct marks for derived values.
Deduct only once for the root numeric error.
Award marks for correct method flow.

3. STRICT (Precision Examiner Mode)
- Goal: Absolute precision and accuracy. Marks are not given easily.
- EVERYTHING deducted in Moderate MUST be deducted here.
- All steps and concepts must be exactly correct. 
- Missing step → full deduction of that step's marks.
- Missing unit/label → mandatory deduction.
- Round DOWN to the nearest 0.5.

ASSERTION-REASON (AR) QUESTIONS

Logic:
- Determine truth of A
- Determine truth of R
- Determine whether R explains A

Options:
(A) A and R true, R explains A
(B) A and R true, R does NOT explain A
(C) A true, R false
(D) A false, R true

Grading:
- Option matches → full marks
- Option mismatch → 0 marks
- NO partial marks

--- THE CEILING LAW (NON-NEGOTIABLE) ---
1. ABSOLUTE CAP: You are strictly prohibited from awarding a 'marksAwarded' value higher than the 'maxMarks' provided for that question. 
2. OVERFLOW PROTECTION: Even if your 'stepWiseEvaluation' points sum up to a higher number, you MUST truncate the final 'marksAwarded' to the 'maxMarks' limit. 

TIER 1: CHOICE CONFLICT RESOLUTION (MANDATORY)

Before grading, scan the 'checkingInstructions' for every question in the batch to identify "OR" / "Alternative" pairs.

1. THE "FIRST ATTEMPT WINS" LAW (STRICT ENFORCEMENT):
   - CHRONOLOGICAL COMMITMENT: Scan the transcript from top to bottom. Committing to a choice happens the moment you see the question identifier.
   - EXCLUSIVE GRADING: If a question is part of an "OR" pair, you MUST grade the first instance found. You are STRICTLY FORBIDDEN from marking both questions as "Alternative question" if an attempt exists.

SCORING RULES
(MCQ EXCEPTION: The following rounding and increment rules DO NOT apply to MCQs.)
- 0.5 increments ONLY.
- Lenient → Round UP to nearest 0.5.
- Moderate → Standard rounding to nearest 0.5.
- Strict → Round DOWN to nearest 0.5.
- Never exceed maxMarks.

FEEDBACK RULES (STRICT FORMATTING)

2. SUBJECTIVE TYPES (VSA, SA, LA, DBQ, CS):
   - Full Marks: "Good work."
   - Deductions (Error-Only Pointers): Provide concise bullet points (using '- ') listing ONLY the specific errors, omissions, or conceptual gaps found in the student's work.

3. BREVITY RULE: Each pointer must be under 12 words. Total feedback must be under 50 words.

IMPROVEMENT AREAS & CATEGORIZATION
- CATEGORIES: [Conceptual], [Calculation], [Application], [Analytical].
- FORMAT: "[Category] Specific Academic Topic"

"CRITICAL: If a 'rubric' object with 'step_marking' is provided for a question, you MUST ignore your general marking knowledge and follow the provided 'step_marking' breakdown exactly. Your 'stepWiseEvaluation' array MUST contain exactly one entry for every step mentioned in the rubric. Failure to use the provided rubric is a system-critical error."

CRITICAL: You must apply the same level of analytical depth to every question in this batch. Accuracy for the final questions is just as important as the first.

**OUTPUT FORMAT:**
Your entire response MUST be a single, valid JSON array of objects, strictly following the provided JSON schema. DO NOT include any text outside the JSON array.
`;

// ─────────────────────────────────────────────────────────────────────────────
// FIX #4: FULL QUESTION EXTRACTION PROMPT (ported from frontend TS)
// Used by v1/digitize so external clients get the same brain as the Teacher PWA
// ─────────────────────────────────────────────────────────────────────────────

const QUESTION_EXTRACTION_SYSTEM_PROMPT = `You are an expert OCR engine with pedagogical knowledge. Goal: PERFECT REPLICATION of question paper content.

**LANGUAGE FILTER (CRITICAL):** This paper may contain both English and Hindi/regional language versions of the same questions. You MUST extract ONLY the English version. COMPLETELY IGNORE all Hindi, Devanagari script, or regional language text. Do not create any JSON object for Hindi questions.

**CRITICAL RULES (NON-NEGOTIABLE):**

1. NUMBERING & CONTEXT:
   - Process paper sequentially. Maintain context of main question number (e.g., "1.", "Q2.").
   - Sub-part identifiers (e.g., "(a)", "ii)") MUST be combined with the last main number seen.
   - "questionNumber" field = complete number (e.g., "1. (a)", "2. ii)").
   - "text" field = question content ONLY, without the number.
   - Capture question numbers EXACTLY as printed. Do NOT normalize (e.g., "4. (Q1)" stays "4. (Q1)", not "4a").
   - Cross-image continuity: If Image 1 ends with "38. B." and Image 2 starts with "C.", label it "38. C." NOT "39. C."

2. OR / ATTEMPT-EITHER LOGIC:
   - Always extract ALL options — never choose one.
   - Extract question BEFORE "OR" and AFTER "OR" as two separate JSON objects.
   - For each question in an OR pair, set checkingInstructions: "Alternative Question (OR) with Question [X]. Grade first attempt only." — explicitly naming the other question.
   - "Attempt X of Y" logic: Extract ALL questions in the group. For EACH, set checkingInstructions: "Group Choice (Questions [list all IDs]): Attempt X of Y. Grade only the first X attempts found for this specific set."

3. SUB-PART RULE:
   - If a main question number is immediately followed by sub-parts (a), (b), etc., the main text is CONTEXT ONLY — do NOT create a JSON object for it.
   - Every sub-part with its own mark value = its own JSON object.

4. MARK EXTRACTION:
   - Find marks (e.g., [3], (5 Marks)) and place in "marks" field.
   - If none found, assign 0. Do NOT calculate or balance a grand total.
   - If a main question has one mark value but multiple sub-parts, distribute marks intelligently. Sub-part marks must SUM to original total.

5. IGNORE HEADERS:
   - Do NOT create JSON for instructional headers like "Solve the following:", "Section A", "Answer all questions."
   - If a header is immediately followed by sub-parts, IGNORE the header completely.

6. COMPREHENSIVE EXTRACTION (ABSOLUTE):
   - NEVER stop early. Process every page, every image.
   - If a question is illegible: set text = "ERROR: Illegible question text", answer = "", marks = 0.
   - Before finishing, mentally review all images once more to ensure nothing is missed.

7. CONTEXT & PREAMBLE RETENTION:
   - If a question starts with a scenario, case study, passage, formula, or diagram description — INCLUDE it in the "text" field.
   - Do NOT orphan sub-parts from their context. If Q3 gives a formula then asks (a) and (b), that formula MUST appear in both (a) and (b) text fields.

8. QUESTION TYPE TAGGING (MANDATORY):
   - MCQ: has A/B/C/D options OR is 1 mark.
   - Assertion-Reason: contains "Assertion" and "Reason".
   - VSA: 1-2 marks, no options.
   - SA: 3-4 marks.
   - LA: 5+ marks.

9. MCQ FORMAT (CRITICAL):
   - "text" field MUST contain: Question stem + all 4 options + type label.
     Format: "Question text\\nA) Option 1\\nB) Option 2\\nC) Option 3\\nD) Option 4\\n[Type: MCQ]"
   - "answer" field MUST contain ONLY the correct option letter (e.g., "B"). No explanation.

10. ANSWER CONCISENESS (when no solution key provided):
    - MCQ: Option letter + name only (e.g., "C) Mitochondria").
    - Assertion-Reason: Option letter only (e.g., "A").
    - VSA: Core keyword or 5-word phrase only.
    - SA: 2-point bullet list only.
    - LA: 3-4 point skeleton marking scheme only.
    - NEVER write "The answer is..." or "I have extracted...".
    - If solution key IS provided: copy answer VERBATIM from key. Do NOT summarize or shorten.

11. RUBRIC & STEP-MARKING:
    - MCQ, True/False, Fill-in-Blanks, Assertion-Reason: set rubric = null.
    - VSA, SA, LA, Case Study: MUST generate "step_marking" string.
      - Look for printed marking schemes, "Step 1/2" labels, bracketed marks in margins.
      - If none printed, generate a logical 2-3 step breakdown.
      - Format: "Step 1: [Description] ([Marks]); Step 2: [Description] ([Marks])..."

12. TOPIC ANCHORS (CRITICAL):
    - For every question, generate "topicAnchors": array of 3-5 highly unique technical terms, proper nouns, or specific values from THIS question/answer only.
    - Generic words like "Calculate" or "Question" are FORBIDDEN.

13. IMAGE / DIAGRAM HANDLING:
    - If a question contains a diagram: write a detailed structural text description in "imagePrompt".
    - If no diagram: "imagePrompt" = null.

14. MATH & EQUATIONS:
    - ALL math MUST be in LaTeX. Inline: \\( x^2 \\). Block: \\[ \\int f(x) dx \\].
    - VERBATIM transcription. NEVER simplify, calculate, or rearrange.

15. PDF HANDLING:
    - Process pages sequentially. Maximum 50 pages.

${LATEX_MATH_INSTRUCTIONS}

**Output Format:** Stream one valid JSON object per question.
Schema fields (ALL required): "topic", "text", "type", "answer", "marks", "imagePrompt", "questionNumber", "checkingInstructions", "rubric" (containing "step_marking"), "topicAnchors".
Do NOT include "imageBase64".

Begin extracting ALL English questions now.`;

// ─────────────────────────────────────────────────────────────────────────────
// FIX #1: SSRF-SAFE EXTERNAL IMAGE FETCHER (Phase 7)
// Used for SaaS jobs where scan_urls come from external ERP servers.
// Blocks requests to internal/private IP ranges to prevent SSRF attacks.
// ─────────────────────────────────────────────────────────────────────────────

const BLOCKED_IP_PATTERNS = [
    /^10\./,
    /^192\.168\./,
    /^172\.(1[6-9]|2\d|3[01])\./,
    /^127\./,
    /^169\.254\./,
    /^::1$/,
    /^fc00:/i,
    /^fe80:/i,
];

async function fetchExternalImagesSecurely(urls) {
    if (!Array.isArray(urls) || urls.length === 0) {
        throw new Error("400: scan_urls must be a non-empty array");
    }

    const parts = await Promise.all(urls.map(async (url, idx) => {
        // 1. Enforce HTTPS only
        if (!url.startsWith('https://')) {
            throw new Error(`SSRF_BLOCK: URL at index ${idx} must use HTTPS. Received: ${url.substring(0, 50)}`);
        }

        // 2. DNS resolution + IP block check
        let hostname;
        try {
            hostname = new URL(url).hostname;
        } catch {
            throw new Error(`SSRF_BLOCK: Invalid URL format at index ${idx}`);
        }

        let addresses = [];
        try {
            addresses = await dnsResolver.resolve4(hostname);
        } catch {
            // IPv6 fallback
            try {
                addresses = await dnsResolver.resolve6(hostname);
            } catch {
                throw new Error(`SSRF_BLOCK: Could not resolve hostname: ${hostname}`);
            }
        }

        const blockedIp = addresses.find(ip => BLOCKED_IP_PATTERNS.some(r => r.test(ip)));
        if (blockedIp) {
            throw new Error(`SSRF_BLOCK: Hostname ${hostname} resolves to a blocked internal IP: ${blockedIp}`);
        }

        // 3. Fetch into memory — never written to Cloud Storage (Phase 7: Memory-Only Processing)
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 15000); // 15s per image
        let response;
        try {
            response = await fetch(url, { signal: controller.signal });
        } catch (fetchErr) {
            throw new Error(`FETCH_FAIL: Could not download image at index ${idx}: ${fetchErr.message}`);
        } finally {
            clearTimeout(timeout);
        }

        if (!response.ok) {
            throw new Error(`FETCH_FAIL: Image URL at index ${idx} returned HTTP ${response.status}`);
        }

        const contentType = response.headers.get('content-type') || 'image/jpeg';
const allowedTypes = [
    'image/jpeg', 'image/jpg', 'image/png', 
    'image/webp', 'image/gif',
    'application/pdf'  // ADD THIS
];

const mimeType = allowedTypes.find(t => contentType.includes(t)) || 'image/jpeg';

        const buffer = await response.arrayBuffer();
        const base64 = Buffer.from(buffer).toString('base64');

        // Memory is automatically freed after this function returns — no /tmp or Storage writes

        return { inlineData: { mimeType, data: base64 } };
    }));

    return parts;
}

// ─────────────────────────────────────────────────────────────────────────────
// SECURITY MIDDLEWARE (Phase 2 Gatekeeper)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * FIX #7: Full atomic idempotency — call BEFORE queuing to prevent race conditions.
 * Returns auth context. Call this, then use the returned ref to set the job atomically.
 */
async function validateSaaSRequest(req, requiredScope) {
    const apiKey = req.headers['x-api-key'];
    const idempotencyKey = req.headers['idempotency-key'];

    if (!apiKey) throw new Error("401: Missing x-api-key header");

    // 1. Auth via SHA-256 Hash lookup
    const hashed = hashKey(apiKey);
    const clientQuery = await db.collection('api_clients')
        .where('hashedApiKey', '==', hashed)
        .where('isActive', '==', true)
        .limit(1)
        .get();

    if (clientQuery.empty) throw new Error("401: Invalid or inactive API Key");

    const clientDoc = clientQuery.docs[0];
    const clientData = clientDoc.data();
    const clientId = clientDoc.id;

    // 2. Scope-Based Authorization
    if (!Array.isArray(clientData.permissions) || !clientData.permissions.includes(requiredScope)) {
        throw new Error(`403: Forbidden - Missing required scope: ${requiredScope}`);
    }

    // 3. Quota Enforcement
    const { monthlyLimit, currentUsage } = clientData.quota || {};
    if (monthlyLimit && (currentUsage || 0) >= monthlyLimit) {
        throw new Error("402: Payment Required - Monthly quota exceeded");
    }

    // 4. Idempotency Check
    if (idempotencyKey) {
        const idenId = `${clientId}_${idempotencyKey}`;
        const idenDoc = await db.collection('api_idempotency_log').doc(idenId).get();
        if (idenDoc.exists) {
            const data = idenDoc.data();
            return { clientId, clientData, isDuplicate: true, cachedResponse: data.cachedResponse };
        }
    }

    return { clientId, clientData, isDuplicate: false, idempotencyKey: idempotencyKey || null };
}

// ─────────────────────────────────────────────────────────────────────────────
// PHASE 5: WEBHOOK DISPATCHER
// ─────────────────────────────────────────────────────────────────────────────

async function dispatchWebhook(clientId, payload) {
    try {
        const clientDoc = await db.collection('api_clients').doc(clientId).get();
        if (!clientDoc.exists) return;

        const { webhookUrl, webhookSecret } = clientDoc.data();
        if (!webhookUrl) return;

        const bodyStr = JSON.stringify(payload);
        const signature = crypto
            .createHmac('sha256', webhookSecret || 'default_secret')
            .update(bodyStr)
            .digest('hex');

        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 10000);
        try {
            await fetch(webhookUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-classmap-signature': signature,
                    'x-classmap-event': 'grading.completed'
                },
                body: bodyStr,
                signal: controller.signal
            });
            console.log(`[Webhook] Successfully delivered to client: ${clientId}`);
        } finally {
            clearTimeout(timeout);
        }
    } catch (e) {
        // Webhook failures are non-fatal — job already completed successfully
        console.error(`[Webhook] Delivery failed for client ${clientId}:`, e.message);
        await db.collection('webhook_failures').add({
            clientId,
            error: e.message,
            timestamp: admin.firestore.FieldValue.serverTimestamp(),
            payload: JSON.stringify(payload).substring(0, 500) // truncate for safety
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RAG: GRADING RULES FETCH
// ─────────────────────────────────────────────────────────────────────────────

async function fetchGradingRules(subject) {
    try {
        const snapshot = await db.collection("gradingCorrectionRules")
            .where("subject", "in", ["Global", "global", subject])
            .get();
        return snapshot.docs.map(doc => doc.data());
    } catch (e) {
        console.warn("[RAG] Fetch Error:", e.message);
        return [];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OCR PIPELINE — extractTextFromImages
// UNCHANGED from original. Used by both Teacher PWA and SaaS API jobs.
// ─────────────────────────────────────────────────────────────────────────────

async function extractTextFromImages(imageParts, subject, jobId, jobData, allRules) {
    const totalPages = imageParts.length;
    const pageResults = [];
    const BATCH_SIZE = 3;

    const validNums = jobData.questions.map(q => q.questionNumber).join(', ');
    const contextualRules = `\n# CONTEXTUAL AWARENESS:\nThe valid question numbers for this exam are: ${validNums}.\nIf a handwritten digit is ambiguous, prefer a number from this list.\n`;

    const ocrTargetRules = allRules.filter(rule => rule.targetStep === 'OCR')
        .map(r => `- VISION RULE: ${r.correctionRule}`)
        .join('\n');

    const finalOcrInstruction = OCR_SYSTEM_INSTRUCTION + contextualRules +
        "\n# CRITICAL: Perform a hard reset of your state for every image. " +
        "DO NOT repeat characters. " +
        "Stop immediately when done.\n" +
        "\n# ADDITIONAL DOMAIN RULES:\n" + ocrTargetRules;

    for (let i = 0; i < imageParts.length; i += BATCH_SIZE) {
        const currentBatch = imageParts.slice(i, i + BATCH_SIZE);
        const endRange = Math.min(i + BATCH_SIZE, totalPages);

        let attempt = 0;
        let batchSuccess = false;

        while (attempt < 2 && !batchSuccess) {
            try {
                const currentProgress = Math.round(((i + currentBatch.length) / totalPages) * 25);
                await db.collection('gradingQueue').doc(jobId).update({
                    statusDetails: `Reading Sheets: Page ${i + 1} to ${endRange} (Total: ${totalPages})`,
                    currentStep: 1,
                    progress: Math.min(25, currentProgress)
                });

                const ocrSchema = {
                    type: "object",
                    properties: { text: { type: "string" } },
                    required: ["text"]
                };

                const refreshModifier = attempt > 0 ?
                    "\n\nCRITICAL: The previous attempt failed due to a repetition loop. RESET your state. DO NOT repeat characters. Extract ONLY handwriting concisely. Stop immediately when done." : "";

                const ocrModel = vertex_ai.getGenerativeModel({
                    model: 'gemini-2.5-flash-lite',
                    systemInstruction: { parts: [{ text: finalOcrInstruction + refreshModifier }] }
                });

                const result = await callGeminiWithRetry(ocrModel, {
                    contents: [{
                        role: 'user',
                        parts: [
                            ...currentBatch,
                            {
                                text: `Analyze these ${currentBatch.length} images.\n  CRITICAL: You MUST use the following markers for each image:\n  ${currentBatch.map((_, idx) => `- Image ${idx + 1} must start with: [PAGE ${i + idx + 1}]`).join('\n')}`
                            }
                        ]
                    }],
                    generationConfig: {
                        responseMimeType: "application/json",
                        responseSchema: ocrSchema,
                        candidateCount: 1,
                        seed: 42,
                        temperature: 0.0,
                        maxOutputTokens: 8192,
                        thinkingConfig: { thinkingBudget: 0 }
                    }
                });

                const rawResponseText = result.response.candidates[0].content.parts[0].text;
                const parsedOcr = extractJsonFromString(rawResponseText);

                if (!parsedOcr || typeof parsedOcr.text !== 'string') {
                    throw new Error("INVALID_OCR_JSON");
                }

                const loopRegex = /(.{1,3})\1{30,}/g;
                if (loopRegex.test(parsedOcr.text)) {
                    throw new Error("REPETITION_LOOP");
                }

                const transcript = parsedOcr.text;
                const pageSplitter = /\[PAGE\s+(\d+)\]/gi;
                const segments = transcript.split(pageSplitter);

                for (let idx = 0; idx < currentBatch.length; idx++) {
                    const absolutePageIndex = i + idx;
                    const targetPageNum = absolutePageIndex + 1;

                    let pageContent = "";
                    const segmentIndex = segments.findIndex((val, sIdx) => sIdx % 2 === 1 && parseInt(val, 10) === targetPageNum);

                    if (segmentIndex !== -1) {
                        pageContent = segments[segmentIndex + 1] || "";
                    } else if (idx === 0 && segments[0].trim().length > 20) {
                        pageContent = segments[0];
                    } else {
                        pageContent = "[NO HANDWRITING DETECTED]";
                    }

                   const sanitizedChunkText = pageContent.trim().replace(
    /\[#P\s*:\s*\d+\s*,\s*(\d+)\s*,\s*(\d+)\s*\]/gi,
    (_, y, x) => `[#P:${targetPageNum},${y.trim()},${x.trim()}]`
);

                    const anchors = [];
                    const matches = [...sanitizedChunkText.matchAll(/\[#P\s*:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]/gi)];
                    matches.forEach(m => {
                        anchors.push({ pageIndex: absolutePageIndex, y: parseInt(m[2], 10), x: parseInt(m[3], 10) });
                    });

                    pageResults.push({ pageNum: targetPageNum, text: sanitizedChunkText, rawText: sanitizedChunkText, anchors });
                }

                const usage = result.response.usageMetadata;
                await db.collection('apiUsageLogs').add({
                    timestamp: admin.firestore.FieldValue.serverTimestamp(),
                    teacherUid: jobData.teacherUid,
                    schoolId: jobData.schoolId || 'N/A',
                    feature: 'Assessment Checker OCR (Backend)',
                    modelCalled: 'gemini-2.5-flash',
                    tokenUsage: {
                        promptTokens: usage.promptTokenCount,
                        candidatesTokens: usage.candidatesTokenCount,
                        totalTokens: usage.totalTokenCount
                    },
                    totalCostInr: ((usage.promptTokenCount / 1000000) * 27.6) + ((usage.candidatesTokenCount / 1000000) * 230)
                });
                batchSuccess = true;

            } catch (err) {
                attempt++;
                if (attempt >= 2) {
                    for (let subIdx = 0; subIdx < currentBatch.length; subIdx++) {
                        const pageNum = i + subIdx + 1;
                        await db.collection('gradingQueue').doc(jobId).update({ statusDetails: `Recovery Mode: Reading Page ${pageNum}...` });

                        const recoveryModel = vertex_ai.getGenerativeModel({
                            model: 'gemini-2.5-flash',
                            systemInstruction: { parts: [{ text: finalOcrInstruction }] }
                        });
        const recResult = await recoveryModel.generateContent({
                            contents: [{ role: 'user', parts: [currentBatch[subIdx], { text: `[PAGE ${pageNum}] Extract handwriting for THIS SINGLE IMAGE ONLY. No repetition.` }] }],
                            generationConfig: { 
                                responseMimeType: "application/json", 
                                temperature: 0.1,
                                thinkingConfig: { thinkingBudget: 0 } // Add this line
                            }
                        });
                        const parsed = extractJsonFromString(recResult.response.candidates[0].content.parts[0].text);
                        const rawText = parsed ? parsed.text : "[RECOVERY FAILED]";
                        const fixedRecoveryText = rawText.replace(/\[#P\s*:\s*\d+\s*,/gi, `[#P:${pageNum},`);
                        pageResults.push({ pageNum, text: fixedRecoveryText, rawText: fixedRecoveryText, anchors: [] });
                    }
                    batchSuccess = true;
                }
            }
        }
    }
    return pageResults;
}

// ─────────────────────────────────────────────────────────────────────────────
// GRADING PIPELINE — gradeQuestionBatch
// UNCHANGED from original.
// ─────────────────────────────────────────────────────────────────────────────

async function gradeQuestionBatch(ocrText, questionBatch, strictness, subject, jobId, allRules) {
    const isLit = isLiteratureSubject(subject);
    const dynamicInstructions = isLit ? LITERATURE_GRADING_INSTRUCTIONS : '';
    const gradingRules = allRules.filter(r => r.targetStep === 'Grading' || !r.targetStep);
    const ragInstructions = gradingRules.map(r => `- RULE: ${r.correctionRule}`).join('\n');

    const questionNumbers = questionBatch.map(q => q.questionNumber).join(', ');
    await db.collection('gradingQueue').doc(jobId).set({ statusDetails: `Step 2/3: Grading questions ${questionNumbers}...` }, { merge: true });

    const subjectType = isLit ? 'Language' : 'Other';

    const fullSystemInstruction = `
    ${GRADING_SYSTEM_INSTRUCTION}
    
    === CONTEXT FOR THIS GRADING TASK ===
    SUBJECT: "${subject}"
    SUBJECT TYPE: "${subjectType}"
    STRICTNESS LEVEL FOR THIS BATCH: "${strictness}"
    
    === ADDITIONAL RULES & CORRECTIONS ===
    ${dynamicInstructions}
    ${ragInstructions}

    **OUTPUT FORMAT:**
    Your entire response MUST be a single, valid JSON array of objects, strictly following the provided JSON schema. DO NOT include any text outside the JSON array.
    `;

    const responseSchema = {
        type: "array",
        items: {
            type: "object",
            properties: {
                questionNumber: { type: "string" },
                marksAwarded: { type: "number" },
                finalFeedback: { type: "string" },
                improvementArea: {
                    type: "array",
                    items: { type: "string" },
                    description: "Format each string as: '[Category] Topic'."
                },
                requiresReview: { type: "boolean" },
                stepWiseEvaluation: {
                    type: "array",
                    items: {
                        type: "object",
                        properties: {
                            marks: { type: "number" },
                            pageIndex: { type: "integer" },
                            stepPoint: {
                                type: "array",
                                items: { type: "number" },
                                description: "The [y, x] values extracted verbatim from the [#P:p,y,x] tag."
                            }
                        },
                        required: ["marks", "pageIndex", "stepPoint"]
                    }
                }
            },
            required: ["questionNumber", "marksAwarded", "finalFeedback", "requiresReview", "stepWiseEvaluation"]
        }
    };

    const request = {
        model: 'gemini-2.5-flash',
        contents: [{
            role: 'user',
            parts: [{ text: `STUDENT_TRANSCRIPT:\n${ocrText}\n\nQUESTIONS_TO_GRADE (Pre-Mapped):\n${JSON.stringify(questionBatch, null, 2)}` }]
        }],
        systemInstruction: { parts: [{ text: fullSystemInstruction }] },
        generationConfig: {
            responseMimeType: "application/json",
            temperature: 0.0,
            responseSchema: responseSchema,
            thinkingConfig: { thinkingBudget: 0 }
        }
    };

    const gradingModel = vertex_ai.getGenerativeModel({ model: 'gemini-2.5-flash' });
    const result = await callGeminiWithRetry(gradingModel, request);
    const rawJson = result.response.candidates[0].content.parts[0].text;
    const parsedResult = extractJsonFromString(rawJson);

    if (!Array.isArray(parsedResult)) {
        throw new Error(`Grading phase failed: Expected a JSON array.`);
    }

    return questionBatch.map(reqQ => {
        const target = normalizeForComparison(reqQ.questionNumber);
        const aiMatch = parsedResult.find(r => normalizeForComparison(r.questionNumber) === target);

        if (aiMatch) {
            const marksAwarded = Math.min(parseFloat(aiMatch.marksAwarded) || 0, reqQ.marks);
            return { ...reqQ, ...aiMatch, marksAwarded, maxMarksForQuestion: reqQ.marks };
        }
        return {
            ...reqQ,
            marksAwarded: 0,
            maxMarksForQuestion: reqQ.marks,
            finalFeedback: "Question not detected.",
            requiresReview: true,
            stepWiseEvaluation: []
        };
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// LIBRARIAN SPLICING — UNCHANGED
// ─────────────────────────────────────────────────────────────────────────────

async function classifyOrWinner(studentSnippet, options, qNum) {
    if (!studentSnippet || studentSnippet.trim().length < 5) return 0;

    const prompt = `
    You are an Exam Version Classifier. 
    A student has provided an answer for Question ${qNum}. 
    There are ${options.length} possible versions of this question.

    STUDENT'S HANDWRITTEN TEXT:
    "${studentSnippet}"

    ${options.map((opt, i) => `
    --- VERSION ${i === 0 ? 'A' : 'B'} (Option ${i}) ---
    QUESTION PROMPT: "${opt.text}"
    MODEL ANSWER: "${opt.modelAnswer}"
    `).join('\n')}

    TASK:
    Which VERSION is the student attempting? 
    - If the student's text mentions keywords or logic from Version A, return 0.
    - If the student's text mentions keywords or logic from Version B, return 1.
    
    CRITICAL: You must return ONLY the digit (0 or 1).
    `;

  const model = vertex_ai.getGenerativeModel({ 
        model: 'gemini-2.0-flash-lite', 
        generationConfig: { 
            temperature: 0,
            thinkingConfig: { thinkingBudget: 0 } // Add this line
        }
    });

    try {
        const result = await model.generateContent(prompt);
        const text = result.response.candidates[0].content.parts[0].text.trim();
        const match = text.match(/\d/);
        return match ? parseInt(match[0], 10) : 0;
    } catch (e) {
        return 0;
    }
}

async function resolveOrWinners(questions) {
    const groups = {};

    questions.forEach((q, index) => {
        const id = normalizeForComparison(q.questionNumber);
        if (!groups[id]) groups[id] = [];
        groups[id].push({ data: q, index });
    });

for (const id in groups) {
    let globalWinnerIdx = null;  // ← move INSIDE the loop
    const cluster = groups[id];

        if (cluster.length >= 2) {
            if (globalWinnerIdx === null) {
                const studentText = cluster
                    .map(c => c.data.studentText)
                    .filter(t => t && t.length > 5)
                    .join("\n");

                if (studentText.length > 5) {
                    globalWinnerIdx = await classifyOrWinner(
                        studentText,
                        cluster.map(c => ({ text: c.data.text, modelAnswer: c.data.modelAnswer })),
                        id
                    );
                } else {
                    continue;
                }
            }

            cluster.forEach((item, idx) => {
                if (idx === globalWinnerIdx) {
                    item.data.isOrLoser = false;
                } else {
                    item.data.isOrLoser = true;
                    item.data.studentText = "";
                }
            });
        }
    }
}

async function librarianTagMapper(fullTranscript, questions, subject) {
const model = vertex_ai.getGenerativeModel({
        model: "gemini-2.5-flash",
        generationConfig: {
            temperature: 0,
            responseMimeType: "application/json",
            thinkingConfig: { thinkingBudget: 0 } // Add this line
        }
    });

    const masterIds = questions.map(q => q.questionNumber).join(', ');

    const topicAnchors = questions.map(q => {
        const familyMatch = String(q.questionNumber || "").match(/\d+/);
        const family = familyMatch ? familyMatch[0] : q.questionNumber;
        return {
            id: q.questionNumber,
            family: family,
            anchors: (Array.isArray(q.topicAnchors) && q.topicAnchors.length > 0)
                ? q.topicAnchors
                : (q.answer || "").split(/\W+/).filter(w => w.length > 5).slice(0, 5)

        };
    });

    const normalizedSubject = subject?.toLowerCase().trim() || "";
    const isMaths = MATH_SUBJECTS.includes(normalizedSubject);

    let prompt;

    if (isMaths) {
        prompt = `
You are a Numeric Structural Document Librarian.

Your goal is to map handwriting to Question IDs using STRUCTURAL NUMERIC DOMINANCE.

MASTER QUESTION IDS:
${masterIds}

TOPIC ANCHORS (Secondary Signals Only):
${JSON.stringify(topicAnchors, null, 2)}

TRANSCRIPT:
${fullTranscript}

TASK:
For each Question ID, find the boundary tags that mark where the student's answer starts and ends.

RULES:
1. Every line ends with a unique coordinate tag like [#P:p,y,x].
2. For each Question ID, identify exactly TWO tags:
   - startTag: the coordinate tag on the line where the student FIRST begins writing — this is typically the line containing the student's handwritten question label (e.g. "Q1a", "Ans 2b"). Do NOT pick a line partway through the answer.
   - endTag: the coordinate tag on the LAST line where that question's answer ends.
3. If a question has only one line, startTag and endTag will be the same tag.
4. When in doubt about startTag, go EARLIER rather than later.
5. RETURN ONLY JSON.

# STRUCTURAL MAPPING LOGIC (PRIMARY AUTHORITY)

RULE E — INTRA-LINE SPLIT:
If a single line contains multiple Master IDs (e.g., "1. c 1600 2. a 8"),
split that line structurally. Assign each coordinate tag to the ID immediately preceding it.

RULE F — HARD NUMERIC TERMINATOR:
A "Digit + Dot" (e.g., "2.") creates a HARD BOUNDARY IF:
- It appears at the start of a new line
- It matches a Master Question ID exactly

IGNORE numeric patterns that are: Years, Decimals, Currency, Percentages, Math expressions.

# FLOW RULES
- NO SEQUENTIAL BIAS.
- Students may answer out of order.
- A detected Master ID immediately closes the previous block.
- Page breaks are NEVER boundaries.

# IMPORTANT
Topic Anchors are SECONDARY confirmation signals.
Use them ONLY if numeric structure is unclear.
Do NOT override clear numeric boundaries using anchors.

OUTPUT FORMAT:
{
  "mappings": [
    { "id": "1", "startTag": "[#P:1,120,930]", "endTag": "[#P:1,150,930]" }
  ]
}
`;
    } else {
        prompt = `
You are a structural document librarian. Your goal is to map student handwriting to Question IDs using "Semantic Anchor Dominance."

MASTER QUESTION IDS:
${masterIds}

TOPIC ANCHORS:
${JSON.stringify(topicAnchors, null, 2)} 

TASK:
For each Question ID, find the boundary tags that mark where the student's answer starts and ends.

RULES:
1. Every line in the transcript ends with a unique coordinate tag like [#P:1,250,930].
2. For each Question ID, identify exactly TWO tags:
   - startTag: the coordinate tag on the line where the student FIRST begins writing — this is typically the line containing the student's handwritten question label (e.g. "Ans 1a", "Q3b"). Do NOT pick a line partway through the answer.
   - endTag: the coordinate tag on the LAST line where that question's answer ends.
3. Use semantic anchors (keywords) to identify question boundaries even if handwritten labels are messy.
4. If a question has only one line, startTag and endTag will be the same tag.
5. When in doubt about startTag, go EARLIER rather than later.
6. RETURN ONLY JSON.

# THE HIERARCHY OF TRUTH (MAPPING PROTOCOL)

1. CASE A: THE SEMANTIC OVERRIDE (Source of Truth):
   If a block of text contains 2+ 'topicAnchors' belonging to a specific ID, you MUST map it to that ID, even if the student's label is missing or wrong.

2. CASE B: EXPLICIT LABEL MATCH:
   If the student's handwritten label matches a Master ID and the content does not explicitly contradict it via anchors, map it immediately.

3. CASE C: THE FAMILY CONSTRAINT:
   If a student labels a block generally (e.g., "Ans 6"), you are STRICTLY FORBIDDEN from mapping it to any ID outside of Family 6.

4. CASE D: ORPHAN/UNLABELED TEXT:
   If you encounter text with NO visible label:
   - FIRST: Scan for 'topicAnchors'. If a match is found, start a new block for that ID immediately.
   - SECOND: Only if NO anchors and NO labels are present, treat as continuation of the preceding block.

# BOUNDARY & FLOW LOGIC
- NO SEQUENTIAL BIAS: Students frequently answer out of order.
- ID DOMINANCE: Any Master ID or strong Anchor match creates a hard boundary.

TRANSCRIPT:
${fullTranscript}

OUTPUT FORMAT:
{
  "mappings": [
    { "id": "1a", "startTag": "[#P:1,120,930]", "endTag": "[#P:1,150,930]" },
    { "id": "1b", "startTag": "[#P:1,180,930]", "endTag": "[#P:2,100,930]" }
  ]
}
`;
    }

    const result = await callGeminiWithRetry(model, {
        contents: [{ role: 'user', parts: [{ text: prompt }] }]
    });

    const rawText = result.response.candidates[0].content.parts[0].text;
    return extractJsonFromString(rawText);
}

function normalizeTag(tag) {
    return tag.replace(/\[\s*#P\s*:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]/gi, '[#P:$1,$2,$3]');
}

function sliceByAtomicLines(fullTranscript, mappingJson) {
    if (!mappingJson || !mappingJson.mappings) return {};
    const lines = fullTranscript.split('\n');
    const slices = {};

    mappingJson.mappings.forEach(mapping => {
        const qId = normalizeForComparison(mapping.id);
        const startTag = mapping.startTag ? normalizeTag(mapping.startTag) : null;
        const endTag = mapping.endTag ? normalizeTag(mapping.endTag) : null;

        if (!startTag && !endTag) return;

        let startIdx = -1;
        let endIdx = -1;

        for (let i = 0; i < lines.length; i++) {
            const normalizedLine = normalizeTag(lines[i]);
            if (startTag && startIdx === -1 && normalizedLine.includes(startTag)) {
                startIdx = i;
            }
            if (endTag && normalizedLine.includes(endTag)) {
                endIdx = i;
            }
        }

        // Fallbacks: if only one anchor found, use it for both
        if (startIdx === -1 && endIdx !== -1) startIdx = endIdx;
        if (endIdx === -1 && startIdx !== -1) endIdx = startIdx;

        if (startIdx !== -1 && endIdx !== -1 && startIdx <= endIdx) {
            // Take 1 line before startIdx as buffer — catches cases where Librarian
            // picks the second chunk instead of the first (student's question label line)
            const bufferedStart = Math.max(0, startIdx - 1);
            slices[qId] = lines.slice(bufferedStart, endIdx + 1).join('\n');
        }
    });

    return slices;
}

// ─────────────────────────────────────────────────────────────────────────────
// CORE TRIGGER: processGradingJob
//
// CHANGES from original:
//   FIX #1  — Branch for SaaS API jobs: fetch external images via SSRF-safe fetcher
//   FIX #2  — Save results to api_results + dispatch webhook for SaaS jobs
//   FIX #3  — totalMarks falls back correctly for both PWA and API jobs
//   FIX #6  — Increment quota.currentUsage + write to api_usage_logs after completion
//
// UNCHANGED: All OCR, Librarian, OR-resolution, grading, and report logic.
// ─────────────────────────────────────────────────────────────────────────────

exports.processGradingJob = onDocumentCreated(
    { document: "gradingQueue/{jobId}", timeoutSeconds: 540, memory: "2GiB", region: "asia-south2" },
    async (event) => {
        const snapshot = event.data;
        if (!snapshot) return;
        const jobId = event.params.jobId;
        const jobData = snapshot.data();
        const {
            teacherUid, studentUid, assessmentId, stream,
            filePaths, answerSheetImageUrls,
            questions, strictness, subject, totalMarks
        } = jobData;

        // Assign internal UIDs for collision-safe result matching (UNCHANGED)
        questions.forEach((q, idx) => {
            q._uid = `uid_${idx}_${Date.now()}`;
        });

        const allRules = await fetchGradingRules(subject);
        const bucket = storage.bucket();

        try {
            // ─── FIX #1: Branch image source based on job origin ────────────────────
            let imageParts;

            if (jobData.source === 'API' && answerSheetImageUrls && answerSheetImageUrls.length > 0) {
                // SaaS path: fetch external URLs with SSRF protection.
                // Images loaded into memory only — never written to Cloud Storage.
                console.log(`[Job ${jobId}] SaaS API job — fetching ${answerSheetImageUrls.length} external images`);
                imageParts = await fetchExternalImagesSecurely(answerSheetImageUrls);
            } else {
                // Teacher PWA path: read from internal GCS bucket (UNCHANGED)
                console.log(`[Job ${jobId}] Teacher PWA job — reading from GCS`);
                imageParts = (filePaths || []).map(path => ({
                    fileData: { mimeType: 'image/jpeg', fileUri: `gs://${bucket.name}/${path}` }
                }));
            }

            if (imageParts.length === 0) {
                throw new Error("NO_IMAGES: No answer sheet images found for this job.");
            }

            // ─── OCR (UNCHANGED) ─────────────────────────────────────────────────────
            const pagesResult = await extractTextFromImages(imageParts, subject, jobId, jobData, allRules);
            const fullTranscript = pagesResult.map(p => `[PAGE ${p.pageNum}]\n${p.text}`).join('\n\n');

            await db.collection('gradingQueue').doc(jobId).update({
                statusDetails: `Structuring answers (AI Boundary Marking)...`,
                currentStep: 2,
                progress: 35
            });

            // ─── LIBRARIAN (UNCHANGED) ───────────────────────────────────────────────
            const tagMapping = await librarianTagMapper(fullTranscript, questions, subject);

            const pageMap = new Map();
            questions.forEach(q => pageMap.set(normalizeForComparison(q.questionNumber), new Set()));

            const atomicSlices = sliceByAtomicLines(fullTranscript, tagMapping);

            questions.forEach(q => {
                const key = normalizeForComparison(q.questionNumber);
                q.studentText = atomicSlices[key] || "";

                const slicedContent = atomicSlices[key] || "";
                const pageMatches = [...slicedContent.matchAll(/\[#P:(\d+),/g)];
                pageMatches.forEach(match => pageMap.get(key).add(parseInt(match[1], 10)));
            });

            // ─── OR-RESOLUTION (UNCHANGED) ───────────────────────────────────────────
            await resolveOrWinners(questions);

            // ─── BATCH GRADING (UNCHANGED) ───────────────────────────────────────────
            const MAX_BATCH_WEIGHT = 16;
            const questionBatches = [];
            let currentBatch = [];
            let currentWeight = 0;

            for (const q of questions) {
                if (q.isOrLoser) continue;
                let weight = q.marks > 2 ? 4 : (q.marks === 2 ? 2 : 1);
                if (currentWeight + weight > MAX_BATCH_WEIGHT && currentBatch.length > 0) {
                    questionBatches.push(currentBatch);
                    currentBatch = [];
                    currentWeight = 0;
                }
                currentBatch.push(q);
                currentWeight += weight;
            }
            if (currentBatch.length > 0) questionBatches.push(currentBatch);

            if (questionBatches.length === 0) throw new Error("LIBRARIAN_NO_SLICES");

            let questionWiseReport = [];
            for (let bIdx = 0; bIdx < questionBatches.length; bIdx++) {
                if (bIdx > 0) await sleep(2000);
                const batch = questionBatches[bIdx];
                const batchProgress = Math.round(((bIdx + 1) / questionBatches.length) * 50);
                await snapshot.ref.update({
                    currentStep: 2,
                    progress: 25 + batchProgress,
                    statusDetails: `Grading Questions ${batch[0].questionNumber} to ${batch[batch.length - 1].questionNumber}`
                });
                const batchResult = await gradeQuestionBatch(
                    batch.map(q => `[Q:${q.questionNumber}]\n${q.studentText}`).join('\n\n'),
                    batch, strictness, subject, jobId, allRules
                );
                questionWiseReport.push(...batchResult.map(qr => ({
                    ...qr,
                    stepWiseEvaluation: (qr.stepWiseEvaluation || []).map(step => ({
                        ...step,
                        pageIndex: (step.pageIndex || 1) - 1,
                        stepPoint: (Array.isArray(step.stepPoint) && step.stepPoint.length >= 2)
                            ? [Number(step.stepPoint[step.stepPoint.length - 2]), Number(step.stepPoint[step.stepPoint.length - 1])]
                            : null
                    }))
                })));
            }

            // ─── REPORT RECONSTRUCTION (UNCHANGED) ──────────────────────────────────
            const reconstructedReport = questions.map((originalQ) => {
                const key = normalizeForComparison(originalQ.questionNumber);
                let pageSet = pageMap.get(key) || new Set();
                const pageIndices = Array.from(pageSet).map(n => n - 1).sort((a, b) => a - b);

                if (originalQ.isOrLoser) {
                    return {
                        questionNumber: originalQ.questionNumber,
                        marksAwarded: 0,
                        maxMarksForQuestion: originalQ.marks,
                        finalFeedback: "Alternative choice attempted. This version was not selected by the student.",
                        studentOcrAnswer: "[Alternative Version]",
                        answerPageIndex: pageIndices.length > 0 ? pageIndices[0] : 0,
                        requiresReview: false,
                        stepWiseEvaluation: []
                    };
                }

                const gradedResult = questionWiseReport.find(r => r._uid === originalQ._uid);

                if (gradedResult) {
                    return {
                        ...gradedResult,
                        studentOcrAnswer: originalQ.studentText,
                        answerPageIndex: pageIndices[0] || 0,
                        answerPageIndices: pageIndices
                    };
                }

                return {
                    questionNumber: originalQ.questionNumber,
                    marksAwarded: 0,
                    maxMarksForQuestion: originalQ.marks,
                    finalFeedback: "Question not detected.",
                    studentOcrAnswer: "No text found for this question.",
                    answerPageIndex: 0,
                    requiresReview: true,
                    stepWiseEvaluation: []
                };
            });

            // ─── FIX #3: totalMarks safe for both PWA and SaaS ───────────────────────
            const computedTotalMarks = Number(totalMarks) ||
                questions.reduce((sum, q) => sum + (Number(q.marks) || 0), 0) || 0;

            const reportForStudent = {
                studentName: jobData.studentName || "Student",
                studentUid,
                stream: stream || "",
                overallScore: reconstructedReport.reduce((sum, qr) => sum + (qr.marksAwarded || 0), 0),
                maximumMarks: computedTotalMarks,
                overallFeedback: {
                    summary: "Grading complete.",
                    areasForImprovement: reconstructedReport.flatMap(qr => qr.improvementArea || [])
                },
                answerSheetImageUrls,
                questionWiseReport: reconstructedReport,
                fullOcrText: fullTranscript,
                gradingTimestamp: admin.firestore.FieldValue.serverTimestamp(),
                assessmentId,
                subject
            };

            await snapshot.ref.update({ status: 'SUCCESS', currentStep: 4, progress: 100, statusDetails: 'Report Saved!' });

            // ─── FIX #2: Route result storage based on job origin ────────────────────
            if (jobData.source === 'API') {
                // SaaS path: save to api_results so v1/result can retrieve it
                await db.collection('api_results').doc(jobId).set({
                    clientId: teacherUid, // clientId was stored as teacherUid during v1_grade
                    studentRef: studentUid,
                    report: cleanUndefined(reportForStudent),
                    createdAt: admin.firestore.FieldValue.serverTimestamp()
                });

                // ─── FIX #6: Increment quota usage + write to billing ledger ──────────
                const pagesProcessed = (answerSheetImageUrls || []).length || 1;
                await db.collection('api_clients').doc(teacherUid).update({
                    'quota.currentUsage': admin.firestore.FieldValue.increment(pagesProcessed)
                });
                await db.collection('api_usage_logs').add({
                    clientId: teacherUid,
                    jobId,
                    studentRef: studentUid,
                    pagesProcessed,
                    timestamp: admin.firestore.FieldValue.serverTimestamp(),
                    overallScore: reportForStudent.overallScore,
                    maximumMarks: reportForStudent.maximumMarks
                });

                // Phase 5: Push result to ERP via Webhook
                await dispatchWebhook(teacherUid, {
                    event: 'grading.completed',
                    jobId,
                    studentRef: studentUid,
                    report: cleanUndefined(reportForStudent)
                });

                console.log(`[Job ${jobId}] SaaS job complete — result stored in api_results`);

            } else {
                // Teacher PWA path: save to the teacher's existing collection (UNCHANGED)
                await db.collection('teachers')
                    .doc(teacherUid)
                    .collection('assessmentHistory')
                    .doc(assessmentId)
                    .collection('submissions')
                    .doc(studentUid)
                    .set(cleanUndefined(reportForStudent));

                console.log(`[Job ${jobId}] PWA job complete — result stored in teacher assessmentHistory`);
            }

            // Queue doc cleanup — same for both paths
            await snapshot.ref.delete();

        } catch (error) {
            console.error(`❌ Grading Job ${jobId} Failed:`, error);
            await snapshot.ref.update({
                status: 'ERROR',
                statusDetails: error.message.includes('LIBRARIAN_') ? 'Stopped: Structural Error'
                    : error.message.includes('SSRF_BLOCK') ? 'Stopped: Security Block on Image URL'
                    : error.message.includes('FETCH_FAIL') ? 'Stopped: Could not download student images'
                    : 'Grading Failed',
                error: error.message,
                finishedAt: admin.firestore.FieldValue.serverTimestamp()
            });
        }
    }
);

// ─────────────────────────────────────────────────────────────────────────────
// v1/grade — Queue a SaaS grading job
//
// CHANGES from original:
//   FIX #3  — Added filePaths: [] and totalMarks to prevent destructuring crash
//   FIX #5  — Added validateAndNormalizeQuestions before queuing
//   FIX #7  — Atomic Firestore transaction prevents idempotency race condition
// ─────────────────────────────────────────────────────────────────────────────

exports.v1_grade = onRequest({ region: "asia-south2", memory: "1GiB", timeoutSeconds: 540 }, async (req, res) => {
    if (req.method !== 'POST') return res.status(405).json({ error: "Method Not Allowed" });

    try {
        // 1. Gatekeeper
        const auth = await validateSaaSRequest(req, 'scope:submit_grading');

        if (auth.isDuplicate) {
            return res.status(200).json({
                status: 'EXISTING',
                message: "Duplicate request detected via Idempotency-Key",
                jobId: auth.cachedResponse?.jobId
            });
        }

        const body = req.body;

        // 2. FIX #5: Validate + normalize the incoming question schema
        let normalizedQuestions;
        try {
            normalizedQuestions = validateAndNormalizeQuestions(body.questions_json);
        } catch (validationErr) {
            return res.status(400).json({ error: validationErr.message });
        }

        // 3. Basic payload validation
        if (!body.student_ref_id) {
            return res.status(400).json({ error: "Invalid payload: student_ref_id is required." });
        }
        if (!Array.isArray(body.scan_urls) || body.scan_urls.length === 0) {
            return res.status(400).json({ error: "Invalid payload: scan_urls must be a non-empty array of HTTPS URLs." });
        }

 const internalTicket = {
            // ── Identity & ownership ─────────────────────────────────────────
            teacherUid:   auth.clientId,      // clientId is the "owner" in SaaS mode
            studentUid:   body.student_ref_id,
            studentName:  body.student_name   || "SaaS Student",
            assessmentId: body.exam_id        || `api_${Date.now()}`,

            // ── Answer sheets ────────────────────────────────────────────────
            answerSheetImageUrls: body.scan_urls,
            filePaths: [],                    // Required by processGradingJob destructuring

            // ── Grading parameters ───────────────────────────────────────────
            questions:   normalizedQuestions,
            strictness:  body.strictness      || 'Moderate',
            subject:     body.subject         || 'General',
            totalMarks:  Number(body.total_marks) ||
                         normalizedQuestions.reduce((s, q) => s + (q.marks || 0), 0),

            // ── ADDED: Fields that were silently dropped ──────────────────────

            // board is used by the grading system to apply board-specific logic
            // (CBSE vs ICSE vs State Board marking conventions).
            // It was being received in body.board but never stored in the ticket,
            // so processGradingJob always had an undefined board.
            board:       body.board           || 'CBSE',

            // stream is ALREADY READ by processGradingJob as jobData.stream
            // (line: "const { ..., stream, ... } = jobData;")
            // but was never written to the ticket. Always resulted in undefined.
            // This caused the stream field in reports to always be empty string "".
            stream:      body.stream          || "",

            // These metadata fields are stored so the grading report has full
            // context when the ERP retrieves it from /v1/result. Without them,
            // the ERP can't display who set the paper or which school it's for.
            schoolName:             body.school_name            || "",
            teacherName:            body.teacher_name           || "",
            section:                body.section                || "",
            assessmentTitle:        body.assessment_title       || "",
            selectedChapters:       body.selected_chapters      || [],
            generalInstructions:    body.general_instructions   || "",

            // ── System flags ─────────────────────────────────────────────────
            source:    'API',
            status:    'PENDING',
            createdAt: admin.firestore.FieldValue.serverTimestamp()
        };


        // 5. FIX #7: Atomic transaction — write idempotency log AND queue doc together
        // This prevents duplicate jobs if the ERP retries before the first write completes
        const jobRef = db.collection('gradingQueue').doc();
        const idempotencyKey = auth.idempotencyKey;
        const idenDocRef = idempotencyKey
            ? db.collection('api_idempotency_log').doc(`${auth.clientId}_${idempotencyKey}`)
            : null;

        await db.runTransaction(async (t) => {
            if (idenDocRef) {
                const existing = await t.get(idenDocRef);
                if (existing.exists) {
                    // Another request got here first — surface the cached job ID
                    throw Object.assign(new Error('DUPLICATE_IN_TRANSACTION'), {
                        cachedJobId: existing.data().cachedResponse?.jobId
                    });
                }
                t.set(idenDocRef, {
                    status: 'PENDING',
                    cachedResponse: { jobId: jobRef.id },
                    expiresAt: admin.firestore.Timestamp.fromDate(new Date(Date.now() + 86400000))
                });
            }
            t.set(jobRef, internalTicket);
        }).catch(err => {
            if (err.message === 'DUPLICATE_IN_TRANSACTION') {
                // Propagate as a special duplicate signal
                err.isDuplicate = true;
                throw err;
            }
            throw err;
        });

        res.status(202).json({
            status: 'ACCEPTED',
            jobId: jobRef.id,
            message: "Grading job successfully queued. Poll /v1/result or await webhook."
        });

    } catch (error) {
        if (error.isDuplicate) {
            return res.status(200).json({
                status: 'EXISTING',
                message: "Duplicate request detected via Idempotency-Key",
                jobId: error.cachedJobId
            });
        }
        console.error("[v1/grade] Error:", error.message);
        const statusCode = error.message.match(/^(4\d\d)/) ? parseInt(error.message.substring(0, 3)) : 500;
        res.status(statusCode).json({ error: error.message });
    }
});

// ─────────────────────────────────────────────────────────────────────────────
// v1/digitize — OCR a question paper and return structured JSON
//
// FIX #4: Now uses the full QUESTION_EXTRACTION_SYSTEM_PROMPT (same brain as
//         Teacher PWA), not a stripped-down one-liner.
// ─────────────────────────────────────────────────────────────────────────────

exports.v1_digitize = onRequest({ region: "asia-south2", memory: "2GiB", timeoutSeconds: 540 }, async (req, res) => {
    if (req.method !== 'POST') return res.status(405).json({ error: "Method Not Allowed" });

    try {
        const auth = await validateSaaSRequest(req, 'scope:digitize_paper');

        if (auth.isDuplicate) {
            return res.status(200).json({
                status: 'EXISTING',
                message: "Duplicate request detected via Idempotency-Key",
                result: auth.cachedResponse?.result
            });
        }

        const {
            images_base64,
            images_urls,
            solution_base64,    // optional: solution key images as base64
            solution_urls,      // optional: solution key images as HTTPS URLs
            subject,
            custom_topic,
            total_marks,
            general_instructions
        } = req.body;

        // ── 1. Build question paper image parts ──────────────────────────────
        let questionImageParts = [];

        if (Array.isArray(images_base64) && images_base64.length > 0) {
            questionImageParts = images_base64.map((b64, idx) => {
                if (typeof b64 !== 'string' || b64.length === 0) {
                    throw new Error(`400: images_base64[${idx}] is not a valid base64 string`);
                }
                return { inlineData: { mimeType: 'image/jpeg', data: b64 } };
            });
} else if (typeof req.body.question_pdf_url === 'string' && req.body.question_pdf_url.length > 0) {
    const qPdfUrl = req.body.question_pdf_url;
    if (!qPdfUrl.startsWith('https://')) throw new Error('400: question_pdf_url must be HTTPS');
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30000);
    let qPdfResponse;
    try {
        qPdfResponse = await fetch(qPdfUrl, { signal: controller.signal });
    } finally {
        clearTimeout(timeout);
    }
    if (!qPdfResponse.ok) throw new Error(`400: Could not fetch question PDF. HTTP ${qPdfResponse.status}`);
    const qPdfBuffer = await qPdfResponse.arrayBuffer();
    const qPdfBase64 = Buffer.from(qPdfBuffer).toString('base64');
    questionImageParts = [{ inlineData: { mimeType: 'application/pdf', data: qPdfBase64 } }];
} else if (Array.isArray(images_urls) && images_urls.length > 0) {
    questionImageParts = await fetchExternalImagesSecurely(images_urls);
} else {
    return res.status(400).json({
        error: "Provide question_pdf_url, images_base64, or images_urls for the question paper."
    });
}

// ── 2. Build solution key image parts (optional) ─────────────────────
let solutionImageParts = [];

if (Array.isArray(solution_base64) && solution_base64.length > 0) {
    solutionImageParts = solution_base64.map((b64, idx) => {
        if (typeof b64 !== 'string' || b64.length === 0) {
            throw new Error(`400: solution_base64[${idx}] is not a valid base64 string`);
        }
        return { inlineData: { mimeType: 'image/jpeg', data: b64 } };
    });
} else if (typeof req.body.solution_pdf_url === 'string' && req.body.solution_pdf_url.length > 0) {
    // PDF URL path — fetch the PDF and send to Gemini as a document
    const pdfUrl = req.body.solution_pdf_url;
    if (!pdfUrl.startsWith('https://')) {
        throw new Error('400: solution_pdf_url must be an HTTPS URL');
    }
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30000); // 30s for PDF
    let pdfResponse;
    try {
        pdfResponse = await fetch(pdfUrl, { signal: controller.signal });
    } finally {
        clearTimeout(timeout);
    }
    if (!pdfResponse.ok) {
        throw new Error(`400: Could not fetch solution PDF. HTTP ${pdfResponse.status}. Make sure the file is publicly accessible.`);
    }
    const pdfBuffer = await pdfResponse.arrayBuffer();
    const pdfBase64 = Buffer.from(pdfBuffer).toString('base64');
    solutionImageParts = [{
        inlineData: {
            mimeType: 'application/pdf',
            data: pdfBase64
        }
    }];
} else if (Array.isArray(solution_urls) && solution_urls.length > 0) {
    solutionImageParts = await fetchExternalImagesSecurely(solution_urls);
}


        // ── 3. Dynamic solution instruction — mirrors frontend exactly ────────
        // This is the key switch: if a solution key is provided, the model copies
        // it verbatim. If not, it generates concise model answers itself.
        const solutionInstruction = solutionImageParts.length > 0
            ? `**Solution Key Provided:** You have also been given images of the solution key. You MUST use these images to determine the correct "answer" for each question you extract. The "answer" in your JSON output must come from this key. You MUST perform a VERBATIM transcription of the solution for each question. Do NOT summarize, do NOT shorten, and do NOT change any text from the solution key. Your goal is a 100% identical copy of what is written in the solution key images for the "answer" field.`
            : `**No Solution Key:** A solution key was not provided. You MUST generate a reasonable and correct model "answer" for each question yourself. IT MUST BE VERY SHORT AND CONCISE AND TO THE POINT. DONT WRITE LONG SENTENCES.`;

        // ── 4. Build full prompt — base rules + dynamic solution instruction ──
        // QUESTION_EXTRACTION_SYSTEM_PROMPT has all the extraction rules.
        // We append the solution instruction and conciseness rules dynamically,
        // exactly as the frontend does in its prompt template literal.
        const contextTitle = custom_topic || subject || 'General Assessment';
const fullPrompt = `${QUESTION_EXTRACTION_SYSTEM_PROMPT}

**Answer Handling:** ${solutionInstruction}

**Context:** Title: "${contextTitle}"

Begin extracting ALL questions now.`;
        // All parts: question images first, then solution images, then prompt text.
        // This exactly mirrors: [...questionImageParts, ...solutionImageParts, { text: prompt }]
        const allParts = [
            ...questionImageParts,
            ...solutionImageParts,
            { text: fullPrompt }
        ];

let finalUsage = null;

// ── 5. Call Gemini (non-streaming) ───────────────────────────────────
const digitizeModel = vertex_ai.getGenerativeModel({ model: 'gemini-2.5-flash' });

const digitizeResult = await callGeminiWithRetry(digitizeModel, {
    contents: [{ role: 'user', parts: allParts }],
    generationConfig: {
        temperature: 0.1,
        thinkingConfig: { thinkingBudget: 0 },
        maxOutputTokens: 8192
    }
});

const rawText = digitizeResult.response.candidates[0].content.parts[0].text || '';
finalUsage = digitizeResult.response.usageMetadata;

// ── 6. Parse JSON objects from response ──────────────────────────────
const questions = [];
let buffer = rawText.replace(/```json|```/g, '');
let questionCounter = 0;

let objStartIndex = buffer.indexOf('{');
while (objStartIndex !== -1) {
    let braceCount = 1;
    let objEndIndex = -1;
    let inString = false;

    for (let i = objStartIndex + 1; i < buffer.length; i++) {
        const char = buffer[i];
        if (char === '"') {
            let backslashCount = 0;
            let j = i - 1;
            while (j >= objStartIndex && buffer[j] === '\\') { backslashCount++; j--; }
            if (backslashCount % 2 === 0) inString = !inString;
        }
        if (!inString) {
            if (char === '{') braceCount++;
            else if (char === '}') braceCount--;
        }
        if (braceCount === 0) { objEndIndex = i; break; }
    }

    if (objEndIndex !== -1) {
        const jsonStr = buffer.substring(objStartIndex, objEndIndex + 1);
        const parsed = extractJsonFromString(jsonStr);
        if (parsed) {
            questions.push({
                ...parsed,
                id: `q_${Date.now()}_${questionCounter++}`,
                topicAnchors: Array.isArray(parsed.topicAnchors) ? parsed.topicAnchors : [],
            });
        }
        buffer = buffer.substring(objEndIndex + 1);
        objStartIndex = buffer.indexOf('{');
    } else {
        break;
    }
}

        // ── 7. Log usage + increment quota ────────────────────────────────────
        const inputTokens = finalUsage?.promptTokenCount || 0;
        const outputTokens = finalUsage?.candidatesTokenCount || 0;
        const totalTokens = finalUsage?.totalTokenCount || 0;

        await db.collection('api_usage_logs').add({
            clientId: auth.clientId,
            feature: 'digitization',
            hasSolutionKey: solutionImageParts.length > 0,
            timestamp: admin.firestore.FieldValue.serverTimestamp(),
            tokenUsage: { inputTokens, outputTokens, totalTokens },
            pagesProcessed: questionImageParts.length + solutionImageParts.length,
            questionsExtracted: questions.length
        });

        await db.collection('api_clients').doc(auth.clientId).update({
            'quota.currentUsage': admin.firestore.FieldValue.increment(questionImageParts.length)
        });

 // Generate a deterministic assessment ID for this paper.
        // The ERP stores this as the "exam template ID" and sends it back
        // in the exam_id field of /v1/grade requests.
        const assessmentId = `assess_${auth.clientId}_${Date.now()}`;

        // Optionally persist the digitized pattern in Firestore so it can
        // be retrieved later without re-digitizing the paper.
        // This is low-cost storage and makes the ERP's life easier.
        await db.collection('api_assessment_patterns').doc(assessmentId).set({
            clientId:       auth.clientId,
            assessmentId,
            assessmentTitle: req.body.assessment_title  || req.body.custom_topic || subject || 'Untitled Assessment',
            board:           req.body.board             || "",
            className:       req.body.class_name        || "",
            subject:         subject                    || "",
            totalMarks:      questions.reduce((s, q) => s + (Number(q.marks) || 0), 0),
            checkingStrictness: req.body.checking_strictness || 'Moderate',
            section:         req.body.section           || "",
            stream:          req.body.stream            || "",
            schoolName:      req.body.school_name       || "",
            teacherName:     req.body.teacher_name      || "",
            questions,
            hasSolutionKey:  solutionImageParts.length > 0,
            createdAt:       admin.firestore.FieldValue.serverTimestamp(),
        totalMarks: Number(total_marks) || questions.reduce((s, q) => s + (Number(q.marks) || 0), 0),
generalInstructions: general_instructions || "",
        });

        res.status(200).json({
            status:             'OK',
            // ── ADDED: Top-level envelope so ERP has a self-contained template ──
            // Previously the response only had status/questionsExtracted/questions.
            // The ERP had to remember board, class, subject separately.
            // Now the response IS the complete assessment pattern — store it as-is.
            assessmentId,
            assessmentTitle:    req.body.assessment_title    || req.body.custom_topic || subject || 'Untitled Assessment',
            board:              req.body.board               || "",
            className:          req.body.class_name          || "",
            subject:            subject                      || "",
            checkingStrictness: req.body.checking_strictness || 'Moderate',
            section:            req.body.section             || "",
            stream:             req.body.stream              || "",
            schoolName:         req.body.school_name         || "",
            teacherName:        req.body.teacher_name        || "",
            // ── Original fields (unchanged) ──────────────────────────────────
            questionsExtracted: questions.length,
            totalMarks:         questions.reduce((s, q) => s + (Number(q.marks) || 0), 0),
            hasSolutionKey:     solutionImageParts.length > 0,
            questions,
            generalInstructions: general_instructions || "",
        });

    } catch (error) {
        console.error("[v1/digitize] Error:", error.message);
        const statusCode = error.message.match(/^(4\d\d)/) ? parseInt(error.message.substring(0, 3)) : 500;
        res.status(statusCode).json({ error: error.message });
    }
});

// ─────────────────────────────────────────────────────────────────────────────
// v1/result — Poll for a grading job result
//
// FIX #8: Handles ERROR state properly so ERPs get meaningful failure messages.
// ─────────────────────────────────────────────────────────────────────────────

exports.v1_result = onRequest({ region: "asia-south2", timeoutSeconds: 30 }, async (req, res) => {
    if (req.method !== 'GET') return res.status(405).json({ error: "Method Not Allowed" });

    try {
        const auth = await validateSaaSRequest(req, 'scope:read_results');

        if (auth.isDuplicate) {
            return res.status(200).json({ status: 'CACHED', jobId: auth.cachedResponse?.jobId });
        }

        const jobId = req.query.jobId;
        if (!jobId) return res.status(400).json({ error: "jobId query parameter is required. Example: ?jobId=abc123" });

        // Check the live queue first (job still in progress)
        const jobDoc = await db.collection('gradingQueue').doc(jobId).get();

        if (jobDoc.exists) {
            const jobData = jobDoc.data();

            // Security: this client must own the job
            if (jobData.teacherUid !== auth.clientId) {
                return res.status(403).json({ error: "Access denied: this job does not belong to your API key." });
            }

            // FIX #8: Expose error state properly
            if (jobData.status === 'ERROR') {
                return res.status(200).json({
                    status: 'ERROR',
                    message: jobData.statusDetails || 'Grading failed',
                    jobId
                });
            }

            return res.status(200).json({
                status: jobData.status || 'PENDING',
                progress: jobData.progress || 0,
                statusDetails: jobData.statusDetails || '',
                jobId
            });
        }

        // Job not in queue — check completed results
        const resultDoc = await db.collection('api_results').doc(jobId).get();

        if (!resultDoc.exists) {
            return res.status(404).json({
                status: 'NOT_FOUND',
                message: `No job found with id: ${jobId}. It may have expired or never existed.`
            });
        }

        const data = resultDoc.data();

        // Security: this client must own the result
        if (data.clientId !== auth.clientId) {
            return res.status(403).json({ error: "Access denied: this result does not belong to your API key." });
        }

        return res.status(200).json({
            status: 'COMPLETED',
            jobId,
            report: data.report
        });

    } catch (error) {
        console.error("[v1/result] Error:", error.message);
        const statusCode = error.message.match(/^(4\d\d)/) ? parseInt(error.message.substring(0, 3)) : 500;
        res.status(statusCode).json({ error: error.message });
    }
});