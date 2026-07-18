'use strict';

const { onDocumentCreated } = require("firebase-functions/v2/firestore");
const { onRequest } = require("firebase-functions/v2/https");
const { onSchedule } = require("firebase-functions/v2/scheduler");
const { VertexAI } = require('@google-cloud/vertexai');
const admin = require('firebase-admin');
const crypto = require('crypto');
const { Resolver } = require('dns').promises;
const sharp = require('sharp');

// ─────────────────────────────────────────────────────────────────────────────
// INITIALIZATION
// ─────────────────────────────────────────────────────────────────────────────

admin.initializeApp();
let vertex_ai = new VertexAI({ project: process.env.GCLOUD_PROJECT, location: 'us-central1' });


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

/**
 * Returns true if the subject is Hindi (any casing: "Hindi", "HINDI", "hindi", "h", "H").
 * Used to inject Hindi-specific OCR and grading instructions.
 */
function isHindiSubject(subject) {
    if (!subject) return false;
    const s = subject.trim().toLowerCase();
    return s === 'hindi' || s === 'h';
}

// ─────────────────────────────────────────────────────────────────────────────
// LANGUAGE-BASED HELPERS (NEW)
// The "answerLanguage" field comes from the PWA "Answer Language:" dropdown.
// A student may write Physics in Hindi OR English — gating must be on
// LANGUAGE, not subject. Default (missing) = English → preserves v42 behaviour.
// ─────────────────────────────────────────────────────────────────────────────
function isHindiLanguage(jobData) {
    if (!jobData) return false;
    const lang = (jobData.answerLanguage || '').trim().toLowerCase();
    if (lang === 'hindi' || lang === 'hi') return true;
    // Fallback: if subject itself is Hindi and no answerLanguage set, treat as Hindi
    const subj = (jobData.subject || '').trim().toLowerCase();
    return (subj === 'hindi' || subj === 'h') && lang === '';
}
function isEnglishLanguage(jobData) {
    if (!jobData) return true; // missing → assume English (v42 default)
    const lang = (jobData.answerLanguage || '').trim().toLowerCase();
    return lang === '' || lang === 'english' || lang === 'en';
}

// ─────────────────────────────────────────────────────────────────────────────
// SUBJECT-BASED HELPERS (NEW) — for content-type OCR/grading addons
// ─────────────────────────────────────────────────────────────────────────────
function isProofSubject(subject) {
    if (!subject) return false;
    const s = subject.trim().toLowerCase();
    return ['physics','chemistry','biology','mathematics','maths','math','science'].includes(s);
}
function isAccountsSubject(subject) {
    if (!subject) return false;
    const s = subject.trim().toLowerCase();
    return s.includes('account') || s.includes('accountancy') ||
           s.includes('bookkeeping') || s.includes('book keeping') ||
           s.includes('commerce') || s.includes('cma') ||
           s.includes('sfm') || s.includes('partnership') ||
           s.includes('financial management') || s.includes('costing') ||
           s.includes('audit') || s.includes('taxation') ||
           s.includes('finance') || s.includes('ca inter') ||
           s.includes('ca final') || s.includes('icai');
}
function isDataTableSubject(subject) {
    if (!subject) return false;
    const s = subject.trim().toLowerCase();
    return ['economics','statistics','business studies','mathematics','maths','math'].includes(s);
}



// ─────────────────────────────────────────────────────────────────────────────
// FEATURE FLAGS — OFF by default → English papers behave EXACTLY like v42.
// Flip individually to true to re-enable each addon after testing.
// ─────────────────────────────────────────────────────────────────────────────
const ENABLE_PROOF_ADDON           = false;
const ENABLE_ACCOUNTS_OCR_ADDON    = true;
const ENABLE_DATATABLE_OCR_ADDON   = false;
const ENABLE_ACCOUNTS_GRADE_ADDON  = true;
const ENABLE_DATATABLE_GRADE_ADDON = false;
const ENABLE_TABLE_MIXED_ADDON     = true;


// AFTER
function extractJsonFromString(text) {
    if (!text || typeof text !== 'string') return null;
    let targetText = text.trim();
    // Strip Gemini thinking blocks that leak through despite thinkingBudget:0
    targetText = targetText.replace(/<thinking>[\s\S]*?<\/thinking>/gi, '').trim();

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

    // 1. Try direct parse first
    try { return JSON.parse(cleaned); } catch (e1) {}

    // 2. Repair bad escaping inside "text" field value.
    // Physics LaTeX causes Gemini to emit unescaped \n, \t, \r and single backslashes
    // (e.g. \frac instead of \\frac), making the JSON string invalid.
    // Fix: walk to find the raw value, patch it, rebuild.
    try {
        const keyIdx = cleaned.search(/"text"\s*:\s*"/);
        if (keyIdx !== -1) {
            const valStart = cleaned.indexOf('"', cleaned.indexOf(':', keyIdx)) + 1;
            let i = valStart;
            while (i < cleaned.length) {
                if (cleaned[i] === '\\') { i += 2; continue; }
                if (cleaned[i] === '"') break;
                i++;
            }
            const rawVal = cleaned.substring(valStart, i);
            const fixed = rawVal
                .replace(/\n/g, '\\n')
                .replace(/\t/g, '\\t')
                .replace(/\r/g, '\\r')
                .replace(/\\(?!["\\/bfnrtu])/g, '\\\\');
            return JSON.parse('{"text":"' + fixed + '"}');
        }
    } catch (e2) {}

    // 3. Substring fallback
    const firstBrace = cleaned.indexOf('{');
    const firstBracket = cleaned.indexOf('[');
    let startIndex = (firstBrace === -1) ? firstBracket : (firstBracket === -1) ? firstBrace : Math.min(firstBrace, firstBracket);
    let endIndex = Math.max(cleaned.lastIndexOf(']'), cleaned.lastIndexOf('}'));
    if (startIndex === -1 || endIndex === -1) return null;
    try {
        return JSON.parse(cleaned.substring(startIndex, endIndex + 1));
    } catch (e3) {
        return null;
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

// ─────────────────────────────────────────────────────────────────────────────
// POST-OCR ORPHAN QLABEL SUPPRESSOR
//
// Problem: OCR emits [QLABEL:?] at "soft" structural shifts inside a single
// answer — e.g. after a diagram block, before a proof continuation, or at
// "Hence proved." The character-position slicer treats EVERY [QLABEL:] as a
// hard boundary, so [QLABEL:?] mid-answer silently severs the proof from the
// parent question. The grader then sees an incomplete answer and deducts marks.
//
// Fix: Remove [QLABEL:?] when it appears BETWEEN two real [QLABEL:N] markers
// (i.e. it is inside a known question's territory). Only keep [QLABEL:?] at
// the very end of the transcript, where it might be a genuine unknown question.
//
// Safe: if the transcript has NO real [QLABEL:N] before the [QLABEL:?], we
// keep it — it may be the only boundary signal for UPSC-style prose answers.
// ─────────────────────────────────────────────────────────────────────────────
function suppressOrphanQLabels(text) {
    if (!text) return text;
    // Split on [QLABEL:?] boundaries
    const parts = text.split(/\[QLABEL:\?\]/g);
    if (parts.length <= 1) return text; // no [QLABEL:?] present — nothing to do

    // Rebuild: for each [QLABEL:?] occurrence, decide whether to keep or drop it.
    // Rule: drop if a real [QLABEL:X] (where X is not '?') appears BEFORE this point.
    const realQLabelRe = /\[QLABEL:(?!\?)[^\]]+\]/;
    let rebuilt = parts[0];
    for (let i = 1; i < parts.length; i++) {
        const textSoFar = rebuilt;
        const hasRealLabelBefore = realQLabelRe.test(textSoFar);
        if (hasRealLabelBefore) {
            // Drop the [QLABEL:?] — it is mid-answer continuation noise
            rebuilt += parts[i];
        } else {
            // Keep it — no real label seen yet, this might be UPSC prose boundary
            rebuilt += '[QLABEL:?]' + parts[i];
        }
    }
    return rebuilt;
}


const MAX_OCR_DIM = 1600;

async function downscaleImagePart(part) {
    // GCS fileData URI — download it, downscale it, return as inlineData
    if (part?.fileData?.fileUri) {
        try {
            const bucket = storage.bucket();
            const gcsPath = part.fileData.fileUri.replace(`gs://${bucket.name}/`, '');
            const [buffer] = await bucket.file(gcsPath).download();
            const meta = await sharp(buffer).metadata();
            const w = meta.width || 0;
            const h = meta.height || 0;

            let finalBuffer = buffer;
            if (w > MAX_OCR_DIM || h > MAX_OCR_DIM) {
finalBuffer = await sharp(buffer)
                    .resize(MAX_OCR_DIM, MAX_OCR_DIM, { fit: 'inside', withoutEnlargement: true })
                    .normalise()
                    .jpeg({ quality: 85 })
                    .toBuffer();
            }
            return { inlineData: { mimeType: 'image/jpeg', data: finalBuffer.toString('base64') } };
        } catch (e) {
            console.warn(`[downscale] GCS download/resize failed, using original fileData: ${e.message}`);
            return part; // fall back to original GCS URI — never break OCR
        }
    }

    // inlineData image (API path / SaaS path)
    if (part?.inlineData?.data) {
        const mime = part.inlineData.mimeType || '';
        if (mime === 'application/pdf') return part;
        try {
            const buf = Buffer.from(part.inlineData.data, 'base64');
            const meta = await sharp(buf).metadata();
            const w = meta.width || 0;
            const h = meta.height || 0;

            if (w <= MAX_OCR_DIM && h <= MAX_OCR_DIM) return part; // already small

const resized = await sharp(buf)
                .resize(MAX_OCR_DIM, MAX_OCR_DIM, { fit: 'inside', withoutEnlargement: true })
                .normalise() // auto-stretch contrast — fixes faint/washed-out ink
                .jpeg({ quality: 85 })
                .toBuffer();
            return { inlineData: { mimeType: 'image/jpeg', data: resized.toString('base64') } };
        } catch (e) {
            console.warn(`[downscale] inlineData resize failed, using original: ${e.message}`);
            return part;
        }
    }

    return part; // placeholder or unknown — pass through
}

function normalizeForComparison(str) {
    if (!str) return '';
    return String(str).toLowerCase()
        .replace(/\b(ans|answer|q|question|sol|solution|pt|part|alt|alternative)\b/gi, '')
        .replace(/[^a-z0-9\u0900-\u097F]/g, '')
        // Strip any leading sequence of q's before a digit.
        // "QQ1"→"1", "Q1"→"1". Handles Teacher PWA IDs (QQ1-QQ30) and
        // API IDs (Q1, Q2). Without this, QLABEL "1)" never matches master "QQ1".
        .replace(/^q+(?=\d)/, '')
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

async function callGeminiWithRetry(model, request, retries = 2) {
    for (let i = 0; i <= retries; i++) {
        try {
            return await model.generateContent(request);
        } catch (error) {
            const errStr = error.message ? error.message.toLowerCase() : "";
            const isRateLimit = errStr.includes('429') || errStr.includes('too many requests');
            if (isRateLimit && i < retries) {
                 const delay = Math.pow(2, i) * 2000 + Math.random() * 3000;
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

// ─────────────────────────────────────────────────────────────────────────────
// HINDI-SPECIFIC OCR INSTRUCTION ADDON
// Injected into the OCR system instruction ONLY when isHindiSubject() is true.
// Does NOT replace the base OCR_SYSTEM_INSTRUCTION — it is appended after it.
// All existing English/Maths OCR rules remain intact for non-Hindi subjects.
// ─────────────────────────────────────────────────────────────────────────────
const HINDI_OCR_ADDON = `
# ═══════════════════════════════════════════════════════════════════════
# HINDI SUBJECT — SPECIAL OCR RULES (ACTIVE FOR THIS PAPER)
# ═══════════════════════════════════════════════════════════════════════

## DEVANAGARI SCRIPT HANDLING

This paper is written in Hindi (Devanagari script). Apply all standard rules above AND the following Hindi-specific rules.

## QUESTION LABEL RECOGNITION — HINDI PAPERS

Hindi question papers use क, ख, ग, घ, ङ as sub-part labels (equivalent to a, b, c, d, e in English).
They also use (क), (ख), (ग) in parentheses for MCQ options or sub-parts.

CRITICAL: You MUST emit [QLABEL] for Hindi sub-part labels exactly like English ones:

Examples:
  Student wrote "क." at start of answer → emit [QLABEL:क.]
  Student wrote "(क)" at start of answer block → emit [QLABEL:(क)]
  Student wrote "ख)" → emit [QLABEL:ख)]
  Student wrote "1." in margin then sub-part "(क)" → emit both:
    1. [QLABEL:1.] ... [#P:p,y,x]
    (क) [QLABEL:(क)] ... [#P:p,y,x]

DO NOT confuse MCQ option labels (क, ख, ग, घ) mid-answer with question boundary labels.
Apply the same KEY TEST: a [QLABEL] is only valid if it appears at the START of a new answer block, not mid-sentence.

## COMPOUND CHARACTER (संयुक्ताक्षर) TRANSCRIPTION

Devanagari compound characters must be transcribed exactly as written.
DO NOT split compound characters into their component parts.

Common compounds to recognize accurately:
  क्ष (not कष or क + ष)
  त्र (not त + र)
  ज्ञ (not ज + ञ)
  श्र (not श + र)
  र्थ (not र + थ)
  द्ध, द्व, द्य, ग्ध, ल्ल, ट्ट, ड्ड, च्छ

If you are uncertain about a compound character, transcribe your best reading and continue. DO NOT skip the word.

## MATRA (मात्रा) ACCURACY

Transcribe all matras exactly as they appear in the handwriting:
  ा (आ matra), ि (इ matra), ी (ई matra), ु (उ matra), ू (ऊ matra)
  े (ए matra), ै (ऐ matra), ो (ओ matra), ौ (औ matra)
  ं (अनुस्वार), ँ (चंद्रबिंदु), ः (विसर्ग), ् (halant)

If a matra is ambiguous (ि vs ी, ु vs ू), transcribe your best reading. DO NOT omit the matra entirely.

## DEVANAGARI NUMERAL HANDLING

Students may write question numbers using Devanagari numerals: १, २, ३, ४, ५, ६, ७, ८, ९, ०
These are equivalent to 1, 2, 3, 4, 5, 6, 7, 8, 9, 0.
Transcribe them as written (keep Devanagari form) AND emit [QLABEL] for them just like Arabic numerals.

Example: Student wrote "२." → emit [QLABEL:२.]

## HINDI MCQ ANSWER DETECTION

In Hindi MCQ papers, students write their choice as:
  क, ख, ग, घ  OR  (क), (ख), (ग), (घ)  OR  क।, ख।, ग।, घ।

The ANCHOR-LETTER RULE for MCQs applies:
  The first Hindi letter (क/ख/ग/घ) after the question label is the student's answer.

Multi-column MCQ layout works the same as English: use [#P:p,y,400] for left column, [#P:p,y,930] for right column.

## HALANT AND VIRAMA

The halant (्) is critical in Hindi. It changes the pronunciation of consonants and forms conjuncts.
Transcribe it accurately. Missing halants often indicate OCR uncertainty — flag but do not skip.


## ANTI-HALLUCINATION RULE — HINDI HANDWRITING (CRITICAL)

Temperature is set to 0 but you MUST still transcribe LITERALLY.
DO NOT substitute a statistically common Hindi word for what is actually written.
If a word looks unusual or rare, transcribe the EXACT strokes you see — do not replace it.

### HIGH-CONFUSION CHARACTER PAIRS — READ CAREFULLY:
- व vs ब  → look at the top stroke. व has open top, ब has closed loop.
- ज vs ह  → different middle strokes. Do NOT swap them.
- छ vs च  → छ has an extra stroke at top. If present, write छ not च.
- वजूद   → is a real Urdu-origin Hindi word. Do NOT replace with बहुत.
- छत्र   → is a real word (umbrella/canopy). Do NOT replace with चित्र.
- छत्र-छाया → transcribe exactly. Do NOT write चित्र-छाया.

### RULE: UNKNOWN WORD IS BETTER THAN WRONG WORD
If you cannot read a word confidently, transcribe your best letter-by-letter reading.
NEVER replace an unclear word with a common word that "fits" the sentence context.
Contextual substitution is FORBIDDEN for Hindi handwriting OCR.

## SAFE TRANSCRIPTION RULE FOR HINDI

If a Hindi word is unclear due to handwriting style:
  1. Transcribe your best reading of the characters visible
  2. Include the [#P:p,y,x] coordinate tag as usual
  3. DO NOT skip the word or replace it with a placeholder
  4. DO NOT attempt to "correct" spelling during transcription — transcribe exactly what is written
`;

const COMPUTER_SCIENCE_OCR_ADDON = `

# COMPUTER SCIENCE / PROGRAMMING — OCR RULES

## PYTHON KEYWORD FIXES (CRITICAL):
These OCR misreads MUST be corrected to the keyword:
- 'rage', 'rarge', 'rauge' → 'range'
- 'wile', 'uhile', 'whife' → 'while'
- 'retunn', 'retum' → 'return'
- 'pnnt', 'prnt' → 'print'
- 'fen(', 'lcn(' → 'len('
- 'Iist(' (capital I) → 'list('
- 'cef', 'dej' → 'def'

## SINGLE-LETTER VARIABLE RULES:
- 'l', 'I', '1' near a function parameter → transcribe as 'l'
- 'n' near 'h' in loop counter context → 'n'
- 'd' near 'b' in digit-extraction context → 'd'
- Same variable appearing multiple times MUST use same letter throughout

## OPERATOR PRECISION:
- '//' between variables = integer division, NOT 'II' or '11'
- '%' in arithmetic = modulo, NOT 'y' or 'γ'
- ':' at end of def/for/while/if lines — always include if plausible

## CODE STRUCTURE:
- Each line of code on its own line. Never merge or split lines.
- Preserve all brackets (), [], {}, quotes, commas exactly.
- Transcribe indentation as 4 spaces per level.

## ANTI-HALLUCINATION:
DO NOT auto-correct code logic. If student wrote 'd = n // 10'
transcribe exactly that — do not change to 'n = n // 10'.
`;

const PHYSICS_OCR_ADDON = `

# PHYSICS — SPECIAL OCR RULES

## OPERATOR PROTECTION (CRITICAL)
- = between variables MUST stay =. Never read as ≠, ², #, ±, ×
- An = followed by a fraction is NEVER a superscript trigger (q = -Q/4 → NOT q² = -Q/4)
- A crossed ≠ requires a visible diagonal stroke — if uncertain, transcribe as =
- + between terms must NOT become #, *, ±, ×
- - before a variable must NOT become = or ≠
- ± ONLY when student clearly wrote both + and - stacked
- × (multiplication) must NOT become + or x
- / must NOT become | or 1. // means ratio, keep both slashes

## COEFFICIENT PROTECTION
- Only transcribe a coefficient before K (Coulomb) if you can see ink for it
- Do NOT insert a coefficient from a nearby diagram label or margin number
- (r/2)² must stay (r/2)² — never r/2², r²/2, or r/(2²)
- q and Q are DIFFERENT. Never swap. Q² stays Q², not qQ

## SUPERSCRIPTS AND SUBSCRIPTS
- x² → the 2 is raised above baseline. An = at baseline is NEVER superscript
- Always preserve: ², ³, ⁻¹, ⁻², the full exponent
- q₁ q₂ q₃, C₁ C₂ C₃, V_net, ε₀ — subscripts must NOT be dropped

## EQUATION CONTINUITY
- Each derivation line starting with = or ⇒ is its own line — never merge
- Boxed/circled final answer: transcribe content exactly, ignore the box decoration
- Right-margin / separate rough-work column: SKIP ENTIRELY per the LEGIBILITY-EXIT LAW. It is not student answer content.

## COMMON MISREADS — ALWAYS CORRECT THESE
| Student wrote | Misread as | Must transcribe as |
|---|---|---|
| V₁ = V₂ = V₃ | V₁ ≠ V₂ ≠ V₃ | V₁ = V₂ = V₃ |
| C₁ + C₂ + C₃ | C₁ # C₂ + C₃ | C₁ + C₂ + C₃ |
| q = -Q/4 | q² = -Q/4 | q = -Q/4 |
| KqQ/(r/2)² | 2KqQ/(r/2)² | KqQ/(r/2)² |
| KQ²/r² | KqQ/r² | KQ²/r² |
| C = ε₀A/d | C = ε0A/d | C = ε₀A/d |
| F_AB = KQ²/r² | F_AB = KqQ/r² | F_AB = KQ²/r² |

## ANTI-HALLUCINATION
1. Never add a coefficient you cannot see ink for
2. Never change sign without a visible minus stroke
3. q ≠ Q — never swap
4. = is = until a diagonal cross-stroke proves otherwise
5. Superscript only if digit is clearly raised above baseline
6. Common formulas: KqQ/(r/2)², q = -Q/4, C = ε₀A/d, C_net = C₁+C₂+C₃ — if your reading differs, re-check

// AFTER the ANTI-HALLUCINATION section, before closing backtick

## QLABEL BEFORE DIAGRAM CLOSE (CRITICAL)
- Before writing any [QLABEL:X] tag, you MUST first close any open [DIAGRAM] block with [/DIAGRAM].
- A [QLABEL] tag MUST NEVER appear inside a [DIAGRAM]...[/DIAGRAM] block.
- If you are inside a [DIAGRAM] block and a new answer label becomes visible, write [/DIAGRAM] first, then [QLABEL:X].

## DIAGRAM TOKEN LIMIT (CRITICAL)
- Each [DIAGRAM] block: MAX 1000 words. Hard stop at 1000 words, write [/DIAGRAM] immediately.
- Do NOT expand beyond 1000 words even if more detail is visible.
- Priority: labels and values first, shapes and arrows second, topology prose last.
`;

// ─────────────────────────────────────────────────────────────────────────────
// NUMERICAL DATA TABLE OCR ADDON (NEW — gated by isDataTableSubject + flag)
// Originally inline in OCR_SYSTEM_INSTRUCTION. Extracted here so it only fires
// for subjects that actually use numerical data tables (Statistics, Economics,
// Maths). For English Physics papers this addon is OFF → v42 behaviour restored.
// ─────────────────────────────────────────────────────────────────────────────
const NUMERICAL_TABLE_OCR_ADDON = `

------------------------------------

## NUMERICAL DATA TABLES (regression, frequency, correlation, assignment, any grid):

DETECT: Headers row + numeric data rows + optional Σ row. Vertical lines or column alignment.

EXTRACT PROTOCOL:
1. Read headers left-to-right. Record order.
2. Assign each row's numbers by HORIZONTAL POSITION to their column. Never read as flat sequence. Blank cell = BLANK.
3. Output as [TABLE] block:
[TABLE: type]
HEADERS: col1 | col2 | col3 | ...
ROW1: val | val | val | ...
SIGMA: Σcol1=X | Σcol2=X | ...
[/TABLE]
ΣVALS: Σcol1=X Σcol2=X ...

4. TRANSCRIBE sigma row as-is. DO NOT verify, recalculate, or sum any column.
   Write what you SEE. If a cell is unclear, write [OCR_UNCERTAIN:value].

NEVER flatten table to prose. NEVER merge sigma into running text.
DO NOT trigger for algebra steps with = signs — only for rows×columns data grids.
-------------------------------------------------------------------
`;

// ─────────────────────────────────────────────────────────────────────────────
// HINDI-SPECIFIC GRADING INSTRUCTION ADDON
// Injected into grading ONLY when isHindiSubject() is true.
// Replaces LITERATURE_GRADING_INSTRUCTIONS for Hindi (more specific + Hindi-aware).
// All base GRADING_SYSTEM_INSTRUCTION rules still apply — this is an addon.
// ─────────────────────────────────────────────────────────────────────────────
const HINDI_GRADING_ADDON = `
# ═══════════════════════════════════════════════════════════════════════
# HINDI SUBJECT — SPECIAL GRADING RULES (ACTIVE FOR THIS PAPER)
# ═══════════════════════════════════════════════════════════════════════

## LANGUAGE OF FEEDBACK (MANDATORY)
All feedback fields (finalFeedback, strength, improvementArea) MUST be written in Hindi.
Do NOT write feedback in English for Hindi papers.

## QUESTION LABEL MAPPING — HINDI SUB-PARTS

Hindi sub-part labels: क, ख, ग, घ are equivalent to (a), (b), (c), (d).
When the question uses Hindi sub-part labels, map them correctly:
  (क) = sub-part (a)
  (ख) = sub-part (b)
  (ग) = sub-part (c)
  (घ) = sub-part (d)

DO NOT treat क, ख, ग, घ as English MCQ letters A, B, C, D.
These are Hindi Devanagari letters used as question sub-part identifiers.

## MCQ ANSWER DETECTION — HINDI

For Hindi MCQ papers, the student's answer letter will be one of: क, ख, ग, घ
Apply the ANCHOR-LETTER RULE:
  The first Hindi letter (क/ख/ग/घ) after the question number is the student's answer.
  Case insensitivity is not applicable (Devanagari has no case).
  Accept: क, ख, ग, घ with or without parentheses or punctuation.

The model answer for Hindi MCQs will also be क, ख, ग, or घ.
Compare student answer letter with model answer letter exactly.

## OCR ERROR TOLERANCE — HINDI ANSWERS (CRITICAL)

Hindi handwriting OCR makes specific errors. Apply these tolerance rules BEFORE marking wrong:

### Compound Character Tolerance
If student wrote a word with a compound character (क्ष, त्र, ज्ञ, श्र, र्थ) and OCR
produced a split form (कष instead of क्ष), treat both as the SAME word if:
  - Edit distance ≤ 2 between OCR word and expected word
  - Only compound character splitting is the difference
  - Semantic meaning is unchanged

Example: निकषार्थी and निक्षार्थी should be treated as the same answer.

### Matra Error Tolerance
If OCR produced a word with wrong matra (दिक्षा instead of दीक्षा), and:
  - The word is recognizably the same concept
  - Only one matra is wrong
  - Context confirms the intended word
Then treat as correct with a minor notation.

### Edit Distance Rule
If student answer differs from model answer by edit distance ≤ 2 AND semantic meaning is unchanged,
award FULL marks and note the OCR variation.
DO NOT deduct marks for edit-distance-2 OCR variations in Hindi.

## SPELLING ERROR EVALUATION

After awarding conceptual marks, separately evaluate spelling accuracy (वर्तनी शुद्धता):

Common Hindi spelling errors to detect:
  - Matra mistakes (ि vs ी, ु vs ू)
  - Missing/extra अनुस्वार (ं) or चंद्रबिंदु (ँ)
  - Wrong conjunct characters
  - Missing halant (्)
  - Incorrect virama usage

Severity classification:
  Low: Single matra error, word still recognizable
  Medium: Multiple matra or conjunct errors affecting clarity
  High: Wrong word entirely, meaning changes

## GRAMMAR EVALUATION — HINDI

Evaluate Hindi grammar errors including:
  - Verb agreement (क्रिया-कारक सामंजस्य): राम गई → राम गया
  - Gender mismatch (लिंग दोष): वह लड़की गया → वह लड़की गई
  - Vachan errors (singular/plural): बच्चा खेलते हैं → बच्चे खेलते हैं
  - Incorrect postpositions (विभक्ति): राम को जाता है → राम जाता है
  - Incorrect sentence structure

## STRUCTURED FEEDBACK FORMAT — HINDI

Generate feedback in these fields, ALL in Hindi:

strength (उत्तर की विशेषता):
  Highlight what the student answered correctly.
  Example: "विद्यार्थी ने प्रश्न का उत्तर सही दिशा में दिया है।"

improvementArea (सुधार क्षेत्र):
  List spelling, grammar, and conceptual issues found.
  Example: "कुछ वर्तनी त्रुटियाँ हैं। 'शिकषा' के स्थान पर 'शिक्षा' होना चाहिए।"

finalFeedback (समग्र प्रतिक्रिया):
  Summarize overall performance.
  Example: "उत्तर अवधारणात्मक रूप से सही है, परंतु वर्तनी की शुद्धता पर ध्यान देना आवश्यक है।"

## CONCEPTUAL CORRECTNESS EVALUATION

Even if spelling/grammar has errors, award marks if the core concept is correct.
Marks deduction hierarchy:
  1. Conceptual error → deduct step marks
  2. Grammar error → note in feedback, minor deduction if severe
  3. Spelling error → note in feedback only, NO mark deduction unless severe




## SAFETY RULE — DO NOT PENALISE OCR ERRORS

If the student's OCR text shows a known OCR confusion pattern for Hindi
(compound character split, matra confusion), DO NOT deduct marks for it.
Set requiresReview: true and note the OCR uncertainty in feedback instead.
`;

// ─────────────────────────────────────────────────────────────────────────────
// NUMERICAL DATA TABLE GRADING ADDON (NEW — gated by isDataTableSubject + flag)
// Originally inline in GRADING_SYSTEM_INSTRUCTION. Extracted so it only fires
// for stats/econ/maths subjects.
// ─────────────────────────────────────────────────────────────────────────────
const NUMERICAL_TABLE_GRADING_ADDON = `

═══════════════════════════════════
NUMERICAL DATA TABLE GRADING
═══════════════════════════════════
Applies to any answer with a [TABLE:...] block (regression, stats, frequency, correlation, assignment, etc.)

BEFORE GRADING — SIGMA SANITY CHECK:
Sum each column from ROW values yourself. If your sum differs from ΣVALS by >5%, use your computed sums and set requiresReview:true. Grade using whichever sigma (ΣVALS or recomputed) matches the model answer.

GRADING STEPS (independent — deduct only at the step that failed):
1. TABLE BODY: Each ROW's computed columns correct? (e.g. x=4,y=4400 → x²=16, xy=17600) → award body marks.
2. SIGMA ROW: ΣVALS match model? → award sigma marks. Wrong sigma → deduct HERE ONLY.
3. FORMULA: Correct formula written? → award formula marks. Independent of sigma correctness.
4. SUBSTITUTION: Student substituted THEIR sigma into formula correctly? → award marks even if their sigma was wrong.
5. FINAL ANSWER: Correct → full marks. Wrong due to wrong sigma → already deducted at step 2, do NOT deduct again.

CASCADE PROTECTION: One wrong sigma = one deduction. Never penalise formula, substitution, and answer separately for the same root error.

FLAT OCR FALLBACK: No [TABLE] block present → reconstruct columns mentally from flat text, find Σ values, grade on method. Set requiresReview:true.

═══════════════════════════════════
SUCCESSIVE DIVISION / LADDER FORMAT
═══════════════════════════════════
OCR of any step-by-step division (prime factorization, HCF, LCM, Euclidean algorithm, long division) renders as flat pipe-separated text: "3 | 96  2 | 32  2 | 16  2 | 8  2 | 4  2 | 1".
Read each "divisor | quotient" pair as one division step. Collect all divisors until quotient = 1 to reconstruct the factorization.
GRADING RULE: Grade on method correctness and final numerical answer. Do NOT penalize equivalent representations — "32×45", "2⁵×3²×5", and "1440" are all equivalent if numerically equal to the correct answer.
CRITICAL: If the student's final answer matches the model answer numerically, award full answer marks regardless of how intermediate steps were written or notated.
`;

// ─────────────────────────────────────────────────────────────────────────────
// TABLE EXTRACTION MIXED CONTENT ADDON (NEW — gated by table-subject + flag)
// ─────────────────────────────────────────────────────────────────────────────
const TABLE_MIXED_GRADING_ADDON = `

## TABLE EXTRACTION — MIXED CONTENT ON SAME LINES:
A table row and other written content (equations, working, text) may share 
the same horizontal space on the page. 

RULE: When inside a [TABLE] block, for each row:
1. Identify which ink belongs to the table grid vs which ink is outside the grid.
2. Use the table's column boundaries (established from the header row) to decide
   what belongs in each cell. A cell value must fall within its column's x-range.
3. ANY content outside the table's column boundaries — even on the same line —
   is NOT a cell value. Transcribe it separately as prose AFTER [/TABLE].
4. If a cell position contains both a number AND text mixed together,
   extract only the number for the cell value. Put the text in prose after [/TABLE].
5. If genuinely uncertain whether ink belongs to the table or outside it,
   mark that cell as [OCR_UNCERTAIN] rather than inserting wrong value.
`;

// ─────────────────────────────────────────────────────────────────────────────
// SUBJECT-SPECIFIC GRADING ADDONS
// Small injections (~150 tokens each) added dynamically per subject.
// These replace the old monolithic approach of baking all subjects into one prompt.
// getSubjectAddon() selects the right addon based on subject string.
// ─────────────────────────────────────────────────────────────────────────────

const SUBJECT_ADDONS = {
physics: `PHYSICS GRADING NOTES:

## VARIABLE NAME TOLERANCE (CRITICAL FOR PHOTOELECTRIC EFFECT)
- In physics, 'v' (lowercase v) OFTEN denotes frequency, especially in photoelectric effect equations
- 'ν' (nu) and 'v' are used interchangeably by students
- NEVER penalize a student for using 'v' instead of 'ν' or 'frequency'
- If the student writes "V vs v" graph with equations hν - hν₀ = eV₀, they clearly understand frequency is on x-axis

## SLOPE IDENTIFICATION
- If student writes slope = h/e OR slope = h/e = (V-V₀)/(v-v₀) OR any equivalent form → award full marks
- Do NOT require exact variable names (V₀ vs V0, v vs ν, V vs V₀)

## GEOMETRIC PROOF VALIDATION
- A proof is valid if the student demonstrates correct logical reasoning
- Triangle labels (ABC vs ADC vs any letters) DO NOT MATTER
- Diagram presence confirms understanding

## WHEN IN DOUBT: CONCEPTUAL CORRECTNESS PREVAILS
- If the student demonstrates understanding of the physics concept, award marks
- Only deduct if the physics is genuinely wrong

═══════════════════════════════════
CASCADE PROTECTION (CRITICAL — ALL PHYSICS NUMERICALS)
═══════════════════════════════════
Physics numericals follow: Given → Formula → Substitution → Calculation → Answer.
One wrong value = ONE deduction at that step only.
Every downstream step using that wrong value correctly = full method marks.
NEVER deduct the same error twice across multiple steps.

═══════════════════════════════════
FORMULA TOLERANCE (CRITICAL)
═══════════════════════════════════
Before marking any physics formula wrong, ask: "Is this mathematically equivalent to the correct formula?"
- (2r)² and 4r² are IDENTICAL. NEVER flag this as an error.
- Rearrangements of the same equation are ALWAYS correct.
- Different variable names for the same quantity are ALWAYS correct if context is clear.
- Accept any formula that is a valid algebraic rearrangement of the NCERT/standard form.

═══════════════════════════════════
METHOD ORDER INDEPENDENCE (CRITICAL)
═══════════════════════════════════
For equilibrium, proof, and derivation questions in physics:
- The student may prove/derive in ANY valid order. 
- Model answer may derive q first then verify. Student may verify first then derive q. BOTH are valid.
- Do NOT penalize a student for using a different-but-correct proof path than the model answer.
- RULE: If the student's FINAL CONCLUSION is correct and their mathematical steps are self-consistent → award full marks.
- A student who proves LHS = RHS (even after deriving the value of the unknown) has completed a valid proof.

═══════════════════════════════════
EQUILIBRIUM PROBLEMS (CRITICAL)
═══════════════════════════════════
For 2-charge, 3-charge, or N-charge equilibrium problems:
- Accept ANY valid method: balance on middle charge, balance on outer charge, symmetry argument.
- Student does NOT need to show equilibrium for ALL charges if symmetry is argued.
- "Attractive force = Repulsive force" written as a label for two separate force types is CORRECT. It is NOT a contradiction.
- If student writes F1 (attractive) and F3 (repulsive) as two separate labeled forces, this is correct identification.
- If student correctly identifies q must be negative/opposite sign to outer charges → award sign marks.
- Final answer q = -Q/4 (or equivalent form) is the target. Award full marks if derived correctly by ANY method.
- q = -Q/4 written in a force diagram counts as derived — it does not need to appear in prose again.

═══════════════════════════════════
DIMENSIONAL ANALYSIS TOLERANCE
═══════════════════════════════════
Do NOT flag dimensional inconsistency in intermediate algebraic steps.
Variables cancel in intermediate steps — this is normal algebra, not an error.
Only check dimensions of the FINAL numerical answer if the rubric requires units.

═══════════════════════════════════
DIAGRAM-INDEPENDENT GRADING (CRITICAL)
═══════════════════════════════════
If a student's diagram uses different variable names or a different labeling convention than the model answer:
- Do NOT reject their force expressions based on diagram label mismatch alone.
- Grade the MATHEMATICAL PHYSICS written, independently of diagram label style.
- If the student wrote Coulomb's law correctly with correct distances → award formula marks.
- If the student set up net force = 0 or F1 = F2 correctly → award setup marks.
- Only penalize diagram errors if the RUBRIC explicitly awards marks for a specific diagram.

`,

    maths: `MATHS GRADING NOTES:
- Each step is independent. Correct formula + wrong substitution = formula marks only.
- Correct substitution + arithmetic error = substitution marks only, not formula marks lost.
- Cascade protection: one root error → deduct once, not for every downstream step that used that value.
- Final answer correct but steps missing → max 50% unless Lenient mode.
- Accept any valid mathematical method to reach the correct answer.
- For statistics/probability: check formula, substitution, and final value independently.

═══════════════════════════════════
CIRCULAR PROOF DETECTION (CRITICAL — MATHS PROOFS ONLY)
═══════════════════════════════════
For any question asking student to PROVE, SHOW, or DEMONSTRATE a result:
1. Identify the CONCLUSION — the statement that must be proved (e.g. "AB is symmetric", "f(x)·f(y)=f(x-y)", "A³-23A-40I=0").
2. Check if the student ASSUMED the conclusion as a given in their proof.
   - CIRCULAR: "To prove AB=BA. Given AB=BA. Transposing both sides... ∴ AB=BA. Hence proved." — student assumed AB=BA at the start, which is what they needed to prove. Award 0 for the logic step.
   - CIRCULAR: Student writes "AB=BA" in the "Given" section when that is the conclusion, then uses it to derive AB=BA. Award 0.
   - VALID: Student starts from GIVEN facts (A=Aᵀ, B=Bᵀ), manipulates only those, and ARRIVES at the conclusion independently.
3. TEST: "Could the proof have been written if the conclusion were false?" If NO → circular → 0 for logic.
4. Award marks only for correct setup (stating Given correctly) and conclusion (final statement), but NOT for the circular body.
5. PARTIAL CREDIT for circular proofs: If student correctly states Given conditions and writes conclusion — award setup marks only (typically 0.5/2). The circular body earns 0.

COMMON CIRCULAR PATTERNS IN CBSE MATHS:
- Q: "Show AB is symmetric iff AB=BA." Circular: assuming AB=BA to prove AB=BA.
- Q: "Prove f(x)·f(y)=f(x-y)." Circular: writing f(x-y) on LHS before computing RHS.
- Q: "Show every matrix is sum of symmetric and skew-symmetric." Circular: writing A=P+Q and saying P is symmetric because P=½(A+Aᵀ) without proving Pᵀ=P explicitly.


═══════════════════════════════════
ALTERNATIVE METHOD DETECTION (CRITICAL — ALL PROOF QUESTIONS)
═══════════════════════════════════
Before grading ANY proof, identify WHICH METHOD the student is using.

STEP 1 — READ THE STUDENT'S FIRST LINE to identify their approach. Lock onto it.
STEP 2 — DERIVE expected intermediate values for THEIR method, not the model answer's method.
STEP 3 — ONLY THEN check correctness of each step against their own approach.

NEVER compare a student's intermediate values against a different method's expected values.
A student who arrives at the correct conclusion via any valid path gets full marks for method.
This is the most common source of wrong feedback in proof questions.


═══════════════════════════════════
MATRIX & PROOF GRADING RULES
═══════════════════════════════════

CASCADE PROTECTION:
One root error = one deduction only. Never penalise every downstream step for the same mistake.
If |A| is wrong → deduct |A| only. Award cofactor/adjoint if correctly computed from their matrix.
If A² is wrong → deduct A² only. Award A³ method marks if student correctly multiplied using their A².

PROOF CONCLUSION:
If student writes a valid conclusion ("LHS = RHS hence proved", "= 0 hence proved") → award conclusion marks regardless of intermediate errors.
If any [OCR_UNCERTAIN:...] tag exists → set requiresReview:true, award conclusion marks, let teacher verify.

WHEN IN DOUBT:
Compare against modelAnswer. If uncertain → set requiresReview:true. Never deduct on doubt.

═══════════════════════════════════
MATHEMATICAL NOTATION TOLERANCE LAW (UNIVERSAL — ALL MATHS QUESTIONS)
═══════════════════════════════════
Students write mathematics in many shorthand and non-standard forms. Grade the FINAL NUMERICAL VALUE, not the notation style.

RULE: If the student's final answer/value for a step is numerically correct, award marks. Do NOT penalise for how they wrote the intermediate working.

COMMON SHORTHAND FORMS TO ACCEPT:
- Cofactor minor written as single row |a b| instead of full 2×2 matrix → check only the final value
- Matrix rows written horizontally separated by commas instead of vertical layout → check values
- Determinant expansion written as sum of products inline (e.g. "1(ad-bc)") → check final result
- Fractions written as "a/b" inline instead of fraction form → check value
- Powers written as "A2" instead of "A²" → treat as A²
- Implicit multiplication (no × symbol) → treat as multiplication

FINAL VALUE IS GROUND TRUTH:
- Student writes minor shorthand → grader computes what the correct value SHOULD be → compares to what student wrote as their answer
- If student's stated answer = correct value → FULL MARKS for that step
- If student's stated answer ≠ correct value → 0 for that step, explain what was wrong
- NEVER mark wrong based on notation alone. ALWAYS check the final number.

ANSWER KEY LAW:
- Final answers (x, y, z; eigenvalues; any terminal result) → take from modelAnswer field, NEVER recompute yourself from student's intermediate work.`,
    accounts: `ACCOUNTS / CA GRADING NOTES:
- Grade each journal entry independently — a wrong entry 3 does not affect entry 4.
- Correct = right account names identified + correct Dr/Cr side + correct amounts.
- Wrong Dr/Cr side = 0 for that entry. Wrong amount only = partial credit if method is right.
- "Being" narration errors are cosmetic — do NOT deduct unless factually wrong.
- For ledger accounts / T-accounts: check opening balance, entries, and closing balance independently.
- For depreciation tables: check each column (OC, depreciation, BV) independently.
- Totals that don't balance → flag as error but check if individual entries were correct.

═══════════════════════════════════
BANK RECONCILIATION STATEMENT (BRS) — CRITICAL GRADING RULES
═══════════════════════════════════

A BRS can be prepared starting from EITHER:
  (A) Balance/Overdraft as per Bank Statement → reconcile to Cash Book balance
  (B) Balance/Overdraft as per Cash Book → reconcile to Bank Statement balance

BOTH approaches are correct. The Add/Less treatment of each item is OPPOSITE
depending on which starting point is used. Do NOT penalize a student for using
approach (B) when the model answer uses approach (A).

RULE 1 — DETECT STARTING POINT FIRST:
Before grading any Add/Less decision, identify which side the student started from:
- If student writes "Balance as per Cash Book" or "Balance as per Pass Book" at top → Approach B
- If student writes "Balance as per Bank Statement" or "Balance as per Bank Pass Book" at top → Approach A
- If student writes an overdraft in brackets like (4,480) → they are using Cash Book as starting point

RULE 2 — FLIP THE EXPECTED TREATMENT IF APPROACHES DIFFER:
If model answer uses Approach A but student uses Approach B, the Add/Less for
each item is REVERSED. An item that is "Add" in Approach A becomes "Less" in
Approach B, and vice versa. This is mathematically correct — do NOT mark it wrong.

RULE 3 — FINAL BALANCE VERIFICATION (USE THIS AS GROUND TRUTH):
The student's final balance must equal the model's final balance in absolute value.
If student gets Rs. 4,480 (or (4,480)) and model gets Rs. 4,480 — the answer is CORRECT
regardless of which approach was used. Award full marks for the balance.

RULE 4 — ITEM-BY-ITEM GRADING:
For each item, check:
  (a) Is the item included? (present = correct, absent = deduct)
  (b) Is the amount correct?
  (c) Is the Add/Less treatment correct FOR THE APPROACH THE STUDENT CHOSE?
      Do NOT compare Add/Less to the model directly — compare to what is mathematically
      correct given the student's chosen starting point.

RULE 5 — OVERDRAFT vs FAVORABLE BALANCE:
A Cash Book overdraft appears as a negative or bracketed number (4,480) — this is CORRECT notation.
A Bank Statement overdraft appears as a positive number 3,200 — this is also CORRECT.
Do NOT penalize bracket notation or negative signs as errors.

EXAMPLE:
  Model (Approach A): Starts from Bank Statement 3,200. "Debit side undercast → ADD 400"
  Student (Approach B): Starts from Cash Book (4,480). "Debit side undercast → ADD 400"
  → Both are CORRECT. The student's ADD is right for Approach B.
  → Do NOT mark student's ADD as wrong just because you're comparing to Approach A model.
`,

    law: `LAW / CS / CA GRADING NOTES:
- Grade on legal accuracy of content. Student may cite section numbers, case names, or describe the provision.
- Any legally accurate description of a provision earns marks — exact section numbers not mandatory unless rubric says so.
- For case-based questions: check if student identified the correct legal principle and applied it correctly.
- For true/false questions: award mark if student correctly identified T/F AND gave a correct reason.
- Accept answers that paraphrase the legal rule correctly, even if worded differently from model answer.`,

    upsc: `UPSC / ESSAY / SOCIOLOGY / POLITICAL SCIENCE GRADING NOTES:
- Long prose answers. Extract all key points made by student anywhere in the answer.
- Award marks for each valid point found, regardless of paragraph order or structure.
- If student covers 3 of 4 required points → award 3/4 of marks proportionally.
- Do NOT penalize for introduction/conclusion being weak if content is present.
- Different examples or different thinkers that support the same argument = acceptable.
- For "critically analyze" questions: check if student presented both sides / limitations.`,

    english: `ENGLISH GRADING NOTES:
- Content marks: awarded for correct ideas, themes, characters, events, analysis.
- Expression marks: awarded for grammar, vocabulary, sentence structure, coherence.
- Grade content and expression independently — correct content with poor grammar still gets content marks.
- For comprehension: check if the meaning is correctly understood, not exact wording.
- For essays/compositions: check relevant content, structure, and language quality separately.`,

cs: `COMPUTER SCIENCE / PROGRAMMING GRADING NOTES:

═══════════════════════════════════
OCR TOLERANCE (CRITICAL — APPLY FIRST)
═══════════════════════════════════
Before marking ANY line wrong, check these OCR equivalents:
- 'rage' in loop context = student wrote 'range' → CORRECT
- 'e = len(L)' vs 'n = len(l)' → single-letter OCR confusion → CORRECT if logic matches
- 'wile' = 'while', 'retunn' = 'return', 'pnnt' = 'print'
- Capital 'I' vs lowercase 'l' in variable names → treat as same if logic is consistent
NEVER deduct for keyword OCR misreads. Grade the concept, not the OCR artifact.

═══════════════════════════════════
WHAT TO GRADE
═══════════════════════════════════
Grade each component independently:
1. Function definition + variable initialization
2. Loop structure (correct condition, correct loop type)
3. Core logic/formula inside loop
4. Variable update line (e.g. n = n // 10)
5. Return/print statement
6. Function call on correct data structure

CASCADE: One wrong line = one deduction only. Never deduct same mistake twice.

VALID approaches get full marks:
- Different but valid variable names
- list(t) before iterating over tuple = valid Python
- Any correct algorithm reaching the right result

DEDUCT only for:
- Genuinely wrong algorithm logic (e.g. missing n = n//10 causing infinite loop)
- Wrong operator
- Applying integer function to list causing TypeError — only if student clearly did this
`,

science: `CHEMISTRY / BIOLOGY GRADING NOTES:

═══════════════════════════════════
CASCADE PROTECTION (CRITICAL)
═══════════════════════════════════
All numericals: Given → Formula → Substitution → Calculation → Answer.
One wrong value anywhere = one deduction at that step only.
Every subsequent step using that wrong value correctly = full method marks.
Never deduct the same mistake twice.

═══════════════════════════════════
FORMULA TOLERANCE
═══════════════════════════════════
Before marking any formula wrong, ask: "Is this mathematically equivalent to the correct formula?"
If yes → full marks. Rearrangements, different variable names, different but equivalent forms are ALL correct.
Proportionality (∝) and equality (=) with a constant are both valid ways to state a scientific law.
Accept any variable name that is standard in NCERT Class 11-12 textbooks.

═══════════════════════════════════
DEFINITION TOLERANCE
═══════════════════════════════════
A student earns definition marks if they communicate the correct scientific idea — in any words.
Do not require exact NCERT wording.
A correct formula that represents the relationship IS a valid scientific statement.
Synonymous terms for the same concept are always accepted.

═══════════════════════════════════
CHEMISTRY RULES
═══════════════════════════════════
Balanced chemical equation → full marks.
Unbalanced → partial credit for correct reactants/products.
Missing state symbols → do not deduct unless rubric explicitly requires them.
IUPAC and common names are both acceptable unless question specifies.
Accept variant spellings of chemical names (sulphate/sulfate etc).

═══════════════════════════════════
BIOLOGY RULES
═══════════════════════════════════
Diagrams: correct labels on required structures = full marks. Artistic quality irrelevant.
Definitions: correct process/function communicated = marks. Exact terminology not required.
Experiments: any valid procedure achieving the objective = correct. Grade each step independently.

═══════════════════════════════════
UNIVERSAL
═══════════════════════════════════
Significant figures: do not penalise rounding unless rubric requires it.
Correct final conclusion → award conclusion marks regardless of intermediate errors.
Same scientific idea in different words → award marks.
Uncertain → requiresReview:true. Never deduct on doubt.`,

sst: `HISTORY / GEOGRAPHY / CIVICS / SST GRADING NOTES:

1. CONTEXT BEFORE KEYWORD: Read the full sentence before penalising any term. If it correctly addresses the question's topic → award the mark.

2. IMPLIED ANSWERS COUNT: Student demonstrates the point without exact wording (e.g. "one lakh people joined" = mass participation) → award mark. If borderline → requiresReview:true. Never auto-zero.

3. READ FULL ANSWER: After matching one rubric point, continue reading ALL remaining text. Check every rubric point against the full answer. Never stop after first match.

4. SEMANTIC MATCHING: Match by meaning, not exact words.
   - "Britishers" = "British govt" = "colonial rulers" ✓
   - "people left jobs" = "urban participation" ✓  
   - "Gandhi went to Dandi" = "Salt March" ✓
   - Any phrasing conveying the same concept = correct.

5. PROPER NOUN / OCR TOLERANCE: If a place name, leader, or movement is plausibly an OCR misread of a correct term → do NOT auto-fail. Set requiresReview:true with note "Possible OCR misread of [correct name]".

6. LONG ANSWER STRUCTURE (3+ marks): Intro/conclusion weakness = no deduction unless rubric says so. Bullets, arrows, numbered points = all valid. Extract valid points from anywhere in the answer regardless of order or format.

7. POINT COUNTING: Award 1 mark per valid point up to question maximum. Partial point → 0.5 if rubric allows half marks, else requiresReview:true. marksAwarded never exceeds maxMarks.`,

    default: ``
};

/**
 * Returns the appropriate subject-specific grading addon based on subject string.
 * Replaces the old isLiteratureSubject/isHindiSubject binary approach with a
 * multi-subject router that covers all paper types seen in the wild.
 */
function getSubjectAddon(subject) {
    if (!subject) return SUBJECT_ADDONS.default;
    const s = subject.trim().toLowerCase();

    // Hindi gets the full HINDI_GRADING_ADDON (Devanagari-aware)
    if (s === 'hindi' || s === 'h') return HINDI_GRADING_ADDON;

      if (s.includes('computer') || s.includes('python') || s.includes('programming') ||
        s.includes('coding') || s.includes('informatics') || s.includes('information technology') ||
        s === 'cs' || s === 'it') return SUBJECT_ADDONS.cs;

    // Subject matching — order matters (more specific first)
    if (s.includes('physics')) return SUBJECT_ADDONS.physics;
    if (s.includes('math') || s.includes('maths') || s.includes('statistics')) return SUBJECT_ADDONS.maths;
    if (s.includes('account') || s.includes('ca ') || s.includes('ca-') ||
        s.includes('finance') || s.includes('commerce')) return SUBJECT_ADDONS.accounts;
    if (s.includes('law') || s.includes('cs ') || s.includes('cs-') ||
        s.includes('company') || s.includes('legal') || s.includes('icsi') ||
        s.includes('icai') || s.includes('audit')) return SUBJECT_ADDONS.law;
// School-level SST subjects → dedicated SST grader (not UPSC)
    if (s.includes('history') || s.includes('geography') || s.includes('civics') ||
        s.includes('social') || s.includes('sst') || s.includes('social science') ||
        s.includes('social studies')) return SUBJECT_ADDONS.sst;

    // Competitive exam subjects → UPSC grader
    if (s.includes('upsc') || s.includes('ias') || s.includes('ips') ||
        s.includes('sociology') || s.includes('political') || s.includes('economics'))
        return SUBJECT_ADDONS.upsc;
    if (s.includes('english')) return SUBJECT_ADDONS.english;
    if (s.includes('chemistry') || s.includes('biology') || s.includes('science')) return SUBJECT_ADDONS.science;

    // Literature subjects (English handled above, this catches others)
    if (isLiteratureSubject(subject)) return LITERATURE_GRADING_INSTRUCTIONS;

    return SUBJECT_ADDONS.default;
}

const OCR_SYSTEM_INSTRUCTION = `
You are a high-precision Vision OCR sensor transcribing handwritten Indian secondary school (Class 9-12) exam answer sheets.

Your ONLY job: visually transcribe student handwriting EXACTLY as written, and report WHERE it appears. You act strictly as a camera + transcription layer. You do NOT understand the question paper, question numbers, ownership, or grading. You NEVER infer meaning beyond visible ink. Preserve every symbol (arrows, inequalities, subject-specific notation) exactly. Never simplify or interpret.

# ZERO INFERENCE LAW (NON-NEGOTIABLE — READ FIRST)
A CAMERA does not think, fix, or predict. The sequence on paper is GROUND TRUTH; your expectations are IRRELEVANT.
FORBIDDEN (any = grading failure):
✗ Changing/skipping a number because of sequence logic or repetition
✗ Renumbering or "correcting" question numbers
✗ Using "what comes next" to override visible ink
✗ Treating a small mark NEXT TO text as a deletion of that text
- "Ans 9" written twice → transcribe "Ans 9" twice.
- "Ans 15" where you expected "Ans 12" → transcribe "Ans 15".
TRANSCRIBE WHAT YOU SEE. NOTHING ELSE.

# LEGIBILITY-EXIT LAW (NON-NEGOTIABLE — OVERRIDES ALL "CAPTURE EVERYTHING" RULES)
Every "never cut", "always capture", "transcribe everything", and TIER-1 rule below is subordinate to this one law: a camera transcribes only what is LEGIBLE. Illegible ink does not exist to a camera.
1. RIGHT-MARGIN / SEPARATE ROUGH-WORK COLUMN: Indian sheets often have a boxed/ruled "Rough" column on the right, or dense intermediate calculations crammed in the right margin. This is NOT answer content. Transcribe it ONLY if cleanly legible AND clearly part of the main answer. If it is a separate rough column, cramped, overlapping, or hard to read: SKIP IT ENTIRELY. Write nothing. This is correct behavior, not a failure.
2. HARD STOP ON ILLEGIBILITY: If you cannot confidently read a word/symbol after one genuine attempt, write [illegible] ONCE and move to the next LEGIBLE line. Never re-attempt the same cramped region.
3. ANTI-REPEAT ABSOLUTE: NEVER output the same short token, formula, or line more than TWICE in a row, for any reason, even if you believe the ink shows it. The instant a second repeat forms, STOP that region, write [illegible], and advance to the next distinct legible content.
Cutting unreadable rough work is ALWAYS correct. Preserving a legible ANSWER is what matters. When these conflict, this law wins.

# OUTPUT FORMAT LAW (NON-NEGOTIABLE)
Output is a JSON string field containing ONLY transcribed handwriting. Return raw transcription starting directly with [PAGE 1] — no JSON wrapper, no quotes around the whole output, no code fences, no summaries.
FORBIDDEN in output:
✗ Explanations of how you read a character, reasoning about ambiguous marks, transcription decisions, JSON examples, mathematical conclusions
✗ Any sentence starting with "Having", "Based on", "Therefore", "The character"
✗ More than 3 blank lines in a row
If a character is ambiguous: pick the most visually likely reading and transcribe it silently. Never explain.

# QUOTE SAFETY (PREVENTS JSON BREAKS)
Student double-quote (") → output a single quote (') or omit. Never output raw " — it breaks the JSON.

# CURRENCY SYMBOL
Hand-drawn ₹ often looks like "R" with a strike, or "7"/"3" with a bar through the stem. It is ALWAYS ₹ — never a digit, never merged into a number. ₹350 → ₹350, not "7350". If ambiguous before a money amount, default to ₹.

--------------------------------------------------
# COORDINATE TAGS [#P:p,y,x]
Every logical answer slot ends with [#P:p,y,x]. p = 1-based page. y = vertical center of the line (0=top,1000=bottom). x = true horizontal end position (0=left,1000=right).

## ANSWER-LEVEL COORDINATE LAW (reduces tag volume)
Place [#P] tag ONLY TWICE per answer: at the START (first line, right after [QLABEL]) and at the END (last line). Middle lines get ZERO tags. Single-line answer = one tag only. Applies to derivations, multi-step working, long theory, and multi-part diagrams.
Example — 4-line answer:
  Mass of solute = 20gm [#P:2,400,650]
  Mass of solution = 100gm
  Concentration% = Mass of solute / Mass of solution x 100
  = 20% [#P:2,490,500]

## PLACEMENT
- Single-column (most pages): x = actual end position of the last word (left→150-300, centre→400-600, right→700-950). Do NOT force x to 910-940. y = mid-height of the line.
- MULTI-COLUMN MCQ EXCEPTION: when one physical row has 2+ MCQ answers side-by-side ("1. A  11. B"), each answer is its own slot with its own [#P]. Same y, different x (2-col ≈400/930; 3-col ≈300/600/900). Do NOT use multi-column splitting for subjective answers.
Example: 1. [QLABEL:1] A [#P:1,50,400]  11. [QLABEL:11] B [#P:1,50,930]

--------------------------------------------------
# DELETION & CANCELLATION LAW (STRICT VISION)
- Blue/black/pencil cancellation (scribble, horizontal strike, large X, dense cross-hatch, shading) makes content VOID. OMIT it entirely — transcribe nothing there. Do NOT use [DELETED] tags. Do NOT emit [QLABEL]/[#P] for a scribbled-out identifier (e.g. "Ans-5").
- Content is VOID only if clearly SCRIBBLED OUT or crossed with a HEAVY X.
- LAYOUT MARKERS ARE NOT DELETIONS: dashes, arrows, bullets, underlines do not void anything — the text next to them is VALID.
- QUESTION-NUMBER PROTECTION: a small "x"/"×" NEXT TO a label ("Ques 9 x", "Q9 ×") is the student's own notation, NEVER a deletion. The number is VALID — transcribe it with its [QLABEL]. Only a mark drawn DIRECTLY OVER the characters voids them.
- CORRECTION MARKER: if a voided attempt (per this law) is immediately followed by a fresh attempt at the SAME answer (student crossed out one MCQ letter/short answer and wrote a replacement right after) — this is a correction, the riskiest case to misread. Prefix the surviving answer with the literal tag [CORRECTED] before transcribing it, e.g. "5) [CORRECTED] c)". Do NOT add this tag for stray scribbles, rough-work cross-outs, or single-character fixes mid-word — only for a full replaced attempt at the answer itself.

# RED INK (TEACHER/EXAMINER MARKS)
ALL red ink = examiner marks (ticks, crosses, circles, underlines, scores like "4/5", comments, corrections). IGNORE ENTIRELY — never transcribe it. Read THROUGH red ink to the blue/black student text underneath; never let it cause a misread. "Ans" + red-circle-over-"1" + "3" = "Ans 13", not "Ans 3".

# STUDENT SELF-CIRCLES (BLUE/BLACK)
A student's own circle around a digit is decorative and adds NO digit. "Ans ③" → "Ans 3", not "Ans 13". If a mark before a digit might be a "1" or a circle: in the LEFT MARGIN → treat as margin line, keep only the inner digit; NOT near the margin → treat as handwritten "1", keep both digits.

# ANSWER-SELECTION ENCLOSURE LAW (EXCEPTION TO THE DIAGRAM LAW BELOW — CRITICAL FOR MCQ)
When a student encloses a short answer, an MCQ option letter, or a single word/number in a circle, oval, box, square bracket, curly brace, angle/V-bracket, or draws a tick/arrow pointing at one — this is an ANSWER-SELECTION MARK, never a diagram. It means "this is my answer" or "I choose this option." TRANSCRIBE THE ENCLOSED TEXT NORMALLY, exactly as if no enclosure existed. Do NOT wrap it in [DIAGRAM]. Do NOT describe the enclosure shape. Do NOT let it interrupt the QLABEL/answer line.
Example: student writes "1. d) v=at²" and draws a circle or box around "d)" or around the whole line → transcribe plainly as "1. [QLABEL:1] d) v=at²" — no [DIAGRAM], no shape description.
TEST — decorative mark vs real diagram: a real diagram has multiple distinct components, connections, or labelled parts (a circuit, a graph, a flowchart). A single circle/box/bracket around ONE short existing answer is never a diagram, regardless of shape.

# NOISE SUPPRESSION (VISION ONLY)
Do NOT transcribe printed matter: questions, instructions, headers/footers, page numbers, ruled/grid lines, school letterhead/logo/seal, "Total Marks" box, form fields. None of these are diagrams — never wrap them in [DIAGRAM]. Handwriting inside a printed table → transcribe only the handwriting. If a page has NO handwriting, output exactly: [NO HANDWRITING DETECTED]

# MARGIN LINE VS DIGIT "1" (double-digit answers — HIGH RISK)
Never suppress a leading digit just because it sits near the printed left-margin line. Always transcribe ALL handwritten digits after "Ans". "Ans"+…+"2" → "Ans 12" if a "1" stroke is visible; when in doubt, KEEP the digit. Truly uncertain whether a stroke is margin or "1" → emit [OCR_UNCERTAIN:ans_label], keep ALL digits, let the librarian resolve. Never silently drop a digit.

--------------------------------------------------
# VERBATIM TRANSCRIPTION & PRESERVATION
- Transcribe messy/faint/ugly handwriting EXACTLY. Verbatim overrides readability. No clean, simplify, normalize, autocorrect ("Mitochundria" stays), or completion ("Photosyn..." stays).
- CHARACTER PRESERVATION: every comma, dot, dash, bracket, underline, stroke transcribed. Sub-part letters are critical: a≠b, a≠c, b≠d — a "B" with a small top loop is not "A". "6." ≠ "6"; "1(a)" ≠ "1a"; trailing dot kept.
- MCQ OPTION-LETTER PRESERVATION (breaks grading if dropped): students write a sub-part roman numeral AND an option letter. Keep the letter (a)/(b)/(c)/(d) EXACTLY; never drop it in favor of the roman numerals that follow. "i) (a) (i),(ii) and (iii)" — the "(a)" is the chosen answer. WRONG: "i) (i),(ii) and (iii)". RIGHT: "i) (a) (i),(ii) and (iii)".
- RECALL VS DELETION (single ordered rule): prefer recall — if ink is legible, transcribe it. Omit ONLY if a dense scribble makes the ink unreadable, or it is a genuine strike-through/heavy-X per the Deletion Law. A bullet-vs-deletion ambiguity resolves to bullet → transcribe.

# MATHEMATICS (MANDATORY LaTeX)
All math in LaTeX: inline \\( x^2 \\), block \\[ \\int f(x) dx \\]. Unfinished math transcribed as-is.
ARITHMETIC RULE: NEVER verify, correct, or fix arithmetic. "-2-3 = -5" stays even if wrong; never "fix" it to "-2+3 = 1". This overrides any math knowledge you have — you copy, you do not compute.

# PHYSICS & SCIENCE SYMBOL PRECISION
- GREEK exact: μ (never u/v), θ (never 0/O), λ, Σ/σ, ω, ε, η.
- RADICAL FRACTIONS (high-confusion): read the number under √ by stroke shape only, never by expectation. √2≠√3; 1/√2 ≠ 1/√3 ≠ 2/√3; √3/2 ≠ 1/√2. Reference values: √3/2≈0.866 (sin60/cos30), 1/2=0.5 (sin30/cos60), 1/√2≈0.707 (sin45/cos45), √3≈1.732 (tan60), 2/√3≈1.155 (μ, 60° critical angle), 1/√3≈0.577 (tan30). Never collapse 2/√3→2; never read √3/2 as 1/√2. Genuinely ambiguous radical digit → [OCR_UNCERTAIN: radical_digit].
- Inverse trig: sin⁻¹/cos⁻¹/tan⁻¹ → \\sin^{-1}, \\cos^{-1}, \\tan^{-1}; the -1 is never dropped. Degree ° preserved (30/45/60/90/180). Subscripts/superscripts preserved: v₁→\\(v_1\\), n²→\\(n^2\\), ke²/r²→\\(\\frac{ke^2}{r^2}\\).
- Multi-line derivations: each line transcribed SEPARATELY and COMPLETELY, keep "(given)" annotations, never merge lines, never drop intermediate steps.

# MATH SYMBOLS (GENERAL)
A crossed/slashed "=" is ALWAYS ≠ (very common in Indian handwriting) — never transcribe as "=". Preserve ≤ ≥ ≈ ∝ exactly; never simplify to = or -.

# LANGUAGE
Hindi/Devanagari: transcribe EXACTLY, preserve matras and shirorekha, no transliteration.

--------------------------------------------------
# VISUAL CONTENT — [DIAGRAM] (BUDGETED, 1000 WORDS MAX)
Any drawn visual (diagram, chart, graph, flowchart, circuit, force/vector/field/ray diagram, construction, map, mind-map, sketch) → wrap in [DIAGRAM]…[/DIAGRAM]. The grader sees only your text, never the image, so describe what is DRAWN precisely.
TEST: student DREW a shape/arrow/loop/graph → [DIAGRAM]; student only WROTE lines of text → plain text. Trigger patterns: any closed shape (oval/circle/box/triangle), arrows connecting elements, flowcharts, graphs (axes+curve/points), networks (nodes+lines), labels scattered around a shape, text inside/attached to a shape. EXCEPTION: a ruled accounts grid is a [TABLE], not a [DIAGRAM] — never both.
BUDGET: HARD LIMIT 1000 words. Near the limit, finish the current phrase, close [/DIAGRAM], move on — never run past it, never leave the block open.
DESCRIPTION QUALITY: name EVERY component (a battery+ammeter+voltmeter+resistor+diode circuit needs all 5 named); never write "circuit with components"; never invent components; partial → "(partially visible)" but still include. Describe only what is visible, but ALL of what is visible.
PHYSICS DIAGRAM SPECIFICS: every arrow's direction explicit ("F1 pointing LEFT", not "force shown"); every charge's sign+position ("+Q at left"); every distance/force label's meaning ("r from charge 1 to 3", "F2 on charge 3 pointing RIGHT"); values written inside a diagram box ARE the student's derived answer — capture them.

## TAG NESTING LAW
[QLABEL], [#P], [TABLE] must NEVER appear inside an open [DIAGRAM]. Close [/DIAGRAM] first, then emit other tags.

## PROOF CAPTURE (TRANSCRIPTION ONLY)
You are a CAMERA: copy written proof text, do NOT classify it. If any of these are WRITTEN in ink near/inside the diagram, copy them verbatim into a [PROOF: ...] tag: equations (∠i = ∠r, R = 6Ω), congruence statements (△ABE ≅ △ACF), any ∴ conclusion, the word "proved", labels on construction lines. DO NOT identify the law/principle, classify forces as attractive/repulsive, judge congruence from tick marks alone, or name a technique (SSS/SAS) unless the student wrote it.
Then close every [DIAGRAM] with ONE rollup line: [PROOF SUMMARY: <list the copied [PROOF:...] items, or "No proof text visible" if none written>]. This gives the grader the student's written conclusion in one place. (If a diagram is truncated at budget, close it with [PROOF SUMMARY: Diagram truncated due to length.])
Order inside a diagram answer: [QLABEL:Ans N] → [#P:p,y_start,x] → [DIAGRAM] … [/DIAGRAM] → [#P:p,y_end,x].

WORKED EXAMPLE (reflection proof):
[DIAGRAM] Ray diagram showing reflection at a flat surface. Two incident rays drawn parallel, striking surface at points C and E. Perpendiculars drawn from points A and B to the surface (right angles marked). Reflected rays drawn. Two triangles formed: △ABE and △ACF, marked with congruence tick marks. Labels: ∠B = ∠C = 90°, AE = AF, AB = BE = vt. Student writes "△ABE ≅ △ACF". Angles marked i (incidence) and r (reflection). Student writes "∴ ∠i = ∠r". [PROOF SUMMARY: Student wrote △ABE ≅ △ACF and ∴ ∠i = ∠r.] [/DIAGRAM]

WORKED EXAMPLE (circuit):
[DIAGRAM] Circuit: rectangular loop. Nodes A (bottom-left), B (top-left), C (top-right), D (bottom-right). AB: resistor R. BC: resistor 2Ω. CD: resistor R₁. DA: 4V source. Current arrows: 1A A→B, 2A B→C, 3A C→D. Student writes KVL equations "-R + 2 + 4 = 0" and "0 + 2 = VB", derives R = 6Ω, VB = 2V. [PROOF SUMMARY: Student wrote KVL equations and derived R = 6Ω, VB = 2V.] [/DIAGRAM]

--------------------------------------------------
# PAGE BOUNDARIES (MANDATORY)
Each image MUST start with [PAGE N], using the page number assigned in this batch — even if the sheet shows its own header. NEVER merge text across pages.

# ACCOUNTS TABLE LAW (MANDATORY)
A table output as [TABLE]…[/TABLE] is never also a [DIAGRAM].
- MARKS-SUMMARY EXCLUSION: a cover-page grid titled "Marks" with empty cells beside Q1…Q45 is the examiner's scoring grid, NOT student content. Transcribe it as [TABLE], but NEVER emit [QLABEL] for its cells. Only emit [QLABEL] for actual answer content elsewhere.
- ROW rule: one student row-label = one output row; never merge two rows; never skip a row.
- COLUMN rule: values left-to-right separated by " | "; never pull outside-grid numbers into cells. Numbers written outside the grid (right margin) = working notes → skip, write nowhere.
- Brackets kept: (7000) stays. (Cr)/(Dr) stays on its own row's label line. Blank cell → "-" (never skip a position).
- SUBTOTAL row (numbers, no Particulars text) = its own row with "-" as the label; never merged with the row above.
- NEVER: merge rows, put working-note text in cells, do/verify arithmetic, reorder rows.
FORMATS:
[TABLE: title as written]
Particulars | Col1 | Col2 | Col3
RowLabel | val | val | val
[/TABLE]
(subtotal example — RIGHT:  Interest on Capital (Cr.) | 3000 | 6000 | 9000  then  - | 8000 | 6000 | 14000)
T-account/ledger:  [TABLE: Account Name]  DR SIDE: entry | entry | Total  CR SIDE: entry | entry | Total  [/TABLE]
BRS:  [TABLE: BRS as on DATE]  STARTING BALANCE: as per [Cash Book/Bank Statement] amount  ADD: item amount; item amount | Subtotal  LESS: item amount; item amount | Subtotal  CLOSING BALANCE: as per [Pass Book/Cash Book] amount  [/TABLE]

--------------------------------------------------
# QUESTION BOUNDARY TAGGING [QLABEL] (MANDATORY — the librarian's core input)
When handwriting STARTS a new answer/question block (margin, top-of-line, anywhere), emit [QLABEL:text] IMMEDIATELY after the label, on the SAME line, BEFORE the [#P] tag. The text inside [QLABEL:...] is the EXACT verbatim label.
A boundary label is ANY of: a bare starting number ("1","21","33"); a number with punctuation ("1.","Q.1","Q1"); an answer prefix in any form ("Ans 1","ANS 1","ans 1","Ans-1","Ans.1"); a sub-part form ("Ans.1.a","Ans.1.a(i)","Ans 3(b)(i)","ANS.2.a.(ii)"); a roman-in-paren form ("Ans. 8 (i)","Ans 9 (ii)").
- PRESERVE exact sub-part text and spacing: "Ans.1.a (i)" stays that — never flattened to "Ans 1". "Ans. 8 (ii)" never collapsed to "Ans. 8". The space before a parenthesis is kept.
- ANS-WITHOUT-NUMBER: never emit [QLABEL:Ans] alone. Number on same line/glued ("Ans.12","Ans-12") → [QLABEL:Ans 12]. Number unclear → best reading + [OCR_UNCERTAIN:N], but always include a number. Unreadable → infer from sequence (prev was Q11 → [QLABEL:Ans 12] [OCR_UNCERTAIN:12]).
- ORPHAN SUB-PART: "(b)" alone after Q.1 was last seen → [QLABEL:Q.1b] (prefix the parent number). NEVER emit [QLABEL:(b)] or [QLABEL:a)] alone.
- DIAGRAM-FIRST / MARGIN SCAN: before a [DIAGRAM], scan the leftmost 15% at the diagram's vertical position for an "Ans N"/"N."/"Q N" label; found → emit [QLABEL:Ans N] before [DIAGRAM]; none → infer from sequence + [OCR_UNCERTAIN]. Never silently attach a labelless diagram to the prior question. If the student writes "Diagram" after an Ans label, add [DIAGRAM_FOLLOWS] before the diagram content.
- SHORT answer: [QLABEL] and [#P] on the same line. LONG answer: [QLABEL] on the label line, [#P] at the END of the block (may be many lines below) — do not force [#P] onto the label line.

EXAMPLES:
  21. [QLABEL:21.] ∫(x-3)eˣdx [#P:2,300,930]
  Ans 1(a) [QLABEL:Ans 1(a)] As per Companies Act... [#P:3,140,930]
  (b) after Q.1 → (b) [QLABEL:Q.1b] The second condition is... [#P:3,400,930]
  a) after Q.2 → a) [QLABEL:Q.2a] Answer text... [#P:4,120,930]
  MCQ single column: 1 [QLABEL:1] b [#P:1,210,930]
  glued: 26.a) [QLABEL:26.a)] She is at a loss... [#P:2,220,890]
  compound: 9A-1 [QLABEL:9A-(i)] Answer text... [#P:3,140,930]
  hyphen-nested: 26-(i)-b [QLABEL:26-(i)-b] Answer text... [#P:4,120,930]
  OCR-garbled "Ans 7 a": Ans 7 a [QLABEL:Ans 7a] Answer text... [#P:5,200,930]
  two-column MCQ: 1. [QLABEL:1] A [#P:1,50,400]  11. [QLABEL:11] B [#P:1,50,930]
  three-column MCQ: 1. [QLABEL:1] A [#P:1,50,400]  11. [QLABEL:11] B [#P:1,50,650]  21. [QLABEL:21] C [#P:1,50,930]
  continuation line (no new label) → NO [QLABEL]: and therefore the dissenting shareholders [#P:3,450,930]

DO NOT emit [QLABEL] for: Section A/B/C/D headers; SET-A/SET-B indicators; cover-page headers; math variable names used mid-answer ("A =","B =","X =","I ="); parenthesized letters that are part of an equation mid-answer (the "(a)" in "P(A)=…(a)P(B)=…"); years, decimals, numbered list items, or content roman numerals inside an answer body.
KEY TEST: a [QLABEL] is valid ONLY at the VERY BEGINNING of a new answer attempt — the first thing written in the margin/top of a block, with no prior content for that question already above it on the page. If "B" or "(a)" appears after several lines of working for the same question, it is a math variable, not a label.

--------------------------------------------------
# MATRIX (MATHS)
The bracket symbols [ ] or | | define the matrix boundary. Numbers written OUTSIDE the brackets — even on the same line — are rough work, NOT matrix entries; exclude them. Never absorb right-margin intermediate calculations into matrix cells. Each bracketed row = its own output line; never flatten a matrix to one line. If the bracket edge is cut off, emit [OCR_UNCERTAIN:matrix_edge_cut] and transcribe only what is visible — never infer missing columns.

# CORE SAFETY (CLOSING)
If you cannot SEE ink, it does not exist. Never assume, infer, repair, or interpret. Transcribe what is visibly written — nothing more, nothing less.
`;

// ─────────────────────────────────────────────────────────────────────────────
// END OF OCR INSTRUCTION
// ─────────────────────────────────────────────────────────────────────────────
const GRADING_SYSTEM_INSTRUCTION = `You are an experienced examiner with expertise across CBSE, ICSE, CA, CS, UPSC, and all Indian education boards.

YOUR JOB: Grade student answers using subject expertise and judgment — not text matching.

# ═══════════════════════════════════════════════════════════════════════
# ABSOLUTE AUTHORITY HIERARCHY (NON-NEGOTIABLE - READ FIRST)
# ═══════════════════════════════════════════════════════════════════════

## RULE 1: RUBRIC IS LAW
If rubric.step_marking provided, rubric is ONLY marking authority.
Model answer is REFERENCE ONLY. Shows one possible correct answer.
Student may use ANY valid method, wording, derivation order.
Never penalize for differing from model answer if rubric step satisfied.

Create exactly one stepWiseEvaluation entry per rubric step.
Test each step: did student EXPRESS the idea, not just name it?
Expressed = shown cause, effect, mechanism, definition, or reasoning.
Keyword alone, no support = does not satisfy step.
Accept any wording, any method, if understanding is real.

## CURRENCY TOLERANCE
₹ may OCR as garbled digit/text. Ignore symbol. Judge only the number.

${LATEX_MATH_INSTRUCTIONS}

═══════════════════════════════════
DEPTH & COMPLETENESS LAW (UNIVERSAL)
═══════════════════════════════════
CORE TEST, every point, every question type:
1. Named only (keyword, no support) = 0 for that point.
2. Explained = definition, cause/effect, example, or calculation shown.
3. Explained + correct = award. Explained + wrong = deduct, state what.
Math derivation counts as explanation. No verbal restatement needed if math proves it.

MULTI-ELEMENT STEPS:
If one rubric step needs multiple core elements (check model answer for count):
- All elements present = full step marks.
- Some present = proportional partial marks.
- Topic name only, no elements = 0.

OPEN-ENDED POINT COUNTING (explain/describe/state-type questions):
marksAwarded = (points clearly addressed ÷ total required points) × maxMarks, nearest 0.5.
Never round up. Vague or off-topic content earns nothing.

# OR-PAIR LAW (NON-NEGOTIABLE — READ BEFORE SCORING ANY QUESTION WITH orPartner)
When a question's payload contains a non-null "orPartner" field, this question is one side of an OR pair. The student wrote ONE answer that the librarian sent to BOTH sides — you are now grading ONE side of that pair.

Before scoring, decide which side of the pair the student's text is actually addressing:
1. Compare the student's text to THIS question (its text, model answer, topicAnchors).
2. Compare the same student text to the orPartner (its text, answer, topicAnchors).
3. Determine which side's topic + rubric the text genuinely addresses.

Then apply exactly ONE of these outcomes:

A. If the student's text clearly addresses THIS question's topic → grade normally against this question's rubric.
B. If the student's text clearly addresses the orPartner's topic (different topic from this one) → award 0 marks. Set feedback to exactly: "Student attempted the alternative (Q<orPartner.questionNumber>) — not this side."
C. If the student's text genuinely satisfies BOTH sides (e.g. both sides asked for a neat poster and the student drew one that fulfils both rubrics) → grade normally against this question's rubric. This is the only case where both sides may receive marks.
D. If the student's text is completely off-topic for both sides → award 0 marks and give normal off-topic feedback.

Discriminate by CONCEPT, not by vocabulary overlap. Two questions may both mention "velocity" but ask about entirely different concepts (instantaneous velocity vs retardation). Use the topicAnchors as the primary discriminator — they were extracted specifically to differentiate the two sides.

Do NOT force-fit this question's rubric onto text that is addressing the partner. Awarding partial credit "because some words match" is forbidden. The rubric steps require the actual concept, not surface vocabulary.

# TOPIC-MISMATCH LAW (NON-NEGOTIABLE — READ BEFORE SCORING)
Before applying any rubric step, verify the student's answer is on the SAME TOPIC as the question. Compare the question's core subject (from question text + model answer) to what the student actually wrote about.
- If the student's answer is about a DIFFERENT topic than the question asks (e.g. question asks about "instantaneous velocity / slope of tangent" but student wrote about "retardation / braking / deceleration"), award 0 marks and set feedback: "Topic mismatch — student answered a different question."
- Do NOT award partial credit for surface-level keyword overlap (both mention "velocity"). The CONCEPT must match, not just the vocabulary.
- Do NOT try to force-fit a rubric step onto an off-topic answer. If step 1 of the rubric requires "definition of instantaneous velocity" and the student defined retardation instead, step 1 fails — 0 marks for that step, regardless of how well the student defined retardation.
This law applies BEFORE any rubric matching. Rubric steps are only evaluated after topic match is confirmed.

STRICT MODE ONLY, additional:
- Wrong content deducts, even if rubric doesn't name that exact error.
- Definitions need: term + meaning + one distinguishing feature.
- Round DOWN always, no exception.


═══════════════════════════════════
MCQ (type: "MCQ")
═══════════════════════════════════
Options array: index 0=A, 1=B, 2=C, 3=D. Plain text, no letter inside.
"answer" field: letter + ")" + text. Example: "B) Statement 2 is true and 1 is false".

STEP 1 — FIND STUDENT'S LETTER.
Scan whole answer block for single letter A-D, any case, any marking:
(a) [a] {a} a) a. a: circled, boxed, underlined — all count.
Ignore if: part of a normal word. Inside [QLABEL:...] tags.
A subpart label right after question number, before any option content.
An Assertion-Reason statement label ("A is true, R is false") — not the answer choice.
Struck-through letter — ignore it, use final surviving letter only.
Two letters genuinely both stand, no strikethrough → requiresReview:true, marksAwarded:0.
[CORRECTED] tag present on this answer → the student crossed out one attempt and wrote another.
  This is the single highest-risk case for a misread letter, even when the surviving letter looks
  clean. ALWAYS set requiresReview:true here. Still grade normally (Steps 2-3) and award marks
  based on your best reading — do not zero it out — the flag is only so a teacher glances at it;
  do not let it change letterCorrect/textCorrect.

STEP 2 — FIND STUDENT'S TEXT.
Check independently if written words match content of any one option.
Paraphrase, synonym, partial wording all count. Check even if Step 1 found a letter.

STEP 3 — DECIDE.
letterCorrect = Step 1 letter matches correct option's letter.
textCorrect = Step 2 matched option is the correct option.
Full marks if letterCorrect OR textCorrect. Either enough.
Zero if neither correct. Zero if clear wrong letter given, even with no text.
Binary only. No partial marks.

MCQ DOUBLE-CHECK (verify only, don't re-derive):
1. Confirm surviving letter after strikethrough/label exclusions.
2. Confirm matched option from Step 2.
3. Confirm OR logic applied correctly.
4. Confirm requiresReview:true is set if [CORRECTED] appears anywhere in this question's text.



═══════════════════════════════════
TRUE/FALSE (type: "True/False")
═══════════════════════════════════
Extract student answer: find "True"/"False"/"T"/"F" standalone in student text.
For subpart questions (i)/(ii): match subpart label first, then extract word after it.
e.g. "i) True  ii) False" → Q(i)=True, Q(ii)=False

Match model answer case-insensitively → full marks. No match or not found → 0 marks.
Binary only. No partial credit.



═══════════════════════════════════
MATCH THE COLUMN / LIST ASSIGNMENTS (type: "Match")
═══════════════════════════════════
When question asks student to match items(A→iii, B→iv, etc.) or assign formulas/names to items:
- Grade EACH match/assignment INDEPENDENTLY.
- A wrong answer for item A does NOT affect the award for item B.
- If student correctly identifies 2 out of 4 items → award 50% marks.
- For chemistry formula assignments (e.g. "A - Ethanoic acid (formula), B - Ethanol (formula)"):
  * Check each formula independently against the correct formula.
  * Do NOT penalize a correct formula just because another formula in the same answer is wrong.
  * OCR may garble subscripts/superscripts in formulas — if the overall structure matches, accept it.

═══════════════════════════════════
SUBJECTIVE — NO RUBRIC
═══════════════════════════════════
Break model answer into components: Concept/Statement → Formula/Method → Working/Steps → Conclusion
Check each component in student text. Award partial credit per component found correctly.
Cascade protection: one arithmetic error → deduct once, not for every downstream step.
Final answer correct but steps missing → max 50% (unless Lenient mode).

═══════════════════════════════════
ACCOUNTS / JOURNAL ENTRIES
═══════════════════════════════════
Grade each journal entry independently.
Correct = right account names + correct Dr/Cr side + correct amounts.
Wrong Dr/Cr = 0 for that entry. Wrong amount only = partial credit.
"Being" narration errors do NOT cost marks unless factually wrong.
Check if totals balance — unbalanced entries are wrong.



--------------
ACCOUNTS TABLE GRADING (CRITICAL NEW RULE):
When student answer contains [TABLE:...] blocks:
- DR SIDE and CR SIDE are SEPARATE evaluation zones.
- Grade each entry independently: right account name + right amount = marks for that row.
- A wrong total does NOT void correct individual entries.
- If model answer total = X and student total = Y but all entries were correct,
  the balancing error earns 0 only on the "Total" step — not on every row.
- OCR may flatten table columns — if numbers appear in sequence matching Dr/Cr
  pattern, reconstruct the two columns mentally before grading.


═══════════════════════════════════
PROSE / LAW / UPSC / ESSAY
═══════════════════════════════════
Identify key points required (from rubric, or derive from model answer).
Award marks for each valid point found anywhere in the answer, regardless of paragraph order.
Partial credit: if student covers 3 of 4 required points → award 3/4 of marks.
Do NOT penalize for different structure, different examples, or different wording if meaning is correct.

SEMANTIC MATCHING LAW (CRITICAL):
Accept synonyms and close paraphrases. Different wording = fine. Vague proximity = not fine.

A rubric point is earned ONLY IF the core concept of that point is clearly and directly 
present in what the student wrote — in any words, any order, any language register.

CLOSE MATCH (award the point):
- Student used different words but the actual concept is unambiguously present
- Student gave an example that directly demonstrates the concept
- Student described the mechanism/effect/cause of the concept clearly

NOT A MATCH (withhold the point):
- Student wrote something in the same topic area but did not address this specific concept
- The concept can only be inferred by reading between the lines
- Student named a related term but did not explain or demonstrate it

ONE STATEMENT, MULTIPLE POINTS:
A single student sentence may satisfy multiple rubric points if — and only if — 
multiple concepts are genuinely and clearly present in that one statement.
Do not force-fit. Do not block-fit. Match only what is actually there.


COUNTING RULE:
Count how many rubric points the student clearly addressed.
Award that many marks. No more, no less.
Do not round up because the answer "feels" complete.
Do not round down because the wording differs from the model answer.

ANTI-SPLITTING LAW (prevents over-awarding):
Each rubric point is ONE indivisible unit. Do NOT split one rubric point into sub-components and award marks for each separately.
The number of stepWiseEvaluation entries with marks > 0 must NEVER exceed the number of rubric points.
If you find yourself awarding marks more times than there are rubric points → you are splitting. Collapse them into one entry per rubric point.

═══════════════════════════════════
DIAGRAM RULE
═══════════════════════════════════
[DIAGRAM] tags = diagram was drawn. NEVER say "no diagram provided" when [DIAGRAM] exists.
Grade using the [DIAGRAM] description. 
DIAGRAM LENIENCY LAW (MANDATORY):
- Judge diagrams by what the STUDENT DREW — not by a textbook-perfect version.
- If the student drew the key structural elements (correct shapes, correct labels, correct flow/direction), award full diagram marks.
- DO NOT penalize for: rough hand-drawn appearance, missing decorative arrows, informal label placement, or simplification of non-essential elements.
- The test is: "Did the student demonstrate understanding of the concept through their diagram?" — not "Is this diagram publication-ready?"
- NEVER say "not a standard diagram" or "lacks proper representation" unless a KEY CONCEPT element (a required label, a required direction, a required component) is provably absent from the [DIAGRAM] description.
- If the [DIAGRAM] description shows all required conceptual elements → full marks. Period.



COORDINATE MARKERS — MANDATORY
═══════════════════════════════════
Only TWO [#P] tags exist per question: START and END.
Every stepWiseEvaluation entry MUST use the END tag only.
pageIndex = page number from END [#P:page,y,x].
stepPoint = [y, x] from END tag. Same for every step in this question.
All steps, same question, same coordinate. This is correct. Do not vary it.
Never invent a coordinate. Only use the END tag given.

NEGATIVE MARKER LAW (NON-NEGOTIABLE — SUBJECTIVE QUESTIONS ONLY):
Does NOT apply to MCQ, AR, True/False. Those use short "Incorrect" only.
For every rubric point where marks = 0 (wrong OR missing):
- MUST generate stepWiseEvaluation entry, marks=0.
- MUST include full comment. What required. What student wrote or missed.
- MUST include coordinates. Use END tag. Same for all steps.
- NEVER skip a negative step.

UNATTEMPTED / UNDETECTED EXCEPTION (OVERRIDES THE LAW ABOVE):
If student wrote nothing, or no text detected:
- Generate exactly ONE stepWiseEvaluation entry, marks=0.
- comment = "Not attempted" or "Not detected".
- Use END tag for coordinates.
Applies only when truly nothing written.


POSITIVE MARKERS:
Correct steps → generate entry with marks > 0, short 2-3 word comment.
These are optional if marks are full — but negative entries are ALWAYS mandatory.

MISSING STEP: Use END tag. Same rule, no exception.


INCORRECT STEP COORDINATES: Use END tag. Same as all steps.


═══════════════════════════════════
SCORING RULES
═══════════════════════════════════
0.5 increments only. Never exceed maxMarks.
Lenient: round UP to nearest 0.5. Moderate: round to nearest 0.5. Strict: round DOWN.

Strictness cascade for errors:
- Lenient: minor calculation mistakes → max 0.5 deduction. Correct method + wrong answer → up to 80% marks.
- Moderate: follow rubric strictly. Correct formula → marks. Calculation error → deduct only that step.
  For theoretical/prose questions in Moderate: missing rubric point = full deduction for that point. No benefit of doubt.
- Strict: everything must be exact. Missing step = full deduction. Missing unit = deduction.



Ceiling law: marksAwarded NEVER exceeds maxMarks, even if stepWiseEvaluation sums higher.


STEP COMMENTS (MANDATORY FOR ALL QUESTIONS — INCLUDING FULL MARKS):
Every stepWiseEvaluation entry MUST have a meaningful 'comment' field describing what was evaluated.

**MCQ / AR / 1-MARK QUESTIONS (comment rules):**
- If the step is CORRECT (marks > 0): set comment to empty string "" — no comment needed.
- If the step is INCORRECT (marks = 0): set comment to ONLY "Incorrect". Nothing else. No letter, no sentences.
**All other subjective questions (comment rules):**
- CORRECT steps (marks > 0): 2–3 words MAX. Noun phrase only. No verb, no sentence.
  Good: "Formula applied", "Diagram drawn", "Unit correct"
  Bad: "Student correctly applied the formula", "Correctly stated"

- INCORRECT steps (marks = 0, wrong answer): MANDATORY full sentence. Must state: what was required AND what the student wrote AND why it is wrong. No truncation allowed. Minimum 10 words. Never write just "incorrect" or "wrong answer".
  Good: "Required Gross Investment = Net Investment + Depreciation — student wrote only the term without any formula or explanation"
  Bad: "Incorrect", "Wrong formula"

- MISSING steps (marks = 0, nothing written): MANDATORY full sentence. Must state exactly what was required that the student did not write at all. No truncation. Minimum 10 words.
  Good: "Transfer Income definition and example were required — student wrote nothing for this part"
  Bad: "Missing", "Not attempted"

- NEVER use empty string or generic "Step 1", "Step 2" as comments for subjective questions.
- NEVER truncate wrong/missing comments. Write the full explanation every time.

═══════════════════════════════════
TOPIC CLASSIFICATION
═══════════════════════════════════
Always populate:
- chapterTopic: GRANULAR sub-concept, NEVER chapter name. Test: if it could be a textbook chapter title, go deeper. Bad: "Matrices". Good: "Cayley-Hamilton Theorem". Fill even for correct answers.

INTELLIGENCE CLASSIFICATION (1 FIELD — MANDATORY)
═══════════════════════════════════
For EVERY question, populate this field.

1. questionType — cognitive demand of this question:
   RECALL, CONCEPTUAL, NUMERICAL, DERIVATION, DIAGRAM, APPLICATION, ASSERTION_REASON.
   ALWAYS set, even full marks.


ANTI-BLEED: Grade each question ONLY against its own text block. Never use content from one question when grading another.

HALLUCINATION PREVENTION: Only reference content that actually appears in the student's OCR text. Never invent student quotes.

OUTPUT: Single valid JSON array. No text outside the array.
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
- MCQ: has A/B/C/D options explicitly listed.
- VSA: 1-2 marks, no options. (one-word, phrase, name, define, state)
   - Assertion-Reason: contains "Assertion" and "Reason".

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
    - VSA, SA, LA, Case Study:
      - **IF a solution key is provided:** Scan the solution key images for any marking scheme, step labels, bracketed marks, or annotation tables next to this question's answer. If found, TRANSCRIBE them VERBATIM into "step_marking". Do NOT generate your own.
      - **IF no solution key, OR if the solution key has no visible rubric for this question:** Generate a logical 2-3 step breakdown based on the answer's complexity.
      - Format: "Step 1: [Description] ([Marks]); Step 2: [Description] ([Marks])..."

12. TOPIC ANCHORS (CRITICAL):
    - For every question, generate "topicAnchors": array of 3-5 highly unique technical terms, proper nouns, or specific values from THIS question/answer only.
    - Generic words like "Calculate" or "Question" are FORBIDDEN.

13. IMAGE / DIAGRAM HANDLING:
    - If a question contains a diagram: write a detailed structural text description in "imagePrompt".
    - If no diagram: "imagePrompt" = null.
    - For PHYSICS questions with charge/force diagrams, imagePrompt MUST include:
      (a) Exact charge signs and positions (e.g. "+Q at left end, +Q at right end, q at center")
      (b) Distance labels between charges (e.g. "distance r between each adjacent charge")
      (c) What the student must show for full marks (e.g. "student must derive q = -Q/4 and show force balance")
      (d) Whether proof order matters (write "any valid proof order accepted" for equilibrium proofs)
    - For equilibrium questions: always add to checkingInstructions: "Accept any valid proof method — force balance on middle charge OR outer charge. Both give q = -Q/4. Method order does not matter."

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
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif', 'application/pdf'];
        const mimeType = allowedTypes.find(t => contentType.includes(t)) || 'image/jpeg';
        const isPdf = mimeType === 'application/pdf' || url.toLowerCase().split('?')[0].endsWith('.pdf');

        const buffer = await response.arrayBuffer();
        const base64 = Buffer.from(buffer).toString('base64');

        // Memory is automatically freed after this function returns — no /tmp or Storage writes

        if (isPdf) {
            // Tag as PDF so processGradingJob can expand into N virtual page slots.
            // _isPdf is routing metadata only — stripped before sending to Gemini.
            return { inlineData: { mimeType: 'application/pdf', data: base64 }, _isPdf: true };
        }
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

async function extractTextFromImages(imageParts, subject, jobId, jobData, allRules, hasSections = false) {

const totalPages = imageParts.length;
const masterIds = jobData.questions.map(q => q.questionNumber);
    const pageResults = [];
    const HEAVY_SUBJECTS = ['physics', 'chemistry', 'biology', 'science', 'accounts', 'accountancy', 'maths', 'mathematics', 'commerce', 'economics', 'statistics'];
    const isHeavySubject = subject && HEAVY_SUBJECTS.some(s => subject.trim().toLowerCase().includes(s));
    const BATCH_SIZE = isHeavySubject ? 2 : 4;

    const validNums = jobData.questions.map(q => q.questionNumber).join(', ');
    const contextualRules = `\n# CONTEXTUAL AWARENESS:\nThe valid question numbers for this exam are: ${validNums}.\nIf a handwritten digit is ambiguous, prefer a number from this list.\n`;

const ocrTargetRules = ''; // Domain rules (Firestore gradingCorrectionRules) removed from OCR prompt per request.

    // Hindi LANGUAGE (not subject!) gets the OCR addon appended AFTER the base
    // instruction. A student may write Physics in Hindi or English — gating is
    // by answerLanguage so the same Physics paper is handled differently
    // depending on what script the student wrote in.
    // Base instruction is NEVER modified.
    const hindiOcrAddon = isHindiLanguage(jobData) ? HINDI_OCR_ADDON : '';
    

    const csOcrAddon = (() => {
    const s = (subject || '').trim().toLowerCase();
    return (s.includes('computer') || s.includes('python') || s.includes('programming') ||
            s.includes('coding') || s.includes('informatics') || s.includes('information technology') ||
            s === 'cs' || s === 'it')
        ? COMPUTER_SCIENCE_OCR_ADDON : '';
})();

const physicsOcrAddon = (() => {
    const s = (subject || '').trim().toLowerCase();
    return s.includes('physics') ? PHYSICS_OCR_ADDON : '';
})();



    // Subject-gated content-type addons. OFF by default → English papers see
    // exactly the v42 OCR prompt → librarian mapping works as before.
const numericalTableOcrAddon = (ENABLE_DATATABLE_OCR_ADDON && isDataTableSubject(subject))
    ? NUMERICAL_TABLE_OCR_ADDON
    : '';




    const finalOcrInstruction = OCR_SYSTEM_INSTRUCTION + contextualRules +
        "\n# CRITICAL: Perform a hard reset of your state for every image. " +
        "DO NOT repeat characters. " +
        "Stop immediately when done.\n" +
      
        numericalTableOcrAddon +
        hindiOcrAddon +
       csOcrAddon +
    physicsOcrAddon;

    // ── PDF FAST PATH ────────────────────────────────────────────────────────
    // Fires ONLY when imageParts[0] is a PDF inlineData blob (API path).
    // The Teacher PWA uses fileData GCS URIs → isPdfJob = false → normal batch loop.
    // Cost: identical to images (1 PDF page = 1 image token, per Google pricing).
    // Why one call instead of batches: a PDF blob can't be sliced without conversion.
    // Sending it once is correct, cheaper (no repeated system prompt), and faster.
const isPdfJob = imageParts.length > 0 && (
    imageParts[0].inlineData?.mimeType === 'application/pdf' ||
    imageParts[0].fileData?.mimeType === 'application/pdf'
);

    if (isPdfJob) {
     const pdfPart = imageParts[0].inlineData
    ? { inlineData: imageParts[0].inlineData }
    : { fileData: imageParts[0].fileData }; // GCS URI path
        const totalPdfPages = imageParts.length; // virtual page count set by processGradingJob

        await db.collection('gradingQueue').doc(jobId).update({
            statusDetails: `Reading PDF answer sheet (${totalPdfPages} pages)...`,
            currentStep: 1,
            progress: 5
        });

        const pageInstructions = Array.from({ length: totalPdfPages }, (_, n) =>
            `- Page ${n + 1} MUST start with: [PAGE ${n + 1}]`
        ).join('\n');

        // Attempt once; on failure fall back to per-page recovery (same as image path)
        let pdfTranscript = null;
        let pdfAttempt = 0;
        while (pdfAttempt < 2 && pdfTranscript === null) {
            try {
                const refreshModifier = pdfAttempt > 0
                    ? "\n\nCRITICAL: Previous attempt failed. RESET state. No repetition. Stop when done."
                    : "";

                const pdfOcrModel = vertex_ai.getGenerativeModel({
                    model: 'gemini-2.5-flash',
                    systemInstruction: { parts: [{ text: finalOcrInstruction + refreshModifier }] }
                });

                const result = await callGeminiWithRetry(pdfOcrModel, {
                    contents: [{
                        role: 'user',
                        parts: [
                            pdfPart,
                            {
                                text: `This is a multi-page PDF student answer sheet with ${totalPdfPages} pages.\nExtract ALL handwriting from EVERY page sequentially.\nCRITICAL PAGE MARKERS (MANDATORY — do not skip any page):\n${pageInstructions}\nIf a page has no handwriting, output exactly: [PAGE N]\\n[NO HANDWRITING DETECTED]. \n\nCRITICAL BUDGET RULE: Each [DIAGRAM]...[/DIAGRAM] block MUST be under 1000 words. Flowcharts must be under 800 words. Physics/geometry diagrams must be under 1000 words.
 EXCEPTION: Accounts tables (journal, ledger, BRS, capital accounts) use [TABLE]...[/TABLE] and must transcribe ALL rows completely — no truncation. After closing [/DIAGRAM] or [/TABLE], immediately continue to the next line of handwriting. DO NOT expand diagram descriptions beyond these limits.`
                            }
                        ]
                    }],
generationConfig: {
                        candidateCount: 1,
                        seed: 42,
temperature: 0.15,
                        topP: 0.5,
                        maxOutputTokens: 16000,
 // 30 pages of handwriting needs ~10k-20k tokens; 4096 truncates at page 3
                        thinkingConfig: { thinkingBudget: 0 }
                    }
                });

const rawText = result.response.candidates[0].content.parts[0].text;
                const parsedOcr = { text: (rawText || '').trim() };

                if (!parsedOcr.text || parsedOcr.text.length < 5) throw new Error("EMPTY_OCR_OUTPUT");

const loopRegex = /(?!\\\\|& |\[#P:|\[PAGE|\[QLABEL|\[DIAGRAM|\[\/DIAGRAM|\[\d|\[0|\[1|\||TO |Dr\.|Cr\.|Capital|TABLE|WORKING|Kq|KQ|ε₀|epsilon|θ|μ|λ|σ|ω|η|→|[\u0900-\u097F]|\\frac|\\pi|sin|cos|tan|[A-Z]_)(?=[^\n]*[a-zA-Z])(.{1,20})\1{450,}/;
if (loopRegex.test(parsedOcr.text)) throw new Error("REPETITION_LOOP");


let sanitizedOcr = parsedOcr.text;
sanitizedOcr = sanitizedOcr.replace(
       /\[DIAGRAM\]([\s\S]{2500,}?)\[\/DIAGRAM\]/g,
    (match, content) => {
        const truncated = content.substring(0, 2000).trim();
        // Find last complete sentence to avoid mid-word cut
        const lastDot = Math.max(truncated.lastIndexOf('. '), truncated.lastIndexOf('.\n'));
        const cleanEnd = lastDot > 400 ? truncated.substring(0, lastDot + 1) : truncated;
        console.warn(`[OCR] Diagram block truncated: was ${content.length} chars, cut to ${cleanEnd.length}`);
        return `[DIAGRAM] ${cleanEnd} [PROOF SUMMARY: Diagram truncated due to length.] [/DIAGRAM]`;
    }
);
pdfTranscript = sanitizedOcr;



                // Log usage OUTSIDE critical path — Firestore failure must NOT trigger OCR retry
                try {
                    const usage = result.response.usageMetadata;
db.collection('apiUsageLogs').add({
    timestamp: admin.firestore.FieldValue.serverTimestamp(),
    teacherUid: jobData.teacherUid,
    teacherName: jobData.teacherName || 'Unknown',
    schoolId: jobData.schoolId || 'N/A',
    schoolName: jobData.schoolName || 'Unknown School',
    appType: 'teacher',
    feature: 'Assessment Checker OCR (Backend)',
    modelCalled: 'gemini-2.5-flash',
    unitsConsumed: {
      imagesProcessed: totalPdfPages
    },
    tokenUsage: {
        promptTokens: usage.promptTokenCount,
        candidatesTokens: usage.candidatesTokenCount,
        thinkingTokens: usage.thoughtTokenCount || 0,
        cachedTokens: usage.cachedContentTokenCount || 0,
        totalTokens: usage.totalTokenCount
    },
// Cached tokens bill at 10% of the standard input rate (implicit caching discount).
totalCostInr: (((usage.promptTokenCount - (usage.cachedContentTokenCount || 0)) / 1000000) * 27.6) + (((usage.cachedContentTokenCount || 0) / 1000000) * 2.76) + ((usage.candidatesTokenCount / 1000000) * 230) + (((usage.thoughtTokenCount || 0) / 1000000) * 34.5)
                    }).catch(e => console.warn('[OCR] PDF usage log write failed (non-critical):', e.message));
                } catch(e) { /* non-critical */ }

            } catch (pdfErr) {
                pdfAttempt++;
                if (pdfAttempt >= 2) {
                    console.warn(`[OCR] PDF OCR failed after 2 attempts: ${pdfErr.message}. Filling pages with RECOVERY FAILED.`);
                    pdfTranscript = "[PDF OCR FAILED]";
                }
            }
        }

// Normalize all PAGE marker variants before splitting
// Handles: [PAGE3], [Page 3], [ PAGE  3 ], [pg 3], etc.
const normalizedTranscript = (pdfTranscript || transcript || "")
    .replace(/\[\s*(?:page|pg)\s*(\d+)\s*\]/gi, '[PAGE $1]')
    .replace(/(\[DIAGRAM\][\s\S]*?)(\[PAGE\s+\d+\])/g,
        (match, diagramPart, pageMarker) => `${diagramPart}[/DIAGRAM]\n${pageMarker}`
    );
const pageSplitter = /\[PAGE\s+(\d+)\]/gi;
const segments = normalizedTranscript.split(pageSplitter);

        for (let pg = 0; pg < totalPdfPages; pg++) {
            const targetPageNum = pg + 1;
            const segmentIndex = segments.findIndex((val, sIdx) => sIdx % 2 === 1 && parseInt(val, 10) === targetPageNum);
            let pageContent = segmentIndex !== -1 ? (segments[segmentIndex + 1] || "") : "[NO HANDWRITING DETECTED]";
            let sanitized = pageContent.trim().replace(/\[#P\s*:\s*\d+\s*,/gi, `[#P:${targetPageNum},`);
            // Strip '#' only when it appears inside LaTeX math delimiters \( ... \) or \[ ... \].
            // Outside math, '#' is valid (e.g. C#, #1, numbered lists) — do NOT touch it.
            // Inside math, '#' is a reserved macro parameter character in TeX/KaTeX and
            // causes: "You can't use 'macro parameter character #' in math mode"
            sanitized = sanitized.replace(/(\\\([\s\S]*?)\\\)/g, (m) => m.replace(/#/g, ''));
            sanitized = sanitized.replace(/(\\\[[\s\S]*?)\\\]/g, (m) => m.replace(/#/g, ''));

             // Post-OCR QLABEL injection (same as image batch path — see comment there)
            // Pass 0: Normalize Unicode circled chars → ASCII.
// Pass 0: Normalize Unicode circled/special chars → ASCII.
// Pass 0: Normalize Unicode circled/special chars → ASCII.
{
    const CN = '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳';
    const CU = 'ⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏ';
    const CL = 'ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩ';

    // Step 1: When circled numbers ①②③ appear RIGHT AFTER a question label
    // (e.g. "Ans-8 ①", "8. ①"), they mean sub-part (i)(ii)(iii) — NOT the digit.
    // Replace BEFORE the global CN→digit pass to avoid "Ans-8 ①" → "81".
    const romanSubs = ['(i)','(ii)','(iii)','(iv)','(v)','(vi)','(vii)','(viii)','(ix)','(x)'];
    sanitized = sanitized.replace(/((?:A[nu][st]s?\.?\s*)?\d{1,2}[.) ]\s*)([①②③④⑤⑥⑦⑧⑨⑩])/g,
        (m, prefix, circ) => {
            const idx = CN.indexOf(circ);
            return (idx >= 0 && idx < romanSubs.length) ? prefix + romanSubs[idx] : m;
        });

    // Step 2: Global circled char → ASCII (remaining cases: standalone ① in answers)
    for (let i = 0; i < CN.length; i++) sanitized = sanitized.split(CN[i]).join(String(i+1));
    for (let i = 0; i < CU.length; i++) {
        sanitized = sanitized.split(CU[i]).join(String.fromCharCode(65+i));
        sanitized = sanitized.split(CL[i]).join(String.fromCharCode(97+i));
    }

    // Step 3: © is visually identical to handwritten (c) — Gemini's most common misread.
    sanitized = sanitized.replace(/©/g, '(c)');
    sanitized = sanitized.replace(/®/g, '(r)');
}
            // Pass 0.5: Mid-line question label injection.
            // Fires ONLY when a known header keyword (Model Test, Section, Paper, Exam)
            // appears before the question number on the same OCR line.
            // e.g. "Model Test - 2 Section - B 18) E₁ = 1.5V [#P:1,170,870]"
            // Deliberately narrow — prevents false-firing on answer lines like "b) Polar [#P:]"
            sanitized = sanitized.replace(
                /^((?:[^\n\[]*?\b(?:model\s*test|section|paper|exam)\b[^\n\[]*?)\s)(\d{1,2}[).])\s(?!.*\[QLABEL:)(.*\[#P:)/gim,
                (match, prefix, label, rest) => {
                    if (match.includes('[QLABEL:')) return match;
                    return `${prefix}${label} [QLABEL:${label}] ${rest}`;
                }
            );
            // Pass 1: Standard "1)" or "1." format  e.g. "1) answer [#P:...]"
            sanitized = sanitized.replace(
                /^(\d{1,2}[).\]])(\s)(?!.*\[QLABEL:)(.*\[#P:)/gm,
                (match, label, space, rest) => `${label} [QLABEL:${label}]${space}${rest}`
            );

            // Pass 1b: Multi-line long answer label injection.
            // Pass 1 requires [#P:] on the SAME line as the question number.
            // For long answers (e.g. "33) B) (i) lambda = n²/R\n= For balmer...\n[#P:7,640,910]")
            // the [#P:] tag is several lines below the label line — Pass 1 misses it.
            // This pass injects [QLABEL:N)] on any line that STARTS with a digit+paren/dot
            // and has NO [QLABEL:] already — regardless of whether [#P:] is present.
            // Guard: line must start with digit, then paren/dot, then a space and more content.
            // This is safe: the boundary resolver ignores QLABELs that don't match any master.
            sanitized = sanitized.replace(
                /^(\d{1,2}[).\]])\s(?!.*\[QLABEL:)/gm,
                (match, label) => {
                    if (match.includes('[QLABEL:')) return match;
                    return `${label} [QLABEL:${label}] `;
                }
            );

            // Pass 1c: Mid-line question label injection after [#P:] tag.
            // Fires when OCR returns the entire page as one long line with inline [#P:] tags.
            // e.g. "...cosφ [#P:6,450,790] 32) A) a) I = I₁+I₂ [#P:6,520,790]"
            // Pass 1 uses ^ anchor so it only catches labels at LINE START — misses mid-line.
            // Pattern: [#P:N,y,x] <space> DIGIT+PAREN <space> (no existing [QLABEL:] after)
            // Safe: only fires after a [#P:] tag, so random numbers in equations won't match.
            sanitized = sanitized.replace(
                /(\[#P:\d+,\d+,\d+\])\s+(\d{1,2}[).])\s(?![^\[]*\[QLABEL:)/g,
                (match, ptag, label) => {
                    if (match.includes('[QLABEL:')) return match;
                    return `${ptag} ${label} [QLABEL:${label}] `;
                }
            );

// ── UNIVERSAL ANS-KEYWORD NORMALIZER ─────────────────────────────────────────
// OCR commonly misreads handwritten "Ans" as: Aus, Auz, Ams, Aas, Ens, AUS,
// ANO, ANB, EUS, and similar. Also handles "Answer no N" box labels from
// students who write full "Answer" word. Normalize ALL to "Ans".
// Circled question numbers after Ans (from Physics CBSE style) also handled:
// "Ans. ①" → "Ans 1", "Ans. ②" → "Ans 2" etc.
sanitized = sanitized.replace(
    /\b(A(?:n[szbdtuw]|u[stz]|e[sn]|m[s])|E(?:us|ns|as)|ANO|ANB|AUS|ENS)\b/gi,
    'Ans'
);
// "Answer no N" / "Answer number N" → normalize to "Ans N" for Pass 9
sanitized = sanitized.replace(
    /\b(Answer\s+(?:no\.?|number|num\.?)\s*)(\d{1,2})\b/gi,
    (m, prefix, num) => `Ans ${num}`
);
// Circled digit immediately after Ans keyword = question number, not sub-part
// "Ans. ①" → "Ans 1", "Ans ②" → "Ans 2"
{
    const CN = '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳';
    sanitized = sanitized.replace(
        /\b(Ans\.?\s*)([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])/gi,
        (m, prefix, circ) => {
            const idx = CN.indexOf(circ);
            return idx >= 0 ? `${prefix}${idx + 1}` : m;
        }
    );
}
            // Pass 2: MCQ "N letter." format  e.g. "1 c. Both age... [#P:...]"
            // OCR writes the question number, then a space, then the answer letter+dot.
            // e.g. "1 c. Both age and soundness of mind [#P:2,188,930]"
            // Inject [QLABEL:N)] after the number so resolver maps it to master Q-N.
            // Result: "1 [QLABEL:1)] c. Both age and soundness of mind [#P:2,188,930]"
            sanitized = sanitized.replace(
                /^(\d{1,2})( [a-eA-E][.)])/gm,
                (match, num, rest) => {
                    if (match.includes('[QLABEL:')) return match;
                    return `${num} [QLABEL:${num})]${rest}`;
                }
            );
            // Pass 3: MCQ bare "N letter [#P:]" format  e.g. "1 B [#P:1,180,930]"
            // OCR writes: question number, space, answer letter, space, then [#P:...]
            // No dot or paren after the letter — the letter IS the entire answer.
            // e.g. "1 B [#P:1,180,930]", "6 C [#P:1,280,930]"
            // Safe: requires line to START with digit — won't match table rows like "A 9 2 7"
            sanitized = sanitized.replace(
                /^(\d{1,2}) ([A-Ea-e]) (\[#P:)/gm,
                (match, num, letter, tag) => {
                    if (match.includes('[QLABEL:')) return match;
                    return `${num} [QLABEL:${num})] ${letter} ${tag}`;
                }
            );
            // Pass 4: Mid-line MCQ pattern — fires when OCR merges a page header
            // with the first MCQ answer on the same line.
            // e.g. "Harakrishna Law Rapid test 1 MCQ 1 c. Both age... [#P:2,188,930]"
            // Lines starting with a digit are already caught by Pass 1/2/3 (^ anchor).
            // This pass handles lines that START with alphabetic text but CONTAIN
            // "N letter. answer [#P:]" mid-line — typically the first MCQ on a page
            // when OCR reads the page header and Q1 as a single continuous line.
            // Guard: only fires on non-digit-start lines with a [#P:] tag present.
            // Safety: the boundary resolver discards any QLABEL with no matching master ID,
            // so false positives in prose are automatically neutralised.
            if (!/^\d/.test(sanitized.trim()) && sanitized.includes('[#P:')) {
                const segParts = sanitized.split(/(\[#P:\d+,\d+,\d+\])/);
                let rebuilt = '';
                for (let si = 0; si < segParts.length; si++) {
                    const seg = segParts[si];
                    if (/^\[#P:\d+,\d+,\d+\]$/.test(seg)) { rebuilt += seg; continue; }
                    const nextSeg = segParts[si + 1];
                    if (!nextSeg || !/^\[#P:/.test(nextSeg)) { rebuilt += seg; continue; }
                    const mcqRe = /(?<![a-zA-Z])(\d{1,2}) ([a-eA-E][.)])\s/g;
                    let lastM = null, mm;
                    while ((mm = mcqRe.exec(seg)) !== null) lastM = mm;
                    if (lastM && !seg.includes('[QLABEL:')) {
                        const insertAt = lastM.index + lastM[1].length;
                        rebuilt += seg.slice(0, insertAt) + ` [QLABEL:${lastM[1]})]` + seg.slice(insertAt);
                    } else {
                        rebuilt += seg;
                    }
                }
 sanitized = rebuilt;
            }
            // Pass 5: "Ans. N (roman)" and "Ans N (roman)" format injection.
            // Handles: "Ans. 8 (i)", "Ans 8 (ii)", "8 (i)", "8 (ii)" etc.
            // These lines start with "Ans" or a digit, contain a parenthesized
            // roman numeral sub-part, and have a [#P:] tag somewhere on the line.
            // If no [QLABEL:] is already present, inject one immediately after the full label.
            sanitized = sanitized.replace(
                /^((?:A[nu][st]s?[-.]?\s*)?\d{1,2}\s*[\(\[](i{1,4}|iv|vi{0,3}|ix|xi{0,3}|ii)[\)\]])(.*?)(\[#P:)/gm,
                (match, label, roman, middle, tag) => {
                    if (match.includes('[QLABEL:')) return match;
                    const cleanLabel = label.trimEnd();
                    return `${cleanLabel} [QLABEL:${cleanLabel}]${middle}${tag}`;
                }
            );

// Pass 6: Mid-line MCQ QLABEL injection.
            // Fires when OCR writes all MCQ answers as one continuous line/paragraph.
            // e.g. "...[#P:1,260,730]2. (d) Either 0V...[#P:1,300,930]3. (b) 3:2..."
            // Passes 1-5 use ^ line-start anchors so they miss labels mid-line.
            // This injects [QLABEL:N.] after each [#P:] tag that is followed by "N. (x)" pattern.
            sanitized = sanitized.replace(
                /(\[#P:\d+,\d+,\d+\])(\s*)(\d{1,2}\.)\s*(\([a-eA-E]\)|[a-eA-E][.)])/g,
                (match, ptag, space, num, letter) => {
                    if (match.includes('[QLABEL:')) return match;
                    const cleanNum = num.replace('.', '');
                    return `${ptag}${space}${num} [QLABEL:${num}] ${letter}`;
                }
            );

            // Pass 6b: Mid-line roman-numeral sub-part QLABEL injection.
            // Fires when OCR writes ALL sub-parts of a question on ONE line:
            // e.g. "Ans 1 [QLABEL:Ans 1] i) (a)... [#P:1,240,490] ii) fixed investments [#P:1,320,490]"
            // The [#P:] tag after sub-part i) is followed by "ii)" — no QLABEL for ii).
            // This pass finds every [#P:...] followed by a roman-numeral sub-part label (i/ii/iii/iv)
            // and injects [QLABEL:PARENTNUM (roman)] using the last parent number seen on this line.
            sanitized = sanitized.replace(
                /(\[#P:\d+,\d+,\d+\])\s+(i{1,4}|iv|vi{0,3}|ix)\s*\)/g,
                (match, ptag, roman) => {
                    if (match.includes('[QLABEL:')) return match;
                    // Find the last "Ans N" parent number visible before this point in the line
                    const beforeMatch = sanitized.substring(0, sanitized.indexOf(match));
                    const parentMatch = beforeMatch.match(/(?:Ans\.?\s*)(\d{1,2})(?:\s|\])/gi);
                    const lastParent = parentMatch
                        ? parentMatch[parentMatch.length - 1].match(/(\d{1,2})/)[1]
                        : null;
 const label = lastParent ? `Ans ${lastParent} (${roman})` : roman;
                    return `${ptag} ${roman}) [QLABEL:${label}]`;
                }
            );

            // Pass 6d (PDF path): Line-start orphan roman sub-part with [#P:] on same line.
            // Handles: "(ii) Yes it is true [#P:1,150,50]" — no parent digit at line start.
            // Guard: [#P:] MUST be on same line (lookahead). This prevents firing on internal
            // step numbering within long answers (those lines typically lack [#P:] anchors).
            // Parent number: look back for last [QLABEL:N] in transcript so far.
            sanitized = (() => {
                let lastParentNum = null;
                return sanitized.replace(
                    /^(\s*)(\[QLABEL:[^\]]*?(\d{1,2})[^\]]*?\])|^(\s*)([(](i{1,4}|iv|vi{0,3}|ix)[)])(\s)(?=.*\[#P:)/gm,
                    (match, i1, qlabel, qnum, i2, romanFull, roman, space) => {
                        if (qlabel) { lastParentNum = qnum; return match; }
                        if (!romanFull) return match;
                        if (match.includes('[QLABEL:')) return match;
                        if (!lastParentNum) return match;
                        return `${i2 || ''}${romanFull} [QLABEL:${lastParentNum} (${roman})]${space}`;
                    }
                );
            })();

                        // Pass 7: MCQ dense format "N) letter" with NO dot after letter (PDF path)
            sanitized = sanitized.replace(
                /^(\d{1,2})\)\s+([a-d])\s+([^[\#]+?)(\[#P:\d+,\d+,\d+\])/gm,
                (match, num, letter, content, tag) => {
                    if (match.includes('[QLABEL:')) return match;
                    return `${num}) [QLABEL:${num})] ${letter} ${content.trim()} ${tag}`;
                }
            );

            // Pass 8: MCQ format "1) c" where answer letter is the ONLY thing before [#P:] (PDF path)
            sanitized = sanitized.replace(
                /^(\d{1,2})\)\s+([a-d])\s+(\[#P:\d+,\d+,\d+\])/gm,
                (match, num, letter, tag) => {
                    if (match.includes('[QLABEL:')) return match;
                    return `${num}) [QLABEL:${num})] ${letter} ${tag}`;
                }
            );


// Pre-Pass: Normalize OCR-corrupted "Ans N" variants → "Ans N" before Pass 9 fires.
// Catches: "Anes10", "Anss 10", "An5 13", "Ane 12", "ANes 9" etc.
// Guard: only fires if digit matches a known master question number.
{
    const masterNumSet = new Set(
        masterIds.map(id => String(id).match(/(\d+)/)?.[1]).filter(Boolean)
    );
    sanitized = sanitized.replace(
        /\b(A[a-z]{1,5}\.?\s{0,2})(\d{1,2})\b/gi,
        (match, prefix, num) => {
            if (match.includes('[QLABEL:')) return match;
            const p = prefix.replace(/[\s.]/g, '').toLowerCase();
            if (!p.startsWith('an')) return match;
            if (p.length > 7) return match;
            if (/^(and|any|ant|anti|another|animal|analysis|angle|annual|answer)/.test(p)) return match;
            if (!masterNumSet.has(num)) return match;
            console.log(`[Pre-Pass PDF] Corrupted Ans normalized: "${match.trim()}" → "Ans ${num}"`);
            return `Ans ${num}`;
        }
    );
}

sanitized = sanitized.replace(
/^(\[?)(\s*(?:answer|ans|ques(?:tion)?|q(?:u[a-z]{0,3}|no?\\.?)?)[\\s._:-]*(?:no\\.?|number|num\\.?)?[\\s._:-]*)(\d{1,2})[xX✓√*]?\s*(?:[.:*-]\s*([a-d])\s*[).]?)?[.:*-]?\]?/gim,
    (match, bracket, prefix, num, letter) => {
        if (match.includes('[QLABEL:')) return match;
        const kw = prefix.trim().split(/[\s.\-_]/)[0] || 'Ans';
        const labelText = letter ? `${num} (${letter})` : num;
        return `${bracket}${prefix}${num}${letter ? '.' + letter + ')' : ''} [QLABEL:${kw} ${labelText}]`;
    }
);



sanitized = sanitized.replace(
    /\b((?:Ans(?:wer)?|ANS)[-.]?\s*)(?:(\d{1,2})|[(\[{](\d{1,2})[)\]}]|([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]))\s*[).\]};:,\-*]?\s*([A-Z]\s*[.:;\-]\s*)?(\([a-dA-D]\)|\([ivxIVX]+\)|\([\u0900-\u097F]\)|[ivx]{1,4}[).]\s*)?/gi,
    (match, keyword, num1, num2, circled, capSection, subPart) => {
        if (match.includes('[QLABEL:')) return match;
        const CIRCLED = '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳';
        const num = num1 || num2 || (circled ? String(CIRCLED.indexOf(circled) + 1) : '');
        if (!num) return match;
        const cap = capSection ? capSection.trim().replace(/[.:;\-\s]+$/, '') : '';
        const sub = subPart ? subPart.trim() : '';
const labelText = [num, cap, sub].filter(Boolean).join(' ');
        return `${match.trimEnd()} [QLABEL:Ans ${labelText}]`;
    }
);

// Pass 9c (HINDI ONLY — PDF path): "Ans 3.(क)" / "Ans 3.(ख)" sub-part injection.
// Pass 9b handles Latin sub-parts like (a)(b)(i)(ii). But Hindi papers use
// Devanagari letters क/ख/ग/घ/ड as sub-part labels: "Ans 3.(क)", "Ans 3 (ख)".
// Pass 9b's optional group now captures \([\u0900-\u097F]\) but if the OCR
// transcript has "Ans 3.(क)" with a dot before paren, Pass 9b may miss the dot.
// This pass is an explicit safety net: fires only for Hindi LANGUAGE papers.
if (isHindiLanguage(jobData)) {
sanitized = sanitized.replace(

        /\b((?:Ans(?:wer)?|ANS)\.?[-]?\s*)(\d{1,2})\s*[.\s]*\(([कखगघङड़])\)/gi,
        (match, keyword, num, hindiLetter) => {
            if (match.includes('[QLABEL:')) return match;
            const labelText = `${num} (${hindiLetter})`;
            return `${match.trimEnd()} [QLABEL:${labelText}]`;
        }
    );
}

// Pass 9d (HINDI LANGUAGE ONLY — PDF path): Bare "N. क)" format — no Ans keyword.
// Catches ALL these student writing styles:/gi,
//   "6. क)"   "6 क)"   "6.(क)"   "6 (क)"   "6. क"   "6 क"
//   Mid-line:  "...text 6. क) answer text..."
// Fires only for Hindi LANGUAGE papers. Safe: boundary resolver discards
// QLABELs with no matching master ID.
if (isHindiLanguage(jobData)) {
sanitized = sanitized.replace(

        /(?<![A-Za-z])(\d{1,2})\s*[.)]*\s*\(?([कखगघङचछजझटठडढणतथदधनपफबभमयरलवशषसहड़ढ़])\)?/gi,
        (match, num, hindiLetter) => {
            if (match.includes('[QLABEL:')) return match;
            const labelText = `${num} (${hindiLetter})`;
            return `${match.trimEnd()} [QLABEL:${labelText}]`;
        }
    );
}
sanitized = sanitized.replace(
    /\b(Ques(?:tion)?[\s.\-]*)(\d{1,2})\b\s*[).]?\s*(?!\S*\[QLABEL:)/gm,
    (match, prefix, num) => {
        if (match.includes('[QLABEL:')) return match;
        return `${match.trimEnd()} [QLABEL:${num}]`;
    }
);

// Pass 9f (ALL PAPERS — PDF path): Hindi "प्रश्नोत्तर सं – N" format.
// Students write "प्रश्नोत्तर सं – 1", "प्रश्नोत्तर सं- 2" etc. instead of "Ans 1".
// None of Passes 1-9e handle this — all miss it → zero QLABELs → unmapped.
// Also handles: "उत्तर – N", "प्रश्न – N", "उत्तर सं N"
sanitized = sanitized.replace(
    /(?:प्रश्नोत्तर|उत्तर|प्रश्न)\s*(?:सं|संख्या|नं|नंबर|सं\.?)?\s*[-–—]?\s*(\d{1,2})\b/gi,

    (match, num) => {
        if (match.includes('[QLABEL:')) return match;
        return `${match} [QLABEL:${num}]`;
    }
);

// Pass 9g: "Ans. N. Ni)" duplicate-number sub-part recovery.
// After nested QLABEL collapse, OCR sometimes produces: Ans. 23. 23i) [QLABEL:Ans 23]
// The repeated number + sub-part letter (23i, 23ii) is an artifact of double OCR.
// This pass rebuilds the correct sub-part QLABEL.
sanitized = sanitized.replace(
    /(\[QLABEL:\s*(?:Ans\.?\s*)?(\d{1,2})\s*\])\s*\2\s*(i{1,4}|iv|vi{0,3}|ix)\s*\)/gi,
    (match, qlabel, num, roman) => {
        return `[QLABEL:Ans ${num}.(${roman})]`;
    }
);

// Pass 9h: Normalize "N.v" → "N.(v)" — dot-letter without parens.
// OCR sometimes emits [QLABEL:22.v] instead of [QLABEL:22.(v)].
sanitized = sanitized.replace(
    /\[QLABEL:(\d{1,2})\.(i{1,4}|iv|vi{0,3}|ix)\]/gi,
    (match, num, roman) => `[QLABEL:${num}.(${roman})]`
);

sanitized = (() => {
    const chunkSoFar = pageResults.map(p => p.rawText || p.text).join('\n') + '\n' + sanitized;
    const numberedQls = [...chunkSoFar.matchAll(/\[QLABEL:\s*(?:Ans\.?\s*)(\d+)/gi)];
    const lastNum = numberedQls.length > 0 ? parseInt(numberedQls[numberedQls.length - 1][1], 10) : 0;
    let nextNum = lastNum;
    return sanitized.replace(
        /\[QLABEL:\s*Ans\.?\s*\](?!\s*\d)/gi,
        () => { nextNum += 1; return `[QLABEL:Ans ${nextNum}]`; }
    );
})();


// Pass 10b: "Ans (d)" or "Ans. (b)" — OCR dropped question number before MCQ letter.
// Infer number by incrementing last seen numbered QLABEL in transcript so far.
sanitized = (() => {
    const chunkSoFar = pageResults.map(p => p.rawText || p.text).join('\n') + '\n' + sanitized;
    const numberedQls = [...chunkSoFar.matchAll(/\[QLABEL:\s*(?:Ans\.?\s*)(\d+)/gi)];
    const lastNum = numberedQls.length > 0 ? parseInt(numberedQls[numberedQls.length - 1][1], 10) : 0;
    let nextNum = lastNum;
    return sanitized.replace(
        /((?:^|[\s\n])Ans\.?\s*)(\([a-eA-E]\)?|\b[a-eA-E]\b)(\s)/gm,
        (match, ansPrefix, letter, space) => {
            if (match.includes('[QLABEL:')) return match;
            nextNum += 1;
            return `${ansPrefix.trim()} ${nextNum} [QLABEL:Ans ${nextNum}] ${letter}${space}`;
        }
    );
})();
            const anchors = [];
            [...sanitized.matchAll(/\[#P\s*:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]/gi)].forEach(m => {
                anchors.push({ pageIndex: pg, y: parseInt(m[2], 10), x: parseInt(m[3], 10) });
            });
            pageResults.push({ pageNum: targetPageNum, text: sanitized, rawText: sanitized, anchors });

            await db.collection('gradingQueue').doc(jobId).update({
                progress: Math.min(25, Math.round((targetPageNum / totalPdfPages) * 25))
            });
        }

        return pageResults; // ← early exit; image batch loop below never runs for PDF jobs
    }

    // ── IMAGE BATCH LOOP (100% unchanged — runs for PWA and image-array API jobs) ──
    for (let i = 0; i < imageParts.length; i += BATCH_SIZE) {
const rawBatch = imageParts.slice(i, i + BATCH_SIZE);
        const currentBatch = await Promise.all(rawBatch.map(p => downscaleImagePart(p)));
        const endRange = Math.min(i + BATCH_SIZE, totalPages);

let attempt = 0;
        let batchSuccess = false;
        let batchWasHeavy = false;
        const batchStartResultsLen = pageResults.length;

        while (attempt < 1 && !batchSuccess) {
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

const refreshModifier = "";

                const ocrModel = vertex_ai.getGenerativeModel({
                    model: 'gemini-2.5-flash',
                    systemInstruction: { parts: [{ text: finalOcrInstruction + refreshModifier }] }
                });

const result = await callGeminiWithRetry(ocrModel, {
                    contents: [{
                        role: 'user',
                        parts: [
                            ...currentBatch,
                            {
text: `Analyze these ${currentBatch.length} images.\n  CRITICAL: You MUST use the following markers for each image:\n  ${currentBatch.map((_, idx) => `- Image ${idx + 1} must start with: [PAGE ${i + idx + 1}]`).join('\n')}\n\nRULE PRIORITY (highest to lowest):\n1. [TABLE] blocks for accounts (journal, ledger, BRS, capital accounts) — transcribe ALL rows, NO truncation, NO word limit.\n2. [DIAGRAM] blocks — HARD LIMIT 1000 words. Truncate at 1000 words, close [/DIAGRAM], move on.\n3. Right-margin / separate rough-work column — SKIP ENTIRELY per the LEGIBILITY-EXIT LAW. Do not transcribe rough work. If any region is cramped or illegible, write [illegible] ONCE and advance. NEVER emit the same formula or line more than twice — a second repeat means STOP that region immediately and move to the next legible line.\n\nCRITICAL: Never write [PAGE N] inside a [DIAGRAM] block. Always close [/DIAGRAM] before writing the next [PAGE N] marker. After closing any block, immediately continue to next handwriting line.`
                            }
                        ]
                    }],
generationConfig: {
                        candidateCount: 1,
                        seed: 42,
                        temperature: 0,
                        topP: 0.5,
                        maxOutputTokens: 16000,
                        thinkingConfig: { thinkingBudget: 0 }
                    }
                });

const ocrParts = result.response.candidates[0].content.parts;
const rawResponseText = (ocrParts.find(p => p.text && !p.thought) || ocrParts[ocrParts.length - 1]).text;
                const parsedOcr = { text: rawResponseText.trim() };

                // Check finishReason — MAX_TOKENS means response was silently truncated.
                // Log it separately so we know the cause (truncation vs hallucination vs API error).
                const finishReason = result.response.candidates[0]?.finishReason;
if (finishReason && finishReason !== 'STOP') {
                    console.warn(`[OCR] Batch pages ${i+1}-${endRange}: finishReason=${finishReason}. Truncation or safety stop.`);
                    if (finishReason === 'MAX_TOKENS') {
                        throw new Error("MAX_TOKENS_TRUNCATION");
                    }
                }

if (!parsedOcr || typeof parsedOcr.text !== 'string' || parsedOcr.text.length < 5) {
                    throw new Error("EMPTY_OCR_OUTPUT");
                }

const loopRegex = /(?!\\\\|& |\[#P:|\[PAGE|\[QLABEL|\[DIAGRAM|\[\/DIAGRAM|\[\d|\[0|\[1|\||TO |Dr\.|Cr\.|Capital|TABLE|WORKING|Kq|KQ|ε₀|epsilon|θ|μ|λ|σ|ω|η|→|[\u0900-\u097F]|\\frac|\\pi|sin|cos|tan|[A-Z]_)(?=[^\n]*[a-zA-Z])(.{1,20})\1{450,}/;
                if (loopRegex.test(parsedOcr.text)) {
                    throw new Error("REPETITION_LOOP");
                }

                // Catches long-phrase loops (e.g. repeated sentence in diagram descriptions)
                // that the short-pattern regex above physically cannot detect.
const phraseLoopRegex = /(?!\[#P:|\[PAGE|\[QLABEL|\[DIAGRAM|\[\/DIAGRAM|\[TABLE|\[\/TABLE|\[illegible\]|\[NO HANDWRITING|\[PROOF|\\frac|\\sin|\\cos|\\tan|\\theta|\\pi|\\sqrt|\\int|\\sum|\\alpha|\\beta|\\gamma|\\mu|\\lambda|\\omega|\\eta|\\Delta|\\rightarrow|[\u0900-\u097F]|TO |Dr\.|Cr\.|Capital)(?=[^\n]*[a-zA-Z])(.{20,100})\1{10,}/;
                if (phraseLoopRegex.test(parsedOcr.text)) {
                    throw new Error("PHRASE_REPETITION_LOOP");
                }

                const transcript = parsedOcr.text;
                // Diagnostic: count how many PAGE markers were found vs expected
const foundPageMarkers = (transcript.match(/\[PAGE\s+\d+\]/gi) || []).length;
if (foundPageMarkers < currentBatch.length) {
    console.warn(`[OCR] MARKER MISMATCH: Expected ${currentBatch.length} [PAGE N] markers, found ${foundPageMarkers}. Pages ${i+1}-${endRange}. Gemini may have skipped markers.`);
}
  // Auto-close any [DIAGRAM] block that swallowed a [PAGE N] marker
const fixedTranscript = transcript.replace(
    /(\[DIAGRAM\][\s\S]*?)(\[PAGE\s+\d+\])/g,
    (match, diagramPart, pageMarker) => `${diagramPart}[/DIAGRAM]\n${pageMarker}`
);
// Auto-close any [DIAGRAM] block that was never closed (truncation mid-diagram)
const fullyFixedTranscript = fixedTranscript.replace(
    /\[DIAGRAM\]([\s\S]*?)(?=\[PAGE\s+\d+\]|$)/g,
    (match, content) => content.includes('[/DIAGRAM]') ? match : `[DIAGRAM]${content}[/DIAGRAM]`
);
const pageSplitter = /\[PAGE\s+(\d+)\]/gi;
const segments = fullyFixedTranscript.split(pageSplitter);

                for (let idx = 0; idx < currentBatch.length; idx++) {
                    const absolutePageIndex = i + idx;
                    const targetPageNum = absolutePageIndex + 1;

                    let pageContent = "";
                    const segmentIndex = segments.findIndex((val, sIdx) => sIdx % 2 === 1 && parseInt(val, 10) === targetPageNum);

                    if (segmentIndex !== -1) {
                        pageContent = segments[segmentIndex + 1] || "";
    } else if (segments[0] && segments[0].trim().length > 20) {
    // Gemini returned content but forgot the [PAGE N] marker entirely
    // For single-image batches or first page, use raw content rather than lose it
    pageContent = idx === 0 ? segments[0] : "[NO HANDWRITING DETECTED]";
    if (idx > 0) {
        console.warn(`[OCR] Page ${targetPageNum} missing [PAGE] marker in batch. Content may be lost.`);
    }
} else {
    pageContent = "[NO HANDWRITING DETECTED]";
}

                    let sanitizedChunkText = pageContent.trim().replace(/\[#P\s*:\s*\d+\s*,/gi, `[#P:${targetPageNum},`);
                    // Strip '#' only inside LaTeX math delimiters — same as PDF path above.
                    sanitizedChunkText = sanitizedChunkText.replace(/(\\\([\s\S]*?)\\\)/g, (m) => m.replace(/#/g, ''));
                    sanitizedChunkText = sanitizedChunkText.replace(/(\\\[[\s\S]*?)\\\]/g, (m) => m.replace(/#/g, ''));

                    // ── POST-OCR QLABEL INJECTION ─────────────────────────────────────────────
                    // Problem: Dense MCQ blocks (e.g. page 11) have clear "10) (C) 2" patterns
                    // that OCR transcribed correctly but did NOT emit [QLABEL:10)] for, because
                    // the visual cues were not strong enough for the model.
                    // Fix: Regex scan for patterns "N)" or "N." at line start WITH a [#P] tag
                    // on the same line, but WITHOUT a preceding [QLABEL:...] on that line.
                    // Inject [QLABEL:N)] immediately after the number pattern.
                    //
                    // Pattern: line starts with digits followed by ) or . then space/letter
                    // and the line has a [#P:...] tag (so it is real content, not a stray number)
                    // and does NOT already have [QLABEL:
                    // Pass 1: Standard "1)" or "1." format
  // Pass 0: Normalize Unicode circled/special chars → ASCII.
{
    const CN = '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳';
    const CU = 'ⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏ';
    const CL = 'ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩ';

    // Step 1: When circled numbers ①②③ appear RIGHT AFTER a question label
    // (e.g. "Ans-8 ①", "8. ①"), they mean sub-part (i)(ii)(iii) — NOT the digit.
    // Replace BEFORE the global CN→digit pass to avoid "Ans-8 ①" → "81".
    const romanSubs = ['(i)','(ii)','(iii)','(iv)','(v)','(vi)','(vii)','(viii)','(ix)','(x)'];
    sanitizedChunkText = sanitizedChunkText.replace(/((?:A[nu][st]s?\.?\s*)?\d{1,2}[.) ]\s*)([①②③④⑤⑥⑦⑧⑨⑩])/g,
        (m, prefix, circ) => {
            const idx = CN.indexOf(circ);
            return (idx >= 0 && idx < romanSubs.length) ? prefix + romanSubs[idx] : m;
        });

    // Step 2: Global circled char → ASCII (remaining cases: standalone ① in answers)
    for (let i = 0; i < CN.length; i++) sanitizedChunkText = sanitizedChunkText.split(CN[i]).join(String(i+1));
    for (let i = 0; i < CU.length; i++) {
        sanitizedChunkText = sanitizedChunkText.split(CU[i]).join(String.fromCharCode(65+i));
        sanitizedChunkText = sanitizedChunkText.split(CL[i]).join(String.fromCharCode(97+i));
    }

    // Step 3: © is visually identical to handwritten (c) — Gemini's most common misread.
    sanitizedChunkText = sanitizedChunkText.replace(/©/g, '(c)');
    sanitizedChunkText = sanitizedChunkText.replace(/®/g, '(r)');
}

                    // Pass 0.5: Mid-line question label injection.
                    // Fires ONLY when a known header keyword (Model Test, Section, Paper, Exam)
                    // appears before the question number on the same OCR line.
                    // e.g. "Model Test - 2 Section - B 18) E₁ = 1.5V [#P:1,170,870]"
                    // Deliberately narrow — prevents false-firing on answer lines like "b) Polar [#P:]"
                    sanitizedChunkText = sanitizedChunkText.replace(
                        /^((?:[^\n\[]*?\b(?:model\s*test|section|paper|exam)\b[^\n\[]*?)\s)(\d{1,2}[).])\s(?!.*\[QLABEL:)(.*\[#P:)/gim,
                        (match, prefix, label, rest) => {
                            if (match.includes('[QLABEL:')) return match;
                            return `${prefix}${label} [QLABEL:${label}] ${rest}`;
                        }
                    );

                    // Pass 1: Standard "1)" or "1." format
                    sanitizedChunkText = sanitizedChunkText.replace(
                        /^(\d{1,2}[).\]])(\s)(?!.*\[QLABEL:)(.*\[#P:)/gm,
                        (match, label, space, rest) => `${label} [QLABEL:${label}]${space}${rest}`
                    );

                    // Pass 1b: Multi-line long answer label injection.
                    // Pass 1 requires [#P:] on the SAME line as the question number.
                    // For long answers (e.g. "33) B) (i) lambda = n²/R\n...\n[#P:7,640,910]")
                    // the [#P:] tag is several lines below — Pass 1 misses it.
                    // This pass injects [QLABEL:N)] on any line starting with digit+paren/dot
                    // that has NO [QLABEL:] already, regardless of [#P:] presence on same line.
                    // Safe: boundary resolver ignores QLABELs that don't match any master ID.
                    sanitizedChunkText = sanitizedChunkText.replace(
                        /^(\d{1,2}[).\]])\s(?!.*\[QLABEL:)/gm,
                        (match, label) => {
                            if (match.includes('[QLABEL:')) return match;
                            return `${label} [QLABEL:${label}] `;
                        }
                    );

                    // Pass 1a-glued: Main number + glued sub-part letter, e.g. "26.a)", "ANS.7.a)", "8.a)"
sanitizedChunkText = sanitizedChunkText.replace(
    /^((?:A[nu][st]s?\.?\s*)?)(\d{1,2})\.\s?([a-eA-E])\)\s*(?!.*\[QLABEL:)/gm,
    (match, prefix, num, letter) => {
        if (match.includes('[QLABEL:')) return match;
        const cleanPrefix = prefix.trim();
        const label = cleanPrefix ? `${cleanPrefix} ${num}.${letter})` : `${num}.${letter})`;
        return `${match.trimEnd()} [QLABEL:${label}] `;
    }
);

// Pass 1b-subpart: Standalone sub-part label on its own line (e.g. "b) def leftshift...")
                    // When student writes "10)a)" on one page and "b) def..." on the next,
                    // OCR sees just "b)" at line start with no parent number.
                    // We look backwards in the FULL transcript so far for the last numeric label,
                    // then inject a composite QLABEL like [QLABEL:10(b)].
                    {
                        const subpartRe = /^([a-d])\)\s(?!.*\[QLABEL:)/gm;
                        let sm;
                        const chunkSoFar = pageResults.map(p => p.rawText || p.text).join('\n') + '\n' + sanitizedChunkText;
                        // Find last parent number in transcript so far
                        const parentMatch = chunkSoFar.match(/\[QLABEL:[^\]]*?(\d{1,2})[^\]]*?\]/g);
                        const lastParentNum = parentMatch
                            ? (parentMatch[parentMatch.length - 1].match(/(\d{1,2})/) || [])[1]
                            : null;
                        if (lastParentNum) {
                            sanitizedChunkText = sanitizedChunkText.replace(subpartRe, (match, letter) => {
                                if (match.includes('[QLABEL:')) return match;
                                return `${letter}) [QLABEL:${lastParentNum}(${letter})] `;
                            });
                        }
                    }

                    // Pass 1c: Mid-line question label injection after [#P:] tag.
                    // Fires when OCR returns the entire page as one long line with inline [#P:] tags.
                    // e.g. "...cosφ [#P:6,450,790] 32) A) a) I = I₁+I₂ [#P:6,520,790]"
                    // Pass 1 uses ^ anchor so it only catches labels at LINE START — misses mid-line.
                    // Safe: only fires after a [#P:] tag, so random numbers in equations won't match.
                    sanitizedChunkText = sanitizedChunkText.replace(
                        /(\[#P:\d+,\d+,\d+\])\s+(\d{1,2}[).])\s(?![^\[]*\[QLABEL:)/g,
                        (match, ptag, label) => {
                            if (match.includes('[QLABEL:')) return match;
                            return `${ptag} ${label} [QLABEL:${label}] `;
                        }
                    );

                    // Pass 2: MCQ "N letter." format  e.g. "1 c. Both age... [#P:...]"
                    sanitizedChunkText = sanitizedChunkText.replace(
                        /^(\d{1,2})( [a-eA-E][.)])/gm,
                        (match, num, rest) => {
                            if (match.includes('[QLABEL:')) return match;
                            return `${num} [QLABEL:${num})]${rest}`;
                        }
                    );
                    // Pass 3: MCQ bare "N letter [#P:]" format  e.g. "1 B [#P:1,180,930]"
                    // OCR writes: digit space letter space [#P:...] — no dot or paren after letter.
                    // This is the most common MCQ format when OCR doesn't add punctuation.
                    sanitizedChunkText = sanitizedChunkText.replace(
                        /^(\d{1,2}) ([A-Ea-e]) (\[#P:)/gm,
                        (match, num, letter, tag) => {
                            if (match.includes('[QLABEL:')) return match;
                            return `${num} [QLABEL:${num})] ${letter} ${tag}`;
                        }
                    );
                    // Pass 4: Mid-line MCQ pattern — same as PDF path above.
                    // Handles OCR merging page header with first MCQ answer line.
                    if (!/^\d/.test(sanitizedChunkText.trim()) && sanitizedChunkText.includes('[#P:')) {
                        const segParts = sanitizedChunkText.split(/(\[#P:\d+,\d+,\d+\])/);
                        let rebuilt = '';
                        for (let si = 0; si < segParts.length; si++) {
                            const seg = segParts[si];
                            if (/^\[#P:\d+,\d+,\d+\]$/.test(seg)) { rebuilt += seg; continue; }
                            const nextSeg = segParts[si + 1];
                            if (!nextSeg || !/^\[#P:/.test(nextSeg)) { rebuilt += seg; continue; }
                            const mcqRe = /(?<![a-zA-Z])(\d{1,2}) ([a-eA-E][.)])\s/g;
                            let lastM = null, mm;
                            while ((mm = mcqRe.exec(seg)) !== null) lastM = mm;
                            if (lastM && !seg.includes('[QLABEL:')) {
                                const insertAt = lastM.index + lastM[1].length;
                                rebuilt += seg.slice(0, insertAt) + ` [QLABEL:${lastM[1]})]` + seg.slice(insertAt);
                            } else {
                                rebuilt += seg;
                            }
                        }
  sanitizedChunkText = rebuilt;
                    }
                    // Pass 5: "Ans. N (roman)" and "Ans N (roman)" format injection.
                    // Handles: "Ans. 8 (i)", "Ans 8 (ii)", "8 (i)", "8 (ii)" etc.
                    // These lines start with "Ans" or a digit, contain a parenthesized
                    // roman numeral sub-part, and have a [#P:] tag somewhere on the line.
                    // If no [QLABEL:] is already present, inject one immediately after the full label.
                    sanitizedChunkText = sanitizedChunkText.replace(
                        /^((?:A[nu][st]s?\.?\s*)?\d{1,2}\s*[\(\[](i{1,4}|iv|vi{0,3}|ix|xi{0,3}|ii)[\)\]])(.*)(\[#P:)/gm,
                        (match, label, roman, middle, tag) => {
                            if (match.includes('[QLABEL:')) return match;
                            const cleanLabel = label.trimEnd();
                            return `${cleanLabel} [QLABEL:${cleanLabel}]${middle}${tag}`;
                        }
                    );
// Pass 6 (image path): Mid-line MCQ QLABEL injection.
                    // Same as PDF path Pass 6 — handles continuous-line MCQ paragraphs.
                    sanitizedChunkText = sanitizedChunkText.replace(
                        /(\[#P:\d+,\d+,\d+\])(\s*)(\d{1,2}\.)(\s*)(\([a-eA-E]\)|[a-eA-E][.)])/g,
                        (match, ptag, space, num, sp2, letter) => {
                            if (match.includes('[QLABEL:')) return match;
                            return `${ptag}${space}${num} [QLABEL:${num}] ${sp2}${letter}`;
                        }
                    );

                    // Pass 6b (image path): Mid-line roman-numeral sub-part QLABEL injection.
                    // Same logic as PDF path Pass 6b above.
                    sanitizedChunkText = sanitizedChunkText.replace(
                        /(\[#P:\d+,\d+,\d+\])\s+(i{1,4}|iv|vi{0,3}|ix)\s*\)/g,
                        (match, ptag, roman) => {
                            if (match.includes('[QLABEL:')) return match;
                            const beforeMatch = sanitizedChunkText.substring(0, sanitizedChunkText.indexOf(match));
                            const parentMatch = beforeMatch.match(/(?:Ans\.?\s*)(\d{1,2})(?:\s|\])/gi);
                            const lastParent = parentMatch
                                ? parentMatch[parentMatch.length - 1].match(/(\d{1,2})/)[1]
                                : null;
   const label = lastParent ? `Ans ${lastParent} (${roman})` : roman;
                            return `${ptag} ${roman}) [QLABEL:${label}]`;
                        }
                    );

                    // Pass 6d (image path): Line-start orphan roman sub-part with [#P:] on same line.
                    // Same logic as PDF path Pass 6d above.
                    sanitizedChunkText = (() => {
                        let lastParentNum = null;
                        return sanitizedChunkText.replace(
                            /^(\s*)(\[QLABEL:[^\]]*?(\d{1,2})[^\]]*?\])|^(\s*)([(](i{1,4}|iv|vi{0,3}|ix)[)])(\s)(?=.*\[#P:)/gm,
                            (match, i1, qlabel, qnum, i2, romanFull, roman, space) => {
                                if (qlabel) { lastParentNum = qnum; return match; }
                                if (!romanFull) return match;
                                if (match.includes('[QLABEL:')) return match;
                                if (!lastParentNum) return match;
                                return `${i2 || ''}${romanFull} [QLABEL:${lastParentNum} (${roman})]${space}`;
                            }
                        );
                    })();

                                        // Pass 7: MCQ dense format "N) letter" with NO dot after letter
                    // Example: "1) c work done... [#P:10,120,900]"
                    // Injects [QLABEL:N)] after the number label
                    sanitizedChunkText = sanitizedChunkText.replace(
                        /^(\d{1,2})\)\s+([a-d])\s+([^[\#]+?)(\[#P:\d+,\d+,\d+\])/gm,
                        (match, num, letter, content, tag) => {
                            if (match.includes('[QLABEL:')) return match;
                            return `${num}) [QLABEL:${num})] ${letter} ${content.trim()} ${tag}`;
                        }
                    );

                    // Pass 8: MCQ format "1) c" where answer letter is the ONLY thing before [#P:]
                    // Example: "1) c [#P:10,210,400]"
                    sanitizedChunkText = sanitizedChunkText.replace(
                        /^(\d{1,2})\)\s+([a-d])\s+(\[#P:\d+,\d+,\d+\])/gm,
                        (match, num, letter, tag) => {
                            if (match.includes('[QLABEL:')) return match;
                            return `${num}) [QLABEL:${num})] ${letter} ${tag}`;
                        }
                    );
// Pre-Pass: Normalize OCR-corrupted "Ans N" variants (image path).
                    {
                        const masterNumSet = new Set(
                            masterIds.map(id => String(id).match(/(\d+)/)?.[1]).filter(Boolean)
                        );
                        sanitizedChunkText = sanitizedChunkText.replace(
                            /\b(A[a-z]{1,5}\.?\s{0,2})(\d{1,2})\b/gi,
                            (match, prefix, num) => {
                                if (match.includes('[QLABEL:')) return match;
                                const p = prefix.replace(/[\s.]/g, '').toLowerCase();
                                if (!p.startsWith('an')) return match;
                                if (p.length > 7) return match;
                                if (/^(and|any|ant|anti|another|animal|analysis|angle|annual|answer)/.test(p)) return match;
                                if (!masterNumSet.has(num)) return match;
                                console.log(`[Pre-Pass IMG] Corrupted Ans normalized: "${match.trim()}" → "Ans ${num}"`);
                                return `Ans ${num}`;
                            }
                        );
                    }

                    // Pass 9: "Answer No N", "Ans No N", "Ans-N", "Answer N", "ANS N" heading lines.
                    // Students write standalone heading lines like:
                    //   "Answer no 2", "Ans-2", "Ans 2", "answer no2", "ANS NO 2", "Answer No. 3"
                    // These lines have NO [#P:] tag (the actual answer content is on subsequent lines).
                    // All existing passes require [#P:] or digit+paren/dot at line start — they all miss this.
                    // This pass: fires on any line that starts with an ans/answer keyword + optional
                    // separator + a 1-2 digit number, and has no [QLABEL:] already on the line.
                    // Extracts the number and injects [QLABEL:N] immediately after the matched header text.
                    // Safe: the boundary resolver ignores QLABELs that don't match any master question ID,
                    // so false positives (e.g. "Answer no 42" when only Q1-Q7 exist) are auto-neutralised.
sanitizedChunkText = sanitizedChunkText.replace(
// AFTER
/^((?:answer|ans(?:wer)?|anes?|ams|aus|aas|ans[a-z]{0,2}|sol(?:ution)?|ques(?:tion)?|q(?:u[a-z]{0,3}|no?\\.?)?|hy)[\s._:-]*(?:no|number|num|no\.)?[\s._:-]*)(\d{1,2})[xX✓√*]?\b/gim,
   (match, prefix, num) => {
        if (match.includes('[QLABEL:')) return match;
        const kw = prefix.trim().split(/[\s.\-_]/)[0] || 'Ans';
        return `${prefix}${num} [QLABEL:${kw} ${num}]`;
    }
);

                    // Pass 9b: Mid-line and any-position Ans/Answer QLABEL injection.
                    //
                    // WHY THIS IS NEEDED:
                    // Pass 9 uses the ^ anchor (line-start only). When OCR collapses all
                    // answers onto one continuous line — which happens frequently when the
                    // student writes short inline answers like "Ans 1 : (d) D  Ans 2 : (b)
                    // Cytokinins  Ans 3 : (a) ..." — Pass 9 only fires once at the very
                    // start of the string. Every subsequent "Ans N" label is missed, meaning
                    // zero [QLABEL] tags are emitted for questions 2 onwards. The librarian
                    // then finds no boundaries, pageMap stays empty for all questions,
                    // answerPageIndex = -1 for everything, and "No questions detected" shows
                    // on every page of the report.
                    //
                    // This pass fires on ANY occurrence of Ans/Answer (any casing, any
                    // punctuation) anywhere in the text — line-start, mid-line, or after a
                    // [#P:] tag. It captures the FULL label:
                    //   - Number (required):          "17", "26", "29"
                    //   - Capital section letter (opt): "A" or "B" in "26 A" or "29 A"
                    //   - Sub-part (opt):              "(a)", "(b)", "(i)", "(ii)"
                    //
                    // The full label is preserved verbatim so that "26 A (a)" and "26 B (a)"
                    // remain distinct — they match different master question IDs.
                    //
                    // Examples:
                    //   "Ans 17 : (c)"     → appends [QLABEL:17]           (no cap, no sub)
                    //   "Ans 26 A: (a)"    → appends [QLABEL:26 A (a)]
                    //   "Ans 26 A: (b)"    → appends [QLABEL:26 A (b)]
                    //   "Ans 26 B: (a)"    → appends [QLABEL:26 B (a)]
                    //   "Ans 29 A: (a)"    → appends [QLABEL:29 A (a)]
                    //   "ANS 11. A. (b)"   → appends [QLABEL:11 A (b)]
                    //   "Answer 16 B (a)"  → appends [QLABEL:16 B (a)]
                    //
                    // Safe: boundary resolver ignores labels with no matching master ID.
                    // Non-destructive: guard prevents double-injection on already-tagged text.
sanitizedChunkText = sanitizedChunkText.replace(
    /\b((?:Ans(?:wer)?[a-z]?|ANS|A\.)[-.]?\s*)(?:(\d{1,2})|[(\[{](\d{1,2})[)\]}]|([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]))\s*[).\]};:,\-*]?\s*([A-Z]\s*[.:;\-]\s*)?(\([a-dA-D]\)|\([ivxIVX]+\)|\([\u0900-\u097F]\)|[ivx]{1,4}[).]\s*)?/gi,
    (match, keyword, num1, num2, circled, capSection, subPart) => {
        if (match.includes('[QLABEL:')) return match;
        const CIRCLED = '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳';
        const num = num1 || num2 || (circled ? String(CIRCLED.indexOf(circled) + 1) : '');
        if (!num) return match;
        const cap = capSection ? capSection.trim().replace(/[.:\-\s]+$/, '') : '';
        const sub = subPart ? subPart.trim() : '';
const labelText = [num, cap, sub].filter(Boolean).join(' ');
        return `${match.trimEnd()} [QLABEL:Ans ${labelText}]`;
    }
);

// Pass 9c (HINDI LANGUAGE ONLY — image path): explicit Devanagari sub-part injection.
if (isHindiLanguage(jobData)) {
    sanitizedChunkText = sanitizedChunkText.replace(
        /\b((?:Ans(?:wer)?|ANS)\.?[-]?\s*)(\d{1,2})\s*[.\s]*\(([कखगघङड़])\)/gi,
        (match, keyword, num, hindiLetter) => {
            if (match.includes('[QLABEL:')) return match;
            const labelText = `${num} (${hindiLetter})`;
            return `${match.trimEnd()} [QLABEL:${labelText}]`;
        }
    );
}

// Pass 9d (HINDI LANGUAGE ONLY — image path): Bare "N. क)" format — no Ans keyword.
// Same logic as PDF path Pass 9d above.
if (isHindiLanguage(jobData)) {
    sanitizedChunkText = sanitizedChunkText.replace(
        /(?<![A-Za-z])(\d{1,2})\s*[.)]*\s*\(?([कखगघङचछजझटठडढणतथदधनपफबभमयरलवशषसहड़ढ़])\)?/g,
        (match, num, hindiLetter) => {
            if (match.includes('[QLABEL:')) return match;
            const labelText = `${num} (${hindiLetter})`;
            return `${match.trimEnd()} [QLABEL:${labelText}]`;
        }
    );
}

sanitizedChunkText = sanitizedChunkText.replace(
    /\b(Ques(?:tion)?[\s.\-]*)(\d{1,2})\b\s*[).]?\s*(?!\S*\[QLABEL:)/gm,
    (match, prefix, num) => {
        if (match.includes('[QLABEL:')) return match;
        return `${match.trimEnd()} [QLABEL:${num}]`;
    }
);

// Pass 9f (ALL PAPERS — image path): Hindi "प्रश्नोत्तर सं – N" format.
// Same logic as PDF path Pass 9f above.
sanitizedChunkText = sanitizedChunkText.replace(
    /(?:प्रश्नोत्तर|उत्तर|प्रश्न)\s*(?:सं|संख्या|नं|नंबर|सं\.?)?\s*[-–—]?\s*(\d{1,2})\b/g,
    (match, num) => {
        if (match.includes('[QLABEL:')) return match;
        return `${match} [QLABEL:${num}]`;
    }
);

// Pass 9g: "Ans. N. Ni)" duplicate-number sub-part recovery.
// After nested QLABEL collapse, OCR sometimes produces: Ans. 23. 23i) [QLABEL:Ans 23]
// The repeated number + sub-part letter (23i, 23ii) is an artifact of double OCR.
// This pass rebuilds the correct sub-part QLABEL.
sanitizedChunkText = sanitizedChunkText.replace(
    /(\[QLABEL:\s*(?:Ans\.?\s*)?(\d{1,2})\s*\])\s*\2\s*(i{1,4}|iv|vi{0,3}|ix)\s*\)/gi,
    (match, qlabel, num, roman) => {
        return `[QLABEL:Ans ${num}.(${roman})]`;
    }
);
// Pass 9h: Normalize "N.v" → "N.(v)" — dot-letter without parens.
// OCR sometimes emits [QLABEL:22.v] instead of [QLABEL:22.(v)].
sanitizedChunkText = sanitizedChunkText.replace(
    /\[QLABEL:(\d{1,2})\.(i{1,4}|iv|vi{0,3}|ix)\]/gi,
    (match, num, roman) => `[QLABEL:${num}.(${roman})]`
);

// Pass 9e: "Ans." alone on a line

// Pass 9e: "Ans." alone on a line (no number) followed by a number
                    // on the very next non-empty line.
                    // Student writes:  "Ans."   ← line 1, OCR missed the number
                    //                  "12"      ← line 2, number in margin
                    // Fix: stitch them and inject [QLABEL:N].
sanitizedChunkText = sanitizedChunkText.replace(
                        /^((?:ans|answer)\.?\s*)(\n[\s]*)(\d{1,2})\b(?![\d])/gim,
                        (match, ansPrefix, gap, num) => {
                            if (match.includes('[QLABEL:')) return match;
                            return `${ansPrefix.trim()} ${num} [QLABEL:${num}]\n`;
                        }
                    );
                ;
                    // ─────────────────────────────────────────────────────────────────────────

                    // Pass 10: Bare [QLABEL:Ans] or [QLABEL:Ans.] with NO number — OCR dropped digit.
sanitizedChunkText = (() => {
    const chunkSoFar = pageResults.map(p => p.rawText || p.text).join('\n') + '\n' + sanitizedChunkText;
    const numberedQls = [...chunkSoFar.matchAll(/\[QLABEL:\s*(?:Ans\.?\s*)(\d+)/gi)];
    const lastNum = numberedQls.length > 0 ? parseInt(numberedQls[numberedQls.length - 1][1], 10) : 0;
    let nextNum = lastNum;
    return sanitizedChunkText.replace(
        /\[QLABEL:\s*Ans\.?\s*\](?!\s*\d)/gi,
        () => { nextNum += 1; return `[QLABEL:Ans ${nextNum}]`; }
    );
})();

// Pass 10b: "Ans (d)" or "Ans. (b)" — OCR dropped question number before MCQ letter.
// Infer number by incrementing last seen numbered QLABEL in transcript so far.
sanitizedChunkText = (() => {
    const chunkSoFar = pageResults.map(p => p.rawText || p.text).join('\n') + '\n' + sanitizedChunkText;
    const numberedQls = [...chunkSoFar.matchAll(/\[QLABEL:\s*(?:Ans\.?\s*)(\d+)/gi)];
    const lastNum = numberedQls.length > 0 ? parseInt(numberedQls[numberedQls.length - 1][1], 10) : 0;
    let nextNum = lastNum;
    return sanitizedChunkText.replace(
        /((?:^|[\s\n])Ans\.?\s*)(\([a-eA-E]\)?|\b[a-eA-E]\b)(\s)/gm,
        (match, ansPrefix, letter, space) => {
            if (match.includes('[QLABEL:')) return match;
            nextNum += 1;
            return `${ansPrefix.trim()} ${nextNum} [QLABEL:Ans ${nextNum}] ${letter}${space}`;
        }
    );
})();

sanitizedChunkText = sanitizedChunkText.replace(
                        /\[DIAGRAM\]([\s\S]{5500,}?)\[\/DIAGRAM\]/g,
                        (match, content) => {
                            const truncated = content.substring(0, 5000).trim();
                            const lastDot = Math.max(truncated.lastIndexOf('. '), truncated.lastIndexOf('.\n'));
                            const cleanEnd = lastDot > 1000 ? truncated.substring(0, lastDot + 1) : truncated;
                            return `[DIAGRAM] ${cleanEnd} [PROOF SUMMARY: Diagram truncated due to length.] [/DIAGRAM]`;
                        }
                    );

                    const anchors = [];
                    const matches = [...sanitizedChunkText.matchAll(/\[#P\s*:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]/gi)];
                    matches.forEach(m => {
                        anchors.push({ pageIndex: absolutePageIndex, y: parseInt(m[2], 10), x: parseInt(m[3], 10) });
                    });

                    pageResults.push({ pageNum: targetPageNum, text: sanitizedChunkText, rawText: sanitizedChunkText, anchors });
                }

   batchSuccess = true;

                // Log usage OUTSIDE critical path — Firestore failure must NOT trigger OCR retry
                try {
                    const usage = result.response.usageMetadata;
db.collection('apiUsageLogs').add({
    timestamp: admin.firestore.FieldValue.serverTimestamp(),
    teacherUid: jobData.teacherUid,
    teacherName: jobData.teacherName || 'Unknown',
    schoolId: jobData.schoolId || 'N/A',
    schoolName: jobData.schoolName || 'Unknown School',
    appType: 'teacher',
    feature: 'Assessment Checker OCR (Backend)',
    modelCalled: 'gemini-2.5-flash',
    unitsConsumed: {
        imagesProcessed: currentBatch.length
    },
    tokenUsage: {
        promptTokens: usage.promptTokenCount,
        candidatesTokens: usage.candidatesTokenCount,
        thinkingTokens: usage.thoughtTokenCount || 0,
        cachedTokens: usage.cachedContentTokenCount || 0,
        totalTokens: usage.totalTokenCount
    },
// Cached tokens bill at 10% of the standard input rate (implicit caching discount).
totalCostInr: (((usage.promptTokenCount - (usage.cachedContentTokenCount || 0)) / 1000000) * 27.6) + (((usage.cachedContentTokenCount || 0) / 1000000) * 2.76) + ((usage.candidatesTokenCount / 1000000) * 230) + (((usage.thoughtTokenCount || 0) / 1000000) * 34.5)
                    }).catch(e => console.warn('[OCR] Usage log write failed (non-critical):', e.message));
                } catch(e) { /* non-critical */ }
} catch (err) {
    attempt++;
    pageResults.length = batchStartResultsLen; // wipe any partial pushes from this failed attempt — stops duplicate pages
    console.error(`[OCR] Batch FAILED (attempt ${attempt}/1). Pages ${i+1}-${endRange}. Error: ${err?.message || err}`);
    if (attempt >= 1) {
 for (let subIdx = 0; subIdx < currentBatch.length; subIdx++) {
                        const pageNum = i + subIdx + 1;
                        await db.collection('gradingQueue').doc(jobId).update({ statusDetails: `Recovery Mode: Reading Page ${pageNum}...` });

                        try {
                            const recoveryModel = vertex_ai.getGenerativeModel({
                                model: 'gemini-2.5-flash',
                                systemInstruction: { parts: [{ text: finalOcrInstruction }] }
                            });
const recResult = await recoveryModel.generateContent({
                               contents: [{ role: 'user', parts: [currentBatch[subIdx], { text: `Analyze this single image.\nCRITICAL: Output MUST start with: [PAGE ${pageNum}]\n\nRULE PRIORITY (highest to lowest):\n1. [TABLE] blocks for accounts (journal, ledger, BRS, capital accounts) — transcribe ALL rows, NO truncation, NO word limit.\n2. [DIAGRAM] blocks — HARD LIMIT 1000 words. Truncate at 1000 words, close [/DIAGRAM], move on.\n3. Right-margin / separate rough-work column — SKIP ENTIRELY per the LEGIBILITY-EXIT LAW. If any region is cramped or illegible, write [illegible] ONCE and advance.\n\nCRITICAL: Never write [PAGE N] inside a [DIAGRAM] block. Always close [/DIAGRAM] before writing the next marker. Output PLAIN TEXT starting with the [PAGE ${pageNum}] header — DO NOT wrap in JSON, code fences, or quotes.` }] }],
generationConfig: {
                                    candidateCount: 1,
                                    seed: 42,
                                    temperature: 0,
                                    topP: 0.5,
                                    maxOutputTokens: 24000,
                                    thinkingConfig: { thinkingBudget: 0 }
                                }
                            });
                            const recParts = recResult.response.candidates[0].content.parts;
                            const rawText = ((recParts.find(p => p.text && !p.thought) || recParts[recParts.length - 1]).text || "[RECOVERY FAILED]").trim();
                            const fixedRecoveryText = rawText.replace(/\[#P\s*:\s*\d+\s*,/gi, `[#P:${pageNum},`);
                            pageResults.push({ pageNum, text: fixedRecoveryText, rawText: fixedRecoveryText, anchors: [] });
                        } catch (pageErr) {
                            console.error(`[OCR] Recovery FAILED for page ${pageNum}: ${pageErr?.message || pageErr}`);
                            pageResults.push({ pageNum, text: "[RECOVERY FAILED — please review manually]", rawText: "", anchors: [] });
                        }
                    }
                    batchSuccess = true;
                }
}
        }
        // Cooldown between batches. One decision per batch.
        // Clean: 1000ms. Heavy: 2500ms. Failed: 4000ms.
        if (i + BATCH_SIZE < imageParts.length) {
            const cooldown = !batchSuccess ? 4000 : (batchWasHeavy ? 2500 : 1000);
            await sleep(cooldown);
        }
    }
    return pageResults;
}
// ─────────────────────────────────────────────────────────────────────────────
// GRADING PIPELINE — gradeQuestionBatch
// UNCHANGED from original.
// ─────────────────────────────────────────────────────────────────────────────

async function gradeQuestionBatch(ocrText, questionBatch, strictness, subject, jobId, allRules, diagramImageParts = [], jobData = null, allQuestions = null) {

    // Use the subject-aware addon router instead of the old binary isHindi/isLit check.
    // getSubjectAddon() returns ~150 tokens of focused subject context.
    // For Hindi, gate by LANGUAGE (not subject) — student may write Physics in Hindi.
const dynamicInstructions = isHindiLanguage(jobData)
        ? HINDI_GRADING_ADDON
        : getSubjectAddon(subject);

    const STEM_SUBJECTS_LOCAL = ['physics','chemistry','maths','mathematics','biology','accounts','accountancy','economics','statistics','applied mathematics','applied maths','computer science'];
    const isSTEMSubject = STEM_SUBJECTS_LOCAL.includes((subject || '').trim().toLowerCase());

    // Subject-gated content-type grading addons. OFF by default for safety.
const numericalTableGradingAddon = NUMERICAL_TABLE_GRADING_ADDON;
    const tableMixedAddon = (ENABLE_TABLE_MIXED_ADDON &&
        (isDataTableSubject(subject) || isAccountsSubject(subject)))
        ? TABLE_MIXED_GRADING_ADDON : '';

    const gradingRules = allRules.filter(r => r.targetStep === 'Grading' || !r.targetStep);
    const ragInstructions = gradingRules.map(r => `- RULE: ${r.correctionRule}`).join('\n');

    const questionNumbers = questionBatch.map(q => q.questionNumber).join(', ');
    await db.collection('gradingQueue').doc(jobId).set({ statusDetails: `Step 2/3: Grading questions ${questionNumbers}...` }, { merge: true });

    const fullSystemInstruction = `
    ${GRADING_SYSTEM_INSTRUCTION}

    === CONTEXT FOR THIS GRADING TASK ===
    SUBJECT: "${subject}"
    STRICTNESS LEVEL: "${strictness}"

    === SUBJECT-SPECIFIC NOTES ===
    ${dynamicInstructions}
    ${numericalTableGradingAddon}
    ${tableMixedAddon}

    === TEACHER CORRECTION RULES ===
    ${ragInstructions}

    OUTPUT FORMAT: Single valid JSON array. No text outside the array.
    `;

    const responseSchema = {
        type: "array",
        items: {
            type: "object",
           properties: {
    questionNumber: { type: "string" },
    marksAwarded: { type: "number" },

    requiresReview: { type: "boolean" },
    chapterTopic: { 
        type: "string",
description: "The GRANULAR sub-concept this question tests. Rule: if your answer is the chapter name from a textbook's table of contents, it is WRONG — go one level deeper to the specific theorem, method, property, or formula. ALWAYS fill this even for correct answers. FORBIDDEN: chapter-level words like the subject name or any top-level chapter title."    },

    questionType: {
        type: "string",
        enum: ["RECALL", "CONCEPTUAL", "NUMERICAL", "DERIVATION", "DIAGRAM", "APPLICATION", "ASSERTION_REASON", ""],
        description: "The cognitive type of this question. RECALL=pure memory/definition. CONCEPTUAL=explain why/how. NUMERICAL=apply formula and calculate. DERIVATION=step-by-step mathematical proof. DIAGRAM=draw and label. APPLICATION=apply concept to new scenario. ASSERTION_REASON=evaluate two statements. Empty string if cannot determine."
    },
    
    stepWiseEvaluation: {
        type: "array",
        items: {
            type: "object",
            properties: {
                marks: { type: "number" },
                comment: { type: "string" },
                pageIndex: { type: "integer" },
                stepPoint: {
                    type: "array",
                    items: { type: "number" }
                }
            },
            required: ["marks", "pageIndex", "stepPoint"]
        }
    }
},
required: ["questionNumber", "marksAwarded", "requiresReview",
           "chapterTopic", "questionType", "stepWiseEvaluation"]
        }
    };
const COMPLEX_TYPES = ['LA', 'CS', 'DBQ'];
const SA_TYPES = ['SA'];
const AR_TYPES = ['AR', 'Assertion-Reason'];

    function isProofLikeQuestion(q) {
        const text = `${q.text || ''} ${q.checkingInstructions || ''} ${q.imagePrompt || ''}`.toLowerCase();
        return !!(
            q.imagePrompt ||
            /\b(prove|verify|derive|derivation|proof|show that|huygens|kirchhoff|theorem|principle|law of|laws of|reflection|refraction)\b/.test(text)
        );
    }
function isComplexQuestion(q) {
    return (
        (COMPLEX_TYPES.includes(q.type) && q.marks >= 3) ||
        (q.marks >= 3 && (q.rubric?.step_marking || isProofLikeQuestion(q))) ||
        (q.marks >= 3 && isSTEMSubject)
    );
}

function isSAQuestion(q) {
    return SA_TYPES.includes(q.type);
}

function isARQuestion(q) {
    return AR_TYPES.includes(q.type);
}

// No separate "lite" model/path — every tier below runs on gemini-2.5-flash via callGrader().
// AR (Assertion-Reason) always gets a thinking budget: it requires evaluating two linked
// logical statements, not just matching a letter, so it is pulled out of the plain-MCQ bucket.
const complexQuestions = questionBatch.filter(q => isComplexQuestion(q));
const arQuestions      = questionBatch.filter(q => !isComplexQuestion(q) && isARQuestion(q));
const saQuestions      = questionBatch.filter(q => !isComplexQuestion(q) && !isARQuestion(q) && isSAQuestion(q));
const simpleQuestions  = questionBatch.filter(q => !isComplexQuestion(q) && !isARQuestion(q) && !isSAQuestion(q));

console.log(`[Grading] 4-tier: ${complexQuestions.length} complex, ${arQuestions.length} AR, ${saQuestions.length} SA, ${simpleQuestions.length} simple (MCQ/TF/FillBlank/VSA)`);

// Look up OR partner in the full paper. Non-OR questions → null.
    const _norm = s => (s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
    const _lookupPool = Array.isArray(allQuestions) && allQuestions.length ? allQuestions : questionBatch;
    function getOrPartnerContext(q) {
        const ci = q.checkingInstructions || "";
        if (!/alternative question \(or\)/i.test(ci)) return null;
        const m = ci.match(/with\s+question\s+([0-9]+(?:[.\-][a-zA-Z0-9]+)?)/i);
        if (!m) return null;
        const target = _norm(m[1]);
        const partner = _lookupPool.find(p => _norm(p.questionNumber) === target);
        if (!partner) return null;
        return {
            questionNumber: partner.questionNumber,
            text:           partner.text || "",
            answer:         partner.answer || "",
            topicAnchors:   Array.isArray(partner.topicAnchors) ? partner.topicAnchors : []
        };
    }

    // Build AI batch — demote model answer to reference for rubric questions
    function buildAIBatch(batch) {
        return batch.map(q => {
            const hasRubric = q.rubric && q.rubric.step_marking &&
                              q.rubric.step_marking !== "Grade based on model answer.";
            return {
                questionNumber:       q._uid,
                text:                 q.text,
                answer: hasRubric
? `[REFERENCE ONLY — student may use any valid method. Do NOT penalize for differing from this if their method is correct. MATH = REASON LAW: If a rubric step requires a "reason" or "explanation", a correct mathematical derivation proving the same conclusion is fully sufficient. Do NOT demand verbal prose if math proves the point.]\n${q.answer}`
                    : q.answer,
                marks:                q.marks,
                type:                 q.type,
                rubric:               q.rubric || null,
checkingInstructions: q.checkingInstructions || "",
                imagePrompt:          q.imagePrompt || null,
                options:              q.options || null,
                ragContext:           q.ragContext || "",
                topicAnchors:         Array.isArray(q.topicAnchors) ? q.topicAnchors : [],
                orPartner:            getOrPartnerContext(q)
            };
        });
    }


    // Single grader call — used for every tier (complex / AR / SA / simple). One code path,
    // one model (gemini-2.5-flash) for all of them, so behavior never silently diverges
    // between question types the way the old separate "lite" path did.
    async function callGrader(batch, useThinking, tierLabel) {
        if (batch.length === 0) return [];

        const aiBatch = buildAIBatch(batch);
        const textPart = {
            text: `STUDENT_TRANSCRIPT:\n${ocrText}\n\nQUESTIONS_TO_GRADE (Pre-Mapped):\n${JSON.stringify(aiBatch)}`
        };

        // Only pass diagram images to complex batch (MCQs never have diagrams)
// Pass images for: (a) thinking-enabled batches, OR (b) any STEM batch with 3+ mark questions
        const batchNeedsImages = diagramImageParts.length > 0 &&
            (useThinking || (isSTEMSubject && batch.some(q => q.marks >= 3)));
        const parts = batchNeedsImages
            ? [
                { text: `DIAGRAM REFERENCE IMAGES: ${diagramImageParts.length} page image(s). Use to visually verify student diagrams, force configurations, and charge setups. A diagram visible in the image but missing or incomplete in OCR is still a valid student attempt.` },
                ...diagramImageParts,
                textPart
              ]
            : [textPart];

 const hasProofQuestion = useThinking && batch.some(q =>
            q.imagePrompt || 
            /\b(prove|verify|derive|derivation|proof|show that|demonstrate|establish|hence|therefore|conclude|similar|congruent|triangle|theorem|principle|law of|reflection|refraction|journal entry|ledger|balance|equation|formula)\b/i.test(q.text || '')
        );
        
let thinkingBudget = 0;
if (useThinking) {
    if (hasProofQuestion) {
        thinkingBudget = 2048;
    } else if (batch.some(q => q.marks >= 5)) {
        thinkingBudget = 1024;
    } else if (batch.some(q => q.marks >= 3)) {
        thinkingBudget = 768;
    } else {
        thinkingBudget = 512;   // SA questions (3-4 marks or lower SA)
    }
}
        
        console.log(`[Grading] thinkingBudget=${thinkingBudget} (proof=${hasProofQuestion})`);

        const request = {
            model: 'gemini-2.5-flash',
            contents: [{ role: 'user', parts }],
            systemInstruction: { parts: [{ text: fullSystemInstruction }] },
            generationConfig: {
                responseMimeType: "application/json",
                temperature: 0.0,
                responseSchema: responseSchema,
                thinkingConfig: { thinkingBudget }
            }
        };

        const gradingModel = vertex_ai.getGenerativeModel({ model: 'gemini-2.5-flash' });
        const result = await callGeminiWithRetry(gradingModel, request);

        // Log usage for every tier — previously only the old "lite" path logged cost, so
        // complex/SA/simple grading (the largest share, given thinking budgets up to 2048)
        // was invisible in apiUsageLogs. Non-blocking — a log failure must never fail grading.
        try {
            const usage = result.response.usageMetadata;
            db.collection('apiUsageLogs').add({
                timestamp: admin.firestore.FieldValue.serverTimestamp(),
                teacherUid: jobData.teacherUid,
                feature: `Grading (${tierLabel || 'Unknown'})`,
                modelCalled: 'gemini-2.5-flash',
                tokenUsage: {
                    promptTokens: usage.promptTokenCount,
                    candidatesTokens: usage.candidatesTokenCount,
                    thinkingTokens: usage.thoughtTokenCount || 0,
                    cachedTokens: usage.cachedContentTokenCount || 0,
                    totalTokens: usage.totalTokenCount
                },
                // Cached tokens bill at 10% of the standard input rate (implicit caching discount).
                totalCostInr: (((usage.promptTokenCount - (usage.cachedContentTokenCount || 0)) / 1000000) * 27.6) + (((usage.cachedContentTokenCount || 0) / 1000000) * 2.76) + ((usage.candidatesTokenCount / 1000000) * 230) + (((usage.thoughtTokenCount || 0) / 1000000) * 34.5)
            }).catch(() => {});
        } catch (e) { /* non-critical */ }

     const parts_arr = result.response.candidates[0].content.parts;
const rawJson = (parts_arr.find(p => p.text && !p.thought) || parts_arr[parts_arr.length - 1]).text;

        const parsedResult = extractJsonFromString(rawJson);

if (!Array.isArray(parsedResult)) {
    throw new Error(`Grading failed for ${tierLabel || (useThinking ? 'complex' : 'simple')} batch: Expected JSON array.`);
}

// After parsedResult is received, before returning, add this:

// Replace "Good work." with actual step pointers for full marks
parsedResult.forEach(result => {
    const originalQ = questionBatch.find(q => q._uid === result.questionNumber);
    if (!originalQ) return;
    
    const isFullMarks = result.marksAwarded >= originalQ.marks && originalQ.marks > 0;
    const hasStepComments = result.stepWiseEvaluation && result.stepWiseEvaluation.length > 0;
    
const isGenericGoodWork = /^good work\.?$/i.test(result.finalFeedback || '');
if (hasStepComments && isGenericGoodWork) {  // removed isFullMarks gate
        // Build feedback from step comments
        const stepPointers = result.stepWiseEvaluation
.filter(step => step.comment && step.comment.trim())
.map(step => step.marks > 0
    ? `✓ ${step.comment}`
    : `✗ ${step.comment}`)
            .join('\n');
        
        if (stepPointers && stepPointers.trim()) {
            result.finalFeedback = stepPointers;
        }
    }
});


return parsedResult;
    }
const useThinkingFull = jobId && !jobId.startsWith('light_') && isSTEMSubject;

const [complexResults, arResults, saResults, simpleResults] = await Promise.all([
    callGrader(complexQuestions, useThinkingFull, 'Complex'),   // LA, CS, DBQ 3+ marks — full thinking
    callGrader(arQuestions, true, 'AR-Reasoning'),              // Assertion-Reason — always thinks (logical evaluation, not letter lookup)
    callGrader(saQuestions, useThinkingFull, 'SA'),             // SA — thinking enabled
    callGrader(simpleQuestions, false, 'Simple-MCQ')            // MCQ, True/False, Fill blanks, VSA — no thinking, pure recall
]);

const parsedResult = [...complexResults, ...arResults, ...saResults, ...simpleResults];

    if (!Array.isArray(parsedResult)) {
        throw new Error(`Grading phase failed: Expected a JSON array.`);
    }

// Match AI results back by _uid — collision-safe even with duplicate question numbers
    return questionBatch.map(reqQ => {
        // Primary match: exact _uid (sent as questionNumber to AI)
        let aiMatch = parsedResult.find(r => r.questionNumber === reqQ._uid);

        // Secondary match: if AI truncated/modified the uid, try matching by
        // normalized real questionNumber as fallback. Prevents total loss when
        // large DIAGRAM blocks cause response truncation and uid is missing.
        if (!aiMatch) {
            const normReal = (reqQ.questionNumber || '').replace(/[^a-z0-9]/gi, '').toLowerCase();
            aiMatch = parsedResult.find(r => {
                const normR = (r.questionNumber || '').replace(/[^a-z0-9]/gi, '').toLowerCase();
                return normR === normReal;
            });
            if (aiMatch) {
                console.warn(`[GradeMatch] Q${reqQ.questionNumber}: uid match failed, recovered via questionNumber match`);
            }
        }

        if (aiMatch) {
            const marksAwarded = Math.min(parseFloat(aiMatch.marksAwarded) || 0, reqQ.marks);
            return { ...reqQ, ...aiMatch, questionNumber: reqQ.questionNumber, marksAwarded, maxMarksForQuestion: reqQ.marks };
        }
        return {
            ...reqQ,
            marksAwarded: 0,
            maxMarksForQuestion: reqQ.marks,
            finalFeedback: "answer not found in OCR.",
            requiresReview: true,
            stepWiseEvaluation: []
        };
    });
}



async function librarianTagMapper(fullTranscript, questions, subject) {
    const model = vertex_ai.getGenerativeModel({
        model: "gemini-2.5-flash",

generationConfig: {
    temperature: 0,
    responseMimeType: "application/json",
    thinkingConfig: { thinkingBudget: 600 }
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
                : [(q.text || "").substring(0, 150)]  // 50 chars is too short for reliable semantic match

        };
    });

    const normalizedSubject = subject?.toLowerCase().trim() || "";
    const isMaths = MATH_SUBJECTS.includes(normalizedSubject);

    let prompt;

    if (isMaths) {
        prompt = `
You are a Numeric Structural Document Librarian.

MASTER QUESTION IDS:
${masterIds}

TOPIC ANCHORS (Secondary Signals Only):
${JSON.stringify(topicAnchors, null, 2)}

IMPORTANT RULES:

1. [QLABEL:text] markers are STRONG BOUNDARY SIGNALS
2. For PROOF QUESTIONS (across ALL subjects): 
   - The student may write the proof ACROSS MULTIPLE lines and [QLABEL] boundaries
   - Do NOT break the proof into separate questions
   - Keep ALL tags that belong to the same proof under the parent question ID
   - If you see [QLABEL:?] inside a proof block, IGNORE it as a boundary

3. DIAGRAM BRIDGE RULE:
   - A [QLABEL:X] remains the ACTIVE OWNER across any following [DIAGRAM] block
   - ALL [#P] tags after [/DIAGRAM] — until the next [QLABEL] — belong to X

4. CROSS-PAGE DIAGRAM CONTINUATION LAW (CRITICAL):
   - When a page starts with [DIAGRAM] and has NO [QLABEL] before it on that page,
     that diagram is a CONTINUATION of the last active question from the previous page.
   - ALL [#P] tags between [/DIAGRAM] and the next [QLABEL] on that page belong to
     that same last question — NOT to the question whose [QLABEL] follows.
   - EXAMPLE: Page 8 ends with [QLABEL:Ans 12]. Page 9 starts with [DIAGRAM]...[/DIAGRAM]
     then [#P:9,498,470] then [QLABEL:Ans. 13].
     → [#P:9,498,470] belongs to Q12. Q13 starts at [QLABEL:Ans. 13].

TRANSCRIPT:
${fullTranscript}

TASK:
Map every coordinate tag [#P:p,y,x] to the correct Question ID.

OUTPUT FORMAT:
{
  "mappings": [
    { "id": "1", "tags": ["[#P:1,120,930]", "[#P:1,150,930]"] }
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
Analyze the transcript and map every coordinate tag [#P:p,y,x] to the correct Question ID.

RULES:
1. Every line in the transcript ends with a unique tag like [#P:1,250,930]. 
2. Group these tags by the Question ID they belong to.
3. Use semantic anchors (keywords) to identify question boundaries even if handwritten labels are messy.
4. If a block of text spans multiple tags, include EVERY tag in that sequence.
5. RETURN ONLY JSON.

# THE HIERARCHY OF TRUTH (MAPPING PROTOCOL)
1. CASE A: THE SEMANTIC OVERRIDE (Use Only When Label Missing Or Wrong):
   If a block of text contains 2+ 'topicAnchors' belonging to a specific ID, map it to that ID —
   BUT ONLY if the student's handwritten label is missing, illegible, or clearly contradicts the anchors.
   If the student's label ALREADY matches a valid Master ID correctly, DO NOT override it with anchors,
   even if anchors seem to point elsewhere. Explicit correct label always wins over semantic guess.

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

IMPORTANT: The transcript contains [QLABEL:text] markers. These are STRONG BOUNDARY SIGNALS — treat every [QLABEL:text] as a high-confidence indicator that a new question block starts at that point. A [QLABEL] on a line means ALL [#P] tags on that line and subsequent lines (until the next [QLABEL]) belong to that question. Use [QLABEL] as your PRIMARY boundary signal, above topic anchors. Only use topic anchors when [QLABEL] is absent.


# ANONYMOUS ANS BLOCK LAW:
If you see "Ans." or "Ans" followed by content but NO number visible,
and the content STRONGLY and CLEARLY matches one specific question's topicAnchors
(2+ distinct anchors, no ambiguity with any other question), assign it to that ID.
If the match is weak, partial, or could plausibly fit more than one question,
DO NOT force an assignment. Leave it unassigned. Downstream fallback will handle it.
Guessing wrong is worse than leaving unresolved.

# DIAGRAM BRIDGE LAW (CRITICAL):
Many questions begin with [DIAGRAM]...[/DIAGRAM] text containing NO [#P] tags inside it.
The first [#P] for these questions appears only AFTER [/DIAGRAM].
RULE: A [QLABEL:X] remains the ACTIVE OWNER across any following [DIAGRAM]...[/DIAGRAM] block.
ALL [#P] tags after [/DIAGRAM] — until the next [QLABEL] — belong to X, not the prior question.
The coordinate-free gap caused by the diagram text is NOT a boundary signal.

# CROSS-PAGE DIAGRAM CONTINUATION LAW (CRITICAL):
When a NEW PAGE starts with [DIAGRAM]...[/DIAGRAM] and has NO [QLABEL] before it on that page,
that diagram is a CONTINUATION of the last active question from the previous page.
ALL [#P] tags between [/DIAGRAM] and the next [QLABEL] on that page belong to that
same last question — NOT to the question whose [QLABEL] follows after the diagram.
EXAMPLE:
  Page 8: ... [QLABEL:Ans 12] ... [#P:8,750,500]
  Page 9: [DIAGRAM] circular flow [/DIAGRAM] Its significance [#P:9,498,470]
           [QLABEL:Ans. 13] Problems of Brain Drain [#P:9,720,600]
  CORRECT: Q12 gets [#P:9,498,470]. Q13 starts at [#P:9,720,600].
TRANSCRIPT:
${fullTranscript}

OUTPUT FORMAT:
{
  "mappings": [
    { "id": "1a", "tags": ["[#P:1,120,930]", "[#P:1,150,930]"], "confidence": "high", "matchedVia": "label" },
    { "id": "1b", "tags": ["[#P:1,180,930]", "[#P:2,100,930]"], "confidence": "medium", "matchedVia": "anchor" }
  ]
}
"confidence" must be "high", "medium", or "low".
"matchedVia" must be "label" (explicit handwritten number matched) or "anchor" (semantic guess only).
`;
    }

    const result = await callGeminiWithRetry(model, {
        contents: [{ role: 'user', parts: [{ text: prompt }] }]
    });

    const rawText = result.response.candidates[0].content.parts[0].text;
    const parsed = extractJsonFromString(rawText);

    // Safety: if LLM returned null/invalid JSON, return empty mappings rather than null.
    // Returning null would make every question get "No specific text assigned".
    if (!parsed || !Array.isArray(parsed.mappings)) {
        console.warn('[LibrarianTagMapper] JSON parse failed or missing mappings array. Returning empty mappings.');
        return { mappings: [] };
    }
    return parsed;
}


// ─────────────────────────────────────────────────────────────────────────────
// sanitizeQLabelTranscript — runs ONCE on fullTranscript before any mapper.
//
// Fixes two OCR hedging problems:
//   1. DEDUP/DENEST: OCR emits multiple QLABELs for the same real label, e.g.
//      [QLABEL:1][QLABEL:Ans 1 [QLABEL:1]] → collapse to [QLABEL:Ans 1]
//      Rule: when a cluster of QLABELs appears within 60 chars of each other,
//      keep the Ans-prefixed form if present, else keep the longest, drop rest.
//   2. DEMOTE IN-ANSWER SUB-POINTS: When paper is ANS-anchored (≥2 "Ans N"
//      labels), bare sub-point QLABELs like [QLABEL:1)] [QLABEL:3)] emitted
//      BETWEEN two Ans labels are student enumeration, NOT boundaries.
//      Strip only the [QLABEL:...] wrapper — keep the visible text as content.
// ─────────────────────────────────────────────────────────────────────────────
function sanitizeQLabelTranscript(transcript, masterIds = []) {
    // ── STEP A: detect ANS-anchored paper ─────────────────────────────────────
const ansLabelRe = /\[QLABEL:\s*(Ans|And|Answer|Ques|Question|Sol|Q)[\s.\-\d]/gi;

    const ansMatches = transcript.match(ansLabelRe) || [];
const bareDigitCountSan = (transcript.match(/\[QLABEL:\s*\d{1,2}[).\s]?\]/g) || []).length;
    const isAnsAnchored = ansMatches.length >= 3 && ansMatches.length > bareDigitCountSan;

    console.log(`[Sanitizer] ANS-anchored: ${isAnsAnchored} (${ansMatches.length} Ans labels found)`);

    // ── STEP B: collapse nested malformed QLABELs ─────────────────────────────
    // Pattern: [QLABEL:Ans 11 [QLABEL:11]] — the inner [QLABEL:11] has no closing
    // bracket because the outer one consumed it. Result after [^\]]+ match:
    // "[QLABEL:Ans 11 [QLABEL:11]" — strip the inner nested part from the text.
    // Simply remove any [QLABEL: that appears INSIDE another QLABEL's text value.
let out = transcript.replace(/\[QLABEL:([^\]]*)\[QLABEL:[^\]]*\]\s*([^\]]*)\]/g, (match, prefix, suffix) => {
    const cleanPrefix = prefix.trim().replace(/\s+$/, '');
    const cleanSuffix = suffix.trim();
    // If suffix contains a sub-part like "i)", "ii)", "(i)", preserve it
    const subPartMatch = cleanSuffix.match(/^(i{1,4}|iv|vi{0,3}|ix|[a-e])\)?/i);
    if (subPartMatch && cleanPrefix) {
        return `[QLABEL:${cleanPrefix} (${subPartMatch[1].toLowerCase()})]`;
    }
    return `[QLABEL:${cleanPrefix}]`;
});

// Add before return out: normalize dot-roman without parens
out = out.replace(
    /\[QLABEL:(\d{1,2})\.(i{1,4}|iv|vi{0,3}|ix)\]/gi,
    (match, num, roman) => `[QLABEL:${num}.(${roman})]`
);


// ── STEP B2: hard-strip QLABELs inside marks-recording grids ──────────────
    function isMarksGrid(tableBody, mIds) {
        const cells = tableBody.split('|').map(c => c.trim());
        const numericLabels = cells.filter(c => /^Q?\.?\s*\d{1,2}$/i.test(c));
        const blanks = cells.filter(c => c === '-' || c === '' || /^\[QLABEL:[^\]]*\]$/.test(c));
        const matchesMasters = numericLabels.filter(c => {
            const n = c.replace(/[^\d]/g, '');
            return mIds.some(id => String(id).replace(/[^\d]/g, '') === n);
        });
        return blanks.length / Math.max(cells.length, 1) > 0.5
            && matchesMasters.length >= Math.min(5, mIds.length * 0.3);
    }

    out = out.replace(
        /(\[TABLE:\s*([^\]]*)\])([\s\S]*?)(\[\/TABLE\])/gi,
        (full, openTag, title, body, closeTag) => {
            const titleHit = /marks|score|examiner|evaluation|grading/i.test(title);
            if (titleHit || isMarksGrid(body, masterIds)) {
                return openTag + body.replace(/\[QLABEL:[^\]]*\]/g, '') + closeTag;
            }
            return full;
        }
    );



 

    // ── STEP C: deduplicate consecutive sibling QLABELs for the same position ─
    // Pattern: [QLABEL:1][QLABEL:1] or [QLABEL:1][QLABEL:Ans 1]
    // Within any run of consecutive [QLABEL:...] tokens (≤60 chars apart),
    // keep ONE: prefer the Ans-prefixed form, else the longest.
out = out.replace(/(\[QLABEL:[^\]]+\])([\s.,:;)\]]{0,8}\[QLABEL:[^\]]+\])+/g, (fullMatch) => {
        const tokens = [...fullMatch.matchAll(/\[QLABEL:([^\]]+)\]/g)].map(m => ({
            full: m[0],
            inner: m[1].trim()
        }));
        // Prefer Ans-prefixed
        const ansPref = tokens.find(t => /^(ans|answer)[\s.\d]/i.test(t.inner));
        if (ansPref) return ansPref.full;
        // Else longest
        tokens.sort((a, b) => b.inner.length - a.inner.length);
        return tokens[0].full;
    });

    // ── STEP D: demote in-answer sub-point QLABELs (ANS-anchored only) ────────
    if (!isAnsAnchored) return out;

    // Split transcript by Ans-boundary QLABELs. Between each pair of Ans labels,
    // remove any QLABEL whose inner text is a bare sub-point (not Ans-prefixed,
    // not matching a master question format — i.e. it's just 1), 2), i), a), etc.)
    //
    // Sub-point pattern: optional open-paren, then digits OR roman OR single alpha,
    // then optional close-paren/dot. Examples: 1) 2) i) ii) a) b) (i) (a) 1. a.
const subPointRe = /^[(]?(\d{1,2}|i{1,4}|iv|vi{0,3}|ix|[a-d])[).)]?$/i;

    // Pre-build normalized master set for fast lookup in Step D
    const normMasterSet = new Set(masterIds.map(id => normalizeForComparison(id)));

    out = out.replace(/\[QLABEL:([^\]]+)\]/g, (match, inner) => {
        const trimmed = inner.trim();
        // Keep ALL Ans-prefixed labels untouched
        if (/^(ans|answer)[\s.\d]/i.test(trimmed)) return match;
        // Keep labels that look like real question numbers (multi-digit, or with dots)
        // i.e. anything NOT a bare sub-point
        if (!subPointRe.test(trimmed)) return match;
        // CRITICAL: even if it looks like a sub-point, keep it if it matches a master ID.
        // e.g. masters are Q1(i), Q1(ii) → "i)" and "ii)" ARE real boundaries.
const normTrimmed = normalizeForComparison(trimmed);
        if (normMasterSet.has(normTrimmed)) {
            // Only keep as boundary if this master ID is itself a subpart
            // (e.g. masters are Q1(i), Q1(ii) → "i" matches master "i")
            // Do NOT keep if the master is a top-level question like Q1, Q2
            // and the subpoint is just the bare digit "1" or "2" — that's
            // student enumeration inside another answer, not a boundary.
            const isTopLevelDigit = /^\d{1,2}$/.test(normTrimmed);
            if (!isTopLevelDigit) {
                console.log(`[Sanitizer] Kept sub-point QLABEL "${trimmed}" — matches master subpart ID`);
                return match;
            }
            // Falls through → demote bare digit even if it matches a top-level master
        }
        // It's a bare sub-point with no master match → strip the QLABEL wrapper
        console.log(`[Sanitizer] Demoted in-answer sub-point QLABEL: "${trimmed}"`);
        return trimmed;
    });

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// buildAnswerSlices — CHARACTER-POSITION based slicer.
//
// WHY THIS REPLACES sliceByAtomicLines:
//   The old line-based slicer broke when the OCR writes multiple answers on a
//   SINGLE line (inline format). Example from a real student paper:
//     "Ans.2.a(i) [QLABEL:Ans.2.a(i)] text [#P:5,170] ... Ans.2.a(ii) [QLABEL:...] text [#P:5,440]"
//   All tags were on LINE 2, so backward/forward extension always returned ALL
//   of line 2 for every question — every "INPUT TO GRADER" was identical and
//   contained the full page instead of just that question's answer.
//
// HOW THIS WORKS:
//   1. Build a flat list of every [#P:] tag and its character position in the string.
//   2. For each question, find the [QLABEL:] that immediately precedes its first tag.
//      That QLABEL's character position is the START of the answer.
//   3. The END is either: the next [QLABEL:] or the next [#P:] tag after the
//      question's last tag — whichever comes first.
//   4. Slice the raw string by [startPos, endPos]. Works identically for both
//      inline (single-line) and multiline OCR output.
// ─────────────────────────────────────────────────────────────────────────────
function sliceByAtomicLines(fullTranscript, mappingJson) {
    if (!mappingJson || !mappingJson.mappings) return {};

    // Build an ordered list of every [#P:p,y,x] tag and its char position
    const allTagPositions = [];
    const tagRegex = /\[#P:(\d+),(\d+),(\d+)\]/g;
    let m;
    while ((m = tagRegex.exec(fullTranscript)) !== null) {
        allTagPositions.push({ tag: m[0], pos: m.index, end: m.index + m[0].length });
    }
    // Fast lookup: tag string → position info
    const tagPosMap = new Map();
    allTagPositions.forEach(t => tagPosMap.set(t.tag, t));

    // ── ANS-ANCHORED PAPER DETECTION ─────────────────────────────────────────
    // If the student consistently prefixes new answers with "Ans"/"Answer"
    // (e.g. "Ans 11", "Ans 12"), then bare sub-point QLABELs like "1)", "2)",
    // "i)" emitted by OCR INSIDE an answer are NOT question boundaries — they
    // are the student's own enumeration. In ANS-anchored mode the slicer must
    // ONLY cut at the next "Ans"-prefixed QLABEL, never on numeric-root diff.
    const ansQlabelRe = /\[QLABEL:\s*(ans|answer)[\s.\d]/i;
    let ansAnchoredCount = 0;
    {
        const allQl = fullTranscript.match(/\[QLABEL:[^\]]+\]/g) || [];
        allQl.forEach(ql => { if (ansQlabelRe.test(ql)) ansAnchoredCount++; });
    }
    // Treat as ANS-anchored when there are at least 2 distinct "Ans N" labels.
const _bareDigitCountSlice = (fullTranscript.match(/\[QLABEL:\s*\d{1,2}[).\s]?\]/g) || []).length;
const isAnsAnchoredPaper = ansAnchoredCount >= 3 && ansAnchoredCount > _bareDigitCountSlice;
    if (isAnsAnchoredPaper) {
        console.log(`[Librarian] ANS-anchored paper detected (${ansAnchoredCount} Ans labels) — slicer will cut ONLY at Ans-prefixed QLABELs.`);
    }

    const slices = {};

    mappingJson.mappings.forEach(mapping => {
        const qId = normalizeForComparison(mapping.id);
        const tags = mapping.tags;

        if (!tags || tags.length === 0) { slices[qId] = ''; return; }

        // Locate each approved tag in the transcript
        const positions = tags
            .map(t => tagPosMap.get(t))
            .filter(Boolean)
            .sort((a, b) => a.pos - b.pos);

        if (positions.length === 0) { slices[qId] = ''; return; }

        // START: find the [QLABEL:] that appears immediately before the first tag.
        // This is the student's "Ans X" label — the true start of their answer.
        const firstTagPos = positions[0].pos;
        const qlabelRegex = /\[QLABEL:([^\]]+)\]/g;
        let lastQlabelBefore = null;
        let qm;
        while ((qm = qlabelRegex.exec(fullTranscript)) !== null) {
            if (qm.index < firstTagPos) lastQlabelBefore = qm;
            else break;
        }
let startPos;
if (lastQlabelBefore) {
    const textBeforeQL = fullTranscript.substring(0, lastQlabelBefore.index);
    const prevNL = textBeforeQL.lastIndexOf('\n');
    startPos = prevNL >= 0 ? prevNL + 1 : 0;
} else {
    startPos = firstTagPos;
}
// END: find the next [QLABEL:] that belongs to a GENUINELY DIFFERENT question.
        // Skipping QLABELs that map to the SAME master prevents truncation when:
        //   a) A question has orphan sub-parts like "(b)", "(c)" — those QLABELs
        //      normalize to the same master and must NOT cut the slice.
        //   b) A student writes "Ans 36" twice (continuation) — same master.
        // Only cut when we find a QLABEL whose normalized master differs from qId.
        const lastTagInfo = positions[positions.length - 1];
        let endPos = fullTranscript.length;
        const qlScanRe = /\[QLABEL:([^\]]+)\]/g;
        qlScanRe.lastIndex = lastTagInfo.end;
        let qlMatch;
        while ((qlMatch = qlScanRe.exec(fullTranscript)) !== null) {
            const rawLabel = (qlMatch[1] || '').trim();
            const qlNorm = normalizeForComparison(rawLabel);
            // If this QLABEL normalizes to our own master → same question, skip
            if (qlNorm === qId) continue;

            // ── ANS-ANCHORED MODE ────────────────────────────────────────────
            // Only an "Ans"/"Answer"-prefixed QLABEL is a real boundary. Bare
            // sub-point labels ("1)", "2)", "i)", "a)") are the student's own
            // enumeration INSIDE this answer — never cut on them.
            if (isAnsAnchoredPaper) {
                if (/^(ans|answer)[\s.\d]/i.test(rawLabel)) {
                    endPos = qlMatch.index;
                    break;
                }
                // Not an Ans-prefixed label → continuation of THIS answer, skip.
                continue;
            }
            // ── END ANS-ANCHORED MODE ────────────────────────────────────────

            // Legacy heuristic (non-ANS-anchored papers):
            // Use a quick prefix check: if qlNorm starts with a DIFFERENT number root, cut here
            const qlNumRoot = qlNorm.match(/^(\d+)/)?.[1];
            const myNumRoot = qId.match(/^(\d+)/)?.[1];
            if (qlNumRoot && myNumRoot && qlNumRoot !== myNumRoot) {
                endPos = qlMatch.index;
                break;
            }
            // If roots are same (sub-part of same parent) → don't cut
        }

        slices[qId] = fullTranscript.substring(startPos, endPos).trim();
    });

    return slices;
}

// ─────────────────────────────────────────────────────────────────────────────
// DETERMINISTIC BOUNDARY RESOLVER
// Zero-LLM splicing engine. Works for all paper types:
//   - MCQ dense packing (Adrija page 1: Q1–Q20 on one page)
//   - Maths with no repeated labels across pages
//   - Prose answers with orphan sub-parts like "(b)", "(ii)"
//   - Mixed papers (IPC01-type deep nesting like "1. (a) (i)")
//
// How it works:
//   1. OCR now emits [QLABEL:text] on the same line as [#P:p,y,x] for every
//      line where a new question/sub-part label is visible.
//   2. We extract all [QLABEL] + their co-located [#P] coordinates.
//   3. Every [#P] tag in the transcript is assigned to the last [QLABEL] whose
//      (page, y) is <= that tag's (page, y). Pure coordinate sorting.
//   4. Orphan sub-labels ("(b)" alone) get resolved via family inheritance:
//      last seen parent "1.(a)" → family "1" → candidate "1b" → master "1. (b)".
//   5. Only truly unresolvable orphans go to a cheap LLM rescue call.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Step A: Parse all [QLABEL:text] markers and their positions from the transcript.
 * Returns array sorted by (pageNum, y) — physical document order.
 */
function parseQLabelBoundaries(fullTranscript, masterIds = []) {
    const lines = fullTranscript.split('\n');
    const boundaries = [];

    lines.forEach((line, lineIndex) => {
        // FIX: use matchAll (not match) to find ALL QLABELs on the same line.
        // OCR places multiple answers inline on one line when students write
        // consecutive short answers (e.g. page 5: all of Ans.2.a(i) through
        // Ans.2.b(ii) appear on a single long OCR line).
        // The old .match() only found the FIRST QLABEL, causing all subsequent
        // boundaries on the same line to be silently dropped — every tag after
        // the first QLABEL was assigned to Q:2.(a)(i) instead of its real owner.
        const labelMatches = [...line.matchAll(/\[QLABEL:([^\]]+)\]/g)];
        if (labelMatches.length === 0) return;

        labelMatches.forEach(labelMatch => {
            // ── ANS-ANCHORED IN-LIST GUARD ──────────────────────────────────────
            // In ANS-anchored papers, a bare number like "4." is a student's list
            // item INSIDE an answer (e.g. Q12 assumption 4), NOT a new question boundary.
            // If the transcript is ANS-anchored AND this QLABEL is a bare digit/dot/paren
            // AND it is NOT preceded by "Ans"/"Answer" → skip registering it as a boundary.
            // The resolver would skip it anyway, but registering it corrupts pageNum/y
            // for the tag sweep and causes the wrong question to own the diagram.
            const rawInner = (labelMatch[1] || '').trim();

// Guard: reject QLABELs whose number exceeds max master question number
            if (masterIds && masterIds.length > 0) {
                const labelNum = parseInt((rawInner.match(/^(\d+)/) || [])[1] || '0', 10);
                const maxMasterNum = Math.max(...masterIds.map(id => {
                    const m = String(id).match(/(\d+)/);
                    return m ? parseInt(m[1], 10) : 0;
                }));
if (labelNum > 0 && maxMasterNum > 0 && labelNum > maxMasterNum + 5) {
                    console.log(`[parseQLabelBoundaries] Rejecting out-of-range label "${rawInner}" num=${labelNum} > maxMaster=${maxMasterNum}`);
                    return;
                }
            }
            const _ansStyleCountPQL = (fullTranscript.match(/\[QLABEL:\s*(Ans|Answer|Ques|Question|Sol(?:ution)?|Q)[\s.\-\d]/gi) || []).length;

            const _bareDigitCountPQL = (fullTranscript.match(/\[QLABEL:\s*\d{1,2}[).\s]?\]/g) || []).length;
            const isAnsAnchored = _ansStyleCountPQL >= 3 && _ansStyleCountPQL > _bareDigitCountPQL;
            const isBareNumeric = /^[(]?\d{1,2}[).\s]?$/.test(rawInner);
            const isAnsPrefix = /^(ans(?:wer)?|sol(?:ution)?|ques(?:tion)?|q)[\s.\-\d]/i.test(rawInner);
if (isAnsAnchored && isBareNumeric && !isAnsPrefix) {
                const bareNum = (rawInner.match(/\d+/) || [])[0] || '';
                const ansVersionExists = bareNum &&
                    new RegExp(`\\[QLABEL:\\s*(Ans|Answer|Sol|Ques)[\\s.\\-]*${bareNum}\\b`, 'i').test(fullTranscript);
                if (ansVersionExists) {
                    console.log(`[parseQLabelBoundaries] Skipping in-list label "${rawInner}" — Ans-version exists`);
                    return;
                }
                // Only keep if this label is the FIRST token on its line
                // (genuine question label, not a list item mid-answer)
                const lineUpToLabel = line.substring(0, labelMatch.index).trim();
                if (lineUpToLabel.length > 0) {
                    console.log(`[parseQLabelBoundaries] Skipping bare label "${rawInner}" — not at line start`);
                    return;
                }
                console.log(`[parseQLabelBoundaries] Keeping bare label "${rawInner}" — no Ans-version found for Q${bareNum}`);
                // fall through → register as boundary
            }
            // ── END ANS-ANCHORED IN-LIST GUARD ─────────────────────────────────

            const afterPos = labelMatch.index + labelMatch[0].length;
const remainingOnLine = line.substring(afterPos);

            // Primary: [#P] on same line, strictly AFTER this QLABEL.
            // CRITICAL: truncate scan at the next [QLABEL:] on the same line.
            // When OCR collapses multiple answers to one line, each QLABEL must
            // only claim the [#P:] tags between itself and the next QLABEL —
            // not steal tags that belong to the next question's boundary.
            const nextQLabelOffsetOnLine = remainingOnLine.search(/\[QLABEL:/);
            const scanWindow = nextQLabelOffsetOnLine > 0
                ? remainingOnLine.substring(0, nextQLabelOffsetOnLine)
                : remainingOnLine;
            let coordMatch = scanWindow.match(/\[#P:(\d+),(\d+),(\d+)\]/);

if (!coordMatch) {
    let insideDiagram = false;
    for (let lookahead = 1; lookahead <= 50; lookahead++) {
        const nextLine = lines[lineIndex + lookahead];
        if (!nextLine) break;

        // If next line has a DIFFERENT question's QLABEL with its own [#P:] tag,
        // stop searching — we've crossed into a new question's territory.
        if (/\[QLABEL:[^\]]+\]/.test(nextLine)) {
            const nextLabelText = (nextLine.match(/\[QLABEL:([^\]]+)\]/) || [])[1] || '';
            const nextNumRoot = nextLabelText.match(/(\d+)/)?.[1];
            const myNumRoot = (labelMatch[1] || '').match(/(\d+)/)?.[1];
            if (nextNumRoot && myNumRoot && nextNumRoot !== myNumRoot) {
                // Only break if this next-question line ALSO has a [#P:] coord.
                // If it has no coord, keep searching — the current Q's coord may be further down.
                const nextHasCoord = /\[#P:\d+,\d+,\d+\]/.test(nextLine);
                if (nextHasCoord) break;
            }
        }

        // Track [DIAGRAM] and [TABLE] blocks — skip coordinate matching inside them
        if (/\[DIAGRAM\]/i.test(nextLine)) { insideDiagram = true; }
        if (/\[\/DIAGRAM\]/i.test(nextLine)) { insideDiagram = false; }
        if (/\[TABLE/i.test(nextLine)) { insideDiagram = true; }
        if (/\[\/TABLE\]/i.test(nextLine)) { insideDiagram = false; }
        if (insideDiagram) continue;

        coordMatch = nextLine.match(/\[#P:(\d+),(\d+),(\d+)\]/);
        if (coordMatch) break;
    }
}
            // If no [#P] found in the lookahead, estimate the page from lineIndex rather
            // than defaulting to 1. A document with ~40 lines per page means lineIndex/40
            // gives a reasonable page estimate. This prevents ALL boundaries from collapsing
            // to page 1 when long answers push [#P] tags far below their [QLABEL] line.
            // The boundary still gets registered correctly — the tag sweep uses (pageNum, y)
            // to sort, so even an estimated page keeps ordering correct.
            const pageNum = coordMatch ? parseInt(coordMatch[1], 10) : Math.max(1, Math.floor(lineIndex / 40) + 1);
            const y = coordMatch ? parseInt(coordMatch[2], 10) : (lineIndex * 10);


            boundaries.push({
                rawLabel: labelMatch[1].trim(),
                pageNum,
                y,
                lineIndex
            });
        });
    });

    boundaries.sort((a, b) =>
        a.pageNum !== b.pageNum ? a.pageNum - b.pageNum : a.y - b.y
    );

    return boundaries;
}

/**
 * Step B: Normalize a raw handwritten label to a canonical alphanumeric form.
 * Strips prefixes like "Ans", "Q", "Answer", punctuation, spaces.
 * "Ans 1(a)" → "1a"   |   "21." → "21"   |   "(b)" → "b"   |   "b." → "b"
 */
function normalizeLabelForMatch(raw, masterIds) {
    if (!raw) return '';
    // If master IDs contain section-prefixed numbers (e.g. "A1", "B3"), this is a
    // sectioned paper — the leading letter is a discriminator, NOT noise to strip.
    const isSectionedPaper = Array.isArray(masterIds) &&
        masterIds.some(id => /^[A-Z]\d/i.test(String(id).trim()));
    const CIRCLED_NUMS  = '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳';
    const CIRCLED_UPPER = 'ⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏ';
    const CIRCLED_LOWER = 'ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩ';
    let normalized = raw;
    // Strip surrounding box brackets: "[Answer no 2]" → "Answer no 2"
    normalized = normalized.replace(/^\[|\]$/g, '').trim();

    normalized = normalized.replace(/\[QLABEL:[^\]]*\]/g, '').trim();
    // FIX 1: Strip trailing student done-markers (x, X, ✓, √, *, ×, ✗)
    // e.g. "Qus-9.x" → "Qus-9", "Q-8.X" → "Q-8", "Q-13x" → "Q-13"
    normalized = normalized.replace(/[\s.\-_]*[xX✓√×✗*][\s.\-_]*$/, '').trim();
    for (let i = 0; i < CIRCLED_NUMS.length; i++) {
        normalized = normalized.split(CIRCLED_NUMS[i]).join(String(i + 1));
    }
    for (let i = 0; i < CIRCLED_UPPER.length; i++) {
        normalized = normalized.split(CIRCLED_UPPER[i]).join(String.fromCharCode(65 + i));
        normalized = normalized.split(CIRCLED_LOWER[i]).join(String.fromCharCode(97 + i));
    }
    
    // NEW: Fix common OCR prefix glitches
normalized = normalized.replace(/^(ns|as|aus|ams|and)\.?\s*(?=\d)/i, '');
normalized = normalized.replace(/^and[-.]?\s*(?=\d)/i, '');
    normalized = normalized.replace(/(\d+)[\s\._\-]+([a-z])/i, '$1$2');
    // Hindi sub-part: collapse "6 (क)" → "6क" and "6क" stays "6क"
// The \u0900-\u097F range survives stripping (Change 2 already done).
// This just removes spaces/parens between number and Devanagari letter.
normalized = normalized.replace(/(\d+)\s*[().]*\s*([\u0900-\u097F])/g, '$1$2');
    normalized = normalized.replace(/\(part\s*\d+\)/gi, '');

    // FIX: Compound/prefixed labels like "B8) b) D", "B8)", "B28) a)"
    // OCR emits these when student writes a section prefix before the question number.
    // Strip the leading letter prefix and trailing garbage, keep number + optional sub-part.
    // "B8) b) D" → "8b", "B8)" → "8", "B28) a)" → "28a"
    // Only fires when label starts with a letter immediately followed by a digit.
// SECTIONED PAPERS: skip this strip — the leading letter IS the section discriminator.
// Also handle "A.N" format (letter dot number): "A.1" → "1", "A.9b" → "9b"
    if (!isSectionedPaper && /^[A-Za-z][.\-]\d/.test(normalized.trim())) {
        const dotMatch = normalized.match(/^[A-Za-z][.\-](\d{1,2})\(?([a-dA-D]?)\)?/);
        if (dotMatch) {
            normalized = dotMatch[1] + (dotMatch[2] || '').toLowerCase();
        }
    }
    if (!isSectionedPaper && /^[A-Za-z]\d/.test(normalized.trim())) {
        const compoundMatch = normalized.match(/^[A-Za-z](\d{1,2})[).\s]*([a-dA-D]?)[).\s]*/);
        if (compoundMatch) {
            normalized = compoundMatch[1] + (compoundMatch[2] || '').toLowerCase();
        }
    }

    // FIX: Normalize capital section letter (A/B/C/D) that appears BETWEEN the question
    // number and the sub-part in question IDs like "26. A. (a)", "29. A.", "16. B. (a)".
    //
    // These question numbering systems use:
    //   Number  → main question  (e.g. "26")
    //   Capital → section/option (e.g. "A" or "B")
    //   Lower   → sub-part       (e.g. "(a)", "(b)")
    //
    // The student writes "Ans 26 A: (a)" → Pass 9b emits [QLABEL:26 A (a)].
    // The master ID is "26. A. (a)".
    // After stripping punctuation both normalize to: digits + uppercase_letter + subpart.
    // We lowercase everything BEFORE stripping non-alphanum, so "26 A (a)" → "26a(a)"
    // which then strips parens → "26aa". Master "26. A. (a)" → same "26aa". MATCH.
    //
    // BUT "26. B. (a)" → "26ba" and student "26 B (a)" → "26ba". ALSO MATCH. ✓
    // AND "26. A. (b)" → "26ab" and student "26 A (b)" → "26ab". ALSO MATCH. ✓
    //
    // This is correct: capital A/B section + lowercase sub-part together form a unique key.
    // No stripping needed — the existing .toLowerCase() + strip-non-alphanum already handles
    // this correctly AS LONG AS the capital letter is not accidentally stripped by the
    // word-boundary replace below. Guard: the word-boundary replace only strips standalone
    // words like "section", "answer", "q" — single letters A/B mid-label are NOT matched
    // by those patterns, so they survive to the final .replace(/[^a-z0-9]/g,'') step.
    // Nothing to change here — documenting that the existing chain is correct for this case.
    
let result = normalized.toLowerCase()
        .replace(/^an[a-z5]{0,4}s?\.?\s*(?=\d)/i, '')
        .replace(/^a[nu][st]s?\.?\s*/i, '')
        .replace(/^answer\.?\s*/i, '')
        .replace(/\b(answer|q(?:uestion|ues)?|sol(?:ution)?|section|pt|to|the|for|of|no)\b/gi, '')
        .replace(/\bpart\b/gi, '')
        // FIX 2: All Q-prefix variants — qu, que, ques, quest, qus, qut, qn, qno, q.no
        // Old ^q+(?=\d) only caught bare q/qq directly before digit
        // New catches q + up to 5 letters + optional separator before digit
        .replace(/^q[a-z]{0,5}[-.\s_]*(?:no\.?\s*)?(?=\d)/i, '')
        // FIX 3: soln, hy, ahs, aas, aws — shorthand/OCR-corrupted prefixes
        .replace(/^(soln?|hy|ahs|aas|aws)[-.\s_]*(?=\d)/i, '')
        .replace(/[^a-z0-9\u0900-\u097F]/g, '')
        .replace(/^q+(?=\d)/, '')
        .trim();

    // FIX 4: Nuclear fallback — strip any remaining pure-alpha prefix before first digit
    // Catches any future unknown prefix not handled above
    // GUARD: skip on sectioned papers (A10, B10 must stay as a10, b10)
    if (!isSectionedPaper) {
        result = result.replace(/^[a-z]+(?=\d)/, '');
    }

    // FIX 5: OCR digit-as-letter repair
    // Repairs common misreads: 1→l/I, 0→o/O when adjacent to real digits
    // e.g. "l2"→"12", "1o"→"10", "lo"→"10", "IO"→"10"
    result = result
        .replace(/^l(\d)/, '1$1')
        .replace(/^I(\d)/, '1$1')
        .replace(/(\d)o$/, '$10')
        .replace(/(\d)O$/, '$10')
        .replace(/^lo$/, '10')
        .replace(/^Io$/, '10')
        .replace(/^lo$/, '10');

    return result.trim();
}

function resolveBoundaryWithConfidence(normLabel, masterIds, lastResolvedId, ansAnchoredIds) {
    const normMasters = masterIds.map(id => ({ id, norm: normalizeLabelForMatch(id) }));
    const isExact = normMasters.some(m => m.norm === normLabel);

    let masterId = matchLabelToMasterId(normLabel, masterIds, lastResolvedId, ansAnchoredIds);
    let confidence = isExact ? 'high' : (masterId ? 'medium' : 'none');

    if (!masterId) {
        masterId = resolveOrphanByFamily(normLabel, lastResolvedId, masterIds);
        confidence = masterId ? 'low' : 'none';
    }

    return { masterId, confidence };
}

/**
 * Step C: Match a normalized label to the best master question ID.
 * Returns the master ID string or null if no match found.
 */
function matchLabelToMasterId(normLabel, masterIds, lastResolvedId = null, ansAnchoredIds = null) {

if (!normLabel) return null;

    // ORPHAN SUBPART RECOVERY: if normLabel is a single letter (a/b/c/d),
    // try compositing it with the last resolved question's number.
    // e.g. normLabel='b', lastResolvedId='10(a)' → try '10b' against masters.
    if (/^[a-d]$/.test(normLabel) && lastResolvedId) {
        const parentNum = String(lastResolvedId).match(/(\d{1,2})/);
        if (parentNum) {
            const composite = parentNum[1] + normLabel;
            const compositeMatch = masterIds.find(id => {
                const nm = normalizeLabelForMatch(String(id), masterIds);
                return nm === composite;
            });
            if (compositeMatch) return compositeMatch;
        }
    }

    const normMasters = masterIds.map(id => ({
        id,
        norm: normalizeLabelForMatch(id)
    }));

 // Exact normalized match — covers 90% of cases
    const exact = normMasters.find(m => m.norm === normLabel);
    if (exact) return exact.id;

    // MCQ ANSWER LETTER STRIP:
    // OCR emits [QLABEL:Ans 1(ii)(c)] — the answer letter (c) is appended to the roman sub-part.
    // normLabel becomes "1iic". Master is "1(ii)" → "1ii". No exact match.
    // Fix: if normLabel ends with roman-suffix + single [a-e], strip the trailing letter and retry.
    // Guard: only strip if the stripped form matches a master AND the original does NOT.
    // This preserves genuine sub-sub-parts like "1(ii)(a)" when masters actually have "1iia".
    const romanMcqStrip = normLabel.match(/^(.+(?:i{1,4}|iv|vi{0,3}|ix))([a-e])$/);
    if (romanMcqStrip) {
        const stripped = romanMcqStrip[1];
        const strippedExact = normMasters.find(m => m.norm === stripped);
        if (strippedExact) {
            console.log(`[Librarian] MCQ-strip: "${normLabel}" → "${stripped}" → master "${strippedExact.id}"`);
            return strippedExact.id;
        }
    }

    // NUMERIC PREFIX FALLBACK: student writes "37)" for a question that only
    // exists as "37. (a)" / "37. (b)" → normLabel="37", masters have "37a","37b"
    // → pick first sub-part in paper order (37.(a)).
    // Only fires when normLabel is purely numeric with no exact match.
    //
    // CRITICAL GUARD: Only fire if the prefix match is UNIQUE (all prefix matches
    // share the same numeric root and there is no ambiguity).
    // DO NOT fire when the bare number is a section header like "Q.2." on a paper
    // where masters are "2.(a)", "2.(b)", "2.(c)" — in that case the numeric label
    // "2" must NOT consume "2.(a)" as a boundary, because "2.(a)" will be correctly
    // claimed by the next [QLABEL:a)] boundary via resolveOrphanByFamily.
    // Rule: fire prefix fallback ONLY when normLabel has no alpha sub-part siblings
    // that could claim the sub-parts themselves via orphan resolution.
    // Heuristic: if ALL prefix matches share the same root AND the transcript has
    // an upcoming alpha/roman label that will resolve them, skip here.
    // Simpler safe rule: only fire when there is exactly 1 prefix match (unambiguous)
    // OR when all prefix matches are numerics only (no alpha suffix) — meaning there
    // truly are no sub-parts to claim.
if (/^\d+$/.test(normLabel)) {
    // Try exact match first
    const exactMatch = normMasters.find(m => m.norm === normLabel);
    if (exactMatch) {
        // GUARD: If this bare number is an ANS-anchored master (student wrote
        // "Ans N" elsewhere), then a BARE "N)" reaching here is almost
        // certainly a student sub-point INSIDE another answer, not a new
        // boundary for QN. Refuse the match — let it fall to orphan/family.
        // Only block when the exact master IS ans-anchored AND this label was
        // not itself ans-prefixed (caller strips "Ans" before normalizing, so
        // we cannot tell here — the ansAnchoredIds membership is the signal).
if (ansAnchoredIds && ansAnchoredIds.has(normLabel)) {
            // Before blocking, attempt sequential rescue.
            // "Ans 12" OCR'd as "Ans 2" → normLabel="2", lastResolvedId="11"
            // "2" < "11"-2 → try "12" → found → rescue succeeds.
            if (lastResolvedId) {
                const lastNum = parseInt(normalizeLabelForMatch(lastResolvedId).match(/(\d+)/)?.[1] || '0');
                const thisNum = parseInt(normLabel);
                if (thisNum > 0 && thisNum < lastNum - 2) {
                    const prefixed = '1' + normLabel;
                    const prefixMatch = normMasters.find(m => m.norm === prefixed);
                    if (prefixMatch) {
                        console.log(`[Librarian] ANS-guard sequential rescue: "${normLabel}" → "${prefixed}" (last="${lastResolvedId}")`);
                        return prefixMatch.id;
                    }
                }
            }
            console.log(`[Librarian] Bare number "${normLabel}" NOT matched to ANS-anchored master "${exactMatch.id}" (suspected in-answer sub-point).`);
            return null;
        }

// Single-digit OCR drop: "4" when student wrote "14", "24", "34" etc.
// Try all prefixed candidates, pick the one closest to lastResolvedId.
if (/^\d$/.test(normLabel)) {
    const exactExists = normMasters.find(m => m.norm === normLabel);
    if (!exactExists) {
        const prefixCandidates = ['1','2','3']
            .map(p => ({ prefixed: p + normLabel, match: normMasters.find(m => m.norm === p + normLabel) }))
            .filter(c => c.match);
        if (prefixCandidates.length === 1) {
            // Only one possible match — unambiguous
            console.log(`[Librarian] Single-digit drop rescue: "${normLabel}" → "${prefixCandidates[0].prefixed}"`);
            return prefixCandidates[0].match.id;
        } else if (prefixCandidates.length > 1 && lastResolvedId) {
            // Multiple candidates — pick closest to lastResolvedId
            const lastNum = parseInt(normalizeLabelForMatch(lastResolvedId).match(/^(\d+)/)?.[1] || '0');
            prefixCandidates.sort((a, b) =>
                Math.abs(parseInt(a.prefixed) - lastNum) - Math.abs(parseInt(b.prefixed) - lastNum)
            );
            console.log(`[Librarian] Single-digit drop rescue (multi): "${normLabel}" → "${prefixCandidates[0].prefixed}" (closest to last="${lastResolvedId}")`);
            return prefixCandidates[0].match.id;
        }
    }
}
    // ── SEQUENTIAL ORDER RESCUE ──────────────────────────────────────────────
// Problem: notebook margin line makes "Ans 12" look like "Ans 2" to OCR.
// "2" matches master "2" exactly — but we just resolved Q11.
// If going backwards by >2 AND "1"+normLabel exists as master → rescue.
if (lastResolvedId) {
    const lastNum = parseInt(normalizeLabelForMatch(lastResolvedId).match(/(\d+)/)?.[1] || '0');
    const thisNum = parseInt(normLabel);
    if (thisNum > 0 && thisNum < lastNum - 2) {
        const prefixed = '1' + normLabel;
        const prefixMatch = normMasters.find(m => m.norm === prefixed);
        if (prefixMatch) {
            console.log(`[Librarian] Sequential rescue: "${normLabel}" → "${prefixed}" (last="${lastResolvedId}")`);
            return prefixMatch.id;
        }
    }
}
// ── OVER-READ RESCUE ─────────────────────────────────────────────────────
// Problem: OCR adds a leading digit. "Ans 12" → "Ans 22", "Ans 13" → "Ans 113"
// thisNum > lastNum + realistic_gap AND stripping first digit gives a valid master
if (lastResolvedId) {
    const lastNum = parseInt(normalizeLabelForMatch(lastResolvedId).match(/(\d+)/)?.[1] || '0');
    const thisNum = parseInt(normLabel);
    if (thisNum > lastNum + 3) {  // jumped too far forward — suspicious
        const stripped = String(thisNum).slice(1); // drop first digit: "22" → "2", "113" → "13"
        const strippedMatch = normMasters.find(m => m.norm === stripped);
        if (strippedMatch) {
            const strippedNum = parseInt(stripped);
            // Only rescue if stripped number is a natural next step
            if (strippedNum > lastNum && strippedNum <= lastNum + 5) {
                console.log(`[Librarian] Over-read rescue: "${normLabel}" → "${stripped}" (last="${lastResolvedId}")`);
                return strippedMatch.id;
            }
        }
    }
}
// ── END OVER-READ RESCUE ──────────────────────────────────────────────────

// ── SEQUENTIAL ORDER RESCUE (UNCONDITIONAL) ──────────────────────────────
// Runs for ALL papers — Ans-anchored or not.
// If "2" matched Q2 exactly but last resolved was Q11 → suspicious.
// Try "12" before accepting "2".
if (lastResolvedId && exactMatch) {
    const lastNum = parseInt(normalizeLabelForMatch(lastResolvedId).match(/(\d+)/)?.[1] || '0');
    const thisNum = parseInt(normLabel);
    if (thisNum > 0 && thisNum < lastNum - 1) {  // going backwards by more than 1
        const prefixed = '1' + normLabel;
        const prefixMatch = normMasters.find(m => m.norm === prefixed);
        if (prefixMatch) {
            console.log(`[Librarian] Sequential rescue (unconditional): "${normLabel}" → "${prefixed}" (last="${lastResolvedId}")`);
            return prefixMatch.id;
        }
    }
}
// ── END SEQUENTIAL ORDER RESCUE ──────────────────────────────────────────


// ── SEQUENTIAL ORDER RESCUE (UNCONDITIONAL) ──────────────────────────────
if (lastResolvedId && exactMatch) {
    const lastNum = parseInt(normalizeLabelForMatch(lastResolvedId).match(/(\d+)/)?.[1] || '0');
    const thisNum = parseInt(normLabel);
    // Only applies to single-digit normLabels — multi-digit misreads are handled elsewhere
    if (/^\d$/.test(normLabel) && lastNum > 0) {
        // Try all plausible prefixes: "13", "23", "33", "43"
        const prefixCandidates = ['1','2','3','4']
            .map(p => ({ prefixed: p + normLabel, num: parseInt(p + normLabel), match: normMasters.find(m => m.norm === p + normLabel) }))
            .filter(c => c.match);
        // Filter: must be a natural next step from lastNum (within 5)
        const naturalNext = prefixCandidates.filter(c => c.num > lastNum && c.num <= lastNum + 5);
        if (naturalNext.length === 1) {
            // Unambiguous — only one candidate is a natural next step
            console.log(`[Librarian] Sequential rescue (single-digit drop): "${normLabel}" → "${naturalNext[0].prefixed}" (last="${lastResolvedId}")`);
            return naturalNext[0].match.id;
        } else if (naturalNext.length > 1) {
            // Multiple candidates — pick closest to lastNum
            naturalNext.sort((a, b) => Math.abs(a.num - lastNum - 1) - Math.abs(b.num - lastNum - 1));
            console.log(`[Librarian] Sequential rescue (multi-candidate): "${normLabel}" → "${naturalNext[0].prefixed}" (last="${lastResolvedId}")`);
            return naturalNext[0].match.id;
        }
        // Backwards case: "3" after Q12 where Q13 exists but is not "natural next" — still try
        if (thisNum < lastNum - 1) {
            const backwardsCandidates = prefixCandidates.filter(c => c.num > lastNum);
            if (backwardsCandidates.length === 1) {
                console.log(`[Librarian] Sequential rescue (backwards drop): "${normLabel}" → "${backwardsCandidates[0].prefixed}" (last="${lastResolvedId}")`);
                return backwardsCandidates[0].match.id;
            }
        }
    }
}
// ── END SEQUENTIAL ORDER RESCUE ──────────────────────────────────────────

        return exactMatch.id;
    }


    
    
// AFTER:
    const startsWithMatch = normMasters.find(m => m.norm.startsWith(normLabel));
    if (startsWithMatch) {
        console.log(`[Librarian] Single number "${normLabel}" mapped to "${startsWithMatch.id}"`);
        return startsWithMatch.id;
    }
}

// ── NEW: Dot-alpha pattern "33.a." → "33a" ───────────────────────────────────
const dotAlphaPattern = normLabel.match(/^(\d+)\.([a-z])\.?$/);
if (dotAlphaPattern) {
    const candidate = dotAlphaPattern[1] + dotAlphaPattern[2];
    const match = normMasters.find(m => m.norm === candidate);
    if (match) {
        console.log(`[Librarian] Dot-alpha pattern: "${normLabel}" → "${candidate}" → master "${match.id}"`);
        return match.id;
    }
}

    // ── NEW: ALPHA-DOT PATTERN "33a." → "33a" ────────────────────────────────────
    const alphaDotPattern = normLabel.match(/^(\d+[a-z])\.$/);
    if (alphaDotPattern) {
        const candidate = alphaDotPattern[1];
        const match = normMasters.find(m => m.norm === candidate);
        if (match) {
            console.log(`[Librarian] Alpha-dot pattern: "${normLabel}" → "${candidate}" → master "${match.id}"`);
            return match.id;
        }
    }

     // NUMERIC+ALPHA PREFIX FALLBACK:
// Student writes "6a", "6b", "7a", "7i" but master only has "6", "7"
// (the teacher set up Q6 and Q7 as single questions, not sub-parts).
// normLabel = "6a" → extract numeric prefix "6" → check if "6" is a master.
// If yes and there is NO master "6a", map to master "6".
// This handles papers where student self-divides a question into sub-parts
// that don't exist as separate master IDs.
const numAlphaMatch = normLabel.match(/^(\d+)([a-z]+)$/);
if (numAlphaMatch) {
    const numPart = numAlphaMatch[1];   // e.g. "2" from "2c"
    const alphaPart = numAlphaMatch[2]; // e.g. "c" from "2c"

    // ── NEW BLOCK 1: ROMAN-SUFFIX STRIPPING ──────────────────────────────────
    // Handles: "1ai"→"1a", "1aii"→"1a", "2ci"→"2c", "1bi"→"1b"
    // Student wrote Ans.1.a(i) but master is only "1.(a)" — strip trailing roman.
    const romanSuffix = /^(.*[a-d])(i{1,3}|iv|vi{0,3}|ix)$/;
    const romanMatch = alphaPart.match(romanSuffix);
    if (romanMatch) {
        const alphaWithoutRoman = numPart + romanMatch[1]; // e.g. "1"+"a" = "1a"
        const masterWithoutRoman = normMasters.find(m => m.norm === alphaWithoutRoman);
        if (masterWithoutRoman && !normMasters.some(m => m.norm === normLabel)) {
            console.log(`[Librarian] Roman-suffix stripped: "${normLabel}" → "${alphaWithoutRoman}" → master "${masterWithoutRoman.id}"`);
            return masterWithoutRoman.id;
        }
    }
    // ── END BLOCK 1 ──────────────────────────────────────────────────────────

    // ── NEW BLOCK 2: ALPHA-IS-ONLY-SUB-PART COLLAPSE ─────────────────────────
    // Handles: student wrote "2c" but master only has "2.(c)(i)" = norm "2ci"
    // i.e., there is NO master "2c" — only deeper "2ci", "2cii" etc.
    // The student's label "2c" is a VALID PARENT reference to the whole "2c" group.
    // Map it to the FIRST master that starts with normLabel.
    //
    // GUARD: Only fire if ALL masters starting with normLabel share the SAME
    // alpha prefix (normLabel itself). This means there are no sibling alpha
    // sub-parts like "2a", "2b" — only roman sub-parts of "2c" exist.
    // If masters have both "2a" and "2b", the student's "2c" is genuinely ambiguous
    // and should NOT collapse — let orphan resolver handle it.
    const prefixMatches = normMasters.filter(m => m.norm.startsWith(normLabel));
    if (prefixMatches.length > 0 && !normMasters.some(m => m.norm === normLabel)) {
        // All prefix matches must start with THIS normLabel (not a sibling like "2ca")
        const allShareSameAlphaRoot = prefixMatches.every(m =>
            m.norm.startsWith(normLabel) &&
            // The remainder after normLabel must be roman numerals only (i, ii, iii, iv…)
            /^(i{1,4}|iv|vi{0,3}|ix)?$/.test(m.norm.slice(normLabel.length))
        );
        if (allShareSameAlphaRoot) {
            const firstMatch = prefixMatches[0];
            console.log(`[Librarian] Alpha-collapse: "${normLabel}" (no exact master) → first roman sub "${firstMatch.id}"`);
            return firstMatch.id;
        }
    }
    // ── END BLOCK 2 ──────────────────────────────────────────────────────────

    // Original guard: no master has this exact label AND no master starts with it
    const anyMasterStartsWith = normMasters.some(m => m.norm.startsWith(normLabel));
    if (!anyMasterStartsWith) {
        // normLabel like "6a" has no masters — check if bare number "6" is a master
        const exactNumMaster = normMasters.find(m => m.norm === numPart);
        if (exactNumMaster) {
            console.log(`[Librarian] Student sub-part "${normLabel}" mapped to parent master "${exactNumMaster.id}"`);
            return exactNumMaster.id;
        }
    }

    // ── ALPHA-SUFFIX STRIPPING FALLBACK ──────────────────────────────────────
    // Pattern: student writes "31) B) a)" → OCR emits [QLABEL:31) B) a)] → normLabel="31ba"
    // Master is "31. b." → norm "31b". The extra trailing 'a' comes from the student
    // writing their sub-part choice letter after the section choice letter.
    // Fix: try stripping 1-2 trailing alpha chars until we find an exact master match.
    // Guard: only for labels with 2+ alpha chars in suffix (single-alpha already handled above).
    if (/^(\d+)([a-z]{2,})$/.test(normLabel)) {
        const nm2 = normLabel.match(/^(\d+)([a-z]{2,})$/);
        const numP = nm2[1];
        const alphaP = nm2[2];
        for (let len = alphaP.length - 1; len >= 0; len--) {


            const candidate = numP + alphaP.substring(0, len);
            const stripMatch = normMasters.find(m => m.norm === candidate);
            if (stripMatch) {
                console.log(`[Librarian] Alpha-suffix stripped: "${normLabel}" → "${candidate}" → master "${stripMatch.id}"`);
                return stripMatch.id;
            }
        }
    }
    // ── END ALPHA-SUFFIX STRIPPING ────────────────────────────────────────────
}

    // ── OCR MISREAD FUZZY FALLBACK ────────────────────────────────────────────
    // Common OCR confusion: handwritten 'c' is frequently misread as '0' (zero).
    // This causes "Ans 2.c(i)" → OCR → "Ans 2.0" → norm "20" → no match for "2ci".
    // Also: "Ans 2.c(i)" → OCR → "Ans 2.0(i)" → norm "20i" → no match for "2ci".
    //
    // Strategy: replace '0' (that could be misread 'c') with letter candidates.
    // Pattern A: normLabel ends with '0'      e.g. "20"  → try "2c", "2ci", "2cii"
    // Pattern B: normLabel has '0i' or '0ii'  e.g. "20i" → try "2ci"; "20ii" → "2cii"
    // Only fires when result is an exact master match — never guesses.
    const fuzzyVariants = [];
    if (/0$/.test(normLabel)) {
        // trailing '0' → try replacing with c, ci, cii, ciii
        const base = normLabel.slice(0, -1);
        ['c', 'ci', 'cii', 'ciii'].forEach(s => fuzzyVariants.push(base + s));
    }
    if (/0i{1,3}$/.test(normLabel)) {
        // trailing '0i', '0ii', '0iii' → replace '0' with 'c'
        fuzzyVariants.push(normLabel.replace(/0(i{1,3})$/, 'c$1'));
    }
    for (const candidate of fuzzyVariants) {
        const fuzzyMatch = normMasters.find(m => m.norm === candidate);
        if (fuzzyMatch) {
            console.log(`[Librarian] OCR c→0 fuzzy rescue: "${normLabel}" → "${candidate}" → "${fuzzyMatch.id}"`);
            return fuzzyMatch.id;
        }
    }
    // ─────────────────────────────────────────────────────────────────────────

    // ROMAN NUMERAL SUB-PART FALLBACK:
    // Student writes sub-parts as "i)", "ii)", "iii)", "iv)" etc.
    // OCR emits [QLABEL:i)], [QLABEL:ii)] — normLabel = "i", "ii", "iii", "iv"
    // Master IDs are "1. i)", "1. ii)" — normalized to "1i", "1ii"
    // Direct match fails. Try prepending each unique parent number from masters.
    // e.g. normLabel="i" → try "1i", "2i", "3i" against all masters.
const romanPattern = /^(i{1,3}|iv|v|vi{0,3}|ix|x|xi{0,3}|xiv|xv|xvi{0,3})$/;
if (romanPattern.test(normLabel)) {
    // Priority 1: use family from lastResolvedId (same as alpha fallback)
    const priorityParents = [];
    if (lastResolvedId) {
        const fm = normalizeLabelForMatch(lastResolvedId).match(/^(\d+)/);
        if (fm) priorityParents.push(fm[1]);
    }
    // Priority 2: all other parent numbers in master order
    const allParents = [...new Set(
        normMasters.map(m => m.norm.match(/^(\d+)/)?.[1]).filter(Boolean)
    )];
    const orderedParents = [
        ...priorityParents,
        ...allParents.filter(p => !priorityParents.includes(p))
    ];
    for (const parent of orderedParents) {
        const candidate = parent + normLabel;
        const match = normMasters.find(m => m.norm === candidate);
        if (match) return match.id;
    }
}

    // ALPHA SUB-PART FALLBACK:
    // Student writes "a)", "b)", "c)" as sub-parts.
    // normLabel = "a", "b", "c" — masters have "1a", "1b", "1c"
    // Same strategy: try prepending each parent number.
if (/^[a-d]$/.test(normLabel)) {
    // Priority 1: use family from lastResolvedId
    const priorityParents = [];
    if (lastResolvedId) {
        const fm = normalizeLabelForMatch(lastResolvedId).match(/^(\d+)/);
        if (fm) priorityParents.push(fm[1]);
    }
    // Priority 2: all other parent numbers
    const allParents = [...new Set(
        normMasters.map(m => m.norm.match(/^(\d+)/)?.[1]).filter(Boolean)
    )];
    const orderedParents = [
        ...priorityParents,
        ...allParents.filter(p => !priorityParents.includes(p))
    ];
    for (const parent of orderedParents) {
        const candidate = parent + normLabel;
        const match = normMasters.find(m => m.norm === candidate);
        if (match) return match.id;
    }
}

// ── FINAL FALLBACK: NUMERIC ROOT EXTRACTION ──────────────────────────────
    // Student wrote "26 B (a)" → norm "26ba" → no master "26ba".
    // Master only has "26" → norm "26". Extract numeric root from normLabel
    // and check if it's an exact master. Covers all "section letter" papers
    // and any case where student subdivides a question the teacher didn't.
    // Guard: only fires if normLabel has a numeric prefix (not pure alpha orphan).
    const numericRootFinal = normLabel.match(/^(\d+)/)?.[1];
    if (numericRootFinal) {
        const rootMaster = normMasters.find(m => m.norm === numericRootFinal);
        if (rootMaster) {
            console.log(`[Librarian] Numeric-root final fallback: "${normLabel}" → master "${rootMaster.id}"`);
            return rootMaster.id;
        }
    }

    return null;
}

/**
 * Step D: Orphan sub-label family resolution (zero LLM).
 * When student writes "(b)" alone, we find the last resolved parent (e.g. "1. (a)")
 * and try to match family + orphan = "1b" against master IDs.
 *
 * Handles multi-level: if last resolved was "1. (a) (i)", family = "1",
 * sub-family = "1a", so "(ii)" → try "1aii" first, then "1ii".
 */
function resolveOrphanByFamily(orphanNorm, lastResolvedId, masterIds) {
    if (!lastResolvedId || !orphanNorm) return null;

    const normMasters = masterIds.map(id => ({ id, norm: normalizeLabelForMatch(id) }));
    const lastNorm = normalizeLabelForMatch(lastResolvedId);

    // Build candidates from longest prefix to shortest.
    // For lastNorm="1ai" (= "1. (a) (i)") and orphanNorm="ii":
    //   candidate[0] = "1a"  + "ii" = "1aii"   ← strip last part of lastNorm, add orphan
    //   candidate[1] = "1"   + "ii" = "1ii"
    //   candidate[2] = ""    + "ii" = "ii"      (only if nothing else matches)
    // For lastNorm="1a" and orphanNorm="b":
    //   candidate[0] = "1"   + "b"  = "1b"      ← correct
    const candidates = [];
    const compoundRoman = /^(\d+)(i{1,3}|iv|v|vi{0,3}|ix|x)$/;
if (compoundRoman.test(orphanNorm)) {
    const direct = normMasters.find(m => m.norm === orphanNorm);
    if (direct) return direct.id;
}

    // Generate progressively shorter prefixes of lastNorm
    // Strip 1 character at a time from the end (each removal drops one sub-level)
    // CRITICAL: start from len-1 not len.
    // Orphan REPLACES the current deepest sub-level, not extends it.
    // e.g. lastNorm="6ci", orphan="ii" → try "6c"+"ii"="6cii" ✓ (not "6ci"+"ii"="6ciii")
    for (let len = lastNorm.length - 1; len >= 0; len--) {
        const prefix = lastNorm.substring(0, len);
        const candidate = prefix + orphanNorm;
        if (candidate && candidate !== orphanNorm) { // skip bare orphan (no context)
            candidates.push(candidate);
        }
    }
    // Also try bare family number + orphan explicitly
    const familyMatch = lastResolvedId.match(/^(\d+)/);
    if (familyMatch) {
        const bare = familyMatch[1] + orphanNorm;
        if (!candidates.includes(bare)) candidates.push(bare);
    }

    // Try each candidate, return first UNIQUE match
    for (const candidate of candidates) {
        const matches = normMasters.filter(m => m.norm === candidate);
        if (matches.length === 1) return matches[0].id;
        // If multiple matches for same candidate, it's ambiguous — try next shorter prefix
    }

    // ── ROMAN-SUFFIX STRIP ON ORPHAN ────────────────────────────────────────
    // Pattern: student writes "b) (i)" → OCR emits [QLABEL:b) (i)] → orphanNorm="bi"
    // No master has roman sub-parts (e.g. master is "31. b." = "31b", not "31bi").
    // Fix: strip the trailing roman suffix from orphanNorm and retry with bare alpha.
    // "bi" → strip "i" → "b" → retry all candidates with "b" → finds "31b" = "31. b."
    // Guard: only fires when orphanNorm ends with a roman suffix AND has a leading alpha.
    const romanSuffixRx = /^([a-z]+?)(i{1,4}|iv|vi{0,3}|ix)$/;
    const romanSuffixMatch = orphanNorm.match(romanSuffixRx);
    if (romanSuffixMatch) {
        const strippedOrphan = romanSuffixMatch[1]; // e.g. "b" from "bi"
        const strippedCandidates = [];
        for (let len = lastNorm.length - 1; len >= 0; len--) {
            const prefix = lastNorm.substring(0, len);
            const candidate = prefix + strippedOrphan;
            if (candidate && candidate !== strippedOrphan) strippedCandidates.push(candidate);
        }
        if (familyMatch) {
            const bare = familyMatch[1] + strippedOrphan;
            if (!strippedCandidates.includes(bare)) strippedCandidates.push(bare);
        }
        for (const candidate of strippedCandidates) {
            const matches = normMasters.filter(m => m.norm === candidate);
            if (matches.length === 1) {
                console.log(`[Librarian] Roman-orphan-strip: "${orphanNorm}" → stripped "${strippedOrphan}" → master "${matches[0].id}"`);
                return matches[0].id;
            }
        }
    }
    // ── END ROMAN-SUFFIX STRIP ON ORPHAN ─────────────────────────────────────

    return null; // truly ambiguous → escalate to LLM rescue
}

function deterministicBoundaryResolver(fullTranscript, masterIds, questions) {
    const lines = fullTranscript.split('\n');
    let boundaries = [];
    
    try {
       boundaries = parseQLabelBoundaries(fullTranscript, masterIds);
    } catch (err) {
        console.warn('[deterministicBoundaryResolver] Failed to parse QLABEL boundaries:', err.message);
        boundaries = [];
    }

    // If no boundaries found, return empty mappings
    if (!boundaries || boundaries.length === 0) {
        console.log('[deterministicBoundaryResolver] No boundaries found, returning empty mappings');
        return { mappings: [], orphanTags: [], unresolvedBoundaries: [] };
    }

    // Collect ALL [#P] tags with their positions
    const allTags = [];
    lines.forEach((line, lineIndex) => {
        const matches = [...line.matchAll(/\[#P:(\d+),(\d+),(\d+)\]/g)];
        matches.forEach(m => {
            allTags.push({
                tag:     m[0],
                pageNum: Math.max(1, parseInt(m[1], 10)),
                y:       parseInt(m[2], 10),
                lineIndex
            });
        });
    });

    // Resolve each boundary label → master question ID
    let lastResolvedId = null;
    const resolvedBoundaries   = [];
    const unresolvedBoundaries = [];

    // Build a fast lookup: normalized master ID → question type.
    const masterTypeMap = {};
    questions.forEach(q => {
masterTypeMap[normalizeLabelForMatch(q.questionNumber, masterIds)] = (q.type || '').toUpperCase();
    });

const ansAnchoredIds = new Set();
    boundaries.forEach(b => {
        const raw = (b.rawLabel || '').trim();
        if (/^(ans(?:wer)?|sol(?:ution)?|ques(?:tion)?|q)[\s.\-]/i.test(raw)) {
            const norm = normalizeLabelForMatch(raw, masterIds);
            if (norm) ansAnchoredIds.add(norm);
        }
    });

    boundaries.forEach(b => {
        const rawLower = (b.rawLabel || '').toLowerCase();
        
        // Skip structural labels
        if (/^(section|set[-\s]?[a-z0-9]|part[-\s]?[ivx\d])/i.test(rawLower)) {
            console.log("[Librarian] Skipping structural label: " + b.rawLabel);
            return;
        }

    let normLabel = normalizeLabelForMatch(b.rawLabel, masterIds);

        // Empty label guard
        if (!normLabel) {
            console.log('[Librarian] Empty label skipped: ' + JSON.stringify(b.rawLabel));
            return;
        }

        // Bare uppercase letter guard (math variables)
        if (/^[A-Z]$/.test(b.rawLabel.trim())) {
            console.log('[Librarian] Bare uppercase letter suppressed (math variable): ' + JSON.stringify(b.rawLabel));
            return;
        }

const isAnsPrefix = /^(ans(?:wer)?|sol(?:ution)?|ques(?:tion)?|q)[\s.\-]/i.test((b.rawLabel || '').trim());

if (!isAnsPrefix && ansAnchoredIds.has(normalizeLabelForMatch(b.rawLabel, masterIds))) {
            console.log(`[Librarian] Non-ANS boundary suppressed (ANS anchor exists): "${b.rawLabel}"`);
            return;
        }

        // MCQ answer-letter guard
        const mcqAnswerPattern = normLabel.match(/^(\d+)([a-d])$/);
        if (mcqAnswerPattern) {
            const numericPart = mcqAnswerPattern[1];
            const numericType = masterTypeMap[numericPart];
            if (numericType === 'MCQ') {
                normLabel = numericPart;
                console.log(`[Librarian] MCQ answer-letter stripped: "${b.rawLabel}" → treating as boundary for Q${numericPart}`);
            }
        }


        // Also strip roman/letter subpart from "N (iii)" or "N(b)" QLABEL format.
// e.g. [QLABEL:6 (iii)] → treat as boundary for Q6 when Q6 is MCQ type.
const subpartStripMatch = normLabel.match(/^(\d+)\s*[\(\[]?(?:i{1,4}|iv|vi{0,3}|ix|[a-e])[\)\]]?$/);
if (subpartStripMatch) {
    const numericPart = subpartStripMatch[1];
    const numericType = masterTypeMap[numericPart];
    if (numericType === 'MCQ') {
        normLabel = numericPart;
        console.log(`[Librarian] MCQ subpart stripped: "${b.rawLabel}" → treating as boundary for Q${numericPart}`);
    }
}

// Guard: reject QLABELs whose leading number exceeds the max master question number.
        // Catches student-written labels like "57.)" and "18.)" when the paper only has
        // Q1-Q13 — these are section/textbook numbers, not answer labels.
        if (masterIds.length > 0) {
            const labelNum = parseInt((normLabel.match(/^(\d+)/) || [])[1] || '0', 10);
            const maxMasterNum = Math.max(...masterIds.map(id => {
                const m = String(id).match(/(\d+)/);
                return m ? parseInt(m[1], 10) : 0;
            }));
if (labelNum > 0 && maxMasterNum > 0 && labelNum > maxMasterNum + 5) {
                console.log(`[Librarian] Rejecting out-of-range label "${b.rawLabel}" → num=${labelNum} > maxMaster=${maxMasterNum}`);
                return;
            }
        }

// Guard: if normLabel is empty or pure-alpha with no digits (e.g. "Ans." → "ans"),
        // it can never match any master ID — skip it to avoid creating a black-hole boundary.
        if (!normLabel || !/\d/.test(normLabel)) {
            console.log(`[Librarian] Skipping unresolvable boundary label: "${b.rawLabel}" → "${normLabel}" (no digits)`);
            unresolvedBoundaries.push({ ...b, normLabel });
            return;
        }


// Try direct match, ranked by confidence
        const { masterId: resolvedMasterId, confidence: matchConfidence } =
            resolveBoundaryWithConfidence(normLabel, masterIds, lastResolvedId, ansAnchoredIds);
        let masterId = resolvedMasterId;

if (masterId) {
            lastResolvedId = masterId;
            // Guard: if duplicate boundary exists, keep whichever has HIGHER confidence.
            const existingBoundaryForId = resolvedBoundaries.find(rb => rb.masterId === masterId);
            if (existingBoundaryForId) {
                const existingConf = existingBoundaryForId.confidence || 'medium';
                const rank = { high: 3, medium: 2, low: 1 };
                if (rank[matchConfidence] > rank[existingConf]) {
                    // Extra guard: if existing boundary is on page 1 and new one
                    // is on a later page, ALWAYS prefer the later page — page 1
                    // is commonly a cover/marks-table page, never real answers.
                    const preferNew = b.pageNum > existingBoundaryForId.pageNum || true;
                    console.log(`[Librarian] Upgrading boundary for ${masterId}: ${existingConf} → ${matchConfidence}`);
                    const idx = resolvedBoundaries.indexOf(existingBoundaryForId);
                    resolvedBoundaries[idx] = { ...b, masterId, confidence: matchConfidence };
                } else {
                    console.log(`[Librarian] Duplicate boundary for ${masterId} on page ${b.pageNum} — keeping existing (${existingConf} >= ${matchConfidence})`);
                }
            } else {
                resolvedBoundaries.push({ ...b, masterId, confidence: matchConfidence });
            }
        } else {
            unresolvedBoundaries.push({ ...b, normLabel });
        }
    });

    // Sort resolved boundaries by document order
    resolvedBoundaries.sort((a, b) =>
        a.pageNum !== b.pageNum ? a.pageNum - b.pageNum : a.y - b.y
    );

        // ── NEW: FALLBACK FOR MCQ DENSE BLOCKS ──────────────────────────────────────
    // If we have unresolved boundaries that look like MCQ patterns ("1)", "2)", etc.)
    // but no resolved boundaries at all, create synthetic boundaries from the transcript.
    if (resolvedBoundaries.length === 0 && unresolvedBoundaries.length > 0) {
        console.log('[Librarian] No resolved boundaries found. Attempting MCQ pattern fallback...');
        
        // Scan the transcript for MCQ patterns like "1) c", "2) b", etc.
        const mcqPattern = /^(\d{1,2})\)\s+[a-d]/gm;
        let mcqMatch;
        const lines = fullTranscript.split('\n');
        
lines.forEach((line, lineIdx) => {
            const match = line.match(/^(\d{1,2})\)\s+[a-d]/);
            if (match) {
                const num = match[1];
                // EXACT match only — no more startsWith("1") matching "1","10","12","18"
                const matchingMaster = masterIds.find(id => {
                    const normId = normalizeLabelForMatch(id, masterIds);
                    return normId === num || normId === num + ')';
                });
                if (matchingMaster) {
                    const pTagMatch = line.match(/\[#P:(\d+),(\d+),(\d+)\]/);
                    let realPage = 1, realY = lineIdx * 10;
                    if (pTagMatch) {
                        realPage = parseInt(pTagMatch[1], 10);
                        realY = parseInt(pTagMatch[2], 10);
                    } else {
                        for (let li = lineIdx - 1; li >= 0; li--) {
                            const prevTag = lines[li].match(/\[#P:(\d+),(\d+),(\d+)\]/);
                            if (prevTag) { realPage = parseInt(prevTag[1], 10); realY = parseInt(prevTag[2], 10) + 5; break; }
                        }
                    }
                    resolvedBoundaries.push({
                        rawLabel: `${num})`,
                        normLabel: num,
                        masterId: matchingMaster,
                        pageNum: realPage,
                        y: realY,
                        lineIndex: lineIdx,
                        confidence: 'medium'
                    });
                    console.log(`[Librarian] MCQ Fallback: Q${matchingMaster} from "${match[0]}" at page ${realPage}`);
                }
            }
        });
        
        // Re-sort after adding synthetic boundaries
        resolvedBoundaries.sort((a, b) =>
            a.pageNum !== b.pageNum ? a.pageNum - b.pageNum : a.y - b.y
        );
    }
    // ─────────────────────────────────────────────────────────────────────────────

    // Assign each [#P] tag to the last resolved boundary
    const mappings = {};
    masterIds.forEach(id => { mappings[id] = []; });
    let orphanTags = [];

    allTags.forEach(tagInfo => {
        let assignedId = null;
        for (let i = resolvedBoundaries.length - 1; i >= 0; i--) {
            const b = resolvedBoundaries[i];
            if (b.pageNum < tagInfo.pageNum || (b.pageNum === tagInfo.pageNum && b.y <= tagInfo.y)) {
                assignedId = b.masterId;
                break;
            }
        }
        if (assignedId && mappings[assignedId] !== undefined) {
            mappings[assignedId].push(tagInfo.tag);
        } else {
            orphanTags.push(tagInfo.tag);
        }
    });

    // Rough-work column absorption (x ≥ 850)
    const absorbedOrphans = [];
    orphanTags.forEach(orphanTag => {
        const xMatch = orphanTag.match(/\[#P:(\d+),(\d+),(\d+)\]/);
        if (!xMatch) return;
        const tagPage = parseInt(xMatch[1], 10);
        const tagX = parseInt(xMatch[3], 10);
        if (tagX >= 850) {
            let targetId = null;
            for (let i = resolvedBoundaries.length - 1; i >= 0; i--) {
                if (resolvedBoundaries[i].pageNum <= tagPage) {
                    targetId = resolvedBoundaries[i].masterId;
                    break;
                }
            }
            if (targetId && mappings[targetId] !== undefined) {
                mappings[targetId].push(orphanTag);
                absorbedOrphans.push(orphanTag);
                console.log(`[RoughWork] Absorbed orphan ${orphanTag} (x≥850) into ${targetId}`);
            }
        }
    });
    orphanTags = orphanTags.filter(t => !absorbedOrphans.includes(t));

    

    // Format output
    const formattedMappings = Object.entries(mappings)
        .filter(([_, tags]) => tags.length > 0)
        .map(([id, tags]) => ({ id, tags }));

// ── MISLABEL SANITY CHECK ──────────────────────────────────────────────────
    // Problem: student writes wrong question number (e.g. "Ques-7" for Q9).
    // Deterministic resolver maps those tags to Q7 (wrong). Q9 gets zero tags.
    // Detection: if a question is in the LATER half of the paper (by index)
    // but ALL its assigned tags are on pages 1-2, it's likely a mislabeled answer.
    // Action: mark _suspectedMislabel on the question object → requiresReview in report.
    const totalQCount = questions.length;
    questions.forEach((q, qIdx) => {
        const assignedTags = mappings[q.questionNumber] || [];
        if (assignedTags.length === 0) return;
        const tagPages = assignedTags
            .map(t => parseInt((t.match(/\[#P:(\d+),/) || [])[1] || '0', 10))
            .filter(p => p > 0);
        if (tagPages.length === 0) return;
        const maxTagPage = Math.max(...tagPages);
        const questionPositionRatio = qIdx / Math.max(totalQCount - 1, 1);
        // Later-half question (position > 60%) with all tags on page 1 or 2 = suspicious
        if (questionPositionRatio > 0.6 && maxTagPage <= 2) {
            console.warn(`[SanityCheck] Q${q.questionNumber} (position ${qIdx}/${totalQCount}) has all tags on pages 1-${maxTagPage} — suspected student mislabel → requiresReview`);
            q._suspectedMislabel = true;
        }
    });
    // ── END MISLABEL SANITY CHECK ─────────────────────────────────────────────

    return { mappings: formattedMappings, orphanTags, unresolvedBoundaries };
}

async function fetchFilesFromStorage(filePaths) {
  const bucket = storage.bucket();
  const parts = [];

  for (const path of filePaths) {
    const file = bucket.file(path);
    const [buffer] = await file.download();

    parts.push({
      inlineData: {
        mimeType: path.endsWith('.pdf')
          ? 'application/pdf'
          : 'image/jpeg',
        data: buffer.toString('base64')
      }
    });
  }

  return parts;
}

/**
 * Gap-span positional assignment.
 * For each unmapped question, finds the character-range gap in the transcript
 * that positionally falls between the questions before and after it (by question number).
 * Assigns that gap text as the question's content. Zero LLM cost.
 * Only fires for questions that got NO tags from the deterministic resolver.
 */
function gapSpanPositionalAssignment(fullTranscript, deterministicMappings, unmappedQuestions, masterIds, hasSections = false) {
if (unmappedQuestions.length === 0) return { mappings: [] };
    // Skip gap-span for sectioned papers — students answer out of order,
    // positional gap assignment would map wrong content to wrong questions.
    if (typeof hasSections !== 'undefined' && hasSections) {
        console.log('[GapSpan] Skipping — sectioned paper, out-of-order answers expected');
        return { mappings: [] };
    }

    // Build QLABEL position index: masterId → char position of its QLABEL in transcript
    const qlabelPositions = [];
    const qlRe = /\[QLABEL:([^\]]+)\]/g;
    let qm;
    while ((qm = qlRe.exec(fullTranscript)) !== null) {
        const raw = (qm[1] || '').trim();
        const normLabel = normalizeLabelForMatch(raw, masterIds);
        if (!normLabel || !/\d/.test(normLabel)) continue;
        qlabelPositions.push({ raw, normLabel, pos: qm.index, end: qm.index + qm[0].length });
    }

    // Map each deterministically-resolved question to its QLABEL char position
    const resolvedPositions = [];
    deterministicMappings.forEach(m => {
        const normId = normalizeForComparison(m.id);
        // Find the QLABEL position whose normLabel matches this masterId
        const ql = qlabelPositions.find(q => normalizeForComparison(q.normLabel) === normId
            || normalizeForComparison(q.raw) === normId);
        resolvedPositions.push({
            id: m.id,
            normId,
            pos: ql ? ql.pos : -1,
            num: parseInt((normId.match(/(\d+)/) || [])[1] || '0')
        });
    });
    resolvedPositions.sort((a, b) => a.pos - b.pos);

    const mappings = [];

    unmappedQuestions.forEach(uq => {
        const uqNum = parseInt((normalizeForComparison(uq.questionNumber).match(/(\d+)/) || [])[1] || '0');
        if (!uqNum) return;

        // Find the resolved questions immediately before and after this unmapped question by number
        const before = resolvedPositions.filter(r => r.num < uqNum && r.pos >= 0)
            .sort((a, b) => b.num - a.num)[0];
        const after = resolvedPositions.filter(r => r.num > uqNum && r.pos >= 0)
            .sort((a, b) => a.num - b.num)[0];

        const gapStart = before ? before.pos + 1 : 0;
        const gapEnd = after ? after.pos : fullTranscript.length;

        if (gapStart >= gapEnd) {
            console.log(`[GapSpan] No gap found for Q${uq.questionNumber} (start=${gapStart} >= end=${gapEnd})`);
            return;
        }

        const gapText = fullTranscript.substring(gapStart, gapEnd).trim();
        if (gapText.length < 5) {
            console.log(`[GapSpan] Gap too short for Q${uq.questionNumber}: ${gapText.length} chars`);
            return;
        }

        // Extract [#P:] tags from the gap for pageMap
        const tags = [...gapText.matchAll(/\[#P:\d+,\d+,\d+\]/g)].map(m => m[0]);

        console.log(`[GapSpan] Q${uq.questionNumber}: gap chars ${gapStart}-${gapEnd}, ${tags.length} tags, ${gapText.length} chars`);
        mappings.push({ id: uq.questionNumber, tags, _gapText: gapText });
    });

    return { mappings };
}

/**
 * LLM Orphan Rescue — fires ONLY when deterministic resolution leaves gaps.
 * Uses gemini-2.0-flash (cheapest model).
 * Sends ONLY the orphan lines + candidate question IDs. NOT the full transcript.
 * Typical prompt: ~500 tokens vs the 50,000-token full librarian call.
 */
async function librarianOrphanRescue(orphanTags, unresolvedBoundaries, fullTranscript, questions, unmappedQuestions = [], resolvedContext = []) {
    if (orphanTags.length === 0 && unresolvedBoundaries.length === 0 && unmappedQuestions.length === 0) {
        return { mappings: [] };
    }

    const lines = fullTranscript.split('\n');

    // Collect only the lines that contain orphan tags
    const orphanLineSet = new Set();
    orphanTags.forEach(tag => {
        const idx = lines.findIndex(l => l.includes(tag));
        if (idx !== -1) orphanLineSet.add(idx);
    });

    // For unresolved boundary labels, include ±2 surrounding lines for context
    unresolvedBoundaries.forEach(b => {
        for (let i = Math.max(0, b.lineIndex - 1); i <= Math.min(lines.length - 1, b.lineIndex + 3); i++) {
            orphanLineSet.add(i);
        }
    });



// When unmapped questions exist, include full transcript BUT exclude lines
    // already cleanly assigned (lines that have no orphan/unresolved tags).
    // Sending the full 9-page transcript to 2.0 Flash overwhelms it.
    // Instead: include orphan lines + ±5 surrounding lines for context.
    // This gives enough context for semantic matching without the noise.
    if (unmappedQuestions.length > 0) {
        // Add ±5 context lines around every orphan line for unmapped question rescue
        orphanTags.forEach(tag => {
            const idx = lines.findIndex(l => l.includes(tag));
            if (idx !== -1) {
                for (let i = Math.max(0, idx - 5); i <= Math.min(lines.length - 1, idx + 5); i++) {
                    orphanLineSet.add(i);
                }
            }
        });
        // Also add all lines that have [#P:] tags not already in resolved boundaries
        lines.forEach((line, idx) => {
            if (line.includes('[#P:')) orphanLineSet.add(idx);
        });
    }
    const transcriptForRescue = [...orphanLineSet].sort((a, b) => a - b).map(i => lines[i]).join('\n');
    const miniTranscript = transcriptForRescue;

    // Build candidate list: only questions whose family number appears in unresolved labels
const candidateQids = questions
        .filter(q => {
            const qFamily = (q.questionNumber.match(/\d+/) || [])[0];
            if (!qFamily) return false;
            const inUnresolved = unresolvedBoundaries.some(b =>
                (b.normLabel && b.normLabel.includes(qFamily)) ||
                (b.inheritedFamily && b.inheritedFamily === qFamily)
            );
            // Only use orphanTags presence as broadener when unmappedQuestions is ALSO set.
            // Without this guard, ANY orphan tag causes ALL questions to become candidates —
            // flooding the rescue LLM with 16+ IDs on a full transcript → unreliable output.
            const hasOrphanTags = orphanTags.length > 0 && unmappedQuestions.length > 0;
            return inUnresolved || hasOrphanTags;
        })
        .map(q => ({
            id:      q.questionNumber,
            anchors: (q.topicAnchors || []).slice(0, 3),
           hint:    (q.text || '').substring(0, 150)
        }));

    // If no candidates can be narrowed down, include ALL unmapped questions
    const assignedIds = new Set();
    // (We can't know mappings here, so just use all questions as fallback)
    const finalCandidates = candidateQids.length > 0 ? candidateQids :
        questions.map(q => ({
            id:      q.questionNumber,
            anchors: (q.topicAnchors || []).slice(0, 2),
            hint:    (q.text || '').substring(0, 40)
        }));

    if (finalCandidates.length === 0) return { mappings: [] };

const model = vertex_ai.getGenerativeModel({
        model: 'gemini-2.0-flash',
        generationConfig: {
            temperature: 0,
            responseMimeType: 'application/json',
            // Thinking budget needed for multi-question semantic matching.
            // Without it, 2.0 Flash hallucinates or skips questions in complex transcripts.
            // 512 tokens is sufficient for rescue scope (orphan lines only, not full transcript).
            thinkingConfig: { thinkingBudget: 512 }
        }
    });

       const prompt = `You are a document router. Assign orphan handwriting blocks to question IDs.
VALID QUESTION IDs (only assign to these):
${JSON.stringify(finalCandidates, null, 2)}

ALREADY CONFIDENTLY ASSIGNED (do NOT reassign these, they are settled):
${JSON.stringify(resolvedContext.filter(r => r.confidence === 'high').map(r => r.id), null, 2)}

ORPHAN LINES (each line has a [#P:...] tag — assign each tag to a question ID):
${miniTranscript}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPECIAL MCQ RULE (CRITICAL):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If a line starts with a pattern like:
  • "1) c"  (number, closing paren, space, letter)
  • "2) b"  (number, closing paren, space, letter)
  • "3) d"  (number, closing paren, space, letter)

Then the NUMBER (1, 2, 3, etc.) is the QUESTION ID.
The LETTER (a, b, c, d) is the student's ANSWER CHOICE for that MCQ.

Map that [#P:] tag to that number as the question ID.

Example:
  Line: "1) c work done in moving a test charge... [#P:10,120,900]"
  → Map [#P:10,120,900] to question ID "1"

  Line: "2) c either 0V or +2V [#P:10,210,400]"
  → Map [#P:10,210,400] to question ID "2"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Each [#P] tag must be assigned to exactly ONE question ID from the valid list above.
2. Use topic anchors and hint text as semantic clues to identify which question the text belongs to.
3. If a label like "(b)" or "(ii)" appears, it is a sub-part — match it to the correct parent question's sub-part ID.
4. If a line starts with a number followed by ")" and a letter (e.g., "1) c"), ALWAYS map to that number.
5. If truly uncertain about a tag, do NOT assign it (skip it).

Return ONLY valid JSON:
{ "mappings": [{ "id": "question_id_exactly_as_listed", "tags": ["[#P:p,y,x]"] }] }`;

    try {
        const result = await callGeminiWithRetry(model, {
            contents: [{ role: 'user', parts: [{ text: prompt }] }]
        });
        const raw = result.response.candidates[0].content.parts[0].text;
        return extractJsonFromString(raw) || { mappings: [] };
    } catch (e) {
        console.warn('[LibrarianRescue] LLM rescue failed, continuing without it:', e.message);
        return { mappings: [] };
    }
}


/**
 * Detects actual PDF page count from base64 data without any library.
 * Reads the /Count field from the PDF cross-reference table.
 * Cost: zero. Pure string parsing.
 */
function detectPdfPageCount(base64Data) {
    try {
        // Convert first 8KB of base64 to string — enough to find page count
        const sample = Buffer.from(base64Data.slice(0, 10000), 'base64').toString('latin1');
        // PDF stores page count as "/Count N" in the Pages dictionary
        const match = sample.match(/\/Count\s+(\d+)/);
        if (match) {
            const count = parseInt(match[1], 10);
            console.log(`[PDF AutoDetect] Detected ${count} pages from PDF header`);
            return count;
        }
    } catch (e) {
        console.warn('[PDF AutoDetect] Failed, using safe default:', e.message);
    }
    // Safe default: 20 (not 50) — reduces worst-case cost by 60%
    return 20;
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
   { document: "gradingQueue/{jobId}", timeoutSeconds: 540, memory: "2GiB", region: "us-central1", concurrency: 1 },
    async (event) => {
        const snapshot = event.data;
        if (!snapshot) return;
        const jobId = event.params.jobId;
        const jobData = snapshot.data();
const {
            teacherUid, studentUid, assessmentId, stream,
            filePaths, answerSheetImageUrls,
            questions, strictness, subject, totalMarks, board
        } = jobData;
        const hasSections = jobData.hasSections || false;




        // Assign internal UIDs for collision-safe result matching (UNCHANGED)
        questions.forEach((q, idx) => {
            q._uid = `uid_${idx}_${Date.now()}`;
        });

const allRules = await fetchGradingRules(subject);
        const bucket = storage.bucket();

// ── SERIAL PROCESSING PER INSTANCE ──────────────────────────────────────
        // concurrency:1 on the trigger guarantees only one job runs per instance
        // at a time. Module-level vertex_ai (line 17) is now safe to reuse across
        // jobs on the same warm instance — no shared-state race possible.
        // 3s cooldown — ensures previous job's QPM usage clears before this job starts.
        await sleep(3000);
        // ────────────────────────────────────────────────────────────────────────

        try {
// REPLACE the current image fetch block (lines 4192-4215) with this:

let imageParts;
let hasSinglePdf = false;
const urls = answerSheetImageUrls || [];
const paths = filePaths || [];

console.log(`[Job ${jobId}] filePaths=${paths.length} urls=${urls.length} source=${jobData.source}`);

if (jobData.source === 'API' && urls.length > 0) {
    // SaaS external path — unchanged
    const rawParts = await fetchExternalImagesSecurely(urls);
    imageParts = [];
    for (const part of rawParts) {
        if (part._isPdf) {
            const pageCount = Number(jobData.pdfPageCount) || await detectPdfPageCount(part.inlineData.data);
            const cleanPdfPart = { inlineData: part.inlineData };
            for (let pg = 0; pg < pageCount; pg++) {
                imageParts.push(pg === 0 ? cleanPdfPart : { _pdfPagePlaceholder: true });
            }
        } else {
            imageParts.push(part);
        }
    }

} else if (paths.length > 0) {
    // Exam path: real GCS paths → Admin SDK (fast, no expiry)
    hasSinglePdf = paths.length === 1 && paths[0].endsWith('.pdf');
    if (hasSinglePdf) {
        const [pdfBuffer] = await bucket.file(paths[0]).download();
        const base64 = pdfBuffer.toString('base64');
        const pageCount = Number(jobData.pdfPageCount) || detectPdfPageCount(base64);
        const pdfPart = { inlineData: { mimeType: 'application/pdf', data: base64 } };
        imageParts = Array.from({ length: pageCount }, (_, pg) =>
            pg === 0 ? pdfPart : { _pdfPagePlaceholder: true }
        );
    } else {
        imageParts = await Promise.all(paths.map(async (p) => {
            const [buf] = await bucket.file(p).download();
            const ext = p.split('.').pop()?.toLowerCase() || 'jpeg';
            return { inlineData: { mimeType: ext === 'png' ? 'image/png' : 'image/jpeg', data: buf.toString('base64') } };
        }));
    }

} else if (urls.length > 0) {
    // ★ HOMEWORK PATH: no filePaths, but has Firebase Storage URLs
    // Use Admin SDK to read directly from GCS — same as exam path, no token expiry
    console.log(`[Job ${jobId}] Homework job — reading ${urls.length} images via Admin SDK (bypasses token expiry)`);
    imageParts = await Promise.all(urls.map(async (url) => {
        // Extract GCS path from Firebase Storage URL
        let filePath = null;
        if (url.includes('firebasestorage.googleapis.com')) {
            const m = url.match(/\/o\/([^?]+)/);
            if (m) filePath = decodeURIComponent(m[1]);
        } else if (url.includes('storage.googleapis.com')) {
            const m = url.match(/storage\.googleapis\.com\/[^/]+\/(.+)/);
            if (m) filePath = decodeURIComponent(m[1]);
        }
        if (!filePath) throw new Error(`FETCH_FAIL: Cannot parse GCS path from URL: ${url.substring(0, 80)}`);

        const [buf] = await bucket.file(filePath).download();
        const ext = filePath.split('.').pop()?.toLowerCase() || 'jpeg';
        return { inlineData: { mimeType: ext === 'png' ? 'image/png' : 'image/jpeg', data: buf.toString('base64') } };
    }));

} else {
    throw new Error("NO_IMAGES: No filePaths or answerSheetImageUrls provided.");
}

if (!imageParts || imageParts.length === 0) {
    throw new Error("NO_IMAGES: Image fetch returned empty array.");
}

            if (imageParts.length === 0) {
                throw new Error("NO_IMAGES: No answer sheet images found for this job.");
            }

            // ─── OCR (UNCHANGED) ─────────────────────────────────────────────────────
const pagesResult = await extractTextFromImages(imageParts, subject, jobId, jobData, allRules, hasSections);
const masterIds = questions.map(q => q.questionNumber);
const fullTranscript = sanitizeQLabelTranscript(pagesResult.map(p => `[PAGE ${p.pageNum}]\n${p.text}`).join('\n\n'), masterIds);

            const fullTranscriptClean = suppressOrphanQLabels(fullTranscript);


            // ── SILENT OCR FAILURE DETECTION ──────────────────────────────────────────
            // If OCR ran but returned empty for every page, the downstream librarian and
            // grader will silently give 0 on everything — no error, no warning to teacher.
            // Cause: rotated pages, very dark photos, Gemini refusing low-quality images,
            // or OCR prompt mismatch. We detect this and fail fast with a clear message.
            const nonEmptyPages = pagesResult.filter(p =>
                p.text && p.text.trim().length > 10 && p.text !== '[NO HANDWRITING DETECTED]'
            );
            const ocrEmptyRatio = 1 - (nonEmptyPages.length / Math.max(pagesResult.length, 1));
            if (ocrEmptyRatio > 0.8) {
                // >80% of pages returned empty — this is a systemic OCR failure, not
                // a student who left pages blank.
                const emptyCount = pagesResult.length - nonEmptyPages.length;
                console.error(`[OCR] SILENT FAILURE DETECTED: ${emptyCount}/${pagesResult.length} pages returned empty transcripts. Aborting grading.`);
                await db.collection('gradingQueue').doc(jobId).update({
                    status: 'ERROR',
                    statusDetails: `OCR could not read ${emptyCount} of ${pagesResult.length} pages. Possible causes: pages scanned upside down, very dark/blurry photos, or poor lighting. Please re-scan and resubmit.`,
                    errorCode: 'OCR_EMPTY_FAILURE',
                    progress: 0
                });
                return; // abort — do not continue to librarian/grader
            }
            // ─────────────────────────────────────────────────────────────────────────

            await db.collection('gradingQueue').doc(jobId).update({
                statusDetails: `Structuring answers (AI Boundary Marking)...`,
                currentStep: 2,
                progress: 35
            });

            // ─── LIBRARIAN: Deterministic-First + LLM Rescue + LLM Fallback ─────────
            // STAGE 1: Deterministic assignment — zero LLM cost.
            //   Reads [QLABEL:text] markers emitted by OCR and assigns every
            //   [#P:p,y,x] coordinate tag to a master question ID by position sweep.
            //   Works for: MCQ (one tag per label), maths (label at top, work below),
            //   prose (Ans 1(a) followed by orphan (b), (c)), deep nesting (1.(a).(i)).


let deterministicMappings = [];
let orphanTags = [];
let unresolvedBoundaries = [];

const result = deterministicBoundaryResolver(fullTranscript, masterIds, questions);

deterministicMappings = result.mappings;
orphanTags = result.orphanTags;
unresolvedBoundaries = result.unresolvedBoundaries;

            const deterministicCoverage = deterministicMappings.length;
            console.log(`[Librarian] Deterministic: ${deterministicCoverage}/${masterIds.length} questions mapped. Orphan tags: ${orphanTags.length}. Unresolved boundaries: ${unresolvedBoundaries.length}`);

            // STAGE 2: LLM Rescue — fires only when there are orphan tags or
            //   unresolved boundary labels. Uses gemini-2.0-flash with a
            //   tiny prompt (just orphan lines, not the full transcript).
  let rescueMappings = { mappings: [] };
            // Also fire rescue for questions deterministic completely missed (no QLABEL emitted at all)
            const deterministicMappedIds = new Set(deterministicMappings.map(m => normalizeForComparison(m.id)));
            const unmappedAfterDeterministic = questions.filter(q => !deterministicMappedIds.has(normalizeForComparison(q.questionNumber)));
if (orphanTags.length > 0 || unresolvedBoundaries.length > 0 || unmappedAfterDeterministic.length > 0) {
                console.log(`[Librarian] Escalating to rescue: ${orphanTags.length} orphan tags, ${unresolvedBoundaries.length} unresolved boundaries, ${unmappedAfterDeterministic.length} fully-missed questions`);

if (unmappedAfterDeterministic.length > 0) {
                    // BOUNDED EXCEPTION for sectioned papers: if an unmapped question
                    // sits DIRECTLY between two already-resolved questions with
                    // consecutive numbers, positional guess is safe even in a sectioned
                    // paper — there's no ambiguity about which gap it is.
                    const resolvedNums = new Set(
                        deterministicMappings.map(m => parseInt((normalizeForComparison(m.id).match(/(\d+)/) || [])[1] || '0', 10))
                    );
                    const boundedSafeList = hasSections
                        ? unmappedAfterDeterministic.filter(uq => {
                            const n = parseInt((normalizeForComparison(uq.questionNumber).match(/(\d+)/) || [])[1] || '0', 10);
                            return n > 0 && resolvedNums.has(n - 1) && resolvedNums.has(n + 1);
                          })
                        : unmappedAfterDeterministic;

const gapMappings = gapSpanPositionalAssignment(
                        fullTranscript, deterministicMappings,
                        hasSections ? boundedSafeList : unmappedAfterDeterministic,
                        masterIds,
                        hasSections ? false : hasSections
                    );
                    if (gapMappings.mappings.length > 0) {
                        console.log(`[GapSpan] Assigned ${gapMappings.mappings.length} questions via gap-span`);
                        gapMappings.mappings.forEach(gm => {
                            const existIdx = deterministicMappings.findIndex(m =>
                                normalizeForComparison(m.id) === normalizeForComparison(gm.id));
                            if (existIdx === -1) deterministicMappings.push(gm);
                        });
                        // Mark these as needing review
                        gapMappings.mappings.forEach(gm => {
                            const q = questions.find(q =>
                                normalizeForComparison(q.questionNumber) === normalizeForComparison(gm.id));
                            if (q) q._gapSpanAssigned = true;
                        });
                    }
                }

                // STAGE 2b: LLM rescue only for remaining orphan tags (misassigned, not missed)
                const stillUnmapped = questions.filter(q =>
                    !deterministicMappings.find(m =>
                        normalizeForComparison(m.id) === normalizeForComparison(q.questionNumber)));
if (orphanTags.length > 0 || unresolvedBoundaries.length > 0) {
                    const resolvedContext = deterministicMappings.map(m => ({
                        id: m.id,
                        confidence: 'medium'
                    }));
                    rescueMappings = await librarianOrphanRescue(
                        orphanTags, unresolvedBoundaries, fullTranscript, questions, stillUnmapped, resolvedContext
                    );
                }
            }

const coverageRatio = deterministicCoverage / Math.max(masterIds.length, 1);
const rescueThreshold = hasSections ? 0.5 : 0.7;
            let tagMapping;

            // UPSC/ESSAY PROSE FALLBACK:
            // Papers like UPSC, sociology essays have NO question labels written by student.
            // Coverage will be 0% because there are no [QLABEL] tags to detect.
            // In this case, assign pages sequentially to questions.
            // Fires when: coverage < 10% AND ≤ 5 master questions (essay/long-answer paper).
            if (coverageRatio < 0.1 && masterIds.length <= 5) {
                console.log('[Librarian] Possible essay/UPSC paper (0 labels, few questions) — using sequential page assignment');
                const sequentialMappings = [];
                pagesResult.forEach((page, pageIdx) => {
                    // Assign each page's tags to the corresponding question (or last question)
                    const questionIdx = Math.min(pageIdx, masterIds.length - 1);
                    const masterId = masterIds[questionIdx];
                    const tags = (page.text.match(/\[#P:\d+,\d+,\d+\]/g) || []);
                    if (tags.length > 0) {
                        const existingIdx = sequentialMappings.findIndex(m => m.id === masterId);
                        if (existingIdx !== -1) {
                            sequentialMappings[existingIdx].tags = [
                                ...new Set([...sequentialMappings[existingIdx].tags, ...tags])
                            ];
                        } else {
                            sequentialMappings.push({ id: masterId, tags });
                        }
                    }
                });
tagMapping = { mappings: sequentialMappings };
console.log(`[Librarian] Sequential assignment: ${sequentialMappings.length} questions mapped across ${pagesResult.length} pages`);
            }
else if (coverageRatio < rescueThreshold) {
                // Merge deterministic + rescue before deciding to fall back.
                // Rescue already ran above and may have recovered unmapped questions.
                // Throwing rescue away and running full LLM is wasteful and unreliable
                // on complex math/science papers. Only fall to full LLM if merged
                // coverage is still critically low (< 30%).
                const mergedForCoverage = [...deterministicMappings];
                (rescueMappings?.mappings || []).forEach(rm => {
                    const existingIdx = mergedForCoverage.findIndex(m =>
                        normalizeForComparison(m.id) === normalizeForComparison(rm.id)
                    );
                    if (existingIdx !== -1) {
                        mergedForCoverage[existingIdx].tags = [
                            ...new Set([...mergedForCoverage[existingIdx].tags, ...rm.tags])
                        ];
                    } else {
                        mergedForCoverage.push(rm);
                    }
                });
                const mergedCoverage = mergedForCoverage.length / Math.max(masterIds.length, 1);
                if (mergedCoverage < 0.3) {
                    console.warn(`[Librarian] Merged coverage still low (${Math.round(mergedCoverage * 100)}%). Falling back to full LLM librarian.`);
                    tagMapping = await librarianTagMapper(fullTranscript, questions, subject);
                } else {
                    console.log(`[Librarian] Rescue elevated coverage to ${Math.round(mergedCoverage * 100)}% — using merged result, skipping full LLM.`);
                    tagMapping = { mappings: mergedForCoverage };
                }
            } else {
                // Merge deterministic + rescue results
                const mergedMappings = [...deterministicMappings];
                (rescueMappings?.mappings || []).forEach(rm => {
                    const existingIdx = mergedMappings.findIndex(m =>
                        normalizeForComparison(m.id) === normalizeForComparison(rm.id)
                    );
                    if (existingIdx !== -1) {
                        // Merge tags, deduplicate
                        mergedMappings[existingIdx].tags = [
                            ...new Set([...mergedMappings[existingIdx].tags, ...rm.tags])
                        ];
                    } else {
                        mergedMappings.push(rm);
                    }
                });
                tagMapping = { mappings: mergedMappings };

                // ── FIX #5: SECOND-PASS ZERO-TAG RESCUE ──────────────────────────────
                // After merging deterministic + first rescue, find questions that still
                // have ZERO assigned tags. These are questions whose content was either:
                //   a) Wrongly assigned to another question (Bug #1: "6A" → "6.(a)"), or
                //   b) Genuinely not written by the student.
                // We can't distinguish (a) from (b) deterministically, so we send the
                // full transcript minus already-mapped lines to the rescue LLM with a
                // targeted candidate list. Cost: cheap — only fires when gaps exist.
                const mappedTagSet = new Set();
                tagMapping.mappings.forEach(m => m.tags.forEach(t => mappedTagSet.add(t)));

// Also rescue questions whose tags are all on wrong pages (stolen by stray QLABEL)
const zeroTagQuestions = questions.filter(q => {
    const key = normalizeForComparison(q.questionNumber);
    const existing = tagMapping.mappings.find(m => normalizeForComparison(m.id) === key);
    if (!existing || existing.tags.length === 0) return true;
    // Check if ALL tags are on page 1-2 but question is in later half of paper
    const qIdx = questions.indexOf(q);
    const positionRatio = qIdx / Math.max(questions.length - 1, 1);
    if (positionRatio > 0.5) {
        const tagPages = existing.tags.map(t => {
            const pm = t.match(/\[#P:(\d+),/);
            return pm ? parseInt(pm[1], 10) : 0;
        }).filter(p => p > 0);
        if (tagPages.length > 0 && Math.max(...tagPages) <= 2) return true;
    }
    return false;
});

                if (zeroTagQuestions.length > 0) {
                    console.log(`[Librarian] Second-pass rescue: ${zeroTagQuestions.length} questions have zero tags.`);

// Build a mini-transcript of UNMAPPED lines only.
                    // EXCEPTION: if zeroTagQuestions have zero tags (not stolen-page case),
                    // the relevant lines ARE mapped (stolen by prior question). Send full transcript.
                    const trueZeroTag = zeroTagQuestions.filter(q => {
                        const key = normalizeForComparison(q.questionNumber);
                        const existing = tagMapping.mappings.find(m => normalizeForComparison(m.id) === key);
                        return !existing || existing.tags.length === 0;
                    });
                    const allLines = fullTranscript.split('\n');
                    const unmappedLines = trueZeroTag.length > 0
                        ? allLines // full transcript — tags are stolen, unmapped filter misses them
                        : allLines.filter(line => {
                            const tagMatch = line.match(/\[#P:\d+,\d+,\d+\]/);
                            if (!tagMatch) return true;
                            return !mappedTagSet.has(tagMatch[0]);
                        });
                    const unmappedTranscript = unmappedLines.join('\n');

if (fullTranscript.trim().length > 20) {
                        // Targeted rescue: sends FULL transcript + explicit corrupted-label instructions.
                        // Old approach sent mini-transcript (unmapped lines only) — missed stolen tags.
                        // New approach: LLM sees everything, knows to look for corrupted Ans labels.
                        const zeroTagIds = zeroTagQuestions.map(q => ({
                            id: q.questionNumber,
                         anchors: (q.topicAnchors && q.topicAnchors.length > 0)
    ? q.topicAnchors.slice(0, 4)
    : [(q.text || '').substring(0, 120)],  // use more text when anchors missing
                            hint: (q.text || '').substring(0, 80)
                        }));

                        const rescueModel = vertex_ai.getGenerativeModel({
                            model: 'gemini-2.0-flash',
                            generationConfig: { temperature: 0, responseMimeType: 'application/json' }
                        });

                        const rescuePrompt = `You are a document librarian. These questions have NO student text assigned, but the student likely wrote answers — the answer labels may be misspelled or corrupted by OCR.

MISSING QUESTIONS (find answers for ONLY these):
${JSON.stringify(zeroTagIds, null, 2)}

FULL TRANSCRIPT:
${fullTranscript}

YOUR TASK:
Search the full transcript for any text that is the student's answer to each missing question.
The student's label may be corrupted — e.g. "Anes10" instead of "Ans 10", "Ans1O" instead of "Ans 10", "Ans I3" instead of "Ans 13", or the label may be entirely absent.
Use topicAnchors and hint to identify which block of text answers each question.
Assign ALL [#P:p,y,x] coordinate tags from that block to the question ID.

RULES:
1. Only assign tags to questions in the MISSING QUESTIONS list.
2. Each [#P] tag can only be assigned to ONE question.
3. If you cannot find an answer for a question, omit it — do not guess.
4. Look for answer text even when the label is absent or corrupted.

Return ONLY valid JSON:
{ "mappings": [{ "id": "question_id_exactly_as_listed", "tags": ["[#P:p,y,x]"] }] }`;

                        try {
                            const rescueResult = await callGeminiWithRetry(rescueModel, {
                                contents: [{ role: 'user', parts: [{ text: rescuePrompt }] }]
                            });
                            const rawRescue = rescueResult.response.candidates[0].content.parts[0].text;
                            const secondRescue = extractJsonFromString(rawRescue) || { mappings: [] };

                            (secondRescue?.mappings || []).forEach(rm => {
                                const existingIdx = tagMapping.mappings.findIndex(m =>
                                    normalizeForComparison(m.id) === normalizeForComparison(rm.id)
                                );
                                if (existingIdx !== -1) {
                                    tagMapping.mappings[existingIdx].tags = [
                                        ...new Set([...tagMapping.mappings[existingIdx].tags, ...rm.tags])
                                    ];
                                } else {
                                    tagMapping.mappings.push(rm);
                                }
                            });
                        } catch (rescueErr) {
                            console.warn(`[Librarian] Zero-tag rescue failed: ${rescueErr.message}`);
                        }
                    }
                }
                // ─────────────────────────────────────────────────────────────────────
            }   // ── RESCUE SANITY FILTER ─────────────────────────────────────────────
                // Problem: The rescue LLM sometimes assigns tags from page 1 (Q1-Q2 region)
                // to questions like Q8/Q9 that live on pages 6-9. This causes Q8/Q9 to
                // inherit Q2's text as their answer. 
                // Fix: For each question, check if ANY of its assigned tags fall on a page
                // that is far earlier than the question's expected page range.
                // If ALL tags are on page 1 but the question appears on page 6+, reject them.
                tagMapping.mappings.forEach(m => {
                    const qObj = questions.find(q => normalizeForComparison(q.questionNumber) === normalizeForComparison(m.id));
                    if (!qObj || !m.tags || m.tags.length === 0) return;

                    // Find what page this question's [QLABEL] was found on (from deterministic pass)
                    const detMapping = deterministicMappings.find(dm => normalizeForComparison(dm.id) === normalizeForComparison(m.id));
                    if (detMapping && detMapping.tags.length > 0) return; // deterministic got it right — don't second-guess

                    // Get pages of ALL assigned tags
                    const tagPages = m.tags.map(t => {
                        const pm = t.match(/\[#P:(\d+),/);
                        return pm ? parseInt(pm[1], 10) : 0;
                    }).filter(p => p > 0);

                    if (tagPages.length === 0) return;

                    const minTagPage = Math.min(...tagPages);
                    const maxTagPage = Math.max(...tagPages);

                    // Find the QLABEL for this question in the full transcript — its page is authoritative
                    const qNorm = normalizeForComparison(m.id);
                    const qlabelRx = /\[QLABEL:([^\]]+)\][\s\S]*?\[#P:(\d+),/g;
                    let qLabelPage = 0;
                    let qlm;
                    while ((qlm = qlabelRx.exec(fullTranscript)) !== null) {
                        if (normalizeForComparison(qlm[1]) === qNorm) {
                            qLabelPage = parseInt(qlm[2], 10);
                            break;
                        }
                    }

  let effectiveQLabelPage = qLabelPage;
if (effectiveQLabelPage === 0) {
    const qIndex = questions.findIndex(q => normalizeForComparison(q.questionNumber) === normalizeForComparison(m.id));
    if (qIndex > 0) {
        // Estimate: earlier questions have lower page numbers
        // Use the page of the nearest preceding question that DID get tags
        for (let qi = qIndex - 1; qi >= 0; qi--) {
            const prevQ = questions[qi];
            const prevMapping = tagMapping.mappings.find(pm => normalizeForComparison(pm.id) === normalizeForComparison(prevQ.questionNumber));
            if (prevMapping && prevMapping.tags.length > 0) {
                const prevPages = prevMapping.tags.map(t => {
                    const pm = t.match(/\[#P:(\d+),/);
                    return pm ? parseInt(pm[1], 10) : 0;
                }).filter(p => p > 0);
                if (prevPages.length > 0) {
                    effectiveQLabelPage = Math.max(...prevPages);
                    break;
                }
            }
        }
    }
}
if (effectiveQLabelPage > 0 && maxTagPage < effectiveQLabelPage) {
                        console.warn(`[SanityFilter] Rejecting rescue tags for ${m.id}: QLABEL on page ${qLabelPage} but tags on pages ${tagPages.join(',')} — likely wrong assignment`);
                        m.tags = []; // clear — question will fall back to "No specific text assigned"
                    }
                });

// AFTER — use _uid as key to survive duplicate question numbers:
const pageMap = new Map();
questions.forEach(q => pageMap.set(q._uid, new Set()));

// Step 1: assign mapping by normKey — SHARED across duplicate-number questions.
// Old code: first question "consumed" the entry, sibling got null.
// New code: all questions with same normKey get the SAME mapping entry.
// OR-winner logic (downstream) decides who keeps content, who gets wiped.
const questionMappings = new Map(); // _uid -> mapping entry
const mappingsByNormKey = new Map(); // normKey -> mapping entry (built once)
const mappingsArr = tagMapping && tagMapping.mappings ? tagMapping.mappings : [];
mappingsArr.forEach(m => {
    const key = normalizeForComparison(m.id);
    if (!mappingsByNormKey.has(key)) mappingsByNormKey.set(key, m);
});
questions.forEach(q => {
    const normKey = normalizeForComparison(q.questionNumber);
    questionMappings.set(q._uid, mappingsByNormKey.get(normKey) || null);
});

const atomicSlices = sliceByAtomicLines(fullTranscript, tagMapping);

// TABLE ANSWER FALLBACK: Questions whose answer is entirely inside a [TABLE] block
// have zero [#P:] tags — OCR doesn't emit coordinate tags inside table rows.
// For these, slice the text directly between their QLABEL and the next QLABEL.
questions.forEach(q => {
    const qKey = normalizeForComparison(q.questionNumber);
    const existing = tagMapping.mappings.find(m => normalizeForComparison(m.id) === qKey);
    if (existing && existing.tags.length > 0) return; // already has tags, skip
    if (atomicSlices[qKey] && atomicSlices[qKey].length > 10) return; // already sliced
    // Find this question's QLABEL in the transcript
    const qlRe = /\[QLABEL:([^\]]+)\]/g;
    let prevQL = null, myQL = null, nextQL = null;
    let m;
    while ((m = qlRe.exec(fullTranscript)) !== null) {
        const norm = normalizeForComparison(m[1]);
        if (norm === qKey && !myQL) { myQL = m; continue; }
        if (myQL) { nextQL = m; break; }
        if (!myQL) prevQL = m; // track last QLABEL before ours
    }
    if (!myQL) return; // no QLABEL found at all
    // FORWARD slice: from myQL to nextQL (normal case — TABLE after label)
    const fwdStart = myQL.index;
    const fwdEnd = nextQL ? nextQL.index : fullTranscript.length;
    const fwdSlice = fullTranscript.substring(fwdStart, fwdEnd).trim();
    if (fwdSlice.includes('[TABLE') && fwdSlice.length > 20) {
        atomicSlices[qKey] = fwdSlice;
        console.log(`[TableFallback] Q${q.questionNumber}: forward-sliced ${fwdSlice.length} chars (TABLE after label)`);
        return;
    }
    // BACKWARD slice: from prevQL (or page boundary) to myQL
    // Handles: OCR emits [TABLE]...[/TABLE] BEFORE the label in linearized output.
    // Example: [TABLE: wrap text | Anchoring ...][/TABLE] Ans-10 [QLABEL:Ans-10]
    const bwdStart = prevQL ? prevQL.index + prevQL[0].length : 0;
    const bwdEnd = myQL.index + myQL[0].length; // include the QLABEL itself
    const bwdSlice = fullTranscript.substring(bwdStart, bwdEnd).trim();
    if (bwdSlice.includes('[TABLE') && bwdSlice.length > 20) {
        // Combine: backward TABLE content + forward label text
        const combined = bwdSlice + '\n' + fwdSlice;
        atomicSlices[qKey] = combined;
        console.log(`[TableFallback] Q${q.questionNumber}: backward-sliced ${bwdSlice.length} chars (TABLE before label)`);
    }
});

// Helper: slice transcript for a specific set of tags.
// Uses the same position-based logic as sliceByAtomicLines above.
// Pre-build tag position map once (reused for all questions).
const _allTagPos = [];
{ const _tr = /\[#P:(\d+),(\d+),(\d+)\]/g; let _m;
  while ((_m = _tr.exec(fullTranscript)) !== null)
      _allTagPos.push({ tag: _m[0], pos: _m.index, end: _m.index + _m[0].length }); }
const _tagPosMap = new Map();
const _normTag = (s) => s.replace(/\s+/g, '').toLowerCase();
_allTagPos.forEach(t => { _tagPosMap.set(t.tag, t); _tagPosMap.set(_normTag(t.tag), t); });

function sliceFromTags(approvedTags, ownerQNum, masterIds, questionType) {
    if (!approvedTags || approvedTags.length === 0) return '';
    const positions = approvedTags
         .map(t => _tagPosMap.get(t) || _tagPosMap.get(_normTag(t))).filter(Boolean)

        .sort((a, b) => a.pos - b.pos);
    if (positions.length === 0) return '';

    const firstTagPos = positions[0].pos;
    const qlRx = /\[QLABEL:([^\]]+)\]/g;
    let lastQL = null; let qm;
    while ((qm = qlRx.exec(fullTranscript)) !== null) {
        if (qm.index < firstTagPos) lastQL = qm; else break;
    }
    // OWNERSHIP CHECK: only use this QLABEL if it belongs to our question.
    // If foreign (e.g. rescue LLM assigned a tag from Q2's region to Q9),
    // start from the first tag's position to avoid inheriting Q2's text.
const _textBeforeFirstTag = fullTranscript.substring(0, firstTagPos);
const _prevNLBeforeTag = _textBeforeFirstTag.lastIndexOf('\n');
let startPos = _prevNLBeforeTag >= 0 ? _prevNLBeforeTag + 1 : 0;
    if (lastQL) {
        const qlNorm = normalizeForComparison(lastQL[1]);
        const ownerNorm = normalizeForComparison(ownerQNum || '');

        // PRIMARY CHECK: exact match
        let qlabelBelongsToOwner = (qlNorm === ownerNorm);

        // SECONDARY CHECK: prefix/suffix tolerance
        if (!qlabelBelongsToOwner && lastQL[1]) {
  const qlNormLib = normalizeLabelForMatch(lastQL[1], masterIds);
            const ownerNormLib = normalizeLabelForMatch(ownerQNum || '', masterIds);
            
            // Check if qlNorm is a prefix of ownerNorm (e.g., "27" vs "27a")
            if (ownerNormLib.startsWith(qlNormLib) && qlNormLib.length >= 2) {
                qlabelBelongsToOwner = true;
            }
            // Check if ownerNorm is a prefix of qlNorm (e.g., "27a" vs "27")
            else if (qlNormLib.startsWith(ownerNormLib) && ownerNormLib.length >= 2) {
                qlabelBelongsToOwner = true;
            }
            // Check stripped roman suffix
            else {
                const stripped = qlNormLib.replace(/(i{1,4}|iv|vi{0,3}|ix)$/, '');
                if (stripped && stripped === ownerNormLib) {
                    qlabelBelongsToOwner = true;
                }
            }
        }

if (!qlabelBelongsToOwner && ownerQNum) {
            const distance = Math.abs((lastQL.index || 0) - firstTagPos);
            const qlHasNumber = /\d/.test(lastQL[1] || '');
            if (distance < 5000 && !qlHasNumber) {
                qlabelBelongsToOwner = true;
            }
        }

        // QUATERNARY CHECK: numeric root match.
        // Handles "Ans 1" (qlNorm="ans1" or "1") vs ownerQNum="Q1" or "1".
        // normalizeForComparison may strip "Q" prefix or "Ans" — leaving just
        // the digit. If both sides share the same digit root, it's our question.
        if (!qlabelBelongsToOwner) {
            const qlDigit  = (lastQL[1] || '').match(/(\d+)/)?.[1];
            const ownDigit = String(ownerQNum || '').match(/(\d+)/)?.[1];
            if (qlDigit && ownDigit && qlDigit === ownDigit) {
                qlabelBelongsToOwner = true;
            }
        }

        if (qlabelBelongsToOwner) {
            // The question number label appears BEFORE [QLABEL:N].
            // Search backwards for THIS question's label in many possible formats:
            //   "N) "  "N. "  "N : "  "Ans N :"  "Ans N."  "Ans. N"
const ownerDigits = String(ownerQNum || '').match(/(\d+)/);
            let foundStart = false;
            if (ownerDigits) {
                const qn = ownerDigits[1];
                const searchRegion = fullTranscript.substring(0, lastQL.index);
                // Try patterns from most specific to least — find LAST match
const patterns = [
new RegExp(`(?:^|[\\s\\n])Ans[-.]?0*${qn}(?:[^0-9]|$)`, 'gi'),
// bare number ONLY — require it's not preceded by a letter/digit (not inside formula)
new RegExp(`(?:^|[\\n])${qn}[).:\\s][\\s)]`, 'g'),
                ];
                for (const re of patterns) {
                    let qnMatch = null, m2;
                    while ((m2 = re.exec(searchRegion)) !== null) qnMatch = m2;
                    if (qnMatch) {
                        const raw = qnMatch.index + (qnMatch[0].match(/^\s|\n/) ? 1 : 0);
// REPLACE WITH:
            if (lastQL.index - raw < 5000) {
                            // Start from the student's "Ans N" / "N)" label — includes full answer
                            startPos = raw;
                            foundStart = true;
                            break;
                        }
                    }
                }
            }
if (!foundStart) {
                // Last resort: search for ANY "Ans N" before the QLABEL
                const fallbackRe = new RegExp(`(?:^|[\\s\\n])(Ans\\.?\\s*${String(ownerQNum||'').replace(/\D/g,'')})`, 'gi');
                let fallbackMatch = null, fm;
                const searchRegion2 = fullTranscript.substring(0, lastQL.index);
                while ((fm = fallbackRe.exec(searchRegion2)) !== null) fallbackMatch = fm;

                if (fallbackMatch) {
                    startPos = fallbackMatch.index + (fallbackMatch[0].match(/^\s|\n/) ? 1 : 0);
                } else {
                    const textBeforeQL = fullTranscript.substring(0, lastQL.index);
const prevNewline = textBeforeQL.lastIndexOf('\n');
startPos = prevNewline >= 0 ? prevNewline + 1 : 0;

                }
            }
        }
    }

const lastTagInfo = positions[positions.length - 1];
// SUBPOINT QLABEL SKIP:
// Student subpoints like "1)", "2)", "3)" emit [QLABEL:1], [QLABEL:2] etc.
// These must NOT be used as endPos — they are inside the current question's answer.
// Only stop at a QLABEL that belongs to a different master question.
// A QLABEL belongs to a different question if its normalized label matches
// a master ID that is NOT the current ownerQNum.
let nextQlPos = -1;
{
    const qlScanRx = /\[QLABEL:([^\]]+)\]/g;
    qlScanRx.lastIndex = lastTagInfo.end;
    let qlm;
    while ((qlm = qlScanRx.exec(fullTranscript)) !== null) {
const qlNorm = normalizeLabelForMatch(qlm[1], masterIds);
        const ownerNorm = normalizeLabelForMatch(ownerQNum || '', masterIds);
        if (qlNorm === ownerNorm) continue;
        const isMaster = masterIds.some(id => normalizeLabelForMatch(id, masterIds) === qlNorm);
        if (!isMaster) continue;
        // This QLABEL belongs to a real different master question — use it as endPos
        nextQlPos = qlm.index;
        break;
    }
}
let endPos = fullTranscript.length;
// Only use nextQlPos as endPos if it comes AFTER startPos.
if (nextQlPos !== -1 && nextQlPos > startPos) {
    endPos = Math.min(endPos, nextQlPos);
}

// MCQ DENSE BLOCK FIX:
// In dense MCQ pages all questions share one [#P] tag on one line.
// The transcript looks like: "[QLABEL:1] work is done... 2) c) [QLABEL:2] either OV..."
// The slice from [QLABEL:1] to [QLABEL:2] still contains "2) c)" text before [QLABEL:2].
// Fix: find the position of the NEXT question-number label (e.g. "2) " or "2. ")
// that appears BEFORE the next [QLABEL:] and use that as endPos instead.
// Only apply when ownerQNum is a simple integer (MCQ-style).
const _isMcqType = (questionType || '').toUpperCase() === 'MCQ';
if (ownerQNum && _isMcqType) {
const ownerDigits = String(ownerQNum).match(/(\d+)/);
    if (ownerDigits) {
        const nextNum = parseInt(ownerDigits[1], 10) + 1;
        const sliceRegion = fullTranscript.substring(startPos, endPos);
        // ONLY match bare "N) " or "N. " format (MCQ inline) — NOT "Ans N" (long-answer).
        const nmatch = sliceRegion.match(new RegExp(
            `(?:^|[\\s\\n])${nextNum}(?:[).][\\s)]|\\s*:)`, 'im'
        ));
        if (nmatch && nmatch.index !== undefined) {
            const candidateEnd = startPos + nmatch.index + (nmatch[0].match(/^[\s\n]/) ? 1 : 0);
            if (candidateEnd > startPos) {
                endPos = Math.min(endPos, candidateEnd);
            }
        }
    }
}

    return fullTranscript.substring(startPos, endPos).trim();
}

questions.forEach(q => {
    const normKey = normalizeForComparison(q.questionNumber);
    const ownMapping = questionMappings.get(q._uid);

if (ownMapping && ownMapping._gapText) {
        // Gap-span assigned: use the gap text directly, no tag sweep needed
        q.studentText = ownMapping._gapText;
        q.requiresReview = true;
    } else if (ownMapping && ownMapping.tags && ownMapping.tags.length > 0) {
        q.studentText = sliceFromTags(ownMapping.tags, q.questionNumber, masterIds, q.type);


// Strip Gemini OCR confidence markers (@@@word@@@) from stored student text
        // BUT first check if any exist — their presence = OCR was uncertain → flag for review
        if (q.studentText) {
          const hasOcrUncertainty = /@{1,3}[^@\n]*@{1,3}/.test(q.studentText)
    || /\[OCR_UNCERTAIN:[^\]]*\]/.test(q.studentText);
            if (hasOcrUncertainty) q._ocrUncertain = true;
            q.studentText = q.studentText.replace(/@{1,3}([^@\n]*)@{1,3}/g, '$1').replace(/(?<![a-zA-Z0-9])@(?![a-zA-Z0-9])/g, '').trim();
        }
    } else {
        q.studentText = atomicSlices[normKey] || "";
    }

    if (q.studentText) {
        const allTagMatches = [...q.studentText.matchAll(/\[#P:(\d+),\d+,\d+\]/g)];
        allTagMatches.forEach(m => {
            pageMap.get(q._uid).add(parseInt(m[1], 10));
        });
    }

    if (ownMapping && ownMapping.tags) {
        ownMapping.tags.forEach(t => {
            const match = t.match(/\[#P:(\d+),/);
            if (match) pageMap.get(q._uid).add(parseInt(match[1], 10));
        });
    }
});
            // ── SUBPART TEXT INHERITANCE ────────────────────────────────────────────
            // Problem: OCR emits ONE [QLABEL:Ans 1] for both Q1(i) and Q1(ii).
            // Deterministic resolver maps all [#P] tags to Q1(i) (first match).
            // Q1(ii) gets zero tags → studentText = "" → grader shows "No specific text".
            //
            // Fix: For each subpart with empty studentText, find its parent question
            // (same numeric prefix, e.g. "1" for "1i" and "1ii"). If the parent or
            // any sibling has non-empty studentText, copy the FULL parent-family text
            // to all empty siblings. The grader can then identify the relevant part.
            //
            // Safety: only fires when subpart has no studentText. Never overwrites.
            // ──────────────────────────────────────────────────────────────────────────
            (function inheritSubpartText(qs) {
                // Group questions by their numeric parent prefix (e.g. "1", "2", "3")
                const familyMap = new Map(); // parentPrefix -> [question, ...]
                qs.forEach(q => {
                    const digits = String(q.questionNumber || '').match(/(\d+)/);
                    if (!digits) return;
                    const prefix = digits[1];
                    if (!familyMap.has(prefix)) familyMap.set(prefix, []);
                    familyMap.get(prefix).push(q);
                });

                familyMap.forEach((members, prefix) => {
                    // Only act on families that have >1 member (i.e. actual subparts)
                    if (members.length < 2) return;

                    // Collect all non-empty studentText from this family
                    const richText = members
                        .map(m => m.studentText || '')
                        .filter(t => t.trim().length > 0)
                        .join('\n');

                    // If entire family empty, look for atomicSlices entry whose
                    // normalised key starts with the parent prefix (e.g. "ans1" for prefix "1").
                    // This fires when OCR emits [QLABEL:Ans 1] but masters are Q1(i)/Q1(ii).
                    let effectiveText = richText;
                    if (!effectiveText) {
                        const prefixNorm = normalizeForComparison(prefix); // e.g. "1"
                        for (const [sliceKey, sliceText] of Object.entries(atomicSlices)) {
                            if (sliceText && sliceText.trim().length > 0 &&
                                (sliceKey === prefixNorm ||
                                 sliceKey.endsWith(prefixNorm) ||
                                 sliceKey.replace(/[^0-9]/g, '') === prefixNorm)) {
                                effectiveText = sliceText;
                                console.log(`[SubpartInherit] Family ${prefix} all empty — using atomicSlices["${sliceKey}"] as fallback`);
                                break;
                            }
                        }
                    }

                    if (!effectiveText) return; // truly nothing to share

                    // Copy full family text to ALL members (so grader sees full context)
    members.forEach(m => {
        m.studentText = effectiveText;
        console.log(`[SubpartInherit] Set full family text for ${m.questionNumber}`);
    });
                });
            })(questions);
            // ─────────────────────────────────────────────────────────────────────────

            // ── SHARED DIAGRAM INHERITANCE ──────────────────────────────────────────
            // Problem: A student often draws ONE diagram that covers multiple sub-parts
            // (e.g. Q17 a) and Q17 b)). The trailing-untagged fix correctly attaches
            // the diagram to the first sub-part (Q17 a)). But Q17 b) — which also
            // expects a diagram per its imagePrompt — never sees it, and the grader
            // says "No diagram provided."
            //
            // Fix (post-slice, non-destructive):
            // For each question that:
            //   1. Has imagePrompt (expects a student-drawn diagram)
            //   2. Its own studentText has NO [DIAGRAM] block
            //   3. A preceding sibling question (same parent prefix, e.g. "17") HAS [DIAGRAM]
            // → Prepend that sibling's [DIAGRAM] block(s) to this question's studentText.
            //
            // "Parent prefix" = the shared numeric prefix before the sub-part letter.
            // Q17 a) norm="17a" → prefix="17"  |  Q17 b) norm="17b" → prefix="17" ✓
            // Q13 a) norm="13a" → prefix="13"  |  Q13 b) norm="13b" → prefix="13"
            //   (Q13 b) has no imagePrompt so it is not touched.)
            //
            // Safety: only fires when imagePrompt is set AND no own [DIAGRAM] exists.
            // Never removes content. Never modifies questions without imagePrompt.
            (function inheritSharedDiagrams(qs) {
                const diagramBlockRe = /\[DIAGRAM\][\s\S]*?\[\/DIAGRAM\]/g;

                function extractDiagramBlocks(text) {
                    const blocks = [];
                    let m;
                    diagramBlockRe.lastIndex = 0;
                    while ((m = diagramBlockRe.exec(text)) !== null) blocks.push(m[0]);
                    return blocks;
                }

                // Parent prefix: strip trailing single alpha char from normalised ID
                // "17a" → "17", "6ci" → "6c", "1ai" → "1a"
                function parentPrefix(normId) {
                    return normId.replace(/[a-z]$/, '');
                }

                qs.forEach((q, idx) => {
                    if (!q.imagePrompt) return;                        // doesn't need a diagram
                    if ((q.studentText || '').includes('[DIAGRAM]')) return; // already has one

                    const normQ  = normalizeForComparison(q.questionNumber);
                    const prefix = parentPrefix(normQ);
                    if (!prefix) return;                               // nothing to match against

                    // Walk backwards through preceding questions to find sibling with [DIAGRAM]
                    for (let i = idx - 1; i >= 0; i--) {
                        const sib     = qs[i];
                        const normSib = normalizeForComparison(sib.questionNumber);
                        if (!normSib.startsWith(prefix)) break;       // left the sibling family
                        const blocks = extractDiagramBlocks(sib.studentText || '');
                        if (blocks.length === 0) continue;

                        // Inherit: prepend sibling's diagram block(s) with a clear label
                        q.studentText = '[NOTE: Student drew a shared diagram in ' + sib.questionNumber + ' — reproduced here for grading]\n' + blocks.join('\n') + '\n' + (q.studentText || '');
                        q.studentText = q.studentText.trim();
                        console.log('[DiagramInherit] ' + q.questionNumber + ' inherited diagram from ' + sib.questionNumber);
                        break;
                    }
                });
            })(questions);

            // ── DIAGRAM INHERITANCE AUDIT ─────────────────────────────────────────────
// After ALL inheritance passes, log any question that still expects a diagram
// but has no [DIAGRAM] in its studentText. This surfaces lost diagrams early.
questions.forEach(q => {
    if (!q.imagePrompt) return;
    if ((q.studentText || '').includes('[DIAGRAM]')) return;
    console.warn(
        `[DiagramAudit] Q${q.questionNumber} expects a diagram (imagePrompt set) ` +
        `but studentText has NO [DIAGRAM] block. ` +
        `studentText length=${( q.studentText || '').length}. ` +
        `This question will likely be graded incorrectly.`
    );
});
// ── END DIAGRAM INHERITANCE AUDIT ────────────────────────────────────────
            // ── END SHARED DIAGRAM INHERITANCE ─────────────────────────────────────

            // ── SUB-PART PAGE INHERITANCE ────────────────────────────────────────────
            // Problem: Q29.(ii), Q29.(iii) share one [QLABEL:29.] with parent.
            // Only first sub-part gets [#P] tags. Others get empty pageMap → index -1.
            // Fix: inherit pages from nearest sibling with same parent number.
            questions.forEach(q => {
                const myPages = pageMap.get(q._uid);
                if (!myPages || myPages.size > 0) return; // already has pages — skip

                const parentMatch = q.questionNumber.match(/(\d+)/);
                if (!parentMatch) return;
                const parentNum = parentMatch[1];

                for (const sibling of questions) {
                    if (sibling._uid === q._uid) continue;
                    const sibMatch = sibling.questionNumber.match(/(\d+)/);
                    if (!sibMatch || sibMatch[1] !== parentNum) continue;
                    const sibPages = pageMap.get(sibling._uid);
                    if (sibPages && sibPages.size > 0) {
                        sibPages.forEach(p => myPages.add(p));
                        console.log(`[PageInherit] Q${q.questionNumber} inherited pages [${[...sibPages].join(',')}] from Q${sibling.questionNumber}`);
                        break;
                    }
                }
            });
// ── END SUB-PART PAGE INHERITANCE ───────────────────────────────────────

questions.forEach(q => {
                if (q.studentText && q.studentText.trim().length > 10) return;
                const parentMatch = q.questionNumber.match(/(\d+)/);
                if (!parentMatch) return;
                const parentNum = parentMatch[1];
                for (const sibling of questions) {
                    if (sibling._uid === q._uid) continue;
                    const sibMatch = sibling.questionNumber.match(/(\d+)/);
                    if (!sibMatch || sibMatch[1] !== parentNum) continue;
                    if (!sibling.studentText || sibling.studentText.trim().length <= 10) continue;
                    q.studentText = sibling.studentText;
                    console.log(`[TextInherit] Q${q.questionNumber} inherited text from Q${sibling.questionNumber}`);
                    break;
                }
            });
            // ── END SUB-PART TEXT INHERITANCE ────────────────────────────────────────

// OR-RESOLUTION happens inside the grader via OR-PAIR LAW (see GRADING_SYSTEM_INSTRUCTION).
            // Losing side hidden post-grading by computeOrLoserQNums() on the frontend.

            // ─── BATCH GRADING (UNCHANGED) ───────────────────────────────────────────
            const MAX_BATCH_WEIGHT = 16;
            const questionBatches = [];
            let currentBatch = [];
            let currentWeight = 0;

for (const q of questions) {
// Extra weight proportional to studentText size — prevents large DIAGRAM
                // blocks from inflating the prompt and truncating later questions' results.
                // Binary hasDiagram+4 is too aggressive on Biology/Science papers with
                // many small diagrams. Use text length instead: only large blocks get
                // extra weight. Short arrow diagrams (<500 chars) get no penalty.
                const studentTextLen = (q.studentText || '').length;
                const textWeight = studentTextLen > 2000 ? 4 : studentTextLen > 500 ? 2 : 0;
                let weight = (q.marks > 2 ? 4 : (q.marks === 2 ? 2 : 1)) + textWeight;
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
                if (bIdx > 0) await sleep(3500);
                const batch = questionBatches[bIdx];
                const batchProgress = Math.round(((bIdx + 1) / questionBatches.length) * 50);
                await snapshot.ref.update({
                    currentStep: 2,
                    progress: 25 + batchProgress,
                    statusDetails: `Grading Questions ${batch[0].questionNumber} to ${batch[batch.length - 1].questionNumber}`
                });

                // T1-2: Collect original page images for any diagram questions in this batch.
                // imagePrompt != null means the question expects a student-drawn diagram.
                // We find the page numbers those questions were answered on (from pageMap),
                // then pass those specific page images to the grader so it can visually verify
                // the diagram. Only fires when diagrams exist — zero cost for text-only batches.
                const diagramImageParts = [];
                const seenDiagramPages = new Set();
                for (const q of batch) {
                    if (!q.imagePrompt) continue; // not a diagram question, skip
      const pagesForQ = pageMap.get(q._uid) || new Set();
                    for (const pgNum of pagesForQ) {
                        if (seenDiagramPages.has(pgNum)) continue; // already added this page
                        seenDiagramPages.add(pgNum);
                        const pageImagePart = imageParts[pgNum - 1]; // imageParts is 0-indexed
                        if (pageImagePart && !pageImagePart._pdfPagePlaceholder) {
                            diagramImageParts.push(pageImagePart);
                        }
                    }
                }

// FIX 2: Build per-question allowed [#P] coordinate map from studentText.
// Each question's studentText contains ONLY the [#P] tags that belong to it
// (sliced by librarian). We extract those and store as the ground-truth
// allowed coords for Fix 3 clamping after grading.
const questionCoordBounds = new Map(); // q._uid -> { allowedCoords: [{page,y,x}], byPage: Map<page, {minY,maxY}> }
for (const q of batch) {
    const text = q.studentText || '';
    const coordRe = /\[#P:(\d+),(\d+),(\d+)\]/g;
    let m;
    const allowed = [];
    while ((m = coordRe.exec(text)) !== null) {
        allowed.push({ page: parseInt(m[1], 10), y: parseInt(m[2], 10), x: parseInt(m[3], 10) });
    }
    // Build per-page y-range for fast clamping
    const byPage = new Map();
    for (const c of allowed) {
        if (!byPage.has(c.page)) byPage.set(c.page, { minY: c.y, maxY: c.y });
        const r = byPage.get(c.page);
        if (c.y < r.minY) r.minY = c.y;
        if (c.y > r.maxY) r.maxY = c.y;
    }
    questionCoordBounds.set(q._uid, { allowed, byPage });
}

const slicedTranscript = batch.map(q => {
    const tags = (tagMapping[q._uid] || []);
    if (tags.length === 0) {
        // Fallback: use studentText assigned by librarian
        return `[Q:${q._uid}]\n${q.studentText || '(no text detected)'}`;
    }

    // STEP 1: Original logic — unchanged.
    const allLines = fullTranscriptClean.split('\n');
    const matchedIndices = new Set(
        allLines
            .map((line, i) => tags.some(tag => line.includes(tag)) ? i : -1)
            .filter(i => i !== -1)
    );

    // STEP 2: Block expansion — only for [TABLE] and [DIAGRAM] blocks.
    // Does nothing for prose, equations, theory — those have no [TABLE] marker.
    const expandedIndices = new Set(matchedIndices);
    let inBlock = false;
    let blockStart = -1;
    let blockCloseTag = '';

    for (let i = 0; i < allLines.length; i++) {
        const line = allLines[i];
        const trimmed = line.trimStart();

        if (!inBlock) {
            if (trimmed.startsWith('[TABLE:') || trimmed.startsWith('[DIAGRAM]')) {
                inBlock = true;
                blockStart = i;
                blockCloseTag = trimmed.startsWith('[TABLE:') ? '[/TABLE]' : '[/DIAGRAM]';
            }
        }

        if (inBlock) {
            if (matchedIndices.has(i)) {
                for (let j = blockStart; j <= i; j++) expandedIndices.add(j);
            }
            if (i === blockStart && (matchedIndices.has(i - 1) || matchedIndices.has(i - 2))) {
                expandedIndices.add(i);
            }
            if (expandedIndices.has(blockStart)) {
                expandedIndices.add(i);
            }
            if (line.includes(blockCloseTag)) {
                if (blockCloseTag === '[/TABLE]' && i + 1 < allLines.length &&
                    allLines[i + 1].trimStart().startsWith('ΣVALS:')) {
                    expandedIndices.add(i + 1);
                }
                inBlock = false;
                blockStart = -1;
                blockCloseTag = '';
            }
        }
    }

    const relevantLines = allLines
        .filter((_, i) => expandedIndices.has(i))
        .join('\n');

    return `[Q:${q._uid}]\n${relevantLines || q.studentText || '(no text detected)'}`;
}).join('\n\n---\n\n');

                // ── SHARED-SUBPART MCQ PRE-RESOLUTION ────────────────────────────────
                // Problem: MCQ subparts (e.g. Q1(i) and Q1(ii)) share the same inherited
                // text block. The deterministic letter extractor can't anchor reliably,
                // and the grader LLM gets confused because it sees the same text for both.
                // Solution: before grading, make ONE cheap gemini-2.0-flash call per
                // subpart family to extract which letter the student wrote for each subpart.
                // Result stamped on q._resolvedMcqLetter — used in MCQ override block.
                // ─────────────────────────────────────────────────────────────────────
                {
                    // Find MCQ subpart families where both siblings share the same text
                    const mcqSubpartFamilies = new Map(); // parentNum -> [q, ...]
                    for (const q of batch) {
                        if (q.type !== 'MCQ' && q.type !== 'AR' && q.type !== 'Assertion-Reason') continue;
                        const qNumRaw = String(q.questionNumber || '');
                        const subPartMatch = qNumRaw.match(/(\d+)[.\s]*(?:\(([ivxIVX]+)\)|([ivxIVX]+))/i);
                        if (!subPartMatch) continue;
                        const parentNum = subPartMatch[1];
                        const subPart = (subPartMatch[2] || subPartMatch[3] || '').toLowerCase();
                        if (!subPart) continue;
                        // Check if text looks like shared multi-subpart text
                        const rawOcr = q.studentText || '';
                        const looksShared = /ii[).\s]|iii[).\s]/i.test(rawOcr) && rawOcr.length > 40;
                        if (!looksShared) continue;
                        if (!mcqSubpartFamilies.has(parentNum)) mcqSubpartFamilies.set(parentNum, []);
                        mcqSubpartFamilies.get(parentNum).push({ q, subPart });
                    }

                    for (const [parentNum, members] of mcqSubpartFamilies) {
                        if (members.length < 2) continue;
                        const sharedText = members[0].q.studentText || '';
                        // Build prompt: list each subpart + its question + model answer
                        const subpartLines = members.map(({ q, subPart }) =>
                            `Subpart (${subPart}): Q: "${q.text || ''}" | Model answer: "${q.answer || ''}"`
                        ).join('\n');

                        const extractPrompt = `You are reading a student's handwritten answer sheet OCR output.
The student answered Question ${parentNum} which has multiple sub-parts.
The full answer text for Q${parentNum} is:
"""
${sharedText.substring(0, 600)}
"""

Sub-parts to identify:
${subpartLines}

For each sub-part:
- If the model answer is an option letter (A/B/C/D), find which letter (a/b/c/d) the student wrote for that sub-part.
  Look for patterns: "i) (a)", "ii) (c)", "i) a)", "ii) b)"
- If the model answer is NOT a letter (e.g. "iii) fixed investment", "True", a phrase), extract the student's actual written answer for that sub-part.
  Look for "ii) [answer text]" — whatever the student wrote after the sub-part Roman numeral.

Roman numeral i = sub-part i, ii = sub-part ii, iii = sub-part iii, etc.

Respond ONLY with a JSON object, no other text:
{"results": [{"subPart": "i", "studentLetter": "A", "studentText": null}, {"subPart": "ii", "studentLetter": null, "studentText": "iii) fixed investments"}]}
Use null for fields that do not apply. If you cannot determine anything for a sub-part, set both to null.`;

                        try {
                            const miniModel = vertex_ai.getGenerativeModel({ model: 'gemini-2.0-flash' });
                            const miniResult = await miniModel.generateContent({
                                contents: [{ role: 'user', parts: [{ text: extractPrompt }] }],
                                generationConfig: { temperature: 0.0, maxOutputTokens: 256 }
                            });
                            const rawText = miniResult.response.candidates[0].content.parts[0].text || '';
                            const parsed = extractJsonFromString(rawText);
                            if (parsed && Array.isArray(parsed.results)) {
                                for (const r of parsed.results) {
                                    if (!r.subPart) continue;
                                    const match = members.find(m => m.subPart === r.subPart.toLowerCase());
                                    if (!match) continue;
                                    if (r.studentLetter) {
                                        match.q._resolvedMcqLetter = r.studentLetter.toUpperCase();
                                        console.log(`[SharedMCQ] Q${match.q.questionNumber} subPart=${r.subPart} resolvedLetter=${match.q._resolvedMcqLetter}`);
                                    } else if (r.studentText) {
                                        // Non-letter answer (e.g. "iii) fixed investment")
                                        // Replace studentText with just this subpart's answer
                                        // so grader LLM is not confused by shared text block
                                        match.q.studentText = r.studentText;
                                        match.q._resolvedSubpartText = true;
                                        console.log(`[SharedMCQ] Q${match.q.questionNumber} subPart=${r.subPart} resolvedText="${r.studentText}"`);
                                    }
                                }
                            }
                        } catch (miniErr) {
                            console.warn(`[SharedMCQ] Mini-resolve failed for Q${parentNum}:`, miniErr.message);
                            // Non-fatal — falls through to existing shared-text guard
                        }
                    }
                }
                // ── END SHARED-SUBPART MCQ PRE-RESOLUTION ────────────────────────────

            const batchResult = await gradeQuestionBatch(
                    slicedTranscript,
                    batch, strictness, subject, jobId, allRules, diagramImageParts, jobData, questions
                );
questionWiseReport.push(...batchResult.map(qr => {
                    const usedYByPage = new Map(); // page -> Set of y already assigned (dedup tracker)
                    // Find this question's known pages from pageMap (built by librarian)
                    const origQ = batch.find(q => q._uid === qr.questionNumber || q._uid === qr._uid);
                    const knownPages = origQ ? Array.from(pageMap.get(origQ._uid) || new Set()) : [];
                    // Use last known page as fallback (not page 1 — long answers end on later pages)
                    const fallbackPage = knownPages.length > 0 ? Math.max(...knownPages) : null;

                    // FIX 2+3: Retrieve the allowed coord bounds for this question
                    const coordBounds = origQ ? (questionCoordBounds.get(origQ._uid) || null) : null;

                    return {
                        ...qr,
                        stepWiseEvaluation: (qr.stepWiseEvaluation || []).map(step => {
                            const rawPage = step.pageIndex;
                            // Use AI's pageIndex if valid (>=1), else last known page, else null
                            const resolvedPage = (rawPage && rawPage >= 1)
                                ? rawPage
                                : (fallbackPage !== null ? fallbackPage : null);

                            // Sanitize stepPoint format (existing logic)
                            let sp = (Array.isArray(step.stepPoint) && step.stepPoint.length >= 2)
                                ? [Number(step.stepPoint[step.stepPoint.length - 2]), Number(step.stepPoint[step.stepPoint.length - 1])]
                                : null;

if (sp && coordBounds && coordBounds.byPage.size > 0) {
                                const targetPage = resolvedPage;
                                const pageRange = coordBounds.byPage.get(targetPage);
                                if (!pageRange && coordBounds.byPage.size > 0) {
                                    // AI put marker on wrong page — fix page only, keep y as-is
                                    let bestPage = null, bestDist = Infinity;
                                    coordBounds.byPage.forEach((r, pg) => {
                                        const d = Math.abs(pg - (targetPage || 1));
                                        if (d < bestDist) { bestDist = d; bestPage = pg; }
                                    });
                                    if (bestPage !== null) {
                                        console.log(`[CoordFix] Q${qr.questionNumber} page mismatch — corrected to page ${bestPage}, y kept at ${sp[0]}`);
                                        // Only fix resolvedPage, not sp — y stays where grader put it
                                    }
                                }
                                // Y clamping removed — grader picks y from actual answer position;
                                // clamping to question label y-range causes markers to appear too high.
                            }



                            return {
                                ...step,
                                pageIndex: resolvedPage,
                                stepPoint: sp
                            };
                        })
                    };
                }));
            }

            // ─── T1-3: POST-GRADING CONSISTENCY ENFORCER (zero LLM cost) ────────────
            // Catches failure modes the grader produces inconsistently:
            //   A) CAT-9: feedback says error but marks = maxMarks (desync)
            //   B) CAT-5: feedback is non-trivial but full marks given
            //   C) CAT-4: requiresReview was false but feedback has negative signals
            //   D) NEW: marksAwarded is not a 0.5 multiple (e.g. 0.75, 1.33) — enforce rounding
            //   E) NEW: finalFeedback bullet deduction numbers don't match 0.5 increments

            // Deterministic 0.5 rounding helper
            function roundToHalf(val, maxMarks, strictnessMode) {
                const n = Number(val) || 0;
                if (strictnessMode === 'Strict') {
                    return Math.min(Math.floor(n * 2) / 2, maxMarks);
                } else if (strictnessMode === 'Lenient') {
                    return Math.min(Math.ceil(n * 2) / 2, maxMarks);
                }
                // Moderate (default): round to nearest 0.5
                return Math.min(Math.round(n * 2) / 2, maxMarks);
            }

            // Sanitize feedback: replace non-0.5 deduction values in bullet points
            // e.g. "(-0.75 marks)" → "(-1 marks)", "(-1.33 marks)" → "(-1.5 marks)"
            function sanitizeFeedbackDeductions(feedbackText, strictnessMode) {
                if (!feedbackText) return feedbackText;
                return feedbackText.replace(/([-−])\s*(\d+(?:\.\d+)?)\s*(marks?)/gi, (match, sign, numStr, marksWord) => {
                    const raw = parseFloat(numStr);
                    if (isNaN(raw)) return match;
                    let rounded;
                    if (strictnessMode === 'Strict') {
                        rounded = Math.ceil(raw * 2) / 2; // strict deductions round UP (more punitive)
                    } else if (strictnessMode === 'Lenient') {
                        rounded = Math.floor(raw * 2) / 2; // lenient deductions round DOWN (less punitive)
                    } else {
                        rounded = Math.round(raw * 2) / 2;
                    }
                    // Only change if it was NOT already a valid 0.5 multiple
                    if (Math.abs(rounded - raw) < 0.001) return match; // already valid
                    const display = rounded === Math.floor(rounded) ? rounded.toFixed(0) : rounded.toFixed(1);
                    return `${sign}${display} ${marksWord}`;
                });
            }

            const NEGATIVE_SIGNALS = [
                'incorrect', 'wrong', 'error', 'missing', 'incomplete', 'not provided',
                'not mentioned', 'does not', "doesn't", 'absent', 'failed', 'no diagram',
                'unattempted', 'not attempted', 'calculation error', 'conceptual error',
                'not matching', 'differs', 'mismatch'
            ];
questionWiseReport = questionWiseReport.map(qr => {
                const maxMarks = qr.maxMarksForQuestion || 0;
                let awarded    = qr.marksAwarded || 0;

                // OCR uncertainty flag — set by text extraction above
                const question = questions.find(q => String(q.questionNumber) === String(qr.questionNumber));
                if (question && question._ocrUncertain && !qr.requiresReview) {
                    const cleanedFbEarly = sanitizeFeedbackDeductions(qr.finalFeedback, strictness);
                    return { ...qr, marksAwarded: awarded, requiresReview: true,
                        finalFeedback: (cleanedFbEarly || '') + ' [OCR uncertain — please verify student handwriting.]' };
                }

                // D: Enforce 0.5 rounding on marksAwarded
                const roundedAwarded = roundToHalf(awarded, maxMarks, strictness);
                awarded = roundedAwarded;

                // E: Sanitize feedback deduction numbers to 0.5 multiples
                const cleanedFeedback = sanitizeFeedbackDeductions(qr.finalFeedback, strictness);

                const feedback = (cleanedFeedback || '').toLowerCase();
                const hasNegativeSignal = NEGATIVE_SIGNALS.some(sig => feedback.includes(sig));

const isMcqFormat = (qr.type === 'MCQ') || (qr.type === 'AR') ||
    (qr.type === 'Assertion-Reason') ||
    (qr.maxMarksForQuestion <= 1 && !!(qr.finalFeedback || '').match(/^[A-Da-d]\s*[-–]/)) ||
    !!(qr.finalFeedback || '').match(/^[A-Da-d]\s*[-–]/);



    if (qr.type === 'True/False') {
    const toTF = (s) => {
        const n = (s||'').toLowerCase().replace(/[^a-z]/g,'');
        if (n === 'true' || n === 't') return 'TRUE';
        if (n === 'false' || n === 'f') return 'FALSE';
        if (/\btrue\b/i.test(s||'')) return 'TRUE';
        if (/\bfalse\b/i.test(s||'')) return 'FALSE';
        return '';
    };
    const modelTF = toTF(qr.answer);
    const studentTF = toTF(qr.studentText || qr.studentOcrAnswer || '');
    if (modelTF && studentTF) {
        const isCorrect = modelTF === studentTF;
        const overrideMarks = isCorrect ? maxMarks : 0;
        const syncedSteps = (qr.stepWiseEvaluation||[]).map((s,i)=>({...s, marks: i===0 ? overrideMarks : 0}));
        return { ...qr, marksAwarded: overrideMarks,
            finalFeedback: isCorrect ? `${studentTF==='TRUE'?'True':'False'} - Good work.` : `${studentTF==='TRUE'?'True':'False'} - Incorrect. Correct answer: ${modelTF==='TRUE'?'True':'False'}.`,
            requiresReview: false, stepWiseEvaluation: syncedSteps };
    }
    return { ...qr, marksAwarded: awarded, finalFeedback: cleanedFeedback, requiresReview: true };
}



                if (isMcqFormat) {
                    // DETERMINISTIC LETTER OVERRIDE:
                    // Extract model answer letter — handle "(a) ...", "a)", "A", "(iii) fixed investments" etc.
                    const _modelRaw = (qr.answer || '').trim();
                    const _modelMatch = _modelRaw.match(/\b([A-Da-d])\b/) || _modelRaw.match(/\(?([A-Da-d])\)?/);
                    const modelLetter = _modelMatch ? _modelMatch[1].toUpperCase() : _modelRaw.charAt(0).toUpperCase();

                    // GUARD: if model answer has no valid A-D letter, this subpart is not
                    // a letter-choice MCQ (e.g. short-answer subpart under MCQ parent type).
                    // Skip deterministic path entirely — let LLM result stand as-is.
                    if (!/^[A-D]$/.test(modelLetter)) {
                        console.log('[MCQ Skip] Q' + qr.questionNumber + ': model answer "' + _modelRaw + '" has no A-D letter — skipping deterministic override');
                    } else {
const rawOcr = (qr.studentText || qr.studentOcrAnswer || '');

// [CORRECTED] = OCR marked a crossed-out-then-rewritten answer (see OCR CORRECTION
// MARKER law). The surviving answer is whatever the student wrote AFTER the tag.
// Without this, the deterministic extractor below grabs the FIRST letter it sees —
// i.e. the struck-out attempt — and wrongly scores the corrected MCQ as 0.
const _hasCorrectedTag = /\[CORRECTED\]/i.test(rawOcr);

let studentLetter = '';
let _extractedSubPart = '';
let _p0SucceededForSubpart = false;

// Pattern 0: find THIS question's answer letter from rawOcr.
// Handles ALL formats:
//   "1) c)"  "1. D"  "1 : (d)"
//   "Ans 1 : (d) D"  "Ans 1: d"  "Ans1 (d)"
//   "29. (i) d)"  "29.(ii) b)"
// Also handles questionNumber like "Q1(i)" where digits not at start.
{
    const qNumRaw = String(qr.questionNumber || '');
    const subPartMatch = qNumRaw.match(/(\d+)[.\s]*(?:\(([ivxIVX]+)\)|([ivxIVX]+))?/i);
    if (subPartMatch) {
        const mainNum = subPartMatch[1];
        const subPart = (subPartMatch[2] || subPartMatch[3] || '').toLowerCase();
        _extractedSubPart = subPart;
        let p0;
        if (subPart) {
            // Sub-part question: find mainNum then (subPart) then first A-D after
            const re = new RegExp(
                `(?:^|[\\s])(?:Ans\\.?\\s*)?${mainNum}[^(]*\\(${subPart}\\)\\s*[^A-Da-d\\n]*?([A-Da-d])(?:[)\\s]|$)`,
                'im'
            );
            p0 = rawOcr.match(re);
            // Fallback: just find (subPart) then first A-D
            if (!p0) {
                const re2 = new RegExp(`\\(${subPart}\\)\\s*([A-Da-d])(?:[)\\s]|$)`, 'im');
                p0 = rawOcr.match(re2);
            }
        } else {
            // Simple question — try multiple formats in priority order:
            // 1. "Ans N : (d)" or "Ans N: d" (most explicit)
            const ansRe = new RegExp(`(?:^|[\\s])Ans\\.?\\s*${mainNum}\\s*[:.]+\\s*\\(?([A-Da-d])\\)?`, 'im');
            p0 = rawOcr.match(ansRe);
            // 2. "N) d" or "N. d" or "N : d"
            if (!p0) {
p0 = rawOcr.match(new RegExp(`(?:^|[\\s\\[])${mainNum}[).:–-]+\\s*\\(?([A-Da-d])\\)?(?:[)\\s]|$)`, 'im'));
            }
        }
        if (p0) {
            studentLetter = p0[1].toUpperCase();
            if (subPart) _p0SucceededForSubpart = true;
        }
    }
}

// Pattern 1: sub-part then answer — "(i) d)" or "(ii) b) text"
if (!studentLetter) {
const p1 = rawOcr.match(/(?:\([ivxIVX]+\)|[ivxIVX]+[).]\s*)\s*\(?([A-Da-d])\)?/);
if (p1) studentLetter = p1[1].toUpperCase();
}

// Pattern 2: "Ans N: (c)" or "N: c"
if (!studentLetter) {
    const p2 = rawOcr.match(/(?:Ans\.?\s*\d+\s*[:.]\s*|^\s*\d+\s*[:.]\s*)\(?([A-Da-d])\)?/im);
    if (p2) studentLetter = p2[1].toUpperCase();
}

// Pattern 3: remove roman numeral words then find A-D
if (!studentLetter) {
    const cleaned = rawOcr
        .replace(/\b(iv|iii|ii|vi|vii|viii|ix|xi|xii|v|x)\b/gi, ' ')
        .replace(/\(i\)/gi, ' ');
    const p3 = cleaned.match(/(?:^|[\s()\[\].])([A-Da-d])(?:[\s()\[\].,)]|$)/);
    if (p3) studentLetter = p3[1].toUpperCase();
}

// Pattern 4: last resort — any isolated A-D
if (!studentLetter) {
    const p4 = rawOcr.match(/\b([A-Da-d])\b/);
    if (p4) studentLetter = p4[1].toUpperCase();
}

// [CORRECTED] OVERRIDE (highest authority): if the OCR tagged a rewritten answer,
// the surviving letter is the one AFTER the last [CORRECTED] marker — it WINS over
// whatever Patterns 0-4 picked (which may have grabbed the struck-out first letter).
if (_hasCorrectedTag) {
    const _tagIdx = rawOcr.toUpperCase().lastIndexOf('[CORRECTED]');
    const _afterTag = rawOcr.slice(_tagIdx + '[CORRECTED]'.length);
    const _pc = _afterTag.match(/\(?\[?([A-Da-d])\]?\)?(?:[)\].\s:,]|$)/);
    if (_pc) {
        studentLetter = _pc[1].toUpperCase();
        console.log(`[MCQ Corrected] Q${qr.questionNumber}: [CORRECTED] surviving letter="${studentLetter}"`);
    }
}

// Shared-text subpart guard:
// When both Q1(i) and Q1(ii) share the same inherited text block, the patterns
// above may extract the wrong letter (e.g. picks option "(i)" text as the answer).
// If this is a subpart question AND Pattern 0 didn't find a clean subpart-anchored
// letter AND the text contains multiple subpart markers → skip deterministic override,
// let the LLM result stand (it saw the full question + answer text and is smarter).
const _hasSharedMultipartText = _extractedSubPart &&
    !_p0SucceededForSubpart &&
    /ii[).\s]|iii[).\s]/i.test(rawOcr);

// If pre-resolution stamped a letter (from shared-subpart mini-LLM call), use it.
if (qr._resolvedMcqLetter) {
    studentLetter = qr._resolvedMcqLetter;
    console.log(`[MCQ] Q${qr.questionNumber}: using pre-resolved letter "${studentLetter}"`);
}



console.log(`[MCQ Extract] Q${qr.questionNumber}: student="${studentLetter}" model="${modelLetter}" subPart="${_extractedSubPart}" p0ok=${_p0SucceededForSubpart} sharedText=${_hasSharedMultipartText} rawOcr="${rawOcr.substring(0,80)}"`);

                    if (modelLetter && studentLetter && (!_hasSharedMultipartText || qr._resolvedMcqLetter)) {
                        const isCorrect = modelLetter === studentLetter;
                        const overrideMarks = isCorrect ? maxMarks : 0;
                        const overrideFeedback = isCorrect
                            ? `${studentLetter} - Good work.`
                            : `${studentLetter} - Incorrect. Correct answer: ${modelLetter}.`;
                        console.log(`[MCQ Override] Q${qr.questionNumber}: student="${studentLetter}" model="${modelLetter}" isCorrect=${isCorrect} overrideMarks=${overrideMarks} maxMarks=${maxMarks} rawOcr="${rawOcr.substring(0,60)}"`);
const syncedSteps = (qr.stepWiseEvaluation || []).map((step, i) => ({
                            ...step,
                            marks: i === 0 ? overrideMarks : 0,
                            comment: i === 0
                                ? (isCorrect ? '' : 'Incorrect')
                                : (step.comment || '')
                        }));
                        // Flag corrected answers for a teacher glance (per [CORRECTED] law),
                        // but keep the awarded marks — do not zero a correctly-read correction.
                        return { ...qr, marksAwarded: overrideMarks, finalFeedback: overrideFeedback, requiresReview: _hasCorrectedTag, stepWiseEvaluation: syncedSteps };
                    }

                    // FALLBACK: studentText was empty (dense MCQ block — Librarian didn't slice it)
                    // Scan the full transcript for patterns like:
                    //   "Ans 20: (c)", "Ans20 (c)", "20. c", "20) c", "20 : c"
const qNum = String(qr.questionNumber).replace(/[^0-9]/g, '').substring(0, 2);
                    let scannedLetter = '';

                    if (qNum && fullTranscript) {
                        // Patterns: "Ans N:", "N:", "N.", "N)" followed by optional space and letter
const scanPatterns = [
    // "29. (i) d)" — question number, sub-part, then answer
    new RegExp(`(?:^|\\n)\\s*(?:Ans\\.?\\s*)?${qNum}\\b[.\\s]*(?:\\([ivxIVX]+\\)|[ivxIVX]+[).]\\s*)\\(?([A-Da-d])\\)?`, 'im'),
    // "Ans 20: (c)" or "Ans20 (c)"
new RegExp(`Ans\\.?\\s*${qNum}\\b\\s*[:\\-.)]?\\s*\\(?([A-Da-d])\\)?`, 'i'),
    // "20: c" or "20. c" or "20) c"  
    new RegExp(`(?:^|\\n)\\s*${qNum}\\b\\s*[:\\-.)]+\\s*\\(?([A-Da-d])\\)?`, 'im'),
];
for (const pat of scanPatterns) {
                            const sm = fullTranscript.match(pat);
                            if (sm) { scannedLetter = sm[1].toUpperCase(); break; }
                        }

                        if (scannedLetter && modelLetter) {
                            const isCorrect = scannedLetter === modelLetter;
                            const overrideMarks = isCorrect ? maxMarks : 0;
                            console.log(`[MCQ Scan] Q${qr.questionNumber}: scanned="${scannedLetter}" model="${modelLetter}" isCorrect=${isCorrect}`);
                            const syncedSteps = (qr.stepWiseEvaluation || []).map((step, i) => ({
                                ...step,
                                marks: i === 0 ? overrideMarks : 0
                            }));
                            return { ...qr, marksAwarded: overrideMarks,
                                finalFeedback: isCorrect ? `${scannedLetter} - Good work.` : `${scannedLetter} - Incorrect. Correct answer: ${modelLetter}.`,
                               requiresReview: !isCorrect, stepWiseEvaluation: syncedSteps };
                        }
                    }


                    console.log(`[MCQ Fallback] Q${qr.questionNumber}: modelLetter="${modelLetter}" studentLetter="${studentLetter}" rawOcr="${rawOcr.substring(0,60)}" type="${qr.type}"`);

                    // Fallback: student letter not extractable — use AI feedback heuristic
                    const feedbackSaysCorrect = feedback.includes('good work') ||
                        (feedback.includes('correct') && !feedback.includes('incorrect'));
                    const feedbackSaysIncorrect = feedback.includes('incorrect') ||
                        feedback.includes('wrong option') || feedback.includes('wrong answer');
                    if (feedbackSaysCorrect && awarded === 0 && maxMarks > 0) {
                        awarded = maxMarks;
                        console.log('[MCQ Parity] Q' + qr.questionNumber + ': feedback=correct but marks=0 → corrected to ' + maxMarks);
                    } else if (feedbackSaysIncorrect && awarded > 0 && maxMarks > 0) {
                        awarded = 0;
                        console.log('[MCQ Parity] Q' + qr.questionNumber + ': feedback=incorrect but marks=' + awarded + ' → corrected to 0');
                    }
                    // Sync steps in fallback path too
                    const fallbackSteps = (qr.stepWiseEvaluation || []).map((step, i) => ({
                        ...step,
                        marks: i === 0 ? awarded : 0
                    }));
                   return { ...qr, marksAwarded: awarded, finalFeedback: cleanedFeedback, requiresReview: true, stepWiseEvaluation: fallbackSteps };
                } // end else (valid modelLetter)
                }
                // Case A+B: negative feedback but full marks → flag for teacher review (non-MCQ only)
                if (hasNegativeSignal && awarded >= maxMarks && maxMarks > 0) {
                    return { ...qr, marksAwarded: awarded, finalFeedback: cleanedFeedback, requiresReview: true };
                }

                // Case C: marks deducted but feedback looks positive → sync issue
                const feedbackLooksPositive = feedback === 'good work.' || feedback === 'good work' || feedback.trim() === '';
                if (awarded < maxMarks && feedbackLooksPositive && !qr.requiresReview) {
                    return { ...qr, marksAwarded: awarded, finalFeedback: cleanedFeedback, requiresReview: true };
                }

                // Flag for review: student attempted but got zero — warrants teacher check
                const isAttempted = (qr.studentOcrAnswer || qr.studentText || '').trim().length > 10;
                if (awarded === 0 && maxMarks > 0 && isAttempted && !qr.requiresReview) {
                    return { ...qr, marksAwarded: awarded, finalFeedback: cleanedFeedback, requiresReview: true };
                }
                return { ...qr, marksAwarded: awarded, finalFeedback: cleanedFeedback };
            });
            // ─────────────────────────────────────────────────────────────────────────
const lastPageIndex = Math.max(0, (pagesResult.length || 1) - 1);
            // ─── REPORT RECONSTRUCTION (UNCHANGED) ──────────────────────────────────
            const reconstructedReport = questions.map((originalQ) => {
let pageSet = pageMap.get(originalQ._uid) || new Set();
                const pageIndices = Array.from(pageSet).map(n => n - 1).sort((a, b) => a - b);



const gradedResult = questionWiseReport.find(r => r._uid === originalQ._uid);

if (gradedResult) {
                const { studentText: _dropped, ...gradedClean } = gradedResult;
                return {
                    ...gradedClean,
                    requiresReview: gradedResult.requiresReview || !!originalQ._suspectedMislabel,
                        studentOcrAnswer: originalQ.studentText,
                        // FIX: never use || 0 — pageIndices[0] can legitimately BE 0 (page 1)
                        // and 0 || 0 = 0 which is correct by accident, but undefined || 0 = 0
                        // which silently snaps every question with empty pageIndices to page 1.
                        // Use -1 (sentinel) when no page is known — frontend hides sentinel questions.
  answerPageIndex: pageIndices.length > 0 ? pageIndices[0] : lastPageIndex,
                        answerPageIndices: pageIndices.length > 0 ? pageIndices : [lastPageIndex]
                    };
                }

                return {
                    questionNumber: originalQ.questionNumber,
                    marksAwarded: 0,
                    maxMarksForQuestion: originalQ.marks,
finalFeedback: "Requires manual review — answer not found in OCR.",
studentOcrAnswer: "Answer not mapped by OCR.",
answerPageIndex: lastPageIndex,   // unmapped → last page, visible & editable
answerPageIndices: [lastPageIndex],
                    requiresReview: true,
                    stepWiseEvaluation: []
                };
            });

            // CLAMP: ensure no question ever gets answerPageIndex < 0
reconstructedReport.forEach(qr => {
    if (typeof qr.answerPageIndex !== 'number' || qr.answerPageIndex < 0) {
        qr.answerPageIndex = lastPageIndex;
    }
    if (!Array.isArray(qr.answerPageIndices) || qr.answerPageIndices.some(p => p < 0)) {
        qr.answerPageIndices = [lastPageIndex];
    }
});




           

            // ─── FIX #3: totalMarks safe for both PWA and SaaS ───────────────────────
            const computedTotalMarks = Number(totalMarks) ||
                questions.reduce((sum, q) => sum + (Number(q.marks) || 0), 0) || 0;


            let reportImageUrls = answerSheetImageUrls;
if (hasSinglePdf && pagesResult.length > 1) {
    const pdfUrl = answerSheetImageUrls?.[0] || '';
    reportImageUrls = pagesResult.map((_, i) => `${pdfUrl}#page=${i + 1}`);
}

const reportForStudent = {
    studentName: jobData.studentName || "Student",
    rollNumber: jobData.rollNumber || "",
    studentUid,
                stream: stream || "",
                overallScore: reconstructedReport.reduce((sum, qr) => sum + (qr.marksAwarded || 0), 0),
                maximumMarks: computedTotalMarks,
  overallFeedback: {
                    summary: "Grading complete.",
                   areasForImprovement: (() => {
const conceptual = [];

    reconstructedReport.forEach(qr => {
        const max = qr.maxMarksForQuestion || 0;
        const awarded = qr.marksAwarded || 0;
        if (awarded >= max || max === 0) return; // full marks — skip

        // Use grader-provided topic and category (new fields)
const topic = (qr.chapterTopic || '').trim();
        
        // FIX: If grader did not provide a topic name, skip this question entirely
        // rather than showing "Q27 — Review Required" in the improvement areas.
        // Topic-less entries pollute the improvement summary with unhelpful labels.
        if (!topic) return;
        const displayTopic = topic;
const item = {
            text: displayTopic,
            questionNumber: qr.questionNumber,
            marksLost: parseFloat((max - awarded).toFixed(1))
        };

        conceptual.push(item);
    });

    const dedup = (arr) => {
        const seen = new Set();
        return arr.filter(item => {
            const key = item.text.toLowerCase();
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        });
    };

return dedup(conceptual);
})()
                },

                // ─── SPRINT 1: DEEP STUDENT INTELLIGENCE ANALYTICS ───────────────────────
                // All computed deterministically from grading output — zero extra LLM calls.
                // Powers the comprehensive parent report with 7-category diagnostics.
                studentIntelligence: (() => {

                    // ── CAT 1: Question-Type Performance ─────────────────────────────────
                    // How does the student perform across RECALL, NUMERICAL, DERIVATION etc.
                    const typeStats = {};
                    reconstructedReport.forEach(qr => {
                        const qType = (qr.questionType || '').trim();
                        if (!qType) return;
                        if (!typeStats[qType]) typeStats[qType] = { scored: 0, max: 0, count: 0 };
                        typeStats[qType].scored += (qr.marksAwarded || 0);
                        typeStats[qType].max    += (qr.maxMarksForQuestion || 0);
                        typeStats[qType].count  += 1;
                    });
                    const questionTypePerformance = Object.entries(typeStats).map(([type, s]) => ({
                        type,
                        scored: parseFloat(s.scored.toFixed(1)),
                        max:    s.max,
                        count:  s.count,
                        pct:    s.max > 0 ? Math.round((s.scored / s.max) * 100) : 100
                    })).sort((a, b) => a.pct - b.pct); // worst first

       



                    // ── SUMMARY SIGNALS ───────────────────────────────────────────────────
                    const totalMarksInPaper = reconstructedReport.reduce((s, qr) => s + (qr.maxMarksForQuestion || 0), 0);
                    const totalScored = reconstructedReport.reduce((s, qr) => s + (qr.marksAwarded || 0), 0);
                    const totalMarksLost = parseFloat((totalMarksInPaper - totalScored).toFixed(1));

                    // Marks lost by question type (teacher-level insight)
                    const marksLostByType = questionTypePerformance
                        .filter(t => t.max > t.scored)
                        .map(t => ({
                            type: t.type,
                            marksLost: parseFloat((t.max - t.scored).toFixed(1)),
                            pct: t.pct
                        }));

 

                    // Is this a "strong thinker, weak recall" profile?
                    const derivationPct = typeStats['DERIVATION']
                        ? Math.round((typeStats['DERIVATION'].scored / typeStats['DERIVATION'].max) * 100) : null;
                    const recallPct = typeStats['RECALL']
                        ? Math.round((typeStats['RECALL'].scored / typeStats['RECALL'].max) * 100) : null;
                    const numericalPct = typeStats['NUMERICAL']
                        ? Math.round((typeStats['NUMERICAL'].scored / typeStats['NUMERICAL'].max) * 100) : null;

return {
                        questionTypePerformance,
                        summary: {
                            totalMarksLost,
                            marksLostByType,
                            derivationPct,
                            recallPct,
                            numericalPct
                        }
                    };
                })(),
                answerSheetImageUrls,
                questionWiseReport: reconstructedReport,
                fullOcrText: fullTranscript,
                gradingTimestamp: admin.firestore.FieldValue.serverTimestamp(),
                assessmentId,
                subject
            };

            await snapshot.ref.update({ status: 'SUCCESS', currentStep: 4, progress: 100, statusDetails: 'Report Saved!' });

// ─── SAVE RESULT: homework → completedHomeworkSubmissions, exam → assessmentHistory
          if (jobData.isHomework) {
    if (!jobData.homeworkSubmissionDocId) {
        throw new Error(`HOMEWORK_SAVE_FAIL: isHomework=true but homeworkSubmissionDocId is null for job ${jobId}. Report not saved.`);
    }
await db.collection('completedHomeworkSubmissions')
    .doc(jobData.homeworkSubmissionDocId)
    .set({
        detailedReport: cleanUndefined(reportForStudent),
        score: reportForStudent.overallScore,
        maximumMarks: reportForStudent.maximumMarks,
        gradedAt: admin.firestore.FieldValue.serverTimestamp()
    }, { merge: true });
} else {
    const submissionRef = db.collection('teachers')
        .doc(teacherUid)
        .collection('assessmentHistory')
        .doc(assessmentId)
        .collection('submissions')
        .doc(studentUid);

    const slimReport = cleanUndefined({
        studentName: reportForStudent.studentName,
        studentUid: reportForStudent.studentUid,
        rollNumber: reportForStudent.rollNumber || '',
        overallScore: reportForStudent.overallScore,
        maximumMarks: reportForStudent.maximumMarks,
        overallFeedback: reportForStudent.overallFeedback,
        studentKeywords: reportForStudent.studentKeywords || [],
        answerSheetImageUrls: reportForStudent.answerSheetImageUrls || [],
        studentIntelligence: reportForStudent.studentIntelligence || null,
        published: false,
        requiresReview: reportForStudent.requiresReview || false,
        gradingTimestamp: reportForStudent.gradingTimestamp,
        assessmentId: reportForStudent.assessmentId,
        subject: reportForStudent.subject,
    });

const detailDoc = cleanUndefined({
        questionWiseReport: (reportForStudent.questionWiseReport || []).map(qr => {
            const { studentText: _t, ...rest } = qr;
            return {
                ...rest,
                studentOcrAnswer: (qr.studentOcrAnswer || '').substring(0, 800),
            };
        }),
        fullOcrText: (reportForStudent.fullOcrText || '').substring(0, 50000),
    });

    await submissionRef.set(slimReport);
    await submissionRef.collection('detail').doc('report').set(detailDoc);
    console.log(`[Job ${jobId}] Exam graded — saved split to assessmentHistory`);
}

            // ─── TRAINING TRACE CAPTURE (immutable Gemini-original snapshot) ──────────
            // Stores Gemini's ORIGINAL per-question grades to GCS *before* any teacher
            // edit on the frontend. Firestore reports get edited in place, so without
            // this we lose the "what Gemini first said" side of every future training
            // pair. Cost: one small JSON object per copy, off the hot path. Never throws.
            try {
                const safeSubject = (subject || 'unknown').trim().toLowerCase().replace(/[^a-z0-9]+/g, '_') || 'unknown';
                const day = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
                const trace = {
                    schemaVersion: 1,
                    capturedAt: new Date().toISOString(),
                    jobId,
                    teacherUid,
                    assessmentId,
                    studentUid,
                    subject,
                    isHomework: !!jobData.isHomework,
                    answerSheetImageUrls: reportForStudent.answerSheetImageUrls || [], // references, not copies
                    // Gemini's original output per question (pre human-edit):
                    questions: (reconstructedReport || []).map(qr => ({
                        questionNumber: qr.questionNumber,
                        text: qr.text || '',
                        model_answer: qr.answer || '',
                        rubric: qr.rubric || null,
                        checking_instructions: qr.checkingInstructions || '',
                        max_marks: qr.maxMarksForQuestion,
                        type: qr.type || qr.questionType || '',
                        student_answer_ocr: (qr.studentOcrAnswer || '').slice(0, 6000),
                        gemini_awarded: qr.marksAwarded,
                        gemini_step_evaluation: qr.stepWiseEvaluation || [],
                        gemini_feedback: qr.finalFeedback || ''
                    }))
                };
                const objectPath = `training-traces/${safeSubject}/${day}/${teacherUid}_${assessmentId}_${studentUid}.json`;
                await storage.bucket().file(objectPath).save(JSON.stringify(trace), {
                    resumable: false,
                    contentType: 'application/json'
                });
            } catch (traceErr) {
                console.warn(`[TrainingTrace] capture failed (non-critical): ${traceErr.message}`);
            }
            // ─── END TRAINING TRACE CAPTURE ──────────────────────────────────────────

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
// EXTRACT ASSESSMENT QUESTIONS — HTTP endpoint (Vertex AI)
// POST: { images: [{mimeType, data}], prompt: string, continuationPrompt: string|null }
// Streams raw text back
// ─────────────────────────────────────────────────────────────────────────────
exports.extractAssessmentQuestions = onRequest(
    { timeoutSeconds: 300, memory: '2GiB', cors: true, region: 'us-central1' },
    async (req, res) => {
        if (req.method !== 'POST') return res.status(405).send('Method Not Allowed');
        try {
            const { images, prompt, continuationPrompt } = req.body;
            if (!images || images.length === 0) return res.status(400).json({ error: 'No images' });

            const localVertex = new VertexAI({ project: process.env.GCLOUD_PROJECT, location: 'us-central1' });
            const model = localVertex.getGenerativeModel({ model: 'gemini-2.5-flash' });

            const imageParts = images.map(img => ({ inlineData: { mimeType: img.mimeType, data: img.data } }));
            const mainParts = [...imageParts, { text: prompt }];

            const contents = continuationPrompt
                ? [
                    { role: 'user', parts: mainParts },
                    { role: 'model', parts: [{ text: '' }] },
                    { role: 'user', parts: [{ text: continuationPrompt }] }
                  ]
                : [{ role: 'user', parts: mainParts }];

            res.setHeader('Content-Type', 'text/plain');
            res.setHeader('Transfer-Encoding', 'chunked');

            const streamResult = await model.generateContentStream({
                contents,
                generationConfig: { temperature: 0.1, maxOutputTokens: 65536 }
            });

            for await (const chunk of streamResult.stream) {
                const text = chunk.candidates?.[0]?.content?.parts?.[0]?.text;
                if (text) res.write(text);
            }
            res.end();

        } catch (err) {
            console.error('[extractAssessmentQuestions] Error:', err);
            if (!res.headersSent) res.status(500).json({ error: err.message });
            else res.end();
        }
    }
);