# ClassMap Backend — Handover for Claude Code Web

Read this first in any new session (local or web) working on the OCR/grading backend.

## 1. Where the code lives

- **This repo** (`github.com/info165/teachers-classmap`, branch `main`) is the backend. The only file that matters for grading: `vertex-ai-backend/index.js` (~475KB, one file — OCR, librarian question-matching, grading, report reconstruction all live here).
- **Frontend is a SEPARATE repo**: `github.com/info165/classmap-teacher-frontend`, file `testfrontend/index.tsx` (~30k lines, not React — plain TSX built with Vite). Different repo, different clone, different deploy. Not needed for Hindi/grading-accuracy work.
- Firebase project for everything (Firestore, Auth, Storage, Functions, Hosting): **`student-database-74297`**.

## 2. Deploying the backend

- **Auto-deploy exists as of 2026-08-10**: `.github/workflows/deploy-functions.yml` deploys `functions` on every push to `main` that touches `vertex-ai-backend/**`. Uses GitHub secret `FIREBASE_SERVICE_ACCOUNT` (encrypted, set via `gh secret set`, value never in any file or chat).
- **Status: blocked on a one-time manual IAM grant.** The deploy identity (`firebase-adminsdk-fbsvc@student-database-74297.iam.gserviceaccount.com`) needs the **"Service Account User"** role on `student-database-74297@appspot.gserviceaccount.com`. Grant it at https://console.cloud.google.com/iam-admin/iam?project=student-database-74297 — only a human/project-Owner can do this, not any Claude session. Until granted, pushing to `main` will NOT actually redeploy (the Action runs and fails at the deploy step).
- To test manually: `gh workflow run deploy-functions.yml --repo info165/teachers-classmap`, then `gh run list --repo info165/teachers-classmap --limit 3`.
- Old manual method (still works if you have local `firebase` CLI login): `firebase deploy --only functions:processGradingJob --project student-database-74297` from `vertex-ai-backend/`.

## 3. Where the data actually is (Firestore, project `student-database-74297`)

- **`auditLogs`** (~49,785 docs) — every human correction a teacher makes to a Gemini grade. Keyed by `studentUid` + `questionNumber` + `assessmentTitle`. `actionType`: `marks_changed` (has `oldValue`=Gemini's mark, `newValue`=teacher's), `verified`, `marker_removed`, `page_changed`, `published`.
- **`teachers/{uid}/assessmentHistory/{assessmentId}/submissions/{studentUid}/detail/report`** — the full graded report for an EXAM submission. Field `questionWiseReport[]` has, per question: `studentOcrAnswer`, `rubric`, `answer` (model answer), `checkingInstructions`, `maxMarksForQuestion`, `marksAwarded`, `stepWiseEvaluation`, `finalFeedback`, `requiresReview`.
- **`completedHomeworkSubmissions/{id}`** (~1,487 docs) — lightweight HOMEWORK summary. `objectiveReport[]` (MCQ, already report-shaped) + `detailedReport.questionDetails[]` (all questions but no images, different field names).
- **`studentGradingResults/{gradingJobId}`** — the real, full homework grading output (same grader as exams), `questionWiseReport[]` for subjective questions, **and the actual answer image URLs** (`answerImageUrls[]`) — the only place those images exist. Referenced from the summary doc's `gradingJobId`. **Client-side reads of this collection are blocked by security rules for teachers** — go through the `getHomeworkSubmissionReport` Cloud Function instead (already deployed), not direct Firestore reads.
- **Training corpus** (for the eventual self-hosted model, separate from live grading): `gs://student-database-74297.firebasestorage.app/training-corpus/backfill-2026-07-17/{subject}.jsonl` — 62,312 questions, 26,234 teacher-corrected.
- Assessment/rubric/question-bank data (model answers, options, per-question rubrics used by the grader) is pulled the same way as everything above — via the Admin SDK, not hardcoded here. Exact collection path wasn't written down in a memory file; re-derive it the same way it was found originally: pull one full graded report and follow the `assessmentId` it references.

## 4. How to query real data (no frontend, no login needed)

Every real-data check this whole project has done (verifying OCR against actual student handwriting, confirming a grading bug against a real report) used the **Firestore Admin SDK directly**, never the web app login:

```js
const admin = require('/Users/nikhar/Desktop/Teachers-classmap-Project/vertex-ai-backend/node_modules/firebase-admin');
admin.initializeApp({ credential: admin.credential.cert(require('/Users/nikhar/Desktop/Teachers-classmap-Project/functions/serviceAccountKey.json.json')) });
const db = admin.firestore();
```

**This key only exists on the local laptop** (`functions/serviceAccountKey.json.json`, gitignored, never committed in this repo). A Claude Code **web** session — which works from a cloned copy of GitHub, not this laptop — will NOT have this file and cannot pull live Firestore data or download real answer-sheet images on its own. Two ways to handle that:
- Do the Hindi *prompt/code* work in the web session, and bring real Hindi student papers into that chat as pasted text/uploaded images yourself, the same way you supplied the Verma/Agarwal photos.
- Or check whether Claude Code web's environment/sandbox settings let you attach a credential file or secret to that specific session yourself (never paste the raw key into chat) — this wasn't verified this session, check the product's own settings.

**Important — do NOT commit this key to any repo.** It was found committed to the frontend repo's git history this session (`functions/serviceAccountKey.json.json`, since untracked but not purged from history) and is what's active in production right now. Left as-is per your call — see section 6.

## 5. Secrets inventory (what exists, where — no values below)

| What | Where it lives | Status |
|---|---|---|
| GCP Admin SDK key (`firebase-adminsdk-fbsvc@...`) | `functions/serviceAccountKey.json.json` (local only) | Untracked in this repo. Also set as GitHub Actions secret `FIREBASE_SERVICE_ACCOUNT` on `teachers-classmap` (encrypted, deploy-only use). Same key sits in `classmap-teacher-frontend`'s git history (committed, private repo, not rotated — user's explicit choice for now). |
| Backend API key | `vertex-ai-backend/env` (local only) | Gitignored, never committed. |
| Frontend API key | `testfrontend/.env.txt` (local only) | Untracked as of this session; WAS committed in frontend repo history, not rotated. |
| A live Google API key hardcoded as literal text | `testfrontend/vite.config.ts` (fallback default value) | **Still live in the deployed bundle** at the production URL — this one is genuinely public (anyone visiting the site can read it from the JS), not just repo-private. Flagged, not fixed yet, user's explicit call to defer. |

## 6. Live URLs

- Teacher web app: `teachersclassmap.web.app` (Firebase Hosting target `teachersclassmap`, deployed manually via `firebase deploy --only hosting:teachersclassmap --project student-database-74297` from `/Users/nikhar/Desktop/TestTeacher` — no auto-deploy CI/CD set up for this one yet, paused when the committed-secret issue above was found).
- Backend has no user-facing URL — it's Cloud Functions triggered by Firestore document writes (`processGradingJob`, `onDocumentCreated`) plus a couple of callable HTTP endpoints (`getHomeworkSubmissionReport`, `extractAssessmentQuestions`) the frontend calls directly.

## 7. Current grading pipeline (what actually runs on a submission)

OCR (Gemini 2.5 Flash, Vertex AI) → OCR self-verification pass (new, compares transcript against the source image, forces `requiresReview:true` on any disagreement rather than silently picking a side) → librarian (matches OCR text to question boundaries) → tiered grader (rubric-based, `gemini-2.5-flash` for everything — explicitly NOT flash-lite, accuracy was too low when tested) → report written back to Firestore.

## 8. Branching, if running Hindi-support work and English-accuracy work in parallel web sessions

Both would touch `vertex-ai-backend/index.js`. To avoid collisions:
- Do Hindi work on a new branch, e.g. `git checkout -b hindi-language-support`, push that branch, keep `main` untouched and still auto-deploying English-only.
- Anything found on the Verma/Agarwal re-audit that needs a real code fix should land on `main` directly (small, independent changes — unrelated code sections to the Hindi addendum, so a later merge is unlikely to conflict).
- Merge `hindi-language-support` into `main` only once it's tested, behind a `language` check so English requests are unaffected either way (see design note below).

## 9. Hindi support — the agreed design (not yet built)

Do NOT fork `index.js` into a second file. Keep one shared base (`OCR_SYSTEM_INSTRUCTION` / `GRADING_SYSTEM_INSTRUCTION` — all the anti-fabrication rules, rubric enforcement) and add a small `HINDI_OCR_ADDENDUM` block appended only when the grading job's language is Hindi. English-language jobs get byte-identical prompts to today (zero cost/behavior change); Hindi jobs pay for the addendum only when needed. This also means every bug fix made to the shared logic (tokenizer, blank-answer guard, truncation handling, rubric step-count enforcement — see git log) automatically applies to Hindi too, instead of needing to be duplicated.

## 10. Where the full history of *why* things are the way they are lives

Most of it is in this machine's Claude memory (`/Users/nikhar/.claude/projects/-Users-nikhar-Desktop-Teachers-classmap-Project/memory/`), which a web session won't have access to. If continuing in web long-term, the durable record is: this file, the git log/commit messages in this repo (each fix has a real "why" in its message), and whatever you paste into the new session.
