
import { marked } from 'marked';

// --- Interfaces ---

export interface AIStepEvaluation { 
    stepDescription: string; 
    marks: number; 
    comment: string; 
    pageIndex?: number;
    stepPoint?: number[]; 
    manualQNum?: string;
}

export interface AIQuestionReport {
    questionNumber: string;
    type?: string;
    marksAwarded: number;
    maxMarksForQuestion: number;
    stepWiseEvaluation: AIStepEvaluation[];
    finalFeedback: string;
    modelAnswer?: string;
    strength?: string[];
    improvementArea?: (string | { text?: string; tag?: string; errorType?: string; detail?: string })[];
    topic?: string; 
    questionText?: string; 
    answerPageIndex?: number;
    answerPageIndices?: number[];
    questionBox?: number[] | null;
    answerEndBox?: number[] | null;
    errorBox?: number[] | null;
    studentOcrAnswer?: string;
    requiresReview?: boolean;
    missingRubricPoints?: string[]; 
    gradingConfidence?: number;     
    auditReason?: string;
    ocrConfidence?: number;
    auditFlags?: string[];
    riskScore?: number;      
    riskSignals?: string[]; 
    questionEndAnchor?: { pageIndex: number; y: number; x: number };
}

export interface AIFullAssessmentReport {
    studentName: string;
    studentUid: string;
    rollNumber?: string;
    overallScore: number;
    maximumMarks: number;
    overallFeedback: { summary: string; strengths: string[]; areasForImprovement: (string | { text?: string; tag?: string; detail?: string; errorType?: string })[]; };
    questionWiseReport: AIQuestionReport[];
    submissionDocId?: string; 
    answerSheetImageUrls?: string[];
    studentKeywords?: string[];
    published?: boolean;
    fullOcrText?: string; 
    pencilAnnotations?: { [pageIndex: number]: string[] }; 
}

// --- Utilities ---

export function normalizeForComparison(str: string | undefined): string {
    if (!str) return '';
    return str.toLowerCase()
              .replace(/\b(ans|answer|q|question|sol|solution|cont|continued|pt|part)\b/gi, '')
              .replace(/[^a-z0-9]/g, '') 
              .trim();
}

export function getOcrTextForPage(fullText: string | undefined, pageNum: number): string {
    if (!fullText) return "No transcript available.";
    const pageMarker = `[PAGE ${pageNum}]`;
    const nextPageMarker = `[PAGE ${pageNum + 1}]`;
    const startIndex = fullText.indexOf(pageMarker);
    if (startIndex === -1) return "Transcript for this page not found.";
    const contentStart = startIndex + pageMarker.length;
    const endIndex = fullText.indexOf(nextPageMarker, contentStart);
    const pageText = (endIndex === -1) 
        ? fullText.substring(contentStart) 
        : fullText.substring(contentStart, endIndex);
    return pageText.trim() || "(Empty Page)";
}

export function formatMathExpressions(text: string): string {
    if (!text || typeof text !== 'string') return '';
    let processedText = text.replace(/#/g, '\\#');
    const commonLatexRegex = /\\(frac|sqrt|int|sum|alpha|beta|theta|pi|phi|neq|le|ge|pm|times|div|approx|cdot|\{|[_^])/g;
    const hasDelimiters = /\\\(|\\\[|\$\$|\$/.test(text);
    if (commonLatexRegex.test(text) && !hasDelimiters) {
        processedText = text.split('\n').map(line => {
            if (commonLatexRegex.test(line) && !/\\\(|\\\[|\$\$|\$/.test(line)) {
                return `\\( ${line.trim()} \\)`;
            }
            return line;
        }).join('\n');
    }
    const mathRegex = /(\\+\[[\s\S]*?\\+\])|(\\+\([\s\S]*?\\+\))|(\$\$[\s\S]*?\$\$)/g;
    const mathExpressions: string[] = [];
    let textWithPlaceholders = processedText.replace(mathRegex, (match) => {
        mathExpressions.push(match);
        return `%%MATH_PLACEHOLDER_${mathExpressions.length - 1}%%`;
    });
    textWithPlaceholders = textWithPlaceholders.replace(/^(\s*)(\d+)\./gm, '$1$2\\.');
    let html = marked.parse(textWithPlaceholders, { breaks: true, gfm: true }) as string;
    html = html.replace(/%%MATH_PLACEHOLDER_(\d+)%%/g, (match, indexStr) => {
        const index = parseInt(indexStr, 10);
        let originalMatch = mathExpressions[index];
        if (!originalMatch) return '';
        return originalMatch.replace(/^\\+\(/, '\\(')
                            .replace(/\\+\)$/, '\\)')
                            .replace(/^\\+\[/, '\\[')
                            .replace(/\\+\]$/, '\\]');
    });
    return html;
}

export async function typesetMathInContainer(element: HTMLElement) {
    if ((window as any).MathJax && typeof (window as any).MathJax.typesetPromise === 'function') {
        await (window as any).MathJax.typesetPromise([element]);
    }
}

export function createHumanAnnotationHtml(report: AIQuestionReport, currentPageIndex: number, qIdx: number, reportIndex: number, isGlobalMarkerEdit: boolean): string {
    const annotations = (report.stepWiseEvaluation || [])
        .filter(s => s.pageIndex === currentPageIndex && s.stepPoint && s.stepPoint.length === 2)
        .map((step, stepIdx) => {
            const [y, x] = step.stepPoint!;
            const isStepCorrect = step.marks > 0 || step.comment === 'Correct';
            const pointerEvents = isGlobalMarkerEdit ? 'auto' : 'none';
            const safeY = (typeof y === 'number' && !isNaN(y)) ? y / 10 : 10;
            const safeX = (typeof x === 'number' && !isNaN(x)) ? x / 10 : 92;

            return `
          <div class="osm-step-marker osm-icon-badge ${isStepCorrect ? 'correct' : 'incorrect'}" 
     style="top:${safeY}%; left:${safeX}%; position:absolute; transform:translate(-50%, -50%); z-index:200; font-size:1.8rem; font-weight:900; 
                        pointer-events:${pointerEvents}; cursor:${isGlobalMarkerEdit ? 'move' : 'default'};"
                 ${isGlobalMarkerEdit ? `draggable="true" ondragstart="window.handleOsmGlobalDragStart(event, ${qIdx}, ${stepIdx})"` : ''}>
                ${isStepCorrect ? '✓' : '✗'}
                ${isGlobalMarkerEdit ? `
                    <div class="osm-delete-btn" onclick="event.stopPropagation(); window.handleOsmRemoveGlobal(${reportIndex}, ${qIdx}, ${stepIdx})" 
                         style="position:absolute; top:-12px; right:-12px; background:red; color:white; border-radius:50%; width:20px; height:20px; font-size:12px; display:flex; align-items:center; justify-content:center; border:2px solid white; z-index:210;">×</div>
                ` : ''}
            </div>`;
        }).join('');
    return annotations;
}

// --- Report Viewer Class ---

export class ReportViewer {
    private report: AIFullAssessmentReport;
    private config: {
        containerId: string;
        apiBaseUrl: string;
        apiKey: string;
        jobId: string;
        onSave?: (updatedReport: AIFullAssessmentReport) => Promise<void>;
    };

    private state = {
        currentReportPageIdx: 0,
        isOsmEditing: false,
        osmEditingContext: null as { reportIndex: number; questionIndex: number } | null,
        activeFsTool: 'pencil' as 'pencil' | 'eraser' | 'tick' | 'cross',
        fsOcrViewMode: 'page' as 'page' | 'spliced',
        isMarkersVisible: true,
    };

    constructor(report: AIFullAssessmentReport, config: any) {
        this.report = report;
        this.config = config;
        
        // Expose necessary functions to window for HTML event handlers
        (window as any).handleManualPageChange = this.handleManualPageChange.bind(this);
        (window as any).toggleFeedbackEdit = this.toggleFeedbackEdit.bind(this);
        (window as any).saveFeedbackEdit = this.saveFeedbackEdit.bind(this);
        (window as any).handleUpdateScoreFromModal = this.handleUpdateScoreFromModal.bind(this);
        (window as any).viewQuestionOcr = this.viewQuestionOcr.bind(this);
        (window as any).toggleOsmEditMode = this.toggleOsmEditMode.bind(this);
        (window as any).saveOsmChanges = this.saveOsmChanges.bind(this);
        (window as any).cancelOsmEdit = this.cancelOsmEdit.bind(this);
        (window as any).handleViewRawOcr = this.handleViewRawOcr.bind(this);
        (window as any).openAiTrainingModal = this.openAiTrainingModal.bind(this);
        (window as any).setFsTool = this.setFsTool.bind(this);
        (window as any).toggleFsOcrViewMode = this.toggleFsOcrViewMode.bind(this);
        (window as any).handleOsmDrop = this.handleOsmDrop.bind(this);
        (window as any).startScribble = this.startScribble.bind(this);
        (window as any).drawScribble = this.drawScribble.bind(this);
        (window as any).endScribble = this.endScribble.bind(this);
        (window as any).removeScribble = this.removeScribble.bind(this);
        (window as any).handleDownloadLibrarianSplice = this.handleDownloadLibrarianSplice.bind(this);
    }

    public render() {
        const container = document.getElementById(this.config.containerId);
        if (!container) return;

        const savedScrollTop = container.scrollTop;
        const report = this.report;
        const reportIndex = 0; // In standalone mode, we usually deal with one report
        const imageUrls = report.answerSheetImageUrls || [];
        
        let html = '';
        for (let i = 0; i < imageUrls.length; i++) {
            const pageNum = i + 1;
            const currentImgUrl = imageUrls[i];
            const pageAnnotations = report.questionWiseReport.map((qr, qIdx) => createHumanAnnotationHtml(qr, i, qIdx, reportIndex, this.state.isOsmEditing)).join('');
            const pageOcr = getOcrTextForPage(report.fullOcrText, pageNum);

            const questionsOnPage = report.questionWiseReport.filter(qr => {
                const feedback = (qr.finalFeedback || "").toLowerCase();
                if (feedback.includes("no attempt") || feedback.includes("not detected") || feedback.includes("not attempted")) return false;
                let actualPageIndex = qr.answerPageIndex || 0;
                return actualPageIndex === i;
            });

            const questionsHtml = questionsOnPage.map(qr => {
                const originalIndex = report.questionWiseReport.indexOf(qr);
                const flagBannerHtml = qr.requiresReview ? `<div class="risk-banner"><div class="risk-banner-header"><span class="risk-icon">⚠️</span><div class="risk-title"><strong>${qr.auditReason || "Needs Review"}</strong></div></div></div>` : '';
                const percentage = qr.maxMarksForQuestion > 0 ? (qr.marksAwarded / qr.maxMarksForQuestion * 100) : 0;
                let scoreClass = percentage > 70 ? 'score-high' : (percentage > 40 ? 'score-medium' : 'score-low');
                const isEditingThisMarker = this.state.isOsmEditing && this.state.osmEditingContext?.questionIndex === originalIndex;

                return `
                <div class="q-card-redesign" id="fs-q-card-ref-${normalizeForComparison(qr.questionNumber)}" style="margin-bottom: 20px; border: 1px solid #e0e0e0; border-radius: 12px; padding: 12px; background: white;">
                    ${flagBannerHtml}
                    <div class="q-card-header" style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                        <h6 style="margin:0; font-size:1.05rem; display:flex; align-items:center; font-weight:700; flex-wrap: wrap; gap: 5px;">
                            Q${qr.questionNumber}
                            <select class="styled-select" onchange="window.handleManualPageChange(${reportIndex}, ${originalIndex}, this.value)">
                                ${imageUrls.map((_, pIdx) => `<option value="${pIdx}" ${pIdx === qr.answerPageIndex ? 'selected' : ''}>Pg ${pIdx + 1}</option>`).join('')}
                            </select>
                        </h6>
                        <span class="score-input-wrapper">
                            <input type="number" class="editable-score-input fs-score-input-compact ${scoreClass}" value="${qr.marksAwarded.toFixed(1)}" onchange="window.handleUpdateScoreFromModal(${reportIndex}, ${originalIndex})">
                            <span style="font-weight:600; color:#666; font-size: 0.8rem;">/ ${qr.maxMarksForQuestion}</span>
                        </span>
                    </div>
                    <div class="always-visible-feedback feedback-container-compact">
                        <div id="fs-feedback-view-${reportIndex}-${originalIndex}">
                            <div class="edit-btn-top-right"><button class="button-tertiary fs-edit-btn-inline" onclick="window.toggleFeedbackEdit(${reportIndex}, ${originalIndex}, true)">✎</button></div>
                            <div class="static-feedback-content ${scoreClass}" style="padding:8px; border-radius:6px; font-size:0.85rem; line-height:1.4; min-height:40px; margin-top: 5px;">
                                ${formatMathExpressions(qr.finalFeedback)}
                            </div>
                        </div>
                        <div id="fs-feedback-edit-${reportIndex}-${originalIndex}" style="display: none;">
                            <textarea class="editable-feedback-textarea" style="width:100%; border-radius:6px; padding:8px; min-height:60px;" rows="2">${qr.finalFeedback}</textarea>
                            <div style="display: flex; gap: 4px; justify-content: flex-end; margin-top: 5px;">
                                <button class="button-secondary extra-small-button" onclick="window.toggleFeedbackEdit(${reportIndex}, ${originalIndex}, false)">Cancel</button>
                                <button class="button-primary extra-small-button" onclick="window.saveFeedbackEdit(${reportIndex}, ${originalIndex})">Save</button>
                            </div>
                        </div>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; border-top:1px solid #f0f0f0; padding-top:8px; margin-top:5px;">
                        <button class="button-secondary icon-btn-compact" onclick="window.viewQuestionOcr(${reportIndex}, ${originalIndex})"><span class="icon">👁️</span></button>
                        <div style="display: flex; gap: 6px;">
                            ${isEditingThisMarker ? `
                                <button class="button-primary extra-small-button" onclick="window.saveOsmChanges()">Save</button>
                                <button class="button-secondary extra-small-button" onclick="window.cancelOsmEdit()">Cancel</button>
                            ` : `
                                <button class="button-tertiary icon-btn-compact" onclick="window.handleViewRawOcr('${report.studentUid}')"><span class="icon">📜</span></button>
                                <button class="button-tertiary icon-btn-compact" onclick="window.toggleOsmEditMode(${reportIndex}, ${originalIndex}, true)"><span class="icon">📍</span></button>
                            `}
                        </div>
                    </div>
                </div>`;
            }).join('') || '<p style="font-size:0.75rem; color:#999; text-align:center; padding:20px;">No questions detected on this page.</p>';

            const toolbarHtml = `
            <div class="fs-page-toolbar">
                <button class="tool-btn ${this.state.activeFsTool === 'pencil' ? 'active' : ''}" onclick="window.setFsTool('pencil', ${i}, ${reportIndex})">✏️</button>
                <button class="tool-btn ${this.state.activeFsTool === 'eraser' ? 'active' : ''}" onclick="window.setFsTool('eraser', ${i}, ${reportIndex})">🧼</button>
                <button class="tool-btn ${this.state.activeFsTool === 'tick' ? 'active' : ''}" onclick="window.setFsTool('tick', ${i}, ${reportIndex})">✅</button>
                <button class="tool-btn ${this.state.activeFsTool === 'cross' ? 'active' : ''}" onclick="window.setFsTool('cross', ${i}, ${reportIndex})">❌</button>
            </div>`;

            html += `
            <div class="fs-page-row" id="fs-page-row-${i}">
                <div class="fs-col fs-image-col" style="position:relative; flex-direction:row;">
                    ${toolbarHtml}
                    <div class="osm-inner-wrapper" style="position: relative; display: inline-block;">
                        <img src="${currentImgUrl}" id="fs-img-${i}">
                        <svg class="drawing-layer" id="fs-svg-${i}" viewBox="0 0 1000 1000" onmousedown="window.startScribble(event, ${i})" onmousemove="window.drawScribble(event, ${i})" onmouseup="window.endScribble(${i}, ${reportIndex})" style="position:absolute; top:0; left:0; width:100%; height:100%; pointer-events: ${this.state.activeFsTool === 'pencil' || this.state.activeFsTool === 'eraser' ? 'auto' : 'none'}; z-index: 120;">
                            ${(report.pencilAnnotations?.[i] || []).map((d, pathIdx) => `<g onclick="if(window.activeFsTool==='eraser') window.removeScribble(${reportIndex}, ${i}, ${pathIdx})"><path d="${d}" stroke="red" stroke-width="3" fill="none" pointer-events="none" /><path d="${d}" stroke="transparent" stroke-width="20" fill="none" style="cursor:pointer" /></g>`).join('')}
                        </svg>
                        <div class="osm-overlay ${!this.state.isMarkersVisible ? 'markers-hidden' : ''}" id="fs-osm-overlay-layer-${i}" ondragover="event.preventDefault()" ondrop="window.handleOsmDrop(event, ${i}, ${reportIndex})" style="position: absolute; top:0; left:0; width:100%; height:100%; pointer-events:auto; z-index: 110;">
                            ${pageAnnotations}
                        </div>
                    </div>
                </div>
                <div class="fs-col">
                    <div class="fs-col-header">${this.state.fsOcrViewMode === 'page' ? 'AI PAGE TRANSCRIPT' : 'LIBRARIAN SPLICED BLOCKS'}</div>
                    <div class="fs-scroll-content ocr-content-box">
                        ${this.state.fsOcrViewMode === 'page' ? formatMathExpressions(pageOcr) : report.questionWiseReport.filter(qr => qr.answerPageIndex === i).map(qr => `<div class="spliced-block"><strong>Q:${qr.questionNumber}</strong><br>${formatMathExpressions(qr.studentOcrAnswer || "")}</div>`).join('')}
                    </div>
                </div>
                <div class="fs-col">
                    <div class="fs-col-header">GRADING & FEEDBACK</div>
                    <div class="fs-scroll-content" style="background:#f8f9fa;">${questionsHtml}</div>
                </div>
            </div>`;
        }

        container.innerHTML = html;
        container.scrollTop = savedScrollTop;
        typesetMathInContainer(container);
    }

    // --- Event Handlers (Simplified for Standalone) ---

    private async handleManualPageChange(reportIndex: number, questionIndex: number, newPageValue: string) {
        const newPageIndex = parseInt(newPageValue, 10);
        const qr = this.report.questionWiseReport[questionIndex];
        qr.answerPageIndex = newPageIndex;
        qr.answerPageIndices = [newPageIndex];
        if (qr.stepWiseEvaluation) qr.stepWiseEvaluation.forEach(step => step.pageIndex = newPageIndex);
        await this.persist();
        this.render();
    }

    private toggleFeedbackEdit(reportIndex: number, questionIndex: number, isEditing: boolean) {
        const view = document.getElementById(`fs-feedback-view-${reportIndex}-${questionIndex}`);
        const edit = document.getElementById(`fs-feedback-edit-${reportIndex}-${questionIndex}`);
        if (view && edit) {
            view.style.display = isEditing ? 'none' : 'block';
            edit.style.display = isEditing ? 'block' : 'none';
        }
    }

    private async saveFeedbackEdit(reportIndex: number, questionIndex: number) {
        const edit = document.getElementById(`fs-feedback-edit-${reportIndex}-${questionIndex}`);
        const textarea = edit?.querySelector('textarea');
        if (textarea) {
            this.report.questionWiseReport[questionIndex].finalFeedback = textarea.value;
            await this.persist();
            this.render();
        }
    }

    private async handleUpdateScoreFromModal(reportIndex: number, questionIndex: number) {
        // Implementation of score update logic
        await this.persist();
        this.render();
    }

    private viewQuestionOcr(reportIndex: number, questionIndex: number) {
        // Implementation of OCR view logic
    }

    private toggleOsmEditMode(reportIndex: number, questionIndex: number, isEditing: boolean) {
        this.state.isOsmEditing = isEditing;
        this.state.osmEditingContext = isEditing ? { reportIndex, questionIndex } : null;
        this.render();
    }

    private async saveOsmChanges() {
        this.state.isOsmEditing = false;
        await this.persist();
        this.render();
    }

    private cancelOsmEdit() {
        this.state.isOsmEditing = false;
        this.render();
    }

    private handleViewRawOcr(studentUid: string) {
        // Implementation of raw OCR view
    }

    private openAiTrainingModal(reportIndex: number, questionIndex: number) {
        // Implementation of AI training modal
    }

    private setFsTool(tool: any, pageIdx: number, reportIdx: number) {
        this.state.activeFsTool = tool;
        this.render();
    }

    private toggleFsOcrViewMode(reportIndex: number) {
        this.state.fsOcrViewMode = this.state.fsOcrViewMode === 'page' ? 'spliced' : 'page';
        this.render();
    }

    private handleOsmDrop(event: DragEvent, pageIdx: number, reportIdx: number) {
        // Implementation of marker drop logic
    }

    private startScribble(event: MouseEvent, pageIdx: number) {
        // Implementation of drawing start
    }

    private drawScribble(event: MouseEvent, pageIdx: number) {
        // Implementation of drawing
    }

    private async endScribble(pageIdx: number, reportIdx: number) {
        // Implementation of drawing end
        await this.persist();
        this.render();
    }

    private async removeScribble(reportIdx: number, pageIdx: number, pathIdx: number) {
        // Implementation of scribble removal
        await this.persist();
        this.render();
    }

    private async handleDownloadLibrarianSplice(reportIndex: number) {
        // Implementation of download logic
    }

    private async persist() {
        if (this.config.onSave) {
            await this.config.onSave(this.report);
        } else {
            // Default API call to backend
            await fetch(`${this.config.apiBaseUrl}/v1_update_result`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'x-api-key': this.config.apiKey 
                },
                body: JSON.stringify({ jobId: this.config.jobId, report: this.report })
            });
        }
    }
}
