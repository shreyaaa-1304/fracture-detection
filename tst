# test.py
import warnings
warnings.filterwarnings("ignore")

import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageFile
from torchvision import transforms
from datetime import datetime

import google.generativeai as genai
genai.configure(api_key="key")

ImageFile.LOAD_TRUNCATED_IMAGES = True

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm as rcm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image as RLImage, Table, TableStyle,
                                 HRFlowable, KeepTogether)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

import config
from model import build_model

# ══════════════════════════════════════════════════════════════════════════
#  Grad-CAM
# ══════════════════════════════════════════════════════════════════════════

class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0).to(config.DEVICE)
        output       = self.model(input_tensor)
        probs        = torch.softmax(output, dim=1)
        pred_idx     = output.argmax(dim=1).item()
        confidence   = probs[0][pred_idx].item()
        self.model.zero_grad()
        output[0, pred_idx].backward()
        pooled_grads = self.gradients.mean(dim=[0, 2, 3])
        cam          = self.activations[0].cpu()
        for i, w in enumerate(pooled_grads.cpu()):
            cam[i] *= w
        cam = cam.mean(dim=0).numpy()
        cam = np.maximum(cam, 0)
        if cam.max() != 0:
            cam = cam / cam.max()
        return cam, pred_idx, confidence


# ══════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════

def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img)
    return tensor, img


def overlay_cam(original_img, cam, alpha=0.5):
    cam_resized = np.array(
        Image.fromarray(np.uint8(255 * cam)).resize(
            original_img.size, Image.BILINEAR)
    ) / 255.0
    heatmap      = cm.jet(cam_resized)[:, :, :3]
    orig_np      = np.array(original_img) / 255.0
    superimposed = alpha * heatmap + (1 - alpha) * orig_np
    return np.clip(superimposed, 0, 1)


# ══════════════════════════════════════════════════════════════════════════
#  Severity (label-aware)
# ══════════════════════════════════════════════════════════════════════════

def compute_severity(confidence, cam, pred_label):
    if pred_label != "fractured":
        return 0.0, "NONE", "No fracture detected. No clinical action required."

    conf_score   = confidence * 60
    spread       = np.sum(cam > 0.5) / cam.size
    spread_score = min(spread * 200, 25)
    peak_score   = float(np.max(cam)) * 15
    total        = round(conf_score + spread_score + peak_score, 2)

    if total >= 80:
        return total, "SEVERE",   "Immediate orthopaedic consultation recommended."
    elif total >= 55:
        return total, "MODERATE", "Further imaging (CT / MRI) advised."
    elif total >= 30:
        return total, "MILD",     "Conservative management may be appropriate. Monitor closely."
    else:
        return total, "MINIMAL",  "Low-grade fracture signal. Clinical review advised."


# ══════════════════════════════════════════════════════════════════════════
#  Gemini API — generate narrative
# ══════════════════════════════════════════════════════════════════════════

def generate_narrative(pred_label, confidence, severity_score,
                       severity_label, cam, image_path):

    is_fractured = pred_label == "fractured"
    peak_act     = round(float(np.max(cam)) * 100, 2)
    mean_act     = round(float(np.mean(cam)) * 100, 2)
    spread_pct   = round(np.sum(cam > 0.5) / cam.size * 100, 2)
    filename     = os.path.basename(image_path)
    frac_status  = "FRACTURED" if is_fractured else "NOT FRACTURED"
    conf_str     = f"{confidence * 100:.2f}%"

    prompt = f"""
You are an expert AI system designed to generate a professional, clinically structured
report for an AI-assisted bone fracture detection system.

Transform the raw model outputs below into well-structured, human-readable
medical-style report sections suitable for academic submission.
The text must NOT feel like raw machine output. It should read like a
diagnostic report with clear explanations and professional language.

INPUT DATA:
- Prediction: {frac_status}
- Confidence: {conf_str}
- Severity Score: {severity_score}/100
- Severity Level: {severity_label}
- Body Part: Hand / Wrist
- Scan Type: X-ray
- Image Name: {filename}
- Grad-CAM Insight:
    Peak Activation:        {peak_act}%
    Mean Activation:        {mean_act}%
    High Activation Area:   {spread_pct}%
    Region Description:     Central metacarpal / wrist region (estimated)

Generate ONLY the following four sections.
Use exactly these headings (in ALL CAPS) and no others:

AI-GENERATED SUMMARY
Write 3-4 sentences. Cover: fracture status, confidence interpretation,
what Grad-CAM indicates, and what the severity implies clinically.

IMAGING ANALYSIS
Write 2-3 sentences explaining what the Grad-CAM heatmap represents,
what red/yellow vs blue regions mean, and where the model is focusing.

FINDINGS
Provide exactly 4 bullet points (start each with a dash):
- Fracture presence
- Suspected location based on heatmap
- Severity interpretation
- One notable clinical observation
Do NOT invent fracture type if not given.

RECOMMENDED ACTIONS
Provide exactly 4 bullet points (start each with a dash) as practical next steps.

RULES:
- No emojis
- No technical ML details (no epochs, PyTorch, checkpoints, loss, training)
- Professional, slightly formal language
- No hallucinated medical details
- Each section heading on its own line in ALL CAPS
- Keep total response under 400 words
"""

    try:
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        response     = gemini_model.generate_content(prompt)
        text         = response.text.strip()
        print("  Gemini narrative generated successfully.")
        return text

    except Exception as e:
        print(f"  ! Gemini API unavailable ({e}). Using fallback narrative.")

        if is_fractured:
            return f"""AI-GENERATED SUMMARY
The AI model has identified a fracture in the submitted X-ray image with a confidence \
of {conf_str}. This high confidence level indicates strong model certainty in the \
prediction. The Grad-CAM analysis highlights concentrated activation in the central \
bone region, suggesting localised structural disruption. The severity score of \
{severity_score}/100 classifies this case as {severity_label}, implying that prompt \
clinical evaluation is warranted.

IMAGING ANALYSIS
The Grad-CAM heatmap visualises the regions of the X-ray that most influenced the \
model prediction. Areas shown in red and yellow represent high activation zones where \
the model detected significant bone irregularities. Blue regions correspond to areas \
of low relevance to the prediction.

FINDINGS
- A fracture has been detected in the submitted X-ray image with high model confidence.
- The heatmap highlights the central metacarpal and wrist region as the primary area of concern.
- The severity score of {severity_score}/100 indicates a {severity_label.lower()} fracture presentation.
- The high activation area suggests a localised fracture zone rather than a diffuse injury pattern.

RECOMMENDED ACTIONS
- Immediate referral to an orthopaedic surgeon is recommended for clinical assessment.
- Advanced imaging such as CT or MRI should be considered for detailed structural evaluation.
- The affected limb should be immobilised pending further clinical review.
- A follow-up X-ray is advised after initial treatment to monitor healing progress."""

        else:
            return f"""AI-GENERATED SUMMARY
The AI model has not detected any fracture in the submitted X-ray image, with a \
confidence of {conf_str}. This high confidence level reflects strong model certainty \
that the bone structure appears intact. The Grad-CAM analysis shows distributed \
activation with no concentrated focal region, consistent with a normal bone \
presentation. No clinical severity has been assigned as the scan does not indicate \
a fracture.

IMAGING ANALYSIS
The Grad-CAM heatmap represents the attention distribution of the model across the \
X-ray image. In this case, the absence of concentrated red or yellow zones indicates \
that no specific region triggered a high-confidence fracture response. The diffuse \
activation pattern is consistent with a structurally normal bone.

FINDINGS
- No fracture has been detected in the submitted X-ray image.
- The Grad-CAM heatmap shows no localised high-activation region indicative of a fracture.
- Severity score is 0/100, consistent with a normal bone presentation.
- Bone contour and density appear within expected normal range for this body region.

RECOMMENDED ACTIONS
- No immediate orthopaedic intervention is required based on this scan.
- If the patient reports persistent pain, a follow-up clinical examination is recommended.
- Repeat imaging may be considered if symptoms develop or worsen over time.
- Maintain routine monitoring as clinically indicated."""


def parse_narrative(text):
    section_keys = [
        "AI-GENERATED SUMMARY",
        "IMAGING ANALYSIS",
        "FINDINGS",
        "RECOMMENDED ACTIONS",
    ]
    result = {}
    for i, key in enumerate(section_keys):
        next_key = section_keys[i + 1] if i + 1 < len(section_keys) else None
        if next_key:
            pattern = rf"{re.escape(key)}\s*(.*?)\s*(?={re.escape(next_key)})"
        else:
            pattern = rf"{re.escape(key)}\s*(.*)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        result[key] = match.group(1).strip() if match else ""
    return result


# ══════════════════════════════════════════════════════════════════════════
#  PDF helpers
# ══════════════════════════════════════════════════════════════════════════

def _bullet_paragraphs(text, td_s, bullet_s):
    paragraphs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("-"):
            paragraphs.append(Paragraph(f"• {line[1:].strip()}", bullet_s))
        else:
            paragraphs.append(Paragraph(line, td_s))
    return paragraphs


# ══════════════════════════════════════════════════════════════════════════
#  PDF Report
# ══════════════════════════════════════════════════════════════════════════

def generate_report(image_path, orig_img, overlay_img, cam,
                    pred_label, confidence, save_dir, narrative_text):

    severity_score, severity_label, clinical_note = compute_severity(
        confidence, cam, pred_label
    )
    sections = parse_narrative(narrative_text)
    is_frac  = pred_label == "fractured"

    # ── Temp images ────────────────────────────────────────────────────────
    orig_tmp    = os.path.join(save_dir, "_tmp_orig.png")
    gradcam_tmp = os.path.join(save_dir, "_tmp_gradcam.png")
    cb_tmp      = os.path.join(save_dir, "_tmp_cb.png")

    orig_img.save(orig_tmp)
    Image.fromarray((overlay_img * 255).astype(np.uint8)).save(gradcam_tmp)

    fig_cb, ax_cb = plt.subplots(figsize=(4, 0.45))
    fig_cb.subplots_adjust(bottom=0.6)
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="jet"),
                 cax=ax_cb, orientation="horizontal")
    ax_cb.set_xlabel(
        "Grad-CAM Activation Intensity  (0 = low  .  1 = high)", fontsize=7)
    plt.savefig(cb_tmp, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig_cb)

    # ── Output path ────────────────────────────────────────────────────────
    filename    = os.path.splitext(os.path.basename(image_path))[0]
    report_path = os.path.join(save_dir, f"{filename}_report.pdf")

    doc = SimpleDocTemplate(
        report_path, pagesize=A4,
        rightMargin=1.8*rcm, leftMargin=1.8*rcm,
        topMargin=1.2*rcm,   bottomMargin=1.5*rcm
    )

    # ── Colour palette ─────────────────────────────────────────────────────
    ACCENT  = colors.HexColor("#8B0000") if is_frac \
              else colors.HexColor("#1A5C1A")
    ACCENT2 = colors.HexColor("#F2DEDE") if is_frac \
              else colors.HexColor("#D6EED6")
    ACCENT3 = colors.HexColor("#FFF8F8") if is_frac \
              else colors.HexColor("#F4FBF4")
    SEV_C   = {
        "SEVERE":   colors.HexColor("#8B0000"),
        "MODERATE": colors.HexColor("#CC6600"),
        "MILD":     colors.HexColor("#B8860B"),
        "MINIMAL":  colors.HexColor("#2E7D32"),
        "NONE":     colors.HexColor("#1A5C1A"),
    }.get(severity_label, ACCENT)

    WHITE  = colors.white
    DARK   = colors.HexColor("#1A1A1A")
    MID    = colors.HexColor("#444444")
    LIGHT  = colors.HexColor("#888888")
    BORDER = colors.HexColor("#CCCCCC")

    # ── Paragraph styles ───────────────────────────────────────────────────
    def ps(name, **kw):
        return ParagraphStyle(name, **kw)

    title_s  = ps("ti", fontSize=19, fontName="Helvetica-Bold",
                  textColor=ACCENT, alignment=TA_CENTER, spaceAfter=2)
    inst_s   = ps("in", fontSize=9,  fontName="Helvetica-Bold",
                  textColor=MID, alignment=TA_CENTER, spaceAfter=2)
    sub_s    = ps("su", fontSize=8,  fontName="Helvetica",
                  textColor=LIGHT, alignment=TA_CENTER, spaceAfter=8)
    sec_s    = ps("se", fontSize=11, fontName="Helvetica-Bold",
                  textColor=ACCENT, spaceBefore=10, spaceAfter=4)
    body_s   = ps("bo", fontSize=9,  fontName="Helvetica",
                  textColor=DARK, leading=14, alignment=TA_JUSTIFY)
    note_s   = ps("no", fontSize=8,  fontName="Helvetica-Oblique",
                  textColor=LIGHT, leading=12)
    alert_s  = ps("al", fontSize=10.5, fontName="Helvetica-Bold",
                  textColor=WHITE, alignment=TA_CENTER)
    label_s  = ps("lb", fontSize=8.5,  fontName="Helvetica-Bold",
                  textColor=MID, alignment=TA_CENTER)
    th_s     = ps("th", fontSize=9,    fontName="Helvetica-Bold",
                  textColor=WHITE)
    td_s     = ps("td", fontSize=8.5,  fontName="Helvetica",
                  textColor=DARK, leading=13)
    tdk_s    = ps("tk", fontSize=8.5,  fontName="Helvetica-Bold",
                  textColor=DARK, leading=13)
    bullet_s = ps("bu", fontSize=8.5,  fontName="Helvetica",
                  textColor=DARK, leading=14, leftIndent=10)
    pid_s    = ps("pi", fontSize=8.5,  fontName="Helvetica",
                  textColor=MID, alignment=TA_CENTER)

    story = []

    # ══════════════════════════════════════════════════════
    #  HEADER
    # ══════════════════════════════════════════════════════
    story.append(Paragraph(
        "AI-Assisted Bone Fracture Detection Report", title_s))

    story.append(Paragraph(
        f"Report Date: {datetime.now().strftime('%d %B %Y')}  "
        f"·  Time: {datetime.now().strftime('%H:%M:%S')}  "
        f"·  Scan Type: X-Ray  ·  Device: {config.DEVICE}",
        sub_s))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=ACCENT, spaceAfter=8))

    # ── Case info row ──────────────────────────────────────────────────────
    case_id  = f"CASE-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    case_t   = Table([[
        Paragraph(f"Case ID: {case_id}",          pid_s),
        Paragraph("Body Region: Hand / Wrist",    pid_s),
        Paragraph("Scan Type: X-Ray",             pid_s),
        Paragraph(f"Status: {pred_label.upper()}", pid_s),
    ]], colWidths=[5*rcm]*4)
    case_t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), ACCENT2),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("PADDING",    (0,0), (-1,-1), 6),
        ("GRID",       (0,0), (-1,-1), 0.4, BORDER),
    ]))
    story.append(case_t)
    story.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════
    #  RESULT BANNER
    # ══════════════════════════════════════════════════════
    banner = Table([[
        Paragraph(
            "FRACTURE DETECTED" if is_frac else "NO FRACTURE DETECTED",
            alert_s),
        Paragraph(f"Confidence<br/>{confidence*100:.2f}%",  alert_s),
        Paragraph(f"Severity<br/>{severity_label}",         alert_s),
        Paragraph(f"Score<br/>{severity_score} / 100",      alert_s),
    ]], colWidths=[6.5*rcm, 4*rcm, 4*rcm, 4*rcm])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,0),  ACCENT),
        ("BACKGROUND", (1,0), (-1,0), SEV_C),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("PADDING",    (0,0), (-1,-1), 13),
        ("LINEAFTER",  (0,0), (2,0),  1, colors.HexColor("#FFFFFF55")),
    ]))
    story.append(KeepTogether([
        Paragraph("Prediction Result", sec_s),
        banner
    ]))
    story.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════
    #  SCAN INFORMATION
    # ══════════════════════════════════════════════════════
    W, H     = orig_img.size
    scan_rows = [
        [Paragraph("Image File",   tdk_s),
         Paragraph(os.path.basename(image_path), td_s),
         Paragraph("Scan Date",    tdk_s),
         Paragraph(datetime.now().strftime("%d %B %Y"), td_s)],
        [Paragraph("Resolution",   tdk_s),
         Paragraph(f"{W} x {H} px", td_s),
         Paragraph("Body Part",    tdk_s),
         Paragraph("Hand / Wrist", td_s)],
        [Paragraph("AI Model",     tdk_s),
         Paragraph("EfficientNetB3 with Transfer Learning", td_s),
         Paragraph("Val Accuracy", tdk_s),
         Paragraph("99.88%",       td_s)],
    ]
    scan_t = Table(scan_rows,
                   colWidths=[3.2*rcm, 6.8*rcm, 3.2*rcm, 6.3*rcm])
    scan_t.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [ACCENT3, WHITE]),
        ("GRID",           (0,0), (-1,-1), 0.4, BORDER),
        ("PADDING",        (0,0), (-1,-1), 6),
        ("VALIGN",         (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(KeepTogether([
        Paragraph("Scan Information", sec_s),
        scan_t
    ]))
    story.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════
    #  AI-GENERATED SUMMARY
    # ══════════════════════════════════════════════════════
    summary_text = sections.get("AI-GENERATED SUMMARY", "")
    if summary_text:
        summary_box = Table(
            [[Paragraph(summary_text.replace("\n", " "), body_s)]],
            colWidths=[19.5*rcm]
        )
        summary_box.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,-1), ACCENT3),
            ("BOX",         (0,0), (-1,-1), 1.2, ACCENT),
            ("PADDING",     (0,0), (-1,-1), 10),
            ("LEFTPADDING", (0,0), (-1,-1), 12),
        ]))
        story.append(KeepTogether([
            Paragraph("AI-Generated Summary", sec_s),
            summary_box
        ]))
        story.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════
    #  IMAGING ANALYSIS (side-by-side images)
    # ══════════════════════════════════════════════════════
    iw = ih = 8.8 * rcm

    img_hdr = Table([[
        Paragraph("Original X-Ray",    label_s),
        Paragraph("Grad-CAM Heatmap",  label_s),
    ]], colWidths=[iw + 0.4*rcm, iw + 0.4*rcm])
    img_hdr.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), ACCENT2),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("PADDING",    (0,0), (-1,-1), 6),
        ("LINEBELOW",  (0,0), (-1,-1), 0.5, BORDER),
    ]))

    img_body = Table([[
        RLImage(orig_tmp,    width=iw, height=ih),
        RLImage(gradcam_tmp, width=iw, height=ih),
    ]], colWidths=[iw + 0.4*rcm, iw + 0.4*rcm])
    img_body.setStyle(TableStyle([
        ("ALIGN",     (0,0), (-1,-1), "CENTER"),
        ("VALIGN",    (0,0), (-1,-1), "MIDDLE"),
        ("PADDING",   (0,0), (-1,-1), 3),
        ("LINEAFTER", (0,0), (0,-1),  0.5, BORDER),
    ]))

    cb_row = Table([[
        Paragraph("", body_s),
        RLImage(cb_tmp, width=iw - 1*rcm, height=0.85*rcm),
    ]], colWidths=[iw + 0.4*rcm, iw + 0.4*rcm])
    cb_row.setStyle(TableStyle([
        ("ALIGN",   (0,0), (-1,-1), "CENTER"),
        ("VALIGN",  (0,0), (-1,-1), "MIDDLE"),
        ("PADDING", (0,0), (-1,-1), 3),
    ]))

    outer = Table(
        [[img_hdr], [img_body], [cb_row]],
        colWidths=[(iw + 0.4*rcm) * 2]
    )
    outer.setStyle(TableStyle([
        ("BOX",     (0,0), (-1,-1), 0.8, BORDER),
        ("PADDING", (0,0), (-1,-1), 0),
    ]))

    imaging_text = sections.get("IMAGING ANALYSIS", "")
    story.append(KeepTogether([
        Paragraph("Imaging Analysis", sec_s),
        outer,
        Spacer(1, 5),
        Paragraph(imaging_text.replace("\n", " "), note_s)
        if imaging_text else Spacer(1, 1),
    ]))
    story.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════
    #  FINDINGS
    # ══════════════════════════════════════════════════════
    findings_text = sections.get("FINDINGS", "")
    if findings_text:
        findings_items = _bullet_paragraphs(findings_text, td_s, bullet_s)
        findings_box   = Table(
            [[item] for item in findings_items],
            colWidths=[19.5*rcm]
        )
        findings_box.setStyle(TableStyle([
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [ACCENT3, WHITE]),
            ("GRID",           (0,0), (-1,-1), 0.3, BORDER),
            ("PADDING",        (0,0), (-1,-1), 7),
            ("LEFTPADDING",    (0,0), (-1,-1), 12),
        ]))
        story.append(KeepTogether([
            Paragraph("Findings", sec_s),
            findings_box
        ]))
        story.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════
    #  GRAD-CAM METRICS TABLE
    # ══════════════════════════════════════════════════════
    peak_act   = round(float(np.max(cam)) * 100, 2)
    mean_act   = round(float(np.mean(cam)) * 100, 2)
    spread_pct = round(np.sum(cam > 0.5) / cam.size * 100, 2)

    gc_rows = [
        [Paragraph("Metric",               th_s),
         Paragraph("Value",                th_s),
         Paragraph("Interpretation",       th_s)],
        [Paragraph("Peak Activation",      tdk_s),
         Paragraph(f"{peak_act}%",         td_s),
         Paragraph("Maximum focus intensity — higher means more localised", td_s)],
        [Paragraph("Mean Activation",      tdk_s),
         Paragraph(f"{mean_act}%",         td_s),
         Paragraph("Average attention spread across entire image",         td_s)],
        [Paragraph("High Activation Area", tdk_s),
         Paragraph(f"{spread_pct}%",       td_s),
         Paragraph("Proportion of image with activation above 50%",        td_s)],
        [Paragraph("Model Confidence",     tdk_s),
         Paragraph(f"{confidence*100:.2f}%", td_s),
         Paragraph("Softmax probability assigned to predicted class",      td_s)],
        [Paragraph("Severity Score",       tdk_s),
         Paragraph(f"{severity_score}/100", td_s),
         Paragraph(f"{severity_label} — {clinical_note}",                  td_s)],
    ]
    gc_t = Table(gc_rows, colWidths=[4.5*rcm, 2.8*rcm, 12.2*rcm])
    gc_t.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0),  ACCENT),
        ("TEXTCOLOR",      (0,0), (-1,0),  WHITE),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [ACCENT3, WHITE]),
        ("GRID",           (0,0), (-1,-1), 0.4, BORDER),
        ("PADDING",        (0,0), (-1,-1), 7),
        ("VALIGN",         (0,0), (-1,-1), "TOP"),
    ]))
    story.append(KeepTogether([
        Paragraph("Grad-CAM Metrics", sec_s),
        gc_t
    ]))
    story.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════
    #  SEVERITY BREAKDOWN (fractured only)
    # ══════════════════════════════════════════════════════
    if is_frac:
        sev_rows = [
            [Paragraph("Component",         th_s),
             Paragraph("Points Scored",     th_s),
             Paragraph("Max Points",        th_s),
             Paragraph("Weight",            th_s)],
            [Paragraph("Model Confidence",  tdk_s),
             Paragraph(f"{round(confidence*60,1)}", td_s),
             Paragraph("60", td_s),
             Paragraph("60%", td_s)],
            [Paragraph("Activation Spread", tdk_s),
             Paragraph(f"{round(min(spread_pct/100*200,25),1)}", td_s),
             Paragraph("25", td_s),
             Paragraph("25%", td_s)],
            [Paragraph("Peak Intensity",    tdk_s),
             Paragraph(f"{round(float(np.max(cam))*15,1)}", td_s),
             Paragraph("15", td_s),
             Paragraph("15%", td_s)],
            [Paragraph("TOTAL",             tdk_s),
             Paragraph(f"{severity_score}", td_s),
             Paragraph("100", td_s),
             Paragraph("", td_s)],
        ]
        sev_t = Table(sev_rows,
                      colWidths=[8.5*rcm, 3.5*rcm, 3.0*rcm, 4.5*rcm])
        sev_t.setStyle(TableStyle([
            ("BACKGROUND",     (0,0),  (-1,0),  ACCENT),
            ("TEXTCOLOR",      (0,0),  (-1,0),  WHITE),
            ("BACKGROUND",     (0,-1), (-1,-1), ACCENT2),
            ("FONTNAME",       (0,-1), (-1,-1), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0,1),  (-1,-2), [ACCENT3, WHITE]),
            ("GRID",           (0,0),  (-1,-1), 0.4, BORDER),
            ("ALIGN",          (1,0),  (-1,-1), "CENTER"),
            ("PADDING",        (0,0),  (-1,-1), 7),
        ]))
        story.append(KeepTogether([
            Paragraph("Severity Score Breakdown", sec_s),
            sev_t
        ]))
        story.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════
    #  CLINICAL ASSESSMENT
    # ══════════════════════════════════════════════════════
    clin_rows = [
        [Paragraph("Parameter",            th_s),
         Paragraph("AI Finding",           th_s),
         Paragraph("Clinical Notes",       th_s)],
        [Paragraph("Fracture Type",        tdk_s),
         Paragraph("Closed / Undetermined",td_s),
         Paragraph("Open vs closed requires physical examination",           td_s)],
        [Paragraph("Location",             tdk_s),
         Paragraph("See heatmap",          td_s),
         Paragraph("Activation zone marks the suspected fracture region",    td_s)],
        [Paragraph("Fracture Pattern",     tdk_s),
         Paragraph("Not determined",       td_s),
         Paragraph("Radiologist review required for pattern classification", td_s)],
        [Paragraph("Alignment",            tdk_s),
         Paragraph("Not quantified",       td_s),
         Paragraph("CT recommended for displacement measurement",            td_s)],
        [Paragraph("Healing Status",       tdk_s),
         Paragraph("Acute (presumed)",     td_s),
         Paragraph("Single time-point image — longitudinal assessment needed", td_s)],
        [Paragraph("Recommended Action",   tdk_s),
         Paragraph(clinical_note,          td_s),
         Paragraph(f"Severity: {severity_score}/100  ({severity_label})",   td_s)],
    ]
    clin_t = Table(clin_rows, colWidths=[4.0*rcm, 5.0*rcm, 10.5*rcm])
    clin_t.setStyle(TableStyle([
        ("BACKGROUND",     (0,0),  (-1,0),  ACCENT),
        ("TEXTCOLOR",      (0,0),  (-1,0),  WHITE),
        ("ROWBACKGROUNDS", (0,1),  (-1,-1), [ACCENT3, WHITE]),
        ("BACKGROUND",     (0,-1), (-1,-1), ACCENT2),
        ("FONTNAME",       (0,-1), (-1,-1), "Helvetica-Bold"),
        ("GRID",           (0,0),  (-1,-1), 0.4, BORDER),
        ("PADDING",        (0,0),  (-1,-1), 7),
        ("VALIGN",         (0,0),  (-1,-1), "TOP"),
    ]))
    story.append(KeepTogether([
        Paragraph("Clinical Fracture Assessment", sec_s),
        clin_t
    ]))
    story.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════
    #  RECOMMENDED ACTIONS
    # ══════════════════════════════════════════════════════
    actions_text = sections.get("RECOMMENDED ACTIONS", "")
    if actions_text:
        action_items = _bullet_paragraphs(actions_text, td_s, bullet_s)
        actions_box  = Table(
            [[item] for item in action_items],
            colWidths=[19.5*rcm]
        )
        actions_box.setStyle(TableStyle([
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [ACCENT3, WHITE]),
            ("GRID",           (0,0), (-1,-1), 0.3, BORDER),
            ("PADDING",        (0,0), (-1,-1), 7),
            ("LEFTPADDING",    (0,0), (-1,-1), 12),
        ]))
        story.append(KeepTogether([
            Paragraph("Recommended Actions", sec_s),
            actions_box
        ]))
        story.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════
    #  LIMITATIONS
    # ══════════════════════════════════════════════════════
    lim_items = [
        "This system analyses a single X-ray image and cannot assess clinical "
        "history, symptoms, or physical examination findings.",
        "The AI model cannot detect soft tissue injuries, ligamentous damage, "
        "or neurovascular complications.",
        "Fracture type (transverse, oblique, comminuted) and exact displacement "
        "cannot be determined from model output alone.",
        "Model performance may vary across different X-ray equipment, image "
        "quality, and patient populations not represented in the training dataset.",
        "This tool is intended to assist, not replace, a qualified radiologist "
        "or orthopaedic surgeon.",
    ]
    lim_box = Table(
        [[Paragraph(f"• {t}", bullet_s)] for t in lim_items],
        colWidths=[19.5*rcm]
    )
    lim_box.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0,0), (-1,-1),
         [colors.HexColor("#FAFAFA"), WHITE]),
        ("GRID",           (0,0), (-1,-1), 0.3, BORDER),
        ("PADDING",        (0,0), (-1,-1), 7),
        ("LEFTPADDING",    (0,0), (-1,-1), 12),
    ]))
    story.append(KeepTogether([
        Paragraph("Limitations", sec_s),
        lim_box
    ]))
    story.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════
    #  DISCLAIMER
    # ══════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=0.8,
                             color=BORDER, spaceAfter=6))
    disc_box = Table([[Paragraph(
        "<b>DISCLAIMER</b><br/>"
        "This report is generated by an AI-based system for research and "
        "educational purposes only and must not be used as a substitute for "
        "professional medical diagnosis or clinical decision-making. All "
        "findings, predictions, and recommendations presented herein are based "
        "solely on the submitted image and must be reviewed and confirmed by a "
        "qualified radiologist or orthopaedic surgeon before any clinical "
        "action is taken. The developers and institution assume no liability "
        "for decisions made on the basis of this report.",
        note_s)
    ]], colWidths=[19.5*rcm])
    disc_box.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#F5F5F5")),
        ("BOX",        (0,0), (-1,-1), 0.5, BORDER),
        ("PADDING",    (0,0), (-1,-1), 9),
    ]))
    story.append(disc_box)

    doc.build(story)

    for f in [orig_tmp, gradcam_tmp, cb_tmp]:
        if os.path.exists(f):
            os.remove(f)

    return report_path


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    print(f"\nDevice : {config.DEVICE}")
    print("=" * 50)

    print("\nLoading model...")
    model      = build_model(freeze_backbone=False).to(config.DEVICE)
    checkpoint = torch.load(
        f"{config.CHECKPOINT_DIR}/best_model.pth",
        map_location=config.DEVICE
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded from epoch   : {checkpoint['epoch']}")
    print(f"  Best Val Accuracy   : {checkpoint['val_acc']*100:.2f}%")

    image_path = input("\nEnter full path to X-ray image: ").strip()
    if not os.path.exists(image_path):
        print(f"\n  File not found: {image_path}")
        return

    gradcam          = GradCAM(model)
    tensor, orig_img = preprocess(image_path)
    cam, pred_idx, confidence = gradcam.generate(tensor)

    pred_label = config.CLASS_NAMES[pred_idx]
    overlay    = overlay_cam(orig_img, cam)
    color      = "red" if pred_label == "fractured" else "green"

    severity_score, severity_label, clinical_note = compute_severity(
        confidence, cam, pred_label
    )

    print(f"\n{'='*50}")
    print(f"  Prediction     : {pred_label.upper()}")
    print(f"  Confidence     : {confidence*100:.2f}%")
    print(f"  Severity Score : {severity_score}/100  ({severity_label})")
    print(f"  Clinical Note  : {clinical_note}")
    print(f"{'='*50}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(orig_img)
    axes[0].set_title("Original X-Ray", fontsize=13)
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Grad-CAM Heatmap\n{pred_label.upper()}  -  {confidence*100:.2f}%",
        fontsize=13, color=color, fontweight="bold"
    )
    axes[1].axis("off")
    plt.suptitle(
        f"Result: {pred_label.upper()}  |  Confidence: {confidence*100:.2f}%",
        fontsize=15, color=color, fontweight="bold"
    )
    plt.tight_layout()

    os.makedirs("outputs/predictions", exist_ok=True)
    filename  = os.path.splitext(os.path.basename(image_path))[0]
    save_path = f"outputs/predictions/{filename}_result.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  Result saved  -> {save_path}")

    # ── Gemini narrative + PDF ────────────────────────────────────────────
    print("\n  Generating AI narrative via Gemini API...")
    narrative = generate_narrative(
        pred_label, confidence, severity_score,
        severity_label, cam, image_path
    )

    report_path = generate_report(
        image_path, orig_img, overlay, cam,
        pred_label, confidence,
        save_dir="outputs/predictions",
        narrative_text=narrative
    )
    print(f"  Report saved  -> {report_path}")


if __name__ == "__main__":
    main()
    
