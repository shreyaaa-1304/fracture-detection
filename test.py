# test.py
import warnings
warnings.filterwarnings("ignore")

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageFile
from torchvision import transforms
from datetime import datetime

ImageFile.LOAD_TRUNCATED_IMAGES = True

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm as rcm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image as RLImage, Table, TableStyle,
                                 HRFlowable, KeepTogether)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT

import config
from model import build_model


# ══════════════════════════════════════════════════════════════════════════
#  Grad-CAM (unchanged)
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
#  Helpers (unchanged)
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
#  FIX 1 — Severity now checks pred_label first
# ══════════════════════════════════════════════════════════════════════════

def compute_severity(confidence, cam, pred_label):
    """
    If NOT FRACTURED  → severity is always NONE regardless of confidence.
    If FRACTURED      → score 0-100 from confidence + CAM spread + peak.
    """
    # ── FIX: normal image = no severity ───────────────────────────────────
    if pred_label != "fractured":
        return 0.0, "NONE", "No fracture detected. No clinical action required."

    conf_score   = confidence * 60
    spread       = np.sum(cam > 0.5) / cam.size
    spread_score = min(spread * 200, 25)
    peak_score   = float(np.max(cam)) * 15
    total        = round(conf_score + spread_score + peak_score, 2)

    if total >= 80:
        sev_label = "SEVERE"
        note      = "Immediate orthopaedic consultation recommended."
    elif total >= 55:
        sev_label = "MODERATE"
        note      = "Further imaging (CT / MRI) advised."
    elif total >= 30:
        sev_label = "MILD"
        note      = "Conservative management may be appropriate. Monitor closely."
    else:
        sev_label = "MINIMAL"
        note      = "Low-grade fracture signal. Clinical review advised."

    return total, sev_label, note


# ══════════════════════════════════════════════════════════════════════════
#  FIX 2 — Better PDF report
# ══════════════════════════════════════════════════════════════════════════

def generate_report(image_path, orig_img, overlay_img, cam,
                    pred_label, confidence, save_dir):

    severity_score, severity_label, clinical_note = compute_severity(
        confidence, cam, pred_label
    )

    is_fractured = (pred_label == "fractured")

    # ── Save temp images ───────────────────────────────────────────────────
    orig_tmp     = os.path.join(save_dir, "_tmp_orig.png")
    gradcam_tmp  = os.path.join(save_dir, "_tmp_gradcam.png")
    colorbar_tmp = os.path.join(save_dir, "_tmp_cb.png")

    orig_img.save(orig_tmp)
    Image.fromarray((overlay_img * 255).astype(np.uint8)).save(gradcam_tmp)

    fig_cb, ax_cb = plt.subplots(figsize=(4, 0.45))
    fig_cb.subplots_adjust(bottom=0.6)
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1), cmap="jet"),
                 cax=ax_cb, orientation="horizontal")
    ax_cb.set_xlabel("Grad-CAM Activation Intensity  (0 = low · 1 = high)",
                     fontsize=7)
    plt.savefig(colorbar_tmp, bbox_inches="tight", dpi=150,
                facecolor="white")
    plt.close(fig_cb)

    # ── Output path ────────────────────────────────────────────────────────
    filename    = os.path.splitext(os.path.basename(image_path))[0]
    report_path = os.path.join(save_dir, f"{filename}_report.pdf")

    doc = SimpleDocTemplate(
        report_path, pagesize=A4,
        rightMargin=1.8*rcm, leftMargin=1.8*rcm,
        topMargin=1.2*rcm,   bottomMargin=1.5*rcm
    )

    # ── Color palette ──────────────────────────────────────────────────────
    if is_fractured:
        ACCENT  = colors.HexColor("#8B0000")   # dark red
        ACCENT2 = colors.HexColor("#F2DEDE")
        ACCENT3 = colors.HexColor("#FFF8F8")
    else:
        ACCENT  = colors.HexColor("#1A5C1A")   # dark green
        ACCENT2 = colors.HexColor("#D6EED6")
        ACCENT3 = colors.HexColor("#F4FBF4")

    SEV_COLORS = {
        "SEVERE":   colors.HexColor("#8B0000"),
        "MODERATE": colors.HexColor("#CC6600"),
        "MILD":     colors.HexColor("#B8860B"),
        "MINIMAL":  colors.HexColor("#2E7D32"),
        "NONE":     colors.HexColor("#1A5C1A"),
    }
    sev_color = SEV_COLORS.get(severity_label, ACCENT)

    DARK  = colors.HexColor("#1A1A1A")
    MID   = colors.HexColor("#444444")
    LIGHT = colors.HexColor("#888888")
    WHITE = colors.white
    BORDER= colors.HexColor("#CCCCCC")
    BG    = colors.HexColor("#F9F9F9")

    # ── Paragraph styles ───────────────────────────────────────────────────
    def ps(name, **kw):
        return ParagraphStyle(name, **kw)

    title_s  = ps("ti", fontSize=20, fontName="Helvetica-Bold",
                  textColor=ACCENT, alignment=TA_CENTER, spaceAfter=2)
    sub_s    = ps("su", fontSize=8.5, fontName="Helvetica",
                  textColor=LIGHT, alignment=TA_CENTER, spaceAfter=2)
    hosp_s   = ps("ho", fontSize=9, fontName="Helvetica-Bold",
                  textColor=MID, alignment=TA_CENTER, spaceAfter=8)
    sec_s    = ps("se", fontSize=11, fontName="Helvetica-Bold",
                  textColor=ACCENT, spaceBefore=10, spaceAfter=4,
                  borderPadding=(0,0,2,0))
    body_s   = ps("bo", fontSize=9, fontName="Helvetica",
                  textColor=DARK, leading=14, alignment=TA_LEFT)
    note_s   = ps("no", fontSize=8, fontName="Helvetica-Oblique",
                  textColor=LIGHT, leading=12)
    alert_s  = ps("al", fontSize=10.5, fontName="Helvetica-Bold",
                  textColor=WHITE, alignment=TA_CENTER)
    label_s  = ps("lb", fontSize=8.5, fontName="Helvetica-Bold",
                  textColor=MID, alignment=TA_CENTER)
    th_s     = ps("th", fontSize=9, fontName="Helvetica-Bold",
                  textColor=WHITE)
    td_s     = ps("td", fontSize=8.5, fontName="Helvetica",
                  textColor=DARK, leading=12)
    tdk_s    = ps("tk", fontSize=8.5, fontName="Helvetica-Bold",
                  textColor=DARK, leading=12)

    story = []

    # ══════════════════════════════════════════════════════
    #  HEADER BLOCK
    # ══════════════════════════════════════════════════════
    story.append(Paragraph(
        "AI-Assisted Bone Fracture Detection Report", title_s))
    story.append(Paragraph(
        "K J Somaiya College of Engineering  •  Fracture Detection System",
        hosp_s))
    story.append(Paragraph(
        f"Report generated: {datetime.now().strftime('%d %B %Y  at  %H:%M:%S')}  "
        f"·  Model: EfficientNetB3  ·  Device: {config.DEVICE}",
        sub_s))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=ACCENT, spaceAfter=10))

    # ══════════════════════════════════════════════════════
    #  RESULT BANNER  (big coloured box)
    # ══════════════════════════════════════════════════════
    banner_bg = ACCENT
    sev_bg    = sev_color

    banner = Table([[
        Paragraph(f"{'✔' if not is_fractured else '⚠'}  {pred_label.upper()}", alert_s),
        Paragraph(f"Confidence<br/>{confidence*100:.2f}%", alert_s),
        Paragraph(f"Severity<br/>{severity_label}", alert_s),
        Paragraph(f"Score<br/>{severity_score}/100", alert_s),
    ]], colWidths=[6.5*rcm, 4*rcm, 4*rcm, 4*rcm])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,0), banner_bg),
        ("BACKGROUND", (1,0), (-1,0), sev_bg),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("PADDING",    (0,0), (-1,-1), 12),
        ("LINEAFTER",  (0,0), (2,0), 1, colors.HexColor("#FFFFFF44")),
        ("ROUNDEDCORNERS", [5]),
    ]))
    story.append(KeepTogether([
        Paragraph("Prediction Result", sec_s),
        banner
    ]))
    story.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════
    #  SCAN INFORMATION
    # ══════════════════════════════════════════════════════
    W, H = orig_img.size
    scan_rows = [
        [Paragraph("Image File",  tdk_s), Paragraph(os.path.basename(image_path), td_s),
         Paragraph("Scan Date",   tdk_s), Paragraph(datetime.now().strftime("%d-%m-%Y"), td_s)],
        [Paragraph("Resolution",  tdk_s), Paragraph(f"{W} × {H} px", td_s),
         Paragraph("Body Region", tdk_s), Paragraph("Hand / Wrist (from X-ray)", td_s)],
        [Paragraph("Framework",   tdk_s), Paragraph("PyTorch 2.x", td_s),
         Paragraph("Architecture",tdk_s), Paragraph("EfficientNetB3 + Transfer Learning", td_s)],
        [Paragraph("Checkpoint",  tdk_s), Paragraph("best_model.pth  (Epoch 23)", td_s),
         Paragraph("Val Accuracy",tdk_s), Paragraph("99.88%", td_s)],
    ]
    scan_t = Table(scan_rows,
                   colWidths=[3.2*rcm, 6.8*rcm, 3.2*rcm, 6.3*rcm])
    scan_t.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,-1), ACCENT3),
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
    #  IMAGING  (side-by-side)
    # ══════════════════════════════════════════════════════
    iw = ih = 8.8 * rcm

    img_header = Table([[
        Paragraph("Original X-Ray", label_s),
        Paragraph("Grad-CAM Heatmap", label_s),
    ]], colWidths=[iw + 0.4*rcm, iw + 0.4*rcm])
    img_header.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), ACCENT2),
        ("PADDING",    (0,0), (-1,-1), 6),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("LINEBELOW",  (0,0), (-1,-1), 0.5, BORDER),
    ]))

    img_body = Table([[
        RLImage(orig_tmp,    width=iw, height=ih),
        RLImage(gradcam_tmp, width=iw, height=ih),
    ]], colWidths=[iw + 0.4*rcm, iw + 0.4*rcm])
    img_body.setStyle(TableStyle([
        ("ALIGN",   (0,0), (-1,-1), "CENTER"),
        ("VALIGN",  (0,0), (-1,-1), "MIDDLE"),
        ("PADDING", (0,0), (-1,-1), 3),
        ("LINEAFTER", (0,0), (0,-1), 0.5, BORDER),
    ]))

    cb_row = Table([[
        Paragraph("", body_s),
        RLImage(colorbar_tmp, width=iw - 1*rcm, height=0.85*rcm),
    ]], colWidths=[iw + 0.4*rcm, iw + 0.4*rcm])
    cb_row.setStyle(TableStyle([
        ("ALIGN",   (0,0), (-1,-1), "CENTER"),
        ("VALIGN",  (0,0), (-1,-1), "MIDDLE"),
        ("PADDING", (0,0), (-1,-1), 3),
    ]))

    outer = Table(
        [[img_header], [img_body], [cb_row]],
        colWidths=[(iw + 0.4*rcm) * 2]
    )
    outer.setStyle(TableStyle([
        ("BOX",    (0,0), (-1,-1), 0.8, BORDER),
        ("PADDING",(0,0), (-1,-1), 0),
    ]))

    story.append(KeepTogether([
        Paragraph("Imaging Analysis", sec_s),
        outer,
        Spacer(1, 4),
        Paragraph(
            "🔴 Red / Yellow = High activation (region model focused on)   "
            "🔵 Blue = Low activation (background / irrelevant region)",
            note_s),
    ]))
    story.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════
    #  GRAD-CAM METRICS
    # ══════════════════════════════════════════════════════
    peak_act   = round(float(np.max(cam)) * 100, 2)
    mean_act   = round(float(np.mean(cam)) * 100, 2)
    spread_pct = round(np.sum(cam > 0.5) / cam.size * 100, 2)

    gc_rows = [
        [Paragraph("Metric", th_s),
         Paragraph("Value",  th_s),
         Paragraph("Interpretation", th_s)],
        [Paragraph("Peak Activation",      tdk_s),
         Paragraph(f"{peak_act}%",         td_s),
         Paragraph("Maximum focus intensity — higher = more localised",  td_s)],
        [Paragraph("Mean Activation",      tdk_s),
         Paragraph(f"{mean_act}%",         td_s),
         Paragraph("Average attention spread across entire image",       td_s)],
        [Paragraph("High Activation Area", tdk_s),
         Paragraph(f"{spread_pct}%",       td_s),
         Paragraph("Proportion of image with activation above 50%",     td_s)],
        [Paragraph("Model Confidence",     tdk_s),
         Paragraph(f"{confidence*100:.2f}%", td_s),
         Paragraph("Softmax probability assigned to predicted class",    td_s)],
        [Paragraph("Severity Score",       tdk_s),
         Paragraph(f"{severity_score}/100",td_s),
         Paragraph(f"{severity_label} — {clinical_note}",               td_s)],
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
        Paragraph("Grad-CAM Interpretation", sec_s),
        gc_t
    ]))
    story.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════
    #  CLINICAL ASSESSMENT
    # ══════════════════════════════════════════════════════
    clin_rows = [
        [Paragraph("Parameter",           th_s),
         Paragraph("AI Finding",          th_s),
         Paragraph("Clinical Notes",      th_s)],
        [Paragraph("Fracture Type",       tdk_s),
         Paragraph("Closed / Undetermined", td_s),
         Paragraph("Open vs closed distinction requires physical examination", td_s)],
        [Paragraph("Location",            tdk_s),
         Paragraph("See heatmap (highlighted zone)", td_s),
         Paragraph("Grad-CAM activation marks the suspected fracture region", td_s)],
        [Paragraph("Fracture Pattern",    tdk_s),
         Paragraph("Not determined",      td_s),
         Paragraph("Transverse / oblique / comminuted — needs radiologist review", td_s)],
        [Paragraph("Alignment",           tdk_s),
         Paragraph("Not quantified",      td_s),
         Paragraph("CT scan recommended for displacement measurement (mm)", td_s)],
        [Paragraph("Associated Injuries", tdk_s),
         Paragraph("Not assessed",        td_s),
         Paragraph("Ligamentous damage and neurovascular status require clinical exam", td_s)],
        [Paragraph("Healing Status",      tdk_s),
         Paragraph("Acute (presumed)",    td_s),
         Paragraph("Assessment based on single time-point image only", td_s)],
        [Paragraph("Gustilo Grade",       tdk_s),
         Paragraph("N/A",                 td_s),
         Paragraph("Applicable only to open fractures confirmed clinically", td_s)],
        [Paragraph("Recommended Action",  tdk_s),
         Paragraph(clinical_note,         td_s),
         Paragraph(f"Severity score: {severity_score}/100  ({severity_label})", td_s)],
    ]
    clin_t = Table(clin_rows, colWidths=[4.0*rcm, 5.0*rcm, 10.5*rcm])
    clin_t.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0),  ACCENT),
        ("TEXTCOLOR",      (0,0), (-1,0),  WHITE),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [ACCENT3, WHITE]),
        ("GRID",           (0,0), (-1,-1), 0.4, BORDER),
        ("PADDING",        (0,0), (-1,-1), 7),
        ("VALIGN",         (0,0), (-1,-1), "TOP"),
        # highlight recommended action row
        ("BACKGROUND",     (0,-1), (-1,-1), ACCENT2),
        ("FONTNAME",       (0,-1), (-1,-1), "Helvetica-Bold"),
    ]))
    story.append(KeepTogether([
        Paragraph("Clinical Fracture Assessment", sec_s),
        clin_t
    ]))
    story.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════
    #  SEVERITY BREAKDOWN  (only for fractured)
    # ══════════════════════════════════════════════════════
    if is_fractured:
        sev_rows = [
            [Paragraph("Component",              th_s),
             Paragraph("Points Scored",          th_s),
             Paragraph("Max Points",             th_s),
             Paragraph("Weight",                 th_s)],
            [Paragraph("Model Confidence",       tdk_s),
             Paragraph(f"{round(confidence*60,1)}", td_s),
             Paragraph("60",                     td_s),
             Paragraph("60%",                    td_s)],
            [Paragraph("Activation Spread",      tdk_s),
             Paragraph(f"{round(min(spread_pct/100*200,25),1)}", td_s),
             Paragraph("25",                     td_s),
             Paragraph("25%",                    td_s)],
            [Paragraph("Peak Intensity",         tdk_s),
             Paragraph(f"{round(float(np.max(cam))*15,1)}", td_s),
             Paragraph("15",                     td_s),
             Paragraph("15%",                    td_s)],
            [Paragraph("TOTAL SEVERITY SCORE",   tdk_s),
             Paragraph(f"{severity_score}",      td_s),
             Paragraph("100",                    td_s),
             Paragraph("",                       td_s)],
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
    #  FOOTER / DISCLAIMER
    # ══════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=0.8,
                             color=BORDER, spaceAfter=5))
    disc_t = Table([[
        Paragraph(
            "<b>⚠  DISCLAIMER</b><br/>"
            "This report is generated by an AI research system and is intended "
            "for educational and research purposes only. It does <b>not</b> "
            "constitute a medical diagnosis. All findings must be reviewed and "
            "confirmed by a qualified radiologist or orthopaedic surgeon before "
            "any clinical decision is made.",
            note_s),
    ]], colWidths=[19.5*rcm])
    disc_t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#F5F5F5")),
        ("BOX",        (0,0), (-1,-1), 0.5, BORDER),
        ("PADDING",    (0,0), (-1,-1), 8),
    ]))
    story.append(disc_t)

    doc.build(story)

    for f in [orig_tmp, gradcam_tmp, colorbar_tmp]:
        if os.path.exists(f):
            os.remove(f)

    return report_path, severity_score, severity_label, clinical_note


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
    print(f"  ✔ Loaded from epoch   : {checkpoint['epoch']}")
    print(f"  ✔ Best Val Accuracy   : {checkpoint['val_acc']*100:.2f}%")

    image_path = input("\nEnter full path to X-ray image: ").strip()
    if not os.path.exists(image_path):
        print(f"\n  ❌ File not found: {image_path}")
        return

    gradcam          = GradCAM(model)
    tensor, orig_img = preprocess(image_path)
    cam, pred_idx, confidence = gradcam.generate(tensor)

    pred_label = config.CLASS_NAMES[pred_idx]
    overlay    = overlay_cam(orig_img, cam)
    color      = "red" if pred_label == "fractured" else "green"

    # ── FIX: pass pred_label to severity ──────────────────────────────────
    severity_score, severity_label, clinical_note = compute_severity(
        confidence, cam, pred_label
    )

    print(f"\n{'='*50}")
    print(f"  Prediction     : {pred_label.upper()}")
    print(f"  Confidence     : {confidence*100:.2f}%")
    print(f"  Severity Score : {severity_score}/100  ({severity_label})")
    print(f"  Clinical Note  : {clinical_note}")
    print(f"{'='*50}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(orig_img)
    axes[0].set_title("Original X-Ray", fontsize=13)
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Grad-CAM Heatmap\n{pred_label.upper()}  —  {confidence*100:.2f}%",
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
    print(f"\n  ✔ Result saved  → {save_path}")

    report_path, *_ = generate_report(
        image_path, orig_img, overlay, cam,
        pred_label, confidence,
        save_dir="outputs/predictions"
    )
    print(f"  ✔ Report saved  → {report_path}")


if __name__ == "__main__":
    main()
