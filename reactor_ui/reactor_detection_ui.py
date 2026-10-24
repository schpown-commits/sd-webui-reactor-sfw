import gradio as gr
from scripts.reactor_swapper import (
    clear_faces,
    clear_faces_list,
    clear_faces_target,
    clear_faces_all
)

# TAB DETECTION & MASKING
def show(show_br: bool = True):
    with gr.Tab("Detection & Masking"):
        with gr.Group():
            gr.Markdown("<div style='font-size: 0.9em; font-weight: bold; margin-bottom: 4px;'>🔍 Gesichtserkennung (Detection Engine)</div>")
            with gr.Row():
                det_thresh = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.5,
                    step=0.01,
                    label="Erkennungs-Schwellenwert (Detection Threshold)",
                    info="Höhere Werte verlangen stärkere Gesichtsmerkmale (0.5 Standard)",
                    scale=2
                )
                det_maxnum = gr.Slider(
                    minimum=0,
                    maximum=20,
                    value=0,
                    step=1,
                    label="Max. Gesichter (0 = Unbegrenzt)",
                    info="Maximale Anzahl an Gesichtern, die im Bild analysiert werden",
                    scale=1
                )
            with gr.Row():
                face_sort_mode = gr.Dropdown(
                    choices=["Left-to-Right", "Right-to-Left", "Top-to-Bottom", "Bottom-to-Top", "Largest First", "Confidence First"],
                    value="Left-to-Right",
                    label="Sortier-Reihenfolge der Gesichter (Sort Mode)",
                    info="Bestimmt, welches Gesicht Index 0, 1, 2 usw. erhält (Links➔Rechts ist Standard)",
                    scale=2
                )
                color_correction = gr.Checkbox(
                    value=True,
                    label="Skin Tone & Lighting Match (Color Match)",
                    info="Gleicht Teint, Farbtemperatur & Lichtverhältnisse automatisch an das Zielbild an",
                    scale=2
                )

        gr.Markdown("<hr style='margin: 10px 0; border: none; border-top: 1px solid rgba(255,255,255,0.08);'>")

        with gr.Group():
            gr.Markdown("<div style='font-size: 0.9em; font-weight: bold; margin-bottom: 4px;'>🎭 Maskierungs-Feinschliff (BiSeNet Mask Tuning)</div>")
            with gr.Row():
                mask_dilation = gr.Slider(
                    minimum=-20,
                    maximum=20,
                    value=0,
                    step=1,
                    label="Maskenausdehnung (Dilation / Erosion)",
                    info="Positiv = Maske erweitern, Negativ = Maske schrumpfen (0 = Standard)",
                    scale=1
                )
                mask_blur = gr.Slider(
                    minimum=0,
                    maximum=30,
                    value=12,
                    step=1,
                    label="Kanten-Weichzeichnung (Mask Feathering / Blur)",
                    info="Weichzeichnungsradius für nahtlosen Übergang (12px = Empfohlen)",
                    scale=1
                )

        gr.Markdown("<hr style='margin: 10px 0; border: none; border-top: 1px solid rgba(255,255,255,0.08);'>")

        with gr.Group():
            with gr.Row():
                gr.Markdown(
                    "<div style='font-size:0.85em; color:var(--text-muted,#888);'>"
                    "<b>⚡ Cache-Info:</b> Erkannte Gesichter werden gehasht, um nachfolgende Swaps zu beschleunigen. Leere den Cache, wenn ein neues Gesicht nicht übernommen wird."
                    "</div>",
                    scale=3
                )
                imgs_hash_clear_all = gr.Button(
                    value="🧹 Gesamten Cache leeren",
                    variant="primary",
                    scale=1
                )

            with gr.Accordion("Erweiterte Cache-Steuerung", open=False):
                with gr.Row():
                    imgs_hash_clear_single = gr.Button("Nur Einzelbild-Cache leeren", size="sm")
                    imgs_hash_clear_multiple = gr.Button("Nur Galerie-Cache leeren", size="sm")
                    imgs_hash_clear_target = gr.Button("Nur Zielbild-Cache leeren", size="sm")

        progressbar_area = gr.Markdown("")
        imgs_hash_clear_single.click(clear_faces, None, [progressbar_area])
        imgs_hash_clear_multiple.click(clear_faces_list, None, [progressbar_area])
        imgs_hash_clear_target.click(clear_faces_target, None, [progressbar_area])
        imgs_hash_clear_all.click(clear_faces_all, None, [progressbar_area])

    return det_thresh, det_maxnum, face_sort_mode, mask_dilation, mask_blur, color_correction
