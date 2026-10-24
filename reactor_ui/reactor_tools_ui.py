import gradio as gr
from scripts.reactor_swapper import build_face_model, blend_faces, render_face_index_preview

# TAB TOOLS & VAULT
def show():
    with gr.Tab("Vault & Tools"):
        with gr.Tab("👤 Face Models Vault (.safetensors)"):
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        gr.Markdown("<div style='font-weight: bold; font-size: 0.9em; margin-bottom: 4px;'>📷 Einzelgesicht als Modell speichern</div>")
                        gr.Markdown("<div style='font-size: 0.8em; color: var(--text-muted,#888); margin-bottom: 8px;'>Erstellt ein wiederverwendbares Gesichtsmodell aus einem Einzelfoto.</div>")
                        img_fm = gr.Image(
                            type="pil",
                            label="Quellbild für Gesichtsmodell",
                        )
                        with gr.Row(equal_height=True):
                            fm_name = gr.Textbox(
                                value="",
                                placeholder="Name eingeben (z. B. Elena_Portrait)",
                                label="Modell-Name",
                                scale=3
                            )
                            save_fm_btn = gr.Button("💾 Erstellen & Speichern", variant="primary", scale=2)
                        save_fm = gr.Markdown("<span style='font-size:0.8em; color:var(--text-muted,#888);'>Modelle werden in <code>models/reactor/faces</code> gespeichert.</span>")
                        save_fm_btn.click(
                            build_face_model,
                            inputs=[img_fm, fm_name],
                            outputs=[save_fm],
                        )

                with gr.Column():
                    with gr.Group():
                        gr.Markdown("<div style='font-weight: bold; font-size: 0.9em; margin-bottom: 4px;'>✨ Super-Embedding Blender</div>")
                        gr.Markdown("<div style='font-size: 0.8em; color: var(--text-muted,#888); margin-bottom: 8px;'>Kombiniert mehrere Fotos derselben Person zu einem optimalen Durchschnitts-Embedding.</div>")
                        imgs_fm = gr.Files(
                            label="Mehrere Fotos derselben Person laden",
                            file_types=["image"]
                        )
                        compute_method = gr.Radio(
                            ["Mean", "Median", "Mode"],
                            value="Mean",
                            label="Berechnungsmethode",
                            type="index",
                            info="Mean (Durchschnitt) liefert die stabilste und konsistenteste Gesichtsähnlichkeit 👍"
                        )
                        shape_check = gr.Checkbox(
                            False,
                            label="Embedding-Shape auf Konsistenz prüfen",
                            info="Überspringt Ausreißer-Gesichter bei unterschiedlichen Bilddimensionen"
                        )
                        with gr.Row(equal_height=True):
                            fm_name_blend = gr.Textbox(
                                value="",
                                placeholder="Name für Blended Model (z. B. Max_SuperModel)",
                                label="Modell-Name",
                                scale=3
                            )
                            save_blend_btn = gr.Button("⚡ Blenden & Speichern", variant="primary", scale=2)
                        save_blend_msg = gr.Markdown("<span style='font-size:0.8em; color:var(--text-muted,#888);'>Modelle werden in <code>models/reactor/faces</code> gespeichert.</span>")
                        save_blend_btn.click(
                            blend_faces,
                            inputs=[imgs_fm, fm_name_blend, compute_method, shape_check],
                            outputs=[save_blend_msg],
                        )

        with gr.Tab("🔍 Face Index Inspector"):
            gr.Markdown("<div style='font-size: 0.85em; color: var(--text-muted,#888); margin-bottom: 8px;'>Lade ein Zielbild hoch, um die exakten Indizes und Bounding-Boxes aller erkannten Gesichter visuell zu überprüfen.</div>")
            with gr.Row():
                preview_img_in = gr.Image(type="pil", label="Zielbild für Index-Prüfung")
                preview_img_out = gr.Image(type="pil", label="Annotierte Vorschau mit Indizes")
            with gr.Row():
                sort_mode_preview = gr.Dropdown(
                    choices=["Left-to-Right", "Right-to-Left", "Top-to-Bottom", "Bottom-to-Top", "Largest First", "Confidence First"],
                    value="Left-to-Right",
                    label="Sortier-Modus für Vorschau",
                    scale=3
                )
                btn_preview = gr.Button("🔎 Gesichts-Indizes ermitteln", variant="primary", scale=1)
            btn_preview.click(
                render_face_index_preview,
                inputs=[preview_img_in, sort_mode_preview],
                outputs=[preview_img_out]
            )
