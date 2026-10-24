import gradio as gr
from scripts.reactor_helpers import (
    get_model_names, 
    get_facemodels
)
from scripts.reactor_swapper import (
    clear_faces_list,
)
from scripts.reactor_compat import gr_update, tool_variant
from modules import shared


def update_fm_list(selected: str):
    return gr_update(gr.Dropdown, value=selected, choices=get_model_names(get_facemodels))


# TAB MAIN / STUDIO
def show(is_img2img: bool, show_br: bool = True, **msgs):

    def on_select_source(evt: gr.SelectData):
        if evt.index == 2:  # Folder
            return {
                control_col_1: gr_update(gr.Column, visible=False),
                control_col_2: gr_update(gr.Column, visible=False),
                control_col_3: gr_update(gr.Column, visible=True),
                imgs_hash_clear: gr_update(gr.Button, visible=True)
            }
        if evt.index == 0:  # Image(s)
            return {
                control_col_1: gr_update(gr.Column, visible=True),
                control_col_2: gr_update(gr.Column, visible=False),
                control_col_3: gr_update(gr.Column, visible=False),
                imgs_hash_clear: gr_update(gr.Button, visible=False)
            }
        if evt.index == 1:  # Face Model
            return {
                control_col_1: gr_update(gr.Column, visible=False),
                control_col_2: gr_update(gr.Column, visible=True),
                control_col_3: gr_update(gr.Column, visible=False),
                imgs_hash_clear: gr_update(gr.Button, visible=False)
            }

    progressbar_area = gr.Markdown("")

    with gr.Tab("Studio"):

        # -------------------------------------------------------------
        # 1. Quick Workflow Presets Bar
        # -------------------------------------------------------------
        with gr.Row(elem_id="reactor_presets_row"):
            gr.Markdown("<div style='font-size:0.85em; font-weight:700; color:var(--primary,#3b82f6); margin-top:6px;'>⚡ PRESETS:</div>", scale=0)
            btn_preset_portrait = gr.Button("📸 Studio Portrait", size="sm", variant="secondary")
            btn_preset_multiswap = gr.Button("👥 Group Multi-Swap", size="sm", variant="secondary")
            btn_preset_fidelity = gr.Button("✨ Max Fidelity", size="sm", variant="secondary")
            btn_preset_draft = gr.Button("⚡ Fast Draft", size="sm", variant="secondary")

        gr.Markdown("<hr style='margin: 8px 0 12px 0; border: none; border-top: 1px solid rgba(255,255,255,0.08);'>")

        # -------------------------------------------------------------
        # 2. Source Selection (Images / Models / Folder)
        # -------------------------------------------------------------
        with gr.Column():
            with gr.Row():
                select_source = gr.Radio(
                    ["Image(s)", "Face Model", "Folder"],
                    value="Image(s)",
                    label="1. Quell-Typ (Source Mode)",
                    type="index",
                    scale=1,
                )
                with gr.Column(visible=False) as control_col_2:
                    with gr.Row():
                        face_models = get_model_names(get_facemodels)
                        face_model = gr.Dropdown(
                            choices=face_models,
                            label="Choose Face Model(s) (.safetensors)",
                            value=["None"],
                            multiselect=True,
                            scale=4,
                        )
                        fm_update = gr.Button(
                            value="🔄",
                            variant=tool_variant(),
                            elem_id="reactor_refresh_facemodels_btn",
                            tooltip="Refresh Face Models List",
                            scale=1,
                            min_width=45,
                        )
                        fm_update.click(
                            update_fm_list, 
                            inputs=[face_model],
                            outputs=[face_model],
                        )
                imgs_hash_clear = gr.Button(
                    value="Clear Source Images Hash",
                    scale=1,
                    visible=False,
                )
                imgs_hash_clear.click(clear_faces_list, None, [progressbar_area])

            # Source Image(s) Dropzone / Tabs
            with gr.Column(visible=True) as control_col_1:
                with gr.Row():
                    selected_tab = gr.Textbox('tab_single', visible=False)
                    with gr.Tabs() as tab_single:
                        with gr.Tab('Single Source Image'):
                            img = gr.Image(
                                type="pil",
                                label="Quell-Bild (Source Face)",
                            )
                        with gr.Tab('Multiple Source Images') as tab_multiple:
                            imgs = gr.Files(
                                label=f"Mehrere Quell-Bilder{msgs.get('extra_multiple_source', '')}",
                                file_types=["image"],
                            )
                    tab_single.select(fn=lambda: 'tab_single', inputs=[], outputs=[selected_tab])
                    tab_multiple.select(fn=lambda: 'tab_multiple', inputs=[], outputs=[selected_tab])

            # Folder Mode
            with gr.Column(visible=False) as control_col_3:
                gr.Markdown("<span style='display:block;text-align:right;font-size:0.85em;color:var(--text-muted,#888);'>Tipp: Cache leeren, falls alte Gesichter geladen werden</span>")
                with gr.Row():
                    source_folder = gr.Textbox(
                        value="",
                        placeholder="Pfad zum Ordner mit Quellgesichtern (z.B. C:/Bilder/Gesichter)",
                        label=f"Quell-Ordner{msgs.get('extra_multiple_source', '')}",
                        scale=3,
                    )
                    random_image = gr.Checkbox(
                        False,
                        label="Zufälliges Bild",
                        info="Wählt pro Durchlauf ein Bild zufällig",
                        scale=1,
                    )

            setattr(face_model, "do_not_save_to_config", True)

            # -------------------------------------------------------------
            # 3. Interactive Visual Face-Routing Hub
            # -------------------------------------------------------------
            with gr.Group():
                gr.Markdown(
                    """
                    <div style="margin-top: 10px; padding: 10px 14px; background: rgba(0,0,0,0.18); border-radius: 8px; border: 1px dashed rgba(255,255,255,0.15);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: bold; font-size: 0.9em; letter-spacing: 0.3px;">🎯 VISUELLES FACE-ROUTING (Source ➔ Target)</span>
                            <span style="font-size: 0.75em; opacity: 0.7;">Klicke ein Preset oder passe die Indizes an</span>
                        </div>
                        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                            <span style="font-size: 0.8em; opacity: 0.85; align-self: center;">Schnell-Zuordnung:</span>
                        </div>
                    </div>
                    """
                )
                
                with gr.Row():
                    btn_route_1to1 = gr.Button("1:1 Standard (0 ➔ 0)", size="sm", variant="secondary")
                    btn_route_multi = gr.Button("Multi-Swap (0,1 ➔ 0,1)", size="sm", variant="secondary")
                    btn_route_cross = gr.Button("Cross-Swap (0,1 ➔ 1,0)", size="sm", variant="secondary")
                    btn_route_all = gr.Button("All-to-All (0 ➔ 0,1,2)", size="sm", variant="secondary")

                # Dynamic mapping display text
                mapping_preview_text = gr.Markdown(
                    "<div style='font-family: monospace; font-size: 0.85em; padding: 6px 12px; background: rgba(59,130,246,0.1); border-left: 3px solid #3b82f6; border-radius: 4px; margin: 4px 0 10px 0;'>"
                    "<b>Aktives Mapping:</b> Quelle [ <span style='color:#4ade80;'>0</span> ] ──➔ Ziel [ <span style='color:#f87171;'>0</span> ] (1:1 Einzeltausch)"
                    "</div>"
                )

                # Source & Target Face Index Inputs + Gender Filter
                with gr.Row():
                    with gr.Column(scale=1):
                        source_faces_index = gr.Textbox(
                            value="0",
                            placeholder="Z. B. 0 oder 0, 1 oder 0, 2, 1",
                            label="Quell-Gesichts-Index (Source Index)",
                            info="Komma- oder leerzeichengetrennte Nummern (0 ist das erste erkannte Gesicht)",
                        )
                        gender_source = gr.Radio(
                            ["No", "Female Only", "Male Only"],
                            value="No",
                            label="Geschlechtsfilter (Quelle)",
                            type="index",
                        )
                    with gr.Column(scale=1):
                        faces_index = gr.Textbox(
                            value="0",
                            placeholder="Z. B. 0 oder 0, 1 oder 1, 0",
                            label="Ziel-Gesichts-Index (Target Index)",
                            info="Ziel-Reihenfolge im generierten Bild (nach Sortier-Modus sortiert)",
                        )
                        gender_target = gr.Radio(
                            ["No", "Female Only", "Male Only"],
                            value="No",
                            label="Geschlechtsfilter (Ziel)",
                            type="index",
                        )

            # Routing presets click handlers
            def on_route_1to1():
                txt = "<div style='font-family: monospace; font-size: 0.85em; padding: 6px 12px; background: rgba(59,130,246,0.1); border-left: 3px solid #3b82f6; border-radius: 4px; margin: 4px 0 10px 0;'><b>Aktives Mapping:</b> Quelle [ <span style='color:#4ade80;'>0</span> ] ──➔ Ziel [ <span style='color:#f87171;'>0</span> ] (1:1 Einzeltausch)</div>"
                return "0", "0", False, txt

            def on_route_multi():
                txt = "<div style='font-family: monospace; font-size: 0.85em; padding: 6px 12px; background: rgba(59,130,246,0.1); border-left: 3px solid #3b82f6; border-radius: 4px; margin: 4px 0 10px 0;'><b>Aktives Mapping:</b> Quelle 0 ──➔ Ziel 0, Quelle 1 ──➔ Ziel 1 (Multi-Swap Simultan)</div>"
                return "0, 1", "0, 1", True, txt

            def on_route_cross():
                txt = "<div style='font-family: monospace; font-size: 0.85em; padding: 6px 12px; background: rgba(59,130,246,0.1); border-left: 3px solid #3b82f6; border-radius: 4px; margin: 4px 0 10px 0;'><b>Aktives Mapping:</b> Quelle 0 ──➔ Ziel 1, Quelle 1 ──➔ Ziel 0 (Überkreuz-Tausch)</div>"
                return "0, 1", "1, 0", True, txt

            def on_route_all():
                txt = "<div style='font-family: monospace; font-size: 0.85em; padding: 6px 12px; background: rgba(59,130,246,0.1); border-left: 3px solid #3b82f6; border-radius: 4px; margin: 4px 0 10px 0;'><b>Aktives Mapping:</b> Quelle 0 ──➔ Ziele 0, 1, 2 (Ein Quellgesicht auf alle Zielpersonen)</div>"
                return "0", "0, 1, 2", True, txt

            btn_route_1to1.click(on_route_1to1, None, [source_faces_index, faces_index, simultaneous_swap if 'simultaneous_swap' in locals() else source_faces_index, mapping_preview_text])
            btn_route_multi.click(on_route_multi, None, [source_faces_index, faces_index, simultaneous_swap if 'simultaneous_swap' in locals() else source_faces_index, mapping_preview_text])
            btn_route_cross.click(on_route_cross, None, [source_faces_index, faces_index, simultaneous_swap if 'simultaneous_swap' in locals() else source_faces_index, mapping_preview_text])
            btn_route_all.click(on_route_all, None, [source_faces_index, faces_index, simultaneous_swap if 'simultaneous_swap' in locals() else source_faces_index, mapping_preview_text])

            # -------------------------------------------------------------
            # 4. Mode Switches (Multi-Swap, Masking, Originals)
            # -------------------------------------------------------------
            with gr.Row():
                simultaneous_swap = gr.Checkbox(
                    False,
                    label="Simultaner Multi-Swap",
                    info="Mehrere Quellgesichter auf unterschiedliche Zielgesichter gleichzeitig übertragen",
                    scale=1,
                )
                blend_source_embeddings = gr.Checkbox(
                    False,
                    label="Super-Embedding Blend",
                    info="Durchschnitts-Embedding aus mehreren Quellfotos derselben Person erzeugen",
                    scale=1,
                )
                mask_face = gr.Checkbox(
                    True,
                    label="Face Mask Correction", 
                    info="Verfeinert Gesichtskonturen via BiSeNet Maskengenerator",
                    scale=1,
                )

            with gr.Row():
                if is_img2img:
                    save_original = gr.Checkbox(
                        False,
                        label="Originalbild mitspeichern", 
                        info="Speichert das ungeswappte Bild zusätzlich im Ausgabeordner"
                    )
                else:
                    save_original = gr.Checkbox(
                        False,
                        label="Originalbild mitspeichern", 
                        info="Speichert das ungeswappte Bild zusätzlich im Ausgabeordner",
                        visible=show_br
                    )

                swap_in_source = gr.Checkbox(
                    False,
                    label="Im Quellbild tauschen (Swap in source)",
                    visible=is_img2img,
                )
                swap_in_generated = gr.Checkbox(
                    True,
                    label="Im generierten Bild tauschen (Swap in generated)",
                    visible=is_img2img,
                )

            # Re-wire buttons to actual simultaneous_swap reference
            btn_route_1to1.click(on_route_1to1, None, [source_faces_index, faces_index, simultaneous_swap, mapping_preview_text])
            btn_route_multi.click(on_route_multi, None, [source_faces_index, faces_index, simultaneous_swap, mapping_preview_text])
            btn_route_cross.click(on_route_cross, None, [source_faces_index, faces_index, simultaneous_swap, mapping_preview_text])
            btn_route_all.click(on_route_all, None, [source_faces_index, faces_index, simultaneous_swap, mapping_preview_text])

            # -------------------------------------------------------------
            # 5. Face Restoration & CodeFormer Quality
            # -------------------------------------------------------------
            with gr.Row():
                face_restorer_name = gr.Radio(
                    label="Face Restorer (Gesichts-Restaurierung)",
                    choices=["None"] + [x.name() for x in shared.face_restorers],
                    value=shared.face_restorers[0].name() if len(shared.face_restorers) > 0 else "None",
                    type="value",
                    scale=1,
                )
                with gr.Column(scale=2):
                    face_restorer_visibility = gr.Slider(
                        0, 1, 1, step=0.05, label="Restorer Visibility (Sichtbarkeit)"
                    )
                    codeformer_weight = gr.Slider(
                        0, 1, 0.65, step=0.05, label="CodeFormer Fidelity (Gewichtung)", 
                        info="0 = Max. Rekonstruktion (stärker geglättet), 1 = Maximale Originaltreue"
                    )

            # Preset Button Click Handlers
            def apply_portrait():
                cf_name = "CodeFormer" if "CodeFormer" in [x.name() for x in shared.face_restorers] else (shared.face_restorers[0].name() if len(shared.face_restorers) > 0 else "None")
                return cf_name, 1.0, 0.65, True, False

            def apply_group_multi():
                cf_name = "CodeFormer" if "CodeFormer" in [x.name() for x in shared.face_restorers] else (shared.face_restorers[0].name() if len(shared.face_restorers) > 0 else "None")
                return cf_name, 1.0, 0.60, True, True

            def apply_max_fidelity():
                rest_name = "GFPGAN" if "GFPGAN" in [x.name() for x in shared.face_restorers] else (shared.face_restorers[0].name() if len(shared.face_restorers) > 0 else "None")
                return rest_name, 0.9, 0.85, True, False

            def apply_fast_draft():
                return "None", 0.0, 1.0, False, False

            btn_preset_portrait.click(apply_portrait, None, [face_restorer_name, face_restorer_visibility, codeformer_weight, mask_face, simultaneous_swap])
            btn_preset_multiswap.click(apply_group_multi, None, [face_restorer_name, face_restorer_visibility, codeformer_weight, mask_face, simultaneous_swap])
            btn_preset_fidelity.click(apply_max_fidelity, None, [face_restorer_name, face_restorer_visibility, codeformer_weight, mask_face, simultaneous_swap])
            btn_preset_draft.click(apply_fast_draft, None, [face_restorer_name, face_restorer_visibility, codeformer_weight, mask_face, simultaneous_swap])

    select_source.select(on_select_source, None, [control_col_1, control_col_2, control_col_3, imgs_hash_clear], show_progress=False)

    return img, imgs, selected_tab, select_source, face_model, source_folder, save_original, mask_face, source_faces_index, gender_source, faces_index, gender_target, face_restorer_name, face_restorer_visibility, codeformer_weight, swap_in_source, swap_in_generated, random_image, simultaneous_swap, blend_source_embeddings
