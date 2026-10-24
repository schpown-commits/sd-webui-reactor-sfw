import gradio as gr
from modules import shared
from scripts.reactor_compat import gr_update, tool_variant

def update_upscalers_list(selected: str):
    return gr_update(gr.Dropdown, value=selected, choices=[upscaler.name for upscaler in shared.sd_upscalers])

# TAB UPSCALE
def show(show_br: bool = True):
    with gr.Tab("Upscale"):
        with gr.Group():
            gr.Markdown("<div style='font-size: 0.9em; font-weight: bold; margin-bottom: 4px;'>🚀 Bild-Skalierung & Reihenfolge (Postprocessing Engine)</div>")
            with gr.Row():
                restore_first = gr.Checkbox(
                    True,
                    label="Reihenfolge: Erst Restaurieren ➔ Dann Hochskalieren",
                    info="Deaktivieren, um zuerst hochzuskalieren und danach das Gesicht zu restaurieren",
                    scale=2
                )
                upscale_force = gr.Checkbox(
                    False,
                    label="Erzwungenes Upscaling (Force Upscale)",
                    info="Skaliert das Bild auch dann hoch, wenn kein Gesicht gefunden wird",
                    scale=1
                )
            with gr.Row():
                upscaler_name = gr.Dropdown(
                    choices=[upscaler.name for upscaler in shared.sd_upscalers],
                    label="Upscaler-Modell",
                    value="None",
                    info="Wähle das gewünschte Modell (z.B. R-ESRGAN 4x+, 4x-UltraSharp, etc.)",
                    scale=4
                )
                upscalers_update = gr.Button(
                    value="🔄",
                    variant=tool_variant(),
                    elem_id="reactor_refresh_upscalers_btn",
                    tooltip="Upscaler-Liste aktualisieren",
                    scale=1,
                    min_width=45
                )
            upscalers_update.click(
                update_upscalers_list, 
                inputs=[upscaler_name],
                outputs=[upscaler_name],
            )
            with gr.Row():
                upscaler_scale = gr.Slider(1, 8, 1, step=0.1, label="Skalierungsfaktor (Scale by)", scale=1)
                upscaler_visibility = gr.Slider(
                    0, 1, 1, step=0.05, label="Upscaler-Sichtbarkeit / Blending (wenn Scale = 1)", scale=1
                )
    return restore_first, upscaler_name, upscaler_scale, upscaler_visibility, upscale_force
