import os, glob
import gradio as gr
from PIL import Image

from typing import List

from modules import scripts, shared, images
from modules.upscaler import Upscaler, UpscalerData
try:
    from modules import scripts_postprocessing
except ImportError:
    scripts_postprocessing = None
from modules.processing import (
    Processed,
    StableDiffusionProcessing,
    StableDiffusionProcessingImg2Img,
)
from modules.face_restoration import FaceRestoration
from modules.images import save_image

from reactor_ui import (
    ui_main,
    ui_upscale,
    ui_tools,
    ui_settings,
    ui_detection,
)
from scripts.reactor_logger import logger
from scripts.reactor_swapper import (
    EnhancementOptions,
    DetectionOptions,
    swap_face,
    check_process_halt,
    reset_messaged,
)
from scripts.reactor_version import version_flag, app_title
from scripts.console_log_patch import apply_logging_patch
from scripts.reactor_helpers import (
    make_grid,
    set_Device,
    get_SDNEXT,
)
from scripts.reactor_globals import SWAPPER_MODELS_PATH #, DEVICE, DEVICE_LIST

def IA_cap(cond: bool, label: str=""):
    return None

try:
    from modules.ui_components import InputAccordion
    NO_IA = False
except:
    NO_IA = True
    InputAccordion = IA_cap


def check_old_webui():
    return NO_IA

import re

def parse_face_indices(val) -> list[int]:
    if isinstance(val, list):
        return [int(x) for x in val if str(x).isdigit()]
    if isinstance(val, int):
        return [val]
    nums = re.findall(r'\d+', str(val))
    return [int(x) for x in nums] if nums else [0]



class FaceSwapScript(scripts.Script):
    def title(self):
        return f"{app_title}"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with (
            gr.Accordion(f"{app_title}", open=False) if check_old_webui() else InputAccordion(False, label=f"{app_title}") as enable
        ):

            # SD.Next or A1111 1.52:
            if get_SDNEXT() or check_old_webui():
                enable = gr.Checkbox(False, label="Enable")
            
            # enable = gr.Checkbox(False, label="Enable", info=f"The Fast and Simple FaceSwap Extension - {version_flag}")
            gr.Markdown(f"<sup>The Fast and Simple FaceSwap Extension - {version_flag}</sup>")

            # TAB MAIN
            msgs: dict = {
                "extra_multiple_source": "",
            }
            img, imgs, selected_tab, select_source, face_model, source_folder, save_original, mask_face, source_faces_index, gender_source, faces_index, gender_target, face_restorer_name, face_restorer_visibility, codeformer_weight, swap_in_source, swap_in_generated, random_image, simultaneous_swap, blend_source_embeddings = ui_main.show(is_img2img=is_img2img, **msgs)
            
            # TAB DETECTION
            det_thresh, det_maxnum, face_sort_mode, mask_dilation, mask_blur, color_correction = ui_detection.show()
            
            # TAB UPSCALE
            restore_first, upscaler_name, upscaler_scale, upscaler_visibility, upscale_force = ui_upscale.show()

            # TAB TOOLS
            ui_tools.show()
            
            # TAB SETTINGS
            model, device, console_logging_level, source_hash_check, target_hash_check = ui_settings.show()
            
            gr.Markdown("<span style='display:block;text-align:right;padding:3px;font-size:0.85em;margin-bottom:-12px;'>by <a style='font-weight:normal;text-decoration:underline;' href='https://github.com/Gourieff' target='_blank' aria-label='Author Eugene Gourieff GitHub'>Eugene Gourieff</a></span>")

        return [
            img,
            enable,
            source_faces_index,
            faces_index,
            model,
            face_restorer_name,
            face_restorer_visibility,
            restore_first,
            upscaler_name,
            upscaler_scale,
            upscaler_visibility,
            swap_in_source,
            swap_in_generated,
            console_logging_level,
            gender_source,
            gender_target,
            save_original,
            codeformer_weight,
            source_hash_check,
            target_hash_check,
            device,
            mask_face,
            select_source,
            face_model,
            source_folder,
            imgs,
            random_image,
            upscale_force,
            det_thresh,
            det_maxnum,
            selected_tab,
            simultaneous_swap,
            face_sort_mode,
            blend_source_embeddings,
            mask_dilation,
            mask_blur,
            color_correction,
        ]


    @property
    def upscaler(self) -> UpscalerData:
        for upscaler in shared.sd_upscalers:
            if upscaler.name == self.upscaler_name:
                return upscaler
        return None

    @property
    def face_restorer(self) -> FaceRestoration:
        for face_restorer in shared.face_restorers:
            if face_restorer.name() == self.face_restorer_name:
                return face_restorer
        return None

    @property
    def enhancement_options(self) -> EnhancementOptions:
        return EnhancementOptions(
            do_restore_first=self.restore_first,
            scale=self.upscaler_scale,
            upscaler=self.upscaler,
            face_restorer=self.face_restorer,
            upscale_visibility=self.upscaler_visibility,
            restorer_visibility=self.face_restorer_visibility,
            codeformer_weight=self.codeformer_weight,
            upscale_force=self.upscale_force
        )
    
    @property
    def detection_options(self) -> DetectionOptions:
        return DetectionOptions(
            det_thresh=self.det_thresh,
            det_maxnum=self.det_maxnum,
            face_sort_mode=self.face_sort_mode,
            mask_dilation=self.mask_dilation,
            mask_blur=self.mask_blur,
            color_correction=self.color_correction,
        )

    def process(
        self,
        p: StableDiffusionProcessing,
        img,
        enable,
        source_faces_index,
        faces_index,
        model,
        face_restorer_name,
        face_restorer_visibility,
        restore_first,
        upscaler_name,
        upscaler_scale,
        upscaler_visibility,
        swap_in_source,
        swap_in_generated,
        console_logging_level,
        gender_source,
        gender_target,
        save_original,
        codeformer_weight,
        source_hash_check,
        target_hash_check,
        device,
        mask_face,
        select_source,
        face_model,
        source_folder,
        imgs,
        random_image,
        upscale_force,
        det_thresh,
        det_maxnum,
        selected_tab,
        simultaneous_swap,
        face_sort_mode,
        blend_source_embeddings,
        mask_dilation,
        mask_blur,
        color_correction,
    ):
        self.enable = enable
        if self.enable:

            logger.debug("*** Start process")

            reset_messaged()
            if check_process_halt():
                return
            
            global SWAPPER_MODELS_PATH
            if selected_tab == "tab_single":
                self.source = img
            else:
                self.source = None
            self.face_restorer_name = face_restorer_name
            self.upscaler_scale = upscaler_scale
            self.upscaler_visibility = upscaler_visibility
            self.face_restorer_visibility = face_restorer_visibility
            self.restore_first = restore_first
            self.upscaler_name = upscaler_name  
            self.swap_in_source = swap_in_source
            self.swap_in_generated = swap_in_generated
            self.model = os.path.join(SWAPPER_MODELS_PATH,model)
            self.console_logging_level = console_logging_level
            self.gender_source = gender_source
            self.gender_target = gender_target
            self.save_original = save_original
            self.codeformer_weight = codeformer_weight
            self.source_hash_check = source_hash_check
            self.target_hash_check = target_hash_check
            self.device = device
            self.mask_face = mask_face
            self.select_source = select_source
            self.face_model = face_model
            self.source_folder = source_folder
            if selected_tab == "tab_single":
                self.source_imgs = None
            else:
                self.source_imgs = imgs
            self.random_image = random_image
            self.upscale_force = upscale_force
            self.det_thresh=det_thresh
            self.det_maxnum=det_maxnum
            self.simultaneous_swap = simultaneous_swap
            self.face_sort_mode = face_sort_mode
            self.blend_source_embeddings = blend_source_embeddings
            self.mask_dilation = mask_dilation
            self.mask_blur = mask_blur
            self.color_correction = color_correction
            if self.gender_source is None or self.gender_source == "No":
                self.gender_source = 0
            if self.gender_target is None or self.gender_target == "No":
                self.gender_target = 0
            self.source_faces_index = parse_face_indices(source_faces_index)
            self.faces_index = parse_face_indices(faces_index)
            self.swapped_in_postprocess_image = False
            if self.save_original is None:
                self.save_original = False
            if self.source_hash_check is None:
                self.source_hash_check = True
            if self.target_hash_check is None:
                self.target_hash_check = False
            if self.mask_face is None:
                self.mask_face = False
            if self.random_image is None:
                self.random_image = False
            if self.upscale_force is None:
                self.upscale_force = False
            
            if shared.state.job_count > 0:
                # logger.debug(f"Job count: {shared.state.job_count}")
                self.face_restorer_visibility = shared.opts.data['restorer_visibility'] if 'restorer_visibility' in shared.opts.data.keys() else face_restorer_visibility
                self.codeformer_weight = shared.opts.data['codeformer_weight'] if 'codeformer_weight' in shared.opts.data.keys() else codeformer_weight
                self.mask_face = shared.opts.data['mask_face'] if 'mask_face' in shared.opts.data.keys() else mask_face
                self.face_model = shared.opts.data['face_model'] if 'face_model' in shared.opts.data.keys() else face_model

            logger.debug("*** Set Device")
            set_Device(self.device)

            if (self.save_original is None or not self.save_original) and (self.select_source == 2 or self.source_imgs is not None):
                p.do_not_save_samples = True
            
            if ((self.source is not None or self.source_imgs is not None) and self.select_source == 0) or ((self.face_model is not None and (self.face_model != "None" if not isinstance(self.face_model, list) else len([m for m in self.face_model if m not in (None, "None")]) > 0)) and self.select_source == 1) or ((self.source_folder is not None and self.source_folder != "") and self.select_source == 2):
                logger.debug("*** Log patch")
                apply_logging_patch(console_logging_level)
                
                if isinstance(p, StableDiffusionProcessingImg2Img) and self.swap_in_source:

                    logger.debug("*** Check process")

                    logger.status("Working: source face index %s, target face index %s", self.source_faces_index, self.faces_index)

                    for i in range(len(p.init_images)):
                        if len(p.init_images) > 1:
                            logger.status("Swap in %s", i)
                        result, output, swapped = swap_face(
                            self.source,
                            p.init_images[i],
                            source_faces_index=self.source_faces_index,
                            faces_index=self.faces_index,
                            model=self.model,
                            enhancement_options=self.enhancement_options,
                            gender_source=self.gender_source,
                            gender_target=self.gender_target,
                            source_hash_check=self.source_hash_check,
                            target_hash_check=self.target_hash_check,
                            device=self.device,
                            mask_face=self.mask_face,
                            select_source=self.select_source,
                            face_model = self.face_model,
                            source_folder = self.source_folder,
                            source_imgs = self.source_imgs,
                            random_image = self.random_image,
                            detection_options=self.detection_options,
                            simultaneous_swap=self.simultaneous_swap,
                            blend_source_embeddings=self.blend_source_embeddings,
                        )
                        p.init_images[i] = result[0] if isinstance(result, list) else result
                        # result_path = get_image_path(p.init_images[i], p.outpath_samples, "", p.all_seeds[i], p.all_prompts[i], "txt", p=p, suffix="-swapped")
                        # if len(output) != 0:
                        #     with open(result_path, 'w', encoding="utf8") as f:
                        #         f.writelines(output)

                        if shared.state.interrupted or shared.state.skipped:
                            return
            
            else:
                logger.error("Please provide a source face")
                return

    def postprocess(self, p: StableDiffusionProcessing, processed: Processed, *args):
        if self.enable:

            logger.debug("*** Check postprocess - before IF")

            reset_messaged()
            if check_process_halt():
                return

            if self.save_original or ((self.select_source == 2 and self.source_folder is not None and self.source_folder != "") or (self.select_source == 0 and self.source_imgs is not None and self.source is None)):

                logger.debug("*** Check postprocess - after IF")

                postprocess_run: bool = True

                orig_images : List[Image.Image] = processed.images[processed.index_of_first_image:]
                orig_infotexts : List[str] = processed.infotexts[processed.index_of_first_image:]

                result_images: List = processed.images
                # result_info: List = processed.infotexts

                if self.swap_in_generated and not getattr(self, 'swapped_in_postprocess_image', False):

                    logger.status("Working: source face index %s, target face index %s", self.source_faces_index, self.faces_index)

                    if self.source is not None:
                        # self.source_folder = None
                        self.source_imgs = None

                    for i,(img,info) in enumerate(zip(orig_images, orig_infotexts)):
                        if check_process_halt():
                            postprocess_run = False
                            break
                        if len(orig_images) > 1:
                            logger.status("Swap in %s", i)
                        result, output, swapped = swap_face(
                            self.source,
                            img,
                            source_faces_index=self.source_faces_index,
                            faces_index=self.faces_index,
                            model=self.model,
                            enhancement_options=self.enhancement_options,
                            gender_source=self.gender_source,
                            gender_target=self.gender_target,
                            source_hash_check=self.source_hash_check,
                            target_hash_check=self.target_hash_check,
                            device=self.device,
                            mask_face=self.mask_face,
                            select_source=self.select_source,
                            face_model = self.face_model,
                            source_folder = self.source_folder,
                            source_imgs = self.source_imgs,
                            random_image = self.random_image,
                            detection_options=self.detection_options,
                            simultaneous_swap=self.simultaneous_swap,
                            blend_source_embeddings=self.blend_source_embeddings,
                        )

                        if self.select_source == 2 or (self.select_source == 0 and self.source_imgs is not None and self.source is None):
                            res_list = result if isinstance(result, list) else [result]
                            if len(res_list) > 0 and swapped > 0:
                                if self.save_original:
                                    result_images.extend(res_list)
                                else:
                                    result_images = res_list
                                suffix = "-swapped"
                                for idx_r, x in enumerate(res_list):
                                    try:
                                        img_path = save_image(x, p.outpath_samples, "", p.all_seeds[0], p.all_prompts[0], "png", info=info, p=p, suffix=suffix)
                                    except:
                                        logger.error("Cannot save a result image - please, check SD WebUI Settings (Saving and Paths)")

                            elif len(res_list) == 0:
                                logger.error("Cannot create a result image")

                        else:
                            if result is not None and swapped > 0:
                                res_img = result[0] if isinstance(result, list) else result
                                if self.save_original:
                                    result_images.append(res_img)
                                else:
                                    target_idx = processed.index_of_first_image + i
                                    if target_idx < len(result_images):
                                        result_images[target_idx] = res_img
                                    else:
                                        result_images.append(res_img)
                                suffix = "-swapped"
                                try:
                                    img_path = save_image(res_img, p.outpath_samples, "", p.all_seeds[0], p.all_prompts[0], "png", info=info, p=p, suffix=suffix)
                                except:
                                    logger.error("Cannot save a result image - please, check SD WebUI Settings (Saving and Paths)")
                            elif result is None:
                                logger.error("Cannot create a result image")
                        
                        # if len(output) != 0:
                        #     split_fullfn = os.path.splitext(img_path[0])
                        #     fullfn = split_fullfn[0] + ".txt"
                        #     with open(fullfn, 'w', encoding="utf8") as f:
                        #         f.writelines(output)
                
                if shared.opts.return_grid and len(result_images) > 2 and postprocess_run:
                    grid = make_grid(result_images)
                    result_images.insert(0, grid)
                    try:
                        save_image(grid, p.outpath_grids, "grid", p.all_seeds[0], p.all_prompts[0], shared.opts.grid_format, info=info, short_filename=not shared.opts.grid_extended_filename, p=p, grid=True)
                    except:
                        logger.error("Cannot save a grid - please, check SD WebUI Settings (Saving and Paths)")
                
                processed.images = result_images
                # processed.infotexts = result_info
            
            elif self.select_source == 0 and self.source is not None and self.source_imgs is not None:

                logger.debug("*** Check postprocess - after ELIF")

                if self.result is not None:
                    orig_infotexts : List[str] = processed.infotexts[processed.index_of_first_image:]
                    processed.images = [self.result]
                    try:
                        img_path = save_image(self.result, p.outpath_samples, "", p.all_seeds[0], p.all_prompts[0], "png", info=orig_infotexts[0], p=p, suffix="")
                    except:
                        logger.error("Cannot save a result image - please, check SD WebUI Settings (Saving and Paths)")
                else:
                    logger.error("Cannot create a result image")

    
    def postprocess_batch(self, p, *args, **kwargs):
        if self.enable and not self.save_original:
            logger.debug("*** Check postprocess_batch")
            images = kwargs["images"]

    def postprocess_image(self, p, script_pp: scripts.PostprocessImageArgs, *args):
        if self.enable and self.swap_in_generated and not self.save_original and ((self.select_source == 0 and self.source is not None) or self.select_source == 1):

            logger.debug("*** Check postprocess_image")

            current_job_number = shared.state.job_no + 1
            job_count = shared.state.job_count
            if current_job_number == job_count:
                reset_messaged()
            if check_process_halt():
                return
            
            # if (self.source is not None and self.select_source == 0) or ((self.face_model is not None and self.face_model != "None") and self.select_source == 1):
            logger.status("Working: source face index %s, target face index %s", self.source_faces_index, self.faces_index)
            image: Image.Image = script_pp.image
            result, output, swapped = swap_face(
                self.source,
                image,
                source_faces_index=self.source_faces_index,
                faces_index=self.faces_index,
                model=self.model,
                enhancement_options=self.enhancement_options,
                gender_source=self.gender_source,
                gender_target=self.gender_target,
                source_hash_check=self.source_hash_check,
                target_hash_check=self.target_hash_check,
                device=self.device,
                mask_face=self.mask_face,
                select_source=self.select_source,
                face_model = self.face_model,
                source_folder = None,
                source_imgs = None,
                random_image = False,
                detection_options=self.detection_options,
                simultaneous_swap=self.simultaneous_swap,
                blend_source_embeddings=self.blend_source_embeddings,
            )
            self.result = result
            self.swapped_in_postprocess_image = True
            try:
                res_img = result if isinstance(result, Image.Image) else result[0]
                if scripts_postprocessing is not None:
                    pp = scripts_postprocessing.PostprocessedImage(res_img)
                    pp.info = {}
                    p.extra_generation_params.update(pp.info)
                    script_pp.image = pp.image
                else:
                    script_pp.image = res_img

                # if len(output) != 0:
                #     result_path = get_image_path(script_pp.image, p.outpath_samples, "", p.all_seeds[0], p.all_prompts[0], "txt", p=p, suffix="-swapped")
                #     if len(output) != 0:
                #         with open(result_path, 'w', encoding="utf8") as f:
                #             f.writelines(output)
            except:
                logger.error("Cannot create a result image")


# Conditional class definition: only register Extras tab if scripts_postprocessing is available
if scripts_postprocessing is not None:
  class FaceSwapScriptExtras(scripts_postprocessing.ScriptPostprocessing):
    name = 'ReActor'
    order = 20000

    def ui(self):
        with (
            gr.Accordion(f"{app_title}", open=False) if check_old_webui() else InputAccordion(False, label=f"{app_title}") as enable
        ):
        # with ui_components.InputAccordion(False, label=f"{app_title}") as enable:
        # with gr.Accordion(f"{app_title}", open=False):
            
            # SD.Next or A1111 1.52:
            if get_SDNEXT() or check_old_webui():
                enable = gr.Checkbox(False, label="Enable")

            # enable = gr.Checkbox(False, label="Enable", info=f"The Fast and Simple FaceSwap Extension - {version_flag}")
            gr.Markdown(f"<span style='display:block;font-size:0.75em;margin-bottom:-24px;'>The Fast and Simple FaceSwap Extension - {version_flag}</span>")

            # TAB MAIN
            msgs: dict = {
                "extra_multiple_source": "",
            }
            img, imgs, selected_tab, select_source, face_model, source_folder, save_original, mask_face, source_faces_index, gender_source, faces_index, gender_target, face_restorer_name, face_restorer_visibility, codeformer_weight, swap_in_source, swap_in_generated, random_image, simultaneous_swap, blend_source_embeddings = ui_main.show(is_img2img=False, show_br=False, **msgs)
            
            # TAB DETECTION
            det_thresh, det_maxnum, face_sort_mode, mask_dilation, mask_blur, color_correction = ui_detection.show()
            
            # TAB UPSCALE
            restore_first, upscaler_name, upscaler_scale, upscaler_visibility, upscale_force = ui_upscale.show(show_br=False)
                        
            # TAB TOOLS
            ui_tools.show()
                        
            # TAB SETTINGS
            model, device, console_logging_level, source_hash_check, target_hash_check = ui_settings.show(hash_check_block=False)
                        
            gr.Markdown("<span style='display:block;text-align:right;padding-right:3px;font-size:0.85em;margin: -9px 0'>by <a style='font-weight:normal;text-decoration:underline;' href='https://github.com/Gourieff' target='_blank' aria-label='Author Eugene Gourieff GitHub'>Eugene Gourieff</a></span>")

        args = {
            'img': img,
            'enable': enable,
            'source_faces_index': source_faces_index,
            'faces_index': faces_index,
            'model': model,
            'face_restorer_name': face_restorer_name,
            'face_restorer_visibility': face_restorer_visibility,
            'restore_first': restore_first,
            'upscaler_name': upscaler_name,
            'upscaler_scale': upscaler_scale,
            'upscaler_visibility': upscaler_visibility,
            'console_logging_level': console_logging_level,
            'gender_source': gender_source,
            'gender_target': gender_target,
            'codeformer_weight': codeformer_weight,
            'device': device,
            'mask_face': mask_face,
            'select_source': select_source,
            'face_model': face_model,
            'source_folder': source_folder,
            'imgs': imgs,
            'random_image': random_image,
            'upscale_force': upscale_force,
            'det_thresh': det_thresh,
            'det_maxnum': det_maxnum,
            'selected_tab': selected_tab,
            'simultaneous_swap': simultaneous_swap,
            'face_sort_mode': face_sort_mode,
            'blend_source_embeddings': blend_source_embeddings,
            'mask_dilation': mask_dilation,
            'mask_blur': mask_blur,
            'color_correction': color_correction,
        }
        return args

    @property
    def upscaler(self) -> UpscalerData:
        for upscaler in shared.sd_upscalers:
            if upscaler.name == self.upscaler_name:
                return upscaler
        return None

    @property
    def face_restorer(self) -> FaceRestoration:
        for face_restorer in shared.face_restorers:
            if face_restorer.name() == self.face_restorer_name:
                return face_restorer
        return None

    @property
    def enhancement_options(self) -> EnhancementOptions:
        return EnhancementOptions(
            do_restore_first=self.restore_first,
            scale=self.upscaler_scale,
            upscaler=self.upscaler,
            face_restorer=self.face_restorer,
            upscale_visibility=self.upscaler_visibility,
            restorer_visibility=self.face_restorer_visibility,
            codeformer_weight=self.codeformer_weight,
            upscale_force=self.upscale_force,
        )
    
    @property
    def detection_options(self) -> DetectionOptions:
        return DetectionOptions(
            det_thresh=self.det_thresh,
            det_maxnum=self.det_maxnum,
            face_sort_mode=self.face_sort_mode,
            mask_dilation=self.mask_dilation,
            mask_blur=self.mask_blur,
            color_correction=self.color_correction,
        )

    def process(self, pp: scripts_postprocessing.PostprocessedImage, **args):
        if args['enable']:
            reset_messaged()
            if check_process_halt():
                return

            global SWAPPER_MODELS_PATH
            if args['selected_tab'] == "tab_single":
                self.source = args['img']
            else:
                self.source = None
            self.face_restorer_name = args['face_restorer_name']
            self.upscaler_scale = args['upscaler_scale']
            self.upscaler_visibility = args['upscaler_visibility']
            self.face_restorer_visibility = args['face_restorer_visibility']
            self.restore_first = args['restore_first']
            self.upscaler_name = args['upscaler_name']
            self.model = os.path.join(SWAPPER_MODELS_PATH, args['model'])
            self.console_logging_level = args['console_logging_level']
            self.gender_source = args['gender_source']
            self.gender_target = args['gender_target']
            self.codeformer_weight = args['codeformer_weight']
            self.device = args['device']
            self.mask_face = args['mask_face']
            self.select_source = args['select_source']
            self.face_model = args['face_model']
            self.source_folder = args['source_folder']
            if args['selected_tab'] == "tab_single":
                self.source_imgs = None
            else:
                self.source_imgs = args['imgs']
            self.random_image = args['random_image']
            self.upscale_force = args['upscale_force']
            self.det_thresh = args['det_thresh']
            self.det_maxnum = args['det_maxnum']
            self.simultaneous_swap = args.get('simultaneous_swap', False)
            self.face_sort_mode = args.get('face_sort_mode', "Left-to-Right")
            self.blend_source_embeddings = args.get('blend_source_embeddings', False)
            self.mask_dilation = args.get('mask_dilation', 0)
            self.mask_blur = args.get('mask_blur', 0)
            self.color_correction = args.get('color_correction', False)
            if self.gender_source is None or self.gender_source == "No":
                self.gender_source = 0
            if self.gender_target is None or self.gender_target == "No":
                self.gender_target = 0
            self.source_faces_index = parse_face_indices(args['source_faces_index'])
            self.faces_index = parse_face_indices(args['faces_index'])
            if self.mask_face is None:
                self.mask_face = False
            if self.random_image is None:
                self.random_image = False
            if self.upscale_force is None:
                self.upscale_force = False

            current_job_number = shared.state.job_no + 1
            job_count = shared.state.job_count
            if current_job_number == job_count:
                reset_messaged()

            set_Device(self.device)

            logger.debug("We're here: process() 1")
            
            if (self.source is not None and self.select_source == 0) or ((self.face_model is not None and (self.face_model != "None" if not isinstance(self.face_model, list) else len([m for m in self.face_model if m not in (None, "None")]) > 0)) and self.select_source == 1) or ((self.source_folder is not None and self.source_folder != "") and self.select_source == 2) or ((self.source_imgs is not None and self.source is None) and self.select_source == 0):

                logger.debug("We're here: process() 2")

                if self.source is not None and self.select_source == 0:
                    self.source_imgs = None

                apply_logging_patch(self.console_logging_level)
                logger.status("Working: source face index %s, target face index %s", self.source_faces_index, self.faces_index)
                # if self.select_source != 2:
                image: Image.Image = pp.image

                # Extract alpha channel
                logger.debug(f"image = {image}")
                if image.mode == 'RGBA':
                    _, _, _, alpha = image.split()
                else:
                    alpha = None
                logger.debug(f"alpha = {alpha}")

                result, output, swapped = swap_face(
                    self.source,
                    image,
                    source_faces_index=self.source_faces_index,
                    faces_index=self.faces_index,
                    model=self.model,
                    enhancement_options=self.enhancement_options,
                    gender_source=self.gender_source,
                    gender_target=self.gender_target,
                    source_hash_check=True,
                    target_hash_check=True,
                    device=self.device,
                    mask_face=self.mask_face,
                    select_source=self.select_source,
                    face_model=self.face_model,
                    source_folder=self.source_folder,
                    source_imgs=self.source_imgs,
                    random_image=self.random_image,
                    detection_options=self.detection_options,
                    simultaneous_swap=self.simultaneous_swap,
                    blend_source_embeddings=self.blend_source_embeddings,
                )
                if self.select_source == 2 or (self.select_source == 0 and self.source_imgs is not None and self.source is None):
                    if len(result) > 0 and swapped > 0:
                        image = result[0]
                        if len(result) > 1:
                            if hasattr(pp, 'extra_images'):
                                image = result[0]
                                pp.extra_images.extend(result[1:])
                            else:
                                grid = make_grid(result)
                                result.insert(0, grid)
                                image = grid
                        pp.info["ReActor"] = True
                        pp.image = image
                        logger.status("---Done!---")
                    else:
                        logger.error("Cannot create a result image")
                else:
                    try:
                        pp.info["ReActor"] = True

                        res_img = result[0] if isinstance(result, list) else result
                        if alpha is not None:
                            logger.debug(f"result = {res_img}")
                            res_img = res_img.convert("RGBA")
                            res_img.putalpha(alpha)
                            logger.debug(f"result_alpha = {res_img}")

                        pp.image = res_img
                        logger.status("---Done!---")
                    except Exception:
                        logger.error("Cannot create a result image")
            else:
                logger.error("Please provide a source face")
