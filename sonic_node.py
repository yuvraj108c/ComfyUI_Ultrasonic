# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
import numpy as np
from omegaconf import OmegaConf
from diffusers import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import WhisperModel, AutoFeatureExtractor
import random
import io
import torchaudio
from .src.models.base.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from .sonic import Sonic, sonic_predata, preprocess_face, crop_face_image
from .src.dataset.test_preprocess import image_audio_to_tensor
from .src.models.audio_adapter.audio_proj import AudioProjModel
from .src.models.audio_adapter.audio_to_bucket import Audio2bucketModel
from .node_utils import tensor2cv, cv2pil
from .src.dataset.face_align.align import AlignImage

import folder_paths

MAX_SEED = np.iinfo(np.int32).max
current_node_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

# add checkpoints dir
SONIC_weigths_path = os.path.join(folder_paths.models_dir, "sonic")
if not os.path.exists(SONIC_weigths_path):
    os.makedirs(SONIC_weigths_path)
folder_paths.add_model_folder_path("sonic", SONIC_weigths_path)


class SONICLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "svd_repo": ("STRING", {"default": "stabilityai/stable-video-diffusion-img2vid-xt-1-1"}),
                "sonic_unet": (["none"] + folder_paths.get_filename_list("sonic"),),
                "ip_audio_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "use_interframe": ("BOOLEAN", {"default": True},),
                "dtype": (["fp16", "fp32", "bf16"],),
            },
        }

    RETURN_TYPES = ("MODEL_SONIC",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loader_main"
    CATEGORY = "SONIC"

    def loader_main(self, svd_repo, sonic_unet, ip_audio_scale, use_interframe, dtype):

        if dtype == "fp16":
            weight_dtype = torch.float16
        elif dtype == "fp32":
            weight_dtype = torch.float32
        elif dtype == "bf16":
            weight_dtype = torch.bfloat16
        else:
            raise ValueError(
                f"Do not support weight dtype: {dtype} during training"
            )

        # check model is exits or not,if not auto downlaod
        flownet_ckpt = os.path.join(SONIC_weigths_path, "RIFE")

        if sonic_unet != "none":
            sonic_unet = folder_paths.get_full_path("sonic", sonic_unet)

        # load model
        print("***********Load model ***********")
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            svd_repo,
            subfolder="vae",
            variant="fp16")

        val_noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            svd_repo,
            subfolder="scheduler")

        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            svd_repo,
            subfolder="unet",
            variant="fp16")
        pipe = Sonic(device, weight_dtype, vae, val_noise_scheduler, unet, flownet_ckpt, sonic_unet,
                     use_interframe, ip_audio_scale)

        print("***********Load model done ***********")
        gc.collect()
        torch.cuda.empty_cache()
        return (pipe,)


class SONIC_PreData:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_vision": ("CLIP_VISION",),
                "audio": ("AUDIO",),
                "image": ("IMAGE",),
                "min_resolution": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64, "display": "number"}),
                "expand_ratio": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
            }}

    RETURN_TYPES = ("SONIC_PREDATA", "BBOX",)
    RETURN_NAMES = ("data_dict", "bbox",)
    FUNCTION = "sampler_main"
    CATEGORY = "SONIC"

    def sampler_main(self, clip_vision, audio, image, min_resolution, expand_ratio):
        config_file = os.path.join(current_node_path, 'config/inference/sonic.yaml')
        config = OmegaConf.load(config_file)

        audio2token_ckpt = os.path.join(SONIC_weigths_path, "audio2token.pth")
        audio2bucket_ckpt = os.path.join(SONIC_weigths_path, "audio2bucket.pth")
        yolo_ckpt = os.path.join(SONIC_weigths_path, "yoloface_v5m.pt")

        if not os.path.exists(audio2bucket_ckpt) or not os.path.exists(audio2token_ckpt) or not os.path.exists(
                yolo_ckpt):
            raise Exception("Please download the model first")
        # init model
        whisper_repo = os.path.join(SONIC_weigths_path, "whisper-tiny")

        whisper = WhisperModel.from_pretrained(whisper_repo).to(device).eval()
        whisper.requires_grad_(False)

        feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_repo)

        audio2token = AudioProjModel(seq_len=10, blocks=5, channels=384, intermediate_dim=1024, output_dim=1024,
                                     context_tokens=32).to(device)
        audio2bucket = Audio2bucketModel(seq_len=50, blocks=1, channels=384, clip_channels=1024, intermediate_dim=1024,
                                         output_dim=1, context_tokens=2).to(device)

        audio2token_dict = torch.load(audio2token_ckpt, map_location="cpu")
        audio2bucket_dict = torch.load(audio2bucket_ckpt, map_location="cpu")
        audio2token.load_state_dict(
            audio2token_dict,
            strict=True,
        )

        audio2bucket.load_state_dict(
            audio2bucket_dict,
            strict=True,
        )
        del audio2token_dict, audio2bucket_dict

        audio_file_prefix = ''.join(random.choice("0123456789") for _ in range(6))
        audio_path = os.path.join(folder_paths.get_input_directory(), f"audio_{audio_file_prefix}_temp.wav")

        # 减少音频数据传递导致的不必要文件存储
        buff = io.BytesIO()
        torchaudio.save(buff, audio["waveform"].squeeze(0), audio["sample_rate"], format="FLAC")
        with open(audio_path, 'wb') as f:
            f.write(buff.getbuffer())
        gc.collect()
        torch.cuda.empty_cache()

        face_det = AlignImage(device, det_path=yolo_ckpt)

        # 先面部裁切处理
        cv_image = tensor2cv(image)
        face_info = preprocess_face(cv_image, face_det, expand_ratio=expand_ratio)
        if face_info['face_num'] > 0:
            crop_image_pil = cv2pil(crop_face_image(cv_image, face_info['crop_bbox']))

        test_data = image_audio_to_tensor(face_det, feature_extractor, crop_image_pil, audio_path,
                                          limit=config.frame_num, image_size=min_resolution, area=config.area)

        step = 2
        for k, v in test_data.items():
            if isinstance(v, torch.Tensor):
                test_data[k] = v.unsqueeze(0).to(device).float()
        ref_img = test_data['ref_img']
        audio_feature = test_data['audio_feature']
        audio_len = test_data['audio_len']

        ref_tensor_list, audio_tensor_list, uncond_audio_tensor_list, motion_buckets, image_embeddings = sonic_predata(
            whisper, audio_feature, audio_len, step, audio2bucket, clip_vision, audio2token, ref_img, image, device)
        del clip_vision, face_det, whisper
        audio2bucket.to("cpu")
        audio2token.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        bbox_c = face_info['crop_bbox']
        bbox = [bbox_c[0], bbox_c[1], bbox_c[2] - bbox_c[0], bbox_c[3] - bbox_c[1]]
        return ({"test_data": test_data, "ref_tensor_list": ref_tensor_list, "config": config,
                 "image_embeddings": image_embeddings,
                 "audio_tensor_list": audio_tensor_list, "uncond_audio_tensor_list": uncond_audio_tensor_list,
                 "motion_buckets": motion_buckets}, bbox,)


class SONICSampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_SONIC",),
                "data_dict": ("SONIC_PREDATA",),  # {}
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "inference_steps": ("INT", {"default": 25, "min": 1, "max": 1024, "step": 1, "display": "number"}),
                "dynamic_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "fps": ("FLOAT", {"default": 25.0, "min": 5.0, "max": 120.0, "step": 0.5}),
            }}

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("image", "fps")
    FUNCTION = "sampler_main"
    CATEGORY = "SONIC"

    def sampler_main(self, model, data_dict, seed, inference_steps, dynamic_scale, fps):
        print("***********Start infer  ***********")
        iamge = model.process(data_dict["audio_tensor_list"],
                              data_dict["uncond_audio_tensor_list"],
                              data_dict["motion_buckets"],
                              data_dict["test_data"],
                              data_dict["config"],
                              image_embeds=data_dict["image_embeddings"],
                              fps=fps,
                              inference_steps=inference_steps,
                              dynamic_scale=dynamic_scale,
                              seed=seed)
        gc.collect()
        torch.cuda.empty_cache()
        return (iamge.permute(0, 2, 3, 4, 1).squeeze(0), fps)


NODE_CLASS_MAPPINGS = {
    "SONICTLoader": SONICLoader,
    "SONIC_PreData": SONIC_PreData,
    "SONICSampler": SONICSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SONICTLoader": "SONICTLoader",
    "SONIC_PreData": "SONIC_PreData",
    "SONICSampler": "SONICSampler",
}
