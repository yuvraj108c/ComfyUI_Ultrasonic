# ComfyUI_Sonic
[Sonic](https://github.com/jixiaozhong/Sonic) is a method about ' Shifting Focus to Global Audio Perception in Portrait Animation',you can use it in comfyUI

# 1. Installation

In the ./ComfyUI/custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_Sonic.git
```
---

# 2. Requirements  

```
pip install -r requirements.txt
```

# 3.Model
* 3.1.1 download  checkpoints  from [google](https://drive.google.com/drive/folders/1oe8VTPUy0-MHHW2a_NJ1F8xL-0VN5G7W) 从Google下载必须的模型,文件结构如下图
* 3.1.2 download [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny/tree/main)
```
--  ComfyUI/models/sonic/
    |-- audio2bucket.pth
    |-- audio2token.pth
    |-- unet.pth
    |-- yoloface_v5m.pt
    |-- whisper-tiny/
        |--config.json
        |--model.safetensors
        |--preprocessor_config.json
    |-- RIFE/
        |--flownet.pkl
```
*  3.2 SVD repo [stabilityai/stable-video-diffusion-img2vid-xt
](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)  or [stabilityai/stable-video-diffusion-img2vid-xt-1-1](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) online or offline   
* if offline
```
--   anypath/stable-video-diffusion-img2vid-xt/  # or stable-video-diffusion-img2vid-xt-1-1 
    ├── model_index.json
    ├── vae...
    ├── unet...
    ├── feature_extractor...
    ├── scheduler...
```
* 3.3 clip_vison
```
--  ComfyUI/models/clip_vision/
    ├── clip_vision_H.safetensors   # or 'stabilityai/stable-video-diffusion-img2vid-xt' image encoder safetensors or ipadapter image encoder
```

# Example
![](https://github.com/smthemex/ComfyUI_Sonic/blob/main/example.png)


# Citation
```
@misc{ji2024sonicshiftingfocusglobal,
      title={Sonic: Shifting Focus to Global Audio Perception in Portrait Animation}, 
      author={Xiaozhong Ji and Xiaobin Hu and Zhihong Xu and Junwei Zhu and Chuming Lin and Qingdong He and Jiangning Zhang and Donghao Luo and Yi Chen and Qin Lin and Qinglin Lu and Chengjie Wang},
      year={2024},
      eprint={2411.16331},
      archivePrefix={arXiv},
      primaryClass={cs.MM},
      url={https://arxiv.org/abs/2411.16331}, 
}
```
