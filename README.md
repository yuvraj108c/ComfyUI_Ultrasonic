# ComfyUI_Sonic
[Sonic](https://github.com/jixiaozhong/Sonic) is a method about ' Shifting Focus to Global Audio Perception in Portrait Animation',you can use it in comfyUI

# Update
* Change the model loading to a monolithic SVD model 模型加载改为单体SVD模型；  
* add frame number to change infer legth. 新增frame number选项，用于控制输出视频的长度（如果无限大，就是基于音频长度）；
* Support output of non square images，OOM 支持非正方形图片的输出，容易OOM；
* image_size is used to control the minimum size of the output image. If OOM, please reduce this value ,image_size用于控制输出图片的最小尺寸，如果OOM请调小这个数值；
* 感谢@civen-cn 提交的PR


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
*  3.2 SVD checkpoints  [svd_xt.safetensors](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)  or [svd_xt_1_1.safetensors](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1)    

```
--   ComfyUI/models/checkpoints
    ├── svd_xt.safetensors  or  svd_xt_1_1.safetensors
```

# Example
![](https://github.com/smthemex/ComfyUI_Sonic/blob/main/exampleA.png)


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
