# TDSNet

## Environment Setup

```bash
conda create -n tdsnet python=3.10

conda activate tdsnet

pip install -r requirements.txt
````

Clone the DINOv3 repository:

```bash
git clone https://github.com/facebookresearch/dinov3.git
```

Download [DINOv3](https://github.com/facebookresearch/dinov3)  pretrained weights and place them in:

```
./web_pth
```

## Datasets

- **[PAL4Inpaint](https://github.com/owenzlz/PAL4Inpaint)**  

- **[PAL4VST](https://github.com/owenzlz/PAL4VST)**  


- **[SynthScars](https://huggingface.co/datasets/khr0516/SynthScars)**  

Organize datasets in the following structure:

```
./datasets\PAL4Inpaint\train\image
./datasets\PAL4Inpaint\train\mask
./datasets\PAL4Inpaint\test\image
./datasets\PAL4Inpaint\test\mask
...

```

## Notes

* Make sure the pretrained DINO weights (`.pth` file) are correctly downloaded and placed under `./web_pth`.
* Modify paths as needed for your environment.
* Training and testing configurations (e.g., image size, batch size, learning rate) can be adjusted via command-line arguments.

## Acknowledgements

We would like to thank the open-source community for their invaluable contributions.  
In particular, we acknowledge the following repositories that made this work possible:

- [DINOv3](https://github.com/facebookresearch/dinov3)   
