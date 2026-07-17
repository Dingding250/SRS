# DINOv3

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

## Notes

* Make sure the pretrained DINO weights (`.pth` file) are correctly downloaded and placed under `./web_pth`.
* Modify paths as needed for your environment.
* Training and testing configurations (e.g., image size, batch size, learning rate) can be adjusted via command-line arguments.

## Acknowledgements

We would like to thank the open-source community for their invaluable contributions.  
In particular, we acknowledge the following repositories that made this work possible:

- [DINOv3](https://github.com/facebookresearch/dinov3)   