# Shift-Net
video restoration

## Get Started

### Installation
```python
python 3.8.5
pytorch 1.8.0
cuda 11.3
```

```
cd Shift-Net/
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

### Dataset:
* download [test set](https://drive.google.com/file/d/1zS9BmfLGNk8EFA6LkTWXQF6UKUnhvB_k/view?usp=sharing) to ./dataset/GOPRO/test/
* download [train set](https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view?usp=sharing) to ./dataset/GOPRO/train/

### Training:

* training script for base model:
```
python3 -m torch.distributed.launch --nproc_per_node=8 basicsr/train1.py -opt options/gopro-gshiftnet1.yml --launcher pytorch
```
* training script for small model:
```
python3 -m torch.distributed.launch --nproc_per_node=8 basicsr/train1.py -opt options/gopro-gshiftnet2.yml --launcher pytorch
```
 
### Testing Flops:
* ```python3 basicsr/models/archs/gshift1.py```
* ```python3 basicsr/models/archs/gshift2.py``` 


### Testing: 
* eval: We provide the pre-trained model for evaluation.
* Please download the model [pretrained model]() to ./checkpoints/gshift_net_gopro.pth
* ```python3 test1.py ``` for base model.
* ```python3 test2.py ``` for small model.



