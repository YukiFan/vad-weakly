## 1. Usage
### 1.1 Dependencies

* Python ≥ 3.8
* PyTorch 1.9.0
* numpy
* Pillow
* torchvision
* scipy

### 1.2 Prepare data

The datasets used in the paper are available at the following links:

* [UCF-Crime](https://stuxidianeducn-my.sharepoint.com/personal/pengwu_stu_xidian_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fpengwu%5Fstu%5Fxidian%5Fedu%5Fcn%2FDocuments%2FUCF%2DCrime%2FI3D&ga=1)

* [XD-Violence](https://roc-ng.github.io/XD-Violence/)

* * *
## 2. Train

```python
python main.py
# you can modify your device number in config.py or main.py
```
* * *
## 3. Evalution
```python
python test.py
```
## 4. Result

**Result on UCF**

<img src="https://github.com/YukiFan/vad-weakly/blob/main/ucf.png" width="50%">

**Result on XD-violence**

<img src="https://github.com/YukiFan/vad-weakly/blob/main/xd.png" width="50%">
