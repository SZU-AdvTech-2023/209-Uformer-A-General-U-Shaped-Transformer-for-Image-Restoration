复现分两个版本：
1. KaagleVersionUformer版本下载后导入Kaggle即可运行。导入的数据集可以在datasets搜索KaShingWong获取。
2. 第二个版本实现在Uformer-main文件夹，项目的依赖是Pytorch1.9.0,Python 3.7,CUDA11.1,可以用以下方式进行安装。
```bash
pip install -r requirements.txt
```
对于训练SIDD的数据可以从[官方网址](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php)进行下载，然后通过下面的方式生成训练数据补丁。
```python
python3 generate_patches_SIDD.py --src_dir ../SIDD_Medium_Srgb/Data --tar_dir ../datasets/denoising/sidd/train
```
若要在SIDD上训练Uformer，可以使用下面的命令：
```bash
sh script/train_denoise.sh
```
注意上面的版本要在linux下使用，如果在windows可能会发生错误。