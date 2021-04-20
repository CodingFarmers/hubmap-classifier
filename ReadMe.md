# 环境配置
```
conda create -n torch17 python==3.7
source activate torch17
pip install -r requirements.txt
```
验证环境配置
```
python test_enviroments.py
```

# 数据集配置
```
ln -s {源数据集路径} ranzcr
```

# 训练
下载对应的预训练权重，放到pretrained目录下  https://www.kaggle.com/ammarali32/startingpointschestx
代码开头指定了GPU 需手动更改
## 1. 训练teacher
```
python kaggle_teacher.py
```
## 2. 训练student
```
python kaggle_student.py
```
## 3. finetune
```
python kaggle_finetune.py
```
## 4. pesudo labeling
