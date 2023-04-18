# CNN卷积核局部与全局信息提取能力测试实验

## Result

### Table:Whether the model converges

| Dataset            | OneKernel | TwoKernels | DCNN | ViT_18 | ViT_36 |
| ------------------ | --------- | ---------- | ---- | ------ | ------ |
| Random Middle Area | √        | ×         | ×   | √     | √     |
| Fix Middle Area    | √        | √         | √   | √     | √     |
| None Middle Area   | √        | √         | √   | √     | √     |
| Cats and Dogs      | ⍻        | ⍻         | √   | ⍻     | ⍻     |

### LOSS

![1681818069564](image/README/1681818069564.png)

### Accuracy

![1681818095649](image/README/1681818095649.png)

## model

### Onekernel

![1681354500244](image/README/1681354500244.png)

### Twokernels

![1681354574429](image/README/1681354574429.png)(730,000,000 FLOPs)

### DCNN

![1681473348019](image/README/1681473348019.png)(3350,000,000,000 FLOPs)

### ViT_36

![1681806196794](image/README/1681806196794.png)

### ViT_18

![1681807567888](image/README/1681807567888.png)

## DataSet

> Random_middle_Area -- dataset.py
>
> Fix_middle_Area -- dataset.py
>
> None_middle_Area -- dataset.py
>
> CatvsDog -- [Cat and Dog | Kaggle](https://www.kaggle.com/datasets/tongpython/cat-and-dog)
