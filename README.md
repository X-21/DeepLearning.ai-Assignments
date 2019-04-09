# DeepLearning.ai Assignments
These are Exercises for DeepLearning (by Andrew Ng) by python.



这是吴恩达老师的深度学习课程的习题。



It's for personal use only. If this project have any infringement, please tell me(or submit an issue) and I will delete this project immediately.



仅供个人使用，若侵权请私信我或直接在ISSUE中提出，我会立即删除该项目。



## Environment 环境
Python ---------- 3.6.8  
TensorFlow ------ 1.13.1  
Keras ------------ 2.2.4

## notes 备注 
1.在第四课第三周的练习中，原来的ipynb里yolo_outputs内四个矩阵的顺序和现在应该略有不同，box_xy和box_wh应该在第1、2个参数，而不是第2、3个参数，否则yolo_eval运行报错（ValueError: Dimensions must be equal, but are 2 and 80 for 'mul_1' (op: 'Mul') with input shapes: [19,19,5,2], [19,19,5,80].），推测是因为yolo_head函数的返回值顺序在后来的版本中发生了改变。


## references 参考
1. @fengdu78 的 [deeplearning_ai_books](https://github.com/fengdu78/deeplearning_ai_books)
<br>



