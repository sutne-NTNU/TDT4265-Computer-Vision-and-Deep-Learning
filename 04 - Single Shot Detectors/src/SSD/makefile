TASK4B 		 = python3 train.py configs/ssd300.py
TASK4C 		 = python3 train.py configs/ssd300_improved.py
TASK4E       = python3 demo.py configs/ssd300_improved.py demo/mnist demo/mnist_output
TASK4F_TRAIN = python3 train.py configs/voc_vgg.py
TASK4F_DEMO  = python3 demo.py configs/voc_vgg.py demo/voc demo/voc_output

# Training will automatically save logs/stats to "/outputs". To open the logs in VSCode's tensorboard:
# 	- Open the VSCode command palette and select: "Python: Launch TensorBoard"
# 	- Select "/outputs" as the root folder when prompted.

task4b:
	clear
	$(TASK4B)

task4c:
	clear
	$(TASK4C)

task4e: 
	clear
	$(TASK4E)

task4f-train:
	clear
	$(TASK4F_TRAIN)

task4f-demo:
	clear
	$(TASK4F_DEMO)
