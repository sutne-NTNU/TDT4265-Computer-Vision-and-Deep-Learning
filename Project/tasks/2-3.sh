cd src

echo -e "\nTask 2.3.1 - ResNet."                     
python3 train.py configs/task_2_3_1.py 

echo -e "\nTask 2.3.2 - ResNet - Adding Focal Loss"   
python3 train.py configs/task_2_3_2.py 

echo -e "\nTask 2.3.3 - ResNet - Adding Deep Heads"   
python3 train.py configs/task_2_3_3.py 

echo -e "\nTask 2.3.4 - ResNet - Adding Improved Weight Init"   
python3 train.py configs/task_2_3_4.py