cd src
echo -e "\nTask 3.2 - Qualitative Analysis"                     
python3 -m performance_assessment.save_comparison_images configs/baseline.py 
python3 -m performance_assessment.save_comparison_images configs/task_2_2.py 
python3 -m performance_assessment.save_comparison_images configs/task_2_3_1.py 
python3 -m performance_assessment.save_comparison_images configs/task_2_3_2.py 
python3 -m performance_assessment.save_comparison_images configs/task_2_3_3.py 
python3 -m performance_assessment.save_comparison_images configs/task_2_3_4.py 
python3 -m performance_assessment.save_comparison_images configs/task_2_4.py 
python3 -m performance_assessment.save_comparison_images configs/updated_dataset.py 

python3 -m performance_assessment.demo_video configs/updated_dataset.py ../videos/Video00003_combined.avi ../videos/output03.avi
python3 -m performance_assessment.demo_video configs/updated_dataset.py ../videos/Video00010_combined.avi ../videos/output10.avi
python3 -m performance_assessment.demo_video configs/updated_dataset.py ../videos/Video00016_combined.avi ../videos/output16.avi
