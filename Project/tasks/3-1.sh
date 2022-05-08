cd src

echo -e "\nTask 3.1 - Performing Performence assessment on all models."                     
python3 assessment/runtime_analysis.py configs/baseline.py 
python3 assessment/runtime_analysis.py configs/task_2_2.py 
python3 assessment/runtime_analysis.py configs/task_2_3_1.py 
python3 assessment/runtime_analysis.py configs/task_2_3_2.py 
python3 assessment/runtime_analysis.py configs/task_2_3_3.py 
python3 assessment/runtime_analysis.py configs/task_2_3_4.py 
python3 assessment/runtime_analysis.py configs/task_2_4.py 
python3 assessment/runtime_analysis.py configs/updated_dataset.py 

echo -e "\nTask 3.1 - Benchmarking Data Loading"    
python3 assessment/benchmark_data_loading.py configs/baseline.py 
python3 assessment/benchmark_data_loading.py configs/task_2_2.py 
