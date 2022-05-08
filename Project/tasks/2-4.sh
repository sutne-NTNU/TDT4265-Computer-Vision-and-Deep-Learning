cd src

echo -e "\nTask 2.4 - Training The Best"                     
python3 train.py configs/task_2_4.py # train on small dataset to see if the model is any good
python3 train.py configs/updated_dataset.py # train on updated dataset before saving results

echo -e "\nTask 2.4 - Saving Validation Results for Leaderboard"   
python3 assessment/save_validation_results.py configs/updated_dataset.py results.json