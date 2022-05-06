cd src
echo -e "\nTask 1.1 - Getting to know your dataset"
echo -e "\nAnalyzing Data"
python3 -m dataset_exploration
echo -e "\nWriting Annotation Images"
python3 -m save_images_with_annotations