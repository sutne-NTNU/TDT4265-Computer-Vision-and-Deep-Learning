echo -e "\nTask 1.1 - Getting to know your dataset"
cd src
echo -e "\nAnalyzing Data"
python -m benchmarks.dataset_exploration
echo -e "\nWriting Annotation Images"
python -m benchmarks.save_images_with_annotations