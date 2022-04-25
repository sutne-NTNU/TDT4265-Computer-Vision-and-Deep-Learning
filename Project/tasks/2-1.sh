echo -e "\nTask 2.1 - Creating your first baseline."
echo -e "Baseline should achieve an mAP@0.5:0.95 of 0.037\n"
cd src
python3 train.py configs/baseline.py
