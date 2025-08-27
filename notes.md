# Project Notes

This is a file for local notes on this project.
This file should be ignored in .gitignore!

Workflow

(step number is in the file name for easier team viewing -- not only does it show the step wise usage of these scripts, but it also reorders them in our script folder for visual cues)

1. We first will load our data in with clean path objects using src/paths.py
2. We'll then do a variety of sanity checking on our loaded data using our scripts/01_data_check.py script
   1. added this to our data check script to ensure paths are working properly.
     print("TRAIN_CSV:", TRAIN_CSV.exists(), TRAIN_CSV)
     print("TEST_CSV:", TEST_CSV.exists(), TEST_CSV)

3. After the dataset has been checked and it looks like we expected it to do, we proceed to EDA by running scripts/02_eda.py
