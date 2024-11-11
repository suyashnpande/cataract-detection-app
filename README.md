 step-by-step guide to running, testing, and deploying your cataract detection project using Streamlit.

 Step 1: Download Dataset folder from link: 
 https://www.kaggle.com/datasets/nandanp6/cataract-image-dataset 
 Add it in model_training folder and rename it as processed_images 

 Step 2: Navigate to Your Project Directory:
        Use the cd command to navigate to your project folder. For example :
        cd path\to\your_project_directory

Step 3: Create and Activate a Virtual Environment
      1. Create a Virtual Environment:
        Run the following command to create a virtual environment called venv:

        
         python -m venv venv
         
2. Activate the Virtual Environment with :
     
        venv\Scripts\activate

Step 4: Install Required Packages
With the virtual environment activated, run:

      python -m pip install --upgrade pip
      python -m pip install -r requirement.txt

Step 5: Train Your Model
        Run the Training Script:
        With the virtual environment still activated, navigate to the model training folder:
        cd model_training
        
Then run the training script:

        python train_model.py
        
This process will take some time as it trains the model. Monitor the command line for any errors or progress messages.

Step 6: Check Model Output
      Verify the Model Files:
      After training completes, ensure you have the following files in your project directory:
      final_model.keras (the trained model)
      best_model_v10.keras (if saved during training)

Step 8: Run the Streamlit Application
With the virtual environment activated and in your project root directory (where cataract_app.py is located), run:

      streamlit run cataract_app.py

Step 9: Deploy Your Streamlit App (optional)


-------------------------------------------------------

your_project_directory/
├── processed_images/
│   ├── train/
│   │   ├── cataract/
│   │   │   └── image1.jpg
│   │   │   └── image2.jpg
│   │   └── normal/
│   │       └── image1.jpg
│   └── test/
│       ├── cataract/
│       └── normal/
├── model_training/
│   ├── train_model.py
│   └── run_training.bat
├── cataract_app.py
├── requirements.txt
└── final_model.keras
( may looks like this )
