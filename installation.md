# Installation Instructions for Accessibility Assistant

Accessibility Assistant is a Python application that requires certain dependencies to be installed in order to function properly. Follow the steps below to install and set up the application:

1. Clone the Repository:
   - Start by cloning the Accessibility Assistant repository from GitHub to your local machine:
     ```
     git clone <repository-url>
     ```

2. Navigate to the Project Directory:
   - Change directory to the root directory of the cloned repository:
     ```
     cd Accessibility-Assistant
     ```

3. Create a Virtual Environment (Optional but Recommended):
   - It's recommended to create a virtual environment to isolate the project dependencies:
     ```
     python -m venv venv
     ```

4. Activate the Virtual Environment (Optional but Recommended):
   - Activate the virtual environment to ensure that the project dependencies are installed in an isolated environment:
     - On Windows:
       ```
       venv\Scripts\activate
       ```
     - On macOS/Linux:
       ```
       source venv/bin/activate
       ```

5. Install Required Dependencies:
   - Use pip to install the required Python dependencies listed in the `requirements.txt` file:
     ```
     pip install -r requirements.txt
     ```

6. Download Pre-trained Models (If Necessary):
   - If the `prediction.h5` file (pre-trained machine learning model) is not included in the repository, download it from the appropriate source and place it in the `models/` directory.

7. Run the Streamlit Application:
   - Once all dependencies are installed, you can run the Streamlit application using the following command:
     ```
     streamlit run main.py
     ```

8. Access the Application:
   - Open a web browser and navigate to the URL provided by Streamlit (usually http://localhost:8501) to access the Accessibility Assistant application.

9. Enjoy Using Accessibility Assistant:
   - You're now ready to use Accessibility Assistant. Explore the functionalities and features provided by the application, and feel free to add more meaningful changes!

For further details on how the app is used, go to [Usage](usage.txt)
