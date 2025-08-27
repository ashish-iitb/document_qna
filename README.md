# Document QnA

This project allows you to upload documents (PDFs) and ask questions to get accurate answers.

---

## 📂 Setup Instructions

### 1. Navigate to the repo folder

```bash
cd <repo-folder>
```

### 2. Create a folder for PDFs
Inside your repository directory, create a folder named `pdfs` and place all your PDF files there.

```bash
mkdir pdfs
```
### 3. Install Python 3.10 or above

Make sure you have Python 3.11 installed.
You can check with:

```bash
python --version
```
or
```bash
python3 --version
```
### 4.Create a Virtual Environment (Optional best practise)
On macOS / Linux:
```bash
python3.11 -m venv your_environment_name_without_qoutes  # this creates the virtual envirnment (takes upto 1 min) replace placeholder with your choice of virtual env name 
source <environment_name_without_qoutes>/bin/activate # activates the environment you can see the environment name in terminal
```

On Windows (PowerShell):
```powershell
python -m venv your_environment_name_without_qoutes  # this creates the virtual envirnment (takes upto 1 min) replace placeholder with your choice of virtual env name
your_environment_name_without_qoutes\Scripts\activate  # activates the environment you can see the environment name in terminal
```
To deactivate:
```bash
deactivate
```
### 5. Install Dependencies
```bash
pip install -r requirements.txt
```

### 6. Create and add .env file 
Add all your environment variables/ secret keys in this file for local testing (never push the secret keys to remote branch)


### 7. Run the Application
```bash
python document_qna.py
```



