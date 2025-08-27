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
if not download it from https://www.python.org/downloads/
You can check with:

```bash
python --version
```
or
```bash
python3 --version
```
### 4.Create a Virtual Environment (Optional best practise)
Suffix here should be your python version (i.e. python3.11)

On macOS / Linux:
```bash
python3.11 -m venv your_environment_name_without_qoutes  # this creates the virtual envirnment (takes upto 1 min) replace placeholder with your choice of virtual env name 
source <environment_name_without_qoutes>/bin/activate # activates the environment you can see the environment name in terminal
```

On Windows (PowerShell):
```powershell
python3.11 -m venv your_environment_name_without_qoutes  # this creates the virtual envirnment (takes upto 1 min) replace placeholder with your choice of virtual env name
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

### Sample output

```json
{
    "trustName": "Noongar Boodja Trust",
    "basicInformation": {
        "dateOfDeed": "not found",
        "settledSum": "not found",
        "governingLaw": "The laws of Western Australia."
    },
    "trustTerm": {
        "commencementDate": "not found",
        "terminationDate": "not found"
    },
    "parties": {
        "settlor": {
            "name": "Noongar Boodja Trust",
            "address": "care of South West Aboriginal Land and Sea Council, Level 2, 100 Royal Street, East Perth, Western Australia",
            "restrictions": "Yes. Clause 22.3(a) provides that, subject to clause 22.3(b), the Settlor (and the Settlor’s legal personal representative or Associate) may not become an Eligible Noongar Entity or directly or indirectly receive any part of the Trust Fund or its income."
        },
        "trustee": {
            "name": "Perpetual Trustee Company Limited",
            "acn": "000 001 007",
            "address": "Level 18, 123 Pitt Street, Sydney, New South Wales",
            "director": "not found"
        },
        "appointor": {
            "name": "Attorney General, Noongar Appointor",
            "powers": [
                "The Attorney General and the Noongar Appointor act jointly as the Appointors. Acting together and by instrument in writing, they may remove the Trustee, appoint additional Trustees who meet the clause 13.2 qualifications, and appoint a new Trustee to replace a Trustee who resigns or ceases by operation of law. Before exercising these powers, they must consult with the Noongar Advisory Company and the existing or outgoing Trustee, and before removing a Trustee they must consider whether the Trustee has breached or failed to act satisfactorily under the Deed. They may request the Trustee to conduct an appropriate selection process for a replacement trustee, and if there is no Trustee, they must conduct that selection process in consultation with the Noongar Advisory Company. When a Nominee Entity is proposed, they must assess whether it meets the Dedicated Trustee Requirements and, if they decide not to appoint it, provide written reasons sufficient to enable it to remedy the issues. Upon exercising their powers, they ensure the incoming and outgoing trustees execute a Deed of Appointment substantially in the form in Schedule 8. The Attorney General must act jointly with the Noongar Appointor for appointments and removals under clause 13.4."
            ]
        }
    }
```

