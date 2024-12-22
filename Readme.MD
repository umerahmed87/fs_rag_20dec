-- creating a virtual environment 
python -m venv ragfs

for windows
the terminal changes thus the code tooo 

we have to run this code
Set-ExecutionPolicy Bypass -Scope Process
for the cmd terminal

ragfs\Scripts\activate.bat

for the powershell 

.\ragfs\Scripts\Activate.ps1


for the unix one

source ragfs\Scripts\activate


-- we convert working directory to local or git repo using git init command
--  to track file we write git add filename , git add.
-- git status to check the files that I am tracking.


--let's commit git commit  -m "v1"





git remote add origin https://github.com/umerahmed87/fs_rag_20dec.git
git branch -M main
git push -u origin main