1. Download latest Python: https://www.python.org/downloads/macos/
2. Make Python 3 the default
```
vi ~/.bash_aliases
add:
  alias python=/usr/local/bin/python3
  alias pip=/Library/Frameworks/Python.framework/Versions/3.12/bin/pip3
source ~/.bash_aliases
```
3. Generate the virtual environment (https://docs.python.org/3/library/venv.html) 
```
python -m venv /Volumes/NVMe/Development/IdeaProjects/AI/chat-with-pdfs/venv
```
4. Install dependencies:
```
pip install -r requirements.txt
```
5. Run the project:
```
streamlit run app.py
```