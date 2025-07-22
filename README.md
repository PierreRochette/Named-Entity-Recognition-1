1. Create venv + activate it
2. `pip install -r requirements.txt`
3. `python3 -m spacy download ${model_name}` (in this tutorial : `en_core_web_md`)

For `second.py` file, if this error pop out : 
```
ValueError: [E955] Can't find table(s) lexeme_norm for language 'en' in spacy-lookups-data. Make sure you have the package installed or provide your own lookup tables if no default lookups are available for your language.
```

Try command : `pip install -U spacy-lookups-data` then restart file
