import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from anjana.anonymity import k_anonymity, utils

app = Flask(__name__)
app.secret_key = 'some_secret_key'  # Change this for production

# Directories for file storage
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("No selected file")
            return redirect(request.url)
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        # Read the CSV to create a preview (first 10 rows)
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            flash("Error reading CSV: " + str(e))
            return redirect(request.url)
        preview_html = df.head(10).to_html(classes="table table-striped", index=False)
        return render_template('upload_preview.html', filename=file.filename, preview=preview_html)
    return render_template('upload.html')

@app.route('/select_columns', methods=['GET', 'POST'])
def select_columns():
    filename = request.args.get('filename')
    if not filename:
        flash("No file provided")
        return redirect(url_for('upload_file'))
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        flash("Error reading CSV: " + str(e))
        return redirect(url_for('upload_file'))
    
    columns = df.columns.tolist()
    original_preview = df.head(10).to_html(classes="table table-striped", index=False)
    
    if request.method == 'POST':
        roles = {}
        for col in columns:
            role = request.form.get(col)
            if role and role != 'none':
                roles[col] = role
        if not roles:
            flash("Please select a role for at least one column!")
            return render_template('select_columns.html', columns=columns, filename=filename, original_preview=original_preview)
        
        # Get k and supp_level values from the form
        try:
            k_value = int(request.form.get('k'))
            supp_level_value = int(request.form.get('supp_level'))
        except (TypeError, ValueError):
            flash("Invalid k or suppression level value. Please enter valid integers.")
            return render_template('select_columns.html', columns=columns, filename=filename, original_preview=original_preview)
        
        # Build lists of column names by role
        quasi_ident = [col for col, role in roles.items() if role == 'quasi']
        ident = [col for col, role in roles.items() if role == 'ident']
        sens_att = [col for col, role in roles.items() if role == 'sensitive']
        
        # Build hierarchies for quasi-identifier columns
        hierarchies = {}
        
        # Hierarchy for ZipCode (or zipcode)
        zipcode_field = None
        if 'ZipCode' in quasi_ident and 'ZipCode' in df.columns:
            zipcode_field = 'ZipCode'
        elif 'zipcode' in quasi_ident and 'zipcode' in df.columns:
            zipcode_field = 'zipcode'
        if zipcode_field:
            hierarchies[zipcode_field] = {
                0: df[zipcode_field].values,
                1: df[zipcode_field].astype(str).str[:-1] + "*",    # Remove last digit
                2: df[zipcode_field].astype(str).str[:-2] + "**",   # Remove last two digits
                3: df[zipcode_field].astype(str).str[:-3] + "***",  # Remove last three digits
                4: np.array(["*"] * len(df[zipcode_field].values))  # Full suppression as last resort
            }
        
        # Hierarchy for Age (or age)
        age_field = None
        if 'Age' in quasi_ident and 'Age' in df.columns:
            age_field = 'Age'
        elif 'age' in quasi_ident and 'age' in df.columns:
            age_field = 'age'
        if age_field:
            hierarchies[age_field] = {
                0: df[age_field].values,
                1: utils.generate_intervals(df[age_field].values, 0, 100, 5),
                2: utils.generate_intervals(df[age_field].values, 0, 100, 10),
            }
        
        # Hierarchies for sensitive attributes:
        # Hierarchy for Gender (or gender)
        gender_field = None
        if 'Gender' in sens_att and 'Gender' in df.columns:
            gender_field = 'Gender'
        elif 'gender' in sens_att and 'gender' in df.columns:
            gender_field = 'gender'
        if gender_field:
            hierarchies[gender_field] = {
                0: df[gender_field].values,
                1: np.array(["*"] * len(df[gender_field].values))  # Full suppression
            }
        
        # Hierarchy for Occupation (or occupation)
        occupation_field = None
        if 'Occupation' in sens_att and 'Occupation' in df.columns:
            occupation_field = 'Occupation'
        elif 'occupation' in sens_att and 'occupation' in df.columns:
            occupation_field = 'occupation'
        if occupation_field:
            hierarchies[occupation_field] = {
                0: df[occupation_field].values,
                1: df[occupation_field].astype(str).str[0] + "*",  # Partial generalization: first letter + *
                2: np.array(["*"] * len(df[occupation_field].values))  # Full suppression
            }
        
        # Run the k-anonymity process; if not possible, catch the exception.
        try:
            data_anon = k_anonymity(df, ident, quasi_ident, k_value, supp_level_value, hierarchies)
        except Exception as e:
            flash("K-anonymity is not possible with the provided parameters: " + str(e))
            return redirect(url_for('select_columns', filename=filename))
        
        # Save the anonymized CSV
        processed_filename = f'anonymized_{filename}'
        processed_filepath = os.path.join(PROCESSED_FOLDER, processed_filename)
        data_anon.to_csv(processed_filepath, index=False)
        
        anonymized_preview = data_anon.head(10).to_html(classes="table table-striped", index=False)
        return render_template('preview.html', anonymized_table=anonymized_preview, download_filename=processed_filename)
    
    return render_template('select_columns.html', columns=columns, filename=filename, original_preview=original_preview)

@app.route('/download/<filename>')
def download_file(filename):
    filepath = os.path.join(PROCESSED_FOLDER, filename)
    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
