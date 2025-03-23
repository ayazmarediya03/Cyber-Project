import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import anjana.anonymity as anonymity
from anjana.anonymity import utils

app = Flask(__name__)
app.secret_key = 'some_secret_key'  # Change for production

# Directories for file storage
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Utility function for numeric intervals
def generate_intervals(values, interval):
    intervals = []
    for v in values:
        lower_bound = (v // interval) * interval
        upper_bound = lower_bound + interval - 1
        intervals.append(f"{lower_bound}-{upper_bound}")
    return np.array(intervals)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET','POST'])
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
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            flash("Error reading CSV: " + str(e))
            return redirect(request.url)
        preview_html = df.head(10).to_html(classes="table table-striped", index=False)
        return render_template('upload_preview.html', filename=file.filename, preview=preview_html)
    return render_template('upload.html')

@app.route('/select_columns', methods=['GET','POST'])
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
    
    # Get the list of columns from the CSV dynamically.
    columns = df.columns.tolist()
    # Build a dictionary with column types (numeric or string)
    col_types = {col: ("numeric" if pd.api.types.is_numeric_dtype(df[col]) else "string") 
                 for col in columns}
    original_preview = df.head(10).to_html(classes="table table-striped", index=False)
    
    if request.method == 'POST':
        # Collect roles for each column.
        roles = {}
        for col in columns:
            role = request.form.get(col)
            if role and role != 'none':
                roles[col] = role
        
        if not roles:
            flash("Please select a role for at least one column!")
            return render_template('select_columns.html', columns=columns, filename=filename,
                                   original_preview=original_preview, col_types=col_types)
        
        # Get the chosen anonymization method.
        method = request.form.get("method")
        
        # Get k and suppression level.
        try:
            k_value = int(request.form.get("k"))
            supp_level_value = int(request.form.get("supp_level"))
        except (TypeError, ValueError):
            flash("Invalid k or suppression level value. Please enter valid integers.")
            return render_template('select_columns.html', columns=columns, filename=filename,
                                   original_preview=original_preview, col_types=col_types)
        
        # If l-diversity is chosen, get the l value.
        if method == "l_diversity":
            try:
                l_div = int(request.form.get("l_div"))
            except (TypeError, ValueError):
                flash("Invalid l value for l-diversity. Please enter a valid integer.")
                return render_template('select_columns.html', columns=columns, filename=filename,
                                       original_preview=original_preview, col_types=col_types)
        
        # Build lists for each role.
        quasi_ident = [col for col, role in roles.items() if role == 'quasi']
        ident = [col for col, role in roles.items() if role == 'ident']
        sens_att_list = [col for col, role in roles.items() if role == 'sensitive']
        
        # Build the dynamic hierarchies.
        hierarchies = {}
        for col in roles:
            # Only apply hierarchy if role is quasi or sensitive.
            if roles[col] in ['quasi', 'sensitive']:
                # Get the selected hierarchy type from the form.
                hier_type = request.form.get(f"hier_{col}", "none")
                if hier_type != "none":
                    if col_types[col] == "numeric":
                        if hier_type == "interval5":
                            hierarchies[col] = {
                                0: df[col].values,
                                1: generate_intervals(df[col].values, 5)
                            }
                        elif hier_type == "interval10":
                            hierarchies[col] = {
                                0: df[col].values,
                                1: generate_intervals(df[col].values, 10)
                            }
                    else:  # string type
                        if hier_type == "substring":
                            # Level 0: original, Level 1: remove last character then append '*', Level 2: full suppression.
                            hierarchies[col] = {
                                0: df[col].values,
                                1: df[col].astype(str).apply(lambda x: x[:-1] + "*" if len(x) > 0 else "*").values,
                                2: np.array(["*"] * len(df[col].values))
                            }
                        elif hier_type == "suppression":
                            hierarchies[col] = {
                                0: df[col].values,
                                1: np.array(["*"] * len(df[col].values))
                            }
        # Debug: you may flash the generated hierarchies for inspection.
        # flash(str(hierarchies))
        
        # Run the anonymization process.
        try:
            if method == "l_diversity":
                # For l-diversity, require exactly one sensitive attribute.
                if len(sens_att_list) != 1:
                    flash("For l-diversity, please select exactly one sensitive attribute.")
                    return render_template('select_columns.html', columns=columns, filename=filename,
                                           original_preview=original_preview, col_types=col_types)
                anonymized_df = anonymity.l_diversity(
                    data=df,
                    ident=ident,
                    quasi_ident=quasi_ident,
                    sens_att=sens_att_list[0],
                    k=k_value,
                    l_div=l_div,
                    supp_level=supp_level_value,
                    hierarchies=hierarchies
                )
            else:
                anonymized_df = anonymity.k_anonymity(
                    data=df,
                    ident=ident,
                    quasi_ident=quasi_ident,
                    k=k_value,
                    supp_level=supp_level_value,
                    hierarchies=hierarchies
                )
        except Exception as e:
            flash("Anonymization error: " + str(e))
            return redirect(url_for('select_columns', filename=filename))
        
        # Save the anonymized CSV.
        processed_filename = f'anonymized_{filename}'
        processed_filepath = os.path.join(PROCESSED_FOLDER, processed_filename)
        anonymized_df.to_csv(processed_filepath, index=False)
        
        anonymized_preview = anonymized_df.head(10).to_html(classes="table table-striped", index=False)
        return render_template('preview.html', anonymized_table=anonymized_preview, download_filename=processed_filename)
    
    return render_template('select_columns.html', columns=columns, filename=filename,
                           original_preview=original_preview, col_types=col_types)

@app.route('/download/<filename>')
def download_file(filename):
    filepath = os.path.join(PROCESSED_FOLDER, filename)
    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
