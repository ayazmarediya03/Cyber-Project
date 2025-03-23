import os
try:
    import pandas as pd
    import numpy as np
    from flask import Flask, render_template, request, redirect, url_for, send_file, flash
    import anjana.anonymity as anonymity
    from anjana.anonymity import utils
except ImportError as e:
    print(f"{e}. Installing missing packages...")
    import os
    os.system("pip install pandas numpy flask anjana")  # Install required libraries
    import pandas as pd
    import numpy as np
    from flask import Flask, render_template, request, redirect, url_for, send_file, flash
    import anjana.anonymity as anonymity
    from anjana.anonymity import utils

# Now you can use all the imported libraries
print("All libraries imported successfully!")

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
        # Get column roles from the form
        roles = {}
        for col in columns:
            role = request.form.get(col)
            if role and role != 'none':
                roles[col] = role
        if not roles:
            flash("Please select a role for at least one column!")
            return render_template('select_columns.html', columns=columns, filename=filename, original_preview=original_preview)
        
        # Get the chosen anonymization method
        method = request.form.get("method")
        
        # Get k and suppression level from the form
        try:
            k_value = int(request.form.get("k"))
            supp_level_value = int(request.form.get("supp_level"))
        except (TypeError, ValueError):
            flash("Invalid k or suppression level value. Please enter valid integers.")
            return render_template('select_columns.html', columns=columns, filename=filename, original_preview=original_preview)
        
        # If l-diversity is chosen, get l value and verify that exactly one sensitive attribute is selected.
        if method == "l_diversity":
            try:
                l_div = int(request.form.get("l_div"))
            except (TypeError, ValueError):
                flash("Invalid l value for l-diversity. Please enter a valid integer.")
                return render_template('select_columns.html', columns=columns, filename=filename, original_preview=original_preview)
        
        # Build lists of column names by role
        quasi_ident = [col for col, role in roles.items() if role == 'quasi']
        ident = [col for col, role in roles.items() if role == 'ident']
        sens_att_list = [col for col, role in roles.items() if role == 'sensitive']
        
        # Build hierarchies for quasi-identifiers and static sensitive hierarchies.
        hierarchies = {}
        
        # Hierarchy for ZipCode / zipcode
        zipcode_field = None
        if 'ZipCode' in quasi_ident and 'ZipCode' in df.columns:
            zipcode_field = 'ZipCode'
        elif 'zipcode' in quasi_ident and 'zipcode' in df.columns:
            zipcode_field = 'zipcode'
        if zipcode_field:
            hierarchies[zipcode_field] = {
                0: df[zipcode_field].values,
                1: df[zipcode_field].astype(str).str[:-1] + "*",
                2: df[zipcode_field].astype(str).str[:-2] + "**",
                3: df[zipcode_field].astype(str).str[:-3] + "***",
                4: np.array(["*"] * len(df[zipcode_field].values))
            }
        
        # Hierarchy for Age / age
        age_field = None
        if 'Age' in quasi_ident and 'Age' in df.columns:
            age_field = 'Age'
        elif 'age' in quasi_ident and 'age' in df.columns:
            age_field = 'age'
        if age_field:
            def generate_intervals(values, interval):
                intervals = []
                for v in values:
                    lower_bound = (v // interval) * interval
                    upper_bound = lower_bound + interval - 1
                    intervals.append(f"{lower_bound}-{upper_bound}")
                return np.array(intervals)
            hierarchies[age_field] = {
                0: df[age_field].values,
                1: generate_intervals(df[age_field].values, 5),
                2: generate_intervals(df[age_field].values, 10)
            }
        
        # Hierarchy for Gender (sensitive)
        if 'Gender' in sens_att_list and 'Gender' in df.columns:
            hierarchies['Gender'] = {
                0: df['Gender'].values,
                1: np.array(["*"] * len(df['Gender'].values))
            }
        
        # Hierarchy for Occupation (sensitive)
        if 'Occupation' in sens_att_list and 'Occupation' in df.columns:
            hierarchies['Occupation'] = {
                0: df['Occupation'].values,
                1: df['Occupation'].astype(str).str[0] + "*",
                2: np.array(["*"] * len(df['Occupation'].values))
            }
        
        # Run the anonymization process according to chosen method
        try:
            if method == "l_diversity":
                if len(sens_att_list) != 1:
                    flash("For l-diversity, please select exactly one sensitive attribute.")
                    return render_template('select_columns.html', columns=columns, filename=filename, original_preview=original_preview)
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
        
        # Save the anonymized CSV
        processed_filename = f'anonymized_{filename}'
        processed_filepath = os.path.join(PROCESSED_FOLDER, processed_filename)
        anonymized_df.to_csv(processed_filepath, index=False)
        
        anonymized_preview = anonymized_df.head(10).to_html(classes="table table-striped", index=False)
        return render_template('preview.html', anonymized_table=anonymized_preview, download_filename=processed_filename)
    
    return render_template('select_columns.html', columns=columns, filename=filename, original_preview=original_preview)

@app.route('/download/<filename>')
def download_file(filename):
    filepath = os.path.join(PROCESSED_FOLDER, filename)
    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
