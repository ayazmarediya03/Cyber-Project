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

# import pandas as pd
# import numpy as np
# from flask import Flask, render_template, request, redirect, url_for, send_file, flash
# import anjana.anonymity as anonymity
# from anjana.anonymity import utils

app = Flask(__name__)
app.secret_key = 'some_secret_key'  # Change this in production

# Folders for storing uploaded and processed files
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def generate_intervals(values, interval):
    """
    Helper function to convert numeric values into interval strings.
    For example, if interval=10 and value=27 -> "20-29"
    """
    intervals = []
    for v in values:
        lower_bound = (v // interval) * interval
        upper_bound = lower_bound + interval - 1
        intervals.append(f"{lower_bound}-{upper_bound}")
    return np.array(intervals)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """
    Upload a CSV file and display its first 10 rows as a preview.
    """
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
    """
    Allows the user to assign roles to columns, pick the anonymization method (k-anonymity or l-diversity),
    choose k, suppression level (supp_level), (and l if l-diversity is selected), and define hierarchy types (none, masking, interval)
    for quasi or sensitive columns.
    """
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
    
    # Determine column types (for our purposes, we use pandas detection)
    col_types = {}
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_types[col] = "numeric"
        else:
            col_types[col] = "string"
    
    # Compute maximum string length for each column using the string representation.
    # This is done for every column so that masking works even for numeric columns.
    max_len_map = {}
    for col in columns:
        max_len_map[col] = df[col].astype(str).apply(len).max()
    
    original_preview = df.head(10).to_html(classes="table table-striped", index=False)
    
    if request.method == 'POST':
        # Collect roles for each column
        roles = {}
        for col in columns:
            role = request.form.get(f"{col}_role", "none")
            if role != "none":
                roles[col] = role
        
        if not roles:
            flash("Please select a role for at least one column!")
            return render_template(
                'select_columns.html',
                columns=columns,
                col_types=col_types,
                max_len_map=max_len_map,
                original_preview=original_preview
            )
        
        # Get the anonymization method (default to k-anonymity)
        method = request.form.get("method", "k_anonymity")
        
        # Get k and suppression level values
        try:
            k_value = int(request.form.get("k", "3"))
            supp_level_value = int(request.form.get("supp_level", "0"))
        except (TypeError, ValueError):
            flash("Invalid k or suppression level. Please enter valid integers.")
            return render_template(
                'select_columns.html',
                columns=columns,
                col_types=col_types,
                max_len_map=max_len_map,
                original_preview=original_preview
            )
        
        # If l-diversity is selected, get the l value.
        if method == "l_diversity":
            try:
                l_div = int(request.form.get("l_div", "2"))
            except (TypeError, ValueError):
                flash("Invalid l value for l-diversity. Please enter a valid integer.")
                return render_template(
                    'select_columns.html',
                    columns=columns,
                    col_types=col_types,
                    max_len_map=max_len_map,
                    original_preview=original_preview
                )
        
        # Build role-based lists
        quasi_ident = [col for col, r in roles.items() if r == 'quasi']
        ident = [col for col, r in roles.items() if r == 'ident']
        sens_att_list = [col for col, r in roles.items() if r == 'sensitive']
        
        # Build dynamic hierarchies based on user input.
        hierarchies = {}
        for i, col in enumerate(columns):
            if col in roles and roles[col] in ['quasi', 'sensitive']:
                hier_type_field = f"hier_type_{i}"
                hier_level_field = f"hier_level_{i}"
                chosen_type = request.form.get(hier_type_field, "none")
                chosen_level = request.form.get(hier_level_field, "none")
                
                # if chosen_type == "none":
                #     hierarchies[col] = {0: df[col].values}
                #     continue
                
                if False: #col_types[col] == "string":
                    # For string columns: if masking is chosen, build levels 0..mask_level
                    if chosen_type == "masking":
                        try:
                            mask_level = int(chosen_level)
                        except:
                            mask_level = 0
                        hierarchy_dict = {0: df[col].values}
                        for lvl in range(1, mask_level + 1):
                            def mask_func(x, lvl=lvl):
                                return x[:-lvl] + ("*" * lvl) if len(x) > lvl else "*" * lvl
                            hierarchy_dict[lvl] = df[col].astype(str).apply(mask_func).values
                        hierarchies[col] = hierarchy_dict
                    elif chosen_type == "interval":
                        try:
                            interval_val = int(chosen_level)
                        except:
                            interval_val = 10
                        if interval_val <= 0:
                            interval_val = 10
                        # For strings, interval is less meaningful – we use a simple suppression alternative.
                        hierarchies[col] = {
                            0: df[col].values,
                            1: np.array(["*"] * len(df[col].values))
                        }
                else:
                    # For numeric columns:
                    if chosen_type == "masking":
                        try:
                            mask_level = int(chosen_level)
                        except:
                            mask_level = 0
                        hierarchy_dict = {0: df[col].values}
                        for lvl in range(1, mask_level + 1):
                            def mask_num(n, lvl=lvl):
                                s = str(int(n))
                                return s[:-lvl] + ("*" * lvl) if len(s) > lvl else "*" * lvl
                            hierarchy_dict[lvl] = df[col].astype(str).apply(mask_num).values
                        hierarchies[col] = hierarchy_dict
                    elif chosen_type == "interval":
                        try:
                            interval_val = int(chosen_level)
                        except:
                            interval_val = 10
                        if interval_val <= 0:
                            interval_val = 10
                        hierarchies[col] = {
                            0: df[col].values,
                            1: generate_intervals(df[col].values, interval_val)
                        }
                    elif chosen_type == 'default':

                        if col == 'relationship':
                           
                            # try:
                            #     mask_level = 2 #int(chosen_level)
                            # except:
                            #     mask_level = 0  # Default to most specific level

                            # Initialize hierarchy dictionary
                            hierarchy_dict = {0: df[col].values}  # Level 0 is original data

                            # Load hierarchy mappings for this column (pre-load this outside the loop in reality)
                            # For this example, we'll assume you have a pre-loaded hierarchy dictionary:
                                
                            column_hierarchy = {
                                'Not-in-family': ['Not-in-family', 'Not Family', 'Relationship'], 
                                'Husband': ['Husband', 'Spouse', 'Relationship'], 
                                'Wife': ['Wife', 'Spouse', 'Relationship'], 
                                'Own-child': ['Own-child', 'Child', 'Relationship'], 
                                'Unmarried': ['Unmarried', 'Not Family', 'Relationship'], 
                                'Other-relative': ['Other-relative', 'Other Rela', 'Relationship']
                                }
                            max_hierarchy_level = len(column_hierarchy[list(column_hierarchy.keys())[0]]) #2 
                            # Create masked versions for each level up to mask_level
                            for lvl in range(1, max_hierarchy_level):
                            # for lvl in range(1, mask_level + 1):
                            
                                # Get the appropriate hierarchy level
                                hierarchy_level = min(lvl, max_hierarchy_level)  # Ensure we don't exceed available levels
                                
                                # Map values through hierarchy
                                hierarchy_dict[lvl] = df[col].map(
                                    lambda x: column_hierarchy.get(x, ["Unknown"] * (max_hierarchy_level + 1))[hierarchy_level]
                                ).values

                            hierarchies[col] = hierarchy_dict

                        if col == 'occupation':
                           
                            # try:
                            #     mask_level = 2 #int(chosen_level)
                            # except:
                            #     mask_level = 0  # Default to most specific level

                            # Initialize hierarchy dictionary
                            hierarchy_dict = {0: df[col].values}  # Level 0 is original data

                            # Load hierarchy mappings for this column (pre-load this outside the loop in reality)
                            # For this example, we'll assume you have a pre-loaded hierarchy dictionary:
                                
                            column_hierarchy = {
                                'Adm-cleric': ['Adm-cleric', 'Clerical/Admin', 'Office/Admin', 'Office/Management', 'Occupation'], 
                                'Exec-managerial': ['Exec-managerial', 'Executive/Managerial', 'Management/Sales', 'Office/Management', 'Occupation'], 
                                'Handlers-cleaners': ['Handlers-cleaners', 'Cleaning/Handling', 'Manual Labor', 'General Labor/Service', 'Occupation'], 
                                'Prof-specialty': ['Prof-specialty', 'Specialized Professional', 'Professional', 'Professional/Specialized', 'Occupation'], 
                                'Other-service': ['Other-service', 'General Services', 'Service', 'General Labor/Service', 'Occupation'], 
                                'Sales': ['Sales', 'Sales/Marketing', 'Management/Sales', 'Office/Management', 'Occupation'], 
                                'Craft-repair': ['Craft-repair', 'Craft/Repair', 'Skilled Trades', 'Skilled Trades/Technical', 'Occupation'], 
                                'Transport-moving': ['Transport-moving', 'Moving/Transport', 'Transportation', 'Skilled Trades/Technical', 'Occupation'], 
                                'Farming-fishing': ['Farming-fishing', 'Agriculture/Fishing', 'Manual Labor', 'General Labor/Service', 'Occupation'], 
                                'Machine-op-inspct': ['Machine-op-inspct', 'Machine Operation', 'Skilled Trades', 'Skilled Trades/Technical', 'Occupation'], 
                                'Tech-support': ['Tech-support', 'Technical Support', 'Office/Admin', 'Office/Management', 'Occupation'], 
                                '?': ['?', 'Unclassified', 'Unknown', 'Other/Unknown', 'Occupation'], 
                                'Protective-serv': ['Protective-serv', 'Protective Services', 'Protective', 'Professional/Specialized', 'Occupation'], 
                                'Armed-Forces': ['Armed-Forces', 'Armed Forces', 'Military', 'Other/Unknown', 'Occupation'], 
                                'Priv-house-serv': ['Priv-house-serv', 'Private Household', 'Service', 'General Labor/Service', 'Occupation']
                                }
                            max_hierarchy_level = len(column_hierarchy[list(column_hierarchy.keys())[0]]) #2 
                            # Create masked versions for each level up to mask_level
                            for lvl in range(1, max_hierarchy_level):
                            # for lvl in range(1, mask_level + 1):
                            
                                # Get the appropriate hierarchy level
                                hierarchy_level = min(lvl, max_hierarchy_level)  # Ensure we don't exceed available levels
                                
                                # Map values through hierarchy
                                hierarchy_dict[lvl] = df[col].map(
                                    lambda x: column_hierarchy.get(x, ["Unknown"] * (max_hierarchy_level + 1))[hierarchy_level]
                                ).values

                            hierarchies[col] = hierarchy_dict
                    
                        if col == 'marital-status':
                           
                            # try:
                            #     mask_level = 2 #int(chosen_level)
                            # except:
                            #     mask_level = 0  # Default to most specific level

                            # Initialize hierarchy dictionary
                            hierarchy_dict = {0: df[col].values}  # Level 0 is original data

                            # Load hierarchy mappings for this column (pre-load this outside the loop in reality)
                            # For this example, we'll assume you have a pre-loaded hierarchy dictionary:
                            column_hierarchy = {
                                'Never-married': ['Never-married', 'Unmarried', 'marital-status'],
                                'Married-civ-spouse': ['Married-civ-spouse', 'Married', 'marital-status'],
                                'Divorced': ['Divorced', 'Unmarried', 'marital-status'],
                                'Married-spouse-absent': ['Married-spouse-absent', 'Married', 'marital-status'],
                                'Separated': ['Separated', 'Unmarried', 'marital-status'],
                                'Married-AF-spouse': ['Married-AF-spouse', 'Married', 'marital-status'], 
                                'Widowed': ['Widowed', 'Unmarried', 'marital-status']
                            }
                            max_hierarchy_level = len(column_hierarchy[list(column_hierarchy.keys())[0]]) #2 
                            # Create masked versions for each level up to mask_level
                            for lvl in range(1, max_hierarchy_level):
                            # for lvl in range(1, mask_level + 1):
                            
                                # Get the appropriate hierarchy level
                                hierarchy_level = min(lvl, max_hierarchy_level)  # Ensure we don't exceed available levels
                                
                                # Map values through hierarchy
                                hierarchy_dict[lvl] = df[col].map(
                                    lambda x: column_hierarchy.get(x, ["Unknown"] * (max_hierarchy_level + 1))[hierarchy_level]
                                ).values

                            hierarchies[col] = hierarchy_dict

                        if col == 'Education': #'education'
                            # location_hierarchy = {
                            #     "Sault Ste. Marie, ON, P6A": ["Sault Ste. Marie, ON, P6A", "Ontario", "Canada"],
                            #     "Toronto, ON, M5A": ["Toronto, ON, M5A", "Ontario", "Canada"],
                            #     "Montreal, QC, H2A": ["Montreal, QC, H2A", "Quebec", "Canada"],
                            # }
                            # Assuming 'hierarchy_df' is a DataFrame loaded from your CSV with columns:
                            # Level0 | Level1 | Level2 (and possibly more levels)
                            # Example:
                            # Level0        | Level1    | Level2
                            # ------------------------------------
                            # Bachelors     | Bachelors | Study
                            # HS-grad       | School    | Study
                            # ...etc

                            # try:
                            #     mask_level = 2 #int(chosen_level)
                            # except:
                            #     mask_level = 0  # Default to most specific level

                            # Initialize hierarchy dictionary
                            hierarchy_dict = {0: df[col].values}  # Level 0 is original data

                            # Load hierarchy mappings for this column (pre-load this outside the loop in reality)
                            # For this example, we'll assume you have a pre-loaded hierarchy dictionary:
                            column_hierarchy = {
                                "HS-grad": ["HS-grad", "School", "Study"],
                                "11th": ["11th", "School", "Study"],
                                "Masters": ["Masters", "Masters", "Study"],
                                "9th": ["9th", "School", "Study"],
                                "Some-college": ["Some-college", "School", "Study"],
                                "Assoc-acdm": ["Assoc-acdm", "School", "Study"],
                                "Assoc-voc": ["Assoc-voc", "School", "Study"],
                                "7th-8th": ["7th-8th", "School", "Study"],
                                "Doctorate": ["Doctorate", "Doctorate", "Study"],
                                "Prof-school": ["Prof-school", "School", "Study"],
                                "5th-6th": ["5th-6th", "School", "Study"],
                                "10th": ["10th", "School", "Study"],
                                "1st-4th": ["1st-4th", "School", "Study"],
                                "Preschool": ["Preschool", "Preschool", "Study"],
                                "12th": ["12th", "School", "Study"],
                                "Bacherlors": ["Bacherlors", "Bacherlors", "Study"],
                            }
                            max_hierarchy_level = len(column_hierarchy[list(column_hierarchy.keys())[0]]) #2 
                            # Create masked versions for each level up to mask_level
                            for lvl in range(1, max_hierarchy_level):
                            # for lvl in range(1, mask_level + 1):

                                # Get the appropriate hierarchy level
                                hierarchy_level = min(lvl, max_hierarchy_level)  # Ensure we don't exceed available levels
                                
                                # Map values through hierarchy
                                hierarchy_dict[lvl] = df[col].map(
                                    lambda x: column_hierarchy.get(x, ["Unknown"] * (max_hierarchy_level + 1))[hierarchy_level]
                                ).values

                            hierarchies[col] = hierarchy_dict

        
        # Run the chosen anonymization
        try:
            if method == "l_diversity":
                if len(sens_att_list) != 1:
                    flash("For l-diversity, please select exactly one sensitive attribute.")
                    return redirect(url_for('select_columns', filename=filename))
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
        
        processed_filename = f'anonymized_{filename}'
        processed_filepath = os.path.join(PROCESSED_FOLDER, processed_filename)
        anonymized_df.to_csv(processed_filepath, index=False)
        
        preview_html = anonymized_df.head(10).to_html(classes="table table-striped", index=False)
        return render_template('preview.html', anonymized_table=preview_html, download_filename=processed_filename)
    
    return render_template(
        'select_columns.html',
        columns=columns,
        col_types=col_types,
        max_len_map=max_len_map,
        original_preview=original_preview
    )

@app.route('/download/<filename>')
def download_file(filename):
    filepath = os.path.join(PROCESSED_FOLDER, filename)
    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
