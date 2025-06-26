import pandas as pd
import base64

def generate_report_link(movement_data, file_name="movement_analysis_report.csv"):
    df = pd.DataFrame(movement_data)
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">Download Movement Analysis Report</a>'
    return href 