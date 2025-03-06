import streamlit as st
import pandas as pd
import numpy as np
import io
import xlsxwriter

def main():
    st.set_page_config(page_title="Tips Calculator", layout="wide")
    
    st.title('Employee Tips Calculator')
    st.write('Upload your employee hours data, select roles for tips, and calculate distributions.')
    
    # Step 1: Upload CSV
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Attempt to read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # List columns that might contain hours
            hour_columns = []
            regular_hour_columns = []
            ot_hour_columns = []
            
            for col in df.columns:
                if 'hour' in col.lower() or ' hr' in col.lower() or 'hrs' in col.lower():
                    hour_columns.append(col)
                    if 'ot' in col.lower() or 'overtime' in col.lower():
                        ot_hour_columns.append(col)
                    else:
                        regular_hour_columns.append(col)
            
            if not hour_columns:
                st.error("No hour columns found in the CSV. Please check your file.")
                return
            
            # Display a preview of the data
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Ensure we have the required columns
            required_cols = ['Employee ID', 'First', 'Last', 'Role']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                return
            
            # Step 2: Select roles for tip distribution
            st.subheader('Configure Tip Distribution')
            
            unique_roles = sorted(df['Role'].unique().tolist())
            selected_roles = st.multiselect(
                'Select roles to include in the tip distribution:',
                options=unique_roles,
                default=[]
            )
            
            # Step 3: Enter tip amount
            tip_amount = st.number_input('Enter the total tip amount ($):', min_value=0.0, value=1000.0, step=100.0)
            
            # Step 4: Generate results
            if st.button('Generate Results', type="primary"):
                if selected_roles and tip_amount > 0:
                    try:
                        # Process the full dataframe for the role breakdown (all roles)
                        full_df = df.copy()
                        
                        # Convert hour columns to numeric in the full dataframe
                        for col in hour_columns:
                            full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
                            full_df[col] = full_df[col].fillna(0)
                        
                        # Create new columns for categorized hours in the full dataframe
                        full_df['Regular Hours'] = 0
                        full_df['OT Hours'] = 0
                        full_df['Total Hours'] = 0
                        
                        # Sum regular and OT hours in the full dataframe
                        for col in regular_hour_columns:
                            full_df['Regular Hours'] += full_df[col]
                        
                        for col in ot_hour_columns:
                            full_df['OT Hours'] += full_df[col]
                        
                        full_df['Total Hours'] = full_df['Regular Hours'] + full_df['OT Hours']
                        
                        # Now filter by selected roles for tip calculation
                        role_filtered = df[df['Role'].isin(selected_roles)].copy()
                        
                        if role_filtered.empty:
                            st.error("No data found for the selected roles")
                            return
                        
                        # Convert hour columns to numeric and sum them for tip calculation
                        for col in hour_columns:
                            role_filtered[col] = pd.to_numeric(role_filtered[col], errors='coerce')
                            role_filtered[col] = role_filtered[col].fillna(0)
                        
                        # Create new columns for categorized hours for tip calculation
                        role_filtered['Regular Hours'] = 0
                        role_filtered['OT Hours'] = 0
                        role_filtered['Total Hours'] = 0
                        
                        # Sum regular hours
                        for col in regular_hour_columns:
                            role_filtered['Regular Hours'] += role_filtered[col]
                        
                        # Sum overtime hours
                        for col in ot_hour_columns:
                            role_filtered['OT Hours'] += role_filtered[col]
                        
                        # Calculate total hours
                        role_filtered['Total Hours'] = role_filtered['Regular Hours'] + role_filtered['OT Hours']
                        
                        # Ensure ID and name columns are strings to avoid grouping issues
                        for col in ['Employee ID', 'First', 'Last']:
                            role_filtered[col] = role_filtered[col].astype(str)
                            full_df[col] = full_df[col].astype(str)
                        
                        # Check for any NaN values in grouping columns
                        for col in ['Employee ID', 'First', 'Last']:
                            if role_filtered[col].isna().any():
                                role_filtered[col] = role_filtered[col].fillna('Unknown')
                            if full_df[col].isna().any():
                                full_df[col] = full_df[col].fillna('Unknown')
                        
                        # Group by employee for tips calculation
                        result = role_filtered.groupby(['Employee ID', 'First', 'Last'], as_index=False)['Total Hours'].sum()
                        
                        if result.empty:
                            st.error("Grouping resulted in empty dataframe - check debug info above")
                            st.write("First 10 rows before grouping:")
                            st.dataframe(role_filtered[['Employee ID', 'First', 'Last', 'Total Hours']].head(10))
                            return
                            
                        # Calculate the total hours
                        total_hours = result['Total Hours'].sum()
                        st.write(f"Total hours: {total_hours}")
                        
                        if total_hours <= 0:
                            st.error("Total hours is zero or negative")
                            return
                        
                        # Calculate tips
                        result['Tips'] = (result['Total Hours'] / total_hours) * tip_amount
                        
                        # Create the final formatted output
                        output = pd.DataFrame()
                        output['Name'] = result['Last'] + ', ' + result['First']
                        output['Hours'] = result['Total Hours'].round(2)
                        output['Earned'] = result['Tips'].round(2)
                        
                        # Sort by hours
                        output = output.sort_values('Hours', ascending=False)
                        
                        # Create formatted output
                        formatted_output = output.copy()
                        formatted_output['Earned'] = formatted_output['Earned'].apply(lambda x: f"${x:,.2f}")
                        
                        # Fill to at least 20 rows
                        rows_needed = max(0, 20 - len(formatted_output))
                        if rows_needed > 0:
                            empty_rows = pd.DataFrame({
                                'Name': ['0'] * rows_needed,
                                'Hours': [0] * rows_needed,
                                'Earned': ['$-'] * rows_needed
                            })
                            formatted_output = pd.concat([formatted_output, empty_rows], ignore_index=True)
                        
                        # Add total row
                        total_row = pd.DataFrame({
                            'Name': ['Total Hours'],
                            'Hours': [total_hours],
                            'Earned': [f"${tip_amount:,.2f}"]
                        })
                        formatted_output = pd.concat([formatted_output, total_row], ignore_index=True)
                        
                        # Display final formatted output
                        st.subheader("Tips Calculator")
                        st.table(formatted_output)
                        
                        # Save the tips calculation results for later export and analysis
                        tips_results = output.copy()
                        
                        # Now create the role breakdown pivot table
                        st.subheader("Hours Breakdown by Employee and Role")
                        
                        role_breakdown = full_df.groupby(['Last', 'First', 'Role'], as_index=False).agg({
                            'Regular Hours': 'sum',
                            'OT Hours': 'sum',
                            'Total Hours': 'sum'
                        })
                        
                        role_breakdown['Name'] = role_breakdown['Last'] + ', ' + role_breakdown['First']
                        for col in ['Regular Hours', 'OT Hours', 'Total Hours']:
                            role_breakdown[col] = role_breakdown[col].round(2)
                        
                        role_breakdown = role_breakdown.sort_values(['Last', 'First'])
                        
                        st.markdown("### Employee Hours by Role")
                        unique_employees = role_breakdown[['Last', 'First', 'Name']].drop_duplicates().sort_values('Name')
                        
                        pivot_rows = []
                        excel_rows = []
                        
                        grand_total_reg = 0
                        grand_total_ot = 0
                        grand_total_hours = 0
                        
                        for idx, employee in unique_employees.iterrows():
                            employee_data = role_breakdown[role_breakdown['Name'] == employee['Name']].sort_values('Role')
                            
                            emp_reg_hours = employee_data['Regular Hours'].sum()
                            emp_ot_hours = employee_data['OT Hours'].sum()
                            emp_total_hours = employee_data['Total Hours'].sum()
                            
                            grand_total_reg += emp_reg_hours
                            grand_total_ot += emp_ot_hours
                            grand_total_hours += emp_total_hours
                            
                            emp_name = employee['Name']
                            
                            st.markdown(f"**{emp_name}** - Total Hours: **{emp_total_hours:.2f}**")
                            
                            role_data = []
                            for _, role_row in employee_data.iterrows():
                                role_data.append({
                                    "Role": role_row['Role'],
                                    "Regular Hours": f"{role_row['Regular Hours']:.2f}",
                                    "OT Hours": f"{role_row['OT Hours']:.2f}",
                                    "Total Hours": f"{role_row['Total Hours']:.2f}"
                                })
                            
                            role_df = pd.DataFrame(role_data)
                            st.dataframe(role_df, hide_index=True, use_container_width=True)
                            st.markdown("---")
                            
                            pivot_rows.append({
                                'Row Labels': f"**{emp_name}**",
                                'Sum of Regular hours': f"**{emp_reg_hours:.2f}**",
                                'Sum of OT hours': f"**{emp_ot_hours:.2f}**",
                                'Sum of Total Hours': f"**{emp_total_hours:.2f}**"
                            })
                            
                            excel_rows.append({
                                'Row Labels': emp_name,
                                'Sum of Regular hours': emp_reg_hours,
                                'Sum of OT hours': emp_ot_hours,
                                'Sum of Total Hours': emp_total_hours,
                                'Is_Header': True
                            })
                            
                            for _, role_row in employee_data.iterrows():
                                pivot_rows.append({
                                    'Row Labels': role_row['Role'],
                                    'Sum of Regular hours': f"{role_row['Regular Hours']:.2f}",
                                    'Sum of OT hours': f"{role_row['OT Hours']:.2f}",
                                    'Sum of Total Hours': f"{role_row['Total Hours']:.2f}"
                                })
                                
                                excel_rows.append({
                                    'Row Labels': role_row['Role'],
                                    'Sum of Regular hours': role_row['Regular Hours'],
                                    'Sum of OT hours': role_row['OT Hours'],
                                    'Sum of Total Hours': role_row['Total Hours'],
                                    'Is_Header': False
                                })
                        
                        st.markdown(f"### Grand Total")
                        st.markdown(f"**Regular Hours**: {grand_total_reg:.2f}")
                        st.markdown(f"**OT Hours**: {grand_total_ot:.2f}")
                        st.markdown(f"**Total Hours**: {grand_total_hours:.2f}")
                        
                        pivot_df = pd.DataFrame(pivot_rows)
                        grand_total = {
                            'Row Labels': '**Grand Total**',
                            'Sum of Regular hours': f"**{grand_total_reg:.2f}**",
                            'Sum of OT hours': f"**{grand_total_ot:.2f}**",
                            'Sum of Total Hours': f"**{grand_total_hours:.2f}**"
                        }
                        pivot_df = pd.concat([pivot_df, pd.DataFrame([grand_total])], ignore_index=True)
                        
                        excel_df = pd.DataFrame(excel_rows)
                        excel_grand_total = {
                            'Row Labels': 'Grand Total',
                            'Sum of Regular hours': grand_total_reg,
                            'Sum of OT hours': grand_total_ot,
                            'Sum of Total Hours': grand_total_hours,
                            'Is_Header': True
                        }
                        excel_df = pd.concat([excel_df, pd.DataFrame([excel_grand_total])], ignore_index=True)
                        
                        role_breakdown_export = role_breakdown.copy()
                        buffer = io.BytesIO()
                        
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            tips_results.to_excel(writer, sheet_name='Tips Distribution', index=False)
                            role_breakdown_export.to_excel(writer, sheet_name='Hours Breakdown', index=False)
                            
                            display_df = excel_df.drop(columns=['Is_Header'])
                            display_df.to_excel(writer, sheet_name='Hours by Role', index=False)
                            
                            workbook = writer.book
                            bold_format = workbook.add_format({
                                'bold': True, 
                                'bg_color': '#E0E0E0'
                            })
                            header_format = workbook.add_format({
                                'bold': True,
                                'bg_color': '#4B88CB',
                                'font_color': 'white',
                                'border': 1
                            })
                            
                            worksheet = writer.sheets['Hours by Role']
                            for col_num, value in enumerate(display_df.columns.values):
                                worksheet.write(0, col_num, value, header_format)
                            
                            for row_num in range(len(excel_df)):
                                if excel_df.iloc[row_num]['Is_Header']:
                                    for col_num in range(len(display_df.columns)):
                                        worksheet.write(
                                            row_num + 1, col_num, 
                                            display_df.iloc[row_num][display_df.columns[col_num]], 
                                            bold_format
                                        )
                            
                            worksheet.set_column('A:A', 25)
                            worksheet.set_column('B:D', 15)
                            
                            num_format = workbook.add_format({'num_format': '0.00'})
                            worksheet.set_column('B:D', 15, num_format)
                            
                            strip_format = workbook.add_format({'bg_color': '#F5F5F5'})
                            for row in range(1, len(excel_df) + 1, 2):
                                if not excel_df.iloc[row-1]['Is_Header']:
                                    worksheet.set_row(row, None, strip_format)
                        
                        buffer.seek(0)
                        st.download_button(
                            label="Download Complete Results (Excel)",
                            data=buffer,
                            file_name="employee_hours_and_tips.xlsx",
                            mime="application/vnd.ms-excel"
                        )
                        
                        # Save the calculation results in session state for analysis
                        st.session_state["analysis_result"] = result
                        st.session_state["analysis_tip_amount"] = tip_amount
                        
                    except Exception as e:
                        st.error(f"Error during grouping or calculation: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                else:
                    if not selected_roles:
                        st.error("Please select at least one role")
                    if tip_amount <= 0:
                        st.error("Please enter a tip amount greater than zero")
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        
        # Analysis Button: Only show if results are stored in session state
        if "analysis_result" in st.session_state:
            if st.button("Show Analysis"):
                analysis_result = st.session_state["analysis_result"]
                analysis_tip_amount = st.session_state["analysis_tip_amount"]
                
                st.subheader("Tip Analysis")
                st.write(f"**Total Employees:** {len(analysis_result)}")
                st.write(f"**Average Hours per Employee:** {analysis_result['Total Hours'].mean():.2f}")
                st.write(f"**Median Hours:** {analysis_result['Total Hours'].median():.2f}")
                st.write(f"**Highest Hours:** {analysis_result['Total Hours'].max():.2f}")
                st.write(f"**Lowest Hours:** {analysis_result['Total Hours'].min():.2f}")
                st.write(f"**Total Tips Distributed:** ${analysis_tip_amount:,.2f}")
                avg_tip_per_hour = analysis_tip_amount / analysis_result['Total Hours'].sum()
                st.write(f"**Average Tip per Hour:** ${avg_tip_per_hour:,.2f}")
                
                st.markdown("### Top 5 Earners")
                st.dataframe(analysis_result.sort_values("Tips", ascending=False).head(5))
                
                # Create a bar chart for tip distribution by employee
                analysis_result["Name"] = analysis_result["Last"] + ", " + analysis_result["First"]
                tip_chart_data = analysis_result.set_index("Name")["Tips"]
                st.markdown("### Tips Distribution Chart")
                st.bar_chart(tip_chart_data)

if __name__ == "__main__":
    main()
