import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import math
import random
import re

st.set_page_config(page_title="Team Reshuffler", layout="wide")

st.title("Military Team Creator & Reshuffler")
st.markdown("Upload an Excel file with personnel data to create new teams or reshuffle existing teams based on specified criteria.")

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

def is_officer(value, comp_col):
    """Check if a value indicates an officer based on ending with 'O'"""
    if isinstance(value, str):
        return value.endswith('O')
    return False

def is_enlisted(value, comp_col):
    """Check if a value indicates enlisted based on ending with 'E'"""
    if isinstance(value, str):
        return value.endswith('E')
    return False

def is_recruit(value, comp_col):
    """Check if a value indicates a 18X based on ending with 'X'"""
    if isinstance(value, str):
        return value.endswith('X')
    return False

def calculate_metrics(original_df, new_teams_df, column_config, is_reshuffle=True):
    metrics = {}
    
    id_col = column_config["id_column"]
    comp_col = column_config.get("comp_column")
    
    # Team size statistics
    team_sizes = new_teams_df.groupby('New Team').size()
    metrics['team_size_min'] = team_sizes.min()
    metrics['team_size_max'] = team_sizes.max()
    metrics['team_size_avg'] = team_sizes.mean()
    
    # If reshuffling, calculate percentage of people who changed teams
    if is_reshuffle and column_config.get("original_team_column") and column_config["original_team_column"] in original_df.columns:
        original_team_col = column_config["original_team_column"]
        total_people = len(original_df)
        changed_teams = sum(original_df[original_team_col] != new_teams_df['New Team'])
        metrics['changed_team_percentage'] = (changed_teams / total_people) * 100
    
    # Calculate ratio statistics for each selected column
    ratio_stats = {}
    
    # If we have a composition column, calculate officer/enlisted/recruit ratios
    if comp_col:
        for team in new_teams_df['New Team'].unique():
            team_data = new_teams_df[new_teams_df['New Team'] == team]
            
            if team not in ratio_stats:
                ratio_stats[team] = {}
            
            # Officer/Enlisted/Recruit ratio
            officers = sum(team_data[comp_col].apply(lambda x: is_officer(x, comp_col)))
            enlisted = sum(team_data[comp_col].apply(lambda x: is_enlisted(x, comp_col)))
            recruits = sum(team_data[comp_col].apply(lambda x: is_recruit(x, comp_col)))
            
            team_size = len(team_data)
            ratio_stats[team]["Officer %"] = officers / team_size * 100
            ratio_stats[team]["Enlisted %"] = enlisted / team_size * 100
            ratio_stats[team]["18X %"] = recruits / team_size * 100
    
    # Calculate ratios for columns with filled/empty values
    for column in column_config["priority_columns"]:
        if column in new_teams_df.columns and new_teams_df[column].notna().any() and new_teams_df[column].isna().any():
            column_name = f"{column} Filled %"
            
            for team in new_teams_df['New Team'].unique():
                team_data = new_teams_df[new_teams_df['New Team'] == team]
                
                if team not in ratio_stats:
                    ratio_stats[team] = {}
                
                # Calculate percentage of filled values
                filled_count = team_data[column].notna().sum()
                team_size = len(team_data)
                ratio_stats[team][column_name] = filled_count / team_size * 100
    
    # Convert ratio_stats to lists for the metrics dict
    for stat_name in set().union(*[stats.keys() for stats in ratio_stats.values()]) if ratio_stats else []:
        stat_values = [stats.get(stat_name, 0) for _, stats in ratio_stats.items()]
        metrics[f"{stat_name}_values"] = stat_values
        metrics[f"{stat_name}_std"] = np.std(stat_values)
    
    # Calculate standard deviations for numeric columns
    for column in column_config["priority_columns"]:
        if column in new_teams_df.columns and pd.api.types.is_numeric_dtype(new_teams_df[column]):
            std_by_team = new_teams_df.groupby('New Team')[column].std().mean()
            metrics[f"{column}_std"] = std_by_team if not pd.isna(std_by_team) else 0
    
    return metrics, ratio_stats

def create_or_reshuffle_teams(df, column_config, is_reshuffle=True, min_team_size=13, max_team_size=18):
    """
    Create new teams or reshuffle existing teams based on provided configuration
    """
    # Copy the original dataframe
    original_df = df.copy()
    
    id_col = column_config["id_column"]
    comp_col = column_config.get("comp_column")
    original_team_col = column_config.get("original_team_column") if is_reshuffle else None
    priority_columns = column_config["priority_columns"]
    
    # Handle missing values in numeric columns
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
    
    # Determine the number of teams needed
    total_people = len(df)
    
    # Calculate minimum number of teams needed to not exceed max_team_size
    min_num_teams = math.ceil(total_people / max_team_size)
    
    # Calculate maximum number of teams that would still maintain min_team_size
    max_num_teams = math.floor(total_people / min_team_size)
    
    # If min > max, we need to prioritize min_team_size
    if min_num_teams > max_num_teams:
        num_teams = min_num_teams
    else:
        # Otherwise use min_num_teams to maximize team sizes
        num_teams = min_num_teams
    
    st.write(f"Creating {num_teams} teams for {total_people} people.")
    
    # If we have a composition column, ensure at least one officer per team
    if comp_col:
        officers = df[df[comp_col].apply(lambda x: is_officer(x, comp_col))].copy() if comp_col else pd.DataFrame()
        
        if not officers.empty and len(officers) < num_teams:
            st.warning(f"Not enough officers ({len(officers)}) to have at least one per team ({num_teams} teams). Continuing with available officers.")
            num_teams = max(1, len(officers))
    else:
        officers = pd.DataFrame()
    
    # Initialize new teams
    new_teams = {i+1: [] for i in range(num_teams)}
    
    # First, distribute officers to ensure at least one per team (if comp_col is provided)
    if not officers.empty:
        # Create sorting keys based on priority columns
        sort_columns = []
        if is_reshuffle and original_team_col:
            sort_columns.append(original_team_col)
        
        for col in priority_columns:
            if col in df.columns and col != original_team_col:
                # For numeric columns, sort by value; for others, for now just use the column
                sort_columns.append(col)
        
        # Sort officers by priority columns
        if sort_columns:
            officers = officers.sort_values(sort_columns)
        
        for i, officer_idx in enumerate(officers.index):
            team_idx = i % num_teams + 1
            new_teams[team_idx].append(officer_idx)
    
    # Get the rest of the personnel
    if not officers.empty:
        rest = df.loc[~df.index.isin(officers.index)].copy()
    else:
        rest = df.copy()
    
    # Sort rest by priority columns
    if sort_columns:
        rest = rest.sort_values(sort_columns)
    
    # Create a scoring function for team assignment
    def score_assignment(person_idx, team_idx):
        person = df.loc[person_idx]
        team_members = [df.loc[idx] for idx in new_teams[team_idx]]
        
        # Start with a base score
        score = 0
        
        # Penalize team size imbalance with higher penalty for exceeding max
        team_size = len(team_members) + 1  # +1 to include the person being added
        
        # If the team is below max, encourage adding more
        if team_size <= max_team_size:
            # Less penalty for teams closer to max size
            size_diff = max_team_size - team_size
            score -= size_diff * 20  # Small penalty - we want to fill teams to max if possible
        else:
            # Higher penalty for exceeding max
            size_diff = team_size - max_team_size
            score -= size_diff * 200  # Stronger penalty for exceeding max
        
        # Add penalties based on priority columns
        penalty_weight = 1000  # Start with high weight and decrease for lower priorities
        
        # If reshuffling, heavily penalize same original team
        if is_reshuffle and original_team_col and original_team_col in df.columns:
            same_team_count = sum(member[original_team_col] == person[original_team_col] for member in team_members)
            score -= same_team_count * penalty_weight
            penalty_weight *= 0.8  # Reduce weight for next priority
        
        # Add penalties for each priority column
        for col in priority_columns:
            if col in df.columns and col != original_team_col:
                # Handle different column types differently
                if pd.api.types.is_numeric_dtype(df[col]):
                    # For numeric columns, aim for similar averages across teams
                    col_values = [member[col] for member in team_members]
                    if col_values:
                        avg_val = sum(col_values) / len(col_values)
                        val_diff = abs(person[col] - avg_val)
                        score -= val_diff * penalty_weight * 0.01  # Scale down for numeric values
                
                elif df[col].notna().any() and df[col].isna().any():
                    # For columns with both filled and empty values, balance them
                    filled_count = sum(pd.notna(member[col]) for member in team_members)
                    empty_count = len(team_members) - filled_count
                    
                    if pd.notna(person[col]) and filled_count > empty_count:
                        # Penalty for adding someone with filled value to a team with more filled values
                        score -= penalty_weight
                    elif pd.isna(person[col]) and empty_count > filled_count:
                        # Penalty for adding someone with empty value to a team with more empty values
                        score -= penalty_weight
                
                else:
                    # For categorical columns, aim for even distribution
                    if not pd.isna(person[col]):
                        col_count = sum(member[col] == person[col] for member in team_members if not pd.isna(member[col]))
                        score -= col_count * penalty_weight * 0.1
                
                penalty_weight *= 0.8  # Reduce weight for next priority
        
        # If we have a composition column, balance officer/enlisted/recruit
        if comp_col and comp_col in df.columns:
            comp_types = [member[comp_col] for member in team_members if not pd.isna(member[comp_col])]
            if comp_types:
                # Count officers, enlisted, recruits
                officer_count = sum(is_officer(comp, comp_col) for comp in comp_types)
                enlisted_count = sum(is_enlisted(comp, comp_col) for comp in comp_types) 
                recruit_count = sum(is_recruit(comp, comp_col) for comp in comp_types)
                
                # Penalize imbalances
                person_comp = person[comp_col]
                if is_officer(person_comp, comp_col) and officer_count > (len(team_members) / 5):  # Roughly 20% officers
                    score -= penalty_weight * 0.5
                elif is_enlisted(person_comp, comp_col) and enlisted_count > (len(team_members) * 0.6):  # Roughly 60% enlisted
                    score -= penalty_weight * 0.5
                elif is_recruit(person_comp, comp_col) and recruit_count > (len(team_members) * 0.3):  # Roughly 30% recruits
                    score -= penalty_weight * 0.5
        
        return score
    
    # Distribute the rest based on the scoring function
    for person_idx in rest.index:
        # Calculate scores for each team
        scores = {team_idx: score_assignment(person_idx, team_idx) for team_idx in new_teams.keys()}
        
        # Assign to the team with the highest score
        best_team = max(scores.items(), key=lambda x: x[1])[0]
        new_teams[best_team].append(person_idx)
    
    # Create a new dataframe with the new team assignments
    new_df = df.copy()
    new_df['New Team'] = 0
    
    for team_idx, members in new_teams.items():
        for member_idx in members:
            new_df.loc[member_idx, 'New Team'] = team_idx
    
    return new_df

def display_metrics(metrics, ratio_stats, column_config, is_reshuffle=True):
    st.subheader("Team Formation Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if is_reshuffle and 'changed_team_percentage' in metrics:
            st.metric("Changed Teams", f"{metrics['changed_team_percentage']:.1f}%")
        st.metric("Min Team Size", metrics['team_size_min'])
        st.metric("Max Team Size", metrics['team_size_max'])
        st.metric("Avg Team Size", f"{metrics['team_size_avg']:.1f}")
    
    with col2:
        # Display standard deviations for key ratios
        for key, value in metrics.items():
            if key.endswith("_std") and "values" not in key:
                nice_name = key.replace("_std", "").replace("_", " ").title()
                if "%" in nice_name:
                    st.metric(f"{nice_name} Std Dev", f"{value:.2f}%")
                else:
                    st.metric(f"{nice_name} Std Dev", f"{value:.2f}")
    
    # Team composition table
    st.subheader("Team Composition")
    
    # Create a DataFrame from ratio_stats
    if ratio_stats:
        teams = sorted(ratio_stats.keys())
        stats = {}
        
        # Add team numbers
        stats['Team'] = [f"Team {team}" for team in teams]
        
        # Add all available stats
        all_stat_names = set()
        for team_stats in ratio_stats.values():
            all_stat_names.update(team_stats.keys())
        
        for stat_name in sorted(all_stat_names):
            stats[stat_name] = [f"{ratio_stats[team].get(stat_name, 0):.1f}%" for team in teams]
        
        composition_data = pd.DataFrame(stats)
        st.dataframe(composition_data)
    else:
        st.write("No team composition statistics available.")

def to_excel(df_dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    processed_data = output.getvalue()
    return processed_data

def detect_column_types(df):
    """
    Detects columns types to suggest configuration
    """
    column_types = {}
    
    # Detect potential ID columns (unique values, often numeric or short strings)
    potential_id_columns = []
    for col in df.columns:
        if df[col].nunique() == len(df):  # Unique values
            potential_id_columns.append(col)
    
    # Detect potential composition columns (looking for values ending in O, E, X)
    potential_comp_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':  # String columns
            # Check if any values end with O, E, or X
            has_o = any(str(val).endswith('O') for val in df[col].dropna())
            has_e = any(str(val).endswith('E') for val in df[col].dropna())
            has_x = any(str(val).endswith('X') for val in df[col].dropna())
            
            if has_o and (has_e or has_x):
                potential_comp_columns.append(col)
    
    # Detect potential original team columns (few unique values, often numeric)
    potential_team_columns = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and 1 < df[col].nunique() < len(df) / 10:
            potential_team_columns.append(col)
        elif df[col].dtype == 'object' and 1 < df[col].nunique() < len(df) / 10:
            # Check if values look like team identifiers
            if any(str(val).lower().startswith('team') for val in df[col].dropna()):
                potential_team_columns.append(col)
    
    # Detect columns with both filled and empty values
    partial_filled_columns = []
    for col in df.columns:
        if df[col].notna().sum() > 0 and df[col].isna().sum() > 0:
            partial_filled_columns.append(col)
    
    column_types['potential_id_columns'] = potential_id_columns
    column_types['potential_comp_columns'] = potential_comp_columns
    column_types['potential_team_columns'] = potential_team_columns
    column_types['partial_filled_columns'] = partial_filled_columns
    
    return column_types

def get_column_config(df):
    """
    Let user configure which columns to use for different purposes
    """
    column_types = detect_column_types(df)
    
    st.subheader("Configure Columns")
    
    # Select ID column
    id_options = column_types['potential_id_columns'] + [col for col in df.columns if col not in column_types['potential_id_columns']]
    id_col = st.selectbox("Select ID column (unique identifier for each person):", 
                         id_options, 
                         index=0 if column_types['potential_id_columns'] else 0)
    
    # Select composition column (optional)
    comp_options = ["None"] + column_types['potential_comp_columns'] + [col for col in df.columns if col not in column_types['potential_comp_columns']]
    comp_col = st.selectbox("Select composition column (personnel types like ADO, NGO, ADE, etc.):", 
                           comp_options,
                           index=1 if column_types['potential_comp_columns'] else 0)
    comp_col = None if comp_col == "None" else comp_col
    
    # For reshuffling, select original team column
    if 'operation' in st.session_state and st.session_state['operation'] == "Reshuffle existing teams":
        team_options = ["None"] + column_types['potential_team_columns'] + [col for col in df.columns if col not in column_types['potential_team_columns']]
        original_team_col = st.selectbox("Select original team column:", 
                                        team_options,
                                        index=1 if column_types['potential_team_columns'] else 0)
        original_team_col = None if original_team_col == "None" else original_team_col
    else:
        original_team_col = None
    
    # Select priority columns
    st.write("Select and order columns for team formation priorities:")
    
    # Highlight columns with partial filled values
    if column_types['partial_filled_columns']:
        st.info(f"Columns with both filled and empty values (good for balancing): {', '.join(column_types['partial_filled_columns'])}")
    
    # Multiselect for column priorities
    available_columns = [col for col in df.columns if col != id_col]
    default_selections = [col for col in available_columns if col in column_types['partial_filled_columns'] or col in column_types['potential_team_columns']]
    
    priority_columns = st.multiselect(
        "Select columns to use for team formation (in priority order):",
        available_columns,
        default=default_selections
    )
    
    column_config = {
        "id_column": id_col,
        "comp_column": comp_col,
        "original_team_column": original_team_col,
        "priority_columns": priority_columns
    }
    
    return column_config

def display_team_details(new_teams_df, column_config, is_reshuffle=True):
    """Display detailed information about each team"""
    id_col = column_config["id_column"]
    comp_col = column_config.get("comp_column")
    original_team_col = column_config.get("original_team_column") if is_reshuffle else None
    priority_columns = column_config["priority_columns"]
    
    st.subheader("Team Details")
    for team in sorted(new_teams_df['New Team'].unique()):
        with st.expander(f"Team {team}"):
            team_data = new_teams_df[new_teams_df['New Team'] == team]
            
            # Basic team stats
            st.write(f"Team Size: {len(team_data)}")
            
            # If we have a composition column, show officer/enlisted/recruit breakdowns
            if comp_col:
                officers = team_data[team_data[comp_col].apply(lambda x: is_officer(x, comp_col))]
                enlisted = team_data[team_data[comp_col].apply(lambda x: is_enlisted(x, comp_col))]
                recruits = team_data[team_data[comp_col].apply(lambda x: is_recruit(x, comp_col))]
                
                st.write(f"Officers: {len(officers)} ({len(officers)/len(team_data)*100:.1f}%)")
                st.write(f"Enlisted: {len(enlisted)} ({len(enlisted)/len(team_data)*100:.1f}%)")
                st.write(f"Recruits: {len(recruits)} ({len(recruits)/len(team_data)*100:.1f}%)")
            
            # Show stats for each priority column
            for col in priority_columns:
                if col in team_data.columns:
                    # For columns with both filled and empty values, show the balance
                    if team_data[col].notna().any() and team_data[col].isna().any():
                        filled = team_data[col].notna().sum()
                        st.write(f"{col} - Filled values: {filled} ({filled/len(team_data)*100:.1f}%)")
                    
                    # For numeric columns, show average
                    if pd.api.types.is_numeric_dtype(team_data[col]):
                        st.write(f"Average {col}: {team_data[col].mean():.2f}")
            
            # For reshuffling, show original team breakdown
            if is_reshuffle and original_team_col and original_team_col in team_data.columns:
                st.write("Members from original teams:")
                original_teams = team_data[original_team_col].value_counts()
                for orig_team, count in original_teams.items():
                    st.write(f"- Team {orig_team}: {count} members")
            
            # Show the team data
            st.dataframe(team_data)

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        
        st.write("Data preview:")
        st.dataframe(df.head())
        
        # Choose operation - create new teams or reshuffle
        operation = st.radio("Choose operation:", 
                         ["Create new teams from scratch", "Reshuffle existing teams"])
        st.session_state['operation'] = operation
        
        # Let user configure columns
        column_config = get_column_config(df)
        
        # For reshuffling, ensure we have an original team column
        if operation == "Reshuffle existing teams":
            if not column_config["original_team_column"]:
                st.warning("No original team column selected. Please either select a column or create random teams:")
                
                create_random = st.checkbox("Create random original teams")
                if create_random:
                    num_teams = st.number_input("Number of original teams:", min_value=1, value=max(1, len(df) // 18))
                    df['Random Original Team'] = [random.randint(1, int(num_teams)) for _ in range(len(df))]
                    column_config["original_team_column"] = 'Random Original Team'
                    st.success("Random original teams created!")
        
        # Team size configuration
        st.subheader("Team Size Settings")
        col1, col2 = st.columns(2)
        with col1:
            min_team_size = st.number_input("Minimum team size:", min_value=1, value=13)
        with col2:
            max_team_size = st.number_input("Maximum team size (ideal size):", min_value=min_team_size, value=18)
        
        # Proceed with team formation
        if st.button("Process Teams"):
            # Ensure we have required configurations
            if operation == "Reshuffle existing teams" and not column_config["original_team_column"]:
                st.error("Please select an original team column or create random teams to proceed with reshuffling.")
            else:
                with st.spinner("Processing teams..."):
                    is_reshuffle = operation == "Reshuffle existing teams"
                    new_teams_df = create_or_reshuffle_teams(df, column_config, is_reshuffle=is_reshuffle, 
                                                            min_team_size=min_team_size, max_team_size=max_team_size)
                    metrics, ratio_stats = calculate_metrics(df, new_teams_df, column_config, is_reshuffle=is_reshuffle)
                    
                    st.success(f"Teams {'reshuffled' if is_reshuffle else 'created'} successfully!")
                    
                    display_metrics(metrics, ratio_stats, column_config, is_reshuffle=is_reshuffle)
                    
                    st.subheader(f"{'New' if is_reshuffle else ''} Team Assignments")
                    st.dataframe(new_teams_df.sort_values(['New Team']))
                    
                    # Prepare Excel download
                    excel_data = to_excel({
                        'Original Data': df,
                        'New Teams': new_teams_df
                    })
                    
                    st.download_button(
                        label=f"Download Excel with {'new' if is_reshuffle else ''} teams",
                        data=excel_data,
                        file_name=f"{'reshuffled' if is_reshuffle else 'new'}_teams.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                    
                    # Show detailed team view
                    display_team_details(new_teams_df, column_config, is_reshuffle=is_reshuffle)
    
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.exception(e)
else:
    st.info("Please upload an Excel file with personnel data.")
    
    # Show example format
    st.subheader("Example Data Format:")
    st.markdown("""
    Upload an Excel file with your team data. The app will help you configure:
    
    1. Which column contains unique identifiers
    2. Which column (if any) indicates personnel types (officers/enlisted/18X)
    3. Which columns to use for team formation priorities
    4. For reshuffling, which column indicates original teams
    5. Minimum and maximum team sizes
    
    The app will automatically detect:
    - Potential ID columns
    - Columns that might indicate personnel types
    - Columns that could represent teams
    - Columns with both filled and empty values (useful for balancing experienced/inexperienced personnel)
    
    You can then customize the column selection and priorities to match your specific needs.
    """)

# Add footer with instructions for GitHub deployment
st.markdown("---")
st.markdown("""
### How to deploy this app:
1. Create a GitHub repository
2. Add this file as `app.py`
3. Create a `requirements.txt` file with the dependencies listed below
4. Deploy to Streamlit Cloud by connecting to your GitHub repository
""")