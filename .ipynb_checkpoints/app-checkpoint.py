import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import math
import random

st.set_page_config(page_title="Team Reshuffler", layout="wide")

st.title("Military Team Reshuffler")
st.markdown("Upload an Excel file with team data to reshuffle teams based on specified criteria.")

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

def is_officer(comp):
    return comp.endswith('O')

def is_enlisted(comp):
    return comp.endswith('E')

def is_recruit(comp):
    return comp.endswith('X')

def calculate_metrics(original_df, new_teams_df):
    metrics = {}
    
    # Percentage of people who changed teams
    total_people = len(original_df)
    changed_teams = sum(original_df['Original Team'] != new_teams_df['New Team'])
    metrics['changed_team_percentage'] = (changed_teams / total_people) * 100
    
    # Team size statistics
    team_sizes = new_teams_df.groupby('New Team').size()
    metrics['team_size_min'] = team_sizes.min()
    metrics['team_size_max'] = team_sizes.max()
    metrics['team_size_avg'] = team_sizes.mean()
    
    # Officer/Enlisted/Recruit ratio per team
    officer_ratio = []
    enlisted_ratio = []
    recruit_ratio = []
    ranger_ratio = []
    
    for team in new_teams_df['New Team'].unique():
        team_data = new_teams_df[new_teams_df['New Team'] == team]
        officers = sum(team_data['COMP'].apply(is_officer))
        enlisted = sum(team_data['COMP'].apply(is_enlisted))
        recruits = sum(team_data['COMP'].apply(is_recruit))
        rangers = sum(team_data['RGR'] == 'Y')
        
        officer_ratio.append(officers / len(team_data) * 100)
        enlisted_ratio.append(enlisted / len(team_data) * 100)
        recruit_ratio.append(recruits / len(team_data) * 100)
        ranger_ratio.append(rangers / len(team_data) * 100)
    
    metrics['officer_ratio'] = officer_ratio
    metrics['enlisted_ratio'] = enlisted_ratio
    metrics['recruit_ratio'] = recruit_ratio
    metrics['ranger_ratio'] = ranger_ratio
    metrics['officer_ratio_std'] = np.std(officer_ratio)
    metrics['enlisted_ratio_std'] = np.std(enlisted_ratio)
    metrics['recruit_ratio_std'] = np.std(recruit_ratio)
    metrics['ranger_ratio_std'] = np.std(ranger_ratio)
    
    # TIS distribution
    tis_std_by_team = new_teams_df.groupby('New Team')['TIS'].std().mean()
    metrics['tis_std'] = tis_std_by_team
    
    # Grade distribution
    # Convert grades to numeric for calculation purposes
    # E grades
    e_grades = [grade for grade in original_df['GRADE'].unique() if grade.startswith('E')]
    o_grades = [grade for grade in original_df['GRADE'].unique() if grade.startswith('O')]
    
    # Sort by numeric part
    e_grades.sort(key=lambda x: int(x[1:]))
    o_grades.sort(key=lambda x: int(x[1:]))
    
    # Combine in order
    all_grades = e_grades + o_grades
    
    rank_mapping = {rank: i for i, rank in enumerate(all_grades)}
    new_teams_df['Grade_Numeric'] = new_teams_df['GRADE'].map(rank_mapping)
    rank_std_by_team = new_teams_df.groupby('New Team')['Grade_Numeric'].std().mean()
    metrics['rank_std'] = rank_std_by_team
    
    # Total Points balance
    # Check if officers have inverse correlation with enlisted/recruits in terms of points
    correlation_metrics = []
    for team in new_teams_df['New Team'].unique():
        team_data = new_teams_df[new_teams_df['New Team'] == team]
        if sum(team_data['COMP'].apply(is_officer)) > 0 and (sum(team_data['COMP'].apply(is_enlisted)) + sum(team_data['COMP'].apply(is_recruit))) > 0:
            officer_points = team_data[team_data['COMP'].apply(is_officer)]['TOTAL'].mean()
            other_points = team_data[~team_data['COMP'].apply(is_officer)]['TOTAL'].mean()
            correlation_metrics.append(officer_points - other_points)
    
    metrics['points_balance'] = np.mean(correlation_metrics)
    
    return metrics

def reshuffle_teams(df):
    # Copy the original dataframe
    original_df = df.copy()
    
    # Replace missing values in TOTAL with the mean
    df['TOTAL'] = df['TOTAL'].fillna(df['TOTAL'].mean())
    
    # Determine the number of teams needed
    total_people = len(df)
    ideal_team_size = 18
    min_team_size = 13
    
    num_teams = max(math.ceil(total_people / ideal_team_size), math.ceil(total_people / min_team_size))
    target_team_size = total_people // num_teams
    
    # Ensure we have at least one officer per team
    officers = df[df['COMP'].apply(is_officer)].copy()
    enlisted = df[df['COMP'].apply(is_enlisted)].copy()
    recruits = df[df['COMP'].apply(is_recruit)].copy()
    
    if len(officers) < num_teams:
        st.warning(f"Not enough officers ({len(officers)}) to have at least one per team ({num_teams} teams). Continuing with available officers.")
        num_teams = len(officers)
    
    # Sort officers, enlisted and recruits by Total Points
    officers = officers.sort_values('TOTAL', ascending=False)
    enlisted = enlisted.sort_values('TOTAL', ascending=True)  # Lower performing enlisted with higher performing officers
    recruits = recruits.sort_values('TOTAL', ascending=True)  # Same for recruits
    
    # Initialize new teams
    new_teams = {i+1: [] for i in range(num_teams)}
    
    # First, distribute officers to ensure at least one per team
    # Higher performing officers go to teams 1, 2, etc.
    for i, officer_idx in enumerate(officers.index):
        team_idx = i % num_teams + 1
        new_teams[team_idx].append(officer_idx)
    
    # Next, distribute enlisted and recruits
    # Sort both lists by original team to maximize team changes
    enlisted = enlisted.sort_values(['Original Team', 'TIS', 'GRADE', 'TOTAL'])
    recruits = recruits.sort_values(['Original Team', 'TIS', 'GRADE', 'TOTAL'])
    
    # Combine and distribute enlisted and recruits
    others = pd.concat([enlisted, recruits])
    
    # Create a scoring function that encourages different teams
    def score_assignment(person_idx, team_idx):
        person = df.loc[person_idx]
        team_members = [df.loc[idx] for idx in new_teams[team_idx]]
        
        # Start with a base score
        score = 0
        
        # Heavily penalize same original team
        same_team_count = sum(member['Original Team'] == person['Original Team'] for member in team_members)
        score -= same_team_count * 100
        
        # Penalize team size imbalance
        target_size = total_people // num_teams
        size_diff = abs(len(team_members) - target_size)
        score -= size_diff * 50
        
        # Consider grade distribution
        grade_count = {}
        for member in team_members:
            grade_count[member['GRADE']] = grade_count.get(member['GRADE'], 0) + 1
        if person['GRADE'] in grade_count:
            score -= grade_count[person['GRADE']] * 10
        
        # Consider TIS distribution
        tis_values = [member['TIS'] for member in team_members]
        if tis_values:
            avg_tis = sum(tis_values) / len(tis_values)
            tis_diff = abs(person['TIS'] - avg_tis)
            score -= tis_diff * 0.5
        
        # Encourage balance of Total Points
        if is_officer(person['COMP']):
            # For officers, prefer teams with lower average points for non-officers
            non_officer_points = [member['TOTAL'] for member in team_members 
                                if not is_officer(member['COMP'])]
            if non_officer_points:
                avg_points = sum(non_officer_points) / len(non_officer_points)
                score += avg_points  # Higher points of non-officers is good for high-point officers
        else:
            # For enlisted/recruits, prefer teams with higher average points for officers
            officer_points = [member['TOTAL'] for member in team_members 
                             if is_officer(member['COMP'])]
            if officer_points:
                avg_points = sum(officer_points) / len(officer_points)
                score += avg_points  # Higher points of officers is good for low-point enlisted/recruits
        
        # Consider RGR distribution (lowest priority)
        rgr_values = [member['RGR'] for member in team_members]
        rgr_count_y = rgr_values.count('Y')
        rgr_count_n = rgr_values.count('N')
        
        if person['RGR'] == 'Y' and rgr_count_y > rgr_count_n:
            score -= 5  # Small penalty if adding another 'Y' to a team that already has more 'Y' than 'N'
        elif person['RGR'] == 'N' and rgr_count_n > rgr_count_y:
            score -= 5  # Small penalty if adding another 'N' to a team that already has more 'N' than 'Y'
        
        return score
    
    # Distribute the rest based on the scoring function
    for person_idx in others.index:
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

def display_metrics(metrics):
    st.subheader("Reshuffling Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Changed Teams", f"{metrics['changed_team_percentage']:.1f}%")
        st.metric("Min Team Size", metrics['team_size_min'])
        st.metric("Max Team Size", metrics['team_size_max'])
        st.metric("Avg Team Size", f"{metrics['team_size_avg']:.1f}")
    
    with col2:
        st.metric("Officer Ratio Std Dev", f"{metrics['officer_ratio_std']:.2f}%")
        st.metric("Enlisted Ratio Std Dev", f"{metrics['enlisted_ratio_std']:.2f}%")
        st.metric("Recruit Ratio Std Dev", f"{metrics['recruit_ratio_std']:.2f}%")
    
    with col3:
        st.metric("Ranger Ratio Std Dev", f"{metrics['ranger_ratio_std']:.2f}%")
        st.metric("TIS Distribution Std Dev", f"{metrics['tis_std']:.2f}")
        st.metric("Grade Distribution Std Dev", f"{metrics['rank_std']:.2f}")
    
    st.subheader("Team Composition")
    composition_data = pd.DataFrame({
        'Team': [f"Team {i+1}" for i in range(len(metrics['officer_ratio']))],
        'Officer %': [f"{x:.1f}%" for x in metrics['officer_ratio']],
        'Enlisted %': [f"{x:.1f}%" for x in metrics['enlisted_ratio']],
        'Recruit %': [f"{x:.1f}%" for x in metrics['recruit_ratio']],
        'Ranger %': [f"{x:.1f}%" for x in metrics['ranger_ratio']]
    })
    st.dataframe(composition_data)

def to_excel(df_dict):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
    for sheet_name, df in df_dict.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def preprocess_data(df):
    # Rename columns if needed
    if 'Original Team' not in df.columns:
        st.warning("No 'Original Team' column found. Please select the column to use as original team assignment.")
        team_col = st.selectbox("Select Original Team column:", df.columns)
        if team_col:
            df['Original Team'] = df[team_col]
        else:
            st.warning("No column selected. Creating random original teams for demonstration.")
            # Create random team assignments for demonstration
            num_teams = max(1, len(df) // 18)  # Aim for teams of around 18
            df['Original Team'] = [random.randint(1, num_teams) for _ in range(len(df))]
    
    # Handle missing values
    if df['TOTAL'].isna().any():
        st.warning(f"Missing values found in TOTAL column. {df['TOTAL'].isna().sum()} values will be replaced with the mean.")
        df['TOTAL'] = df['TOTAL'].fillna(df['TOTAL'].mean())
    
    if df['TIS'].isna().any():
        st.warning(f"Missing values found in TIS column. {df['TIS'].isna().sum()} values will be replaced with 0.")
        df['TIS'] = df['TIS'].fillna(0)
    
    return df

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        
        # Check if the dataframe has the required columns
        required_columns = ['ROSTER', 'COMP', 'GRADE', 'RGR', 'TIS', 'TOTAL']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
        else:
            # Preprocess data
            df = preprocess_data(df)
            
            st.write("Original data preview:")
            st.dataframe(df.head())
            
            # Option to manually set original team if not in data
            if 'Original Team' not in df.columns:
                method = st.radio("Choose how to assign Original Teams:", 
                                 ["Randomly assign teams", "Use existing column"])
                
                if method == "Use existing column":
                    team_col = st.selectbox("Select column to use as Original Team:", df.columns)
                    df['Original Team'] = df[team_col]
                else:
                    num_teams = st.number_input("Number of original teams:", min_value=1, value=max(1, len(df) // 18))
                    df['Original Team'] = [random.randint(1, int(num_teams)) for _ in range(len(df))]
            
            if st.button("Reshuffle Teams"):
                with st.spinner("Reshuffling teams..."):
                    new_teams_df = reshuffle_teams(df)
                    metrics = calculate_metrics(df, new_teams_df)
                    
                    st.success("Teams reshuffled successfully!")
                    
                    display_metrics(metrics)
                    
                    st.subheader("New Team Assignments")
                    st.dataframe(new_teams_df.sort_values(['New Team', 'COMP']))
                    
                    # Prepare Excel download
                    excel_data = to_excel({
                        'Original Data': df,
                        'New Teams': new_teams_df
                    })
                    
                    st.download_button(
                        label="Download Excel with new teams",
                        data=excel_data,
                        file_name="reshuffled_teams.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                    
                    # Show detailed team view
                    st.subheader("Team Details")
                    for team in sorted(new_teams_df['New Team'].unique()):
                        with st.expander(f"Team {team}"):
                            team_data = new_teams_df[new_teams_df['New Team'] == team]
                            officers = team_data[team_data['COMP'].apply(is_officer)]
                            enlisted = team_data[team_data['COMP'].apply(is_enlisted)]
                            recruits = team_data[team_data['COMP'].apply(is_recruit)]
                            rangers = team_data[team_data['RGR'] == 'Y']
                            
                            st.write(f"Team Size: {len(team_data)}")
                            st.write(f"Officers: {len(officers)} ({len(officers)/len(team_data)*100:.1f}%)")
                            st.write(f"Enlisted: {len(enlisted)} ({len(enlisted)/len(team_data)*100:.1f}%)")
                            st.write(f"Recruits: {len(recruits)} ({len(recruits)/len(team_data)*100:.1f}%)")
                            st.write(f"Rangers: {len(rangers)} ({len(rangers)/len(team_data)*100:.1f}%)")
                            
                            st.write(f"Average TIS: {team_data['TIS'].mean():.1f}")
                            st.write(f"Average Points: {team_data['TOTAL'].mean():.1f}")
                            
                            # Show how many members are from the same original team
                            original_teams = team_data['Original Team'].value_counts()
                            st.write("Members from original teams:")
                            for orig_team, count in original_teams.items():
                                st.write(f"- Team {orig_team}: {count} members")
                            
                            st.dataframe(team_data)
    
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload an Excel file with team data.")
    
    # Show example format
    st.subheader("Expected Excel Format:")
    example_data = {
        'ROSTER': [1, 2, 3, 4, 5],
        'COMP': ['ADO', 'NGO', 'ADE', 'NGE', 'AD18X'],
        'GRADE': ['O3', 'O2', 'E6', 'E4', 'E3'],
        'RGR': ['Y', 'N', 'Y', 'N', 'N'],
        'TIS': [10, 8, 6, 4, 0],
        'TOTAL': [95, 85, 75, 60, 50],
        'Original Team': [1, 2, 1, 3, 2]
    }
    example_df = pd.DataFrame(example_data)
    st.dataframe(example_df)
    
    st.markdown("""
    ### File Requirements:
    - Excel format (.xlsx)
    - Must contain all required columns:
      - **ROSTER**: Unique identifier for each person
      - **COMP**: Type of person (ADO, NGO, ADE, AD18X, NGE, NG18X)
      - **GRADE**: Military grade (O3, O2, E6, E5, etc.)
      - **RGR**: Ranger status (Y/N)
      - **TIS**: Time in service (years)
      - **TOTAL**: Performance points earned
      - **Original Team**: Current team number (if not available, will be requested during processing)
    
    ### Reshuffling Logic:
    1. Teams will be created with 13-18 members (ideal is 18)
    2. Each team will have at least one officer
    3. Teams will be balanced for:
       - Different people than original teams (highest priority)
       - Rank distribution
       - Total points (high-performing officers with low-performing enlisted/recruits)
       - Time in service (TIS)
       - Ranger status (RGR) (lowest priority)
    """)

# Add footer with instructions for GitHub deployment
st.markdown("---")
st.markdown("""
### How to deploy this app:
1. Create a GitHub repository
2. Add this file as `app.py`
3. Create a `requirements.txt` 