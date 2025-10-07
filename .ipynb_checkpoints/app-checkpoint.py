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
    """Check if a value indicates a recruit based on ending with 'X'"""
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
    
    # Calculate teams based on min and max team size
    max_possible_teams = math.floor(total_people / min_team_size)
    min_needed_teams = math.ceil(total_people / max_team_size)
    
    # Use the minimum number of teams that satisfies both constraints
    num_teams = max(min_needed_teams, 1)
    
    # If min team size would make teams too large, warn and adjust
    if max_possible_teams < num_teams:
        st.warning(f"Can't create teams with at least {min_team_size} members each. Will create {num_teams} teams instead.")
        min_team_size = math.floor(total_people / num_teams)
    
    st.write(f"Creating {num_teams} teams for {total_people} people.")
    st.write(f"Team size range: {min_team_size}-{max_team_size} members (ideal: max)")
    
    # If we have a composition column, ensure at least one officer per team
    if comp_col:
        officers = df[df[comp_col].apply(lambda x: is_officer(x, comp_col))].copy() if comp_col else pd.DataFrame()
        
        if not officers.empty and len(officers) < num_teams:
            st.warning(f"Not enough officers ({len(officers)}) to have at least one per team ({num_teams} teams). Continuing with available officers.")
            # In this case, adjust the number of teams to match available officers
            if len(officers) > 0:  # Make sure we have at least one officer
                num_teams = len(officers)
                st.write(f"Adjusted to {num_teams} teams due to officer constraint.")
    else:
        officers = pd.DataFrame()
    
    # Initialize new teams
    new_teams = {i+1: [] for i in range(num_teams)}
    
    # Create a tracking dictionary to store current team sizes
    team_sizes = {i+1: 0 for i in range(num_teams)}
    
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
            team_sizes[team_idx] += 1
    
    # Get the rest of the personnel
    if not officers.empty:
        rest = df.loc[~df.index.isin(officers.index)].copy()
    else:
        rest = df.copy()
    
    # Sort rest by priority columns
    if 'sort_columns' in locals() and sort_columns:
        rest = rest.sort_values(sort_columns)
    
    # Calculate overall target ratios if comp_col is provided
    if comp_col:
        total_officers = sum(df[comp_col].apply(lambda x: is_officer(x, comp_col)))
        total_enlisted = sum(df[comp_col].apply(lambda x: is_enlisted(x, comp_col)))
        total_recruits = sum(df[comp_col].apply(lambda x: is_recruit(x, comp_col)))
        
        target_officer_ratio = total_officers / len(df)
        target_enlisted_ratio = total_enlisted / len(df)
        target_recruit_ratio = total_recruits / len(df)
    
    # Create a scoring function for team assignment with strict max size enforcement
    def score_assignment(person_idx, team_idx):
        person = df.loc[person_idx]
        team_members = [df.loc[idx] for idx in new_teams[team_idx]]
        team_size = team_sizes[team_idx]
        
        # If team is already at max size, make it ineligible
        if team_size >= max_team_size:
            return float('-inf')  # Return negative infinity to ensure this team is not selected
        
        # If adding to this team would make it below min size, strongly encourage it
        total_remaining = len(rest) - len([p for p in rest.index if p in [member for members in new_teams.values() for member in members]])
        teams_below_min = [t for t, size in team_sizes.items() if size < min_team_size]
        
        if len(teams_below_min) > 0 and total_remaining <= len(teams_below_min) * (min_team_size - team_size):
            if team_size < min_team_size:
                return 100000  # Very high score to ensure teams reach minimum size
        
        # Start with a base score
        score = 0
        
        # Penalize teams closer to max size less (to fill them up first)
        # But still maintain a slight penalty to balance other factors
        size_penalty = (max_team_size - team_size) * 10
        score -= size_penalty
        
        # Add penalties based on priority columns
        penalty_weight = 1000  # Start with high weight and decrease for lower priorities
        
        # If reshuffling, heavily penalize same original team
        if is_reshuffle and original_team_col and original_team_col in df.columns:
            same_team_count = sum(member[original_team_col] == person[original_team_col] for member in team_members)
            score -= same_team_count * penalty_weight
            penalty_weight *= 0.8  # Reduce weight for next priority
        
        # If we have a composition column, balance officer/enlisted/recruit (higher priority)
        if comp_col and comp_col in df.columns:
            comp_types = [member[comp_col] for member in team_members if not pd.isna(member[comp_col])]
            if comp_types:
                # Count officers, enlisted, recruits
                officer_count = sum(is_officer(comp, comp_col) for comp in comp_types)
                enlisted_count = sum(is_enlisted(comp, comp_col) for comp in comp_types) 
                recruit_count = sum(is_recruit(comp, comp_col) for comp in comp_types)
                
                # Calculate current team ratios
                current_team_size = len(team_members)
                current_officer_ratio = officer_count / current_team_size if current_team_size > 0 else 0
                current_enlisted_ratio = enlisted_count / current_team_size if current_team_size > 0 else 0
                current_recruit_ratio = recruit_count / current_team_size if current_team_size > 0 else 0
                
                # Check if adding this person would improve the ratio balance
                person_comp = person[comp_col]
                new_team_size = current_team_size + 1
                
                if is_officer(person_comp, comp_col):
                    new_officer_ratio = (officer_count + 1) / new_team_size
                    # Penalty if adding officer makes ratio deviate more from target
                    ratio_deviation = abs(new_officer_ratio - target_officer_ratio)
                    current_deviation = abs(current_officer_ratio - target_officer_ratio)
                    score -= (ratio_deviation - current_deviation) * penalty_weight * 300
                elif is_enlisted(person_comp, comp_col):
                    new_enlisted_ratio = (enlisted_count + 1) / new_team_size
                    # Penalty if adding enlisted makes ratio deviate more from target
                    ratio_deviation = abs(new_enlisted_ratio - target_enlisted_ratio)
                    current_deviation = abs(current_enlisted_ratio - target_enlisted_ratio)
                    score -= (ratio_deviation - current_deviation) * penalty_weight * 300
                elif is_recruit(person_comp, comp_col):
                    new_recruit_ratio = (recruit_count + 1) / new_team_size
                    # Penalty if adding recruit makes ratio deviate more from target
                    ratio_deviation = abs(new_recruit_ratio - target_recruit_ratio)
                    current_deviation = abs(current_recruit_ratio - target_recruit_ratio)
                    score -= (ratio_deviation - current_deviation) * penalty_weight * 300
            
            penalty_weight *= 0.7  # Reduce weight for next priority
        
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
        
        return score
    
    # Distribute the rest based on the scoring function
    for person_idx in rest.index:
        # Calculate scores for each team
        scores = {team_idx: score_assignment(person_idx, team_idx) for team_idx in new_teams.keys()}
        
        # Check if any team is eligible (not at max capacity)
        if all(score == float('-inf') for score in scores.values()):
            # If all teams are at max capacity, we need to add a new team
            new_team_idx = len(new_teams) + 1
            new_teams[new_team_idx] = [person_idx]
            team_sizes[new_team_idx] = 1
            st.warning(f"Added additional team {new_team_idx} because all existing teams reached maximum size.")
        else:
            # Assign to the team with the highest score
            best_team = max(scores.items(), key=lambda x: x[1])[0]
            new_teams[best_team].append(person_idx)
            team_sizes[best_team] += 1
    
    # Perform a second pass to balance composition if comp_col is provided
    if comp_col:
        st.write("Performing composition balancing...")
        
        # Get composition stats for each team
        team_comp_stats = {}
        for team_idx in range(1, num_teams + 1):
            team_members = new_teams.get(team_idx, [])
            if team_members:
                team_data = df.loc[team_members]
                officer_count = sum(team_data[comp_col].apply(lambda x: is_officer(x, comp_col)))
                enlisted_count = sum(team_data[comp_col].apply(lambda x: is_enlisted(x, comp_col)))
                recruit_count = sum(team_data[comp_col].apply(lambda x: is_recruit(x, comp_col)))
                
                team_comp_stats[team_idx] = {
                    'officer_ratio': officer_count / len(team_members),
                    'enlisted_ratio': enlisted_count / len(team_members),
                    'recruit_ratio': recruit_count / len(team_members),
                    'size': len(team_members)
                }
        
        # Calculate overall composition ratios
        total_officers = sum(df[comp_col].apply(lambda x: is_officer(x, comp_col)))
        total_enlisted = sum(df[comp_col].apply(lambda x: is_enlisted(x, comp_col)))
        total_recruits = sum(df[comp_col].apply(lambda x: is_recruit(x, comp_col)))
        
        target_officer_ratio = total_officers / total_people
        target_enlisted_ratio = total_enlisted / total_people
        target_recruit_ratio = total_recruits / total_people
        
        # Perform composition balancing iterations
        max_iterations = 20
        total_swaps = 0
        for iteration in range(max_iterations):
            made_swap = False
            
            # For each team with high/low officer ratio, try to swap with a team that has opposite imbalance
            for team_a_idx in range(1, num_teams + 1):
                if team_a_idx not in team_comp_stats:
                    continue
                    
                team_a_stats = team_comp_stats[team_a_idx]
                
                # Check if this team has composition imbalance
                a_officer_imbalance = team_a_stats['officer_ratio'] - target_officer_ratio
                a_enlisted_imbalance = team_a_stats['enlisted_ratio'] - target_enlisted_ratio
                a_recruit_imbalance = team_a_stats['recruit_ratio'] - target_recruit_ratio
                
                # Find a team with opposite imbalance to swap with
                for team_b_idx in range(1, num_teams + 1):
                    if team_b_idx == team_a_idx or team_b_idx not in team_comp_stats:
                        continue
                        
                    team_b_stats = team_comp_stats[team_b_idx]
                    
                    # Check if team B has opposite imbalance
                    b_officer_imbalance = team_b_stats['officer_ratio'] - target_officer_ratio
                    b_enlisted_imbalance = team_b_stats['enlisted_ratio'] - target_enlisted_ratio
                    b_recruit_imbalance = team_b_stats['recruit_ratio'] - target_recruit_ratio
                    
                    # If teams have opposite imbalances, try to swap members
                    if (a_officer_imbalance * b_officer_imbalance < 0 or 
                        a_enlisted_imbalance * b_enlisted_imbalance < 0 or
                        a_recruit_imbalance * b_recruit_imbalance < 0):
                        
                        # Find candidates for swapping
                        team_a_members = new_teams[team_a_idx]
                        team_b_members = new_teams[team_b_idx]
                        
                        # Identify what we need to swap:
                        swap_type = None
                        if abs(a_officer_imbalance) > 0.05 and abs(b_officer_imbalance) > 0.05:
                            if a_officer_imbalance > 0 and b_officer_imbalance < 0:
                                # Team A has too many officers, team B has too few
                                swap_type = ('officer', 'non_officer')
                            elif a_officer_imbalance < 0 and b_officer_imbalance > 0:
                                # Team B has too many officers, team A has too few
                                swap_type = ('non_officer', 'officer')
                        elif abs(a_enlisted_imbalance) > 0.05 and abs(b_enlisted_imbalance) > 0.05:
                            if a_enlisted_imbalance > 0 and b_enlisted_imbalance < 0:
                                # Team A has too many enlisted, team B has too few
                                swap_type = ('enlisted', 'non_enlisted')
                            elif a_enlisted_imbalance < 0 and b_enlisted_imbalance > 0:
                                # Team B has too many enlisted, team A has too few
                                swap_type = ('non_enlisted', 'enlisted')
                        elif abs(a_recruit_imbalance) > 0.05 and abs(b_recruit_imbalance) > 0.05:
                            if a_recruit_imbalance > 0 and b_recruit_imbalance < 0:
                                # Team A has too many recruits, team B has too few
                                swap_type = ('recruit', 'non_recruit')
                            elif a_recruit_imbalance < 0 and b_recruit_imbalance > 0:
                                # Team B has too many recruits, team A has too few
                                swap_type = ('non_recruit', 'recruit')
                        
                        if swap_type:
                            # Find candidates based on swap type
                            from_type, to_type = swap_type
                            from_team_idx = team_a_idx
                            to_team_idx = team_b_idx
                            
                            # Select candidates from each team
                            if from_type == 'officer':
                                from_candidates = [idx for idx in team_a_members 
                                              if is_officer(df.loc[idx, comp_col], comp_col)]
                            elif from_type == 'non_officer':
                                from_candidates = [idx for idx in team_a_members 
                                              if not is_officer(df.loc[idx, comp_col], comp_col)]
                            elif from_type == 'enlisted':
                                from_candidates = [idx for idx in team_a_members 
                                              if is_enlisted(df.loc[idx, comp_col], comp_col)]
                            elif from_type == 'non_enlisted':
                                from_candidates = [idx for idx in team_a_members 
                                              if not is_enlisted(df.loc[idx, comp_col], comp_col)]
                            elif from_type == 'recruit':
                                from_candidates = [idx for idx in team_a_members 
                                              if is_recruit(df.loc[idx, comp_col], comp_col)]
                            elif from_type == 'non_recruit':
                                from_candidates = [idx for idx in team_a_members 
                                              if not is_recruit(df.loc[idx, comp_col], comp_col)]
                            
                            if to_type == 'officer':
                                to_candidates = [idx for idx in team_b_members 
                                            if is_officer(df.loc[idx, comp_col], comp_col)]
                            elif to_type == 'non_officer':
                                to_candidates = [idx for idx in team_b_members 
                                            if not is_officer(df.loc[idx, comp_col], comp_col)]
                            elif to_type == 'enlisted':
                                to_candidates = [idx for idx in team_b_members 
                                            if is_enlisted(df.loc[idx, comp_col], comp_col)]
                            elif to_type == 'non_enlisted':
                                to_candidates = [idx for idx in team_b_members 
                                            if not is_enlisted(df.loc[idx, comp_col], comp_col)]
                            elif to_type == 'recruit':
                                to_candidates = [idx for idx in team_b_members 
                                            if is_recruit(df.loc[idx, comp_col], comp_col)]
                            elif to_type == 'non_recruit':
                                to_candidates = [idx for idx in team_b_members 
                                            if not is_recruit(df.loc[idx, comp_col], comp_col)]
                            
                            # If we have candidates, perform a swap
                            if from_candidates and to_candidates:
                                # Choose candidates at random
                                from_idx = random.choice(from_candidates)
                                to_idx = random.choice(to_candidates)
                                
                                # Swap the members
                                new_teams[from_team_idx].remove(from_idx)
                                new_teams[from_team_idx].append(to_idx)
                                new_teams[to_team_idx].remove(to_idx)
                                new_teams[to_team_idx].append(from_idx)
                                
                                # Update team composition stats
                                for team_idx in [from_team_idx, to_team_idx]:
                                    team_members = new_teams[team_idx]
                                    team_data = df.loc[team_members]
                                    officer_count = sum(team_data[comp_col].apply(lambda x: is_officer(x, comp_col)))
                                    enlisted_count = sum(team_data[comp_col].apply(lambda x: is_enlisted(x, comp_col)))
                                    recruit_count = sum(team_data[comp_col].apply(lambda x: is_recruit(x, comp_col)))
                                    
                                    team_comp_stats[team_idx] = {
                                        'officer_ratio': officer_count / len(team_members),
                                        'enlisted_ratio': enlisted_count / len(team_members),
                                        'recruit_ratio': recruit_count / len(team_members),
                                        'size': len(team_members)
                                    }
                                
                                total_swaps += 1
                                made_swap = True
                                break
                
                if made_swap:
                    break
            
            if not made_swap:
                break
        
        if total_swaps > 0:
            st.write(f"Made {total_swaps} swaps to improve composition balance")
    
    # Make one final pass to ensure minimum team size
    team_sizes = {team_idx: len(members) for team_idx, members in new_teams.items()}
    teams_below_min = {team_idx: size for team_idx, size in team_sizes.items() if size < min_team_size}
    
    if teams_below_min:
        st.warning(f"Some teams are below the minimum size: {teams_below_min}")
        
        # Try to redistribute from largest teams to meet minimum size
        teams_above_min = {team_idx: size for team_idx, size in team_sizes.items() if size > min_team_size}
        
        # Sort teams by size (largest first)
        sorted_teams_above_min = sorted(teams_above_min.items(), key=lambda x: x[1], reverse=True)
        
        for small_team_idx, small_team_size in teams_below_min.items():
            members_needed = min_team_size - small_team_size
            
            for large_team_idx, large_team_size in sorted_teams_above_min:
                available_to_move = large_team_size - min_team_size
                
                if available_to_move > 0:
                    # Find members to move
                    num_to_move = min(members_needed, available_to_move)
                    
                    if num_to_move > 0:
                        # Try to select diverse members to maintain balance
                        members_to_move = []
                        large_team_members = new_teams[large_team_idx]
                        
                        if comp_col:
                            # Get a mix of member types
                            large_team_data = df.loc[large_team_members]
                            officers = large_team_data[large_team_data[comp_col].apply(lambda x: is_officer(x, comp_col))]
                            enlisted = large_team_data[large_team_data[comp_col].apply(lambda x: is_enlisted(x, comp_col))]
                            recruits = large_team_data[large_team_data[comp_col].apply(lambda x: is_recruit(x, comp_col))]
                            
                            # Determine how many of each to move based on overall ratios
                            officers_to_move = min(len(officers), int(num_to_move * target_officer_ratio) + 1)
                            enlisted_to_move = min(len(enlisted), int(num_to_move * target_enlisted_ratio) + 1)
                            recruits_to_move = min(len(recruits), num_to_move - officers_to_move - enlisted_to_move)
                            
                            # Adjust if we're not moving enough
                            total_moving = officers_to_move + enlisted_to_move + recruits_to_move
                            if total_moving < num_to_move:
                                # Add more from the largest category
                                if len(enlisted) > len(officers) and len(enlisted) > len(recruits):
                                    enlisted_to_move += num_to_move - total_moving
                                elif len(recruits) > len(officers) and len(recruits) > len(enlisted):
                                    recruits_to_move += num_to_move - total_moving
                                else:
                                    officers_to_move += num_to_move - total_moving
                            
                            # Select members to move
                            if officers_to_move > 0:
                                members_to_move.extend(officers.sample(min(officers_to_move, len(officers))).index.tolist())
                            if enlisted_to_move > 0:
                                members_to_move.extend(enlisted.sample(min(enlisted_to_move, len(enlisted))).index.tolist())
                            if recruits_to_move > 0:
                                members_to_move.extend(recruits.sample(min(recruits_to_move, len(recruits))).index.tolist())
                        else:
                            # Just select random members
                            members_to_move = random.sample(large_team_members, num_to_move)
                        
                        # Move the members
                        for member_idx in members_to_move:
                            new_teams[large_team_idx].remove(member_idx)
                            new_teams[small_team_idx].append(member_idx)
                        
                        # Update tracking
                        members_needed -= len(members_to_move)
                        team_sizes[large_team_idx] -= len(members_to_move)
                        team_sizes[small_team_idx] += len(members_to_move)
                        
                        st.write(f"Moved {len(members_to_move)} members from Team {large_team_idx} to Team {small_team_idx}")
                
                if members_needed <= 0:
                    break
    
    # Create a new dataframe with the new team assignments
    new_df = df.copy()
    new_df['New Team'] = 0
    
    for team_idx, members in new_teams.items():
        for member_idx in members:
            new_df.loc[member_idx, 'New Team'] = team_idx
    
    # Verify no team exceeds maximum size
    team_size_check = new_df.groupby('New Team').size()
    team_size_dict = {int(team): int(size) for team, size in team_size_check.items()}
    st.write("Final team sizes:")
    team_size_df = pd.DataFrame({
        'Team': list(team_size_dict.keys()),
        'Size': list(team_size_dict.values())
    })
    st.table(team_size_df)
    
    if team_size_check.min() < min_team_size:
        st.warning(f"Some teams are still below minimum size. Smallest team has {team_size_check.min()} members (minimum: {min_team_size}).")
    
    if team_size_check.max() > max_team_size:
        st.error(f"Error: Team size check failed. Max team size: {team_size_check.max()} (should be ≤ {max_team_size})")
    
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
                st.write(f"18Xs: {len(recruits)} ({len(recruits)/len(team_data)*100:.1f}%)")
            
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

def edit_teams(new_teams_df, column_config):
    """
    Allow users to manually edit team assignments
    """
    id_col = column_config["id_column"]
    comp_col = column_config.get("comp_column")
    
    # Create a copy to avoid modifying the original
    if 'current_edited_df' not in st.session_state:
        st.session_state['current_edited_df'] = new_teams_df.copy()
    
    edited_df = st.session_state['current_edited_df']
    
    # Create two columns for the interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Move Member to Different Team")
        
        # Select a team to view
        teams = sorted(edited_df['New Team'].unique())
        source_team = st.selectbox("Select source team:", teams, key="source_team")
        
        # Display team members
        team_data = edited_df[edited_df['New Team'] == source_team]
        
        # Create a nice display name for each member
        if comp_col:
            team_data['Display Name'] = team_data.apply(
                lambda row: f"{row[id_col]} - {row[comp_col]} - {row['GRADE'] if 'GRADE' in row else ''}", 
                axis=1
            )
        else:
            # Use ID and any other relevant columns
            additional_cols = ['GRADE', 'RANK', 'Name'] 
            display_cols = [col for col in additional_cols if col in team_data.columns]
            
            if display_cols:
                team_data['Display Name'] = team_data.apply(
                    lambda row: f"{row[id_col]} - " + " - ".join([str(row[col]) for col in display_cols]), 
                    axis=1
                )
            else:
                team_data['Display Name'] = team_data[id_col].astype(str)
        
        # Select a member to move
        if not team_data.empty:
            member_options = team_data['Display Name'].tolist()
            selected_member = st.selectbox("Select member to move:", member_options, key="member_select")
            
            # Find the selected member's ID
            selected_member_row = team_data[team_data['Display Name'] == selected_member].iloc[0]
            selected_member_id = selected_member_row[id_col]
            
            # Select destination team
            other_teams = [team for team in teams if team != source_team]
            if other_teams:
                destination_team = st.selectbox(
                    "Select destination team:", 
                    other_teams,
                    key="destination_team"
                )
                
                # Check if destination team is at max capacity
                dest_team_size = len(edited_df[edited_df['New Team'] == destination_team])
                max_team_size = column_config.get("max_team_size", 18)
                
                if dest_team_size >= max_team_size:
                    st.warning(f"Team {destination_team} already has {dest_team_size} members (maximum: {max_team_size}).")
                    st.write("You can still proceed, but the team will exceed the maximum size.")
                
                if st.button("Move Member"):
                    # Update the team assignment
                    edited_df.loc[edited_df[id_col] == selected_member_id, 'New Team'] = destination_team
                    st.session_state['current_edited_df'] = edited_df  # Update stored dataframe
                    st.success(f"Moved {selected_member} from Team {source_team} to Team {destination_team}")
                    
                    # Check team sizes
                    source_size = len(edited_df[edited_df['New Team'] == source_team])
                    dest_size = len(edited_df[edited_df['New Team'] == destination_team])
                    
                    if source_size < column_config.get("min_team_size", 13):
                        st.warning(f"Team {source_team} now has {source_size} members (below minimum)")
                    if dest_size > column_config.get("max_team_size", 18):
                        st.warning(f"Team {destination_team} now has {dest_size} members (above maximum)")
                    
                    # Rerun to update the UI
                    st.experimental_rerun()
            else:
                st.info("No other teams available for moving members.")
        else:
            st.info(f"Team {source_team} has no members.")
    
    with col2:
        st.subheader("Swap Two Members")
        
        # Select first team
        first_team = st.selectbox("Select first team:", teams, key="first_team")
        
        # Display first team members
        first_team_data = edited_df[edited_df['New Team'] == first_team]
        if comp_col:
            first_team_data['Display Name'] = first_team_data.apply(
                lambda row: f"{row[id_col]} - {row[comp_col]} - {row['GRADE'] if 'GRADE' in row else ''}", 
                axis=1
            )
        else:
            first_team_data['Display Name'] = first_team_data[id_col].astype(str)
        
        # Select a member from first team
        if not first_team_data.empty:
            first_member = st.selectbox(
                "Select member from first team:", 
                first_team_data['Display Name'].tolist(),
                key="first_member"
            )
            
            # Find the selected member's ID
            first_member_id = first_team_data[first_team_data['Display Name'] == first_member].iloc[0][id_col]
            
            # Select second team
            other_teams = [team for team in teams if team != first_team]
            if other_teams:
                second_team = st.selectbox(
                    "Select second team:", 
                    other_teams,
                    key="second_team"
                )
                
                # Display second team members
                second_team_data = edited_df[edited_df['New Team'] == second_team]
                if comp_col:
                    second_team_data['Display Name'] = second_team_data.apply(
                        lambda row: f"{row[id_col]} - {row[comp_col]} - {row['GRADE'] if 'GRADE' in row else ''}", 
                        axis=1
                    )
                else:
                    second_team_data['Display Name'] = second_team_data[id_col].astype(str)
                
                # Select a member from second team
                if not second_team_data.empty:
                    second_member = st.selectbox(
                        "Select member from second team:", 
                        second_team_data['Display Name'].tolist(),
                        key="second_member"
                    )
                    
                    # Find the selected member's ID
                    second_member_id = second_team_data[second_team_data['Display Name'] == second_member].iloc[0][id_col]
                    
                    if st.button("Swap Members"):
                        # Update the team assignments
                        edited_df.loc[edited_df[id_col] == first_member_id, 'New Team'] = second_team
                        edited_df.loc[edited_df[id_col] == second_member_id, 'New Team'] = first_team
                        
                        st.session_state['current_edited_df'] = edited_df  # Update stored dataframe
                        st.success(f"Swapped {first_member} and {second_member}")
                        
                        # Rerun to update the UI
                        st.experimental_rerun()
                else:
                    st.info(f"Team {second_team} has no members.")
            else:
                st.info("No other teams available for swapping.")
        else:
            st.info(f"Team {first_team} has no members.")
    
    # Display the updated team assignments
    st.subheader("Updated Team Assignments")
    st.dataframe(edited_df.sort_values(['New Team', id_col]))
    
    # Calculate new metrics
    metrics, ratio_stats = calculate_metrics(edited_df, edited_df, column_config, is_reshuffle=False)
    display_metrics(metrics, ratio_stats, column_config, is_reshuffle=False)
    
    # Prepare Excel download for edited teams
    excel_data = to_excel({
        'Original Teams': new_teams_df,
        'Edited Teams': edited_df
    })
    
    st.download_button(
        label="Download Excel with edited teams",
        data=excel_data,
        file_name="edited_teams.xlsx",
        mime="application/vnd.ms-excel"
    )
    
    return edited_df

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
                    
                    # Store the results in session state
                    st.session_state['new_teams_df'] = new_teams_df
                    st.session_state['metrics'] = metrics
                    st.session_state['ratio_stats'] = ratio_stats
                    st.session_state['is_reshuffle'] = is_reshuffle
                    st.session_state['min_team_size'] = min_team_size
                    st.session_state['max_team_size'] = max_team_size
                    st.session_state['teams_processed'] = True
                    
                    # Clear any previous edited teams
                    if 'current_edited_df' in st.session_state:
                        del st.session_state['current_edited_df']
                    
                    st.success(f"Teams {'reshuffled' if is_reshuffle else 'created'} successfully!")
        
        # Check if we have processed teams in session state
        if 'teams_processed' in st.session_state and st.session_state['teams_processed']:
            # Retrieve values from session state
            new_teams_df = st.session_state['new_teams_df']
            metrics = st.session_state['metrics']
            ratio_stats = st.session_state['ratio_stats']
            is_reshuffle = st.session_state['is_reshuffle']
            min_team_size = st.session_state['min_team_size']
            max_team_size = st.session_state['max_team_size']
            
            # Display metrics and team assignments
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
            
            # Add manual editing option
            st.subheader("Edit Team Assignments")
            if 'edit_mode' not in st.session_state:
                st.session_state['edit_mode'] = False
            
            edit_mode = st.checkbox("Enable manual team editing", value=st.session_state['edit_mode'])
            st.session_state['edit_mode'] = edit_mode
            
            if edit_mode:
                # Store minimum and maximum team size in column_config for reference
                column_config["min_team_size"] = min_team_size
                column_config["max_team_size"] = max_team_size
                
                # If we have edited teams already, use those as the starting point
                starting_df = st.session_state.get('current_edited_df', new_teams_df)
                
                edited_teams_df = edit_teams(new_teams_df, column_config)
                
                # Store the edited teams
                st.session_state['edited_teams_df'] = edited_teams_df
    
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
    2. Which column (if any) indicates personnel types (officers/enlisted/recruits)
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