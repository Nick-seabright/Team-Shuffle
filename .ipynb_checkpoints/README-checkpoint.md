# Team Reshuffler

A flexible Streamlit application for creating and reshuffling teams based on configurable criteria and priorities.

## Overview

This Team Reshuffler application is designed to help organize personnel into balanced teams. It offers two main functionalities:

1. **Create new teams from scratch**: Form teams based on balanced distribution of attributes
2. **Reshuffle existing teams**: Reorganize teams while minimizing people remaining on the same team

The application is highly configurable, allowing users to select which data columns to use and their priority order in team formation.

## Features

- **Flexible Data Input**: Works with any Excel file format - column names don't matter
- **Dynamic Configuration**: Select which columns to use for different purposes (ID, team assignment, priorities)
- **Intelligent Column Detection**: Automatically identifies potential ID columns, team columns, and columns with filled/empty values
- **Customizable Priorities**: Set the order of importance for different attributes when forming teams
- **Comprehensive Metrics**: View detailed statistics on team balance across all selected attributes
- **Team Details**: Explore the composition and statistics for each team
- **Export Results**: Download the team assignments as an Excel file

## How It Works

1. **Upload Data**: Start by uploading an Excel file with your personnel data
2. **Select Operation**: Choose whether to create new teams or reshuffle existing ones
3. **Configure Columns**:
   - Select which column contains unique identifiers
   - Choose which column (if any) represents personnel types
   - For reshuffling, select which column contains original team assignments
   - Choose and prioritize columns to use for team formation
4. **Process Teams**: The algorithm creates balanced teams based on your configuration
5. **View Results**: Examine team metrics and detailed breakdowns
6. **Download**: Export the new team assignments to Excel

## Special Features

- **Personnel Type Detection**: Automatically identifies columns that might represent personnel types (e.g., officers, enlisted, recruits)
- **Filled/Empty Value Balancing**: Automatically detects and balances columns with both filled and empty values (useful for experienced/inexperienced members)
- **Team Size Optimization**: Creates teams of 13-18 members with ideal size being 18
- **Officer Distribution**: Ensures at least one officer per team (if personnel type column is provided)
- **Priority-Based Scoring**: Assigns people to teams using a sophisticated scoring system based on your priority order

## Team Formation Logic

The application uses a scoring algorithm that:
1. Assigns greater weight to higher-priority columns
2. Ensures even distribution of categorical values (like ranks, types, etc.)
3. Balances numeric values (like time in service, performance points)
4. Evenly distributes personnel with filled vs. empty values for relevant columns
5. When reshuffling, minimizes the number of people who remain on their original team

## Local Development

To run this application locally:

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   streamlit run app.py
   ```

## Deployment

This application is designed to be easily deployed to Streamlit Cloud:

1. Fork or clone this repository to your GitHub account
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy the application

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- openpyxl
- xlsxwriter

## License

[MIT License](LICENSE)

## Contact

For questions or support, please open an issue on this repository.

---

*Originally developed for military team management but adaptable for any team organization needs.*