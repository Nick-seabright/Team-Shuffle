# Team Reshuffler

This Streamlit application reshuffles military teams based on various criteria to ensure optimal team composition and personnel distribution.

## Overview

The Team Reshuffler is designed to take an existing team structure and create new, balanced teams while ensuring:

- Teams have 13-18 members (ideally 18)
- Each team has at least one officer
- Team members are distributed to minimize people remaining on the same team as before
- Even distribution of ranks, time in service, and performance points
- Balance of officers and enlisted/recruits based on performance
- Proportional distribution of rangers across teams

## Features

- **Data Import**: Upload your team data as an Excel file
- **Intelligent Reshuffling**: Algorithm prioritizes different team members first, followed by rank distribution, total points, TIS, and ranger status
- **Comprehensive Metrics**: View statistics on team composition and how well the reshuffling meets criteria
- **Team Details**: Explore detailed breakdowns of each new team
- **Export Results**: Download the new team assignments as an Excel file

## Required Data Format

The application expects an Excel file with the following columns:

| Column | Description |
|--------|-------------|
| ROSTER | Unique identifier for each person |
| COMP | Type of person (ADO, NGO, ADE, NGE, AD18X, NG18X) - Officers end in O, enlisted end in E, recruits end in X |
| GRADE | Military grade (e.g., O3, O2, E6, E5) |
| RGR | Ranger status (Y/N) |
| TIS | Time in service (years) |
| TOTAL | Performance points earned |
| Original Team | Current team assignment |

## Reshuffling Logic

The application uses a scoring algorithm with the following priorities (from highest to lowest):

1. Minimize personnel remaining on the same team
2. Ensure even distribution of ranks
3. Balance total points (high-performing officers with low-performing enlisted and vice versa)
4. Even distribution of time in service
5. Balanced distribution of ranger-qualified personnel

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

*Note: This application is designed for military team management but can be adapted for other team reshuffling needs with appropriate data modifications.*