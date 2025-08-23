import pandas as pd

class TennisDataProcessor:
    def __init__(self):
        self.matches_df = None
        self.rankings_df = None

    def load_matches_data(self, data_path="data/raw/"):
        """Load and combine all match files"""
        dfs = []
        for year in range(2020, 2025):
            try:
                df = pd.read_csv(f"{data_path}atp_matches_{year}.csv")
                dfs.append(df)
            except FileNotFoundError:
                print(f"No data found for {year}")

        if not dfs:
            raise ValueError("No match data files found! Please run data collection.")
    
        self.matches_df = pd.concat(dfs, ignore_index=True)
        print(f"Total matches loaded: {len(self.matches_df)}")
        return self.matches_df

    def clean_matches_data(self):
        """"Clean and standardize match data with intelligent missing value handling"""
        df = self.matches_df.copy()

        # Convert date to datetime
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')

        # Remove matches with missing critical data (red flags)
        critical_fields = ['winner_name', 'loser_name', 'surface']
        initial_count = len(df)
        df = df.dropna(subset=critical_fields)
        dropped_critical = initial_count - len(df)
        print(f"Dropped {dropped_critical} matches with missing critical data")

        # Standardize surface names
        surface_mapping = {
            'Hard': 'Hard',
            'Clay': 'Clay',
            'Grass': 'Grass',
            'Carpet': 'Hard'
        }
        df['surface'] = df['surface'].map(surface_mapping)

        # Handle missing rankings intelligently (normal in tennis)
        df['winner_rank_original'] = df['winner_rank']  # Keep original for analysis
        df['loser_rank_original'] = df['loser_rank']
        
        df['winner_rank'] = df['winner_rank'].fillna(999)  # Unranked players
        df['loser_rank'] = df['loser_rank'].fillna(999)

        # Create additional features from missing data patterns
        df['winner_is_ranked'] = df['winner_rank_original'].notna()
        df['loser_is_ranked'] = df['loser_rank_original'].notna()
        df['both_ranked'] = df['winner_is_ranked'] & df['loser_is_ranked']

        # Document data quality for portfolio
        quality_metrics = {
            'original_matches': initial_count,
            'final_matches': len(df),
            'retention_rate': len(df) / initial_count * 100,
            'unranked_winners': (~df['winner_is_ranked']).sum(),
            'unranked_losers': (~df['loser_is_ranked']).sum()
        }

        print("=== DATA CLEANING SUMMARY ===")
        for metric, value in quality_metrics.items():
            if 'rate' in metric:
                print(f"{metric}: {value:.1f}%")
            else:
                print(f"{metric}: {value}")
        
        self.matches_df = df
        self.quality_metrics = quality_metrics  # Store for later reference
        return df
