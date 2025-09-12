import pandas as pd
import numpy as np

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
    
    # def calculate_elo_ratings(self, initial_rating=1500, k_factor=32):
    #     """Calculate ELO ratings for all players"""
    #     df = self.matches_df.copy().sort_values('tourney_date')

    #     # Initialize player ratings
    #     player_ratings = {}

    #     # Track ratings over time
    #     match_data = []

    #     for idx, match in df.iterrows():
    #         winner = match['winner_name']
    #         loser = match['loser_name']

    #         # Initialize ratings if players are new
    #         if winner not in player_ratings:
    #             player_ratings[winner] = initial_rating
    #         if loser not in player_ratings:
    #             player_ratings[loser] = initial_rating

    #         # Get current ratings
    #         winner_rating = player_ratings[winner]
    #         loser_rating = player_ratings[loser]

    #         # Calculate expected scores
    #         winner_expected = 1 / (1 + 10**((loser_rating - winner_rating) / 400))
    #         loser_expected = 1 - winner_expected

    #         # Update ratings
    #         player_ratings[winner] += k_factor * (1 - winner_expected)
    #         player_ratings[loser] += k_factor * (0 - loser_expected)

    #         # Store match with pre-match ratings
    #         match_data.append({
    #             'tourney_date': match['tourney_date'],
    #             'winner': winner,
    #             'loser': loser,
    #             'winner_elo_before': winner_rating,
    #             'loser_elo_before': loser_rating,
    #             'winner_elo_after': player_ratings[winner],
    #             'loser_elo_after': player_ratings[loser],
    #             'surface': match['surface'],
    #             'tourney_level': match['tourney_level']
    #         })
        
    #     return pd.DataFrame(match_data), player_ratings
    
    # def calculate_surface_elo(self, surfaces=['Hard', 'Clay', 'Grass']):
    #     """Calculate separate ELO ratings for each surface"""
    #     surface_ratings = {}
    #     surface_matches = {}

    #     print("=== CALCULATING SURFACE-SPECIFIC ELO RATINGS ===")

    #     for surface in surfaces:
    #         surface_data = self.matches_df[self.matches_df['surface'] == surface]
    #         print(f"\n{surface} court: {len(surface_data)} matches found")

    #         if len(surface_data) > 0:
    #             matches, ratings = self.calculate_elo_ratings_for_surface(surface_data, surface)
    #             surface_ratings[surface] = ratings
    #             surface_matches[surface] = matches
        
    #     return surface_ratings, surface_matches
    
    # def calculate_elo_ratings_for_surface(self, df, surface_name, initial_rating=1500, k_factor=32):
    #     """Calculate ELO ratings for a specific surface"""
    #     df_surface = df.copy().sort_values('tourney_date')

    #     # Initialize player ratings for this surface
    #     player_ratings = {}

    #     # Track ratings over time for this surface
    #     match_data = []

    #     for idx, match in df_surface.iterrows():
    #         winner = match['winner_name']
    #         loser = match['loser_name']

    #         # Initialize ratings if new players on this surface
    #         if winner not in player_ratings:
    #             player_ratings[winner] = initial_rating
    #         if loser not in player_ratings:
    #             player_ratings[loser] = initial_rating
            
    #         # Get current ratings for this surface
    #         winner_rating = player_ratings[winner]
    #         loser_rating = player_ratings[loser]

    #         # Calculate expected scores
    #         winner_expected = 1 / (1 + 10**((loser_rating - winner_rating) / 400))
    #         loser_expected = 1 - winner_expected

    #         # Update ratings for this surface
    #         player_ratings[winner] += k_factor * (1 - winner_expected)
    #         player_ratings[loser] += k_factor * (0 - loser_expected)

    #         # Store match with pre-match surface ratings
    #         match_data.append({
    #             'tourney_date': match['tourney_date'],
    #             'winner': winner,
    #             'loser': loser,
    #             'surface': surface_name,
    #             'winner_surface_elo_before': winner_rating,
    #             'loser_surface_elo_before': loser_rating,
    #             'winner_surface_elo_after': player_ratings[winner],
    #             'loser_surface_elo_after': player_ratings[loser],
    #             'tourney_level': match['tourney_level']
    #         })

    #     return pd.DataFrame(match_data), player_ratings
    
    # def get_player_surface_comparison(self, player_name, surface_ratings):
    #     """Compare a player's relative performance across different surfaces"""
    #     player_surfaces = {}
    #     surface_stats = {}

    #     # Calculate surface-specific statistics
    #     for surface, ratings in surface_ratings.items():
    #         ratings_list = list(ratings.values())
    #         surface_stats[surface] = {
    #         'mean': np.mean(ratings_list),
    #         'std': np.std(ratings_list),
    #         'max': max(ratings_list),
    #         'min': min(ratings_list),
    #         'total_players': len(ratings_list)
    #     }
            
    #     # Calculate player's relative performance on each surface
    #     for surface, ratings in surface_ratings.items():
    #         if player_name in ratings:
    #             raw_rating = ratings[player_name]
                
    #             # Calculate percentile ranking (0-100)
    #             sorted_ratings = sorted(ratings.values(), reverse=True)
    #             rank = sorted_ratings.index(raw_rating) + 1
    #             percentile = (rank / len(sorted_ratings)) * 100
                
    #             # Calculate z-score 
    #             z_score = (raw_rating - surface_stats[surface]['mean']) / surface_stats[surface]['std']
                
    #             # Calculate relative strength (percentage of surface maximum)
    #             max_rating = surface_stats[surface]['max']
    #             min_rating = surface_stats[surface]['min']
    #             relative_strength = ((raw_rating - min_rating) / (max_rating - min_rating)) * 100
                
    #             player_surfaces[surface] = {
    #                 'raw_rating': raw_rating,
    #                 'rank': rank,
    #                 'percentile': percentile,
    #                 'z_score': z_score,
    #                 'relative_strength': relative_strength,
    #                 'surface_stats': surface_stats[surface]
    #             }
    #         else:
    #             player_surfaces[surface] = None # No match data for this player on this surface

    #     return player_surfaces

    # def analyze_player_surface_specialization(self, player_name, surface_ratings):
    #     """Analyze a player's surface specialization with proper normalization"""
    #     comparison = self.get_player_surface_comparison(player_name, surface_ratings)
        
    #     # Filter out surfaces where player hasn't played
    #     played_surfaces = {s: data for s, data in comparison.items() if data is not None}
        
    #     if len(played_surfaces) < 2:
    #         return "Insufficient data for surface comparison"
        
    #     print(f"\n=== {player_name.upper()} SURFACE ANALYSIS ===")
    #     print(f"{'Surface':<10} {'Raw ELO':<8} {'Rank':<6} {'Percentile':<10} {'Z-Score':<8} {'Rel.Strength':<12}")
    #     print("-" * 70)
        
    #     surface_performance = {}
        
    #     for surface, data in played_surfaces.items():
    #         print(f"{surface:<10} {data['raw_rating']:<8.0f} {data['rank']:<6} "
    #             f"{data['percentile']:<10.1f}% {data['z_score']:<8.2f} {data['relative_strength']:<12.1f}%")
            
    #         # Store for specialization analysis
    #         surface_performance[surface] = {
    #             'percentile': data['percentile'],
    #             'z_score': data['z_score'],
    #             'rank': data['rank']
    #         }
        
    #     # Determine true specialization based on percentile rankings
    #     best_surface_by_percentile = max(surface_performance.items(), 
    #                                 key=lambda x: x[1]['percentile'])
    #     worst_surface_by_percentile = min(surface_performance.items(), 
    #                                     key=lambda x: x[1]['percentile'])
        
    #     print(f"\n=== SPECIALIZATION ANALYSIS ===")
    #     print(f"Best surface: {best_surface_by_percentile[0]} "
    #         f"({best_surface_by_percentile[1]['percentile']:.1f}th percentile, "
    #         f"rank #{best_surface_by_percentile[1]['rank']})")
    #     print(f"Worst surface: {worst_surface_by_percentile[0]} "
    #         f"({worst_surface_by_percentile[1]['percentile']:.1f}th percentile, "
    #         f"rank #{worst_surface_by_percentile[1]['rank']})")
        
    #     percentile_gap = (best_surface_by_percentile[1]['percentile'] - 
    #                     worst_surface_by_percentile[1]['percentile'])
        
    #     if percentile_gap > 20:
    #         print(f"STRONG SURFACE SPECIALIST: {percentile_gap:.1f} percentile point difference")
    #     elif percentile_gap > 10:
    #         print(f"MODERATE SURFACE PREFERENCE: {percentile_gap:.1f} percentile point difference")
    #     else:
    #         print(f"WELL-ROUNDED PLAYER: Only {percentile_gap:.1f} percentile point difference")
        
    #     return surface_performance
    
    # def calculate_recent_form(self, window_days=90):
    #     """Calculate recent form metrics for each player"""
    #     df = self.matches_df.copy().sort_values('tourney_date')

    #     form_data = []

    #     for idx, match in df.iterrows():
    #         match_date = match['tourney_date']
    #         winner = match['winner_name']
    #         loser = match['loser_name']

    #         # Calculate form for both players
    #         for player in [winner, loser]:
    #             # Get matches in the last window_days
    #             cutoff_date = match_date - pd.Timedelta(days=window_days)

    #             recent_matches = df[
    #                 (df['tourney_date'] >= cutoff_date) &
    #                 (df['tourney_date'] < match_date) &
    #                 ((df['winner_name'] == player) | (df['loser_name'] == player))     
    #             ]

    #             if len(recent_matches) > 0:
    #                 wins = len(recent_matches[recent_matches['winner_name'] == player])
    #                 total = len(recent_matches)
    #                 win_rate = wins / total

    #                 # Exponentially weighted win rate (more recent matches weighted higher)
    #                 weights = np.exp(-0.1 * (match_date - recent_matches['tourney_date']).dt.days)
    #                 weighted_wins = np.sum(weights * (recent_matches['winner_name'] == player))
    #                 weighted_total =  np.sum(weights)
    #                 weighted_win_rate = weighted_wins / weighted_total
    #             else:
    #                 win_rate = 0.5 # Default for no recent matches
    #                 weighted_win_rate = 0.5
                
    #             form_data.append({
    #                 'date': match_date,
    #                 'player': player,
    #                 'recent_matches': len(recent_matches),
    #                 'win_rate': win_rate,
    #                 'weighted_win_rate': weighted_win_rate
    #             })
        
    #     return pd.DataFrame(form_data)
    
    # def calculate_head_to_head(self):
    #     """Calculate head-to-head records between players"""
    #     df = self.matches_df.copy().sort_values('tourney_date')

    #     h2h_records = {}

    #     for idx, match in df.iterrows():
    #         winner = match['winner_name']
    #         loser = match['loser_name']

    #         # Create consistent pairing key
    #         pair = tuple(sorted([winner, loser]))

    #         if pair not in h2h_records:
    #             h2h_records[pair] = {
    #                 'player1': pair[0],
    #                 'player2': pair[1],
    #                 'player1_wins': 0,
    #                 'player2_wins': 0,
    #                 'total_matches': 0,
    #                 'surfaces': {},
    #                 'last_match_date': None
    #             }
            
    #         # Update record
    #         h2h_records[pair]['total_matches'] += 1
    #         h2h_records[pair]['last_match_date'] = match['tourney_date']

    #         # Update wins
    #         if winner == pair[0]:
    #             h2h_records[pair]['player1_wins'] += 1
    #         else:
    #             h2h_records[pair]['player2_wins'] += 1
            
    #         # Update surface-specific records
    #         surface = match['surface']
    #         if surface not in h2h_records[pair]['surfaces']:
    #             h2h_records[pair]['surfaces'][surface] = {
    #                 'player1_wins': 0, 'player2_wins': 0
    #             }
        
    #         if winner == pair[0]:
    #             h2h_records[pair]['surfaces'][surface]['player1_wins'] += 1
    #         else:
    #             h2h_records[pair]['surfaces'][surface]['player2_wins'] += 1
    
    #     return h2h_records





