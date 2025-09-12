import pandas as pd
import numpy as np
from datetime import timedelta

class TennisFeatureEngineer:
    def __init__(self, data_processor):
        self.processor = data_processor
        self.elo_cache = {}
        self.surface_elo_cache = {}
        self.form_cache = {}
        self.h2h_cache = {}
        self.h2h_records = None
    
    def create_match_features(self, match_date, playerA, playerB, surface):
        """Create all features for a specific match prediction"""
        features = {}
        players = sorted([playerA, playerB])
        player1, player2 = players[0], players[1]

        # ELO features
        current_ratings = self._get_elo_ratings_at_date(match_date)
        features['player1_elo'] = current_ratings.get(player1, 1500)
        features['player2_elo'] = current_ratings.get(player2, 1500)
        features['elo_diff'] = features['player1_elo'] - features['player2_elo']

        # Surface specialization
        surface_ratings = self._get_surface_elo_at_date(match_date, surface)
        player1_surface_elo = surface_ratings.get(player1, 1500)
        player2_surface_elo = surface_ratings.get(player2, 1500)

        features['player1_surface_specialization'] = self._get_surface_specialization_at_date(match_date, surface, player1, player1_surface_elo, features['player1_elo'])
        features['player2_surface_specialization'] = self._get_surface_specialization_at_date(match_date, surface, player2, player2_surface_elo, features['player2_elo'])
        features['surface_specialization_diff'] = features['player1_surface_specialization'] - features['player2_surface_specialization']

        # Recent form (90-day window)
        form_data = self._get_form_at_date(match_date, [player1, player2])
        features['player1_recent_form'] = form_data.get(player1, 0.5)
        features['player2_recent_form'] = form_data.get(player2, 0.5)
        features['form_diff'] = features['player1_recent_form'] - features['player2_recent_form']

        # Head-to-head (overall and surface-specific)
        h2h_overall, h2h_surface = self._get_h2h_at_date(match_date, player1, player2, surface)
        features['h2h_overall_advantage'] = h2h_overall
        features['h2h_surface_advantage'] = h2h_surface

        # Surface encoding
        features['surface_hard'] = 1 if surface == 'Hard' else 0
        features['surface_clay'] = 1 if surface == 'Clay' else 0
        features['surface_grass'] = 1 if surface == 'Grass' else 0

        return features
    
    def _get_elo_ratings_at_date(self, match_date):
        """Get ELO ratings as they were on a specific date"""
        if match_date in self.elo_cache:
            return self.elo_cache[match_date]
        
        # Calculate ELO ratings up to this date
        historical_matches = self.processor.matches_df[
            self.processor.matches_df['tourney_date'] < match_date
        ].sort_values('tourney_date')
        
        player_ratings = {}

        for idx, match in historical_matches.iterrows():
            winner = match['winner_name']
            loser = match['loser_name']
            
            # Initialize ratings if new players
            if winner not in player_ratings:
                player_ratings[winner] = 1500
            if loser not in player_ratings:
                player_ratings[loser] = 1500
            
            winner_rating = player_ratings[winner]
            loser_rating = player_ratings[loser]

            # Calculate expected scores
            winner_expected = 1 / (1 + 10**((loser_rating - winner_rating) / 400))
            loser_expected = 1 - winner_expected

            # Update ratings
            k_factor = 32
            player_ratings[winner] += k_factor * (1 - winner_expected)
            player_ratings[loser] += k_factor * (0 - loser_expected)
        
        self.elo_cache[match_date] = player_ratings
        return player_ratings
    
    def _get_surface_elo_at_date(self, match_date, surface):
        """Get surface-specific ELO ratings as they were on a specific date"""
        cache_key = (match_date, surface)
        if cache_key in self.surface_elo_cache:
            return self.surface_elo_cache[cache_key]
        
        # Calculate surface-specific ELO ratings up to this date
        surface_matches = self.processor.matches_df[
            (self.processor.matches_df['tourney_date'] < match_date) &
            (self.processor.matches_df['surface'] == surface)
        ].sort_values('tourney_date')

        player_ratings = {}

        for idx, match in surface_matches.iterrows():
            winner = match['winner_name']
            loser = match['loser_name']
            
            # Initialize ratings if new players
            if winner not in player_ratings:
                player_ratings[winner] = 1500
            if loser not in player_ratings:
                player_ratings[loser] = 1500
            
            winner_rating = player_ratings[winner]
            loser_rating = player_ratings[loser]

            # Calculate expected scores
            winner_expected = 1 / (1 + 10**((loser_rating - winner_rating) / 400))
            loser_expected = 1 - winner_expected

            # Update ratings
            k_factor = 32
            player_ratings[winner] += k_factor * (1 - winner_expected)
            player_ratings[loser] += k_factor * (0 - loser_expected)
        
        self.surface_elo_cache[cache_key] = player_ratings
        return player_ratings
    
    def _get_surface_specialization_at_date(self, match_date, surface, player, surface_elo, overall_elo):
        """Get player confidence_weighted surface specialization as of a specific date"""

        surface_matches = self.processor.matches_df[
            ((self.processor.matches_df['tourney_date'] < match_date) &
            (self.processor.matches_df['surface'] == surface) &
            ((self.processor.matches_df['winner_name'] == player) |
             (self.processor.matches_df['loser_name'] == player)))
        ]
        
        confidence = len(surface_matches) / (len(surface_matches) + 15)
        surface_specialization = confidence * (surface_elo - overall_elo)

        return surface_specialization
    
    def _get_form_at_date(self, match_date, players, window_days=90):
        """Get recent form for players as of a specific date"""
        cache_key = (match_date, tuple(players))
        if cache_key in self.form_cache:
            return self.form_cache[cache_key]
        
        cutoff_date = match_date - timedelta(days=window_days)
        form_data = {}

        for player in players:
            # Get matches in the form window
            recent_matches = self.processor.matches_df[
                (self.processor.matches_df['tourney_date'] >= cutoff_date) &
                (self.processor.matches_df['tourney_date'] < match_date) &
                ((self.processor.matches_df['winner_name'] == player) | 
                 (self.processor.matches_df['loser_name'] == player))
            ]

            if len(recent_matches) > 0:
                weights = np.exp(-0.1 * (match_date - recent_matches['tourney_date']).dt.days)
                weighted_wins = np.sum(weights * (recent_matches['winner_name'] == player))
                weighted_total =  np.sum(weights)

                form_data[player] = weighted_wins / weighted_total if weighted_total > 0 else 0.5
            else:
                form_data[player] = 0.5
            
        self.form_cache[cache_key] = form_data
        return form_data
    
    def _get_h2h_records(self):
        """Get H2H records (cached)"""
        if self.h2h_records is None:
            df = self.processor.matches_df.copy().sort_values('tourney_date')

            h2h_records = {}

            for idx, match in df.iterrows():
                winner = match['winner_name']
                loser = match['loser_name']

                # Create consistent pairing key
                pair = tuple(sorted([winner, loser]))

                if pair not in h2h_records:
                    h2h_records[pair] = {
                        'player1': pair[0],
                        'player2': pair[1],
                        'player1_wins': 0,
                        'player2_wins': 0,
                        'total_matches': 0,
                        'surfaces': {},
                        'last_match_date': None
                    }
                
                # Update record
                h2h_records[pair]['total_matches'] += 1
                h2h_records[pair]['last_match_date'] = match['tourney_date']

                # Update wins
                if winner == pair[0]:
                    h2h_records[pair]['player1_wins'] += 1
                else:
                    h2h_records[pair]['player2_wins'] += 1
                
                # Update surface-specific records
                surface = match['surface']
                if surface not in h2h_records[pair]['surfaces']:
                    h2h_records[pair]['surfaces'][surface] = {
                        'player1_wins': 0, 'player2_wins': 0
                    }
            
                if winner == pair[0]:
                    h2h_records[pair]['surfaces'][surface]['player1_wins'] += 1
                else:
                    h2h_records[pair]['surfaces'][surface]['player2_wins'] += 1
            
            self.h2h_records = h2h_records
        
        return self.h2h_records
    
    def _get_h2h_at_date(self, match_date, player1, player2, surface):
        """Get head-to-head record as of a specific date"""
        cache_key = (match_date, (player1, player2), surface)
        if cache_key in self.h2h_cache:
            return self.h2h_cache[cache_key]
        
        # Get all H2H records
        all_h2h = self._get_h2h_records()
        pair = tuple(sorted([player1, player2]))

        if pair not in all_h2h:
            self.h2h_cache[cache_key] = (0.0, 0.0)
            return (0.0, 0.0)
        
        # Get historical matches between these players
        historical_h2h = self.processor.matches_df[
            (self.processor.matches_df['tourney_date'] < match_date) &
            (((self.processor.matches_df['winner_name'] == player1) & 
              (self.processor.matches_df['loser_name'] == player2)) |
             ((self.processor.matches_df['winner_name'] == player2) & 
              (self.processor.matches_df['loser_name'] == player1)))
        ]

        # Overall H2H
        if len(historical_h2h) > 0:
            player1_wins = len(historical_h2h[historical_h2h['winner_name'] == player1])
            total_matches = len(historical_h2h)
            h2h_overall = (player1_wins / total_matches) - 0.5  # Center around 0
        else:
            h2h_overall = 0.0  # No history

        # Surface-specific H2H
        surface_h2h = historical_h2h[historical_h2h['surface'] == surface]
        if len(surface_h2h) > 0:
            player1_surface_wins = len(surface_h2h[surface_h2h['winner_name'] == player1])
            total_surface_matches = len(surface_h2h)
            h2h_surface = (player1_surface_wins / total_surface_matches) - 0.5
        else:
            h2h_surface = 0.0  # No surface history
        
        result = (h2h_overall, h2h_surface)
        self.h2h_cache[cache_key] = result
        return result
    
    def create_training_dataset(self, start_date=None, end_date=None, min_matches_for_inclusion=5):
        """Create a complete training dataset with all features"""
        matches_df = self.processor.matches_df.copy()

        if start_date:
            matches_df = matches_df[matches_df['tourney_date'] >= start_date]
        if end_date:
            matches_df = matches_df[matches_df['tourney_date'] <= end_date]
        
        # Filter out players with too few matches for reliable features
        player_match_counts = pd.concat([
            matches_df['winner_name'], 
            matches_df['loser_name']
        ]).value_counts()

        frequent_players = set(player_match_counts[
            player_match_counts >= min_matches_for_inclusion
        ].index)

        matches_df = matches_df[
            matches_df['winner_name'].isin(frequent_players) &
            matches_df['loser_name'].isin(frequent_players)
        ]

        print(f"Creating training dataset from {len(matches_df)} matches")
        print(f"Filtered to {len(frequent_players)} players with ≥{min_matches_for_inclusion} matches")
        
        training_data = []

        for idx, match in matches_df.iterrows():
            
            # Create features for this match
            features = self.create_match_features(
                match['tourney_date'],
                match['winner_name'],
                match['loser_name'], 
                match['surface']
            )

            winner = match['winner_name']
            sorted_players = sorted([match['winner_name'], match['loser_name']])
            target = 1 if winner == sorted_players[0] else 0

            # Add match metadata
            features['match_date'] = match['tourney_date']
            features['player1'] = sorted_players[0] 
            features['player2'] = sorted_players[1]
            features['surface'] = match['surface']
            features['tourney_level'] = match['tourney_level']
            features['target'] = target  
            
            training_data.append(features)
        
        training_df = pd.DataFrame(training_data)
        print(f"Training dataset created: {len(training_df)} samples")
        
        return training_df
    


