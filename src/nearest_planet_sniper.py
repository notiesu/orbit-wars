from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
import math


class Logger:
    def __init__(self):
        self.timestep = 0
        self.fleets_sent = 0
        self.planets_gained = 0
        self.actions = []
        self.hit_rate_data = []
        self.timestep_data = []
    
    def log_action(self, action):
        """Log a move action: [planet_id, angle, ships]"""
        self.actions.append(action)
        self.fleets_sent += 1
    
    def log_planets_gained(self, count):
        """Log planets conquered this timestep"""
        self.planets_gained += count
    
    def log_timestep(self, my_planets_count, target_planets_count):
        """Log stats for current timestep"""
        self.timestep += 1
        hit_rate = self.fleets_sent / max(1, self.timestep) if self.fleets_sent > 0 else 0
        self.hit_rate_data.append(hit_rate)
        self.timestep_data.append({
            'timestep': self.timestep,
            'fleets_sent': self.fleets_sent,
            'planets_gained': self.planets_gained,
            'my_planets': my_planets_count,
            'target_planets': target_planets_count,
            'hit_rate': hit_rate
        })
    
    def get_stats(self):
        """Return summary statistics"""
        return {
            'total_timesteps': self.timestep,
            'total_fleets_sent': self.fleets_sent,
            'total_planets_gained': self.planets_gained,
            'total_actions': len(self.actions),
            'avg_hit_rate': sum(self.hit_rate_data) / len(self.hit_rate_data) if self.hit_rate_data else 0,
            'all_actions': self.actions,
            'timestep_history': self.timestep_data
        }
    
    def reset(self):
        """Reset logger for new episode"""
        self.timestep = 0
        self.fleets_sent = 0
        self.planets_gained = 0
        self.actions = []
        self.hit_rate_data = []
        self.timestep_data = []


def nearest_planet_sniper(obs):
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]

    # Separate our planets from targets
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]

    if not targets:
        return moves

    for mine in my_planets:
        # Find the nearest planet we don't own
        nearest = None
        min_dist = float('inf')
        for t in targets:
            dist = math.sqrt((mine.x - t.x)**2 + (mine.y - t.y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest = t

        if nearest is None:
            continue

        # How many ships do we need? Target's garrison + 1
        ships_needed = max(nearest.ships + 1, 20)

        # Only send if we have enough
        if mine.ships >= ships_needed:
            # Calculate angle from our planet to the target
            angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            moves.append([mine.id, angle, ships_needed])

    return moves