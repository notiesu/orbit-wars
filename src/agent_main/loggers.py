


import csv
import math
from dataclasses import dataclass
from typing import Any, Dict, List

from simple_agent import FleetState, PlanetState, _parse_fleets, _parse_planets, detect_hits

# ----------------------------
# Logging and Statistics
# ----------------------------

@dataclass
class ActionLog:
	"""Records details of a single action (fleet launch)."""
	step: int
	source_id: int
	target_id: int
	ships_sent: int
	angle: float
	score: float
	roi: float
	ships_needed: int
	pressure: float
	distance: float
	target_production: int
	target_owner: int


@dataclass
class OutcomeLog:
	"""Records the outcome of an action."""
	action_index: int
	step_resolved: int
	success: bool  # Did we acquire the target?
	surviving_ships: int
	target_final_owner: int


@dataclass
class HitLog:
	"""Records a fleet hitting a planet (regardless of capture outcome)."""
	step: int
	fleet_id: int
	fleet_owner: int
	planet_id: int
	planet_owner: int
	fleet_ships: int
	old_distance_to_planet: float  # Distance before step
	new_distance_to_planet: float  # Distance after step (0 if hit)
	hit_occurred: bool


class AgentLogger:
	"""Comprehensive logging for agent decisions and outcomes."""

	def __init__(self):
		self.actions: List[ActionLog] = []
		self.outcomes: Dict[int, OutcomeLog] = {}  # action_index -> outcome
		self.state_history: List[Dict[str, Any]] = []  # Track state over time
		self.hits: List[HitLog] = []  # Track fleet-planet hits
		self.action_counter = 0

	def log_action(self, step: int, source: PlanetState, target: PlanetState,
				   ships_sent: int, angle: float, score: float, roi: float,
				   ships_needed: int, pressure: float, distance: float) -> int:
		"""Log an action taken by the agent. Returns action_index for later outcome matching."""
		action = ActionLog(
			step=step,
			source_id=source.id,
			target_id=target.id,
			ships_sent=ships_sent,
			angle=angle,
			score=score,
			roi=roi,
			ships_needed=ships_needed,
			pressure=pressure,
			distance=distance,
			target_production=target.production,
			target_owner=target.owner
		)
		self.actions.append(action)
		action_index = self.action_counter
		self.action_counter += 1
		return action_index

	def log_outcome(self, action_index: int, step_resolved: int, success: bool,
					surviving_ships: int, target_final_owner: int):
		"""Log the outcome of a launched fleet."""
		outcome = OutcomeLog(
			action_index=action_index,
			step_resolved=step_resolved,
			success=success,
			surviving_ships=surviving_ships,
			target_final_owner=target_final_owner
		)
		self.outcomes[action_index] = outcome

	def log_hit(self, step: int, fleet_id: int, fleet_owner: int, planet_id: int,
			   planet_owner: int, fleet_ships: int, old_distance: float, hit_occurred: bool):
		"""Log a fleet-planet hit event."""
		hit = HitLog(
			step=step,
			fleet_id=fleet_id,
			fleet_owner=fleet_owner,
			planet_id=planet_id,
			planet_owner=planet_owner,
			fleet_ships=fleet_ships,
			old_distance_to_planet=old_distance,
			new_distance_to_planet=0.0 if hit_occurred else old_distance,
			hit_occurred=hit_occurred
		)
		self.hits.append(hit)

	def process_game_history(self, env_steps: List[Any]):
		"""
		Process environment game history to detect and log all fleet-planet hits.
		Call this after a game completes: logger.process_game_history(env.steps)
		"""
		for step_idx in range(1, len(env_steps)):
			step_before = env_steps[step_idx - 1]
			step_after = env_steps[step_idx]
			
			fleets_before = _parse_fleets(step_before[0].observation)
			fleets_after = _parse_fleets(step_after[0].observation)
			planets_before = _parse_planets(step_before[0].observation)
			planets_after = _parse_planets(step_after[0].observation)
			
			detect_hits(fleets_before, fleets_after, planets_before, planets_after, step_idx, logger=self)

	def log_state(self, step: int, planets: List[PlanetState], fleets: List[FleetState],
				  player: int, my_score: float, enemy_score: float, pressure: float):
		"""Log the current game state at a step."""
		self.state_history.append({
			'step': step,
			'player_score': my_score,
			'enemy_score': enemy_score,
			'pressure': pressure,
			'planets_owned': sum(1 for p in planets if p.owner == player),
			'total_ships': sum(p.ships for p in planets if p.owner == player),
			'fleets_in_transit': sum(1 for f in fleets if f.owner == player),
			'fleets_ships': sum(f.ships for f in fleets if f.owner == player)
		})

	def get_summary(self) -> Dict[str, Any]:
		"""Return summary statistics."""
		if not self.actions:
			return {
				'total_actions': 0,
				'successful_captures': 0,
				'hit_rate': 0.0,
				'avg_roi': 0.0,
				'avg_score': 0.0,
				'total_ships_sent': 0,
				'avg_ships_per_action': 0.0
			}

		# Calculate successful captures from state history
		successful_captures = 0
		if self.state_history:
			initial_planets = self.state_history[0].get('planets_owned', 0)
			max_planets = max(s.get('planets_owned', 0) for s in self.state_history)
			successful_captures = max_planets - initial_planets

		# Hit rate: actual hits detected / total actions sent
		total_actions = len(self.actions)
		hit_rate = len(self.hits) / total_actions if total_actions > 0 else 0.0

		return {
			'total_actions': total_actions,
			'successful_captures': successful_captures,
			'hit_rate': hit_rate,
			'avg_roi': sum(a.roi for a in self.actions) / len(self.actions) if self.actions else 0.0,
			'avg_score': sum(a.score for a in self.actions) / len(self.actions) if self.actions else 0.0,
			'total_ships_sent': sum(a.ships_sent for a in self.actions),
			'avg_ships_per_action': sum(a.ships_sent for a in self.actions) / len(self.actions) if self.actions else 0.0,
			'avg_pressure': sum(a.pressure for a in self.actions) / len(self.actions) if self.actions else 0.0,
			'avg_distance': sum(a.distance for a in self.actions) / len(self.actions) if self.actions else 0.0,
		}

	def to_csv(self, filepath: str):
		"""Export action logs to CSV."""
		
		with open(filepath, 'w', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=[
				'step', 'source_id', 'target_id', 'ships_sent', 'angle_rad', 'angle_deg',
				'score', 'roi', 'ships_needed', 'pressure', 'distance',
				'target_production', 'target_owner', 'outcome_success', 'outcome_surviving_ships'
			])
			writer.writeheader()
			for i, action in enumerate(self.actions):
				outcome = self.outcomes.get(i)
				writer.writerow({
					'step': action.step,
					'source_id': action.source_id,
					'target_id': action.target_id,
					'ships_sent': action.ships_sent,
					'angle_rad': f"{action.angle:.4f}",
					'angle_deg': f"{math.degrees(action.angle):.1f}",
					'score': f"{action.score:.4f}",
					'roi': f"{action.roi:.4f}",
					'ships_needed': action.ships_needed,
					'pressure': f"{action.pressure:.4f}",
					'distance': f"{action.distance:.4f}",
					'target_production': action.target_production,
					'target_owner': action.target_owner,
					'outcome_success': outcome.success if outcome else '',
					'outcome_surviving_ships': outcome.surviving_ships if outcome else ''
				})

	def to_hits_csv(self, filepath: str):
		"""Export hit logs to CSV."""
		if not self.hits:
			return
		with open(filepath, 'w', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=[
				'step', 'fleet_id', 'fleet_owner', 'planet_id', 'planet_owner',
				'fleet_ships', 'old_distance_to_planet', 'new_distance_to_planet', 'hit_occurred'
			])
			writer.writeheader()
			for hit in self.hits:
				writer.writerow({
					'step': hit.step,
					'fleet_id': hit.fleet_id,
					'fleet_owner': hit.fleet_owner,
					'planet_id': hit.planet_id,
					'planet_owner': hit.planet_owner,
					'fleet_ships': hit.fleet_ships,
					'old_distance_to_planet': f"{hit.old_distance_to_planet:.4f}",
					'new_distance_to_planet': f"{hit.new_distance_to_planet:.4f}",
					'hit_occurred': hit.hit_occurred
				})

	def to_state_csv(self, filepath: str):
		"""Export state history to CSV."""
		if not self.state_history:
			return
		with open(filepath, 'w', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=self.state_history[0].keys())
			writer.writeheader()
			writer.writerows(self.state_history)

	def print_summary(self):
		"""Print summary statistics to console."""
		summary = self.get_summary()
		print("\n" + "="*60)
		print("AGENT STATISTICS")
		print("="*60)
		print(f"Total Actions:        {summary['total_actions']}")
		print(f"Successful Captures:  {summary['successful_captures']}")
		print(f"Hit Rate:             {summary['hit_rate']:.1%}")
		print(f"Avg ROI:              {summary['avg_roi']:.4f}")
		print(f"Avg Score:            {summary['avg_score']:.4f}")
		print(f"Avg Pressure:         {summary['avg_pressure']:.4f}")
		print(f"Total Ships Sent:     {summary['total_ships_sent']}")
		print(f"Avg Ships/Action:     {summary['avg_ships_per_action']:.1f}")
		print(f"Avg Distance:         {summary['avg_distance']:.2f}")
		print("="*60 + "\n")
