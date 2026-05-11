"""Heuristic Orbit Wars agent.

The agent scores candidate target planets by expected return on investment and
only launches fleets when the projected gain is worth the ship cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
import csv

from mechanics import BOARD_SIZE, SUN_POS, SUN_RADIUS, compute_fleet_speed, dist, line_intersects_circle, calculate_score, calculate_interception_angle, is_inner_planet

@dataclass(frozen=True)
class PlanetState:
	id: int
	owner: int
	x: float
	y: float
	radius: float
	ships: int
	production: int


@dataclass(frozen=True)
class FleetState:
	id: int
	owner: int
	x: float
	y: float
	angle: float
	from_planet_id: int
	ships: int


def _obs_value(obs: Any, key: str, default: Any = None) -> Any:
	if isinstance(obs, dict):
		return obs.get(key, default)
	return getattr(obs, key, default)


def _parse_planets(obs: Any) -> List[PlanetState]:
	raw_planets = _obs_value(obs, "planets", []) or []
	return [PlanetState(*p) for p in raw_planets]


def _parse_fleets(obs: Any) -> List[FleetState]:
	raw_fleets = _obs_value(obs, "fleets", []) or []
	return [FleetState(*f) for f in raw_fleets]


def detect_hits(fleets_before: List[FleetState], fleets_after: List[FleetState],
				planets_before: List[PlanetState], planets_after: List[PlanetState],
				step: int, logger: 'AgentLogger | None' = None) -> List[HitLog]:
	"""
	Detect fleet-planet hits by comparing before/after fleet positions.
	A fleet that existed before but is gone after likely hit a planet.
	Returns list of HitLog entries and optionally logs them to the provided logger.
	"""
	hits: List[HitLog] = []
	
	# Build lookup by fleet id
	fleets_after_by_id = {f.id: f for f in fleets_after}
	
	# Find fleets that disappeared (likely hit)
	for fleet_before in fleets_before:
		if fleet_before.id not in fleets_after_by_id:
			# Fleet is gone—find which planet it likely hit (closest one)
			closest_planet = None
			closest_distance = float('inf')
			
			for planet in planets_before:
				distance = dist((fleet_before.x, fleet_before.y), (planet.x, planet.y))
				if distance < closest_distance:
					closest_distance = distance
					closest_planet = planet
			
			if closest_planet:
				hit = HitLog(
					step=step,
					fleet_id=fleet_before.id,
					fleet_owner=fleet_before.owner,
					planet_id=closest_planet.id,
					planet_owner=closest_planet.owner,
					fleet_ships=fleet_before.ships,
					old_distance_to_planet=closest_distance,
					new_distance_to_planet=0.0,
					hit_occurred=True
				)
				hits.append(hit)
				if logger:
					logger.log_hit(step, fleet_before.id, fleet_before.owner, closest_planet.id,
								   closest_planet.owner, fleet_before.ships, closest_distance, True)
	
	return hits


# def _get_angular_velocity(obs: Any, planet_id: int | None = None, default: float = 0.0) -> float:
# 	"""Parse angular velocity from observation, optionally by planet index."""
# 	raw_velocity = _obs_value(obs, "angular_velocity", default)
# 	if isinstance(raw_velocity, (list, tuple)):
# 		if planet_id is None:
# 			return raw_velocity[0] if raw_velocity else default
# 		if 0 <= planet_id < len(raw_velocity):
# 			return raw_velocity[planet_id] or 0.0
# 		return default
# 	return raw_velocity or default


def _predict_planet_position(planet: PlanetState, travel_turns: float, angular_velocity: float) -> Tuple[float, float]:
	"""Predict where a planet will be after travel_turns, accounting for orbital rotation."""
	if angular_velocity == 0:
		return planet.x, planet.y

	# Get current angle and orbital radius
	cx, cy = SUN_POS
	r = dist((planet.x, planet.y), SUN_POS)
	current_angle = math.atan2(planet.y - cy, planet.x - cx)

	# Predict future angle
	future_angle = current_angle + (angular_velocity * travel_turns)

	# Calculate future position
	future_x = cx + r * math.cos(future_angle)
	future_y = cy + r * math.sin(future_angle)

	return future_x, future_y


def _distance_to_planet(a: PlanetState, b: PlanetState) -> float:
	return dist((a.x, a.y), (b.x, b.y))


def _distance_to_planet_predicted(source: PlanetState, target: PlanetState, travel_turns: float, angular_velocity: float) -> float:
	"""Calculate distance from source to target's predicted future position."""
	future_x, future_y = _predict_planet_position(target, travel_turns, angular_velocity)
	return dist((source.x, source.y), (future_x, future_y))


def _is_path_safe(source: PlanetState, target: PlanetState, planets: Sequence[PlanetState], travel_turns: float = 0.0, angular_velocity: float = 0.0, source_angular_velocity: float = 0.0, launch_delay: float = 1.0) -> bool:
	"""Reject obvious bad shots that cross the sun or another planet, accounting for orbital motion."""

	start = _predict_planet_position(source, launch_delay, source_angular_velocity) if source_angular_velocity != 0 else (source.x, source.y)
	# Aim at target's predicted position if we have travel time info
	if travel_turns > 0 and angular_velocity != 0:
		end = _predict_planet_position(target, launch_delay + travel_turns, angular_velocity)
	else:
		end = _predict_planet_position(target, launch_delay, angular_velocity) if angular_velocity != 0 else (target.x, target.y)

	if line_intersects_circle(start, end, SUN_POS, SUN_RADIUS):
		return False

	for p in planets:
		if p.id in (source.id, target.id):
			continue
		if line_intersects_circle(start, end, (p.x, p.y), p.radius):
			return False

	return True


def _defense_reserve(source: PlanetState, fleets: Sequence[FleetState], player: int) -> int:
	"""Keep a small garrison back unless the planet is very safe."""

	reserve = max(8, source.production * 4)

	enemy_fleets = [f for f in fleets if f.owner != player]
	if enemy_fleets:
		nearest_enemy = min(dist((source.x, source.y), (f.x, f.y)) for f in enemy_fleets)
		if nearest_enemy < 18.0:
			reserve += int((18.0 - nearest_enemy) * 1.5)

	return min(reserve, max(0, source.ships - 1))


def _estimate_required_ships(source: PlanetState, target: PlanetState, remaining_turns: int, angular_velocity: float = 0.0) -> Tuple[int, float]:
	# First pass: estimate using current distance
	distance = _distance_to_planet(source, target)
	guessed_speed = compute_fleet_speed(max(1, source.ships // 3))
	travel_turns = distance / max(1.0, guessed_speed)

	# Second pass: refine with orbital prediction if applicable
	if angular_velocity != 0:
		predicted_distance = _distance_to_planet_predicted(source, target, travel_turns, angular_velocity)
		travel_turns = predicted_distance / max(1.0, guessed_speed)

	# Account for the target continuing to produce while the fleet is in transit.
	future_garrison = target.ships + target.production * max(1, int(math.ceil(travel_turns)))

	# Keep a tiny cushion so we do not bounce off by one ship on capture.
	ships_needed = future_garrison + 1

	# If the fleet arrives very late, the planet is less valuable.
	effective_turns = max(0.0, remaining_turns - travel_turns)
	return ships_needed, effective_turns


def _opponent_contest_penalty(
	source: PlanetState,
	target: PlanetState,
	remaining_turns: int,
	ships_needed: int,
	planets: Sequence[PlanetState] | None,
	fleets: Sequence[FleetState] | None,
	player: int | None,
	angular_velocity: float = 0.0,
) -> float:
	"""Estimate how likely opponents can outbid this capture and punish crowded races."""
	if planets is None or player is None:
		return 0.0

	opponents = [p for p in planets if p.owner not in (-1, player)]
	if not opponents:
		return 0.0

	best_enemy_arrival = float("inf")
	best_enemy_capacity = 0.0
	best_enemy_roi = 0.0

	for enemy_source in opponents:
		ed = _distance_to_planet(enemy_source, target)
		enemy_speed = compute_fleet_speed(max(1, enemy_source.ships // 3))
		enemy_travel_turns = ed / max(1.0, enemy_speed)

		enemy_required, enemy_effective_turns = _estimate_required_ships(enemy_source, target, remaining_turns, angular_velocity)
		enemy_available = max(0.0, enemy_source.ships - _defense_reserve(enemy_source, fleets or [], enemy_source.owner))

		# Enemy attractiveness proxy (same components as our core value model).
		enemy_production_value = target.production * enemy_effective_turns
		if target.owner != -1:
			enemy_production_value *= 1.35
		enemy_size_value = target.radius * 5.0
		enemy_ship_value = 0.0 if target.owner == -1 else target.ships * 0.5 + 10.0
		enemy_value = enemy_production_value + enemy_size_value + enemy_ship_value
		enemy_roi = enemy_value / max(1.0, enemy_required)

		if enemy_travel_turns < best_enemy_arrival:
			best_enemy_arrival = enemy_travel_turns
			best_enemy_capacity = enemy_available

		if enemy_roi > best_enemy_roi:
			best_enemy_roi = enemy_roi

	if best_enemy_arrival == float("inf"):
		return 0.0

	# Use predicted distance for our arrival calculation
	our_distance = _distance_to_planet(source, target)
	our_speed = compute_fleet_speed(max(1, source.ships // 3))
	our_arrival = our_distance / max(1.0, our_speed)
	if angular_velocity != 0:
		predicted_our_distance = _distance_to_planet_predicted(source, target, our_arrival, angular_velocity)
		our_arrival = predicted_our_distance / max(1.0, our_speed)

	arrival_advantage = (our_arrival - best_enemy_arrival) / max(1.0, our_arrival)
	capacity_gap = (ships_needed - best_enemy_capacity) / max(1.0, ships_needed)

	# Penalize if enemy can get there earlier, has enough ships, and target ROI is good for them.
	enemy_interest = min(3.0, best_enemy_roi)
	penalty = max(0.0, arrival_advantage) * 1.8 + max(0.0, -capacity_gap) * 1.2
	penalty *= 0.6 + 0.4 * enemy_interest

	return penalty


def _score_target(
	source: PlanetState,
	target: PlanetState,
	remaining_turns: int,
	planets: Sequence[PlanetState] | None = None,
	fleets: Sequence[FleetState] | None = None,
	player: int | None = None,
	angular_velocity: float = 0.0,
) -> Tuple[float, int, float]:
	ships_needed, effective_turns = _estimate_required_ships(source, target, remaining_turns, angular_velocity)
	if ships_needed <= 0:
		return float("-inf"), 0, 0.0

	distance = _distance_to_planet(source, target)
	# Refine with predicted distance if orbital motion exists
	if angular_velocity != 0:
		guessed_speed = compute_fleet_speed(max(1, source.ships // 3))
		travel_turns = distance / max(1.0, guessed_speed)
		distance = _distance_to_planet_predicted(source, target, travel_turns, angular_velocity)

	# Production matters most early; production on enemy planets is even more valuable.
	production_value = target.production * effective_turns
	if target.owner != -1:
		production_value *= 1.35

	# Bigger planets are slightly more valuable because they are usually more durable.
	size_value = target.radius * 5.0

	# Taking ships off an enemy planet is worth more than taking a neutral one.
	ship_value = 0.0 if target.owner == -1 else target.ships * 0.5 + 10.0

	value = production_value + size_value + ship_value
	risk_penalty = 0.18 * distance + 0.06 * ships_needed
	risk_penalty += _opponent_contest_penalty(source, target, remaining_turns, ships_needed, planets, fleets, player, angular_velocity)

	roi = value / max(1.0, ships_needed)
	score = roi - risk_penalty / max(1.0, ships_needed)

	return score, ships_needed, roi


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


class SimpleAgent: 

	def __init__(self, roi_threshold: int):
		self.roi_threshold = roi_threshold

	def predict(self, obs: Any):
		planets = _parse_planets(obs)
		fleets = _parse_fleets(obs)
		player = _obs_value(obs, "player", 0)
		step = _obs_value(obs, "step", 0) or 0
		remaining_turns = max(1, 500 - int(step))

		my_planets = [p for p in planets if p.owner == player]
		targets = [p for p in planets if p.owner != player]

		if not my_planets or not targets:
			return []

		actions: List[Tuple[float, List[int]]] = []

		for source in my_planets:
			source_angular_velocity = self._estimate_source_angular_velocity(source)

			#THIS RESULTS IN MORE ACCURATE REUSLTS
			launch_delay = 0.0
			launch_source_pos = self._launch_position(source, source_angular_velocity, launch_delay)
			reserve = _defense_reserve(source, fleets, player)
			available = source.ships - reserve
			if available <= 1:
				continue

			best_action = None
			best_score = float("-inf")

			for target in targets:
				if target.id == source.id:
					continue

				target_angular_velocity = 0.0

				score, ships_needed, roi = _score_target(source, target, remaining_turns, planets, fleets, player, target_angular_velocity)
				if ships_needed > available:
					continue

				# Calculate travel time for path safety check and angle calculation
				guessed_speed = compute_fleet_speed(max(1, source.ships // 3))
				dist_to_target = dist(launch_source_pos, (target.x, target.y))
				travel_turns = dist_to_target / max(1.0, guessed_speed)
				if not _is_path_safe(source, target, planets, travel_turns, target_angular_velocity, source_angular_velocity, launch_delay):
					continue

				# High-quality captures should be materially positive in ROI terms.
				threshold = 1.10 if target.owner == -1 else 1.25
				if roi < self.roi_threshold or score <= 0:
					continue

				# Aim at the predicted future position of the target
				future_x, future_y = _predict_planet_position(target, launch_delay + travel_turns, target_angular_velocity)
				angle = math.atan2(future_y - launch_source_pos[1], future_x - launch_source_pos[0])
				action = [source.id, angle, ships_needed]

				if score > best_score:
					best_score = score
					best_action = action

			if best_action is not None:
				actions.append((best_score, best_action))

		actions.sort(key=lambda item: item[0], reverse=True)

		# Do not over-commit; keep the strongest opportunities only.
		return [action for _, action in actions[:4]]

import math
from typing import Any, List, Tuple


class SimpleAgent2:

	def __init__(self,
				 initial_roi_threshold: float = 0.5,
				 ending_roi_threshold: float = 1.45):

		self.initial_roi_threshold = initial_roi_threshold
		self.ending_roi_threshold = ending_roi_threshold
		self.current_roi_threshold = initial_roi_threshold
		self.steps = 0
		self.logger = AgentLogger()
		self.last_planet_positions: Dict[int, Tuple[float, float]] = {}
		self.recent_targets: Dict[int, int] = {}  # target_id -> step_when_available_again

	# -------------------------
	# time-based ROI schedule
	# -------------------------

	def early_game_score(self, source: PlanetState, target: PlanetState) -> float:
		# Early game heuristic: prioritize high production and close planets.
		distance = _distance_to_planet(source, target)
		production_value = target.production * 10.0
		size_value = target.radius * 5.0
		return production_value + size_value - distance

	def roi_threshold_time(self, progress):
		k = 8
		s = 1 / (1 + math.exp(-k * (progress - 0.5)))

		return self.initial_roi_threshold + (
			self.ending_roi_threshold - self.initial_roi_threshold
		) * s

	# -------------------------
	# state-based pressure (losing -> aggressive)
	# -------------------------

	def pressure(self, my_score, enemy_score):

		try:
			adv = float(my_score) - float(enemy_score)

			# normalize gently (avoid division explosion)
			denom = max(50.0, abs(float(enemy_score)) + 1e-6)
			x = adv / denom

			# keep sigmoid input in a safe range
			x = max(-60.0, min(60.0, x))

			value = 1.0 / (1.0 + math.exp(-x))
			return min(max(value, 0.0), 0.8)
		except Exception:
			return 0.0

	def mark_target_sent(self, target_id: int, current_step: int, travel_turns: float):
		"""Mark a target as recently sent to; avoid resending for travel_turns steps."""
		steps_until_arrival = int(math.ceil(travel_turns))
		self.recent_targets[target_id] = current_step + steps_until_arrival

	def is_target_recently_sent(self, target_id: int, current_step: int) -> bool:
		"""Check if a target was recently sent to and is still in cooldown."""
		return target_id in self.recent_targets and self.recent_targets[target_id] > current_step

	def cleanup_recent_targets(self, current_step: int):
		"""Remove expired entries from recent targets memory."""
		self.recent_targets = {
			target_id: step for target_id, step in self.recent_targets.items()
			if step > current_step
		}

	def _estimate_source_angular_velocity(self, planet: PlanetState) -> float:
		"""Estimate the source planet's angular velocity from its previous position."""
		previous = self.last_planet_positions.get(planet.id)
		if previous is None:
			return 0.0

		previous_angle = math.atan2(previous[1] - SUN_POS[1], previous[0] - SUN_POS[0])
		current_angle = math.atan2(planet.y - SUN_POS[1], planet.x - SUN_POS[0])
		delta = current_angle - previous_angle
		while delta > math.pi:
			delta -= 2 * math.pi
		while delta < -math.pi:
			delta += 2 * math.pi
		return delta

	def _launch_position(self, planet: PlanetState, angular_velocity: float, launch_delay: float = 1.0) -> Tuple[float, float]:
		"""Predict the position at the actual launch timestep."""
		if angular_velocity == 0.0:
			return planet.x, planet.y
		return _predict_planet_position(planet, launch_delay, angular_velocity)
	


    # -------------------------
    # simple state evaluator
    # -------------------------

	def state_score(self, obs, player, planets, fleets):
		score = 0.0

		for p in planets:
			if p.owner == player:
				score += p.ships

		for f in fleets:
			if f.owner == player:
				score += f.ships

		return score

	# def prune_actions(self, actions: List[Tuple[float, List[int]]], max_actions: int = 4) -> List[List[int]]:
	# 	#find the best combintination of actions that fulfill resource constraints of each planet
	# 	pruned_actions = []


	# -------------------------
	# main policy
	# -------------------------
	def predict(self, obs: Any):

		planets = _parse_planets(obs)
		fleets = _parse_fleets(obs)
		comets = _obs_value(obs, "comets", {}) or {}
		comet_ids = {id for id in comets[0]["planet_ids"]} if comets else set()
		player = _obs_value(obs, "player", 0)
		step = _obs_value(obs, "step", 0) or 0
		remaining_turns = max(1, 500 - int(step))

		#inner planet angular velocity - single int
		inner_planet_angular_velocity = _obs_value(obs, 'angular_velocity', [])

		
		my_planets = [p for p in planets if p.owner == player]
		enemy_planets = [p for p in planets if p.owner != player]

		if not my_planets or not enemy_planets:
			return []

		# -------------------------
		# compute state advantage
		# -------------------------
		self.steps += 1
		my_score = self.state_score(obs, player, planets, fleets) if self.steps > 50 else self.early_game_score(my_planets[0], enemy_planets[0])
		enemy_score = sum(
			p.ships for p in planets if p.owner != player
		)

		pressure = self.pressure(my_score, enemy_score)

		# -------------------------
		# update ROI threshold (time + state override)
		# -------------------------
		progress = step / 500.0
		base_roi = self.roi_threshold_time(progress)

		self.current_roi_threshold = base_roi * (1 - 0.35 * pressure)

		# -------------------------
		# generate actions
		# -------------------------
		actions: List[Tuple[float, List[int]]] = []

		for source in my_planets:
			source_angular_velocity = inner_planet_angular_velocity if is_inner_planet((source.x, source.y), source.radius) else 0.0
			launch_source_pos = self._launch_position(source, source_angular_velocity, 1.0)

			reserve = _defense_reserve(source, fleets, player)
			available = source.ships - reserve

			if available <= 1:
				continue

			for target in enemy_planets:
				#TODO - ACCOUNT FOR COMETS
				if target.id in comet_ids:
					continue
				target_angular_velocity = inner_planet_angular_velocity if is_inner_planet((target.x, target.y), target.radius) else 0.0

				# Skip if we recently sent a fleet to this target
				if self.is_target_recently_sent(target.id, step):
					continue
				
				# Calculate travel time for path safety check and angle calculation
				guessed_speed = compute_fleet_speed(max(1, source.ships // 3))
				dist_to_target = dist(launch_source_pos, (target.x, target.y))
				travel_turns = dist_to_target / max(1.0, guessed_speed)
				if not _is_path_safe(source, target, planets, travel_turns, target_angular_velocity, source_angular_velocity, 1.0):
					continue

				score, ships_needed, roi = _score_target(
					source, target, remaining_turns, planets, fleets, player, target_angular_velocity
				)

				if ships_needed <= 0 or ships_needed > available:
					continue

				# -------------------------
				# ROI now is a signal, not a gate
				# -------------------------
				roi_signal = (roi - self.current_roi_threshold)

				# Use predicted distance for scoring
				if target_angular_velocity != 0:
					projected_target_pos = _predict_planet_position(target, 1.0 + travel_turns, target_angular_velocity)
					distance = dist(launch_source_pos, projected_target_pos)
				else:
					distance = math.hypot(target.x - launch_source_pos[0], target.y - launch_source_pos[1])

				# -------------------------
				# final score (ALL components matter)
				# -------------------------
				final_score = (
					score
					+ 2.0 * roi_signal
					+ 0.4 * pressure * target.production
					- distance / 120.0
				)

				# -------------------------
				# aggression affects commitment
				# -------------------------
				commit = 1.1 + 0.5 * pressure
				ships_to_send = int(ships_needed * commit)
				ships_to_send = min(ships_to_send, available)

				# Calculate intercept angle using iterative prediction
				angle = calculate_interception_angle(
					source_pos=(source.x, source.y),
					target_pos=(target.x, target.y),
					target_angular_velocity=target_angular_velocity,
					ships=ships_to_send,
					source_angular_velocity=source_angular_velocity,
					launch_delay=1.0
				)

				#recalculate if this angle is safe, if not fallback to direct angle
				if angle is not None:
					intercept_x = target.x + target.radius * math.cos(angle)
					intercept_y = target.y + target.radius * math.sin(angle)
					intercept_distance = dist(launch_source_pos, (intercept_x, intercept_y))
					travel_turns = intercept_distance / max(1.0, guessed_speed)

					if not _is_path_safe(source, target, planets, travel_turns, target_angular_velocity, source_angular_velocity, 1.0):
						continue
				# #print the target planet for debugging
				# print(f"Step {step}: Evaluating target Planet {target.id} (owner={target.owner}, prod={target.production}, ships={target.ships}) from Planet {source.id} with score {final_score:.2f}, ROI {roi:.2f}, pressure {pressure:.2f}, sending {ships_to_send} ships at angle {math.degrees(angle):.1f}")

				actions.append((final_score, [source.id, angle, ships_to_send]))

				# Mark target as recently sent to avoid duplicate fleets
				# Log action immediately while we have access to source and target
				self.logger.log_action(
					step=step,
					source=source,
					target=target,
					ships_sent=ships_to_send,
					angle=angle,
					score=final_score,
					roi=roi,
					ships_needed=ships_needed,
					pressure=pressure,
					distance=distance
				)

				# Prevent re-sending while the fleet is in transit.
				self.mark_target_sent(target.id, step, 1.0 + travel_turns)

		# -------------------------
		# global selection
		# -------------------------
		actions.sort(key=lambda x: x[0], reverse=True)

		# Clean up expired target memory
		self.cleanup_recent_targets(step)

		# Log state snapshot
		self.logger.log_state(step, planets, fleets, player, my_score, enemy_score, pressure)
		self.last_planet_positions = {p.id: (p.x, p.y) for p in planets}

		return [a[1] for a in actions]
	
