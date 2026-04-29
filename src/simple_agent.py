"""Heuristic Orbit Wars agent.

The agent scores candidate target planets by expected return on investment and
only launches fleets when the projected gain is worth the ship cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .mechanics import BOARD_SIZE, SUN_POS, SUN_RADIUS, compute_fleet_speed, dist, line_intersects_circle

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


def _distance_to_planet(a: PlanetState, b: PlanetState) -> float:
	return dist((a.x, a.y), (b.x, b.y))


def _is_path_safe(source: PlanetState, target: PlanetState, planets: Sequence[PlanetState]) -> bool:
	"""Reject obvious bad shots that cross the sun or another planet."""

	start = (source.x, source.y)
	end = (target.x, target.y)

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


def _estimate_required_ships(source: PlanetState, target: PlanetState, remaining_turns: int) -> Tuple[int, float]:
	distance = _distance_to_planet(source, target)
	guessed_speed = compute_fleet_speed(max(1, source.ships // 3))
	travel_turns = distance / max(1.0, guessed_speed)

	# Account for the target continuing to produce while the fleet is in transit.
	future_garrison = target.ships + target.production * max(1, int(math.ceil(travel_turns)))

	# Keep a tiny cushion so we do not bounce off by one ship on capture.
	ships_needed = future_garrison + 1

	# If the fleet arrives very late, the planet is less valuable.
	effective_turns = max(0.0, remaining_turns - travel_turns)
	return ships_needed, effective_turns


def _score_target(source: PlanetState, target: PlanetState, remaining_turns: int) -> Tuple[float, int, float]:
	ships_needed, effective_turns = _estimate_required_ships(source, target, remaining_turns)
	if ships_needed <= 0:
		return float("-inf"), 0, 0.0

	distance = _distance_to_planet(source, target)

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

	roi = value / max(1.0, ships_needed)
	score = roi - risk_penalty / max(1.0, ships_needed)

	return score, ships_needed, roi

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
			reserve = _defense_reserve(source, fleets, player)
			available = source.ships - reserve
			if available <= 1:
				continue

			best_action = None
			best_score = float("-inf")

			for target in targets:
				if target.id == source.id:
					continue
				if not _is_path_safe(source, target, planets):
					continue

				score, ships_needed, roi = _score_target(source, target, remaining_turns)
				if ships_needed > available:
					continue

				# High-quality captures should be materially positive in ROI terms.
				threshold = 1.10 if target.owner == -1 else 1.25
				if roi < self.roi_threshold or score <= 0:
					continue

				angle = math.atan2(target.y - source.y, target.x - source.x)
				action = [source.id, angle, ships_needed]

				if score > best_score:
					best_score = score
					best_action = action

			if best_action is not None:
				actions.append((best_score, best_action))

		actions.sort(key=lambda item: item[0], reverse=True)

		# Do not over-commit; keep the strongest opportunities only.
		return [action for _, action in actions[:4]]

