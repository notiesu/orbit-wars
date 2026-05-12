"""Pure incremental ROI Orbit Wars agent.

No minimum ship estimation.
No hard capture feasibility gate.

Architecture:
- Every target starts with a tiny commitment.
- Ships are added incrementally while marginal ROI improves.
- Best commitment size is discovered dynamically.
"""

#######TODOS#########################

#TODO: change pure distance penalty to account for time of intersection + direction
#done

#TODO: distinct value for moving and stable planets
#done - expansion factor accounted

#TODO: RL for tuning parameters

#TODO: Train on simple PPO agent to learn better parameters and maybe even a value function for lookahead

#TODO: Resolve comets

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple
try:    from mechanics import (
        SUN_POS,
        SUN_RADIUS,
        calculate_interception_angle,
        compute_fleet_speed,
        dist,
        line_intersects_circle,
        is_inner_planet,
        angle_intersects_sun,
        predict_orbit_position
    )
except ImportError:
    from .mechanics import (
        SUN_POS,
        SUN_RADIUS,
        calculate_interception_angle,
        compute_fleet_speed,
        dist,
        line_intersects_circle,
        is_inner_planet,
        angle_intersects_sun,
        predict_orbit_position
    )

# =========================================================
# CONFIGURABLE MACROS / CONSTANTS
# =========================================================
# Path / search radius
PATH_SEARCH_RADIUS_MIN = 18.0
PATH_SEARCH_RADIUS_FACTOR = 5.0

# Local value multipliers
PLANET_VALUE_PRODUCTION_FACTOR = 10.0
PLANET_VALUE_SHIP_FACTOR = 0.03
PRODUCTION_VALUE_FACTOR = 10.0

# Defense reserve
DEFENSE_RESERVE_FACTOR = 0.2
# is now dynamically calculated as 20% of current ships
DEFENSE_RESERVE_PROD_MULT = 0.75

# Enemy proximity handling
NEAREST_ENEMY_INIT = 9999.0
ENEMY_PROXIMITY_THRESHOLD = 50.0
ENEMY_PROXIMITY_MULTIPLIER = 0.5
ENEMY_FLEET_SIZE_MULTIPLIER = 0.04

# Fleet speed / commitment
COMMITMENT_SPEED_DIVISOR = 3
MIN_FLEET_SPEED = 1.0

# Capture probability / uncertainty
DEFENSE_UNCERTAINTY_FACTOR = 0.06
# Ratio-based sigmoid: log(commitment/defense) scaled by K
# K=2.0 gives: ratio 1.1 -> ~55% prob, ratio 2.0 -> ~80% prob, ratio 10 -> ~99% prob
CAPTURE_SIGMOID_K = 2.0

# Economic / pressure bonuses
# INCREMENTAL_OWNERSHIP_BONUS_NEUTRAL = 1.0
# INCREMENTAL_OWNERSHIP_BONUS_CAPTURED = 1.3
PRESSURE_BONUS_CAP = 1.2
PRESSURE_RATIO_FACTOR = 0.25

# Ownership-specific capture shaping
NEUTRAL_CAPTURE_COST_FACTOR = 0.04
CONTEST_CAPTURE_COST_FACTOR = 0.015
CONTEST_PRESSURE_MULTIPLIER = 0.6

# Early enemy-contest shaping
EARLY_CONTEST_SHIP_SCALE = 30.0
EARLY_CONTEST_PROD_SCALE = 4.0
EARLY_CONTEST_BONUS_CAP = 0.75
EARLY_CONTEST_BONUS_MULTIPLIER = 0.9
CONTEST_RATIO_THRESHOLD = 0.5

# Overkill / distance penalties
OVERKILL_PENALTY_FACTOR = 0.03
OVERKILL_PENALTY_CAP = 1.5
# Distance is already represented through travel time, so keep this soft.
DISTANCE_PENALTY_FACTOR = 0.005

# Movement-aware capture shaping
MOVING_AWAY_DISTANCE_PENALTY_FACTOR = 0.75
# Agent / commitment defaults
AVAILABLE_MIN_TO_CONSIDER = 2
INITIAL_COMMITMENT = 2
COMMITMENT_INCREMENT = 1
COMMITMENT_STOP_MIN = 6

# Global limits and defaults
LAUNCH_ACTIONS_LIMIT = 4
MAX_STEPS_DEFAULT = 500
BEST_SCORE_INIT = -1e18
# ROI sigmoid shaping
ROI_SIGMOID_SHIFT = 0.45
ROI_SIGMOID_SCALE = 5.0

# Observation defaults
OBS_DEFAULT_PLAYER = 0
OBS_DEFAULT_STEP = 0

# Progress / denominators
PROGRESS_DENOMINATOR_MIN = 1

LAUNCH_DELAY_DEFAULT = 0.0

# Defense / return constants
MIN_RETURN_SHIPS = 0
SHIP_MIN_BUFFER = 1

# Minimum commitment for speed calculation
MIN_COMMITMENT_FOR_SPEED = 1
MIN_LAUNCH_COMMITMENT = 0

# Expansion / consolidation tuning


EXPANSION_FACTOR_MIN = 0.5
EXPANSION_FACTOR_MAX = 2.5

# optional shaping exponent
EXPANSION_FACTOR_CURVE = 1.25

# =====================================================
# Defense / pressure modeling
# =====================================================


DEFENSE_PRESSURE_FACTOR = 0.015

# Ownership-aware scaling for pressure-based defense inflation.
# Enemy targets retain full pressure impact, while neutral/friendly
# targets are only lightly adjusted.
ENEMY_DEFENSE_WEIGHT = 1.0
NEUTRAL_DEFENSE_WEIGHT = 0.25
FRIENDLY_DEFENSE_WEIGHT = 0.25

# =====================================================
# Tempo / payback
# =====================================================

TEMPO_PENALTY_FACTOR = 0.01
PAYBACK_COMMITMENT_EXPONENT = 0.7

# Encourage larger waves to reach a reliable capture force.
ASSAULT_BUFFER_FACTOR = 1.1
UNDERCOMMITMENT_PENALTY_FACTOR = 0.75

# Reward lower arrival times from added ships, while keeping the scale bounded.
ARRIVAL_TIME_DELTA_FACTOR = 0.5
ARRIVAL_TIME_DELTA_CAP = 1.5

# =====================================================
# Production delta
# =====================================================

# Bonus for capturing high-production targets from low-production sources.
# Encourages fast seizure of production capability upgrades.
# Ratio-based: target_prod / (source_prod + 1) scaled by factor
PRODUCTION_DELTA_BONUS_FACTOR = 0.15

# =====================================================
# Overcommitment
# =====================================================

OVERCOMMITMENT_PENALTY_FACTOR = 0.0

# =====================================================
# Commitment lookahead
# =====================================================

# When searching for best commitment, use lookahead to avoid fragmenting into multiple small launches.
# If the next increment's marginal ROI is also strong (above this ratio of threshold),
# the current commitment is too small—keep searching.
COMMITMENT_LOOKAHEAD_RATIO = 0.7  # If next marginal is > 70% of threshold, keep searching

# =====================================================
# Contest / local support
# =====================================================

# Logarithmic scaling: log1p(contest_ratio) * FACTOR, capped at CAP
# Tames extreme ratios while staying aggressive
# contest_ratio ~2 -> 1.5x multiplier
# contest_ratio ~10 -> 2.4x multiplier
# contest_ratio ~100 -> 3.7x multiplier (capped)
CONTEST_BONUS_FACTOR = 1.0
CONTEST_BONUS_CAP = 3.0

# Directional expansion bonus
# Encourages spreading across board rather than one-directional pushes
DIRECTIONAL_WEAKNESS_BONUS_FACTOR = 0.05

# =====================================================
# Ownership modifiers
# ======================================================

OWNERSHIP_NEUTRAL_FACTOR = 1.0
OWNERSHIP_FRIENDLY_FACTOR = 0.7
OWNERSHIP_ENEMY_FACTOR = 2.0

# =========================================================
# DATA
# =========================================================

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


@dataclass(frozen=True)
class AgentGameState:
    planets: List[PlanetState]
    fleets: List[FleetState]
    player: int
    step: int
    max_steps: int


# =========================================================
# OBS PARSING
# =========================================================

def _obs_value(obs: Any, key: str, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _parse_planets(obs: Any) -> List[PlanetState]:
    return [PlanetState(*p) for p in _obs_value(obs, "planets", [])]


def _parse_fleets(obs: Any) -> List[FleetState]:
    return [FleetState(*f) for f in _obs_value(obs, "fleets", [])]


# =========================================================
# UTILS
# =========================================================

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def entity_pos(entity):
    return (entity.x, entity.y)


def projected_separation(
    source: PlanetState,
    target: PlanetState,
    travel_turns: float,
    source_angular_velocity: float,
    target_angular_velocity: float,
) -> float:

    source_pos = entity_pos(source)
    target_pos = entity_pos(target)

    if source_angular_velocity != 0.0:
        source_pos = predict_orbit_position(
            source_pos,
            travel_turns,
            source_angular_velocity,
        )

    if target_angular_velocity != 0.0:
        target_pos = predict_orbit_position(
            target_pos,
            travel_turns,
            target_angular_velocity,
        )

    return dist(source_pos, target_pos)


# =========================================================
# LOCAL PRESSURE
# =========================================================

def local_strength(
    target: PlanetState,
    game_state: AgentGameState,
    owner_filter,
) -> float:

    strength = 0.0
    radius = max(
        PATH_SEARCH_RADIUS_MIN,
        target.radius * PATH_SEARCH_RADIUS_FACTOR,
    )

    for planet in game_state.planets:

        if planet.id == target.id:
            continue

        if not owner_filter(planet.owner):
            continue

        d = dist(
            entity_pos(target),
            entity_pos(planet),
        )

        if d > radius:
            continue

        value = (
            planet.ships
            + planet.production * PLANET_VALUE_PRODUCTION_FACTOR
        )

        strength += value / (1.0 + d)

    for fleet in game_state.fleets:

        if not owner_filter(fleet.owner):
            continue

        d = dist(
            entity_pos(target),
            entity_pos(fleet),
        )

        if d > radius:
            continue

        strength += fleet.ships / (1.0 + d)

    return strength


def allied_strength(target, game_state):
    return local_strength(
        target,
        game_state,
        lambda o: o == game_state.player,
    )


def enemy_strength(target, game_state):
    return local_strength(
        target,
        game_state,
        lambda o: o not in (-1, game_state.player),
    )


def defense_weight_for_target(target_owner: int, player: int) -> float:
    if target_owner == player:
        return FRIENDLY_DEFENSE_WEIGHT
    if target_owner == -1:
        return NEUTRAL_DEFENSE_WEIGHT
    return ENEMY_DEFENSE_WEIGHT


def early_enemy_contest_bonus(target: PlanetState, player: int) -> float:
    if target.owner in (-1, player):
        return 1.0

    target_scale = (
        target.ships / EARLY_CONTEST_SHIP_SCALE
        + target.production / EARLY_CONTEST_PROD_SCALE
    )
    urgency = 1.0 / (1.0 + target_scale)

    return 1.0 + min(
        EARLY_CONTEST_BONUS_CAP,
        urgency * EARLY_CONTEST_BONUS_MULTIPLIER,
    )


def directional_weakness_bonus(target: PlanetState, game_state: AgentGameState, player: int) -> float:
    """Bonus for expanding into directionally weak areas.
    Returns multiplier ~1.0 to ~1.3 based on target direction.
    """
    allied_planets = [p for p in game_state.planets if p.owner == player]
    if not allied_planets:
        return 1.0

    # Player's center of mass
    center_x = sum(p.x for p in allied_planets) / len(allied_planets)
    center_y = sum(p.y for p in allied_planets) / len(allied_planets)

    # Angle from center to target
    target_angle = math.atan2(target.y - center_y, target.x - center_x)

    # Measure allied strength in target direction vs opposite
    strength_toward = 0.0
    strength_away = 0.0

    for planet in game_state.planets:
        if planet.owner != player or planet.id == target.id:
            continue

        angle = math.atan2(planet.y - center_y, planet.x - center_x)
        angle_diff = abs(angle - target_angle)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff

        strength = planet.ships + planet.production * PLANET_VALUE_PRODUCTION_FACTOR
        if angle_diff < math.pi / 4:  # 45° cone toward target
            strength_toward += strength
        elif angle_diff > 3 * math.pi / 4:  # opposite direction
            strength_away += strength

    # Bonus if we're weak in target direction
    weakness_ratio = (strength_away + 1.0) / (strength_toward + 1.0)
    return 1.0 + min(0.3, weakness_ratio * DIRECTIONAL_WEAKNESS_BONUS_FACTOR)


# =========================================================
# PATHING
# =========================================================

def path_safe(
    source: PlanetState,
    target: PlanetState,
    planets: Sequence[PlanetState],
) -> bool:

    start = (source.x, source.y)
    end = (target.x, target.y)
    if line_intersects_circle(
        start,
        end,
        SUN_POS,
        SUN_RADIUS,
    ):
        return False

    for p in planets:

        if p.id in (source.id, target.id):
            continue

        if line_intersects_circle(
            start,
            end,
            (p.x, p.y),
            p.radius,
        ):
            return False

    return True


# =========================================================
# DEFENSE RESERVE
# =========================================================

def defense_reserve(
    source: PlanetState,
    fleets: Sequence[FleetState],
    player: int,
) -> int:

    # Normalize defense reserve min to 20% of current ships (dynamic from current timestep)
    normalized_reserve_min = max(1, int(source.ships * DEFENSE_RESERVE_FACTOR))
    reserve = max(
        normalized_reserve_min,
        int(source.production * DEFENSE_RESERVE_PROD_MULT),
    )

    nearest_enemy = NEAREST_ENEMY_INIT
    enemy_fleet_pressure = 0.0

    for fleet in fleets:

        if fleet.owner == player:
            continue

        d = dist(
            entity_pos(source),
            entity_pos(fleet),
        )

        nearest_enemy = min(
            nearest_enemy,
            d,
        )

        proximity_weight = min(1.0, ENEMY_PROXIMITY_THRESHOLD / max(1.0, d))
        enemy_fleet_pressure += fleet.ships * proximity_weight

    if nearest_enemy < ENEMY_PROXIMITY_THRESHOLD:
        reserve += int((ENEMY_PROXIMITY_THRESHOLD - nearest_enemy) * ENEMY_PROXIMITY_MULTIPLIER)

    if enemy_fleet_pressure > 0.0:
        reserve += int(enemy_fleet_pressure * ENEMY_FLEET_SIZE_MULTIPLIER)

    return min(
        reserve,
        max(MIN_RETURN_SHIPS, source.ships - SHIP_MIN_BUFFER),
    )


# =========================================================
# MARGINAL ROI
# =========================================================

def incremental_roi(
    target: PlanetState,
    source: PlanetState,
    ships_to_add: int,
    current_commitment: int,
    game_state: AgentGameState,
    player: int = None,
    expansion_factor: float = 1.0,
    source_angular_velocity: float = 0.0,
    target_angular_velocity: float = 0.0,
    travel_turns_cache: dict | None = None,
) -> float:

    # =====================================================
    # Commitment
    # =====================================================

    new_commitment = current_commitment + ships_to_add

    if new_commitment <= 0:
        return 0.0

    # =====================================================
    # Travel
    # =====================================================

    distance = dist(entity_pos(source), entity_pos(target))

    if travel_turns_cache is None:
        travel_turns_cache = {}

    def effective_travel_turns_for(commitment: int) -> float:
        cached_turns = travel_turns_cache.get(commitment)
        if cached_turns is not None:
            return cached_turns

        speed = compute_fleet_speed(
            max(
                MIN_COMMITMENT_FOR_SPEED,
                commitment // COMMITMENT_SPEED_DIVISOR,
            )
        )

        travel_turns = distance / max(MIN_FLEET_SPEED, speed)

        predicted_separation = projected_separation(
            source,
            target,
            travel_turns,
            source_angular_velocity,
            target_angular_velocity,
        )

        separation_delta = max(0.0, predicted_separation - distance)
        moving_away_turns = (
            separation_delta
            / max(MIN_FLEET_SPEED, speed)
            * MOVING_AWAY_DISTANCE_PENALTY_FACTOR
        )

        cached_turns = travel_turns + moving_away_turns
        travel_turns_cache[commitment] = cached_turns
        return cached_turns

    baseline_travel_turns = effective_travel_turns_for(current_commitment)
    effective_travel_turns = effective_travel_turns_for(new_commitment)
    arrival_time_delta = baseline_travel_turns - effective_travel_turns

    arrival_time_multiplier = 1.0 + max(
        -ARRIVAL_TIME_DELTA_CAP,
        min(
            ARRIVAL_TIME_DELTA_CAP,
            arrival_time_delta * ARRIVAL_TIME_DELTA_FACTOR,
        ),
    )

    # =====================================================
    # Defensive projection
    # =====================================================

    projected_defense = (
        target.ships * (1.0 + DEFENSE_UNCERTAINTY_FACTOR * effective_travel_turns)
        + target.production * effective_travel_turns
    )

    local_enemy = enemy_strength(target, game_state)
    local_allied = allied_strength(target, game_state)

    pressure_ratio = (
        (local_enemy + 1.0)
        / (local_allied + 1.0)
    )

    defense_multiplier = (
        1.0
        + pressure_ratio * DEFENSE_PRESSURE_FACTOR
    )

    defense_weight = defense_weight_for_target(target.owner, player)
    normalized_defense_multiplier = (
        1.0
        + (defense_multiplier - 1.0) * defense_weight
    )

    effective_defense = (
        projected_defense * normalized_defense_multiplier
    )

    # =====================================================
    # Capture probability (ratio-based)
    # =====================================================

    # Use commitment ratio vs defense for natural incremental scaling
    commitment_ratio = new_commitment / max(1.0, effective_defense)

    def safe_sigmoid(x: float) -> float:
        if x > 50:
            return 1.0
        if x < -50:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    capture_probability = safe_sigmoid(
        math.log(max(0.1, commitment_ratio)) * CAPTURE_SIGMOID_K
    )

    # =====================================================
    # Remaining economic lifetime
    # =====================================================

    remaining_turns = max(
        0.0,
        game_state.max_steps
        - game_state.step
        - effective_travel_turns
    )

    # =====================================================
    # Raw economic value
    # =====================================================

    production_value = (
        target.production * PRODUCTION_VALUE_FACTOR * remaining_turns
    )

    # =====================================================
    # Tempo / payback modeling
    # =====================================================

    payback_turns = (
        (new_commitment ** PAYBACK_COMMITMENT_EXPONENT)
        / max(1.0, target.production)
    )

    true_payback = (
        effective_travel_turns + payback_turns
    )

    tempo_penalty = (
        1.0
        + true_payback * TEMPO_PENALTY_FACTOR
    )

    # =====================================================
    # Overcommitment penalty
    # =====================================================

    required_force = effective_defense

    overcommitment = max(
        0.0,
        new_commitment - required_force
    )

    overcommitment_ratio = (
        overcommitment
        / max(1.0, required_force)
    )

    overcommitment_penalty = (
        1.0
        + overcommitment_ratio
        * OVERCOMMITMENT_PENALTY_FACTOR
    )

    assault_requirement = (
        effective_defense
        * ASSAULT_BUFFER_FACTOR
    )

    undercommitment_ratio = (
        max(0.0, assault_requirement - new_commitment)
        / max(1.0, assault_requirement)
    )

    undercommitment_penalty = (
        1.0
        + undercommitment_ratio
        * UNDERCOMMITMENT_PENALTY_FACTOR
    )

    # =====================================================
    # Distance penalty
    # =====================================================

    distance_penalty = (
        1.0
        + distance * DISTANCE_PENALTY_FACTOR
    )

    # =====================================================
    # Contestability
    # =====================================================

    contest_ratio = (
        (local_allied + 1.0)
        / (local_enemy + 1.0)
    )

    #if local enemy is strong, penalize heavily to encourage building local support first
    if contest_ratio < CONTEST_RATIO_THRESHOLD:
        contest_penalty = (
            1.0
            + (CONTEST_RATIO_THRESHOLD - contest_ratio)
            * CONTEST_PRESSURE_MULTIPLIER
        )


    contest_bonus = (
        1.0
        + min(
            CONTEST_BONUS_CAP,
            math.log1p(contest_ratio) * CONTEST_BONUS_FACTOR,
        )
    )

    # =====================================================
    # Ownership modifier
    # =====================================================

    if target.owner == -1:
        # Penalty for neutrals with high defense cost relative to production
        # If defense >> production, this is a resource sink we should avoid
        defense_to_production_ratio = effective_defense / max(1.0, target.production)
        neutral_defense_penalty = 1.0 / (1.0 + defense_to_production_ratio * 0.1)
        ownership_modifier = (
            OWNERSHIP_NEUTRAL_FACTOR * neutral_defense_penalty
        )

    elif target.owner == player:
        ownership_modifier = (
            OWNERSHIP_FRIENDLY_FACTOR
        )

    else:
        ownership_modifier = (
            OWNERSHIP_ENEMY_FACTOR
            * early_enemy_contest_bonus(target, player)
        )

    # =====================================================
    # Expansion modifier
    # =====================================================

    # =====================================================
    # Production delta bonus
    # =====================================================

    # Reward capturing high-production targets from low-production sources
    production_delta = target.production - source.production
    if production_delta > 0:
        production_delta_ratio = production_delta / max(1.0, source.production)
        production_delta_bonus = (
            1.0
            + min(
                0.5,  # Cap the bonus to avoid extreme values
                production_delta_ratio * PRODUCTION_DELTA_BONUS_FACTOR,
            )
        )
    else:
        production_delta_bonus = 1.0

    # =====================================================
    # Final value assembly
    # =====================================================

   
    directional_bonus = directional_weakness_bonus(target, game_state, player)

    gross_value = (

        production_value
        * contest_bonus
        * ownership_modifier
        * expansion_factor
        * directional_bonus
        * production_delta_bonus
        * arrival_time_multiplier
    )

    

    expected_value = (
        gross_value
        * capture_probability
    )

    total_penalty = (
        tempo_penalty
        * overcommitment_penalty
        * undercommitment_penalty
        * distance_penalty
        * (contest_penalty if contest_ratio < CONTEST_RATIO_THRESHOLD else 1.0)
    )

    roi = expected_value / max(1e-6, total_penalty)

    # =====================================================
    # Normalize ROI range
    # =====================================================

    roi = math.log1p(max(0.0, roi))

    return roi

def calculate_base(planets, inner_planets, outer_planets):
    """
    Computes total strategic value of inner vs outer holdings.

    Production is weighted much more heavily than ships because
    production represents long-term economic control while ships
    are temporary/local.
    """

    inner_value = 0.0
    outer_value = 0.0

    for planet in planets:
        value = (
            planet.production * PLANET_VALUE_PRODUCTION_FACTOR
            + planet.ships * PLANET_VALUE_SHIP_FACTOR
        )

        if planet.id in inner_planets:
            inner_value += value

        elif planet.id in outer_planets:
            outer_value += value

    return inner_value, outer_value


def expansion_factor(inner_value, outer_value):
    """
    Returns expansion aggressiveness multiplier.

    ~0.5:
        weak outer shell -> discourage risky inward expansion

    ~1.0:
        balanced

    ~1.5:
        strong outer shell -> encourage aggressive expansion
    """

    total = inner_value + outer_value

    if total <= 1e-6:
        return 1.0

    outer_ratio = outer_value / total

    # map [0,1] -> [0.5,1.5]
    factor = EXPANSION_FACTOR_MIN + (
        outer_ratio * (EXPANSION_FACTOR_MAX - EXPANSION_FACTOR_MIN)
    )

    # optional nonlinear shaping
    factor = factor ** EXPANSION_FACTOR_CURVE

    return factor


# =========================================================
# =========================================================
# AGENT
# =========================================================

class IncrementalAgent:

    def __init__(
        self,
        initial_roi_threshold=0.01,
        ending_roi_threshold=0.08,
    ):

        self.initial_roi_threshold = initial_roi_threshold
        self.ending_roi_threshold = ending_roi_threshold
        self.player = None
        
        #For scoring stationary planets vs. moving planets

        # self.logger = AgentLogger("IncrementalAgent")

    # -----------------------------------------------------

    def roi_threshold(self, progress):

        s = sigmoid(
            (progress - ROI_SIGMOID_SHIFT) * ROI_SIGMOID_SCALE
        )

        return (
            self.initial_roi_threshold
            + (
                self.ending_roi_threshold
                - self.initial_roi_threshold
            ) * s
        )

    # -----------------------------------------------------

    def predict(self, obs):
        #TODO - DO NOT CONSIDER COMETS FOR NOW
        if self.player is None:
            self.player = _obs_value(obs, "player", OBS_DEFAULT_PLAYER)
        inner_planet_angular_velocity = _obs_value(obs, 'angular_velocity', [])
        planets = _parse_planets(obs)
        fleets = _parse_fleets(obs)

        player = _obs_value(obs, "player", OBS_DEFAULT_PLAYER)

        step = int(
            _obs_value(obs, "step", OBS_DEFAULT_STEP)
        )

        max_steps = int(
            _obs_value(obs, "max_steps", MAX_STEPS_DEFAULT)
        )

        game_state = AgentGameState(
            planets=planets,
            fleets=fleets,
            player=player,
            step=step,
            max_steps=max_steps,
        )

        my_planets = [
            p for p in planets
            if p.owner == player
        ]

        #separate stationary and moving planets
        inner_planets = set()
        outer_planets = set()
        for planet in planets:
            if is_inner_planet((planet.x, planet.y), planet.radius):
                inner_planets.add(planet.id)
            else:
                outer_planets.add(planet.id)

        
        targets = [
            p for p in planets
            if p.owner != player
        ]

        if not my_planets or not targets:
            return []

        progress = (
            step / max(PROGRESS_DENOMINATOR_MIN, max_steps)
        )

        roi_threshold = self.roi_threshold(
            progress
        )

        candidate_actions = []

        comets = _obs_value(obs, "comets", {}) or {}
        comet_ids = {id for id in comets[0]["planet_ids"]} if comets else set()

        # =================================================
        # SOURCE LOOP
        # =================================================

        for source in my_planets:

            source_angular_velocity = inner_planet_angular_velocity if is_inner_planet((source.x, source.y), source.radius) else 0.0

            reserve = defense_reserve(
                source,
                fleets,
                player,
            )

            available = (
                source.ships
                - reserve
            )

            if available <= AVAILABLE_MIN_TO_CONSIDER:
                continue

            # =============================================
            # TARGET LOOP
            # =============================================

            for target in targets:
                target_angular_velocity = inner_planet_angular_velocity if is_inner_planet((target.x, target.y), target.radius) else 0.0
                if target.id == source.id or target.id in comet_ids:
                    continue

                if not path_safe(
                    source,
                    target,
                    planets,
                ):
                    continue

                # -----------------------------------------
                # Pure incremental allocation
                # -----------------------------------------

                commitment = INITIAL_COMMITMENT

                #scale initial commitment with distance to prevent over-commitment on long paths
                distance = dist(entity_pos(source), entity_pos(target))
                commitment = max(commitment, int(distance / COMMITMENT_SPEED_DIVISOR))

                increment = COMMITMENT_INCREMENT

                total_score = 0.0

                best_score = BEST_SCORE_INIT
                best_commitment = 0
                travel_turns_cache = {}

                speed_est = compute_fleet_speed(
                    max(
                        MIN_COMMITMENT_FOR_SPEED,
                        commitment // COMMITMENT_SPEED_DIVISOR,
                    )
                )
                travel_turns_est = distance / max(MIN_FLEET_SPEED, speed_est)
                projected_defense_est = (
                    target.ships
                    + target.production * travel_turns_est
                )
                local_enemy_est = enemy_strength(target, game_state)
                local_allied_est = allied_strength(target, game_state)
                pressure_ratio_est = (
                    (local_enemy_est + 1.0)
                    / (local_allied_est + 1.0)
                )
                defense_multiplier_est = (
                    1.0
                    + pressure_ratio_est * DEFENSE_PRESSURE_FACTOR
                )
                defense_weight_est = defense_weight_for_target(target.owner, player)
                normalized_defense_multiplier_est = (
                    1.0
                    + (defense_multiplier_est - 1.0) * defense_weight_est
                )
                effective_defense_est = (
                    projected_defense_est * normalized_defense_multiplier_est
                )
                min_assault_commitment = int(
                    effective_defense_est * ASSAULT_BUFFER_FACTOR
                )

                ex_factor = expansion_factor(*calculate_base(planets, inner_planets, outer_planets))
                while commitment <= available:
                    
                    marginal = incremental_roi(
                        target,
                        source,
                        increment,
                        commitment - increment,
                        game_state,
                        player=player,
                        expansion_factor=ex_factor if target.id in inner_planets else 1.0,
                        source_angular_velocity=source_angular_velocity,
                        target_angular_velocity=target_angular_velocity,
                        travel_turns_cache=travel_turns_cache,
                    )

                    total_score += marginal

                    if total_score > best_score:
                        best_score = total_score
                        best_commitment = commitment

                    # stop when marginal utility collapses, BUT with lookahead:
                    # if the next increment would also be strong, don't stop yet (avoid fragmentation)
                    should_stop = False
                    if (
                        marginal < roi_threshold
                        and commitment > COMMITMENT_STOP_MIN
                        and commitment > min_assault_commitment
                    ):
                        # Lookahead: would the next increment also be weak?
                        next_commitment = commitment + increment
                        if next_commitment <= available:
                            next_marginal = incremental_roi(
                                target,
                                source,
                                increment,
                                next_commitment - increment,
                                game_state,
                                player=player,
                                expansion_factor=ex_factor if target.id in inner_planets else 1.0,
                                source_angular_velocity=source_angular_velocity,
                                target_angular_velocity=target_angular_velocity,
                                travel_turns_cache=travel_turns_cache,
                            )
                            # Only stop if next increment is also weak (below lookahead threshold)
                            if next_marginal < roi_threshold * COMMITMENT_LOOKAHEAD_RATIO:
                                should_stop = True
                        else:
                            should_stop = True

                    if should_stop:
                        break

                    commitment += increment

                # if the maximized total incremental ROI is below the threshold,
                # do not launch at all
                if best_score < roi_threshold:
                    # print(f"Skipping launch from {source.id} to {target.id} due to low ROI: {best_score:.4f} with {best_commitment} ships (threshold: {roi_threshold:.4f})")
                    continue

                if best_commitment <= MIN_LAUNCH_COMMITMENT:
                    continue

                # -----------------------------------------
                # Final launch
                # -----------------------------------------

                angle = calculate_interception_angle(
                    source_pos=(
                        source.x,
                        source.y,
                    ),
                    target_pos=(
                        target.x,
                        target.y,
                    ),
                    target_angular_velocity=target_angular_velocity,
                    ships=best_commitment,
                    source_angular_velocity=source_angular_velocity,
                    launch_delay=LAUNCH_DELAY_DEFAULT,
                    source_radius=source.radius,
                )

                #if angle intersects sun, skip launch
                if angle_intersects_sun(
                    source=(
                        source.x,
                        source.y,
                    ),
                    angle=angle,
                    sun_pos=SUN_POS,
                    sun_radius=SUN_RADIUS,
                    source_radius=source.radius,
                ):
                    # print(f"Skipping launch from {source.id} to {target.id} due to sun collision at angle {math.degrees(angle):.2f}°")
                    continue
                
                if angle is None:

                    angle = math.atan2(
                        target.y - source.y,
                        target.x - source.x,
                    )

                candidate_actions.append(
                    (
                        best_score,
                        [
                            source.id,
                            angle,
                            best_commitment,
                        ],
                    )
                )

        # =================================================
        # GLOBAL SELECTION
        # =================================================

        candidate_actions.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            action
            for _, action
            in candidate_actions[:LAUNCH_ACTIONS_LIMIT]
        ]