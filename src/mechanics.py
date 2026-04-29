import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

BOARD_SIZE = 100.0
SUN_POS = (50.0, 50.0)
SUN_RADIUS = 10.0

MAX_FLEET_SPEED = 6.0


# ----------------------------
# Basic geometry helpers
# ----------------------------

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def line_intersects_circle(p1, p2, center, radius) -> bool:
    # checks if segment p1->p2 intersects circle
    (x1, y1), (x2, y2) = p1, p2
    cx, cy = center

    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - cx, y1 - cy

    a = dx*dx + dy*dy
    b = 2 * (fx*dx + fy*dy)
    c = fx*fx + fy*fy - radius*radius

    if a == 0:
        return dist(p1, center) <= radius

    discriminant = b*b - 4*a*c
    if discriminant < 0:
        return False

    discriminant = math.sqrt(discriminant)

    t1 = (-b - discriminant) / (2*a)
    t2 = (-b + discriminant) / (2*a)

    return (0 <= t1 <= 1) or (0 <= t2 <= 1)


# ----------------------------
# Data models
# ----------------------------

@dataclass
class Planet:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int
    angular_velocity: float = 0.0
    angle: float = 0.0  # for orbiting planets


@dataclass
class Fleet:
    id: int
    owner: int
    x: float
    y: float
    angle: float
    ships: int
    speed: float


# ----------------------------
# Planet dynamics
# ----------------------------

def update_orbiting_planet(p: Planet):
    if p.angular_velocity == 0:
        return

    cx, cy = SUN_POS
    r = dist((p.x, p.y), SUN_POS)

    p.angle += p.angular_velocity

    p.x = cx + r * math.cos(p.angle)
    p.y = cy + r * math.sin(p.angle)


# ----------------------------
# Fleet dynamics
# ----------------------------

def compute_fleet_speed(ships: int) -> float:
    if ships <= 1:
        return 1.0
    return 1.0 + (MAX_FLEET_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5


def move_fleet(f: Fleet):
    dx = math.cos(f.angle) * f.speed
    dy = math.sin(f.angle) * f.speed

    new_x = f.x + dx
    new_y = f.y + dy

    old_pos = (f.x, f.y)
    new_pos = (new_x, new_y)

    f.x, f.y = new_x, new_y

    return old_pos, new_pos


# ----------------------------
# Collision checks
# ----------------------------

def fleet_hits_sun(old_pos, new_pos) -> bool:
    return line_intersects_circle(old_pos, new_pos, SUN_POS, SUN_RADIUS)


def fleet_hits_planet(old_pos, new_pos, planet: Planet) -> bool:
    return line_intersects_circle(old_pos, new_pos, (planet.x, planet.y), planet.radius)


def out_of_bounds(pos) -> bool:
    x, y = pos
    return x < 0 or y < 0 or x > BOARD_SIZE or y > BOARD_SIZE


# ----------------------------
# Combat resolution (simplified core)
# ----------------------------

def resolve_combat(fleets: List[Fleet], planet: Planet):
    """
    Minimal deterministic version:
    - group by owner
    - sum ships per owner
    - largest vs second largest
    """

    if not fleets:
        return planet

    grouped = {}
    for f in fleets:
        grouped[f.owner] = grouped.get(f.owner, 0) + f.ships

    sorted_factions = sorted(grouped.items(), key=lambda x: x[1], reverse=True)

    top_owner, top_ships = sorted_factions[0]
    second_ships = sorted_factions[1][1] if len(sorted_factions) > 1 else 0

    surviving = top_ships - second_ships
    if surviving <= 0:
        return planet  # all attackers die

    if top_owner == planet.owner:
        planet.ships += surviving
    else:
        if surviving > planet.ships:
            planet.owner = top_owner
            planet.ships = surviving - planet.ships
        else:
            planet.ships -= surviving

    return planet


# ----------------------------
# Core simulation step
# ----------------------------

def simulate_step(planets: List[Planet], fleets: List[Fleet]):
    # 1. update planets
    for p in planets:
        update_orbiting_planet(p)

    # 2. move fleets
    new_fleets = []
    planet_hits = {p.id: [] for p in planets}

    for f in fleets:
        old_pos, new_pos = move_fleet(f)

        if out_of_bounds(new_pos):
            continue

        if fleet_hits_sun(old_pos, new_pos):
            continue

        hit = False
        for p in planets:
            if fleet_hits_planet(old_pos, new_pos, p):
                planet_hits[p.id].append(f)
                hit = True
                break

        if not hit:
            new_fleets.append(f)

    # 3. resolve combat
    for p in planets:
        if planet_hits[p.id]:
            p = resolve_combat(planet_hits[p.id], p)

    # 4. production
    for p in planets:
        if p.owner != -1:
            p.ships += p.production

    return planets, new_fleets

def calculate_score(obs, player_id):
    score = 0.0

    planets = obs.get("planets", [])
    fleets = obs.get("fleets", [])

    # -------------------------
    # 1. Planets
    # -------------------------
    for p in planets:
        # p format: [id, owner, x, y, radius, ships, production]
        owner = p[1]
        ships = p[5]

        if owner == player_id:
            score += ships

    # -------------------------
    # 2. Fleets (discounted)
    # -------------------------
    for f in fleets:
        # f format: [id, owner, x, y, angle, from_planet_id, ships]
        owner = f[1]
        ships = f[6]

        if owner == player_id:
            score += 0.8 * ships

    return score