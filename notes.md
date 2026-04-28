1. Production = exponential snowball
Bigger radius → more ships per turn → faster scaling
Early expansion = everything

If you fall behind early, you’re dead unless opponent throws.

👉 You should:

Estimate ROI of capturing a planet
Compare:
cost to capture vs ships generated over time horizon
2. Travel time is the entire game

Fleets don’t teleport.

You need:

Distance / speed → arrival time
Speed depends on fleet size → nonlinear timing

This creates:

Small fleet = slow but cheap
Big fleet = fast but expensive

👉 That tradeoff is huge. Most people ignore it and lose.

3. Combat is just subtraction

No randomness. Deterministic.

If:

fleet_size > garrison_at_arrival

→ you win

So the real problem is:
👉 Predict garrison at arrival time

That means:

current ships
production over time
± incoming fleets (both sides)

If you're not simulating future state, your agent is blind.

4. Rotation (this is where most bots fail)

Inner planets move.

So:

Distance is time-dependent
Angles matter
A shot now may miss later

👉 This turns it into:

interception problem, not static targeting

You either:

simulate forward positions
or
approximate with short horizon targeting

If you ignore this → your fleets literally miss or take suboptimal paths.

5. Sun = constraint / obstacle
Deletes fleets → creates no-go zones
Forces path planning

👉 You need at least:

line–circle intersection check

Otherwise your agent suicides fleets.

6. Comets = temporary high ROI spikes

They’re basically:

moving, temporary factories

👉 Strategy:

Only worth it if:
you can capture early
and hold long enough to profit

This is a timing + prediction problem, not just “grab it”.

Game theory layer (this is where you get edge)
1. Greedy expansion vs defense

Classic tension:

Expand fast → vulnerable
Turtle → lose map control

👉 You need a dynamic policy, not fixed:

Early: aggressive expansion
Mid: contest key planets
Late: efficiency + denial
2. Overcommit punishment

If opponent sends a big fleet:

their source planet becomes weak

👉 You should:

detect outgoing fleets
counterattack immediately

This is free wins vs bad bots.

3. Timing attacks (most important mechanic)

Because of travel delay:

You can:

send fleets that arrive simultaneously
overwhelm production

👉 This is basically:
multi-source coordination

If you’re not doing this, you’re playing at half strength.

4. Fake pressure / forcing moves

Even deterministic games have pressure:

Send small fleets to force defense
Opponent overreacts → loses tempo

Not mandatory, but strong vs heuristic bots.

5. Territory control ≠ planet count

What matters:

total production rate
positioning (distance graph)

Sometimes fewer planets > better planets.

What you should actually implement (minimum viable competitive bot)

If you don’t do these, don’t expect results:

1. Forward simulator

At least:

simulate planet state over time
include incoming fleets
2. Target evaluation function

Score planets based on:

production
distance
current owner
contest level
3. Attack feasibility check

Before sending:

simulate arrival
check if capture succeeds
4. Multi-fleet coordination

Don’t send random fleets.
Bundle attacks:

same arrival time
from multiple planets
5. Basic path validation
avoid sun
account for moving targets (even roughly)
Advanced stuff (if you want to actually win)
Interception math for rotating planets
Opponent modeling (predict where they expand)
Value of time (discount future production)
Min-cost flow / resource allocation framing
Monte Carlo rollout (if allowed by time constraints)
Brutal truth

Most Kaggle bots:

don’t simulate properly
don’t coordinate fleets
ignore timing

If you just:

simulate correctly
coordinate attacks
exploit overextensions


Reward ideas:



