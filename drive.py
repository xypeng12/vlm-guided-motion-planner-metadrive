"""
drive.py  –  Manually drive through your custom MetaDrive map
=============================================================
Opens a 3-D window. Use keyboard to steer.

Controls (in-game):
    W / Up     Accelerate
    S / Down   Brake / reverse
    A / Left   Steer left
    D / Right  Steer right
    R          Reset the scenario
    Q          Quit

Usage:
    # Custom long highway with incidents (recommended)
    python drive.py --highway
    python drive.py --highway --lanes 3 --traffic 0.2 --incident 0.6

    # Generic sandbox maps
    python drive.py [--map <preset>] [--lanes N] [--seed N] [--traffic N]

Map presets (edit MAP_PRESETS in my_env.py to add more):
    straight | curve | intersection | roundabout |
    highway  | t_junction | mixed | random_3 | random_5

Examples:
    python drive.py --highway                          # long highway, incidents ON
    python drive.py --highway --lanes 4 --traffic 0.3
    python drive.py --highway --incident 0.8           # more incidents
    python drive.py --map roundabout                   # generic roundabout
    python drive.py --map random_5 --traffic 0.2
"""

import argparse
from my_env import make_env, make_highway_env, MAP_PRESETS, HIGHWAY_BLOCK_SEQUENCE
from logger import RunLogger

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="MAE252 – Manual driving sandbox")

# -- mode selection --
parser.add_argument("--highway",  action="store_true",
                    help="Launch the custom long highway environment (overrides --map)")

# -- generic map args (ignored when --highway is set) --
parser.add_argument("--map",      default="mixed",     choices=list(MAP_PRESETS.keys()),
                    help="Map layout preset (default: mixed)")

# -- shared args --
parser.add_argument("--lanes",    type=int,   default=3,   help="Base lanes per road (default: 3)")
parser.add_argument("--seed",     type=int,   default=0,   help="Scenario seed (default: 0)")
parser.add_argument("--traffic",  type=float, default=0.15,
                    help="Traffic density 0-1 (default: 0.15)")

# -- highway-only args --
parser.add_argument("--incident", type=float, default=0.5,
                    help="[highway only] Incident probability 0-1 (default: 0.5)")

# -- logging --
parser.add_argument("--no-log", action="store_true",
                    help="Disable CSV telemetry logging")
parser.add_argument("--run-name", type=str, default="",
                    help="Optional label appended to the log folder name")
parser.add_argument("--record", action="store_true",
                    help="Save a 3-D window MP4 video to the log folder")
parser.add_argument("--debug", action="store_true",
                    help="Enable MetaDrive debug mode (extra log output; press 1 to toggle physics colliders)")

args = parser.parse_args()

# ── print header ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  MAE252 MetaDrive – Manual Driving Sandbox")
print("=" * 60)

if args.highway:
    print(f"  Mode         : HIGHWAY  (long, incidents + ramps)")
    print(f"  Block seq    : {HIGHWAY_BLOCK_SEQUENCE}")
    print(f"  Base lanes   : {args.lanes}")
    print(f"  Traffic      : {args.traffic}")
    print(f"  Incident prob: {args.incident}  (roadside cones / stopped cars)")
    print(f"  Seed         : {args.seed}")
else:
    print(f"  Mode         : SANDBOX")
    print(f"  Map preset   : {args.map}  →  {MAP_PRESETS[args.map]}")
    print(f"  Lanes        : {args.lanes}")
    print(f"  Traffic      : {args.traffic}")
    print(f"  Seed         : {args.seed}")

print("=" * 60)
print("  Controls: W/S/A/D or arrow keys | R = reset | Q = quit")
if not args.no_log:
    run_label = args.run_name or ("highway" if args.highway else args.map)
    print(f"  Logging to : logs/{run_label}_<timestamp>/telemetry.csv")
print("=" * 60)

# ── build env ────────────────────────────────────────────────────────────────
if args.highway:
    env = make_highway_env(
        base_lanes=args.lanes,
        traffic_density=args.traffic,
        accident_prob=args.incident,
        seed=args.seed,
        use_render=True,
        manual_control=True,
        debug=args.debug,
    )
else:
    env = make_env(
        map_preset=args.map,
        num_lanes=args.lanes,
        traffic_density=args.traffic,
        seed=args.seed,
        use_render=True,
        manual_control=True,
        debug=args.debug,
    )

# ── logger setup ────────────────────────────────────────────────────────────
run_label = args.run_name or ("highway" if args.highway else args.map)
logger = RunLogger(run_label, record_video=args.record) if not args.no_log else None

# ── run loop ─────────────────────────────────────────────────────────────
obs, info = env.reset(seed=args.seed)

try:
    while True:
        # manual_control=True: keyboard drives the car; env.step() just ticks the sim.
        obs, reward, terminated, truncated, info = env.step([0, 0])

        if logger:
            logger.record(env, reward=reward, info=info)

        if terminated or truncated:
            cost = info.get("cost", 0)
            print(f"Episode ended (cumulative cost: {cost:.1f}) – resetting …")
            obs, info = env.reset()

finally:
    if logger:
        logger.close()
    env.close()
    print("Closed cleanly.")
