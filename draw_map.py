"""
draw_map.py  –  Save a top-down PNG of any map preset or the custom highway
============================================================================
Usage:
    python draw_map.py                  # draws the highway map (default)
    python draw_map.py --map roundabout
    python draw_map.py --map mixed
    python draw_map.py --seed 5         # different random seed

Output: map_<name>.png  in the current folder
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from metadrive.utils.draw_top_down_map import draw_top_down_map
from my_env import make_env, make_highway_env, MAP_PRESETS
import os

parser = argparse.ArgumentParser(description="Save a top-down map image")
parser.add_argument("--map",  default=None, choices=list(MAP_PRESETS.keys()),
                    help="Generic map preset. Omit to draw the custom highway.")
parser.add_argument("--lanes",  type=int,   default=3,  help="Base lane count (default: 3)")
parser.add_argument("--seed",   type=int,   default=0,  help="Scenario seed (default: 0)")
parser.add_argument("--size",   type=int,   default=2048, help="Image resolution in pixels (default: 2048)")
parser.add_argument("--out",    type=str,   default=None, help="Output filename (default: map_<name>.png)")
args = parser.parse_args()

# ── build env (no window, no manual control) ──────────────────────────────────
if args.map is None:
    name = "highway"
    print(f"Building custom highway map (seed={args.seed}) …")
    env = make_highway_env(
        base_lanes=args.lanes,
        traffic_density=0.0,   # no traffic – cleaner map view
        accident_prob=0.5,
        seed=args.seed,
        use_render=False,
        manual_control=False,
    )
else:
    name = args.map
    print(f"Building map preset '{name}' (seed={args.seed}) …")
    env = make_env(
        map_preset=args.map,
        num_lanes=args.lanes,
        traffic_density=0.0,
        seed=args.seed,
        use_render=False,
        manual_control=False,
    )

# ── render the map ────────────────────────────────────────────────────────────
env.reset(seed=args.seed)

img = draw_top_down_map(
    env.current_map,
    resolution=(args.size, args.size),
    semantic_map=False,   # False → returns numpy array (RGB); True → pygame Surface
)

env.close()

# ── save figure ───────────────────────────────────────────────────────────────
out_file = args.out or f"map_{name}_seed{args.seed}.png"

fig, ax = plt.subplots(figsize=(14, 14), dpi=150)
ax.imshow(img, cmap="bone")
ax.set_title(f"MetaDrive Top-Down Map: {name}  (seed={args.seed})", fontsize=16, pad=12)
ax.axis("off")

# legend
legend_items = [
    mpatches.Patch(color="white",  label="Road / Lane"),
    mpatches.Patch(color="gray",   label="Shoulder / Boundary"),
    mpatches.Patch(color="black",  label="Off-road"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=10,
          facecolor="#222", labelcolor="white", framealpha=0.8)

plt.tight_layout()
os.makedirs("maps", exist_ok=True)
out_file = os.path.join("maps", out_file)
plt.savefig(out_file, bbox_inches="tight")
plt.close()

print(f"Saved → {out_file}")
