"""
my_env.py  –  MAE252 custom MetaDrive environments
==================================================
Provides two factory functions:
  make_env()         – generic sandbox with any map preset
  make_highway_env() – long custom highway with incidents & ramps

Map block IDs (use as a string, e.g. map="SXO"):
    S  Straight          C  Circular (curve)
    X  Intersection      O  Roundabout
    T  T-Intersection    r  InRamp
    R  OutRamp           y  Merge
    Y  Split             P  Parking Lot
    $  Tollgate

Or pass an integer to get that many randomly-assembled blocks, e.g. map=5.
"""

from metadrive import MetaDriveEnv
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod


# ── map presets you can choose from in drive.py ──────────────────────────────
MAP_PRESETS = {
    "straight":       "SSSSSSSSSS",  # ten straight segments
    "curve":          "CCCC",        # winding curves
    "intersection":   "SXS",         # straight → intersection → straight
    "roundabout":     "SOC",         # straight → roundabout → curve
    "highway":        "SrRSSS",      # basic on-ramp / off-ramp stretch
    "simple_highway": "SSrSRSSrSS",  # 2 on-ramps + 1 off-ramp (VLM demo map)
    "t_junction":     "STT",         # two T-intersections
    "mixed":          "SXOCrR",      # a bit of everything
    "random_3":       3,             # 3 randomly assembled blocks
    "random_5":       5,             # 5 randomly assembled blocks
}

# ── long highway map block sequence ──────────────────────────────────────────
# Layout
#   SS          : opening straights
#   r           : on-ramp  (traffic merges in, lane count +1)
#   SSS         : cruise at wider road
#   C           : gentle left-hand curve
#   SS          : straight after curve
#   R           : off-ramp (lane count -1)
#   SSS         : mid-highway cruise
#   r           : second on-ramp
#   SS          : straight
#   C           : gentle right-hand curve
#   SS          : straight
#   r           : third on-ramp
#   SSS         : long straight run
#   C           : final sweeping curve
#   SS          : straight
#   R           : second off-ramp
#   SS          : closing straights
# Total: 24 blocks, roughly 2+ km of highway
HIGHWAY_BLOCK_SEQUENCE = "SSSSSrSSSSCSSSRSSSrSSSSSrSSRSSSS"


def make_env(map_preset: str = "mixed",
             num_lanes: int = 4,
             lane_width: float = 3.5,
             traffic_density: float = 0.5,
             seed: int = 0,
             random_traffic: bool = True,
             use_render: bool = True,
             manual_control: bool = True,
             debug: bool = False) -> MetaDriveEnv:
    """
    Build and return a configured MetaDriveEnv.

    Parameters
    ----------
    map_preset      : key from MAP_PRESETS, or any raw string/int for map=
    num_lanes       : lanes per road section (1-4)
    lane_width      : width of each lane in metres
    traffic_density : 0.0 = no traffic, 0.1 = light, 0.5 = heavy
    seed            : which procedural scenario to load
    use_render      : open a 3-D window
    manual_control  : WASD / arrow keys to drive
    """
    map_value = MAP_PRESETS.get(map_preset, map_preset)  # fall back to raw value

    # Determine generate type: int → BIG_BLOCK_NUM, string → BIG_BLOCK_SEQUENCE
    if isinstance(map_value, int):
        generate_type = MapGenerateMethod.BIG_BLOCK_NUM
    else:
        generate_type = MapGenerateMethod.BIG_BLOCK_SEQUENCE

    config = dict(
        # ── map ──────────────────────────────────────────────────────────────
        map_config={
            BaseMap.GENERATE_TYPE: generate_type,
            BaseMap.GENERATE_CONFIG: map_value,
            BaseMap.LANE_WIDTH: lane_width,
            BaseMap.LANE_NUM: num_lanes,
        },

        # ── traffic ──────────────────────────────────────────────────────────
        traffic_density=traffic_density,        # random_traffic=False locks NPC routes to the seed → same scenario every run
        random_traffic=random_traffic,
        # ── rendering & control ──────────────────────────────────────────────
        use_render=use_render,
        manual_control=manual_control,

        # ── hide navigation waypoint markers ─────────────────────────────────
        # The planner does NOT follow MetaDrive's navigation waypoints;
        # hiding them removes visual clutter and prevents confusion.
        show_interface_navi_mark=False,   # top-level: hides UI navi mark
        vehicle_config=dict(
            show_navi_mark=False,         # vehicle-level: hides 3D checkpoint spheres
            show_line_to_navi_mark=False,

            # ── Sensors ──────────────────────────────────────────────────────
            # lidar: 240-ray rotating lidar, 50 m range.
            #   num_others=4  → the obs vector also encodes the 4 closest vehicles'
            #                   relative position + velocity (16 extra dims), giving
            #                   the planner's block/TTC detection richer signal.
            #   gaussian_noise / dropout_prob = 0 → clean signal for the baseline;
            #   set > 0 to simulate real sensor imperfection.
            lidar=dict(
                num_lasers=240,        # angular resolution: 1.5° per ray
                distance=50,           # max range [m]  – matches FRONT_RANGE_M
                num_others=4,          # closest-NPC kinematics appended to obs
                gaussian_noise=0.0,    # additive Gaussian noise on distance readings
                dropout_prob=0.0,      # fraction of rays randomly zeroed
            ),

            # side_detector: sideways-facing rays that measure distance to road
            #   borders (kerbs / barriers). 4 rays cover left-front, left-rear,
            #   right-front, right-rear.  These replace the two scalar
            #   lateral-distance values in the obs vector (4 dims instead of 2).
            side_detector=dict(num_lasers=4, distance=50,
                               gaussian_noise=0.0, dropout_prob=0.0),

            # lane_line_detector: detects lane markings ahead of the vehicle.
            #   4 rays sampling the current lane boundaries.  Replaces the
            #   single lateral-offset scalar in the obs vector (4 dims instead of 1).
            lane_line_detector=dict(num_lasers=4, distance=20,
                                    gaussian_noise=0.0, dropout_prob=0.0),
        ),

        # ── misc ─────────────────────────────────────────────────────────────
        num_scenarios=1,
        start_seed=seed,

        # Don't end the episode on crash – useful while exploring the map
        crash_vehicle_done=False,
        out_of_road_done=False,

        # No time limit – drive as long as you want
        horizon=None,

        # Debug mode: extra console output + press 1 to see physics colliders
        debug=debug,
    )

    return MetaDriveEnv(config)


def make_highway_env(lane_width: float = 3.5,
                    base_lanes: int = 3,
                    traffic_density: float = 0.08,
                    accident_prob: float = 0.0,
                    seed: int = 0,
                    use_render: bool = True,
                    manual_control: bool = True,
                    debug: bool = False) -> SafeMetaDriveEnv:
    """
    Long highway environment with:
    - ~2 km of road built from straights, gentle curves, on-ramps, off-ramps
    - Lane count varies naturally at each ramp junction
    - Roadside incidents (cones / stopped cars) on the RIGHT shoulder,
      controlled by accident_prob (0 = none, 1 = very dense)
    - Light background traffic so the road feels alive

    Uses SafeMetaDriveEnv so incidents are cost-tracked but do NOT
    terminate the episode — good for long exploratory drives.

    Parameters
    ----------
    lane_width      : width of each lane in metres (default 3.5)
    base_lanes      : lanes on the main carriageway (default 3)
    traffic_density : NPC traffic density 0-1 (default 0.5)
    accident_prob   : probability of an incident block 0-1 (default 0.5)
    seed            : scenario seed
    use_render      : open a 3-D window
    manual_control  : WASD / arrow keys to drive
    """
    config = dict(
        # ── map ──────────────────────────────────────────────────────────────
        map_config={
            BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
            BaseMap.GENERATE_CONFIG: HIGHWAY_BLOCK_SEQUENCE,
            BaseMap.LANE_WIDTH: lane_width,
            BaseMap.LANE_NUM: base_lanes,
        },

        # ── incidents on the right shoulder ──────────────────────────────────
        # accident_prob=0.5 means ~50% of eligible blocks get an obstacle
        accident_prob=accident_prob,
        # Static objects (cones, stopped cars) don't move
        static_traffic_object=True,

        # ── background traffic ───────────────────────────────────────────────
        traffic_density=traffic_density,
        traffic_mode="respawn",        # vehicles continuously circulate
        need_inverse_traffic=False,    # all traffic goes the same direction

        # ── safety / termination ─────────────────────────────────────────────
        # Don't end on crash — let you keep driving after hitting a cone
        crash_vehicle_done=False,
        crash_object_done=False,
        out_of_road_done=False,
        # Cost IS recorded so you can log safety metrics later
        crash_object_cost=1,
        crash_vehicle_cost=1,
        out_of_road_cost=1,

        # ── rendering & control ──────────────────────────────────────────────
        use_render=use_render,
        manual_control=manual_control,

        # ── misc ─────────────────────────────────────────────────────────────
        num_scenarios=1,
        start_seed=seed,

        # No time limit – drive the full map without auto-reset
        horizon=None,

        # Debug mode: extra console output + press 1 to see physics colliders
        debug=debug,
    )

    return SafeMetaDriveEnv(config)
