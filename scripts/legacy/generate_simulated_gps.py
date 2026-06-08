import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


OUTPUT_DIR = Path("data/raw/gps_simulated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STEP_SECONDS = 1
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


ROUTES = {
    # S01 - Complex -> AEM, tram, morning
    "gps_complex_aem.csv": {
        "route_id": "complex_aem",
        "transport_mode": "tram",
        "start_time": datetime(2026, 5, 14, 8, 21, 0),
        "duration_minutes": 32,
        "avg_speed": 20,
        "speed_std": 7,
        "points": [
            (45.7470, 21.2390),  # Complex
            (45.7455, 21.2430),
            (45.7430, 21.2485),
            (45.7405, 21.2545),
            (45.7365, 21.2605),
            (45.7320, 21.2645),
            (45.7290, 21.2675),  # AEM
        ],
    },

    # S01 - Complex -> AEM, tram, evening variant
    "gps_complex_aem_evening.csv": {
        "route_id": "complex_aem",
        "transport_mode": "tram",
        "start_time": datetime(2026, 5, 14, 18, 12, 0),
        "duration_minutes": 38,
        "avg_speed": 17,
        "speed_std": 7,
        "points": [
            (45.7470, 21.2390),
            (45.7455, 21.2430),
            (45.7430, 21.2485),
            (45.7405, 21.2545),
            (45.7365, 21.2605),
            (45.7320, 21.2645),
            (45.7290, 21.2675),
        ],
    },

    # S02 - Piata Victoriei -> Iulius, walking
    "gps_victoriei_iulius.csv": {
        "route_id": "victoriei_iulius",
        "transport_mode": "walking",
        "start_time": datetime(2026, 5, 19, 14, 18, 0),
        "duration_minutes": 30,
        "avg_speed": 4.8,
        "speed_std": 0.8,
        "points": [
            (45.7537, 21.2257),  # Piata Victoriei
            (45.7560, 21.2262),
            (45.7588, 21.2269),
            (45.7620, 21.2278),
            (45.7660, 21.2288),
            (45.7711, 21.2295),  # Iulius Town
        ],
    },

    # S02 - Piata Victoriei -> Iulius, walking, evening variant
    "gps_victoriei_iulius_evening.csv": {
        "route_id": "victoriei_iulius",
        "transport_mode": "walking",
        "start_time": datetime(2026, 5, 19, 18, 30, 0),
        "duration_minutes": 34,
        "avg_speed": 4.4,
        "speed_std": 0.9,
        "points": [
            (45.7537, 21.2257),
            (45.7560, 21.2262),
            (45.7588, 21.2269),
            (45.7620, 21.2278),
            (45.7660, 21.2288),
            (45.7711, 21.2295),
        ],
    },

    # S03 - Iulius -> Aeroport, car
    "gps_iulius_airport.csv": {
        "route_id": "iulius_airport",
        "transport_mode": "car",
        "start_time": datetime(2026, 5, 21, 17, 44, 0),
        "duration_minutes": 34,
        "avg_speed": 48,
        "speed_std": 14,
        "points": [
            (45.7711, 21.2295),  # Iulius
            (45.7735, 21.2425),
            (45.7765, 21.2600),
            (45.7790, 21.2850),
            (45.7820, 21.3150),
            (45.7851, 21.3370),  # Aeroport
        ],
    },

    # S03 - Iulius -> Aeroport, car, continuation
    "gps_iulius_airport_part2.csv": {
        "route_id": "iulius_airport",
        "transport_mode": "car",
        "start_time": datetime(2026, 5, 21, 17, 46, 0),
        "duration_minutes": 26,
        "avg_speed": 52,
        "speed_std": 13,
        "points": [
            (45.7760, 21.2700),
            (45.7782, 21.2870),
            (45.7805, 21.3050),
            (45.7830, 21.3220),
            (45.7851, 21.3370),
        ],
    },

    # S03 - Iulius -> Aeroport, morning variant
    "gps_iulius_airport_morning.csv": {
        "route_id": "iulius_airport",
        "transport_mode": "car",
        "start_time": datetime(2026, 5, 21, 8, 20, 0),
        "duration_minutes": 38,
        "avg_speed": 42,
        "speed_std": 15,
        "points": [
            (45.7711, 21.2295),
            (45.7735, 21.2425),
            (45.7765, 21.2600),
            (45.7790, 21.2850),
            (45.7820, 21.3150),
            (45.7851, 21.3370),
        ],
    },

    # S04 - Gara -> Piata Traian, tram
    "gps_gara_traian.csv": {
        "route_id": "gara_traian",
        "transport_mode": "tram",
        "start_time": datetime(2026, 5, 13, 18, 16, 0),
        "duration_minutes": 36,
        "avg_speed": 18,
        "speed_std": 6,
        "points": [
            (45.7575, 21.2076),  # Gara de Nord
            (45.7548, 21.2145),
            (45.7535, 21.2210),
            (45.7548, 21.2280),
            (45.7565, 21.2350),
            (45.7580, 21.2418),  # Piata Traian
        ],
    },

    # S04 - Gara -> Piata Traian, morning variant
    "gps_gara_traian_morning.csv": {
        "route_id": "gara_traian",
        "transport_mode": "tram",
        "start_time": datetime(2026, 5, 13, 8, 5, 0),
        "duration_minutes": 42,
        "avg_speed": 16,
        "speed_std": 7,
        "points": [
            (45.7575, 21.2076),
            (45.7548, 21.2145),
            (45.7535, 21.2210),
            (45.7548, 21.2280),
            (45.7565, 21.2350),
            (45.7580, 21.2418),
        ],
    },

    # S05 - Bega / parcuri, walking
    "gps_bega_parcuri.csv": {
        "route_id": "bega_parcuri",
        "transport_mode": "walking",
        "start_time": datetime(2026, 5, 15, 11, 44, 0),
        "duration_minutes": 31,
        "avg_speed": 4.5,
        "speed_std": 0.7,
        "points": [
            (45.7475, 21.2200),  # zona Catedrala / Bega
            (45.7472, 21.2245),
            (45.7470, 21.2290),
            (45.7465, 21.2330),
            (45.7460, 21.2365),
            (45.7455, 21.2400),  # zona Michelangelo
        ],
    },

    # S06 - Victoriei -> Michelangelo -> Complex, bus
    "gps_01_victoriei_michelangelo_complex.csv": {
        "route_id": "victoriei_michelangelo_complex",
        "transport_mode": "bus",
        "start_time": datetime(2026, 5, 16, 10, 58, 0),
        "duration_minutes": 27,
        "avg_speed": 23,
        "speed_std": 8,
        "points": [
            (45.7537, 21.2257),  # Victoriei
            (45.7520, 21.2280),
            (45.7505, 21.2308),  # Michelangelo
            (45.7485, 21.2340),
            (45.7470, 21.2390),  # Complex
        ],
    },

    # S07 - Complex -> Giroc, bus
    "gps_complex_giroc.csv": {
        "route_id": "complex_giroc",
        "transport_mode": "bus",
        "start_time": datetime(2026, 5, 16, 17, 20, 0),
        "duration_minutes": 38,
        "avg_speed": 27,
        "speed_std": 9,
        "points": [
            (45.7470, 21.2390),  # Complex
            (45.7420, 21.2425),
            (45.7365, 21.2475),
            (45.7305, 21.2525),
            (45.7245, 21.2570),
            (45.7190, 21.2605),  # Giroc aproximativ
        ],
    },

    # S08 - Iulius -> Dumbravita, car/bus
    "gps_iulius_dumbravita.csv": {
        "route_id": "iulius_dumbravita",
        "transport_mode": "car",
        "start_time": datetime(2026, 5, 21, 7, 5, 0),
        "duration_minutes": 24,
        "avg_speed": 36,
        "speed_std": 11,
        "points": [
            (45.7711, 21.2295),  # Iulius
            (45.7765, 21.2315),
            (45.7820, 21.2345),
            (45.7880, 21.2370),
            (45.7940, 21.2400),  # Dumbravita aproximativ
        ],
    },

    # S09 - Shopping City / Sagului, bus/car
    "gps_shopping_city_centru.csv": {
        "route_id": "shopping_city_centru",
        "transport_mode": "bus",
        "start_time": datetime(2026, 5, 30, 19, 11, 0),
        "duration_minutes": 34,
        "avg_speed": 25,
        "speed_std": 9,
        "points": [
            (45.7248, 21.1990),  # Shopping City / Calea Sagului aprox.
            (45.7315, 21.2070),
            (45.7390, 21.2145),
            (45.7475, 21.2210),
            (45.7537, 21.2257),  # Victoriei
        ],
    },

    # S10 - Complex -> AEM reverse, tram, evening
    "gps_aem_complex_evening.csv": {
        "route_id": "aem_complex",
        "transport_mode": "tram",
        "start_time": datetime(2026, 5, 29, 18, 18, 0),
        "duration_minutes": 34,
        "avg_speed": 19,
        "speed_std": 7,
        "points": [
            (45.7290, 21.2675),  # AEM
            (45.7320, 21.2645),
            (45.7365, 21.2605),
            (45.7405, 21.2545),
            (45.7430, 21.2485),
            (45.7455, 21.2430),
            (45.7470, 21.2390),  # Complex
        ],
    },
}


def is_peak_hour(timestamp):
    return int(timestamp.hour in [7, 8, 9, 16, 17, 18, 19])


def get_noise_for_mode(transport_mode):
    if transport_mode == "walking":
        return 0.000025
    if transport_mode in ["tram", "bus"]:
        return 0.000045
    if transport_mode == "car":
        return 0.000060
    return 0.000050


def interpolate_route(points, total_steps, i):
    progress = i / total_steps

    segment_count = len(points) - 1
    segment_float = progress * segment_count
    segment_index = min(int(segment_float), segment_count - 1)
    local_progress = segment_float - segment_index

    lat1, lon1 = points[segment_index]
    lat2, lon2 = points[segment_index + 1]

    lat = lat1 + (lat2 - lat1) * local_progress
    lon = lon1 + (lon2 - lon1) * local_progress

    return lat, lon


def generate_base_speed(avg_speed, speed_std, transport_mode, i, total_steps):
    progress = i / total_steps

    if transport_mode == "walking":
        speed = np.random.normal(avg_speed, speed_std)

        # mici opriri random, ca la treceri/pauze
        if np.random.rand() < 0.025:
            speed = np.random.normal(0.8, 0.35)

        speed = min(speed, 7.0)

    elif transport_mode == "tram":
        speed = np.random.normal(avg_speed, speed_std)

        # oprire in statie la fiecare ~3 minute
        stop_every = 180
        stop_duration = 18

        if i % stop_every < stop_duration:
            speed = np.random.normal(1.2, 0.6)

        # incetiniri random
        if np.random.rand() < 0.035:
            speed *= 0.35

    elif transport_mode == "bus":
        speed = np.random.normal(avg_speed, speed_std)

        # bus-ul opreste putin mai des
        stop_every = 150
        stop_duration = 20

        if i % stop_every < stop_duration:
            speed = np.random.normal(1.0, 0.6)

        # trafic/semafoare
        if np.random.rand() < 0.07:
            speed *= 0.45

    elif transport_mode == "car":
        speed = np.random.normal(avg_speed, speed_std)

        # semafor / trafic
        if np.random.rand() < 0.07:
            speed = np.random.normal(8, 4)

        # bucati mai libere
        if np.random.rand() < 0.10:
            speed += np.random.normal(14, 5)

    else:
        speed = np.random.normal(avg_speed, speed_std)

    # pornire si final mai lente
    if progress < 0.04 or progress > 0.96:
        speed *= 0.35

    return max(0, speed)


def smooth_speeds(speeds):
    """
    Netestez un pic vitezele ca sa nu sara irealist de la 0 la 70.
    """
    smoothed = []

    for i, speed in enumerate(speeds):
        if i == 0:
            smoothed.append(speed)
        else:
            previous = smoothed[-1]
            # limitam cresterea/scaderea brusca
            max_delta = 8
            delta = speed - previous

            if delta > max_delta:
                speed = previous + max_delta
            elif delta < -max_delta:
                speed = previous - max_delta

            smoothed.append(max(0, speed))

    return smoothed


def generate_route(filename, config):
    total_steps = int((config["duration_minutes"] * 60) / STEP_SECONDS)

    coordinates = []
    speeds = []

    for i in range(total_steps):
        lat, lon = interpolate_route(config["points"], total_steps, i)

        noise = get_noise_for_mode(config["transport_mode"])
        lat += np.random.normal(0, noise)
        lon += np.random.normal(0, noise)

        speed = generate_base_speed(
            config["avg_speed"],
            config["speed_std"],
            config["transport_mode"],
            i,
            total_steps
        )

        coordinates.append((lat, lon))
        speeds.append(speed)

    speeds = smooth_speeds(speeds)

    rows = []

    for i, ((lat, lon), speed) in enumerate(zip(coordinates, speeds)):
        timestamp = config["start_time"] + timedelta(seconds=i * STEP_SECONDS)

        rows.append({
            "timestamp": timestamp,
            "latitude": round(lat, 7),
            "longitude": round(lon, 7),
            "speed_kmh": round(speed, 2),
            "route_id": config["route_id"],
            "transport_mode": config["transport_mode"],
            "is_peak_hour": is_peak_hour(timestamp),
            "source": "synthetic_gps"
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / filename, index=False)

    print(
        f"Generated {filename} | "
        f"rows={len(df)} | "
        f"route={config['route_id']} | "
        f"mode={config['transport_mode']} | "
        f"start={config['start_time']}"
    )


def main():
    for filename, config in ROUTES.items():
        generate_route(filename, config)

    print()
    print(f"All GPS files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()