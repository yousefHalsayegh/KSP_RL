import time

import krpc


START_DELAY = 5.0
MAX_TEST_TIME = 30.0       # Wall-clock timeout
TURN_ALTITUDE_GAIN = 100.0 # Metres above starting altitude
TURN_DURATION = 5.0        # Game seconds
LAUNCH_THROTTLE = 1.0

def apply_action(vessel, throttle, pitch):
    vessel.control.throttle = max(0.0, min(1.0, throttle))
    vessel.auto_pilot.target_pitch_and_heading(
        max(0.0, min(90.0, pitch)),
        90.0,  # East
    )


def step(conn, vessel, throttle, pitch, duration=0.2):
    sc = conn.space_center
    flight = vessel.flight(vessel.orbit.body.reference_frame)

    start_ut = sc.ut
    deadline = time.monotonic() + 5.0

    apply_action(vessel, throttle, pitch)

    while sc.ut - start_ut < duration:
        if time.monotonic() >= deadline:
            raise TimeoutError("Game time stopped advancing.")
        time.sleep(0.01)

    return {
        "ut": sc.ut,
        "altitude": flight.mean_altitude,
        "vertical_speed": flight.vertical_speed,
        "horizontal_speed": flight.horizontal_speed,
        "apoapsis": vessel.orbit.apoapsis_altitude,
        "periapsis": vessel.orbit.periapsis_altitude,
        "mass": vessel.mass,
    }



def main():
    conn = krpc.connect(name="Launchpad Control Test")

    vessel = conn.space_center.active_vessel
    control = vessel.control
    ap = vessel.auto_pilot
    sc = conn.space_center

    # Local orientation and surface-relative motion.
    surface_frame = vessel.surface_reference_frame
    #logging 
    solid_fuel = conn.add_stream(vessel.resources.amount, "SolidFuel")
    solid_fuel.rate = 10
    boosters_separated = False

    try:
        print(f"Connected to: {vessel.name}")
        print("Next stage must ignite engines and release clamps.")
        print(f"Launching in {START_DELAY:.0f} seconds. Switch to KSP.")
        time.sleep(START_DELAY)

        # Clear old commands.
        ap.engaged = False
        control.throttle = 0.0
        control.pitch = 0.0
        control.yaw = 0.0
        control.roll = 0.0
        control.sas = False

        # Point straight up. In this frame, +X is up.
        ap.reference_frame = surface_frame
        ap.target_roll = float("nan")
        ap.target_direction = (1.0, 0.0, 0.0)
        ap.engaged = True


        # Launch: throttle command alone does not ignite inactive engines.
        control.throttle = LAUNCH_THROTTLE
        control.activate_next_stage()

        print("\nStage activated. Ascending vertically.")

        for _ in range(100):
            observation = step(conn, vessel, throttle=1.0, pitch=80.0)
            if (
                not boosters_separated
                and vessel.control.current_stage == 5
                and solid_fuel() <= 0.0001
            ):
                boosters_separated = True  # Prevent another staging command
                vessel.control.activate_next_stage()  # Stage 4: detach boosters
                print("Empty boosters detached; liquid engine continues.")
            print(observation)
        else:
            print("\nTimeout: test stopped before completion.")

    except KeyboardInterrupt:
        print("\nTest interrupted.")

    finally:
        try:
            control.throttle = 0.0
            ap.engaged = False
            control.pitch = 0.0
            control.yaw = 0.0
            control.roll = 0.0

            control.sas = True
            control.sas_mode = sc.SASMode.stability_assist

            print("Reverting to launch in 5 seconds...")
            time.sleep(5)

            if sc.can_revert_to_launch():
                sc.revert_to_launch()
        finally:
            conn.close()


if __name__ == "__main__":
    main()