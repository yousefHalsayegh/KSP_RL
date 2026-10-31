
import time

from client import KSPClient
from environment import KSPEnvironment
import numpy as np

def test():
    connection = KSPClient(name="Environment Reward Test")

    try:
        env = KSPEnvironment(connection, task="orbit", episode_seconds=10)
        state = env.reset()

        print("Launching in 5 seconds. Switch to KSP.")
        time.sleep(5)
        

        for episode in range(2):
            print(f"\nEpisode {episode + 1}")

            # Request launch once.
            action = [1.0, 0.7, 0.0, 0.0, 1.0]
            while True:
                state, reward, terminated, truncated, info = env.step(action)
                observation = info["telemetry"]

                print(
                    f"State size: {len(state)} | "
                    f"Altitude: {observation['altitude']:.1f}m | "
                    f"Reward: {reward:+.6f}"
                )

                if terminated or truncated:
                    print(f"Episode ended: {info['end_reason']}")
                    break

            state = env.reset()
            print("Reset complete.")
    except KeyboardInterrupt:
        print("Test interrupted.")

if __name__ == "__main__":
    test()