using System;
using KRPC.Service.Attributes;
using UnityEngine;

namespace KspRl
{
    [DefaultExecutionOrder(-10000)]
    public sealed class CheckpointHook : MonoBehaviour
    {
        private static CheckpointHook instance;
        private bool flightReadySubscribed;
        private bool stateLoadSubscribed;

        internal static void EnsureInstalled()
        {
            if (instance != null)
                return;

            // Arm runs on the game thread after the kRPC server is available.
            // Keep the listener on its own root object across flight reloads.
            var host = new GameObject("KspRlCheckpointHook");
            try
            {
                DontDestroyOnLoad(host);
                var listener = host.AddComponent<CheckpointHook>();
                // Use instance delegates so KSP's event wrapper has a target.
                GameEvents.onFlightReady.Add(listener.HandleFlightReady);
                listener.flightReadySubscribed = true;
                GameEvents.onGameStateLoad.Add(listener.HandleGameStateLoad);
                listener.stateLoadSubscribed = true;
                instance = listener;
                Debug.Log("[KSP-RL checkpoint] Flight-ready hook installed on demand.");
            }
            catch
            {
                Destroy(host);
                throw;
            }
        }

        private void HandleFlightReady()
        {
            CheckpointControl.OnFlightReady();
        }

        private void HandleGameStateLoad(ConfigNode state)
        {
            CheckpointControl.OnGameStateLoad();
        }

        public void Update()
        {
            CheckpointControl.Expire();
            if (CheckpointControl.StartQueuedLoad())
                return;
            CheckpointControl.TryPauseLoadedFlight("Update");
        }

        public void FixedUpdate()
        {
            CheckpointControl.TryPauseLoadedFlight("FixedUpdate");
        }

        public void OnDestroy()
        {
            if (flightReadySubscribed)
                GameEvents.onFlightReady.Remove(HandleFlightReady);
            if (stateLoadSubscribed)
                GameEvents.onGameStateLoad.Remove(HandleGameStateLoad);
            if (ReferenceEquals(instance, this))
                instance = null;
        }
    }

    [KRPCService]
    public static class CheckpointControl
    {
        private static string token;
        private static string saveName;
        private static double savedUt;
        private static double pausedUt;
        private static string status;
        private static DateTime expires;
        private static Game armedGame;
        private static Vessel armedVessel;
        private static double lastUt;
        private static bool sawStateLoad;
        private static bool sawFlightReady;
        private static bool sawRewind;
        private static string pauseSource;
        private static bool loadQueued;
        private static string loadError;

        [KRPCProcedure]
        public static void BeginLoad(string request, string name, double ut)
        {
            if (string.IsNullOrEmpty(name) || !System.Text.RegularExpressions.Regex.IsMatch(
                    name, @"\A[A-Za-z0-9_-]+\z"))
                throw new ArgumentException("Expected a named checkpoint without a path or extension.");

            CheckpointHook.EnsureInstalled();
            Expire();
            // A reconnect can supersede an interrupted request on this worker.
            if (status == "paused")
                FlightDriver.SetPause(false);
            token = request;
            saveName = name;
            savedUt = ut;
            armedGame = HighLogic.CurrentGame;
            armedVessel = FlightGlobals.ActiveVessel;
            lastUt = Planetarium.GetUniversalTime();
            sawStateLoad = false;
            sawFlightReady = false;
            sawRewind = false;
            pauseSource = null;
            loadError = null;
            loadQueued = true;
            status = "armed";
            expires = DateTime.UtcNow.AddSeconds(60);
            Debug.Log($"[KSP-RL checkpoint] Armed and queued {saveName}, token={token}, saved UT={savedUt:F3}");
        }

        internal static bool StartQueuedLoad()
        {
            if (status != "armed" || !loadQueued)
                return false;

            // Launch the scene change from Unity, outside the kRPC handler.
            // There is no synchronous SpaceCenter.Load call waiting for flight-ready.
            loadQueued = false;
            try
            {
                var game = GamePersistence.LoadGame(saveName, HighLogic.SaveFolder, true, true);
                if (game == null || game.flightState == null || game.flightState.activeVesselIdx < 0)
                    throw new InvalidOperationException($"Save {saveName} has no loadable active vessel.");
                Debug.Log($"[KSP-RL checkpoint] Loading {saveName}, token={token}");
                FlightDriver.StartAndFocusVessel(game, game.flightState.activeVesselIdx);
            }
            catch (Exception error)
            {
                loadError = error.Message;
                status = "load_failed";
                Debug.LogError($"[KSP-RL checkpoint] Failed to load {saveName}: {error}");
            }
            return true;
        }

        internal static void OnFlightReady()
        {
            if (status != "armed")
                return;
            sawFlightReady = true;
            TryPauseLoadedFlight("onFlightReady");
        }

        internal static void OnGameStateLoad()
        {
            if (status == "armed")
                sawStateLoad = true;
        }

        internal static void TryPauseLoadedFlight(string source)
        {
            if (status != "armed" || loadQueued || !HighLogic.LoadedSceneIsFlight)
                return;

            var ut = Planetarium.GetUniversalTime();
            if (ut < lastUt - 0.02)
                sawRewind = true;
            lastUt = ut;

            var vessel = FlightGlobals.ActiveVessel;
            if (vessel == null || !vessel.loaded || vessel.rootPart == null)
                return;

            var flightChanged = !ReferenceEquals(armedGame, HighLogic.CurrentGame)
                || !ReferenceEquals(armedVessel, vessel);
            // A scene event can precede the new active vessel. Require the
            // saved timeline as well before pausing an in-place load.
            if (ut < savedUt - 0.10 || ut > savedUt + 2.50
                || !(sawFlightReady || flightChanged || sawRewind))
                return;

            try
            {
                FlightDriver.SetPause(true);
                pausedUt = Planetarium.GetUniversalTime();
                pauseSource = source;
                status = "paused";
                Debug.Log($"[KSP-RL checkpoint] Paused {saveName}, token={token}, " +
                          $"source={source}, saved UT={savedUt:F3}, live UT={pausedUt:F3}, " +
                          $"delta={pausedUt - savedUt:+0.000;-0.000;0.000}s");
            }
            catch (Exception error)
            {
                status = "pause_failed";
                Debug.LogError($"[KSP-RL checkpoint] Failed to pause {saveName}: {error}");
            }
        }

        [KRPCProcedure]
        public static string GetStatus(string request)
        {
            return token == request ? status : "unknown";
        }

        [KRPCProcedure]
        public static string GetDiagnostics(string request)
        {
            if (token != request)
                return "request unknown";
            var vessel = FlightGlobals.ActiveVessel;
            return $"state={status}, scene={HighLogic.LoadedScene}, " +
                   $"gameStateLoad={sawStateLoad}, flightReady={sawFlightReady}, " +
                   $"rewind={sawRewind}, savedUT={savedUt:F3}, lastUT={lastUt:F3}, " +
                   $"activeVessel={(vessel == null ? "<none>" : vessel.vesselName)}, " +
                   $"pauseSource={pauseSource ?? "<none>"}, loadError={loadError ?? "<none>"}";
        }

        [KRPCProcedure]
        public static double GetPausedUt(string request)
        {
            if (token != request || status != "paused")
                throw new InvalidOperationException("Checkpoint pause has not been acknowledged.");
            return pausedUt;
        }

        [KRPCProcedure]
        public static void Release(string request)
        {
            if (token != request || status != "paused")
                throw new InvalidOperationException("No checkpoint pause is held for this request.");
            FlightDriver.SetPause(false);
            Debug.Log($"[KSP-RL checkpoint] Released {saveName}, token={token}");
            Clear();
        }

        [KRPCProcedure]
        public static void Cancel(string request)
        {
            if (token != request)
                return;
            if (status == "paused")
                FlightDriver.SetPause(false);
            Clear();
        }

        internal static void Expire()
        {
            if (status != "armed" || DateTime.UtcNow < expires)
                return;

            status = "expired";
            Debug.LogWarning($"[KSP-RL checkpoint] Load expired for {saveName}, token={token}");
        }

        private static void Clear()
        {
            token = null;
            saveName = null;
            status = null;
            armedGame = null;
            armedVessel = null;
            loadQueued = false;
            loadError = null;
        }
    }
}
