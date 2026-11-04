using System;
using System.Collections;
using System.Linq;
using UnityEngine;

namespace KSPRL
{
    [KSPAddon(KSPAddon.Startup.AllGameScenes, false)]
    public sealed class WorkerServerSelector : MonoBehaviour
    {
        private IEnumerator Start()
        {
            const string prefix = "--ksp-rl-server=";
            string serverName = null;

            foreach (string argument in Environment.GetCommandLineArgs())
            {
                if (argument.StartsWith(prefix, StringComparison.Ordinal))
                    serverName = argument.Substring(prefix.Length);
            }

            if (string.IsNullOrWhiteSpace(serverName))
                yield break;

            float deadline = Time.realtimeSinceStartup + 30f;
            while (KRPC.Addon.Instance == null)
            {
                if (Time.realtimeSinceStartup >= deadline)
                {
                    Debug.LogError("[KSPRL] kRPC did not initialize.");
                    yield break;
                }
                yield return null;
            }

            var core = KRPC.Core.Instance;
            var matches = core.Servers
                .Where(server => server.Name == serverName)
                .ToList();

            if (matches.Count != 1)
            {
                Debug.LogError(
                    "[KSPRL] Expected exactly one saved server named '" +
                    serverName + "', found " + matches.Count + "."
                );
                yield break;
            }

            var selected = matches[0];
            foreach (var server in core.Servers)
            {
                if (server.Id != selected.Id)
                    server.Stop();
            }

            if (selected.Running)
                yield break;

            try
            {
                selected.Start();
                Debug.Log(
                    "[KSPRL] Started " + selected.Name + ": " +
                    selected.Address
                );
            }
            catch (Exception error)
            {
                selected.Stop();
                Debug.LogError(
                    "[KSPRL] Failed to start " + serverName + ": " + error
                );
            }
        }
    }
}