using System;
using System.Collections;
using System.Linq;
using UnityEngine;

namespace KSPRL
{
    [KSPAddon(KSPAddon.Startup.AllGameScenes, false)]
    public sealed class WorkerServerSelector : MonoBehaviour
    {
        private static string ReadArgument(string prefix)
        {
            foreach (string argument in Environment.GetCommandLineArgs())
            {
                if (argument.StartsWith(prefix, StringComparison.Ordinal))
                    return argument.Substring(prefix.Length);
            }
            return null;
        }

        private IEnumerator Start()
        {
            string name = ReadArgument("--ksp-rl-server=");
            if (string.IsNullOrWhiteSpace(name))
                yield break;

            int rpcPort;
            int streamPort;

            if (
                !int.TryParse(ReadArgument("--ksp-rl-rpc-port="), out rpcPort) ||
                !int.TryParse(ReadArgument("--ksp-rl-stream-port="), out streamPort) ||
                rpcPort < 1 || rpcPort > 65535 ||
                streamPort < 1 || streamPort > 65535 ||
                rpcPort == streamPort
            )
            {
                Debug.LogError("[KSPRL] Missing or invalid worker ports.");
                yield break;
            }

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

            KRPC.Server.Server selected = null;

            try
            {
                var core = KRPC.Core.Instance;
                var configuration = KRPC.Configuration.Instance;
                var matches = configuration.Servers
                    .Where(server => server.Name == name)
                    .ToList();

                if (matches.Count > 1)
                    throw new InvalidOperationException(
                        "Multiple saved servers have the name " + name
                    );

                var definition = matches.SingleOrDefault();
                bool changed = false;

                if (definition == null)
                {
                    definition = new KRPC.Configuration.Server();
                    definition.Name = name;
                    configuration.Servers.Add(definition);
                    changed = true;
                }

                string existingRpc;
                string existingStream;
                string existingAddress;

                changed |=
                    definition.Protocol != KRPC.Server.Protocol.ProtocolBuffersOverTCP ||
                    !definition.Settings.TryGetValue("rpc_port", out existingRpc) ||
                    existingRpc != rpcPort.ToString() ||
                    !definition.Settings.TryGetValue("stream_port", out existingStream) ||
                    existingStream != streamPort.ToString() ||
                    !definition.Settings.TryGetValue("address", out existingAddress) ||
                    existingAddress != "127.0.0.1";

                definition.Protocol = KRPC.Server.Protocol.ProtocolBuffersOverTCP;
                definition.Settings["address"] = "127.0.0.1";
                definition.Settings["rpc_port"] = rpcPort.ToString();
                definition.Settings["stream_port"] = streamPort.ToString();

                // Stop other listeners without deleting their definitions.
                foreach (var server in core.Servers.ToArray())
                {
                    if (server.Id != definition.Id)
                        server.Stop();
                }

                selected = core.Servers.FirstOrDefault(
                    server => server.Id == definition.Id
                );

                if (selected == null)
                    core.Add(definition.Create());
                else if (changed)
                    core.Replace(definition.Create());

                selected = core.Servers.First(
                    server => server.Id == definition.Id
                );

                // Preserve the connection across ordinary scene changes.
                if (!selected.Running)
                    selected.Start();

                Debug.Log("[KSPRL] Ready: " + name + " | " + selected.Address);
            }
            catch (Exception error)
            {
                if (selected != null)
                    selected.Stop();

                Debug.LogError("[KSPRL] Server setup failed: " + error);
            }
        }
    }
}