#!/usr/bin/env python3
"""
Azure IoT Bridge Example

This example shows how to initialize the AzureBridge, map a device to a ReGenNexus entity,
and send a device-to-cloud message. Requires the Azure IoT Device SDK for full functionality.
"""
import asyncio
from regennexus.bridges.azure_bridge import AzureBridge

async def main():
    # Replace with your Azure IoT Hub connection string
    connection_string = "<Your IoT Hub Connection String>"
    bridge = AzureBridge(connection_string)
    await bridge.initialize()

    if not bridge.azure_initialized:
        print("Azure bridge not initialized. Ensure Azure IoT Device SDK is installed and the connection string is correct.")
        return

    # Map a device ID to an internal entity ID
    device_id = "myDevice"
    entity_id = "myEntity"
    await bridge.map_device_to_entity(device_id, entity_id)
    print(f"Mapped device '{device_id}' to entity '{entity_id}'")

    # Send a device-to-cloud message
    payload = {"temperature": 23.5, "humidity": 62}
    print(f"Sending telemetry from '{device_id}': {payload}")
    await bridge.send_device_to_cloud_message(device_id, payload)

    # Allow some time for the message to be sent
    await asyncio.sleep(2)

    # Shut down the bridge connection
    await bridge.shutdown()
    print("Azure bridge shut down.")

if __name__ == '__main__':
    asyncio.run(main())