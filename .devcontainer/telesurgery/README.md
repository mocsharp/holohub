# VS Code Dev Container for Telesurgery Application

This dev container is used to run the Telesurgery application.

## Running the Dev Container

To run the dev container, you can use the following command:

```bash
./dev_container vscode telesurgery
```

> [!IMPORTANT]
> Ensure that all devices are connected to the host machine before running the dev container.

## Configurations for Telesurgery

In the `devcontainer.json` file, you can find the following configurations that are specific to the Telesurgery application:

```json
"runArgs": [
        "--net=host",
        "--device=/dev/input/js0",
        "--device=/dev/input/event0",
        "--device=/dev/input/event1",
        "--device=/dev/input/event2",
        "--device=/dev/input/event3",
        "--device=/dev/input/event4",
        "--device=/dev/input/event5",
        "--device=/dev/input/event6",
        "--device=/dev/input/event7",
        "--device=/dev/input/event8",
        "--device=/dev/input/event9",
        "--device=/dev/input/event10",
        "--device=/dev/input/event11",
        "--device=/dev/input/event12",
        "--device=/dev/input/event13",
        "--device=/dev/input/event14",
        "--device=/dev/input/event15",
        "--volume=${localEnv:HOME}/rti_connext_dds-7.3.0:/opt/rti.com/rti_connext_dds-7.3.0/",
    //"<env>"
],
```

- `--device=/dev/input/js*`: These are the devices that will be used to read the joystick events.
- `--device=/dev/input/event*`: This is the device that will be used to read the keyboard and mouse events.
- `--volume=${localEnv:HOME}/rti_connext_dds-7.3.0:/opt/rti.com/rti_connext_dds-7.3.0/`: This is the path to the RTI Connext DDS installation.

> [!TIP]
> Your devices may be connected to a different path on your host machine. Please check the output of the `ls -l /dev/input` or `ls -l /dev/input/by-id` command in the terminal to find the correct path.

> [!WARNING]
> Some devices may require `sudo` permissions to access. Therefore, you may need to run the dev container with `sudo` privileges.
> The other options is to add yourself to the `input` group.
>
> ```bash
> sudo usermod -a -G input $USER
> newgrp input
> ```

For other configurations, please refer to the main [Dev Container README](../README.md).
