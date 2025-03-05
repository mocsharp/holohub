# DDS Telesurgery Application

The DDS Telesurgery application demonstrates how a surgeon can perform a telesurgery procedure using a Holoscan application to control a robot remotely.

The application can be run as either a surgeon or a robot. In either case, it will use both the [InputCommand](../../operators/dds/hid/InputCommand.idl) data topic and the [VideoFrame](../../operators/dds/video/VideoFrame.idl) data topic registered by the `DDSHIDPublisherOp` or `DDSHIDSubscriberOp` operators and the `DDSVideoPublisherOp` or `DDSVideoSubscriberOp` operators, respectively. These operators read and write Human Interface Device (HID) events and video frame data to/from the DDS databus.

When run as a surgeon, the source for the input events will come from the configured HID devices.

When run as a robot, the application will capture live video from the configured V4L2 device and use Holoviz to render the received HID events. The rendered data is also published to the DDS databus to the surgeon application.

## Prerequisites

- This application requires [RTI Connext](https://content.rti.com/l/983311/2024-04-30/pz1wms) to be installed and configured with a valid RTI Connext license prior to use.
- V4L2 capable device
- One or more Human Interface Devices (HID), e.g., joystick, keyboard, mouse, etc.

> [!NOTE]
> Instructions below are based on the `.run` installer from RTI Connext. Refer to the [Linux installation](https://community.rti.com/static/documentation/developers/get-started/full-install.html) for details.

## Quick Start

```bash
# Start the publisher
./dev_container build_and_run telesurgery --container_args "-v $HOME/rti_connext_dds-7.3.0:/opt/rti.com/rti_connext_dds-7.3.0/" --run_args "-r"

# Start the subscriber
./dev_container build_and_run telesurgery --container_args "-v $HOME/rti_connext_dds-7.3.0:/opt/rti.com/rti_connext_dds-7.3.0/ --device=/dev/input/js0 --device=/dev/input/event4 --device=/dev/input/event6" --run_args "-s"
```

> [!NOTE]
> Use `--device` to pass the HID devices to the container.

## Building the Application

To build on an IGX devkit (using the `armv8` architecture), follow the [instructions to build Connext DDS applications for embedded Arm targets](https://community.rti.com/kb/how-do-i-create-connext-dds-application-rti-code-generator-and-build-it-my-embedded-target-arm) up to, and including, step 5 (Installing Java and setting JREHOME).

To build the application, the `RTI_CONNEXT_DDS_DIR` CMake variable must point to the installation path for RTI Connext. This can be done automatically by setting the `NDDSHOME` environment variable to the RTI Connext installation directory (such as when using the RTI `setenv` scripts), or manually at build time, e.g.:

```sh
$ ./run build telesurgery --configure-args -DRTI_CONNEXT_DDS_DIR=~/rti/rti_connext_dds-7.3.0
```

### Building with a Container

Due to the license requirements of RTI Connext, it is not currently supported to install RTI Connext into a development container. Instead, Connext should be installed onto the host as above, and then the development container can be launched with the RTI Connext folder mounted at runtime. To do so, ensure that the `NDDSHOME` and `CONNEXTDDS_ARCH` environment variables are set (which can be done using the RTI `setenv` script) and use the following:

```sh
# 1. Build the container
./dev_container build --docker_file applications/dds/Dockerfile
# 2. Launch the container
./dev_container launch --docker_opts "-v $HOME/rti_connext_dds-7.3.0:/opt/rti.com/rti_connext_dds-7.3.0/"
# 3. Build the application
./run build telesurgery
# Continue to the next section to run the application with the robot. 
# Open a new terminal to repeat step #2 and launch a new container for the surgeon.
```

## Running the Application

Both a publisher and subscriber process must be launched to see the result of writing to and reading the video stream from DDS, respectively.

To run the robot process, use the `-r` option:

```sh
$ ./run launch telesurgery --extra_args "-r"
```

To run the surgeon process, use the `-s` option:

```sh
$ ./run launch telesurgery --extra_args "-s"
```

If running the application generates an error about `RTI Connext DDS No Source for License information`, ensure that the RTI Connext license has either been installed system-wide or the `NDDSHOME` environment variable has been set to point to your user's RTI Connext installation path.

Note that these processes can be run on the same or different systems, so long as they are both discoverable by the other via RTI Connext. If the processes are run on different systems, they will communicate using UDPv4, for which optimizations have been defined in the default `qos_profiles.xml` file. These optimizations include increasing the buffer size used by RTI Connext for network sockets, and so the systems running the application must also be configured to increase their maximum send and receive socket buffer sizes. This can be done by running the `set_socket_buffer_sizes.sh` script within this directory:

```sh
$ ./set_socket_buffer_sizes.sh
```

For more details, see the [RTI Connext Guide to Improve DDS Network Performance on Linux Systems](https://community.rti.com/howto/improve-rti-connext-dds-network-performance-linux-systems).

The QoS profiles used by the application can also be modified by editing the `qos_profiles.xml` file in the application directory. For more information about modifying the QoS profiles, see the [RTI Connext Basic QoS](https://community.rti.com/static/documentation/connext-dds/7.3.0/doc/manuals/connext_dds_professional/getting_started_guide/cpp11/intro_qos.html) tutorial or the [RTI Connext QoS Reference Guide](https://community.rti.com/static/documentation/connext-dds/7.3.0/doc/manuals/connext_dds_professional/qos_reference/index.htm).

## Configure the Application

The `telesurgery.yaml` file is used to configure the application. The `surgeon` section is used to configure the surgeon application, and the `robot` section is used to configure the robot application.

### Surgeon Configuration

The `surgeon` section is used to configure the surgeon application.

```yaml
surgeon:
  video:
    domain_id: 0
    stream_id: 0
    participant_qos: "HoloscanDDSTransport::SHMEM+LAN"
    reader_qos: "HoloscanDDSDataFlow::Video"
  holoviz:
    window_title: "Telesurgery - Surgeon"
    width: 1024
    height: 576
    tensors:
      - name: ""
        type: color
        opacity: 1.0
        priority: 0
  hid:
    domain_id: 0
    participant_qos: HoloscanDDSTransport::SHMEM+LAN
    writer_qos: HoloscanDDSDataFlow::Command
    hid_devices:
      - name: joystick1
        path: /dev/input/js0
        type: joystick
      - name: keyboard1
        path: /dev/input/event4
        type: keyboard
      - name: mouse1
        path: /dev/input/event6
        type: mouse
```

The `hid_devices` section is used to configure which HID devices are used by the surgeon. Each HID device is configured with a `name`, `path`, and `type`. The `name` is used to identify the HID device in the application. The `path` is the path to the HID device on the system. The `type` is the type of HID device, which can be `joystick`, `keyboard`, or `mouse`.

### Robot Configuration

The `robot` section is used to configure the robot application.

```yaml
robot:
  video:
    width: 1024
    height: 576
    device: /dev/video0
  hid:
    domain_id: 0
    participant_qos: "HoloscanDDSTransport::SHMEM+LAN"
    reader_qos: "HoloscanDDSDataFlow::Command"
    hid_device_filters:
      - /dev/input/js0
      - /dev/input/event4
      - /dev/input/event6
  holoviz:
    window_title: "Telesurgery - Robot"
    width: 1024
    height: 576
    enable_render_buffer_output: true
    tensors:
      - name: ""
        type: color
        opacity: 1.0
        priority: 0
      - name: joystick1
        type: crosses
        line_width: 2.0
        color: [0.0, 1.0, 0.0, 1.0]
        priority: 1
        opacity: 1.0
      - name: keyboard1
        type: ovals
        color: [0.5, 0.0, 1.0, 1.0]
        priority: 2
        line_width: 2.0
        opacity: 1.0
      - name: mouse1
        type: triangles
        color: [1.0, 0.0, 1.0, 1.0]
        line_width: 3.0
        point_size: 4.0
        priority: 3
  video_publisher:
    domain_id: 0
    stream_id: 0
    participant_qos: "HoloscanDDSTransport::SHMEM+LAN"
    writer_qos: "HoloscanDDSDataFlow::Video"
```

`video.device` is the path to the V4L2 device to use for the robot's video stream. 

`hid.hid_device_filters` is a list of HID devices to subscribe to. Each HID device is configured with a `path` and `type`. The `path` is the path to the HID device on the system. The `type` is the type of HID device, which can be `joystick`, `keyboard`, or `mouse`.

For each HID device that you want to render, you must add a `holoviz.tensors` entry with the same `name` as the `hid.hid_devices.name` and the `type` (aka shape) you want to render.
