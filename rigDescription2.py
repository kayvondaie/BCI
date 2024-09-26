from datetime import date, datetime, timezone
from aind_data_schema_models.modalities import Modality
import aind_data_schema.components.devices as d
import aind_data_schema.core.rig as r



rig = r.Rig(
    rig_id="Photostim_BCI",
    modification_date=date(2023, 1, 4),
    modalities=[Modality.POPHYS, Modality.BEHAVIOR, Modality.BEHAVIOR_VIDEOS],  # Correct modality for the setup
    cameras=[
        d.CameraAssembly(
            name="BehaviorVideography_FaceCamera",
            camera_target=d.CameraTarget.SIDE,
            camera=d.Camera(
                name="Face Camera",
                detector_type="Camera",
                serial_number="",
                manufacturer=d.Organization.FLIR,
                model="Flea3 FL3-U3-13Y3M",
                notes="",
                data_interface="USB",
                computer_name="",
                max_frame_rate=120,
                sensor_width=640,
                sensor_height=480,
                chroma="Color",
                cooling="Air",
                bin_mode="Additive",
                recording_software=d.Software(name="Bonsai", version="2.5"),
            ),
            lens=d.Lens(
                name="Face Camera Lens",
                model="XC0922LENS",
                serial_number="unknown",
                manufacturer=d.Organization.OTHER,
                max_aperture="f/1.4",
                notes='Focal Length 9-22mm 1/3" IR F1.4',
            ),
        ),
        d.CameraAssembly(
            name="BehaviorVideography_SideCamera",
            camera_target=d.CameraTarget.SIDE,
            camera=d.Camera(
                name="Side Camera",
                detector_type="Camera",
                serial_number="",
                manufacturer=d.Organization.FLIR,
                model="Flea3 FL3-U3-13Y3M",
                notes="",
                data_interface="USB",
                computer_name="",
                max_frame_rate=120,
                sensor_width=640,
                sensor_height=480,
                chroma="Color",
                cooling="Air",
                bin_mode="Additive",
                recording_software=d.Software(name="Bonsai", version="2.5"),
            ),
            lens=d.Lens(
                name="Side Camera Lens",
                model="XC0922LENS",
                serial_number="unknown",
                manufacturer=d.Organization.OTHER,
                max_aperture="f/1.4",
                notes='Focal Length 9-22mm 1/3" IR F1.4',
            ),
        ),
    ],
    detectors=[
        d.Detector(
            name="Red PMT",
            serial_number="",
            manufacturer=d.Organization.HAMAMATSU,
            model="AF7695",
            detector_type="Photomultiplier Tube",
            data_interface="PXI",
            cooling="Air",
        ),
        d.Detector(
            name="Green PMT",
            serial_number="",
            manufacturer=d.Organization.HAMAMATSU,
            model="AF7695",
            detector_type="Photomultiplier Tube",
            data_interface="PXI",
            cooling="Air",
        ),
    ],
    light_sources=[
        d.Laser(
            name="Monaco Laser",
            manufacturer=d.Organization.COHERENT_SCIENTIFIC,
            model="Monaco",
            serial_number="0918012925",
            wavelength=1035,
            maximum_power=40000,
            coupling="Free-space",
        ),
        d.Laser(
            name="Chameleon Laser",
            manufacturer=d.Organization.COHERENT_SCIENTIFIC,
            model="Chameleon",
            serial_number="GDP.1185374.8460",
            wavelength=920,
            maximum_power=4000,
            coupling="Free-space",
        ),
    ],
    mouse_platform=d.Tube(
        name="Standard Mouse Tube",
        diameter=3.0,  # Add the required diameter field
        notes="Mouse sits in a plastic tube",
    ),
    objectives=[
        d.Objective(
            name="10x Objective",
            serial_number="12345",
            manufacturer=d.Organization.NIKON,
            model="CFI Plan Apochromat Lambda D",
            numerical_aperture=0.45,
            magnification=10,
            immersion="air"
        )
    ],
    calibrations=[
        d.Calibration(
            calibration_date=datetime(2023, 1, 4, tzinfo=timezone.utc),
            device_name="Monaco Laser",
            description="Laser Power Calibration",
            input={"power_setting": [1, 2, 3]},
            output={"power_output": [5, 10, 20]},
            notes="Calibration of the laser power output at different settings"
        )
    ],
    stimulus_devices=[
        d.StimulusDevice(
            name="Reward Valve",
            serial_number="",
            manufacturer=d.Organization.THORLABS,
            model="Model XYZ",
            notes="Reward delivery system",
        )
    ]
)

# Serialize to JSON
json_data = rig.model_dump_json()

# Write JSON data to a file
with open('microscope_metadata.json', 'w') as json_file:
    json_file.write(json_data)

print("JSON file created successfully.")

#missing stuff: reward valve info, additional devices Bergamo thorlabs and model
#Future: add position of cameras