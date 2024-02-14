# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 13:14:19 2023

@author: kayvon.daie
"""

""" example Bergamo ophys session """
import datetime

from aind_data_schema.ophys.ophys_session import Camera, Detector, FieldOfView, Laser, TwoPhotonOphysSession
from aind_data_schema.stimulus import PhotoStimulation, PhotoStimulationGroup, StimulusEpoch

from ScanImageTiffReader import ScanImageTiffReader
import numpy as np
import folder_props_fun
import extract_scanimage_metadata
folder = r'\\allen\aind\scratch\BCI\2p-raw\BCI54\072423/'
folder_props = folder_props_fun.folder_props_fun(folder)
file = folder + folder_props['siFiles'][0]
siHeader = extract_scanimage_metadata.extract_scanimage_metadata(file)

photostim_groups = siHeader['metadata']['json']['RoiGroups']['photostimRoiGroups']
photostim_groups[0]['rois'][1]['scanfields']['powers']

t = datetime.datetime(2022, 7, 12, 7, 00, 00)
t2 = datetime.time(7, 00, 00)

s = TwoPhotonOphysSession(
    experimenter_full_name=["John Doe"],
    session_start_time=t,
    session_end_time=t,
    subject_id="652567",
    session_type="BCI",
    iacuc_protocol="2115",
    rig_id="Bergamo photostim.",
    light_sources=[
        Laser(
            name="Laser A",
            wavelength=920,
            wavelength_unit="nanometer",
            excitation_power=int(siHeader['metadata']['hBeams']['powers'][1:-1].split()[0]),
            excitation_power_unit="percent",
        ),
    ],
    detectors=[
        Detector(
            name="PMT A",
            exposure_time=0.1,
            trigger_type="Internal",
        ),
    ],
    cameras=[Camera(name="Side Camera")],
    stimulus_epochs=[
        StimulusEpoch(
            stimulus=PhotoStimulation(
                stimulus_name="PhotoStimulation",
                number_groups=2,
                groups=[
                    PhotoStimulationGroup(
                        group_index=0,
                        number_of_neurons=int(np.array(photostim_groups[0]['rois'][1]['scanfields']['slmPattern']).shape[0]),
                        stimulation_laser_power=int(photostim_groups[0]['rois'][1]['scanfields']['powers']),
                        number_trials=5,
                        number_spirals=int(photostim_groups[0]['rois'][1]['scanfields']['repetitions']),
                        spiral_duration=photostim_groups[0]['rois'][1]['scanfields']['duration'],
                        inter_spiral_interval=photostim_groups[0]['rois'][2]['scanfields']['duration'],
                    ),
                    PhotoStimulationGroup(
                        group_index=0,
                        number_of_neurons=int(np.array(photostim_groups[0]['rois'][1]['scanfields']['slmPattern']).shape[0]),
                        stimulation_laser_power=int(photostim_groups[0]['rois'][1]['scanfields']['powers']),
                        number_trials=5,
                        number_spirals=int(photostim_groups[0]['rois'][1]['scanfields']['repetitions']),
                        spiral_duration=photostim_groups[0]['rois'][1]['scanfields']['duration'],
                        inter_spiral_interval=photostim_groups[0]['rois'][2]['scanfields']['duration'],
                    ),
                ],
                inter_trial_interval=10,
            ),
            stimulus_start_time=t2,
            stimulus_end_time=t2,
        )
    ],
    fovs=[
        FieldOfView(
            index=0,
            imaging_depth=150,
            targeted_structure="M1",
            fov_coordinate_ml=1.5,
            fov_coordinate_ap=1.5,
            fov_reference="Bregma",
            fov_width=int(siHeader['metadata']['hRoiManager']['pixelsPerLine']),
            fov_height=int(siHeader['metadata']['hRoiManager']['linesPerFrame']),
            magnification="16x",
            fov_scale_factor=float(siHeader['metadata']['hRoiManager']['scanZoomFactor']),
            frame_rate=float(siHeader['metadata']['hRoiManager']['scanFrameRate']),
        ),
    ],
)

s.write_standard_file(prefix="bergamo")