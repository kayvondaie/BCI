# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 09:01:19 2023

@author: kayvon.daie
"""

""" example BCI_photostim session """

import datetime

from aind_data_schema.ophys.ophys_session import ,TwoPhotonOphysSession, Laser

t = datetime.datetime(2023, 9, 05, 2, 23, 00)

s = TwoPhotonOphysSession(
    experimenter_full_name=["Kayvon Daie"],
    session_start_time=t,
    session_end_time=t,
    subject_id="652567",
    session_type="Supervised BCI",
    iacuc_protocol="2115",
    rig_id="BCI_photostim",
    light_sources=[
        Laser(
            name="Chameleon",
            wavelength=920,
            wavelength_unit="nanometer",
            excitation_power=13,
            excitation_power_unit="percent",
        ),
        Laser(
            name="Monaco",
            wavelength=1035,
            wavelength_unit="nanometer",
            excitation_power=10,
            excitation_power_unit="percent",
        ),
    ],
    
    notes="Single neuron connection mapping, Supervised BCI, Single neuron connection mapping (same neurons)",
)

s.write_standard_file()