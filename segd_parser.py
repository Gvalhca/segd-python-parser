# -*- coding: utf8 -*-
"""
SEG-D modified Parser based on ObsPy core module.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import ast
import ntpath
import os
import sys
from pathlib import Path
from typing import List

from future.builtins import *  # NOQA
from future.utils import PY2

from collections import OrderedDict
from struct import unpack
from array import array

import numpy as np
from obspy import UTCDateTime, Trace, Stream
from typing.io import BinaryIO

from decoder import Decoder


class SEGDNotImplemented(Exception):
    pass


class SEGDScanTypeError(Exception):
    pass


class SegDParser:
    """
    SegDParser(segd_path : str)
        Seg-D files parser. Call read_segd() to execute.

        Parameters
        ----------
        segd_path : path to Seg-D file

        Attributes
        ----------
        header_block : bytearray
            Bytearray of general header #1 + general header #2 + general header #3 + scan type header +
            extended header + external header

        traceh_list : List[bytearray]
            Bytearrays of trace header + trace header extensions #1 ... #7 for each trace

        traces_data : np.ndarray
            2D array of shape(num_of_traces, num_of_samples) containing amplitude values in float
    """
    _byte_stream: BinaryIO
    traces_data: np.ndarray
    traceh_list: List[bytearray]
    header_block: bytearray
    _segd_dir: str
    _segd_filename: str
    _output_dir: str

    def __init__(self, segd_path: str):
        self._filepath = segd_path
        self.header_block = bytearray(b'')
        self.traceh_list = []

    def _read_general_hdr1(self):
        """Read general header block #1."""
        buf = self._byte_stream.read(32)
        self.header_block.extend(bytearray(buf))

        general_hdr1 = OrderedDict()
        general_hdr1['file_number'] = Decoder.decode_bcd(buf[0:2])
        _format_code = Decoder.decode_bcd(buf[2:4])
        if _format_code != 8058:
            raise SEGDNotImplemented('Only 32 bit IEEE demultiplexed data '
                                     'is currently supported')
        general_hdr1['format_code'] = Decoder.decode_bcd(buf[2:4])
        general_hdr1['general_constants'] = [Decoder.decode_bcd(b) for b in buf[4:10]]  # unsure
        _year = Decoder.decode_bcd(buf[10:11]) + 2000
        _nblocks, _jday = Decoder.bcd(buf[11])
        general_hdr1['n_additional_blocks'] = _nblocks
        _jday *= 100
        _jday += Decoder.decode_bcd(buf[12:13])
        _hour = Decoder.decode_bcd(buf[13:14])
        _min = Decoder.decode_bcd(buf[14:15])
        _sec = Decoder.decode_bcd(buf[15:16])
        general_hdr1['time'] = UTCDateTime(year=_year, julday=_jday,
                                           hour=_hour, minute=_min, second=_sec)
        general_hdr1['manufacture_code'] = Decoder.decode_bcd(buf[16:17])
        general_hdr1['manufacture_serial_number'] = Decoder.decode_bcd(buf[17:19])
        general_hdr1['bytes_per_scan'] = Decoder.decode_bcd(buf[19:22])
        _bsi = Decoder.decode_bcd(buf[22:23])
        if _bsi < 10:
            _bsi = 1. / _bsi
        else:
            _bsi /= 10.
        general_hdr1['base_scan_interval_in_ms'] = _bsi
        _pol, _ = Decoder.bcd(buf[23])
        general_hdr1['polarity'] = _pol
        # 23L-24 : not used
        _rec_type, _rec_len = Decoder.bcd(buf[25])
        # general_hdr['record_type'] = record_types[_rec_type]
        _rec_len = 0x100 * _rec_len
        _rec_len += Decoder.decode_bin(buf[26:27])
        if _rec_len == 0xFFF:
            _rec_len = None
        general_hdr1['record_length'] = _rec_len
        general_hdr1['scan_type_per_record'] = Decoder.decode_bcd(buf[27:28])
        general_hdr1['n_channel_sets_per_record'] = Decoder.decode_bcd(buf[28:29])
        general_hdr1['n_sample_skew_32bit_extensions'] = Decoder.decode_bcd(buf[29:30])
        general_hdr1['extended_header_length'] = Decoder.decode_bcd(buf[30:31])
        _ehl = Decoder.decode_bcd(buf[31:32])
        # If more than 99 External Header blocks are used,
        # then this field is set to FF and General Header block #2 (bytes 8-9)
        # indicates the number of External Header blocks.
        if _ehl == 0xFF:
            _ehl = None
        general_hdr1['external_header_length'] = _ehl
        return general_hdr1

    def _read_general_hdr2(self):
        """Read general header block #2."""
        buf = self._byte_stream.read(32)
        self.header_block.extend(bytearray(buf))

        general_hdr2 = OrderedDict()
        general_hdr2['expanded_file_number'] = Decoder.decode_bin(buf[0:3])
        # 3-6 : not used
        general_hdr2['external_header_blocks'] = Decoder.decode_bin(buf[7:9])
        # 9 : not used
        _rev = ord(buf[10:11])
        _rev += ord(buf[11:12]) / 10.
        general_hdr2['segd_revision_number'] = _rev
        general_hdr2['no_blocks_of_general_trailer'] = Decoder.decode_bin(buf[12:14])
        general_hdr2['extended_record_length_in_ms'] = Decoder.decode_bin(buf[14:17])
        # 17 : not used
        general_hdr2['general_header_block_number'] = Decoder.decode_bin(buf[18:19])
        # 19-32 : not used
        return general_hdr2

    def _read_general_hdr3(self):
        """Read general header block #3."""
        buf = self._byte_stream.read(32)
        self.header_block.extend(bytearray(buf))

        general_hdr3 = OrderedDict()
        general_hdr3['expanded_file_number'] = Decoder.decode_bin(buf[0:3])
        _sln = Decoder.decode_bin(buf[3:6])
        _sln += Decoder.decode_fraction(buf[6:8])
        general_hdr3['source_line_number'] = _sln
        _spn = Decoder.decode_bin(buf[8:11])
        _spn += Decoder.decode_fraction(buf[11:13])
        general_hdr3['source_point_number'] = _spn
        general_hdr3['phase_control'] = Decoder.decode_bin(buf[14:15])
        general_hdr3['vibrator_type'] = Decoder.decode_bin(buf[15:16])
        general_hdr3['phase_angle'] = Decoder.decode_bin(buf[16:18])
        general_hdr3['general_header_block_number'] = Decoder.decode_bin(buf[18:19])
        general_hdr3['source_set_number'] = Decoder.decode_bin(buf[19:20])
        # 20-32 : not used
        return general_hdr3

    def _read_sch(self):
        """Read scan type header."""
        buf = self._byte_stream.read(32)
        self.header_block.extend(bytearray(buf))

        # check if all the bytes are zero:
        if PY2:
            # convert buf to a list of ints
            _sum = sum(map(ord, buf))
        else:
            # in PY3 this is already the case
            _sum = sum(buf)
        if _sum == 0:
            raise SEGDScanTypeError('Empty scan type header')
        sch = OrderedDict()
        sch['scan_type_header'] = Decoder.decode_bcd(buf[0:1])
        sch['channel_set_number'] = Decoder.decode_bcd(buf[1:2])
        sch['channel_set_starting_time'] = Decoder.decode_bin(buf[2:4])
        sch['channel_set_end_time'] = Decoder.decode_bin(buf[4:6])
        _dm = Decoder.decode_bin(buf[6:8][::-1])
        # sch['descale_multiplier_in_mV'] = _descale_multiplier.get(_dm, int(str(_dm), base=16))
        # print(sch['descale_multiplier_in_mV'])
        sch['number_of_channels'] = Decoder.decode_bcd(buf[8:10])
        _ctid, _ = Decoder.bcd(buf[10])
        sch['channel_type_id'] = _ctid
        _nse, _cgcm = Decoder.bcd(buf[11])
        sch['number_of_subscans_exponent'] = _nse
        sch['channel_gain_control_method'] = _cgcm
        sch['alias_filter_freq_at_-3dB_in_Hz'] = Decoder.decode_bcd(buf[12:14])
        sch['alias_filter_slope_in_dB/octave'] = Decoder.decode_bcd(buf[14:16])
        sch['low-cut_filter_freq_in_Hz'] = Decoder.decode_bcd(buf[16:18])
        sch['low-cut_filter_slope_in_dB/octave'] = Decoder.decode_bcd(buf[18:20])
        sch['first_notch_freq'] = Decoder.decode_bcd(buf[20:22])
        sch['second_notch_freq'] = Decoder.decode_bcd(buf[22:24])
        sch['third_notch_freq'] = Decoder.decode_bcd(buf[24:26])
        sch['extended_channel_set_number'] = Decoder.decode_bcd(buf[26:28])
        _ehf, _the = Decoder.bcd(buf[28])
        sch['extended_header_flag'] = _ehf
        sch['trace_header_extensions'] = _the
        sch['vertical_stack'] = Decoder.decode_bin(buf[29:30])
        sch['streamer_cable_number'] = Decoder.decode_bin(buf[30:31])
        sch['array_forming'] = Decoder.decode_bin(buf[31:32])
        return sch

    def _read_extended_header(self, size):
        """Read extended header."""
        buf = self._byte_stream.read(size)
        self.header_block.extend(bytearray(buf))

        extdh = OrderedDict()
        # SERCEL extended header format
        extdh['acquisition_length_in_ms'] = Decoder.decode_bin(buf[0:4])
        extdh['sample_rate_in_us'] = Decoder.decode_bin(buf[4:8])
        extdh['total_number_of_traces'] = Decoder.decode_bin(buf[8:12])
        extdh['number_of_auxes'] = Decoder.decode_bin(buf[12:16])
        extdh['number_of_seis_traces'] = Decoder.decode_bin(buf[16:20])
        extdh['number_of_dead_seis_traces'] = Decoder.decode_bin(buf[20:24])
        extdh['number_of_live_seis_traces'] = Decoder.decode_bin(buf[24:28])
        _tos = Decoder.decode_bin(buf[28:32])
        # extdh['type_of_source'] = _source_types.get(_tos, None)
        num_of_samples_raw = buf[32:36]
        num_of_samples = Decoder.decode_bin(num_of_samples_raw)
        extdh['number_of_samples_in_trace'] = Decoder.decode_bin(buf[32:36])
        extdh['shot_number'] = Decoder.decode_bin(buf[36:40])
        extdh['TB_window_in_s'] = Decoder.decode_flt(buf[40:44])
        _trt = Decoder.decode_bin(buf[44:48])
        # extdh['test_record_type'] = _test_record_types[_trt]
        extdh['spread_first_line'] = Decoder.decode_bin(buf[48:52])
        extdh['spread_first_number'] = Decoder.decode_bin(buf[52:56])
        extdh['spread_number'] = Decoder.decode_bin(buf[56:60])
        _st = Decoder.decode_bin(buf[60:64])
        # extdh['spread_type'] = _spread_types[_st]
        extdh['time_break_in_us'] = Decoder.decode_bin(buf[64:68])
        extdh['uphole_time_in_us'] = Decoder.decode_bin(buf[68:72])
        extdh['blaster_id'] = Decoder.decode_bin(buf[72:76])
        extdh['blaster_status'] = Decoder.decode_bin(buf[76:80])
        extdh['refraction_delay_in_ms'] = Decoder.decode_bin(buf[80:84])
        extdh['TB_to_T0_time_in_us'] = Decoder.decode_bin(buf[84:88])
        extdh['internal_time_break'] = Decoder.decode_bin_bool(buf[88:92])
        extdh['prestack_within_field_units'] = Decoder.decode_bin_bool(buf[92:96])
        _net = Decoder.decode_bin(buf[96:100])
        # extdh['noise_elimination_type'] = _noise_elimination_types.get(_net, None)
        extdh['low_trace_percentage'] = Decoder.decode_bin(buf[100:104])
        extdh['low_trace_value_in_dB'] = Decoder.decode_bin(buf[104:108])
        _value1 = Decoder.decode_bin(buf[108:112])
        _value2 = Decoder.decode_bin(buf[112:116])
        if _net == 2:
            # Diversity Stack
            extdh['number_of_windows'] = _value1
        elif _net == 3:
            # Historic
            # extdh['historic_editing_type'] = _historic_editing_types[_value2]
            extdh['historic_range'] = Decoder.decode_bin(buf[120:124])
            extdh['historic_taper_length_2_exponent'] = Decoder.decode_bin(buf[124:128])
            extdh['historic_threshold_init_value'] = Decoder.decode_bin(buf[132:136])
            extdh['historic_zeroing_length'] = Decoder.decode_bin(buf[136:140])
        elif _net == 4:
            # Enhanced Diversity Stack
            extdh['window_length'] = _value1
            extdh['overlap'] = _value2
        extdh['noisy_trace_percentage'] = Decoder.decode_bin(buf[116:120])
        _thv = Decoder.decode_bin(buf[128:132])
        # extdh['threshold_hold/var'] = _threshold_types.get(_thv, None)
        _top = Decoder.decode_bin(buf[140:144])
        # extdh['type_of_process'] = _process_types.get(_top, None)
        extdh['acquisition_type_tables'] = \
            [Decoder.decode_bin(buf[144 + n * 4:144 + (n + 1) * 4]) for n in range(32)]
        extdh['threshold_type_tables'] = \
            [Decoder.decode_bin(buf[272 + n * 4:272 + (n + 1) * 4]) for n in range(32)]
        extdh['stacking_fold'] = Decoder.decode_bin(buf[400:404])
        # 404-483 : not used
        extdh['record_length_in_ms'] = Decoder.decode_bin(buf[484:488])
        extdh['autocorrelation_peak_time_in_ms'] = Decoder.decode_bin(buf[488:492])
        # 492-495 : not used
        extdh['correlation_pilot_number'] = Decoder.decode_bin(buf[496:500])
        extdh['pilot_length_in_ms'] = Decoder.decode_bin(buf[500:504])
        extdh['sweep_length_in_ms'] = Decoder.decode_bin(buf[504:508])
        extdh['acquisition_number'] = Decoder.decode_bin(buf[508:512])
        extdh['max_of_max_aux'] = Decoder.decode_flt(buf[512:516])
        extdh['max_of_max_seis'] = Decoder.decode_flt(buf[516:520])
        extdh['dump_stacking_fold'] = Decoder.decode_bin(buf[520:524])
        extdh['tape_label'] = Decoder.decode_asc(buf[524:540])
        extdh['tape_number'] = Decoder.decode_bin(buf[540:544])
        extdh['software_version'] = Decoder.decode_asc(buf[544:560])
        extdh['date'] = Decoder.decode_asc(buf[560:572])
        extdh['source_easting'] = Decoder.decode_dbl(buf[572:580])
        extdh['source_northing'] = Decoder.decode_dbl(buf[580:588])
        extdh['source_elevation'] = Decoder.decode_flt(buf[588:592])
        extdh['slip_sweep_mode_used'] = Decoder.decode_bin_bool(buf[592:596])
        extdh['files_per_tape'] = Decoder.decode_bin(buf[596:600])
        extdh['file_count'] = Decoder.decode_bin(buf[600:604])
        extdh['acquisition_error_description'] = Decoder.decode_asc(buf[604:764])
        _ft = Decoder.decode_bin(buf[764:768])
        # extdh['filter_type'] = _filter_types.get(_ft, None)
        extdh['stack_is_dumped'] = Decoder.decode_bin_bool(buf[768:772])
        _ss = Decoder.decode_bin(buf[772:776])
        if _ss == 2:
            _ss = -1
        extdh['stack_sign'] = _ss
        extdh['PRM_tilt_correction_used'] = Decoder.decode_bin_bool(buf[776:780])
        extdh['swath_name'] = Decoder.decode_asc(buf[780:844])
        _om = Decoder.decode_bin(buf[844:848])
        # XXX: here I suppose that several operating modes are possible
        _op_mode = []
        # for key in _operating_modes:
        #     try:
        #         k = _om & key
        #         _op_mode.append(_operating_modes[k])
        #     except KeyError:
        #         continue
        extdh['operating_mode'] = _op_mode
        # 848-851 : reserved
        extdh['no_log'] = Decoder.decode_bin_bool(buf[852:856])
        extdh['listening_time_in_ms'] = Decoder.decode_bin(buf[856:860])
        _tod = Decoder.decode_bin(buf[860:864])
        # extdh['type_of_dump'] = _dump_types[_tod]
        # 864-867 : reserved
        extdh['swath_id'] = Decoder.decode_bin(buf[868:872])
        extdh['seismic_trace_offset_removal_is_disabled'] = \
            Decoder.decode_bin_bool(buf[872:876])
        # _gps_microseconds = unpack('>Q', buf[876:884])[0]
        # _gps_time = UTCDateTime('19800106') + _gps_microseconds / 1e6
        # _gps_time includes leap seconds (17 as for November 2016)
        # extdh['gps_time_of_acquisition'] = _gps_time
        # 884-963 : reserved
        # 964-1023 : not used
        return extdh

    def _read_external_header(self, size):
        """Read external header."""
        buf = self._byte_stream.read(size)
        self.header_block.extend(bytearray(buf))

        return Decoder.decode_asc(buf)

    def _read_traceh(self):
        """Read trace header."""
        buf = self._byte_stream.read(20)
        self.traceh_list.append(bytearray(buf))

        traceh = OrderedDict()
        _fn = Decoder.decode_bcd(buf[0:2])
        if _fn == 0xFFFF:
            _fn = None
        traceh['file_number'] = _fn
        traceh['scan_type_number'] = Decoder.decode_bcd(buf[2:3])
        traceh['channel_set_number'] = Decoder.decode_bcd(buf[3:4])
        traceh['trace_number'] = Decoder.decode_bcd(buf[4:6])
        traceh['first_timing_word_in_ms'] = Decoder.decode_bin(buf[6:9]) * 1. / 256
        traceh['trace_header_extension'] = Decoder.decode_bin(buf[9:10])
        traceh['sample_skew'] = Decoder.decode_bin(buf[10:11])
        traceh['trace_edit'] = Decoder.decode_bin(buf[11:12])
        traceh['time_break_window'] = Decoder.decode_bin(buf[12:14])
        traceh['time_break_window'] += Decoder.decode_bin(buf[14:15]) / 100.
        traceh['extended_channel_set_number'] = Decoder.decode_bin(buf[15:16])
        traceh['extended_file_number'] = Decoder.decode_bin(buf[17:20])
        return traceh

    def _read_traceh_eb1(self):
        """Read trace header extension block #1, SEGD standard."""
        buf = self._byte_stream.read(32)
        self.traceh_list[-1].extend(bytearray(buf))

        traceh = OrderedDict()
        _rln = Decoder.decode_bin(buf[0:3])
        if _rln == 0xFFFFFF:
            _rln = None
        traceh['receiver_line_number'] = _rln
        _rpn = Decoder.decode_bin(buf[3:6])
        if _rpn == 0xFFFFFF:
            _rpn = None
        traceh['receiver_point_number'] = _rpn
        traceh['receiver_point_index'] = Decoder.decode_bin(buf[6:7])
        traceh['number_of_samples_per_trace'] = Decoder.decode_bin(buf[7:10])
        return traceh

    def _read_traceh_eb2(self):
        """Read trace header extension block #2, SERCEL format."""
        buf = self._byte_stream.read(32)
        self.traceh_list[-1].extend(bytearray(buf))

        traceh = OrderedDict()
        traceh['receiver_point_easting'] = Decoder.decode_dbl(buf[0:8])
        traceh['receiver_point_northing'] = Decoder.decode_dbl(buf[8:16])
        traceh['receiver_point_elevation'] = Decoder.decode_flt(buf[16:20])
        traceh['sensor_type_number'] = Decoder.decode_bin(buf[20:21])
        # 21-23 : not used
        traceh['DSD_identification_number'] = Decoder.decode_bin(buf[24:28])
        traceh['extended_trace_number'] = Decoder.decode_bin(buf[28:32])
        return traceh

    def _read_traceh_eb3(self):
        """Read trace header extension block #3, SERCEL format."""
        buf = self._byte_stream.read(32)
        self.traceh_list[-1].extend(bytearray(buf))

        traceh = OrderedDict()
        traceh['resistance_low_limit'] = Decoder.decode_flt(buf[0:4])
        traceh['resistance_high_limit'] = Decoder.decode_flt(buf[4:8])
        traceh['resistance_calue_in_ohms'] = Decoder.decode_flt(buf[8:12])
        traceh['tilt_limit'] = Decoder.decode_flt(buf[12:16])
        traceh['tilt_value'] = Decoder.decode_flt(buf[16:20])
        traceh['resistance_error'] = Decoder.decode_bin_bool(buf[20:21])
        traceh['tilt_error'] = Decoder.decode_bin_bool(buf[21:22])
        # 22-31 : not used
        return traceh

    def _read_traceh_eb4(self):
        """Read trace header extension block #4, SERCEL format."""
        buf = self._byte_stream.read(32)
        self.traceh_list[-1].extend(bytearray(buf))

        traceh = OrderedDict()
        traceh['capacitance_low_limit'] = Decoder.decode_flt(buf[0:4])
        traceh['capacitance_high_limit'] = Decoder.decode_flt(buf[4:8])
        traceh['capacitance_value_in_nano_farads'] = Decoder.decode_flt(buf[8:12])
        traceh['cutoff_low_limit'] = Decoder.decode_flt(buf[12:16])
        traceh['cutoff_high_limit'] = Decoder.decode_flt(buf[16:20])
        traceh['cutoff_value_in_Hz'] = Decoder.decode_flt(buf[20:24])
        traceh['capacitance_error'] = Decoder.decode_bin_bool(buf[24:25])
        traceh['cutoff_error'] = Decoder.decode_bin_bool(buf[25:26])
        # 26-31 : not used
        return traceh

    def _read_traceh_eb5(self):
        """Read trace header extension block #5, SERCEL format."""
        buf = self._byte_stream.read(32)
        self.traceh_list[-1].extend(bytearray(buf))

        traceh = OrderedDict()
        traceh['leakage_limit'] = Decoder.decode_flt(buf[0:4])
        traceh['leakage_value_in_megahoms'] = Decoder.decode_flt(buf[4:8])
        traceh['instrument_longitude'] = Decoder.decode_dbl(buf[8:16])
        traceh['instrument_latitude'] = Decoder.decode_dbl(buf[16:24])
        traceh['leakage_error'] = Decoder.decode_bin_bool(buf[24:25])
        traceh['instrument_horizontal_position_accuracy_in_mm'] = \
            Decoder.decode_bin(buf[25:28])
        traceh['instrument_elevation_in_mm'] = Decoder.decode_flt(buf[28:32])
        return traceh

    def _read_traceh_eb6(self):
        """Read trace header extension block #6, SERCEL format."""
        buf = self._byte_stream.read(32)
        self.traceh_list[-1].extend(bytearray(buf))

        traceh = OrderedDict()
        _ut = Decoder.decode_bin(buf[0:1])
        # traceh['unit_type'] = _unit_types[_ut]
        traceh['unit_serial_number'] = Decoder.decode_bin(buf[1:4])
        traceh['channel_number'] = Decoder.decode_bin(buf[4:5])
        # 5-7 : not used
        traceh['assembly_type'] = Decoder.decode_bin(buf[8:9])
        traceh['assembly_serial_number'] = Decoder.decode_bin(buf[9:12])
        traceh['location_in_assembly'] = Decoder.decode_bin(buf[12:13])
        # 13-15 : not used
        _st = Decoder.decode_bin(buf[16:17])
        # traceh['subunit_type'] = _subunit_types[_st]
        _ct = Decoder.decode_bin(buf[17:18])
        # traceh['channel_type'] = _channel_types[_ct]
        # 18-19 : not used
        traceh['sensor_sensitivity_in_mV/m/s/s'] = Decoder.decode_flt(buf[20:24])
        # 24-31 : not used
        return traceh

    def _read_traceh_eb7(self):
        """Read trace header extension block #7, SERCEL format."""
        buf = self._byte_stream.read(32)
        self.traceh_list[-1].extend(bytearray(buf))

        traceh = OrderedDict()
        _cut = Decoder.decode_bin(buf[0:1])
        # traceh['control_unit_type'] = _control_unit_types[_cut]
        # traceh['control_unit_serial_number'] = Decoder.decode_bin(buf[1:4])
        # traceh['channel_gain_scale'] = Decoder.decode_bin(buf[4:5])
        # traceh['channel_filter'] = Decoder.decode_bin(buf[5:6])
        # traceh['channel_data_error_overscaling'] = Decoder.decode_bin(buf[6:7])
        # _ces = Decoder.decode_bin(buf[7:8])
        # traceh['channel_edited_status'] = _channel_edited_statuses[_ces]
        # traceh['channel_sample_to_mV_conversion_factor'] = Decoder.decode_flt(buf[8:12])
        # traceh['number_of_stacks_noisy'] = Decoder.decode_bin(buf[12:13])
        # traceh['number_of_stacks_low'] = Decoder.decode_bin(buf[13:14])
        # _channel_type_ids = {1: 'seis', 9: 'aux'}
        # _cti = Decoder.decode_bin(buf[14:15])
        # traceh['channel_type_id'] = _channel_type_ids[_cti]
        # _cp = Decoder.decode_bin(buf[15:16])
        # traceh['channel_process'] = _channel_processes[_cp]
        traceh['trace_max_value'] = Decoder.decode_flt(buf[16:20])
        traceh['trace_max_time_in_us'] = Decoder.decode_bin(buf[20:24])
        traceh['number_of_interpolations'] = Decoder.decode_bin(buf[24:28])
        traceh['seismic_trace_offset_value'] = Decoder.decode_bin(buf[28:32])
        return traceh

    def _read_trace_data(self, size):
        buf = array('f')
        # buf.fromfile(fp, size) doesn't work with py2
        tmp = self._byte_stream.read(size * 4)
        buf.fromstring(tmp)
        buf.byteswap()
        buf = np.array(buf, dtype=np.float32)
        return buf

    def _read_trace_data_block(self, size):
        traceh = self._read_traceh()
        th_ext = traceh['trace_header_extension']

        _read_traceh_eb = [self._read_traceh_eb1, self._read_traceh_eb2,
                           self._read_traceh_eb3, self._read_traceh_eb4,
                           self._read_traceh_eb5, self._read_traceh_eb6,
                           self._read_traceh_eb7]

        for n in range(th_ext):
            traceh.update(_read_traceh_eb[n]())

        data = self._read_trace_data(size)
        return traceh, data

    def _build_segd_header(self, generalh, sch, extdh, extrh, traceh):
        segd = OrderedDict()
        segd.update(generalh)
        channel_set_number = traceh['channel_set_number']
        segd.update(sch[channel_set_number])
        segd.update(extdh)
        segd['external_header'] = extrh
        segd.update(traceh)
        return segd

    def _print_dict(self, dict, title):
        print(title)
        for key, val in dict.items():
            print('{}: {}'.format(key, val))

    def read_segd(self):
        self._byte_stream = open(self._filepath, 'rb')
        generalh = self._read_general_hdr1()
        generalh.update(self._read_general_hdr2())
        generalh.update(self._read_general_hdr3())
        sch = {}
        for n in range(generalh['n_channel_sets_per_record']):
            try:
                _sch = self._read_sch()
            except SEGDScanTypeError:
                continue
            sch[_sch['channel_set_number']] = _sch
        size = generalh['extended_header_length'] * 32
        extdh = self._read_extended_header(size)
        ext_hdr_lng = generalh['external_header_length']
        if ext_hdr_lng == 165:
            ext_hdr_lng = generalh['external_header_blocks']
        size = ext_hdr_lng * 32
        extrh = self._read_external_header(size)
        sample_rate = extdh['sample_rate_in_us'] / 1e6
        npts = extdh['number_of_samples_in_trace']
        size = npts
        st = Stream()
        convert_to_int = True
        for n in range(extdh['total_number_of_traces']):
            traceh, data = self._read_trace_data_block(size)
            # check if all traces can be converted to int
            convert_to_int = convert_to_int and np.all(np.mod(data, 1) == 0)
            # _print_dict(traceh, '***TRACEH:')
            tr = Trace(data)
            # tr.stats.station = str(traceh['unit_serial_number'])
            tr.stats.channel = Decoder.band_code(1. / sample_rate)
            # tr.stats.channel += _instrument_orientation_code[traceh['sensor_code']]
            tr.stats.delta = sample_rate
            tr.stats.starttime = generalh['time']
            tr.stats.segd = self._build_segd_header(generalh, sch, extdh, extrh, traceh)
            st.append(tr)
        self._byte_stream.close()
        self.traces_data = np.zeros((extdh['total_number_of_traces'], extdh['number_of_samples_in_trace']))
        for i in range(len(st)):
            self.traces_data[i, :] = np.asarray(st[i].data)
        # for n, _sch in sch.iteritems():
        #     _print_dict(_sch, '***SCH %d:' % n)
        # _print_dict(extdh, '***EXTDH:')
        # print('***EXTRH:\n %s' % extrh)
        # _print_dict(generalh, '***GENERALH:')
        if convert_to_int:
            for tr in st:
                tr.data = tr.data.astype(np.int32)
        # return st

    def path_leaf(self):
        head, tail = os.path.split(self._filepath)
        return tail or os.path.basename(head)

    def save_parsed_files(self, output_dir=''):
        self._segd_dir = os.path.dirname(self._filepath)
        self._segd_filename = self.path_leaf()

        if output_dir == '':
            self._output_dir = os.path.join(self._segd_dir, 'parsed', self._segd_filename.split('.')[0])
        else:
            self._output_dir = os.path.join(output_dir, self._segd_filename.split('.')[0])

        # Create output dir
        Path(self._output_dir).mkdir(parents=True, exist_ok=True)

        self.write_header_block()
        self.write_trace_headers()
        self.write_trace_data()
        print(f'Saved {self.path_leaf()} to {self._output_dir}')

    def write_header_block(self):
        header_block_filepath = os.path.join(self._output_dir, self._segd_filename.split('.')[0]) + '.hdr_block'
        with open(header_block_filepath, 'wb') as f:
            f.write(self.header_block)
        print(f'Wrote header block of {self.path_leaf()} to {header_block_filepath}')

    def write_trace_headers(self):
        traces_headers_dir = os.path.join(self._output_dir, 'trace_headers')
        Path(traces_headers_dir).mkdir(parents=True, exist_ok=True)
        for i in range(len(self.traceh_list)):
            trace_filepath = os.path.join(traces_headers_dir,
                                          self._segd_filename.split('.')[0]) + f'.trace_{i + 1}.headers'
            with open(trace_filepath, 'wb') as f:
                f.write(self.traceh_list[i])
        print(f'Wrote traces headers of {self.path_leaf()} to {traces_headers_dir}')

    def write_trace_data(self):
        trace_data_filepath = os.path.join(self._output_dir, self._segd_filename.split('.')[0]) + '.trace_data'
        np.set_printoptions(threshold=sys.maxsize, suppress=True, formatter={'float_kind': '{:0.16f}'.format})
        np.savetxt(trace_data_filepath, self.traces_data, fmt='%0.16f')
        print(f'Wrote traces data of {self.path_leaf()} with shape: {self.traces_data.shape} to {trace_data_filepath}')
