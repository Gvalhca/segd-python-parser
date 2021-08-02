from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str, PY2
from struct import unpack
import string
import math


class Decoder:
    # band codes matching sample rate, for a short-period instrument
    @staticmethod
    def band_code(sample_rate):
        if sample_rate >= 1000:
            return 'G'
        if sample_rate >= 250:
            return 'D'
        if sample_rate >= 80:
            return 'E'
        if sample_rate >= 10:
            return 'S'

    @staticmethod
    def bcd(byte):
        """Decode 1-byte binary code decimals."""

        if isinstance(byte, (native_str, str)):
            try:
                byte = ord(byte)
            except TypeError:
                raise ValueError('not a byte')
        elif isinstance(byte, int):
            if byte > 255:
                raise ValueError('not a byte')
        else:
            raise ValueError('not a byte')
        v1 = byte >> 4
        v2 = byte & 0xF
        return v1, v2

    @staticmethod
    def decode_bcd(bytes_in):
        """Decode arbitrary length binary code decimals."""
        v = 0
        if isinstance(bytes_in, int):
            bytes_in = bytes([bytes_in])
        n = len(bytes_in)
        n = n * 2 - 1  # 2 values per byte
        for byte in bytes_in:
            v1, v2 = Decoder.bcd(byte)
            v += v1 * 10 ** n + v2 * 10 ** (n - 1)
            n -= 2
        return v

    @staticmethod
    def decode_bin(bytes_in):
        """Decode unsigned ints."""
        if isinstance(bytes_in, int):
            bytes_in = bytes([bytes_in])
        ll = len(bytes_in)
        # zero-pad to 4 bytes
        b = (chr(0) * (4 - ll)).encode()
        b += bytes_in
        return unpack('>I', b)[0]

    @staticmethod
    def decode_bin_bool(bytes_in):
        """Decode unsigned ints as booleans."""
        b = Decoder.decode_bin(bytes_in)
        return b > 0

    @staticmethod
    def decode_fraction(bytes_in):
        """Decode positive binary fractions."""
        if PY2:
            # transform bytes_in to a list of ints
            bytes_ord = map(ord, bytes_in)
        else:
            # in PY3 this is already the case
            bytes_ord = bytes_in
        bit = ''.join('{:08b}'.format(b) for b in bytes_ord)
        return sum(int(x) * 2 ** -n for n, x in enumerate(bit, 1))

    @staticmethod
    def decode_flt(bytes_in):
        """Decode single-precision floats."""
        if isinstance(bytes_in, int):
            bytes_in = bytes([bytes_in])
        ll = len(bytes_in)
        # zero-pad to 4 bytes
        b = (chr(0) * (4 - ll)).encode()
        b += bytes_in
        f = unpack('>f', b)[0]
        if math.isnan(f):
            f = None
        return f

    @staticmethod
    def decode_dbl(bytes_in):
        """Decode double-precision floats."""
        return unpack('>d', bytes_in)[0]

    @staticmethod
    def decode_asc(bytes_in):
        """Decode ascii."""
        if PY2:
            # transform bytes_in to a list of ints
            bytes_ord = map(ord, bytes_in)
        else:
            # in PY3 this is already the case
            bytes_ord = bytes_in
        printable = map(ord, string.printable)
        s = ''.join(chr(x) for x in bytes_ord if x in printable)
        if not s:
            s = None
        return s
