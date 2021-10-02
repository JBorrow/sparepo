"""
Particle type enum, to be used throughout
in the place of integers.
"""

from enum import Enum


class ParticleType(Enum):
    GAS = 0
    DARK_MATTER = 1
    TRACER = 3
    STAR = 4
    BLACK_HOLE = 5
