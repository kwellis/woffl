import math

import numpy as np

# eventually make dataclass called PumpCatalog, with different manufacturers


class JetPump:
    # Champion X jet pump nozzle sizes, inches
    nozzle_dia = [
        0.0553,
        0.0628,
        0.0705,
        0.0798,
        0.0903,  # matches 1 throat
        0.1016,  # matches 2 throat
        0.1145,  # doesn't match the 3 throat
        0.1291,  # matches 4 throat
        0.1458,  # matches 5 throat
        0.1643,
        0.1858,
        0.2099,
        0.237,
        0.2675,
        0.3017,
        0.3404,
        0.3841,
        0.4335,
        0.4981,
        0.5519,
    ]

    # Champion X jet pump throat sizes, inches
    throat_dia = [
        0.0903,
        0.1016,
        0.1151,
        0.1291,
        0.1458,
        0.1643,
        0.1858,
        0.2099,
        0.237,
        0.2675,
        0.3017,
        0.3404,
        0.3841,
        0.4335,
        0.4981,
        0.5519,
        0.6228,
        0.7027,
        0.7929,
        0.8947,
    ]

    area_code = {"X": -1, "A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

    def __init__(
        self,
        nozzle_no: str,
        area_ratio: str,
        knz: float = 0.01,
        ken: float = 0.03,
        kth: float = 0.3,
        kdi: float = 0.3,
    ) -> None:
        """Jet Pump Initialization

        Create a Jet Pump object.

        Args:
            nozzle_no (string): Represents the Nozzle Size
            area_ratio (string): Represents the Area Ratio
            knz (float): Nozzle Friction Factor, unitless
            ken (float): Enterance Friction Factor, unitless
            kth (float): Throat Friction Factor, unitless
            kdi (float): Diffuser Friction Factor, unitless

        Returns:
            None
        """

        self.noz_no = str(nozzle_no)
        self.rat_ar = str(area_ratio)

        self.knz = knz
        self.ken = ken
        self.kth = kth
        self.kdi = kdi

        nozzle_idx = int(nozzle_no) - 1
        throat_idx = nozzle_idx + JetPump.area_code[area_ratio]

        dnz = None
        dth = None

        try:
            dnz = JetPump.nozzle_dia[nozzle_idx]
        except IndexError:
            print(f"Nozzle Size {nozzle_no} not recognized")

        try:
            dth = JetPump.throat_dia[throat_idx]
        except IndexError:
            print(f"Nozzle Throat Combo {str(nozzle_no) + str(area_ratio)} not recognized")

        self.dnz = dnz
        self.dth = dth

    def __repr__(self):
        return f"{self.noz_no + self.rat_ar} Jet Pump, Nozzle: {self.dnz} inches, Throat: {self.dth} inches"

    @staticmethod
    def area_circle(diameter):
        """Area of a Circle

        Args:
            diameter (float): diameter of a circle

        Returns:
            area (float): area of a circle
        """
        area = math.pi * (diameter**2) / 4
        return area

    @property
    def anz(self) -> float:
        """Area of Nozzle, ft2"""
        return self.area_circle(self.dnz) / 144

    @property
    def ath(self) -> float:
        """Area of Throat, ft2"""
        return self.area_circle(self.dth) / 144

    @property
    def ate(self) -> float:
        """Throat Enterance Area, ft2"""
        return self.ath - self.anz
