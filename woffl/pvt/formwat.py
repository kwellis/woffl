class FormWater:
    def __init__(self, wat_sg: float) -> None:
        """Initialize a water stream

        Args:
            wat_sg (float): Water Specific Gravity, 0.5 to 1.5

        Returns:
            self
        """

        if (0.5 < wat_sg < 1.5) is False:
            raise ValueError(f"Water SG {wat_sg} Outside Range")

        self.wat_sg = wat_sg

    def __repr__(self) -> str:
        return f"Water {self.wat_sg} Specific Gravity"

    @classmethod
    def schrader(cls):
        """Schrader Bluff Generic Formation Water

        Args:
            wat_sg (float): 1.02"""
        return cls(wat_sg=1.02)

    @classmethod
    def kuparuk(cls):
        """Kuparuk Generic Formation Water

        Args:
            wat_sg (float): 1.02"""
        return cls(wat_sg=1.02)

    @classmethod
    def sag(cls):
        """Sagavanirktok Generic Formation Water

        Args:
            wat_sg (float): 1.02
        """
        return cls(wat_sg=1.02)

    @classmethod
    def ugnu(cls):
        """Ugnu Generic Formation Water

        Args:
            wat_sg (float): 1.02
        """
        return cls(wat_sg=1.02)

    def condition(self, press: float, temp: float):
        """Set condition of evaluation

        Args:
            press (float): Pressure of the water, psig
            temp (float): Temperature of the water, deg F

        Returns:
            self
        """
        self.press = press  # psig
        self.temp = temp  # deg fahr

        self.pabs = self.press + 14.7  # convert psig to psia
        self.tabs = self.temp + 459.67  # convert fahr to rankine
        return self

    @property
    def density(self) -> float:
        """Water Density

        Calculate water density

        Args:
            None

        Returns:
            rho_wat (float): water density, lbm/ft3
        """

        # leave it simple now, asume no compressibility
        rho_wat = self.wat_sg * 62.4
        rho_wat = round(rho_wat, 3)

        return rho_wat

    @property
    def viscosity(self) -> float:
        """Water Viscosity

        Calculate water viscosity

        Args:
            None

        Returns:
            uwat (float): water viscosity, cP
        """

        # come back later and rewrite
        uwat = 0.75  # leave as 0.75 cP for now
        return uwat

    @property
    def compress(self) -> float:
        """Water Isothermal Compressibility

        Calculate isothermal water compressibility.
        Inverse of the bulk modulus of elasticity.
        Used water compressibility at 62.4 lbm/ft3 since we don't change density.

        Args:
            None

        Returns:
            wo (float): water compressibility, psi**-1

        References:
            https://roymech.org/Related/Fluids/Fluids_Water_Props.html
        """
        cw_si = 0.0004543  # 1/MPa
        cw = cw_si / 145.038  # 1/psi
        return cw

    @property
    def tension(self) -> float:
        """Water Surface Tension

        Calculate the water surface tension

        Args:
            None

        Returns:
            sigw (float): Water Surface Tension, lbf/ft
        """
        sigw_cgs = 72.8  # dyne /cm
        sigw = sigw_cgs * 0.0000685  # lbf/ft
        return sigw

    def gas_solubility(self) -> float:
        """Gas Solubility in the Water

        WOFFL assumes the formation water has no gas solubility as a
        simplifying assumption. This code is here to support other calcs.

        Args:
            None

        Returns:
            rsw (float): Gas Solubility in the water, scf/stb
        """
        return self.solubility_hawkins(self.pabs, self.temp)

    @staticmethod
    def solubility_hawkins(pabs: float, teval: float) -> float:
        """Craft and Hawkins Gas Solubility in Water

        Follows the Craft and Hawkins (1951) gas solubility method. Does
        the water need to have a bubblepoint pressure to limit this?

        Args:
            pabs (float): Eval Absolute Pressure, psia
            teval (float): Evaluated Temperature, deg F

        Returns:
            rsw (float): Gas Solubility in the water, scf/stb
        """
        a = 2.12 + 3.45e-3 * teval - 3.59e-5 * teval**2
        b = 0.0107 - 5.26e-5 * teval + 1.48e-7 * teval**2
        c = -8.75e-7 + 3.9e-9 * teval - 1.02e-11 * teval**2
        rsw = a + b * pabs + c * pabs**2
        return rsw
