import math

import numpy as np


class BlackOil:
    """Black Oil Stream

    Set a condition and calculate density, viscosity or compressibility
    """

    def __init__(self, oil_api: float, bubblepoint: float, gas_sg: float) -> None:
        """Initialize a Black Oil Stream

        Args:
            oil_api (float): Oil API, 10 to 40
            bubblepoint (float): Bubble Point Pressure, PSIG
            gas_sg (float): In-Situ Gas Specific Gravity

        Returns:
            Self
        """
        api_min = 10
        api_max = 40
        if (api_min < oil_api < api_max) is False:
            raise ValueError(f"Oil API {oil_api} Outside Range of {api_min} to {api_max}")

        bub_min = 1000
        bub_max = 3000
        if (bub_min < bubblepoint < bub_max) is False:
            raise ValueError(f"Bubblepoint {bubblepoint} Outside Range of {bub_min} to {bub_max}")

        sg_min = 0.5
        sg_max = 1.2
        if (sg_min < gas_sg < sg_max) is False:
            raise ValueError(f"Gas SG {gas_sg} Outside Range of {sg_min} to {sg_max}")

        self.oil_api = oil_api
        self.pbp = bubblepoint
        self.gas_sg = gas_sg

    def __repr__(self):
        return f"Oil: {self.oil_api} API and a {round(self.gas_sg, 2)} SG Gas"

    @classmethod
    def schrader(cls):
        """Schrader Bluff Black Oil

        Generic Schrader Bluff black oil with preset properties

        Args:
            oil_api (float): 22 API
            bubblepoint (float): 1750 psig
            gas_sg (float): 0.65
        """
        return cls(oil_api=22, bubblepoint=1750, gas_sg=0.65)

    @classmethod
    def kuparuk(cls):
        """Kuparuk Black Oil

        Generic Kuparuk black oil with preset properties

        Args:
            oil_api (float): 24 API
            bubblepoint (float): 2250 psig
            gas_sg (float): 0.65
        """
        return cls(oil_api=24, bubblepoint=2250, gas_sg=0.65)

    @classmethod
    def sag(cls):
        """Sagavanirktok Black Oil

        Generic Sagavanirktok black oil with preset properties

        Args:
            oil_api (float): 28 API
            bubblepoint (float): 2700 psig
            gas_sg (float): 0.65
        """
        return cls(oil_api=28, bubblepoint=2700, gas_sg=0.65)

    @classmethod
    def ugnu(cls):
        """Ugnu Black Oil

        Generic Ugnu black oil with preset properties

        Args:
            oil_api (float): 14 API
            bubblepoint (float): 1500 psig
            gas_sg (float): 0.65
        """
        return cls(oil_api=14, bubblepoint=1500, gas_sg=0.65)

    @classmethod
    def test_oil(cls):
        """Test Oil

        Used to run tests on the black oil methods. Uses 0.55 gas
        specific gravity with is the same as pure methane.
        """
        return cls(oil_api=22, bubblepoint=1750, gas_sg=0.55)

    def condition(self, press: float, temp: float):
        """Set condition of evaluation

        Args:
            press (float): Pressure of the oil, psig
            temp (float): Temperature of the oil, deg F

        Returns:
            self
        """
        self.press = press
        self.temp = temp
        return self

    def gas_solubility(self) -> float:
        """Gas Solubility in Oil (Solution GOR)

        Return the gas solubility in the oil, scf/stb.

        Args:
            None

        Returns:
            rs (float): Gas solubility in the oil, scf/stb
        """
        if self.press > self.pbp:  # above bubblepoint pressure
            press = self.pbp  # calculate gas solubility using bubblepoint pressure
        else:
            press = self.press
        rs = self.solubility_kartoatmodjo(press, self.temp, self.oil_api, self.gas_sg)
        return rs

    @property
    def compress(self) -> float:
        """Oil Compressibility Isothermal

        Calculate isothermal oil compressibility.

        Args:
            None

        Returns:
            co (float): oil compressibility, psi**-1
        """
        rs = self.gas_solubility()  # solubility of gas in the oil
        if self.press > self.pbp:  # above bubblepoint
            co = self.compressibility_vasquez_above(self.press, self.temp, self.oil_api, self.gas_sg, rs)
        else:  # below the bubblepoint
            co = self.compressibility_mccain_below(self.press, self.temp, self.oil_api, self.gas_sg, rs)
            # co = self.compressibility_kartoatmodjo_above(self.press, self.temp, self.oil_api, self.gas_sg, rs)
        return co

    def oil_fvf(self) -> float:
        """Oil Formation Volume Factor, Bo

        Calculate formation volume factor for the oil phase.

        Args:
            None

        Returns:
            bo (float): oil formation volume, rb/stb
        """
        rs = self.gas_solubility()
        bo = self.fvf_kartoatmodjo_below(self.temp, self.oil_api, self.gas_sg, rs)
        if self.press > self.pbp:  # above bubblepoint pressure
            bob = bo
            co = self.compress
            bo = self.fvf_vasquez_above(self.press, self.pbp, bob, co)
        return bo

    @property  # property decorator makes it so you don't need brackets
    def density(self) -> float:
        """Live oil density, lbm/ft3

        Calculate the live density of the oil.

        Args:
            None

        Returns:
            rho_oil (float): density of the oil, lbm/ft3
        """
        rs = self.gas_solubility()
        bo = self.oil_fvf()
        rho_oil = self.live_oil_density(self.oil_api, self.gas_sg, rs, bo)
        return rho_oil

    @property
    def viscosity(self) -> float:
        """Live oil viscosity, cP

        Return the live oil viscosity, cP.
        Requires pressure and temperature condition to previously be set.

        Args:
            None

        Returns:
            uoil (float): live oil viscosity, cP
        """
        uod = self.viscosity_dead_kartoatmodjo(self.temp, self.oil_api)
        uol = self.viscosity_live_kartoatmodjo_below(uod, self.gas_solubility())
        if self.press > self.pbp:  # above bubblepoint
            uob = uol
            uol = self.viscosity_live_kartoatmodjo_above(uob, self.press, self.pbp)
        return uol

    @property
    def tension(self) -> float:
        """Live Oil Surface Tension

        Calculate the live oil surface tension

        Args:
            None

        Returns:
            sigo (float): Live Oil Surface Tension, lbf/ft"""
        sigod = self.tension_dead_abdul(self.temp, self.oil_api)
        sigol = self.tension_live_abdul(sigod, self.gas_solubility())  # dyne/cm
        sigo = sigol * 0.0000685  # lbf/ft
        return sigo

    @staticmethod
    def fvf_standing_below(temp: float, oil_api: float, gas_sg: float, rs: float) -> float:
        """Standing Oil Formation Volume Factor

        Calculate formation volume factor for the oil phase.
        Uses the Standing California 1980 correlation.

        Args:
            temp (float): Temperature of the oil, deg F
            oil_api (float): Oil API Degrees
            gas_sg (float): Gas Specific Gravity, air
            rs (float): Solubility of Gas in Oil, SCF/STB

        Returns:
            bo (float): Oil FVF at / below bubblepoint, rb/stb
        """
        oil_sg = 141.5 / (oil_api + 131.5)  # oil specific gravity
        bo = 0.972 + 1.47 * 10**-4 * (rs * (gas_sg / oil_sg) ** 0.5 + 1.25 * temp) ** 1.175
        return bo

    @staticmethod
    def fvf_almarhoun_below(temp: float, oil_api: float, gas_sg: float, rs: float) -> float:
        """Al-Marhoun FVF Below Bubblepoint

        Calculate the formation volume factor at / or below bubblepoint.
        The is method is different than the 1985 method commonly referenced.

        Args:
            temp (float): Temperature of the oil, deg F
            oil_api (float): Oil API Degrees
            gas_sg (float): Gas Specific Gravity, relative to air
            rs (float): Solubility of Gas in Oil, SCF/STB

        Returns:
            bo (float): Oil FVF at / below bubblepoint, rb/stb

        References:
            - New Correlations for FVF of Oil and Gas, M. Al-Marhoun (1992) PETSOC-92-03-02
            - Black Oil Property Correlations State of the Art, M. Al-Marhoun (2015) SPE-172833-MS
        """
        a1 = 0.177342 * 10**-3
        a2 = 0.220163 * 10**-3
        a3 = 4.292580 * 10**-6
        a4 = 0.528707 * 10**-3
        oil_sg = 141.5 / (oil_api + 131.5)  # oil specific gravity
        bo = 1 + a1 * rs + a2 * rs * gas_sg / oil_sg + a3 * rs * (temp - 60) * (1 - oil_sg) + a4 * (temp - 60)
        return bo

    @staticmethod
    def fvf_kartoatmodjo_below(temp: float, oil_api: float, gas_sg: float, rs: float) -> float:
        """Kartoatmodjo and Schmidt FVF Below Bubblepoint

        Calculate the formation volume factor at / or below bubblepoint.
        Correlation uses gas_sg at 100 psig, which is the composition of the free gas
        released at 100 psig. Assumption is made that the difference in composition of the
        free gas between 100 psig and 0 psig is negligible.

        Args:
            temp (float): Temperature of the oil, deg F
            oil_api (float): Oil API Degrees
            gas_sg (float): Gas Specific Gravity, relative to air
            rs (float): Solubility of Gas in Oil, SCF/STB

        Returns:
            bo (float): Oil FVF at / below Bubblepoint, rb/stb

        References:
            - New Correlations for Crude Oil Physical Properties, Kartoatmodjo and Schmidt (1991) SPE-23556
            - Applied Multiphase Flow in Pipes (2017) Al-Safran and Brill, Page 285
        """
        oil_sg = 141.5 / (oil_api + 131.5)  # oil specific gravity
        f = rs**0.755 * gas_sg**0.25 * oil_sg**-1.5 + 0.45 * temp
        bo = 0.98496 + 0.0001 * f**1.5
        return bo

    @staticmethod
    def fvf_vasquez_above(press: float, pbp: float, bob: float, co: float) -> float:
        """Vasquez and Beggs FVF Above Bubblepoint

        Calculate formation volume factor above bubblepoint:

        Args:
            press (float): Pressure of the oil, psig
            pbp (float): Bubblepoint Pressure, psig
            bob (float): Oil FVF at Bubblepoint, rb/stb
            co (float): Oil Isothermal Compressibility, psi-1

        Returns:
            bo (float): Oil FVF above bubblepoint, rb/stb

        References:
            - Applied Multiphase Flow in Pipes (2017) Al-Safran and Brill, Page 286
        """
        bo = bob * math.exp(co * (pbp - press))
        return bo

    @staticmethod
    def solubility_kartoatmodjo(press: float, temp: float, oil_api: float, gas_sg: float) -> float:
        """Kartoatmodjo and Schmidt Solubility of Gas in Oil

        Calculate the Solubility of Gas in Oil.
        Correlation uses gas_sg at 100 psig, which is the composition of the free gas
        released at 100 psig. Assumption is made that the difference in composition of the
        free gas between 100 psig and 0 psig is negligible.

        Args:
            press (float): Pressure of the oil, psig
            temp (float): Temperature of the oil, deg F
            oil_api (float): Oil API Degrees
            gas_sg (float): Gas Specific Gravity, relative to air

        Returns:
            rs (float): Solubility of the Gas in the Oil, scf/stb

        References:
            - New Correlations for Crude Oil Physical Properties, Kartoatmodjo and Schmidt (1991) SPE-23556
            - Applied Multiphase Flow in Pipes (2017) Al-Safran and Brill, Page 285
        """
        if oil_api <= 30:
            c1, c2, c3, c4 = [0.05958, 0.7972, 1.0014, 13.1405]
        else:  # oilAPI greater than 30API
            c1, c2, c3, c4 = [0.0315, 0.7587, 1.0937, 11.289]
        pabs = press + 14.7  # absolute pressure
        # print(pabs)
        with np.errstate(invalid="raise"):
            rs = c1 * gas_sg**c2 * pabs**c3 * 10 ** (c4 * oil_api / (temp + 460))
        # print(rs)
        return rs

    @staticmethod
    def solubility_vasquez(press: float, temp: float, oil_api: float, gas_sg: float) -> float:
        """Vasquez and Beggs Solubility of Gas in Oil

        Calculate solubility of the gas in oil. Using the Vasquez and Beggs Methodology

        Args:
            press (float): Pressure of the oil, psig
            temp (float): Temperature of the oil, deg F
            oil_api (float): Oil API Degrees
            gas_sg (float): Gas Specific Gravity, relative to air

        Returns:
            rs (float): Solubility of the Gas in the Oil, scf/stb

        References:
            - Fundamental Principles of Reservoir Engineering, B.Towler (2002) Page 22
            - Correlations for Fluid Physical Property... Vasquez and Beggs (1980)
        """
        if oil_api <= 30:
            c1, c2, c3 = [0.0362, 1.0937, 25.7240]
        else:  # if oilAPI is greater than 30API
            c1, c2, c3 = [0.0178, 1.187, 23.9310]
        pabs = press + 14.7  # absolute pressure
        rs = c1 * gas_sg * pabs**c2 * math.exp(c3 * oil_api / (temp + 460))
        return rs

    @staticmethod
    def compressibility_vasquez_above(press: float, temp: float, oil_api: float, gas_sg: float, rs: float) -> float:
        """Vasquez and Beggs compressibility of oil above bubblepoint

        Calculate isothermal oil compressibility. Inverse of the bulk modulus of elasticity.
        Valid for pressures that are above the oil bubblepoint.

        Args:
            press (float): Pressure of the oil, psig
            temp (float): Temperature of the oil, deg F
            oil_api (float): Oil API Degrees
            gas_sg (float): Gas Specific Gravity, relative to air
            rs (float): Solubility of Gas in the oil, scf/stb

        Returns:
            co (float): oil compressibility, psi**-1

        References:
            - Fundamental Principles of Reservoir Engineering, B.Towler (2002) Page 22
            - Correlations for Fluid Physical Property... Vasquez and Beggs (1980)
        """
        pabs = press + 14.7
        co = (5 * rs + 17.2 * temp - 1180 * gas_sg + 12.61 * oil_api - 1433) / (pabs * 10**5)
        return co

    @staticmethod
    def compressibility_kartoatmodjo_above(
        press: float, temp: float, oil_api: float, gas_sg: float, rs: float
    ) -> float:
        """Kartoatmodjo compressibility of oil above bubblepoint

        Calculate isothermal oil compressibility. Valid for all (?) pressures.
        Error in the paper, oil_api is ommitted and temp coef is not 0.7606.
        Textbook has the correct equation, verified via internets.

        Args:
            press (float): Pressure of the oil, psig
            temp (float): Temperature of the oil, deg F
            oil_api (float): Oil API Degrees
            gas_sg (float): Gas Specific Gravity, relative to air
            rs (float): Solubility of Gas in the oil, scf/stb

        Returns:
            co (float): oil compressibility, psi**-1

        References:
            - New Correlations for Crude Oil Physical Properties, Kartoatmodjo and Schmidt (1991) SPE-23556
            - Applied Multiphase Flow in Pipes (2017) Al-Safran and Brill, Page 286
        """
        pa = press + 14.7
        co = 6.8257 * 10**-6 * rs**0.5002 * oil_api**0.3613 * temp**0.7606 * gas_sg**-0.35505 / pa
        return co

    @staticmethod
    def compressibility_mccain_below(press: float, temp: float, oil_api: float, gas_sg: float, rs: float) -> float:
        """McCain et. al compressibility of oil below bubblepoint

        Calculate isothermal oil compressibility. Valid for pressures below bubblepoint.
        Equation 5 in the referenced paper by McCain.

        Args:
            press (float): Pressure of the oil, psig
            temp (float): Temperature of the oil, deg F
            oil_api (float): Oil API Degrees
            gas_sg (float): Gas Specific Gravity, relative to air
            rs (float): Solubility of Gas in the oil, scf/stb

        Returns:
            co (float): oil compressibility, psi**-1

        References:
            - Coefficient of Isothermal Compressibility of Black Oil, McCain, Rollins, Villena (1988) SPE-15664-PA
        """
        # gas_sg isn't used, eqn 4 in paper takes oil bubblepoint as input
        pa = press + 14.7
        co = math.exp(
            -7.633 - 1.497 * math.log(pa) + 1.115 * math.log(temp) + 0.533 * math.log(oil_api) + 0.184 * math.log(rs)
        )
        return co

    @staticmethod
    def live_oil_density(oil_api: float, gas_sg: float, rs: float, bo: float) -> float:
        """Live Oil Density, lbm/ft3

        Calculate the live density of the oil.

        Args:
            oil_api (float): Oil API Degrees
            gas_sg (float): Gas Specific Gravity, relative to air
            rs (float): Solubility of Gas in the oil, scf/stb
            bo (float): Oil FVF, rb/stb

        Returns:
            rho_oil (float): density of the oil, lbm/ft3

        References:
            - Applied Multiphase Flow in Pipes (2017) Al-Safran and Brill, Page 288, Eqn B-34
        """
        oil_sg = 141.5 / (oil_api + 131.5)  # oil specific gravity
        rho_oil = (62.42796 * oil_sg + (0.0136 * gas_sg * rs)) / bo
        return rho_oil

    @staticmethod
    def viscosity_dead_beggs(temp: float, oil_api: float) -> float:
        """Beggs Dead Oil Viscosity

        Dead oil viscosity following Beggs methodology.

        Args:
            temp (float): Oil Temperature, deg F
            oil_api (float): Oil API Degrees

        Returns:
            uod (float): Dead Oil Viscosity, cP

        References:
            - Fundamental Principles of Reservoir Engineering, B.Towler (2002) Page 23
            - Correlations for Fluid Physical Property... Vasquez and Beggs (1980)
        """
        x = temp**-1.163 * math.exp(6.9824 - 0.04658 * oil_api)
        uod = 10**x - 1
        return uod

    @staticmethod
    def viscosity_live_beggs_below(uod: float, rs: float) -> float:
        """Live Oil Viscosity, Below Bubble Point

        Live oil viscosity below bubblepoint following Beggs methodology.

        Args:
            uod (float): Dead Oil Viscosity, cP
            rs (float): Oil API Degrees

        Returns:
            uol (float): Live Oil Viscosity, cP

        References:
            - Fundamental Principles of Reservoir Engineering, B.Towler (2002) Page 23
            - Correlations for Fluid Physical Property... Vasquez and Beggs (1980)
        """
        a = 10.715 * (rs + 100) ** -0.515
        b = 5.44 * (rs + 150) ** -0.338
        uol = a * (uod**b)
        return uol

    @staticmethod
    def viscosity_live_beggs_above(uob: float, press: float, pbp: float) -> float:
        """Live Oil Viscosity, Above Bubble Point

        Live oil viscosity above bubblepoint following Beggs methodology.

        Args:
            uob (float): Live Oil Viscosity at Bubblepoint, cP
            press (float): Pressure, psig
            pbp (float): Bubblepoint Pressure, psig

        Returns:
            uol (float): Live Oil Viscosity, cP

        References:
            - Fundamental Principles of Reservoir Engineering, B.Towler (2002) Page 23
            - Correlations for Fluid Physical Property... Vasquez and Beggs (1980)
        """
        pa = press + 14.7
        m = 2.6 * press**1.187 * math.exp(-11.513 - 8.98 * 10**-5 * press)
        uol = uob * (pa / pbp) ** m
        return uol

    @staticmethod
    def viscosity_dead_kartoatmodjo(temp: float, oil_api: float) -> float:
        """Kartoatmodjo Dead Oil Viscosity

        Dead oil viscosity following Kartoatmodjo methodology.

        Args:
            temp (float): Oil Temperature, deg F
            oil_api (float): Oil API Degrees

        Returns:
            uod (float): Dead Oil Viscosity, cP

        References:
            - New Correlations for Crude Oil Physical Properties, Kartoatmodjo and Schmidt (1991) SPE-23556
            - Applied Multiphase Flow in Pipes (2017) Al-Safran and Brill, Page 289
        """
        uod = (16 * 10**8) * temp**-2.8177 * (math.log(oil_api, 10)) ** (5.7526 * math.log(temp, 10) - 26.9718)
        return uod

    @staticmethod
    def viscosity_live_kartoatmodjo_below(uod: float, rs: float) -> float:
        """Live Oil Viscosity, Below Bubble Point

        Live oil viscosity below bubblepoint following Kartoatmodjo methodology.

        Args:
            uod (float): Dead Oil Viscosity, cP
            rs (float): Gas Solubility, scf/stb

        Returns:
            uol (float): Live Oil Viscosity, cP

        References:
            - New Correlations for Crude Oil Physical Properties, Kartoatmodjo and Schmidt (1991) SPE-23556
            - Applied Multiphase Flow in Pipes (2017) Al-Safran and Brill, Page 289
        """
        y = 10 ** (-0.00081 * rs)
        f = (0.2001 + 0.8428 * 10 ** (-0.000845 * rs)) * uod ** (0.43 + 0.5165 * y)
        uol = -0.06821 + 0.9824 * f + 0.0004034 * f**2
        return uol

    @staticmethod
    def viscosity_live_kartoatmodjo_above(uob: float, press: float, pbp: float) -> float:
        """Live Oil Viscosity, Above Bubble Point

        Live oil viscosity above bubblepoint following Kartoatmodjo methodology.

        Args:
            uob (float): Live Oil Viscosity at Bubblepoint, cP
            press (float): Pressure, psig
            pbp (float): Bubblepoint Pressure, psig

        Returns:
            uol (float): Live Oil Viscosity, cP

        References:
            - New Correlations for Crude Oil Physical Properties, Kartoatmodjo and Schmidt (1991) SPE-23556
            - Applied Multiphase Flow in Pipes (2017) Al-Safran and Brill, Page 290
        """
        uol = 1.00081 * uob + 0.001127 * (press - pbp) * (-0.006517 * uob**1.8148 + 0.038 * uob**1.590)
        return uol

    @staticmethod
    def tension_dead_abdul(temp: float, oil_api: float) -> float:
        """Surface Tension Dead Oil

        Dead oil surface tension from the Abdul-Majeed 2000 Method

        Args:
            temp (float): Oil Temperature, deg F
            oil_api (float): Oil API Degrees

        Returns:
            sigod (float): Dead Oil Surface Tension, dyne/cm

        References:
            - Estimation of gas - oil surface tension, G.H. Abdul-Majeed, J Petroleum Science (2000)"""

        temp_c = (5 / 9) * (temp - 32)
        A = 1.11591 - 0.00305 * temp_c
        sigod = A * (38.085 - 0.259 * oil_api)
        return sigod

    @staticmethod
    def tension_live_abdul(sigod: float, rs: float) -> float:
        """Surface Tension Live Oil

        Live oil surface tension from the Abdul-Majeed 2000 Method

        Args:
            sigod (float): Dead Oil Surface Tension, dyne/cm
            oil_api (float): Gas Solubility, scf/stb

        Returns:
            sigol (float): Live Oil Surface Tension, dyne/cm

        References:
            - Estimation of gas - oil surface tension, G.H. Abdul-Majeed, J Petroleum Science (2000)
        """
        rs_si = rs * 7.4801 / 42  # units of ft3/ft3, equivalent to m3/m3
        if rs_si < 50:
            sigol = sigod / (1 + 0.02549 * rs_si**1.0157)
        else:
            sigol = sigod * 32.0436 * rs_si**-1.1367
        return sigol
