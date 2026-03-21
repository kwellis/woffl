import math


class FormGas:
    # ideal gas constant
    _R = 10.73  # psia*ft^3/(lbmol*Rankine)

    def __init__(self, gas_sg) -> None:
        """Initialize a Formation Gas Stream

        Args:
            gas_sg (float): Free Gas Specific Gravity

        Returns:
            Self
        """

        if (0.5 < gas_sg < 1.2) is False:
            # do I need more here?
            raise ValueError(f"Gas SG {gas_sg} Outside Range")

        # define a lot of properties just on the gas' specific gravity
        self.gas_sg = gas_sg
        self.ppc, self.tpc = self._sutton_pseudo_crit(gas_sg)
        self.mw = 28.96443 * gas_sg  # molecular weight

    def __repr__(self):
        return f"Gas: {self.gas_sg} SG and {self.mw} Mol Weight"

    @classmethod
    def schrader(cls):
        """Schrader Bluff Generic Formation Gas

        Args:
            gas_sg (float): 0.65"""
        return cls(gas_sg=0.65)

    @classmethod
    def kuparuk(cls):
        """Kuparuk Generic Formation Gas

        Args:
            gas_sg (float): 0.65"""
        return cls(gas_sg=0.65)

    @classmethod
    def sag(cls):
        """Sagavanirktok Generic Formation Gas

        Args:
            gas_sg (float): 0.65
        """
        return cls(gas_sg=0.65)

    @classmethod
    def ugnu(cls):
        """Ugnu Generic Formation Gas

        Args:
            gas_sg (float): 0.65
        """
        return cls(gas_sg=0.65)

    @classmethod
    def methane_gas(cls):
        """Pure Methane Formation Gas

        Args:
            gas_sg (float): 0.55"""
        return cls(gas_sg=0.55)

    # almost need seperate function to change pressure / temperature
    def condition(self, press, temp):
        """Set condition of evaluation

        Args:
            press (float): Pressure of the gas, psig
            temp (float): Temperature of the gas, deg F

        Returns:
            Self
        """
        self.press = press
        self.temp = temp

        self.pabs = press + 14.7  # convert psig to psia
        self.tabs = temp + 459.67  # convert fahr to rankine

        # not adjusted for non-hydrocarbon gas such as H2S or CO2
        self.ppr = self.pabs / self.ppc  # unitless, pressure pseudo reduced
        self.tpr = self.tabs / self.tpc  # unitless, temperature pseudo reduced
        return self

    @property
    def zfactor(self) -> float:
        """Gas Z-Factor Compressibility

        Args:
            None

        Returns:
            zfactor(float): gas zfactor, no units
        """
        zfactor = self._zfactor_grad_school(self.ppr, self.tpr)
        # zfactor = self._zfactor_dak(self.ppr, self.tpr)
        return zfactor

    @property
    def density(self) -> float:
        """Gas Density, lbm/ft3

        Return the density of gas.

        Args:
            None

        Returns:
            rho_gas (float): density of the gas, lbm/ft3

        References:
            Fundamental Principles of Reservoir Engineering, B.Towler (2002) Page 16
            Applied Multiphase Flow in Pipes..., Al-Safran and Brill (2017) Page 305
        """
        rho_gas = self.pabs * self.mw / (self.zfactor * FormGas._R * self.tabs)
        return rho_gas

    @property
    def viscosity(self) -> float:
        """Gas Viscosity, cP

        Return the gas viscosity, cP

        Args:
            None

        Returns:
            ugas (float): gas viscosity, cP
        """
        ug = self._viscosity_lee(self.tabs, self.mw, self.density)
        return ug

    @property
    def compress(self) -> float:
        """Gas Compressibility Isothermal

        Calculate isothermal gas compressibility.

        Args:
            None

        Returns:
            cg (float): gas compressibility, psi**-1

        References:
            Fundamental Principles of Reservoir Engineering, B.Towler (2002) Page 16
            Applied Multiphase Flow in Pipes..., Al-Safran and Brill (2017) Page 305
        """
        temp = self.temp
        p1 = self.press
        p2 = p1 + 10  # add 100 psi to evaluate a different condition for compressibility

        z1 = self.zfactor
        z2 = self.condition(p2, temp).zfactor

        cg = 1 / p1 - (1 / z1) * ((z2 - z1) / (p2 - p1))
        return cg

    def hydrate_check(self) -> tuple[bool, float]:
        """DO NOT USE

        Hydrate Formation Check
        Checks to see if the specified pressure and temperature is at risk of hydrates.
        Calculate hydrate formation temperature at specified pressure.
        Requires a pressure and temperature to be specified.

        Args:
            None

        Returns:
            hydrate risk (bool): True or False
            hydrate temp (float): Return Temperature of hydrate formation at pressure

        References:
            - Applied Multiphase Flow in Pipes (2017) Al-Safran and Brill Page 176
        """
        # placeholder for future work
        return True, 55

    @staticmethod
    def _viscosity_lee(tempr: float, mw: float, rho_gas: float) -> float:
        """Lee et. al Gas Viscosity

        Gas viscosity following Lee 1966 et al methodology.

        Args:
            tempr (float): Absolute Temperature of Gas, deg R
            mw (float): Molecular Weight of Gas Mixture, lb/lb-mol
            rho_gas (float): In-Situ Density of Gas, lbm/ft3

        Returns:
            ug (float): Gas Viscosity, cP

        References:
            - Applied Multiphase Flow in Pipes (2017) Al-Safran and Brill, Page 304
        """
        # tempr = temp + 459.67
        K = ((9.4 + 0.02 * mw) * tempr**1.5) / (209 + 19 * mw + tempr)
        X = 3.5 + (986 / tempr) + 0.01 * mw
        Y = 2.4 - 0.2 * X
        ug = (10**-4) * K * math.exp(X * (rho_gas / 62.4) ** Y)
        return ug

    @staticmethod
    def _sutton_pseudo_crit(gas_sg: float) -> tuple[float, float]:
        """Sutton Correlation for Pseudo Critical Pressure and Temperature

        Args:
            gas_sg (float): Gas Specific Gravity, unitless

        Returns:
            ppc (float): Pseudo Critical Pressure, psia
            tpc (float): Pseudo Critical Temperature, deg R

        References:
            - Compressibility Factors for High MW Gases (1985) Sutton, R.P, SPE 14265
        """
        ppc = 756.8 - 131.07 * gas_sg - 3.6 * gas_sg**2  # psia
        tpc = 169.2 + 349.5 * gas_sg - 74.0 * gas_sg**2  # deg R
        return ppc, tpc

    @staticmethod
    def _zfactor_grad_school(ppr: float, tpr: float) -> float:
        """Z Factor Grad School Method

        A method that was on on graduate school excel file that I borrowed.
        Much faster than the Dranchuk Abu Kassem method, unknown accuracy.

        Args:
            ppr (float): Pseudo Reduced Pressure, unitless
            tpr (float): Pseudo Reduced Pressure, unitless

        Returns:
            zfactor (float): Compressibility of Natural Gas, unitless
        """
        m = 0.51 * (tpr) ** (-4.133)
        n = 0.038 - 0.026 * (tpr) ** (1 / 2)
        zfactor = 1 - m * ppr + n * ppr**2 + 0.0003 * ppr**3
        return zfactor

    @staticmethod
    def _zfactor_dak(ppr: float, tpr: float) -> float:
        """Z Factor Dranchuk and Abu-Kassem

        Blasingame (1988) PVT Appendix has good description.

        Args:
            ppr (float): Pseudo Reduced Pressure, unitless
            tpr (float): Pseudo Reduced Temperature, unitless

        Return:
            zfactor (float): Gas Compressibility, unitless
        """
        z_ray = [0.95, 0.91]  # start calculation with a guess
        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = FormGas._dak_constants()
        c1 = FormGas._dak_c1(tpr, a1, a2, a3, a4, a5)
        c2 = FormGas._dak_c2(tpr, a6, a7, a8)
        c3 = FormGas._dak_c3(tpr, a7, a8, a9)

        # need to loop this section for z-factor convergence
        while abs(z_ray[-2] - z_ray[-1]) > 0.001:
            rho_pr = FormGas._dak_rho_pr(z_ray[-1], ppr, tpr)
            c4 = FormGas._dak_c4(rho_pr, tpr, a10, a11)
            zfun = FormGas._dak_zfun(z_ray[-1], rho_pr, c1, c2, c3, c4)
            c5 = FormGas._dak_c5(z_ray[-1], rho_pr, tpr, a10, a11)
            zder = FormGas._dak_zderv(z_ray[-1], rho_pr, c1, c2, c3, c5)
            znew = z_ray[-1] - zfun / zder  # newtons method
            z_ray.append(znew)

        return z_ray[-1]

    @staticmethod
    def _dak_constants() -> tuple[float, float, float, float, float, float, float, float, float, float, float]:
        """Dranchuk and Abu-Kassem Constants

        Returns the constants used in the Dranchuk and Abu-Kassem Z factor Correlation
        """
        a1 = 0.3265
        a2 = -1.07
        a3 = -0.5339
        a4 = 0.01569
        a5 = -0.05165
        a6 = 0.5475
        a7 = -0.7361
        a8 = 0.1844
        a9 = 0.1056
        a10 = 0.6134
        a11 = 0.7210
        return a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11

    @staticmethod
    def _dak_c1(tpr: float, a1: float, a2: float, a3: float, a4: float, a5: float) -> float:
        """Dranchuk and Abu-Kassem C1

        First Part of the Dranchuk and Abu-Kassem Equation

        Args:
            tpr (float): Pseudo Reduced Temperature, unitless

        Returns:
            c1 (float): Dranchuk and Abu-Kassem C1
        """
        c1 = a1 + a2 / tpr + a3 / tpr**3 + a4 / tpr**4 + a5 / tpr**5
        return c1

    @staticmethod
    def _dak_c2(tpr: float, a6: float, a7: float, a8: float) -> float:
        """Dranchuk and Abu-Kassem C2

        Second Part of the Dranchuk and Abu-Kassem Equation

        Args:
            tpr (float): Pseudo Reduced Temperature, unitless

        Returns:
            c2 (float): Dranchuk and Abu-Kassem C2
        """
        c2 = a6 + a7 / tpr + a8 / tpr**2
        return c2

    @staticmethod
    def _dak_c3(tpr: float, a7: float, a8: float, a9: float) -> float:
        """Dranchuk and Abu-Kassem C3

        Third Part of the Dranchuk and Abu-Kassem Equation

        Args:
            tpr (float): Pseudo Reduced Temperature, unitless

        Returns:
            c3 (float): Dranchuk and Abu-Kassem C3
        """
        c3 = a9 * (a7 / tpr + a8 / tpr**2)
        return c3

    @staticmethod
    def _dak_c4(rho_pr: float, tpr: float, a10: float, a11: float) -> float:
        """Dranchuk and Abu-Kassem C4

        Fourth Part of the Dranchuk and Abu-Kassem Equation

        Args:
            rho_pr (float): Density Pseudo Reduced, units...?
            tpr (float): Temperature Pseudo Reduced, unitless

        Returns:
            c4 (float): Dranchuk and Abu-Kassem C4
        """
        c4 = a10 * (1 + a11 * rho_pr**2) * (rho_pr**2 / tpr**3) * math.exp(-a11 * rho_pr**2)
        return c4

    @staticmethod
    def _dak_rho_pr(zf: float, ppr: float, tpr: float) -> float:
        """Pseudo Reduced Density

        Used in the Dranchuk and Abu-Kassem Calculation

        Args:
            zf (float): zfactor, unitless
            ppr (float): Pressure Pseudo Reduced, unitless
            tpr (float): Temperature Pseudo Reduced, unitless

        Returns:
            rho_pr (float): Density Pseudo Reduced, units...?
        """
        rho_pr = 0.27 * ppr / (zf * tpr)
        return rho_pr

    @staticmethod
    def _dak_zfun(zf: float, rho_pr: float, c1: float, c2: float, c3: float, c4: float) -> float:
        """Calculated Z Factor with Dranchuk and Abu-Kassem Calculation

        Used to compare to the guess value to create a residual to minimize.
        Outputs the residual of the guess vs calculated zfactor. Needed to minimize.

        Args:
            zf (float): Input zfactor, unitless
            rho_pr (float): Density Pseudo Reduced, units...?
            tpr (float): Temperature Pseudo Reduced, unitless
            c1 (float): DAK C1, unitless
            c2 (float): DAK C2, unitless
            c3 (float): DAK C3, unitless
            c4 (float): DAK C4, unitless

        Return:
            zres (float): Residual of the Input zfactor vs calculated
        """
        zres = zf - (1 + c1 * rho_pr + c2 * rho_pr**2 - c3 * rho_pr**5 + c4)
        return zres

    @staticmethod
    def _dak_c5(zf: float, rho_pr: float, tpr: float, a10: float, a11: float) -> float:
        """Dranchuk and Abu-Kassem C4

        Used to compare to the guess value to create a residual to minimize.
        Add in the zf, which is the guess value, get the residual?

        Args:
            zf (float): zfactor, unitless
            rho_pr (float): Density Pseudo Reduced, units...?
            tpr (float): Temperature Pseudo Reduced, unitless
            a10 (float): DAK a1, unitless
            a11 (float): DAK a2, unitless

        Return:
            c5 (float): DAK C5, used in derivative equation
        """
        r1 = 2 * a10 * rho_pr**2 / (tpr**3 * zf)
        r2 = 1 + a11 * rho_pr**2 - (a11 * rho_pr**2) ** 2
        c5 = r1 * r2 * math.exp(-a11 * rho_pr**2)
        return c5

    @staticmethod
    def _dak_zderv(zf: float, rho_pr: float, c1: float, c2: float, c3: float, c5: float) -> float:
        """Z Factor Derivative from Dranchuk and Abu-Kassem

        Analytical derivative of the DAK zfun residual function.
        Used in Newtons Method.

        Args:
            zf (float): zfactor
            rho_pr (float): Density Pseudo Reduced, units...?
            tpr (float): Temperature Pseudo Reduced, unitless
            c1 (float): DAK C1, unitless
            c2 (float): DAK C2, unitless
            c3 (float): DAK C3, unitless
            c5 (float): DAK C5, unitless

        Returns:
            zderv (float): Z Factor Derivative
        """
        zderv = 1 + c1 * rho_pr / zf + 2 * c2 * rho_pr**2 / zf - 5 * c3 * rho_pr**5 / zf + c5
        return zderv
