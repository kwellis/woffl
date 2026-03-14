import math

from woffl.pvt.blackoil import BlackOil
from woffl.pvt.formgas import FormGas
from woffl.pvt.formwat import FormWater


class ResMix:
    def __init__(self, wc: float, fgor: float, oil: BlackOil, wat: FormWater, gas: FormGas) -> None:
        """Reservoir Mixture

        Mixture of oil, water and natural gas. Define the watercut, fgor and classes
        of the BlackOil, FormWater and FormGas.

        Args:
            wc (float): Watercut of the Mixture, 0 to 1
            fgor (int): Formation GOR of the Mixture, scf/stb
            oil (BlackOil): Class with Oil_API, Gas_SG, Bubblepoint
            wat (FormWater): Class with Water SG
            gas (FormGas): Class with Gas SG

        Returns:
            Self
        """

        self.wc = wc  # specify a decimal
        self.fgor = fgor
        self.oil = oil
        self.wat = wat
        self.gas = gas

        # store standard densities so you don't have to call continually
        pstd = 0  # psig standard pressure
        tstd = 60  # deg f standard temperature
        self.rho_oil_std = oil.condition(pstd, tstd).density
        self.rho_wat_std = wat.condition(pstd, tstd).density
        self.rho_gas_std = gas.condition(pstd, tstd).density

    def __repr__(self) -> str:
        # return(f'Mixture with {self.oil.oil_api} API Oil')
        return f"Mixture at {100 * self.wc}% Watercut and {self.fgor} SCF/STB FGOR"

    def condition(self, press: float, temp: float):
        """Set condition of evaluation

        Args:
            press (float): Pressure of the mixture, psig
            temp (float): Temperature of the mixture, deg F

        Returns:
            Self
        """
        # add code that prevents pressure from going negative?
        self.press = press
        self.temp = temp

        self.oil = self.oil.condition(press, temp)
        self.wat = self.wat.condition(press, temp)
        self.gas = self.gas.condition(press, temp)

        # input press in psig
        # input temp in deg F
        self._pressa = self.press + 14.7  # convert psig to psia
        self._tempr = self.temp + 459.67  # convert fahr to rankine
        return self

    def rho_comp(self) -> tuple[float, float, float]:
        """Density Components

        Return the density of the oil, water and gas from the mixture.
        Requires a pressure and temperature condition to previously be set.

        Args:
            None

        Returns:
            rho_oil (float): density of oil, lbm/ft3
            rho_wat (float): density of water, lbm/ft3
            rho_gas (float): density of gas, lbm/ft3
        """
        rho_oil = self.oil.density
        rho_wat = self.wat.density
        rho_gas = self.gas.density
        return rho_oil, rho_wat, rho_gas

    def rho_two(self) -> tuple[float, float]:
        """Density of Two Phases, Liquid and Gas

        Computes the liquid mixture density. Used in multiphase equations for piping.

        Args:
            None

        Returns:
            rho_liq (float): Density of Liquid, lbm/ft3
            rho_gas (float): Density of Gas, lbm/ft3
        """
        rho_oil, rho_wat, rho_gas = self.rho_comp()
        yoil, ywat, ygas = self.volm_fract()  # volume fractions
        rho_liq = self._homogenous_liquid(yoil, ywat, rho_oil, rho_wat)
        return rho_liq, rho_gas

    def rho_mix(self) -> float:
        """Homogenous Mixture Density

        Return the homogenous density of the mixture.
        Requires a pressure and temperature condition to previously be set.

        Args:
            None

        Returns:
            pmix (float): density of mixture, lbm/ft3
        """
        xoil, xwat, xgas = self.mass_fract()
        rho_oil, rho_wat, rho_gas = self.rho_comp()
        rho_mix = self._homogenous_density(xoil, xwat, xgas, rho_oil, rho_wat, rho_gas)
        return rho_mix

    def visc_comp(self) -> tuple[float, float, float]:
        """Viscosity Components

        Return the viscosity of the oil, water and gas from the mixture.
        Requires a pressure and temperature condition to previously be set.

        Args:
            None

        Returns:
            uoil (float): viscosity of oil, cP
            uwat (float): viscosity of water, cP
            ugas (float): viscosity of gas, cP
        """
        uoil = self.oil.viscosity()
        uwat = self.wat.viscosity()
        ugas = self.gas.viscosity()
        return uoil, uwat, ugas

    def visc_two(self) -> tuple[float, float]:
        """Viscosity of Two Phases, Liquid and Gas

        Computes the liquid mixture viscosity. Used in multiphase equations for piping.

        Args:
            None

        Returns:
            uliq (float): Viscosity of Liquid, cP
            ugas (float): Viscosity of Gas, cP
        """
        uoil, uwat, ugas = self.visc_comp()
        yoil, ywat, ygas = self.volm_fract()  # volume fractions
        uliq = self._homogenous_liquid(yoil, ywat, uoil, uwat)
        return uliq, ugas

    def visc_mix(self) -> float:
        """Viscosity of Homogenous Mixture

        Args:
            None

        Returns:
            umix (float): Viscosity of Mixture, cP
        """
        yoil, ywat, ygas = self.volm_fract()
        uoil, uwat, ugas = self.visc_comp()
        umix = self._homogenous_mixture(yoil, ywat, ygas, uoil, uwat, ugas)
        return umix

    def tension(self) -> float:
        """Surface Tension of Liquid Phase

        Args:
            None

        Returns:
            sig_liq (float): Surface Tension Liquid, lbf/ft
        """
        sig_oil = self.oil.tension()
        sig_wat = self.wat.tension()
        yoil, ywat, ygas = self.volm_fract()  # volume fractions
        sig_liq = self._homogenous_liquid(yoil, ywat, sig_oil, sig_wat)
        return sig_liq

    def comp_comp(self) -> tuple[float, float, float]:
        """Compressibility Components

        Return the compressibility of the oil, water and gas from the mixture.
        Requires a pressure and temperature condition to previously be set.

        Args:
            None

        Returns:
            co (float): compressibility of oil, psi**-1
            cw (float): compressibility of oil, psi**-1
            cg (float): compressibility of oil, psi**-1
        """
        co = self.oil.compress()
        cw = self.wat.compress()
        cg = self.gas.compress()
        return co, cw, cg

    def mass_fract(self) -> tuple[float, float, float]:
        """Mass Fractions

        Return the mass fractions of the oil, water and gas from the mixture.
        Requires a pressure and temperature condition to previously be set.
        Uses algebraic relationships of wc, gor and density to solve for mass fractions.

        Args:
            None

        Returns:
            xoil (float): mass fraction of oil in the mixture
            xwat (float): mass fraction of water in the mixture
            xgas (float): mass fraction of gas in the mixture

        References:
            Derivations from Kaelin available on request
        """
        # use static method from below to run the calcs
        xoil, xwat, xgas = self._owg_mass_fraction(
            self.wc, self.fgor, self.oil.gas_solubility(), self.rho_oil_std, self.rho_wat_std, self.rho_gas_std
        )
        return xoil, xwat, xgas

    def volm_fract(self) -> tuple[float, float, float]:
        """Volume Fractions

        Return the volume fractions of the oil, water and gas from the mixture.
        Requires a pressure and temperature condition to previously be set.

        Args:
            None

        Returns:
            yoil (float): volume fraction of oil in the mixture
            ywat (float): volume fraction of water in the mixture
            ygas (float): volume fraction of gas in the mixture
        """
        xoil, xwat, xgas = self.mass_fract()
        rho_oil, rho_wat, rho_gas = self.rho_comp()
        rho_mix = self._homogenous_density(xoil, xwat, xgas, rho_oil, rho_wat, rho_gas)

        yoil = xoil * rho_mix / rho_oil
        ywat = xwat * rho_mix / rho_wat
        ygas = xgas * rho_mix / rho_gas

        return yoil, ywat, ygas

    def nslh(self) -> float:
        """No Slip Liquid Holdup

        Return the no slip liquid holdup of the mixture.

        Args:
            None

        Returns:
            nslh (float): No Slip Liquid Holdup, unitless
        """
        yoil, ywat, ygas = self.volm_fract()
        nslh = self._no_slip_liquid_holdup(yoil, ywat, ygas)
        return nslh

    def cmix(self) -> float:
        """Mixture Speed of Sound

        Return the adiabatic? Speed of Sound in the mixture.
        Requires a pressure and temperature condition to previously be set.

        Args:
            None

        Returns:
            snd_mix (float): speed of sound in the mixture, ft/s

        References:
            Sound Speed in the Mixture Water-Air D.Himr (2009)
        """
        co, cw, cg = self.comp_comp()  # isothermal compressibility
        yoil, ywat, ygas = self.volm_fract()  # volume fractions
        rho_s = self.rho_mix()
        cs = self._homogenous_mixture(yoil, ywat, ygas, co, cw, cg)  # mixture compressibility
        ks = 1 / cs  # mixture bulk modulus of elasticity
        snd_mix = math.sqrt(32.174 * 144 * ks / rho_s)  # speed of sound, ft/s
        return snd_mix

    def insitu_volm_flow(self, qoil_std: float) -> tuple[float, float, float]:
        """Insitu Volumetric Flow of Components

        Calculate the insitu volumetric flow rates of the oil, water and gas in ft3/s

        Args:
            qoil_std (float): Oil Rate, BOPD

        Returns:
            qoil (float): Oil Volumetric Flow, Insitu ft3/s
            qwat (float): Water Volumetric Flow, Insitu ft3/s
            qgas (float): Gas Volumetric Flow, Insitu ft3/s
        """
        yoil, ywat, ygas = self.volm_fract()
        qoil, qwat, qgas = self._static_insitu_volm_flow(qoil_std, self.rho_oil_std, self.oil.density, yoil, ywat, ygas)
        return qoil, qwat, qgas

    @staticmethod
    def _static_insitu_volm_flow(
        qoil_std: float, rho_oil_std: float, rho_oil: float, yoil: float, ywat: float, ygas: float
    ) -> tuple[float, float, float]:
        """Insitu Volumetric Flow of Components

        Calculate the insitu volumetric flow rates of the oil, water and gas in ft3/s

        Args:
            qoil_std (float): Oil Rate, BOPD
            rho_oil_std (float): Density Oil at Std Cond, lbm/ft3
            rho_oil (float): Density Oil Insitu Cond, lbm/ft3
            yoil (float): Volm Fraction Oil Insitu Cond, ft3/ft3
            ywat (float): Volm Fraction Water Insitu Cond, ft3/ft3
            ygas (float): Volm Fraction Gas Insitu Cond, ft3/ft3

        Returns:
            qoil (float): Oil Volumetric Flow, Insitu ft3/s
            qwat (float): Water Volumetric Flow, Insitu ft3/s
            qgas (float): Gas Volumetric Flow, Insitu ft3/s
        """
        # 42 gal/bbl, 7.48052 gal/ft3, 24 hr/day, 60min/hour, 60sec/min
        qoil_cfs = qoil_std * 42 / (24 * 60 * 60 * 7.48052)  # ft3/s at standard conditions
        moil = qoil_cfs * rho_oil_std  # mass flow of oil
        qoil = moil / rho_oil  # actual flow, ft3/s

        qtot = qoil / yoil  # oil flow divided by oil total fraction
        qwat = ywat * qtot
        qgas = ygas * qtot
        return qoil, qwat, qgas

    def insitu_mass_flow(self, qoil_std: float) -> tuple[float, float, float]:
        """Insitu Mass Flow of Components

        Calculate the insitu mass flow rates of the oil, water and gas in lbm/s
        The fractions change with pressure and temperature from the gas solubility in the oil.
        As a result the mass fractions cannot be assumed to be constant in the process.

        Args:
            qoil_std (float): Oil Rate, BOPD

        Returns:
            moil (float): Oil Mass Flow, lbm/s
            mwat (float): Water Mass Flow, lbm/s
            mgas (float): Gas Mass Flow, lbm/s
        """
        xoil, xwat, xgas = self.mass_fract()
        moil, mwat, mgas = self._static_insitu_mass_flow(qoil_std, self.rho_oil_std, xoil, xwat, xgas)
        return moil, mwat, mgas

    @staticmethod
    def _static_insitu_mass_flow(
        qoil_std: float, rho_oil_std: float, xoil: float, xwat: float, xgas: float
    ) -> tuple[float, float, float]:
        """Insitu Mass Flow of Components

        Calculate the insitu mass flow rates of the oil, water and gas in lbm/s
        The fractions change with pressure and temperature from the gas solubility in the oil.
        As a result the mass fractions cannot be assumed to be constant in the process.

        Args:
            qoil_std (float): Oil Rate, BOPD
            rho_oil_std (float): Density Oil at Std Cond, lbm/ft3
            xoil (float): Mass Fraction Oil Insitu Cond, lbm/lbm
            xwat (float): Mass Fraction Water Insitu Cond, lbm/lbm
            xgas (float): Mass Fraction Gas Insitu Cond, lbm/lbm

        Returns:
            moil (float): Oil Mass Flow, lbm/s
            mwat (float): Water Mass Flow, lbm/s
            mgas (float): Gas Mass Flow, lbm/s
        """
        # 42 gal/bbl, 7.48052 gal/ft3, 24 hr/day, 60min/hour, 60sec/min
        qoil_cfs = qoil_std * 42 / (24 * 60 * 60 * 7.48052)  # ft3/s at standard conditions
        moil = qoil_cfs * rho_oil_std  # mass flow of oil
        mtot = moil / xoil
        mwat = xwat * mtot
        mgas = xgas * mtot
        return moil, mwat, mgas

    @staticmethod
    def _owg_mass_fraction(
        wc: float, fgor: float, rs: float, rho_oil_std: float, rho_wat_std: float, rho_gas_std: float
    ) -> tuple[float, float, float]:
        """Oil Water and Gas Mass Fractions

        Return the mass fractions of the oil, water and gas from the mixture.
        Uses algebraic relationships of wc, gor and density to solve for mass fractions.
        Densities are evaluated at standard conditions

        Args:
            wc (float): Watercut of the Mixture, 0 to 1
            fgor (float): Formation GOR of the Mixture, scf/stb
            rs (float): Gas Solubility of the Oil, scf/stb
            rho_oil_std (float): Oil Density Standard Conditions, lbm/ft3
            rho_wat_std (float): Water Density Standard Conditions, lbm/ft3
            rho_gas_std (float): Gas Density Standard Conditions, lbm/ft3

        Returns:
            xoil (float): mass fraction of oil in the mixture
            xwat (float): mass fraction of water in the mixture
            xgas (float): mass fraction of gas in the mixture

        References:
            Derivations from Kaelin available on request
        """
        # mass based gas solubility
        mrs = (7.48 / 42) * rs * rho_gas_std / rho_oil_std
        # mass formation gas oil ratio, convert from scf/bbl to scf/cf
        mfgor = (7.48 / 42) * fgor * rho_gas_std / rho_oil_std
        # mass watercut
        mwc = rho_wat_std * wc / (rho_wat_std * wc + rho_oil_std * (1 - wc))
        # mass formation gas to liquid ratio
        mfglr = (7.48 / 42) * fgor * rho_gas_std * (1 - wc) / (wc * rho_wat_std + (1 - wc) * rho_oil_std)
        # mass fraction of gas
        xgas = mfglr / (1 + mfglr)
        # mass fraction of oil
        xoil = (1 - mwc) / (1 + mfgor * (1 - mwc))
        # mass fraction of water
        xwat = 1 - xoil - xgas
        # correct for the gas that is inside the oil
        xrs = xoil * mrs
        # calculate new mass fraction of free gas
        xgas = xgas - xrs
        # make sure mass fraction of gas is always zero or above
        xgas = max(xgas, 0)
        xoil = 1 - xwat - xgas
        # mass fractions
        return xoil, xwat, xgas

    @staticmethod
    def _homogenous_liquid(yoil: float, ywat: float, prop_oil: float, prop_wat: float) -> float:
        """Mixture Property of Homogenous Liquid

        Uses common assumption of homogenous liquid for calculating properties.
        Properties could be density, viscosity or surface tension.

        Args:
            yoil (float): Volume fraction of oil in the mixture
            ywat (float): Volume fraction of water in the mixture
            prop_oil (float): Property of the Oil
            prop_wat (float): Property of the Wat

        Return:
            prop_liq (float): Property of the Liquid

        References:
            - Applied Multiphase Flow in Pipes (2017) Al-Safran and Brill, Page 35
        """
        fw = ywat / (yoil + ywat)  # water fraction of the liquid
        prop_liq = prop_oil * (1 - fw) + prop_wat * fw
        return prop_liq

    @staticmethod
    def _homogenous_mixture(
        yoil: float, ywat: float, ygas: float, prop_oil: float, prop_wat: float, prop_gas: float
    ) -> float:
        """Property of Homogenous Mixture

        Uses common assumption of homogenous mixture for calculating properties.
        Only used for homogenous viscosity, density uses mass fractions instead.

        Args:
            yoil (float): Volume fraction of oil in the mixture
            ywat (float): Volume fraction of water in the mixture
            ygas (float): Volume fraction of gas in the mixture
            prop_oil (float): Property of the Oil
            prop_wat (float): Property of the Water
            prop_gas (float): Property of the Gas

        Return:
            prop_mix (float): Property of the Homogenous Mixture
        """
        prop_mix = yoil * prop_oil + ywat * prop_wat + ygas * prop_gas
        return prop_mix

    @staticmethod
    def _homogenous_density(
        xoil: float, xwat: float, xgas: float, rho_oil: float, rho_wat: float, rho_gas: float
    ) -> float:
        """Density of Homogenous Mixture

        Density uses mass fractions instead of volume fractions.
        That will prevent a weird circular logic loop from occuring.

        Args:
            xoil (float): Mass fraction of oil in the mixture
            xwat (float): Mass fraction of water in the mixture
            xgas (float): Mass fraction of gas in the mixture
            rho_oil (float): Density of the Oil
            rho_wat (float): Density of the Water
            rho_gas (float): Density of the Gas

        Return:
            rho_mix (float): Density of the Homogenous Mixture
        """
        vol_mix = (xoil / rho_oil) + (xwat / rho_wat) + (xgas / rho_gas)  # mixture specific volume
        rho_mix = 1 / vol_mix
        return rho_mix

    @staticmethod
    def _no_slip_liquid_holdup(yoil: float, ywat: float, ygas: float) -> float:
        """No Slip Liquid Holdup

        Use the volume fractions of the oil, water and gas.
        Calculate the no slip liquid holdup.

        Args:
            yoil (float): Volume fraction of oil in the mixture
            ywat (float): Volume fraction of water in the mixture
            ygas (float): Volume fraction of gas in the mixture

        Return:
            nslh (float): No Slip Liquid Holdup
        """
        nslh = (yoil + ywat) / (yoil + ywat + ygas)
        return nslh
