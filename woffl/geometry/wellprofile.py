import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize


class WellProfile:
    """Well Profile Class

    Create a wellprofile, which is the subsurface geometry of the measured depth versus vertical depth.
    Can be used to interpolate values for understanding how measured depth relates to vertical depth.
    """

    def __init__(self, md_list: list | np.ndarray, vd_list: list | np.ndarray, jetpump_md: float) -> None:
        """Create a Well Profile

        Args:
            md_list (list): List of measured depths
            vd_list (list): List of vertical depths
            jetpump_md (float): Measured depth of Jet Pump, feet

        Returns:
            Self
        """
        if len(md_list) != len(vd_list):
            raise ValueError("Lists for Measured Depth and Vertical Depth need to be the same length")

        if max(md_list) < max(vd_list):
            raise ValueError("Measured Depth needs to extend farther than Vertical Depth")

        if jetpump_md > max(md_list):
            raise ValueError("Jet pump not inside well profile measured depth")

        md_ray, vd_ray = sort_profile(np.array(md_list), np.array(vd_list))

        self.md_ray = md_ray
        self.vd_ray = vd_ray
        self.hd_ray = self._horz_dist(self.md_ray, self.vd_ray)
        self.jetpump_md = jetpump_md

        # here is the real question, do I even need the raw data? just filter the data
        # at the __init__ and be done? Is the raw data just more stuff to wade through?
        self.hd_fit, self.vd_fit, self.md_fit = self.filter()  # run the method

    def __repr__(self):
        final_md = round(self.md_ray[-1], 0)
        final_vd = round(self.vd_ray[-1], 0)

        return f"Profile is {final_md} ft. long and {final_vd} ft. deep"

    def vd_interp(self, md_dpth: float) -> float:
        """Vertical Depth Interpolation

        Args:
            md_dpth (float): Measured Depth, feet

        Returns:
            vd_dpth (float): Vertical Depth, feet
        """
        return self._depth_interp(md_dpth, self.md_ray, self.vd_ray)

    def hd_interp(self, md_dpth: float) -> float:
        """Horizontal Distance Interpolation

        Args:
            md_dpth (float): Measured Depth, feet

        Returns:
            hd_dist (float): Horizontal Distance, feet
        """
        return self._depth_interp(md_dpth, self.md_ray, self.hd_ray)

    def md_interp(self, vd_dpth: float) -> float:
        """Measured Depth Interpolation

        Args:
            vd_dpth (float): Vertical Depth, feet

        Returns:
            md_dpth (float): Measured Depth, feet
        """
        return self._depth_interp(vd_dpth, self.vd_ray, self.md_ray)

    @property
    def jetpump_vd(self) -> float:
        """Jet Pump True Vertical Depth, Feet"""
        jp_vd = self.vd_interp(self.jetpump_md)
        return jp_vd

    @property
    def jetpump_hd(self) -> float:
        """Jet Pump Horizontal Distance, Feet"""
        jp_hd = self.hd_interp(self.jetpump_md)
        return jp_hd

    # make a plot of the filtered data on top of the raw data
    def plot_raw(self) -> None:
        """Plot the Raw Profile Data"""

        self._profileplot(self.hd_ray, self.vd_ray, self.md_ray, self.jetpump_hd, self.jetpump_vd, self.jetpump_md)
        return None

    def plot_filter(self) -> None:
        """Plot the Filtered Data"""
        self._profileplot(self.hd_fit, self.vd_fit, self.md_fit, self.jetpump_hd, self.jetpump_vd, self.jetpump_md)
        return None

    def filter(self):
        """Filter WellProfile to the Minimal Data Points

        Feed the method the raw measured depth and raw true vertical depth data.
        Method will assess the raw data and represent it in the smallest number of data points.
        Method uses the segments fit function from above.

        Args:

        Returns:
            md_fit (np array): Measured Depth Filtered Data
            vd_fit (np array): Vertical Depth Filter Data
            hd_fit (np array): Horizontal Dist Filter Data
        """
        # have to use md since the valve will always be increasing
        md_fit, vd_fit = segments_fit(self.md_ray, self.vd_ray)
        md_fit[0], vd_fit[0] = 0, 0  # first values always need to start at zero
        idx = np.searchsorted(self.md_ray, md_fit)  # why is this pulling one more? because md_ray and

        hd_fit = self.hd_ray[idx]
        return hd_fit, vd_fit, md_fit

    def outflow_spacing(self, seg_len: float) -> tuple[np.ndarray, np.ndarray]:
        """Outflow Piping Spacing

        Break the outflow piping into nodes that can be fed  with piping dimension
        flowrates and etc to calculate differential pressure across them. Outflow is
        assumed to start at the jetpump.

        Args:
            seg_len (float): Segment Length of Outflow Piping, feet

        Returns:
            md_seg (np array): Measured depth Broken into segments
            vd_seg (np array): Vertical depth broken into segments
        """
        return self._outflow_spacing(self.md_fit, self.vd_fit, self.jetpump_md, seg_len)

    @staticmethod
    def _outflow_spacing(
        md_fit: np.ndarray, vd_fit: np.ndarray, outflow_md: float, seg_len: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Outflow Piping Spacing

        Break the outflow piping into nodes that can be fed  with piping dimension
        flowrates and etc to calculate differential pressure across them.

        Args:
            md_fit (np array): Filtered Measured Depth Data, feet
            vd_fit (np array): Filtered Vertical Depth Data, feet
            outflow_md (float): Depth the outflow node ends, feet
            seg_len (float): Segment Length of Outflow Piping, feet

        Returns:
            md_seg (np array): Measured depth Broken into segments
            vd_seg (np array): Vertical depth broken into segments
        """
        # need to break it up
        outflow_vd = np.interp(outflow_md, md_fit, vd_fit)
        vd_fit = vd_fit[md_fit <= outflow_md]
        md_fit = md_fit[md_fit <= outflow_md]  # keep values less than outflow_md
        md_fit = np.append(md_fit, outflow_md)  # add the final outflow md to end
        vd_fit = np.append(vd_fit, outflow_vd)
        md1 = md_fit[:-1]  # everything but the last character
        md2 = md_fit[1:]  # everything but the first character
        dist = md2 - md1  # distance between points
        md_seg = np.array([])
        for i, dis in enumerate(dist):
            # force there to always be at least three (?) spaces?
            dis = max(int(np.ceil(dis / seg_len)), 3)  # evenly space out the spaces
            md_seg = np.append(md_seg, np.linspace(md1[i], md2[i], dis))  # double counting
        md_seg = np.unique(md_seg)  # get rid of weird double counts from linspace
        vd_seg = np.interp(md_seg, md_fit, vd_fit)
        return md_seg, vd_seg

    @staticmethod
    def _depth_interp(in_dpth: float, in_ray: np.ndarray, out_ray: np.ndarray) -> float:
        """Depth Interpolation

        Args:
            in_dpth (float): Known Depth, feet
            in_ray (list): Known List of Depths, feet
            out_ray (list): Unknown List of Depths, feet

        Returns:
            out_dpth (float): Unknown Depth, Feet
        """
        if (min(in_ray) < in_dpth < max(in_ray)) is False:
            raise ValueError(f"{in_dpth} feet is not inside survey boundary")

        out_dpth = np.interp(in_dpth, in_ray, out_ray)
        return float(out_dpth)

    @staticmethod
    def _horz_dist(md_ray: np.ndarray, vd_ray: np.ndarray) -> np.ndarray:
        """Horizontal Distance from Wellhead

        Args:
            md_ray (np array): Measured Depth array, feet
            vd_ray (np array): Vertical Depth array, feet

        Returns:
            hd_ray (np array): Horizontal Dist array, feet
        """
        md_diff = np.diff(md_ray, n=1)  # difference between values in array
        vd_diff = np.diff(vd_ray, n=1)  # difference between values in array
        hd_diff = np.zeros(1)  # start with zero at top to make array match original size
        hd_diff = np.append(hd_diff, np.sqrt(md_diff**2 - vd_diff**2))  # pythagorean theorem
        hd_ray = np.cumsum(hd_diff)  # rolling sum, previous values are finite differences
        return hd_ray

    @staticmethod
    def _profileplot(
        hd_ray: np.ndarray, vd_ray: np.ndarray, md_ray: np.ndarray, hd_jp: float, vd_jp: float, md_jp: float
    ) -> None:
        """Create a Well Profile Plot

        Annotate the graph will a label of the measured depth every 1000 feet of md.
        Future note, should the hd and vd axis scales be forced to match? How hard is that?

        Args:
            hd_ray (np array): Horizontal distance, feet
            td_ray (np array): Vertical depth, feet
            md_ray (np arary): Measured depth, feet
            hd_jp (float): Horizontal distance jetpump, feet
            vd_jp (float): Vertical depth jetpump, feet
            md_jp (float): Measured depth jetpump, feet
        """
        if len(md_ray) > 20:
            plt.scatter(hd_ray, vd_ray, label="Survey")
        else:
            plt.plot(hd_ray, vd_ray, marker="o", linestyle="--", label="Survey")

        # plot jetpump location
        plt.plot(hd_jp, vd_jp, marker="o", color="r", linestyle="", label=f"Jetpump MD: {int(md_jp)} ft")

        plt.gca().invert_yaxis()
        plt.title(f"Dir Survey, Length: {max(md_ray)} ft")
        plt.xlabel("Horizontal Distance, Feet")
        plt.ylabel("True Vertical Depth, Feet")

        # find the position in the measured depth array closest to every 1000'
        md_match = np.arange(1000, max(md_ray), 1000)
        idxs = np.searchsorted(md_ray, md_match)
        idxs = np.unique(idxs)  # don't repeat values, issue on filtered data

        # annotate every ~1000' of measured depth
        for idx in idxs:
            plt.annotate(
                text=f"{int(md_ray[idx])} ft", xy=(hd_ray[idx] + 5, vd_ray[idx] - 10), rotation=30  # type: ignore
            )
        plt.legend()
        plt.axis("equal")
        plt.show()

    @classmethod
    def schrader(cls):
        """Schrader Bluff Generic Well Profile

        Generic Schrader Bluff well profile based on MPE-42 geometry.
        Imported from generic_wprof.json

        Args:
            md_list (list): MPE-42 Measured Depth
            vd_list (list): MPE-42 Vertical Depth
            jetpump_md (float): 6693 MD, feet
        """
        e42_md, e42_vd = survey_data("MPE-42")
        return cls(md_list=e42_md, vd_list=e42_vd, jetpump_md=6693)

    @classmethod
    def kuparuk(cls):
        """Kuparuk Generic Well Profile

        Generic Kuparuk well profile based on MPC-23 geometry.
        MPC-23 is a slant Kuparuk Well, so not a perfect canidate.
        Imported from generic_wprof.json

        Args:
            md_list (list): MPC-23 Measured Depth
            vd_list (list): MPC-23 Vertical Depth
            jetpump_md (float): 7926 MD, feet
        """
        c23_md, c23_vd = survey_data("MPC-23")
        return cls(md_list=c23_md, vd_list=c23_vd, jetpump_md=7926)


def sort_profile(md_ray: np.ndarray, vd_ray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort Well Profile

    Take in the raw data from various databases. Sort the measured
    depth in ascending order from smallest to largest. Mirror the new
    ordered sort on the vertical array.

    Args:
        md_ray (np array): Measured Depth array, feet
        vd_ray (np array): Vertical Depth array, feet

    Returns:
        md_sort (np array): Sorted Measured Depth array, feet
        vd_sort (np array): Sorted Vertical Depth array, feet
    """
    sort_idxs = np.argsort(md_ray)
    md_sort = md_ray[sort_idxs]
    vd_sort = vd_ray[sort_idxs]
    return md_sort, vd_sort


def segments_fit(X: np.ndarray, Y: np.ndarray, maxcount: int = 18) -> tuple[np.ndarray, np.ndarray]:
    """Segments Fit DankOC

    Feed the method the raw data. Function will assess the raw data.
    Then return the smallest smallest number of data points to fit.

    Args:
        X (numpy array): Raw X Data to be fit
        Y (numpy array): Raw Y Data to be fit
        maxcount (int): Max number of points to represent the profile by

    Returns:
        X_fit (np.array): Filtered X Data
        Y_fit (np.array): Filtered Y Data

    References:
        - https://gist.github.com/ruoyu0088/70effade57483355bbd18b31dc370f2a
        - https://discovery.ucl.ac.uk/id/eprint/10070516/1/AIC_BIC_Paper.pdf
        - Comment from user: dankoc
    """
    xmin, xmax = X.min(), X.max()

    n = len(X)

    best_AIC = float("inf")
    best_BIC = float("inf")
    best_fit = None

    for count in range(1, maxcount + 1):
        seg = np.full(count - 1, (xmax - xmin) / count)

        px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.1].mean() for x in px_init])

        def func(p):
            seg = p[: count - 1]
            py = p[count - 1 :]  # noqa E203
            px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
            return px, py

        def err(p):  # This is RSS / n
            px, py = func(p)
            Y2 = np.interp(X, px, py)
            return np.mean((Y - Y2) ** 2)

        res = optimize.minimize(err, x0=np.r_[seg, py_init], method="Nelder-Mead")

        # Compute AIC/ BIC.
        AIC = n * np.log(err(res.x)) + 4 * count
        BIC = n * np.log(err(res.x)) + 2 * count * np.log(n)

        if (BIC < best_BIC) & (AIC < best_AIC):  # Continue adding complexity.
            best_fit = res
            best_AIC, best_BIC = AIC, BIC
        else:  # Stop.
            count = count - 1
            break

    px, py = func(best_fit.x)  # type: ignore
    px, py = segments_guardrails(px, py, X.max(), Y.max())

    return px, py  # type: ignore [return the last (n-1)]


def segments_guardrails(px: np.ndarray, py: np.ndarray, xmax: float, ymax: float) -> tuple[np.ndarray, np.ndarray]:
    """Segments Guardrails

    Add some guardrails to ensure the returned datapoints don't go past the max values.
    Eliminate one of the values that are redundant. Maybe add something where the first
    values in the array need to be 0,0 as well? That code is repeated in other places.

    Args:
        px (np.array): Filtered x data
        py (np.array): Filtered y data
        xmax (float): Maximum x value from the non-filtered data
        ymax (float): Maximum y value from the non-filtered data

    Returns:
        px (np.array): Filtered x data with guardrails
        py (np.array): Filtered y data with guardrails
    """
    px = np.clip(px, None, xmax)  # Replace any px values longer than xmax with xmax.
    py[px == xmax] = ymax  # Any place that xmax exists in px, replace the corresponding py with ymax.

    combo = np.stack((px, py), axis=-1)  # zip together 1D into 2D that you can review
    unq_vals, unq_idxs = np.unique(combo, return_index=True, axis=0)

    px, py = px[unq_idxs], py[unq_idxs]
    return px, py


def survey_data(well_name: str) -> tuple[list, list]:
    """Survey Data

    Use the survey data from a .JSON file

    Args:
        well_name (str): Well Name

    Returns:
        md_dpth (list): List of Measured Depth, feet
        vd_dpth (list): List of Vertical Depth, feet
    """

    json_path = Path(__file__).parents[0] / "generic_wprof.json"

    # verify the well_name actually exists in the JSON file
    with open(json_path, "r") as json_file:
        survey_dict = json.load(json_file)

    # check if the name exists in the survey dictionary
    if well_name in survey_dict:
        md_dpth = survey_dict[well_name]["md_dpth"]
        vd_dpth = survey_dict[well_name]["vd_dpth"]
        return md_dpth, vd_dpth
    else:
        raise ValueError(f"Well Name {well_name} not found in the Survey Data")
