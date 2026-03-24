import json
from pathlib import Path

import pandas as pd


def nunnuh(x):
    return None


filename = (
    Path(__file__).parents[3]
    / "Facility_Engineer"  # noqa: W503
    / "Jet Pump"  # noqa: W503
    / "subsurface_dimensions"  # noqa: W503
    / "dims_nozzle_throat_rev0.xlsx"  # noqa: W503
)

excel_file = pd.ExcelFile(filename)

sheet_names = excel_file.sheet_names

df_dict = pd.read_excel(excel_file, sheet_name=["national", "kobe", "guiberson", "petrolift"])

jp_dict = {}
long_data = []

# just store inner diameter in inches, the area can be calculated very
brand_dict = {}
for key, value in df_dict.items():
    piece_dict = {}
    for piece in ["nozzle", "throat", "ratio"]:

        id_list = list(map(str, value[value["type"] == piece]["number"].to_list()))

        if piece == "ratio" and key != "guiberson":
            var_list = list(map(int, value[value["type"] == piece]["dia_in"].to_list()))
        elif piece == "ratio" and key == "guiberson":
            var_list = list(map(nunnuh, value[value["type"] == piece]["dia_in"].to_list()))
        else:
            var_list = value[value["type"] == piece]["dia_in"].to_list()
            # print(value[value["type"] == piece]["area_in2"].to_list())
        piece_dict.update({piece: dict(zip(id_list, var_list))})

        if piece != "ratio":  # Exclude ratio from CSV output
            for id_val, var_val in zip(id_list, var_list):
                long_data.append([key, piece, id_val, var_val])

    brand_dict.update({key: piece_dict})


jetpump_path = Path(__file__).parents[1] / "data" / "jetpump_dimensions.json"
with open(jetpump_path, "w") as json_file:
    json.dump(brand_dict, json_file, indent=4)

# Save CSV in long format
csv_path = Path(__file__).parents[1] / "data" / "jetpump_dimensions.csv"
df_long = pd.DataFrame(long_data, columns=["brand", "type", "identity", "dia_in"])
df_long.to_csv(csv_path, index=False)
