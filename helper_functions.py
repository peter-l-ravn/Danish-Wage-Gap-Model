import numpy as np

def clean_name(name):
    """
    Convert python-style names to LaTeX-safe command names.
    
    Example:
    theta_l_y -> thetaLY
    wage_h_o_ss -> wageHOSS
    """

    parts = name.split("_")

    return parts[0] + "".join(p for p in parts[1:])


def params_to_latex(par, filename="params.tex", prefix="par"):
    
    lines = []

    for key, value in vars(par).items():

        latex_key = clean_name(key)

        # handle None
        if value is None:
            value_str = "None"

        # handle numpy scalars
        elif isinstance(value, np.generic):
            value_str = f"{value.item()}"

        # handle floats
        elif isinstance(value, float):
            value_str = f"{value:.10g}"

        else:
            value_str = str(value)

        line = rf"\newcommand{{\{prefix}{latex_key}}}{{{value_str}}}"
        lines.append(line)

    latex_code = "\n".join(lines)

    with open(filename, "w") as f:
        f.write(latex_code)

    # return latex_code