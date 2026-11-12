"""Regimes, label sets and the product label map (frozen 2026-10-31, before any model was run).

Break dates are taken from the data (first/last day each product label appears, see
results/label_dates.md): R0 ends 2017-04-22 (no narratives on 04-23), R1 starts 2017-04-24;
R1 labels last appear 2023-08-25 (16 stragglers), R2 labels start 2023-08-25.

Edge types:
  same      label unchanged
  rename    1:1, new name
  merge     several old labels -> one new label
  successor old label splits; the new label that keeps the old name takes the carry-over,
            the carved-out part is a NEW class
  none      old label splits with no name-continuous successor: no carry-over
New classes (no carry-over at the break): R1 'Vehicle loan or lease';
R2 'Prepaid card', 'Debt or credit management'.
"""
import pandas as pd

BREAKS = {"R1": "2017-04-24", "R2": "2023-08-25"}

R0 = ["Bank account or service", "Consumer Loan", "Credit card", "Credit reporting",
      "Debt collection", "Money transfers", "Mortgage", "Payday loan", "Student loan",
      "Prepaid card", "Other financial service", "Virtual currency"]
R1 = ["Checking or savings account", "Credit card or prepaid card",
      "Credit reporting, credit repair services, or other personal consumer reports",
      "Debt collection", "Money transfer, virtual currency, or money service", "Mortgage",
      "Payday loan, title loan, or personal loan", "Student loan", "Vehicle loan or lease"]
R2 = ["Checking or savings account", "Credit card", "Prepaid card",
      "Credit reporting or other personal consumer reports", "Debt collection",
      "Debt or credit management", "Money transfer, virtual currency, or money service",
      "Mortgage", "Payday loan, title loan, personal loan, or advance loan", "Student loan",
      "Vehicle loan or lease"]
LABELS = {"R0": R0, "R1": R1, "R2": R2}

# old label -> (new label or None, edge type)
MAP = {
    ("R0", "R1"): {
        "Bank account or service": ("Checking or savings account", "rename"),
        "Credit card": ("Credit card or prepaid card", "merge"),
        "Prepaid card": ("Credit card or prepaid card", "merge"),
        "Credit reporting": ("Credit reporting, credit repair services, or other personal consumer reports", "merge"),
        "Money transfers": ("Money transfer, virtual currency, or money service", "merge"),
        "Virtual currency": ("Money transfer, virtual currency, or money service", "merge"),
        "Payday loan": ("Payday loan, title loan, or personal loan", "merge"),
        "Debt collection": ("Debt collection", "same"),
        "Mortgage": ("Mortgage", "same"),
        "Student loan": ("Student loan", "same"),
        "Consumer Loan": (None, "none"),
        "Other financial service": (None, "none"),
    },
    ("R1", "R2"): {
        "Checking or savings account": ("Checking or savings account", "same"),
        "Credit card or prepaid card": ("Credit card", "successor"),
        "Credit reporting, credit repair services, or other personal consumer reports":
            ("Credit reporting or other personal consumer reports", "successor"),
        "Debt collection": ("Debt collection", "same"),
        "Money transfer, virtual currency, or money service":
            ("Money transfer, virtual currency, or money service", "same"),
        "Mortgage": ("Mortgage", "same"),
        "Payday loan, title loan, or personal loan":
            ("Payday loan, title loan, personal loan, or advance loan", "rename"),
        "Student loan": ("Student loan", "same"),
        "Vehicle loan or lease": ("Vehicle loan or lease", "same"),
    },
}
NEW_CLASSES = {"R1": ["Vehicle loan or lease"], "R2": ["Prepaid card", "Debt or credit management"]}


def regime_of(dates: pd.Series) -> pd.Series:
    r = pd.Series("R0", index=dates.index)
    r[dates >= BREAKS["R1"]] = "R1"
    r[dates >= BREAKS["R2"]] = "R2"
    return r


def map_label(label: str, src: str, dst: str):
    """Map a label from regime src to regime dst (dst later than src); None if no carry-over."""
    order = ["R0", "R1", "R2"]
    cur = label
    for a, b in zip(order[order.index(src):order.index(dst)], order[order.index(src) + 1:order.index(dst) + 1]):
        if cur is None:
            return None
        cur = MAP[(a, b)].get(cur, (None, "none"))[0]
    return cur
