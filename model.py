#!/usr/bin/env python3
"""
Seed-stage financial model.
Reads assumptions.yaml, projects forward, answers:
  1. How long is my runway?
  2. When should I raise?
  3. How much should I raise?
  4. When am I Series A ready?
"""

import yaml
import json
import os
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Load assumptions ───────────────────────────

BASE_DIR = Path(__file__).parent
with open(BASE_DIR / "assumptions.yaml") as f:
    A = yaml.safe_load(f)

OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── Projection engine ─────────────────────────

def project(scenario_name="base"):
    """Run month-by-month projection. Returns list of monthly snapshots."""

    cfg = A
    months = cfg["projection"]["months"]

    # Find scenario multipliers
    scenario = next(
        (s for s in cfg["projection"]["scenarios"] if s["name"] == scenario_name),
        {"revenue_growth_mult": 1.0, "burn_growth_mult": 1.0}
    )
    rev_mult = scenario["revenue_growth_mult"]
    burn_mult = scenario["burn_growth_mult"]

    # Starting values
    cash = cfg["current"]["cash_in_bank"]
    customers = cfg["revenue"]["current_customers"]
    monthly_growth = cfg["revenue"]["monthly_growth_rate"] * rev_mult
    churn = cfg["revenue"]["churn_rate"]
    price = cfg["revenue"]["price_per_unit"]

    headcount = cfg["team"]["current_headcount"]
    avg_salary_monthly = cfg["team"]["avg_salary"] / 12

    infra = cfg["costs"]["infrastructure"]
    infra_growth = cfg["costs"]["infrastructure_growth"] * burn_mult
    marketing = cfg["costs"]["marketing"]
    mktg_growth = cfg["costs"]["marketing_growth"] * burn_mult
    fixed_costs = cfg["costs"]["office"] + cfg["costs"]["legal_accounting"] + cfg["costs"]["other"]

    start_date = datetime.strptime(cfg["current"]["date"], "%Y-%m")

    # Planned hires schedule
    hires = {h["month"]: h for h in cfg["team"]["planned_hires"]}

    snapshots = []

    for m in range(months + 1):
        date = start_date + relativedelta(months=m)

        # Revenue
        mrr = customers * price
        arr = mrr * 12

        # Costs
        if m in hires:
            headcount += 1
        payroll = headcount * avg_salary_monthly
        total_costs = payroll + infra + marketing + fixed_costs

        # Net burn
        net_burn = total_costs - mrr

        # Runway
        runway_months = cash / net_burn if net_burn > 0 else float('inf')

        snapshots.append({
            "month": m,
            "date": date,
            "date_str": date.strftime("%Y-%m"),
            "customers": round(customers),
            "mrr": round(mrr),
            "arr": round(arr),
            "headcount": headcount,
            "payroll": round(payroll),
            "infra": round(infra),
            "marketing": round(marketing),
            "total_costs": round(total_costs),
            "net_burn": round(net_burn),
            "cash": round(cash),
            "runway_months": round(runway_months, 1) if runway_months != float('inf') else 999,
        })

        # Advance to next month
        new_customers = customers * monthly_growth
        churned = customers * churn
        customers = customers + new_customers - churned
        cash -= net_burn
        infra *= (1 + infra_growth)
        marketing *= (1 + mktg_growth)

    return snapshots


def find_raise_timing(snapshots):
    """Determine when to raise based on runway threshold."""
    min_runway = A["targets"]["min_runway_months"]
    fundraise_duration = A["targets"]["fundraise_duration_months"]

    # When does runway drop below threshold?
    raise_deadline = None
    for s in snapshots:
        if s["runway_months"] < min_runway and s["cash"] > 0:
            raise_deadline = s
            break

    # Start fundraising `fundraise_duration` months before deadline
    start_fundraising = None
    if raise_deadline:
        target_month = raise_deadline["month"] - fundraise_duration
        if target_month >= 0:
            start_fundraising = snapshots[target_month]
        else:
            start_fundraising = snapshots[0]  # start now!

    return start_fundraising, raise_deadline


def recommend_raise_amount(snapshots):
    """How much to raise: enough for target_runway_months of projected burn."""
    target_months = A["targets"]["target_runway_months"]

    # Find the month we'd raise (or month 0 if no deadline)
    start, _ = find_raise_timing(snapshots)
    raise_month = start["month"] if start else 0

    # Average monthly burn over the target period after raising
    future_burns = []
    for s in snapshots:
        if raise_month <= s["month"] <= raise_month + target_months:
            future_burns.append(s["net_burn"])

    if not future_burns:
        future_burns = [snapshots[raise_month]["net_burn"]]

    avg_burn = sum(future_burns) / len(future_burns)
    recommended = avg_burn * target_months

    # Round up to nearest $250K
    recommended = max(recommended, 0)
    recommended = int(np.ceil(recommended / 250000) * 250000)

    return recommended, avg_burn


def series_a_readiness(snapshots):
    """When will the company hit Series A benchmarks?"""
    targets = A["targets"]["series_a"]
    min_arr = targets["min_arr"]
    min_growth = targets["min_arr_growth"]

    for i, s in enumerate(snapshots):
        if s["arr"] >= min_arr:
            # Check growth rate too
            if i > 0:
                prev_mrr = snapshots[i-1]["mrr"]
                if prev_mrr > 0:
                    growth = (s["mrr"] - prev_mrr) / prev_mrr
                    if growth >= min_growth:
                        return s, growth

    return None, None


# ── Run all scenarios ──────────────────────────

scenarios = {}
for sc in A["projection"]["scenarios"]:
    scenarios[sc["name"]] = project(sc["name"])

base = scenarios["base"]

# ── Analysis ───────────────────────────────────

start_raise, raise_deadline = find_raise_timing(base)
raise_amount, avg_burn = recommend_raise_amount(base)
sa_ready, sa_growth = series_a_readiness(base)

# ── Summary output ─────────────────────────────

def fmt_money(n):
    if abs(n) >= 1_000_000:
        return f"${n/1_000_000:.1f}M"
    elif abs(n) >= 1_000:
        return f"${n/1_000:.0f}K"
    else:
        return f"${n:,.0f}"

summary_lines = []
summary_lines.append("=" * 60)
summary_lines.append(f"  {A['company']['name']} -- Financial Model Summary")
summary_lines.append("=" * 60)
summary_lines.append("")

# Current state
s0 = base[0]
summary_lines.append("CURRENT STATE")
summary_lines.append(f"  Cash in bank:      {fmt_money(s0['cash'])}")
summary_lines.append(f"  Monthly revenue:   {fmt_money(s0['mrr'])}")
summary_lines.append(f"  Monthly burn:      {fmt_money(s0['total_costs'])}")
summary_lines.append(f"  Net burn:          {fmt_money(s0['net_burn'])}")
summary_lines.append(f"  Runway:            {s0['runway_months']:.0f} months")
summary_lines.append(f"  Team:              {s0['headcount']} people")
summary_lines.append("")

# Fundraising recommendation
summary_lines.append("FUNDRAISING")
if start_raise:
    summary_lines.append(f"  Start raising by:  {start_raise['date_str']}  (runway hits {A['targets']['min_runway_months']}mo threshold)")
    summary_lines.append(f"  Close by:          {raise_deadline['date_str']}")
else:
    summary_lines.append(f"  Runway is strong -- no urgent need to raise in the projection window")
summary_lines.append(f"  Recommended raise: {fmt_money(raise_amount)}  ({A['targets']['target_runway_months']}mo of projected burn)")
summary_lines.append(f"  Avg future burn:   {fmt_money(avg_burn)}/mo")
summary_lines.append("")

# Series A readiness
summary_lines.append("SERIES A READINESS")
if sa_ready:
    summary_lines.append(f"  Hit {fmt_money(A['targets']['series_a']['min_arr'])} ARR:  {sa_ready['date_str']}  (month {sa_ready['month']})")
    summary_lines.append(f"  MRR at that point: {fmt_money(sa_ready['mrr'])}")
    summary_lines.append(f"  MoM growth:        {sa_growth:.0%}")
    val_est = sa_ready["mrr"] * A["targets"]["series_a"]["target_valuation_multiple"]
    summary_lines.append(f"  Est. valuation:    {fmt_money(val_est)}  ({A['targets']['series_a']['target_valuation_multiple']}x MRR)")
    if sa_ready["cash"] > 0:
        summary_lines.append(f"  Cash at that date: {fmt_money(sa_ready['cash'])}")
    else:
        summary_lines.append(f"  WARNING: Cash runs out before Series A threshold!")
        summary_lines.append(f"  You need to raise before {sa_ready['date_str']}")
else:
    summary_lines.append(f"  Does not reach {fmt_money(A['targets']['series_a']['min_arr'])} ARR in {A['projection']['months']}-month window")
    last = base[-1]
    summary_lines.append(f"  ARR at month {last['month']}: {fmt_money(last['arr'])}")
    summary_lines.append(f"  Consider: faster growth, higher pricing, or longer runway")
summary_lines.append("")

# Monthly projections table (base case)
summary_lines.append("MONTHLY PROJECTIONS (Base Case)")
summary_lines.append(f"  {'Mo':>3}  {'Date':>7}  {'MRR':>8}  {'ARR':>9}  {'Costs':>8}  {'Net Burn':>9}  {'Cash':>10}  {'Runway':>7}  {'Team':>4}")
summary_lines.append("  " + "-" * 78)
for s in base:
    if s["month"] % 3 == 0 or s["month"] == len(base) - 1:  # show quarterly + last
        runway_str = f"{s['runway_months']:.0f}mo" if s['runway_months'] < 999 else "inf"
        cash_str = fmt_money(s['cash']) if s['cash'] > 0 else f"({fmt_money(abs(s['cash']))})"
        summary_lines.append(
            f"  {s['month']:>3}  {s['date_str']:>7}  {fmt_money(s['mrr']):>8}  {fmt_money(s['arr']):>9}"
            f"  {fmt_money(s['total_costs']):>8}  {fmt_money(s['net_burn']):>9}  {cash_str:>10}  {runway_str:>7}  {s['headcount']:>4}"
        )

summary_text = "\n".join(summary_lines)
print(summary_text)

with open(OUT_DIR / "summary.txt", "w") as f:
    f.write(summary_text)

# ── JSON output for dashboard ──────────────────

summary_json = {
    "company_name": A["company"]["name"],
    "current": {
        "cash": s0["cash"],
        "mrr": s0["mrr"],
        "arr": s0["arr"],
        "total_costs": s0["total_costs"],
        "net_burn": s0["net_burn"],
        "runway_months": s0["runway_months"],
        "headcount": s0["headcount"],
    },
    "fundraising": {
        "start_raising": start_raise["date_str"] if start_raise else None,
        "close_by": raise_deadline["date_str"] if raise_deadline else None,
        "recommended_amount": raise_amount,
        "avg_burn": round(avg_burn),
        "target_runway": A["targets"]["target_runway_months"],
        "min_runway_threshold": A["targets"]["min_runway_months"],
    },
    "series_a": {
        "ready_date": sa_ready["date_str"] if sa_ready else None,
        "arr_at_ready": sa_ready["arr"] if sa_ready else None,
        "mrr_at_ready": sa_ready["mrr"] if sa_ready else None,
        "growth_at_ready": round(sa_growth, 3) if sa_growth else None,
        "est_valuation": sa_ready["mrr"] * A["targets"]["series_a"]["target_valuation_multiple"] if sa_ready else None,
        "final_arr": base[-1]["arr"],
    },
    "projections": [
        {
            "date": s["date_str"],
            "mrr": s["mrr"],
            "arr": s["arr"],
            "costs": s["total_costs"],
            "net_burn": s["net_burn"],
            "cash": s["cash"],
            "runway": s["runway_months"],
            "headcount": s["headcount"],
            "customers": s["customers"],
        }
        for s in base
    ],
}

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary_json, f, indent=2)

# ── Charts ─────────────────────────────────────

# Chart style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'figure.dpi': 150,
})

months_arr = [s["month"] for s in base]
dates = [s["date"].strftime("%b\n%y") for s in base]

# Only show every 3rd month label
tick_positions = [i for i in months_arr if i % 3 == 0]
tick_labels = [dates[i] for i in tick_positions]

# 1. Cash + Runway Chart
fig, ax1 = plt.subplots(figsize=(10, 5))

for sc_name, sc_data in scenarios.items():
    cash_values = [s["cash"] / 1000 for s in sc_data]
    style = {"base": "-", "upside": "--", "downside": ":"}
    alpha = {"base": 1.0, "upside": 0.6, "downside": 0.6}
    color = {"base": "#2563EB", "upside": "#16A34A", "downside": "#DC2626"}
    ax1.plot(months_arr, cash_values, style.get(sc_name, "-"),
             color=color.get(sc_name, "#666"), alpha=alpha.get(sc_name, 0.8),
             label=f"{sc_name} case", linewidth=2)

# Zero line
ax1.axhline(y=0, color='#DC2626', linewidth=0.8, linestyle='-', alpha=0.3)

# Fundraising markers
if start_raise:
    ax1.axvline(x=start_raise["month"], color='#F59E0B', linewidth=1.5, linestyle='--', alpha=0.7)
    ax1.annotate(f'Start raising\n{start_raise["date_str"]}',
                xy=(start_raise["month"], start_raise["cash"]/1000),
                xytext=(15, 30), textcoords='offset points',
                fontsize=9, color='#F59E0B',
                arrowprops=dict(arrowstyle='->', color='#F59E0B', lw=1.5))

ax1.set_xlabel("Month")
ax1.set_ylabel("Cash ($K)")
ax1.set_title(f"{A['company']['name']} -- Cash Runway", fontweight='bold', fontsize=14)
ax1.legend(loc='upper right', framealpha=0.9)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}K"))
ax1.set_xticks(tick_positions)
ax1.set_xticklabels(tick_labels, fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "runway_chart.png", bbox_inches='tight')
plt.close()

# 2. Revenue Chart
fig, ax = plt.subplots(figsize=(10, 5))

for sc_name, sc_data in scenarios.items():
    mrr_values = [s["mrr"] / 1000 for s in sc_data]
    style = {"base": "-", "upside": "--", "downside": ":"}
    color = {"base": "#16A34A", "upside": "#2563EB", "downside": "#DC2626"}
    ax.plot(months_arr, mrr_values, style.get(sc_name, "-"),
            color=color.get(sc_name, "#666"), alpha=0.8,
            label=f"{sc_name} case", linewidth=2)

# Series A threshold
sa_min_mrr = A["targets"]["series_a"]["min_arr"] / 12 / 1000
ax.axhline(y=sa_min_mrr, color='#8B5CF6', linewidth=1.5, linestyle='--', alpha=0.5)
ax.annotate(f'Series A threshold\n({fmt_money(A["targets"]["series_a"]["min_arr"])} ARR)',
            xy=(months_arr[-1], sa_min_mrr),
            xytext=(-15, 15), textcoords='offset points',
            fontsize=9, color='#8B5CF6', ha='right',
            arrowprops=dict(arrowstyle='->', color='#8B5CF6', lw=1.5))

ax.set_xlabel("Month")
ax.set_ylabel("MRR ($K)")
ax.set_title(f"{A['company']['name']} -- Monthly Recurring Revenue", fontweight='bold', fontsize=14)
ax.legend(loc='upper left', framealpha=0.9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}K"))
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "revenue_chart.png", bbox_inches='tight')
plt.close()

# 3. Fundraising Timeline Chart
fig, ax = plt.subplots(figsize=(10, 4))

# Cash curve (base)
cash_values = [s["cash"] / 1000 for s in base]
ax.fill_between(months_arr, cash_values, 0, alpha=0.1, color='#2563EB')
ax.plot(months_arr, cash_values, '-', color='#2563EB', linewidth=2, label='Cash')

# Zones
for s in base:
    if s["runway_months"] < A["targets"]["min_runway_months"] and s["cash"] > 0:
        ax.axvspan(s["month"] - 0.5, s["month"] + 0.5, alpha=0.15, color='#DC2626')

if start_raise:
    ax.axvspan(start_raise["month"] - 0.5,
               raise_deadline["month"] + 0.5 if raise_deadline else start_raise["month"] + A["targets"]["fundraise_duration_months"] + 0.5,
               alpha=0.1, color='#F59E0B', label='Fundraising window')

# Post-raise projection (if raise happens)
if start_raise:
    raise_month = start_raise["month"] + A["targets"]["fundraise_duration_months"]
    if raise_month < len(base):
        post_raise_cash = base[raise_month]["cash"] + raise_amount
        ax.annotate(f'+{fmt_money(raise_amount)} raise\n= {fmt_money(post_raise_cash)}',
                    xy=(raise_month, base[raise_month]["cash"]/1000),
                    xytext=(0, 40), textcoords='offset points',
                    fontsize=9, color='#16A34A', ha='center', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#16A34A', lw=2))

# Series A marker
if sa_ready:
    ax.axvline(x=sa_ready["month"], color='#8B5CF6', linewidth=1.5, linestyle=':', alpha=0.5)
    ax.annotate(f'Series A ready\n{sa_ready["date_str"]}',
                xy=(sa_ready["month"], 0),
                xytext=(0, -30), textcoords='offset points',
                fontsize=9, color='#8B5CF6', ha='center',
                arrowprops=dict(arrowstyle='->', color='#8B5CF6', lw=1.5))

ax.set_xlabel("Month")
ax.set_ylabel("Cash ($K)")
ax.set_title(f"{A['company']['name']} -- Fundraising Timeline", fontweight='bold', fontsize=14)
ax.legend(loc='upper right', framealpha=0.9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}K"))
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "fundraising_timeline.png", bbox_inches='tight')
plt.close()

print(f"\nCharts saved to {OUT_DIR}/")
print(f"  - runway_chart.png")
print(f"  - revenue_chart.png")
print(f"  - fundraising_timeline.png")
