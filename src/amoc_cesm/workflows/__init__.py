"""High-level analysis workflows.

Two of them, sharing everything below the reduction step:

- `steady_state` — quasi-steady-state, the last 50 years of each run treated as
  an equilibrium climate. Figures now; tables to follow.
- `transient` — time-dependent, year-for-year against the control. To be built.

Both consume `amoc_cesm.plotting.grid_3x3`, which is agnostic about which
reduction produced the fields it draws.
"""
