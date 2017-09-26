Survey Planning Tile Prioritization Rules
=========================================

Rules for how tiles are prioritized are contained in a YAML file,
with a baseline survey example in `py/desisurvey/data/rules.yaml`.
Each tile group is separately scheduled and prioritized.
All tiles in the footprint must be assigned to exactly one group.

Group names are arbitrary strings. The tiles associated with a
group can be optionally restricted using:

  * cap = N or S.
  * dec_min or dec_max limits.
  * covers keyword specifies previously defined subgroups that the tiles in
    this group cover (for fiber assignment).

Groups are also associated with a list of passes, with each pass
defining a subgroup.

Each subgroup has associated rules that specify its relative weight
when selecting the next tile in the scheduler.

Rules are assigned separately for each subgroup using the notation::

  GROUP_NAME(PASS): { ...rules... }

By default, each subgroup has an initial weight of zero when the
survey starts (so will not be scheduled), unless it has an explicit
start rule::

  START: INITIAL_WEIGHT

Additional rules of the form::

  GROUP_NAME(PASS): NEW_WEIGHT

specify the subgroup weight that will be assigned when the specified
subgroup has been completely observed. Forward references to subgroups
that have not been defined yet are ok.
