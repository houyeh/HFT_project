# ML Result Key Numbers

- `val`: baseline `1,218,095`, after ML veto `1,227,410`, incremental `9,315`, accepted useful ML `True`.
- `test`: baseline `2,312,590`, after ML veto `2,312,590`, incremental `0`, accepted useful ML `False`.
- `new4`: baseline `6,348,770`, after ML veto `6,341,385`, incremental `-7,385`, accepted useful ML `False`.

The safe conclusion is that residual ML did not pass the incremental value hurdle. It should remain a future-work route, not part of the final core controller.

Epoch-level training curves are shown as a placeholder because earlier Phase 28/30 scripts did not save epoch logs. The validation threshold curve is provided as the available training-selection diagnostic.
