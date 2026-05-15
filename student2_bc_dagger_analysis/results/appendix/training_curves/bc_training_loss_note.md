# BC Training Loss Diagnostic Note

This figure is an appendix diagnostic for BC training, not a main performance result.

BC is trained as supervised learning on state-action pairs using mean-squared error (MSE) loss. The plotted curves are copied from existing BC training outputs for representative noisy ratios; no new training or evaluation was run.

Final performance should still be reported using closed-loop rollout success rate and mean episode steps, because lower validation loss does not necessarily guarantee task success during autonomous rollout.

In general, clean data tends to produce lower validation loss, while noisy demonstrations increase supervised fitting difficulty. However, validation loss and final success rate need not be monotonic with each other. This mismatch is one of the reasons closed-loop evaluation is important for covariate shift analysis.

Representative ratios used: noise0, noise30, noise50.
