"""Old module name of the critique and revise half of the layer 4 modules, kept for importers outside this package.
``run`` is the critique and revise driver (the draft step is run by ``layer2_draft.run``); every public and underscore name
of the new modules is available here under the name it had."""
import importlib

for _new_module_name in ("sdf_pipeline.layer4_draft_critique_and_revise_shared", "sdf_pipeline.steps.critique",
                         "sdf_pipeline.steps.revise", "sdf_pipeline.steps.critique_and_revise_in_one_call",
                         "sdf_pipeline.layer4_draft_critique_and_revise"):
    globals().update({name: value for name, value in vars(importlib.import_module(_new_module_name)).items()
                      if not name.startswith("__")})
run = run_critique_and_revise  # noqa: F821
_MAX_TOKENS = MAX_TOKENS  # noqa: F821
_CRITIQUE_MAX_TOKENS = CRITIQUE_MAX_TOKENS  # noqa: F821
