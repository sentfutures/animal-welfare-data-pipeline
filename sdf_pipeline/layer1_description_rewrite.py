"""Old module name of the new modules ``sdf_pipeline.steps.description_rewrite``, kept for importers outside this package.
Every public and underscore name of those modules is available here under the name it had."""
import importlib

for _new_module_name in ['sdf_pipeline.steps.description_rewrite']:
    globals().update({name: value for name, value in vars(importlib.import_module(_new_module_name)).items()
                      if not name.startswith("__")})
