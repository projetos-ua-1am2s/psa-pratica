from pathlib import Path
try:
    # Prefer relative import when `visao` is used as a package
    from .person_tracker import PersonTracker
except ImportError:
    # Fallback for running this file directly as a script
    from person_tracker import PersonTracker

data_config_path = Path(__file__).resolve().parent / "data.yaml"

surveillance_system = PersonTracker()
surveillance_system.validate(data_config=str(data_config_path))
# surveillance_system.run()
