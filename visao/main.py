from pathlib import Path
from person_tracker import PersonTracker

data_config_path = Path(__file__).resolve().parent / "data.yaml"

survaillence_system = PersonTracker()
survaillence_system.validate(data_config=str(data_config_path))
# survaillence_system.run()