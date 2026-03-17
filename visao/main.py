from pathlib import Path
from person_tracker import PersonTracker

data_config_path = Path(__file__).resolve().parent / "data.yaml"

surveillance_system = PersonTracker()
surveillance_system.validate(data_config=str(data_config_path))
# surveillance_system.run()