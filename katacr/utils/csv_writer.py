from pathlib import Path
import csv
from typing import Sequence

class CSVWriter:
  def __init__(self, path_csv: Path, title: Sequence = None):
    assert path_csv.suffix == '.csv'
    self.path_csv = path_csv
    self.path_csv.parent.mkdir(parents=True, exist_ok=True)
    if title is not None:
      self.write(title)
  
  def write(self, data: Sequence):
    with self.path_csv.open('a') as file:
      writer = csv.writer(file)
      writer.writerow(data)

if __name__ == '__main__':
  path_root = Path(__file__).parents[2]
  path_logs = path_root / 'logs'
  csv_writer = CSVWriter(path_logs / "test.csv", title=['A', 'B', 'C', '4'])
  csv_writer.write([321,123,4,2])
  csv_writer.write([1,2,3,4])
