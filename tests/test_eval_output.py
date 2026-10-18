import unittest
from pathlib import Path

from katacr.policy.offline.eval_output import resolve_output_dir


class EvalOutputTest(unittest.TestCase):
  def test_disabled_saving_returns_no_output_directory(self):
    self.assertIsNone(resolve_output_dir("workspace", "match.mp4", save=False))

  def test_video_outputs_are_grouped_by_video_name(self):
    output = resolve_output_dir(
      "workspace",
      Path("recordings") / "match.mp4",
      save=True,
      timestamp="20261018_120000",
    )

    self.assertEqual(
      output,
      Path("workspace") / "logs" / "offline_eval" / "match" / "20261018_120000",
    )

  def test_live_outputs_keep_the_interaction_layout(self):
    output = resolve_output_dir(
      "workspace",
      save=True,
      timestamp="20261018_120000",
    )

    self.assertEqual(
      output,
      Path("workspace") / "logs" / "interaction" / "20261018_120000",
    )


if __name__ == "__main__":
  unittest.main()
