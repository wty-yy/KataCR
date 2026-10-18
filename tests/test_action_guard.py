import unittest

from katacr.policy.env.action_guard import ActionGuard


class ActionGuardTest(unittest.TestCase):
  def setUp(self):
    self.guard = ActionGuard({
      "next": {
        "home": "battle",
        "post_match": "post-match",
      },
      "confirm": {"post_match": "confirm"},
    })

  def test_unknown_state_denies_action(self):
    decision = self.guard.decide("next", "unknown", confirmed=True)

    self.assertFalse(decision.allowed)
    self.assertEqual(decision.action, "none")

  def test_mismatched_state_denies_action(self):
    decision = self.guard.decide("confirm", "home", confirmed=True)

    self.assertFalse(decision.allowed)
    self.assertEqual(decision.action, "none")

  def test_matching_state_suggests_without_confirmation(self):
    decision = self.guard.decide("next", "home")

    self.assertFalse(decision.allowed)
    self.assertEqual(decision.action, "battle")
    self.assertIn("confirmation", decision.reason)

  def test_matching_state_allows_confirmed_action(self):
    decision = self.guard.decide("confirm", "post_match", confirmed=True)

    self.assertTrue(decision.allowed)
    self.assertEqual(decision.action, "confirm")

  def test_force_allows_explicit_recovery(self):
    decision = self.guard.decide("confirm", "unknown", force=True)

    self.assertTrue(decision.allowed)
    self.assertEqual(decision.action, "confirm")


if __name__ == "__main__":
  unittest.main()
