from katacr.utils import Stopwatch

class SARBuilder:
  def __init__(self, verbose=True):
    from katacr.policy.visualization.visual_fusion import VisualFusion
    from katacr.policy.perceptron.state_builder import StateBuilder
    from katacr.policy.perceptron.action_builder import ActionBuilder
    from katacr.policy.perceptron.reward_builder import RewardBuilder
    if verbose:
      print("Building SAR builder...")
    self.visual_fusion = VisualFusion()
    ocr = self.visual_fusion.ocr
    self.state_builder = StateBuilder(ocr=ocr)
    self.action_builder = ActionBuilder(ocr=ocr)
    self.reward_builder = RewardBuilder(ocr=ocr)
    self.sw = [Stopwatch() for _ in range(5)]
    self.reset()
    if verbose:
      print("SAR builder complete!")
  
  def reset(self):
    self.s, self.a, self.r = [None] * 3
    if self.visual_fusion.yolo.tracker is not None:
      self.visual_fusion.yolo.tracker.reset()
    self.state_builder.reset()
    self.action_builder.reset()
    self.reward_builder.reset()
  
  def update(self, img):
    """ Update builders' visual info (img: BGR) """
    with self.sw[0]:
      info = self.visual_info = self.visual_fusion.process(img)
    box = info['arena'].get_data()
    if box.shape[-1] != 8:
      print(f"Warning(state): The last dim should be 8, but get {box.shape[-1]}")
      return None
    with self.sw[1]:
      self.action_builder.update(info)
    with self.sw[2]:
      self.state_builder.update(info, self.action_builder.deploy_cards)
    with self.sw[3]:
      self.reward_builder.update(info)
    return info, [self.sw[i].dt for i in range(4)]
    
  def get_sar(self, verbose=False):
    with self.sw[4]:
      self.s = self.state_builder.get_state(verbose=verbose)
      self.a = self.action_builder.get_action(verbose=verbose)
      self.r = self.reward_builder.get_reward(verbose=verbose)
    return self.s, self.a, self.r, self.sw[4].dt
    
  def render(self):
    img = self.state_builder.render(self.a)
    img = self.reward_builder.render(img, self.r)
    return img
  