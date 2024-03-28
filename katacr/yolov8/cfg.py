unit_nums = 40
map_update_mode = 'dynamic'
intersect_ratio_thre = 0.6
train_datasize = 20000
img_size = (576, 896)  # width, height
assert img_size[0] % 32 == 0 and img_size[1] % 32 == 0, "The shape must be 32 multiple"
