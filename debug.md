Q1: pyarrow.lib.ArrowInvalid: Column 3 named pixel_values expected length 98 but got length 25088

A1:pixel_values 是图像数据的张量，通常应为 [batch_size, num_channels, height, width] 的形状（例如 [98, 3, 224, 224]），但你的输出显示为 [25088, 1176]，说明处理器（processor）的输出形状与预期不符
