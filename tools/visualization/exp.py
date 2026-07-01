import re

logs = [
    "2025-05-27 10:22:45,204 - Uchiha - INFO - Mean PSNR: 36.04 dB",
    "2025-05-27 10:22:45,204 - Uchiha - INFO - Mean SSIM: 0.9579"
]

# 提取 PSNR 和 SSIM 的正则表达式
psnr_pattern = r'PSNR:\s([\d.]+)'
ssim_pattern = r'SSIM:\s([\d.]+)'

for log in logs:
    if "PSNR" in log:
        psnr = re.search(psnr_pattern, log).group(1)
        print(f"PSNR: {psnr} dB")
    elif "SSIM" in log:
        ssim = re.search(ssim_pattern, log).group(1)
        print(f"SSIM: {ssim}")

# pattern = r'lr:([\d.e+-]+)'
# r'epoch:\[(\d+)/(\d+)\]'
# r'iter:\[(\d+)/(\d+)\]'
# loss:\s*([\d.]+)


log_line = "epoch:[23/100]    iter:[50/2520]    loss: 0.012178    lr:0.00026502364424628715"

# 匹配 epoch:[数字/数字]
match = re.search(r'epoch:\[(\d+)/(\d+)\]\s*iter:\[(\d+)/(\d+)\]\s*loss:\s*([\d.]+)\s*lr:([\d.e+-]+)', log_line)
if match:
    current_epoch, total_epoch = match.groups()
    print(f"Current epoch: {current_epoch}, Total epochs: {total_epoch}")
