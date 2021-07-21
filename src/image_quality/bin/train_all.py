from image_quality.bin.train_koniq import main as train_k
from image_quality.bin.train_spaq import main as train_s
from image_quality.bin.train_koniq_mos import main as train_s_mos


try:
    train_k()
except Exception as e:
    print(e)

try:
    train_s()
except Exception as e:
    print(e)

try:
    train_s_mos()
except Exception as e:
    print(e)